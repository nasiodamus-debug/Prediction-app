"""Stories 7.1 and 7.4: NFL and college football stats from ESPN's site API (scoreboard weeks, team statistics) and the wind
figure from Open-Meteo through the stadium table. Recorded samples (tests/data/espn_*.json, open_meteo_wind_sample.json,
probed 2026-09-30) check the parsers on real shapes; fabricated multi-team leagues (httpx.MockTransport) check what is
derived, how it is covered and what is "insufficient data". No live call."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from predictions.adapters.espn import football as fb
from predictions.adapters.weather import openmeteo
from predictions.adapters.weather.stadiums import NFL_VENUES, ROOFS, stadium_for_city
from predictions.core.ports import StatsFetchError
from predictions.core.stats import AMBIGUOUS, INSUFFICIENT, READY
from predictions.core.types import Fixture, MatchId

DATA = Path(__file__).resolve().parent / "data"
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
KICK = datetime(2026, 10, 4, 17, 0, tzinfo=timezone.utc)


def run(coro):
    return asyncio.run(coro)


def sample(name):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


# ================================================================ recorded samples
def test_the_recorded_nfl_scoreboard_gives_final_scores_and_quarter_scores():
    board = fb.parse_scoreboard(sample("espn_nfl_scoreboard_sample.json"))
    assert board.week == 3 and board.year == 2026 and board.season_type == 2
    game = next(g for g in board.played if g.home.name == "Buffalo Bills")
    assert (game.home.score, game.away.score) == (24, 16)
    assert game.home.quarters == (0.0, 10.0, 0.0, 14.0) and game.away.quarters == (10.0, 0.0, 3.0, 3.0)
    assert any(s.city == "Orchard Park" for s in board.scheduled)


def test_the_recorded_upcoming_scoreboard_has_the_venue_city_of_scheduled_games():
    board = fb.parse_scoreboard(sample("espn_nfl_upcoming_sample.json"))
    assert not board.played and len(board.scheduled) == 3
    cities = {s.home: s.city for s in board.scheduled}
    assert cities["Buffalo Bills"] == "Orchard Park" and cities["Chicago Bears"] == "Chicago"
    assert cities["Washington Commanders"] == "London"      # the Colts-Commanders game was a London game: venue, not home team


def test_the_recorded_college_scoreboard_parses_the_same_shape():
    board = fb.parse_scoreboard(sample("espn_college_scoreboard_sample.json"))
    assert board.played and all(len(g.home.quarters) >= 4 for g in board.played)
    assert stadium_for_city(board.scheduled[0].city) is None                    # college cities are not in the table


def test_the_recorded_team_statistics_give_pace_and_rush_rate():
    pace, rush = fb.parse_team_statistics(sample("espn_nfl_team_statistics_sample.json"))
    assert pace == pytest.approx(5166.0 / 184.0) and 28.0 < pace < 28.2
    assert rush == pytest.approx(90.0 / 184.0 * 100.0)


@pytest.mark.parametrize("payload", [{}, {"results": {}}, {"results": {"stats": {"categories": []}}}])
def test_team_statistics_without_the_needed_totals_give_no_pace(payload):
    assert fb.parse_team_statistics(payload) is None


def test_the_recorded_open_meteo_response_gives_the_highest_wind_from_kickoff_to_three_hours_later():
    payload = sample("open_meteo_wind_sample.json")
    speeds = dict(zip(payload["hourly"]["time"], payload["hourly"]["wind_speed_10m"]))
    kickoff = datetime(2026, 10, 4, 17, 30, tzinfo=timezone.utc)
    expected = max(speeds[f"2026-10-04T{h}:00"] for h in ("17", "18", "19", "20"))
    assert openmeteo.parse_wind(payload, kickoff) == expected
    assert payload["hourly_units"]["wind_speed_10m"] == "mp/h"


@pytest.mark.parametrize("bad", [None, {}, {"hourly": {}}, {"hourly": {"time": ["2026-10-04T17:00"], "wind_speed_10m": []}}])
def test_an_unreadable_wind_response_is_none(bad):
    assert openmeteo.parse_wind(bad, KICK) is None


def wind_payload(speeds):
    return {"hourly": {"time": [f"2026-10-04T{h:02d}:00" for h in range(24)], "wind_speed_10m": speeds}}


def test_the_wind_window_runs_from_the_kickoff_hour_through_three_hours_later_inclusive():
    speeds = [0.0] * 24
    speeds[17], speeds[20] = 9.0, 14.9
    assert openmeteo.parse_wind(wind_payload(speeds), KICK) == 14.9          # hour +3 counts
    speeds[20], speeds[21] = 0.0, 30.0
    assert openmeteo.parse_wind(wind_payload(speeds), KICK) == 9.0           # hour +4 does not
    speeds[16] = 40.0
    assert openmeteo.parse_wind(wind_payload(speeds), KICK) == 9.0           # the hour before kickoff does not


def test_a_missing_or_nonnumeric_hour_in_the_window_means_no_figure_never_a_guess():
    speeds = [5.0] * 24
    speeds[19] = None
    assert openmeteo.parse_wind(wind_payload(speeds), KICK) is None
    speeds[19] = float("nan")
    assert openmeteo.parse_wind(wind_payload(speeds), KICK) is None


def test_the_forecast_is_only_asked_within_its_horizon_and_a_failure_is_none():
    calls = []

    def handler(request):
        calls.append(request.url)
        return httpx.Response(200, json=wind_payload([11.0] * 24))

    async def go(kickoff, transport=handler):
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            return await openmeteo.fetch_wind(client, NFL_VENUES["orchard park"], kickoff, NOW)
    assert run(go(KICK)) == 11.0
    params = dict(calls[0].params)
    assert params["wind_speed_unit"] == "mph" and params["hourly"] == "wind_speed_10m" and params["timezone"] == "UTC"
    assert run(go(NOW + timedelta(days=16))) is None and len(calls) == 1          # beyond the horizon: no request
    assert run(go(NOW - timedelta(days=2))) is None and len(calls) == 1          # already played: no request
    assert run(go(KICK, lambda request: httpx.Response(500))) is None
    assert run(go(KICK, lambda request: httpx.Response(200, text="not json"))) is None


# ================================================================ the stadium table
def test_every_stadium_row_is_well_formed_and_looked_up_by_the_venue_city_as_espn_spells_it():
    assert len(NFL_VENUES) >= 38
    for stadium in NFL_VENUES.values():
        assert stadium.roof in ROOFS and -90 <= stadium.latitude <= 90 and -180 <= stadium.longitude <= 180
        assert stadium.name
    assert stadium_for_city("Orchard Park").name == "Highmark Stadium"
    assert stadium_for_city("  orchard  park ").roof == "open"
    assert stadium_for_city("Detroit").roof == "dome" and stadium_for_city("Minneapolis").roof == "dome"
    assert stadium_for_city("London") is not None and stadium_for_city("Mexico City") is not None
    assert stadium_for_city("Springfield") is None and stadium_for_city("") is None and stadium_for_city(None) is None


# ================================================================ fabricated leagues
NFL_TEAMS = ["Buffalo Bills", "Chicago Bears", "Detroit Lions", "Cleveland Browns", "Miami Dolphins", "Denver Broncos",
             "Seattle Seahawks", "Green Bay Packers", "New York Giants", "New York Jets", "Kansas City Chiefs", "Dallas Cowboys"]
PAIRS = {1: [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11)],
         2: [(0, 2), (1, 3), (4, 6), (5, 7), (8, 10), (9, 11)],
         3: [(0, 3), (1, 2), (4, 7), (5, 6), (8, 11), (9, 10)]}
Q2, Q3, Q4 = 7.0, 3.0, 4.0


def team_q1(i):
    return float(11 - i)          # team 0 has the best first quarter; every game of a team is the same


def event(eid, when, home, away, city, completed=True, season_type=2, hq=None, aq=None):
    def competitor(kind, name, index, quarters):
        line = {"homeAway": kind, "team": {"id": str(index + 1), "displayName": name}}
        if completed:
            line["score"] = str(int(sum(quarters)))
            line["linescores"] = [{"value": q, "period": n + 1} for n, q in enumerate(quarters)]
        return line
    h = hq or [team_q1(NFL_TEAMS.index(home)), Q2, Q3, Q4]
    a = aq or [team_q1(NFL_TEAMS.index(away)), Q2, Q3, Q4]
    return {"id": str(eid), "date": when.strftime("%Y-%m-%dT%H:%MZ"), "season": {"year": 2026, "type": season_type},
            "status": {"type": {"completed": completed}},
            "competitions": [{"venue": {"address": {"city": city}},
                              "competitors": [competitor("home", home, NFL_TEAMS.index(home), h),
                                              competitor("away", away, NFL_TEAMS.index(away), a)]}]}


def week_board(week, events):
    return {"season": {"year": 2026, "type": 2}, "week": {"number": week}, "events": events}


def history(weeks=(1, 2, 3)):
    boards = {}
    for w in weeks:
        day = datetime(2026, 9, 6, 17, 0, tzinfo=timezone.utc) + timedelta(days=7 * (w - 1))
        boards[w] = week_board(w, [event(w * 100 + n, day, NFL_TEAMS[a], NFL_TEAMS[b], "Somewhere")
                                   for n, (a, b) in enumerate(PAIRS[w])])
    return boards


UPCOMING = [
    ("Buffalo Bills", "Chicago Bears", "Orchard Park", datetime(2026, 10, 4, 17, 0, tzinfo=timezone.utc)),
    ("Detroit Lions", "Cleveland Browns", "Detroit", datetime(2026, 10, 4, 17, 0, tzinfo=timezone.utc)),
    ("Miami Dolphins", "Denver Broncos", "London", datetime(2026, 10, 4, 13, 30, tzinfo=timezone.utc)),
    ("Seattle Seahawks", "Green Bay Packers", "Nowhereville", datetime(2026, 10, 4, 20, 25, tzinfo=timezone.utc)),
]


def current_board(events=None):
    if events is None:
        events = [event(400 + n, when, home, away, city, completed=False) for n, (home, away, city, when) in enumerate(UPCOMING)]
    return week_board(4, events)


def team_stats(poss, plays, rush):
    return {"results": {"stats": {"categories": [
        {"name": "rushing", "stats": [{"name": "rushingAttempts", "value": rush}, {"name": "totalOffensivePlays", "value": plays}]},
        {"name": "miscellaneous", "stats": [{"name": "possessionTimeSeconds", "value": poss}]}]}}}


class Espn:
    """An ESPN + Open-Meteo double. `boards` maps a week to a scoreboard, `stats` a team id to its statistics payload."""

    def __init__(self, boards=None, current=None, stats=None, wind=12.0, weather_status=200, espn_status=200, next_week=None):
        self.boards = history() if boards is None else boards
        self.current = current_board() if current is None else current
        self.stats = stats if stats is not None else {}
        self.wind, self.weather_status, self.espn_status = wind, weather_status, espn_status
        self.next_week = week_board(5, []) if next_week is None else next_week
        self.requests = []

    def handler(self, request: httpx.Request):
        self.requests.append(request)
        url, params = str(request.url), dict(request.url.params)
        if "open-meteo" in url:
            if self.weather_status != 200:
                return httpx.Response(self.weather_status)
            return httpx.Response(200, json=wind_payload([self.wind] * 24))
        if self.espn_status != 200:
            return httpx.Response(self.espn_status)
        if request.url.path.endswith("/scoreboard"):
            if "week" not in params:
                return httpx.Response(200, json=self.current)
            week = int(params["week"])
            return httpx.Response(200, json=self.boards.get(week, self.next_week if week == 5 else week_board(week, [])))
        if "/teams/" in request.url.path and request.url.path.endswith("/statistics"):
            team_id = request.url.path.split("/")[-2]
            payload = self.stats.get(team_id)
            return httpx.Response(200, json=payload) if payload else httpx.Response(404)
        return httpx.Response(404)

    def source(self):
        return fb.EspnFootballStats(httpx.AsyncClient(transport=httpx.MockTransport(self.handler)), clock=lambda: NOW)

    def espn_requests(self):
        return [r for r in self.requests if "open-meteo" not in str(r.url)]


def fixture(home, away, kickoff=KICK, competition="USA - NFL", sport="american_football"):
    mid = MatchId(sport, competition, kickoff, home, away)
    return Fixture(mid, competition, home, away, kickoff, False)


def fetch(api, *fixtures):
    return run(api.source().fetch_stats(list(fixtures)))


BILLS = fixture("Buffalo Bills", "Chicago Bears")
LIONS = fixture("Detroit Lions", "Cleveland Browns")
LONDON = fixture("Miami Dolphins", "Denver Broncos", datetime(2026, 10, 4, 13, 30, tzinfo=timezone.utc))
NOWHERE = fixture("Seattle Seahawks", "Green Bay Packers", datetime(2026, 10, 4, 20, 25, tzinfo=timezone.utc))
STATS = {"1": team_stats(5166.0, 184.0, 90.0), "2": team_stats(5400.0, 180.0, 99.0)}


def values(result):
    return {i.name: i for i in result.inputs}


def test_points_quarters_and_league_levels_are_derived_from_the_last_games_with_provenance():
    result = fetch(Espn(stats=STATS), BILLS)[BILLS.match_id]
    assert result.status == READY
    got = values(result)
    assert got["home.matches"].value == 3 and got["home.points_for"].value == pytest.approx(25.0)
    assert got["home.points_against"].value == pytest.approx(23.0)          # opponents scored 24, 23, 22
    assert got["home.q1_points_for"].value == 11.0
    assert got["home.fh_points_for"].value == 18.0 and got["home.fh_points_against"].value == pytest.approx(16.0)
    assert got["away.points_for"].value == pytest.approx(24.0)
    assert got["match.league_ppg"].value == pytest.approx(19.5)
    assert all(i.source.startswith("espn-football:nfl") or i.source.startswith("open-meteo") for i in result.inputs)
    assert got["home.points_for"].source == "espn-football:nfl" and got["home.points_for"].unit == "points per game"


def test_the_first_quarter_rank_proxy_ranks_teams_by_first_quarter_points_per_game():
    got = values(fetch(Espn(stats=STATS), BILLS)[BILLS.match_id])
    assert got["home.q1_rank"].value == 1.0 and got["away.q1_rank"].value == 2.0
    assert "1st-quarter points" in got["home.q1_rank"].unit
    low = values(fetch(Espn(), fixture("Green Bay Packers", "Seattle Seahawks"))[fixture("Green Bay Packers", "Seattle Seahawks").match_id])
    assert low["home.q1_rank"].value == 8.0 and low["away.q1_rank"].value == 7.0


def test_ties_in_the_first_quarter_share_the_better_rank_and_the_top_ten_boundary_is_rank_ten():
    def game(gid, a, b, qa, qb, day):
        side = lambda tid, q: fb.Side(str(tid), f"T{tid}", int(q + 14), (q, 7.0, 3.0, 4.0))  # noqa: E731
        return fb.PlayedGame(str(gid), datetime(2026, 9, day, tzinfo=timezone.utc), side(a, qa), side(b, qb))
    games = []
    n = 0
    for w, day in enumerate((6, 13, 20)):        # 12 teams; team i scores i // 2 first-quarter points (so pairs tie)
        for a, b in PAIRS[w + 1]:
            n += 1
            games.append(game(n, a, b, a // 2, b // 2, day))
    ranks = fb.q1_ranks(games)
    assert ranks["10"] == ranks["11"] == 1 and ranks["0"] == ranks["1"] == 11    # ten teams are ahead: rank 11, outside the top 10
    assert len(ranks) == 12


def test_a_team_needs_three_games_to_be_ranked():
    side = lambda tid: fb.Side(str(tid), f"T{tid}", 20, (3.0, 7.0, 3.0, 7.0))  # noqa: E731
    games = [fb.PlayedGame("1", datetime(2026, 9, 6, tzinfo=timezone.utc), side(1), side(2)),
             fb.PlayedGame("2", datetime(2026, 9, 13, tzinfo=timezone.utc), side(1), side(3)),
             fb.PlayedGame("3", datetime(2026, 9, 20, tzinfo=timezone.utc), side(1), side(2))]
    assert set(fb.q1_ranks(games)) == {"1"}          # team 1 has three games, team 2 two, team 3 one


def test_pace_and_rush_rate_come_from_the_season_totals_with_units():
    got = values(fetch(Espn(stats=STATS), BILLS)[BILLS.match_id])
    assert got["home.pace"].value == pytest.approx(5166.0 / 184.0) and got["home.pace"].unit == "seconds per offensive play"
    assert got["home.rush_rate"].value == pytest.approx(90.0 / 184.0 * 100.0)
    assert got["away.pace"].value == pytest.approx(5400.0 / 180.0) and got["away.rush_rate"].value == pytest.approx(55.0)


def test_missing_team_statistics_only_drop_the_pace_inputs_the_fixture_stays_ready():
    result = fetch(Espn(stats={}), BILLS)[BILLS.match_id]
    assert result.status == READY and "home.pace" not in values(result) and "home.rush_rate" not in values(result)


def test_wind_for_an_outdoor_stadium_comes_from_open_meteo_with_the_stadium_in_the_provenance():
    api = Espn(stats=STATS, wind=12.5)
    wind = values(fetch(api, BILLS)[BILLS.match_id])["match.wind_mph"]
    assert wind.value == 12.5 and wind.unit == "mph" and wind.source.startswith("open-meteo: Highmark Stadium")
    forecast = [r for r in api.requests if "open-meteo" in str(r.url)]
    assert len(forecast) == 1 and dict(forecast[0].url.params)["latitude"].startswith("42.77")


def test_a_fixed_roof_stadium_has_zero_wind_by_the_table_and_no_forecast_is_read():
    api = Espn()
    wind = values(fetch(api, LIONS)[LIONS.match_id])["match.wind_mph"]
    assert wind.value == 0.0 and "stadium-table" in wind.source and "fixed roof" in wind.source
    assert not [r for r in api.requests if "open-meteo" in str(r.url)]


def test_a_neutral_or_international_venue_uses_the_venue_city_not_the_home_teams_stadium():
    api = Espn(wind=9.0)
    wind = values(fetch(api, LONDON)[LONDON.match_id])["match.wind_mph"]
    assert wind.value == 9.0 and "London" in wind.source
    assert dict([r for r in api.requests if "open-meteo" in str(r.url)][0].url.params)["latitude"].startswith("51.5")


def test_an_unknown_venue_city_gives_no_wind_input_so_an_over_will_be_insufficient_data():
    result = fetch(Espn(), NOWHERE)[NOWHERE.match_id]
    assert result.status == READY and "match.wind_mph" not in values(result)


def test_no_scheduled_game_in_the_window_gives_no_wind_input():
    later = fixture("Buffalo Bills", "Chicago Bears", datetime(2026, 10, 12, 17, 0, tzinfo=timezone.utc))
    assert "match.wind_mph" not in values(fetch(Espn(), later)[later.match_id])


def test_a_kickoff_beyond_the_forecast_horizon_or_a_weather_outage_gives_no_wind_but_never_fails_the_slate():
    api = Espn(weather_status=500)
    result = fetch(api, BILLS)[BILLS.match_id]
    assert result.status == READY and "match.wind_mph" not in values(result)
    far = NOW + timedelta(days=20)
    board = current_board([event(500, far, "Buffalo Bills", "Chicago Bears", "Orchard Park", completed=False)])
    api = Espn(current=board)
    far_fixture = fixture("Buffalo Bills", "Chicago Bears", far)
    assert "match.wind_mph" not in values(fetch(api, far_fixture)[far_fixture.match_id])
    assert not [r for r in api.requests if "open-meteo" in str(r.url)]


def test_a_slate_reads_a_handful_of_pages_not_one_per_game():
    api = Espn(stats=STATS)
    fetch(api, BILLS, LIONS, LONDON, NOWHERE)
    boards = [r for r in api.espn_requests() if r.url.path.endswith("/scoreboard")]
    assert len(boards) == 1 + 3 + 1          # the current week, weeks 1 to 3 (none before week 1), the next week
    assert len([r for r in api.espn_requests() if r.url.path.endswith("/statistics")]) == 8    # one per team, none twice
    assert all("custom" not in r.headers.get("user-agent", "") for r in api.requests)


def test_preseason_and_unfinished_games_are_not_counted():
    boards = history()
    boards[1]["events"].append(event(190, datetime(2026, 8, 20, tzinfo=timezone.utc), "Buffalo Bills", "Chicago Bears", "X",
                                     season_type=1, hq=[30.0, 30.0, 30.0, 30.0], aq=[0.0, 0.0, 0.0, 0.0]))
    boards[2]["events"].append(event(191, datetime(2026, 9, 13, tzinfo=timezone.utc), "Buffalo Bills", "Chicago Bears", "X",
                                     completed=False))
    got = values(fetch(Espn(boards=boards), BILLS)[BILLS.match_id])
    assert got["home.matches"].value == 3 and got["home.points_for"].value == pytest.approx(25.0)


def test_fewer_than_three_games_is_insufficient_data():
    result = fetch(Espn(boards=history(weeks=(3,))), BILLS)[BILLS.match_id]
    assert result.status == INSUFFICIENT and any("missing required input" in r for r in result.reasons)


def test_no_games_at_all_is_insufficient_data_naming_the_team():
    result = fetch(Espn(boards={}), BILLS)[BILLS.match_id]
    assert result.status == INSUFFICIENT and "Buffalo Bills has no completed regular-season games" in result.reasons[0]


def test_a_team_espn_does_not_know_is_insufficient_and_an_ambiguous_name_is_a_question():
    unknown = fixture("Atlantis Krakens", "Chicago Bears")
    assert fetch(Espn(), unknown)[unknown.match_id].status == INSUFFICIENT
    vague = fixture("New York", "Chicago Bears")
    result = fetch(Espn(), vague)[vague.match_id]
    assert result.status == AMBIGUOUS and set(result.options) == {"New York Giants", "New York Jets"}


def test_other_competitions_and_sports_are_insufficient_with_the_reason_and_nothing_is_read():
    cfl = fixture("Ottawa Redblacks", "Hamilton Tiger-Cats", competition="Canada - CFL")
    api = Espn()
    result = fetch(api, cfl)[cfl.match_id]
    assert result.status == INSUFFICIENT and "no free stats source covers Canada - CFL" in result.reasons[0]
    assert api.requests == []


def test_an_espn_outage_raises_a_stats_fetch_error_with_no_detail():
    with pytest.raises(StatsFetchError) as error:
        fetch(Espn(espn_status=500), BILLS)
    assert "HTTP 500" in str(error.value)
    with pytest.raises(StatsFetchError):
        fetch(Espn(current={"nope": 1}), BILLS)


def test_a_changed_scoreboard_shape_fails_loudly():
    with pytest.raises(StatsFetchError):
        fb.parse_scoreboard({"events": [{"id": 1}]})
    with pytest.raises(StatsFetchError):
        fb.parse_scoreboard([])


# ---- college
COLLEGE = "USA - NCAA, Regular Season"
COLLEGE_TEAMS = ["Delaware Blue Hens", "Liberty Flames", "Tulsa Golden Hurricane", "North Texas Mean Green", "Miami Hurricanes",
                 "Miami (OH) RedHawks"]


def college_event(eid, home, away, day):
    def side(kind, name):
        i = COLLEGE_TEAMS.index(name)
        return {"homeAway": kind, "team": {"id": str(100 + i), "displayName": name}, "score": "31",
                "linescores": [{"value": 7.0}, {"value": 10.0}, {"value": 7.0}, {"value": 7.0}]}
    return {"id": str(eid), "date": (datetime(2026, 9, 5, 17, 0, tzinfo=timezone.utc) + timedelta(days=day)).strftime("%Y-%m-%dT%H:%MZ"),
            "season": {"year": 2026, "type": 2}, "status": {"type": {"completed": True}},
            "competitions": [{"venue": {"address": {"city": "Minneapolis"}},
                              "competitors": [side("home", home), side("away", away)]}]}


def college_api():
    api = Espn(boards={w: week_board(w, [college_event(w * 10 + 1, "Delaware Blue Hens", "Liberty Flames", 7 * w),
                                          college_event(w * 10 + 2, "Tulsa Golden Hurricane", "North Texas Mean Green", 7 * w),
                                          college_event(w * 10 + 3, "Miami Hurricanes", "Miami (OH) RedHawks", 7 * w)])
                       for w in (1, 2, 3)}, current=week_board(4, []))
    return api


def test_college_teams_are_covered_through_the_fbs_scoreboard_with_a_spelling_alias_and_a_home_stadium_wind():
    api = college_api()
    game = fixture("Delaware Fightin´ Blue Hens", "Liberty Flames", competition=COLLEGE)
    result = fetch(api, game)[game.match_id]
    assert result.status == READY
    got = values(result)
    assert got["home.points_for"].value == 31.0 and got["home.points_for"].source == "espn-football:college-football"
    assert got["match.wind_mph"].source.startswith("open-meteo: Delaware Stadium")      # college stadium table, by home team
    groups = {dict(r.url.params).get("groups") for r in api.espn_requests() if r.url.path.endswith("/scoreboard")}
    assert groups == {"80", "81"}                            # FBS and FCS are both read and merged
    assert "college-football" in str(api.requests[0].url)


def test_a_college_team_espn_does_not_list_is_insufficient_and_the_analyst_may_fill_it():
    game = fixture("Idaho Vandals", "Liberty Flames", competition=COLLEGE)
    result = fetch(college_api(), game)[game.match_id]
    assert result.status == INSUFFICIENT and "Idaho Vandals was not found among the COLLEGE teams" in result.reasons[0]


def test_college_and_nfl_fixtures_in_one_slate_use_their_own_league_reads():
    nfl_api = Espn(stats=STATS)
    game = fixture("Buffalo Bills", "Chicago Bears")
    result = fetch(nfl_api, game, fixture("Ottawa Redblacks", "Hamilton Tiger-Cats", competition="Canada - CFL"))
    assert result[game.match_id].status == READY and len(result) == 2
    assert all("/nfl/" in str(r.url) for r in nfl_api.espn_requests())


# ================================================================ the college stadium table
def test_college_stadiums_are_looked_up_by_home_team_and_skipped_when_the_game_is_elsewhere():
    from predictions.adapters.weather.college_stadiums import COLLEGE_VENUES, college_stadium_for
    for _city, stadium in COLLEGE_VENUES.values():
        assert stadium.roof in ROOFS and -90 <= stadium.latitude <= 90 and -180 <= stadium.longitude <= 180
    assert college_stadium_for("Western Kentucky Hilltoppers", "Bowling Green").name.endswith("L.T. Smith Stadium")
    assert college_stadium_for("Arkansas State Red Wolves", None).roof == "open"       # no scheduled venue: the home stadium
    assert college_stadium_for("UTSA Roadrunners", "San Antonio").roof == "dome"
    assert college_stadium_for("Alabama State Hornets", "Montgomery") is not None
    assert college_stadium_for("Arkansas State Red Wolves", "Arlington") is None        # neutral site: wrong weather place
    assert college_stadium_for("Some Unlisted College", None) is None


def test_the_los_chankas_spelling_resolves_to_the_stats_source_name():
    from predictions.core.identity import resolve_team
    assert resolve_team("Los Chankas CYC", ["Atletico Grau", "Los Chankas"]).method == "alias"
