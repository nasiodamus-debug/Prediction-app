"""Story 6.1: the ESPN basketball stats adapter, from recorded samples of ESPN's public API (tests/data/espn_nba_*.json,
trimmed from real responses probed on 2026-09-30). No test makes a live call."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from predictions.adapters.espn import stats as espn
from predictions.adapters.espn.stats import EspnBasketballStats, parse_scoreboard, parse_team_statistics, parse_teams
from predictions.core.ports import StatsFetchError
from predictions.core.stats import INSUFFICIENT, READY
from predictions.core.types import Fixture, MatchId

DATA = Path(__file__).parent / "data"
TEAMS = json.loads((DATA / "espn_nba_teams_sample.json").read_text(encoding="utf-8"))
BOARD = json.loads((DATA / "espn_nba_scoreboard_sample.json").read_text(encoding="utf-8"))
STATS = json.loads((DATA / "espn_nba_team_statistics_sample.json").read_text(encoding="utf-8"))
NOW = datetime(2026, 4, 13, 12, 0, tzinfo=timezone.utc)  # the day after the sample's last game
KICKOFF = datetime(2026, 4, 14, 23, 0, tzinfo=timezone.utc)


def fixture(home="Detroit Pistons", away="Boston Celtics", competition="USA - NBA", sport="basketball", kickoff=KICKOFF):
    return Fixture(MatchId(sport, competition, kickoff, home, away), competition, home, away, kickoff, False)


class Api:
    """A fake of the parts of ESPN's API the adapter uses, served from the recorded samples."""

    def __init__(self, board=BOARD, teams=TEAMS, stats=STATS, fail=None):
        self.board, self.teams, self.stats, self.fail = board, teams, stats, fail
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        if self.fail:
            return self.fail(request)
        path = request.url.path
        if path.endswith("/teams"):
            return httpx.Response(200, json=self.teams)
        if path.endswith("/scoreboard"):
            return httpx.Response(200, json=self.board.get(request.url.params["dates"], {"events": []}))
        if path.endswith("/statistics"):
            team = path.split("/")[-2]
            return httpx.Response(200, json=self.stats[team]) if team in self.stats else httpx.Response(404)
        return httpx.Response(404)


def fetch(fixtures, api=None, now=NOW):
    api = api or Api()

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(api)) as client:
            return await EspnBasketballStats(client, clock=lambda: now).fetch_stats(fixtures)

    return asyncio.run(go()), api


def games_of(team_id, until=NOW):
    """The team's completed regular-season games in the sample, newest first, as (points, opp points, fga, fta, 1H)."""
    rows = []
    for day, payload in BOARD.items():
        for game in parse_scoreboard(payload):
            if game.when > until:
                continue
            for own, opp in ((game.home, game.away), (game.away, game.home)):
                if own.team_id == team_id:
                    rows.append((game.when, own, opp))
    return sorted(rows, key=lambda r: r[0], reverse=True)


# ---------------------------------------------------------------- parsing
def test_scoreboard_keeps_only_completed_regular_season_and_playoff_games():
    payload = {"events": [
        {"date": "2026-04-01T00:00Z", "season": {"year": 2026, "type": 2}, "status": {"type": {"completed": True}},
         "competitions": [{"competitors": [
             {"homeAway": "home", "team": {"id": "1"}, "score": "100", "linescores": [{"period": 1, "value": 25.0}, {"period": 2, "value": 25.0}],
              "statistics": [{"name": "fieldGoalsAttempted", "displayValue": "80"}, {"name": "freeThrowsAttempted", "displayValue": "20"}]},
             {"homeAway": "away", "team": {"id": "2"}, "score": "90", "linescores": [], "statistics": []}]}]},
        {"date": "2026-04-01T00:00Z", "season": {"year": 2026, "type": 1}, "status": {"type": {"completed": True}},
         "competitions": [{"competitors": []}]},                      # preseason: ignored
        {"date": "2026-04-02T00:00Z", "season": {"year": 2026, "type": 2}, "status": {"type": {"completed": False}},
         "competitions": [{"competitors": []}]},                      # not finished: ignored
    ]}
    (game,) = parse_scoreboard(payload)
    assert game.home.points == 100 and game.home.fga == 80 and game.home.fta == 20 and game.home.fh_points == 50
    assert game.away.fga is None and game.away.fh_points is None


@pytest.mark.parametrize("bad", [None, [], {}, {"events": "x"}, {"events": [1]}])
def test_a_payload_that_is_not_a_scoreboard_is_an_error(bad):
    with pytest.raises(StatsFetchError):
        parse_scoreboard(bad)


def test_teams_and_statistics_are_read_or_refused():
    assert parse_teams(TEAMS)["8"] == "Detroit Pistons"
    with pytest.raises(StatsFetchError):
        parse_teams({"sports": []})
    assert parse_team_statistics(STATS["8"], 2026) == pytest.approx((13.085366, 14.182927))
    assert parse_team_statistics(STATS["8"], 2025) is None  # another season's numbers are never used
    assert parse_team_statistics({"requestedSeason": {"year": 2026}, "results": {}}, 2026) is None


# ---------------------------------------------------------------- the inputs, checked against an independent calculation
def test_a_covered_fixture_gets_the_documented_inputs_with_provenance():
    result, _ = fetch([fixture()])
    stats = result[fixture().match_id]
    assert stats.status == READY
    inputs = {i.name: i for i in stats.inputs}
    assert all(i.source == "espn:nba" for i in inputs.values())
    for side, team_id in (("home", "8"), ("away", "2")):
        window = games_of(team_id)[: espn.LAST_N]
        oreb, tov = parse_team_statistics(STATS[team_id], 2026)
        poss = [own.fga + 0.44 * own.fta - oreb + tov for _, own, _ in window]
        points, against = sum(o.points for _, o, _ in window), sum(p.points for _, _, p in window)
        assert inputs[f"{side}.matches"].value == 10
        assert inputs[f"{side}.ortg"].value == pytest.approx(100 * points / sum(poss))
        assert inputs[f"{side}.drtg"].value == pytest.approx(100 * against / sum(poss))
        assert inputs[f"{side}.pace"].value == pytest.approx(sum(poss) / 10)
        assert inputs[f"{side}.points_for"].value == pytest.approx(points / 10)
        assert inputs[f"{side}.fh_points_for"].value == pytest.approx(sum(o.fh_points for _, o, _ in window) / 10)
        assert inputs[f"{side}.fh_points_against"].value == pytest.approx(sum(p.fh_points for _, _, p in window) / 10)
    assert 90 < inputs["match.league_pace"].value < 105
    assert inputs["home.ortg"].unit == "points per 100 possessions" and inputs["home.pace"].unit == "possessions per game"


def test_only_the_last_ten_games_before_the_request_count():
    result, _ = fetch([fixture()])
    inputs = {i.name: i for i in result[fixture().match_id].inputs}
    older = games_of("8")[espn.LAST_N:]
    assert older  # the sample holds more than ten games for the team, and they are not used
    assert inputs["home.matches"].value == 10


def test_a_game_after_the_request_time_is_not_used():
    early = datetime(2026, 4, 3, 12, 0, tzinfo=timezone.utc)
    f = fixture(kickoff=early + timedelta(days=1))
    result, _ = fetch([f], now=early)
    inputs = {i.name: i for i in result[f.match_id].inputs}
    expected = min(espn.LAST_N, len(games_of("8", until=early)))
    assert expected >= 5 and inputs["home.matches"].value == expected
    assert max(g[0] for g in games_of("8", until=early)) <= early  # nothing from the future entered the window


# ---------------------------------------------------------------- insufficient data, never guessed
def test_no_recent_games_is_insufficient_data():
    offseason = NOW + timedelta(days=200)
    f = fixture(kickoff=offseason + timedelta(days=1))
    result, api = fetch([f], now=offseason)
    stats = result[f.match_id]
    assert stats.status == INSUFFICIENT and "no completed games" in stats.reasons[0]
    scoreboards = [r for r in api.requests if r.url.path.endswith("/scoreboard")]
    assert len(scoreboards) <= espn.LOOKBACK_DAYS  # the search stops at the lookback limit


def test_preseason_games_are_ignored_so_the_offseason_gives_no_picks():
    preseason = {"events": [{"date": "2026-10-01T00:00Z", "season": {"year": 2027, "type": 1},
                             "status": {"type": {"completed": True}}, "competitions": [{"competitors": []}]}]}
    board = {f"202610{d:02d}": preseason for d in range(1, 10)}
    now = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)
    f = fixture(kickoff=now + timedelta(days=1))
    result, _ = fetch([f], api=Api(board=board), now=now)
    assert result[f.match_id].status == INSUFFICIENT


def test_fewer_than_five_games_is_insufficient_data():
    few = {}
    for day, payload in BOARD.items():
        few[day] = {"events": [e for e in payload["events"]
                               if any(c["team"]["id"] in ("8", "2") for c in e["competitions"][0]["competitors"])]}
    keep = {d: p for d, p in sorted(few.items(), reverse=True)[:4]}  # only the four latest days
    result, _ = fetch([fixture()], api=Api(board=keep))
    stats = result[fixture().match_id]
    assert stats.status == INSUFFICIENT and any("matches" in r for r in stats.reasons)


def test_a_league_no_free_source_covers_is_insufficient_data():
    f = fixture("Panathinaikos BC", "ASVEL Lyon-Villeurbanne", competition="International - Euroleague")
    result, api = fetch([f])
    assert result[f.match_id].status == INSUFFICIENT and "no free stats source covers" in result[f.match_id].reasons[0]
    assert api.requests == []  # nothing was asked for a league that is not covered


def test_an_unknown_team_is_insufficient_data():
    f = fixture("Nowhere Nomads", "Boston Celtics")
    result, _ = fetch([f])
    assert result[f.match_id].status == INSUFFICIENT and "Nowhere Nomads" in result[f.match_id].reasons[0]


def test_missing_season_statistics_is_insufficient_data():
    api = Api(stats={"2": STATS["2"]})  # no statistics for team 8: 404
    result, _ = fetch([fixture()], api=api)
    stats = result[fixture().match_id]
    assert stats.status == INSUFFICIENT and "rebounds and turnovers" in stats.reasons[0]


def test_sportybet_spelling_of_the_lakers_is_matched_by_alias():
    names = parse_teams(TEAMS).values()
    assert espn.resolve_team("LA Lakers", names, espn.ALIASES).team == "Los Angeles Lakers"
    assert espn.resolve_team("Los Angeles Lakers", names, espn.ALIASES).team == "Los Angeles Lakers"


# ---------------------------------------------------------------- failures are reported, not retried or partial
@pytest.mark.parametrize("status", [403, 429, 500])
def test_an_http_error_is_a_stats_fetch_error_and_is_not_retried(status):
    api = Api(fail=lambda request: httpx.Response(status))
    with pytest.raises(StatsFetchError) as caught:
        fetch([fixture()], api=api)
    assert str(status) in str(caught.value) and "http" not in str(caught.value).lower().replace("http " + str(status), "")
    assert len(api.requests) == 1


def test_a_timeout_and_unreadable_data_are_stats_fetch_errors():
    def timeout(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(StatsFetchError):
        fetch([fixture()], api=Api(fail=timeout))
    with pytest.raises(StatsFetchError):
        fetch([fixture()], api=Api(fail=lambda r: httpx.Response(200, text="<html>not json</html>")))


def test_requests_are_polite_read_only_gets_with_no_custom_headers_and_bounded_concurrency():
    result, api = fetch([fixture()])
    assert all(r.method == "GET" for r in api.requests)
    assert all("user-agent" not in {k.lower() for k in r.headers} or r.headers["user-agent"].startswith("python-httpx")
               for r in api.requests)  # ESPN answers 403 to a custom User-Agent: the library's own is the only one used
    assert espn.CONCURRENCY <= 5
    assert len([r for r in api.requests if r.url.path.endswith("/scoreboard")]) <= 30  # ten games take about 20 days; it stops once both teams have them


def test_the_adapter_stops_searching_days_once_every_team_has_ten_games():
    _, api = fetch([fixture()])
    days = sorted(r.url.params["dates"] for r in api.requests if r.url.path.endswith("/scoreboard"))
    assert len(days) < espn.LOOKBACK_DAYS


def test_two_fixtures_share_one_game_log():
    f1, f2 = fixture(), fixture("Orlando Magic", "Atlanta Hawks")
    result, api = fetch([f1, f2])
    assert result[f1.match_id].status == READY and result[f2.match_id].status in (READY, INSUFFICIENT)
    teams_calls = [r for r in api.requests if r.url.path.endswith("/teams")]
    assert len(teams_calls) == 1
