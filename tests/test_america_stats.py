"""Stories 6.3 and 6.4: the MLB and NHL stats sources behind the StatsSource port. Both are exercised through
httpx.MockTransport with FABRICATED API answers shaped like the real ones (probed 2026-09-30), plus a trimmed RECORDED
NHL play-by-play and schedule (tests/data/nhl_*_sample.json). No test makes a live call."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from predictions.adapters.mlb import stats as mlb_stats
from predictions.adapters.nhl import stats as nhl_stats
from predictions.core.inputs import BASEBALL_REGISTRY, HOCKEY_REGISTRY, check_inputs
from predictions.core.ports import StatsFetchError
from predictions.core.stats import AMBIGUOUS, INSUFFICIENT, READY
from predictions.core.types import Fixture, MatchId

DATA = Path(__file__).resolve().parent / "data"
NOW = datetime(2026, 9, 30, 8, 0, tzinfo=timezone.utc)
KICKOFF = datetime(2026, 9, 30, 18, 0, tzinfo=timezone.utc)


def run(coro):
    return asyncio.run(coro)


def fixture(sport, league, home, away, kickoff=KICKOFF):
    return Fixture(MatchId(sport, league, kickoff, home, away), league, home, away, kickoff, False, ())


def by_name(stats):
    return {i.name: i for i in stats.inputs}


# ================================================================ MLB
TEAMS = {144: "Atlanta Braves", 143: "Philadelphia Phillies", 117: "Houston Astros", 145: "Chicago White Sox"}
HOME_INNINGS = [1, 0, 2, 0, 1, 0, 1, 0, 0]   # 5 runs, 4 of them in innings 1 to 5
AWAY_INNINGS = [0, 1, 0, 0, 2, 0, 0, 0, 0]   # 3 runs, 3 in innings 1 to 5


def mlb_game(pk, day, home, away, hi=HOME_INNINGS, ai=AWAY_INNINGS, kind="R", state="Final", innings=None):
    when = f"2026-09-{day:02d}T23:05:00Z"
    line = [{"home": {"runs": h}, "away": {"runs": a}} for h, a in zip(hi, ai)] if innings is None else innings
    return {"gamePk": pk, "gameType": kind, "gameDate": when, "status": {"abstractGameState": state},
            "teams": {"home": {"team": {"id": home}, "score": sum(hi)}, "away": {"team": {"id": away}, "score": sum(ai)}},
            "linescore": {"innings": line}}


def mlb_history(extra=()):
    games, pk = [], 1
    for day in (20, 21, 22):
        for home, away in ((144, 143), (117, 145), (144, 117), (143, 145)):
            games.append(mlb_game(pk, day, home, away))
            pk += 1
    return games + list(extra)


def mlb_today(pitchers=(111, 222)):
    def game(home, away, hour, hp, ap):
        team = lambda tid, pid: {"team": {"id": tid}, **({"probablePitcher": {"id": pid}} if pid else {})}  # noqa: E731
        return {"gamePk": 900 + home, "gameType": "R", "gameDate": f"2026-09-30T{hour}:00Z",
                "status": {"abstractGameState": "Preview"}, "teams": {"home": team(home, hp), "away": team(away, ap)}}
    return [game(144, 143, "18:00", *pitchers), game(117, 145, "21:00", None, None)]


def mlb_people():
    def person(pid, runs, outs, starts, season="2026"):
        return {"id": pid, "stats": [{"group": {"displayName": "pitching"},
                                      "splits": [{"season": season, "stat": {"runs": runs, "outs": outs, "gamesStarted": starts}}]}]}
    return {"people": [person(111, 60, 540, 20), person(222, 90, 360, 20)]}


class MlbApi:
    def __init__(self, history=None, today=None, people=None, fail=None, teams=None):
        self.history = mlb_history() if history is None else history
        self.today = mlb_today() if today is None else today
        self.people = mlb_people() if people is None else people
        self.fail, self.teams, self.requests = fail, teams or TEAMS, []

    def handler(self, request: httpx.Request):
        self.requests.append(request)
        if self.fail:
            return httpx.Response(self.fail)
        path, params = request.url.path, dict(request.url.params)
        if path.endswith("/teams"):
            return httpx.Response(200, json={"teams": [{"id": i, "name": n} for i, n in self.teams.items()]})
        if path.endswith("/schedule"):
            games = self.history if params.get("hydrate") == "linescore" else self.today
            return httpx.Response(200, json={"dates": [{"games": games}]})
        if path.endswith("/people"):
            return httpx.Response(200, json=self.people)
        return httpx.Response(404)

    def source(self):
        client = httpx.AsyncClient(transport=httpx.MockTransport(self.handler))
        return mlb_stats.MlbStats(client, clock=lambda: NOW)


def mlb_fetch(api, *fixtures):
    return run(api.source().fetch_stats(list(fixtures)))


BRAVES = fixture("baseball", "USA - MLB", "Atlanta Braves", "Philadelphia Phillies")
ASTROS = fixture("baseball", "USA - MLB", "Houston Astros", "Chicago White Sox", KICKOFF + timedelta(hours=3))


def test_mlb_inputs_are_derived_from_the_games_with_provenance():
    stats = mlb_fetch(MlbApi(), BRAVES)[BRAVES.match_id]
    assert stats.status == READY
    got = by_name(stats)
    assert got["home.matches"].value == 6 and got["home.runs_for"].value == 5.0 and got["home.runs_against"].value == 3.0
    assert got["home.f5_runs_for"].value == 4.0 and got["home.f5_runs_against"].value == 3.0
    assert got["home.sp_runs_per9"].value == pytest.approx(3.0) and got["home.sp_starts"].value == 20
    assert got["away.sp_runs_per9"].value == pytest.approx(6.75)   # 90 runs in 120 innings
    assert got["away.runs_for"].value == 4.0 and got["away.runs_against"].value == 4.0   # Phillies: 3 games away (3-5), 3 at home (5-3)
    assert got["match.league_runs"].value == pytest.approx(4.0) and got["match.league_f5_runs"].value == pytest.approx(3.5)
    assert all(i.source == "mlb-statsapi" for i in stats.inputs)
    kept, dropped = check_inputs(stats.inputs, BASEBALL_REGISTRY)
    assert len(kept) == len(stats.inputs) and not dropped   # every derived input is inside the registry


def test_one_slate_reads_the_api_a_handful_of_times_not_once_per_game():
    api = MlbApi()
    mlb_fetch(api, BRAVES, ASTROS)
    assert len(api.requests) == 4 and all(r.method == "GET" for r in api.requests)


def test_only_mlb_is_covered_other_baseball_leagues_are_insufficient_data():
    kbo = fixture("baseball", "South Korea - KBO", "Doosan Bears", "LG Twins")
    stats = mlb_fetch(MlbApi(), kbo, BRAVES)
    assert stats[kbo.match_id].status == INSUFFICIENT and "no free stats source covers" in stats[kbo.match_id].reasons[0]
    assert stats[BRAVES.match_id].status == READY


def test_fewer_than_five_completed_games_is_insufficient_data():
    api = MlbApi(history=mlb_history()[:5])   # the Braves are in only 3 of these games
    stats = mlb_fetch(api, BRAVES)[BRAVES.match_id]
    assert stats.status == INSUFFICIENT and "missing required input" in stats.reasons[0]


def test_an_offseason_with_no_games_at_all_is_insufficient_data():
    stats = mlb_fetch(MlbApi(history=[]), BRAVES)[BRAVES.match_id]
    assert stats.status == INSUFFICIENT and "no completed games" in stats.reasons[0]


def test_spring_training_and_unfinished_games_are_ignored():
    extra = [mlb_game(50, 25, 144, 143, hi=[9] * 9, kind="S"), mlb_game(51, 26, 144, 143, hi=[9] * 9, state="Live")]
    got = by_name(mlb_fetch(MlbApi(history=mlb_history(extra)), BRAVES)[BRAVES.match_id])
    assert got["home.matches"].value == 6 and got["home.runs_for"].value == 5.0


def test_f5_inputs_are_absent_when_a_game_has_no_full_five_innings_of_linescore():
    short = mlb_game(60, 27, 144, 143, innings=[{"home": {"runs": 1}, "away": {"runs": 0}}] * 4)
    got = by_name(mlb_fetch(MlbApi(history=mlb_history([short])), BRAVES)[BRAVES.match_id])
    assert "home.f5_runs_for" not in got and got["home.runs_for"].value > 0


def test_a_missing_probable_pitcher_leaves_the_pitcher_inputs_out_but_the_game_is_still_ready():
    stats = mlb_fetch(MlbApi(today=mlb_today((None, None))), BRAVES)[BRAVES.match_id]
    got = by_name(stats)
    assert stats.status == READY and "home.sp_runs_per9" not in got and "away.sp_starts" not in got


def test_a_pitcher_without_a_season_line_is_left_out_not_guessed():
    people = {"people": [{"id": 111, "stats": []}, mlb_people()["people"][1]]}
    got = by_name(mlb_fetch(MlbApi(people=people), BRAVES)[BRAVES.match_id])
    assert "home.sp_runs_per9" not in got and "away.sp_runs_per9" in got


def test_an_unknown_team_is_insufficient_data():
    odd = fixture("baseball", "USA - MLB", "Nowhere Nine", "Philadelphia Phillies")
    stats = mlb_fetch(MlbApi(), odd)[odd.match_id]
    assert stats.status == INSUFFICIENT and "Nowhere Nine was not found" in stats.reasons[0]


def test_an_ambiguous_team_name_asks_and_is_never_guessed():
    api = MlbApi(teams={**TEAMS, 146: "Chicago Cubs"})
    ask = fixture("baseball", "USA - MLB", "Chicago", "Philadelphia Phillies")
    stats = mlb_fetch(api, ask)[ask.match_id]
    assert stats.status == AMBIGUOUS and set(stats.options) == {"Chicago White Sox", "Chicago Cubs"}


@pytest.mark.parametrize("failure", [500, 404])
def test_an_outage_is_a_stats_error_with_no_partial_result(failure):
    with pytest.raises(StatsFetchError) as error:
        mlb_fetch(MlbApi(fail=failure), BRAVES)
    assert "http" not in str(error.value).lower() or "HTTP" in str(error.value)


def test_an_unexpected_answer_shape_fails_loudly_instead_of_yielding_numbers():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"oops": True})))
    with pytest.raises(StatsFetchError):
        run(mlb_stats.MlbStats(client, clock=lambda: NOW).fetch_stats([BRAVES]))


# ================================================================ NHL
TEAM_ROWS = {6: ("Boston", "Bruins"), 14: ("Tampa Bay", "Lightning"), 2: ("New York", "Islanders"), 9: ("Ottawa", "Senators")}


def team_json(tid, score=0):
    place, common = TEAM_ROWS[tid]
    return {"id": tid, "placeName": {"default": place}, "commonName": {"default": common}, "abbrev": common[:3].upper(),
            "score": score}


def nhl_game(gid, day, home, away, kind=2, state="OFF"):
    return {"id": gid, "gameType": kind, "gameState": state, "startTimeUTC": f"2026-09-{day:02d}T23:00:00Z",
            "homeTeam": team_json(home), "awayTeam": team_json(away)}


def play(kind, period, team, x, y, side="left", goalie=99):
    details = {"xCoord": x, "yCoord": y, "eventOwnerTeamId": team, "zoneCode": "O"}
    if goalie is not None:
        details["goalieInNetId"] = goalie
    return {"typeDescKey": kind, "periodDescriptor": {"number": period, "periodType": "REG"},
            "homeTeamDefendingSide": side, "details": details}


def nhl_pbp(gid, home, away, kind=2, state="OFF"):
    """The home side attacks the right end (+89), the away side the left end (-89). Distances in feet from the net."""
    plays = [
        play("goal", 1, home, 80, 0),                 # 9 ft: worth 0.20
        play("shot-on-goal", 2, home, 64, 0),         # 25 ft: 0.06
        play("missed-shot", 3, home, 44, 0),          # 45 ft: 0.025
        play("blocked-shot", 1, home, 70, 0),         # never counted
        play("goal", 4, home, 80, 0),                 # overtime: never counted
        play("goal", 3, away, -80, 0),                # 9 ft: 0.20
        play("shot-on-goal", 1, away, -49, 0),        # 40 ft: 0.04
        play("goal", 3, away, -30, 0, goalie=None),   # empty net: a goal, worth no xG
    ]
    plays += [play("shot-on-goal", 2, home, 0, 0) for _ in range(4)]    # from centre ice: 89 ft, worth 0.012 each
    plays += [play("shot-on-goal", 2, away, 0, 0) for _ in range(3)]
    return {"id": gid, "gameType": kind, "gameState": state, "homeTeam": {"id": home}, "awayTeam": {"id": away},
            "plays": plays}


def nhl_history():
    games, gid = [], 2026020001
    for day in (20, 21, 22):
        for home, away in ((6, 14), (2, 9), (6, 2), (14, 9)):
            games.append(nhl_game(gid, day, home, away))
            gid += 1
    return games


class NhlApi:
    def __init__(self, games=None, pbp=None, fail=None):
        self.games = nhl_history() if games is None else games
        self.pbp, self.fail, self.requests = pbp or {}, fail, []

    def handler(self, request: httpx.Request):
        self.requests.append(request)
        if self.fail:
            return httpx.Response(self.fail)
        path = request.url.path
        if path.startswith("/v1/schedule/"):
            return httpx.Response(200, json={"gameWeek": [{"date": "x", "games": self.games}]})
        if "/play-by-play" in path:
            gid = int(path.split("/")[3])
            if gid in self.pbp:
                return httpx.Response(200, json=self.pbp[gid])
            game = next(g for g in self.games if g["id"] == gid)
            return httpx.Response(200, json=nhl_pbp(gid, game["homeTeam"]["id"], game["awayTeam"]["id"],
                                                    game["gameType"], game["gameState"]))
        return httpx.Response(404)

    def source(self):
        return nhl_stats.NhlStats(httpx.AsyncClient(transport=httpx.MockTransport(self.handler)), clock=lambda: NOW)


def nhl_fetch(api, *fixtures):
    return run(api.source().fetch_stats(list(fixtures)))


BRUINS = fixture("hockey", "USA - NHL", "Boston Bruins", "Tampa Bay Lightning")
ISLES = fixture("hockey", "USA - NHL", "New York Islanders", "Ottawa Senators", KICKOFF + timedelta(hours=1))


def test_the_xg_proxy_uses_only_distance_regulation_and_unblocked_attempts_and_is_rescaled_to_league_goals():
    stats = nhl_fetch(NhlApi(), BRUINS)[BRUINS.match_id]
    assert stats.status == READY
    got = by_name(stats)
    # per game: home 1 goal and 0.333 raw xG; away 2 goals (one on an empty net) and 0.276 raw xG; league 1.5 goals per team
    home_raw, away_raw = 0.285 + 4 * 0.012, 0.24 + 3 * 0.012
    factor = 3.0 / (home_raw + away_raw)
    assert got["match.league_goals"].value == pytest.approx(1.5)
    assert got["home.goals_for"].value == pytest.approx(1.0) and got["home.goals_against"].value == pytest.approx(2.0)
    assert got["home.xg_for"].value == pytest.approx(factor * home_raw) and got["home.xg_against"].value == pytest.approx(factor * away_raw)
    assert got["home.shots_for"].value == 6 and got["home.shots_against"].value == 6   # goals count as shots on goal
    assert got["home.p1_goals_for"].value == 1 and got["home.p1_goals_against"].value == 0
    assert got["home.p1_xg_for"].value == pytest.approx(factor * 0.20) and got["home.p1_xg_against"].value == pytest.approx(factor * 0.04)
    assert got["home.matches"].value == 6


def test_expected_goals_are_labeled_as_a_proxy_in_their_unit_and_every_input_has_its_source():
    got = by_name(nhl_fetch(NhlApi(), BRUINS)[BRUINS.match_id])
    assert "proxy" in got["home.xg_for"].unit and "proxy" in got["away.p1_xg_against"].unit
    assert all(i.source == "nhl-api" for i in got.values())
    kept, dropped = check_inputs(list(got.values()), HOCKEY_REGISTRY)
    assert len(kept) == len(got) and not dropped


def test_the_league_level_makes_the_proxy_average_equal_actual_goals():
    stats = nhl_fetch(NhlApi(), BRUINS, ISLES)
    got = by_name(stats[BRUINS.match_id])
    both = got["home.xg_for"].value + got["home.xg_against"].value
    assert both == pytest.approx(3.0)   # the two sides of every game sum to the league's 3 goals


def test_the_distance_table_and_geometry_of_the_proxy():
    assert nhl_stats.attempt_xg(nhl_stats.distance_to_net(80, 0, True)) == 0.20
    assert nhl_stats.attempt_xg(nhl_stats.distance_to_net(-80, 0, False)) == 0.20
    assert nhl_stats.attempt_xg(nhl_stats.distance_to_net(80, 0, False)) == 0.012   # a shot at the wrong end is far away
    assert nhl_stats.attempt_xg(40.0) == 0.04 and nhl_stats.attempt_xg(40.01) == 0.025


def test_only_nhl_is_covered_other_hockey_leagues_are_insufficient_data():
    shl = fixture("hockey", "Sweden - SHL", "Djurgardens IF", "Frolunda HC")
    stats = nhl_fetch(NhlApi(), shl, BRUINS)
    assert stats[shl.match_id].status == INSUFFICIENT and "no free stats source covers" in stats[shl.match_id].reasons[0]
    assert stats[BRUINS.match_id].status == READY


def test_preseason_and_unfinished_games_are_ignored():
    games = nhl_history() + [nhl_game(1, 25, 6, 14, kind=1), nhl_game(2, 26, 6, 14, state="LIVE")]
    api = NhlApi(games=games)
    got = by_name(nhl_fetch(api, BRUINS)[BRUINS.match_id])
    assert got["home.matches"].value == 6
    assert not [r for r in api.requests if r.url.path.endswith("/1/play-by-play") or "/2/play-by-play" in r.url.path]


def test_fewer_than_five_completed_games_is_insufficient_data():
    stats = nhl_fetch(NhlApi(games=nhl_history()[:6]), BRUINS)[BRUINS.match_id]
    assert stats.status == INSUFFICIENT and "missing required input" in stats.reasons[0]


def test_an_offseason_is_insufficient_data():
    stats = nhl_fetch(NhlApi(games=[]), BRUINS)[BRUINS.match_id]
    assert stats.status == INSUFFICIENT


def test_an_unknown_team_is_insufficient_data():
    odd = fixture("hockey", "USA - NHL", "Nowhere Ninjas", "Ottawa Senators")
    stats = nhl_fetch(NhlApi(), odd)[odd.match_id]
    assert stats.status == INSUFFICIENT and "was not found among the NHL teams" in stats.reasons[0]


def test_play_by_play_is_read_once_per_distinct_game_and_only_for_the_teams_needed():
    api = NhlApi()
    nhl_fetch(api, BRUINS)
    pbp = [r for r in api.requests if "/play-by-play" in r.url.path]
    assert len(pbp) == len({r.url.path for r in pbp}) and 0 < len(pbp) <= 12 and all(r.method == "GET" for r in api.requests)


@pytest.mark.parametrize("failure", [500, 503])
def test_an_outage_is_a_stats_error(failure):
    with pytest.raises(StatsFetchError):
        nhl_fetch(NhlApi(fail=failure), BRUINS)


def test_an_unexpected_answer_shape_fails_loudly():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"nope": 1})))
    with pytest.raises(StatsFetchError):
        run(nhl_stats.NhlStats(client, clock=lambda: NOW).fetch_stats([BRUINS]))


def test_the_recorded_play_by_play_parses_to_the_games_own_totals():
    payload = json.loads((DATA / "nhl_play_by_play_sample.json").read_text(encoding="utf-8"))
    facts = nhl_stats.parse_play_by_play(payload)
    assert facts.goals == {6: 1, 14: 2} and facts.shots == {6: 21, 14: 24}   # Boston 1 Tampa Bay 2, 21 and 24 shots on goal
    assert 1.5 < facts.xg_raw[6] < 4.5 and 1.5 < facts.xg_raw[14] < 4.5 and sum(facts.p1_goals.values()) == 0


def test_the_recorded_schedule_parses_to_finished_games_and_club_names():
    payload = json.loads((DATA / "nhl_schedule_sample.json").read_text(encoding="utf-8"))
    games, names = nhl_stats.parse_schedule(payload)
    assert len(games) == 18 and names[6] == "Boston Bruins" and names[14] == "Tampa Bay Lightning"


def test_an_unfinished_game_has_no_facts():
    assert nhl_stats.parse_play_by_play(nhl_pbp(1, 6, 14, state="LIVE")) is None
