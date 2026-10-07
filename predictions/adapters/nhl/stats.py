"""NHL team stats from the NHL's public web API (api-web.nhle.com), behind the StatsSource port (Story 6.4).

The API needs no key and no login. It publishes NO expected goals, so every "xg" value below is a PROXY built here from
shot locations (see "What is a proxy"). Every field is validated and a change in its shape fails loudly (StatsFetchError)
instead of yielding wrong numbers. No retries: a failed read is reported (AD-14). Custom headers are not sent.

Read-only GETs (probed 2026-09-30, see spec-6-4 Design Notes):
  * `v1/schedule/<date>`: the week that starts at <date>, every game with its final state, teams and score. Asked in
    7-day steps over the last LOOKBACK_DAYS days; the club names come from here (place name + common name).
  * `v1/gamecenter/<id>/play-by-play`: every shot attempt with its coordinates and every goal, per period, for the games
    in a team's last LAST_N. One read per distinct game (a slate needs at most a few dozen, CONCURRENCY at a time).

What is derived (the source string is `nhl-api`):
  * Window: a team's last LAST_N completed REGULAR-SEASON or PLAYOFF games found in the last LOOKBACK_DAYS days; preseason
    games are ignored. Fewer than 5 games (the registry minimum) is "insufficient data", so the offseason and a season's
    first days give no picks.
  * REGULATION only: goals and shots in periods 1 to 3. Overtime and shootout goals are ignored so the numbers match a
    model of regulation play (the engine adds overtime itself).
  * `goals_for` / `goals_against` and `p1_goals_for` / `p1_goals_against`: mean regulation goals, and first-period goals,
    per game over the window.
  * `shots_for` / `shots_against`: mean shots on goal in regulation (shots on goal plus goals).

What is a proxy (nothing here is a published expected-goals figure):
  * `xg_for` / `xg_against` / `p1_xg_for` / `p1_xg_against`. Each unblocked shot attempt (shot on goal, missed shot,
    goal) that has coordinates and a goalie in net is worth a probability that depends only on its DISTANCE to the net it
    attacks (DISTANCE_XG below: closer is worth more). Empty-net shots are worth nothing. The per-shot table is
    then rescaled by ONE factor so the league's proxy total over the sampled games equals the league's actual goals over
    the same games: the proxy therefore measures shot quality and volume relative to the league, not absolute
    finishing. Shot type, angle beyond distance, rebounds, rush and screens are NOT modelled.
Only "USA - NHL" is covered; every other hockey league is "insufficient data" (the analyst may fill it).
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable

import httpx

from predictions.core.identity import resolve_team
from predictions.core.inputs import Input
from predictions.core.ports import StatsFetchError
from predictions.core.stats import FixtureStats, ambiguous, assess, insufficient
from predictions.core.ttl_cache import STATS_CACHE, TTL_FORM_SECONDS, request_key
from predictions.core.types import Fixture, MatchId, normalize_team

NAME = "nhl-api"
BASE_URL = "https://api-web.nhle.com/v1"
TIMEOUT_SECONDS = 30.0
OVERALL_TIMEOUT_SECONDS = 240.0
CONCURRENCY = 4              # polite: at most this many play-by-play reads at once
LAST_N = 10                  # a team's inputs come from its last LAST_N completed games
LOOKBACK_DAYS = 28           # how far back the game log is searched; older games are never used
WEEK_DAYS = 7                # the schedule endpoint answers one week per date
COUNTED_GAME_TYPES = (2, 3)  # 1 preseason (ignored), 2 regular season, 3 playoffs
FINAL_STATES = ("OFF", "FINAL")
REGULATION_PERIODS = 3
NET_X = 89.0                 # the goal line sits this far from centre ice (feet); coordinates run -100 to 100
UNBLOCKED_TYPES = ("shot-on-goal", "missed-shot", "goal")
SHOT_ON_GOAL_TYPES = ("shot-on-goal", "goal")

# PROXY: probability of a goal from an unblocked attempt by distance to the net (feet), (upper bound, probability).
# Rounded shape of public shot-location models; the table is rescaled to the league's goals (see the module doc).
DISTANCE_XG = ((10.0, 0.20), (20.0, 0.11), (30.0, 0.06), (40.0, 0.04), (55.0, 0.025), (float("inf"), 0.012))

LEAGUES = frozenset({"usa nhl"})
# SportyBet's spelling of a team -> the API's (both normalized). Everything else matches by the identity resolver.
ALIASES: dict[str, str] = {
    "utah hockey club": "utah mammoth",
    "ny rangers": "new york rangers",
    "ny islanders": "new york islanders",
    "la kings": "los angeles kings",
}


@dataclass(frozen=True)
class FinishedGame:
    game_id: int
    when: datetime
    home_id: int
    away_id: int


@dataclass(frozen=True)
class GameFacts:
    """One game's regulation facts per team id: goals, shots on goal and proxy xG, all for the whole game and period 1."""
    goals: dict[int, int]
    shots: dict[int, int]
    xg_raw: dict[int, float]
    p1_goals: dict[int, int]
    p1_xg_raw: dict[int, float]


def _bad() -> StatsFetchError:
    return StatsFetchError("the NHL's stats service returned an unexpected response")


def _when(text) -> datetime:
    try:
        return datetime.strptime(str(text), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        raise _bad() from None


def _name(team) -> str:
    try:
        place, common = team["placeName"]["default"], team["commonName"]["default"]
    except (KeyError, TypeError):
        raise _bad() from None
    return f"{place} {common}"


def parse_schedule(payload) -> tuple[list[FinishedGame], dict[int, str]]:
    """(completed regular-season and playoff games, team id -> club name) of one schedule response."""
    if not isinstance(payload, dict) or not isinstance(payload.get("gameWeek"), list):
        raise _bad()
    games, names = [], {}
    for day in payload["gameWeek"]:
        if not isinstance(day, dict) or not isinstance(day.get("games"), list):
            raise _bad()
        for game in day["games"]:
            if not isinstance(game, dict):
                raise _bad()
            try:
                home, away = game["homeTeam"], game["awayTeam"]
                home_id, away_id = int(home["id"]), int(away["id"])
                names[home_id], names[away_id] = _name(home), _name(away)
                state, kind, game_id = game["gameState"], game["gameType"], int(game["id"])
                when = _when(game["startTimeUTC"])
            except (KeyError, TypeError, ValueError):
                raise _bad() from None
            if state in FINAL_STATES and kind in COUNTED_GAME_TYPES:
                games.append(FinishedGame(game_id, when, home_id, away_id))
    return games, names


def distance_to_net(x: float, y: float, attacking_right: bool) -> float:
    return math.hypot((NET_X if attacking_right else -NET_X) - x, y)


def attempt_xg(distance: float) -> float:
    for bound, value in DISTANCE_XG:
        if distance <= bound:
            return value
    return DISTANCE_XG[-1][1]  # unreachable: the last bound is infinite


def parse_play_by_play(payload) -> GameFacts | None:
    """Regulation goals, shots on goal and raw proxy xG per team, or None for a game that is not a completed counted game.
    A play without the fields it needs is skipped for the xG proxy (a goal is always counted as a goal)."""
    if not isinstance(payload, dict) or not isinstance(payload.get("plays"), list):
        raise _bad()
    try:
        home_id, away_id = int(payload["homeTeam"]["id"]), int(payload["awayTeam"]["id"])
    except (KeyError, TypeError, ValueError):
        raise _bad() from None
    if payload.get("gameState") not in FINAL_STATES or payload.get("gameType") not in COUNTED_GAME_TYPES:
        return None
    goals = {home_id: 0, away_id: 0}
    shots = {home_id: 0, away_id: 0}
    xg = {home_id: 0.0, away_id: 0.0}
    p1_goals = {home_id: 0, away_id: 0}
    p1_xg = {home_id: 0.0, away_id: 0.0}
    for play in payload["plays"]:
        if not isinstance(play, dict):
            raise _bad()
        kind = play.get("typeDescKey")
        if kind not in UNBLOCKED_TYPES:
            continue
        period = (play.get("periodDescriptor") or {}).get("number")
        details = play.get("details") or {}
        team = details.get("eventOwnerTeamId")
        if not isinstance(period, int) or not 1 <= period <= REGULATION_PERIODS or team not in goals:
            continue
        if kind == "goal":
            goals[team] += 1
            if period == 1:
                p1_goals[team] += 1
        if kind in SHOT_ON_GOAL_TYPES:
            shots[team] += 1
        x, y, side = details.get("xCoord"), details.get("yCoord"), play.get("homeTeamDefendingSide")
        if (not isinstance(x, (int, float)) or not isinstance(y, (int, float)) or side not in ("left", "right")
                or details.get("goalieInNetId") is None):
            continue  # no location or an empty net: no proxy xG for this attempt
        # The home side attacks the end it does not defend; the away side attacks the end the home side defends.
        attacking_right = (side == "left") if team == home_id else (side == "right")
        value = attempt_xg(distance_to_net(float(x), float(y), attacking_right))
        xg[team] += value
        if period == 1:
            p1_xg[team] += value
    return GameFacts(goals, shots, xg, p1_goals, p1_xg)


def window_of(games: list[FinishedGame], team_id: int) -> list[FinishedGame]:
    own = [g for g in games if team_id in (g.home_id, g.away_id)]
    return sorted(own, key=lambda g: g.when, reverse=True)[:LAST_N]


def league_factor(facts: dict[int, GameFacts]) -> float | None:
    """The one rescaling factor of the xG proxy: league goals over league raw proxy total (both regulation)."""
    goals = sum(sum(f.goals.values()) for f in facts.values())
    raw = sum(sum(f.xg_raw.values()) for f in facts.values())
    return goals / raw if goals > 0 and raw > 0 else None


def league_inputs(facts: dict[int, GameFacts], source: str) -> list[Input]:
    team_games = 2 * len(facts)
    if not team_games:
        return []
    inputs = [Input("match.league_goals", sum(sum(f.goals.values()) for f in facts.values()) / team_games,
                    "goals per team per game", source)]
    p1 = sum(sum(f.p1_goals.values()) for f in facts.values())
    inputs.append(Input("match.league_p1_goals", p1 / team_games, "1st-period goals per team per game", source))
    return inputs


def team_inputs(side: str, team_id: int, window: list[FinishedGame], facts: dict[int, GameFacts], factor: float,
                source: str) -> list[Input]:
    rows = []
    for game in window:
        found = facts.get(game.game_id)
        if found is None:
            continue
        other = game.away_id if game.home_id == team_id else game.home_id
        rows.append((found, team_id, other))
    n = len(rows)
    if n == 0:
        return []

    def mean(pick) -> float:
        return sum(pick(f, me, opp) for f, me, opp in rows) / n

    xg_unit = "expected goals per game (proxy)"
    p1_xg_unit = "1st-period expected goals per game (proxy)"
    return [
        Input(f"{side}.matches", float(n), "games", source),
        Input(f"{side}.goals_for", mean(lambda f, me, opp: f.goals[me]), "goals per game", source),
        Input(f"{side}.goals_against", mean(lambda f, me, opp: f.goals[opp]), "goals per game", source),
        Input(f"{side}.xg_for", factor * mean(lambda f, me, opp: f.xg_raw[me]), xg_unit, source),
        Input(f"{side}.xg_against", factor * mean(lambda f, me, opp: f.xg_raw[opp]), xg_unit, source),
        Input(f"{side}.shots_for", mean(lambda f, me, opp: f.shots[me]), "shots on goal per game", source),
        Input(f"{side}.shots_against", mean(lambda f, me, opp: f.shots[opp]), "shots on goal per game", source),
        Input(f"{side}.p1_goals_for", mean(lambda f, me, opp: f.p1_goals[me]), "1st-period goals per game", source),
        Input(f"{side}.p1_goals_against", mean(lambda f, me, opp: f.p1_goals[opp]), "1st-period goals per game", source),
        Input(f"{side}.p1_xg_for", factor * mean(lambda f, me, opp: f.p1_xg_raw[me]), p1_xg_unit, source),
        Input(f"{side}.p1_xg_against", factor * mean(lambda f, me, opp: f.p1_xg_raw[opp]), p1_xg_unit, source),
    ]


class NhlStats:
    """StatsSource for hockey backed by the NHL's public web API. `clock` is injectable for tests."""

    def __init__(self, client: httpx.AsyncClient | None = None, clock: Callable[[], datetime] | None = None):
        self._client = client
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def fetch_stats(self, fixtures: list[Fixture]) -> dict[MatchId, FixtureStats]:
        try:
            if self._client is not None:
                return await asyncio.wait_for(self._run(self._client, fixtures), OVERALL_TIMEOUT_SECONDS)
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
                return await asyncio.wait_for(self._run(client, fixtures), OVERALL_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raise StatsFetchError("the NHL's stats service took too long to answer") from None

    async def _get(self, client: httpx.AsyncClient, path: str):
        """One GET, remembered for a few hours (core/ttl_cache.py): schedules and finished games only, never an error."""
        return await STATS_CACHE.get_or_fetch(request_key("nhl", path), TTL_FORM_SECONDS, lambda: self._read(client, path))

    async def _read(self, client: httpx.AsyncClient, path: str):
        try:
            response = await client.get(BASE_URL + path)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            raise StatsFetchError("the NHL's stats service did not answer in time") from None
        except httpx.HTTPStatusError as error:
            raise StatsFetchError(f"the NHL's stats service returned HTTP {error.response.status_code}") from None
        except (httpx.HTTPError, ValueError):
            raise StatsFetchError("the NHL's stats service could not be reached or gave unreadable data") from None

    async def _run(self, client: httpx.AsyncClient, fixtures: list[Fixture]) -> dict[MatchId, FixtureStats]:
        results: dict[MatchId, FixtureStats] = {}
        covered = []
        for fixture in fixtures:
            if fixture.match_id.sport != "hockey" or normalize_team(fixture.competition) not in LEAGUES:
                results[fixture.match_id] = insufficient(
                    fixture.match_id, f"no free stats source covers {fixture.competition} (NHL API: NHL only)")
            else:
                covered.append(fixture)
        if not covered:
            return results
        now = self._clock().astimezone(timezone.utc)
        games: dict[int, FinishedGame] = {}
        names: dict[int, str] = {}
        first = now.date() - timedelta(days=LOOKBACK_DAYS)
        start = first
        while start <= now.date():
            found, found_names = parse_schedule(await self._get(client, f"/schedule/{start.isoformat()}"))
            games.update({g.game_id: g for g in found})
            names.update(found_names)
            start += timedelta(days=WEEK_DAYS)
        history = [g for g in games.values() if now - timedelta(days=LOOKBACK_DAYS) <= g.when <= now]
        by_name = {v: k for k, v in names.items()}
        wanted, questions = {}, {}
        for fixture in covered:
            ids = []
            for side_name in (fixture.home, fixture.away):
                found = resolve_team(side_name, names.values(), ALIASES)
                if found.status == "matched":
                    ids.append(by_name[found.team])
                elif found.status == "ambiguous":
                    questions[fixture.match_id] = (f"Which team is '{side_name}'? " + " or ".join(found.matches) + "?",
                                                   found.matches)
                    break
                else:
                    results[fixture.match_id] = insufficient(fixture.match_id,
                                                             f"{side_name} was not found among the NHL teams")
                    break
            else:
                wanted[fixture.match_id] = (ids[0], ids[1])
        for match_id, (question, options) in questions.items():
            results[match_id] = ambiguous(match_id, question, options)
        needed = sorted({g.game_id for pair in wanted.values() for team in pair for g in window_of(history, team)})
        facts = await self._facts(client, needed)
        factor = league_factor(facts)
        league = league_inputs(facts, NAME)
        for match_id, (home_id, away_id) in wanted.items():
            results[match_id] = self._assess(match_id, names, history, facts, factor, home_id, away_id, league)
        return results

    async def _facts(self, client: httpx.AsyncClient, game_ids: list[int]) -> dict[int, GameFacts]:
        gate = asyncio.Semaphore(CONCURRENCY)

        async def one(game_id: int):
            async with gate:
                return game_id, parse_play_by_play(await self._get(client, f"/gamecenter/{game_id}/play-by-play"))

        found = await asyncio.gather(*(one(g) for g in game_ids))
        return {game_id: f for game_id, f in found if f is not None}

    @staticmethod
    def _assess(match_id, names, history, facts, factor, home_id, away_id, league) -> FixtureStats:
        if factor is None:
            return insufficient(match_id, "no completed NHL games in the last "
                                          f"{LOOKBACK_DAYS} days to measure the league level")
        raw: list[Input] = []
        problems: list[str] = []
        for side, team in (("home", home_id), ("away", away_id)):
            window = window_of(history, team)
            inputs = team_inputs(side, team, window, facts, factor, NAME)
            if not inputs:
                problems.append(f"{names[team]} has no completed games in the last {LOOKBACK_DAYS} days")
            else:
                raw.extend(inputs)
        if problems:
            return insufficient(match_id, *problems)
        raw.extend(league)
        return assess(match_id, raw)
