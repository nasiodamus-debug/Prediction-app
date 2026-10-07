"""MLB team and starting-pitcher stats from MLB's official public Stats API (statsapi.mlb.com), behind the StatsSource
port (Story 6.3). The API needs no key and no login. Every field is validated, and a change in its shape fails loudly
(StatsFetchError) instead of yielding wrong numbers. No retries: a failed read is reported (AD-14).

Three read-only GETs cover a whole slate (probed 2026-09-30, see spec-6-3 Design Notes):
  * `teams?sportId=1`: the 30 club names and ids (SportyBet's names are matched to these by the identity resolver);
  * `schedule?sportId=1&startDate&endDate&hydrate=linescore`: every game of the last LOOKBACK_DAYS days with its final
    score and the runs of every inning, so team run scoring, run prevention and the first five innings (F5) are DERIVED
    here from the games themselves (the API has no ready-made F5 splits);
  * `schedule?...&hydrate=probablePitcher` for the fixture dates, then `people?personIds&hydrate=stats(...)` for those
    pitchers' season line.

What is derived (nothing is invented; every value has the source `mlb-statsapi`):
  * Window: a team's last LAST_N completed games of the CURRENT season (regular season or postseason; spring training and
    exhibitions are ignored), found in the last LOOKBACK_DAYS days. Fewer than 5 games (the registry minimum) is
    "insufficient data", so a season's first days and the offseason give no picks.
  * `runs_for` / `runs_against` = mean runs scored / allowed per game over the window (extra innings included).
  * `f5_runs_for` / `f5_runs_against` = mean runs in innings 1 to 5, only when EVERY game of the window has at least five
    innings of linescore (a game ended earlier is missing them, and then the F5 inputs are simply absent).
  * `sp_runs_per9` = the probable starter's season runs allowed (earned or not) per nine innings = 9 * runs / (outs / 3);
    `sp_starts` = his season starts. A pitcher without a season line, or a game without a probable pitcher, has none.
  * `match.league_runs` / `match.league_f5_runs` = the mean over all teams' games in the same window: the level the
    model measures teams and starters against.
Only MLB is covered ("USA - MLB"); KBO, NPB, CPBL and every other league is "insufficient data" (the analyst may fill it).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Callable

import httpx

from predictions.core.identity import resolve_team
from predictions.core.inputs import Input
from predictions.core.ports import StatsFetchError
from predictions.core.stats import FixtureStats, ambiguous, assess, insufficient
from predictions.core.ttl_cache import STATS_CACHE, TTL_FORM_SECONDS, TTL_WEATHER_SECONDS, request_key
from predictions.core.types import Fixture, MatchId, normalize_team

NAME = "mlb-statsapi"
BASE_URL = "https://statsapi.mlb.com/api/v1"
TIMEOUT_SECONDS = 30.0
OVERALL_TIMEOUT_SECONDS = 120.0
LAST_N = 10                 # a team's inputs come from its last LAST_N completed games
LOOKBACK_DAYS = 21          # how far back the game log is searched; older games are never used
F5_INNINGS = 5
PITCHER_WINDOW_HOURS = 8    # the API's first pitch may differ from SportyBet's kickoff by this much
COUNTED_GAME_TYPES = ("R", "F", "D", "L", "W")   # regular season, wild card, division, league championship, World Series

# SportyBet "<country> - <league>" (normalized) -> covered. Anything else is not covered.
LEAGUES = frozenset({"usa mlb"})
# SportyBet's spelling of a team -> the API's (both normalized). Everything else matches by the identity resolver.
ALIASES: dict[str, str] = {
    "oakland athletics": "athletics",
    "sacramento athletics": "athletics",
    "arizona d backs": "arizona diamondbacks",
    "chi white sox": "chicago white sox",
    "la dodgers": "los angeles dodgers",
    "la angels": "los angeles angels",
}


@dataclass(frozen=True)
class PlayedGame:
    game_pk: int
    when: datetime          # UTC first pitch
    home_id: int
    away_id: int
    home_runs: int
    away_runs: int
    home_f5: int | None     # runs in innings 1 to 5, None when the game has fewer innings of linescore
    away_f5: int | None


@dataclass(frozen=True)
class Probables:
    when: datetime
    home_id: int
    away_id: int
    home_pitcher: int | None
    away_pitcher: int | None


@dataclass(frozen=True)
class PitcherLine:
    runs_per9: float
    starts: int


def _bad() -> StatsFetchError:
    return StatsFetchError("MLB's stats service returned an unexpected response")


def _int(value) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number == int(number) and number >= 0 else None


def _when(text) -> datetime:
    try:
        return datetime.strptime(str(text), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        raise _bad() from None


def parse_teams(payload) -> dict[int, str]:
    """team id -> club name."""
    try:
        found = {int(t["id"]): str(t["name"]) for t in payload["teams"]}
    except (KeyError, TypeError, ValueError):
        raise _bad() from None
    if not found:
        raise _bad()
    return found


def _games_of(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("dates"), list):
        raise _bad()
    for day in payload["dates"]:
        if not isinstance(day, dict) or not isinstance(day.get("games"), list):
            raise _bad()
        yield from day["games"]


def _f5(linescore, side: str) -> int | None:
    innings = linescore.get("innings") if isinstance(linescore, dict) else None
    if not isinstance(innings, list) or len(innings) < F5_INNINGS:
        return None
    total = 0
    for inning in innings[:F5_INNINGS]:
        runs = _int(((inning or {}).get(side) or {}).get("runs"))
        if runs is None:
            return None
        total += runs
    return total


def parse_history(payload) -> list[PlayedGame]:
    """The completed regular-season and postseason games of a schedule response. A game that is not final, is not a
    counted type, or lacks a score is left out; a payload that is not a schedule is an error."""
    games: dict[int, PlayedGame] = {}
    for game in _games_of(payload):
        if not isinstance(game, dict):
            raise _bad()
        status = game.get("status") or {}
        if status.get("abstractGameState") != "Final" or game.get("gameType") not in COUNTED_GAME_TYPES:
            continue
        teams = game.get("teams") or {}
        try:
            home, away = teams["home"], teams["away"]
            pk = int(game["gamePk"])
            home_id, away_id = int(home["team"]["id"]), int(away["team"]["id"])
        except (KeyError, TypeError, ValueError):
            raise _bad() from None
        home_runs, away_runs = _int(home.get("score")), _int(away.get("score"))
        if home_runs is None or away_runs is None or home_runs == away_runs:
            continue  # no score, or a game that did not finish (a tie is not a result in baseball)
        line = game.get("linescore")
        games[pk] = PlayedGame(pk, _when(game.get("gameDate")), home_id, away_id, home_runs, away_runs,
                               _f5(line, "home"), _f5(line, "away"))
    return sorted(games.values(), key=lambda g: g.when)


def parse_probables(payload) -> list[Probables]:
    """Scheduled games with their probable pitchers (either may be missing)."""
    found = []
    for game in _games_of(payload):
        if not isinstance(game, dict) or (game.get("status") or {}).get("abstractGameState") == "Final":
            continue
        teams = game.get("teams") or {}
        try:
            home, away = teams["home"], teams["away"]
            home_id, away_id = int(home["team"]["id"]), int(away["team"]["id"])
        except (KeyError, TypeError, ValueError):
            raise _bad() from None
        found.append(Probables(_when(game.get("gameDate")), home_id, away_id,
                               _int((home.get("probablePitcher") or {}).get("id")),
                               _int((away.get("probablePitcher") or {}).get("id"))))
    return found


def parse_pitchers(payload, season: int) -> dict[int, PitcherLine]:
    """pitcher id -> season runs per nine innings and starts. A pitcher with no season line is absent (never guessed)."""
    if not isinstance(payload, dict) or not isinstance(payload.get("people"), list):
        raise _bad()
    found: dict[int, PitcherLine] = {}
    for person in payload["people"]:
        pid = _int((person or {}).get("id"))
        for block in (person or {}).get("stats") or []:
            if (block.get("group") or {}).get("displayName") != "pitching":
                continue
            for split in block.get("splits") or []:
                if str(split.get("season")) != str(season):
                    continue
                stat = split.get("stat") or {}
                runs, outs, starts = _int(stat.get("runs")), _int(stat.get("outs")), _int(stat.get("gamesStarted"))
                if pid is not None and runs is not None and outs and starts is not None:
                    found[pid] = PitcherLine(9.0 * runs / (outs / 3.0), starts)
    return found


def team_window(games: list[PlayedGame], team_id: int) -> list[tuple[PlayedGame, int, int, int | None, int | None]]:
    """A team's games newest first, as (game, runs for, runs against, F5 for, F5 against), at most LAST_N."""
    own = []
    for g in games:
        if g.home_id == team_id:
            own.append((g, g.home_runs, g.away_runs, g.home_f5, g.away_f5))
        elif g.away_id == team_id:
            own.append((g, g.away_runs, g.home_runs, g.away_f5, g.home_f5))
    return sorted(own, key=lambda x: x[0].when, reverse=True)[:LAST_N]


def team_inputs(side: str, window, source: str) -> list[Input]:
    n = len(window)
    if n == 0:
        return []
    inputs = [
        Input(f"{side}.matches", float(n), "games", source),
        Input(f"{side}.runs_for", sum(w[1] for w in window) / n, "runs per game", source),
        Input(f"{side}.runs_against", sum(w[2] for w in window) / n, "runs per game", source),
    ]
    if all(w[3] is not None and w[4] is not None for w in window):
        inputs.append(Input(f"{side}.f5_runs_for", sum(w[3] for w in window) / n, "runs per 5 innings", source))
        inputs.append(Input(f"{side}.f5_runs_against", sum(w[4] for w in window) / n, "runs per 5 innings", source))
    return inputs


def league_inputs(games: list[PlayedGame], source: str) -> list[Input]:
    """The level of runs per team-game (and per team in innings 1 to 5) over every game in the window."""
    inputs = []
    if games:
        runs = [r for g in games for r in (g.home_runs, g.away_runs)]
        inputs.append(Input("match.league_runs", sum(runs) / len(runs), "runs per team per game", source))
    f5 = [r for g in games for r in (g.home_f5, g.away_f5) if r is not None]
    if f5:
        inputs.append(Input("match.league_f5_runs", sum(f5) / len(f5), "runs per team per 5 innings", source))
    return inputs


def pitcher_inputs(side: str, pitcher: PitcherLine | None, source: str) -> list[Input]:
    if pitcher is None:
        return []
    return [Input(f"{side}.sp_runs_per9", pitcher.runs_per9, "runs allowed per 9 innings", source),
            Input(f"{side}.sp_starts", float(pitcher.starts), "starts", source)]


class MlbStats:
    """StatsSource for baseball backed by MLB's public Stats API. `clock` is injectable for tests."""

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
            raise StatsFetchError("MLB's stats service took too long to answer") from None

    async def _get(self, client: httpx.AsyncClient, path: str, **params):
        """One GET, remembered (core/ttl_cache.py): form and schedules for 3 hours, announced starting pitchers for 1 hour."""
        ttl = TTL_WEATHER_SECONDS if "probablePitcher" in str(params.get("hydrate", "")) else TTL_FORM_SECONDS
        return await STATS_CACHE.get_or_fetch(request_key("mlb", path, params), ttl,
                                              lambda: self._read(client, path, **params))

    async def _read(self, client: httpx.AsyncClient, path: str, **params):
        try:
            response = await client.get(BASE_URL + path, params=params)
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            raise StatsFetchError("MLB's stats service did not answer in time") from None
        except httpx.HTTPStatusError as error:
            raise StatsFetchError(f"MLB's stats service returned HTTP {error.response.status_code}") from None
        except (httpx.HTTPError, ValueError):
            raise StatsFetchError("MLB's stats service could not be reached or gave unreadable data") from None

    async def _run(self, client: httpx.AsyncClient, fixtures: list[Fixture]) -> dict[MatchId, FixtureStats]:
        results: dict[MatchId, FixtureStats] = {}
        covered = []
        for fixture in fixtures:
            if fixture.match_id.sport != "baseball" or normalize_team(fixture.competition) not in LEAGUES:
                results[fixture.match_id] = insufficient(
                    fixture.match_id, f"no free stats source covers {fixture.competition} (MLB Stats API: MLB only)")
            else:
                covered.append(fixture)
        if not covered:
            return results
        today = self._clock().astimezone(timezone.utc).date()
        teams = parse_teams(await self._get(client, "/teams", sportId=1))
        history = parse_history(await self._get(
            client, "/schedule", sportId=1, startDate=(today - timedelta(days=LOOKBACK_DAYS)).isoformat(),
            endDate=today.isoformat(), hydrate="linescore"))
        cutoff = self._clock() - timedelta(days=LOOKBACK_DAYS)
        history = [g for g in history if cutoff <= g.when <= self._clock()]
        days = [f.kickoff_utc.date() for f in covered]
        probables = parse_probables(await self._get(
            client, "/schedule", sportId=1, startDate=(min(days) - timedelta(days=1)).isoformat(),
            endDate=(max(days) + timedelta(days=1)).isoformat(), hydrate="probablePitcher"))
        by_name = {v: k for k, v in teams.items()}
        wanted, questions = {}, {}
        for fixture in covered:
            ids = []
            for side_name in (fixture.home, fixture.away):
                found = resolve_team(side_name, teams.values(), ALIASES)
                if found.status == "matched":
                    ids.append(by_name[found.team])
                elif found.status == "ambiguous":
                    questions[fixture.match_id] = (f"Which team is '{side_name}'? " + " or ".join(found.matches) + "?",
                                                   found.matches)
                    break
                else:
                    results[fixture.match_id] = insufficient(fixture.match_id,
                                                             f"{side_name} was not found among the MLB teams")
                    break
            else:
                wanted[fixture.match_id] = (fixture, ids[0], ids[1])
        for match_id, (question, options) in questions.items():
            results[match_id] = ambiguous(match_id, question, options)
        pitchers_needed = {}
        for match_id, (fixture, home_id, away_id) in wanted.items():
            pitchers_needed[match_id] = self._probable(probables, fixture, home_id, away_id)
        ids = sorted({p for pair in pitchers_needed.values() for p in pair if p is not None})
        season = max((f.kickoff_utc.year for f in covered), default=today.year)
        lines: dict[int, PitcherLine] = {}
        if ids:
            lines = parse_pitchers(await self._get(
                client, "/people", personIds=",".join(str(i) for i in ids),
                hydrate=f"stats(group=[pitching],type=[season],season={season})"), season)
        league = league_inputs(history, NAME)
        for match_id, (fixture, home_id, away_id) in wanted.items():
            results[match_id] = self._assess(match_id, teams, history, home_id, away_id, pitchers_needed[match_id],
                                             lines, league)
        return results

    @staticmethod
    def _probable(probables: list[Probables], fixture: Fixture, home_id: int, away_id: int):
        near = [p for p in probables if p.home_id == home_id and p.away_id == away_id
                and abs(p.when - fixture.kickoff_utc) <= timedelta(hours=PITCHER_WINDOW_HOURS)]
        if not near:
            return (None, None)
        chosen = min(near, key=lambda p: abs(p.when - fixture.kickoff_utc))
        return (chosen.home_pitcher, chosen.away_pitcher)

    @staticmethod
    def _assess(match_id, teams, history, home_id, away_id, pitchers, lines, league) -> FixtureStats:
        raw: list[Input] = []
        problems: list[str] = []
        for side, team in (("home", home_id), ("away", away_id)):
            window = team_window(history, team)
            if not window:
                problems.append(f"{teams[team]} has no completed games in the last {LOOKBACK_DAYS} days")
            else:
                raw.extend(team_inputs(side, window, NAME))
        if problems:
            return insufficient(match_id, *problems)
        for side, pid in zip(("home", "away"), pitchers):
            raw.extend(pitcher_inputs(side, lines.get(pid) if pid is not None else None, NAME))
        raw.extend(league)
        return assess(match_id, raw)
