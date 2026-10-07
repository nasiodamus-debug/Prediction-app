"""Basketball team stats from ESPN's public site API, behind the StatsSource port (Story 6.1).

ESPN's `site.api.espn.com` needs no key and no login. It is an unofficial endpoint, so every field is validated and a
change in its shape fails loudly (StatsFetchError) instead of yielding wrong numbers. Two quirks found by probing
(2026-09-30): the API answers 403 to a custom or browser-like User-Agent, so requests use httpx's own default header
and nothing else; and the schedule endpoint of a team has no quarter scores, so the game log is built from the
per-date `scoreboard` (which has quarter scores, points, field-goal and free-throw attempts for every game).

What is computed (see spec-6-1 Design Notes for the approximations):
  * Rolling window: a team's last LAST_N completed REGULAR-SEASON or PLAYOFF games (preseason is ignored), found by
    walking back day by day from today for at most LOOKBACK_DAYS. Fewer than 5 games (the registry minimum) is
    "insufficient data", so an offseason or a season's first days give no picks. Nothing is carried over from an
    older season.
  * Possessions per game are ESTIMATED: FGA + 0.44 * FTA - OREB + TOV. The scoreboard has FGA and FTA per game; the
    team's SEASON averages of offensive rebounds and turnovers (team statistics endpoint) supply the OREB/TOV part.
  * ORTG = 100 * points scored / possessions, DRTG = 100 * points allowed / the same possessions (the opponent's
    possessions are not looked up, they differ by about one per game), pace = possessions per game, over the window.
  * 1st-half points = periods 1 and 2 of the linescores; only given when every game of the window has them.
  * `match.league_pace` = the league's average of FGA + 0.44 * FTA over the sampled games plus the average of the
    looked-up teams' (TOV - OREB): the yardstick of the "explosive start" proxy.
Only leagues in LEAGUES are covered; every other league is "insufficient data" (the analyst may fill it).
No retries: a failed read is reported (AD-14).
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
from predictions.core.ttl_cache import STATS_CACHE, TTL_FORM_SECONDS, request_key
from predictions.core.types import Fixture, MatchId, normalize_team

NAME = "espn"
BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/{league}"
TIMEOUT_SECONDS = 20.0
OVERALL_TIMEOUT_SECONDS = 240.0
CONCURRENCY = 4           # polite: at most this many requests at once
BATCH_DAYS = 6            # scoreboard days asked for together, newest first
LAST_N = 10               # a team's inputs come from its last LAST_N completed games
LOOKBACK_DAYS = 45        # how far back the game log is searched; older games are never used
FT_POSSESSION_FACTOR = 0.44
COUNTED_SEASON_TYPES = (2, 3)   # ESPN season types: 1 preseason, 2 regular season, 3 playoffs

# SportyBet "<country> - <league>" (normalized) -> ESPN league slug. Anything else is not covered.
LEAGUES: dict[str, str] = {
    "usa nba": "nba",
    "usa wnba": "wnba",
}
# SportyBet's spelling of a team -> ESPN's (both normalized). Everything else matches by the identity resolver.
ALIASES: dict[str, str] = {
    "la lakers": "los angeles lakers",
    "los angeles clippers": "la clippers",
}


@dataclass(frozen=True)
class GameTeam:
    team_id: str
    points: int
    fga: float | None
    fta: float | None
    fh_points: float | None  # periods 1 and 2, when the linescores have them


@dataclass(frozen=True)
class PlayedGame:
    when: datetime  # UTC
    season_year: int
    home: GameTeam
    away: GameTeam


def _bad() -> StatsFetchError:
    return StatsFetchError("ESPN returned an unexpected response")


def _int(value) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number == int(number) and number >= 0 else None


def _stat(competitor: dict, name: str) -> float | None:
    for item in competitor.get("statistics") or []:
        if isinstance(item, dict) and item.get("name") == name:
            try:
                value = float(item.get("displayValue"))
            except (TypeError, ValueError):
                return None
            return value if value >= 0 else None
    return None


def _first_half(competitor: dict) -> float | None:
    periods = {}
    for line in competitor.get("linescores") or []:
        if isinstance(line, dict) and isinstance(line.get("period"), int):
            try:
                periods[line["period"]] = float(line.get("value"))
            except (TypeError, ValueError):
                return None
    return periods[1] + periods[2] if 1 in periods and 2 in periods else None


def _when(text) -> datetime:
    try:
        return datetime.strptime(str(text), "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
    except ValueError:
        raise _bad() from None


def parse_scoreboard(payload) -> list[PlayedGame]:
    """The completed regular-season and playoff games of one scoreboard day. A game that is not finished, is a
    preseason game, or lacks a score is left out; a payload that is not a scoreboard is an error."""
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise _bad()
    games: list[PlayedGame] = []
    for event in payload["events"]:
        if not isinstance(event, dict):
            raise _bad()
        status = ((event.get("status") or {}).get("type") or {})
        season = event.get("season") or {}
        if status.get("completed") is not True or season.get("type") not in COUNTED_SEASON_TYPES:
            continue
        year = season.get("year")
        competitions = event.get("competitions")
        if type(year) is not int or not isinstance(competitions, list) or len(competitions) != 1:
            raise _bad()
        sides: dict[str, GameTeam] = {}
        for competitor in competitions[0].get("competitors") or []:
            team, where = (competitor.get("team") or {}), competitor.get("homeAway")
            points = _int(competitor.get("score"))
            if where not in ("home", "away") or not team.get("id") or points is None:
                sides = {}
                break
            sides[where] = GameTeam(str(team["id"]), points, _stat(competitor, "fieldGoalsAttempted"),
                                    _stat(competitor, "freeThrowsAttempted"), _first_half(competitor))
        if set(sides) == {"home", "away"}:
            games.append(PlayedGame(_when(event.get("date")), year, sides["home"], sides["away"]))
    return games


def parse_teams(payload) -> dict[str, str]:
    """team id -> display name."""
    try:
        teams = payload["sports"][0]["leagues"][0]["teams"]
        found = {str(t["team"]["id"]): str(t["team"]["displayName"]) for t in teams}
    except (KeyError, IndexError, TypeError):
        raise _bad() from None
    if not found:
        raise _bad()
    return found


def parse_team_statistics(payload, season: int) -> tuple[float, float] | None:
    """(season average offensive rebounds, season average turnovers) of the requested season, or None when the
    payload does not hold them for that season (never guessed)."""
    try:
        if (payload.get("requestedSeason") or {}).get("year") != season:
            return None
        values = {}
        for category in payload["results"]["stats"]["categories"]:
            for stat in category.get("stats") or []:
                values[stat.get("name")] = stat.get("value")
        oreb, tov = float(values["avgOffensiveRebounds"]), float(values["avgTurnovers"])
    except (KeyError, TypeError, ValueError, AttributeError):
        return None
    return (oreb, tov) if 0 <= oreb <= 30 and 0 <= tov <= 30 else None


def team_window(games: list[tuple[PlayedGame, GameTeam, GameTeam]]) -> list[tuple[PlayedGame, GameTeam, GameTeam]]:
    """A team's games (game, own side, opposing side), newest first, only those with the box score numbers needed."""
    usable = [g for g in games if g[1].fga is not None and g[1].fta is not None]
    return sorted(usable, key=lambda g: g[0].when, reverse=True)[:LAST_N]


def team_inputs(side: str, window, oreb: float, tov: float, source: str) -> list[Input]:
    """The model inputs of one team from its window: matches, ORTG, DRTG, pace, points, first-half points."""
    n = len(window)
    poss = [own.fga + FT_POSSESSION_FACTOR * own.fta - oreb + tov for _, own, _ in window]
    total_poss = sum(poss)
    if n == 0 or total_poss <= 0:
        return []
    points = sum(own.points for _, own, _ in window)
    against = sum(opp.points for _, _, opp in window)
    inputs = [
        Input(f"{side}.matches", float(n), "games", source),
        Input(f"{side}.ortg", 100.0 * points / total_poss, "points per 100 possessions", source),
        Input(f"{side}.drtg", 100.0 * against / total_poss, "points per 100 possessions", source),
        Input(f"{side}.pace", total_poss / n, "possessions per game", source),
        Input(f"{side}.points_for", points / n, "points per game", source),
        Input(f"{side}.points_against", against / n, "points per game", source),
    ]
    if all(own.fh_points is not None and opp.fh_points is not None for _, own, opp in window):
        inputs.append(Input(f"{side}.fh_points_for", sum(o.fh_points for _, o, _ in window) / n,
                            "1st-half points per game", source))
        inputs.append(Input(f"{side}.fh_points_against", sum(p.fh_points for _, _, p in window) / n,
                            "1st-half points per game", source))
    return inputs


class EspnBasketballStats:
    """StatsSource for basketball backed by ESPN's public site API. `clock` is injectable for tests."""

    def __init__(self, client: httpx.AsyncClient | None = None, clock: Callable[[], datetime] | None = None):
        self._client = client
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    async def fetch_stats(self, fixtures: list[Fixture]) -> dict[MatchId, FixtureStats]:
        try:
            if self._client is not None:
                return await asyncio.wait_for(self._run(self._client, fixtures), OVERALL_TIMEOUT_SECONDS)
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:  # httpx's own headers: see the module note
                return await asyncio.wait_for(self._run(client, fixtures), OVERALL_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raise StatsFetchError("ESPN took too long to answer") from None

    async def _get(self, client: httpx.AsyncClient, gate: asyncio.Semaphore, league: str, path: str, **params):
        """One GET, remembered for a few hours (core/ttl_cache.py): team form and schedules only, never an error."""
        return await STATS_CACHE.get_or_fetch(request_key("espn-basketball", f"{league}{path}", params), TTL_FORM_SECONDS,
                                              lambda: self._read(client, gate, league, path, **params))

    async def _read(self, client: httpx.AsyncClient, gate: asyncio.Semaphore, league: str, path: str, **params):
        """One GET. Returns the JSON, or None for a 404 (the thing does not exist); anything else that goes wrong is a
        StatsFetchError. No retries."""
        async with gate:
            try:
                response = await client.get(BASE_URL.format(league=league) + path, params=params)
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException:
                raise StatsFetchError("ESPN did not answer in time") from None
            except httpx.HTTPStatusError as error:
                raise StatsFetchError(f"ESPN returned HTTP {error.response.status_code}") from None
            except (httpx.HTTPError, ValueError):
                raise StatsFetchError("ESPN could not be reached or gave unreadable data") from None

    async def _run(self, client: httpx.AsyncClient, fixtures: list[Fixture]) -> dict[MatchId, FixtureStats]:
        results: dict[MatchId, FixtureStats] = {}
        by_league: dict[str, list[Fixture]] = {}
        for fixture in fixtures:
            league = LEAGUES.get(normalize_team(fixture.competition))
            if fixture.match_id.sport != "basketball" or league is None:
                results[fixture.match_id] = insufficient(
                    fixture.match_id, f"no free stats source covers {fixture.competition} (ESPN: NBA and WNBA only)")
            else:
                by_league.setdefault(league, []).append(fixture)
        gate = asyncio.Semaphore(CONCURRENCY)
        for league, games in by_league.items():
            results.update(await self._league(client, gate, league, games))
        return results

    async def _league(self, client, gate, league: str, fixtures: list[Fixture]) -> dict[MatchId, FixtureStats]:
        payload = await self._get(client, gate, league, "/teams")
        if payload is None:
            raise _bad()
        names = parse_teams(payload)
        by_name = {v: k for k, v in names.items()}
        results: dict[MatchId, FixtureStats] = {}
        wanted: dict[MatchId, tuple[str, str]] = {}
        for fixture in fixtures:
            ids, question = [], None
            for side_name in (fixture.home, fixture.away):
                found = resolve_team(side_name, names.values(), ALIASES)
                if found.status == "matched":
                    ids.append(by_name[found.team])
                elif found.status == "ambiguous":
                    question = (f"Which team is '{side_name}'? " + " or ".join(found.matches) + "?")
                    results[fixture.match_id] = ambiguous(fixture.match_id, question, found.matches)
                    break
                else:
                    results[fixture.match_id] = insufficient(fixture.match_id,
                                                             f"{side_name} was not found among the {league.upper()} teams")
                    break
            if question is None and len(ids) == 2:
                wanted[fixture.match_id] = (ids[0], ids[1])
        if not wanted:
            return results
        need = {t for pair in wanted.values() for t in pair}
        games = await self._game_log(client, gate, league, need)
        source = f"{NAME}:{league}"
        stats_by_team: dict[str, tuple[float, float] | None] = {}
        windows: dict[str, list] = {}
        for team in need:
            windows[team] = team_window([(g, *((g.home, g.away) if g.home.team_id == team else (g.away, g.home)))
                                         for g in games if team in (g.home.team_id, g.away.team_id)])
        seasons = {team: max(g[0].season_year for g in w) for team, w in windows.items() if w}
        with_games = sorted(t for t in need if windows[t])
        payloads = await asyncio.gather(*(self._get(client, gate, league, f"/teams/{t}/statistics", season=seasons[t])
                                          for t in with_games))
        for team, payload in zip(with_games, payloads):
            stats_by_team[team] = parse_team_statistics(payload, seasons[team]) if payload else None
        league_pace = self._league_pace(games, need, stats_by_team)
        for match_id, (home_id, away_id) in wanted.items():
            results[match_id] = self._assess(match_id, names, home_id, away_id, windows, stats_by_team, league_pace,
                                             source)
        return results

    async def _game_log(self, client, gate, league: str, need: set[str]) -> list[PlayedGame]:
        """Completed games, newest days first, until every needed team has LAST_N of them or the lookback ends."""
        today = self._clock().astimezone(timezone.utc).date()
        found: dict[tuple, PlayedGame] = {}
        counts = {t: 0 for t in need}
        day = 0
        while day < LOOKBACK_DAYS and any(c < LAST_N for c in counts.values()):
            days = [today - timedelta(days=d) for d in range(day, min(day + BATCH_DAYS, LOOKBACK_DAYS))]
            batches = await asyncio.gather(*(self._scoreboard(client, gate, league, d) for d in days))
            for games in batches:
                for game in games:
                    if game.when <= self._clock() and (game.when, game.home.team_id, game.away.team_id) not in found:
                        found[(game.when, game.home.team_id, game.away.team_id)] = game
                        for t in (game.home.team_id, game.away.team_id):
                            if t in counts and game.home.fga is not None and game.away.fga is not None:
                                counts[t] += 1
            day += BATCH_DAYS
        return sorted(found.values(), key=lambda g: g.when)

    async def _scoreboard(self, client, gate, league: str, day: date) -> list[PlayedGame]:
        payload = await self._get(client, gate, league, "/scoreboard", dates=day.strftime("%Y%m%d"))
        return parse_scoreboard(payload) if payload is not None else []

    @staticmethod
    def _league_pace(games: list[PlayedGame], need: set[str], stats_by_team) -> float | None:
        raw = [t.fga + FT_POSSESSION_FACTOR * t.fta for g in games for t in (g.home, g.away)
               if t.fga is not None and t.fta is not None]
        corrections = [tov - oreb for pair in stats_by_team.values() if pair for oreb, tov in [pair]]
        if not raw or not corrections:
            return None
        return sum(raw) / len(raw) + sum(corrections) / len(corrections)

    @staticmethod
    def _assess(match_id, names, home_id, away_id, windows, stats_by_team, league_pace, source) -> FixtureStats:
        raw: list[Input] = []
        problems: list[str] = []
        for side, team in (("home", home_id), ("away", away_id)):
            window, pair = windows.get(team, []), stats_by_team.get(team)
            if not window:
                problems.append(f"{names[team]} has no completed games in the last {LOOKBACK_DAYS} days")
            elif pair is None:
                problems.append(f"{names[team]}: season rebounds and turnovers (needed for possessions) are not available")
            else:
                raw.extend(team_inputs(side, window, pair[0], pair[1], source))
        if problems:
            return insufficient(match_id, *problems)
        if league_pace is not None:
            raw.append(Input("match.league_pace", league_pace, "possessions per game", source))
        return assess(match_id, raw)
