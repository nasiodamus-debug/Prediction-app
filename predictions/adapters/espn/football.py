"""NFL and college football team stats from ESPN's public site API, behind the StatsSource port (Stories 7.1 to 7.4).

ESPN's `site.api.espn.com` needs no key and no login (custom User-Agents are answered with 403, so requests carry httpx's
default header only). It is unofficial, so every field is validated and a changed shape fails loudly (StatsFetchError)
instead of yielding wrong numbers. No retries: a failed read is reported (AD-14). Read-only GETs (probed 2026-09-30,
spec-7-1 Design Notes):
  * `<league>/scoreboard` (the current week) then `<league>/scoreboard?dates=<year>&seasontype=2&week=<n>` for the weeks
    before it (and the next one, for upcoming venues). A week holds every game with its final score, its QUARTER scores
    (linescores) and, for scheduled games, the venue with its city. College asks `groups=80` (FBS) and `groups=81` (FCS) with `limit=400` and merges them. A date
    RANGE is rejected by ESPN (HTTP 400), so weeks are used: a slate reads about 7 pages per league, not one per game.
  * `<league>/teams/<id>/statistics`: season totals, one read per team in the slate.
  * Open-Meteo (adapters/weather) for the wind of NFL games.

What is derived (source `espn-football:<league>`):
  * Window: a team's last LAST_N completed REGULAR-SEASON games found in the last LOOKBACK_WEEKS weeks; preseason and
    postseason are ignored. Fewer than 3 games (the registry minimum for football) is "insufficient data", so the offseason
    and a season's first two weeks give no picks.
  * `points_for/against`: mean points per game including overtime; `fh_points_*`: quarters 1 and 2 (only when every game
    of the window has them); `q1_points_for`: mean first-quarter points.
  * `q1_rank` (PROXY for "first-quarter offensive efficiency", which no free source publishes): the team's rank, 1 = best, by
    mean first-quarter points per game over the same window, among every team with at least MIN_RANK_GAMES games in the
    league sample (ties share the better rank). Points per quarter is a blunt efficiency measure: it ignores drive count,
    field position and opponent strength.
  * `pace` = possession seconds per offensive play and `rush_rate` = rushing attempts as a percent of offensive plays, from
    the team's SEASON totals (possessionTimeSeconds, totalOffensivePlays, rushingAttempts). The 28.5 s/play line of the
    rule is used as an absolute threshold; the league's "bottom 20%" is not ranked (that would need every team's totals).
  * `match.league_ppg` / `match.league_fh_ppg`: league means over the sampled games.
  * `match.wind_mph`: NFL: the schedule's venue city -> stadium table -> Open-Meteo forecast, or 0 for a fixed-roof
    stadium. College: the home team -> college stadium table (weather/college_stadiums.py), skipped when the schedule names
    a different venue city. A game with no scheduled venue (NFL), a city/team not in the table, an unreadable forecast, or a
    kickoff beyond the forecast horizon has NO wind input.
Only "USA - NFL" and "USA - NCAA, ..." are covered (FBS and FCS teams); every other competition, such as the CFL, is
"insufficient data" (the analyst may fill the required inputs).
"""
from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable

import httpx

from predictions.adapters.weather.openmeteo import fetch_wind
from predictions.adapters.weather.college_stadiums import college_stadium_for
from predictions.adapters.weather.stadiums import stadium_for_city
from predictions.core.engines.american_football import COLLEGE, MIN_GAMES, NFL, league_of
from predictions.core.identity import resolve_team
from predictions.core.inputs import Input
from predictions.core.ports import StatsFetchError
from predictions.core.ratings import adjusted_ratings, results_of
from predictions.core.stats import FixtureStats, ambiguous, assess, insufficient
from predictions.core.ttl_cache import STATS_CACHE, TTL_FORM_SECONDS, request_key
from predictions.core.types import Fixture, MatchId, normalize_team

NAME = "espn-football"
BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/{slug}"
SLUGS = {NFL: "nfl", COLLEGE: "college-football"}
EXTRA_PARAMS = {NFL: {}, COLLEGE: {"limit": "400"}}
# ESPN scoreboard groups per league: college is read for FBS (80) AND FCS (81) (probed 2026-10-01: groups=81 lists the 57
# lower-division games of a Saturday). A FCS team's games against an FBS side only appear in the FBS board, so both are merged.
GROUPS = {NFL: (None,), COLLEGE: ("80", "81")}
TIMEOUT_SECONDS = 30.0
OVERALL_TIMEOUT_SECONDS = 240.0
CONCURRENCY = 4
LOOKBACK_WEEKS = 6           # weeks before the current one that are read
LAST_N = 8                   # a team's inputs come from its last LAST_N completed games
MIN_RANK_GAMES = 3           # a team needs this many games to be ranked
REGULAR_SEASON = 2
MATCH_WINDOW = timedelta(hours=36)   # a scheduled game counts as the fixture's game within this of its kickoff

# SportyBet's spelling of a team -> ESPN's (both normalized). Everything else matches by the identity resolver, and a
# name that fits more than one team becomes a question to the owner, never a guess.
ALIASES: dict[str, str] = {
    "delaware fightin blue hens": "delaware blue hens",
    "delaware fightina blue hens": "delaware blue hens",
    "umass minutemen": "massachusetts minutemen",
    "hawai i rainbow warriors": "hawaii rainbow warriors",
    "la rams": "los angeles rams",
    "la chargers": "los angeles chargers",
    "ny giants": "new york giants",
    "ny jets": "new york jets",
}


@dataclass(frozen=True)
class Side:
    team_id: str
    name: str
    score: int
    quarters: tuple[float, ...]


@dataclass(frozen=True)
class PlayedGame:
    game_id: str
    when: datetime
    home: Side
    away: Side


@dataclass(frozen=True)
class Scheduled:
    when: datetime
    home: str
    away: str
    city: str


@dataclass(frozen=True)
class Board:
    played: tuple[PlayedGame, ...]
    scheduled: tuple[Scheduled, ...]
    week: int | None
    year: int | None
    season_type: int | None


def _bad() -> StatsFetchError:
    return StatsFetchError("ESPN's football service returned an unexpected response")


def _when(text) -> datetime:
    try:
        return datetime.strptime(str(text), "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
    except ValueError:
        raise _bad() from None


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _side(competitor) -> Side | None:
    """One competitor of a finished game, or None when it lacks a score."""
    team = competitor.get("team") or {}
    score = _number(competitor.get("score"))
    if score is None or team.get("id") is None or not team.get("displayName"):
        return None
    quarters = []
    for line in competitor.get("linescores") or []:
        value = _number(line.get("value")) if isinstance(line, dict) else None
        if value is None:
            quarters = []
            break
        quarters.append(value)
    return Side(str(team["id"]), str(team["displayName"]), int(score), tuple(quarters))


def parse_scoreboard(payload) -> Board:
    """The completed regular-season games and the venues of every game on one scoreboard page."""
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise _bad()
    played, scheduled = [], []
    for event in payload["events"]:
        if not isinstance(event, dict):
            raise _bad()
        try:
            competition = event["competitions"][0]
            competitors = competition["competitors"]
            when = _when(event["date"])
            home = next(c for c in competitors if c.get("homeAway") == "home")
            away = next(c for c in competitors if c.get("homeAway") == "away")
            state = event["status"]["type"]
            season_type = (event.get("season") or {}).get("type")
        except (KeyError, IndexError, TypeError, StopIteration, AttributeError):
            raise _bad() from None
        home_name = (home.get("team") or {}).get("displayName")
        away_name = (away.get("team") or {}).get("displayName")
        city = ((competition.get("venue") or {}).get("address") or {}).get("city")
        if home_name and away_name:
            scheduled.append(Scheduled(when, str(home_name), str(away_name), str(city or "")))
        if not state.get("completed") or season_type != REGULAR_SEASON:
            continue
        home_side, away_side = _side(home), _side(away)
        if home_side is not None and away_side is not None:
            played.append(PlayedGame(str(event.get("id")), when, home_side, away_side))
    week = _number((payload.get("week") or {}).get("number")) if isinstance(payload.get("week"), dict) else None
    season = payload.get("season") if isinstance(payload.get("season"), dict) else {}
    year, kind = _number(season.get("year")), _number(season.get("type"))
    return Board(tuple(played), tuple(scheduled), None if week is None else int(week),
                 None if year is None else int(year), None if kind is None else int(kind))


def parse_team_statistics(payload) -> tuple[float, float] | None:
    """(seconds per offensive play, rush percent of plays) from a team's season totals, or None when they are absent."""
    try:
        categories = payload["results"]["stats"]["categories"]
    except (KeyError, TypeError):
        return None
    found: dict[str, float] = {}
    for category in categories if isinstance(categories, list) else []:
        for stat in category.get("stats") or []:
            if isinstance(stat, dict) and stat.get("name") in ("possessionTimeSeconds", "totalOffensivePlays",
                                                              "rushingAttempts"):
                value = _number(stat.get("value"))
                if value is not None:
                    found.setdefault(stat["name"], value)
    plays, seconds, rushes = (found.get("totalOffensivePlays"), found.get("possessionTimeSeconds"),
                              found.get("rushingAttempts"))
    if not plays or not seconds or rushes is None or plays <= 0 or seconds <= 0 or rushes < 0 or rushes > plays:
        return None
    return seconds / plays, rushes / plays * 100.0


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values)


def team_window(games: list[PlayedGame], team_id: str) -> list[tuple[Side, Side]]:
    """(the team's side, the opponent's side) of its last LAST_N games, newest first."""
    own = [g for g in games if team_id in (g.home.team_id, g.away.team_id)]
    own.sort(key=lambda g: g.when, reverse=True)
    return [(g.home, g.away) if g.home.team_id == team_id else (g.away, g.home) for g in own[:LAST_N]]


def q1_mean(window: list[tuple[Side, Side]]) -> float | None:
    if not window or any(len(mine.quarters) < 1 for mine, _ in window):
        return None
    return _mean(mine.quarters[0] for mine, _ in window)


def q1_ranks(games: list[PlayedGame]) -> dict[str, int]:
    """team id -> rank (1 = best) by mean first-quarter points over its window; ties share the better rank."""
    means = {}
    ids = {s.team_id for g in games for s in (g.home, g.away)}
    for team_id in ids:
        window = team_window(games, team_id)
        value = q1_mean(window) if len(window) >= MIN_RANK_GAMES else None
        if value is not None:
            means[team_id] = value
    return {t: 1 + sum(1 for other in means.values() if other > v) for t, v in means.items()}


def team_inputs(side: str, window: list[tuple[Side, Side]], rank: int | None, source: str) -> list[Input]:
    n = len(window)
    if n == 0:
        return []
    inputs = [Input(f"{side}.matches", float(n), "games", source),
              Input(f"{side}.points_for", _mean(m.score for m, _ in window), "points per game", source),
              Input(f"{side}.points_against", _mean(o.score for _, o in window), "points per game", source)]
    q1 = q1_mean(window)
    if q1 is not None:
        inputs.append(Input(f"{side}.q1_points_for", q1, "1st-quarter points per game", source))
    if all(len(m.quarters) >= 2 and len(o.quarters) >= 2 for m, o in window):
        inputs.append(Input(f"{side}.fh_points_for", _mean(sum(m.quarters[:2]) for m, _ in window),
                            "1st-half points per game", source))
        inputs.append(Input(f"{side}.fh_points_against", _mean(sum(o.quarters[:2]) for _, o in window),
                            "1st-half points per game", source))
    if rank is not None:
        inputs.append(Input(f"{side}.q1_rank", float(rank), "rank by 1st-quarter points per game (1 = best)", source))
    return inputs


def adjusted_inputs(games: list[PlayedGame], home_id: str, away_id: str, source: str) -> list[Input]:
    """Opponent-adjusted points for and against for the two teams (core/ratings.py, no ridge), found over every game of the window.
    The backtest (tools/football_backtest.py, 1,149 college games) found these predict college margins and winners better than the
    raw averages (margin miss 13.7 -> 13.3, winners right 69% -> 71.5%) and totals slightly worse, so the engine uses them for
    handicaps and winners only. [] when the games give no ratings."""
    ratings = adjusted_ratings(results_of(games), ridge=0.0)
    if ratings is None:
        return []
    unit = "points per game, opponent-adjusted"
    found: list[Input] = []
    for side, team_id in (("home", home_id), ("away", away_id)):
        rate_for, rate_against = ratings.rate_for(team_id), ratings.rate_against(team_id)
        if rate_for is not None and rate_against is not None:
            found += [Input(f"{side}.adj_points_for", rate_for, unit, source), Input(f"{side}.adj_points_against", rate_against, unit, source)]
    return found


def league_inputs(games: list[PlayedGame], source: str) -> list[Input]:
    if not games:
        return []
    inputs = [Input("match.league_ppg", _mean(s.score for g in games for s in (g.home, g.away)),
                    "points per team per game", source)]
    with_halves = [s for g in games for s in (g.home, g.away) if len(s.quarters) >= 2]
    if with_halves:
        inputs.append(Input("match.league_fh_ppg", _mean(sum(s.quarters[:2]) for s in with_halves),
                            "1st-half points per team per game", source))
    return inputs


class EspnFootballStats:
    """StatsSource for American football (NFL and college). `clock` is injectable for tests."""

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
            raise StatsFetchError("ESPN's football service took too long to answer") from None

    async def _get(self, client: httpx.AsyncClient, gate: asyncio.Semaphore, league: str, path: str, **params):
        """One GET, remembered for a few hours (core/ttl_cache.py): team form and schedules only, never an error."""
        return await STATS_CACHE.get_or_fetch(request_key("espn-football", f"{league}{path}", params), TTL_FORM_SECONDS,
                                              lambda: self._read(client, gate, league, path, **params))

    async def _read(self, client: httpx.AsyncClient, gate: asyncio.Semaphore, league: str, path: str, **params):
        async with gate:
            try:
                response = await client.get(BASE_URL.format(slug=SLUGS[league]) + path,
                                            params={**EXTRA_PARAMS[league], **params} if path == "/scoreboard" else params)
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
            league = league_of(fixture.competition)
            if fixture.match_id.sport != "american_football" or league is None:
                results[fixture.match_id] = insufficient(
                    fixture.match_id, f"no free stats source covers {fixture.competition} (ESPN: NFL and college FBS/FCS only)")
            else:
                by_league.setdefault(league, []).append(fixture)
        gate = asyncio.Semaphore(CONCURRENCY)
        for league, games in by_league.items():
            results.update(await self._league(client, gate, league, games))
        return results

    async def _boards(self, client, gate, league: str) -> tuple[list[PlayedGame], list[Scheduled], int | None]:
        played: dict[str, PlayedGame] = {}
        scheduled: list[Scheduled] = []
        week = None
        for group in GROUPS[league]:
            extra = {} if group is None else {"groups": group}
            got, sched, current_week = await self._group_boards(client, gate, league, extra)
            played.update({g.game_id: g for g in got})
            scheduled.extend(sched)
            week = week if week is not None else current_week
        return list(played.values()), scheduled, week

    async def _group_boards(self, client, gate, league: str, extra: dict) -> tuple[list[PlayedGame], list[Scheduled], int | None]:
        current = parse_scoreboard(await self._get(client, gate, league, "/scoreboard", **extra) or {})
        played = {g.game_id: g for g in current.played}
        scheduled = list(current.scheduled)
        if current.week is None or current.year is None or current.season_type != REGULAR_SEASON:
            return list(played.values()), scheduled, current.week
        weeks = [w for w in range(current.week - LOOKBACK_WEEKS, current.week) if w >= 1]

        async def one(week: int):
            payload = await self._get(client, gate, league, "/scoreboard", dates=str(current.year),
                                      seasontype=str(REGULAR_SEASON), week=str(week), **extra)
            return parse_scoreboard(payload or {})

        for board in await asyncio.gather(*(one(w) for w in weeks)):
            played.update({g.game_id: g for g in board.played})
            scheduled.extend(board.scheduled)
        try:  # the next week only adds venues of later games: a failure there costs wind figures, not the slate
            board = await one(current.week + 1)
            scheduled.extend(board.scheduled)
        except StatsFetchError:
            pass
        return list(played.values()), scheduled, current.week

    async def _league(self, client, gate, league: str, fixtures: list[Fixture]) -> dict[MatchId, FixtureStats]:
        games, scheduled, _ = await self._boards(client, gate, league)
        now = self._clock().astimezone(timezone.utc)
        games = [g for g in games if g.when <= now]
        names: dict[str, str] = {}
        for g in games:
            names[g.home.team_id], names[g.away.team_id] = g.home.name, g.away.name
        known = set(names.values()) | {n for s in scheduled for n in (s.home, s.away)}
        by_name = {v: k for k, v in names.items()}
        results: dict[MatchId, FixtureStats] = {}
        wanted: dict[MatchId, tuple[str, str, str, str]] = {}
        for fixture in fixtures:
            resolved, problem = [], None
            for side_name in (fixture.home, fixture.away):
                found = resolve_team(side_name, known, ALIASES)
                if found.status == "matched":
                    resolved.append(found.team)
                elif found.status == "ambiguous":
                    problem = ambiguous(fixture.match_id, f"Which team is '{side_name}'? " + " or ".join(found.matches) + "?",
                                        found.matches)
                    break
                else:
                    problem = insufficient(fixture.match_id, f"{side_name} was not found among the {league.upper()} teams "
                                                             f"ESPN lists")
                    break
            if problem is not None:
                results[fixture.match_id] = problem
            else:
                wanted[fixture.match_id] = (resolved[0], resolved[1], by_name.get(resolved[0], ""), by_name.get(resolved[1], ""))
        if not wanted:
            return results
        source = f"{NAME}:{SLUGS[league]}"
        ranks = q1_ranks(games)
        league_in = league_inputs(games, source)
        team_ids = sorted({t for v in wanted.values() for t in v[2:] if t})
        pace = await self._pace(client, gate, league, team_ids)
        fixture_by_id = {f.match_id: f for f in fixtures}
        for match_id, (home_name, away_name, home_id, away_id) in wanted.items():
            fixture = fixture_by_id[match_id]
            raw: list[Input] = []
            problems = []
            for side, name, team_id in (("home", home_name, home_id), ("away", away_name, away_id)):
                window = team_window(games, team_id) if team_id else []
                if not window:
                    problems.append(f"{name} has no completed regular-season games in the last {LOOKBACK_WEEKS} weeks")
                    continue
                if len(window) < MIN_GAMES.get(league, 3):
                    problems.append(f"{name} has only {len(window)} completed regular-season game(s) in the last {LOOKBACK_WEEKS} "
                                    f"weeks; {MIN_GAMES[league]} are needed")
                    continue
                raw.extend(team_inputs(side, window, ranks.get(team_id), source))
                if pace.get(team_id) is not None:
                    seconds, rush = pace[team_id]
                    raw.append(Input(f"{side}.pace", seconds, "seconds per offensive play", source))
                    raw.append(Input(f"{side}.rush_rate", rush, "percent of offensive plays that are rushes", source))
            if problems:
                results[match_id] = insufficient(match_id, *problems)
                continue
            raw.extend(league_in)
            if league == COLLEGE:   # verified on college games only; the NFL keeps the raw averages
                raw.extend(adjusted_inputs(games, home_id, away_id, source))
            wind = await self._wind(client, league, fixture, home_name, away_name, scheduled, now)
            if wind is not None:
                raw.append(wind)
            results[match_id] = assess(match_id, raw)
        return results

    async def _pace(self, client, gate, league: str, team_ids: list[str]) -> dict[str, tuple[float, float] | None]:
        async def one(team_id: str):
            try:
                payload = await self._get(client, gate, league, f"/teams/{team_id}/statistics")
            except StatsFetchError:
                return team_id, None  # optional inputs: missing pace means the clock-control rule does not apply
            return team_id, parse_team_statistics(payload) if payload else None
        return dict(await asyncio.gather(*(one(t) for t in team_ids)))

    async def _wind(self, client, league: str, fixture: Fixture, home: str, away: str, scheduled: list[Scheduled],
                    now: datetime) -> Input | None:
        match = [s for s in scheduled if normalize_team(s.home) == normalize_team(home)
                 and normalize_team(s.away) == normalize_team(away) and abs(s.when - fixture.kickoff_utc) <= MATCH_WINDOW]
        if league == NFL:
            if len(match) < 1:
                return None
            stadium = stadium_for_city(match[0].city)
        else:  # college: the home team's stadium, unless the schedule puts the game in another city
            stadium = college_stadium_for(home, match[0].city if match else None)
        if stadium is None:
            return None
        if stadium.roof == "dome":
            return Input("match.wind_mph", 0.0, "mph", f"stadium-table: {stadium.name} (fixed roof, indoors)")
        speed = await fetch_wind(client, stadium, fixture.kickoff_utc, now)
        if speed is None:
            return None
        return Input("match.wind_mph", speed, "mph", f"open-meteo: {stadium.name} forecast, highest hourly wind "
                                                       f"from kickoff to +3 h")
