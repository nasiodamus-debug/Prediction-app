"""Team history and style stats from Understat's public league data, behind the StatsSource port.

Understat is free and needs no key or login. Its league pages load one JSON document per league and
season (`getLeagueData/<league>/<season>`); that is an unofficial endpoint, so every field is validated
and a change in its shape fails loudly instead of yielding wrong numbers. It covers only six leagues.
Times in it are UTC. No retries: a failed read is reported (AD-14).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from predictions.core.identity import TeamMatch, resolve_team
from predictions.core.inputs import Input
from predictions.core.ports import StatsFetchError
from predictions.core.stats import FixtureStats, ambiguous, assess, insufficient
from predictions.core.ttl_cache import STATS_CACHE, TTL_FORM_SECONDS, request_key
from predictions.core.types import Fixture, MatchId, normalize_team

NAME = "understat"
URL = "https://understat.com/getLeagueData/{league}/{season}"
REFERER = "https://understat.com/league/{league}/{season}"
TIMEOUT_SECONDS = 20.0
OVERALL_TIMEOUT_SECONDS = 90.0
LAST_N = 10  # a team's inputs come from its last LAST_N matches before kickoff
GAME_WINDOW = timedelta(hours=36)  # the source's kickoff may differ from the bookmaker's by this much
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "application/json",
    "X-Requested-With": "XMLHttpRequest",
}

# SportyBet "<country> - <league>" (normalized) -> Understat league code. Anything else is not covered.
LEAGUES: dict[str, str] = {
    "england premier league": "EPL",
    "spain laliga": "La_liga",
    "spain la liga": "La_liga",
    "germany bundesliga": "Bundesliga",
    "italy serie a": "Serie_A",
    "france ligue 1": "Ligue_1",
    "russia premier league": "RFPL",
}


@dataclass(frozen=True)
class PlayedMatch:
    when: datetime  # UTC
    xg_for: float
    xg_against: float
    goals_for: int
    goals_against: int
    ppda_att: float
    ppda_def: float
    deep: float
    deep_allowed: float


@dataclass(frozen=True)
class ScheduledGame:
    when: datetime  # UTC
    home: str
    away: str


@dataclass(frozen=True)
class LeagueSeason:
    history: dict[str, list[PlayedMatch]]  # by team title, oldest first
    schedule: list[ScheduledGame]


def _bad() -> StatsFetchError:
    return StatsFetchError("Understat returned data in an unexpected format")


def _when(text) -> datetime:
    try:
        return datetime.strptime(str(text), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except ValueError:
        raise _bad() from None


def _num(value) -> float:
    if isinstance(value, bool):
        raise _bad()
    try:
        return float(value)
    except (TypeError, ValueError):
        raise _bad() from None


def parse_league(payload) -> LeagueSeason:
    """One raw league-season document to history per team and the game schedule. Raises on a bad shape."""
    if not isinstance(payload, dict) or not isinstance(payload.get("teams"), dict) \
            or not isinstance(payload.get("dates"), list):
        raise _bad()
    history: dict[str, list[PlayedMatch]] = {}
    for team in payload["teams"].values():
        if not isinstance(team, dict) or not team.get("title") or not isinstance(team.get("history"), list):
            raise _bad()
        played = []
        for row in team["history"]:
            if not isinstance(row, dict) or not isinstance(row.get("ppda"), dict):
                raise _bad()
            played.append(PlayedMatch(
                _when(row.get("date")), _num(row.get("xG")), _num(row.get("xGA")),
                int(_num(row.get("scored"))), int(_num(row.get("missed"))),
                _num(row["ppda"].get("att")), _num(row["ppda"].get("def")),
                _num(row.get("deep")), _num(row.get("deep_allowed"))))
        played.sort(key=lambda m: m.when)
        history[str(team["title"])] = played
    schedule = []
    for game in payload["dates"]:
        if not isinstance(game, dict) or not isinstance(game.get("h"), dict) or not isinstance(game.get("a"), dict):
            raise _bad()
        if game["h"].get("title") and game["a"].get("title"):
            schedule.append(ScheduledGame(_when(game.get("datetime")), str(game["h"]["title"]),
                                          str(game["a"]["title"])))
    return LeagueSeason(history, schedule)


def season_of(kickoff: datetime) -> int:
    """The season's starting year: July onward belongs to the season that starts that year."""
    return kickoff.year if kickoff.month >= 7 else kickoff.year - 1


def team_inputs(side: str, matches: list[PlayedMatch], source: str) -> list[Input]:
    """Averages over `matches` (already the last N before kickoff)."""
    n = len(matches)
    if n == 0:
        return []

    def mean(values) -> float:
        return round(sum(values) / n, 4)

    inputs = [
        Input(f"{side}.matches", float(n), "matches", source),
        Input(f"{side}.xg_for", mean(m.xg_for for m in matches), "xG per match", source),
        Input(f"{side}.xg_against", mean(m.xg_against for m in matches), "xG per match", source),
        Input(f"{side}.goals_for", mean(m.goals_for for m in matches), "goals per match", source),
        Input(f"{side}.goals_against", mean(m.goals_against for m in matches), "goals per match", source),
        Input(f"{side}.deep", mean(m.deep for m in matches), "deep completions per match", source),
        Input(f"{side}.deep_allowed", mean(m.deep_allowed for m in matches),
              "deep completions allowed per match", source),
    ]
    defensive = sum(m.ppda_def for m in matches)
    if defensive > 0:
        inputs.append(Input(f"{side}.ppda", round(sum(m.ppda_att for m in matches) / defensive, 4),
                            "opponent passes per defensive action", source))
    return inputs


class UnderstatStats:
    """StatsSource backed by Understat. One request per league and season needed, at most two seasons."""

    def __init__(self, client: httpx.AsyncClient | None = None):
        self._client = client

    async def fetch_stats(self, fixtures: list[Fixture]) -> dict[MatchId, FixtureStats]:
        try:
            if self._client is not None:
                return await asyncio.wait_for(self._run(self._client, fixtures), OVERALL_TIMEOUT_SECONDS)
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, headers=_HEADERS) as client:
                return await asyncio.wait_for(self._run(client, fixtures), OVERALL_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raise StatsFetchError("Understat took too long to answer") from None

    async def _league(self, client, cache, league: str, season: int) -> LeagueSeason:
        key = (league, season)
        if key not in cache:   # per request; across requests the parsed season is remembered for a few hours
            cache[key] = await STATS_CACHE.get_or_fetch(request_key("understat", league, {"season": season}),
                                                        TTL_FORM_SECONDS, lambda: self._read(client, league, season))
        return cache[key]

    async def _read(self, client, league: str, season: int) -> LeagueSeason:
        try:
            response = await client.get(URL.format(league=league, season=season),
                                        headers={"Referer": REFERER.format(league=league, season=season)})
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException:
            raise StatsFetchError("Understat did not answer in time") from None
        except httpx.HTTPStatusError as error:
            raise StatsFetchError(f"Understat returned HTTP {error.response.status_code}") from None
        except (httpx.HTTPError, ValueError):
            raise StatsFetchError("Understat could not be reached or gave unreadable data") from None
        return parse_league(payload)

    async def _run(self, client, fixtures: list[Fixture]) -> dict[MatchId, FixtureStats]:
        cache: dict = {}
        results: dict[MatchId, FixtureStats] = {}
        for fixture in fixtures:
            league = LEAGUES.get(normalize_team(fixture.competition))
            if league is None:
                results[fixture.match_id] = insufficient(
                    fixture.match_id, "this competition is not covered by the stats source")
                continue
            results[fixture.match_id] = await self._one(client, cache, fixture, league)
        return results

    async def _one(self, client, cache, fixture: Fixture, league: str) -> FixtureStats:
        mid = fixture.match_id
        season = season_of(fixture.kickoff_utc)
        current = await self._league(client, cache, league, season)
        if not current.schedule:
            return insufficient(mid, f"the stats source has no {league} data for the {season} season yet")
        teams = list(current.history)
        resolved = {}
        for side, name in (("home", fixture.home), ("away", fixture.away)):
            match = resolve_team(name, teams)
            if match.status == "ambiguous" and len(match.matches) == 1:
                # One candidate that only differs by an extra word: the schedule check below (this exact pair near this
                # kickoff) confirms it; if that fails the fixture is not evaluated.
                match = TeamMatch("matched", match.matches, "partial")
            if match.status == "ambiguous":
                return ambiguous(
                    mid, f"Which team is '{name}' ({side})? It could be: {', '.join(match.matches)}.",
                    match.matches)
            if match.status == "none":
                return insufficient(mid, f"the {side} team '{name}' is not in the stats source's {league} list")
            resolved[side] = match.team
        if resolved["home"] == resolved["away"]:
            return ambiguous(mid, f"'{fixture.home}' and '{fixture.away}' both match {resolved['home']}; "
                                  "which is which?", (resolved["home"],))
        games = [g for g in current.schedule if g.home == resolved["home"] and g.away == resolved["away"]
                 and abs(g.when - fixture.kickoff_utc) <= GAME_WINDOW]
        if not games:
            return insufficient(mid, f"the stats source lists no {resolved['home']} v {resolved['away']} game "
                                     "near this kickoff")
        if len(games) > 1:
            return ambiguous(mid, f"More than one {resolved['home']} v {resolved['away']} game is listed "
                                  "near this kickoff; which one is it?",
                             [g.when.strftime("%Y-%m-%d %H:%M UTC") for g in games])

        seasons_used = [season]
        histories = {side: list(current.history.get(team, [])) for side, team in resolved.items()}
        if any(sum(1 for m in h if m.when < fixture.kickoff_utc) < LAST_N for h in histories.values()):
            previous = await self._league(client, cache, league, season - 1)
            seasons_used.append(season - 1)
            for side, team in resolved.items():
                histories[side] = previous.history.get(team, []) + histories[side]
        source = f"{NAME}:{league}:" + "+".join(str(s) for s in seasons_used)
        raw: list[Input] = []
        for side, played in histories.items():
            before = sorted((m for m in played if m.when < fixture.kickoff_utc), key=lambda m: m.when)
            raw.extend(team_inputs(side, before[-LAST_N:], source))
        return assess(mid, raw)
