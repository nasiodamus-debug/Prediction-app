"""Goals-only team stats from football-data.co.uk's season CSVs, behind the StatsSource port (the last source in the soccer chain).

Understat (xG, six leagues) answers first and ESPN (goals only) second; this source is asked only about what neither covers, which is
mostly the lower divisions SportyBet lists (England's Championship, League One, League Two and National League, the Scottish lower
leagues, Spain's LaLiga 2, Italy's Serie B, France's Ligue 2, Germany's 2. Bundesliga). It reads the same files as the corners and cards
source (adapters/footballdata/stats.py: two season files per league, remembered for a few hours): a team's inputs are matches,
goals_for and goals_against over its last LAST_N completed games before kickoff, and fewer than the goals-only minimum (8) is
"insufficient data". The engine prices such a fixture with its goals-only model and labels it so. No key, no login, no retries.
"""
from __future__ import annotations

import asyncio

import httpx

from predictions.adapters.footballdata import stats as fd
from predictions.core.identity import resolve_team
from predictions.core.inputs import GOALS_ONLY_MIN_MATCHES, Input
from predictions.core.ports import StatsFetchError
from predictions.core.stats import FixtureStats, assess_goals_only, insufficient
from predictions.core.types import Fixture, MatchId, normalize_team

NAME = "footballdata"
NOT_COVERED = "this competition is not covered by the stats source"
LAST_N = fd.LAST_N


def parse_goals(text: str) -> list[tuple]:
    """(day, home, away, home goals, away goals) for every played row; a row without both scores is skipped, never read as 0-0."""
    games = []
    for row in fd._rows(text):
        scores = [fd._int(row.get(k)) for k in ("FTHG", "FTAG")]
        if any(s is None for s in scores):
            continue
        games.append((fd._day(row["Date"]), row["HomeTeam"].strip(), row["AwayTeam"].strip(), scores[0], scores[1]))
    return sorted(games, key=lambda g: g[0])


def team_inputs(side: str, team: str, games: list[tuple], source: str) -> list[Input]:
    n = len(games)
    if n == 0:
        return []
    gf = sum(h if home == team else a for _, home, _, h, a in games) / n
    ga = sum(a if home == team else h for _, home, _, h, a in games) / n
    return [Input(f"{side}.matches", float(n), "matches", source),
            Input(f"{side}.goals_for", round(gf, 4), "goals per match", source),
            Input(f"{side}.goals_against", round(ga, 4), "goals per match", source)]


class FootballDataGoalsStats:
    """StatsSource backed by football-data.co.uk. Covers only the competitions in fd.LEAGUES."""

    def __init__(self, client: httpx.AsyncClient | None = None):
        self._reader = fd.FootballDataInputs(client)
        self._client = client

    async def fetch_stats(self, fixtures: list[Fixture]) -> dict[MatchId, FixtureStats]:
        try:
            if self._client is not None:
                return await asyncio.wait_for(self._run(self._client, fixtures), fd.OVERALL_TIMEOUT_SECONDS)
            async with httpx.AsyncClient(timeout=fd.TIMEOUT_SECONDS, headers=fd._HEADERS, follow_redirects=True) as client:
                return await asyncio.wait_for(self._run(client, fixtures), fd.OVERALL_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raise StatsFetchError("football-data.co.uk took too long to answer") from None

    async def _run(self, client, fixtures: list[Fixture]) -> dict[MatchId, FixtureStats]:
        seasons: dict[tuple[str, str], list[tuple]] = {}
        results: dict[MatchId, FixtureStats] = {}
        for fixture in fixtures:
            mid = fixture.match_id
            code = fd.LEAGUES.get(normalize_team(fixture.competition))
            if mid.sport != "soccer" or code is None:
                results[mid] = insufficient(mid, NOT_COVERED)
                continue
            season = fd.season_code(fixture.kickoff_utc)
            games: list[tuple] = []
            for s in (fd.previous_code(season), season):
                if (code, s) not in seasons:
                    seasons[(code, s)] = parse_goals(await self._reader._get(client, fd.SEASON_URL.format(season=s, code=code)))
                games += seasons[(code, s)]
            teams = {g[1] for g in games} | {g[2] for g in games}
            resolved, problems = {}, []
            for side, name in (("home", fixture.home), ("away", fixture.away)):
                match = resolve_team(name, teams, fd.FD_ALIASES)
                if match.status != "matched":
                    problems.append(f"the {side} team '{name}' is not in the stats source's {code} list")
                else:
                    resolved[side] = match.team
            if problems:
                results[mid] = insufficient(mid, *problems)
                continue
            day = fixture.kickoff_utc.date()
            source = f"{NAME}:goals-only:{code}:{season}"
            raw: list[Input] = []
            for side, team in resolved.items():
                before = [g for g in games if team in (g[1], g[2]) and g[0] < day][-LAST_N:]
                if len(before) < GOALS_ONLY_MIN_MATCHES:
                    problems.append(f"too little history: {team} has {len(before)} completed match(es) before this game on file, "
                                    f"at least {GOALS_ONLY_MIN_MATCHES} are needed (goals only, no xG)")
                else:
                    raw += team_inputs(side, team, before, source)
            results[mid] = insufficient(mid, *problems) if problems else assess_goals_only(mid, raw)
        return results
