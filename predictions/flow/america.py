"""Run an America request (MLB, NHL and American football, Stories 6.3 to 7.4): the same slate runner as Matchday, over several sports.

One request lists the games of every sport it covers in ONE reply and ONE list. Only the fixture source and the stats
source differ by sport; the gate, the list and everything after it are sport-agnostic (`ENGINES` is keyed by sport).
American football (Epic 7) was added by registering its sport in `app.py`'s AMERICA_SPORTS (a listing and a stats
source), its engine in `core/engines`, and its odds line in `telegram/render.py::SPORT_ODDS`. Nothing here changed.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Callable, Mapping

from predictions.core.ports import Analyst, FixtureSource, StatsFetchError, StatsSource
from predictions.core.stats import FixtureStats, insufficient
from predictions.core.types import Fixture, MatchId
from predictions.flow.evaluate import GateResult
from predictions.flow.slate_run import AMERICA, run_slate


class MultiSportFixtures:
    """A FixtureSource made of one source per sport, read at the same time (spec-14-1; each sport is a separate listing).
    If any sport's listing cannot be read completely nothing is returned (never a partial list), exactly as for one sport:
    the first failure ends the read and the other reads are cancelled. Games come back in the order of the sports."""

    def __init__(self, sources: Mapping[str, FixtureSource]):
        self._sources = dict(sources)

    async def fetch_fixtures(self) -> list[Fixture]:
        tasks = [asyncio.ensure_future(source.fetch_fixtures()) for source in self._sources.values()]
        try:
            lists = await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        return [fixture for games in lists for fixture in games]


class MultiSportStats:
    """A StatsSource made of one source per sport. A sport whose source cannot be read makes only ITS games "insufficient
    data" (the other sports are still evaluated); when every sport that has games fails, the error is raised as usual."""

    def __init__(self, sources: Mapping[str, StatsSource], sanitize: Callable[[object], str] = str):
        self._sources = dict(sources)
        self._sanitize = sanitize

    async def fetch_stats(self, fixtures: list[Fixture]) -> dict[MatchId, FixtureStats]:
        results: dict[MatchId, FixtureStats] = {}
        failures: list[StatsFetchError] = []
        used = 0
        for fixture in fixtures:
            if fixture.match_id.sport not in self._sources:
                results[fixture.match_id] = insufficient(fixture.match_id,
                                                         f"no stats source is registered for {fixture.match_id.sport}")
        for sport, source in self._sources.items():
            own = [f for f in fixtures if f.match_id.sport == sport]
            if not own:
                continue
            used += 1
            try:
                results.update(await source.fetch_stats(own))
            except StatsFetchError as error:
                failures.append(error)
                for fixture in own:
                    results[fixture.match_id] = insufficient(
                        fixture.match_id, f"the {sport} stats service could not be read: {self._sanitize(error)}")
        if used and len(failures) == used:
            raise failures[0]
        return results


async def run_america(argument: str, source: FixtureSource, sanitize: Callable[[object], str],
                      now: datetime | None = None, stats_source: StatsSource | None = None,
                      on_result: Callable[[GateResult], None] | None = None,
                      analyst: Analyst | None = None, analyst_off_note: str = "", day_cache=None) -> list[str]:
    """The message chunks for `America <dates>`. `source` lists the games of every covered sport (a MultiSportFixtures)
    and `stats_source` gives their stats with provenance (a MultiSportStats). Games that started, and games with no
    data, are handled exactly as for Matchday; when no pick passes the reply is NO VALID PLAY - PASS."""
    return await run_slate(AMERICA, argument, source, sanitize, now, stats_source, None, on_result, analyst,
                           analyst_off_note, day_cache=day_cache)
