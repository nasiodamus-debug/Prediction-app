"""Attach stats to fixtures. The flow asks the source; deciding what counts as enough data is the core's job."""
from __future__ import annotations

from typing import Callable, Sequence

from predictions.core.ports import InputsSource, StatsFetchError, StatsSource
from predictions.core.stats import (AMBIGUOUS, CAT_NOT_COVERED, INSUFFICIENT, READY, FixtureStats, insufficient,
                                    merge_inputs, skip_category)
from predictions.core.types import Fixture, MatchId


async def gather_stats(fixtures: list[Fixture], source: StatsSource) -> dict[MatchId, FixtureStats]:
    """Stats for every fixture, keyed by MatchId. Raises StatsFetchError if the source cannot be read.
    A fixture the source does not answer for is "insufficient data", never assumed fine."""
    if not fixtures:
        return {}
    found = await source.fetch_stats(fixtures)
    result = {}
    for fixture in fixtures:
        stats = found.get(fixture.match_id)
        if stats is None or stats.match_id != fixture.match_id:
            stats = insufficient(fixture.match_id, "the stats source gave no answer for this game")
        result[fixture.match_id] = stats
    return result


def evaluable(fixtures: list[Fixture], stats: dict[MatchId, FixtureStats]) -> list[Fixture]:
    """Only fixtures whose required inputs are all present may go on to evaluation (Story 2.3)."""
    return [f for f in fixtures if f.match_id in stats and stats[f.match_id].ready]


def summarize(stats: dict[MatchId, FixtureStats]) -> tuple[int, int, int]:
    """(ready, insufficient data, needs the owner's choice)."""
    values = list(stats.values())
    return (sum(s.status == READY for s in values), sum(s.status == INSUFFICIENT for s in values),
            sum(s.status == AMBIGUOUS for s in values))


async def add_extra_inputs(fixtures: list[Fixture], stats: dict[MatchId, FixtureStats],
                           source: InputsSource) -> dict[MatchId, FixtureStats]:
    """Corner, card and referee inputs added to fixtures that are READY. Raises StatsFetchError if the source
    cannot be read; the caller decides what that means (the fixtures keep the stats they had)."""
    ready = [f for f in fixtures if f.match_id in stats and stats[f.match_id].ready]
    if not ready:
        return stats
    extra = await source.fetch_inputs(ready)
    return {mid: merge_inputs(entry, extra.get(mid, ())) for mid, entry in stats.items()}


class FirstCoveringStats:
    """A StatsSource made of several, in order of preference (Story 9.3). Every source is asked only about the fixtures the
    sources before it did not cover ("not covered by the stats source"), so a fixture covered by two sources is answered by
    the FIRST one (Understat, which has xG, before ESPN, which has goals only). A source that cannot be read leaves its
    fixtures to the next source; whatever then stays uncovered is "insufficient data: ... could not be read" (never "not
    covered"). The error is raised only when no source could be read at all."""

    def __init__(self, sources: Sequence[StatsSource], sanitize: Callable[[object], str] = str):
        self._sources = list(sources)
        self._sanitize = sanitize

    async def fetch_stats(self, fixtures: list[Fixture]) -> dict[MatchId, FixtureStats]:
        results: dict[MatchId, FixtureStats] = {}
        remaining = list(fixtures)
        errors: list[StatsFetchError] = []
        for source in self._sources:
            if not remaining:
                break
            try:
                answered = await source.fetch_stats(remaining)
            except StatsFetchError as error:
                errors.append(error)
                continue
            left = []
            for fixture in remaining:
                entry = answered.get(fixture.match_id)
                if entry is not None and entry.status == INSUFFICIENT and skip_category(entry) == CAT_NOT_COVERED:
                    left.append(fixture)
                    results.setdefault(fixture.match_id, entry)  # kept if no later source covers it
                elif entry is not None:
                    results[fixture.match_id] = entry
                else:
                    left.append(fixture)
            remaining = left
        if errors and len(errors) == len(self._sources):
            raise errors[0]
        for fixture in remaining:
            entry = results.get(fixture.match_id)
            if errors and (entry is None or skip_category(entry) == CAT_NOT_COVERED):
                results[fixture.match_id] = insufficient(
                    fixture.match_id, f"a stats service could not be read: {self._sanitize(errors[0])}")
        return results
