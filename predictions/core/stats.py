"""What the bot knows about a fixture before any estimate is made. Standard library only."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

from predictions.core.inputs import GOALS_ONLY_REGISTRY, Input, check_inputs, missing_required, registry_for
from predictions.core.types import MatchId

READY = "ready"
INSUFFICIENT = "insufficient data"
AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class FixtureStats:
    """`status` READY: every required input is present, in range and has a source. INSUFFICIENT: the fixture
    is not evaluated further, `reasons` says why. AMBIGUOUS: `question` must be put to the owner first;
    nothing was guessed."""
    match_id: MatchId
    status: str
    inputs: tuple[Input, ...] = ()
    reasons: tuple[str, ...] = ()
    question: str = ""
    options: tuple[str, ...] = ()
    partial: tuple[Input, ...] = ()  # INSUFFICIENT only: the valid inputs the free sources did supply

    @property
    def ready(self) -> bool:
        return self.status == READY


def insufficient(match_id: MatchId, *reasons: str) -> FixtureStats:
    return FixtureStats(match_id, INSUFFICIENT, (), tuple(reasons))


def ambiguous(match_id: MatchId, question: str, options: Iterable[str]) -> FixtureStats:
    return FixtureStats(match_id, AMBIGUOUS, (), (question,), question, tuple(options))


def assess_goals_only(match_id: MatchId, raw_inputs: Iterable[Input]) -> FixtureStats:
    """Soccer without xG (Story 9.3): the goals-only registry applies (goals, and at least GOALS_ONLY_MIN_MATCHES
    matches). Too little history is "insufficient data", never an estimate."""
    return assess(match_id, raw_inputs, GOALS_ONLY_REGISTRY)


def assess(match_id: MatchId, raw_inputs: Iterable[Input], registry=None) -> FixtureStats:
    """Apply the registry; a missing required input makes the fixture "insufficient data"."""
    registry = registry_for(match_id.sport) if registry is None else registry
    kept, discarded = check_inputs(raw_inputs, registry)
    missing = missing_required(kept, registry)
    if missing:
        reasons = ["missing required input: " + ", ".join(missing)]
        reasons.extend(f"discarded {d}" for d in discarded)
        return FixtureStats(match_id, INSUFFICIENT, (), tuple(reasons), partial=kept)
    return FixtureStats(match_id, READY, kept)


def merge_inputs(stats: FixtureStats, extra: Iterable[Input]) -> FixtureStats:
    """A READY fixture with more inputs (corners, cards, referee). Checked by the registry like any input;
    the fixture's status never gets worse because an optional extra input was absent or discarded."""
    if not stats.ready:
        return stats
    kept, _ = check_inputs([*stats.inputs, *extra], registry_for(stats.match_id.sport))
    return FixtureStats(stats.match_id, READY, kept)


def missing_for(stats: FixtureStats) -> tuple[str, ...]:
    """The required inputs an INSUFFICIENT fixture still lacks (what the analyst may be asked for)."""
    return (missing_required(stats.partial, registry_for(stats.match_id.sport))
            if stats.status == INSUFFICIENT else ())


def merge_analyst(stats: FixtureStats, analyst_inputs: Iterable[Input]) -> FixtureStats:
    """Fill an INSUFFICIENT fixture from analyst inputs (Story 3.1). Everything goes through the registry:
    out-of-range or wrong-unit values are discarded and stay missing, and a free-source value always beats an
    analyst value for the same name. Only when every required input is then present is the fixture READY;
    otherwise it stays "insufficient data". A fixture that is not INSUFFICIENT is returned unchanged."""
    if stats.status != INSUFFICIENT:
        return stats
    result = assess(stats.match_id, [*stats.partial, *analyst_inputs])
    if result.ready:
        return result
    return replace(result, reasons=(*(f"after the analyst lookup: {r}" for r in result.reasons), *stats.reasons))


def add_reason(stats: FixtureStats, reason: str) -> FixtureStats:
    return replace(stats, reasons=(reason, *stats.reasons))


# ---- why a fixture was not evaluated (Story 9.2): plain-language groups for the reply. Matching is by keyword on the
# reasons the adapters give; anything unrecognised is "other", never hidden.
CAT_NOT_COVERED = "not covered by the stats source"
CAT_HISTORY = "too little history"
CAT_TEAM_UNKNOWN = "team not found in the stats source"
CAT_GAME_UNLISTED = "game not listed by the stats source"
CAT_AMBIGUOUS = "ambiguous team match (needs your choice)"
CAT_UNREADABLE = "stats source could not be read"
CAT_ANALYST = "the AI analyst could not fill the gap"
CAT_OTHER = "other data problem"
COMPETITION_LEVEL = frozenset({CAT_NOT_COVERED})  # the same reason for every game of a competition


def skip_category(stats: FixtureStats) -> str:
    """The plain-language group of a fixture that is not READY (see the CAT_ constants)."""
    if stats.status == AMBIGUOUS:
        return CAT_AMBIGUOUS
    text = (stats.reasons[0] if stats.reasons else "").lower()  # the first reason is the one the reply shows
    if "analyst" in text:
        return CAT_ANALYST
    if "not covered" in text or "stats source covers" in text or "no stats source is registered" in text:
        return CAT_NOT_COVERED
    if "could not be read" in text or "gave no answer" in text:
        return CAT_UNREADABLE
    if "too little history" in text or "missing required input" in text or "completed games" in text:
        return CAT_HISTORY
    if "is not in the stats source" in text or "was not found among" in text or "not found in" in text:
        return CAT_TEAM_UNKNOWN
    if "lists no" in text or "no data for" in text:
        return CAT_GAME_UNLISTED
    return CAT_OTHER
