"""The validation stage: evaluate every ready fixture and split picks into passed and rejected."""
from __future__ import annotations

from dataclasses import dataclass

from predictions.core.analyst import RULE_ANALYST_FLAG, AnalystFlag, describe_flags
from predictions.core.engines import ENGINES
from predictions.core.market_rules import is_under, is_whole_or_quarter_line, reject_under, reject_whole_line
from predictions.core.model import Evaluation, FixturePass
from predictions.core.stats import FixtureStats
from predictions.core.types import Fixture, MatchId


@dataclass(frozen=True)
class GateResult:
    passed: tuple[Evaluation, ...]
    rejected: tuple[Evaluation, ...]
    evaluated: int  # fixtures that were evaluated
    skipped: int    # fixtures not ready (insufficient data or awaiting a choice)
    fixtures: dict[MatchId, Fixture]
    passes: tuple[FixturePass, ...] = ()  # fixtures refused outright (derby, low-block cup tie)
    footer: tuple[str, ...] = ()  # Story 10.1: the short lines (games ignored, analyst cost, stats warning) for the end of a passing list
    under_rejected: tuple[Evaluation, ...] = ()  # AD-20: Under evaluations refused (never in `rejected`, never listed one by one)
    whole_rejected: tuple[Evaluation, ...] = ()  # AD-21: whole/quarter-line evaluations refused (never in `rejected`, never listed one by one)
    unpriced_games: tuple[MatchId, ...] = ()  # spec 11-1: games with a corners or cards market that could not be priced (no data): not listed as rejections

    @property
    def whole_ignored(self) -> int:
        return len(self.whole_rejected)

    @property
    def under_ignored(self) -> int:
        return len(self.under_rejected)


def _best_per_game(passed: list[Evaluation], fixtures: list[Fixture]) -> list[Evaluation]:
    """Spec 11-1: an engine that sets `max_plays_per_game` (soccer: 2) keeps only its best passing plays of each game, ranked
    by probability of success, then higher EV, then selection text (the same order the list advises in). Other engines are
    untouched."""
    limits = {}
    for fixture in fixtures:
        engine = ENGINES.get(fixture.match_id.sport)
        limit = getattr(engine, "max_plays_per_game", None)
        if limit:
            limits[fixture.match_id] = limit
    if not limits:
        return passed
    grouped: dict[MatchId, list[Evaluation]] = {}
    for ev in passed:
        grouped.setdefault(ev.match_id, []).append(ev)
    kept: list[Evaluation] = []
    for match_id, plays in grouped.items():
        if match_id in limits:
            plays = sorted(plays, key=lambda e: (-e.estimate.probability, -e.ev, e.selection))[:limits[match_id]]
        kept.extend(plays)
    return kept


def run_gate(fixtures: list[Fixture], stats: dict[MatchId, FixtureStats],
             flags: dict[MatchId, tuple[AnalystFlag, ...]] | None = None) -> GateResult:
    """`flags` are analyst context flags (Story 3.2). A flagged fixture can only get STRICTER: it becomes a
    NO VALID PLAY - PASS and none of its picks is evaluated. Flags never touch an estimate."""
    flags = flags or {}
    passed, rejected, passes, evaluated, skipped = [], [], [], 0, 0
    unders: list[Evaluation] = []
    wholes: list[Evaluation] = []
    unpriced: dict[MatchId, None] = {}
    for fixture in fixtures:
        entry = stats.get(fixture.match_id)
        engine = ENGINES.get(fixture.match_id.sport)
        if engine is None:
            skipped += 1
            continue
        ready = entry is not None and entry.ready
        # Screening comes first: a derby is passed even when the stats are missing (it needs none).
        refusal = engine.screen(fixture, entry.inputs if ready else ())
        if refusal is not None:
            passes.append(refusal)
            evaluated += ready
            continue
        if refusal is None and flags.get(fixture.match_id):
            refusal = FixturePass(fixture.match_id, RULE_ANALYST_FLAG, describe_flags(flags[fixture.match_id]))
            passes.append(refusal)
            evaluated += ready
            continue
        if not ready:
            skipped += 1
            continue
        evaluated += 1
        for ev in engine.evaluate(fixture, entry.inputs, fixture.markets):
            if is_under(ev):   # owner rule (AD-20): the one central gate, whatever an engine returned
                unders.append(reject_under(ev))
                continue
            if is_whole_or_quarter_line(ev):   # owner rule (AD-21): no whole or quarter line, whatever an engine returned
                wholes.append(reject_whole_line(ev))
                continue
            if ev.insufficient and getattr(engine, "quiet_insufficient", False):
                unpriced[fixture.match_id] = None   # spec 11-1: a play with no data behind it is simply not considered
                continue
            (passed if ev.passed else rejected).append(ev)
    passed = _best_per_game(passed, fixtures)
    passed.sort(key=lambda e: (-e.estimate.probability, str(e.match_id), e.selection))
    rejected.sort(key=lambda e: (-e.estimate.probability, str(e.match_id), e.selection))
    return GateResult(tuple(passed), tuple(rejected), evaluated, skipped,
                      {f.match_id: f for f in fixtures}, tuple(passes), under_rejected=tuple(unders),
                      whole_rejected=tuple(wholes),
                      unpriced_games=tuple(unpriced))
