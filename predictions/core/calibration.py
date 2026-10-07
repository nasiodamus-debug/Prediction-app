"""Calibration of the model's estimates against SportyBet's own graded results. Pure functions, standard library only.

Only PLACED legs that SportyBet has graded are counted. A void or a push is counted apart: it is neither a win nor a loss,
and the win rate is wins / (wins + losses). The estimate stored for a push-protected pick is the SUCCESS probability
P(win) + P(push) (see core/model.py), so a pushed leg is listed as a push and never as a win.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from predictions.core.model import PREVIOUS_FLOOR
from predictions.core.results import LOSS, PUSH, VOID, WIN

MIN_BAND_SAMPLE = 20   # a band with fewer decided legs (wins + losses) is too small to judge
# (label, lower bound inclusive, upper bound exclusive); estimates below the floor cannot pass the gate but are kept honestly.
BANDS = (("under 65%", 0.0, PREVIOUS_FLOOR), ("65-70%", PREVIOUS_FLOOR, 0.70), ("70-75%", 0.70, 0.75), ("75-80%", 0.75, 0.80),
         ("80%+", 0.80, 1.0000001))


@dataclass(frozen=True)
class GradedLeg:
    probability: float   # the stored estimate
    result: str          # WIN, LOSS, VOID or PUSH


@dataclass(frozen=True)
class BandSummary:
    label: str
    legs: int
    wins: int
    losses: int
    voids: int
    pushes: int
    average_estimate: float | None   # mean stored estimate of the band's legs

    @property
    def decided(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float | None:
        return self.wins / self.decided if self.decided else None

    @property
    def too_small(self) -> bool:
        return self.decided < MIN_BAND_SAMPLE


def band_label(probability: float) -> str:
    for label, low, high in BANDS:
        if low <= probability < high:
            return label
    return BANDS[-1][0]


def summarise(legs: Iterable[GradedLeg]) -> list[BandSummary]:
    """One BandSummary per band that holds at least one leg, lowest band first."""
    grouped: dict[str, list[GradedLeg]] = {}
    for leg in legs:
        if leg.result in (WIN, LOSS, VOID, PUSH):
            grouped.setdefault(band_label(leg.probability), []).append(leg)
    out = []
    for label, _, _ in BANDS:
        rows = grouped.get(label)
        if rows:
            out.append(BandSummary(label, len(rows), sum(r.result == WIN for r in rows),
                                   sum(r.result == LOSS for r in rows), sum(r.result == VOID for r in rows),
                                   sum(r.result == PUSH for r in rows),
                                   sum(r.probability for r in rows) / len(rows)))
    return out
