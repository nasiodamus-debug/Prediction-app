"""Estimate, Evaluation and the validation gate (AD-2, AD-4). Standard library only.

Only deterministic code produces an Estimate, and only from provenance inputs. The floor is defined here
and nowhere else.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_FLOOR, Decimal
from typing import Protocol, Sequence

from predictions.core.inputs import Input
from predictions.core.types import Fixture, MarketOffer, MatchId

PROBABILITY_FLOOR = 0.70  # the one definition of the floor; everything else imports it (owner raised it from 65% on 2026-10-03)
PREVIOUS_FLOOR = 0.65  # the floor until 2026-10-03; calibration keeps a 65-70% band so earlier graded picks still count
PROBABILITY_DECIMALS = 3  # estimates are kept to 0.1%, so what is shown is what is gated

REASON_PASSED = "passed"


def _floor_dp(value: float) -> Decimal:
    # Round DOWN: a probability just under the floor must never be lifted over it by rounding.
    step = Decimal(1).scaleb(-PROBABILITY_DECIMALS)
    return Decimal(str(value)).quantize(step, rounding=ROUND_FLOOR)


@dataclass(frozen=True)
class Estimate:
    """A probability from 0 to 1, rounded to PROBABILITY_DECIMALS, with the inputs it came from.

    For a push-protected pick (a whole goal line, where one exact result refunds the stake) `probability`
    is the SUCCESS probability P(win) + P(push) and `push` is the refunded part, so P(win) is
    `probability - push`. That is the one definition the floor is applied to (see spec-2-4 Open Issues).
    """
    probability: float
    inputs: tuple[Input, ...] = ()
    model: str = ""
    push: float = 0.0

    def __post_init__(self):
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("an estimate must be between 0 and 1")
        if not 0.0 <= self.push <= self.probability:
            raise ValueError("the push probability must be between 0 and the estimate")
        object.__setattr__(self, "probability", float(_floor_dp(self.probability)))
        object.__setattr__(self, "push", float(_floor_dp(self.push)))

    @classmethod
    def push_protected(cls, p_win: float, p_push: float, inputs: tuple[Input, ...] = (),
                       model: str = "") -> "Estimate":
        """Success = P(win) + P(push), each rounded down first so the sum can never round up."""
        win, push = _floor_dp(min(p_win, 1.0)), _floor_dp(min(p_push, 1.0))
        return cls(min(float(win + push), 1.0), inputs, model, float(push))

    @property
    def win(self) -> float:
        return float(Decimal(str(self.probability)) - Decimal(str(self.push)))


def expected_value(probability: float, odds: Decimal, push: float = 0.0) -> Decimal:
    """EV per unit staked = P(win) * (odds - 1) - P(lose).

    `probability` is the success probability (win + push); `push` is the refunded part. A refund returns
    the stake, so it adds nothing and costs nothing. With push = 0 this is probability * odds - 1."""
    win = Decimal(str(probability)) - Decimal(str(push))
    lose = 1 - Decimal(str(probability))
    return win * (Decimal(odds) - 1) - lose


def percent(value) -> str:
    return f"{float(value) * 100:.1f}%"


def signed_percent(value) -> str:
    return f"{float(value) * 100:+.1f}%"


@dataclass(frozen=True)
class Evaluation:
    """One pick judged by the gate. `passed` needs estimate >= floor AND EV > 0."""
    match_id: MatchId
    market: str        # for example "Over/Under 2.5"
    selection: str     # for example "Over 2.5"
    odds: Decimal
    estimate: Estimate
    ev: Decimal
    passed: bool
    reason: str
    rule: str = ""              # the Optimization rule applied, "" when none
    original: str = ""          # the pick as first offered, when a rule changed it
    notes: tuple[str, ...] = ()  # plain-text lines shown with the pick (inputs, refund terms)
    insufficient: bool = False   # rejected for missing data: there is no estimate, `estimate` is a placeholder


@dataclass(frozen=True)
class FixturePass:
    """A fixture the engine refuses to play at all (NO VALID PLAY - PASS), with the reason."""
    match_id: MatchId
    rule: str
    reason: str


def validate(estimate: Estimate, odds: Decimal) -> tuple[bool, Decimal, str]:
    """The floor and EV gate. Returns (passed, ev, reason)."""
    ev = expected_value(estimate.probability, odds, estimate.push)
    if estimate.probability < PROBABILITY_FLOOR:
        detail = f" (win {percent(estimate.win)} + refund {percent(estimate.push)})" if estimate.push else ""
        return False, ev, f"estimate {percent(estimate.probability)}{detail} is below the {PROBABILITY_FLOOR:.0%} floor"
    if ev <= 0:
        return False, ev, f"expected value {signed_percent(ev)} is not positive at odds {odds:.2f}"
    return True, ev, REASON_PASSED


class Engine(Protocol):
    def evaluate(self, fixture: Fixture, inputs: Sequence[Input],
                 market_options: Sequence[MarketOffer]) -> list[Evaluation]:
        """Judge each pick the engine understands among `market_options`. Deterministic."""
        ...

    def screen(self, fixture: Fixture, inputs: Sequence[Input]) -> FixturePass | None:
        """A fixture-level NO VALID PLAY - PASS, or None when the fixture may be evaluated."""
        ...

    def understands(self, offer: MarketOffer) -> bool:
        """Whether the engine has rules for this market (Review, Story 4.2). False means "no rules for this market"."""
        ...

    def evaluate_exact(self, fixture: Fixture, inputs: Sequence[Input], offer: MarketOffer) -> list[Evaluation]:
        """Judge exactly the outcomes in `offer` at the odds it carries, with no optimization or substitution.
        Every sport added later (Epics 6 and 7) implements this and registers in `core.engines.ENGINES`."""
        ...
