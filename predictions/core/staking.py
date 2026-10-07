"""Stake parsing and the Kelly suggestion (FR-9, AD-4, AD-15). Standard library only.

What a "unit" is
    A unit is the owner's OWN standard stake size. The bot does not know the owner's bankroll and never invents it, so
    the suggestion is shown in units, never in money. By the common convention used here, 1 unit = 1% of the bankroll.

The formula
    Legs are treated as independent. The multiple's win probability is p = product of each leg's WIN probability
    (a refunded half-line counts as not winning, which is conservative). With total odds O the net odds are b = O - 1
    and the full Kelly fraction of the bankroll is f = (p * b - (1 - p)) / b.
    Full Kelly is far too aggressive for a model that is not calibrated yet, so the bot uses QUARTER Kelly:
        raw units = f * 100 * KELLY_MULTIPLIER / UNIT_PERCENT_OF_BANKROLL
    and clamps the result into the product range of 1.5 to 3.5 units (PRD FR-9), to one decimal. When there is no
    positive edge on the model's numbers the minimum is shown, flagged as "no positive edge".

The suggestion is only a suggestion. The stake the owner types is final; the bot never changes it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_UP, Decimal
from typing import Sequence

KELLY_MULTIPLIER = Decimal("0.25")            # quarter Kelly
UNIT_PERCENT_OF_BANKROLL = Decimal("1")       # 1 unit = 1% of the owner's bankroll (a convention, not a fact)
MIN_UNITS = Decimal("1.5")
MAX_UNITS = Decimal("3.5")
UNIT_STEP = Decimal("0.1")

_STAKE_PLAIN = re.compile(r"^\d+(\.\d{1,2})?$")
_STAKE_GROUPED = re.compile(r"^\d{1,3}(,\d{3})+(\.\d{1,2})?$")

DEFAULT_MAX_STAKE = Decimal("100000")   # NGN, the same units the owner types; the app reads MAX_STAKE and passes it down

MSG_BAD_STAKE = ("That is not a valid stake. Send a positive number with at most two decimals, "
                 "for example 500 or 1250.50.")


class StakeError(ValueError):
    """The text is not a valid stake. The message is safe to show the owner."""


def msg_over_max(max_stake: Decimal) -> str:
    return (f"That stake is above the limit of {format(Decimal(max_stake), ',f')}. Send a smaller stake. To raise the "
            "limit, set MAX_STAKE in config.env and restart the bot.")


def parse_max_stake(raw) -> Decimal:
    """The MAX_STAKE setting. Unset or empty keeps the default. A value that is present but is not a positive number
    raises ValueError (the bot must not start with a limit the owner did not intend). The message never echoes the value."""
    if raw is None or not str(raw).strip():
        return DEFAULT_MAX_STAKE
    try:
        value = Decimal(str(raw).strip().replace(",", ""))
    except Exception:
        value = None
    if value is None or not value.is_finite() or value <= 0:
        raise ValueError("MAX_STAKE in config.env is not a positive number (for example 100000). "
                         "Fix it or remove the line to use the default of 100,000, then start the bot again.")
    return value


def parse_stake(text, max_stake: Decimal | None = DEFAULT_MAX_STAKE) -> Decimal:
    """Parse the owner's typed stake ONCE into a Decimal: a positive number with at most two decimals (AD-15), not
    above `max_stake` (None means no upper limit; the app always passes one).
    Anything else (empty, zero, negative, letters, exponents, three decimals, unicode digits) is rejected."""
    if not isinstance(text, str):
        raise StakeError(MSG_BAD_STAKE)
    cleaned = text.strip()
    if not cleaned.isascii() or not (_STAKE_PLAIN.match(cleaned) or _STAKE_GROUPED.match(cleaned)):
        raise StakeError(MSG_BAD_STAKE)
    value = Decimal(cleaned.replace(",", ""))
    if value <= 0:
        raise StakeError(MSG_BAD_STAKE)
    if max_stake is not None and value > Decimal(max_stake):
        raise StakeError(msg_over_max(max_stake))
    return value


_CENT = Decimal("0.01")


def potential_winnings(total_odds, stake) -> Decimal:
    """What the bet pays back if it wins: total odds x stake, INCLUDING the stake, in Decimal (AD-15) and rounded DOWN to
    two decimals so the screen never overstates. It is an estimate: SportyBet may adjust for bonuses or limits."""
    odds, amount = Decimal(total_odds), Decimal(stake)
    if not odds.is_finite() or not amount.is_finite() or odds <= 0 or amount <= 0:
        raise ValueError("total odds and stake must be positive numbers")
    return (odds * amount).quantize(_CENT, rounding=ROUND_DOWN)


def potential_profit(total_odds, stake) -> Decimal:
    """Potential winnings minus the stake (what is gained if the bet wins), also never overstated."""
    return potential_winnings(total_odds, stake) - Decimal(stake).quantize(_CENT, rounding=ROUND_UP)


def format_money(value) -> str:
    """Thousands separators and two decimals, like the stake on the summary ('1,250.50')."""
    return f"{Decimal(value):,.2f}"


@dataclass(frozen=True)
class StakeSuggestion:
    units: Decimal            # what to show: always within MIN_UNITS..MAX_UNITS
    kelly_fraction: Decimal   # the full Kelly fraction f (may be zero or negative)
    raw_units: Decimal        # before clamping
    bound: str                # "min" or "max" when the raw value was clamped, else ""
    edge: bool                # False when the model sees no positive edge on this multiple


def suggest_stake(win_probabilities: Sequence[float], total_odds: Decimal) -> StakeSuggestion:
    """A quarter-Kelly suggestion in units for a multiple of independent legs. See the module docstring."""
    if not win_probabilities:
        raise ValueError("a suggestion needs at least one leg")
    odds = Decimal(total_odds)
    if not odds.is_finite() or odds <= 1:
        raise ValueError("total odds must be above 1")
    p = Decimal(1)
    for probability in win_probabilities:
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError("a probability must be between 0 and 1")
        p *= Decimal(str(probability))
    b = odds - 1
    fraction = (p * b - (1 - p)) / b
    raw = fraction * 100 * KELLY_MULTIPLIER / UNIT_PERCENT_OF_BANKROLL
    if fraction <= 0:
        return StakeSuggestion(MIN_UNITS, fraction, raw, "min", False)
    units = raw
    bound = ""
    if units < MIN_UNITS:
        units, bound = MIN_UNITS, "min"
    elif units > MAX_UNITS:
        units, bound = MAX_UNITS, "max"
    return StakeSuggestion(units.quantize(UNIT_STEP), fraction, raw, bound, True)
