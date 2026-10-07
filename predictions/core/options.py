"""The closest available option that still clears the floor (FR-8, AD-4). Standard library only.

When a game is on the bookmaker but the selected option is not, the bot may OFFER one replacement: the option in the
same market and on the same side with the nearest line, provided its own estimate clears the probability floor (and its
expected value is positive, exactly as for any pick). An option below the floor is never offered. The owner decides.

Markets without a line (1X2) have no "nearest" option, so nothing is offered for them.
"""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Iterable

from predictions.core.betting import (AWAY_TOTAL_MARKET, BB_AWAY_TOTAL_MARKET, BB_FH_TOTAL_MARKET,
                                      BB_HOME_TOTAL_MARKET, BB_TOTAL_MARKET, HOME_TOTAL_MARKET, TOTALS_MARKET, Leg,
                                      UnsupportedLeg, leg_from_evaluation)
from predictions.core.engines import mlb as _mlb
from predictions.core.engines import nhl as _nhl
from predictions.core.market_rules import is_under, is_whole_or_quarter_line
from predictions.core.model import PROBABILITY_FLOOR, Evaluation

# Markets with one over/under line: the closest option is the nearest line on the same side. (Handicaps and
# winner markets are not here: a different handicap is a different bet, so nothing is offered for them.)
LINE_MARKETS = frozenset({TOTALS_MARKET, HOME_TOTAL_MARKET, AWAY_TOTAL_MARKET, BB_TOTAL_MARKET,
                          BB_HOME_TOTAL_MARKET, BB_AWAY_TOTAL_MARKET, BB_FH_TOTAL_MARKET,
                          # MLB (Story 6.3) and NHL (Stories 6.4, 6.5) over/under lines
                          _mlb.TOTAL_MARKET_ID, _mlb.HOME_TOTAL_MARKET_ID, _mlb.AWAY_TOTAL_MARKET_ID,
                          _mlb.F5_TOTAL_MARKET_ID, _nhl.TOTAL_OT_MARKET_ID, _nhl.PERIOD_TOTAL_MARKET_ID})
RULE_SUBSTITUTE = "Closest option"


def closest_option(original: Leg, candidates: Iterable[Evaluation]) -> Evaluation | None:
    """The candidate to offer in place of `original`, or None when nothing qualifies.

    A candidate qualifies when it is for the same match, market and side, on a different line, has an estimate at or
    above PROBABILITY_FLOOR and passed the gate (floor and positive expected value). Among those the nearest line wins;
    a tie goes to the higher estimate, then the lower line, so the choice is deterministic."""
    if original.market_id not in LINE_MARKETS or original.line is None or is_under(original):
        return None   # an Under is never replaced by a "closest" option either (AD-20)
    if is_whole_or_quarter_line(original):
        return None   # nor is a whole line (AD-21): the only choice is to skip
    best: tuple[tuple[Decimal, float, Decimal], Evaluation] | None = None
    for candidate in candidates:
        if (candidate.insufficient or not candidate.passed or candidate.estimate.probability < PROBABILITY_FLOOR
                or is_under(candidate) or is_whole_or_quarter_line(candidate)):
            continue
        if str(candidate.match_id) != str(original.match_id):
            continue
        try:
            leg = leg_from_evaluation(candidate)
        except UnsupportedLeg:
            continue
        if (leg.market_id != original.market_id or leg.side != original.side or leg.line is None
                or leg.line == original.line or is_under(leg) or is_whole_or_quarter_line(leg)):
            continue
        rank = (abs(leg.line - original.line), -candidate.estimate.probability, leg.line)
        if best is None or rank < best[0]:
            best = (rank, candidate)
    return best[1] if best else None


def as_substitute(original: Evaluation, chosen: Evaluation) -> Evaluation:
    """`chosen` marked as the replacement for `original`, so the list and the summary show what was swapped."""
    was = f"{original.selection} @ {original.odds:.2f}"
    return replace(chosen, rule=f"{RULE_SUBSTITUTE}: SportyBet does not offer {original.selection}",
                   original=was)
