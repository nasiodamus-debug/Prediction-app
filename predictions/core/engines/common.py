"""Building blocks shared by the MLB and NHL engines (Stories 6.3 to 6.5). Standard library only.

Two kinds of helper live here and nothing sport-specific does:
  * specifier and outcome reading for SportyBet offers ("total=1.5|periodnr=1", "Over 3.5", "Home (-1.5)");
  * discrete score distributions (Poisson, negative binomial) and the settlement of a line against one, so a
    whole line (refund on the exact number) and a half line are priced by the same function, and line shopping
    (one pick per market and side) with the "rule changed this pick" bookkeeping.
The floor is NOT here: engines import it from `core.model`.
"""
from __future__ import annotations

import math
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from typing import Mapping, NamedTuple, Sequence

from predictions.core.model import Estimate, Evaluation, percent
from predictions.core.types import MarketOffer

TAIL_MASS = 1e-12      # a distribution is cut once the remaining mass is below this
MAX_SUPPORT = 80       # and never longer than this many values (runs or goals)


# ---------------------------------------------------------------- reading SportyBet offers
def parse_specifier(specifier: str) -> dict[str, str]:
    """"total=1.5|periodnr=1" -> {"total": "1.5", "periodnr": "1"}. Order does not matter, junk parts are ignored."""
    found: dict[str, str] = {}
    for part in (specifier or "").split("|"):
        key, sep, value = part.partition("=")
        if sep and key.strip():
            found[key.strip()] = value.strip()
    return found


def spec_decimal(specifier: str, key: str) -> Decimal | None:
    value = parse_specifier(specifier).get(key)
    if value is None:
        return None
    try:
        number = Decimal(value)
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def spec_is_only(specifier: str, *keys: str) -> bool:
    """True when the specifier has exactly these keys (so a period or inning qualifier cannot be missed)."""
    return set(parse_specifier(specifier)) == set(keys)


def fmt_line(line: Decimal) -> str:
    return format(line.normalize(), "f")


def fmt_signed(line: Decimal) -> str:
    text = fmt_line(line)
    return text if text.startswith("-") or line == 0 else "+" + text


def find_outcome(offer: MarketOffer, side: str):
    """The one outcome whose name starts with `side` ("over", "home", "draw"), else None."""
    found = [o for o in offer.outcomes if o.name.strip().lower().startswith(side)]
    return found[0] if len(found) == 1 else None


# ---------------------------------------------------------------- discrete distributions
def _cut(pmf: list[float]) -> list[float]:
    total = sum(pmf)
    return [p / total for p in pmf] if total > 0 else pmf


def poisson_pmf(mean: float) -> list[float]:
    """P(X = k) for k = 0.. until the tail is negligible, normalised to sum to 1."""
    mean = max(mean, 1e-9)
    term = math.exp(-mean)
    pmf = [term]
    while len(pmf) < MAX_SUPPORT and (1.0 - sum(pmf)) > TAIL_MASS:
        term *= mean / len(pmf)
        pmf.append(term)
    return _cut(pmf)


def negbin_pmf(mean: float, dispersion: float) -> list[float]:
    """Negative binomial with this mean and variance = dispersion * mean (dispersion above 1; 1 or less is Poisson).
    Baseball runs are over-dispersed, hockey goals are close to Poisson."""
    mean = max(mean, 1e-9)
    if dispersion <= 1.0 + 1e-9:
        return poisson_pmf(mean)
    r = mean / (dispersion - 1.0)
    p0 = (r / (r + mean)) ** r
    ratio = mean / (r + mean)
    pmf = [p0]
    while len(pmf) < MAX_SUPPORT and (1.0 - sum(pmf)) > TAIL_MASS:
        k = len(pmf) - 1
        pmf.append(pmf[-1] * (k + r) / (k + 1) * ratio)
    return _cut(pmf)


Dist = Mapping[int, float]


def joint(pmf_home: Sequence[float], pmf_away: Sequence[float]):
    """(home, away) -> probability for two independent scores."""
    return {(h, a): ph * pa for h, ph in enumerate(pmf_home) for a, pa in enumerate(pmf_away)}


def margin_and_total(grid: Mapping[tuple[int, int], float], tie_home_win: float | None = None,
                     tie_extra: int = 0) -> tuple[dict[int, float], dict[int, float]]:
    """(margin distribution, total distribution) of the home-minus-away margin and the summed score.

    A tie is resolved when `tie_home_win` is given (overtime or extra innings): the home side wins the tied game
    with that probability by one, the away side otherwise, and `tie_extra` is added to the total (the tied score
    plus at least this much). Without it a tie stays a tie (margin 0)."""
    margin: dict[int, float] = {}
    total: dict[int, float] = {}
    for (h, a), p in grid.items():
        diff = h - a
        if diff == 0 and tie_home_win is not None:
            margin[1] = margin.get(1, 0.0) + p * tie_home_win
            margin[-1] = margin.get(-1, 0.0) + p * (1.0 - tie_home_win)
            key = h + a + tie_extra
            total[key] = total.get(key, 0.0) + p
        else:
            margin[diff] = margin.get(diff, 0.0) + p
            total[h + a] = total.get(h + a, 0.0) + p
    return margin, total


def line_probabilities(dist: Dist, line: Decimal) -> tuple[float, float, float] | None:
    """(P(over wins), P(under wins), P(push)) of an integer-valued outcome against a line.

    Half line: no push. Whole line N: exactly N is a push (the stake is refunded). Quarter lines are not supported."""
    if line % 1 == Decimal("0.5") or line % 1 == Decimal("-0.5"):
        over = sum(p for k, p in dist.items() if k > float(line))
        return over, 1.0 - over, 0.0
    if line % 1 == 0:
        n = int(line)
        over = sum(p for k, p in dist.items() if k > n)
        under = sum(p for k, p in dist.items() if k < n)
        return over, under, max(0.0, 1.0 - over - under)
    return None


def handicap_probabilities(margin: Dist, hcp: Decimal) -> tuple[float, float, float] | None:
    """(P(home covers), P(away covers), P(push)) when the HOME side is given `hcp` goals or runs: home covers when
    margin + hcp is above 0, pushes when it is exactly 0. A whole handicap refunds on the exact margin."""
    probs = line_probabilities(margin, -hcp)
    return None if probs is None else (probs[0], probs[1], probs[2])


# ---------------------------------------------------------------- priced picks and line shopping
class Priced(NamedTuple):
    kind: str          # winner, handicap, total, home_total, away_total, f5_total, ...
    side: str          # over / under / home / away / draw
    line: Decimal | None
    evaluation: Evaluation


def with_(evaluation: Evaluation, **changes) -> Evaluation:
    return replace(evaluation, **changes)


def describe(evaluation: Evaluation) -> str:
    return f"{evaluation.selection} @ {evaluation.odds:.2f} (estimate {percent(evaluation.estimate.probability)})"


def push_note(estimate: Estimate, text: str) -> str:
    if not estimate.push:
        return ""
    return (f"refund on {text}: win {percent(estimate.win)} + refund {percent(estimate.push)} = "
            f"success {percent(estimate.probability)}")


def rank(p: Priced):
    """A passing pick beats a rejected one; then the higher estimate wins (the owner's goal is success at the floor or
    better, not the longest price), then the higher EV, then the higher odds."""
    e = p.evaluation
    return (e.passed, e.estimate.probability, e.ev, e.odds)


def best(candidates: Sequence[Priced]) -> Priced | None:
    """The pick the owner would take among these: a passing one with the highest estimate (so the safest line that
    still has positive EV), else the closest to passing."""
    live = [p for p in candidates if not p.evaluation.insufficient]
    return max(live, key=rank) if live else None


def shop(candidates: Sequence[Priced]) -> list[Evaluation]:
    """One evaluation per (kind, side): the passing line with the highest estimate, or, when none passes, the closest
    rejection. A listing has 5 to 30 lines per market: the owner is shown one line, not every line. Choosing by estimate
    (not EV) keeps far tail lines from winning just because their price is long."""
    groups: dict[tuple[str, str], list[Priced]] = {}
    for p in candidates:
        groups.setdefault((p.kind, p.side), []).append(p)
    out: list[Evaluation] = []
    for key in sorted(groups):
        members = groups[key]
        real = [p for p in members if not p.evaluation.insufficient]
        chosen = best(real) if real else members[0]
        if chosen is not None:
            out.append(chosen.evaluation)
    return out


def converted(evaluations: Sequence[Evaluation], rule: str, original: str) -> list[Evaluation]:
    """A rule turned an original pick into these picks: show the rule and the pick it replaced."""
    return [with_(e, rule=rule, original=original) for e in evaluations]
