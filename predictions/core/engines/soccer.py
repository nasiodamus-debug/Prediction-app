"""Soccer engine (Stories 2.3, 2.4, 2.5). Standard library only. Everything here is deterministic.

Model: each side's scoring rate blends its xG and actual goals; the expected goals for a side average
its attack with the opponent's defence. Goals of the two sides are independent Poisson variables, so the
total is Poisson with the summed rate. From that one model come Over/Under totals (half and whole lines),
1X2, and team totals. Corners and cards use their own count model (negative binomial) from their own inputs.

Pipeline per fixture: screen (derby / cup pass) -> Validation (floor + EV, `core.model.validate`) ->
Optimization (1X2 swap). Nothing here defines the floor; it is imported.

No whole-line or quarter-line play is ever offered (owner rule, AD-21, spec-13-1): `evaluate` prices half lines only. The Asian goal-line
shield is gone. Review (`evaluate_exact`) still prices a whole line found on a booking code: a push-protected selection has the
success probability P(win) + P(push), the floor is applied to it, and EV = P(win) * (odds - 1) - P(lose). See spec-2-4.
"""
from __future__ import annotations

import math
import re
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from typing import Callable, Sequence

from predictions.core.engines.soccer_tables import find_derby, is_cup
from predictions.core.inputs import GOALS_ONLY_MIN_MATCHES, Input
from predictions.core.market_rules import (SIDES_PLAYED, offers_without_whole_lines, reject_under, without_unders,
                                           without_whole_lines)
from predictions.core.model import Estimate, Evaluation, FixturePass, PROBABILITY_FLOOR, percent, validate
from predictions.core.types import Fixture, MarketOffer, normalize_team

# ---- model (Story 2.3) ----
MODEL_NAME = "poisson-totals-v1"
XG_WEIGHT = 0.7                  # weight of xG against actual goals in a team's rate
GOALS_WEIGHT = 1.0 - XG_WEIGHT
MIN_EXPECTED_GOALS = 0.3         # total expected goals is kept inside these bounds
MAX_EXPECTED_GOALS = 6.0
FIRST_HALF_SHARE = 0.45         # share of a game's goals scored in the first half (league average, an ASSUMPTION: no half data is stored)
SECOND_HALF_SHARE = 0.55        # ... and in the second half (used by the second-half and both-halves markets, Review only)
GRID_MAX_GOALS = 25              # scores beyond this are ignored in 1X2 (mass is negligible)
TOTALS_MARKET_ID = "18"          # SportyBet Over/Under (match goals); lists half AND whole lines (only half lines are offered, AD-21)
RESULT_MARKET_ID = "1"           # SportyBet 1X2
DOUBLE_CHANCE_MARKET_ID = "10"   # SportyBet Double Chance: "Home or Draw", "Home or Away", "Draw or Away" (probed 2026-10-01, spec-11-1)
CORNERS_MARKET_ID = "166"        # SportyBet "Corners - Over/Under" (match total corners; only on the event endpoint, spec-11-1)
ASIAN_HANDICAP_MARKET_ID = "16"  # SportyBet full-time Asian Handicap (specifier hcp=<home handicap>; the 1st half one is 66). REVIEW ONLY; id taken from the 1st-half sample's numbering, confirm on the first real code
CARDS_MARKET_ID = "139"          # SportyBet "Bookings - Over/Under" (match total bookings = cards; only on the event endpoint)
HOME_TOTAL_MARKET_ID = "19"      # SportyBet Home team Over/Under
AWAY_TOTAL_MARKET_ID = "20"      # SportyBet Away team Over/Under
MIN_LINE = Decimal("0.5")
MAX_LINE = Decimal("6.5")
LINE_STEP = Decimal("1")
LINE_OFFSET = Decimal("0.5")

# ---- Story 9.3: GOALS-ONLY path (no xG), for competitions whose only source is ESPN ----
# Used when a fixture has goals but no xG inputs. Deterministic and deliberately MORE CONSERVATIVE than the xG model:
#  * a bigger sample is required (GOALS_ONLY_MIN_MATCHES, imported from the input registry: 8 matches, not 5);
#  * each team's goals for and against are SHRUNK toward a league-average prior worth GOALS_ONLY_PRIOR_MATCHES matches,
#    so with a 10-match sample the prior carries half the weight (an xG rate gets no shrinkage).
# The prior (goals per team per match) is a judgement, about the average of senior international and club-friendly
# football; it is not fitted. The floor and the EV gate are applied by `validate` exactly as for the xG model.
GOALS_ONLY_MODEL_NAME = "goals-only-poisson-v1"
GOALS_ONLY_PRIOR_GOALS = 1.30
GOALS_ONLY_PRIOR_MATCHES = 10.0
GOALS_ONLY_LABEL = "goals-only estimate (no xG)"
# "Too good to be true" cap (AD-4). The goals-only model has no opponent-strength adjustment, no home advantage and no
# calibration, so a big gap to the market is far more likely a model artifact than a real edge (the 2026-10-01 live probe
# passed 50 picks on 23 games with EVs up to +58%). A goals-only pick whose EV per unit staked is ABOVE this value is
# rejected as unreliable, in addition to the floor and the positive-EV gate (exactly this value still passes). It applies
# to every goals-only evaluation: Matchday picks, swap-converted picks, Review (`evaluate_exact`),
# team totals and the corners/cards that lean on goals-only inputs. The xG model is not capped. 0.20 is a judgement.
GOALS_ONLY_MAX_EV = 0.20
GOALS_ONLY_CAP_REASON = ("goals-only estimate beats the market by more than {cap:.0%} EV: treated as unreliable "
                         "(rough model, no opponent strength)")
XG_INPUT_NAMES = ("home.xg_for", "home.xg_against", "away.xg_for", "away.xg_against")

# ---- Spec 11-1: the five play types Matchday considers (owner request of 2026-10-01) ----
# (a) total goals OVER half lines of 1.5 and above (no whole or quarter line, AD-21); (b) match winner
# (Home or Away from the 1X2; a Draw alone is not a play); (c) double chance Home or Draw / Away or Draw; (d) total yellow cards
# (SportyBet "Bookings") OVER 2.5 and above; (e) total corners OVER 7.5 and above. Everything else is not considered in Matchday.
# At most the two plays with the highest probability of success are kept per game (applied by the gate, see run_gate).
MATCHDAY_MIN_GOAL_LINE = Decimal("1.5")
MATCHDAY_MIN_CARDS_LINE = Decimal("2.5")
MATCHDAY_MIN_CORNERS_LINE = Decimal("7.5")
MATCHDAY_MAX_PLAYS_PER_GAME = 2
DOUBLE_CHANCE_CHOICES = (("Home or Draw", "home or draw"), ("Away or Draw", "draw or away"))  # (our text, SportyBet's outcome)

# ---- Story 2.4: 1X2 swap ----
SWAP_MIN_ODDS = Decimal("2.00")                       # only a selection ABOVE this is swapped
SWAP_MIN_ESTIMATE = 0.50                              # ...and only if it is a plausible pick: a long shot is
                                                      # rejected, not turned into a different bet
SWAP_TOTAL_LINES = (Decimal("2.5"),)                  # Over 2.5 match goals (the Over 2.0 target is gone: no whole line, AD-21)
# (Spec 11-1: a 1X2 swap may only land on an Over match-goals line; a team total is never produced in Matchday.)

# ---- Story 2.4: derby and cup pass ----
# Derbies come from a static table (soccer_tables.DERBIES). A cup tie is a PROXY: the competition's name has a
# cup word (soccer_tables.CUP_WORDS). "Strong low block" is a PROXY too: a side whose PPDA (opponent passes
# per defensive action, higher = presses less = sits deeper) is at least LOW_BLOCK_PPDA, in a game where the
# model expects at most CUP_TIGHT_MAX_TOTAL_GOALS goals. With no PPDA input the low block cannot be ruled
# out, so the tie is passed when CUP_PASS_WITHOUT_PPDA is true.
LOW_BLOCK_PPDA = 13.0
CUP_TIGHT_MAX_TOTAL_GOALS = 2.4
CUP_PASS_WITHOUT_PPDA = True

# ---- Story 2.5: cards ----
CARDS_MIN_REFEREE_AVG = 3.5          # the referee must average at least this many cards per game
CARDS_MAX_EXPECTED_GOAL_DIFF = 2.0   # blowout risk: expected goal difference must be UNDER this
REFEREE_MIN_GAMES = 8                # a referee average over fewer games is not enough data
CARDS_REFEREE_WEIGHT = 0.5           # expected cards = weight * referee avg + (1 - weight) * both teams' rates
CARDS_DISPERSION = 1.2               # variance / mean of cards per game (judgement, not fitted)
MIN_EXPECTED_CARDS = 1.0
MAX_EXPECTED_CARDS = 12.0
CARDS_MODEL = "negbin-cards-v1"

# ---- Story 2.5: corners ----
CORNERS_DISPERSION = 1.3             # variance / mean of total corners per game (judgement, not fitted)
MIN_EXPECTED_CORNERS = 4.0
MAX_EXPECTED_CORNERS = 20.0
CORNERS_MODEL = "negbin-corners-v1"
TEAM_CORNERS_DISPERSION = 1.15       # variance / mean of ONE team's corners per game (judgement, not fitted; below the match figure)
MIN_EXPECTED_TEAM_CORNERS = 2.0
MAX_EXPECTED_TEAM_CORNERS = 10.0

# Corners and cards lines are recognised by market ID (166 and 139, spec 11-1: both only appear on SportyBet's per-event
# endpoint, not in the listing) and, for a booking code that spells them differently, by market NAME. Names with these words are
# other markets (halves, teams, handicaps, points).
PROP_LINE_MIN = Decimal("0.5")
PROP_LINE_MAX = Decimal("20.5")
PROP_EXCLUDED_WORDS = ("half", "1st", "2nd", "home", "away", "team", "handicap", "race", "first", "last",
                       "odd", "even", "exact", "booking", "point", "&", " and ")

RULE_SWAP = "1X2 swap"
RULE_DERBY = "Derby pass"
RULE_CUP = "Cup tie pass"
RULE_CARDS = "Cards rules"
INSUFFICIENT_DATA = "insufficient data"


class MissingInput(Exception):
    pass


def _value(by_name: dict[str, Input], name: str) -> float:
    if name not in by_name:
        raise MissingInput(name)
    return by_name[name].value


# ---------------------------------------------------------------- goals model
def is_goals_only(inputs: Sequence[Input]) -> bool:
    """True when the inputs lack xG (the ESPN path): the goals-only model prices the game."""
    names = {i.name for i in inputs}
    return not set(XG_INPUT_NAMES) <= names


def _shrunk_goals(by_name: dict[str, Input], side: str, kind: str) -> float:
    """A side's goals for/against per match pulled toward the prior by its sample size (goals-only model)."""
    matches = _value(by_name, f"{side}.matches")
    if matches < GOALS_ONLY_MIN_MATCHES:
        raise MissingInput(f"{side}.matches")  # too little history: never estimated
    goals = _value(by_name, f"{side}.goals_{kind}")
    return (matches * goals + GOALS_ONLY_PRIOR_MATCHES * GOALS_ONLY_PRIOR_GOALS) / (matches + GOALS_ONLY_PRIOR_MATCHES)


def expected_goals(inputs: Sequence[Input]) -> tuple[float, float]:
    """(home, away) expected goals from the inputs only."""
    by_name = {i.name: i for i in inputs}
    if is_goals_only(inputs):
        return ((_shrunk_goals(by_name, "home", "for") + _shrunk_goals(by_name, "away", "against")) / 2,
                (_shrunk_goals(by_name, "away", "for") + _shrunk_goals(by_name, "home", "against")) / 2)

    def rate(side: str, kind: str) -> float:  # kind is "for" or "against"
        return (XG_WEIGHT * _value(by_name, f"{side}.xg_{kind}")
                + GOALS_WEIGHT * _value(by_name, f"{side}.goals_{kind}"))

    home = (rate("home", "for") + rate("away", "against")) / 2
    away = (rate("away", "for") + rate("home", "against")) / 2
    return home, away


def scaled_rates(inputs: Sequence[Input]) -> tuple[float, float]:
    """(home, away) rates with their sum kept inside MIN/MAX_EXPECTED_GOALS. One source of truth for totals,
    team totals and 1X2, so the three can never disagree about the game's expected goals."""
    home, away = expected_goals(inputs)
    total = home + away
    target = min(max(total, MIN_EXPECTED_GOALS), MAX_EXPECTED_GOALS)
    if total <= 0:
        return target / 2, target / 2
    factor = target / total
    return home * factor, away * factor


def poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    term = math.exp(-lam)
    total = term
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return min(total, 1.0)


def negbin_cdf(k: int, mean: float, dispersion: float) -> float:
    """CDF of a count with the given mean and variance = dispersion * mean (falls back to Poisson at 1)."""
    if k < 0:
        return 0.0
    if dispersion <= 1.0:
        return poisson_cdf(k, mean)
    r = mean / (dispersion - 1.0)
    p = 1.0 / dispersion
    total = 0.0
    for i in range(k + 1):
        log_pmf = (math.lgamma(i + r) - math.lgamma(r) - math.lgamma(i + 1) + r * math.log(p)
                   + i * math.log1p(-p))
        total += math.exp(log_pmf)
    return min(total, 1.0)


def line_probabilities(cdf: Callable[[int], float], line: Decimal) -> tuple[float, float, float]:
    """(P(over wins), P(under wins), P(push)) for a half or whole line on a non-negative count.

    Half line x.5: over wins on count >= x + 1, no push. Whole line N: over wins on count > N, under wins on
    count < N, and count == N is a push (the stake is refunded). Quarter lines are not supported."""
    if line % 1 == LINE_OFFSET:
        k = int(line)
        under = cdf(k)
        return 1.0 - under, under, 0.0
    if line % 1 == 0:
        n = int(line)
        over_win = 1.0 - cdf(n)
        under_win = cdf(n - 1)
        return over_win, under_win, max(0.0, 1.0 - over_win - under_win)
    raise ValueError("only half and whole lines are supported")


def probability_over(inputs: Sequence[Input], line: Decimal) -> float:
    """P(total goals > line) for a half line."""
    home, away = scaled_rates(inputs)
    return line_probabilities(lambda k: poisson_cdf(k, home + away), line)[0]


_RESULT_OR_OVER = re.compile(r"(home team|away team|draw) or over (\d+(?:\.\d+)?)")
_THREE_WAY = re.compile(r"handicap (\d+):(\d+)")
_BOTH_HALVES_UNDER = re.compile(r"both halves under (\d+(?:\.\d+)?)")
_TEAM_CORNERS = re.compile(r"(home|away) team total corners")
_WIN_EITHER_HALF = re.compile(r"(home|away) team to win either half")
_THREE_WAY_OUTCOME = re.compile(r"^(home|draw|away)\s*\(\s*(\d+):(\d+)\s*\)$", re.IGNORECASE)
_ASIAN_OUTCOME = re.compile(r"^(home|away)\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*\)$", re.IGNORECASE)
_MULTIGOALS_RANGE = re.compile(r"^(\d+)\s*-\s*(\d+)$")
_MULTIGOALS_PLUS = re.compile(r"^(\d+)\s*\+$")


def score_grid(home: float, away: float) -> dict[tuple[int, int], float]:
    """P(home goals = i, away goals = j) with independent Poisson goals, the grid result_probabilities sums over."""
    def pmf(lam):
        term = math.exp(-lam)
        out = [term]
        for i in range(1, GRID_MAX_GOALS + 1):
            term *= lam / i
            out.append(term)
        return out

    ph, pa = pmf(home), pmf(away)
    return {(i, j): x * y for i, x in enumerate(ph) for j, y in enumerate(pa)}


def result_probabilities(home: float, away: float) -> tuple[float, float, float]:
    """(P(home win), P(draw), P(away win)) with independent Poisson goals."""
    def pmf(lam):
        term = math.exp(-lam)
        out = [term]
        for i in range(1, GRID_MAX_GOALS + 1):
            term *= lam / i
            out.append(term)
        return out

    ph, pa = pmf(home), pmf(away)
    win_h = draw = win_a = 0.0
    for i, x in enumerate(ph):
        for j, y in enumerate(pa):
            if i > j:
                win_h += x * y
            elif i == j:
                draw += x * y
            else:
                win_a += x * y
    return win_h, draw, win_a


# ---------------------------------------------------------------- market helpers
def _parse_line(specifier: str, low: Decimal, high: Decimal, whole: bool, half: bool) -> Decimal | None:
    key, _, value = specifier.partition("=")
    if key.strip() != "total":
        return None
    try:
        line = Decimal(value.strip())
    except InvalidOperation:
        return None
    if not line.is_finite() or not low <= line <= high:
        return None
    is_half = line % LINE_STEP == LINE_OFFSET
    is_whole = line % LINE_STEP == 0
    return line if (is_half and half) or (is_whole and whole) else None


def totals_line(specifier: str) -> Decimal | None:
    """A half line x.5 from 0.5 to 6.5, else None."""
    return _parse_line(specifier, MIN_LINE, MAX_LINE, whole=False, half=True)


def whole_total_line(specifier: str) -> Decimal | None:
    """A whole line 1 to 6 (refund on exactly that many goals), else None."""
    return _parse_line(specifier, Decimal("1"), Decimal("6"), whole=True, half=False)


def fmt_line(line: Decimal) -> str:
    return f"{line:.1f}" if line % 1 == 0 else str(line)


def _outcome(offer: MarketOffer, side: str):
    found = [o for o in offer.outcomes if o.name.strip().lower().startswith(side)]
    return found[0] if len(found) == 1 else None


def market_kind(offer: MarketOffer) -> str | None:
    if offer.market_id == RESULT_MARKET_ID:
        return "result"
    if offer.market_id == TOTALS_MARKET_ID:
        return "goals"
    if offer.market_id == HOME_TOTAL_MARKET_ID:
        return "home_total"
    if offer.market_id == AWAY_TOTAL_MARKET_ID:
        return "away_total"
    if offer.market_id == DOUBLE_CHANCE_MARKET_ID:
        return "double_chance"
    if offer.market_id == CORNERS_MARKET_ID:
        return "corners"
    if offer.market_id == CARDS_MARKET_ID:
        return "cards"
    if offer.market_id == ASIAN_HANDICAP_MARKET_ID:
        return "asian_handicap"
    name_key = " ".join(offer.name.lower().split())
    if name_key == "multigoals":   # full time only
        return "multigoals"
    # Review-only markets, priced from the same scoreline model. Matched by name (their SportyBet ids were not probed).
    if offer.market_id == "29" or name_key == "gg/ng":
        return "btts"
    if offer.market_id == "11" or name_key == "draw no bet":
        return "dnb"
    if _RESULT_OR_OVER.fullmatch(name_key):
        return "result_or_over"
    if name_key.startswith("1x2 - 2up"):
        return "result_2up"
    if _THREE_WAY.fullmatch(name_key):
        return "handicap3"
    if name_key == "2nd half - multigoals":
        return "multigoals_2h"
    if name_key == "2nd half - double chance":
        return "double_chance_2h"
    if _BOTH_HALVES_UNDER.fullmatch(name_key):
        return "both_halves_under"
    team_corners = _TEAM_CORNERS.fullmatch(name_key)
    if team_corners:
        return f"{team_corners.group(1)}_corners"
    either_half = _WIN_EITHER_HALF.fullmatch(name_key)
    if either_half:
        return f"{either_half.group(1)}_win_either_half"
    name = f" {offer.name.lower()} "
    if any(word in name for word in PROP_EXCLUDED_WORDS):
        return None
    if "corner" in name:
        return "corners"
    if "card" in name:
        return "cards"
    return None


def _kept_inputs(inputs: Sequence[Input]) -> tuple[Input, ...]:
    return tuple(sorted(inputs, key=lambda i: i.name))


# ---------------------------------------------------------------- engine
class SoccerEngine:
    sport = "soccer"

    # ---- fixture-level pass (Story 2.4 c) ----
    def screen(self, fixture: Fixture, inputs: Sequence[Input]) -> FixturePass | None:
        derby = find_derby(fixture.competition, fixture.home, fixture.away)
        if derby is not None:
            return FixturePass(fixture.match_id, RULE_DERBY,
                               f"derby ({derby.name}): tight, low-event games are not played")
        if not is_cup(fixture.competition) or not inputs:
            return None
        by_name = {i.name: i.value for i in inputs}
        ppdas = [by_name[k] for k in ("home.ppda", "away.ppda") if k in by_name]
        if not ppdas:
            if CUP_PASS_WITHOUT_PPDA:
                return FixturePass(fixture.match_id, RULE_CUP,
                                   "cup tie and no PPDA input to rule out a low block")
            return None
        try:
            home, away = expected_goals(tuple(inputs))
        except MissingInput:
            return None
        total = home + away
        if max(ppdas) >= LOW_BLOCK_PPDA and total <= CUP_TIGHT_MAX_TOTAL_GOALS:
            return FixturePass(fixture.match_id, RULE_CUP,
                               f"cup tie with a low block (PPDA {max(ppdas):.1f}, at least {LOW_BLOCK_PPDA:.1f}) "
                               f"and a tight game ({total:.1f} expected goals, at most "
                               f"{CUP_TIGHT_MAX_TOTAL_GOALS:.1f})")
        return None

    # ---- picks ----
    # Spec 11-1: the gate keeps at most this many passing plays per game (the best by probability of success), and a corners or
    # cards play with no data behind it is not considered at all (it is not listed as a rejection either).
    max_plays_per_game = MATCHDAY_MAX_PLAYS_PER_GAME
    quiet_insufficient = True

    def evaluate(self, fixture: Fixture, inputs: Sequence[Input],
                 market_options: Sequence[MarketOffer]) -> list[Evaluation]:
        """Every Matchday play judged: over half goal lines from 1.5, match winner, double chance, over yellow cards from 2.5 and
        over corners from 7.5 (spec 11-1); nothing else is considered. A screened fixture (derby, low-block cup tie) gives
        nothing here: see `screen`."""
        used = _kept_inputs(inputs)
        if self.screen(fixture, used) is not None:
            return []
        try:
            home, away = scaled_rates(used)
        except MissingInput:
            return []
        ctx = _Context(fixture, used, home, away, offers_without_whole_lines(market_options))   # AD-21: half lines only
        results: list[Evaluation] = []
        results.extend(ctx.goal_totals())
        results.extend(ctx.results_and_swaps(results))
        results.extend(ctx.double_chances())
        results.extend(ctx.corners(MATCHDAY_MIN_CORNERS_LINE))
        results.extend(ctx.cards(MATCHDAY_MIN_CARDS_LINE))
        return without_whole_lines(without_unders(results))   # owner rules: no Under (AD-20), no whole line (AD-21)

    # ---- Story 4.2: one exact selection from a booking code ----
    def understands(self, offer: MarketOffer) -> bool:
        """True when this engine has rules for the offer's market (and only then may it judge it)."""
        return market_kind(offer) is not None

    def evaluate_exact(self, fixture: Fixture, inputs: Sequence[Input], offer: MarketOffer) -> list[Evaluation]:
        return without_unders(self._evaluate_exact(fixture, inputs, offer))   # owner rule: no Under (AD-20)

    def _evaluate_exact(self, fixture: Fixture, inputs: Sequence[Input], offer: MarketOffer) -> list[Evaluation]:
        """Judge exactly the outcomes in `offer` at the odds it carries (the odds ON THE CODE): no Asian shielding,
        no 1X2 swap, no other line. Same model, same floor and EV gate as a Matchday pick. Returns [] when the
        market or line is not one the engine understands or the inputs cannot price it; a screened fixture
        (derby, low-block cup tie) is refused by `screen`, which the caller applies first."""
        used = _kept_inputs(inputs)
        try:
            home, away = scaled_rates(used)
        except MissingInput:
            return []
        ctx = _Context(fixture, used, home, away, [offer])
        kind = market_kind(offer)
        if kind == "goals":
            return ctx.exact_totals()
        if kind == "result":
            return ctx.exact_results()
        if kind == "double_chance":
            return ctx.double_chances(exact=True)
        if kind in ("home_total", "away_total"):
            return ctx.exact_team_totals(kind)
        if kind == "corners":
            return ctx.corners()
        if kind == "cards":
            return ctx.cards()
        if kind == "asian_handicap":
            return ctx.exact_asian_handicap()
        if kind == "multigoals":
            return ctx.exact_multigoals()
        extra = {"btts": ctx.exact_btts, "dnb": ctx.exact_draw_no_bet, "result_or_over": ctx.exact_result_or_over,
                 "result_2up": ctx.exact_result_2up, "handicap3": ctx.exact_three_way_handicap,
                 "multigoals_2h": ctx.exact_multigoals_second_half, "double_chance_2h": ctx.exact_double_chance_second_half,
                 "both_halves_under": ctx.exact_both_halves_under,
                 "home_corners": lambda: ctx.exact_team_corners("home"), "away_corners": lambda: ctx.exact_team_corners("away"),
                 "home_win_either_half": lambda: ctx.exact_win_either_half("home"),
                 "away_win_either_half": lambda: ctx.exact_win_either_half("away")}.get(kind)
        return extra() if extra is not None else []


class _Context:
    """One fixture's model, so every market is priced from the same rates."""

    def __init__(self, fixture, used, home, away, offers):
        self.fixture, self.used, self.home, self.away = fixture, used, home, away
        self.offers = list(offers)
        self.total = home + away
        self.by_name = {i.name: i for i in used}
        self.goals_only = is_goals_only(used)
        self.model = GOALS_ONLY_MODEL_NAME if self.goals_only else MODEL_NAME

    # -- shared
    def _offers(self, kind: str):
        return [o for o in self.offers if market_kind(o) == kind]

    def _evaluation(self, market: str, selection: str, odds: Decimal, estimate: Estimate, rule: str = "",
                    original: str = "", notes: tuple[str, ...] = ()) -> Evaluation:
        passed, ev, reason = validate(estimate, odds)
        if self.goals_only and passed and ev > Decimal(str(GOALS_ONLY_MAX_EV)):
            passed, reason = False, GOALS_ONLY_CAP_REASON.format(cap=GOALS_ONLY_MAX_EV)
        if self.goals_only and estimate.model != GOALS_ONLY_MODEL_NAME:
            notes = (*notes, GOALS_ONLY_LABEL)  # corners and cards still lean on goals-only expected goals
        return Evaluation(self.fixture.match_id, market, selection, odds, estimate, ev, passed, reason,
                          rule, original, tuple(n for n in notes if n))

    def _insufficient(self, market: str, selection: str, odds: Decimal, why: str, model: str) -> Evaluation:
        return Evaluation(self.fixture.match_id, market, selection, odds, Estimate(0.0, self.used, model),
                          Decimal(0), False, f"{INSUFFICIENT_DATA}: {why}", "", "", (), True)

    def _estimate(self, over_win, under_win, push, side: str, model: str | None = None) -> Estimate:
        model = model or self.model
        win = over_win if side == "over" else under_win
        if push:
            return Estimate.push_protected(win, push, self.used, model)
        return Estimate(min(win, 1.0), self.used, model)

    # -- goal totals (Story 2.4 a, spec 11-1, AD-21): Over HALF lines from 1.5 only; a whole or quarter line is never priced here
    def goal_totals(self) -> list[Evaluation]:
        cdf = lambda k: poisson_cdf(k, self.total)  # noqa: E731
        results: list[Evaluation] = []
        for offer in self._offers("goals"):
            line = totals_line(offer.specifier)
            if line is None or line < MATCHDAY_MIN_GOAL_LINE:
                continue
            over_win, under_win, _ = line_probabilities(cdf, line)
            for side in SIDES_PLAYED:
                outcome = _outcome(offer, side)
                if outcome is None:
                    continue
                results.append(self._evaluation(f"Over/Under {line}", f"{side.title()} {line}", outcome.odds,
                                                self._estimate(over_win, under_win, 0.0, side)))
        return results

    # -- exact selections for Review (Story 4.2)
    def exact_totals(self) -> list[Evaluation]:
        results = []
        for offer in self._offers("goals"):
            line = whole_total_line(offer.specifier) or totals_line(offer.specifier)
            if line is None:
                continue
            over_win, under_win, push = line_probabilities(lambda k: poisson_cdf(k, self.total), line)
            for side in SIDES_PLAYED:
                outcome = _outcome(offer, side)
                if outcome is None:
                    continue
                estimate = self._estimate(over_win, under_win, push, side)
                notes = (_push_note(estimate, line),) if push else ()
                results.append(self._evaluation(f"Over/Under {fmt_line(line)}", f"{side.title()} {fmt_line(line)}",
                                                outcome.odds, estimate, "", "", notes))
        return results

    def exact_team_totals(self, kind: str) -> list[Evaluation]:
        team = "home" if kind == "home_total" else "away"
        lam = self.home if team == "home" else self.away
        results = []
        for offer in self._offers(kind):
            line = _parse_line(offer.specifier, MIN_LINE, MAX_LINE, True, True)
            if line is None:
                continue
            over_win, under_win, push = line_probabilities(lambda k: poisson_cdf(k, lam), line)
            for side in SIDES_PLAYED:
                outcome = _outcome(offer, side)
                if outcome is None:
                    continue
                estimate = self._estimate(over_win, under_win, push, side)
                notes = (_push_note(estimate, line),) if push else ()
                results.append(self._evaluation(f"{team.title()} team total {fmt_line(line)}",
                                                f"{team.title()} team {side.title()} {fmt_line(line)}",
                                                outcome.odds, estimate, "", "", notes))
        return results

    # -- Review only: Asian handicap and multigoals, priced from the same Poisson scoreline grid as the 1X2 (so they cannot
    #    disagree with it). A whole handicap refunds the stake when the margin equals it; a quarter handicap splits the stake over
    #    the two lines either side (half win, half refund or half lose). Success = win + refund, as for a whole goal line.
    def exact_asian_handicap(self) -> list[Evaluation]:
        grid = score_grid(self.home, self.away)
        results = []
        for offer in self._offers("asian_handicap"):
            for outcome in offer.outcomes:
                found = _ASIAN_OUTCOME.match(outcome.name.strip())
                if found is None:
                    continue
                side, handicap = found.group(1).lower(), Decimal(found.group(2))
                if (handicap * 4) % 1 != 0:
                    continue   # not a quarter step
                legs = [handicap] if (handicap * 2) % 1 == 0 else [handicap - Decimal("0.25"), handicap + Decimal("0.25")]
                win = push = 0.0
                for (i, j), p in grid.items():
                    margin = (i - j) if side == "home" else (j - i)
                    for leg in legs:
                        value = margin + float(leg)
                        if value > 0:
                            win += p / len(legs)
                        elif value == 0:
                            push += p / len(legs)
                estimate = Estimate.push_protected(min(win, 1.0), min(push, 1.0 - min(win, 1.0)), self.used, self.model) if push                     else Estimate(min(win, 1.0), self.used, self.model)
                notes = ((f"a stake is refunded (half of it on a quarter handicap) when the margin equals the handicap: win "
                          f"{percent(estimate.win)} + refund {percent(estimate.push)} = success {percent(estimate.probability)}",)
                         if estimate.push else ())
                results.append(self._evaluation(f"Asian Handicap {handicap:+}", outcome.name.strip(), outcome.odds, estimate,
                                                "", "", notes))
        return results

    def exact_multigoals(self) -> list[Evaluation]:
        results = []
        for offer in self._offers("multigoals"):
            for outcome in offer.outcomes:
                text = outcome.name.strip()
                ranged, plus = _MULTIGOALS_RANGE.match(text), _MULTIGOALS_PLUS.match(text)
                cdf = lambda k: poisson_cdf(k, self.total)  # noqa: E731
                if ranged is not None and int(ranged.group(1)) <= int(ranged.group(2)):
                    low, high = int(ranged.group(1)), int(ranged.group(2))
                    p = cdf(high) - cdf(low - 1)
                elif plus is not None:
                    p = 1.0 - cdf(int(plus.group(1)) - 1)
                else:
                    continue
                results.append(self._evaluation("Multigoals", text, outcome.odds,
                                                Estimate(min(max(p, 0.0), 1.0), self.used, self.model)))
        return results

    # -- more Review-only markets (the scoreline model; see the half-share assumption above)
    def _under_type(self, market: str, selection: str, odds, probability: float, why: str) -> Evaluation:
        """A side that is an Under in disguise (it needs FEWER goals): priced for the record but refused by the owner's no-Under rule."""
        evaluation = self._evaluation(market, selection, odds, Estimate(min(max(probability, 0.0), 1.0), self.used, self.model),
                                      notes=(why,))
        return reject_under(evaluation)

    def exact_btts(self) -> list[Evaluation]:
        p_yes = (1 - math.exp(-self.home)) * (1 - math.exp(-self.away))
        results = []
        for offer in self._offers("btts"):
            for outcome in offer.outcomes:
                side = outcome.name.strip().lower()
                if side == "yes":
                    results.append(self._evaluation("GG/NG", "Yes", outcome.odds, Estimate(min(p_yes, 1.0), self.used, self.model)))
                elif side == "no":
                    results.append(self._under_type("GG/NG", "No", outcome.odds, 1 - p_yes,
                                                    "'No' needs a team to stay scoreless, the opposite of a goals pick"))
        return results

    def exact_draw_no_bet(self) -> list[Evaluation]:
        win_h, draw, win_a = result_probabilities(self.home, self.away)
        results = []
        for offer in self._offers("dnb"):
            for outcome in offer.outcomes:
                side = outcome.name.strip().lower()
                if side not in ("home", "away"):
                    continue
                win = win_h if side == "home" else win_a
                estimate = Estimate.push_protected(min(win, 1.0), min(draw, 1.0 - min(win, 1.0)), self.used, self.model)
                note = (f"refund on a draw: win {percent(estimate.win)} + refund {percent(estimate.push)} = success "
                        f"{percent(estimate.probability)}",)
                results.append(self._evaluation("Draw No Bet", side.title(), outcome.odds, estimate, "", "", note))
        return results

    def exact_result_or_over(self) -> list[Evaluation]:
        grid = score_grid(self.home, self.away)
        results = []
        for offer in self._offers("result_or_over"):
            found = _RESULT_OR_OVER.fullmatch(" ".join(offer.name.lower().split()))
            who, line = found.group(1), float(found.group(2))
            wins = {"home team": lambda i, j: i > j, "away team": lambda i, j: j > i, "draw": lambda i, j: i == j}[who]
            p_yes = sum(p for (i, j), p in grid.items() if wins(i, j) or i + j > line)
            title = f"{found.group(1).title()} or Over {found.group(2)}"
            for outcome in offer.outcomes:
                side = outcome.name.strip().lower()
                if side == "yes":
                    results.append(self._evaluation(title, "Yes", outcome.odds, Estimate(min(p_yes, 1.0), self.used, self.model)))
                elif side == "no":
                    results.append(self._under_type(title, "No", outcome.odds, 1 - p_yes,
                                                    "'No' needs fewer goals than the line and no win for that side: an Under in disguise"))
        return results

    def exact_result_2up(self) -> list[Evaluation]:
        """1X2 with the early-payout promotion (a side that leads by two goals at any moment is paid as a winner). The promotion only
        ever helps the pick and needs goal times we do not have, so the price is the plain 1X2 chance: a conservative estimate."""
        results = []
        probs = result_probabilities(self.home, self.away)
        for offer in self._offers("result_2up"):
            for name, p in zip(("home", "draw", "away"), probs):
                outcome = _outcome(offer, name)
                if outcome is not None:
                    results.append(self._evaluation("1X2 - 2UP", name.title(), outcome.odds, Estimate(min(p, 1.0), self.used, self.model),
                                                    notes=("the 2-goals-up early payout is not counted: the estimate is the plain 1X2 chance",)))
        return results

    def exact_three_way_handicap(self) -> list[Evaluation]:
        grid = score_grid(self.home, self.away)
        results = []
        for offer in self._offers("handicap3"):
            found = _THREE_WAY.fullmatch(" ".join(offer.name.lower().split()))
            start_home, start_away = int(found.group(1)), int(found.group(2))
            probs = {"home": 0.0, "draw": 0.0, "away": 0.0}
            for (i, j), p in grid.items():
                margin = (i + start_home) - (j + start_away)
                probs["home" if margin > 0 else "draw" if margin == 0 else "away"] += p
            for outcome in offer.outcomes:
                parsed = _THREE_WAY_OUTCOME.match(outcome.name.strip())
                if parsed is None or (int(parsed.group(2)), int(parsed.group(3))) != (start_home, start_away):
                    continue
                side = parsed.group(1).lower()
                results.append(self._evaluation(f"Handicap {start_home}:{start_away}", outcome.name.strip(), outcome.odds,
                                                Estimate(min(probs[side], 1.0), self.used, self.model)))
        return results

    def exact_multigoals_second_half(self) -> list[Evaluation]:
        results = []
        rate = self.total * SECOND_HALF_SHARE
        for offer in self._offers("multigoals_2h"):
            for outcome in offer.outcomes:
                text = outcome.name.strip()
                ranged, plus = _MULTIGOALS_RANGE.match(text), _MULTIGOALS_PLUS.match(text)
                if ranged is not None and int(ranged.group(1)) <= int(ranged.group(2)):
                    low, high = int(ranged.group(1)), int(ranged.group(2))
                    p = poisson_cdf(high, rate) - poisson_cdf(low - 1, rate)
                elif plus is not None:
                    p = 1.0 - poisson_cdf(int(plus.group(1)) - 1, rate)
                else:
                    continue
                results.append(self._evaluation("2nd Half - Multigoals", text, outcome.odds, Estimate(min(max(p, 0.0), 1.0), self.used, self.model),
                                                notes=(f"second-half goals assumed to be {SECOND_HALF_SHARE:.0%} of the game's",)))
        return results

    def exact_double_chance_second_half(self) -> list[Evaluation]:
        win_h, draw, win_a = result_probabilities(self.home * SECOND_HALF_SHARE, self.away * SECOND_HALF_SHARE)
        chances = {"home or draw": win_h + draw, "draw or away": win_a + draw, "away or draw": win_a + draw, "home or away": win_h + win_a}
        results = []
        for offer in self._offers("double_chance_2h"):
            for outcome in offer.outcomes:
                key = " ".join(outcome.name.lower().split())
                if key not in chances:
                    continue
                results.append(self._evaluation("2nd Half - Double Chance", outcome.name.strip(), outcome.odds,
                                                Estimate(min(chances[key], 1.0), self.used, self.model),
                                                notes=(f"second-half goals assumed to be {SECOND_HALF_SHARE:.0%} of the game's",)))
        return results

    def exact_both_halves_under(self) -> list[Evaluation]:
        results = []
        for offer in self._offers("both_halves_under"):
            line = float(_BOTH_HALVES_UNDER.fullmatch(" ".join(offer.name.lower().split())).group(1))
            limit = int(line)   # a half line L.5 allows up to L goals in each half
            p_both = poisson_cdf(limit, self.total * FIRST_HALF_SHARE) * poisson_cdf(limit, self.total * SECOND_HALF_SHARE)
            title = f"Both Halves Under {offer.name.split()[-1]}"
            for outcome in offer.outcomes:
                side = outcome.name.strip().lower()
                if side == "no":
                    results.append(self._evaluation(title, "No", outcome.odds, Estimate(min(max(1 - p_both, 0.0), 1.0), self.used, self.model),
                                                    notes=(f"half goals assumed {FIRST_HALF_SHARE:.0%} / {SECOND_HALF_SHARE:.0%} of the game's",)))
                elif side == "yes":
                    results.append(self._under_type(title, "Yes", outcome.odds, p_both, "'Yes' is an Under (fewer goals in both halves)"))
        return results

    def exact_team_corners(self, team: str) -> list[Evaluation]:
        """One team's own corners (SportyBet "Home/Away Team Total Corners"): the team's corners for, averaged with what the
        opponent concedes, through the same negative binomial as match corners but with a lower dispersion. Review only."""
        offers = self._offers(f"{team}_corners")
        names = (f"{team}.corners_for", f"{'away' if team == 'home' else 'home'}.corners_against")
        missing = [n for n in names if n not in self.by_name]
        results = []
        if not missing:
            mean = min(max((self.by_name[names[0]].value + self.by_name[names[1]].value) / 2, MIN_EXPECTED_TEAM_CORNERS),
                       MAX_EXPECTED_TEAM_CORNERS)
            notes = (_inputs_note(tuple(self.by_name[n] for n in names),
                                  f"expected {team} team corners {mean:.1f}, negative binomial, "
                                  f"dispersion {TEAM_CORNERS_DISPERSION}"),)
        for offer in offers:
            line = _parse_line(offer.specifier, PROP_LINE_MIN, PROP_LINE_MAX, True, True)
            if line is None:
                continue
            for side in SIDES_PLAYED:
                outcome = _outcome(offer, side)
                if outcome is None:
                    continue
                market, selection = f"{team.title()} team corners {fmt_line(line)}", f"{team.title()} team corners {side.title()} {fmt_line(line)}"
                if missing:
                    results.append(self._insufficient(market, selection, outcome.odds,
                                                      "missing corner data: " + ", ".join(missing), CORNERS_MODEL))
                    continue
                o, u, push = line_probabilities(lambda k: negbin_cdf(k, mean, TEAM_CORNERS_DISPERSION), line)
                estimate = self._estimate(o, u, push, side, CORNERS_MODEL)
                extra = (_push_note(estimate, line, "corner(s)"),) if push else ()
                results.append(self._evaluation(market, selection, outcome.odds, estimate, "", "", notes + extra))
        return results

    def exact_win_either_half(self, team: str) -> list[Evaluation]:
        """"Team to Win Either Half", Yes only ("No" is the opposite of a goals pick and is not priced). The team wins a half when
        it outscores the opponent in it; the halves are treated as independent, with the assumed 45% / 55% split of goals."""
        win_1 = result_probabilities(self.home * FIRST_HALF_SHARE, self.away * FIRST_HALF_SHARE)
        win_2 = result_probabilities(self.home * SECOND_HALF_SHARE, self.away * SECOND_HALF_SHARE)
        index = 0 if team == "home" else 2
        p = 1.0 - (1.0 - win_1[index]) * (1.0 - win_2[index])
        results = []
        for offer in self._offers(f"{team}_win_either_half"):
            for outcome in offer.outcomes:
                if outcome.name.strip().lower() == "yes":
                    results.append(self._evaluation(
                        f"{team.title()} Team to Win Either Half", "Yes", outcome.odds,
                        Estimate(min(max(p, 0.0), 1.0), self.used, self.model),
                        notes=(f"half goals assumed {FIRST_HALF_SHARE:.0%} / {SECOND_HALF_SHARE:.0%} of the game's, halves independent",)))
        return results

    def exact_results(self) -> list[Evaluation]:
        results = []
        probs = result_probabilities(self.home, self.away)
        for offer in self._offers("result"):
            for name, p in zip(("home", "draw", "away"), probs):
                outcome = _outcome(offer, name)
                if outcome is not None:
                    results.append(self._evaluation("1X2", name.title(), outcome.odds,
                                                    Estimate(min(p, 1.0), self.used, self.model)))
        return results

    # -- double chance (spec 11-1): the same scoreline grid as the 1X2, so the three can never disagree
    def double_chances(self, exact: bool = False) -> list[Evaluation]:
        offers = self._offers("double_chance")
        if not offers:
            return []
        win_h, draw, win_a = result_probabilities(self.home, self.away)
        choices = [*DOUBLE_CHANCE_CHOICES, ("Home or Away", "home or away")] if exact else list(DOUBLE_CHANCE_CHOICES)
        chances = {"Home or Draw": win_h + draw, "Away or Draw": win_a + draw, "Home or Away": win_h + win_a}
        results = []
        for offer in offers:
            for text, listed in choices:
                found = [o for o in offer.outcomes if " ".join(o.name.lower().split()) == listed]
                if len(found) != 1:
                    continue
                estimate = Estimate(min(chances[text], 1.0), self.used, self.model)
                results.append(self._evaluation("Double Chance", text, found[0].odds, estimate))
        return results

    # -- 1X2 and the swap (Story 2.4 b)
    def results_and_swaps(self, existing: list[Evaluation]) -> list[Evaluation]:
        offers = self._offers("result")
        if not offers:
            return []
        probs = result_probabilities(self.home, self.away)
        results = []
        for offer in offers:
            for name, p in zip(("home", "draw", "away"), probs):
                if name == "draw":   # spec 11-1: a Draw on its own is not a Matchday play
                    continue
                outcome = _outcome(offer, name)
                if outcome is None:
                    continue
                estimate = Estimate(min(p, 1.0), self.used, self.model)
                base = self._evaluation("1X2", name.title(), outcome.odds, estimate)
                if (outcome.odds <= SWAP_MIN_ODDS or estimate.probability >= PROBABILITY_FLOOR
                        or estimate.probability < SWAP_MIN_ESTIMATE):
                    results.append(base)
                    continue
                results.append(self._swap(name, base, existing))
        return results

    def _candidates(self, team: str):
        """(market, selection, odds, estimate, line) for every listed conversion target: Over match-goals lines only."""
        found = []
        cdf_total = lambda k: poisson_cdf(k, self.total)  # noqa: E731
        for offer in self._offers("goals"):
            line = totals_line(offer.specifier)
            if line is None or line not in SWAP_TOTAL_LINES:
                continue
            outcome = _outcome(offer, "over")
            if outcome is None:
                continue
            over_win, under_win, push = line_probabilities(cdf_total, line)
            found.append((f"Over/Under {fmt_line(line)}", f"Over {fmt_line(line)}", outcome.odds,
                          self._estimate(over_win, under_win, push, "over"), line))
        return found

    def _swap(self, team_name: str, base: Evaluation, existing: list[Evaluation]) -> Evaluation:
        candidates = [(m, sel, odds, est, line) for m, sel, odds, est, line in self._candidates(team_name)]
        rule = (f"{RULE_SWAP}: {team_name} above {SWAP_MIN_ODDS:.2f} odds with an estimate from "
                f"{SWAP_MIN_ESTIMATE:.0%} up to the {PROBABILITY_FLOOR:.0%} floor is tried as a goal line")
        if not candidates:
            return _with(base, rule=rule + "; no goal line is listed to convert to",
                         reason=base.reason + "; no goal line is listed to convert to")
        scored = []
        for market, selection, odds, estimate, line in candidates:
            evaluation = self._evaluation(market, selection, odds, estimate, rule, _describe(base),
                                          (_push_note(estimate, line),) if estimate.push else ())
            scored.append(evaluation)
        passing = [e for e in scored if e.passed]
        if passing:
            best = max(passing, key=lambda e: (e.ev, e.estimate.probability, e.selection))
            # An identical pick already listed by the goal-lines section is replaced by the swap version.
            existing[:] = [e for e in existing if (e.market, e.selection) != (best.market, best.selection)]
            return best
        closest = max(scored, key=lambda e: (e.estimate.probability, e.selection))
        note = (f"; conversion also fails: {closest.selection} @ {closest.odds:.2f} has "
                f"{percent(closest.estimate.probability)}, {closest.reason}")
        return _with(base, rule=rule, reason=base.reason + note)

    # -- corners (Story 2.5)
    def corners(self, min_line: Decimal | None = None) -> list[Evaluation]:
        offers = self._offers("corners")
        if not offers:
            return []
        names = ("home.corners_for", "home.corners_against", "away.corners_for", "away.corners_against")
        missing = [n for n in names if n not in self.by_name]
        results = []
        if not missing:
            home_exp = (self.by_name["home.corners_for"].value + self.by_name["away.corners_against"].value) / 2
            away_exp = (self.by_name["away.corners_for"].value + self.by_name["home.corners_against"].value) / 2
            mean = min(max(home_exp + away_exp, MIN_EXPECTED_CORNERS), MAX_EXPECTED_CORNERS)
            shown = tuple(self.by_name[n] for n in names)
            notes = (_inputs_note(shown, f"expected corners {mean:.1f}, negative binomial, "
                                         f"dispersion {CORNERS_DISPERSION}"),)
        for offer in offers:
            line = _parse_line(offer.specifier, PROP_LINE_MIN, PROP_LINE_MAX, True, True)
            if line is None or (min_line is not None and line < min_line):
                continue
            for side in SIDES_PLAYED:
                outcome = _outcome(offer, side)
                if outcome is None:
                    continue
                market, selection = f"Corners {fmt_line(line)}", f"Corners {side.title()} {fmt_line(line)}"
                if missing:
                    results.append(self._insufficient(market, selection, outcome.odds,
                                                      "missing corner data: " + ", ".join(missing), CORNERS_MODEL))
                    continue
                o, u, push = line_probabilities(lambda k: negbin_cdf(k, mean, CORNERS_DISPERSION), line)
                estimate = self._estimate(o, u, push, side, CORNERS_MODEL)
                extra = (_push_note(estimate, line, "corner(s)"),) if push else ()
                results.append(self._evaluation(market, selection, outcome.odds, estimate, "", "", notes + extra))
        return results

    # -- cards (Story 2.5)
    def cards(self, min_line: Decimal | None = None) -> list[Evaluation]:
        offers = self._offers("cards")
        if not offers:
            return []
        problems = []
        ref = self.by_name.get("match.referee_cards_avg")
        games = self.by_name.get("match.referee_games")
        if ref is None:
            problems.append("no referee average for this game")
        elif games is not None and games.value < REFEREE_MIN_GAMES:
            problems.append(f"referee average is over only {games.value:g} games (need {REFEREE_MIN_GAMES})")
        team_missing = [n for n in ("home.cards_for", "away.cards_for") if n not in self.by_name]
        if team_missing:
            problems.append("missing card data: " + ", ".join(team_missing))
        results = []
        disqualify = []
        if not problems:
            if ref.value < CARDS_MIN_REFEREE_AVG:
                disqualify.append(f"referee averages {ref.value:.1f} cards per game, under the required "
                                  f"{CARDS_MIN_REFEREE_AVG:.1f}")
            diff = abs(self.home - self.away)
            if diff >= CARDS_MAX_EXPECTED_GOAL_DIFF:
                disqualify.append(f"blowout risk: expected goal difference {diff:.1f} is not under "
                                  f"{CARDS_MAX_EXPECTED_GOAL_DIFF:.1f}")
            team_rate = self.by_name["home.cards_for"].value + self.by_name["away.cards_for"].value
            mean = CARDS_REFEREE_WEIGHT * ref.value + (1 - CARDS_REFEREE_WEIGHT) * team_rate
            mean = min(max(mean, MIN_EXPECTED_CARDS), MAX_EXPECTED_CARDS)
            shown = tuple(x for x in (ref, games, self.by_name["home.cards_for"],
                                      self.by_name["away.cards_for"]) if x is not None)
            notes = (_inputs_note(shown, f"expected cards {mean:.1f}, goal difference {diff:.1f}, "
                                         f"negative binomial, dispersion {CARDS_DISPERSION}"),)
        for offer in offers:
            line = _parse_line(offer.specifier, PROP_LINE_MIN, PROP_LINE_MAX, True, True)
            if line is None or (min_line is not None and line < min_line):
                continue
            for side in SIDES_PLAYED:
                outcome = _outcome(offer, side)
                if outcome is None:
                    continue
                market, selection = f"Cards {fmt_line(line)}", f"Cards {side.title()} {fmt_line(line)}"
                if problems:
                    results.append(self._insufficient(market, selection, outcome.odds, "; ".join(problems),
                                                      CARDS_MODEL))
                    continue
                o, u, push = line_probabilities(lambda k: negbin_cdf(k, mean, CARDS_DISPERSION), line)
                estimate = self._estimate(o, u, push, side, CARDS_MODEL)
                extra = (_push_note(estimate, line, "card(s)"),) if push else ()
                evaluation = self._evaluation(market, selection, outcome.odds, estimate, "", "", notes + extra)
                if disqualify:
                    evaluation = _with(evaluation, passed=False, rule=RULE_CARDS,
                                       reason="disqualified: " + "; ".join(disqualify))
                results.append(evaluation)
        return results


def _with(evaluation: Evaluation, **changes) -> Evaluation:
    return replace(evaluation, **changes)


def _describe(evaluation: Evaluation) -> str:
    return (f"{evaluation.selection} @ {evaluation.odds:.2f} (estimate "
            f"{percent(evaluation.estimate.probability)})")


def _push_note(estimate: Estimate, line: Decimal, unit: str = "goal(s)") -> str:
    if not estimate.push:
        return ""
    return (f"refund on exactly {int(line)} {unit}: win {percent(estimate.win)} + refund {percent(estimate.push)} = "
            f"success {percent(estimate.probability)}")


def _inputs_note(shown: Sequence[Input], summary: str) -> str:
    parts = ", ".join(f"{i.name} {i.value:g} {i.unit} [{i.source}]" for i in shown)
    return f"inputs: {parts}; {summary}"
