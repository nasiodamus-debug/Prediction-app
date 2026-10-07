"""Basketball engine (Stories 6.1 and 6.2). Standard library only. Everything here is deterministic.

Model: the expected points of each side come from the two teams' rolling efficiency and pace,
    pace     = (home pace + away pace) / 2                       possessions per game
    home pts = pace * (home ORTG + away DRTG) / 200 + HOME_COURT_POINTS / 2
    away pts = pace * (away ORTG + home DRTG) / 200 - HOME_COURT_POINTS / 2
(ORTG / DRTG are points scored / allowed per 100 possessions.) Points are integers, so each market is a normal
distribution with a continuity correction: game total, team total, first-half total, margin (spread, moneyline).
Standard deviations are FRACTIONS of the expected points (so the model works for NBA, WNBA and other scoring
levels) and are judgement values, not fitted (spec-6-1 Design Notes). The model is uncalibrated: calibration comes
with Epic 8.

Pipeline per fixture: Validation (floor + EV, `core.model.validate`) -> Optimization (Story 6.2):
  * ORTG lock: a team whose rolling ORTG is at least ORTG_LOCK_THRESHOLD locks Team Total Over as the primary
    output; full-game totals are bypassed (the best full-game total is shown as the original pick).
  * 1H fast-pace route: an "explosive-starting" team (proxy below) routes the total to a 1st-half total Over.
  A converted pick stays only if it clears the floor with positive EV; otherwise it is rejected with its reason.
Nothing here defines the floor; it is imported.

Only half lines are offered (AD-21, spec-13-1). Review (`evaluate_exact`) still prices a whole line, push-protected (the stake is
refunded on the exact number): the floor applies to P(win) + P(push) and EV = P(win) * (odds - 1) - P(lose), as for soccer (spec-2-4).
"""
from __future__ import annotations

import math
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from typing import NamedTuple, Sequence

from predictions.core.inputs import Input
from predictions.core.market_rules import SIDES_PLAYED, offers_without_whole_lines, without_unders, without_whole_lines
from predictions.core.model import Estimate, Evaluation, FixturePass, percent, validate
from predictions.core.types import Fixture, MarketOffer

SPORT = "basketball"

# ---- model (Story 6.1) ----
MODEL_NAME = "normal-points-v1"
HOME_COURT_POINTS = 2.0          # home-court edge in points, moves the margin only, never the game total
TOTAL_SD_FRACTION = 0.085        # sd of the game total = this * expected total (about 19 points at 225)
TEAM_SD_FRACTION = 0.115         # sd of one team's points = this * its expected points (about 13 at 112)
MARGIN_SD_FRACTION = 0.058       # sd of the margin = this * expected game total (about 13 points at 225)
FH_SD_FRACTION = 0.115           # sd of the 1st-half total = this * its expected value (about 13 at 112)
MIN_LINE = Decimal("20")         # lines outside this range are not a basketball points line
MAX_LINE = Decimal("400")
MAX_SPREAD = Decimal("40")

# ---- Story 6.2: ORTG lock ----
ORTG_LOCK_THRESHOLD = 108.0      # rolling offensive rating (points per 100 possessions) that locks Team Total Over

# ---- Story 6.2: 1H fast-pace route. "Explosive-starting" is a PROXY (spec-6-2 Design Notes): a team is explosive
# when it scores at least EXPLOSIVE_MIN_FH_SHARE of its points in the first half AND plays at least
# EXPLOSIVE_MIN_PACE_RATIO times the league's average pace. With no `match.league_pace` input, or no first-half
# data, nobody is explosive (fail closed: the rule simply does not apply).
EXPLOSIVE_MIN_FH_SHARE = 0.515
EXPLOSIVE_MIN_PACE_RATIO = 1.0

# ---- SportyBet market ids (probed on the public listing, spec-6-1) ----
WINNER_MARKET_ID = "219"         # Winner (incl. overtime): Home / Away
HANDICAP_MARKET_ID = "223"       # Handicap (incl. overtime), specifier hcp=<home handicap>
TOTAL_MARKET_ID = "225"          # Over/Under (incl. overtime), specifier total=<line>
HOME_TOTAL_MARKET_ID = "227"     # Home O/U (incl. overtime)
AWAY_TOTAL_MARKET_ID = "228"     # Away O/U (incl. overtime)
FH_TOTAL_MARKET_ID = "68"        # 1st half - total
MARKET_KINDS = {WINNER_MARKET_ID: "winner", HANDICAP_MARKET_ID: "handicap", TOTAL_MARKET_ID: "total",
                HOME_TOTAL_MARKET_ID: "home_total", AWAY_TOTAL_MARKET_ID: "away_total", FH_TOTAL_MARKET_ID: "fh_total"}
LIST_MARKET_IDS = ",".join(MARKET_KINDS)  # what the fixture listing asks SportyBet for

RULE_ORTG = "ORTG lock"
RULE_FAST_1H = "1H fast-pace route"
INSUFFICIENT_DATA = "insufficient data"
SIDES = ("home", "away")


class MissingInput(Exception):
    pass


# ---------------------------------------------------------------- probability helpers
def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def line_probabilities(mu: float, sd: float, line: Decimal) -> tuple[float, float, float] | None:
    """(P(over wins), P(under wins), P(push)) for an integer-valued outcome with mean `mu` and spread `sd`.

    Half line x.5: no push. Whole line N: N itself is a push (refund). Quarter lines are not supported (None)."""
    if sd <= 0:
        return None
    if abs(line) % 1 == Decimal("0.5"):
        under = normal_cdf((float(line) - mu) / sd)
        return 1.0 - under, under, 0.0
    if line % 1 == 0:
        n = float(line)
        over = 1.0 - normal_cdf((n + 0.5 - mu) / sd)
        under = normal_cdf((n - 0.5 - mu) / sd)
        return over, under, max(0.0, 1.0 - over - under)
    return None


def moneyline_probability(margin_mu: float, margin_sd: float) -> float:
    """P(home wins). Overtime means no draw, so the margin is never 0 and the boundary is 0."""
    return 1.0 - normal_cdf((0.0 - margin_mu) / margin_sd)


def fmt_line(line: Decimal) -> str:
    return format(line.normalize(), "f")


def fmt_signed(line: Decimal) -> str:
    text = fmt_line(line)
    return text if text.startswith("-") or line == 0 else "+" + text


def _spec_value(specifier: str, key: str) -> Decimal | None:
    name, _, value = specifier.partition("=")
    if name.strip() != key:
        return None
    try:
        number = Decimal(value.strip())
    except InvalidOperation:
        return None
    return number if number.is_finite() else None


def total_line(specifier: str) -> Decimal | None:
    line = _spec_value(specifier, "total")
    return line if line is not None and MIN_LINE <= line <= MAX_LINE else None


def handicap_line(specifier: str) -> Decimal | None:
    line = _spec_value(specifier, "hcp")
    return line if line is not None and abs(line) <= MAX_SPREAD else None


def market_kind(offer: MarketOffer) -> str | None:
    return MARKET_KINDS.get(offer.market_id)


def _outcome(offer: MarketOffer, side: str):
    found = [o for o in offer.outcomes if o.name.strip().lower().startswith(side)]
    return found[0] if len(found) == 1 else None


# ---------------------------------------------------------------- the model of one fixture
class _Model:
    """Everything priced from one place, so totals, team totals, first half and margin never disagree."""

    def __init__(self, inputs: Sequence[Input]):
        self.used = tuple(sorted(inputs, key=lambda i: i.name))
        self.by = {i.name: i for i in self.used}

        def need(name: str) -> float:
            if name not in self.by:
                raise MissingInput(name)
            return self.by[name].value

        self.ortg = {s: need(f"{s}.ortg") for s in SIDES}
        self.drtg = {s: need(f"{s}.drtg") for s in SIDES}
        self.pace_side = {s: need(f"{s}.pace") for s in SIDES}
        self.pace = (self.pace_side["home"] + self.pace_side["away"]) / 2
        self.home = self.pace * (self.ortg["home"] + self.drtg["away"]) / 200 + HOME_COURT_POINTS / 2
        self.away = self.pace * (self.ortg["away"] + self.drtg["home"]) / 200 - HOME_COURT_POINTS / 2
        self.total = self.home + self.away
        self.margin = self.home - self.away
        fh = [f"{s}.fh_points_{k}" for s in SIDES for k in ("for", "against")]
        self.has_fh = all(n in self.by for n in fh)
        if self.has_fh:
            self.fh_home = (self.by["home.fh_points_for"].value + self.by["away.fh_points_against"].value) / 2
            self.fh_away = (self.by["away.fh_points_for"].value + self.by["home.fh_points_against"].value) / 2
            self.fh_total = self.fh_home + self.fh_away
        self.league_pace = self.by["match.league_pace"].value if "match.league_pace" in self.by else None

    def team_mu(self, side: str) -> float:
        return self.home if side == "home" else self.away

    def points_for(self, side: str) -> float:
        item = self.by.get(f"{side}.points_for")
        return item.value if item is not None else self.ortg[side] * self.pace_side[side] / 100

    def explosive(self, side: str) -> bool:
        """The documented proxy for an explosive-starting team (see the constants above)."""
        if not self.has_fh or self.league_pace is None or self.league_pace <= 0:
            return False
        points = self.points_for(side)
        if points <= 0:
            return False
        share = self.by[f"{side}.fh_points_for"].value / points
        return share >= EXPLOSIVE_MIN_FH_SHARE and self.pace_side[side] >= EXPLOSIVE_MIN_PACE_RATIO * self.league_pace

    def summary(self, *names: str) -> str:
        shown = [self.by[n] for n in names if n in self.by]
        parts = ", ".join(f"{i.name} {i.value:g} {i.unit} [{i.source}]" for i in shown)
        return (f"inputs: {parts}; expected points {self.home:.1f} - {self.away:.1f} (total {self.total:.1f}), "
                f"{MODEL_NAME}")


class _Priced(NamedTuple):
    kind: str   # winner, handicap, total, home_total, away_total, fh_total
    side: str   # over / under / home / away
    line: Decimal | None
    evaluation: Evaluation


def _with(evaluation: Evaluation, **changes) -> Evaluation:
    return replace(evaluation, **changes)


def _describe(evaluation: Evaluation) -> str:
    return f"{evaluation.selection} @ {evaluation.odds:.2f} (estimate {percent(evaluation.estimate.probability)})"


def _push_note(estimate: Estimate, line: Decimal) -> str:
    if not estimate.push:
        return ""
    return (f"refund on exactly {fmt_line(line)}: win {percent(estimate.win)} + refund {percent(estimate.push)} = "
            f"success {percent(estimate.probability)}")


class BasketballEngine:
    sport = SPORT

    # ---- fixture-level pass: basketball has no derby or cup rule ----
    def screen(self, fixture: Fixture, inputs: Sequence[Input]) -> FixturePass | None:
        return None

    def understands(self, offer: MarketOffer) -> bool:
        return market_kind(offer) is not None

    # ---- picks with Optimization (Matchday-style slate) ----
    def evaluate(self, fixture: Fixture, inputs: Sequence[Input],
                 market_options: Sequence[MarketOffer]) -> list[Evaluation]:
        # owner rules: no Under (AD-20), no whole or quarter line (AD-21): half-line offers only are ever priced
        return without_whole_lines(without_unders(self._evaluate(fixture, inputs, offers_without_whole_lines(market_options))))

    def _evaluate(self, fixture: Fixture, inputs: Sequence[Input],
                  market_options: Sequence[MarketOffer]) -> list[Evaluation]:
        try:
            model = _Model(inputs)
        except MissingInput:
            return []
        priced = _Pricer(fixture, model, market_options)
        results: list[Evaluation] = []

        full = priced.line_market("total")
        locked = [s for s in SIDES if model.ortg[s] >= ORTG_LOCK_THRESHOLD]
        explosive = [s for s in SIDES if model.explosive(s)]
        original = _best(full)
        original_text = _describe(original.evaluation) if original is not None else ""

        if locked or explosive:
            pass  # full-game totals are bypassed: the converted picks below are the output
        else:
            results.extend(_shop(full))

        # -- team totals: a locked team gets Team Total Over only (the rule); the other side is priced as usual
        for side in SIDES:
            kind = f"{side}_total"
            candidates = priced.line_market(kind)
            if side in locked:
                team = fixture.home if side == "home" else fixture.away
                rule = (f"{RULE_ORTG}: {team} rolling ORTG {model.ortg[side]:.1f} is at least "
                        f"{ORTG_LOCK_THRESHOLD:.1f}, so Team Total Over is the primary output and full-game "
                        f"totals are bypassed")
                overs = [p for p in candidates if p.side == "over"]
                results.extend(_converted(_shop(overs), rule, original_text))
            else:
                results.extend(_shop(candidates))

        # -- first half: explosive starts route to the 1st-half total Over; otherwise both sides are priced
        fh = priced.line_market("fh_total")
        if explosive:
            who = " and ".join(fixture.home if s == "home" else fixture.away for s in explosive)
            rule = (f"{RULE_FAST_1H}: {who} start fast (at least {EXPLOSIVE_MIN_FH_SHARE:.1%} of points in the first "
                    f"half, pace at least {EXPLOSIVE_MIN_PACE_RATIO:.2f} x the league's), so the pick routes to a "
                    f"1st-half total")
            results.extend(_converted(_shop([p for p in fh if p.side == "over"]), rule, original_text))
        else:
            results.extend(_shop(fh))

        results.extend(_shop(priced.line_market("handicap")))
        results.extend(_shop(priced.line_market("winner")))
        return results

    # ---- exact selections for Review: no optimization, the code's own odds ----
    def evaluate_exact(self, fixture: Fixture, inputs: Sequence[Input], offer: MarketOffer) -> list[Evaluation]:
        return without_unders(self._evaluate_exact(fixture, inputs, offer))   # owner rule: no Under (AD-20)

    def _evaluate_exact(self, fixture: Fixture, inputs: Sequence[Input], offer: MarketOffer) -> list[Evaluation]:
        try:
            model = _Model(inputs)
        except MissingInput:
            return []
        kind = market_kind(offer)
        if kind is None:
            return []
        found = _Pricer(fixture, model, [offer]).line_market(kind, exact=True)
        results = [p.evaluation for p in found]
        if kind in ("total", "home_total", "away_total", "fh_total") and results:
            lock_note = self._review_note(fixture, model, kind)
            if lock_note:
                results = [_with(e, notes=(*e.notes, lock_note)) for e in results]
        return results

    @staticmethod
    def _review_note(fixture: Fixture, model: _Model, kind: str) -> str:
        if kind != "total":
            return ""
        locked = [fixture.home if s == "home" else fixture.away for s in SIDES if model.ortg[s] >= ORTG_LOCK_THRESHOLD]
        if not locked:
            return ""
        return (f"{RULE_ORTG} applies to {' and '.join(locked)} (ORTG at least {ORTG_LOCK_THRESHOLD:.1f}): a Team "
                f"Total Over would be the primary pick. A Review judges this selection as it is and never swaps it.")


class _Pricer:
    def __init__(self, fixture: Fixture, model: _Model, offers: Sequence[MarketOffer]):
        self.fixture, self.model, self.offers = fixture, model, list(offers)

    def _evaluation(self, market: str, selection: str, odds: Decimal, estimate: Estimate,
                    notes: tuple[str, ...]) -> Evaluation:
        passed, ev, reason = validate(estimate, odds)
        return Evaluation(self.fixture.match_id, market, selection, odds, estimate, ev, passed, reason,
                          "", "", tuple(n for n in notes if n))

    def _insufficient(self, market: str, selection: str, odds: Decimal, why: str) -> Evaluation:
        return Evaluation(self.fixture.match_id, market, selection, odds, Estimate(0.0, self.model.used, MODEL_NAME),
                          Decimal(0), False, f"{INSUFFICIENT_DATA}: {why}", "", "", (), True)

    def _estimate(self, over: float, under: float, push: float, side: str) -> Estimate:
        win = over if side in ("over", "home") else under
        if push:
            return Estimate.push_protected(win, push, self.model.used, MODEL_NAME)
        return Estimate(min(win, 1.0), self.model.used, MODEL_NAME)

    def line_market(self, kind: str, exact: bool = False) -> list[_Priced]:
        """Every listed line of one kind, priced on each listed side. `exact` (Review) keeps only listed outcomes."""
        offers = [o for o in self.offers if market_kind(o) == kind]
        if kind == "winner":
            return self._winner(offers)
        if kind == "handicap":
            return self._handicap(offers)
        return self._points(kind, offers)

    # -- totals of points: game, team, first half
    def _points(self, kind: str, offers: list[MarketOffer]) -> list[_Priced]:
        m = self.model
        if kind == "total":
            mu, sd, label, prefix = m.total, TOTAL_SD_FRACTION * m.total, "Total points", ""
        elif kind == "fh_total":
            if not m.has_fh:
                return self._insufficient_fh(offers)
            mu, sd, label, prefix = m.fh_total, FH_SD_FRACTION * m.fh_total, "1st half total points", "1st half "
        else:
            team = "home" if kind == "home_total" else "away"
            mu, sd = m.team_mu(team), TEAM_SD_FRACTION * m.team_mu(team)
            label, prefix = f"{team.title()} team total points", f"{team.title()} team "
        shown = [f"{s}.{k}" for s in SIDES for k in ("ortg", "drtg", "pace")]
        if kind == "fh_total":
            shown = [f"{s}.fh_points_{k}" for s in SIDES for k in ("for", "against")]
        note = m.summary(*shown) + f"; normal sd {sd:.1f}"
        results: list[_Priced] = []
        for offer in offers:
            line = total_line(offer.specifier)
            probs = line_probabilities(mu, sd, line) if line is not None else None
            if probs is None:
                continue
            over, under, push = probs
            for side in SIDES_PLAYED:
                outcome = _outcome(offer, side)
                if outcome is None:
                    continue
                estimate = self._estimate(over, under, push, side)
                evaluation = self._evaluation(f"{label} {fmt_line(line)}", f"{prefix}{side.title()} {fmt_line(line)}",
                                              outcome.odds, estimate, (note, _push_note(estimate, line)))
                results.append(_Priced(kind, side, line, evaluation))
        return results

    def _insufficient_fh(self, offers: list[MarketOffer]) -> list[_Priced]:
        results = []
        for offer in offers:
            line = total_line(offer.specifier)
            if line is None:
                continue
            for side in SIDES_PLAYED:
                outcome = _outcome(offer, side)
                if outcome is not None:
                    why = "no first-half scoring data for one or both teams"
                    results.append(_Priced("fh_total", side, line, self._insufficient(
                        f"1st half total points {fmt_line(line)}", f"1st half {side.title()} {fmt_line(line)}",
                        outcome.odds, why)))
        return results

    # -- winner (moneyline, overtime included)
    def _winner(self, offers: list[MarketOffer]) -> list[_Priced]:
        m = self.model
        margin_sd = MARGIN_SD_FRACTION * m.total
        p_home = moneyline_probability(m.margin, margin_sd)
        note = m.summary("home.ortg", "home.drtg", "away.ortg", "away.drtg") + (
            f"; expected margin {m.margin:+.1f} (home court {HOME_COURT_POINTS:g}), normal sd {margin_sd:.1f}")
        results = []
        for offer in offers:
            for side, p in (("home", p_home), ("away", 1.0 - p_home)):
                outcome = _outcome(offer, side)
                if outcome is not None:
                    results.append(_Priced("winner", side, None, self._evaluation(
                        "Winner (incl. OT)", side.title(), outcome.odds, Estimate(min(p, 1.0), m.used, MODEL_NAME),
                        (note,))))
        return results

    # -- handicap (spread, overtime included). The specifier is the HOME handicap.
    def _handicap(self, offers: list[MarketOffer]) -> list[_Priced]:
        m = self.model
        margin_sd = MARGIN_SD_FRACTION * m.total
        note = m.summary("home.ortg", "home.drtg", "away.ortg", "away.drtg") + (
            f"; expected margin {m.margin:+.1f}, normal sd {margin_sd:.1f}")
        results = []
        for offer in offers:
            hcp = handicap_line(offer.specifier)
            if hcp is None:
                continue
            # Home covers when margin + hcp > 0, i.e. margin > -hcp; away covers when margin < -hcp.
            probs = line_probabilities(m.margin, margin_sd, -hcp)
            if probs is None:
                continue
            over, under, push = probs
            for side, shown in (("home", hcp), ("away", -hcp)):
                outcome = _outcome(offer, side)
                if outcome is None:
                    continue
                estimate = self._estimate(over, under, push, "home" if side == "home" else "away")
                results.append(_Priced("handicap", side, hcp, self._evaluation(
                    f"Handicap {fmt_signed(hcp)}", f"{side.title()} ({fmt_signed(shown)})", outcome.odds, estimate,
                    (note, _push_note(estimate, -hcp)))))
        return results


# ---------------------------------------------------------------- line shopping
def _rank(p: _Priced):
    """A passing pick beats a rejected one; then the higher estimate wins (the owner's goal is success at 65% or
    better, not the longest price), then the higher EV, then the higher odds."""
    e = p.evaluation
    return (e.passed, e.estimate.probability, e.ev, e.odds)


def _best(candidates: Sequence[_Priced]) -> _Priced | None:
    """The pick the owner would take among these: a passing one with the highest estimate (so the safest line that
    still has positive EV), else the closest to passing."""
    live = [p for p in candidates if not p.evaluation.insufficient]
    return max(live, key=_rank) if live else None


def _shop(candidates: Sequence[_Priced]) -> list[Evaluation]:
    """One evaluation per (kind, side): the passing line with the highest estimate (positive EV is part of passing),
    or, when none passes, the closest rejection. A listing has 10 to 30 lines per market: the owner is shown one
    line, not every line. Choosing by estimate, not by EV, keeps the tail lines (a 99.9% estimate at odds of 4.20,
    which only an over-confident model would show) from winning just because their price is long."""
    groups: dict[tuple[str, str], list[_Priced]] = {}
    for p in candidates:
        groups.setdefault((p.kind, p.side), []).append(p)
    out: list[Evaluation] = []
    for key in sorted(groups):
        members = groups[key]
        real = [p for p in members if not p.evaluation.insufficient]
        chosen = _best(real) if real else members[0]
        if chosen is not None:
            out.append(chosen.evaluation)
    return out


def _converted(shopped: Sequence[Evaluation], rule: str, original: str) -> list[Evaluation]:
    """A rule turned the full-game total into these picks: show the rule and the pick it replaced."""
    return [_with(e, rule=rule, original=original) for e in shopped]
