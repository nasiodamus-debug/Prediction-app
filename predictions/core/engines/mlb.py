"""MLB engine (Story 6.3, sport "baseball"). Standard library only. Everything here is deterministic.

Model (`negbin-runs-v1`). Each side's runs in nine innings are a negative-binomial variable (baseball runs are
over-dispersed: variance about twice the mean), the two sides independent. The mean of a side combines its own
scoring with the opponent's run prevention against the league level (a "log5" style ratio),
    runs(A) = A runs scored * B runs allowed / league runs per team-game
nudged by the probable starting pitcher of the opposing side, then by a small home factor. The first five innings (F5)
use the same recipe on the teams' F5 figures with a larger pitcher weight, because a starter throws most of them.
From the distributions come the full-game total, both team totals, the run line and the winner (a tie after nine
innings goes to extra innings: the home side wins it with EXTRA_INNINGS_HOME_WIN and the total is the tied score plus
at least one run), and the F5 total, F5 run line and F5 result (a tie after five is a real result, the "draw").

Optimization (Story 6.3): an F5 TEAM TOTAL Over whose estimate is 65% to 72% steps down to Combined F5 Game Total
Over 3.5 and stays only if that clears the floor with positive EV; the original and the rule are shown. SportyBet NG
does not list F5 team totals (probed 2026-09-30), so the rule is dormant on today's listing; it runs on any offer whose
market name is an F5 team total and is covered by tests with a fabricated offer (spec-6-3 Open Issues).

Only half lines are offered (AD-21, spec-13-1). Review (`evaluate_exact`) still prices a whole line, push-protected (refund on the
exact number): the floor applies to P(win) + P(push) and EV = P(win) * (odds - 1) - P(lose), as for soccer (spec-2-4). The floor is imported, never defined here.
"""
from __future__ import annotations

import re
from decimal import Decimal
from typing import Sequence

from predictions.core.engines.common import (Priced, best, describe, find_outcome, fmt_line, fmt_signed,
                                             handicap_probabilities, joint, line_probabilities, margin_and_total,
                                             negbin_pmf, push_note, shop, spec_decimal, spec_is_only, with_)
from predictions.core.inputs import Input
from predictions.core.market_rules import SIDES_PLAYED, offers_without_whole_lines, without_unders, without_whole_lines
from predictions.core.model import PROBABILITY_FLOOR, Estimate, Evaluation, FixturePass, percent, validate
from predictions.core.types import Fixture, MarketOffer, normalize_team

SPORT = "baseball"

# ---- model ----
MODEL_NAME = "negbin-runs-v1"
RUN_DISPERSION = 2.0             # variance / mean of a team's runs in nine innings (judgement, MLB is about 2.1)
F5_DISPERSION = 1.8              # the same for the first five innings (judgement)
HOME_RUNS_FACTOR = 1.02          # the home side scores this much more, the away side this much less (judgement)
EXTRA_INNINGS_HOME_WIN = 0.52    # the home side bats last in extra innings (judgement)
EXTRA_INNINGS_MIN_RUNS = 1       # a game decided in extra innings has at least this many runs more than the tied score
SP_WEIGHT_FULL = 0.30            # share of the starter's effect applied to a nine-inning mean (judgement)
SP_WEIGHT_F5 = 0.50              # ... and to the F5 mean (a starter throws most of the first five)
SP_RATIO_MIN, SP_RATIO_MAX = 0.60, 1.60   # a starter's runs per 9 against the league level is kept inside these
SP_MIN_STARTS = 3                # a starter with fewer starts than this is not used (no sample)
MIN_TEAM_RUNS, MAX_TEAM_RUNS = 1.0, 9.0       # a side's nine-inning mean is kept inside these
MIN_F5_RUNS, MAX_F5_RUNS = 0.5, 6.0
MIN_LINE = Decimal("0.5")
MAX_LINE = Decimal("30")
MAX_SPREAD = Decimal("12")

# ---- Story 6.3: F5 team total step-down ----
F5_STEP_MAX = 0.72                   # an estimate from the floor up to this steps down (the floor itself is core.model's)
F5_STEP_LINE = Decimal("3.5")        # Combined F5 Game Total Over 3.5
F5_TEAM_TOTAL_WORDS = ("innings 1 to 5", "first 5 innings", "first five innings", "f5")
RULE_F5 = "F5 step-down"

# ---- SportyBet market ids (probed on the public listing and game page, spec-6-3) ----
WINNER_MARKET_ID = "251"         # Winner (incl. extra innings): Home / Away
HANDICAP_MARKET_ID = "256"       # Handicap (incl. extra innings), specifier hcp=<home handicap>
TOTAL_MARKET_ID = "258"          # Total (incl. extra innings), specifier total=<line>
HOME_TOTAL_MARKET_ID = "260"     # Competitor1 (home) total
AWAY_TOTAL_MARKET_ID = "261"     # Competitor2 (away) total
F5_RESULT_MARKET_ID = "274"      # Innings 1 to 5 - 1x2: Home / Draw / Away
F5_HANDICAP_MARKET_ID = "275"    # Innings 1 to 5 - handicap, hcp=<home handicap>
F5_TOTAL_MARKET_ID = "276"       # Innings 1 to 5 - total, total=<line> (whole and half lines)
MARKET_KINDS = {WINNER_MARKET_ID: "winner", HANDICAP_MARKET_ID: "handicap", TOTAL_MARKET_ID: "total",
                HOME_TOTAL_MARKET_ID: "home_total", AWAY_TOTAL_MARKET_ID: "away_total",
                F5_RESULT_MARKET_ID: "f5_result", F5_HANDICAP_MARKET_ID: "f5_handicap", F5_TOTAL_MARKET_ID: "f5_total"}
LIST_MARKET_IDS = ",".join(MARKET_KINDS)  # what the fixture listing asks SportyBet for

WINNER_MARKET_TEXT = "Winner (incl. extra innings)"
F5_RESULT_MARKET_TEXT = "F5 result (1X2)"
INSUFFICIENT_DATA = "insufficient data"
SIDES = ("home", "away")


class MissingInput(Exception):
    pass


def market_kind(offer: MarketOffer) -> str | None:
    """The kind of market, by SportyBet id; an F5 team total (no id known, none listed) by its name."""
    kind = MARKET_KINDS.get(offer.market_id)
    if kind is not None:
        return kind
    name = normalize_team(offer.name)
    if "total" in name and any(w in name for w in F5_TEAM_TOTAL_WORDS) and re.search(
            r"\b(competitor ?[12]|home|away)\b", name):
        return "f5_team_total"
    return None


def f5_team_side(offer: MarketOffer, fixture: Fixture | None = None) -> str | None:
    """"home" or "away" for an F5 team total offer (by competitor number, side word, or the fixture's team name)."""
    name = normalize_team(offer.name)
    if re.search(r"\bcompetitor ?1\b|\bhome\b", name) or (fixture and normalize_team(fixture.home) in name):
        return "home"
    if re.search(r"\bcompetitor ?2\b|\baway\b", name) or (fixture and normalize_team(fixture.away) in name):
        return "away"
    return None


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


# ---------------------------------------------------------------- the model of one fixture
class _Model:
    def __init__(self, inputs: Sequence[Input]):
        self.used = tuple(sorted(inputs, key=lambda i: i.name))
        self.by = {i.name: i for i in self.used}

        def need(name: str) -> float:
            if name not in self.by:
                raise MissingInput(name)
            return self.by[name].value

        self.rf = {s: need(f"{s}.runs_for") for s in SIDES}
        self.ra = {s: need(f"{s}.runs_against") for s in SIDES}
        league = self.by.get("match.league_runs")
        self.league = league.value if league is not None else (sum(self.rf.values()) + sum(self.ra.values())) / 4
        # the starter of side s is judged against the level of runs per nine innings of the league
        self.sp = {}
        for s in SIDES:
            era, starts = self.by.get(f"{s}.sp_runs_per9"), self.by.get(f"{s}.sp_starts")
            ok = era is not None and starts is not None and starts.value >= SP_MIN_STARTS
            self.sp[s] = _clamp(era.value / self.league, SP_RATIO_MIN, SP_RATIO_MAX) if ok else None
        self.mu = {s: self._full(s) for s in SIDES}
        f5 = [f"{s}.f5_runs_{k}" for s in SIDES for k in ("for", "against")]
        self.has_f5 = all(n in self.by for n in f5) and all(self.sp[s] is not None for s in SIDES)
        self.f5_missing = ("first-five-inning scoring data for one or both teams"
                           if not all(n in self.by for n in f5) else
                           "no probable starting pitcher with at least "
                           f"{SP_MIN_STARTS} starts for one or both teams")
        if self.has_f5:
            league5 = self.by.get("match.league_f5_runs")
            self.league5 = league5.value if league5 is not None else sum(self.by[n].value for n in f5) / 4
            self.mu5 = {s: self._f5(s) for s in SIDES}
        self._grid = None
        self._grid5 = None

    def _home(self, side: str) -> float:
        return HOME_RUNS_FACTOR if side == "home" else 2.0 - HOME_RUNS_FACTOR

    def _full(self, side: str) -> float:
        other = "away" if side == "home" else "home"
        base = self.rf[side] * self.ra[other] / self.league
        if self.sp[other] is not None:
            base *= 1.0 + SP_WEIGHT_FULL * (self.sp[other] - 1.0)
        return _clamp(base * self._home(side), MIN_TEAM_RUNS, MAX_TEAM_RUNS)

    def _f5(self, side: str) -> float:
        other = "away" if side == "home" else "home"
        base = self.by[f"{side}.f5_runs_for"].value * self.by[f"{other}.f5_runs_against"].value / self.league5
        base *= 1.0 + SP_WEIGHT_F5 * (self.sp[other] - 1.0)
        return _clamp(base * self._home(side), MIN_F5_RUNS, MAX_F5_RUNS)

    # ---- distributions (built once, on demand)
    def grid(self):
        if self._grid is None:
            self._grid = joint(negbin_pmf(self.mu["home"], RUN_DISPERSION), negbin_pmf(self.mu["away"], RUN_DISPERSION))
        return self._grid

    def full(self):
        """(margin, total) with extra innings resolving ties."""
        return margin_and_total(self.grid(), EXTRA_INNINGS_HOME_WIN, EXTRA_INNINGS_MIN_RUNS)

    def team_dist(self, side: str) -> dict[int, float]:
        pmf = negbin_pmf(self.mu[side], RUN_DISPERSION)
        return dict(enumerate(pmf))

    def grid5(self):
        if self._grid5 is None:
            self._grid5 = joint(negbin_pmf(self.mu5["home"], F5_DISPERSION), negbin_pmf(self.mu5["away"], F5_DISPERSION))
        return self._grid5

    def f5(self):
        """(margin, total) after five innings; a tie stays a tie."""
        return margin_and_total(self.grid5())

    def f5_team_dist(self, side: str) -> dict[int, float]:
        return dict(enumerate(negbin_pmf(self.mu5[side], F5_DISPERSION)))

    def summary(self, *names: str, f5: bool = False) -> str:
        shown = [self.by[n] for n in names if n in self.by]
        parts = ", ".join(f"{i.name} {i.value:g} {i.unit} [{i.source}]" for i in shown)
        expected = (f"F5 expected runs {self.mu5['home']:.2f} - {self.mu5['away']:.2f}" if f5 else
                    f"expected runs {self.mu['home']:.2f} - {self.mu['away']:.2f} (total {sum(self.mu.values()):.2f})")
        starters = ", ".join(f"{s} starter {self.sp[s]:.2f} x league" for s in SIDES if self.sp[s] is not None)
        return f"inputs: {parts}; {expected}" + (f"; {starters}" if starters else "") + f", {MODEL_NAME}"


# ---------------------------------------------------------------- pricing the offers
class _Pricer:
    def __init__(self, fixture: Fixture, model: _Model, offers: Sequence[MarketOffer]):
        self.fixture, self.model, self.offers = fixture, model, list(offers)

    def _evaluation(self, market: str, selection: str, odds: Decimal, estimate: Estimate,
                    notes: tuple[str, ...]) -> Evaluation:
        passed, ev, reason = validate(estimate, odds)
        return Evaluation(self.fixture.match_id, market, selection, odds, estimate, ev, passed, reason,
                          "", "", tuple(n for n in notes if n))

    def _insufficient(self, market: str, selection: str, odds: Decimal, why: str) -> Evaluation:
        return Evaluation(self.fixture.match_id, market, selection, odds,
                          Estimate(0.0, self.model.used, MODEL_NAME), Decimal(0), False,
                          f"{INSUFFICIENT_DATA}: {why}", "", "", (), True)

    def _estimate(self, over: float, under: float, push: float, side: str) -> Estimate:
        win = over if side in ("over", "home") else under
        if push:
            return Estimate.push_protected(win, push, self.model.used, MODEL_NAME)
        return Estimate(min(win, 1.0), self.model.used, MODEL_NAME)

    def market(self, kind: str, exact: bool = False) -> list[Priced]:
        offers = [o for o in self.offers if market_kind(o) == kind]
        if kind in ("f5_result", "f5_handicap", "f5_total", "f5_team_total") and not self.model.has_f5:
            return self._no_f5(kind, offers)
        if kind == "winner":
            return self._winner(offers)
        if kind == "handicap":
            return self._handicap(offers, "handicap", "Run line", "", self.model.full()[0])
        if kind == "f5_handicap":
            return self._handicap(offers, "f5_handicap", "F5 handicap", "F5 ", self.model.f5()[0])
        if kind == "total":
            return self._totals(offers, "total", "Total runs", "", self.model.full()[1],
                                self.model.summary("home.runs_for", "home.runs_against", "away.runs_for",
                                                   "away.runs_against", "home.sp_runs_per9", "away.sp_runs_per9"))
        if kind in ("home_total", "away_total"):
            side = kind.split("_")[0]
            return self._totals(offers, kind, f"{side.title()} team total runs", f"{side.title()} team ",
                                self.model.team_dist(side),
                                self.model.summary(f"{side}.runs_for", f"{'away' if side == 'home' else 'home'}"
                                                   f".runs_against", f"{'away' if side == 'home' else 'home'}.sp_runs_per9"))
        if kind == "f5_total":
            return self._totals(offers, kind, "F5 total runs", "F5 ", self.model.f5()[1], self._f5_summary())
        if kind == "f5_result":
            return self._f5_result(offers)
        if kind == "f5_team_total":
            return self._f5_team_totals(offers)
        return []

    def _f5_summary(self) -> str:
        m = self.model
        return m.summary("home.f5_runs_for", "home.f5_runs_against", "away.f5_runs_for", "away.f5_runs_against",
                         "home.sp_runs_per9", "away.sp_runs_per9", f5=True)

    def _no_f5(self, kind: str, offers: list[MarketOffer]) -> list[Priced]:
        results = []
        for offer in offers:
            line = spec_decimal(offer.specifier, "total") if kind != "f5_result" else None
            if kind == "f5_handicap":
                line = spec_decimal(offer.specifier, "hcp")
            for outcome in offer.outcomes:
                side = outcome.name.strip().lower().split(" ")[0]
                if side not in ("over", "home", "away", "draw"):   # no Under (AD-20)
                    continue
                label = f"{fmt_line(line)}" if line is not None else ""
                results.append(Priced(kind, side, line, self._insufficient(
                    f"F5 {kind.replace('f5_', '').replace('_', ' ')} {label}".strip(), f"F5 {outcome.name}",
                    outcome.odds, self.model.f5_missing)))
        return results

    def _totals(self, offers, kind, label, prefix, dist, note) -> list[Priced]:
        results: list[Priced] = []
        for offer in offers:
            if not spec_is_only(offer.specifier, "total"):
                continue
            line = spec_decimal(offer.specifier, "total")
            if line is None or not MIN_LINE <= line <= MAX_LINE:
                continue
            probs = line_probabilities(dist, line)
            if probs is None:
                continue
            over, under, push = probs
            for side in SIDES_PLAYED:
                outcome = find_outcome(offer, side)
                if outcome is None:
                    continue
                estimate = self._estimate(over, under, push, side)
                results.append(Priced(kind, side, line, self._evaluation(
                    f"{label} {fmt_line(line)}", f"{prefix}{side.title()} {fmt_line(line)}", outcome.odds, estimate,
                    (note, push_note(estimate, f"exactly {fmt_line(line)}")))))
        return results

    def _winner(self, offers) -> list[Priced]:
        margin, _ = self.model.full()
        p_home = sum(p for k, p in margin.items() if k > 0)
        note = self.model.summary("home.runs_for", "home.runs_against", "away.runs_for", "away.runs_against",
                                  "home.sp_runs_per9", "away.sp_runs_per9")
        results = []
        for offer in offers:
            for side, p in (("home", p_home), ("away", 1.0 - p_home)):
                outcome = find_outcome(offer, side)
                if outcome is not None:
                    results.append(Priced("winner", side, None, self._evaluation(
                        WINNER_MARKET_TEXT, side.title(), outcome.odds,
                        Estimate(min(p, 1.0), self.model.used, MODEL_NAME), (note,))))
        return results

    def _handicap(self, offers, kind, label, prefix, margin) -> list[Priced]:
        note = (self.model.summary("home.runs_for", "home.runs_against", "away.runs_for", "away.runs_against")
                if kind == "handicap" else self._f5_summary())
        results = []
        for offer in offers:
            if not spec_is_only(offer.specifier, "hcp"):
                continue
            hcp = spec_decimal(offer.specifier, "hcp")
            if hcp is None or abs(hcp) > MAX_SPREAD:
                continue
            probs = handicap_probabilities(margin, hcp)
            if probs is None:
                continue
            home, away, push = probs
            for side, shown in (("home", hcp), ("away", -hcp)):
                outcome = find_outcome(offer, side)
                if outcome is None:
                    continue
                estimate = self._estimate(home, away, push, side)
                results.append(Priced(kind, side, hcp, self._evaluation(
                    f"{label} {fmt_signed(hcp)}", f"{prefix}{side.title()} ({fmt_signed(shown)})", outcome.odds,
                    estimate, (note, push_note(estimate, f"a margin of exactly {fmt_line(abs(hcp))}")))))
        return results

    def _f5_result(self, offers) -> list[Priced]:
        margin, _ = self.model.f5()
        probs = {"home": sum(p for k, p in margin.items() if k > 0), "draw": margin.get(0, 0.0),
                 "away": sum(p for k, p in margin.items() if k < 0)}
        results = []
        for offer in offers:
            for side, p in probs.items():
                outcome = find_outcome(offer, side)
                if outcome is not None:
                    results.append(Priced("f5_result", side, None, self._evaluation(
                        F5_RESULT_MARKET_TEXT, f"F5 {side.title()}", outcome.odds,
                        Estimate(min(p, 1.0), self.model.used, MODEL_NAME), (self._f5_summary(),))))
        return results

    def _f5_team_totals(self, offers) -> list[Priced]:
        results = []
        for offer in offers:
            side = f5_team_side(offer, self.fixture)
            line = spec_decimal(offer.specifier, "total")
            if side is None or line is None or not MIN_LINE <= line <= MAX_LINE:
                continue
            over, under, push = line_probabilities(self.model.f5_team_dist(side), line) or (0, 0, 0)
            if not (over or under):
                continue
            for pick in SIDES_PLAYED:
                outcome = find_outcome(offer, pick)
                if outcome is None:
                    continue
                estimate = self._estimate(over, under, push, pick)
                results.append(Priced("f5_team_total_" + side, pick, line, self._evaluation(
                    f"F5 {side.title()} team total runs {fmt_line(line)}",
                    f"F5 {side.title()} team {pick.title()} {fmt_line(line)}", outcome.odds, estimate,
                    (self._f5_summary(), push_note(estimate, f"exactly {fmt_line(line)}")))))
        return results


# ---------------------------------------------------------------- the engine
class MlbEngine:
    sport = SPORT

    def screen(self, fixture: Fixture, inputs: Sequence[Input]) -> FixturePass | None:
        return None  # no derby or weather rule for baseball in the specification

    def understands(self, offer: MarketOffer) -> bool:
        return market_kind(offer) is not None

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
        pricer = _Pricer(fixture, model, market_options)
        results: list[Evaluation] = []
        for kind in ("winner", "handicap", "total", "home_total", "away_total", "f5_result", "f5_handicap"):
            results.extend(shop(pricer.market(kind)))
        f5_totals = pricer.market("f5_total")
        results.extend(shop(f5_totals))
        stepped = self._f5_team_total_rule(pricer, f5_totals)
        keys = {(e.market, e.selection) for e in stepped}
        results = [e for e in results if (e.market, e.selection) not in keys or e.rule]
        return results + stepped

    # ---- Story 6.3: F5 team total Over at 65% to 72% steps down to Combined F5 Game Total Over 3.5
    @staticmethod
    def _f5_team_total_rule(pricer: _Pricer, f5_totals: Sequence[Priced]) -> list[Evaluation]:
        team_totals = pricer.market("f5_team_total")
        overs = [p for p in team_totals if p.side == "over"]
        out: list[Evaluation] = []
        groups: dict[str, list[Priced]] = {}
        for p in overs:
            groups.setdefault(p.kind, []).append(p)
        for kind in sorted(groups):
            original = best(groups[kind])
            if original is None or original.evaluation.insufficient:
                if original is None and groups[kind]:
                    out.append(groups[kind][0].evaluation)
                continue
            e = original.evaluation
            if not PROBABILITY_FLOOR <= e.estimate.probability <= F5_STEP_MAX:
                out.extend(shop(groups[kind]))  # outside the zone the pick stands or falls on its own
                continue
            rule = (f"{RULE_F5}: an F5 team total Over estimated at {percent(e.estimate.probability)} (between "
                    f"{PROBABILITY_FLOOR:.0%} and {F5_STEP_MAX:.0%}) steps down to Combined F5 Game Total Over "
                    f"{fmt_line(F5_STEP_LINE)}")
            target = [p for p in f5_totals if p.side == "over" and p.line == F5_STEP_LINE
                      and not p.evaluation.insufficient]
            if target:
                out.append(with_(target[0].evaluation, rule=rule, original=describe(e)))
            else:
                out.append(with_(e, passed=False, ev=e.ev, rule=rule, original=describe(e),
                                 reason=f"the step-down target, Combined F5 Game Total Over {fmt_line(F5_STEP_LINE)}, "
                                        f"is not offered by SportyBet, so the original is not played"))
        return out

    # ---- exact selections for Review: no optimization, the code's own odds
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
        found = _Pricer(fixture, model, [offer]).market(kind, exact=True)
        return [p.evaluation for p in found]
