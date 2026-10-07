"""NHL engine (Stories 6.4 and 6.5, sport "hockey"). Standard library only. Everything here is deterministic.

Model (`poisson-goals-v1`). Each side's goals in regulation are Poisson, the two sides independent. A side's scoring
rate blends its expected goals for (a PROXY built from shot locations, see adapters/nhl/stats.py; the NHL publishes
no xG) with its actual goals, and the mean of a side is a "log5" style ratio against the league level,
    goals(A) = A rate for * B rate against / league goals per team-game     (home 3% up, away 3% down)
A regulation tie goes to overtime or a shootout: the home side wins it with a probability that leans on its share of
the expected goals, and a game decided there has one more goal than the tied score. From the one score grid come the
winner and handicaps (with or without overtime), full-game totals (with or without overtime), regulation team totals
and the first-period total (its own Poisson from the first-period rates).

Optimization (Story 6.5), thresholds as named constants below; every converted pick stays only if it clears the floor
with positive EV, and the original pick and the rule are shown:
  * PUCK LINE SHIELD: a -1.5 puck line on a heavy favourite converts to the underdog +1.5 (the -1.0 Asian puck line is gone:
    no whole-line play is ever offered, AD-21, spec-13-1); no listed +1.5 means the -1.5 is simply not played.
  * PERIOD 1 ROUTE: combined first-period expected goals ABOVE 2.20 route the pick to Period 1 Over 1.5 goals.
  * TEAM TOTAL LOCK: a team whose rolling expected goals for (xGF) is AT LEAST 3.40 locks the team total Over at the NEAREST
    LISTED HALF LINE at or above 3 (Over 3.5 when listed, else the next listed half line up); with no such half line listed the
    lock is dropped and the full-game totals are not bypassed (AD-21).
Only half lines are priced for offers. Review (`evaluate_exact`) still prices a whole line on a booking code, push-protected exactly
as in soccer (spec-2-4). The floor is imported, never defined here.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Sequence

from predictions.core.engines.common import (Priced, best, converted, describe, find_outcome, fmt_line, fmt_signed,
                                             handicap_probabilities, joint, line_probabilities, margin_and_total,
                                             parse_specifier, poisson_pmf, push_note, shop, spec_decimal,
                                             spec_is_only, with_)
from predictions.core.inputs import Input
from predictions.core.market_rules import SIDES_PLAYED, offers_without_whole_lines, without_unders, without_whole_lines
from predictions.core.model import Estimate, Evaluation, FixturePass, percent, validate
from predictions.core.types import Fixture, MarketOffer

SPORT = "hockey"

# ---- model (Story 6.4) ----
MODEL_NAME = "poisson-goals-v1"
XG_WEIGHT = 0.5                  # weight of the xG proxy against actual goals in a rate (judgement: the proxy is crude)
GOALS_WEIGHT = 1.0 - XG_WEIGHT
HOME_GOALS_FACTOR = 1.03         # the home side scores this much more, the away side this much less (judgement)
OT_LEAN = 0.5                    # how much of the expected-goal share tilts an overtime winner (0 = a coin flip)
OT_MIN_EXTRA_GOALS = 1           # a game decided after regulation has at least this many goals more than the tie
MIN_TEAM_GOALS, MAX_TEAM_GOALS = 1.0, 5.0     # a side's regulation mean is kept inside these
MIN_P1_GOALS, MAX_P1_GOALS = 0.25, 1.8        # ... and its first-period mean
MIN_LINE = Decimal("0.5")
MAX_LINE = Decimal("12")
MAX_SPREAD = Decimal("6")
P1_PERIOD = "1"                  # the only period the engine prices

# ---- Story 6.5: puck-line shield ----
HEAVY_FAVOURITE_MIN_WIN = 0.60   # a side whose win probability (overtime included) is at least this is a heavy favourite
PUCK_LINE = Decimal("-1.5")      # the favourite's puck line that is shielded
UNDERDOG_LINE = Decimal("1.5")   # the alternative shield: the underdog with a goal and a half
RULE_PUCK = "Puck line shield"

# ---- Story 6.5: first-period route ----
P1_COMBINED_XG_THRESHOLD = 2.20  # combined P1 expected goals ABOVE this route to Period 1 Over 1.5 (exactly 2.20 does not)
P1_ROUTE_LINE = Decimal("1.5")
RULE_P1 = "Period 1 route"

# ---- Story 6.5: team-total lock ----
XGF_LOCK_THRESHOLD = 3.40        # a team with rolling xGF AT LEAST this locks the team total Over at the nearest listed half line
TEAM_TOTAL_LOCK_LINE = Decimal("3")   # ... at or above this (so Over 3.5, else the next listed half line up; none listed: no lock)
RULE_LOCK = "xGF lock"

# ---- SportyBet market ids (probed on the public listing and game page, spec-6-4) ----
WINNER_MARKET_ID = "406"         # Winner (incl. OT and penalties): Home / Away
HANDICAP_OT_MARKET_ID = "410"    # Handicap (incl. OT and penalties), the puck line, hcp=<home handicap>
HANDICAP_REG_MARKET_ID = "16"    # "Handicap 3 way" in regulation time; half lines have 2 outcomes, whole lines add a Draw (skipped)
TOTAL_OT_MARKET_ID = "412"       # Total (incl. OT and penalties), whole and half lines
TOTAL_REG_MARKET_ID = "18"       # Over/Under in regulation time
HOME_TOTAL_MARKET_ID = "19"      # Home team over/under (regulation time)
AWAY_TOTAL_MARKET_ID = "20"      # Away team over/under (regulation time)
PERIOD_TOTAL_MARKET_ID = "446"   # Xth period total, specifier total=<line>|periodnr=<n>
MARKET_KINDS = {WINNER_MARKET_ID: "winner", HANDICAP_OT_MARKET_ID: "handicap_ot", HANDICAP_REG_MARKET_ID: "handicap_reg",
                TOTAL_OT_MARKET_ID: "total_ot", TOTAL_REG_MARKET_ID: "total_reg", HOME_TOTAL_MARKET_ID: "home_total",
                AWAY_TOTAL_MARKET_ID: "away_total", PERIOD_TOTAL_MARKET_ID: "p1_total"}
LIST_MARKET_IDS = ",".join(MARKET_KINDS)  # what the fixture listing asks SportyBet for

WINNER_MARKET_TEXT = "Winner (incl. OT and pen.)"
INSUFFICIENT_DATA = "insufficient data"
SIDES = ("home", "away")


class MissingInput(Exception):
    pass


def market_kind(offer: MarketOffer) -> str | None:
    kind = MARKET_KINDS.get(offer.market_id)
    if kind == "p1_total" and parse_specifier(offer.specifier).get("periodnr") != P1_PERIOD:
        return None  # only the first period is priced
    return kind


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

        self.xgf = {s: need(f"{s}.xg_for") for s in SIDES}
        self.rate_for = {s: XG_WEIGHT * need(f"{s}.xg_for") + GOALS_WEIGHT * need(f"{s}.goals_for") for s in SIDES}
        self.rate_against = {s: XG_WEIGHT * need(f"{s}.xg_against") + GOALS_WEIGHT * need(f"{s}.goals_against")
                             for s in SIDES}
        league = self.by.get("match.league_goals")
        self.league = league.value if league is not None else (sum(self.rate_for.values())
                                                               + sum(self.rate_against.values())) / 4
        self.mu = {s: self._goals(s) for s in SIDES}
        p1 = [f"{s}.p1_{k}" for s in SIDES for k in ("goals_for", "goals_against", "xg_for", "xg_against")]
        self.has_p1 = all(n in self.by for n in p1)
        if self.has_p1:
            p1_for = {s: XG_WEIGHT * self.by[f"{s}.p1_xg_for"].value + GOALS_WEIGHT * self.by[f"{s}.p1_goals_for"].value
                      for s in SIDES}
            p1_against = {s: XG_WEIGHT * self.by[f"{s}.p1_xg_against"].value
                          + GOALS_WEIGHT * self.by[f"{s}.p1_goals_against"].value for s in SIDES}
            league1 = self.by.get("match.league_p1_goals")
            self.league1 = league1.value if league1 is not None else (sum(p1_for.values()) + sum(p1_against.values())) / 4
            self.mu1 = {}
            for s in SIDES:
                other = "away" if s == "home" else "home"
                self.mu1[s] = _clamp(p1_for[s] * p1_against[other] / self.league1 * self._home(s),
                                     MIN_P1_GOALS, MAX_P1_GOALS)
            # The routing figure is the pure proxy (no goals blended in): the first-period xG of the matchup.
            self.p1_combined_xg = sum((self.by[f"{s}.p1_xg_for"].value
                                       + self.by[f"{'away' if s == 'home' else 'home'}.p1_xg_against"].value) / 2
                                      for s in SIDES)
        self._grid = None

    @staticmethod
    def _home(side: str) -> float:
        return HOME_GOALS_FACTOR if side == "home" else 2.0 - HOME_GOALS_FACTOR

    def _goals(self, side: str) -> float:
        other = "away" if side == "home" else "home"
        base = self.rate_for[side] * self.rate_against[other] / self.league
        return _clamp(base * self._home(side), MIN_TEAM_GOALS, MAX_TEAM_GOALS)

    def grid(self):
        if self._grid is None:
            self._grid = joint(poisson_pmf(self.mu["home"]), poisson_pmf(self.mu["away"]))
        return self._grid

    @property
    def ot_home_win(self) -> float:
        share = self.mu["home"] / (self.mu["home"] + self.mu["away"])
        return 0.5 + OT_LEAN * (share - 0.5)

    def regulation(self):
        return margin_and_total(self.grid())

    def with_overtime(self):
        return margin_and_total(self.grid(), self.ot_home_win, OT_MIN_EXTRA_GOALS)

    def team_dist(self, side: str) -> dict[int, float]:
        return dict(enumerate(poisson_pmf(self.mu[side])))

    def p1_dist(self) -> dict[int, float]:
        pmf = joint(poisson_pmf(self.mu1["home"]), poisson_pmf(self.mu1["away"]))
        return margin_and_total(pmf)[1]

    def p_home_wins(self) -> float:
        margin, _ = self.with_overtime()
        return sum(p for k, p in margin.items() if k > 0)

    def summary(self, *names: str, first_period: bool = False) -> str:
        shown = [self.by[n] for n in names if n in self.by]
        parts = ", ".join(f"{i.name} {i.value:g} {i.unit} [{i.source}]" for i in shown)
        if first_period:
            expected = (f"1st-period expected goals {self.mu1['home']:.2f} - {self.mu1['away']:.2f} "
                        f"(combined xG {self.p1_combined_xg:.2f})")
        else:
            expected = (f"expected goals {self.mu['home']:.2f} - {self.mu['away']:.2f} "
                        f"(total {sum(self.mu.values()):.2f})")
        return f"inputs: {parts}; {expected}, {MODEL_NAME}"


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

    def market(self, kind: str) -> list[Priced]:
        offers = [o for o in self.offers if market_kind(o) == kind]
        m = self.model
        core = ("home.xg_for", "home.goals_for", "away.xg_for", "away.goals_for", "home.xg_against",
                "away.xg_against", "home.goals_against", "away.goals_against")
        if kind == "winner":
            return self._winner(offers, core)
        if kind == "handicap_ot":
            return self._handicap(offers, kind, "Puck line (incl. OT)", m.with_overtime()[0], core)
        if kind == "handicap_reg":
            return self._handicap(offers, kind, "Handicap (regulation)", m.regulation()[0], core)
        if kind == "total_ot":
            return self._totals(offers, kind, "Total goals (incl. OT)", "", m.with_overtime()[1], m.summary(*core))
        if kind == "total_reg":
            return self._totals(offers, kind, "Total goals (regulation)", "", m.regulation()[1], m.summary(*core))
        if kind in ("home_total", "away_total"):
            side = kind.split("_")[0]
            return self._totals(offers, kind, f"{side.title()} team total goals", f"{side.title()} team ",
                                m.team_dist(side), m.summary(*core))
        if kind == "p1_total":
            if not m.has_p1:
                return self._no_p1(offers)
            names = tuple(f"{s}.p1_{k}" for s in SIDES for k in ("xg_for", "xg_against", "goals_for", "goals_against"))
            return self._totals(offers, kind, "1st period total goals", "1st period ", m.p1_dist(),
                                m.summary(*names, first_period=True), extras=("total", "periodnr"))
        return []

    def _no_p1(self, offers: list[MarketOffer]) -> list[Priced]:
        results = []
        for offer in offers:
            line = spec_decimal(offer.specifier, "total")
            if line is None:
                continue
            for side in SIDES_PLAYED:
                outcome = find_outcome(offer, side)
                if outcome is not None:
                    results.append(Priced("p1_total", side, line, self._insufficient(
                        f"1st period total goals {fmt_line(line)}", f"1st period {side.title()} {fmt_line(line)}",
                        outcome.odds, "no first-period scoring data for one or both teams")))
        return results

    def _totals(self, offers, kind, label, prefix, dist, note, extras=("total",)) -> list[Priced]:
        results: list[Priced] = []
        for offer in offers:
            if not spec_is_only(offer.specifier, *extras):
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
                    (note, push_note(estimate, f"exactly {fmt_line(line)} goals")))))
        return results

    def _winner(self, offers, core) -> list[Priced]:
        p_home = self.model.p_home_wins()
        note = self.model.summary(*core) + f"; overtime winner: home {self.model.ot_home_win:.1%}"
        results = []
        for offer in offers:
            for side, p in (("home", p_home), ("away", 1.0 - p_home)):
                outcome = find_outcome(offer, side)
                if outcome is not None:
                    results.append(Priced("winner", side, None, self._evaluation(
                        WINNER_MARKET_TEXT, side.title(), outcome.odds,
                        Estimate(min(p, 1.0), self.model.used, MODEL_NAME), (note,))))
        return results

    def _handicap(self, offers, kind, label, margin, core) -> list[Priced]:
        note = self.model.summary(*core)
        results = []
        for offer in offers:
            if not spec_is_only(offer.specifier, "hcp"):
                continue
            hcp = spec_decimal(offer.specifier, "hcp")
            if hcp is None or abs(hcp) > MAX_SPREAD:
                continue
            probs = handicap_probabilities(margin, hcp)
            if probs is None:
                continue  # quarter lines are not priced
            if hcp % 1 == 0 and find_outcome(offer, "draw") is not None:
                continue  # a 3-way handicap on a whole line has a Draw outcome and NO refund: not a push-protected line
            home, away, push = probs
            for side, shown in (("home", hcp), ("away", -hcp)):
                outcome = find_outcome(offer, side)
                if outcome is None:
                    continue
                estimate = self._estimate(home, away, push, side)
                results.append(Priced(kind, side, hcp, self._evaluation(
                    f"{label} {fmt_signed(hcp)}", f"{side.title()} ({fmt_signed(shown)})", outcome.odds, estimate,
                    (note, push_note(estimate, f"a one-goal margin" if abs(hcp) == 1 else
                                     f"a margin of exactly {fmt_line(abs(hcp))}")))))
        return results


def shown_line(p: Priced) -> Decimal | None:
    """The handicap as the pick's own side sees it: the home handicap for a home pick, its negation for an away pick."""
    if p.line is None:
        return None
    return p.line if p.side == "home" else -p.line


# ---------------------------------------------------------------- the engine
class NhlEngine:
    sport = SPORT

    def screen(self, fixture: Fixture, inputs: Sequence[Input]) -> FixturePass | None:
        return None  # no derby or weather rule for hockey in the specification

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

        locked = [s for s in SIDES if model.xgf[s] >= XGF_LOCK_THRESHOLD and self._lock_target(pricer.market(f"{s}_total"))]
        routed = model.has_p1 and model.p1_combined_xg > P1_COMBINED_XG_THRESHOLD

        # -- full-game totals: bypassed when either rule produces the output instead
        full = pricer.market("total_ot") + pricer.market("total_reg")
        top = best(full)
        original_text = describe(top.evaluation) if top is not None else ""
        if not (locked or routed):
            results.extend(shop(full))

        # -- team totals: a locked team gets Team Total Over 3.0 only; the other side is priced as usual
        for side in SIDES:
            candidates = pricer.market(f"{side}_total")
            if side in locked:
                results.extend(self._lock(fixture, model, side, candidates, original_text))
            else:
                results.extend(shop(candidates))

        # -- first period: routed to Over 1.5 above the threshold, otherwise both sides are priced
        first = pricer.market("p1_total")
        if routed:
            rule = (f"{RULE_P1}: combined first-period expected goals {model.p1_combined_xg:.2f} are above "
                    f"{P1_COMBINED_XG_THRESHOLD:.2f}, so the pick routes to Period 1 Over {fmt_line(P1_ROUTE_LINE)} goals")
            target = [p for p in first if p.side == "over" and p.line == P1_ROUTE_LINE]
            results.extend(converted(shop(target), rule, original_text))
        else:
            results.extend(shop(first))

        results.extend(shop(pricer.market("winner")))
        results.extend(self._handicaps(fixture, model, pricer))
        return results

    # ---- Story 6.5: xGF lock (AD-21: the nearest LISTED HALF line at or above 3, never a whole line)
    @staticmethod
    def _lock_target(candidates: Sequence[Priced]) -> Priced | None:
        eligible = [p for p in candidates if p.side == "over" and p.line is not None and p.line >= TEAM_TOTAL_LOCK_LINE
                    and not p.evaluation.insufficient]
        return min(eligible, key=lambda p: p.line) if eligible else None

    def _lock(self, fixture: Fixture, model: _Model, side: str, candidates: Sequence[Priced],
              original_text: str) -> list[Evaluation]:
        team = fixture.home if side == "home" else fixture.away
        target = self._lock_target(candidates)
        if target is None:
            return shop(candidates)   # no listed half line at or above 3: the lock is dropped
        rule = (f"{RULE_LOCK}: {team} rolling xGF {model.xgf[side]:.2f} is at least {XGF_LOCK_THRESHOLD:.2f}, so Team "
                f"Total Over {fmt_line(target.line)} (the nearest listed half line at or above {fmt_line(TEAM_TOTAL_LOCK_LINE)}) "
                f"is the primary output and full-game totals are bypassed")
        return converted(shop([target]), rule, original_text)

    # ---- Story 6.5: puck-line shield
    def _handicaps(self, fixture: Fixture, model: _Model, pricer: _Pricer) -> list[Evaluation]:
        priced = pricer.market("handicap_ot") + pricer.market("handicap_reg")
        p_home = model.p_home_wins()
        fav, p_fav = ("home", p_home) if p_home >= 0.5 else ("away", 1.0 - p_home)
        originals = [p for p in priced if p.side == fav and shown_line(p) == PUCK_LINE]
        if p_fav < HEAVY_FAVOURITE_MIN_WIN or not originals:
            return shop(priced)  # not a heavy favourite (or no -1.5 listed): plain line shopping
        opp = "away" if fav == "home" else "home"
        team = fixture.home if fav == "home" else fixture.away
        original = best(originals)
        dog = [p for p in priced if p.side == opp and shown_line(p) == UNDERDOG_LINE]
        rule = (f"{RULE_PUCK}: {team} is a heavy favourite (win {percent(p_fav)}, at least "
                f"{HEAVY_FAVOURITE_MIN_WIN:.0%}), so the {fmt_signed(PUCK_LINE)} puck line converts to the "
                f"underdog {fmt_signed(UNDERDOG_LINE)}")
        out = [e for e in shop([p for p in priced if p not in originals and p not in dog])]
        choice = best(dog)
        if choice is not None:
            picked = with_(choice.evaluation, rule=rule, original=describe(original.evaluation))
            out = [e for e in out if (e.market, e.selection) != (picked.market, picked.selection)]
            out.append(picked)
        else:
            out.append(with_(original.evaluation, passed=False, rule=rule, original=describe(original.evaluation),
                             reason=(f"the shield is not offered: SportyBet lists no underdog {fmt_signed(UNDERDOG_LINE)} for this game, "
                                     f"and the plain {fmt_signed(PUCK_LINE)} is not played")))
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
        return [p.evaluation for p in _Pricer(fixture, model, [offer]).market(kind)]
