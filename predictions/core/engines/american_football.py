"""American football engine (Stories 7.1 to 7.4, sport "american_football": NFL and college). Standard library only.
Everything here is deterministic; every threshold is a named constant (AD-4). The floor is imported, never defined here.

Model (`normal-points-kn-v1`). Each side's points are the "log5" style ratio of its own scoring and the opponent's
points allowed against the league level, after SHRINK_GAMES games' worth of shrinkage towards the league (a football
team has only a handful of games in the sample), plus a home edge:
    points(A) = rate for(A) * rate against(B) / league points per team        (home +HFA/2, away -HFA/2)
Points are integers, so each market is a normal distribution with a continuity correction (like basketball). The one
place football differs is the MARGIN: a game is often decided by exactly 3 or 7. The margin distribution is the normal
one RE-WEIGHTED at the key numbers with a documented frequency table (KEY_NUMBERS below: the owner's approximate
frequencies for NFL 3 and 7 and the owner's ranking for the others, spec-7-2), so a line on a key number is priced
differently from the half point next to it. Team totals, the game total and the first-half totals are plain normals.
The model is uncalibrated (calibration is Epic 8).

Optimization, each rule only ever REPLACES a validated pick and the converted pick stays only if it clears the floor
with positive EV (otherwise it is shown as rejected with its reason); the original and the rule are always shown:
  * Story 7.2 key-number shield (handicaps). The ORIGINAL pick of a side is its highest-EV listed line that passes the
    gate (else the closest to passing). NFL favourite at -3.5 or -7.5 with an estimate of at least the floor steps
    down to -2.5 / -6.5 (the "buy the half point" branch to the whole line -3 / -7 is gone: no whole-line play, AD-21).
    NFL underdog at +2.5 / +6.5 moves to +3.5 / +7.5. College favourite from -7.5 to -9.5 steps down to -6.5, or
    converts to the Moneyline when the model's win probability is at least 72%. Only lines SportyBet really lists
    are used.
  * Story 7.3 first-half. A team in the top 10 for first-quarter offensive efficiency (PROXY: first-quarter points per
    game, ranked over the league sample) routes the pick to a 1st-half Team Total Over (only if SportyBet lists one:
    it does not today) or to the 1st-half Game Total Over (full-game Overs are not played). A 1st-half Over at 24.5
    with an estimate from the floor to 72% steps down to 23.5 or a single-team 1st-half total.
  * Story 7.4 wind and pace. Every Over-type total (WIND_GUARDED_KINDS) needs a wind figure: none means "insufficient
    data" (never assumed calm); projected sustained wind of at least 15 mph disqualifies a full-game Over (and the other
    Overs) whatever the model says and is not replaced by anything (the owner allows no Under plays in any sport, AD-20: the old shift to a Team
    Total Under is gone, the Over is simply not played). A team slower than 28.5 seconds per play that also rushes on more
    than 52% of its plays sets a clock-control baseline: full-game Overs and its own team-total Over are not played, and
    nothing is played in their place.
Only half lines are offered (AD-21, spec-13-1). Review (`evaluate_exact`) still prices a whole line on a booking code, push-protected
as in soccer (spec-2-4): success = P(win) + P(push), EV = P(win)*(odds-1) - P(lose).
"""
from __future__ import annotations

import math
import re
from decimal import Decimal
from typing import Mapping, Sequence

from predictions.core.engines.common import (Priced, best, converted, describe, find_outcome, fmt_line, fmt_signed,
                                             handicap_probabilities, line_probabilities, push_note,
                                             shop, spec_decimal, spec_is_only, with_)
from predictions.core.inputs import Input
from predictions.core.market_rules import SIDES_PLAYED, offers_without_whole_lines, without_unders, without_whole_lines
from predictions.core.model import PROBABILITY_FLOOR, Estimate, Evaluation, FixturePass, percent, validate
from predictions.core.types import Fixture, MarketOffer, normalize_team

SPORT = "american_football"
NFL, COLLEGE = "nfl", "college"


def league_of(competition: str) -> str | None:
    """"nfl" for SportyBet's "USA - NFL", "college" for "USA - NCAA, ...", None for anything else (the CFL, ...): no
    rules and no stats for it."""
    name = normalize_team(competition)
    if name == "usa nfl":
        return NFL
    if name.startswith("usa ncaa"):
        return COLLEGE
    return None


# ---- model ----
MODEL_NAME = "normal-points-kn-v1"
SHRINK_GAMES = {NFL: 4.0, COLLEGE: 4.0}   # a team's rate is (n * own + K * league) / (n + K): football samples are tiny (judgement).
                                          # College was 6 for a few hours on 2026-10-10 without evidence; the backtest found 4 at least as
                                          # good for margins and winners and within 0.1 point for totals, so it is back to 4.
HOME_FIELD_POINTS = {NFL: 2.0, COLLEGE: 2.5}        # margin edge in points, split half and half (judgement)
FH_HOME_SHARE = 0.5                # the home edge in the first half is this share of the full-game edge (judgement)
# College standard deviations are MEASURED (2026-10-10 walk-forward backtest, tools/football_backtest.py: 1,149 games of 2025 and 2026,
# weeks 5-16, predicted from the earlier weeks only): actual margin sd 17.7, total 15.5, one side 11.8, 1st-half total 11.1. The NFL values
# are still judgement.
MARGIN_SD = {NFL: 13.5, COLLEGE: 17.7}              # sd of the final margin
TOTAL_SD = {NFL: 10.5, COLLEGE: 15.5}               # sd of the game total
TEAM_SD = {NFL: 7.5, COLLEGE: 11.8}                 # sd of one side's points
FH_TOTAL_SD = {NFL: 7.0, COLLEGE: 11.1}             # sd of the first-half total
FH_TEAM_SD = {NFL: 5.2, COLLEGE: 6.6}               # sd of one side's first-half points
MIN_TEAM_POINTS, MAX_TEAM_POINTS = 3.0, 55.0
MIN_FH_POINTS, MAX_FH_POINTS = 1.0, 35.0
MAX_MARGIN = 80                    # the margin distribution runs from -MAX_MARGIN to MAX_MARGIN
MAX_POINTS = 200                   # totals run from 0 to this
MIN_LINE, MAX_LINE = Decimal("5"), Decimal("120")
MAX_SPREAD = Decimal("40")

# ---- limits on what the model may claim (2026-10-10 review of the first settled week: 54 legs, 72.8% predicted, 53.7% won;
#      picks with an expected value above +50% won 1 of 8, and overs 2 of 10). College only: the NFL data was not separable.
MAX_EV = {NFL: None, COLLEGE: 0.20}      # a pick claiming more edge than this is not passed: a bigger claimed edge over the
                                          # bookmaker is more likely a model error (same idea as the soccer goals-only cap)
COLLEGE_OVERS_PAUSED = True               # no college Over (game, team or 1st-half total) is played until the rules are re-reviewed
MIN_GAMES = {NFL: 3, COLLEGE: 4}          # completed regular-season games a team needs before it is priced (read by the ESPN source)
RULE_PAUSED = "College Overs paused"
PAUSED_REASON = ("disqualified: college football Overs are paused (the first settled week: 2 wins from 10), so none is played "
                 "whatever the model says")

# The key-number table: the share of all games decided by exactly this many points (either team). The owner's figures
# are NFL 3 = about 14.5% and 7 = about 9.8%; for the others the owner gave only the ORDER (NFL: 3, 7, then 6, 10, 14;
# college: 7, 3, 10, 14, 4), so those frequencies are judgement values that keep that order (spec-7-2 Design Notes).
NFL_KEY_NUMBERS: Mapping[int, float] = {3: 0.145, 7: 0.098, 6: 0.062, 10: 0.052, 14: 0.043}
COLLEGE_KEY_NUMBERS: Mapping[int, float] = {7: 0.072, 3: 0.065, 10: 0.058, 14: 0.052, 4: 0.049}
KEY_NUMBERS = {NFL: NFL_KEY_NUMBERS, COLLEGE: COLLEGE_KEY_NUMBERS}

# ---- Story 7.2: key-number shield ----
NFL_FAVORITE_STEPS: Mapping[Decimal, tuple[Decimal, ...]] = {   # favourite line -> the stepped-down half line
    Decimal("-3.5"): (Decimal("-2.5"),),
    Decimal("-7.5"): (Decimal("-6.5"),),
}
NFL_UNDERDOG_STEPS: Mapping[Decimal, tuple[Decimal, ...]] = {   # underdog line -> the line on the safe side of the key number
    Decimal("2.5"): (Decimal("3.5"),),
    Decimal("6.5"): (Decimal("7.5"),),
}
COLLEGE_SHIELD_FROM = Decimal("-9.5")     # a college favourite from -9.5 ...
COLLEGE_SHIELD_TO = Decimal("-7.5")       # ... to -7.5 (half lines only) is shielded
COLLEGE_STEP_LINE = Decimal("-6.5")
COLLEGE_MONEYLINE_MIN_WIN = 0.72          # win probability at or above this converts to the Moneyline
RULE_KEY = "Key-number shield"

# ---- Story 7.3: first-half ----
TOP_Q1_RANK = 10                          # rank 10 or better (1 = best) routes to the first half
FH_STEP_LINE = Decimal("24.5")            # a 1st-half Over at this line ...
FH_STEP_MAX = 0.72                        # ... with an estimate from the floor up to this (inclusive) steps down
FH_STEP_TO = Decimal("23.5")
RULE_Q1 = "1H route"
RULE_FH_STEP = "1H step-down"

# ---- Story 7.4: wind and pace ----
WIND_LIMIT_MPH = 15.0                     # projected sustained wind AT LEAST this disqualifies an Over (14.9 does not)
PACE_SLOW_SECONDS = 28.5                  # seconds per play STRICTLY above this is slow (the bottom 20%)
RUSH_RATE_MIN_PERCENT = 52.0              # rush share of plays STRICTLY above this
WIND_GUARDED_KINDS = frozenset({"total", "home_total", "away_total", "fh_total", "fh_home_total", "fh_away_total"})
PACE_GUARDED_KINDS = frozenset({"total", "home_total", "away_total"})
RULE_WIND = "Wind filter"
RULE_PACE = "Clock-control baseline"

# ---- SportyBet market ids (probed on the public listing 2026-09-30, sr:sport:16; same ids as basketball) ----
WINNER_MARKET_ID = "219"         # Winner (incl. overtime): Home / Away
HANDICAP_MARKET_ID = "223"       # Handicap (incl. overtime), hcp=<home handicap>
TOTAL_MARKET_ID = "225"          # Over/Under (incl. overtime), total=<line>
HOME_TOTAL_MARKET_ID = "227"     # Home Over/Under (incl. overtime), listed for a few games only
AWAY_TOTAL_MARKET_ID = "228"
FH_TOTAL_MARKET_ID = "68"        # 1st half - total, total=<line>
MARKET_KINDS = {WINNER_MARKET_ID: "winner", HANDICAP_MARKET_ID: "handicap", TOTAL_MARKET_ID: "total",
                HOME_TOTAL_MARKET_ID: "home_total", AWAY_TOTAL_MARKET_ID: "away_total", FH_TOTAL_MARKET_ID: "fh_total"}
LIST_MARKET_IDS = ",".join(MARKET_KINDS)  # what the fixture listing asks SportyBet for
WINNER_MARKET_TEXT = "Winner (incl. OT)"
FH_TEAM_TOTAL_WORDS = ("1st half", "first half")
INSUFFICIENT_DATA = "insufficient data"
SIDES = ("home", "away")


class MissingInput(Exception):
    pass


def market_kind(offer: MarketOffer) -> str | None:
    """The kind of market, by SportyBet id; a 1st-half TEAM total (no id known, none listed) by its name."""
    kind = MARKET_KINDS.get(offer.market_id)
    if kind is not None:
        return kind
    name = normalize_team(offer.name)
    if "total" in name and any(w in name for w in FH_TEAM_TOTAL_WORDS) and re.search(
            r"\b(competitor ?[12]|home|away)\b", name):
        return "fh_team_total"
    return None


def fh_team_side(offer: MarketOffer) -> str | None:
    name = normalize_team(offer.name)
    if re.search(r"\bcompetitor ?1\b|\bhome\b", name):
        return "home"
    if re.search(r"\bcompetitor ?2\b|\baway\b", name):
        return "away"
    return None


# ---------------------------------------------------------------- discrete distributions
def _cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def normal_pmf(mu: float, sd: float, low: int, high: int) -> dict[int, float]:
    """P(X = k) for integer k in [low, high]: a normal with a continuity correction, normalised over that range."""
    pmf = {k: _cdf((k + 0.5 - mu) / sd) - _cdf((k - 0.5 - mu) / sd) for k in range(low, high + 1)}
    total = sum(pmf.values())
    return {k: p / total for k, p in pmf.items()}


def key_weights(sd: float, table: Mapping[int, float]) -> dict[int, float]:
    """weight(k) = table frequency of a margin of exactly k / the frequency a plain zero-mean normal gives it."""
    zero = normal_pmf(0.0, sd, -MAX_MARGIN, MAX_MARGIN)
    return {k: table[k] / (zero[k] + zero[-k]) for k in table}


def margin_pmf(mu: float, sd: float, table: Mapping[int, float]) -> dict[int, float]:
    """The home-minus-away margin: the normal, re-weighted at the key numbers, renormalised."""
    weights = key_weights(sd, table)
    base = normal_pmf(mu, sd, -MAX_MARGIN, MAX_MARGIN)
    weighted = {m: p * weights.get(abs(m), 1.0) for m, p in base.items()}
    total = sum(weighted.values())
    return {m: p / total for m, p in weighted.items()}


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


# ---------------------------------------------------------------- the model of one fixture
class _Model:
    def __init__(self, inputs: Sequence[Input], league: str):
        self.league = league
        self.used = tuple(sorted(inputs, key=lambda i: i.name))
        self.by = {i.name: i for i in self.used}

        def need(name: str) -> float:
            if name not in self.by:
                raise MissingInput(name)
            return self.by[name].value

        self.n = {s: need(f"{s}.matches") for s in SIDES}
        pf = {s: need(f"{s}.points_for") for s in SIDES}
        pa = {s: need(f"{s}.points_against") for s in SIDES}
        league_ppg = self.by.get("match.league_ppg")
        self.league_ppg = league_ppg.value if league_ppg is not None else (sum(pf.values()) + sum(pa.values())) / 4
        self.rate_for = {s: self._shrink(pf[s], self.n[s], self.league_ppg) for s in SIDES}
        self.rate_against = {s: self._shrink(pa[s], self.n[s], self.league_ppg) for s in SIDES}
        edge = HOME_FIELD_POINTS[league] / 2.0
        self.mu = {}
        for s in SIDES:
            other = "away" if s == "home" else "home"
            base = self.rate_for[s] * self.rate_against[other] / self.league_ppg
            self.mu[s] = _clamp(base + (edge if s == "home" else -edge), MIN_TEAM_POINTS, MAX_TEAM_POINTS)
        # opponent-adjusted rates (college): used for the MARGIN only (handicaps and the winner); totals stay on the raw rates
        adj = [f"{s}.adj_points_{k}" for s in SIDES for k in ("for", "against")]
        self.adjusted = all(n in self.by for n in adj)
        if self.adjusted:
            adj_for = {s: self._shrink(self.by[f"{s}.adj_points_for"].value, self.n[s], self.league_ppg) for s in SIDES}
            adj_against = {s: self._shrink(self.by[f"{s}.adj_points_against"].value, self.n[s], self.league_ppg) for s in SIDES}
            self.mu_margin = {}
            for s in SIDES:
                other = "away" if s == "home" else "home"
                base = adj_for[s] * adj_against[other] / self.league_ppg
                self.mu_margin[s] = _clamp(base + (edge if s == "home" else -edge), MIN_TEAM_POINTS, MAX_TEAM_POINTS)
        else:
            self.mu_margin = self.mu
        fh = [f"{s}.fh_points_{k}" for s in SIDES for k in ("for", "against")]
        self.has_fh = all(n in self.by for n in fh)
        if self.has_fh:
            league_fh = self.by.get("match.league_fh_ppg")
            self.league_fh = league_fh.value if league_fh is not None else (
                sum(self.by[f"{s}.fh_points_for"].value + self.by[f"{s}.fh_points_against"].value for s in SIDES) / 4)
            fh_for = {s: self._shrink(self.by[f"{s}.fh_points_for"].value, self.n[s], self.league_fh) for s in SIDES}
            fh_against = {s: self._shrink(self.by[f"{s}.fh_points_against"].value, self.n[s], self.league_fh)
                          for s in SIDES}
            fh_edge = edge * FH_HOME_SHARE
            self.mu1 = {}
            for s in SIDES:
                other = "away" if s == "home" else "home"
                base = fh_for[s] * fh_against[other] / self.league_fh
                self.mu1[s] = _clamp(base + (fh_edge if s == "home" else -fh_edge), MIN_FH_POINTS, MAX_FH_POINTS)
        self.wind = self.by.get("match.wind_mph")
        self._margin = None

    def _shrink(self, own: float, games: float, league: float) -> float:
        k = SHRINK_GAMES[self.league]
        return (games * own + k * league) / (games + k)

    # ---- distributions
    def margin(self) -> dict[int, float]:
        if self._margin is None:
            self._margin = margin_pmf(self.mu_margin["home"] - self.mu_margin["away"], MARGIN_SD[self.league], KEY_NUMBERS[self.league])
        return self._margin

    def p_home_wins(self) -> float:
        """A tie (margin 0) goes to overtime, which is a coin flip."""
        margin = self.margin()
        return sum(p for k, p in margin.items() if k > 0) + 0.5 * margin.get(0, 0.0)

    def total_dist(self) -> dict[int, float]:
        return normal_pmf(sum(self.mu.values()), TOTAL_SD[self.league], 0, MAX_POINTS)

    def team_dist(self, side: str) -> dict[int, float]:
        return normal_pmf(self.mu[side], TEAM_SD[self.league], 0, MAX_POINTS)

    def fh_dist(self) -> dict[int, float]:
        return normal_pmf(sum(self.mu1.values()), FH_TOTAL_SD[self.league], 0, MAX_POINTS)

    def fh_team_dist(self, side: str) -> dict[int, float]:
        return normal_pmf(self.mu1[side], FH_TEAM_SD[self.league], 0, MAX_POINTS)

    # ---- the rule inputs
    def q1_top(self, side: str) -> bool:
        rank = self.by.get(f"{side}.q1_rank")
        return rank is not None and rank.value <= TOP_Q1_RANK

    def slow_and_rushing(self, side: str) -> bool:
        pace, rush = self.by.get(f"{side}.pace"), self.by.get(f"{side}.rush_rate")
        return (pace is not None and rush is not None and pace.value > PACE_SLOW_SECONDS
                and rush.value > RUSH_RATE_MIN_PERCENT)

    def summary(self, *names: str, first_half: bool = False) -> str:
        shown = [self.by[n] for n in names if n in self.by]
        parts = ", ".join(f"{i.name} {i.value:g} {i.unit} [{i.source}]" for i in shown)
        if first_half:
            expected = f"1st-half expected points {self.mu1['home']:.1f} - {self.mu1['away']:.1f}"
        else:
            expected = (f"expected points {self.mu['home']:.1f} - {self.mu['away']:.1f} "
                        f"(total {sum(self.mu.values()):.1f})")
            if self.adjusted:
                expected += (f"; margin and winner from opponent-adjusted rates (expected margin "
                             f"{self.mu_margin['home'] - self.mu_margin['away']:+.1f})")
        return f"inputs: {parts}; {expected}, {MODEL_NAME}"


CORE_INPUTS = ("home.points_for", "home.points_against", "away.points_for", "away.points_against")
FH_INPUTS = ("home.fh_points_for", "home.fh_points_against", "away.fh_points_for", "away.fh_points_against")
WIND_INPUTS = ("match.wind_mph",)


# ---------------------------------------------------------------- pricing the offers
class _Pricer:
    def __init__(self, fixture: Fixture, model: _Model, offers: Sequence[MarketOffer]):
        self.fixture, self.model, self.offers = fixture, model, list(offers)

    def _evaluation(self, market: str, selection: str, odds: Decimal, estimate: Estimate,
                    notes: tuple[str, ...]) -> Evaluation:
        passed, ev, reason = validate(estimate, odds)
        cap = MAX_EV.get(self.model.league)
        if passed and cap is not None and ev > Decimal(str(cap)):
            passed = False
            reason = (f"expected value {percent(float(ev))} is above the {percent(cap)} cap for college football: a claimed edge "
                      f"that large over the bookmaker is more likely a model error than a bargain")
        return Evaluation(self.fixture.match_id, market, selection, odds, estimate, ev, passed, reason,
                          "", "", tuple(n for n in notes if n))

    def _estimate(self, over: float, under: float, push: float, side: str) -> Estimate:
        win = over if side in ("over", "home") else under
        if push:
            return Estimate.push_protected(win, push, self.model.used, MODEL_NAME)
        return Estimate(min(win, 1.0), self.model.used, MODEL_NAME)

    def market(self, kind: str) -> list[Priced]:
        m = self.model
        note = m.summary(*CORE_INPUTS, *WIND_INPUTS)
        if kind == "winner":
            return self._winner([o for o in self.offers if market_kind(o) == "winner"], note)
        if kind == "handicap":
            return self._handicap([o for o in self.offers if market_kind(o) == "handicap"], note)
        if kind == "total":
            return self._totals("total", "Total points", "", m.total_dist(), note,
                                [o for o in self.offers if market_kind(o) == "total"])
        if kind in ("home_total", "away_total"):
            side = kind.split("_")[0]
            return self._totals(kind, f"{side.title()} team total points", f"{side.title()} team ", m.team_dist(side),
                                note, [o for o in self.offers if market_kind(o) == kind])
        fh_note = None
        if kind in ("fh_total", "fh_home_total", "fh_away_total"):
            offers = ([o for o in self.offers if market_kind(o) == "fh_total"] if kind == "fh_total"
                      else [o for o in self.offers if market_kind(o) == "fh_team_total"
                            and fh_team_side(o) == kind.split("_")[1]])
            if not offers:
                return []
            if not m.has_fh:
                return self._no_fh(kind, offers)
            fh_note = m.summary(*FH_INPUTS, *WIND_INPUTS, first_half=True)
            if kind == "fh_total":
                return self._totals(kind, "1st half total points", "1st half ", m.fh_dist(), fh_note, offers)
            side = kind.split("_")[1]
            return self._totals(kind, f"{side.title()} 1st half team total points", f"{side.title()} 1st half team ",
                                m.fh_team_dist(side), fh_note, offers)
        return []

    def _no_fh(self, kind: str, offers: list[MarketOffer]) -> list[Priced]:
        results = []
        for offer in offers:
            line = spec_decimal(offer.specifier, "total")
            if line is None:
                continue
            for side in SIDES_PLAYED:
                outcome = find_outcome(offer, side)
                if outcome is not None:
                    results.append(Priced(kind, side, line, Evaluation(
                        self.fixture.match_id, f"1st half total points {fmt_line(line)}",
                        f"1st half {side.title()} {fmt_line(line)}", outcome.odds,
                        Estimate(0.0, self.model.used, MODEL_NAME), Decimal(0), False,
                        f"{INSUFFICIENT_DATA}: no first-half scoring data for one or both teams", "", "", (), True)))
        return results

    def _totals(self, kind, label, prefix, dist, note, offers) -> list[Priced]:
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
                    (note, push_note(estimate, f"exactly {fmt_line(line)} points")))))
        return results

    def _winner(self, offers, note) -> list[Priced]:
        p_home = self.model.p_home_wins()
        results = []
        for offer in offers:
            for side, p in (("home", p_home), ("away", 1.0 - p_home)):
                outcome = find_outcome(offer, side)
                if outcome is not None:
                    results.append(Priced("winner", side, None, self._evaluation(
                        WINNER_MARKET_TEXT, side.title(), outcome.odds,
                        Estimate(min(p, 1.0), self.model.used, MODEL_NAME), (note,))))
        return results

    def _handicap(self, offers, note) -> list[Priced]:
        margin = self.model.margin()
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
                continue  # a 3-way handicap has a Draw outcome and NO refund: not a push-protected line
            home, away, push = probs
            for side, shown in (("home", hcp), ("away", -hcp)):
                outcome = find_outcome(offer, side)
                if outcome is None:
                    continue
                estimate = self._estimate(home, away, push, side)
                results.append(Priced("handicap", side, hcp, self._evaluation(
                    f"Handicap {fmt_signed(hcp)}", f"{side.title()} ({fmt_signed(shown)})", outcome.odds, estimate,
                    (note, push_note(estimate, f"a margin of exactly {fmt_line(abs(hcp))}")))))
        return results


def shown_line(p: Priced) -> Decimal | None:
    """The handicap as the pick's own side sees it: the home handicap for a home pick, its negation for an away pick."""
    if p.line is None:
        return None
    return p.line if p.side == "home" else -p.line


def original_pick(candidates: Sequence[Priced]) -> Priced | None:
    """The pick before Optimization: the passing candidate with the highest EV (the value line), else the one closest to
    passing. Story 7.2 and 7.3 rules act on it."""
    live = [p for p in candidates if not p.evaluation.insufficient]
    passing = [p for p in live if p.evaluation.passed]
    if passing:
        return max(passing, key=lambda p: (p.evaluation.ev, p.evaluation.estimate.probability, p.evaluation.odds))
    return best(live)


# ---------------------------------------------------------------- the engine
class AmericanFootballEngine:
    sport = SPORT

    def screen(self, fixture: Fixture, inputs: Sequence[Input]) -> FixturePass | None:
        return None  # wind and pace are rules on Over picks (Story 7.4), not a whole-game pass

    def understands(self, offer: MarketOffer) -> bool:
        return market_kind(offer) is not None

    # ---- Story 7.4: the Over guard
    @staticmethod
    def _block(model: _Model, kind: str, team: str | None) -> tuple[str, str, str] | None:
        """(how, rule, reason) when an Over of this kind may not be played, else None. `team` is the side of a team total.
        Missing wind comes first (never assumed calm), then the wind limit, then the clock-control baseline."""
        if kind not in WIND_GUARDED_KINDS:
            return None
        if COLLEGE_OVERS_PAUSED and model.league == COLLEGE:
            return ("paused", RULE_PAUSED, PAUSED_REASON)
        if model.wind is None:
            return ("insufficient", RULE_WIND,
                    f"{INSUFFICIENT_DATA}: no wind forecast for this game, so an Over is never assumed calm")
        if model.wind.value >= WIND_LIMIT_MPH:
            return ("wind", RULE_WIND,
                    f"disqualified: projected sustained wind {model.wind.value:g} mph is at least {WIND_LIMIT_MPH:g} mph, "
                    f"so an Over is not played whatever the model says")
        if kind in PACE_GUARDED_KINDS:
            slow = [s for s in SIDES if model.slow_and_rushing(s) and (team is None or s == team)]
            if slow:
                names = " and ".join(f"{s} {model.by[f'{s}.pace'].value:.1f} s/play, {model.by[f'{s}.rush_rate'].value:.1f}% rush"
                                     for s in slow)
                return ("pace", RULE_PACE,
                        f"disqualified: slow, run-heavy team ({names}; more than {PACE_SLOW_SECONDS:g} s/play and more "
                        f"than {RULE_RUSH_TEXT}): the clock-control baseline applies, so an Over is not played")
        return None

    @staticmethod
    def _blocked(model: _Model, overs: Sequence[Priced], block: tuple[str, str, str]) -> list[Evaluation]:
        live = [p for p in overs if not p.evaluation.insufficient]
        top = best(live) or (overs[0] if overs else None)
        if top is None:
            return []
        how, rule, reason = block
        if how == "insufficient":
            return [with_(top.evaluation, estimate=Estimate(0.0, model.used, MODEL_NAME), ev=Decimal(0), passed=False,
                          reason=reason, rule=rule, insufficient=True)]
        return [with_(top.evaluation, passed=False, reason=f"{reason} (model estimate {percent(top.evaluation.estimate.probability)})",
                      rule=rule)]

    def evaluate(self, fixture: Fixture, inputs: Sequence[Input],
                 market_options: Sequence[MarketOffer]) -> list[Evaluation]:
        # owner rules: no Under (AD-20), no whole or quarter line (AD-21): half-line offers only are ever priced
        return without_whole_lines(without_unders(self._evaluate(fixture, inputs, offers_without_whole_lines(market_options))))

    def _evaluate(self, fixture: Fixture, inputs: Sequence[Input],
                  market_options: Sequence[MarketOffer]) -> list[Evaluation]:
        league = league_of(fixture.competition)
        if league is None:
            return []
        try:
            model = _Model(inputs, league)
        except MissingInput:
            return []
        pricer = _Pricer(fixture, model, market_options)
        out: list[Evaluation] = []

        def split(priced):
            return [p for p in priced if p.side == "over"]   # no Under is ever kept (AD-20)

        # -- full-game total: guarded Overs only (no Under plays in any sport, AD-20)
        full_over = split(pricer.market("total"))
        block = self._block(model, "total", None)
        routed = [s for s in SIDES if model.q1_top(s)]
        original_text = ""
        top = best(full_over)
        if top is not None:
            original_text = describe(top.evaluation)
        if full_over and block is not None:
            out.extend(self._blocked(model, full_over, block))
        elif full_over and routed and model.has_fh:
            pass  # bypassed: the 1st-half route below produces the output (Story 7.3)
        else:
            out.extend(shop(full_over))

        # -- team totals: Over guarded per side (wind, clock control of that team); a wind block plays nothing instead
        for side in SIDES:
            over = split(pricer.market(f"{side}_total"))
            team_block = self._block(model, f"{side}_total", side)
            if over and team_block is not None:
                out.extend(self._blocked(model, over, team_block))
            else:
                out.extend(shop(over))

        # -- first half: routed (Story 7.3) and stepped down; no Unders
        out.extend(self._first_half(model, pricer, routed, original_text, block, full_over))

        out.extend(shop(pricer.market("winner")))
        out.extend(self._handicaps(fixture, model, pricer))
        return _dedupe(out)

    # ---- Story 7.3
    def _first_half(self, model: _Model, pricer: _Pricer, routed: Sequence[str], original_text: str,
                    full_block, full_over) -> list[Evaluation]:
        out: list[Evaluation] = []
        game_over = [p for p in pricer.market("fh_total") if p.side == "over"]
        team_over = {s: [p for p in pricer.market(f"fh_{s}_total") if p.side == "over"] for s in SIDES}
        # the Over candidates, with the guard applied per kind
        pools: dict[str, list[Priced]] = {"fh_total": game_over}
        for s in SIDES:
            pools[f"fh_{s}_total"] = team_over[s]
        allowed: list[Priced] = []
        for kind, pool in pools.items():
            if not pool:
                continue
            block = self._block(model, kind, None)
            if block is not None:
                out.extend(self._blocked(model, pool, block))
            else:
                allowed.extend(pool)
        route_rule = ""
        if routed and model.has_fh:
            names = " and ".join(f"{s} (first-quarter points rank {model.by[f'{s}.q1_rank'].value:g})" for s in routed)
            route_rule = (f"{RULE_Q1}: {names} in the top {TOP_Q1_RANK} for first-quarter offensive efficiency (proxy: "
                          f"first-quarter points per game), so the pick routes to a 1st-half Over instead of a full-game Over")
            listed_team = [p for p in allowed if p.kind != "fh_total" and p.kind.split("_")[1] in routed]
            candidates = listed_team + [p for p in allowed if p.kind == "fh_total"]
        else:
            candidates = allowed
        if not candidates:
            if route_rule and full_over and not [p for p in full_over if p.evaluation.insufficient] and full_block is None:
                top = best(full_over)
                if top is not None:
                    out.append(with_(top.evaluation, passed=False, rule=route_rule, original=original_text,
                                     reason="routed to a 1st-half Over, but SportyBet lists none for this game or "
                                            "its wind figure blocks it: the full-game Over is not played"))
            return out
        # one pick per market kind: the original (value line), then the 24.5 step-down
        by_kind: dict[str, list[Priced]] = {}
        for p in candidates:
            by_kind.setdefault(p.kind, []).append(p)
        for kind in sorted(by_kind):
            picked = original_pick(by_kind[kind])
            if picked is None:
                out.extend(shop(by_kind[kind]))   # no first-half data: the insufficient Over is shown, not dropped
                continue
            out.extend(self._fh_over(model, picked, by_kind[kind], allowed, route_rule, original_text))
        return out

    def _fh_over(self, model: _Model, picked: Priced, kind_pool: Sequence[Priced], allowed: Sequence[Priced],
                 route_rule: str, original_text: str) -> list[Evaluation]:
        evaluation = picked.evaluation
        original = describe(evaluation)
        rules = [route_rule] if route_rule else []
        if (picked.line == FH_STEP_LINE and PROBABILITY_FLOOR <= evaluation.estimate.probability <= FH_STEP_MAX
                and not evaluation.insufficient):
            step = [p for p in kind_pool if p.line == FH_STEP_TO]
            singles = [p for p in allowed if p.kind != "fh_total" and p.evaluation.estimate.probability > 0]
            choice = best(step + singles)
            rule = (f"{RULE_FH_STEP}: a 1st-half Over {fmt_line(FH_STEP_LINE)} with an estimate of "
                    f"{percent(evaluation.estimate.probability)} (from {PROBABILITY_FLOOR:.0%} to {FH_STEP_MAX:.0%}) steps "
                    f"down to Over {fmt_line(FH_STEP_TO)} or a single-team 1st-half total")
            rules.append(rule)
            if choice is None:
                return [with_(evaluation, passed=False, rule="; ".join(rules), original=original_text or original,
                              reason=(f"the step-down is not offered: SportyBet lists no 1st-half Over "
                                      f"{fmt_line(FH_STEP_TO)} or single-team 1st-half total for this game, and the plain "
                                      f"Over {fmt_line(FH_STEP_LINE)} is not played"))]
            return [with_(choice.evaluation, rule="; ".join(rules), original=original_text or original)]
        if rules:
            return [with_(evaluation, rule="; ".join(rules), original=original_text or original)]
        return [evaluation]

    # ---- Story 7.2
    def _handicaps(self, fixture: Fixture, model: _Model, pricer: _Pricer) -> list[Evaluation]:
        priced = pricer.market("handicap")
        winners = {p.side: p for p in pricer.market("winner")}
        out: list[Evaluation] = []
        for side in SIDES:
            mine = [p for p in priced if p.side == side]
            picked = original_pick(mine)
            if picked is None:
                continue
            shielded = self._shield(fixture, model, side, picked, priced, winners.get(side))
            out.extend(shielded if shielded is not None else [picked.evaluation])
        return out

    def _shield(self, fixture: Fixture, model: _Model, side: str, picked: Priced, priced: Sequence[Priced],
                winner: Priced | None) -> list[Evaluation] | None:
        line = shown_line(picked)
        evaluation = picked.evaluation
        if line is None or evaluation.estimate.probability < PROBABILITY_FLOOR:
            return None  # only a validated pick is optimized
        team = fixture.home if side == "home" else fixture.away
        original = describe(evaluation)
        mine = [p for p in priced if p.side == side]
        if model.league == NFL:
            targets = NFL_FAVORITE_STEPS.get(line) or NFL_UNDERDOG_STEPS.get(line)
            if targets is None:
                return None
            wanted = [p for p in mine if shown_line(p) in targets]
            if line < 0:
                what = (f"{team} is a favourite at {fmt_signed(line)}, a hook off a key number, so the pick steps down to "
                        f"{fmt_signed(targets[0])}")
            else:
                what = (f"{team} is an underdog at {fmt_signed(line)}, just short of a key number, so the pick moves to "
                        f"{fmt_signed(targets[0])}")
            rule = f"{RULE_KEY}: {what} (the margin table puts games on 3 and 7)"
        else:
            if not COLLEGE_SHIELD_FROM <= line <= COLLEGE_SHIELD_TO:
                return None
            wanted = [p for p in mine if p.line is not None and shown_line(p) == COLLEGE_STEP_LINE]
            win = winner.evaluation.estimate.probability if winner is not None else 0.0
            eligible_ml = winner is not None and win >= COLLEGE_MONEYLINE_MIN_WIN
            if eligible_ml:
                wanted = wanted + [winner]
            rule = (f"{RULE_KEY}: college favourite {team} at {fmt_signed(line)} (between {fmt_signed(COLLEGE_SHIELD_FROM)} "
                    f"and {fmt_signed(COLLEGE_SHIELD_TO)}) steps down to {fmt_signed(COLLEGE_STEP_LINE)}"
                    + (f" or converts to the Moneyline (win probability {percent(win)} is at least "
                       f"{COLLEGE_MONEYLINE_MIN_WIN:.0%})" if eligible_ml else
                       f" (no Moneyline conversion: win probability {percent(win)} is below {COLLEGE_MONEYLINE_MIN_WIN:.0%})"))
        choice = best(wanted)
        if choice is None:
            return [with_(evaluation, passed=False, rule=rule, original=original,
                          reason=("the shield is not offered: SportyBet lists none of the shielded lines for this game, "
                                  "and the original line is not played"))]
        return [with_(choice.evaluation, rule=rule, original=original)]

    # ---- exact selections for Review: no optimization, the code's own odds; the wind guard still applies to Overs
    def evaluate_exact(self, fixture: Fixture, inputs: Sequence[Input], offer: MarketOffer) -> list[Evaluation]:
        return without_unders(self._evaluate_exact(fixture, inputs, offer))   # owner rule: no Under (AD-20)

    def _evaluate_exact(self, fixture: Fixture, inputs: Sequence[Input], offer: MarketOffer) -> list[Evaluation]:
        league = league_of(fixture.competition)
        if league is None:
            return []
        try:
            model = _Model(inputs, league)
        except MissingInput:
            return []
        kind = market_kind(offer)
        if kind is None:
            return []
        if kind == "fh_team_total":
            side = fh_team_side(offer)
            if side is None:
                return []
            kind = f"fh_{side}_total"
        priced = _Pricer(fixture, model, [offer]).market(kind)
        team = kind.split("_")[1] if kind in ("fh_home_total", "fh_away_total") else kind.split("_")[0]
        team = team if team in SIDES else None
        block = self._block(model, kind, team)
        results: list[Evaluation] = []
        for p in priced:
            if p.side == "over" and block is not None:
                results.extend(self._blocked(model, [p], block))
            else:
                results.append(p.evaluation)
        return results


RULE_RUSH_TEXT = f"{RUSH_RATE_MIN_PERCENT:g}% rushing plays"


def _dedupe(evaluations: Sequence[Evaluation]) -> list[Evaluation]:
    """One evaluation per (market, selection); a converted one (with a rule) beats a plain one."""
    chosen: dict[tuple[str, str], Evaluation] = {}
    order: list[tuple[str, str]] = []
    for e in evaluations:
        key = (e.market, e.selection)
        if key not in chosen:
            order.append(key)
            chosen[key] = e
        elif e.rule and not chosen[key].rule:
            chosen[key] = e
    return [chosen[k] for k in order]
