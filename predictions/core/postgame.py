"""The Monday post-game review's analysis: why legs of lost bets failed, and detailed, evidence-backed recommendations for the review
engine. Pure functions, standard library only (design: docs/weekly-review-design.md).

Principles the team agreed (and the tests pin):
  * The unit of analysis is the LEG, never a bet and never one loss. A recommendation needs enough settled legs (Wilson interval,
    minimum counts, several weeks); a single bet can only ever be a note.
  * A loss is explained by ONE primary reason chosen by priority, and "unexplained" is allowed: nothing is forced.
  * The default explanation for a loss at an estimate of 70% or more is variance: "no change warranted from this bet".
  * Every threshold recommendation is replayed on the pooled history before it is shown (what it would have removed and the flat-stake
    effect), and the owner decides; nothing here changes the engine.
  * The bookmaker's result is the official one: our settlement is a diagnosis and says so.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal

from predictions.core.settle import HALF_LOSS, HALF_WIN, LOSS, PUSH, UNVERIFIABLE, VOID, WIN, FinalScore, Settlement, settle_leg

# ---- reasons (priority order, first = strongest)
R_VOID = "void or unsettleable"
R_RULE = "rule artefact"
R_DATA = "data quality"
R_MODEL = "model miss"
R_VARIANCE = "variance"
R_UNEXPLAINED = "unexplained"
PRIORITY = (R_VOID, R_RULE, R_DATA, R_MODEL, R_VARIANCE)

ANALYST_PREFIX = "analyst-sourced:"
GOALS_ONLY_MODEL = "goals-only-poisson-v1"
SMALL_SAMPLE_MATCHES = {"soccer": 10, "american_football": 6, "basketball": 10, "baseball": 10, "hockey": 10}
MODEL_MISS_Z = 1.5            # the result missed the model's expectation by this many standard deviations against the pick
VARIANCE_FLOOR = 0.70         # at or above this estimate, a loss with nothing else firing is variance

# ---- evidence levels (guardrails)
ANECDOTE_MAX, WEAK_MAX, MODERATE_MIN, STRONG_MIN, STRONG_WEEKS = 9, 19, 20, 30, 3
ANECDOTE, WEAK, MODERATE, STRONG = "anecdote", "weak", "moderate", "strong"


def wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval for a hit rate."""
    if n <= 0:
        return 0.0, 1.0
    p = wins / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def evidence_level(n: int, weeks: int, mechanism: bool = False) -> str:
    if mechanism and n >= 1:
        return STRONG
    if n <= ANECDOTE_MAX:
        return ANECDOTE
    if n <= WEAK_MAX:
        return WEAK
    return STRONG if n >= STRONG_MIN and weeks >= STRONG_WEEKS else MODERATE


# ---------------------------------------------------------------- one leg
@dataclass(frozen=True)
class LegRecord:
    """A placed (or approved) leg joined to the evaluation it was picked on and the game's final score."""
    leg_id: str                       # "placed:12" or "eval:345"
    week: str
    placed: bool
    bet_id: str
    match_id: str
    sport: str
    league: str
    home: str
    away: str
    market_id: str
    specifier: str
    side: str
    market_text: str
    selection_text: str
    odds: float
    probability: float | None = None
    model: str = ""
    rule: str = ""
    original: str = ""
    ev: float | None = None
    inputs: tuple[tuple[str, float, str], ...] = ()   # (name, value, source)
    score: FinalScore | None = None
    score_note: str = ""
    expected: tuple[float, float] | None = None       # (expected home goals/points, expected away) when it could be recomputed
    sd_total: float | None = None


@dataclass(frozen=True)
class LegDiagnosis:
    leg: LegRecord
    settlement: Settlement
    primary: str
    reasons: tuple[str, ...]          # every category that fired, strongest first
    detail: tuple[str, ...]           # the evidence lines for the owner
    z: float | None = None
    counterfactual: str = ""          # what the ORIGINAL pick would have done, for rule artefacts


def settle(leg: LegRecord) -> Settlement:
    return settle_leg(leg.sport, leg.market_id, leg.specifier, leg.side, leg.score)


def market_kind(leg: LegRecord) -> str:
    """A short, stable name for the market, for pooling ('total over', 'handicap', '1st half total', ...)."""
    mid, sport = str(leg.market_id), leg.sport
    if sport == "soccer":
        return {"1": "1x2", "10": "double chance", "18": f"goals {leg.side}", "19": "home team total", "20": "away team total",
                "16": "asian handicap", "166": "corners", "139": "cards"}.get(mid, f"market {mid}")
    names = {"219": "winner", "223": "handicap", "225": f"total {leg.side}", "227": "home team total", "228": "away team total",
             "68": f"1st half total {leg.side}", "251": "winner", "256": "run line", "258": f"total {leg.side}",
             "274": "f5 result", "275": "f5 handicap", "276": f"f5 total {leg.side}", "406": "winner", "410": "puck line",
             "412": f"total {leg.side}"}
    return names.get(mid, f"market {mid}")


def cut_key(leg: LegRecord) -> str:
    return f"{leg.sport}|{market_kind(leg)}"


def analyst_sourced(leg: LegRecord) -> list[str]:
    return [src for _, _, src in leg.inputs if src.startswith(ANALYST_PREFIX)]


def min_matches(leg: LegRecord) -> float | None:
    values = [v for name, v, _ in leg.inputs if name.endswith(".matches")]
    return min(values) if values else None


_ORIGINAL_1X2 = re.compile(r"^(Home|Away|Draw)\s*@", re.IGNORECASE)


def original_settlement(leg: LegRecord) -> Settlement | None:
    """What the pick as FIRST OFFERED would have done, when we can read it: only the soccer 1X2 swap leaves a readable original."""
    if not leg.original or leg.sport != "soccer" or leg.score is None:
        return None
    found = _ORIGINAL_1X2.match(leg.original.strip())
    return settle_leg("soccer", "1", "", found.group(1).lower(), leg.score) if found else None


def model_miss_z(leg: LegRecord, settlement: Settlement) -> float | None:
    """How many standard deviations the result fell against the pick's side of the model's expectation, when the expectation was
    recomputed from the stored inputs (soccer totals, team totals, 1X2 / handicap margin). Positive = against the pick."""
    if leg.expected is None or leg.score is None or leg.sport != "soccer":
        return None
    eh, ea = leg.expected
    home, away = leg.score.home, leg.score.away
    try:
        line = float(leg.specifier.split("=")[1]) if "=" in leg.specifier else None
    except ValueError:
        line = None
    mid = str(leg.market_id)
    if mid == "18" and line is not None:
        sd = math.sqrt(max(eh + ea, 1e-9))
        gap = (home + away) - (eh + ea)
        return -gap / sd if leg.side == "over" else gap / sd
    if mid in ("19", "20") and line is not None:
        lam, count = (eh, home) if mid == "19" else (ea, away)
        sd = math.sqrt(max(lam, 1e-9))
        gap = count - lam
        return -gap / sd if leg.side == "over" else gap / sd
    if mid in ("1", "10", "16"):
        sd = math.sqrt(max(eh + ea, 1e-9))
        margin_gap = (home - away) - (eh - ea)
        wants_home = leg.side in ("home", "home or draw", "home or away")
        return -margin_gap / sd if wants_home else margin_gap / sd
    return None


def diagnose_leg(leg: LegRecord) -> LegDiagnosis:
    settlement = settle(leg)
    detail: list[str] = []
    reasons: list[str] = []
    counterfactual = ""
    z = None
    if settlement.verdict in (VOID, UNVERIFIABLE, PUSH, WIN, HALF_WIN):
        why = {VOID: "the leg was void", UNVERIFIABLE: "cannot be settled: " + (leg.score_note or settlement.reason),
               PUSH: "the leg pushed (stake refunded)", WIN: "the leg won", HALF_WIN: "the leg half-won"}[settlement.verdict]
        primary = R_VOID if settlement.verdict in (VOID, UNVERIFIABLE, PUSH) else ""
        return LegDiagnosis(leg, settlement, primary, (primary,) if primary else (), (why + f" ({settlement.reason})",), None)
    # a loss or half loss: collect every category that fires
    detail.append(f"lost: {settlement.reason}")
    original = original_settlement(leg)
    if leg.rule and original is not None:
        if original.verdict in (WIN, HALF_WIN):
            reasons.append(R_RULE)
            counterfactual = f"the original pick ({leg.original.split('(')[0].strip()}) would have won ({original.reason})"
            detail.append(f"rule '{leg.rule}' changed the pick and {counterfactual}")
        else:
            detail.append(f"rule '{leg.rule}' changed the pick but the original would also have lost: the rule is not blamed")
    quality = []
    cited = analyst_sourced(leg)
    if cited:
        quality.append(f"{len(cited)} input(s) came from the AI analyst, not a free source ({cited[0][:70]})")
    if leg.model == GOALS_ONLY_MODEL:
        quality.append("priced with the goals-only model (no xG)")
    games = min_matches(leg)
    floor = SMALL_SAMPLE_MATCHES.get(leg.sport)
    if games is not None and floor is not None and games < floor:
        quality.append(f"small sample: the smaller team had {games:g} games (< {floor})")
    if quality:
        reasons.append(R_DATA)
        detail.extend(quality)
    z = model_miss_z(leg, settlement)
    if z is not None:
        detail.append(f"the result was {z:+.1f} standard deviations against the model's expectation")
        if z >= MODEL_MISS_Z:
            reasons.append(R_MODEL)
    probability = leg.probability
    if probability is not None and probability >= VARIANCE_FLOOR and not reasons:
        reasons.append(R_VARIANCE)
        detail.append(f"estimated {probability:.0%}: a loss like this is expected {1 - probability:.0%} of the time; "
                      "no change is warranted from this bet")
    elif probability is not None and not reasons and z is not None and z < MODEL_MISS_Z:
        reasons.append(R_VARIANCE)
        detail.append("the result was within normal spread of the model's expectation")
    primary = next((r for r in PRIORITY if r in reasons), R_UNEXPLAINED)
    return LegDiagnosis(leg, settlement, primary, tuple(r for r in PRIORITY if r in reasons), tuple(detail), z, counterfactual)


# ---------------------------------------------------------------- one bet
@dataclass(frozen=True)
class BetReview:
    bet_id: str
    legs: tuple[LegDiagnosis, ...]
    total_odds: float | None
    stake: float | None
    bookmaker_result: str = ""

    @property
    def failed_legs(self) -> tuple[LegDiagnosis, ...]:
        return tuple(d for d in self.legs if d.settlement.lost)

    @property
    def joint_probability(self) -> float | None:
        values = [d.leg.probability for d in self.legs if d.leg.probability is not None]
        if len(values) != len(self.legs) or not values:
            return None
        out = 1.0
        for v in values:
            out *= v
        return out

    @property
    def weakest_failed(self) -> LegDiagnosis | None:
        failed = [d for d in self.failed_legs if d.leg.probability is not None]
        return min(failed, key=lambda d: d.leg.probability) if failed else None

    @property
    def verified(self) -> bool:
        return all(d.settlement.verdict != UNVERIFIABLE for d in self.legs)

    @property
    def cause(self) -> str:
        """The bet's headline reason: unverified, or the primary reason of its weakest failed leg."""
        if not self.failed_legs:
            return "cause unverified (no leg could be shown to have lost)"
        weakest = self.weakest_failed
        joint = self.joint_probability
        if joint is not None and joint < 0.5 and all(d.primary in (R_VARIANCE, R_UNEXPLAINED, "") for d in self.failed_legs):
            return f"variance by construction (all {len(self.legs)} legs together were only {joint:.0%} likely)"
        return (weakest or self.failed_legs[0]).primary


def review_bet(bet_id: str, legs: list[LegRecord], total_odds: float | None, stake: float | None, bookmaker_result: str = "") -> BetReview:
    return BetReview(bet_id, tuple(diagnose_leg(leg) for leg in legs), total_odds, stake, bookmaker_result)


# ---------------------------------------------------------------- pooled history, calibration and replay
@dataclass(frozen=True)
class SettledLeg:
    """One settled leg kept for pooled evidence (placed legs and labelled extra evidence from approved, unplaced picks)."""
    leg_id: str
    week: str
    placed: bool
    cut: str
    sport: str
    kind: str
    model: str
    rule: str
    analyst: bool
    small_sample: bool
    probability: float
    odds: float
    verdict: str
    ev: float | None = None


def settled_from(leg: LegRecord, settlement: Settlement) -> SettledLeg | None:
    """A history row, or None when the leg has no decided result or no estimate (it cannot calibrate anything)."""
    if settlement.verdict not in (WIN, LOSS) or leg.probability is None:
        return None
    games = min_matches(leg)
    floor = SMALL_SAMPLE_MATCHES.get(leg.sport)
    return SettledLeg(leg.leg_id, leg.week, leg.placed, cut_key(leg), leg.sport, market_kind(leg), leg.model, leg.rule,
                      bool(analyst_sourced(leg)), games is not None and floor is not None and games < floor,
                      leg.probability, leg.odds, settlement.verdict, leg.ev)


@dataclass(frozen=True)
class CutStats:
    n: int
    wins: int
    mean_estimate: float
    low: float
    high: float
    weeks: int

    @property
    def hit_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    @property
    def gap(self) -> float:
        return self.mean_estimate - self.hit_rate   # positive = the model is over-confident


def cut_stats(legs: list[SettledLeg]) -> CutStats:
    n = len(legs)
    wins = sum(1 for leg in legs if leg.verdict == WIN)
    low, high = wilson_interval(wins, n)
    mean = sum(leg.probability for leg in legs) / n if n else 0.0
    return CutStats(n, wins, mean, low, high, len({leg.week for leg in legs}))


def flat_profit(legs: list[SettledLeg]) -> float:
    """Units won staking 1 on each leg: odds - 1 on a win, -1 on a loss."""
    return sum((leg.odds - 1) if leg.verdict == WIN else -1.0 for leg in legs)


def replay_floor(legs: list[SettledLeg], new_floor: float) -> dict:
    """What raising the probability floor for these legs to `new_floor` would have done to the pooled history."""
    kept = [leg for leg in legs if leg.probability >= new_floor]
    removed = [leg for leg in legs if leg.probability < new_floor]
    kept_stats = cut_stats(kept)
    return {"floor": new_floor, "kept": len(kept), "removed": len(removed), "kept_hit_rate": kept_stats.hit_rate if kept else None,
            "kept_profit": round(flat_profit(kept), 2), "removed_profit": round(flat_profit(removed), 2),
            "removed_hit_rate": cut_stats(removed).hit_rate if removed else None}


# ---------------------------------------------------------------- recommendations
@dataclass(frozen=True)
class Recommendation:
    topic_key: str
    title: str
    category: str
    finding: str
    evidence: tuple[str, ...]
    proposed_change: str
    expected_effect: str
    risk: str
    effort: str
    confidence: str
    validation: str
    rollback: str
    n: int = 0
    weeks: int = 0
    mechanism: bool = False


def _pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def _calibration_finding(stats: CutStats) -> str:
    return (f"{stats.wins} of {stats.n} legs won ({_pct(stats.hit_rate)}; 95% interval {_pct(stats.low)} to {_pct(stats.high)}) against "
            f"a mean estimate of {_pct(stats.mean_estimate)}, over {stats.weeks} week(s)")


def _replay_text(replay: dict) -> str:
    if not replay["kept"] and not replay["removed"]:
        return "no history to replay"
    kept_rate = "n/a" if replay["kept_hit_rate"] is None else _pct(replay["kept_hit_rate"])
    return (f"replayed on this history, a floor of {_pct(replay['floor'])} would have removed {replay['removed']} leg(s) "
            f"(flat-stake result of the removed legs {replay['removed_profit']:+.1f} units) and kept {replay['kept']} "
            f"(hit rate {kept_rate}, {replay['kept_profit']:+.1f} units)")


def recommend(history: list[SettledLeg], coverage: "Coverage | None" = None, bets: list[BetReview] | None = None) -> list[Recommendation]:
    """Detailed recommendations from the pooled settled legs, this week's coverage gaps and this week's lost bets. A cut with too few
    legs yields no recommendation (anecdotes are only listed in the report once)."""
    out: list[Recommendation] = []
    out.extend(_calibration_recommendations(history))
    out.extend(_source_recommendations(history))
    if bets:
        out.extend(_accumulator_recommendations(bets))
    if coverage is not None:
        out.extend(coverage.recommendations())
    return out


def _calibration_recommendations(history: list[SettledLeg]) -> list[Recommendation]:
    out = []
    groups: dict[str, list[SettledLeg]] = defaultdict(list)
    for leg in history:
        groups[leg.cut].append(leg)
    for cut, legs in sorted(groups.items()):
        stats = cut_stats(legs)
        level = evidence_level(stats.n, stats.weeks)
        if level in (ANECDOTE, WEAK) or stats.gap < 0.08 or stats.high >= stats.mean_estimate:
            continue
        sport, kind = cut.split("|", 1)
        replay = replay_floor(legs, 0.75 if stats.mean_estimate < 0.80 else 0.80)
        out.append(Recommendation(
            f"calibration:{cut}", f"{sport.replace('_', ' ').title()} {kind}: the model is over-confident",
            "market calibration", _calibration_finding(stats) + f": the estimate sits above the top of the interval.",
            tuple(leg.leg_id for leg in legs if leg.verdict == LOSS)[:12],
            f"For {sport} '{kind}' picks only, shrink the estimate toward the observed rate (multiply by about "
            f"{stats.hit_rate / stats.mean_estimate:.2f}) or raise the probability floor to {_pct(replay['floor'])} in the engine "
            f"(predictions/core/engines/{sport if sport != 'soccer' else 'soccer'}.py; the shared floor is core/model.py PROBABILITY_FLOOR).",
            _replay_text(replay),
            "Good picks just above the new floor are lost; with a small sample the observed rate may itself be unlucky.",
            "M: one constant or a per-market floor, plus a test", level,
            f"Recheck after 20 more settled legs in this cut: the hit rate should sit inside the mean estimate's interval.",
            "Revert the shrink factor or floor; detect by the same hit-rate comparison.", stats.n, stats.weeks))
    return out


def _source_recommendations(history: list[SettledLeg]) -> list[Recommendation]:
    out = []
    probes = (("analyst", lambda leg: leg.analyst, "analyst-sourced inputs",
               "Block or haircut picks priced from AI-analyst inputs: require probability >= 80% for them, or demand a citation that lists "
               "the games used (predictions/core/analyst.py / flow/analyst.py merge_analyst).",
               "ANALYST inputs"),
              ("goals-only", lambda leg: leg.model == GOALS_ONLY_MODEL, "goals-only soccer estimates",
               "Lower soccer GOALS_ONLY_MAX_EV (predictions/core/engines/soccer.py, now 0.20) toward 0.12, or raise GOALS_ONLY_PRIOR_MATCHES "
               "(now 10) so goals-only estimates shrink harder toward the league average.", "GOALS-ONLY model"),
              ("small-sample", lambda leg: leg.small_sample, "small-sample picks",
               "Raise the minimum games before a team is priced (inputs.py MIN_MATCHES 5 / GOALS_ONLY_MIN_MATCHES 8, football SHRINK_GAMES) "
               "so thin samples become 'insufficient data'.", "SMALL-SAMPLE"))
    for key, test, label, change, _ in probes:
        legs = [leg for leg in history if test(leg)]
        rest = [leg for leg in history if not test(leg)]
        if not legs:
            continue
        stats, others = cut_stats(legs), cut_stats(rest)
        level = evidence_level(stats.n, stats.weeks)
        if level in (ANECDOTE, WEAK) or stats.gap < 0.06 or stats.high >= stats.mean_estimate:
            continue
        replay = replay_floor(legs, 0.80)
        out.append(Recommendation(
            f"source:{key}", f"Legs with {label} lose more than their estimate",
            "data quality", _calibration_finding(stats) + (f"; all other legs: {_pct(others.hit_rate)} against {_pct(others.mean_estimate)}"
                                                           if others.n else ""),
            tuple(leg.leg_id for leg in legs if leg.verdict == LOSS)[:12], change, _replay_text(replay),
            "Removes picks that may be fine; the gap may come from the leagues these picks cluster in, not the source itself.",
            "S to M: one constant or rule", level,
            "Recheck after 20 more settled legs of this kind.", "Revert the constant; compare the same hit rates.", stats.n, stats.weeks))
    return out


def _accumulator_recommendations(bets: list[BetReview]) -> list[Recommendation]:
    long_lost = [b for b in bets if len(b.legs) >= 3 and b.failed_legs]
    if len(long_lost) < 2:
        return []
    joints = [b.joint_probability for b in long_lost if b.joint_probability is not None]
    if not joints or sum(joints) / len(joints) >= 0.5:
        return []
    return [Recommendation(
        "accumulator:size", "Long accumulators were expected to lose most of the time",
        "bet construction", f"{len(long_lost)} lost bets had 3 or more legs and averaged a joint probability of {_pct(sum(joints) / len(joints))} "
                            "(the product of their leg estimates).",
        tuple(b.bet_id for b in long_lost),
        "Show the joint probability on the final list and cap the legs per slip (predictions/core/slip.py or staking.py), or warn below 50%.",
        "Fewer legs per ticket: a lower payout but a higher chance each ticket wins.",
        "Limits your choice of ticket; a cap is only a guard, not an edge.", "M: slip.py plus the summary screen", ANECDOTE if len(long_lost) < 10 else WEAK,
        "Compare the share of winning tickets before and after, over several weeks.", "Remove the cap.", len(long_lost), 1)]


# ---------------------------------------------------------------- coverage: why the Review engine could not judge selections
@dataclass
class Coverage:
    """What the review engine could NOT judge in the week, grouped by reason, so the owner sees where data or rules are missing."""
    total: int = 0
    judged: int = 0
    reasons: Counter = field(default_factory=Counter)             # category -> count
    examples: dict[str, list[str]] = field(default_factory=dict)  # category -> example games or markets
    groups: dict[str, Counter] = field(default_factory=dict)      # category -> counts per league / sport, to show where the gap is
    unclear_names: list[str] = field(default_factory=list)
    unverifiable_legs: int = 0
    placed_legs: int = 0

    def add(self, category: str, example: str, group: str = "") -> None:
        self.reasons[category] += 1
        if group:
            self.groups.setdefault(category, Counter())[group] += 1
        bucket = self.examples.setdefault(category, [])
        if example not in bucket and len(bucket) < 6:
            bucket.append(example)

    def recommendations(self) -> list[Recommendation]:
        out = []
        share = (self.total - self.judged) / self.total if self.total else 0.0
        specs = (
            ("no rules for market", "markets",
             "Add the market to the engine: price it from the same scoreline model and add it to understands() / evaluate_exact() "
             "(predictions/core/engines/soccer.py for soccer), as done for Asian handicap and multigoals.", "L (a model per market)"),
            ("no rules for sport", "sports",
             "Add an engine for this sport (core/engines/ and ENGINES in core/engines/__init__.py) with a stats source.", "L"),
            ("insufficient data", "games",
             "Add a free stats source for these leagues, or accept the goals-only fallback for them (predictions/flow/analyst.py "
             "goals_only_fallback is already on for Review).", "M to L (a source adapter)"),
            ("analyst call limit", "games",
             "Raise ANALYST_CALL_LIMIT in config.env (default 40) if big codes are being cut off.", "S (a setting)"),
        )
        for category, noun, change, effort in specs:
            count = self.reasons.get(category, 0)
            if count < 3 or count / max(self.total, 1) < 0.10:
                continue
            out.append(Recommendation(
                f"coverage:{category}", f"Review could not judge {count} selection(s): {category}", "coverage",
                f"{count} of {self.total} reviewed selections ({_pct(count / self.total)}) hit '{category}'."
                + (" Most affected: " + ", ".join(f"{g} {n}" for g, n in self.groups[category].most_common(6)) + "."
                   if self.groups.get(category) else "")
                + " Examples: " + "; ".join(self.examples.get(category, [])[:4]),
                tuple(self.examples.get(category, [])), change,
                f"Would turn up to {count} disqualified selection(s) a week into priced ones (they still must clear the floor).",
                "More priced selections is not more approvals; each new market needs its own validation.", effort,
                STRONG if count >= STRONG_MIN else MODERATE if count >= MODERATE_MIN else WEAK,
                "Count of selections with this reason next week.", "Remove the market or source.", count, 1, True))
        if len(set(self.unclear_names)) >= 2:
            names = sorted(set(self.unclear_names))
            out.append(Recommendation(
                "coverage:team names", f"{len(names)} team name(s) could not be matched", "coverage",
                "Unmatched or ambiguous team names: " + "; ".join(names[:8]),
                tuple(names[:12]),
                "Add each pair to the alias tables (predictions/core/identity.py ALIASES, adapters/espn/soccer.py ALIASES) so those games are priced.",
                "Each alias unlocks the games of that team.", "A wrong alias would mix up two clubs: check each one.", "S (data only)",
                STRONG, "No 'team names are unclear' lines for these teams next week.", "Remove the alias.", len(names), 1, True))
        if self.unverifiable_legs and self.placed_legs and self.unverifiable_legs / self.placed_legs >= 0.25:
            out.append(Recommendation(
                "coverage:settlement", "A quarter or more of your placed legs could not be settled for this review", "coverage",
                f"{self.unverifiable_legs} of {self.placed_legs} placed legs have no free final score or a market a score cannot settle.",
                (), "Add score sources for those leagues (football-data.co.uk CSVs, the EuroLeague API) in predictions/adapters/espn/scores.py "
                    "and corner/card counts from football-data.co.uk.", "Turns 'unverifiable' legs into diagnosed ones.",
                "More sources are more code to keep working.", "M to L", MODERATE, "Share of unverifiable placed legs next week.",
                "Remove the source.", self.unverifiable_legs, 1, True))
        if share >= 0.5 and self.total >= 10 and not out:
            out.append(Recommendation(
                "coverage:overall", f"{_pct(share)} of reviewed selections could not be judged", "coverage",
                f"{self.total - self.judged} of {self.total} selections were disqualified for missing data or rules.", (),
                "Review the reasons in the report; no single reason passed the threshold.", "n/a", "n/a", "n/a", WEAK,
                "Next week's share.", "n/a", self.total - self.judged, 1))
        return out
