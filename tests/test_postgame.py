"""The weekly post-game analysis: diagnoses, guardrails, replay and recommendations. Pure functions."""
import pytest

from predictions.core.postgame import (ANECDOTE, MODERATE, STRONG, WEAK, Coverage, LegRecord, R_DATA, R_MODEL, R_RULE,
                                       R_UNEXPLAINED, R_VARIANCE, R_VOID, SettledLeg, cut_stats, diagnose_leg, evidence_level,
                                       flat_profit, recommend, replay_floor, review_bet, settled_from, wilson_interval)
from predictions.core.settle import LOSS, WIN, FinalScore, settle_leg


def leg(**kw):
    base = dict(leg_id="placed:1", week="2026-W41", placed=True, bet_id="b1", match_id="soccer:l:20261001T1900Z:a:b", sport="soccer",
                league="l", home="A", away="B", market_id="18", specifier="total=2.5", side="over", market_text="Over/Under 2.5",
                selection_text="Over 2.5", odds=1.8, probability=0.75, model="poisson-xg-v1",
                inputs=(("home.matches", 12.0, "understat"), ("away.matches", 12.0, "understat")),
                score=FinalScore(1, 0), expected=(1.6, 1.3))
    base.update(kw)
    return LegRecord(**base)


def test_wilson_interval_and_evidence_levels():
    low, high = wilson_interval(15, 30)
    assert 0.33 < low < 0.5 < high < 0.67
    assert wilson_interval(0, 0) == (0.0, 1.0)
    assert evidence_level(5, 1) == ANECDOTE and evidence_level(15, 2) == WEAK and evidence_level(25, 2) == MODERATE
    assert evidence_level(30, 2) == MODERATE and evidence_level(30, 3) == STRONG
    assert evidence_level(1, 1, mechanism=True) == STRONG


# ---------------------------------------------------------------- diagnoses
def test_a_high_estimate_loss_with_nothing_else_is_variance():
    d = diagnose_leg(leg(score=FinalScore(1, 1), expected=(1.1, 1.0)))
    assert d.settlement.verdict == LOSS and d.primary == R_VARIANCE and "no change is warranted" in " ".join(d.detail)


def test_a_result_far_below_the_model_expectation_is_a_model_miss():
    d = diagnose_leg(leg(score=FinalScore(0, 0), expected=(2.0, 1.8)))
    assert d.z is not None and d.z >= 1.5 and R_MODEL in d.reasons and d.primary == R_MODEL


def test_analyst_goals_only_and_small_samples_are_data_quality_and_outrank_a_model_miss():
    analyst = leg(inputs=(("home.matches", 12.0, "analyst-sourced:example.com/page"), ("away.matches", 12.0, "understat")),
                  score=FinalScore(0, 0), expected=(2.0, 1.8))
    assert diagnose_leg(analyst).primary == R_DATA
    assert R_DATA in diagnose_leg(leg(model="goals-only-poisson-v1", score=FinalScore(0, 0))).reasons
    thin = leg(inputs=(("home.matches", 4.0, "espn"), ("away.matches", 12.0, "espn")), score=FinalScore(0, 0))
    assert "small sample" in " ".join(diagnose_leg(thin).detail)


def test_a_rule_that_changed_the_pick_is_blamed_only_when_the_original_would_have_won():
    # the swap picked Over 2.5; 2-0 is two goals so it lost, while the original Home pick would have won
    swapped = leg(rule="1X2 swap", original="Home @ 2.10 (estimate 57.8%)", score=FinalScore(2, 0))
    d = diagnose_leg(swapped)
    assert d.primary == R_RULE and "would have won" in d.counterfactual
    both_lose = diagnose_leg(leg(rule="1X2 swap", original="Home @ 2.10 (estimate 57.8%)", score=FinalScore(0, 1)))
    assert R_RULE not in both_lose.reasons and "not blamed" in " ".join(both_lose.detail)


def test_void_pushed_and_unverifiable_legs_are_never_blamed_on_the_model():
    assert diagnose_leg(leg(score=None, score_note="no free score source")).primary == R_VOID
    assert diagnose_leg(leg(market_id="166")).primary == R_VOID
    pushed = diagnose_leg(leg(specifier="total=2", score=FinalScore(1, 1)))
    assert pushed.primary == R_VOID and "pushed" in pushed.detail[0]
    won = diagnose_leg(leg(score=FinalScore(2, 1)))
    assert won.settlement.verdict == WIN and won.primary == ""


def test_a_loss_below_the_variance_floor_with_no_expectation_is_unexplained_not_forced():
    d = diagnose_leg(leg(probability=0.66, expected=None, score=FinalScore(0, 0)))
    assert d.primary == R_UNEXPLAINED


# ---------------------------------------------------------------- bets
def test_a_long_multiple_with_a_low_joint_probability_is_variance_by_construction():
    legs = [leg(leg_id=f"placed:{i}", probability=0.72, score=FinalScore(1, 1), expected=(1.1, 1.0)) for i in range(3)]
    bet = review_bet("b1", legs, 5.0, 10.0)
    assert bet.joint_probability == pytest.approx(0.72 ** 3) and bet.cause.startswith("variance by construction")
    assert bet.verified and bet.weakest_failed is not None


def test_a_bet_with_no_provably_lost_leg_is_cause_unverified():
    bet = review_bet("b2", [leg(score=None, score_note="no source")], 2.0, 10.0)
    assert not bet.verified and bet.cause.startswith("cause unverified")


# ---------------------------------------------------------------- history, replay, recommendations
def hist(n, hit_rate, p=0.8, week_count=1, **kw):
    wins = round(n * hit_rate)
    return [SettledLeg(f"placed:{i}", f"2026-W{41 + i % week_count}", True, "soccer|goals over", "soccer", "goals over",
                       kw.get("model", "poisson-xg-v1"), "", kw.get("analyst", False), False, p, 1.5, WIN if i < wins else LOSS)
            for i in range(n)]


def test_cut_stats_and_replay():
    legs = hist(40, 0.5, p=0.8)
    stats = cut_stats(legs)
    assert stats.n == 40 and stats.wins == 20 and stats.gap == pytest.approx(0.3)
    assert flat_profit(legs) == pytest.approx(20 * 0.5 - 20)
    extra = SettledLeg("x", "w", True, "c", "soccer", "k", "m", "", False, False, 0.72, 2.0, LOSS)
    replay = replay_floor(legs + [extra], 0.75)
    assert replay["removed"] == 1 and replay["removed_profit"] == -1.0 and replay["kept"] == 40


def test_a_miscalibrated_cut_gets_a_detailed_recommendation_only_with_enough_legs_and_weeks():
    assert recommend(hist(8, 0.4)) == []                       # anecdote: never recommended
    assert recommend(hist(15, 0.4)) == []                      # weak: watch only
    moderate = recommend(hist(25, 0.4, week_count=2))
    assert moderate and moderate[0].confidence == MODERATE and "over-confident" in moderate[0].title
    r = recommend(hist(40, 0.4, week_count=3))[0]
    assert r.confidence == STRONG and "PROBABILITY_FLOOR" in r.proposed_change and "removed" in r.expected_effect
    assert r.validation and r.rollback and r.effort and r.risk and r.finding
    assert r.topic_key == "calibration:soccer|goals over"
    assert recommend(hist(40, 0.78, week_count=3)) == []      # well calibrated: nothing to recommend


def test_the_analyst_and_goals_only_sources_get_their_own_recommendations():
    legs = hist(30, 0.4, week_count=3, analyst=True) + hist(30, 0.8, p=0.8, week_count=3)
    assert "source:analyst" in {r.topic_key for r in recommend(legs)}
    goals_only = hist(30, 0.4, week_count=3, model="goals-only-poisson-v1")
    assert "source:goals-only" in {r.topic_key for r in recommend(goals_only)}


def test_settled_from_keeps_only_decided_legs_with_an_estimate():
    win = settle_leg("soccer", "18", "total=2.5", "over", FinalScore(2, 1))
    assert settled_from(leg(), win).verdict == WIN
    assert settled_from(leg(probability=None), win) is None
    assert settled_from(leg(), settle_leg("soccer", "166", "", "over", FinalScore(1, 1))) is None


# ---------------------------------------------------------------- coverage
def test_coverage_gaps_become_recommendations_with_counts_and_examples():
    cov = Coverage(total=30, judged=14, placed_legs=4, unverifiable_legs=2)
    for i in range(6):
        cov.add("no rules for market", f"Goal Bounds game {i}")
    for i in range(4):
        cov.add("insufficient data", f"Lower league game {i}")
    cov.unclear_names += ["Alianza FC Valledupar", "Atletico-MG", "Alianza FC Valledupar"]
    keys = {r.topic_key: r for r in cov.recommendations()}
    assert {"coverage:no rules for market", "coverage:insufficient data", "coverage:team names", "coverage:settlement"} <= set(keys)
    assert "Goal Bounds" in keys["coverage:no rules for market"].finding and keys["coverage:team names"].n == 2
    assert "ALIASES" in keys["coverage:team names"].proposed_change
    assert Coverage(total=30, judged=29).recommendations() == []


def test_accumulator_recommendation_needs_two_long_lost_bets_with_low_joint_probability():
    def lost_bet(i):
        legs = [leg(leg_id=f"placed:{i}{j}", probability=0.72, score=FinalScore(1, 1), expected=(1.1, 1.0)) for j in range(4)]
        return review_bet(f"b{i}", legs, 6.0, 10.0)
    assert recommend([], bets=[lost_bet(1)]) == []
    recs = recommend([], bets=[lost_bet(1), lost_bet(2)])
    assert recs and recs[0].topic_key == "accumulator:size"


def test_coverage_findings_name_the_most_affected_leagues():
    cov = Coverage(total=40, judged=20)
    for i in range(8):
        cov.add("insufficient data", f"game {i}", "International - Euroleague (basketball)")
    for i in range(4):
        cov.add("insufficient data", f"other {i}", "Germany - BBL (basketball)")
    finding = {r.topic_key: r for r in cov.recommendations()}["coverage:insufficient data"].finding
    assert "Most affected: International - Euroleague (basketball) 8, Germany - BBL (basketball) 4" in finding
