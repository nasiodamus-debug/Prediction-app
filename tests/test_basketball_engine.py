"""Stories 6.1 and 6.2: the basketball engine (model, ORTG lock, 1H fast-pace route, floor and EV on converted picks),
its input registry, its market-to-betslip mapping and the closest-option rule for basketball lines."""
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from predictions.core import betting
from predictions.core.betting import UnsupportedLeg, leg_from_evaluation
from predictions.core.engines import ENGINES
from predictions.core.engines import basketball as bb
from predictions.core.engines.basketball import BasketballEngine
from predictions.core.inputs import BASKETBALL_REGISTRY, Input, check_inputs, missing_required
from predictions.core.model import PROBABILITY_FLOOR, Estimate, validate
from predictions.core.options import closest_option
from predictions.core.stats import INSUFFICIENT, READY, assess, merge_analyst
from predictions.core.types import Fixture, MarketOffer, MatchId, Outcome

KICKOFF = datetime(2026, 10, 21, 0, 0, tzinfo=timezone.utc)
D = Decimal


def match_id(sport="basketball"):
    return MatchId(sport, "USA - NBA", KICKOFF, "Home Team", "Away Team")


def make_fixture(offers=()):
    return Fixture(match_id(), "USA - NBA", "Home Team", "Away Team", KICKOFF, False, tuple(offers))


def team_inputs(side, ortg=112.0, drtg=112.0, pace=100.0, fh_for=None, fh_against=None, points_for=None):
    src = "test-source"
    out = [Input(f"{side}.matches", 10.0, "games", src),
           Input(f"{side}.ortg", ortg, "points per 100 possessions", src),
           Input(f"{side}.drtg", drtg, "points per 100 possessions", src),
           Input(f"{side}.pace", pace, "possessions per game", src),
           Input(f"{side}.points_for", points_for if points_for is not None else ortg * pace / 100,
                 "points per game", src)]
    if fh_for is not None:
        out.append(Input(f"{side}.fh_points_for", fh_for, "1st-half points per game", src))
        out.append(Input(f"{side}.fh_points_against", fh_against, "1st-half points per game", src))
    return out


def inputs(home=None, away=None, league_pace=None):
    found = team_inputs("home", **(home or {})) + team_inputs("away", **(away or {}))
    if league_pace is not None:
        found.append(Input("match.league_pace", league_pace, "possessions per game", "test-source"))
    return found


def offer(market_id, name, spec_key, lines, over_odds="1.60", under_odds="1.60", sides=("Over", "Under")):
    """One offer per line, like the SportyBet listing."""
    return [MarketOffer(market_id, name, f"{spec_key}={line}",
                        tuple(Outcome(f"{side} {line}", D(over_odds if side == "Over" else under_odds)) for side in sides))
            for line in lines]


def totals(lines=("214.5", "220.5", "224.5", "228.5"), **kw):
    return offer(bb.TOTAL_MARKET_ID, "Over/Under (incl. overtime)", "total", lines, **kw)


def team_totals(side, lines=("104.5", "108.5", "112.5"), **kw):
    mid = bb.HOME_TOTAL_MARKET_ID if side == "home" else bb.AWAY_TOTAL_MARKET_ID
    return offer(mid, f"{side.title()} O/U (incl. overtime)", "total", lines, **kw)


def fh_totals(lines=("100.5", "108.5", "112.5"), **kw):
    return offer(bb.FH_TOTAL_MARKET_ID, "1st half - total", "total", lines, **kw)


def run(inps, offers):
    fixture = make_fixture(offers)
    return ENGINES["basketball"].evaluate(fixture, inps, fixture.markets)


def by(evals, text):
    return [e for e in evals if text in e.market or text in e.selection]


# ------------------------------------------------------------------ registry and model basics
def test_basketball_is_registered_in_the_engine_registry():
    assert isinstance(ENGINES["basketball"], BasketballEngine) and "soccer" in ENGINES


def test_the_floor_is_imported_not_redefined_in_the_basketball_engine():
    source = open(bb.__file__, encoding="utf-8").read()
    assert "PROBABILITY_FLOOR =" not in source and "0.65" not in source.replace("0.65 ", "")


def test_thresholds_are_named_constants():
    assert bb.ORTG_LOCK_THRESHOLD == 108.0 and 0 < bb.EXPLOSIVE_MIN_FH_SHARE < 1 and bb.EXPLOSIVE_MIN_PACE_RATIO > 0


def test_half_line_probabilities_add_up_and_a_whole_line_has_a_push():
    over, under, push = bb.line_probabilities(224.0, 19.0, D("224.5"))
    assert push == 0.0 and abs(over + under - 1.0) < 1e-12 and 0.45 < over < 0.55
    over, under, push = bb.line_probabilities(224.0, 19.0, D("224"))
    assert 0.015 < push < 0.03 and abs(over + under + push - 1.0) < 1e-9
    assert bb.line_probabilities(224.0, 19.0, D("224.25")) is None  # quarter lines are not supported
    assert bb.line_probabilities(224.0, 0.0, D("224.5")) is None


def test_the_estimate_is_deterministic():
    a = [(e.market, e.selection, e.estimate.probability, e.ev) for e in run(inputs(), totals() + team_totals("home"))]
    b = [(e.market, e.selection, e.estimate.probability, e.ev) for e in run(inputs(), totals() + team_totals("home"))]
    assert a == b and a


def test_missing_required_input_prices_nothing():
    partial = [i for i in inputs() if i.name != "home.ortg"]
    assert run(partial, totals()) == []


# ------------------------------------------------------------------ Story 6.1: markets, shopping, floor and EV
def test_no_rule_prices_game_totals_and_team_totals_on_both_sides_with_one_line_each():
    # ORTG 105 (under 108) and no first-half data: nothing routes anywhere, everything is priced as listed
    weak = dict(ortg=105.0, drtg=105.0)
    evals = run(inputs(weak, weak), totals() + team_totals("home") + team_totals("away"))
    game = by(evals, "Total points")
    assert sorted(e.selection.split()[0] for e in game) == ["Over"]  # ONE Over line, not every listed line; no Under (AD-20)
    assert len(by(evals, "Home team total")) == 1 and len(by(evals, "Away team total")) == 1
    assert all(e.rule == "" and e.original == "" for e in evals)


def test_one_line_per_side_is_shown_the_passing_line_with_the_highest_estimate():
    weak = dict(ortg=105.0, drtg=105.0)  # expected game total about 210
    lines = ("190.5", "200.5", "205.5")
    evals = run(inputs(weak, weak), totals(lines, over_odds="1.50", under_odds="2.50"))
    overs = [e for e in evals if e.selection.startswith("Over")]
    assert len(overs) == 1 and overs[0].passed
    singles = [run(inputs(weak, weak), totals((line,), over_odds="1.50", under_odds="2.50"))[0] for line in lines]
    singles = [e for e in singles if e.selection.startswith("Over") and e.passed]
    assert len(singles) == 2 and overs[0].estimate.probability == max(e.estimate.probability for e in singles)
    assert overs[0].selection == "Over 190.5"  # the safest line that still has positive EV, not the longest price


def test_a_rejected_side_shows_the_closest_rejection_only():
    weak = dict(ortg=105.0, drtg=105.0)
    evals = run(inputs(weak, weak), totals(("212.5", "216.5", "220.5"), over_odds="1.90", under_odds="1.90"))
    assert len([e for e in evals if e.selection.startswith("Over")]) == 1
    assert not any(e.passed for e in evals if e.selection.startswith("Over"))


def test_a_pick_below_the_floor_is_rejected_with_its_estimate_and_reason():
    weak = dict(ortg=105.0, drtg=105.0)
    evals = run(inputs(weak, weak), totals(("210.5",), over_odds="1.90", under_odds="1.90"))
    over = by(evals, "Over")[0]
    assert not over.passed and over.estimate.probability < PROBABILITY_FLOOR and "below the 70% floor" in over.reason


def test_a_pick_at_or_above_the_floor_needs_positive_ev():
    weak = dict(ortg=105.0, drtg=105.0)
    cheap = run(inputs(weak, weak), totals(("190.5",), over_odds="1.05"))  # very likely but priced too short
    over = by(cheap, "Over")[0]
    assert over.estimate.probability >= PROBABILITY_FLOOR and not over.passed and "expected value" in over.reason
    fair = run(inputs(weak, weak), totals(("190.5",), over_odds="1.60"))
    assert by(fair, "Over")[0].passed


def test_the_gate_is_the_shared_one_for_70_percent():
    assert validate(Estimate(0.699), D("2.0"))[0] is False
    assert validate(Estimate(0.70), D("2.0"))[0] is True


def handicap_offers(*lines, odds="1.90"):
    """SportyBet spells the outcomes "Home (-9.5)" / "Away (+9.5)" for the specifier hcp=-9.5."""
    offers = []
    for h in lines:
        value = D(h)
        away = f"{-value:+}".replace("+0", "0") if value != value.to_integral() else f"{int(-value):+d}"
        offers.append(MarketOffer(bb.HANDICAP_MARKET_ID, "Handicap (incl. overtime)", f"hcp={h}",
                                  (Outcome(f"Home ({value:+})", D(odds)), Outcome(f"Away ({away})", D(odds)))))
    return offers


def test_the_winner_and_handicap_are_priced_from_the_same_model():
    strong = dict(ortg=120.0, drtg=105.0)
    weak = dict(ortg=105.0, drtg=118.0)
    winner = MarketOffer(bb.WINNER_MARKET_ID, "Winner (incl. overtime)", "",
                         (Outcome("Home", D("1.30")), Outcome("Away", D("3.60"))))
    evals = run(inputs(strong, weak), [winner, *handicap_offers("-9.5", "-5.5", "3.5")])
    won = {e.selection: e for e in by(evals, "Winner")}
    assert won["Home"].estimate.probability > 0.85
    assert won["Home"].estimate.probability + won["Away"].estimate.probability <= 1.0
    hcp = by(evals, "Handicap")
    assert {e.selection.split()[0] for e in hcp} == {"Home", "Away"}  # one per side
    assert all(e.market.startswith("Handicap ") for e in hcp)


def test_a_whole_handicap_line_is_not_offered_but_review_prices_it_with_the_refund_maths():
    strong = dict(ortg=120.0, drtg=105.0)
    weak = dict(ortg=105.0, drtg=118.0)
    whole = [MarketOffer(bb.HANDICAP_MARKET_ID, "Handicap (incl. overtime)", "hcp=-12",
                         (Outcome("Home (-12)", D("2.00")), Outcome("Away (+12)", D("1.80"))))]
    assert run(inputs(strong, weak), whole) == []  # AD-21: a whole handicap is never offered
    fixture = make_fixture(whole)
    evals = ENGINES["basketball"].evaluate_exact(fixture, inputs(strong, weak), whole[0])  # Review: exactly 12 points refunds
    home = by(evals, "Home (-12)")[0]
    assert home.estimate.push > 0 and any("refund" in n for n in home.notes)


# ------------------------------------------------------------------ Story 6.2: ORTG lock
def test_an_ortg_of_exactly_108_locks_team_total_over_and_bypasses_full_game_totals():
    locked = dict(ortg=108.0, drtg=112.0)
    other = dict(ortg=100.0, drtg=112.0)
    evals = run(inputs(locked, other), totals() + team_totals("home", lines=("96.5", "100.5")) + team_totals("away"))
    assert not by(evals, "Total points")  # full-game totals are bypassed
    home_totals = by(evals, "Home team total")
    assert [e.selection.split()[2] for e in home_totals] == ["Over"]  # Team Total OVER only for the locked team
    e = home_totals[0]
    assert e.rule.startswith("ORTG lock") and "108.0" in e.rule and "108.0 rolling" not in e.rule
    assert e.original.startswith(("Over ", "Under ")) and "estimate" in e.original  # the original full-game pick is shown
    assert len(by(evals, "Away team total")) == 1  # the other team is priced as usual (its Over; no Under, AD-20)


def test_an_ortg_just_under_108_does_not_lock():
    evals = run(inputs(dict(ortg=107.9, drtg=112.0), dict(ortg=107.9, drtg=112.0)), totals() + team_totals("home"))
    assert by(evals, "Total points") and not any(e.rule for e in evals)


def test_when_both_teams_lock_both_get_a_team_total_over():
    evals = run(inputs(dict(ortg=112.0), dict(ortg=113.0)), totals() + team_totals("home") + team_totals("away"))
    assert not by(evals, "Total points")
    assert {e.market.split()[0] for e in evals if e.rule.startswith("ORTG lock")} == {"Home", "Away"}
    assert all("Under" not in e.selection for e in evals)


def test_the_converted_team_total_stays_only_if_it_clears_the_floor_with_positive_ev():
    locked = dict(ortg=112.0)
    good = run(inputs(locked, dict(ortg=100.0)), team_totals("home", lines=("104.5",), over_odds="1.60"))
    assert good[0].passed and good[0].rule.startswith("ORTG lock")
    below = run(inputs(locked, dict(ortg=100.0)), team_totals("home", lines=("116.5",), over_odds="2.60"))
    assert not below[0].passed and below[0].estimate.probability < PROBABILITY_FLOOR
    assert below[0].rule.startswith("ORTG lock")  # the rejection still shows the rule and the estimate
    short = run(inputs(locked, dict(ortg=100.0)), team_totals("home", lines=("104.5",), over_odds="1.05"))
    assert not short[0].passed and "expected value" in short[0].reason


def test_the_lock_with_no_listed_team_total_gives_no_full_game_pick_either():
    evals = run(inputs(dict(ortg=112.0), dict(ortg=100.0)), totals())
    assert not by(evals, "Total points") and not by(evals, "team total")


def test_the_original_pick_shown_is_the_best_full_game_total():
    evals = run(inputs(dict(ortg=112.0), dict(ortg=100.0)), totals(("190.5", "224.5"), over_odds="1.60") +
                team_totals("home", lines=("104.5",)))
    converted = by(evals, "Home team total")[0]
    assert "Over 190.5" in converted.original  # the line the bettor would have taken without the rule


# ------------------------------------------------------------------ Story 6.2: 1H fast-pace route
FAST = dict(ortg=105.0, drtg=108.0, pace=104.0, fh_for=56.0, fh_against=52.0, points_for=105.0 * 104.0 / 100)  # share .513


def fast_team(share=0.53, pace=104.0):
    points = 110.0
    return dict(ortg=105.0, drtg=108.0, pace=pace, fh_for=points * share, fh_against=54.0, points_for=points)


def test_an_explosive_start_routes_to_the_first_half_total_over_and_bypasses_full_game_totals():
    home, away = fast_team(), dict(ortg=104.0, drtg=108.0, pace=98.0, fh_for=53.0, fh_against=53.0, points_for=103.0)
    evals = run(inputs(home, away, league_pace=100.0),
                totals() + team_totals("home") + fh_totals(("100.5", "104.5"), over_odds="1.60"))
    assert not by(evals, "Total points")
    fh = by(evals, "1st half total")
    assert [e.selection.split()[2] for e in fh] == ["Over"]
    e = fh[0]
    assert e.rule.startswith("1H fast-pace route") and "Home Team" in e.rule and e.original
    assert not any(x.rule.startswith("ORTG lock") for x in evals)  # ORTG 105 does not lock


def test_a_converted_first_half_pick_is_rejected_when_it_does_not_clear_the_floor():
    home, away = fast_team(), dict(ortg=104.0, drtg=108.0, pace=98.0, fh_for=53.0, fh_against=53.0, points_for=103.0)
    evals = run(inputs(home, away, league_pace=100.0), fh_totals(("112.5",), over_odds="1.90"))
    e = by(evals, "1st half total")[0]
    assert not e.passed and e.rule.startswith("1H fast-pace route") and "below the 70% floor" in e.reason


@pytest.mark.parametrize("home_kw,league_pace", [
    (dict(share=0.50), 100.0),            # scores half of its points early: not front-loaded
    (dict(share=0.53, pace=95.0), 100.0),  # front-loaded but slower than the league
    (dict(share=0.53), None),             # no league pace to compare with: the rule cannot apply (fail closed)
])
def test_not_explosive_means_no_routing(home_kw, league_pace):
    home = fast_team(**home_kw)
    away = dict(ortg=104.0, drtg=108.0, pace=98.0, fh_for=53.0, fh_against=53.0, points_for=103.0)
    evals = run(inputs(home, away, league_pace=league_pace), totals() + fh_totals(("100.5",)))
    assert by(evals, "Total points") and not any(e.rule.startswith("1H fast-pace") for e in evals)
    assert {e.selection.split()[2] for e in by(evals, "1st half total")} == {"Over"}  # 1H priced as usual (no Under, AD-20)


def test_the_fast_pace_boundary_is_at_least_the_share_and_at_least_the_league_pace():
    home = fast_team(share=bb.EXPLOSIVE_MIN_FH_SHARE, pace=100.0)  # exactly on both thresholds
    away = dict(ortg=104.0, drtg=108.0, pace=98.0, fh_for=53.0, fh_against=53.0, points_for=103.0)
    evals = run(inputs(home, away, league_pace=100.0), totals() + fh_totals(("100.5",)))
    assert any(e.rule.startswith("1H fast-pace") for e in evals)


def test_first_half_picks_without_first_half_data_are_insufficient_never_guessed():
    evals = run(inputs(dict(ortg=105.0), dict(ortg=105.0)), fh_totals(("100.5",)))
    assert evals and all(e.insufficient and not e.passed and e.reason.startswith("insufficient data") for e in evals)


def test_lock_and_fast_pace_can_both_apply_and_both_outputs_are_listed():
    home = dict(ortg=112.0, drtg=108.0, pace=104.0, fh_for=60.0, fh_against=54.0, points_for=116.0)
    away = dict(ortg=104.0, drtg=108.0, pace=98.0, fh_for=53.0, fh_against=53.0, points_for=103.0)
    evals = run(inputs(home, away, league_pace=100.0),
                totals() + team_totals("home", lines=("104.5",)) + fh_totals(("100.5",)))
    rules = {e.rule.split(":")[0] for e in evals if e.rule}
    assert rules == {"ORTG lock", "1H fast-pace route"} and not by(evals, "Total points")


# ------------------------------------------------------------------ Review: exact selections at the code's odds
def test_understands_only_listed_basketball_markets():
    engine = ENGINES["basketball"]
    assert engine.understands(totals()[0]) and engine.understands(fh_totals()[0])
    assert not engine.understands(MarketOffer("18", "Over/Under", "total=2.5", ()))
    assert not engine.understands(MarketOffer("74", "1st half - odd/even", "", ()))


def test_exact_judgement_uses_the_code_odds_and_never_converts():
    fixture = make_fixture()
    engine = ENGINES["basketball"]
    one = MarketOffer(bb.TOTAL_MARKET_ID, "Over/Under (incl. overtime)", "total=190.5", (Outcome("Over 190.5", D("1.60")),))
    (e,) = engine.evaluate_exact(fixture, inputs(dict(ortg=105.0), dict(ortg=105.0)), one)
    assert e.passed and e.odds == D("1.60") and e.rule == "" and e.original == ""
    worse = MarketOffer(bb.TOTAL_MARKET_ID, "Over/Under (incl. overtime)", "total=190.5", (Outcome("Over 190.5", D("1.03")),))
    (w,) = engine.evaluate_exact(fixture, inputs(dict(ortg=105.0), dict(ortg=105.0)), worse)
    assert w.estimate.probability == e.estimate.probability and not w.passed


def test_exact_judgement_of_a_full_game_total_notes_the_ortg_lock_without_swapping_it():
    fixture = make_fixture()
    one = MarketOffer(bb.TOTAL_MARKET_ID, "Over/Under (incl. overtime)", "total=190.5", (Outcome("Over 190.5", D("1.60")),))
    (e,) = ENGINES["basketball"].evaluate_exact(fixture, inputs(dict(ortg=112.0), dict(ortg=100.0)), one)
    assert e.selection == "Over 190.5" and any(n.startswith("ORTG lock applies to Home Team") for n in e.notes)


def test_exact_first_half_without_data_is_an_insufficient_evaluation():
    fixture = make_fixture()
    one = MarketOffer(bb.FH_TOTAL_MARKET_ID, "1st half - total", "total=100.5", (Outcome("Over 100.5", D("1.60")),))
    (e,) = ENGINES["basketball"].evaluate_exact(fixture, inputs(), one)
    assert e.insufficient and not e.passed


# ------------------------------------------------------------------ input registry
def test_the_basketball_registry_rejects_out_of_range_and_wrong_unit_values():
    kept, dropped = check_inputs([Input("home.ortg", 300.0, "points per 100 possessions", "x"),
                                  Input("home.pace", 100.0, "goals per match", "x"),
                                  Input("home.drtg", 110.0, "points per 100 possessions", "x"),
                                  Input("home.xg_for", 1.0, "xG per match", "x")], BASKETBALL_REGISTRY)
    assert [i.name for i in kept] == ["home.drtg"] and len(dropped) == 3


def test_missing_required_basketball_inputs_make_the_fixture_insufficient():
    stats = assess(match_id(), [i for i in inputs() if i.name != "away.pace"])
    assert stats.status == INSUFFICIENT and "away.pace" in stats.reasons[0]
    assert assess(match_id(), inputs()).status == READY


def test_first_half_and_league_pace_are_optional_inputs():
    required = missing_required([], BASKETBALL_REGISTRY)
    assert "home.fh_points_for" not in required and "match.league_pace" not in required and "home.ortg" in required


def test_analyst_values_fill_missing_inputs_but_never_beat_a_free_source():
    partial = [i for i in inputs() if not i.name.endswith("ortg")]
    stats = assess(match_id(), partial)
    assert stats.status == INSUFFICIENT
    analyst = [Input("home.ortg", 110.0, "points per 100 possessions", "analyst-sourced:example.com"),
               Input("away.ortg", 109.0, "points per 100 possessions", "analyst-sourced:example.com"),
               Input("home.pace", 60.0, "possessions per game", "analyst-sourced:example.com")]
    filled = merge_analyst(stats, analyst)
    assert filled.status == READY
    by_name = {i.name: i for i in filled.inputs}
    assert by_name["home.ortg"].source == "analyst-sourced:example.com"
    assert by_name["home.pace"].source == "test-source" and by_name["home.pace"].value == 100.0  # free source wins


# ------------------------------------------------------------------ mapping picks to the SportyBet betslip
def _leg(market, selection, sport="basketball"):
    from predictions.core.model import Evaluation
    return leg_from_evaluation(Evaluation(match_id(sport), market, selection, D("1.90"), Estimate(0.7), D("0.3"), True, "passed"))


@pytest.mark.parametrize("market,selection,market_id,specifier,side", [
    ("Total points 220.5", "Over 220.5", "225", "total=220.5", "over"),
    ("Total points 220", "Under 220", "225", "total=220", "under"),
    ("Home team total points 110.5", "Home team Over 110.5", "227", "total=110.5", "over"),
    ("Away team total points 108.5", "Away team Under 108.5", "228", "total=108.5", "under"),
    ("1st half total points 108.5", "1st half Over 108.5", "68", "total=108.5", "over"),
    ("Winner (incl. OT)", "Home", "219", "", "home"),
    ("Winner (incl. OT)", "Away", "219", "", "away"),
    ("Handicap -6.5", "Home (-6.5)", "223", "hcp=-6.5", "home"),
    ("Handicap -6.5", "Away (+6.5)", "223", "hcp=-6.5", "away"),
    ("Handicap +3.5", "Home (+3.5)", "223", "hcp=3.5", "home"),
])
def test_basketball_picks_map_to_the_sportybet_market_ids(market, selection, market_id, specifier, side):
    leg = _leg(market, selection)
    assert (leg.market_id, leg.specifier, leg.side) == (market_id, specifier, side)


@pytest.mark.parametrize("market,selection", [
    ("Total points 220.5", "Over 221.5"),
    ("Handicap -6.5", "Away (-6.5)"),
    ("Home team total points 110.5", "Away team Over 110.5"),
    ("Corners 9.5", "Corners Over 9.5"),
])
def test_a_basketball_pick_that_does_not_match_its_market_is_unsupported(market, selection):
    with pytest.raises(UnsupportedLeg):
        _leg(market, selection)


def test_soccer_mapping_is_unchanged_and_line_keys_are_told_apart():
    assert _leg("Over/Under 2.5", "Over 2.5", "soccer").market_id == "18"
    assert betting.specifiers_equal("hcp=-6.5", "hcp=-6.50") and not betting.specifiers_equal("hcp=6.5", "total=6.5")
    assert betting.normalized_specifier("hcp=-6.50") == "hcp=-6.5" and betting.normalized_specifier("total=220.0") == "total=220"


def test_every_evaluation_the_engine_can_return_can_be_put_on_the_betslip():
    strong, weak = dict(ortg=112.0, fh_for=60.0, fh_against=54.0), dict(ortg=104.0, fh_for=52.0, fh_against=52.0)
    hcp = [MarketOffer(bb.HANDICAP_MARKET_ID, "Handicap", "hcp=-3.5",
                       (Outcome("Home (-3.5)", D("1.90")), Outcome("Away (+3.5)", D("1.90"))))]
    win = [MarketOffer(bb.WINNER_MARKET_ID, "Winner", "", (Outcome("Home", D("1.5")), Outcome("Away", D("2.5"))))]
    evals = run(inputs(strong, weak, 95.0), totals() + team_totals("home") + team_totals("away") + fh_totals() + hcp + win)
    assert evals
    for e in evals:
        leg = leg_from_evaluation(e)  # raises UnsupportedLeg if the wording were not recognised
        assert leg.match_id == e.match_id


def test_the_closest_option_works_for_basketball_lines():
    weak = dict(ortg=105.0, drtg=105.0)
    fixture = make_fixture(totals(("188.5", "190.5", "196.5"), over_odds="1.60"))
    candidates = ENGINES["basketball"].evaluate_exact(fixture, inputs(weak, weak), fixture.markets[0])
    candidates += ENGINES["basketball"].evaluate_exact(fixture, inputs(weak, weak), fixture.markets[1])
    candidates += ENGINES["basketball"].evaluate_exact(fixture, inputs(weak, weak), fixture.markets[2])
    original = _leg("Total points 192.5", "Over 192.5")
    chosen = closest_option(original, candidates)
    assert chosen is not None and chosen.selection in ("Over 190.5", "Over 196.5") and chosen.passed
    assert chosen.selection == "Over 190.5"  # nearer to 192.5 than 196.5
