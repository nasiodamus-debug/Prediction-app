"""Story 6.3: the MLB engine (model, markets, whole-line push maths, F5 team-total step-down), its input registry, its
market-to-betslip mapping and the closest-option rule for MLB lines. No test makes a live call."""
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from predictions.adapters.sportybet.fixtures import parse_page
from predictions.core.betting import UnsupportedLeg, leg_from_evaluation
from predictions.core.engines import ENGINES
from predictions.core.engines import mlb
from predictions.core.engines.mlb import MlbEngine
from predictions.core.inputs import BASEBALL_REGISTRY, Input, check_inputs
from predictions.core.model import PROBABILITY_FLOOR
from predictions.core.options import closest_option
from predictions.core.types import Fixture, MarketOffer, MatchId, Outcome

D = Decimal
KICKOFF = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)
DATA = Path(__file__).resolve().parent / "data"


def match_id():
    return MatchId("baseball", "USA - MLB", KICKOFF, "Home Team", "Away Team")


def make_fixture(offers=()):
    return Fixture(match_id(), "USA - MLB", "Home Team", "Away Team", KICKOFF, False, tuple(offers))


def side_inputs(side, rf=4.5, ra=4.5, f5_for=2.4, f5_against=2.4, sp=4.5, starts=20.0):
    src = "test-source"
    out = [Input(f"{side}.matches", 10.0, "games", src),
           Input(f"{side}.runs_for", rf, "runs per game", src),
           Input(f"{side}.runs_against", ra, "runs per game", src)]
    if f5_for is not None:
        out += [Input(f"{side}.f5_runs_for", f5_for, "runs per 5 innings", src),
                Input(f"{side}.f5_runs_against", f5_against, "runs per 5 innings", src)]
    if sp is not None:
        out += [Input(f"{side}.sp_runs_per9", sp, "runs allowed per 9 innings", src),
                Input(f"{side}.sp_starts", starts, "starts", src)]
    return out


def inputs(home=None, away=None, league=4.5, league5=2.4):
    found = side_inputs("home", **(home or {})) + side_inputs("away", **(away or {}))
    if league is not None:
        found.append(Input("match.league_runs", league, "runs per team per game", "test-source"))
    if league5 is not None:
        found.append(Input("match.league_f5_runs", league5, "runs per team per 5 innings", "test-source"))
    return found


def line_offers(market_id, name, key, lines, over="1.90", under="1.90"):
    return [MarketOffer(market_id, name, f"{key}={line}",
                        (Outcome(f"Over {line}", D(over)), Outcome(f"Under {line}", D(under)))) for line in lines]


def totals(lines=("6.5", "7.5", "8.5", "9.5", "10.5"), **kw):
    return line_offers(mlb.TOTAL_MARKET_ID, "Total (incl. extra innings)", "total", lines, **kw)


def team_totals(side, lines=("2.5", "3.5", "4.5"), **kw):
    mid = mlb.HOME_TOTAL_MARKET_ID if side == "home" else mlb.AWAY_TOTAL_MARKET_ID
    return line_offers(mid, f"Competitor{1 if side == 'home' else 2} total", "total", lines, **kw)


def f5_totals(lines=("3", "3.5", "4.5"), **kw):
    return line_offers(mlb.F5_TOTAL_MARKET_ID, "Innings 1 to 5 - total", "total", lines, **kw)


def f5_team_totals(lines=("1.5", "2.5"), over="1.80", under="1.90", side="Competitor1"):
    """SportyBet NG does not list F5 team totals (probed); this is a FABRICATED offer to exercise the rule."""
    return line_offers("999", f"Innings 1 to 5 - {side} total", "total", lines, over=over, under=under)


def handicaps(market_id, name, lines, home="1.90", away="1.90"):
    return [MarketOffer(market_id, name, f"hcp={line}",
                        (Outcome(f"Home ({D(line):+})", D(home)), Outcome(f"Away ({-D(line):+})", D(away))))
            for line in lines]


def run(inps, offers):
    fixture = make_fixture(offers)
    return ENGINES["baseball"].evaluate(fixture, inps, fixture.markets)


def by(evals, text):
    return [e for e in evals if text in e.market or text in e.selection]


# a strong scoring environment: the game's Over prices well above the floor
HOT = dict(rf=5.2, ra=5.0, f5_for=2.9, f5_against=2.8)


# ------------------------------------------------------------------ registry, constants, determinism
def test_baseball_is_registered_and_the_floor_is_imported_not_redefined():
    assert isinstance(ENGINES["baseball"], MlbEngine)
    source = open(mlb.__file__, encoding="utf-8").read()
    assert "PROBABILITY_FLOOR =" not in source and "0.65" not in source.replace("0.65 ", "")


def test_thresholds_are_named_constants():
    assert mlb.F5_STEP_MAX == 0.72 and mlb.F5_STEP_LINE == D("3.5") and mlb.RUN_DISPERSION > 1


def test_the_estimate_is_deterministic_and_missing_required_input_prices_nothing():
    offers = totals() + team_totals("home") + f5_totals()
    a = [(e.market, e.selection, e.estimate.probability, e.ev) for e in run(inputs(HOT, HOT), offers)]
    b = [(e.market, e.selection, e.estimate.probability, e.ev) for e in run(inputs(HOT, HOT), offers)]
    assert a == b and a
    assert run([i for i in inputs() if i.name != "home.runs_for"], totals()) == []


def test_the_registry_discards_out_of_range_values_and_wrong_units():
    kept, dropped = check_inputs([Input("home.runs_for", 40.0, "runs per game", "x"),
                                  Input("home.runs_against", 4.0, "goals per game", "x"),
                                  Input("home.runs_for", 4.0, "runs per game", "x")], BASEBALL_REGISTRY)
    assert [i.name for i in kept] == ["home.runs_for"] and len(dropped) == 2


# ------------------------------------------------------------------ pricing: totals, team totals, winner, run line
def test_hot_offence_makes_a_high_over_pass_and_shows_inputs_with_sources():
    evals = run(inputs(HOT, HOT), totals())
    over = by(evals, "Over")
    assert over and over[0].passed and over[0].estimate.probability >= PROBABILITY_FLOOR and over[0].ev > 0
    assert all(i.source == "test-source" for i in over[0].estimate.inputs) and over[0].estimate.inputs
    assert "expected runs" in " ".join(over[0].notes)


def test_a_full_game_total_shows_one_over_line_and_no_under():
    evals = run(inputs(HOT, HOT), totals())
    assert len(by(evals, "Over")) == 1 and len(by(evals, "Under")) == 0   # owner rule: no Under (AD-20)


def test_positive_ev_is_required_even_above_the_floor():
    evals = run(inputs(HOT, HOT), totals(lines=("6.5",), over="1.05", under="1.05"))
    over = by(evals, "Over")[0]
    assert over.estimate.probability > PROBABILITY_FLOOR and not over.passed and "expected value" in over.reason


def test_team_totals_use_the_right_team():
    strong_home = dict(rf=6.5, ra=4.5)
    evals = run(inputs(strong_home, None), team_totals("home", ("4.5",)) + team_totals("away", ("4.5",)))
    home, away = by(evals, "Home team Over")[0], by(evals, "Away team Over")[0]
    assert home.estimate.probability > away.estimate.probability


def test_the_starting_pitcher_moves_the_estimate_and_a_starter_without_a_sample_is_ignored():
    ace = run(inputs(None, dict(sp=2.5)), totals(("8.5",)))
    plain = run(inputs(None, None), totals(("8.5",)))
    thin = run(inputs(None, dict(sp=2.5, starts=2.0)), totals(("8.5",)))
    p = lambda evals: by(evals, "Over")[0].estimate.probability  # noqa: E731
    assert p(ace) < p(plain) and p(thin) == p(plain)


def test_moneyline_and_run_line_are_priced_by_side():
    strong = dict(rf=5.6, ra=3.4, f5_for=3.0, f5_against=1.9, sp=3.0)
    offers = [MarketOffer(mlb.WINNER_MARKET_ID, "Winner (incl. extra innings)", "",
                          (Outcome("Home", D("1.50")), Outcome("Away", D("2.60"))))]
    offers += handicaps(mlb.HANDICAP_MARKET_ID, "Handicap (incl. extra innings)", ("-1.5", "1.5"), "2.10", "1.70")
    evals = run(inputs(strong, None), offers)
    winner = [e for e in evals if e.market == mlb.WINNER_MARKET_TEXT]
    home, away = (next(e for e in winner if e.selection == s) for s in ("Home", "Away"))
    assert abs(home.estimate.probability + away.estimate.probability - 1.0) < 0.003 and home.estimate.probability > 0.6
    run_line = by(evals, "Run line")
    assert sorted(e.selection.split()[0] for e in run_line) == ["Away", "Home"]  # one line per side is shown


def test_no_starter_or_f5_data_makes_f5_markets_insufficient_while_full_game_is_priced():
    thin = dict(f5_for=None, sp=None)
    evals = run(inputs(thin, thin, league5=None), totals(("7.5",)) + f5_totals(("3.5",)))
    f5 = [e for e in evals if e.market.startswith("F5")]
    assert f5 and all(e.insufficient and not e.passed and e.reason.startswith("insufficient data") for e in f5)
    assert any(e.market.startswith("Total runs") and not e.insufficient for e in evals)


def test_an_f5_whole_line_is_not_offered_but_review_prices_it_with_the_push_formula():
    offers = f5_totals(("3",), over="1.70", under="2.00")
    assert run(inputs(HOT, HOT), offers) == []   # AD-21: no whole line is offered
    fixture = make_fixture(offers)
    evals = ENGINES["baseball"].evaluate_exact(fixture, inputs(HOT, HOT), offers[0])
    over = by(evals, "F5 Over 3")[0]
    est = over.estimate
    assert est.push > 0.05 and abs(est.probability - (est.win + est.push)) < 1e-9
    assert abs(float(over.ev) - (est.win * (1.70 - 1) - (1 - est.win - est.push))) < 0.002
    assert "refund" in " ".join(over.notes)


def test_f5_result_prices_the_draw_as_a_real_outcome():
    offers = [MarketOffer(mlb.F5_RESULT_MARKET_ID, "Innings 1 to 5 - 1x2", "",
                          (Outcome("Home", D("2.10")), Outcome("Draw", D("4.20")), Outcome("Away", D("2.40"))))]
    evals = run(inputs(), offers)
    probs = {e.selection: e.estimate.probability for e in evals}
    assert set(probs) == {"F5 Home", "F5 Draw", "F5 Away"} and abs(sum(probs.values()) - 1.0) < 0.004
    assert 0.1 < probs["F5 Draw"] < 0.25


def test_a_period_qualified_or_unknown_market_is_not_understood():
    engine = ENGINES["baseball"]
    assert not engine.understands(MarketOffer("999", "Winner of inning 1", "", (Outcome("Home", D("2")),)))
    assert engine.understands(totals()[0])


# ------------------------------------------------------------------ the F5 team-total step-down
def stepdown(home, away, offers=None):
    offers = offers if offers is not None else f5_team_totals() + f5_totals()
    return run(inputs(home, away), offers)


IN_ZONE = ({"f5_for": 2.75, "f5_against": 2.6}, {"f5_for": 2.75, "f5_against": 2.6})   # team Over 1.5 at 71.1%: inside 70% to 72%


def test_an_f5_team_total_over_between_70_and_72_percent_steps_down_to_combined_f5_over_3_5():
    evals = stepdown(*IN_ZONE)
    converted = [e for e in evals if e.rule]
    assert len(converted) == 1
    e = converted[0]
    assert e.selection == "F5 Over 3.5" and e.market == "F5 total runs 3.5"
    assert e.passed and e.estimate.probability >= PROBABILITY_FLOOR and e.ev > 0
    assert "F5 Home team Over 1.5" in e.original and "71.1%" in e.original  # the original pick is shown with its estimate
    assert "F5 step-down" in e.rule and "Combined F5 Game Total Over 3.5" in e.rule
    assert not [x for x in evals if "team Over" in x.selection]  # the team total itself is not offered any more


def test_a_converted_pick_below_the_floor_is_rejected_and_shows_both_the_original_and_the_rule():
    home = {"f5_for": 2.7, "f5_against": 2.4}
    away = {"f5_for": 0.9, "f5_against": 2.6}   # a strong home F5 offence against a feeble opponent: combined 3.5 is a coin flip
    evals = stepdown(home, away)
    e = next(x for x in evals if x.rule)
    assert e.selection == "F5 Over 3.5" and not e.passed
    assert e.estimate.probability < PROBABILITY_FLOOR and "below the 70% floor" in e.reason
    assert "F5 Home team Over 1.5" in e.original and "70.3%" in e.original


def test_a_converted_pick_needs_positive_ev_too():
    home = dict(HOT, f5_for=2.75, f5_against=2.6)
    evals = stepdown(home, home, f5_team_totals() + f5_totals(("3.5",), over="1.10", under="1.10"))
    e = next(x for x in evals if x.rule)
    assert e.estimate.probability >= PROBABILITY_FLOOR and not e.passed and "expected value" in e.reason


def test_an_f5_team_total_above_72_percent_is_not_stepped_down():
    hot = {"f5_for": 3.3, "f5_against": 3.0}
    evals = stepdown(hot, hot)
    team = [e for e in evals if "team Over" in e.selection]
    assert team and team[0].passed and team[0].estimate.probability > mlb.F5_STEP_MAX and not team[0].rule
    assert not [e for e in evals if e.rule]


def test_an_f5_team_total_below_the_floor_is_not_stepped_down_it_is_rejected_as_is():
    weak = {"f5_for": 1.9, "f5_against": 1.9}
    evals = stepdown(weak, weak)
    team = [e for e in evals if "team Over" in e.selection]
    assert team and not team[0].passed and not team[0].rule and team[0].estimate.probability < PROBABILITY_FLOOR


def test_when_the_step_down_target_is_not_offered_the_original_is_not_played():
    evals = stepdown(*IN_ZONE, offers=f5_team_totals() + f5_totals(("3", "4.5")))
    e = next(x for x in evals if x.rule)
    assert not e.passed and "not offered" in e.reason and "F5 Home team Over 1.5" in e.original


def test_the_away_side_and_competitor_names_are_understood():
    away_zone = ({"f5_for": 2.9, "f5_against": 2.6}, {"f5_for": 2.9, "f5_against": 2.6})   # the away team Over 1.5 lands in 70% to 72%
    evals = stepdown(*away_zone, offers=f5_team_totals(side="Competitor2") + f5_totals())
    assert "F5 Away team Over 1.5" in next(x for x in evals if x.rule).original


# ------------------------------------------------------------------ the real listing sample
def sample_fixtures():
    payload = json.loads((DATA / "sportybet_mlb_sample.json").read_text(encoding="utf-8"))
    return parse_page(payload, "baseball")[0]


def test_the_recorded_mlb_listing_prices_end_to_end_and_every_pick_maps_to_a_betslip_leg():
    fixture = sample_fixtures()[0]
    assert fixture.match_id.sport == "baseball" and fixture.competition == "USA - MLB"
    evals = ENGINES["baseball"].evaluate(fixture, inputs(HOT, HOT), fixture.markets)
    assert evals
    ids = {m.market_id for m in fixture.markets}
    assert ids <= set(mlb.MARKET_KINDS) and {"251", "258", "276"} <= ids
    for e in evals:
        leg = leg_from_evaluation(e)
        assert leg.match_id == e.match_id and leg.market_id in mlb.MARKET_KINDS


def test_legs_carry_the_market_id_specifier_and_side_sportybet_uses():
    fixture = sample_fixtures()[0]
    evals = ENGINES["baseball"].evaluate(fixture, inputs(HOT, HOT), fixture.markets)
    total = next(e for e in evals if e.market.startswith("Total runs"))
    leg = leg_from_evaluation(total)
    assert leg.market_id == "258" and leg.side == "over" and leg.specifier.startswith("total=")
    f5 = next(e for e in evals if e.market.startswith("F5 total runs"))
    assert leg_from_evaluation(f5).market_id == "276"
    with pytest.raises(UnsupportedLeg):
        leg_from_evaluation(type(total)(**{**total.__dict__, "market": "Total runs 8.5x"}))


def test_the_closest_option_rule_knows_mlb_lines():
    original = leg_from_evaluation(by(run(inputs(HOT, HOT), totals(("8.5",))), "Over")[0])
    near = by(run(inputs(HOT, HOT), totals(("7.5",))), "Over")[0]
    far = by(run(inputs(HOT, HOT), totals(("6.5",))), "Over")[0]
    assert near.passed and far.passed
    assert closest_option(original, [far, near]) is near   # the nearest line that clears the floor with positive EV
    weak = by(run(inputs(HOT, HOT), totals(("7.5",), over="1.02", under="1.02")), "Over")[0]
    assert not weak.passed and closest_option(original, [weak]) is None
