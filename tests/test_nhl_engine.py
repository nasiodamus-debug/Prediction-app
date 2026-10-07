"""Stories 6.4 and 6.5: the NHL engine (model, markets, whole-line push maths), the puck-line shield, the first-period
route, the xGF team-total lock (threshold tested at exactly 3.40 and 2.20), its registry and betslip mapping.
Expected goals are a PROXY (adapters/nhl/stats.py); the engine treats the inputs as given. No live call."""
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from predictions.adapters.sportybet.fixtures import parse_page
from predictions.core.betting import UnsupportedLeg, leg_from_evaluation
from predictions.core.engines import ENGINES
from predictions.core.market_rules import is_whole_or_quarter_line
from predictions.core.engines import nhl
from predictions.core.engines.nhl import NhlEngine
from predictions.core.inputs import HOCKEY_REGISTRY, Input
from predictions.core.model import PROBABILITY_FLOOR
from predictions.core.options import closest_option
from predictions.core.types import Fixture, MarketOffer, MatchId, Outcome

D = Decimal
KICKOFF = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)
DATA = Path(__file__).resolve().parent / "data"


def make_fixture(offers=()):
    mid = MatchId("hockey", "USA - NHL", KICKOFF, "Home Team", "Away Team")
    return Fixture(mid, "USA - NHL", "Home Team", "Away Team", KICKOFF, False, tuple(offers))


def side_inputs(side, gf=3.0, ga=3.0, xf=3.0, xa=3.0, p1gf=0.9, p1ga=0.9, p1xf=0.95, p1xa=0.95, p1=True):
    src = "test-source"
    xg = "expected goals per game (proxy)"
    out = [Input(f"{side}.matches", 10.0, "games", src),
           Input(f"{side}.goals_for", gf, "goals per game", src), Input(f"{side}.goals_against", ga, "goals per game", src),
           Input(f"{side}.xg_for", xf, xg, src), Input(f"{side}.xg_against", xa, xg, src)]
    if p1:
        p1xg = "1st-period expected goals per game (proxy)"
        out += [Input(f"{side}.p1_goals_for", p1gf, "1st-period goals per game", src),
                Input(f"{side}.p1_goals_against", p1ga, "1st-period goals per game", src),
                Input(f"{side}.p1_xg_for", p1xf, p1xg, src), Input(f"{side}.p1_xg_against", p1xa, p1xg, src)]
    return out


def inputs(home=None, away=None, league=3.0):
    found = side_inputs("home", **(home or {})) + side_inputs("away", **(away or {}))
    if league is not None:
        found.append(Input("match.league_goals", league, "goals per team per game", "test-source"))
    return found


def line_offers(market_id, name, key, lines, over="1.90", under="1.90", extra=""):
    return [MarketOffer(market_id, name, f"{key}={line}{extra}",
                        (Outcome(f"Over {line}", D(over)), Outcome(f"Under {line}", D(under)))) for line in lines]


def totals(lines=("5", "5.5", "6.5"), **kw):
    return line_offers(nhl.TOTAL_OT_MARKET_ID, "Total (OT and P)", "total", lines, **kw)


def team_totals(side="home", lines=("1.5", "2.5", "3", "3.5"), **kw):
    mid = nhl.HOME_TOTAL_MARKET_ID if side == "home" else nhl.AWAY_TOTAL_MARKET_ID
    return line_offers(mid, f"Competitor{1 if side == 'home' else 2} total", "total", lines, **kw)


def period1(lines=("0.5", "1.5", "2.5"), **kw):
    return line_offers(nhl.PERIOD_TOTAL_MARKET_ID, "Xth period - total", "total", lines, extra="|periodnr=1", **kw)


def puck(lines, home="1.90", away="1.90", market_id=nhl.HANDICAP_OT_MARKET_ID, name="Handicap 2 (incl. OT and pen.)"):
    return [MarketOffer(market_id, name, f"hcp={line}",
                        (Outcome(f"Home ({D(line):+})", D(home)), Outcome(f"Away ({-D(line):+})", D(away))))
            for line in lines]


def run(inps, offers):
    fixture = make_fixture(offers)
    return ENGINES["hockey"].evaluate(fixture, inps, fixture.markets)


def by(evals, text):
    return [e for e in evals if text in e.market or text in e.selection]


def ruled(evals, name):
    return [e for e in evals if e.rule.startswith(name)]


HEAVY_HOME = dict(gf=3.9, ga=2.2, xf=3.3, xa=2.4)
WEAK_AWAY = dict(gf=2.4, ga=3.4, xf=2.4, xa=3.3)


# ------------------------------------------------------------------ registry, constants, basics
def test_hockey_is_registered_and_the_floor_is_imported_not_redefined():
    assert isinstance(ENGINES["hockey"], NhlEngine)
    source = open(nhl.__file__, encoding="utf-8").read()
    assert "PROBABILITY_FLOOR =" not in source and "0.65" not in source.replace("0.65 ", "")


def test_thresholds_are_named_constants_with_the_specified_values():
    assert nhl.XGF_LOCK_THRESHOLD == 3.40 and nhl.TEAM_TOTAL_LOCK_LINE == D("3")
    assert nhl.P1_COMBINED_XG_THRESHOLD == 2.20 and nhl.P1_ROUTE_LINE == D("1.5")
    assert nhl.PUCK_LINE == D("-1.5") and nhl.UNDERDOG_LINE == D("1.5")
    assert not hasattr(nhl, "ASIAN_PUCK_LINE") and not hasattr(nhl, "TEAM_TOTAL_LOCK_FALLBACK_LINE")   # no whole line anywhere (AD-21)


def test_the_estimate_is_deterministic_and_missing_required_input_prices_nothing():
    offers = totals() + team_totals() + puck(("-1.5", "1.5"))
    a = [(e.market, e.selection, e.estimate.probability, e.ev) for e in run(inputs(), offers)]
    assert a == [(e.market, e.selection, e.estimate.probability, e.ev) for e in run(inputs(), offers)] and a
    assert run([i for i in inputs() if i.name != "home.xg_for"], totals()) == []


def test_the_registry_labels_expected_goals_as_a_proxy():
    assert "proxy" in HOCKEY_REGISTRY["xg_for"].unit and "proxy" in HOCKEY_REGISTRY["p1_xg_against"].unit


# ------------------------------------------------------------------ pricing (Story 6.4)
def test_full_game_total_team_total_and_winner_are_priced_with_sources_shown():
    winner = [MarketOffer(nhl.WINNER_MARKET_ID, "Winner (incl. OT and pen.)", "",
                          (Outcome("Home", D("1.80")), Outcome("Away", D("2.10"))))]
    evals = run(inputs(), totals() + team_totals() + winner)
    total = by(evals, "Total goals")
    assert total and all(e.estimate.inputs and all(i.source == "test-source" for i in e.estimate.inputs) for e in total)
    win = {e.selection: e.estimate.probability for e in evals if e.market == nhl.WINNER_MARKET_TEXT}
    assert abs(sum(win.values()) - 1.0) < 0.003 and 0.5 < win["Home"] < 0.56  # home ice and overtime lean, nothing more
    assert by(evals, "Home team total goals")


def test_one_over_line_is_shown_per_market_and_no_under():
    evals = run(inputs(), totals(("5.5", "6.5")))
    assert len(by(evals, "Over")) == 1 and len(by(evals, "Under")) == 0   # owner rule: no Under (AD-20)


def test_positive_ev_is_required():
    e = by(run(inputs(), totals(("4.5",), over="1.10", under="1.10")), "Over")[0]
    assert e.estimate.probability > PROBABILITY_FLOOR and not e.passed and "expected value" in e.reason


def test_a_whole_line_total_is_not_offered_but_review_prices_it_with_the_push_formula():
    offers = totals(("5",), over="1.80", under="2.00")
    assert run(inputs(), offers) == []   # AD-21: no whole line is offered
    e = by(ENGINES["hockey"].evaluate_exact(make_fixture(offers), inputs(), offers[0]), "Over 5")[0]
    est = e.estimate
    assert est.push > 0.1 and abs(est.probability - (est.win + est.push)) < 1e-9
    assert abs(float(e.ev) - (est.win * 0.80 - (1 - est.win - est.push))) < 0.002


def test_a_three_way_whole_line_handicap_is_never_priced_as_a_refund_line():
    three_way = MarketOffer(nhl.HANDICAP_REG_MARKET_ID, "Handicap 3 way", "hcp=-1",
                            (Outcome("Home (-1)", D("3.0")), Outcome("Draw", D("4.0")), Outcome("Away (+1)", D("1.9"))))
    two_way = puck(("-1.5",), market_id=nhl.HANDICAP_REG_MARKET_ID, name="Handicap 3 way")
    evals = run(inputs(), [three_way] + two_way)
    assert not [e for e in evals if e.market.endswith(" -1")] and by(evals, "Handicap (regulation)")


def test_missing_first_period_data_makes_period_picks_insufficient_while_the_game_is_priced():
    evals = run(inputs(dict(p1=False), dict(p1=False)), totals(("5.5",)) + period1())
    p1 = [e for e in evals if e.market.startswith("1st period")]
    assert p1 and all(e.insufficient and not e.passed and e.reason.startswith("insufficient data") for e in p1)
    assert by(evals, "Total goals")


def test_only_the_first_period_is_priced():
    second = MarketOffer(nhl.PERIOD_TOTAL_MARKET_ID, "Xth period - total", "total=1.5|periodnr=2",
                         (Outcome("Over 1.5", D("1.9")), Outcome("Under 1.5", D("1.9"))))
    assert not ENGINES["hockey"].understands(second) and run(inputs(), [second]) == []


# ------------------------------------------------------------------ Story 6.5: puck-line shield
def test_a_minus_1_5_puck_line_on_a_heavy_favourite_never_converts_to_the_asian_minus_1_0():
    offers = puck(("-1.5", "-1", "1.5"), "2.40", "1.60")   # a listed -1 (whole) line is not a play and not a shield target (AD-21)
    evals = run(inputs(HEAVY_HOME, WEAK_AWAY), offers)
    e = ruled(evals, "Puck line shield")
    assert len(e) == 1
    e = e[0]
    assert e.selection == "Away (+1.5)" and e.market == "Puck line (incl. OT) -1.5" and e.estimate.push == 0
    assert "Home (-1.5) @ 2.40" in e.original and "heavy favourite" in e.rule and "refund" not in e.rule and "Asian" not in e.rule
    assert not [x for x in evals if x.selection in ("Home (-1.5)", "Home (-1)")]
    assert not any(is_whole_or_quarter_line(x) for x in evals)


def test_a_whole_puck_line_is_priced_only_by_review_with_the_refund_maths():
    offer = MarketOffer(nhl.HANDICAP_OT_MARKET_ID, "Handicap 2 (incl. OT and pen.)", "hcp=-1",
                        (Outcome("Home (-1)", D("2.40")),))
    e = ENGINES["hockey"].evaluate_exact(make_fixture([offer]), inputs(HEAVY_HOME, WEAK_AWAY), offer)[0]
    est = e.estimate
    assert e.selection == "Home (-1)" and est.push > 0.1 and abs(est.probability - (est.win + est.push)) < 1e-9
    assert abs(float(e.ev) - (est.win * (2.40 - 1) - (1 - est.win - est.push))) < 0.002  # EV = P(win)*(odds-1) - P(lose)
    assert run(inputs(HEAVY_HOME, WEAK_AWAY), [offer]) == []


def test_the_floor_applies_to_win_plus_push_and_ev_uses_win_only():
    # Review only: a whole puck line at a short price: the estimate still clears the floor but EV is not positive
    offer = MarketOffer(nhl.HANDICAP_OT_MARKET_ID, "Handicap 2 (incl. OT and pen.)", "hcp=-1", (Outcome("Home (-1)", D("1.05")),))
    e = ENGINES["hockey"].evaluate_exact(make_fixture([offer]), inputs(HEAVY_HOME, WEAK_AWAY), offer)[0]
    assert e.estimate.probability >= PROBABILITY_FLOOR and not e.passed and "expected value" in e.reason


def test_without_a_minus_1_line_the_underdog_plus_1_5_is_the_shield_and_must_clear_the_floor():
    offers = puck(("-1.5", "1.5"), "2.40", "1.60")
    evals = run(inputs(HEAVY_HOME, WEAK_AWAY), offers)
    e = ruled(evals, "Puck line shield")[0]
    assert e.selection == "Away (+1.5)" and not e.passed and e.estimate.probability < PROBABILITY_FLOOR
    assert "Home (-1.5)" in e.original


def test_neither_shield_offered_means_the_original_is_not_played():
    only_home = [MarketOffer(nhl.HANDICAP_OT_MARKET_ID, "Handicap 2 (incl. OT and pen.)", "hcp=-1.5",
                             (Outcome("Home (-1.5)", D("2.40")),))]   # no underdog price and no -1 line on offer
    e = ruled(run(inputs(HEAVY_HOME, WEAK_AWAY), only_home), "Puck line shield")[0]
    assert not e.passed and "not offered" in e.reason and e.selection == "Home (-1.5)"


def test_a_favourite_below_the_heavy_threshold_or_a_different_line_is_plain_line_shopping():
    assert not ruled(run(inputs(), puck(("-1.5", "1.5"))), "Puck line shield")
    assert not ruled(run(inputs(HEAVY_HOME, WEAK_AWAY), puck(("-2.5", "2.5"))), "Puck line shield")


# ------------------------------------------------------------------ Story 6.5: first-period route
def p1_inputs(xg):
    return dict(p1xf=xg, p1xa=xg, p1gf=1.3, p1ga=1.3)


def test_combined_first_period_xg_above_2_20_routes_to_period_1_over_1_5_and_bypasses_full_game_totals():
    hot = p1_inputs(1.4)
    evals = run(inputs(hot, hot), period1(over="1.80", under="2.00") + totals(("5.5", "6.5")))
    routed = ruled(evals, "Period 1 route")
    assert len(routed) == 1 and routed[0].selection == "1st period Over 1.5" and routed[0].passed
    assert routed[0].estimate.probability >= PROBABILITY_FLOOR and "combined first-period expected goals 2.80" in routed[0].rule
    assert routed[0].original
    assert not by(evals, "Total goals") and not by(evals, "1st period Under")  # nothing else is offered for the total or P1


def test_the_first_period_threshold_is_strictly_above_2_20():
    at = p1_inputs(1.1)      # (1.1 + 1.1) / 2 per side = 1.10 each, combined exactly 2.20
    over = p1_inputs(1.11)   # combined 2.22
    offers = period1() + totals(("5.5", "6.5"))
    assert not ruled(run(inputs(at, at), offers), "Period 1 route") and by(run(inputs(at, at), offers), "Total goals")
    assert ruled(run(inputs(over, over), offers), "Period 1 route")


def test_a_routed_pick_below_the_floor_is_rejected_and_shows_the_original_and_the_rule():
    just_over = dict(p1xf=1.11, p1xa=1.11)   # combined 2.22: routed, but the first-period goals are ordinary
    evals = run(inputs(just_over, just_over), period1() + totals(("5.5", "6.5")))
    e = ruled(evals, "Period 1 route")[0]
    assert not e.passed and e.estimate.probability < PROBABILITY_FLOOR and "below the 70% floor" in e.reason
    assert e.original and "Period 1 route" in e.rule


def test_the_route_needs_first_period_data():
    evals = run(inputs(dict(p1=False), dict(p1=False)), period1() + totals(("5.5",)))
    assert not ruled(evals, "Period 1 route") and by(evals, "Total goals")


# ------------------------------------------------------------------ Story 6.5: expected-goals-for lock
def test_a_team_with_xgf_of_exactly_3_40_locks_the_nearest_listed_half_line_at_or_above_3():
    evals = run(inputs(dict(xf=3.4), {}), team_totals(lines=("2", "2.5", "3", "3.5", "4.5")) + totals())
    locked = ruled(evals, "xGF lock")
    assert len(locked) == 1 and locked[0].selection == "Home team Over 3.5" and "3.40" in locked[0].rule
    assert "nearest listed half line" in locked[0].rule
    assert not [e for e in evals if e.selection.startswith("Home team") and "Over 3" not in e.selection]
    assert not by(evals, "Total goals") and locked[0].original  # full-game totals are bypassed; the original is shown


def test_a_team_just_below_3_40_is_not_locked():
    evals = run(inputs(dict(xf=3.39), {}), team_totals(lines=("2", "2.5", "3", "3.5")) + totals())
    assert not ruled(evals, "xGF lock") and by(evals, "Total goals")


def test_the_floor_applies_to_the_converted_lock_pick():
    hot = dict(gf=4.2, ga=2.6, xf=3.6, xa=2.8)
    e = ruled(run(inputs(hot, {}), team_totals(lines=("3.5",), over="1.80")), "xGF lock")[0]
    assert e.estimate.push == 0 and e.selection == "Home team Over 3.5"
    below = ruled(run(inputs(dict(xf=3.4), {}), team_totals(lines=("3.5",))), "xGF lock")[0]
    assert below.estimate.probability < PROBABILITY_FLOOR and not below.passed and "below the 70% floor" in below.reason


def test_the_lock_is_dropped_when_only_a_whole_line_or_a_lower_half_line_is_listed():
    # Over 3.0 (whole) is gone and Over 2.5 is below the stated threshold: no conversion, and the full-game totals stay (AD-21)
    for lines in (("3",), ("2.5",), ("2", "2.5", "3")):
        evals = run(inputs(dict(xf=3.6, gf=3.9), {}), team_totals(lines=lines) + totals())
        assert not ruled(evals, "xGF lock") and by(evals, "Total goals")
        assert not any(is_whole_or_quarter_line(e) for e in evals)


def test_the_lock_moves_up_to_the_next_listed_half_line_when_3_5_is_not_listed():
    e = ruled(run(inputs(dict(xf=3.6, gf=3.9), {}), team_totals(lines=("2.5", "4.5"))), "xGF lock")[0]
    assert e.selection == "Home team Over 4.5" and "nearest listed half line at or above 3" in e.rule


def test_the_away_side_locks_too_and_only_its_own_team_total():
    evals = run(inputs({}, dict(xf=3.5)), team_totals("home") + team_totals("away") + totals())
    locked = ruled(evals, "xGF lock")
    assert len(locked) == 1 and locked[0].selection.startswith("Away team")
    assert by(evals, "Home team total goals")  # the other side is priced as usual


# ------------------------------------------------------------------ the real listing sample and the betslip mapping
def sample_fixtures():
    payload = json.loads((DATA / "sportybet_nhl_sample.json").read_text(encoding="utf-8"))
    return parse_page(payload, "hockey")[0]


def test_the_recorded_nhl_listing_prices_end_to_end_and_every_pick_maps_to_a_betslip_leg():
    fixture = sample_fixtures()[0]
    assert fixture.match_id.sport == "hockey" and fixture.competition == "USA - NHL"
    evals = ENGINES["hockey"].evaluate(fixture, inputs(HEAVY_HOME, WEAK_AWAY), fixture.markets)
    assert evals and {m.market_id for m in fixture.markets} <= set(nhl.MARKET_KINDS)
    for e in evals:
        leg = leg_from_evaluation(e)
        assert leg.match_id == e.match_id and leg.market_id in nhl.MARKET_KINDS
    assert any(e.rule for e in evals) and all("Asian" not in e.market for e in evals)


def test_legs_carry_the_specifier_and_side_sportybet_uses():
    fixture = sample_fixtures()[0]
    hot = dict(p1xf=1.4, p1xa=1.4, p1gf=1.3, p1ga=1.3)
    evals = ENGINES["hockey"].evaluate(fixture, inputs(hot, hot), fixture.markets)
    p1 = next(e for e in evals if e.market.startswith("1st period"))
    leg = leg_from_evaluation(p1)
    assert leg.market_id == "446" and leg.side == "over" and "periodnr=1" in leg.specifier and "total=1.5" in leg.specifier
    shield = ruled(run(inputs(HEAVY_HOME, WEAK_AWAY), puck(("-1.5", "1.5"), "2.4", "1.6")), "Puck line shield")[0]
    leg = leg_from_evaluation(shield)
    assert leg.market_id == "410" and leg.side == "away" and leg.specifier == "hcp=-1.5"
    with pytest.raises(UnsupportedLeg):
        leg_from_evaluation(type(p1)(**{**p1.__dict__, "market": "Hockey magic 3"}))


def test_the_closest_option_rule_knows_nhl_period_lines():
    hot = p1_inputs(1.4)
    original = leg_from_evaluation(ruled(run(inputs(hot, hot), period1(("1.5",))), "Period 1 route")[0])
    calm = dict(p1xf=1.0, p1xa=1.0, p1gf=1.3, p1ga=1.3)   # combined 2.00: no route, plain first-period pricing
    other = by(run(inputs(calm, calm), period1(("0.5",))), "Over")[0]
    assert other.passed and closest_option(original, [other]) is other
