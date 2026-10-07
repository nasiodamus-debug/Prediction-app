"""AD-21: the owner's rule "leave out whole-line then".

A whole-line or quarter-line play (any line that is not a half line x.5: a play that can end in a refund or a half refund) is never
produced, ranked, offered, converted to, kept, loaded or placed, in Matchday, Gameday or America. REVIEW is exempt: it judges whatever
is on the booking code with the push maths. Nothing here uses the network."""
import asyncio
import itertools
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

import test_basketball_engine as tb
import test_football_engine as tf
import test_gate as tg
import test_mlb_engine as tm
import test_nhl_engine as tn
import test_no_under as tnu
import test_soccer_optimization as tso
from fakes_bookmaker import FakeBookmaker, make_fixture, pick_evaluation, totals_offer
from predictions.adapters.sportybet.betslip import SportyBetBookmaker
from predictions.core import market_rules
from predictions.core.betting import Leg, leg_from_evaluation
from predictions.core.engines import ENGINES
from predictions.core.market_rules import (RULE_NO_WHOLE_LINE, WHOLE_LINE_REASON, is_whole_or_quarter_line,
                                           offers_without_whole_lines, reject_whole_line, without_whole_lines)
from predictions.core.model import Estimate, Evaluation
from predictions.core.options import closest_option
from predictions.core.ports import BookmakerError
from predictions.core.review import judge_selection
from predictions.core.slip import Pick, Slip, build_games, make_pick, make_pick_id
from predictions.core.stats import READY, FixtureStats
from predictions.core.types import MarketOffer, Outcome, Owner
from predictions.flow import evaluate as evaluate_flow
from predictions.flow.evaluate import GateResult, run_gate
from predictions.flow.loading import KIND_FAILED, alternative_for, run_load
from predictions.flow.session import Action, SessionManager, State, Status
from predictions.telegram.render import gate_lines, why_lines

D = Decimal
OWNER = Owner(111222333)
SID = "abcd1234"


# ------------------------------------------------------------------ the one definition
@pytest.mark.parametrize("value", [
    2, 2.0, 3.0, 2.25, 2.75, 0, 0.0, -1, -1.0, -3, 44, 44.0, 152, 1.0, 6.25, -0.25, D("2"), D("2.0"), D("2.25"), D("-1.0"), D("0"),
    "2", "2.0", "3.0", "2.25", "2.75", "-1.0", "-1", "0", "0.0", "+3", " 2.0 ", "2,0", "2.25",
    "total=2", "total=2.0", "total=2.25", "total=3|periodnr=1", "hcp=-1", "hcp=-1.0", "hcp=-3", "hcp=0", "hcp=+7",
    "Over 2.0", "Over 2", "Over 2.25", "Over 3.0", "Over 44", "Home (-1)", "Home (-3)", "Away (+7)", "Home (-1.0)",
    "Home (-1.25)", "Away (+0)", "Corners Over 9.0", "Cards Over 3", "1st half Over 24", "F5 Over 3", "Home team Over 3.0",
    "Handicap -3", "Puck line (incl. OT) -1", "Over/Under 2.0", "Total points 44", "1st period Over 1",
    "Under 2.0", "Total runs 8",
])
def test_whole_and_quarter_spellings_are_rejected(value):
    assert is_whole_or_quarter_line(value) is True


@pytest.mark.parametrize("value", [
    0.5, 1.5, 2.5, 3.5, 152.5, -1.5, -0.5, 7.5, D("2.5"), D("-3.5"), D("152.5"),
    "2.5", "1.5", "-1.5", "+2.5", "152.5", "0.5", "2,5",
    "total=2.5", "total=1.5|periodnr=1", "hcp=-1.5", "hcp=+3.5", "hcp=-0.5",
    "Over 2.5", "Over 1.5", "Home (-1.5)", "Away (+2.5)", "Home (-3.5)", "Corners Over 8.5", "Cards Over 3.5",
    "1st half Over 24.5", "F5 Over 4.5", "Home team Over 0.5", "1st period Over 1.5", "Over/Under 2.5", "Total points 44.5",
    "Home", "Away", "Draw", "Home or Draw", "Away or Draw", "Home or Away", "1X2", "Winner (incl. OT)", "F5 result (1X2)",
    "", "   ", None, True, "Over", "Winner (incl. extra innings)",
])
def test_half_lines_and_lineless_selections_are_fine(value):
    assert is_whole_or_quarter_line(value) is False


def test_it_reads_evaluations_codeselections_legs_picks_markets_and_outcomes():
    assert is_whole_or_quarter_line(pick_evaluation(market="Over/Under 2.0", selection="Over 2.0"))
    assert not is_whole_or_quarter_line(pick_evaluation())
    assert is_whole_or_quarter_line(pick_evaluation(market="Over/Under 2.5", selection="Over 2.0"))   # any readable field counts
    assert is_whole_or_quarter_line(tnu.code_selection("Over 2.0", specifier="total=2"))
    assert is_whole_or_quarter_line(tnu.code_selection("Over 2.5", specifier="total=2"))
    assert not is_whole_or_quarter_line(tnu.code_selection("Over 2.5"))
    # a bookmaker's market NAME with a number in it is not a line ("Innings 1 to 5"), the specifier is
    assert not is_whole_or_quarter_line(tnu.code_selection("Over 3.5", name="Innings 1 to 5 - total", specifier="total=3.5"))
    assert is_whole_or_quarter_line(leg_from_evaluation(pick_evaluation(market="Over/Under 2.0", selection="Over 2.0")))
    assert not is_whole_or_quarter_line(leg_from_evaluation(pick_evaluation()))
    pick = Pick("x", pick_evaluation(market="Over/Under 2.0", selection="Over 2.0"))
    assert is_whole_or_quarter_line(pick) and not is_whole_or_quarter_line(Pick("y", pick_evaluation()))
    assert is_whole_or_quarter_line(Outcome("Over 2.0", D("1.5"))) and not is_whole_or_quarter_line(Outcome("Home", D("1.5")))
    offer = tso.total_offer("2", "1.5", "2.5")
    assert is_whole_or_quarter_line(offer) and not is_whole_or_quarter_line(tso.total_offer("2.5", "1.5", "2.5"))
    assert not is_whole_or_quarter_line(tso.one_x_two("2.0", "3.0", "4.0"))


def test_the_helpers_reject_strip_and_filter_offers():
    whole = pick_evaluation(market="Over/Under 2.0", selection="Over 2.0")
    half = pick_evaluation()
    refused = reject_whole_line(whole)
    assert not refused.passed and refused.reason == WHOLE_LINE_REASON == "Whole-line plays are not allowed"
    assert refused.rule == RULE_NO_WHOLE_LINE and refused.estimate == whole.estimate
    assert without_whole_lines([whole, half]) == [half]
    offers = [tso.total_offer(x, "1.5", "2.5") for x in ("1.5", "2", "2.25", "2.5", "2.75", "3")] + [tso.one_x_two("2", "3", "4")]
    assert [o.specifier for o in offers_without_whole_lines(offers)] == ["total=1.5", "total=2.5", ""]


def test_the_market_rules_module_stays_a_standard_library_leaf():
    source = open(market_rules.__file__, encoding="utf-8").read()
    assert "import predictions" not in source and "from predictions" not in source


# ------------------------------------------------------------------ the property grid: every sport's engine, whole lines at good odds
GOOD = "2.60"   # a long price: a whole line with its refund share would be the best play of the game at these odds


def soccer_cases():
    for competition in (tso.MID, tso.TOP, "Testland - League"):
        for attack, defence in ((1.0, 1.0), (1.4, 1.2), (2.0, 1.0), (2.5, 1.5)):
            half = [tso.total_offer(x, "1.80", "2.00") for x in ("1.5", "2.5", "3.5")]
            extra = [tso.total_offer(x, GOOD, "1.30") for x in ("2", "3", "4", "2.25", "2.75", "1.75", "1.25", "1")]
            dc = MarketOffer("10", "Double Chance", "", (Outcome("Home or Draw", D("1.30")), Outcome("Draw or Away", D("1.60")),
                                                         Outcome("Home or Away", D("1.20"))))
            rest = [tso.one_x_two("2.60", "3.10", "2.80"), dc]
            inputs = tso.make_inputs(attack, defence, attack, defence)
            yield ("soccer", tso.fixture(half + extra + rest, competition=competition),
                   tso.fixture(half + rest, competition=competition), inputs)


def basketball_cases():
    for ortg, pace in ((95.0, 94.0), (105.0, 98.0), (112.0, 100.0), (118.0, 104.0)):
        side = dict(ortg=ortg, drtg=ortg, pace=pace, fh_for=ortg / 2, fh_against=ortg / 2)
        strong = dict(ortg=ortg + 8, drtg=ortg - 8, pace=pace, fh_for=ortg / 2, fh_against=ortg / 2)
        halves = (tb.totals(("214.5", "220.5", "224.5"), over_odds=GOOD) + tb.team_totals("home", ("104.5", "108.5"), over_odds=GOOD)
                  + tb.fh_totals(("100.5", "108.5"), over_odds=GOOD) + tb.handicap_offers("-9.5", "-5.5", "3.5"))
        extra = (tb.totals(("214", "220", "224", "219.25"), over_odds=GOOD) + tb.team_totals("away", ("104", "108"), over_odds=GOOD)
                 + tb.handicap_offers("-9", "-5", "3", "-2"))
        yield ("basketball", tb.make_fixture(halves + extra), tb.make_fixture(halves), tb.inputs(strong, side))


def baseball_cases():
    for rf, ra in ((3.0, 3.0), (4.5, 4.5), (5.4, 5.0)):
        side = dict(rf=rf, ra=ra, f5_for=rf / 2, f5_against=ra / 2)
        halves = (tm.totals(("6.5", "7.5", "8.5"), over=GOOD) + tm.team_totals("home", ("2.5", "3.5"), over=GOOD)
                  + tm.f5_totals(("3.5", "4.5"), over=GOOD) + tm.handicaps(tm.mlb.HANDICAP_MARKET_ID, "Handicap", ("-1.5", "1.5")))
        extra = (tm.totals(("7", "8", "9", "7.25"), over=GOOD) + tm.f5_totals(("3", "4"), over=GOOD)
                 + tm.handicaps(tm.mlb.HANDICAP_MARKET_ID, "Handicap", ("-1", "1", "-1.25")))
        yield ("baseball", tm.make_fixture(halves + extra), tm.make_fixture(halves), tm.inputs(side, side))


def hockey_cases():
    for gf, ga in ((2.0, 2.0), (3.0, 3.0), (3.6, 3.4), (4.0, 2.0)):
        side = dict(gf=gf, ga=ga, xf=gf, xa=ga, p1gf=gf / 3, p1ga=ga / 3, p1xf=gf / 3, p1xa=ga / 3)
        halves = (tn.totals(("5.5", "6.5"), over=GOOD) + tn.team_totals("home", ("1.5", "2.5", "3.5"), over=GOOD)
                  + tn.period1(("0.5", "1.5"), over=GOOD) + tn.puck(("-1.5", "1.5")))
        extra = (tn.totals(("5", "6", "5.25"), over=GOOD) + tn.team_totals("home", ("3", "2", "3.25"), over=GOOD)
                 + tn.period1(("1", "2"), over=GOOD) + tn.puck(("-1", "1", "-1.25")))
        yield ("hockey", tn.make_fixture(halves + extra), tn.make_fixture(halves), tn.inputs(side, side))


def football_cases():
    for competition in ("USA - NFL", "USA - NCAA, Division I"):
        for points in (13.0, 20.0, 24.0, 31.0):
            side = dict(pf=points, pa=points, fh_for=points / 2, fh_against=points / 2, q1_rank=3)
            halves = (tf.totals(("40.5", "44.5"), over=GOOD) + tf.team_totals("home", ("20.5", "23.5"), over=GOOD)
                      + tf.fh_totals(("20.5", "23.5", "24.5"), over=GOOD) + tf.moneyline()
                      + tf.spreads({"-3.5": ("1.9", "1.9"), "-2.5": ("1.9", "1.9"), "-7.5": ("1.9", "1.9"), "7.5": ("1.9", "1.9"),
                                    "2.5": ("1.9", "1.9"), "-9.5": ("1.9", "1.9"), "-6.5": ("1.9", "1.9")}))
            extra = (tf.totals(("40", "44", "44.25"), over=GOOD) + tf.fh_totals(("21", "24"), over=GOOD)
                     + tf.spreads({"-3": ("1.9", "1.9"), "-7": ("1.9", "1.9"), "3": ("1.9", "1.9"), "-6": ("1.9", "1.9"),
                                   "-3.25": ("1.9", "1.9")}))
            yield ("american_football", tf.make_fixture(halves + extra, competition), tf.make_fixture(halves, competition),
                   tf.inputs(side, side, wind=5.0))


def grid():
    return itertools.chain(soccer_cases(), basketball_cases(), baseball_cases(), hockey_cases(), football_cases())


def assert_plain(evaluations):
    for e in evaluations:
        assert not is_whole_or_quarter_line(e), (e.market, e.selection)
        assert not e.original or not is_whole_or_quarter_line(e.original.split(" @ ")[0]), e.original   # nor the pick it replaced


def test_property_no_engine_returns_a_whole_or_quarter_line_and_half_lines_are_unaffected():
    sports, results = set(), 0
    for sport, with_whole, half_only, inputs in grid():
        found = ENGINES[sport].evaluate(with_whole, inputs, with_whole.markets)
        assert_plain(found)
        assert found == ENGINES[sport].evaluate(half_only, inputs, half_only.markets)   # the half-line outcome is untouched
        sports.add(sport)
        results += len(found)
    assert sports == {"soccer", "basketball", "baseball", "hockey", "american_football"} and results > 150


def test_property_the_run_gate_passes_ranks_and_rejects_no_whole_line_on_the_grid():
    passed_total = 0
    for sport, with_whole, half_only, inputs in grid():
        stats = {with_whole.match_id: FixtureStats(with_whole.match_id, READY, tuple(inputs))}
        gate = run_gate([with_whole], stats)
        assert not [e for e in (*gate.passed, *gate.rejected) if is_whole_or_quarter_line(e)]
        assert not gate.whole_rejected
        half_stats = {half_only.match_id: FixtureStats(half_only.match_id, READY, tuple(inputs))}
        assert run_gate([half_only], half_stats).passed == gate.passed
        passed_total += len(gate.passed)
    assert passed_total > 20


def test_property_review_still_prices_every_whole_line_offer_on_the_grid():
    priced = 0
    for sport, with_whole, half_only, inputs in itertools.islice(grid(), 0, None, 3):
        engine = ENGINES[sport]
        for offer in with_whole.markets:
            if is_whole_or_quarter_line(offer):
                found = engine.evaluate_exact(with_whole, inputs, offer)
                priced += len(found)
                assert all(not e.insufficient or e.reason for e in found)
    assert priced > 50   # Review judges whole lines (the quarter lines of some sports have no model and stay unpriced)


# ------------------------------------------------------------------ the rules that used to land on a whole line
EVEN = tso.EVEN


def test_soccer_over_2_5_stays_over_2_5_and_no_shield_is_applied():
    for competition in (tso.MID, tso.TOP, "Testland - League"):
        f = tso.fixture([tso.total_offer("2.5", "1.90", "1.90"), tso.total_offer("2", "1.45", "2.90")], competition=competition)
        found = tso.by_selection(tso.evaluate(f, EVEN))
        assert list(found) == ["Over 2.5"] and found["Over 2.5"].odds == D("1.90") and not found["Over 2.5"].rule


def test_a_swap_lands_on_over_2_5_only_and_never_on_the_whole_line_2_0():
    f = tso.swap_fixture()
    found = tso.evaluate(f, tso.HOME_FAVOURITE)
    assert [e.selection for e in found if e.rule.startswith("1X2 swap")] == ["Over 2.5"]
    only_whole = tso.fixture([tso.one_x_two("2.10", "3.60", "3.40"), tso.total_offer("2", "1.40", "2.90")], competition=tso.TOP)
    home = tso.by_selection(tso.evaluate(only_whole, tso.HOME_FAVOURITE))["Home"]
    assert not home.passed and "no goal line is listed to convert to" in home.reason


def test_the_nhl_puck_line_and_lock_land_on_half_lines_or_drop():
    evals = tn.run(tn.inputs(tn.HEAVY_HOME, tn.WEAK_AWAY), tn.puck(("-1.5", "-1", "1.5"), "2.40", "1.60"))
    assert [e.selection for e in evals if e.rule.startswith("Puck line shield")] == ["Away (+1.5)"]
    lock = tn.run(tn.inputs(dict(xf=3.6, gf=3.9), {}), tn.team_totals(lines=("3", "3.5", "4")))
    assert [e.selection for e in lock if e.rule.startswith("xGF lock")] == ["Home team Over 3.5"]
    dropped = tn.run(tn.inputs(dict(xf=3.6, gf=3.9), {}), tn.team_totals(lines=("3", "4")) + tn.totals())
    assert not [e for e in dropped if e.rule.startswith("xGF lock")] and tn.by(dropped, "Total goals")
    assert not any(is_whole_or_quarter_line(e) for e in (*evals, *lock, *dropped))


def test_the_football_key_number_shield_has_no_buy_the_half_point_branch():
    inps = tf.nfl_fav("-3.5", 12.0)["inps"]
    only_whole = tf.away_spreads({"-3.5": ("1.50", "3.00"), "-3": ("1.55", "2.20"), "-7": ("1.5", "2.0")})
    away = [e for e in tf.run(inps, only_whole) if e.selection.startswith("Away (")]
    assert [e.selection for e in away] == ["Away (-3.5)"] and not away[0].passed and "not offered" in away[0].reason
    stepped = tf.away_spreads({"-3.5": ("1.50", "3.00"), "-3": ("1.55", "2.20"), "-2.5": ("1.60", "1.55")})
    assert [e.selection for e in tf.run(inps, stepped) if e.selection.startswith("Away (")] == ["Away (-2.5)"]
    underdog = [e.selection for e in tf.run(tf.nfl_fav("-3.5", 12.0)["inps"], tf.spreads({"2.5": ("1.9", "1.9"), "3": ("1.9", "1.9")}))]
    assert not any(is_whole_or_quarter_line(s) for s in underdog)


# ------------------------------------------------------------------ the central gate and the best-two ranking
class Injecting:
    """An engine that breaks the rule on purpose (and keeps soccer's best-two limit): the central gate must still catch it."""
    max_plays_per_game = 2

    def __init__(self, evaluations):
        self.evaluations = evaluations

    def screen(self, fixture, inputs):
        return None

    def evaluate(self, fixture, inputs, markets):
        return list(self.evaluations)


def _gate(monkeypatch, evaluations):
    f = tg.fixture(markets=[tg.totals("2.5", "1.90", "1.90")])
    monkeypatch.setitem(evaluate_flow.ENGINES, "soccer", Injecting(evaluations))
    stats = {f.match_id: FixtureStats(f.match_id, READY, tuple(tg.inputs_for(2.0, 1.0)))}
    return run_gate([f], stats)


def injected(selection, prob, odds="2.00", passed=True, market=None):
    market = market or f"Over/Under {selection.split()[-1]}"
    e = replace(pick_evaluation(selection=selection, market=market, odds=odds, prob=prob), match_id=tg.fixture().match_id, passed=passed)
    return e


def test_the_central_gate_refuses_an_injected_whole_line_before_the_best_two_cut(monkeypatch):
    whole = injected("Over 2.0", 0.95)
    quarter = injected("Over 2.25", 0.90)
    plain = [injected("Over 1.5", 0.80), injected("Over 2.5", 0.75), injected("Over 3.5", 0.70)]
    gate = _gate(monkeypatch, [whole, quarter, *plain])
    assert [e.selection for e in gate.passed] == ["Over 1.5", "Over 2.5"]   # the best two PLAIN plays, not pushed out
    assert sorted(e.selection for e in gate.whole_rejected) == ["Over 2.0", "Over 2.25"] and gate.whole_ignored == 2
    assert all(not e.passed and e.reason == WHOLE_LINE_REASON for e in gate.whole_rejected)
    assert not [e for e in gate.rejected if is_whole_or_quarter_line(e)]


def test_a_game_whose_only_passing_plays_were_whole_lines_shows_its_plain_plays_or_none(monkeypatch):
    gate = _gate(monkeypatch, [injected("Over 2.0", 0.95), injected("Over 3.0", 0.90), injected("Over 1.5", 0.60, passed=False)])
    assert gate.passed == () and [e.selection for e in gate.rejected] == ["Over 1.5"] and gate.whole_ignored == 2
    shown = _gate(monkeypatch, [injected("Over 2.0", 0.95), injected("Over 1.5", 0.80)])
    assert [e.selection for e in shown.passed] == ["Over 1.5"]


def test_the_reply_counts_whole_lines_in_one_line_only_when_nothing_passes(monkeypatch):
    gate = _gate(monkeypatch, [injected("Over 2.0", 0.95), injected("Over 3.0", 0.90), injected("Over 2.5", 0.60, passed=False)])
    assert not gate.passed
    text = "\n".join(gate_lines(gate))
    assert "2 whole-line play(s) ignored: whole-line plays are not allowed." in text and text.count("whole-line play(s)") == 1
    assert "Over 2.0" not in text and "Over 3.0" not in text
    why = "\n".join(why_lines(gate, None))
    assert "2 whole-line play(s) ignored" in why and "Over 2.0" not in why
    passing = _gate(monkeypatch, [injected("Over 2.0", 0.95), injected("Over 1.5", 0.80)])
    assert "whole" not in "\n".join(gate_lines(passing)).lower()


def test_a_whole_line_that_slipped_into_rejected_is_hidden_and_counted():
    f = tg.fixture()
    leaked = injected("Over 2.0", 0.7, passed=False)
    gate = GateResult((), (leaked, injected("Over 2.5", 0.6, passed=False)), 1, 0, {f.match_id: f})
    text = "\n".join(gate_lines(gate))
    assert "Over 2.0" not in text and "1 whole-line play(s) ignored" in text and "Over 2.5" in text


# ------------------------------------------------------------------ the closest option and alternatives
def test_the_closest_option_never_offers_or_replaces_a_whole_or_quarter_line():
    leg = leg_from_evaluation(pick_evaluation(selection="Over 2.5"))
    candidates = [pick_evaluation(market=f"Over/Under {x}", selection=f"Over {x}", prob=0.9) for x in ("2.0", "3.0", "2.25", "3.0")]
    assert closest_option(leg, candidates) is None
    half = pick_evaluation(market="Over/Under 3.5", selection="Over 3.5", prob=0.8)
    assert closest_option(leg, [*candidates, half]).selection == "Over 3.5"
    assert closest_option(leg_from_evaluation(candidates[0]), [half]) is None


def test_alternative_for_a_whole_line_pick_or_leg_offers_nothing():
    whole = Pick("w", pick_evaluation(market="Over/Under 2.0", selection="Over 2.0"))
    fixture = make_fixture("Alpha", "Beta", [totals_offer("3.5", "1.50", "1.50")])
    assert alternative_for(whole, SimpleNamespace(leg=leg_from_evaluation(whole.evaluation), fixture=fixture)) is None
    half = Pick("h", pick_evaluation())
    wholeleg = leg_from_evaluation(whole.evaluation)
    assert alternative_for(half, SimpleNamespace(leg=wholeleg, fixture=fixture)) is None


def test_the_alternative_search_skips_whole_line_listings():
    half = Pick("h", pick_evaluation())
    leg = leg_from_evaluation(half.evaluation)
    fixture = make_fixture("Alpha", "Beta", [totals_offer("2", "1.60", "2.30"), totals_offer("3.5", "2.30", "1.60"),
                                             totals_offer("1.5", "1.30", "3.20")])
    found = alternative_for(half, SimpleNamespace(leg=leg, fixture=fixture))
    assert found is not None and found.selection == "Over 1.5"


# ------------------------------------------------------------------ Slip and Game construction
def test_slip_refuses_a_whole_line_pick_clearly():
    whole = pick_evaluation(market="Over/Under 2.0", selection="Over 2.0", odds="1.60")
    with pytest.raises(ValueError, match=WHOLE_LINE_REASON):
        make_pick(whole)
    over = make_pick(pick_evaluation())
    with pytest.raises(ValueError, match=WHOLE_LINE_REASON):
        Slip((over,)).substitute(over.pick_id, whole)


def test_from_evaluations_leaves_a_whole_line_out_without_crashing(caplog):
    over = pick_evaluation("Alpha", "Beta")
    whole = pick_evaluation("Gamma", "Delta", market="Over/Under 2.0", selection="Over 2.0", odds="1.60")
    with caplog.at_level("WARNING"):
        slip = Slip.from_evaluations([over, whole])
    assert [p.evaluation.selection for p in slip.picks] == ["Over 2.5"]
    assert any(WHOLE_LINE_REASON in r.message for r in caplog.records)


def test_games_never_carry_a_whole_line_option_even_if_one_was_built_by_hand():
    over = make_pick(pick_evaluation())
    whole_eval = pick_evaluation(market="Over/Under 2.0", selection="Over 2.0", odds="1.60")
    smuggled = Pick(make_pick_id(whole_eval), whole_eval)
    games = build_games((over, smuggled))
    assert len(games) == 1 and [o.evaluation.selection for o in games[0].options] == ["Over 2.5"]
    assert not build_games((smuggled,))


def test_a_session_start_ignores_a_whole_line_evaluation():
    manager = SessionManager()
    session = manager.start([pick_evaluation(market="Over/Under 2.0", selection="Over 2.0", odds="1.60")], SID)
    assert len(session.menu) == 0 and len(session.slip) == 0


# ------------------------------------------------------------------ the betslip loader and the adapter
def test_the_loader_fails_closed_on_a_whole_line_pick_and_never_touches_the_bookmaker():
    manager = SessionManager()
    manager.start([pick_evaluation(odds="1.85")], SID)
    session = manager.active
    assert manager.act(OWNER, SID, session.state_version, Action.DONE).status is Status.APPLIED
    assert manager.act(OWNER, SID, manager.active.state_version, Action.CONFIRM).status is Status.APPLIED
    session = manager.active
    assert session.state is State.LOADING_SLIP
    whole = pick_evaluation(market="Over/Under 2.0", selection="Over 2.0", odds="1.60")
    manager._set(replace(session, slip=Slip((Pick(make_pick_id(whole), whole),))))   # smuggled past every other guard
    bookmaker = FakeBookmaker([make_fixture("Alpha", "Beta", [totals_offer("2", "1.60", "2.30")])])
    outcome = asyncio.run(run_load(manager, bookmaker, SID, manager.active.state_version, str))
    assert outcome.kind == KIND_FAILED and WHOLE_LINE_REASON in outcome.message
    assert bookmaker.calls == [] and bookmaker.loaded is None


def test_the_loader_also_refuses_a_whole_line_leg_whose_text_looks_plain():
    # the text says a half line but the leg's own specifier is whole: the leg check catches it
    manager = SessionManager()
    manager.start([pick_evaluation(odds="1.85")], SID)
    assert manager.act(OWNER, SID, manager.active.state_version, Action.DONE).status is Status.APPLIED
    assert manager.act(OWNER, SID, manager.active.state_version, Action.CONFIRM).status is Status.APPLIED
    sneaky = pick_evaluation(odds="1.85")
    leg = leg_from_evaluation(sneaky)
    assert not is_whole_or_quarter_line(leg)
    assert is_whole_or_quarter_line(replace(leg, specifier="total=2"))


def test_the_sportybet_adapter_refuses_to_load_a_whole_line_leg():
    leg = leg_from_evaluation(pick_evaluation(market="Over/Under 2.0", selection="Over 2.0", odds="1.60"))
    bookmaker = SportyBetBookmaker(dry_run=True)
    with pytest.raises(BookmakerError, match=WHOLE_LINE_REASON):
        asyncio.run(bookmaker.load_slip([leg]))


# ------------------------------------------------------------------ snapshot restore
def test_a_restored_snapshot_drops_a_saved_whole_line_pick_and_alternative():
    from predictions.flow import snapshot
    manager = SessionManager()
    over = pick_evaluation("Alpha", "Beta")
    whole = pick_evaluation("Gamma", "Delta", market="Over/Under 2.0", selection="Over 2.0", odds="1.60")
    manager.start([over], SID)
    data = snapshot.session_to_data(manager.active)
    smuggled = snapshot.session_to_data(replace(manager.active, slip=Slip((Pick(make_pick_id(whole), whole),)),
                                                menu=(Pick(make_pick_id(whole), whole),)))
    restored = snapshot.session_from_data(smuggled)
    assert len(restored.slip) == 0 and len(restored.menu) == 0
    assert len(snapshot.session_from_data(data).menu) == 1
    from predictions.flow.session import Issue, ISSUE_OPTION
    issue = Issue(make_pick_id(over), ISSUE_OPTION, "gone", whole)
    with_alt = snapshot.session_to_data(replace(manager.active, issues=(issue,)))
    assert snapshot.session_from_data(with_alt).issues[0].alternative is None


# ------------------------------------------------------------------ Review is NOT restricted
def test_review_still_judges_a_whole_line_selection_with_the_push_maths():
    f = tso.fixture([], competition=tso.TOP)
    selection = tnu.code_selection("Over 2.0", specifier="total=2", competition=tso.TOP, odds="1.45")
    verdict = judge_selection(selection, f, FixtureStats(f.match_id, READY, tuple(EVEN)))
    assert verdict.selection == "Over 2.0" and not verdict.insufficient and verdict.estimate.push > 0.2
    assert verdict.passed and verdict.estimate.probability == pytest.approx(0.8, abs=2e-3)
    assert abs(float(verdict.ev) - (verdict.estimate.win * 0.45 - (1 - verdict.estimate.probability))) < 0.002
    assert verdict.rule != RULE_NO_WHOLE_LINE and verdict.reason != WHOLE_LINE_REASON


def test_review_hands_on_no_whole_line_to_a_bet_but_still_lists_it_as_judged():
    from predictions.core.review import CodeSelection, Judgement, ReviewResult
    from predictions.flow.review import approved_evaluations
    f = tso.fixture([], competition=tso.TOP)
    stats = FixtureStats(f.match_id, READY, tuple(EVEN))
    whole = judge_selection(tnu.code_selection("Over 2.0", specifier="total=2", competition=tso.TOP, odds="1.45"), f, stats)
    half = judge_selection(tnu.code_selection("Over 1.5", specifier="total=1.5", competition=tso.TOP, odds="1.35"), f, stats)
    result = ReviewResult("ABC123", (Judgement(tnu.code_selection("Over 2.0", specifier="total=2", competition=tso.TOP), whole),
                                     Judgement(tnu.code_selection("Over 1.5", specifier="total=1.5", competition=tso.TOP), half)), ())
    assert result.approved and len(result.approved) == 2          # both are judged and approved on the review itself
    assert [e.selection for e in approved_evaluations(result)] == ["Over 1.5"]   # but only the plain one can become a bet
