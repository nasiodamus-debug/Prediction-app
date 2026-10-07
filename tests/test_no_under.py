"""AD-20: the owner's rule "In all sports no under predictions are allowed".

An Under is never produced, listed, offered, converted to, kept, loaded or placed, in any sport, trigger or market.
Nothing here uses the network."""
import asyncio
import itertools
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

import test_basketball_engine as tb
import test_corners_cards as tcc
import test_football_engine as tf
import test_gate as tg
import test_mlb_engine as tm
import test_nhl_engine as tn
import test_soccer_optimization as tso
from fakes_bookmaker import FakeBookmaker, make_fixture, pick_evaluation, totals_offer
from predictions.adapters.sportybet.betslip import SportyBetBookmaker
from predictions.core.betting import leg_from_evaluation
from predictions.core.engines import ENGINES
from predictions.core.market_rules import (RULE_NO_UNDER, SIDES_PLAYED, UNDER_REASON, is_under, reject_under,
                                           without_unders)
from predictions.core.options import closest_option
from predictions.core.ports import BookmakerError
from predictions.core.review import CodeSelection, judge_selection
from predictions.core.slip import Pick, Slip, build_games, make_pick, make_pick_id
from predictions.core.stats import READY, FixtureStats
from predictions.core.types import Fixture, Outcome, Owner
from predictions.flow import evaluate as evaluate_flow
from predictions.flow.evaluate import GateResult, run_gate
from predictions.flow.loading import KIND_FAILED, alternative_for, run_load
from predictions.flow.review import run_review
from predictions.flow.session import Action, SessionManager, State, Status
from predictions.telegram.render import gate_lines, why_lines

D = Decimal
OWNER = Owner(111222333)
SID = "abcd1234"


def make_pick_obj(evaluation):
    return Pick(make_pick_id(evaluation), evaluation)


def code_selection(outcome, sport="soccer", home="Arsenal", away="Leeds United", market_id="18", name="Over/Under",
                   specifier="total=2.5", competition="England - Premier League", odds="2.00"):
    return CodeSelection(sport, "Football", competition, home, away, datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc),
                         False, market_id, name, specifier, outcome, D(odds))


# ------------------------------------------------------------------ the one definition
@pytest.mark.parametrize("text", [
    "Under 2.5", "under 2.5", "UNDER 2.5", "Under 3.0", "Under 3", "Under 152.5", "Under 24.5", "  Under   0.5 ",
    "U 2.5", "u2.5", "U2.5", "Home team Under 1.5", "Away team Under 2.5", "Home team U 1.5", "1st half Under 24.5",
    "1st Half Under 20.5", "Home 1st half team Under 8.5", "1st period Under 1.5", "F5 Under 4.5",
    "F5 Home team Under 1.5", "Corners Under 9.5", "Cards Under 4.5", "Corners U 9.5", "Under", "Under(2.5)",
    "Under_2.5"])
def test_is_under_is_true_for_every_spelling(text):
    assert is_under(text)


@pytest.mark.parametrize("text", [
    "Over 2.5", "Over 2.0", "over 152.5", "Home", "Away", "Draw", "Home (+3.5)", "Away (-1.5)", "Home team Over 1.5",
    "1st half Over 24.5", "Corners Over 9.5", "Cards Over 4.5", "F5 Over 4.5", "Underdog +1.5", "Over/Under 2.5",
    "1st Half - Over/Under", "Team U 21", "", "Manchester United", "Alpha"])
def test_is_under_is_false_for_everything_else(text):
    assert not is_under(text)


def test_is_under_reads_evaluations_codeselections_legs_picks_and_outcomes():
    under, over = pick_evaluation(selection="Under 2.5", odds="2.00"), pick_evaluation(selection="Over 2.5")
    assert is_under(under) and not is_under(over)
    assert is_under(make_pick_obj(under)) and not is_under(make_pick_obj(over))
    assert is_under(leg_from_evaluation(under)) and not is_under(leg_from_evaluation(over))
    assert is_under(Outcome("Under 2.5", D("2.0"))) and not is_under(Outcome("Over 2.5", D("2.0")))
    assert is_under(code_selection("Under 2.5")) and not is_under(code_selection("Over 2.5"))
    assert is_under(SimpleNamespace(side="under")) and not is_under(SimpleNamespace(side="over")) and not is_under(None)


def test_the_helpers_reject_and_strip():
    under, over = pick_evaluation(selection="Under 2.5", odds="2.00"), pick_evaluation(selection="Over 2.5")
    refused = reject_under(under)
    assert not refused.passed and refused.reason == UNDER_REASON and refused.rule == RULE_NO_UNDER
    assert without_unders([under, over]) == [over] and SIDES_PLAYED == ("over",)


# ------------------------------------------------------------------ every engine, Under strongly favoured
def soccer_cases():
    for competition in (tso.TOP, tso.MID, "Testland - League"):
        for attack, defence in ((0.7, 0.7), (1.0, 1.0), (1.5, 1.5), (2.0, 1.0), (2.8, 2.2)):
            for under_odds in ("1.30", "1.90", "2.50", "4.00"):
                markets = [tso.total_offer(line, "1.90", under_odds) for line in ("0.5", "1.5", "2.5", "3.5", "4.5")]
                markets += [tso.total_offer(line, "1.50", under_odds) for line in ("1", "2", "3", "4")]
                markets += [tso.team_total("19", "1.5", "1.80", under_odds),
                            tso.team_total("20", "0.5", "1.40", under_odds),
                            tso.team_total("19", "2", "2.10", under_odds), tso.one_x_two("2.60", "3.10", "2.80"),
                            tcc.prop_offer("Total Corners", "9.5", "1.80", under_odds),
                            tcc.prop_offer("Total Cards", "4.5", "1.80", under_odds)]
                f = tso.fixture(markets, competition=competition)
                inputs = tso.make_inputs(attack, defence, attack, defence) + tcc.extra()
                yield ("soccer", f, inputs)


def basketball_cases():
    for ortg, pace in ((95.0, 94.0), (105.0, 98.0), (112.0, 100.0), (118.0, 104.0)):
        for under_odds in ("1.40", "2.00", "3.00"):
            side = dict(ortg=ortg, drtg=ortg, pace=pace, fh_for=ortg / 2, fh_against=ortg / 2)
            offers = (tb.totals(under_odds=under_odds) + tb.team_totals("home", under_odds=under_odds)
                      + tb.team_totals("away", under_odds=under_odds) + tb.fh_totals(under_odds=under_odds))
            yield ("basketball", tb.make_fixture(offers), tb.inputs(side, side))


def baseball_cases():
    for rf, ra in ((3.0, 3.0), (4.0, 4.2), (4.5, 4.5), (5.4, 5.0)):
        for under_odds in ("1.40", "2.00", "3.00"):
            side = dict(rf=rf, ra=ra, f5_for=rf / 2, f5_against=ra / 2)
            offers = (tm.totals(under=under_odds) + tm.team_totals("home", under=under_odds)
                      + tm.team_totals("away", under=under_odds) + tm.f5_totals(under=under_odds)
                      + tm.f5_team_totals(under=under_odds))
            yield ("baseball", tm.make_fixture(offers), tm.inputs(side, side))


def hockey_cases():
    for gf, ga in ((2.0, 2.0), (2.6, 2.6), (3.0, 3.0), (3.6, 3.4)):
        for under_odds in ("1.40", "2.00", "3.00"):
            side = dict(gf=gf, ga=ga, xf=gf, xa=ga, p1gf=gf / 3, p1ga=ga / 3, p1xf=gf / 3, p1xa=ga / 3)
            offers = (tn.totals(under=under_odds) + tn.team_totals("home", under=under_odds)
                      + tn.team_totals("away", under=under_odds) + tn.period1(under=under_odds)
                      + tn.puck(("-1.5", "1.5")))
            yield ("hockey", tn.make_fixture(offers), tn.inputs(side, side))


def football_cases():
    for competition in ("USA - NFL", "USA - NCAA, Division I"):
        for points in (13.0, 20.0, 24.0, 31.0):
            for wind in (None, 5.0, 14.9, 15.0, 25.0):
                for pace, rush in ((None, None), (29.5, 56.0), (27.0, 45.0)):
                    side = dict(pf=points, pa=points, fh_for=points / 2, fh_against=points / 2, q1_rank=3,
                                pace=pace, rush=rush)
                    offers = (tf.totals(under="2.40") + tf.team_totals("home", under="2.40")
                              + tf.team_totals("away", under="2.40") + tf.fh_totals(under="2.40")
                              + tf.spreads({"-3.5": ("1.9", "1.9"), "7.5": ("1.9", "1.9")}) + tf.moneyline())
                    yield ("american_football", tf.make_fixture(offers, competition), tf.inputs(side, side, wind=wind))


def all_cases():
    return itertools.chain(soccer_cases(), basketball_cases(), baseball_cases(), hockey_cases(), football_cases())


def assert_no_under(evaluations):
    for e in evaluations:
        assert not is_under(e), e.selection
        assert "Under" not in e.rule and "Under" not in e.original, (e.selection, e.rule, e.original)


def test_property_no_engine_returns_an_under_anywhere_on_the_grid():
    count = 0
    for sport, fixture, inputs in all_cases():
        evals = ENGINES[sport].evaluate(fixture, inputs, fixture.markets)
        assert_no_under(evals)
        count += 1
    assert count > 200


def test_property_evaluate_exact_never_returns_an_under_for_any_offer_on_the_grid():
    seen_offers = 0
    for sport, fixture, inputs in itertools.islice(all_cases(), 0, None, 3):
        engine = ENGINES[sport]
        for offer in fixture.markets:
            assert_no_under(engine.evaluate_exact(fixture, inputs, offer))
            seen_offers += 1
    assert seen_offers > 1000


def test_property_the_run_gate_passes_no_under_on_the_grid():
    for sport, fixture, inputs in itertools.islice(all_cases(), 0, None, 7):
        stats = {fixture.match_id: FixtureStats(fixture.match_id, READY, tuple(inputs))}
        gate = run_gate([fixture], stats)
        assert not [e for e in gate.passed + gate.rejected if is_under(e)]


def test_an_under_that_the_model_loves_is_not_played_and_the_over_is_unaffected_soccer():
    # a defensive game: Under 2.5 is about 90% likely at 2.50, the old engine would have played it
    under_fav = tso.make_inputs(0.7, 0.7, 0.7, 0.7)
    f = tso.fixture([tso.total_offer("2.5", "1.90", "2.50")], competition=tso.TOP)
    evals = tso.evaluate(f, under_fav)
    assert [e.selection for e in evals] == ["Over 2.5"] and not evals[0].passed
    # an attacking game keeps its Over exactly as before
    f2 = tso.fixture([tso.total_offer("1.5", "1.30", "3.20")], competition=tso.TOP)
    over = tso.by_selection(tso.evaluate(f2, tg.inputs_for(2.5, 1.5)))["Over 1.5"]
    assert over.passed and over.estimate.probability == pytest.approx(0.908, abs=1e-3)


def test_soccer_shield_swap_and_conversion_never_produce_an_under():
    # no shield any more (AD-21): Over 2.5 stays Over 2.5; the Under 2.5 -> Under 3.0 path is gone
    f = tso.fixture([tso.total_offer("2.5", "1.90", "1.90"), tso.total_offer("2", "1.45", "2.90"),
                     tso.total_offer("3", "1.30", "3.40")])
    evals = tso.by_selection(tso.evaluate(f, tso.EVEN))
    assert set(evals) == {"Over 2.5"} and "Under 3.0" not in evals and "Under 2.0" not in evals
    assert_no_under(evals.values())
    # swap: a 1X2 favourite is converted to a goal line or team total, never to an Under
    f = tso.fixture([tso.one_x_two("1.70", "3.60", "5.00"), tso.total_offer("1.5", "1.30", "3.40"),
                     tso.total_offer("2.5", "1.90", "1.90"), tso.team_total("19", "1.5", "1.80", "1.90")])
    assert_no_under(tso.evaluate(f, tso.HOME_FAVOURITE))


def test_basketball_locked_team_rule_never_produces_an_under():
    locked = dict(ortg=118.0, drtg=105.0, pace=104.0)
    evals = tb.run(tb.inputs(locked, locked), tb.totals(under_odds="2.50") + tb.team_totals("home", under_odds="2.50"))
    assert evals and not [e for e in evals if "Under" in e.selection]
    assert any(e.selection.startswith("Home team Over") for e in evals)   # the Over route is untouched


def test_mlb_overs_are_unaffected_when_the_under_is_strongly_favoured():
    cold = dict(rf=2.8, ra=2.8, f5_for=1.5, f5_against=1.5)
    evals = tm.run(tm.inputs(cold, cold), tm.totals(under="2.40"))
    assert evals and not [e for e in evals if "Under" in e.selection]
    hot = tm.run(tm.inputs(tm.HOT, tm.HOT), tm.totals())
    assert hot and hot[0].passed and hot[0].selection.startswith("Over")


def test_football_wind_and_pace_rules_produce_a_plain_pass_never_an_under():
    # wind: the full-game Over is disqualified, and nothing takes its place (the old Team Total Under shift is gone)
    inps = tf.inputs(home=dict(pf=17.0, pa=17.0), away=dict(pf=17.0, pa=17.0), wind=20.0)
    offers = (tf.totals(("34.5",), under="2.40") + tf.team_totals("home", ("26.5",), under="2.40")
              + tf.team_totals("away", ("26.5",), under="2.40"))
    evals = tf.run(inps, offers)
    assert_no_under(evals)
    assert not any(e.passed for e in evals)
    blocked = [e for e in evals if e.rule.startswith(tf.af.RULE_WIND)]
    assert blocked and all(not e.passed for e in blocked)
    # pace: a slow run-heavy team blocks the Over, with no Under baseline play
    home = dict(pf=32.0, pa=20.0, pace=28.6, rush=52.1)
    evals = tf.run(tf.inputs(home=home, away=tf.BOOST["away"], wind=5.0), tf.totals(("40.5", "44.5"), under="2.40"))
    over = tf.over_evals(evals)
    assert len(over) == 1 and not over[0].passed and over[0].rule.startswith(tf.af.RULE_PACE)
    assert_no_under(evals)


# ------------------------------------------------------------------ the central gate
class InjectingEngine:
    """An engine that breaks the rule on purpose: the central gate must still catch it."""

    def __init__(self, evaluations):
        self.evaluations = evaluations

    def screen(self, fixture, inputs):
        return None

    def evaluate(self, fixture, inputs, markets):
        return list(self.evaluations)


def gate_with(monkeypatch, evaluations):
    f = tg.fixture(markets=[tg.totals("2.5", "1.90", "1.90")])
    monkeypatch.setitem(evaluate_flow.ENGINES, "soccer", InjectingEngine(evaluations))
    stats = {f.match_id: FixtureStats(f.match_id, READY, tuple(tg.inputs_for(2.0, 1.0)))}
    return f, run_gate([f], stats)


def injected(selection, odds="2.00", passed=True):
    return replace(pick_evaluation(selection=selection, odds=odds), match_id=tg.fixture().match_id, passed=passed)


def test_the_central_gate_rejects_an_injected_under_evaluation(monkeypatch):
    _, gate = gate_with(monkeypatch, [injected("Under 2.5"), injected("Over 2.5", "1.85")])
    assert [e.selection for e in gate.passed] == ["Over 2.5"]
    assert [e.selection for e in gate.under_rejected] == ["Under 2.5"] and gate.under_ignored == 1
    refused = gate.under_rejected[0]
    assert not refused.passed and refused.reason == UNDER_REASON
    assert not [e for e in gate.rejected if is_under(e)]


def test_the_reply_counts_unders_in_one_line_only_when_nothing_passes_and_never_lists_them(monkeypatch):
    _, gate = gate_with(monkeypatch, [injected("Under 2.5"), injected("Under 3.5"),
                                      injected("Over 2.5", "1.85", passed=False)])
    assert not gate.passed and gate.under_ignored == 2
    text = "\n".join(gate_lines(gate))
    assert "2 under play(s) ignored: under plays are not allowed." in text
    assert "Under 2.5" not in text and "Under 3.5" not in text and text.count("under play(s)") == 1
    why = "\n".join(why_lines(gate, None))
    assert "2 under play(s) ignored: under plays are not allowed." in why and "Under 2.5" not in why


def test_the_reply_does_not_mention_unders_when_something_passes(monkeypatch):
    _, gate = gate_with(monkeypatch, [injected("Under 2.5"), injected("Over 2.5", "1.85")])
    text = "\n".join(gate_lines(gate)).lower()
    assert gate.passed and "under" not in text


def test_an_under_that_slipped_into_rejected_is_hidden_and_counted():
    f = tg.fixture()
    leaked = injected("Under 2.5", passed=False)
    gate = GateResult((), (leaked, injected("Over 2.5", "1.85", passed=False)), 1, 0, {f.match_id: f})
    text = "\n".join(gate_lines(gate))
    assert "Under 2.5" not in text and "1 under play(s) ignored" in text and "Over 2.5" in text


# ------------------------------------------------------------------ Review
class TripwireAnalyst:
    """Any use of the analyst fails the test: an Under selection must cost no analyst call."""

    calls = 0

    async def research(self, query):
        TripwireAnalyst.calls += 1
        raise AssertionError("the analyst was called")


class Reader:
    def __init__(self, selections):
        self.selections = selections

    async def read_code(self, code):
        return list(self.selections)


def run_review_with(selections, **kwargs):
    import test_review as tr
    stats = tr.FakeStats()
    seen = []
    chunks = asyncio.run(run_review(" ABC123", Reader(selections), str, stats_sources={"soccer": stats}, now=tr.NOW,
                                    on_result=seen.append, **kwargs))
    return chunks, seen, stats


@pytest.mark.parametrize("sport,market_id,name,spec,outcome", [
    ("soccer", "18", "Over/Under", "total=2.5", "Under 2.5"),
    ("soccer", "18", "Over/Under", "total=3", "Under 3.0"),
    ("soccer", "19", "Home Over/Under", "total=1.5", "Home team Under 1.5"),
    ("basketball", "225", "Over/Under (incl. overtime)", "total=152.5", "Under 152.5"),
    ("baseball", "225", "Total", "total=8.5", "Under 8.5"),
    ("hockey", "18", "Total", "total=5.5", "Under 5.5"),
    ("american_football", "225", "Over/Under (incl. overtime)", "total=24.5", "Under 24.5"),
])
def test_review_disqualifies_an_under_before_any_stats_or_analyst_work(sport, market_id, name, spec, outcome):
    TripwireAnalyst.calls = 0
    s = code_selection(outcome, sport=sport, market_id=market_id, name=name, specifier=spec)
    chunks, seen, stats = run_review_with([s], analyst=TripwireAnalyst())
    text = "\n".join(chunks)
    assert UNDER_REASON in text
    assert stats.asked == [] and TripwireAnalyst.calls == 0
    result = seen[0]
    assert not result.approved and len(result.disqualified) == 1
    ev = result.disqualified[0].evaluation
    assert ev.reason == UNDER_REASON and ev.rule == RULE_NO_UNDER and not ev.passed


def test_judge_selection_disqualifies_an_under_even_with_ready_stats_and_a_great_price():
    import test_review as tr
    s = code_selection("Under 2.5", odds="3.50")
    fixture = Fixture(s.match_id, s.competition, s.home, s.away, s.kickoff_utc, False)
    stats = FixtureStats(s.match_id, READY, tuple(tr.inputs_for(0.6, 0.6)))
    ev = judge_selection(s, fixture, stats)
    assert not ev.passed and ev.reason == UNDER_REASON


def test_review_still_judges_the_over_next_to_a_disqualified_under():
    TripwireAnalyst.calls = 0
    chunks, seen, stats = run_review_with(
        [code_selection("Under 2.5", home="Arsenal", away="Leeds United"),
         code_selection("Over 2.5", home="Chelsea", away="Everton", odds="1.85")])
    assert stats.asked == ["Chelsea"]            # only the Over's game is looked up
    assert any(j.evaluation.reason == UNDER_REASON for j in seen[0].judgements)
    assert any(j.selection.outcome_name == "Over 2.5" and j.evaluation.reason != UNDER_REASON
               for j in seen[0].judgements)


# ------------------------------------------------------------------ alternatives for a missing option
def passed_eval(selection, line, odds="1.85", prob=0.80):
    return pick_evaluation(market=f"Over/Under {line}", selection=selection, odds=odds, prob=prob)


def test_the_closest_option_is_never_an_under():
    original_over = leg_from_evaluation(pick_evaluation(selection="Over 2.5"))
    candidates = [passed_eval("Under 3.5", "3.5", "1.50", 0.95), passed_eval("Over 3.5", "3.5", "1.95", 0.78)]
    chosen = closest_option(original_over, candidates)
    assert chosen is not None and chosen.selection == "Over 3.5"
    assert closest_option(original_over, [passed_eval("Under 3.5", "3.5", "1.50", 0.95)]) is None
    original_under = leg_from_evaluation(pick_evaluation(selection="Under 2.5", odds="2.00"))
    assert closest_option(original_under, [passed_eval("Under 3.5", "3.5", "1.50", 0.95)]) is None
    assert closest_option(original_under, candidates) is None


def test_alternative_for_an_under_pick_offers_nothing():
    under = make_pick_obj(pick_evaluation(selection="Under 2.5", odds="2.00"))
    leg = leg_from_evaluation(under.evaluation)
    fixture = make_fixture("Alpha", "Beta", [totals_offer("3.5", "1.50", "1.50")])
    found = SimpleNamespace(leg=leg, fixture=fixture)
    assert alternative_for(under, found) is None


# ------------------------------------------------------------------ Slip and Game construction
def test_slip_refuses_an_under_pick_clearly():
    under = pick_evaluation(selection="Under 2.5", odds="2.00")
    with pytest.raises(ValueError, match=UNDER_REASON):
        make_pick(under)
    over = make_pick(pick_evaluation(selection="Over 2.5"))
    with pytest.raises(ValueError, match=UNDER_REASON):
        Slip((over,)).substitute(over.pick_id, replace(under, odds=D("1.85")))


def test_from_evaluations_leaves_an_under_out_without_crashing(caplog):
    over = pick_evaluation("Alpha", "Beta", selection="Over 2.5")
    under = pick_evaluation("Gamma", "Delta", market="Over/Under 2.5", selection="Under 2.5", odds="2.00")
    with caplog.at_level("WARNING"):
        slip = Slip.from_evaluations([over, under])
    assert [p.evaluation.selection for p in slip.picks] == ["Over 2.5"]
    assert any(UNDER_REASON in r.message for r in caplog.records)


def test_games_never_carry_an_under_option_even_if_one_was_built_by_hand():
    over = make_pick(pick_evaluation(selection="Over 2.5"))
    under_eval = pick_evaluation(selection="Under 2.5", odds="2.00")
    smuggled = Pick(make_pick_id(under_eval), under_eval)
    games = build_games((over, smuggled))
    assert len(games) == 1 and [o.evaluation.selection for o in games[0].options] == ["Over 2.5"]
    assert not build_games((smuggled,))


def test_a_session_start_ignores_an_under_evaluation():
    manager = SessionManager()
    session = manager.start([pick_evaluation(selection="Under 2.5", odds="2.00")], SID)
    assert len(session.menu) == 0 and len(session.slip) == 0


# ------------------------------------------------------------------ the betslip loader
def test_the_loader_fails_closed_on_an_under_pick_and_never_touches_the_bookmaker():
    manager = SessionManager()
    manager.start([pick_evaluation(selection="Over 2.5", odds="1.85")], SID)
    session = manager.active
    assert manager.act(OWNER, SID, session.state_version, Action.DONE).status is Status.APPLIED
    assert manager.act(OWNER, SID, manager.active.state_version, Action.CONFIRM).status is Status.APPLIED
    session = manager.active
    assert session.state is State.LOADING_SLIP
    under = pick_evaluation(selection="Under 2.5", odds="2.00")
    manager._set(replace(session, slip=Slip((Pick(make_pick_id(under), under),))))   # smuggled past every other guard
    bookmaker = FakeBookmaker([make_fixture("Alpha", "Beta", [totals_offer("2.5", "1.85", "2.00")])])
    outcome = asyncio.run(run_load(manager, bookmaker, SID, manager.active.state_version, str))
    assert outcome.kind == KIND_FAILED and UNDER_REASON in outcome.message
    assert bookmaker.calls == [] and bookmaker.loaded is None


def test_the_sportybet_adapter_refuses_to_load_an_under_leg():
    leg = leg_from_evaluation(pick_evaluation(selection="Under 2.5", odds="2.00"))
    bookmaker = SportyBetBookmaker(dry_run=True)
    with pytest.raises(BookmakerError, match=UNDER_REASON):
        asyncio.run(bookmaker.load_slip([leg]))


def test_a_restored_snapshot_drops_a_saved_under_pick():
    from predictions.flow import snapshot
    manager = SessionManager()
    over = pick_evaluation("Alpha", "Beta", selection="Over 2.5")
    under = pick_evaluation("Gamma", "Delta", selection="Under 2.5", odds="2.00")
    manager.start([over], SID)
    data = snapshot.session_to_data(manager.active)
    smuggled = snapshot.session_to_data(replace(manager.active, slip=Slip((Pick(make_pick_id(under), under),)),
                                                menu=(Pick(make_pick_id(under), under),)))
    restored = snapshot.session_from_data(smuggled)
    assert len(restored.slip) == 0 and len(restored.menu) == 0
    assert len(snapshot.session_from_data(data).menu) == 1
