"""Story 2.4: 1X2 swap, derby and cup pass. The Asian goal-line shield and every whole-line play are gone (AD-21, spec-13-1)."""
import asyncio
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from listview import detail_text, list_text
from predictions.core.engines import soccer
from predictions.core.engines.soccer import (
    SoccerEngine, line_probabilities, poisson_cdf, result_probabilities, scaled_rates)
from predictions.core.engines.soccer_tables import LEAGUE_PROFILES, find_derby, is_cup
from predictions.core.inputs import Input, SOCCER_REGISTRY
from predictions.core.market_rules import is_whole_or_quarter_line
from predictions.core.model import PROBABILITY_FLOOR, Estimate, expected_value, validate
from predictions.core.stats import READY, FixtureStats, insufficient
from predictions.core.types import Fixture, MarketOffer, MatchId, Outcome
from predictions.flow.evaluate import run_gate
from predictions.flow.matchday import run_matchday

KICKOFF = datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
MID = "Russia - Premier League"       # tier 2: shielded
TOP = "Germany - Bundesliga"          # tier 1, 3.15 goals a game: half lines stand


def make_inputs(hf, ha, af, aa, ppda=None):
    vals = {"home": (hf, ha), "away": (af, aa)}
    out = []
    for side, (attack, defence) in vals.items():
        out.append(Input(f"{side}.matches", 10.0, "matches", "test"))
        for stat, v in (("xg_for", attack), ("goals_for", attack), ("xg_against", defence),
                        ("goals_against", defence)):
            out.append(Input(f"{side}.{stat}", float(v), SOCCER_REGISTRY[stat].unit, "test"))
    for side, value in (ppda or {}).items():
        out.append(Input(f"{side}.ppda", value, SOCCER_REGISTRY["ppda"].unit, "test"))
    return out


EVEN = make_inputs(2.0, 1.0, 2.0, 1.0)      # both sides expect 1.5 goals: total 3.0
HOME_FAVOURITE = make_inputs(2.8, 1.0, 1.8, 2.0)  # home wins 59.6%, 2.5+ goals 73.1% (so a swap can clear the 70% floor)


def total_offer(line, over, under):
    return MarketOffer("18", "Over/Under", f"total={line}",
                       (Outcome(f"Over {line}", Decimal(over)), Outcome(f"Under {line}", Decimal(under))))


def team_total(kind, line, over, under):
    return MarketOffer(kind, "Home O/U" if kind == "19" else "Away O/U", f"total={line}",
                       (Outcome(f"Over {line}", Decimal(over)), Outcome(f"Under {line}", Decimal(under))))


def one_x_two(home, draw, away):
    return MarketOffer("1", "1X2", "", (Outcome("Home", Decimal(home)), Outcome("Draw", Decimal(draw)),
                                        Outcome("Away", Decimal(away))))


def fixture(markets=(), competition=MID, home="Alpha", away="Beta"):
    mid = MatchId("soccer", competition, KICKOFF, home, away)
    return Fixture(mid, competition, home, away, KICKOFF, False, tuple(markets))


def evaluate(f, inputs):
    return SoccerEngine().evaluate(f, inputs, f.markets)


def by_selection(evals):
    return {e.selection: e for e in evals}


# ------------------------------------------------------------------ push-protected maths
def test_whole_line_probabilities_sum_to_one_and_push_is_the_exact_count():
    over_win, under_win, push = line_probabilities(lambda k: poisson_cdf(k, 3.0), Decimal("2"))
    assert over_win == pytest.approx(0.5768, abs=1e-3)      # P(3 or more goals)
    assert push == pytest.approx(0.2240, abs=1e-3)          # P(exactly 2 goals)
    assert under_win == pytest.approx(0.1992, abs=1e-3)     # P(0 or 1 goals)
    assert over_win + under_win + push == pytest.approx(1.0)


def test_half_line_has_no_push():
    over_win, under_win, push = line_probabilities(lambda k: poisson_cdf(k, 3.0), Decimal("2.5"))
    assert push == 0.0 and over_win + under_win == pytest.approx(1.0)


def test_quarter_line_is_refused():
    with pytest.raises(ValueError):
        line_probabilities(lambda k: 0.5, Decimal("2.25"))


def test_success_is_win_plus_push_and_rounds_down():
    estimate = Estimate.push_protected(0.57689, 0.22399)
    assert estimate.probability == 0.799  # 0.576 + 0.223, each part floored first
    assert estimate.push == 0.223 and estimate.win == pytest.approx(0.576)


def test_ev_is_win_times_odds_minus_one_minus_lose():
    # P(win) 50%, P(push) 20%, P(lose) 30%, odds 1.60: EV = 0.5 * 0.6 - 0.3 = 0.0
    assert expected_value(0.70, Decimal("1.60"), 0.20) == Decimal("0.00")
    # the plain success * odds - 1 formula would call this +12% and let it through
    assert expected_value(0.70, Decimal("1.60")) == Decimal("0.12")


def test_push_pick_needs_positive_ev_even_when_success_clears_the_floor():
    estimate = Estimate.push_protected(0.50, 0.20)  # success 70%, above the floor
    ok, ev, reason = validate(estimate, Decimal("1.60"))
    assert not ok and ev == 0 and "expected value" in reason


def test_push_pick_below_floor_is_rejected_and_reason_shows_the_split():
    estimate = Estimate.push_protected(0.42, 0.22)  # success 64%
    ok, ev, reason = validate(estimate, Decimal("3.00"))
    assert not ok and "64.0%" in reason and "win 42.0% + refund 22.0%" in reason and "floor" in reason


def test_push_pick_at_the_floor_passes_with_positive_ev():
    estimate = Estimate.push_protected(0.55, 0.15)
    assert estimate.probability == PROBABILITY_FLOOR
    ok, ev, _ = validate(estimate, Decimal("1.80"))
    assert ok and ev == Decimal("0.55") * Decimal("0.8") - Decimal("0.30")


def test_estimate_rejects_a_push_larger_than_the_estimate():
    with pytest.raises(ValueError):
        Estimate(0.5, push=0.6)


# ------------------------------------------------------------------ league table
def test_league_table_is_well_formed():
    assert all(p.tier in (1, 2, 3) and 1.5 < p.avg_goals < 4.0 for p in LEAGUE_PROFILES.values())
    assert all(key == key.lower() for key in LEAGUE_PROFILES)


# ------------------------------------------------------------------ AD-21: no Asian goal lines, no shield
@pytest.mark.parametrize("competition", [MID, TOP, "Testland - League"])
def test_over_2_5_stays_over_2_5_in_every_league_there_is_no_shield(competition):
    f = fixture([total_offer("2.5", "1.90", "1.90"), total_offer("2", "1.45", "2.90")], competition=competition)
    found = by_selection(evaluate(f, EVEN))
    assert set(found) == {"Over 2.5"} and found["Over 2.5"].odds == Decimal("1.90")
    assert not found["Over 2.5"].rule and not found["Over 2.5"].original
    assert not found["Over 2.5"].passed and found["Over 2.5"].estimate.probability == pytest.approx(0.576, abs=1e-3)


def test_whole_and_quarter_lines_are_never_priced_returned_or_ranked():
    f = fixture([total_offer("1.5", "1.40", "2.80"), total_offer("2", "1.20", "4.40"), total_offer("2.25", "1.45", "2.60"),
                 total_offer("2.75", "1.70", "2.10"), total_offer("3", "2.40", "1.55")])
    found = evaluate(f, EVEN)
    assert [e.selection for e in found] == ["Over 1.5"]
    gate = run_gate([f], {f.match_id: ready(f, EVEN)})
    assert [e.selection for e in gate.passed] == ["Over 1.5"] and not gate.whole_rejected


def test_a_game_whose_only_listed_over_is_a_whole_line_has_no_play():
    f = fixture([total_offer("2", "1.45", "2.90")], competition=TOP)
    assert evaluate(f, EVEN) == []


def test_whole_lines_do_not_push_plain_plays_out_of_the_best_two():
    # the whole line 2.0 would have an 80% success (57.6% win + 22.4% refund) and outrank every plain play of the game
    f = fixture([total_offer("1.5", "1.30", "3.20"), total_offer("2", "1.45", "2.90"), total_offer("2.5", "1.90", "1.90"),
                 one_x_two("1.80", "3.60", "4.50"),
                 MarketOffer("10", "Double Chance", "", (Outcome("Home or Draw", Decimal("1.20")),
                                                         Outcome("Draw or Away", Decimal("1.90")),
                                                         Outcome("Home or Away", Decimal("1.30"))))])
    gate = run_gate([f], {f.match_id: ready(f, EVEN)})
    assert 1 <= len(gate.passed) <= 2 and not any(is_whole_or_quarter_line(e) for e in gate.passed)
    assert "Over 1.5" in {e.selection for e in gate.passed}


def test_review_still_prices_a_whole_line_with_the_refund_maths():
    f = fixture([], competition=TOP)
    offer = total_offer("2", "1.45", "2.90")
    found = SoccerEngine().evaluate_exact(f, EVEN, MarketOffer(offer.market_id, offer.name, offer.specifier, (offer.outcomes[0],)))
    assert [e.selection for e in found] == ["Over 2.0"] and found[0].estimate.push == pytest.approx(0.224, abs=1e-3)
    assert "refund on exactly 2 goal(s)" in found[0].notes[0] and found[0].estimate.probability == pytest.approx(0.8, abs=2e-3)


def test_goal_lines_below_one_and_a_half_are_not_matchday_plays():
    # Spec 11-1: only Over lines of 1.5 and above; Over 0.5, Over 1.0 (whole) are never considered
    f = fixture([total_offer("0.5", "1.10", "6.00"), total_offer("1", "1.20", "4.00")])
    assert evaluate(f, EVEN) == []


# ------------------------------------------------------------------ AC2: 1X2 swap
def swap_fixture(one_x_two_odds=("2.10", "3.60", "3.40"), extra=(), competition=TOP):
    return fixture([one_x_two(*one_x_two_odds), total_offer("2.5", "1.70", "2.10"),
                    total_offer("2", "1.40", "2.90"), *extra], competition=competition)


def test_sixty_percent_home_at_odds_above_two_converts_to_a_goal_line_that_clears_the_floor():
    f = swap_fixture()
    evals = by_selection(evaluate(f, HOME_FAVOURITE))
    assert "Home" not in evals                                # the 1X2 pick is not kept
    converted = evals["Over 2.5"]
    assert converted.passed and converted.estimate.probability >= PROBABILITY_FLOOR and converted.ev > 0
    assert converted.original.startswith("Home @ 2.10") and "59.6%" in converted.original
    assert "1X2 swap" in converted.rule and "2.00" in converted.rule


def test_swap_picks_the_candidate_with_the_best_ev_among_those_that_clear():
    f = swap_fixture(extra=[team_total("19", "1.5", "1.60", "2.30")])
    converted = [e for e in evaluate(f, HOME_FAVOURITE) if e.original]
    assert len(converted) == 1
    # the only conversion target is Over 2.5 @ 1.70 (73.1%, EV +24.3%): the whole line Over 2.0 is never a target (AD-21)
    assert converted[0].selection == "Over 2.5" and not any(e.selection == "Over 2.0" for e in evaluate(f, HOME_FAVOURITE))


def test_swap_never_lands_on_a_team_total():
    # Spec 11-1: a team total is not a Matchday play, so a long-odds home win is never converted into one
    f = fixture([one_x_two("2.10", "3.60", "3.40"), team_total("19", "1.5", "1.90", "1.80")], competition=TOP)
    stronger = make_inputs(2.8, 0.8, 1.5, 2.0)                # home wins 64.9%, scores 2+ with 69.2%
    for inputs in (HOME_FAVOURITE, stronger):
        evals = evaluate(f, inputs)
        assert not any(e.passed for e in evals) and not any("team" in e.selection.lower() for e in evals)
        assert "no goal line is listed to convert to" in by_selection(evals)["Home"].reason


def test_a_swap_that_stays_below_the_floor_leaves_the_original_rejected_with_the_reason():
    low_scoring = make_inputs(1.6, 0.5, 0.8, 1.3)
    f = swap_fixture()
    evals = evaluate(f, low_scoring)
    assert not any(e.passed for e in evals)
    home = by_selection(evals)["Home"]
    assert home.rule.startswith("1X2 swap") and "conversion also fails" in home.reason
    assert "floor" in home.reason


def test_swap_with_nothing_listed_says_so():
    f = fixture([one_x_two("2.10", "3.60", "3.40")], competition=TOP)
    home = by_selection(evaluate(f, HOME_FAVOURITE))["Home"]
    assert not home.passed and "no goal line is listed" in home.reason


def test_converted_pick_with_ev_at_or_below_zero_is_dropped():
    f = fixture([one_x_two("2.10", "3.60", "3.40"), total_offer("2.5", "1.35", "3.20")], competition=TOP)
    # Over 2.5 is 73.1% but 0.731 * 1.35 - 1 < 0
    evals = by_selection(evaluate(f, HOME_FAVOURITE))
    assert not any(e.passed for e in evals.values())
    assert "expected value" in evals["Home"].reason


def test_odds_of_exactly_two_or_less_are_not_swapped():
    f = swap_fixture(("2.00", "3.60", "3.40"))
    evals = by_selection(evaluate(f, HOME_FAVOURITE))
    assert "Home" in evals and not evals["Home"].rule and not evals["Home"].passed


def test_a_1x2_at_or_over_the_floor_is_kept_as_1x2():
    strong = make_inputs(3.4, 0.8, 1.3, 2.3)                  # home wins 74%
    f = swap_fixture(("2.30", "4.00", "6.00"))
    home = by_selection(evaluate(f, strong))["Home"]
    assert home.passed and not home.original and home.estimate.probability >= PROBABILITY_FLOOR


def test_a_long_shot_is_rejected_not_swapped():
    f = swap_fixture(("2.10", "3.60", "3.40"))
    away = by_selection(evaluate(f, HOME_FAVOURITE))["Away"]    # 21.9%: nobody would back it
    assert not away.passed and not away.rule and not away.original


def test_a_draw_is_not_a_matchday_play_at_all():
    # Spec 11-1: a Draw on its own is never priced, swapped or listed (Review may still judge it through evaluate_exact)
    f = swap_fixture(("2.10", "3.60", "3.40"))
    assert "Draw" not in by_selection(evaluate(f, HOME_FAVOURITE))


def test_result_probabilities_sum_to_one_and_follow_the_rates():
    home, draw, away = result_probabilities(2.2, 1.3)
    assert home + draw + away == pytest.approx(1.0, abs=1e-9)
    assert home > away and 0.15 < draw < 0.25
    assert result_probabilities(1.4, 1.4)[0] == pytest.approx(result_probabilities(1.4, 1.4)[2])


def test_rates_are_scaled_consistently_when_the_total_is_clamped():
    h, a = scaled_rates(make_inputs(6.0, 6.0, 6.0, 6.0))
    assert h + a == pytest.approx(soccer.MAX_EXPECTED_GOALS) and h == pytest.approx(a)


# ------------------------------------------------------------------ AC3: derby and cup pass
@pytest.mark.parametrize("home,away,competition", [
    ("Arsenal", "Tottenham Hotspur", "England - Premier League"),
    ("Tottenham", "Arsenal FC", "England - FA Cup"),
    ("Manchester City", "Man United", "England - Premier League"),
    ("Real Madrid", "FC Barcelona", "Spain - LaLiga"),
    ("Inter Milan", "AC Milan", "Italy - Serie A"),
    ("Borussia Dortmund", "FC Bayern Munchen", "Germany - Bundesliga"),
    ("Paris Saint-Germain", "Olympique Marseille", "France - Ligue 1"),
    ("Spartak Moscow", "CSKA Moscow", "Russia - Premier League"),
])
def test_derbies_are_found_whatever_the_spelling_or_order(home, away, competition):
    assert find_derby(competition, home, away) is not None


@pytest.mark.parametrize("home,away,competition", [
    ("Arsenal", "Leeds United", "England - Premier League"),
    ("Arsenal", "Tottenham Hotspur", "Spain - LaLiga"),          # right clubs, wrong country
    ("Inter Miami", "AC Milan", "USA - MLS"),                     # a partial name is not the derby club
    ("Arsenal", "Arsenal", "England - Premier League"),
])
def test_non_derbies_are_not_flagged(home, away, competition):
    assert find_derby(competition, home, away) is None


def test_a_derby_is_no_valid_play_pass_with_the_reason():
    f = fixture([total_offer("1.5", "1.30", "3.20")], competition="England - Premier League",
                home="Arsenal", away="Tottenham Hotspur")
    refusal = SoccerEngine().screen(f, EVEN)
    assert refusal.rule == "Derby pass" and "North London derby" in refusal.reason
    assert evaluate(f, EVEN) == []                              # nothing is priced, even a 91% Over 1.5


def test_derby_pass_needs_no_stats_and_goes_through_the_gate():
    f = fixture([total_offer("1.5", "1.30", "3.20")], competition="England - Premier League",
                home="Arsenal", away="Tottenham Hotspur")
    gate = run_gate([f], {f.match_id: insufficient(f.match_id, "no data")})
    assert [p.rule for p in gate.passes] == ["Derby pass"] and gate.skipped == 0 and not gate.passed


class Fixtures:
    def __init__(self, fixtures):
        self.fixtures = fixtures

    async def fetch_fixtures(self):
        return self.fixtures


class Stats:
    def __init__(self, table):
        self.table = table

    async def fetch_stats(self, fixtures):
        return {f.match_id: self.table(f) for f in fixtures}


def matchday(fixtures, table, **kwargs):
    return asyncio.run(run_matchday("10 October", Fixtures(fixtures), str, NOW, stats_source=Stats(table), **kwargs))


def matchday_gate(fixtures, table, **kwargs):
    """The reply chunks and the GateResult (what the interactive list is built from when something passes)."""
    gates = []
    chunks = matchday(fixtures, table, on_result=gates.append, **kwargs)
    return chunks, gates[0]


def ready(f, inputs):
    return FixtureStats(f.match_id, READY, tuple(inputs))


def test_reply_for_a_derby_is_no_valid_play_pass_with_the_reason():
    derby = fixture([total_offer("1.5", "1.30", "3.20")], competition="England - Premier League",
                    home="Arsenal", away="Tottenham Hotspur")
    chunks = matchday([derby], lambda f: ready(f, EVEN))
    assert chunks[0] == "NO VALID PLAY - PASS"
    text = "\n".join(chunks)
    assert "NO VALID PLAY - PASS (Derby pass: derby (North London derby)" in text and "Arsenal vs Tottenham" in text


def test_derby_is_reported_beside_valid_plays_of_other_games():
    derby = fixture([total_offer("1.5", "1.30", "3.20")], competition="England - Premier League",
                    home="Arsenal", away="Tottenham Hotspur")
    other = fixture([total_offer("1.5", "1.30", "3.20")], competition=TOP, home="Gamma", away="Delta")
    chunks, gate = matchday_gate([derby, other], lambda f: ready(f, make_inputs(2.5, 1.5, 2.5, 1.5)))
    shown = list_text(gate.passed)
    assert chunks == [] and "Gamma vs Delta" in shown and "Over 1.5 goals @ 1.30" in shown   # only the game with a play
    assert "Arsenal" not in shown and any(r.rule == "Derby pass" for r in gate.passes)   # the derby is not listed


@pytest.mark.parametrize("name,cup", [("England - FA Cup", True), ("Spain - Copa del Rey", True),
                                      ("Germany - DFB Pokal", True), ("France - Coupe de France", True),
                                      ("England - Premier League", False), ("International - UEFA Nations League", False)])
def test_cup_proxy_is_the_competition_name(name, cup):
    assert is_cup(name) is cup


TIGHT = make_inputs(1.1, 1.0, 1.0, 1.1)   # about 2.1 expected goals


def test_tight_cup_tie_with_a_low_block_is_passed():
    f = fixture([total_offer("2.5", "1.90", "1.90")], competition="England - FA Cup")
    refusal = SoccerEngine().screen(f, make_inputs(1.1, 1.0, 1.0, 1.1, {"home": 8.0, "away": 15.0}))
    assert refusal.rule == "Cup tie pass" and "low block" in refusal.reason and "15.0" in refusal.reason


def test_cup_tie_with_a_pressing_side_on_both_ends_is_not_passed():
    inputs = make_inputs(1.1, 1.0, 1.0, 1.1, {"home": 8.0, "away": 9.0})
    assert SoccerEngine().screen(fixture(competition="England - FA Cup"), inputs) is None


def test_cup_tie_with_a_low_block_but_a_high_scoring_forecast_is_not_passed():
    inputs = make_inputs(2.2, 1.6, 2.0, 1.8, {"home": 8.0, "away": 15.0})
    assert SoccerEngine().screen(fixture(competition="England - FA Cup"), inputs) is None


def test_league_game_with_a_passive_side_is_not_passed():
    inputs = make_inputs(1.1, 1.0, 1.0, 1.1, {"home": 8.0, "away": 15.0})
    assert SoccerEngine().screen(fixture(competition=TOP), inputs) is None


def test_cup_tie_without_passivity_data_is_passed_because_a_low_block_cannot_be_ruled_out():
    refusal = SoccerEngine().screen(fixture(competition="England - FA Cup"), TIGHT)
    assert refusal.rule == "Cup tie pass" and "no PPDA input" in refusal.reason


def test_cup_rule_can_be_relaxed_by_its_constant(monkeypatch):
    monkeypatch.setattr(soccer, "CUP_PASS_WITHOUT_PPDA", False)
    assert SoccerEngine().screen(fixture(competition="England - FA Cup"), TIGHT) is None


# ------------------------------------------------------------------ reply text
def test_reply_lists_the_half_line_and_no_asian_line_or_shield():
    f = fixture([total_offer("1.5", "1.30", "3.20"), total_offer("2.5", "1.90", "1.90"), total_offer("2", "1.45", "2.90")])
    chunks, gate = matchday_gate([f], lambda fx: ready(fx, EVEN))
    shown, detail = list_text(gate.passed), detail_text(gate.passed)
    assert chunks == [] and "Over 1.5 goals @ 1.30" in shown
    assert "Asian" not in shown + detail and "2.0" not in shown + detail and "refund" not in shown + detail


def test_reply_for_a_swap_shows_the_1x2_original():
    f = swap_fixture()
    chunks, gate = matchday_gate([f], lambda fx: ready(fx, HOME_FAVOURITE))
    shown, detail = list_text(gate.passed), detail_text(gate.passed)
    assert chunks == [] and "Over 2.5 goals @ 1.70" in shown and "(1X2 swap)" in shown
    assert "Original pick: Home @ 2.10 (estimate 59.6%)" in detail and "Rule applied: 1X2 swap" in detail


def test_nothing_below_the_floor_or_with_zero_ev_ever_passes_in_any_rule_path():
    fixtures = []
    for i, league in enumerate((MID, TOP, "Testland - League")):
        for odds in ("1.05", "1.45", "2.60"):
            fixtures.append(fixture([total_offer("2.5", odds, odds), total_offer("2", odds, odds),
                                     total_offer("3", odds, odds), total_offer("1.5", odds, odds),
                                     one_x_two(odds, "3.5", "3.4"), team_total("19", "1.5", odds, odds)],
                                    competition=league, home=f"H{i}{odds}", away=f"A{i}{odds}"))
    for inputs in (EVEN, HOME_FAVOURITE, make_inputs(1.6, 0.5, 0.8, 1.3)):
        for f in fixtures:
            for e in evaluate(f, inputs):
                if e.passed:
                    assert e.estimate.probability >= PROBABILITY_FLOOR and e.ev > 0
                    assert e.ev == expected_value(e.estimate.probability, e.odds, e.estimate.push)
