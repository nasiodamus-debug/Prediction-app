"""More soccer markets Review can now price (scoreline model, Review only): GG/NG, Draw No Bet, result-or-over combos, 1X2 2UP, the 3-way
handicap, second-half multigoals and double chance, and Both Halves Under. Sides that are Unders in disguise are refused, never approved."""
import math
from decimal import Decimal as D

import pytest

from predictions.core.engines.soccer import (FIRST_HALF_SHARE, SECOND_HALF_SHARE, SoccerEngine, poisson_cdf, result_probabilities,
                                              scaled_rates)
from predictions.core.market_rules import UNDER_REASON
from predictions.core.types import MarketOffer, Outcome
from test_soccer_optimization import HOME_FAVOURITE, TOP, fixture

RATES = scaled_rates(HOME_FAVOURITE)
TOTAL = sum(RATES)


def price(name, specifier, outcome, market_id="0", odds="1.80"):
    offer = MarketOffer(market_id, name, specifier, (Outcome(outcome, D(odds)),))
    found = SoccerEngine().evaluate_exact(fixture([], competition=TOP), HOME_FAVOURITE, offer)
    assert len(found) == 1, (name, outcome, found)
    return found[0]


def test_every_new_market_is_understood_by_name_and_matchday_never_prices_them():
    engine = SoccerEngine()
    for name, mid in (("GG/NG", "29"), ("Draw No Bet", "11"), ("Home Team or Over 2.5", "0"), ("Draw or Over 2.5", "0"),
                      ("Away Team or Over 3.5", "0"), ("1X2 - 2UP", "0"), ("Handicap 0:1", "0"), ("2nd Half - Multigoals", "0"),
                      ("2nd Half - Double Chance", "0"), ("Both Halves Under 1.5", "0")):
        assert engine.understands(MarketOffer(mid, name, "", ())), name
    f = fixture([MarketOffer("29", "GG/NG", "", (Outcome("Yes", D("1.8")),)), MarketOffer("11", "Draw No Bet", "", (Outcome("Home", D("1.8")),))],
                competition=TOP)
    assert SoccerEngine().evaluate(f, HOME_FAVOURITE, f.markets) == []
    assert not engine.understands(MarketOffer("0", "Corners and cards combo", "", ()))


def test_both_teams_to_score_uses_the_scoring_chances_and_no_is_refused_as_an_under():
    expected = (1 - math.exp(-RATES[0])) * (1 - math.exp(-RATES[1]))
    yes = price("GG/NG", "", "Yes", "29")
    assert yes.estimate.probability == pytest.approx(expected, abs=2e-3)
    no = price("GG/NG", "", "No", "29")
    assert not no.passed and no.reason == UNDER_REASON and no.estimate.probability == pytest.approx(1 - expected, abs=2e-3)


def test_draw_no_bet_refunds_the_draw_so_both_sides_and_the_refund_add_up_to_one():
    home, away = price("Draw No Bet", "", "Home", "11"), price("Draw No Bet", "", "Away", "11")
    win_h, draw, win_a = result_probabilities(*RATES)
    assert home.estimate.win == pytest.approx(win_h, abs=2e-3) and home.estimate.push == pytest.approx(draw, abs=2e-3)
    assert home.estimate.win + away.estimate.win + home.estimate.push == pytest.approx(1.0, abs=4e-3)
    assert "refund on a draw" in home.notes[0]


def test_a_result_or_over_combo_is_the_union_and_no_is_refused():
    win_h, draw, win_a = result_probabilities(*RATES)
    home_or_over = price("Home Team or Over 2.5", "total=2.5", "Yes")
    plain_over = 1 - poisson_cdf(2, TOTAL)
    assert home_or_over.estimate.probability >= max(win_h, plain_over) - 2e-3     # the union is at least each part
    assert home_or_over.estimate.probability <= win_h + plain_over + 2e-3
    draw_or_over = price("Draw or Over 2.5", "total=2.5", "Yes")
    assert draw_or_over.estimate.probability >= plain_over - 2e-3
    refused = price("Away Team or Over 2.5", "total=2.5", "No")
    assert not refused.passed and refused.reason == UNDER_REASON


def test_one_x_two_2up_is_priced_as_the_plain_chance_and_says_the_early_payout_is_not_counted():
    win_h, draw, win_a = result_probabilities(*RATES)
    home = price("1X2 - 2UP", "", "Home")
    assert home.estimate.probability == pytest.approx(win_h, abs=2e-3) and "early payout is not counted" in home.notes[0]
    assert home.market == "1X2 - 2UP"


def test_the_three_way_handicap_sides_add_up_to_one_and_follow_the_starting_score():
    sides = [price("Handicap 0:1", "hcp=0:1", f"{s} (0:1)").estimate.probability for s in ("Home", "Draw", "Away")]
    assert sum(sides) == pytest.approx(1.0, abs=4e-3)
    even = [price("Handicap 1:0", "hcp=1:0", f"{s} (1:0)").estimate.probability for s in ("Home", "Draw", "Away")]
    assert even[0] > sides[0]                        # a one-goal head start helps the home side
    offer = MarketOffer("0", "Handicap 0:1", "hcp=0:1", (Outcome("Home (1:0)", D("1.8")),))
    assert SoccerEngine().evaluate_exact(fixture([], competition=TOP), HOME_FAVOURITE, offer) == []   # the outcome must match the market


def test_second_half_markets_use_the_assumed_half_share_and_say_so():
    rate = TOTAL * SECOND_HALF_SHARE
    multi = price("2nd Half - Multigoals", "", "1-3")
    assert multi.estimate.probability == pytest.approx(poisson_cdf(3, rate) - poisson_cdf(0, rate), abs=2e-3) and "assumed" in multi.notes[0]
    assert price("2nd Half - Multigoals", "", "2+").estimate.probability == pytest.approx(1 - poisson_cdf(1, rate), abs=2e-3)
    win_h, draw, win_a = result_probabilities(RATES[0] * SECOND_HALF_SHARE, RATES[1] * SECOND_HALF_SHARE)
    dc = price("2nd Half - Double Chance", "", "Home or Draw")
    assert dc.estimate.probability == pytest.approx(win_h + draw, abs=2e-3)
    assert price("2nd Half - Double Chance", "", "Home or Away").estimate.probability == pytest.approx(win_h + win_a, abs=2e-3)


def test_both_halves_under_prices_no_and_refuses_yes_as_an_under():
    both = poisson_cdf(1, TOTAL * FIRST_HALF_SHARE) * poisson_cdf(1, TOTAL * SECOND_HALF_SHARE)
    no = price("Both Halves Under 1.5", "total=1.5", "No")
    assert no.estimate.probability == pytest.approx(1 - both, abs=2e-3) and no.market == "Both Halves Under 1.5"
    yes = price("Both Halves Under 1.5", "total=1.5", "Yes")
    assert not yes.passed and yes.reason == UNDER_REASON and yes.estimate.probability == pytest.approx(both, abs=2e-3)


def test_unreadable_outcomes_are_skipped_not_priced():
    for name, outcome in (("2nd Half - Multigoals", "lots"), ("2nd Half - Double Chance", "Home or Nobody"), ("GG/NG", "Maybe")):
        offer = MarketOffer("0", name, "", (Outcome(outcome, D("1.5")),))
        assert SoccerEngine().evaluate_exact(fixture([], competition=TOP), HOME_FAVOURITE, offer) == []
