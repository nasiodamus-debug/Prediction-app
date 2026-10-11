"""The soccer market blend (2026-10-11): result, double chance and goal totals use 75% of the bookmaker's own price (margin removed)."""
from decimal import Decimal

import pytest

from predictions.core.engines import soccer
from predictions.core.engines.soccer import SoccerEngine, market_overs, market_result
from predictions.tools import soccer_backtest as bt
from test_soccer_optimization import EVEN, TOP, fixture, make_inputs, one_x_two, total_offer


@pytest.fixture(autouse=True)
def blend_on(monkeypatch):
    monkeypatch.setattr(soccer, "MARKET_WEIGHT", 0.75)


def evals(f, inputs=EVEN):
    return {e.selection: e for e in SoccerEngine().evaluate(f, inputs, f.markets)}


def test_default_weight_is_three_quarters_market():
    from pathlib import Path
    assert "MARKET_WEIGHT = 0.75" in Path(soccer.__file__).read_text(encoding="utf-8").splitlines()


def test_fair_probabilities_remove_the_margin():
    fair = market_result([one_x_two("2.00", "4.00", "4.00")])
    assert fair == pytest.approx((0.5, 0.25, 0.25))
    assert market_overs([total_offer("2.5", "1.80", "1.80")])[Decimal("2.5")] == pytest.approx(0.5)


def test_an_incomplete_price_gives_no_market():
    from predictions.core.types import MarketOffer, Outcome
    partial = MarketOffer("1", "1X2", "", (Outcome("Home", Decimal("2.0")),))
    assert market_result([partial]) is None
    only_over = MarketOffer("18", "Over/Under", "total=2.5", (Outcome("Over 2.5", Decimal("1.9")),))
    assert market_overs([only_over]) == {}


def test_over_goals_are_pulled_toward_the_market():
    model_only = None
    f = fixture([total_offer("2.5", "1.50", "2.60")], competition=TOP)
    blended = evals(f)["Over 2.5"].estimate.probability
    soccer.MARKET_WEIGHT = 0.0
    model_only = evals(f)["Over 2.5"].estimate.probability
    soccer.MARKET_WEIGHT = 0.75
    market = soccer._fair([Decimal("1.50"), Decimal("2.60")])[0]
    assert blended == pytest.approx(0.75 * market + 0.25 * model_only, abs=1e-3)
    assert "probability blended 75% with the bookmaker's own price" in evals(f)["Over 2.5"].notes


def test_a_pick_the_market_does_not_back_no_longer_passes():
    # the model loves the home side (about 60%), the market prices it at 40%: the blend sits near 45%, below the floor and below value
    f = fixture([one_x_two("2.60", "3.40", "2.80")], competition=TOP)
    home = make_inputs(2.8, 1.0, 1.8, 2.0)
    assert not evals(f, home)["Home"].passed


def test_no_market_price_means_model_alone_with_a_note():
    f = fixture([], competition=TOP)
    ctx = soccer._Context(f, tuple(EVEN), 1.5, 1.5, [])
    assert ctx.result_probs() == pytest.approx(soccer.result_probabilities(1.5, 1.5))
    assert ctx._market_note(False) == (soccer.MARKET_MISSING_NOTE,)


def test_double_chance_comes_from_the_blended_result():
    from predictions.core.types import MarketOffer, Outcome
    dc = MarketOffer("10", "Double Chance", "", (Outcome("Home or Draw", Decimal("1.30")),
                                                  Outcome("Draw or Away", Decimal("1.90")), Outcome("Home or Away", Decimal("1.25"))))
    f = fixture([one_x_two("2.10", "3.30", "3.60"), dc], competition=TOP)
    ctx = soccer._Context(f, tuple(EVEN), 1.5, 1.5, f.markets)
    h, d, a = ctx.result_probs()
    assert h + d + a == pytest.approx(1.0)
    got = evals(f)
    assert got["Home or Draw"].estimate.probability == pytest.approx(h + d, abs=1e-3)


# ------------------------------------------------------------------ backtest tool
CSV = ("Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,AvgCH,AvgCD,AvgCA,AvgC>2.5,AvgC<2.5\n"
       "E0,10/08/2024,A,B,2,1,2.0,4.0,4.0,1.8,1.8\n"
       "E0,17/08/2024,B,A,0,0,,,,,\n"
       "E0,bad,B,A,1,1,2,3,4,1.9,1.9\n")


def test_backtest_reads_results_and_fair_prices():
    games = bt.parse_games(CSV)
    assert len(games) == 2
    assert games[0].market_1x2 == pytest.approx((0.5, 0.25, 0.25))
    assert games[0].market_over == pytest.approx(0.5)
    assert games[1].market_1x2 is None


def test_backtest_scores_prefer_the_better_forecast():
    row = bt.Row((0.6, 0.2, 0.2), 0.6, (0.9, 0.05, 0.05), 0.9, 0, True)
    table = bt.score_table([row])
    by_label = {t[0]: t for t in table}
    assert by_label["market alone"][1] < by_label["model alone"][1]
    assert bt.log_loss([0.5, 0.5], 0) == pytest.approx(0.6931, abs=1e-3)
