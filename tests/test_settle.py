"""Settling one leg from a final score (diagnosis only). Pure functions, no network."""
import pytest

from predictions.core.settle import (HALF_LOSS, HALF_WIN, LOSS, PUSH, UNVERIFIABLE, VOID, WIN, FinalScore, parse_specifier,
                                      settle_leg)


def verdict(sport, market, spec, side, score):
    return settle_leg(sport, market, spec, side, score).verdict


def test_the_specifier_is_read_as_key_value_pairs():
    assert parse_specifier("total=8.5|periodnr=1") == {"total": "8.5", "periodnr": "1"}
    assert parse_specifier("") == {}


# ---------------------------------------------------------------- soccer
def test_soccer_1x2_and_double_chance_use_the_ninety_minute_score():
    assert verdict("soccer", "1", "", "home", FinalScore(2, 1)) == WIN
    assert verdict("soccer", "1", "", "draw", FinalScore(2, 1)) == LOSS
    assert verdict("soccer", "10", "", "home or draw", FinalScore(1, 1)) == WIN
    assert verdict("soccer", "10", "", "draw or away", FinalScore(2, 0)) == LOSS
    assert verdict("soccer", "10", "", "home or away", FinalScore(1, 1)) == LOSS


def test_a_game_that_went_to_extra_time_is_unverifiable_unless_the_regulation_score_is_known():
    assert verdict("soccer", "1", "", "home", FinalScore(2, 1, extra_time=True)) == UNVERIFIABLE
    assert verdict("soccer", "1", "", "draw", FinalScore(2, 1, extra_time=True, regulation=(1, 1))) == WIN


def test_soccer_totals_half_whole_and_quarter_lines():
    assert verdict("soccer", "18", "total=2.5", "over", FinalScore(2, 1)) == WIN
    assert verdict("soccer", "18", "total=2.5", "over", FinalScore(1, 1)) == LOSS
    assert verdict("soccer", "18", "total=3", "over", FinalScore(2, 1)) == PUSH        # exactly the line: refund
    assert verdict("soccer", "18", "total=2.75", "over", FinalScore(2, 1)) == HALF_WIN   # 3 goals: wins the 2.5 half, pushes the 3.0 half
    assert verdict("soccer", "18", "total=2.25", "over", FinalScore(2, 1)) == WIN       # 3 beats both 2.0 and 2.5
    assert verdict("soccer", "18", "total=2.25", "over", FinalScore(1, 1)) == HALF_LOSS  # 2: pushes 2.0, loses 2.5


def test_soccer_team_totals_and_asian_handicap_use_the_home_handicap():
    assert verdict("soccer", "19", "total=1.5", "over", FinalScore(2, 0)) == WIN
    assert verdict("soccer", "20", "total=0.5", "over", FinalScore(2, 0)) == LOSS
    assert verdict("soccer", "16", "hcp=-1", "home", FinalScore(2, 1)) == PUSH           # wins by exactly one
    assert verdict("soccer", "16", "hcp=-1", "away", FinalScore(2, 1)) == PUSH
    assert verdict("soccer", "16", "hcp=-1.5", "home", FinalScore(2, 1)) == LOSS
    assert verdict("soccer", "16", "hcp=-1.5", "away", FinalScore(2, 1)) == WIN         # away line is +1.5
    assert verdict("soccer", "16", "hcp=-0.25", "home", FinalScore(1, 1)) == HALF_LOSS
    assert verdict("soccer", "16", "hcp=0.25", "home", FinalScore(1, 1)) == HALF_WIN


def test_soccer_corners_cards_and_unknown_markets_are_unverifiable():
    for market in ("166", "139", "999"):
        assert verdict("soccer", market, "total=9.5", "over", FinalScore(3, 3)) == UNVERIFIABLE


# ---------------------------------------------------------------- basketball and american football
def test_us_winner_handicap_and_totals_include_overtime():
    assert verdict("basketball", "219", "", "home", FinalScore(101, 99)) == WIN
    assert verdict("basketball", "223", "hcp=21.5", "home", FinalScore(70, 100)) == LOSS    # -30 + 21.5 is below zero
    assert verdict("basketball", "223", "hcp=21.5", "home", FinalScore(80, 100)) == WIN     # -20 + 21.5 covers
    assert verdict("basketball", "223", "hcp=21.5", "away", FinalScore(80, 100)) == LOSS    # the away line is -21.5
    assert verdict("basketball", "225", "total=190.5", "over", FinalScore(100, 91)) == WIN
    assert verdict("american_football", "225", "total=44", "over", FinalScore(24, 20)) == PUSH
    assert verdict("american_football", "227", "total=27.5", "over", FinalScore(28, 10)) == WIN
    assert verdict("american_football", "228", "total=27.5", "over", FinalScore(28, 10)) == LOSS


def test_a_tied_winner_market_is_unverifiable_not_a_loss():
    assert verdict("american_football", "219", "", "home", FinalScore(20, 20)) == UNVERIFIABLE


def test_first_half_totals_use_the_first_two_quarters_and_need_them():
    score = FinalScore(100, 90, periods=((20, 25), (30, 20), (25, 25), (25, 20)))
    assert verdict("basketball", "68", "total=94.5", "over", score) == WIN      # first half 50 + 45 = 95, above 94.5
    assert verdict("basketball", "68", "total=95.5", "over", score) == LOSS
    assert verdict("basketball", "68", "total=71.5", "over", FinalScore(100, 90)) == UNVERIFIABLE


# ---------------------------------------------------------------- baseball
def test_baseball_full_game_and_first_five_innings():
    assert verdict("baseball", "258", "total=4.5", "over", FinalScore(3, 2)) == WIN
    assert verdict("baseball", "256", "hcp=-1.5", "home", FinalScore(5, 3)) == WIN
    innings = tuple((1, 0) for _ in range(5)) + ((0, 3),) * 4
    score = FinalScore(5, 12, periods=innings)
    assert verdict("baseball", "274", "", "home", score) == WIN                # 5-0 after five
    assert verdict("baseball", "276", "total=4.5", "over", score) == WIN
    assert verdict("baseball", "275", "hcp=-0.5", "away", score) == LOSS
    assert verdict("baseball", "276", "total=4.5", "over", FinalScore(2, 1, periods=((1, 0), (1, 1)))) == VOID   # ended before 5
    assert verdict("baseball", "276", "total=4.5", "over", FinalScore(2, 1)) == UNVERIFIABLE


# ---------------------------------------------------------------- hockey and status
def test_hockey_full_game_markets_and_regulation_markets():
    assert verdict("hockey", "406", "", "home", FinalScore(4, 3)) == WIN
    assert verdict("hockey", "412", "total=5.5", "over", FinalScore(4, 3)) == WIN
    assert verdict("hockey", "446", "total=1.5", "over", FinalScore(4, 3)) == UNVERIFIABLE


def test_postponed_abandoned_and_missing_scores():
    assert verdict("soccer", "1", "", "home", FinalScore(0, 0, status="postponed")) == VOID
    assert verdict("soccer", "1", "", "home", FinalScore(0, 0, status="abandoned")) == VOID
    assert verdict("soccer", "1", "", "home", FinalScore(0, 0, status="suspended")) == UNVERIFIABLE
    assert verdict("soccer", "1", "", "home", None) == UNVERIFIABLE
    assert verdict("tennis", "1", "", "home", FinalScore(2, 0)) == UNVERIFIABLE


def test_only_losses_count_as_the_leg_that_lost():
    assert settle_leg("soccer", "1", "", "away", FinalScore(2, 1)).lost
    assert settle_leg("soccer", "18", "total=3", "over", FinalScore(2, 1)).lost is False   # a push cannot be the losing leg
