"""Opponent-adjusted scoring ratings (core/ratings.py): pure arithmetic over played games."""
from types import SimpleNamespace

import pytest

from predictions.core.ratings import Result, adjusted_ratings, results_of


def game(home, away, hs, aws):
    return SimpleNamespace(home=SimpleNamespace(team_id=home, score=hs), away=SimpleNamespace(team_id=away, score=aws))


def test_results_of_gives_one_result_from_each_side():
    assert results_of([game("A", "B", 30, 10)]) == [Result("A", "B", 30.0, 10.0), Result("B", "A", 10.0, 30.0)]


def test_no_results_give_no_ratings():
    assert adjusted_ratings([]) is None


def test_a_team_that_only_beat_a_weak_opponent_is_rated_below_a_team_with_the_same_scores_against_a_strong_one():
    # S (strong) and W (weak) play others: S scores a lot and allows little; W the reverse.
    games = [game("S", "M1", 40, 10), game("S", "M2", 38, 12), game("W", "M1", 10, 40), game("W", "M2", 12, 38),
             # X and Y both score 35 and allow 14, but X's opponent is the weak team and Y's is the strong one
             game("X", "W", 35, 14), game("Y", "S", 35, 14), game("M1", "M2", 24, 24)]
    ratings = adjusted_ratings(results_of(games), ridge=0.0, iterations=200)
    assert ratings.rate_for("Y") > ratings.rate_for("X")             # the same 35 points mean more against a strong defence
    assert ratings.rate_against("Y") < ratings.rate_against("X")


def test_the_league_level_is_the_mean_score_and_attack_and_defence_are_centred_on_it():
    games = [game("A", "B", 30, 20), game("B", "C", 27, 17), game("C", "A", 14, 21), game("A", "C", 24, 10)]
    ratings = adjusted_ratings(results_of(games), ridge=0.0, iterations=300)
    assert ratings.league == pytest.approx(sum([30, 20, 27, 17, 14, 21, 24, 10]) / 8)
    assert sum(ratings.attack.values()) == pytest.approx(0.0, abs=1e-9) and sum(ratings.defence.values()) == pytest.approx(0.0, abs=1e-9)


def test_the_ratings_recover_the_attack_and_defence_of_a_league_that_is_exactly_additive():
    attack = {"A": 6.0, "B": 2.0, "C": -2.0, "D": -6.0}
    defence = {"A": 3.0, "B": -1.0, "C": 1.0, "D": -3.0}
    games = []
    for i, a in enumerate("ABCD"):
        for b in "ABCD"[i + 1:]:
            games.append(game(a, b, 24 + attack[a] + defence[b], 24 + attack[b] + defence[a]))
    ratings = adjusted_ratings(results_of(games), ridge=0.0, iterations=800)
    for team in "ABCD":
        assert ratings.attack[team] == pytest.approx(attack[team], abs=1e-3)
        assert ratings.defence[team] == pytest.approx(defence[team], abs=1e-3)
        assert ratings.rate_for(team) == pytest.approx(24 + attack[team], abs=1e-3)
    assert ratings.rate_for("Z") is None and ratings.rate_against("Z") is None


def test_a_ridge_keeps_a_team_with_one_game_close_to_the_league_level():
    games = [game("A", "B", 60, 10), game("B", "C", 20, 20), game("C", "D", 20, 20), game("D", "B", 20, 20), game("C", "B", 20, 20)]
    loose = adjusted_ratings(results_of(games), ridge=0.0, iterations=200)
    tight = adjusted_ratings(results_of(games), ridge=6.0, iterations=200)
    assert abs(tight.attack["A"]) < abs(loose.attack["A"])                    # A has a single (huge) game: the ridge tempers it


def test_the_ratings_are_deterministic():
    games = [game("A", "B", 30, 20), game("B", "C", 27, 17), game("C", "A", 14, 21)]
    assert adjusted_ratings(results_of(games)) == adjusted_ratings(results_of(games))
