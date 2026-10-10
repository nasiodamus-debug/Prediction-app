"""The walk-forward football backtest (tools/football_backtest.py): no leakage from the future, a readable report, a local copy of
a season, and the way a season is read from ESPN. No network: seasons are fabricated and ESPN is a MockTransport."""
import json
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from predictions.adapters.espn import football as fb
from predictions.core.engines import american_football as af
from predictions.tools import football_backtest as bt

START = datetime(2026, 9, 5, 17, 0, tzinfo=timezone.utc)


def side(i, score):
    return fb.Side(str(i), f"Team {i}", score, (7.0, 7.0, 7.0, float(score - 21) if score >= 21 else 0.0))


def season(weeks=8, teams=8, tweak=None):
    """A round robin: every team plays every week, with fixed scores that depend on the team ids (so strengths are real)."""
    games = []
    ids = list(range(teams))
    for week in range(1, weeks + 1):
        order = ids[week % teams:] + ids[:week % teams]
        for k in range(0, teams, 2):
            home, away = order[k], order[k + 1]
            hs, aws = 20 + 3 * home + (week % 3), 20 + 3 * away + (week % 2)
            if tweak:
                hs, aws = tweak(week, home, away, hs, aws)
            when = START + timedelta(days=7 * (week - 1), hours=k)
            games.append(bt.Game(week, fb.PlayedGame(f"{week}-{home}-{away}", when, side(home, hs), side(away, aws))))
    return games


def test_the_replay_predicts_from_the_weeks_before_and_never_from_the_game_itself_or_later():
    base = season()
    predictions, skipped = bt.backtest(base, af.COLLEGE, min_week=5)
    assert predictions and {p.week for p in predictions} == {5, 6, 7, 8}
    # change every score from week 6 onwards: the week 5 expectations must not move at all
    changed = season(tweak=lambda week, h, a, hs, aws: (hs + 40, aws) if week >= 6 else (hs, aws))
    again, _ = bt.backtest(changed, af.COLLEGE, min_week=5)
    week5 = [(p.mu_margin, p.mu_total) for p in predictions if p.week == 5]
    assert week5 == [(p.mu_margin, p.mu_total) for p in again if p.week == 5]
    later = [(p.mu_margin) for p in predictions if p.week == 7]
    assert later != [p.mu_margin for p in again if p.week == 7]               # while a later week DOES see the new results


def test_a_team_without_enough_games_is_skipped_not_guessed():
    only_two_weeks = [g for g in season() if g.week in (1, 2, 3)]
    predictions, skipped = bt.backtest(only_two_weeks + [bt.Game(4, season()[-1].game)], af.COLLEGE, min_week=4)
    assert predictions == [] and skipped >= 1                                # 3 games of history, college needs 4


def test_every_prediction_carries_probabilities_and_outcomes_the_calibration_can_use():
    predictions, _ = bt.backtest(season(), af.COLLEGE, min_week=5)
    events = [e for p in predictions for e in p.events]
    assert {e.kind for e in events} == {"winner", "handicap", "total", "team_total"}
    assert all(0.0 <= e.p <= 1.0 for e in events)
    rows = bt.calibration(events)
    assert sum(n for _, n, _, _ in rows) == len(events)                      # every event lands in exactly one bucket of its favoured side
    assert all(0.5 <= predicted <= 1.0 and 0.0 <= actual <= 1.0 for _, _, predicted, actual in rows)


def test_the_accuracy_numbers_match_a_hand_calculation():
    p = bt.Prediction(5, mu_margin=3.0, mu_total=44.0, mu_home=23.5, mu_away=20.5, margin=10.0, total=50.0,
                      home_points=30.0, away_points=20.0, events=[bt.Event("winner", 0.7, True)])
    q = bt.Prediction(5, mu_margin=-3.0, mu_total=44.0, mu_home=20.5, mu_away=23.5, margin=-1.0, total=38.0,
                      home_points=17.0, away_points=21.0, events=[bt.Event("winner", 0.4, True)])
    acc = bt.accuracy([p, q])
    assert acc["margin_bias"] == pytest.approx((7.0 + 2.0) / 2) and acc["margin_mae"] == pytest.approx(4.5)
    assert acc["total_bias"] == pytest.approx((6.0 - 6.0) / 2) and acc["total_mae"] == pytest.approx(6.0)
    assert acc["winner_accuracy"] == pytest.approx(0.5) and acc["winner_brier"] == pytest.approx(((0.3) ** 2 + (0.4 - 1) ** 2) / 2)


def test_the_report_names_what_the_model_assumes_and_what_happened():
    predictions, skipped = bt.backtest(season(), af.COLLEGE, min_week=5)
    text = bt.format_report(predictions, af.COLLEGE, "test", skipped)
    assert (f"model assumes {af.MARGIN_SD[af.COLLEGE]:g}" in text and f"model assumes {af.TOTAL_SD[af.COLLEGE]:g}" in text
            and f"model assumes {af.TEAM_SD[af.COLLEGE]:g}" in text and f"model assumes {af.FH_TOTAL_SD[af.COLLEGE]:g}" in text)
    assert "When the model favours a side" in text and "winner:" in text and "handicap:" in text
    assert "no games could be predicted" in bt.format_report([], af.COLLEGE, "empty")


def test_a_season_is_kept_locally_and_read_back_identically(tmp_path):
    games = season()
    path = tmp_path / "x.json"
    bt.save_games(path, games)
    assert bt.load_games(path) == games
    path.write_text("not json")
    assert bt.load_games(path) is None and bt.load_games(tmp_path / "missing.json") is None


def scoreboard(week, events):
    return {"season": {"year": 2025, "type": 2}, "week": {"number": week}, "events": events}


def event(eid, home, away, hs, aws, week):
    def side_json(kind, i, score):
        return {"homeAway": kind, "team": {"id": str(i), "displayName": f"Team {i}"}, "score": str(score),
                "linescores": [{"value": 7.0}] * 4}
    return {"id": str(eid), "date": (START + timedelta(days=7 * week)).strftime("%Y-%m-%dT%H:%MZ"), "season": {"year": 2025, "type": 2},
            "status": {"type": {"completed": True}},
            "competitions": [{"venue": {"address": {"city": "X"}}, "competitors": [side_json("home", home, hs), side_json("away", away, aws)]}]}


def test_a_season_is_read_from_both_college_divisions_week_by_week_and_a_finished_year_is_cached(tmp_path):
    seen = []

    def handler(request):
        params = dict(request.url.params)
        seen.append((params.get("groups"), params.get("week"), params.get("dates"), params.get("seasontype")))
        week = int(params["week"])
        if week > 2:
            return httpx.Response(404)
        eid = 100 * week + (1 if params.get("groups") == "80" else 2)
        return httpx.Response(200, json=scoreboard(week, [event(eid, 1 + week, 5, 24, 17, week)]))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    today = datetime(2026, 10, 10, tzinfo=timezone.utc)
    games = bt.season_games(af.COLLEGE, 2025, client, cache_dir=tmp_path, today=today)
    assert {g for g, *_ in seen} == {"80", "81"} and {w for _, w, *_ in seen} == {str(w) for w in range(1, bt.MAX_WEEK + 1)}
    assert all(d == "2025" and t == "2" for _, _, d, t in seen)
    assert len(games) == 4 and [g.week for g in games] == [1, 1, 2, 2]
    calls = len(seen)
    again = bt.season_games(af.COLLEGE, 2025, client, cache_dir=tmp_path, today=today)     # a finished year: the local copy
    assert again == games and len(seen) == calls
    bt.season_games(af.COLLEGE, 2026, client, cache_dir=tmp_path, today=today)             # the current year is always read again
    assert len(seen) > calls


def test_the_production_variant_is_registered_and_applies_the_minimum_games_per_league():
    assert bt.VARIANTS["production"] is bt.production_inputs
    games = season(weeks=4)
    history = [g.game for g in games if g.week <= 3]
    assert bt.production_inputs(history, "0", "1", af.COLLEGE) is None                    # three games: college needs four
    assert bt.production_inputs(history, "0", "1", af.NFL) is not None


def test_one_malformed_entry_on_a_page_loses_only_that_entry_in_a_backtest():
    good = event(1, 1, 2, 24, 17, 1)
    page = scoreboard(1, [good, {}, event(2, 3, 4, 10, 7, 1)])
    with pytest.raises(Exception):
        fb.parse_scoreboard(page)                                   # the bot's own parser still refuses the whole page
    assert [g.game_id for g in bt._parse_page(page)] == ["1", "2"]
    assert bt._parse_page("not a page") == [] and bt._parse_page({"events": "x"}) == []


def test_production_is_what_the_bot_builds_now_college_gets_the_adjusted_rates_and_raw_does_not():
    games = season(weeks=6)
    history = [g.game for g in games if g.week <= 5]
    college = {i.name for i in bt.production_inputs(history, "0", "1", af.COLLEGE)}
    assert "home.adj_points_for" in college and "away.adj_points_against" in college
    assert not any(".adj_" in i.name for i in bt.raw_inputs(history, "0", "1", af.COLLEGE))
    assert not any(".adj_" in i.name for i in bt.production_inputs(history, "0", "1", af.NFL))     # the NFL keeps the raw averages
    assert bt.VARIANTS["raw"] is bt.raw_inputs


def test_production_is_what_the_bot_builds_now_college_gets_the_adjusted_rates_and_raw_does_not():
    games = season(weeks=6)
    history = [g.game for g in games if g.week <= 5]
    college = {i.name for i in bt.production_inputs(history, "0", "1", af.COLLEGE)}
    assert "home.adj_points_for" in college and "away.adj_points_against" in college
    assert not any(".adj_" in i.name for i in bt.raw_inputs(history, "0", "1", af.COLLEGE))
    assert not any(".adj_" in i.name for i in bt.production_inputs(history, "0", "1", af.NFL))     # the NFL keeps the raw averages
    assert bt.VARIANTS["raw"] is bt.raw_inputs
