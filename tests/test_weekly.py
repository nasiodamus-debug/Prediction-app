"""The Monday weekly review flow: gathering, settling, reasons, recommendations, suppression and idempotency. No network."""
import asyncio
import json
from datetime import date, datetime, timezone

import pytest

from predictions.core.dates import LAGOS
from predictions.core.postgame import Recommendation
from predictions.core.settle import FinalScore
from predictions.flow.store import EvaluationStore
from predictions.flow.weekly import (WEEKLY_JOB_NAME, _select, last_completed_week, parse_weekly_time, run_weekly, week_containing,
                                     week_key)

NOW = datetime(2026, 10, 5, 8, 30, tzinfo=timezone.utc)      # Monday 09:30 in Lagos
SOCCER = "soccer:england premier league:20260930T1900Z:arsenal:chelsea"
WNBA = "basketball:usa wnba:20261002T0100Z:las vegas aces:indiana fever"
UNKNOWN = "soccer:made up league:20261001T1900Z:alpha:beta"


class FakeScores:
    def __init__(self, table=None, down=False):
        self.table, self.down, self.asked = table or {}, down, []

    async def final_score(self, match_id):
        self.asked.append(match_id)
        if self.down:
            raise RuntimeError("down")
        return self.table.get(match_id, (None, "no free score source")) if self.table else (None, "no free score source")


@pytest.fixture
def store(tmp_path):
    with EvaluationStore.open(tmp_path / "db.sqlite", lambda s: s) as s:
        yield s


def add_evaluation(store, match_id, market, selection, probability=0.75, model="poisson-xg-v1", passed=1, notes="expected goals 1.6 - 1.3",
                   rule="", original="", odds="1.80", insufficient=0, reason="passed", kickoff=None, league="England - Premier League"):
    parts = match_id.split(":")
    kick = kickoff or datetime.strptime(parts[2], "%Y%m%dT%H%MZ").replace(tzinfo=timezone.utc).isoformat()
    cur = store._conn.execute(
        "INSERT INTO evaluations (session_id, evaluated_at_utc, match_id, sport, league, home, away, kickoff_utc, market, selection, odds, "
        "probability, push, model, ev, passed, insufficient, reason, rule, original, notes, placed) "
        "VALUES ('s', '2026-09-29T10:00:00+00:00', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, '0.10', ?, ?, ?, ?, ?, ?, 0)",
        (match_id, parts[0], league, parts[3].title(), parts[4].title(), kick, market, selection, odds, probability, model, passed,
         insufficient, reason, rule, original, notes))
    eid = cur.lastrowid
    store._conn.execute("INSERT INTO inputs (evaluation_id, name, value, unit, source) VALUES (?, 'home.matches', 12, 'matches', 'understat')", (eid,))
    store._conn.execute("INSERT INTO inputs (evaluation_id, name, value, unit, source) VALUES (?, 'away.matches', 12, 'matches', 'understat')", (eid,))
    return eid


def add_bet(store, legs, state="PLACED_BY_OWNER", token="t1", stake="500", total="3.5"):
    store._conn.execute(
        "INSERT INTO placements (token_hash, session_id, state, dry_run, stake, total_odds, legs, expires_at_utc, created_at_utc, "
        "updated_at_utc, bet_id, booking_code, detail, source, owner_stake) VALUES (?, 's', ?, 0, ?, ?, ?, '2026-10-30T00:00:00+00:00', "
        "'2026-09-29T10:00:00+00:00', '2026-09-29T10:00:00+00:00', '', 'ABC', '', 'owner-confirmed', ?)",
        (token, state, stake, total, json.dumps(legs), stake))


def soccer_leg(match_id=SOCCER, side="over", spec="total=2.5", selection="Over 2.5"):
    return {"leg_key": [match_id, "18", spec, side], "market": "Over/Under 2.5", "selection": selection, "odds": "1.80", "outcome": selection}


def wnba_leg():
    return {"leg_key": [WNBA, "219", "", "home"], "market": "Winner (incl. overtime)", "selection": "Home", "odds": "1.55", "outcome": "Home"}


SCORES = {SOCCER: (FinalScore(1, 0), ""), WNBA: (FinalScore(94, 83), "")}


def run(store, scores=None, **kw):
    return asyncio.run(run_weekly(store, scores or FakeScores(SCORES), NOW, **kw))


# ---------------------------------------------------------------- weeks and settings
def test_the_review_covers_the_last_completed_monday_to_sunday_week():
    assert last_completed_week(date(2026, 10, 5)) == (date(2026, 9, 28), date(2026, 10, 4))
    assert last_completed_week(date(2026, 10, 7)) == (date(2026, 9, 28), date(2026, 10, 4))
    assert week_containing(date(2026, 10, 3)) == (date(2026, 9, 28), date(2026, 10, 4))
    assert week_key(date(2026, 9, 28)) == "2026-W40" and WEEKLY_JOB_NAME == "weekly-review"


def test_the_weekly_time_setting_defaults_and_ignores_garbage():
    assert parse_weekly_time(None).hour == 9 and parse_weekly_time(None).minute == 30
    assert (parse_weekly_time("07:05").hour, parse_weekly_time("07:05").minute) == (7, 5)
    assert parse_weekly_time("25:99").hour == 9 and parse_weekly_time("soon").minute == 30
    assert parse_weekly_time("07:05").tzinfo == LAGOS


# ---------------------------------------------------------------- the report
def test_a_lost_bet_is_listed_with_its_failed_leg_the_reason_and_the_sources(store):
    add_evaluation(store, SOCCER, "Over/Under 2.5", "Over 2.5")
    add_evaluation(store, WNBA, "Winner (incl. overtime)", "Home", probability=0.8, model="normal-points-v1", notes="", league="USA - WNBA")
    add_bet(store, [soccer_leg(), wnba_leg()])
    out = run(store)
    text = "\n".join(out.summary)
    assert out.week_key == "2026-W40" and out.status == "posted"
    assert "WEEKLY REVIEW 2026-W40" in text and "Lost (shown below): 1" in text
    assert "Arsenal v Chelsea" in text and "LOSS" in text and "WIN" in text and "OUR settlement" in text
    assert "Cause: variance" in text and "no change is warranted from this bet" in text
    assert "input home.matches = 12 [understat]" in out.doc_text and "primary reason" in out.doc_text
    assert store.weekly_review("2026-09-28")["status"] == "building"   # the app marks it posted after it has sent


def test_a_week_is_built_once_unless_forced_and_a_dry_run_writes_nothing(store):
    add_evaluation(store, SOCCER, "Over/Under 2.5", "Over 2.5")
    add_bet(store, [soccer_leg()])
    dry = run(store, dry=True)
    assert dry.status == "dry" and store.weekly_review("2026-09-28") is None and store.list_leg_settlements() == []
    first = run(store)
    store.finish_weekly_review("2026-09-28", "posted", "{}", [1, 2], "", NOW)
    again = run(store)
    assert again.status == "skipped" and "already built" in again.note
    forced = run(store, force=True)
    assert forced.status == "posted" and store.weekly_review("2026-09-28")["revision"] == 2 and first.status == "posted"


def test_a_failed_build_can_be_run_again_straight_away(store):
    add_evaluation(store, SOCCER, "Over/Under 2.5", "Over 2.5")
    add_bet(store, [soccer_leg()])

    class Boom(FakeScores):
        async def final_score(self, match_id):
            raise KeyboardInterrupt   # not an Exception: it escapes the per-game guard and aborts the build

    with pytest.raises(KeyboardInterrupt):
        run(store, Boom())
    assert run(store).status == "posted"


def test_no_bets_gives_a_short_report_and_still_records_the_week(store):
    out = run(store)
    text = "\n".join(out.summary)
    assert "Bets you placed this week: 0" in text and "nothing to explain" in text and out.recommendations == []
    assert store.weekly_review("2026-09-28")["status"] == "building"


def test_when_no_score_source_works_the_report_says_so_and_gives_no_reasons(store):
    add_evaluation(store, SOCCER, "Over/Under 2.5", "Over 2.5")
    add_bet(store, [soccer_leg()])
    out = run(store, FakeScores(down=True))
    text = "\n".join(out.summary)
    assert out.status == "degraded" and "no score source could be reached" in text
    assert "cause unverified" in text


def test_a_bet_outside_the_week_a_dry_run_bet_and_a_booked_bet_are_ignored(store):
    old = "soccer:england premier league:20260915T1900Z:arsenal:chelsea"
    add_evaluation(store, old, "Over/Under 2.5", "Over 2.5")
    add_bet(store, [soccer_leg(old)], token="old")
    add_bet(store, [soccer_leg()], state="BOOKED", token="booked")
    out = run(store)
    assert "Bets you placed this week: 0" in "\n".join(out.summary)


# ---------------------------------------------------------------- recommendations and decisions
def seed_history(store, n=40, hit=0.4):
    for i in range(n):
        store._conn.execute(
            "INSERT INTO leg_settlements (leg_id, week_key, placed, cut, sport, kind, model, rule, analyst, small_sample, probability, odds, "
            "verdict, ev, settled_at_utc) VALUES (?, ?, 1, 'soccer|goals over', 'soccer', 'goals over', 'poisson-xg-v1', '', 0, 0, 0.8, 1.5, ?, 0.1, 'x')",
            (f"old:{i}", f"2026-W{37 + i % 3}", "win" if i < n * hit else "loss"))


def test_a_miscalibrated_cut_in_the_pooled_history_becomes_a_numbered_recommendation(store):
    seed_history(store)
    add_evaluation(store, SOCCER, "Over/Under 2.5", "Over 2.5")
    add_bet(store, [soccer_leg()])
    out = run(store)
    assert out.recommendations and out.recommendations[0][0] == "2026-W40-01"
    text = out.recommendations[0][1]
    assert "over-confident" in text and "Proposed change:" in text and "Rollback:" in text and "Expected effect:" in text
    row = store.get_recommendation("2026-W40-01")
    assert row["status"] == "new" and row["topic_key"] == "calibration:soccer|goals over" and json.loads(row["body_json"])["n"] >= 40


def rec(n=40, topic="calibration:soccer|goals over"):
    return Recommendation(topic, "t", "c", "f", (), "p", "e", "r", "S", "strong", "v", "rb", n, 3)


def seed_recommendation(store, rec_id, week, status, n=40, ignored_n=None, resurface=None):
    store._conn.execute("INSERT OR IGNORE INTO weekly_reviews (week_start, week_key, status, built_at_utc) VALUES (?, ?, 'posted', 'x')", (week, rec_id[:8]))
    store.save_recommendation(rec_id, week, 1, "calibration:soccer|goals over", "t", "c", "{}", n, 3, "strong", NOW)
    store.set_recommendation_status(rec_id, ("new",), status, ignored_n=ignored_n, resurface_after=resurface, now=NOW)


def test_an_ignored_finding_stays_out_until_its_evidence_grows(store):
    seed_recommendation(store, "2026-W39-01", "2026-09-21", "ignore", n=40, ignored_n=40)
    chosen, suppressed = _select(store, [rec(n=44)], "2026-W40", NOW)
    assert chosen == [] and suppressed["ignored"] == 1
    chosen, _ = _select(store, [rec(n=70)], "2026-W40", NOW)
    assert len(chosen) == 1 and "re-raised" in chosen[0][1]


def test_implement_later_and_done_decisions_shape_the_next_post(store):
    seed_recommendation(store, "2026-W39-01", "2026-09-21", "implement")
    assert _select(store, [rec()], "2026-W40", NOW)[1]["awaiting"] == 1
    store._conn.execute("DELETE FROM recommendations")
    seed_recommendation(store, "2026-W39-01", "2026-09-21", "later", resurface="2026-10-19")
    assert _select(store, [rec()], "2026-W40", NOW)[1]["later"] == 1
    due = _select(store, [rec()], "2026-W42", datetime(2026, 10, 20, 8, 30, tzinfo=timezone.utc))
    assert due[0] and "revisited" in due[0][0][1]
    store._conn.execute("DELETE FROM recommendations")
    seed_recommendation(store, "2026-W39-01", "2026-09-21", "done")
    assert _select(store, [rec()], "2026-W40", NOW)[1]["done"] == 1


def test_at_most_five_recommendations_are_posted(store):
    chosen, suppressed = _select(store, [rec(topic=f"t{i}") for i in range(8)], "2026-W40", NOW)
    assert len(chosen) == 5 and suppressed["over the cap"] == 3


def test_a_decision_is_idempotent_and_records_when_it_was_made(store):
    seed_recommendation(store, "2026-W40-01", "2026-09-28", "new")
    store._conn.execute("UPDATE recommendations SET status = 'new', decided_at_utc = NULL WHERE id = '2026-W40-01'")
    assert store.set_recommendation_status("2026-W40-01", ("new", "later", "ignore"), "implement", now=NOW) == 1
    assert store.set_recommendation_status("2026-W40-01", ("new",), "implement", now=NOW) == 0   # a second tap changes nothing
    assert store.get_recommendation("2026-W40-01")["decided_at_utc"] is not None


def test_games_not_finished_yet_are_timing_not_a_coverage_gap(store):
    class NotFinished:
        async def final_score(self, match_id):
            return None, "the game was not on the scoreboard (or not finished)"

    add_evaluation(store, SOCCER, "Over/Under 2.5", "Over 2.5")
    add_bet(store, [soccer_leg()])
    out = run(store, NotFinished())
    assert not any("could not be settled" in text for _, text in out.recommendations)
    assert "cause unverified" in "\n".join(out.summary)
