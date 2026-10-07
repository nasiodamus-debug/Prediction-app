"""The owner's buttons and files for the weekly review: callback data, idempotent decisions, Decisions and APPROVED.md."""
import json
from datetime import datetime, timezone

import pytest

from predictions.flow.store import EvaluationStore
from predictions.telegram.weeklyview import (CALLBACK_LIMIT, apply_decision, approved_markdown, decided_text, decisions_text,
                                              rec_keyboard, set_done, set_note, split_id, w_decode, w_encode, write_approved)

NOW = datetime(2026, 10, 5, 8, 41, tzinfo=timezone.utc)


@pytest.fixture
def store(tmp_path):
    with EvaluationStore.open(tmp_path / "db.sqlite", lambda s: s) as s:
        s._conn.execute("INSERT INTO weekly_reviews (week_start, week_key, status, built_at_utc) VALUES ('2026-09-28', '2026-W40', 'posted', 'x')")
        body = {"title": "t", "category": "market calibration", "confidence": "strong", "n": 40, "weeks": 3, "finding": "30 of 40 lost",
                "proposed_change": "raise the floor", "expected_effect": "fewer picks", "risk": "lost picks", "effort": "S",
                "validation": "recheck", "rollback": "revert", "evidence": ["placed:1"]}
        s.save_recommendation("2026-W40-01", "2026-09-28", 1, "calibration:x", "Soccer goals: over-confident", "market calibration",
                              json.dumps(body), 40, 3, "strong", NOW)
        yield s


def test_callback_data_round_trips_and_stays_short():
    data = w_encode("2026-W40", 1, "i")
    assert data == "w:2026W40:01:i" and len(data.encode()) <= CALLBACK_LIMIT and w_decode(data) == ("2026-W40", 1, "i")
    assert w_decode("w:bad") is None and w_decode("s:2026W40:01:i") is None and w_decode(None) is None
    with pytest.raises(ValueError):
        w_encode("2026-W40", 1, "x")
    assert split_id("2026-W40-03") == ("2026-W40", 3) and split_id("nope") is None
    assert all(len(d.encode()) <= CALLBACK_LIMIT for row in rec_keyboard("2026-W40", 5) for _, d in row)


def test_a_decision_is_recorded_once_and_a_repeat_tap_changes_nothing(store):
    first = apply_decision(store, "2026-W40", 1, "i", NOW)
    assert first.changed and first.status == "implement" and "IMPLEMENT" in first.notice and first.keyboard
    second = apply_decision(store, "2026-W40", 1, "i", NOW)
    assert not second.changed and second.notice == "Already recorded."
    assert store.get_recommendation("2026-W40-01")["status"] == "implement"


def test_later_sets_a_return_date_ignore_remembers_the_evidence_and_undo_restores(store):
    later = apply_decision(store, "2026-W40", 1, "l", NOW)
    row = store.get_recommendation("2026-W40-01")
    assert row["status"] == "later" and row["resurface_after"] == "2026-10-19" and "2026-10-19" in later.notice
    apply_decision(store, "2026-W40", 1, "g", NOW)
    row = store.get_recommendation("2026-W40-01")
    assert row["status"] == "ignore" and row["ignored_n"] == 40
    undone = apply_decision(store, "2026-W40", 1, "u", NOW)
    assert undone.status == "new" and apply_decision(store, "2026-W40", 1, "u", NOW).notice.startswith("Already recorded")


def test_details_shows_the_full_text_and_unknown_or_done_items_are_handled(store):
    details = apply_decision(store, "2026-W40", 1, "d", NOW).details
    for needle in ("Finding: 30 of 40 lost", "Proposed change: raise the floor", "Rollback: revert", "Evidence ids: placed:1"):
        assert needle in details
    assert "not on file" in apply_decision(store, "2026-W40", 9, "i", NOW).notice
    apply_decision(store, "2026-W40", 1, "i", NOW)
    assert set_done(store, "2026-W40-01", NOW) == "2026-W40-01 marked done."
    assert "already marked done" in apply_decision(store, "2026-W40", 1, "g", NOW).notice
    assert "not waiting" in set_done(store, "2026-W40-01", NOW)


def test_decided_text_keeps_the_title_and_says_what_was_decided():
    text = decided_text("[2026-W40-01] Soccer goals: over-confident\nlong body...", "implement", NOW)
    assert text.startswith("[2026-W40-01] Soccer goals") and "=> IMPLEMENT" in text and "09:41 Lagos" in text and "long body" not in text


def test_decisions_lists_what_is_approved_and_the_file_matches(store, tmp_path):
    assert "Nothing is waiting" in decisions_text(store)
    apply_decision(store, "2026-W40", 1, "i", NOW)
    assert set_note(store, "2026-W40-01", "do it after the weekend") == "Note saved on 2026-W40-01."
    text = decisions_text(store)
    assert "Waiting to be implemented: 1" in text and "raise the floor" in text and "do it after the weekend" in text
    path = write_approved(store, tmp_path / "weekly")
    content = path.read_text(encoding="utf-8")
    assert content == approved_markdown(store) and "2026-W40-01" in content and "Proposed change: raise the floor" in content
    assert not (tmp_path / "weekly" / "APPROVED.md.tmp").exists()
    assert "2026-W40-01" in decisions_text(store, "all")
