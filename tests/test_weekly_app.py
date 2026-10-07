"""The weekly review wired into the bot: schedule, restart catch-up, posting (dry and real) and the decision buttons. No network."""
import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from telegram.ext import ApplicationBuilder

import app as bot
from predictions.core.settle import FinalScore
from predictions.flow.store import EvaluationStore
from predictions.flow.weekly import DEFAULT_WEEKLY_TIME, WEEKLY_JOB_NAME, weekly_catch_up_due
from predictions.telegram import weeklyview

NOW = datetime(2026, 10, 5, 8, 30, tzinfo=timezone.utc)   # Monday 09:30 in Lagos


@pytest.fixture
def store(tmp_path, monkeypatch):
    with EvaluationStore.open(tmp_path / "db.sqlite", lambda s: s) as s:
        monkeypatch.setattr(bot, "STORE", s)
        monkeypatch.setattr(bot, "WEEKLY_DIR", tmp_path / "weekly")
        yield s


class FakeBot:
    def __init__(self):
        self.sent, self.documents = [], []

    async def send_message(self, chat_id, text, reply_markup=None, **kw):
        self.sent.append((text, reply_markup))
        return SimpleNamespace(message_id=len(self.sent))

    async def send_document(self, chat_id, document, filename, **kw):
        self.documents.append((filename, document.read().decode("utf-8")))


class NoScores:
    async def final_score(self, match_id):
        return None, "no free score source"


def test_the_monday_job_is_scheduled_once_a_week_with_a_short_grace(monkeypatch):
    monkeypatch.setattr(bot, "WEEKLY_ENABLED", True)
    application = ApplicationBuilder().token("123456:TEST-TOKEN-NOT-REAL").build()
    job = bot.schedule_weekly_review(application)
    assert job is not None and job.name == WEEKLY_JOB_NAME and job.job.misfire_grace_time <= 60 and job.job.coalesce
    trigger = str(job.job.trigger)
    assert "day_of_week='mon'" in trigger or "day_of_week='1'" in trigger.replace("mon", "1")
    assert "hour='9'" in trigger and "minute='30'" in trigger


def test_the_job_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(bot, "WEEKLY_ENABLED", False)
    assert bot.schedule_weekly_review(ApplicationBuilder().token("123456:TEST-TOKEN-NOT-REAL").build()) is None


# ---------------------------------------------------------------- catch-up after a missed Monday
def test_a_missed_monday_is_made_up_once_within_the_catch_up_days(store):
    def due(day, hour=8):
        return weekly_catch_up_due(store, datetime(2026, 10, day, hour, 0, tzinfo=timezone.utc), DEFAULT_WEEKLY_TIME, 3)
    assert not due(5, hour=7)                    # Monday before 09:30 Lagos: the scheduled run is still to come
    assert due(5, hour=9) and due(6) and due(8)  # Monday after the time, Tuesday, Thursday
    assert not due(9) and not due(10)            # Friday and the weekend: too late, send Weekly by hand
    store._conn.execute("INSERT INTO weekly_reviews (week_start, week_key, status, built_at_utc) VALUES ('2026-09-28', '2026-W40', 'posted', 'x')")
    assert not due(6)
    store._conn.execute("UPDATE weekly_reviews SET status = 'building', built_at_utc = ? WHERE week_start = '2026-09-28'", (NOW.isoformat(),))
    assert not weekly_catch_up_due(store, datetime(2026, 10, 5, 8, 40, tzinfo=timezone.utc), DEFAULT_WEEKLY_TIME, 3)   # building 10 minutes ago
    assert weekly_catch_up_due(store, datetime(2026, 10, 5, 8, 50, tzinfo=timezone.utc), DEFAULT_WEEKLY_TIME, 3)       # stale: retry


# ---------------------------------------------------------------- posting
def test_a_real_post_sends_the_summary_and_the_footer_saves_the_file_locally_and_marks_the_week_posted(store, monkeypatch):
    monkeypatch.setattr(bot, "EspnScores", NoScores)
    fake = FakeBot()
    assert asyncio.run(bot.post_weekly(fake, week_start=datetime(2026, 9, 28).date())) == ""
    assert "WEEKLY REVIEW 2026-W40" in fake.sent[0][0]
    assert fake.documents == []   # the bot never sends files (a guard test pins that); the full text is saved locally
    assert store.weekly_review("2026-09-28")["status"] in ("posted", "degraded")
    assert (bot.WEEKLY_DIR / "APPROVED.md").exists() and (bot.WEEKLY_DIR / "2026-W40.md").exists()
    again = asyncio.run(bot.post_weekly(fake, week_start=datetime(2026, 9, 28).date()))
    assert "already built" in again


def test_a_dry_run_sends_without_buttons_and_saves_nothing(store, monkeypatch):
    monkeypatch.setattr(bot, "EspnScores", NoScores)
    fake = FakeBot()
    assert asyncio.run(bot.post_weekly(fake, week_start=datetime(2026, 9, 28).date(), dry=True)) == ""
    assert fake.sent[0][0].startswith("DRY RUN") and all(markup is None for _, markup in fake.sent) and fake.documents == []
    assert store.weekly_review("2026-09-28") is None and not bot.WEEKLY_DIR.exists()


def test_a_second_build_while_one_runs_is_refused(store, monkeypatch):
    monkeypatch.setattr(bot, "EspnScores", NoScores)
    assert bot.WEEKLY_GUARD.try_begin()
    try:
        assert asyncio.run(bot.post_weekly(FakeBot())) == "A weekly review is already being built."
    finally:
        bot.WEEKLY_GUARD.end()


# ---------------------------------------------------------------- the decision buttons
class FakeQuery:
    def __init__(self, data, text="[2026-W40-01] Soccer: over-confident\nlong body"):
        self.data, self.answers, self.edits, self.replies = data, [], [], []
        self.message = SimpleNamespace(text=text, reply_text=self._reply)

    async def answer(self, text="", **kw):
        self.answers.append(text)

    async def edit_message_text(self, text, reply_markup=None, **kw):
        self.edits.append((text, reply_markup))

    async def _reply(self, text, **kw):
        self.replies.append(text)


def seed(store):
    store._conn.execute("INSERT INTO weekly_reviews (week_start, week_key, status, built_at_utc) VALUES ('2026-09-28', '2026-W40', 'posted', 'x')")
    body = {"finding": "30 of 40 lost", "proposed_change": "raise the floor", "n": 40, "weeks": 3, "confidence": "strong"}
    store.save_recommendation("2026-W40-01", "2026-09-28", 1, "calibration:x", "Soccer: over-confident", "market calibration", json.dumps(body),
                              40, 3, "strong", NOW)


def tap(data):
    query = FakeQuery(data)
    asyncio.run(bot.weekly_callback(SimpleNamespace(callback_query=query), None))
    return query


def test_implement_updates_the_message_the_database_and_the_developer_file(store):
    seed(store)
    query = tap(weeklyview.w_encode("2026-W40", 1, "i"))
    assert store.get_recommendation("2026-W40-01")["status"] == "implement"
    assert query.answers == ["Recorded: IMPLEMENT."] and "=> IMPLEMENT" in query.edits[0][0] and "long body" not in query.edits[0][0]
    assert "raise the floor" in (bot.WEEKLY_DIR / "APPROVED.md").read_text(encoding="utf-8")


def test_a_repeat_tap_and_an_unknown_button_change_nothing(store):
    seed(store)
    tap(weeklyview.w_encode("2026-W40", 1, "i"))
    again = tap(weeklyview.w_encode("2026-W40", 1, "i"))
    assert again.answers == ["Already recorded."] and again.edits == []
    assert tap("w:bad").answers == ["That button is not valid any more."]


def test_details_replies_with_the_full_text_and_changes_nothing(store):
    seed(store)
    query = tap(weeklyview.w_encode("2026-W40", 1, "d"))
    assert any("Proposed change: raise the floor" in r for r in query.replies) and query.edits == []
    assert store.get_recommendation("2026-W40-01")["status"] == "new"
