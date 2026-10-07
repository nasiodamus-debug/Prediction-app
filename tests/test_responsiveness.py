"""AD-22: the bot stays responsive while a long request runs.

The owner's live symptoms were "This list changed" on Final confirm and "Query is too old" in the console. Root cause:
python-telegram-bot handles updates one after the other, and the Matchday handler awaited the whole evaluation, so every
button press waited in the queue behind it, was answered after Telegram's window closed, and was then judged against a
list that had since been replaced. These tests run the REAL handler chain through PTB's real (sequential) update queue."""
import asyncio
import logging
from types import SimpleNamespace

import pytest
from telegram import Update, User
from telegram.error import BadRequest
from telegram.ext import ApplicationBuilder, ExtBot

import app as bot
from fakes_bookmaker import NOW, two_pick_setup
from handled import drop_ack
from predictions.flow.evaluate import GateResult
from predictions.flow.jobs import BackgroundJobs
from predictions.flow.session import Action, SessionManager, State
from predictions.flow.store import EvaluationStore
from predictions.telegram.slate import list_screen
from test_epic5_app import live_chain, to_stake  # noqa: F401  (the fixture, used by name)
from test_final_list import find, message

OWNER = bot.OWNER_ID


def gate_of(passed):
    return GateResult(tuple(passed), (), 2, 0, {})


class Harness:
    """A real Application with the real handlers, fed through its real update queue, with Telegram faked."""

    def __init__(self, monkeypatch, tmp_path):
        self.log = []          # ("answer" | "edit" | "send" | "slow-start" | "slow-end", text)
        self.release = None    # an asyncio.Event made inside the loop
        self.n = 0
        self.fail_matchday = None

        async def no_network(this):
            this._bot_user = User(id=1, first_name="TestBot", is_bot=True, username="test_bot")

        async def send_message(this, chat_id, text, **kwargs):
            self.log.append(("send", text))

        async def edit_message_text(this, text, *args, **kwargs):
            self.log.append(("edit", text))

        async def answer_callback_query(this, callback_query_id, text=None, show_alert=None, **kwargs):
            self.log.append(("answer", text))

        monkeypatch.setattr(ExtBot, "initialize", no_network)
        monkeypatch.setattr(ExtBot, "send_message", send_message)
        monkeypatch.setattr(ExtBot, "edit_message_text", edit_message_text)
        monkeypatch.setattr(ExtBot, "answer_callback_query", answer_callback_query)
        self.evals, _ = two_pick_setup()

        async def slow_matchday(argument, source, sanitize, **kwargs):
            self.log.append(("slow-start", argument))
            await self.release.wait()
            self.log.append(("slow-end", argument))
            if self.fail_matchday is not None:
                raise self.fail_matchday
            kwargs["on_result"](gate_of(self.evals))
            return ["Matchday fixtures ..."]

        monkeypatch.setattr(bot, "run_matchday", slow_matchday)
        monkeypatch.setattr(bot.ClaudeAnalyst, "from_environment", classmethod(lambda cls, mode="full": None))
        monkeypatch.setattr(bot, "SESSIONS", SessionManager(clock=lambda: NOW))
        monkeypatch.setattr(bot, "JOBS", BackgroundJobs())
        monkeypatch.setattr(bot, "_recent_alerts", {})
        self.store = EvaluationStore.open(tmp_path / "t.db", bot.redact)
        monkeypatch.setattr(bot, "STORE", self.store)
        self.application = ApplicationBuilder().token("123456:TEST-TOKEN-NOT-REAL").build()
        bot.register_handlers(self.application)

    def text_update(self, text):
        self.n += 1
        return Update.de_json({"update_id": self.n, "message": message(OWNER, OWNER, text)}, self.application.bot)

    def press_update(self, data):
        self.n += 1
        return Update.de_json({"update_id": self.n, "callback_query": {
            "id": str(self.n), "chat_instance": "x", "data": data,
            "from": {"id": OWNER, "is_bot": False, "first_name": "T"}, "message": message(OWNER, OWNER, "list")}},
            self.application.bot)

    def kinds(self):
        return [kind for kind, _ in self.log]

    async def until(self, predicate, seconds=5.0):
        end = asyncio.get_running_loop().time() + seconds
        while not predicate():
            assert asyncio.get_running_loop().time() < end, f"timed out; log={self.log}"
            await asyncio.sleep(0.005)

    def run(self, scenario):
        async def go():
            self.release = asyncio.Event()
            async with self.application:
                await self.application.start()  # the real update queue: one update at a time
                try:
                    await scenario(self)
                finally:
                    self.release.set()
                    await bot.JOBS.wait()
                    await self.application.stop()

        asyncio.run(go())


@pytest.fixture
def harness(monkeypatch, tmp_path):
    h = Harness(monkeypatch, tmp_path)
    yield h
    h.store.close()


def open_slate_and_delete_button(h):
    session = bot.SESSIONS.start(h.evals, origin="matchday")
    return find([(label, data) for row in list_screen(session).buttons for label, data in row], "1 Delete")


async def put(h, update):
    await h.application.update_queue.put(update)


# ---------------------------------------------------------------- the fix
def test_a_press_is_answered_and_applied_while_a_slow_matchday_runs(harness):
    h = harness
    button = open_slate_and_delete_button(h)
    version_before = bot.SESSIONS.active.state_version

    async def scenario(h):
        await put(h, h.text_update("Matchday 10 October"))
        await h.until(lambda: "slow-start" in h.kinds())          # the evaluation is now running and NOT finished
        await put(h, h.press_update(button))
        await h.until(lambda: "answer" in h.kinds())              # the press is answered ...
        assert "slow-end" not in h.kinds()                        # ... while the evaluation is still running
        assert bot.SESSIONS.active.state_version == version_before + 1   # and applied (the game was deleted)
        assert "edit" in h.kinds() and h.kinds().index("answer") < h.kinds().index("edit")
        h.release.set()                                           # the late evaluation result still arrives
        await h.until(lambda: any(k == "send" and t.startswith("You already have a slate") for k, t in h.log))

    h.run(scenario)
    texts = [t for k, t in h.log if k == "send"]
    assert texts[0].startswith("Analysing Matchday for 10 October")       # the short, honest acknowledgement first
    assert "Matchday fixtures ..." in texts
    assert [k for k, _ in h.log if k == "answer"] == ["answer"]            # exactly one answer for the one press


def test_reproduction_an_inline_handler_holds_every_press_behind_it(harness, monkeypatch):
    """The OLD behaviour: the handler awaits the whole request. The press is only answered after the slow work finishes."""
    h = harness

    async def inline(update, kind, word, argument, work):
        await work(update, argument)

    monkeypatch.setattr(bot, "start_long_request", inline)
    button = open_slate_and_delete_button(h)

    async def scenario(h):
        await put(h, h.text_update("Matchday 10 October"))
        await h.until(lambda: "slow-start" in h.kinds())
        await put(h, h.press_update(button))
        await asyncio.sleep(0.2)                                  # a long time for a button press
        assert "answer" not in h.kinds()                          # still queued behind the handler: this IS the bug
        h.release.set()
        await h.until(lambda: "answer" in h.kinds())
        assert h.kinds().index("slow-end") < h.kinds().index("answer")

    h.run(scenario)
    # and by then the press belongs to a list that the finished handler replaced or asked about
    assert h.log[-1][0] in ("send", "edit", "answer")


def test_a_second_request_of_the_same_kind_gets_a_still_working_reply(harness):
    h = harness

    async def scenario(h):
        await put(h, h.text_update("Matchday 10 October"))
        await h.until(lambda: "slow-start" in h.kinds())
        await put(h, h.text_update("Matchday 11 October"))
        await h.until(lambda: any(k == "send" and "still running" in t for k, t in h.log))
        assert [t for k, t in h.log if k == "slow-start"] == [" 10 October"]   # never a second run
        assert bot.JOBS.active_keys() == ["MATCHDAY"]

    h.run(scenario)


def test_a_failing_background_request_reports_a_sanitized_error_and_the_bot_keeps_working(harness):
    h = harness
    h.fail_matchday = RuntimeError("boom at C:\\Users\\hp\\secret\\file.py")
    button = open_slate_and_delete_button(h)

    async def scenario(h):
        await put(h, h.text_update("Matchday 10 October"))
        h.release.set()
        await h.until(lambda: any(k == "send" and "Matchday failed" in t for k, t in h.log))
        failure = next(t for k, t in h.log if k == "send" and "Matchday failed" in t)
        assert "secret" not in failure and "C:\\" not in failure and "Nothing was placed" in failure
        await put(h, h.press_update(button))                      # the bot is alive and answers
        await h.until(lambda: "answer" in h.kinds())
        assert not bot.JOBS.running("MATCHDAY")                   # and a new request may start

    h.run(scenario)


def test_shutdown_cancels_a_running_request_cleanly(harness):
    h = harness

    async def scenario(h):
        await put(h, h.text_update("Matchday 10 October"))
        await h.until(lambda: "slow-start" in h.kinds())
        await bot.stop_background_jobs(h.application)
        assert bot.JOBS.active_keys() == [] and "slow-end" not in h.kinds()

    h.run(scenario)


def test_the_background_runner_cancels_and_reports_failures_by_itself():
    seen = []

    async def go():
        jobs = BackgroundJobs(clock=iter(range(0, 1000, 2)).__next__)
        gate = asyncio.Event()

        async def waits():
            await gate.wait()

        async def breaks():
            raise ValueError("x")

        async def report(error):
            seen.append(type(error).__name__)

        assert jobs.start("a", waits) is not None and jobs.start("a", waits) is None    # single flight
        await jobs.cancel_all()
        assert not jobs.running("a")
        jobs.start("b", breaks, "b job", report)
        await jobs.wait()
        assert seen == ["ValueError"]

    asyncio.run(go())


def test_the_long_request_logs_how_long_it_took(harness, caplog):
    h = harness
    caplog.set_level(logging.INFO)

    async def scenario(h):
        await put(h, h.text_update("Matchday 10 October"))
        await h.until(lambda: "slow-start" in h.kinds())
        h.release.set()
        await bot.JOBS.wait()

    h.run(scenario)
    assert any("Matchday request took" in r.getMessage() and r.getMessage().endswith(" s") for r in caplog.records)


# ---------------------------------------------------------------- answer before slow work
def test_a_press_is_answered_before_any_sqlite_write(harness):
    h = harness
    button = open_slate_and_delete_button(h)
    real_save = h.store.save_session

    def save_session(*args, **kwargs):
        h.log.append(("sqlite", "save_session"))
        return real_save(*args, **kwargs)

    bot.attach_store(bot.SESSIONS, h.store)          # the slate is saved after every step, as in the real app
    h.store.save_session = save_session

    async def scenario(h):
        await put(h, h.press_update(button))
        await h.until(lambda: "edit" in h.kinds())

    h.run(scenario)
    kinds = h.kinds()
    assert kinds.index("answer") < kinds.index("sqlite")      # the save waited for the answer ...
    assert "sqlite" in kinds                                  # ... but still happened (every step is saved)


def test_final_confirm_is_answered_before_the_registry_read_the_bookmaker_or_any_save(live_chain, monkeypatch):
    chain = live_chain
    to_stake(chain)
    chain.text(OWNER, "500")
    button = chain.buttons(-1)[0][1]
    order = []
    real_find, real_save = chain.store.find_blocking_legs, chain.store.save_session

    def find_blocking_legs(identities):
        order.append("sqlite-registry")
        return real_find(identities)

    def save_session(*args, **kwargs):
        order.append("sqlite-save")
        return real_save(*args, **kwargs)

    chain.store.find_blocking_legs, chain.store.save_session = find_blocking_legs, save_session
    bot.attach_store(bot.SESSIONS, chain.store)      # the slate is saved after every step, as in the real app
    calls_before = len(chain.fake.calls)

    async def answer(this, callback_query_id, text=None, show_alert=None, **kwargs):
        order.append(("answer", len(chain.fake.calls) - calls_before))
        chain.answers.append((text, show_alert))

    monkeypatch.setattr(ExtBot, "answer_callback_query", answer)
    chain.press(OWNER, button)
    assert order[0] == ("answer", 0)                                      # nothing slow came first, no bookmaker call yet
    assert "sqlite-registry" in order and "sqlite-save" in order
    assert [c[0] for c in chain.fake.calls].count("place") == 1


def test_a_late_answer_that_telegram_refuses_never_stops_the_work(harness, monkeypatch):
    h = harness
    button = open_slate_and_delete_button(h)

    async def too_old(this, callback_query_id, text=None, show_alert=None, **kwargs):
        raise BadRequest("Query is too old and response timeout expired or query id is invalid")

    monkeypatch.setattr(ExtBot, "answer_callback_query", too_old)
    version_before = bot.SESSIONS.active.state_version

    async def scenario(h):
        await put(h, h.press_update(button))
        await h.until(lambda: "edit" in h.kinds())

    h.run(scenario)
    assert bot.SESSIONS.active.state_version == version_before + 1
