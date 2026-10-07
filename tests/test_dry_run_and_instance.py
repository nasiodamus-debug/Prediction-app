"""Story 1.3: dry run, single instance, startup message, crash alerts."""
import asyncio
import io
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import app as bot
from telegram import Update, User
from telegram.ext import ApplicationBuilder, ExtBot

OWNER = 111222333
TOKEN = "123456:TEST-TOKEN-NOT-REAL"
PASSWORD = "test-password-not-real"
REPO = Path(__file__).resolve().parent.parent


# ---------- DRY_RUN setting ----------

@pytest.mark.parametrize("raw", [None, "", "  ", "true", "TRUE", " True ", "1", "yes", "off", "0", "garbage"])
def test_dry_run_is_on_unless_explicitly_false(raw):
    assert bot.parse_dry_run(raw) is True


@pytest.mark.parametrize("raw", ["false", "FALSE", " False "])
def test_only_false_turns_dry_run_off(raw):
    assert bot.parse_dry_run(raw) is False


# ---------- the old placement script is gone (Epic 5 replaced it) ----------

PICKS = [{"id": 0, "team": "Arsenal", "market": "Over 2.0 Asian Goal Line", "search_term": "Arsenal", "note": "n"}]


def test_the_app_has_no_direct_playwright_use_any_more():
    source = (REPO / "app.py").read_text(encoding="utf-8")
    assert "async_playwright" not in source and ".click()" not in source


def owner_callback(data):
    sent = []

    async def reply_text(message, **kwargs):
        sent.append(message)

    async def edit_message_text(message, **kwargs):
        sent.append(message)

    async def answer(*args, **kwargs):
        return None

    message = SimpleNamespace(reply_text=reply_text)
    query = SimpleNamespace(data=data, answer=answer, edit_message_text=edit_message_text, message=message)
    update = SimpleNamespace(effective_user=SimpleNamespace(id=OWNER), callback_query=query, message=None)
    return update, sent


# ---------- single instance ----------

def test_second_copy_is_refused_and_told_why(tmp_path):
    lock = tmp_path / "bot.lock"
    first = bot.acquire_instance_lock(lock)
    with pytest.raises(bot.AlreadyRunning) as err:
        bot.acquire_instance_lock(lock)
    assert "already running" in str(err.value)
    first.close()
    bot.acquire_instance_lock(lock).close()  # free again once the first copy lets go


def test_lock_is_freed_when_a_copy_crashes(tmp_path):
    lock = tmp_path / "bot.lock"
    script = (
        "import sys, time; sys.path.insert(0, sys.argv[1]); import app; from pathlib import Path;"
        "h = app.acquire_instance_lock(Path(sys.argv[2])); print('locked', flush=True); time.sleep(120)"
    )
    proc = subprocess.Popen([sys.executable, "-c", script, str(REPO), str(lock)],
                            stdout=subprocess.PIPE, text=True, env=os.environ.copy())
    try:
        assert proc.stdout.readline().strip() == "locked"
        with pytest.raises(bot.AlreadyRunning):
            bot.acquire_instance_lock(lock)
    finally:
        proc.kill()
        proc.wait(timeout=30)
    time.sleep(0.2)
    bot.acquire_instance_lock(lock).close()


# ---------- startup message ----------

def fake_application(fail=False):
    sent = []

    async def send_message(chat_id, text, **kwargs):
        if fail:
            raise RuntimeError("Forbidden: bot can't initiate conversation")
        sent.append((chat_id, text))

    return SimpleNamespace(bot=SimpleNamespace(send_message=send_message)), sent


def test_startup_message_says_dry_run_is_on(monkeypatch):
    monkeypatch.setattr(bot, "DRY_RUN", True)
    application, sent = fake_application()
    asyncio.run(bot.announce_startup(application))
    assert sent[0][0] == OWNER
    assert "Bot started" in sent[0][1] and "ON" in sent[0][1] and "no bet will be placed" in sent[0][1]


def test_startup_message_warns_when_dry_run_is_off(monkeypatch):
    monkeypatch.setattr(bot, "DRY_RUN", False)
    application, sent = fake_application()
    asyncio.run(bot.announce_startup(application))
    assert "OFF" in sent[0][1] and "WILL be placed" in sent[0][1]


def test_a_failed_startup_message_does_not_stop_the_bot():
    application, _ = fake_application(fail=True)
    asyncio.run(bot.announce_startup(application))  # must not raise


# ---------- crash alerts through the real handler chain ----------

def message(user_id, chat_id, text):
    return {
        "message_id": 1, "date": 1700000000,
        "chat": {"id": chat_id, "type": "private"},
        "from": {"id": user_id, "is_bot": False, "first_name": "T"},
        "text": text,
    }


@pytest.fixture
def wired(monkeypatch):
    alerts, handled = [], []

    async def crashing_text(update, context):
        if update.message.text == "crash":
            raise RuntimeError(f"boom {TOKEN} {PASSWORD} in {REPO}\\app.py")
        handled.append(update.message.text)

    async def no_network(self):
        self._bot_user = User(id=1, first_name="TestBot", is_bot=True, username="test_bot")

    async def record_send(self, chat_id, text, **kwargs):
        alerts.append((chat_id, text))

    monkeypatch.setattr(bot, "_recent_alerts", {})
    monkeypatch.setattr(bot, "handle_message", crashing_text)
    monkeypatch.setattr(ExtBot, "initialize", no_network)
    monkeypatch.setattr(ExtBot, "send_message", record_send)
    application = ApplicationBuilder().token(TOKEN).build()
    bot.register_handlers(application)
    return application, alerts, handled


def feed(application, user_id, text):
    update = Update.de_json({"update_id": 5, "message": message(user_id, user_id, text)}, application.bot)

    async def go():
        await application.initialize()
        await application.process_update(update)

    asyncio.run(go())


def test_unhandled_error_sends_the_owner_a_clean_short_alert(wired):
    application, alerts, handled = wired
    feed(application, OWNER, "crash")
    assert len(alerts) == 1
    chat_id, text = alerts[0]
    assert chat_id == OWNER and "Unexpected error" in text and "RuntimeError" in text
    for secret in (TOKEN, PASSWORD, str(REPO), "app.py"):
        assert secret not in text
    assert len(text) < 500


def test_the_bot_keeps_working_after_an_error(wired):
    application, alerts, handled = wired
    feed(application, OWNER, "crash")
    feed(application, OWNER, "Matchday 20 September")
    assert handled == ["Matchday 20 September"]


def test_a_stranger_causes_no_error_and_no_alert(wired):
    application, alerts, handled = wired
    feed(application, 999888777, "crash")
    assert alerts == [] and handled == []


# ---------- deferred from Story 1.2: output that bypassed the redactor ----------

@pytest.fixture
def clean_logging(monkeypatch):
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    monkeypatch.setattr(sys, "excepthook", sys.excepthook)  # restored after the test
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in saved[0]:
        root.addHandler(handler)
    root.setLevel(saved[1])


def test_any_handler_gets_redacted_messages(clean_logging):
    bot.configure_logging()
    stream = io.StringIO()
    plain = logging.StreamHandler(stream)  # a handler that does NOT use the redacting formatter
    plain.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("other.library").addHandler(plain)
    try:
        logging.getLogger("other.library").warning("token %s and password %s", TOKEN, PASSWORD)
    finally:
        logging.getLogger("other.library").removeHandler(plain)
    output = stream.getvalue()
    assert "token" in output
    assert TOKEN not in output and PASSWORD not in output


def test_uncaught_exceptions_go_through_the_redacting_logger(clean_logging):
    bot.configure_logging()
    stream = io.StringIO()
    logging.getLogger().handlers[0].setStream(stream)
    assert sys.excepthook is bot._log_uncaught
    try:
        raise RuntimeError(f"startup failed with {PASSWORD}")
    except RuntimeError:
        sys.excepthook(*sys.exc_info())
    output = stream.getvalue()
    assert "Uncaught exception" in output and "startup failed" in output
    assert PASSWORD not in output


def test_playwright_debug_output_is_switched_off(clean_logging, monkeypatch):
    monkeypatch.setenv("DEBUG", "pw:api")
    bot.configure_logging()
    assert "DEBUG" not in os.environ


# ---------- review fixes ----------

def test_startup_message_flags_an_unrecognised_dry_run_value(monkeypatch):
    monkeypatch.setattr(bot, "DRY_RUN_RAW", "no")
    monkeypatch.setattr(bot, "DRY_RUN", bot.parse_dry_run("no"))
    application, sent = fake_application()
    asyncio.run(bot.announce_startup(application))
    assert "not 'true' or 'false'" in sent[0][1] and "stays on" in sent[0][1]


def test_lock_problems_are_not_reported_as_a_second_copy(tmp_path):
    # A folder in place of the lock file cannot be opened; that is a real error, not "already running".
    with pytest.raises(OSError) as err:
        bot.acquire_instance_lock(tmp_path)
    assert not isinstance(err.value, bot.AlreadyRunning)


def test_the_same_error_alert_is_not_repeated_within_a_minute(wired, monkeypatch):
    application, alerts, handled = wired
    clock = [1000.0]
    monkeypatch.setattr(bot.time, "monotonic", lambda: clock[0])
    feed(application, OWNER, "crash")
    feed(application, OWNER, "crash")
    assert len(alerts) == 1
    clock[0] += bot.ALERT_REPEAT_SECONDS + 1
    feed(application, OWNER, "crash")
    assert len(alerts) == 2


def test_long_owner_messages_fit_telegram(monkeypatch):
    application, sent = fake_application()
    asyncio.run(bot.send_to_owner(application.bot, "x" * 10000))
    assert len(sent[0][1]) <= 4000


def test_tracebacks_are_redacted_for_every_handler(clean_logging):
    bot.configure_logging()
    stream = io.StringIO()
    plain = logging.StreamHandler(stream)
    plain.setFormatter(logging.Formatter("%(message)s"))  # no redacting formatter
    logger = logging.getLogger("other.library")
    logger.addHandler(plain)
    try:
        try:
            raise RuntimeError(f"failed with {PASSWORD} and {TOKEN}")
        except RuntimeError:
            logger.exception("it broke")
    finally:
        logger.removeHandler(plain)
    output = stream.getvalue()
    assert "it broke" in output and "RuntimeError" in output
    assert PASSWORD not in output and TOKEN not in output