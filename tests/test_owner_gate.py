"""Story 1.1: owner-only gate.

Values below are test-only. Real config.env values are never needed: the
environment is set before app.py is imported, and load_dotenv does not
override variables that are already set.
"""
import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

OWNER = 111222333
STRANGER = 999888777

os.environ["OWNER_ID"] = str(OWNER)
os.environ["TELEGRAM_BOT_TOKEN"] = "123456:TEST-TOKEN-NOT-REAL"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as bot  # noqa: E402
from telegram import Update, User  # noqa: E402
from telegram.ext import ApplicationBuilder, ApplicationHandlerStop, ExtBot, TypeHandler  # noqa: E402


# ---------- OWNER_ID parsing ----------

@pytest.mark.parametrize("raw,expected", [("123", 123), ("  456  ", 456), ("111222333", 111222333)])
def test_parse_owner_id_accepts_positive_whole_numbers(raw, expected):
    assert bot.parse_owner_id(raw) == expected


@pytest.mark.parametrize("raw", [None, "", "   ", "abc", "-5", "0", "12.5", "1 2", "١٢٣"])
def test_parse_owner_id_rejects_bad_values(raw):
    with pytest.raises(ValueError):
        bot.parse_owner_id(raw)


def test_parse_owner_id_error_never_echoes_the_value():
    with pytest.raises(ValueError) as err:
        bot.parse_owner_id("secret-value-42x")
    assert "secret-value-42x" not in str(err.value)


# ---------- is_owner_update ----------

def fake_update(user_id, chat_id, chat_type="private", has_user=True, has_chat=True):
    return SimpleNamespace(
        effective_user=SimpleNamespace(id=user_id) if has_user else None,
        effective_chat=SimpleNamespace(id=chat_id, type=chat_type) if has_chat else None,
    )


def test_owner_in_private_chat_is_accepted():
    assert bot.is_owner_update(fake_update(OWNER, OWNER), OWNER)


@pytest.mark.parametrize("update", [
    fake_update(STRANGER, STRANGER),                 # another account
    fake_update(OWNER, -100123, chat_type="group"),  # owner, but in a group
    fake_update(OWNER, -100123, chat_type="supergroup"),
    fake_update(OWNER, OWNER, chat_type="channel"),
    fake_update(STRANGER, OWNER),                    # mismatched sender and chat
    fake_update(OWNER, STRANGER),
    fake_update(OWNER, OWNER, has_user=False),       # update types with no sender
    fake_update(OWNER, OWNER, has_chat=False),
])
def test_everyone_else_is_rejected(update):
    assert not bot.is_owner_update(update, OWNER)


# ---------- the gate handler ----------

def run_gate(update):
    return asyncio.run(bot.owner_gate(update, None))


def test_gate_lets_the_owner_through():
    update = Update.de_json({"update_id": 1, "message": message(OWNER, OWNER, "private", "hi")}, None)
    assert run_gate(update) is None


def test_gate_drops_a_stranger():
    update = Update.de_json({"update_id": 2, "message": message(STRANGER, STRANGER, "private", "hi")}, None)
    with pytest.raises(ApplicationHandlerStop):
        run_gate(update)


def test_gate_drops_things_that_are_not_updates():
    with pytest.raises(ApplicationHandlerStop):
        run_gate(object())


# ---------- wiring: the real handler chain ----------

def message(user_id, chat_id, chat_type, text, command=False):
    data = {
        "message_id": 1,
        "date": 1700000000,
        "chat": {"id": chat_id, "type": chat_type},
        "from": {"id": user_id, "is_bot": False, "first_name": "T"},
        "text": text,
    }
    if command:
        data["entities"] = [{"type": "bot_command", "offset": 0, "length": len(text)}]
    return data


@pytest.fixture
def wired(monkeypatch):
    """A real Application with the real gate, and recorders in place of the real handlers."""
    calls = []

    async def rec_start(update, context):
        calls.append("start")

    async def rec_text(update, context):
        calls.append("text")

    async def rec_callback(update, context):
        calls.append("callback")

    monkeypatch.setattr(bot, "start_cmd", rec_start)
    monkeypatch.setattr(bot, "handle_message", rec_text)
    monkeypatch.setattr(bot, "slate_callback", rec_callback)

    async def no_network(self):
        """Application.initialize() would call Telegram's getMe; keep the tests offline."""
        self._bot_user = User(id=1, first_name="TestBot", is_bot=True, username="test_bot")

    monkeypatch.setattr(ExtBot, "initialize", no_network)
    application = ApplicationBuilder().token("123456:TEST-TOKEN-NOT-REAL").build()
    bot.register_handlers(application)
    return application, calls


def feed(application, payload):
    update = Update.de_json({"update_id": 10, **payload}, application.bot)

    async def go():
        await application.initialize()
        await application.process_update(update)

    asyncio.run(go())


def test_gate_is_registered_first(wired):
    application, _ = wired
    assert min(application.handlers) == -1
    first = application.handlers[-1][0]
    assert isinstance(first, TypeHandler) and first.callback is bot.owner_gate


def test_stranger_start_command_gets_no_reaction(wired):
    application, calls = wired
    feed(application, {"message": message(STRANGER, STRANGER, "private", "/start", command=True)})
    assert calls == []


def test_stranger_text_and_button_get_no_reaction(wired):
    application, calls = wired
    feed(application, {"message": message(STRANGER, STRANGER, "private", "Matchday 20 September")})
    feed(application, {"callback_query": {
        "id": "1", "chat_instance": "x", "data": "EXECUTE_BET",
        "from": {"id": STRANGER, "is_bot": False, "first_name": "T"},
        "message": message(STRANGER, STRANGER, "private", "panel"),
    }})
    assert calls == []


def test_owner_in_a_group_gets_no_reaction(wired):
    application, calls = wired
    feed(application, {"message": message(OWNER, -100123, "group", "/start", command=True)})
    assert calls == []


def test_owner_in_private_chat_still_works(wired):
    application, calls = wired
    feed(application, {"message": message(OWNER, OWNER, "private", "/start", command=True)})
    feed(application, {"message": message(OWNER, OWNER, "private", "Matchday 20 September")})
    feed(application, {"callback_query": {
        "id": "2", "chat_instance": "x", "data": "s:x",
        "from": {"id": OWNER, "is_bot": False, "first_name": "T"},
        "message": message(OWNER, OWNER, "private", "panel"),
    }})
    assert calls == ["start", "text", "callback"]


# ---------- startup fails closed ----------

@pytest.mark.parametrize("bad", ["", "notanumber-77", "-1", "0"])
def test_startup_refuses_without_a_valid_owner_id(bad):
    """Importing app.py runs the startup checks. A bad OWNER_ID must stop it."""
    import subprocess

    repo = Path(__file__).resolve().parent.parent
    env = {**os.environ, "OWNER_ID": bad, "TELEGRAM_BOT_TOKEN": "123456:TEST-TOKEN-NOT-REAL"}
    result = subprocess.run([sys.executable, "-c", "import app"], cwd=repo, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "OWNER_ID" in result.stderr
    if len(bad) > 3:  # short values like "0" would match line numbers in the traceback
        assert bad not in result.stderr
