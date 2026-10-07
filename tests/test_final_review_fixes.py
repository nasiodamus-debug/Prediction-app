"""Final review fixes: result wording, MAX_STAKE, edited messages, ASCII startup print."""
import asyncio
import io
import os

import pytest

import app as bot
from predictions.core.results import LOSS, PENDING, UNKNOWN, WIN, classify_bet_status
from telegram import Update, User
from telegram.ext import ApplicationBuilder, ExtBot

OWNER = 111222333
TOKEN = "123456:TEST-TOKEN-NOT-REAL"


# ---- 1. only the exact words Won / Lost / Running are understood; every other word is UNKNOWN (never a win or a loss)
@pytest.mark.parametrize("word", ["Winner", "Win", "Winning", "Lose", "Losing", "Tie", "Open", "Live", "Status: banana",
                                  "Cashed Out", "Void", "Cancelled", "Half Won", "Settled", ""])
def test_ambiguous_or_unseen_words_are_unknown_never_a_result(word):
    assert classify_bet_status(word) == UNKNOWN


def test_the_three_words_seen_live_are_understood_whatever_the_spacing_or_case():
    assert classify_bet_status(" Won ") == WIN and classify_bet_status("LOST") == LOSS
    assert classify_bet_status("Running") == PENDING and classify_bet_status("  running ") == PENDING



# ---- 2. MAX_STAKE at startup
def test_startup_message_states_the_limit(monkeypatch):
    sent = []

    async def fake_send(bot_, text, **kw):
        sent.append(text)

    async def no_recovery(b):
        return []

    monkeypatch.setattr(bot, "send_to_owner", fake_send)
    monkeypatch.setattr(bot, "recover_after_start", no_recovery)
    asyncio.run(bot.announce_startup(type("A", (), {"bot": object()})()))
    assert "Stake limit" in sent[0] and format(bot.MAX_STAKE, ",f") in sent[0]


# ---- 5. edited messages
def test_an_edited_message_causes_no_error_and_no_reaction(monkeypatch):
    alerts, handled, started = [], [], []

    async def text_handler(update, context):
        handled.append(update)

    async def start_handler(update, context):
        started.append(update)

    async def no_network(self):
        self._bot_user = User(id=1, first_name="T", is_bot=True, username="t_bot")

    async def record_send(self, chat_id, text, **kwargs):
        alerts.append(text)

    monkeypatch.setattr(bot, "_recent_alerts", {})
    monkeypatch.setattr(ExtBot, "initialize", no_network)
    monkeypatch.setattr(ExtBot, "send_message", record_send)
    application = ApplicationBuilder().token(TOKEN).build()
    real = bot.handle_message
    bot.register_handlers(application)  # the REAL handler chain, real handle_message and start_cmd

    def msg(text):
        return {"message_id": 1, "date": 1700000000, "edit_date": 1700000005,
                "chat": {"id": OWNER, "type": "private"},
                "from": {"id": OWNER, "is_bot": False, "first_name": "T"}, "text": text}

    async def go():
        await application.initialize()
        for n, text in enumerate(("Matchday 20 September", "/start")):
            update = Update.de_json({"update_id": n + 1, "edited_message": msg(text)}, application.bot)
            await application.process_update(update)

    asyncio.run(go())
    assert alerts == [] and real is bot.handle_message


# ---- 6. ASCII startup print
def test_the_startup_print_is_ascii_and_survives_cp1252():
    line = bot.startup_banner()
    assert line.isascii()
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    print(line, file=stream)
    stream.flush()
