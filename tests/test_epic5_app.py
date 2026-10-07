"""Epic 5 through the real python-telegram-bot handler chain (owner gate, buttons, typed stake) with a FakeBookmaker,
and the owner-run smoke script (never against a real site)."""
import asyncio
import subprocess
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from telegram.ext import ApplicationBuilder, ExtBot
from telegram import User

import app as bot
from fakes_bookmaker import NOW, FakeBookmaker, make_fixture, totals_offer, two_pick_setup
from predictions.core.betting import leg_from_evaluation
from predictions.flow.evaluate import GateResult
from predictions.flow.session import SessionManager, State
from predictions.flow.store import EvaluationStore
from predictions.tools import betslip_smoke
from test_final_list import Chain, find

REPO = Path(__file__).resolve().parent.parent
OWNER_ID = 111222333
STRANGER = 999888777


def gate_of(passed):
    return GateResult(tuple(passed), (), 2, 0, {})


@pytest.fixture
def live_chain(monkeypatch, tmp_path):
    """The handler chain with the REAL run_slate_job and a FakeBookmaker in place of SportyBet."""
    sent, edits, answers, events = [], [], [], []
    scenario = {}
    evals, listing = two_pick_setup()
    fake = FakeBookmaker(listing, dry_run=True)

    async def no_network(self):
        self._bot_user = User(id=1, first_name="TestBot", is_bot=True, username="test_bot")

    async def send_message(self, chat_id, text, **kwargs):
        sent.append((text, kwargs))
        events.append((text, kwargs))

    async def edit_message_text(self, text, *args, **kwargs):
        edits.append((text, kwargs))
        events.append((text, kwargs))

    async def answer_callback_query(self, callback_query_id, text=None, show_alert=None, **kwargs):
        answers.append((text, show_alert))

    async def fake_run(argument, source, sanitize, **kwargs):
        kwargs["on_result"](gate_of(evals))
        return ["Matchday fixtures ..."]

    # a message.reply_text inside a callback goes through the bot's send_message
    monkeypatch.setattr(ExtBot, "initialize", no_network)
    monkeypatch.setattr(ExtBot, "send_message", send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", edit_message_text)
    monkeypatch.setattr(ExtBot, "answer_callback_query", answer_callback_query)
    monkeypatch.setattr(bot, "run_matchday", fake_run)
    monkeypatch.setattr(bot, "SESSIONS", SessionManager(clock=lambda: NOW))
    monkeypatch.setattr(bot, "BOOKMAKER", fake)
    monkeypatch.setattr(bot, "DRY_RUN", True)
    store = EvaluationStore.open(tmp_path / "t.db", bot.redact)
    monkeypatch.setattr(bot, "STORE", store)
    application = ApplicationBuilder().token("123456:TEST-TOKEN-NOT-REAL").build()
    bot.register_handlers(application)
    chain = Chain(application, sent, edits, answers)
    chain.scenario, chain.store, chain.events, chain.fake, chain.evals = scenario, store, events, fake, evals
    yield chain
    store.close()


def texts(chain):
    return [t for t, _ in chain.sent]


def to_stake(chain):
    chain.text(OWNER_ID, "Matchday 10 October")
    chain.press(OWNER_ID, find(chain.buttons(-1), "Final confirm"))           # the final edit screen first
    chain.press(OWNER_ID, find(chain.buttons(-1), "Confirm games"))


def test_confirm_loads_shows_odds_first_and_only_then_asks_for_the_stake(live_chain):
    chain = live_chain
    to_stake(chain)
    shown = texts(chain)
    odds_at = next(i for i, t in enumerate(shown) if t.startswith("Loaded on SportyBet"))
    stake_at = next(i for i, t in enumerate(shown) if "How much do you want to stake" in t)
    assert odds_at < stake_at and "Total odds: 2.41" in shown[odds_at] and "stake" not in shown[odds_at].lower()
    assert chain.fake.clicks == 0 and [c[0] for c in chain.fake.calls] == ["check", "load"]
    assert bot.SESSIONS.active.state is State.AWAITING_STAKE and bot.SESSIONS.active.odds_shown


def test_the_typed_stake_goes_through_the_owner_gate_and_a_stranger_cannot_set_one(live_chain):
    chain = live_chain
    to_stake(chain)
    count = len(chain.sent)
    chain.text(STRANGER, "500")                                      # dropped by the gate: no reply, no effect
    assert len(chain.sent) == count and bot.SESSIONS.active.state is State.AWAITING_STAKE
    chain.text(OWNER_ID, "ten thousand")
    assert "not a valid stake" in chain.sent[-1][0] and bot.SESSIONS.active.state is State.AWAITING_STAKE
    chain.text(OWNER_ID, "500")
    text, kwargs = chain.sent[-1]
    assert "Final summary" in text and "Stake: NGN 500.00" in text and "DRY RUN is ON" in text
    assert [b.text for row in kwargs["reply_markup"].inline_keyboard for b in row] == ["Confirm (dry run: nothing is placed)", "Change amount"]


def test_the_stranger_cannot_press_the_final_button(live_chain):
    chain = live_chain
    to_stake(chain)
    chain.text(OWNER_ID, "500")
    button = chain.buttons(-1)[0][1]
    chain.press(STRANGER, button)
    assert bot.SESSIONS.active.state is State.AWAITING_FINAL_CONFIRM and chain.fake.calls[-1][0] != "place"


def test_the_whole_dry_run_journey_reports_and_never_clicks_place(live_chain):
    chain = live_chain
    to_stake(chain)
    chain.text(OWNER_ID, "500")
    button = chain.buttons(-1)[0][1]
    chain.press(OWNER_ID, button)
    final = chain.sent[-1][0]
    assert "DRY RUN" in final and "Place button was never clicked" in final and "What would have been placed" in final
    assert chain.fake.clicks == 0 and [c[0] for c in chain.fake.calls].count("place") == 1
    assert bot.SESSIONS.active.state is State.DONE
    (row,) = chain.store.list_placements()
    assert row["state"] == "DRY_RUN" and row["stake"] == "500" and chain.store.list_placed_legs() == []
    # a second tap on the same button places nothing and does not run again
    chain.press(OWNER_ID, button)
    assert [c[0] for c in chain.fake.calls].count("place") == 1


def test_a_double_tap_on_the_final_button_places_once(live_chain):
    chain = live_chain
    to_stake(chain)
    chain.text(OWNER_ID, "500")
    button = chain.buttons(-1)[0][1]
    chain.press(OWNER_ID, button)
    chain.press(OWNER_ID, button)
    # Final confirm is answered first with no text (AD-22); the notice is then shown in the chat
    assert chain.answers[-1][0] is None and (chain.sent[-1][0] == "Already done." or chain.sent[-1][0].startswith("This list changed"))
    assert [c[0] for c in chain.fake.calls].count("place") == 1


def test_changing_the_stake_gives_a_new_summary_and_the_old_button_stops_working(live_chain):
    chain = live_chain
    to_stake(chain)
    chain.text(OWNER_ID, "500")
    old_button = chain.buttons(-1)[0][1]
    chain.text(OWNER_ID, "750")
    new_button = chain.buttons(-1)[0][1]
    assert new_button != old_button and "Stake: NGN 750.00" in chain.sent[-1][0]
    chain.press(OWNER_ID, old_button)
    assert chain.answers[-1][0] is None and "This list changed" in (chain.edits[-1][0] + chain.sent[-1][0])
    assert not any(c[0] == "place" for c in chain.fake.calls)


def test_missing_game_flow_asks_and_skip_button_resumes_loading(live_chain):
    chain = live_chain
    chain.fake.listing = chain.fake.listing[:1]                      # the second game is not on SportyBet
    to_stake(chain)
    text, kwargs = chain.sent[-1]
    assert "cannot be loaded as they are" in text and "Please skip this game" in text
    assert bot.SESSIONS.active.state is State.RESOLVING_AVAILABILITY and len(bot.SESSIONS.active.slip) == 2
    labels = [b.text for row in kwargs["reply_markup"].inline_keyboard for b in row]
    assert labels == ["Skip 1"]
    skip = kwargs["reply_markup"].inline_keyboard[0][0].callback_data
    chain.press(OWNER_ID, skip)
    assert bot.SESSIONS.active.state is State.AWAITING_STAKE and len(bot.SESSIONS.active.slip) == 1
    assert any("How much do you want to stake" in t for t in texts(chain))
    assert chain.fake.clicks == 0


def test_a_loading_failure_is_reported_sanitized_and_ends_the_session(live_chain):
    chain = live_chain
    from predictions.core.ports import BookmakerError
    chain.fake.load_error = BookmakerError("could not load 08031234567 test-password-not-real")
    to_stake(chain)
    final = chain.sent[-1][0]
    assert "Loading your games onto SportyBet failed" in final and "Nothing was placed" in final
    assert "08031234567" not in final and "test-password-not-real" not in final
    assert bot.SESSIONS.active.state is State.FAILED and [c[0] for c in chain.fake.calls] == ["check", "load"]


def test_real_mode_placement_is_reported_with_the_booking_code_only_after_proof(live_chain, monkeypatch):
    chain = live_chain
    live = FakeBookmaker(chain.fake.listing, dry_run=False)
    monkeypatch.setattr(bot, "BOOKMAKER", live)
    monkeypatch.setattr(bot, "DRY_RUN", False)
    to_stake(chain)
    chain.text(OWNER_ID, "500")
    assert "DRY RUN is OFF" in chain.sent[-1][0]
    chain.press(OWNER_ID, chain.buttons(-1)[0][1])
    final = chain.sent[-1][0]
    assert "Bet placed" in final and "Booking code: ABC123" in final and live.clicks == 1


# ================================================================ the owner-run smoke script

class ListFixtures:
    def __init__(self, fixtures):
        self.fixtures = fixtures

    async def fetch_fixtures(self):
        return self.fixtures


def smoke_listing():
    return [make_fixture("Alpha", "Beta", [totals_offer("2.5", "1.85", "2.00")], 14),
            make_fixture("Gamma", "Delta", [totals_offer("2.5", "1.50", "2.60")], 16),
            make_fixture("Old", "Game", [totals_offer("2.5", "1.50", "2.60")], 12, started=True)]


def test_the_smoke_script_walks_every_step_in_dry_run_and_never_places():
    listing = smoke_listing()
    fake = FakeBookmaker(listing, dry_run=True)
    lines = []
    failed = asyncio.run(betslip_smoke.run_smoke(fake, ListFixtures(listing), NOW, 2, "", lines.append))
    assert failed == 0 and fake.clicks == 0
    text = "\n".join(lines)
    for n in range(1, 6):
        assert f"STEP {n} OK" in text
    assert "Place button is NOT clicked" in text and "Old v Game" not in text
    assert [c[0] for c in fake.calls] == ["check", "load", "place", "history"]


def test_the_smoke_script_refuses_to_run_against_a_bookmaker_that_can_place():
    fake = FakeBookmaker(smoke_listing(), dry_run=False)
    with pytest.raises(SystemExit):
        asyncio.run(betslip_smoke.run_smoke(fake, ListFixtures(smoke_listing()), NOW, 2, "", lambda s: None))
    assert fake.calls == []


def test_the_smoke_script_reports_a_failing_step_and_says_which_selectors_to_check():
    from predictions.core.ports import BookmakerError
    fake = FakeBookmaker(smoke_listing(), dry_run=True)
    fake.load_error = BookmakerError("The betslip could not be emptied.")
    lines = []
    failed = asyncio.run(betslip_smoke.run_smoke(fake, ListFixtures(smoke_listing()), NOW, 2, "", lines.append))
    text = "\n".join(lines)
    assert failed == 1 and "STEP 3 FAILED" in text and "selector group to check" in text and "STEP 4" not in text


def test_the_smoke_script_prefers_matching_names_and_skips_started_games():
    got = betslip_smoke.pick_fixtures(smoke_listing(), NOW, 1, "gamma")
    assert [f.home for f in got] == ["Gamma"]
    assert [f.home for f in betslip_smoke.pick_fixtures(smoke_listing(), NOW, 5)] == ["Alpha", "Gamma"]


def test_the_smoke_script_is_dry_run_only_by_construction_and_documents_how_to_run():
    source = (REPO / "predictions" / "tools" / "betslip_smoke.py").read_text(encoding="utf-8")
    assert "dry_run=True" in source and "dry_run=False" not in source and "python -m predictions.tools.betslip_smoke" in source
    result = subprocess.run([sys.executable, "-m", "predictions.tools.betslip_smoke", "--help"], cwd=REPO,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and "--headed" in result.stdout


def test_smoke_step_5_describes_the_list_entries_without_any_bet_id_odds_or_picks():
    from fakes_bookmaker import history_entry
    listing = smoke_listing()
    fake = FakeBookmaker(listing, dry_run=True)
    fake.history = (history_entry(("Alpha v Beta", "Gamma v Delta"), "500"), history_entry(("A v B",), "100", "Won"))
    lines = []
    failed = asyncio.run(betslip_smoke.run_smoke(fake, ListFixtures(listing), NOW, 2, "", lines.append))
    text = "\n".join(lines)
    assert failed == 0 and "STEP 5 OK: read the first page of the bet history" in text and "2 bet(s) read" in text
    assert "Multiple, Running, stake 500" in text and "Singles, Won, stake 100" in text and "2 game(s) (2 named)" in text
    assert "shows no bet id, booking code, odds or picks" in text


def test_smoke_step_5_flags_an_unreadable_history_and_an_ambiguous_key():
    from fakes_bookmaker import history_entry
    from predictions.core.ports import BookmakerError
    listing = smoke_listing()
    fake = FakeBookmaker(listing, dry_run=True)
    fake.history = (history_entry(("A v B",), "100", ambiguous=True),)
    lines = []
    asyncio.run(betslip_smoke.run_smoke(fake, ListFixtures(listing), NOW, 2, "", lines.append))
    assert "AMBIGUOUS key" in "\n".join(lines)
    broken = FakeBookmaker(listing, dry_run=True)

    async def unreadable():
        raise BookmakerError("The bet history could not be read (no entries and no empty-list marker).")
    broken.bet_history = unreadable
    lines = []
    failed = asyncio.run(betslip_smoke.run_smoke(broken, ListFixtures(listing), NOW, 2, "", lines.append))
    text = "\n".join(lines)
    assert failed == 1 and "STEP 5 FAILED" in text and "history_items" in text and "could not be read" in text
