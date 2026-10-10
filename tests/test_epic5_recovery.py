"""Stories 5.5, 5.6 and 5.7: recovery after a crash, continuing from a Review, and one slate at a time.

Everything runs against the FakeBookmaker (the Bookmaker port) and a real SQLite store in a temp folder. Nothing here
logs in anywhere, and no test can place a real bet: the fake only counts "clicks"."""
import asyncio
import json
import sqlite3
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from telegram.ext import ApplicationBuilder, ExtBot
from telegram import User

import app as bot
from fakes_bookmaker import (KICKOFF, NOW, FakeBookmaker, make_fixture, pick_evaluation, totals_offer, two_pick_setup,
                             xg_inputs)
from predictions.core.betting import HISTORY_LIMITS_NOTE, HistoryBet, leg_identity
from predictions.core.model import Estimate, Evaluation
from predictions.core.placement import DuplicateLeg
from predictions.core.ports import BookmakerError
from predictions.core.slip import Slip, make_pick_id
from predictions.core.types import Owner
from predictions.flow import recovery
from predictions.flow.loading import KIND_LOADED, run_load
from predictions.flow.placement import OUTCOME_FAILED, OUTCOME_PLACED, OUTCOME_UNKNOWN, run_placement
from predictions.flow.reconcile import reconcile_placement
from predictions.flow.recovery import (KIND_ENDED, KIND_RECONCILED, KIND_RELOAD, KIND_RESUMED, KIND_UNREADABLE,
                                       attach_store, recover_on_startup)
from predictions.flow.review import approved_evaluations
from predictions.flow.session import Action, Session, SessionManager, State, Status
from predictions.flow.snapshot import SnapshotError, from_json, to_json
from predictions.flow.store import MIGRATIONS, EvaluationStore
from predictions.telegram import slate
from predictions.telegram.slate import decode, encode, handle_callback, summary_screen
from test_final_list import Chain, find, gate_of
from test_review import FakeCodeReader, FakeStats, selections
from predictions.core.review import ReviewResult
from predictions.core.stats import READY, FixtureStats
from predictions.core.types import Fixture

OWNER_ID = 111222333
STRANGER = 999888777
OWNER = Owner(OWNER_ID)
SID = "abcd1234"
REPO = Path(__file__).resolve().parent.parent


def run(coro):
    return asyncio.run(coro)


def history_bet(bet_id="B777", stake="500", shown=("Alpha v Beta", "Gamma v Delta"), code="CODE77", leg_count=2,
                placed_at=None, status="Running", bet_type=None):
    """A history LIST entry as the real page shows it (games, stake, status, time; no odds, no picks, no real bet id)."""
    when = placed_at or max(NOW, datetime.now(timezone.utc))
    kind = bet_type or ("Singles" if leg_count == 1 else "Multiple")
    return HistoryBet(bet_id, code, Decimal(stake), None, (), " | ".join(shown), leg_count, when, tuple(shown), status, kind)


class Rig:
    """One slate, its state machine attached to a real store, and a fake bookmaker."""

    def __init__(self, tmp_path, dry_run=False, evals=None, listing=None, sid=SID, store=None):
        default_evals, default_listing = two_pick_setup()
        self.evals = evals if evals is not None else default_evals
        self.listing = listing if listing is not None else default_listing
        self.now = NOW
        self.store = store or EvaluationStore.open(tmp_path / "r.db", str)
        self.manager = SessionManager(clock=lambda: self.now)
        attach_store(self.manager, self.store)
        self.sid = sid
        self.manager.start(self.evals, sid, "matchday")
        self.bm = FakeBookmaker(self.listing, dry_run)

    @property
    def session(self):
        return self.manager.active

    def confirm(self):
        assert self.manager.act(OWNER, self.sid, self.session.state_version, Action.DONE).status is Status.APPLIED
        result = self.manager.act(OWNER, self.sid, self.session.state_version, Action.CONFIRM)
        assert result.status is Status.APPLIED
        return result

    def load(self):
        return run(run_load(self.manager, self.bm, self.sid, self.session.state_version, str))

    def to_stake(self):
        self.confirm()
        assert self.load().kind == KIND_LOADED
        self.manager.mark_odds_shown(self.sid, self.session.state_version)

    def to_summary(self, stake="500"):
        self.to_stake()
        assert self.manager.submit_stake(OWNER, stake).status is Status.APPLIED
        return self.session

    def press_final(self):
        s = self.session
        return self.manager.act(OWNER, self.sid, s.state_version, Action.FINAL, s.summary.summary_id)

    def place(self, token):
        return run(run_placement(self.manager, self.store, self.bm, token, str))

    def confirm_and_place(self, stake="500"):
        self.to_summary(stake)
        result = self.press_final()
        assert result.status is Status.APPLIED, result.status
        return self.place(result.token)

    def restart(self, bookmaker=None, now=None, get_bookmaker=None):
        """A new process: a fresh SessionManager on the same database, then the start-up recovery."""
        manager = SessionManager(clock=lambda: now or self.now)
        attach_store(manager, self.store)
        bm = bookmaker or self.bm
        reports = run(recover_on_startup(manager, self.store, get_bookmaker or (lambda: bm), str))
        self.manager = manager
        return reports

    def close(self):
        self.store.close()


@pytest.fixture
def rig(tmp_path):
    r = Rig(tmp_path)
    yield r
    r.close()


@pytest.fixture
def dry_rig(tmp_path):
    r = Rig(tmp_path, dry_run=True)
    yield r
    r.close()


def calls(rig):
    return [c[0] for c in rig.bm.calls]


# ================================================================ 5.5: saving the slate, deterministically

def test_a_session_serialises_deterministically_and_round_trips_in_every_state(rig):
    seen = [rig.session]
    rig.confirm()
    seen.append(rig.session)
    assert rig.load().kind == KIND_LOADED
    seen.append(rig.session)
    rig.manager.mark_odds_shown(SID, rig.session.state_version)
    rig.manager.submit_stake(OWNER, "500")
    seen.append(rig.session)
    rig.press_final()
    seen.append(rig.session)
    assert [s.state for s in seen] == [State.REVIEWING_LIST, State.LOADING_SLIP, State.AWAITING_STAKE,
                                       State.AWAITING_FINAL_CONFIRM, State.PLACING]
    for session in seen:
        text = to_json(session)
        assert text == to_json(session) and from_json(text) == session and to_json(from_json(text)) == text
        assert json.loads(text)["session_id"] == SID


def test_a_session_with_open_issues_and_an_alternative_round_trips(tmp_path):
    a = pick_evaluation("Alpha", "Beta", "Over/Under 2.5", "Over 2.5", "1.85")
    listing = [make_fixture("Alpha", "Beta", [totals_offer("2.0", "1.50", "2.30")], 14)]   # 2.5 is not offered
    r = Rig(tmp_path, evals=[a], listing=listing)
    r.confirm()
    assert r.load().kind == "issues" and r.session.state is State.RESOLVING_AVAILABILITY
    assert from_json(to_json(r.session)) == r.session
    r.close()


def test_the_slate_is_saved_after_each_step_and_holds_no_secret(tmp_path):
    secret = "test-password-not-real"
    store = EvaluationStore.open(tmp_path / "s.db", lambda t: t.replace(secret, "[hidden]"))
    evals = [pick_evaluation("Alpha " + secret, "Beta", "Over/Under 2.5", "Over 2.5", "1.85")]
    listing = [make_fixture("Alpha " + secret, "Beta", [totals_offer("2.5", "1.85", "2.00")], 14)]
    r = Rig(tmp_path, evals=evals, listing=listing, store=store)
    steps = [(r.session.state, store.load_latest_session()["state"])]
    r.confirm()
    steps.append((r.session.state, store.load_latest_session()["state"]))
    r.load()
    steps.append((r.session.state, store.load_latest_session()["state"]))
    r.manager.mark_odds_shown(SID, r.session.state_version)
    r.manager.submit_stake(OWNER, "500")
    steps.append((r.session.state, store.load_latest_session()["state"]))
    assert all(a.value == b for a, b in steps) and len(steps) == 4
    assert store.load_latest_session()["state_version"] == r.session.state_version
    r.store.close()
    raw = (tmp_path / "s.db").read_bytes()
    assert secret.encode() not in raw and b"[hidden]" in raw
    assert r.manager.persist_errors == []


def test_a_saved_session_that_was_tampered_with_is_refused():
    with pytest.raises(SnapshotError):
        from_json("not json")
    with pytest.raises(SnapshotError):
        from_json("[]")
    data = {"snapshot": 1}
    with pytest.raises(SnapshotError):
        from_json(json.dumps(data))


def test_a_pick_whose_id_does_not_match_is_refused(rig):
    data = json.loads(to_json(rig.session))
    data["picks"][0]["pick_id"] = "000000000000"
    with pytest.raises(SnapshotError):
        from_json(json.dumps(data))


def test_migration_v3_is_forward_only_and_keeps_earlier_rows(tmp_path):
    db = tmp_path / "m.db"
    with EvaluationStore.open(db, str, migrations=MIGRATIONS[:2]) as old:
        assert old.schema_version == 2
        evals, _ = two_pick_setup()
        old.save_evaluations("s", evals)
    backups = tmp_path / "backups"
    with EvaluationStore.open(db, str, backup_dir=backups) as new:
        assert new.schema_version == 7 == len(MIGRATIONS)
        assert len(new.list_evaluations()) == 2 and new.load_latest_session() is None
    assert list(backups.glob("*-before-v3.db"))


def test_a_failed_save_is_recorded_and_never_changes_the_flow(rig):
    def broken(session):
        raise RuntimeError("disk full")

    rig.manager.attach(broken, rig.manager._registry)
    rig.confirm()
    assert rig.session.state is State.LOADING_SLIP and rig.manager.persist_errors == ["RuntimeError", "RuntimeError"]  # Done and Confirm: each save failed


# ================================================================ 5.5: an error after the click

def test_a_crash_after_the_click_says_unknown_checks_history_and_never_places_again(rig):
    rig.bm.place_mode = "unknown"
    outcome = rig.confirm_and_place()
    assert outcome.kind == OUTCOME_UNKNOWN and rig.session.state is State.PLACEMENT_UNKNOWN
    assert "UNKNOWN" in outcome.message and "I will not place again" in outcome.message
    assert "did not find this bet" in outcome.message
    assert calls(rig) == ["check", "load", "place", "history"] and rig.bm.clicks == 1
    (row,) = rig.store.list_placements()
    assert row["state"] == "PLACEMENT_UNKNOWN" and rig.store.list_placed_legs() == []
    # the same token can never be used again
    token_calls = calls(rig).count("place")
    again = rig.press_final()
    assert again.status is Status.STALE and calls(rig).count("place") == token_calls == 1


def test_an_unclear_result_is_resolved_from_history_when_the_bet_is_there(rig):
    rig.bm.place_mode = "unknown"
    rig.bm.history = (history_bet("B4242"),)
    outcome = rig.confirm_and_place()
    assert outcome.kind == OUTCOME_PLACED and "EXISTS" in outcome.message and "B4242" in outcome.message
    assert rig.session.state is State.DONE and rig.bm.clicks == 1 and calls(rig).count("place") == 1
    (row,) = rig.store.list_placements()
    assert row["state"] == "PLACED" and row["bet_id"] == "B4242" and len(rig.store.list_placed_legs()) == 2
    assert "Booking code: CODE77" in slate.placement_text(outcome)


def test_a_history_that_cannot_be_read_stays_unknown_and_places_nothing(rig):
    async def broken():
        raise BookmakerError("could not reach 08031234567")

    rig.bm.place_mode = "unknown"
    rig.bm.bet_history = broken
    outcome = rig.confirm_and_place()
    assert outcome.kind == OUTCOME_UNKNOWN and "could not read your SportyBet bet history" in outcome.message
    assert rig.session.state is State.PLACEMENT_UNKNOWN and rig.bm.clicks == 1


def test_an_unexpected_error_while_placing_is_unknown_never_no_bet(rig):
    rig.bm.place_mode = "unexpected"
    outcome = rig.confirm_and_place()
    assert outcome.kind == OUTCOME_UNKNOWN and "UNKNOWN" in outcome.message
    assert rig.session.state is State.PLACEMENT_UNKNOWN and calls(rig).count("place") == 1


def test_several_matching_bets_are_a_loud_warning_not_a_guess(rig):
    rig.bm.place_mode = "unknown"
    rig.bm.history = (history_bet("B1"), history_bet("B2"))
    outcome = rig.confirm_and_place()
    assert outcome.kind == OUTCOME_UNKNOWN and "WARNING: 2 entries" in outcome.message
    assert rig.store.list_placed_legs() == [] and rig.session.state is State.PLACEMENT_UNKNOWN


def test_a_bet_that_is_already_another_placement_is_not_mistaken_for_this_one(rig):
    rig.bm.place_mode = "unknown"
    rig.to_summary()
    token = rig.press_final().token
    rig.store.record_placing(replace_token(token), False)
    rig.store.update_placement(replace_token(token).token_hash, "PLACED", bet_id="B9", detail="earlier")
    rig.bm.history = (history_bet("B9"),)
    rig.store.record_placing(token, False)
    outcome = run(reconcile_placement(rig.store, rig.bm, token.token_hash, str))
    assert not outcome.found  # B9 belongs to the earlier placement, so this one is still unproven


def replace_token(token):
    """A second, different token (another stake) so the earlier placement has its own row."""
    from predictions.core.placement import MINT_AUTHORITY, PlacementToken, summary_digest
    stake = Decimal("123")
    digest = summary_digest(token.session_id, token.legs, token.total_odds, stake, token.expires_at)
    return PlacementToken(token.session_id, token.legs, token.total_odds, stake, token.expires_at, digest, MINT_AUTHORITY)


def test_dry_run_never_reads_history_or_clicks(dry_rig):
    dry_rig.bm.place_mode = "unexpected"
    outcome = dry_rig.confirm_and_place()
    assert outcome.kind == OUTCOME_UNKNOWN and "history" not in calls(dry_rig) and dry_rig.bm.clicks == 0


# ================================================================ 5.5: restart in each state

def test_a_slate_saved_before_loading_resumes_exactly_where_it_was(rig):
    old_buttons = slate.list_screen(rig.session).buttons
    version = rig.session.state_version
    reports = rig.restart()
    (report,) = reports
    assert report.kind == KIND_RESUMED and report.show_list and not report.reload
    assert rig.manager.active == rig.session and rig.session.state is State.REVIEWING_LIST
    assert rig.session.state_version == version and calls(rig) == []      # the bookmaker was not touched
    delete = old_buttons[0][1][1]                                          # a button from BEFORE the restart still works
    reply = handle_callback(delete, OWNER, rig.manager)
    assert reply.answer == "Deleted." and len(rig.session.slip) == 1


@pytest.mark.parametrize("stop", ["loading", "stake", "summary", "resolving"])
def test_a_slate_at_or_after_loading_resumes_at_loading_slip_and_asks_again(tmp_path, stop):
    r = Rig(tmp_path)
    if stop == "resolving":
        r.listing = r.listing[:1]
        r.bm.listing = r.listing
    r.confirm()
    if stop != "loading":
        outcome = r.load()
        if stop != "resolving":
            assert outcome.kind == KIND_LOADED
            r.manager.mark_odds_shown(SID, r.session.state_version)
        if stop == "summary":
            r.manager.submit_stake(OWNER, "500")
    before = r.session
    stale_final = None
    if stop == "summary":
        stale_final = encode(SID, before.state_version, Action.FINAL, before.summary.summary_id)
    r.bm.calls.clear()
    r.bm.listing = two_pick_setup()[1]
    (report,) = r.restart()
    assert report.kind == KIND_RELOAD and report.reload and r.session.state is State.LOADING_SLIP
    assert r.session.state_version == before.state_version + 1 and r.bm.calls == []
    assert r.session.stake is None and r.session.summary is None and r.session.loaded is None and not r.session.odds_shown
    assert len(r.session.slip) == len(before.slip)
    if stale_final:
        assert handle_callback(stale_final, OWNER, r.manager, dry_run=False).answer.startswith("This list changed")
    # the caller reloads the slip, re-reads the odds and only then asks for the stake, and a fresh confirmation
    assert r.load().kind == KIND_LOADED and r.session.state is State.AWAITING_STAKE
    assert calls(r) == ["check", "load"] and r.bm.clicks == 0
    assert r.session.summary is None
    r.close()


def test_a_slate_too_old_to_resume_is_ended_not_reloaded(rig):
    rig.to_stake()
    rig.bm.calls.clear()
    (report,) = rig.restart(now=NOW + timedelta(hours=7))
    assert report.kind == KIND_ENDED and not report.reload and rig.manager.active.state is State.CANCELLED
    assert rig.bm.calls == []


def test_a_slate_that_ended_is_not_recovered(rig):
    rig.to_stake()
    rig.manager.finish(SID, State.FAILED, "x")
    assert rig.restart() == [] and rig.manager.active is None


def test_a_bet_in_placing_at_restart_is_checked_in_history_before_anything_is_offered_found(rig):
    rig.to_summary()
    token = rig.press_final().token
    rig.store.record_placing(token, False)                 # PLACING was written; then the bot died around the click
    assert rig.session.state is State.PLACING
    rig.bm.calls.clear()
    rig.bm.history = (history_bet("B5150"),)
    (report,) = rig.restart()
    assert report.kind == KIND_RECONCILED and "EXISTS" in report.text and "B5150" in report.text
    assert not report.reload and not report.show_list                                 # nothing is offered
    assert calls(rig) == ["history"] and rig.bm.clicks == 0
    assert rig.manager.active.state is State.DONE
    (row,) = rig.store.list_placements()
    assert row["state"] == "PLACED" and row["bet_id"] == "B5150" and len(rig.store.list_placed_legs()) == 2


def test_a_bet_in_placing_at_restart_that_is_not_in_history_stays_unknown_and_blocked(rig):
    rig.to_summary()
    token = rig.press_final().token
    rig.store.record_placing(token, False)
    rig.bm.calls.clear()
    (report,) = rig.restart()
    assert report.kind == KIND_RECONCILED and "did not find this bet" in report.text and "will not place it again" in report.text
    assert calls(rig) == ["history"] and rig.manager.active.state is State.PLACEMENT_UNKNOWN
    assert rig.store.list_placements()[0]["state"] == "PLACEMENT_UNKNOWN"
    assert {d.source for d in rig.store.find_blocking_legs([leg_identity(l.leg_key) for l in token.legs])} == {"unknown"}
    assert not report.reload and rig.bm.clicks == 0


def test_placing_at_restart_with_the_bookmaker_unavailable_is_reported_as_unreadable(rig):
    rig.to_summary()
    token = rig.press_final().token
    rig.store.record_placing(token, False)

    def no_bookmaker():
        raise BookmakerError("no login details")

    (report,) = rig.restart(get_bookmaker=no_bookmaker)
    assert "could not read your SportyBet bet history" in report.text
    assert rig.manager.active.state is State.PLACEMENT_UNKNOWN and rig.bm.clicks == 0


def test_a_dry_run_placing_row_at_restart_is_resolved_without_history(dry_rig):
    dry_rig.to_summary()
    token = dry_rig.press_final().token
    dry_rig.store.record_placing(token, True)
    dry_rig.bm.calls.clear()
    (report,) = dry_rig.restart()
    assert "nothing was placed" in report.text and calls(dry_rig) == [] and dry_rig.manager.active.state is State.DONE
    assert dry_rig.store.list_placements()[0]["state"] == "DRY_RUN"


def test_a_placing_row_wins_over_a_stale_session_snapshot(rig):
    rig.to_summary()
    token = rig.press_final().token
    rig.store.record_placing(token, False)
    # the saved slate is older than the PLACING row (the last save was lost): it must NOT be reloaded and offered again
    stale = replace(rig.session, state=State.AWAITING_FINAL_CONFIRM, token_hash="")
    rig.store.save_session(SID, stale.state.value, stale.state_version, to_json(stale), "matchday")
    rig.bm.history = (history_bet("B31"),)
    rig.bm.calls.clear()
    reports = rig.restart()
    assert [r.kind for r in reports] == [KIND_RECONCILED] and not any(r.reload for r in reports)
    assert rig.manager.active.state is State.DONE and rig.bm.calls == [("history",)]


def test_a_session_in_placing_with_no_placement_row_is_checked_and_nothing_was_clicked(rig):
    rig.to_summary()
    rig.press_final()                                      # PLACING saved, but the crash came before record_placing
    rig.bm.calls.clear()
    (report,) = rig.restart()
    assert "No bet was placed" in report.text and calls(rig) == ["history"] and rig.manager.active.state is State.FAILED


def test_an_unknown_result_is_checked_again_at_the_next_restart(rig):
    rig.bm.place_mode = "unknown"
    rig.confirm_and_place()
    assert rig.session.state is State.PLACEMENT_UNKNOWN
    rig.bm.history = (history_bet("B88"),)                 # the history has caught up by now
    (report,) = rig.restart()
    assert "EXISTS" in report.text and rig.manager.active.state is State.DONE
    assert rig.store.list_placements()[0]["state"] == "PLACED" and len(rig.store.list_placed_legs()) == 2


def test_an_unreadable_saved_slate_is_reported_and_nothing_is_done(rig):
    rig.store.save_session(SID, "AWAITING_STAKE", 5, json.dumps({"snapshot": 1, "session_id": SID}), "matchday")
    (report,) = rig.restart()
    assert report.kind == KIND_UNREADABLE and "Nothing was resumed and nothing was placed" in report.text
    assert rig.manager.active is None and rig.bm.calls == []


def test_a_cancelled_slate_is_never_brought_back(tmp_path):
    r = Rig(tmp_path)
    r.manager.request_start(OWNER, [pick_evaluation("Echo", "Foxtrot", "Over/Under 1.5", "Over 1.5", "1.3")], "matchday",
                            "eeee5555")
    pending = r.manager.pending
    result = r.manager.act(OWNER, SID, 1, Action.REPLACE, pending.pending_id)
    assert result.status is Status.APPLIED
    states = {row["session_id"]: row["state"] for row in r.store.list_sessions()}
    assert states == {SID: "CANCELLED", "eeee5555": "REVIEWING_LIST"}
    (report,) = r.restart()
    assert report.session.session_id == "eeee5555" and r.manager.active.session_id == "eeee5555"
    r.close()


# ================================================================ 5.5: the start-up path in app.py

class FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append((chat_id, text, kwargs))


def startup(rig, monkeypatch):
    manager = SessionManager(clock=lambda: NOW)
    attach_store(manager, rig.store)
    fake_bot, tasks = FakeBot(), []
    application = SimpleNamespace(bot=fake_bot, create_task=lambda coro: tasks.append(coro))
    monkeypatch.setattr(bot, "SESSIONS", manager)
    monkeypatch.setattr(bot, "STORE", rig.store)
    monkeypatch.setattr(bot, "BOOKMAKER", rig.bm)
    monkeypatch.setattr(bot, "DRY_RUN", False)
    rig.bm.calls.clear()
    run(bot.announce_startup(application))
    return manager, fake_bot, tasks, application


class FakeResumeQuery:
    def __init__(self, data):
        self.data, self.answers, self.cleared = data, [], 0
        self.message = SimpleNamespace(text="prompt")

    async def answer(self, text="", **kwargs):
        self.answers.append(text)

    async def edit_message_reply_markup(self, reply_markup=None, **kwargs):
        self.cleared += 1


def press(data, fake_bot, application):
    query = FakeResumeQuery(data)
    context = SimpleNamespace(bot=fake_bot, application=application)
    run(bot.resume_callback(SimpleNamespace(callback_query=query), context))
    return query


def resume_buttons(fake_bot):
    markups = [kw["reply_markup"] for _, _, kw in fake_bot.sent if kw.get("reply_markup") is not None]
    assert markups, "the restart prompt carried no buttons"
    return [b.callback_data for b in markups[-1].inline_keyboard[0]]


def test_startup_shows_the_last_confirmed_games_and_asks_before_reloading_anything(rig, monkeypatch):
    rig.to_stake()
    manager, fake_bot, tasks, _ = startup(rig, monkeypatch)
    texts = [t for _, t, _ in fake_bot.sent]
    assert all(chat == OWNER_ID for chat, _, _ in fake_bot.sent)
    assert texts[0].startswith("🚀 Bot started") and "Recovered 1 session(s)" in texts[1]
    assert "last confirmed slate (2 games)" in texts[2] and "NOTHING was placed" in texts[2] and "Press Continue" in texts[2]
    assert "Your last confirmed games (nothing was placed):" in texts[3] and "1." in texts[3] and "2." in texts[3]
    data = resume_buttons(fake_bot)
    assert len(data) == 2 and data[0].endswith(":c") and data[1].endswith(":d") and all(d.startswith("r:") and len(d) <= 64 for d in data)
    assert tasks == [] and rig.bm.calls == [] and rig.bm.clicks == 0           # nothing was reloaded or placed on its own
    assert manager.active.state is State.LOADING_SLIP


def test_continue_reloads_the_games_then_asks_for_the_stake_and_never_places(rig, monkeypatch):
    rig.to_stake()
    manager, fake_bot, tasks, application = startup(rig, monkeypatch)
    query = press(resume_buttons(fake_bot)[0], fake_bot, application)
    assert query.answers == ["Reloading the games on SportyBet..."] and query.cleared == 1

    async def finish():
        for task in tasks:
            await task
    run(finish())
    texts = [t for _, t, _ in fake_bot.sent]
    assert any(t.startswith("Loaded on SportyBet") for t in texts) and any("How much do you want to stake" in t for t in texts)
    assert manager.active.state is State.AWAITING_STAKE and calls(rig) == ["check", "load"] and rig.bm.clicks == 0


def test_discard_drops_the_slate_and_a_repeat_or_stale_tap_does_nothing(rig, monkeypatch):
    rig.to_stake()
    manager, fake_bot, tasks, application = startup(rig, monkeypatch)
    data = resume_buttons(fake_bot)
    query = press(data[1], fake_bot, application)
    assert query.answers == ["Discarded. Nothing was placed."] and manager.active.state is State.FAILED
    assert any("Discarded. Nothing was placed" in t for _, t, _ in fake_bot.sent) and rig.bm.calls == [] and tasks == []
    assert press(data[0], fake_bot, application).answers == ["That prompt is out of date."] and tasks == []
    assert press("r:bad", fake_bot, application).answers == ["That button is not valid any more."]


def test_a_slate_that_failed_before_the_click_is_offered_again_after_a_restart(rig, monkeypatch):
    rig.to_stake()
    sid, version = rig.manager.active.session_id, rig.manager.active.state_version
    rig.manager.finish(sid, State.FAILED, "SportyBet is not logged in. Nothing was placed.", version, retryable=True)
    manager, fake_bot, tasks, _ = startup(rig, monkeypatch)
    texts = [t for _, t, _ in fake_bot.sent]
    assert "failed before anything was placed" in texts[2] and "Press Continue" in texts[2]
    assert len(resume_buttons(fake_bot)) == 2 and manager.active.state is State.LOADING_SLIP and tasks == [] and rig.bm.calls == []


def test_a_failed_slate_that_cannot_be_retried_is_not_offered(rig, monkeypatch):
    rig.to_stake()
    sid, version = rig.manager.active.session_id, rig.manager.active.state_version
    rig.manager.finish(sid, State.FAILED, "x", version, retryable=False)
    manager, fake_bot, tasks, _ = startup(rig, monkeypatch)
    assert all(kw.get("reply_markup") is None for _, _, kw in fake_bot.sent)


def test_startup_reconciles_a_bet_and_offers_nothing(rig, monkeypatch):
    rig.to_summary()
    token = rig.press_final().token
    rig.store.record_placing(token, False)
    rig.bm.history = (history_bet("B2024"),)
    manager = SessionManager(clock=lambda: NOW)
    attach_store(manager, rig.store)
    fake_bot, tasks = FakeBot(), []
    application = SimpleNamespace(bot=fake_bot, create_task=lambda coro: tasks.append(coro))
    for name, value in (("SESSIONS", manager), ("STORE", rig.store), ("BOOKMAKER", rig.bm)):
        monkeypatch.setattr(bot, name, value)
    rig.bm.calls.clear()
    run(bot.announce_startup(application))
    texts = [t for _, t, _ in fake_bot.sent]
    assert "Recovered 1 session(s)" in texts[1] and "B2024" in texts[2] and tasks == []
    assert calls(rig) == ["history"] and rig.bm.clicks == 0 and not any("stake" in t.replace(HISTORY_LIMITS_NOTE, "").lower() for t in texts[1:])


def test_startup_without_a_store_recovers_nothing_and_still_starts(monkeypatch):
    fake_bot = FakeBot()
    monkeypatch.setattr(bot, "STORE", None)
    run(bot.announce_startup(SimpleNamespace(bot=fake_bot, create_task=lambda c: None)))
    assert len(fake_bot.sent) == 1 and fake_bot.sent[0][1].startswith("🚀 Bot started")


def test_the_main_block_attaches_the_store_and_the_startup_hook_runs_recovery():
    source = (REPO / "app.py").read_text(encoding="utf-8")
    main = source[source.index('if __name__ == "__main__":'):]
    assert "attach_store(SESSIONS, STORE)" in main and ".post_init(announce_startup)" in main
    announce = source[source.index("async def announce_startup"):source.index("async def recover_after_start")]
    assert "recover_after_start(" in announce and "run_slate_job(" in announce


# ================================================================ 5.5: the placed-leg registry

def test_a_leg_already_placed_is_blocked_unless_the_owner_overrides_it(tmp_path):
    first = Rig(tmp_path)
    assert first.confirm_and_place().kind == OUTCOME_PLACED and first.bm.clicks == 1
    second = Rig(tmp_path, sid="bbbb2222", store=first.store)
    second.to_summary("500")
    blocked = second.press_final()
    assert blocked.status is Status.DUPLICATE and blocked.token is None
    assert {d.source for d in blocked.duplicates} == {"placed"} and len(blocked.duplicates) == 2
    assert second.session.state is State.AWAITING_FINAL_CONFIRM and calls(second).count("place") == 0
    # the blocked screen names the games and offers ONE dedicated button
    screen = slate.duplicate_screen(second.session, blocked.duplicates)
    assert "already placed (bet id B1234567)" in screen.text and "Alpha v Beta" in screen.text
    assert [label for label, _ in screen.buttons[0]] == ["Override: place it anyway"]
    # the ordinary confirm button keeps saying no
    summary = summary_screen(second.session, dry_run=False)
    reply = handle_callback(summary.buttons[0][0][1], OWNER, second.manager, dry_run=False)
    assert reply.job is None and reply.screen.buttons[0][0][0] == "Override: place it anyway"
    # the override is explicit, records nothing else, and still needs a new confirmation
    override = decode(screen.buttons[0][0][1])
    result = second.manager.act(OWNER, override[0], override[1], override[2], override[3])
    assert result.status is Status.APPLIED and second.bm.clicks == 0
    final = second.press_final()
    assert final.status is Status.APPLIED and final.token is not None
    outcome = second.place(final.token)
    assert outcome.kind == OUTCOME_PLACED and second.bm.clicks == 1
    first.close()


def test_an_override_only_covers_the_exact_summary_it_was_given_for(tmp_path):
    first = Rig(tmp_path)
    first.confirm_and_place()
    second = Rig(tmp_path, sid="bbbb2222", store=first.store)
    second.to_summary("500")
    dup = second.press_final()
    button = slate.duplicate_screen(second.session, dup.duplicates).buttons[0][0][1]
    parsed = decode(button)
    assert second.manager.act(OWNER, *parsed).status is Status.APPLIED
    # a repeated tap of the override does nothing more
    assert second.manager.act(OWNER, *parsed).status is Status.REPEAT
    # changing the stake gives a new summary: the override no longer applies and the duplicate is caught again
    assert second.manager.submit_stake(OWNER, "750").status is Status.APPLIED
    assert second.press_final().status is Status.DUPLICATE
    first.close()


def test_a_leg_is_blocked_by_match_market_and_line_whatever_the_side_and_a_new_line_is_not(tmp_path):
    first = Rig(tmp_path, evals=[pick_evaluation("Alpha", "Beta", "Over/Under 2.5", "Over 2.5", "1.85")],
                listing=[make_fixture("Alpha", "Beta", [totals_offer("2.5", "1.85", "2.00"), totals_offer("3.5", "3.00", "1.30")], 14)])
    first.confirm_and_place()
    under = Rig(tmp_path, sid="bbbb2222", store=first.store,
                evals=[pick_evaluation("Alpha", "Beta", "Over/Under 2.5", "Over 2.5", "1.95")],
                listing=first.listing)
    under.to_summary()
    assert under.press_final().status is Status.DUPLICATE          # the same line at other odds: blocked (no Under exists, AD-20)
    other = Rig(tmp_path, sid="cccc3333", store=first.store,
                evals=[pick_evaluation("Alpha", "Beta", "Over/Under 3.5", "Over 3.5", "3.00")], listing=first.listing)
    other.to_summary()
    assert other.press_final().status is Status.APPLIED            # a different line is a different leg
    first.close()


def test_legs_of_an_unproven_placement_block_too_but_a_dry_run_never_does(tmp_path):
    first = Rig(tmp_path)
    first.bm.place_mode = "unknown"
    first.confirm_and_place()
    second = Rig(tmp_path, sid="bbbb2222", store=first.store)
    second.to_summary()
    result = second.press_final()
    assert result.status is Status.DUPLICATE and {d.source for d in result.duplicates} == {"unknown"}
    assert "may already exist" in slate.duplicate_screen(second.session, result.duplicates).text
    third_dir = tmp_path / "dry"
    third_dir.mkdir()
    dry = Rig(third_dir, dry_run=True)
    dry.confirm_and_place()
    again = Rig(third_dir, sid="bbbb2222", store=dry.store, dry_run=True)
    again.to_summary()
    assert again.press_final().status is Status.APPLIED            # dry run registers nothing and blocks nothing
    first.close()
    dry.close()


def test_the_registry_is_checked_again_at_placement_time_without_the_state_machine(tmp_path):
    first = Rig(tmp_path)
    first.confirm_and_place()
    second = Rig(tmp_path, sid="bbbb2222", store=first.store)
    second.to_summary()
    second.manager.attach(None, None)                       # the state machine's own check is off ...
    token = second.press_final().token
    outcome = second.place(token)                           # ... run_placement still refuses, before recording anything
    assert outcome.kind == OUTCOME_FAILED and "already placed" in outcome.message and second.bm.clicks == 0
    assert calls(second).count("place") == 0 and len(first.store.list_placements()) == 1
    first.close()


def test_an_unreadable_registry_fails_closed(rig):
    def broken(identities):
        raise RuntimeError("db locked")

    rig.manager.attach(rig.manager._persist, broken)
    rig.to_summary()
    result = rig.press_final()
    assert result.status is Status.BLOCKED and "nothing was placed" in result.notice and result.token is None
    assert rig.session.state is State.AWAITING_FINAL_CONFIRM


def test_the_registry_query_ignores_other_games_and_returns_what_it_knows(rig):
    rig.confirm_and_place()
    known = rig.store.find_blocking_legs([("nope", "18", "total=2.5")])
    assert known == []
    legs = rig.session.summary.legs
    found = rig.store.find_blocking_legs([leg_identity(legs[0].leg_key)])
    assert len(found) == 1 and isinstance(found[0], DuplicateLeg) and found[0].bet_id == "B1234567"


# ================================================================ 5.6: continue from a Review

def review_result(stats=None):
    """A ReviewResult from the fabricated sample: Arsenal approved, others not (Chelsea is below the floor, the
    basketball and tennis selections have no rules)."""
    from test_review import review
    _, seen, *_ = review()
    return seen[0]


class StrongStats(FakeStats):
    """Arsenal and Gor Mahia both clear the floor."""

    async def fetch_stats(self, fixtures):
        from test_review import inputs_for
        table = await super().fetch_stats(fixtures)
        for f in fixtures:
            if f.home == "Gor Mahia":
                table[f.match_id] = FixtureStats(f.match_id, READY, tuple(inputs_for(2.5, 2.0)))
        return table


def strong_review():
    from test_review import review
    _, seen, *_ = review(stats=StrongStats())
    return seen[0]


def test_only_approved_selections_carry_on_from_a_review_with_what_the_next_steps_need():
    result = review_result()
    assert len(result.judgements) == 6 and len(result.approved) == 1
    kept = approved_evaluations(result)
    (ev,) = kept
    assert ev is result.approved[0].evaluation and ev.passed and not ev.insufficient
    assert ev.market == "Over/Under 2.5" and ev.selection == "Over 2.5" and ev.odds == Decimal("1.85")
    assert ev.estimate.probability >= 0.65 and ev.estimate.inputs and ev.match_id.home == "Arsenal"
    from predictions.core.betting import leg_from_evaluation
    leg = leg_from_evaluation(ev)                            # it maps straight onto a betslip leg (market, line, side)
    assert (leg.market_id, leg.specifier, leg.side, leg.odds) == ("18", "total=2.5", "over", Decimal("1.85"))


def test_a_disqualified_selection_can_never_be_entered_or_offered():
    result = review_result()
    disqualified = [j.evaluation for j in result.disqualified]
    assert len(disqualified) == 5
    for ev in disqualified:
        with pytest.raises(ValueError):
            Slip.from_evaluations([ev])
    manager = SessionManager(clock=lambda: NOW)
    with pytest.raises(ValueError):
        manager.request_start(OWNER, [*approved_evaluations(result), disqualified[0]], "review:ABC123")
    assert manager.active is None
    # even a forged "approved" flag never lets a priceless or below-floor selection through
    good = result.approved[0]
    for forged in (replace(good.evaluation, insufficient=True), replace(good.evaluation, estimate=Estimate(0.5))):
        assert approved_evaluations(ReviewResult("X", (replace(good, evaluation=forged),), ())) == ()


def test_a_session_from_a_review_lists_only_approved_picks_and_disqualified_ids_are_unknown_buttons():
    result = review_result()
    manager = SessionManager(clock=lambda: NOW)
    started = manager.request_start(OWNER, approved_evaluations(result), "review:ABC123", SID)
    assert started.status is Status.APPLIED and len(started.session.slip) == 1
    for j in result.disqualified:
        pid = make_pick_id(j.evaluation)
        for action in (Action.DELETE, Action.RESTORE, Action.SKIP):
            assert manager.act(OWNER, SID, 1, action, pid).status in (Status.UNKNOWN_PICK, Status.STALE)
    assert len(manager.active.slip) == 1 and manager.active.state_version == 1


@pytest.fixture
def app_chain(monkeypatch, tmp_path):
    """The real handler chain for Matchday AND Review, a fake bookmaker, and a real store."""
    sent, edits, answers, events = [], [], [], []
    scenario = {"gate": None}
    evals, listing = two_pick_setup()

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

    async def fake_matchday(argument, source, sanitize, **kwargs):
        if scenario["gate"] is not None:
            kwargs["on_result"](scenario["gate"])
        return ["Matchday fixtures ..."]

    reader = FakeCodeReader(selections())
    monkeypatch.setattr(ExtBot, "initialize", no_network)
    monkeypatch.setattr(ExtBot, "send_message", send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", edit_message_text)
    monkeypatch.setattr(ExtBot, "answer_callback_query", answer_callback_query)
    monkeypatch.setattr(bot, "run_matchday", fake_matchday)
    monkeypatch.setattr(bot, "SportyBetCodeReader", lambda: reader)
    monkeypatch.setattr(bot, "review_stats_sources", lambda: {"soccer": scenario.get("stats") or StrongStats()})
    monkeypatch.setattr(bot.ClaudeAnalyst, "from_environment", classmethod(lambda cls, mode="full": None))
    store = EvaluationStore.open(tmp_path / "t.db", bot.redact)
    manager = SessionManager(clock=lambda: NOW)
    attach_store(manager, store)
    fake = FakeBookmaker(listing + review_listing(), dry_run=True)
    monkeypatch.setattr(bot, "SESSIONS", manager)
    monkeypatch.setattr(bot, "STORE", store)
    monkeypatch.setattr(bot, "BOOKMAKER", fake)
    monkeypatch.setattr(bot, "DRY_RUN", True)
    application = ApplicationBuilder().token("123456:TEST-TOKEN-NOT-REAL").build()
    bot.register_handlers(application)
    chain = Chain(application, sent, edits, answers)
    chain.scenario, chain.store, chain.events, chain.fake, chain.evals, chain.reader = scenario, store, events, fake, evals, reader
    yield chain
    store.close()


def review_listing():
    """SportyBet's live listing of the two games StrongStats approves, at slightly different odds than on the code."""
    out = []
    for s, over in ((selections()[0], "1.90"), (selections()[6], "1.45")):
        line = s.line
        fx = Fixture(s.match_id, s.competition, s.home, s.away, s.kickoff_utc, False,
                     (totals_offer(line, over, "2.10"),), f"sr:match:{s.home}")
        out.append(fx)
    return out


def press_by_label(chain, label):
    chain.press(OWNER_ID, find(chain.buttons(-1), label))


def test_review_keeps_its_text_and_adds_the_interactive_list_after_it(app_chain, monkeypatch):
    from test_review import freeze_review_clock
    freeze_review_clock(monkeypatch)
    chain = app_chain
    chain.text(OWNER_ID, "Review ABC123")
    texts = [t for t, _ in chain.sent]
    assert "APPROVED (2)" in texts[0] and "DISQUALIFIED" in texts[0]
    assert all(kw.get("reply_markup") is None for _, kw in chain.sent[:-1])
    text, kwargs = chain.sent[-1]
    assert text.startswith("Approved selections from Review ABC123: Kept 2 of 2 game(s)") and "%" in text
    assert "Disqualified selections are not here and cannot be added" in text
    labels = [b.text for row in kwargs["reply_markup"].inline_keyboard for b in row]
    assert labels == ["1 Keep", "1 Delete", "2 Keep", "2 Delete", "Final confirm"]
    assert chain.fake.calls == []                                   # Review itself never touched the betslip
    session = bot.SESSIONS.active
    assert session.origin == "review:ABC123" and {p.match_id.home for p in session.slip.picks} == {"Arsenal", "Gor Mahia"}
    rows = chain.store.list_evaluations(session.session_id)
    assert len(rows) == 6                                           # the review's evaluations are saved under this session


def test_the_owner_drops_an_approved_selection_and_the_rest_loads_fresh_onto_the_betslip(app_chain):
    chain = app_chain
    chain.text(OWNER_ID, "Review ABC123")
    text = chain.sent[-1][0]
    first = "Gor Mahia" if text.index("Gor Mahia") < text.index("Arsenal") else "Arsenal"
    press_by_label(chain, "1 Delete")
    assert len(bot.SESSIONS.active.slip) == 1 and bot.SESSIONS.active.slip.picks[0].match_id.home != first
    press_by_label(chain, "Final confirm")
    press_by_label(chain, "Confirm games")
    session = bot.SESSIONS.active
    assert session.state is State.AWAITING_STAKE and chain.fake.clicks == 0
    assert [c[0] for c in chain.fake.calls] == ["check", "load"]    # loaded fresh: the code's own slip is never used
    loaded_keys = chain.fake.calls[1][1]
    assert len(loaded_keys) == 1 and session.slip.picks[0].match_id.home.lower() in loaded_keys[0][0]
    odds_text = next(t for t, _ in chain.sent if t.startswith("Loaded on SportyBet"))
    assert "evaluation used" in odds_text                           # the code's odds vs SportyBet's odds are both shown
    assert any("How much do you want to stake" in t for t, _ in chain.sent)


def test_a_review_with_nothing_approved_offers_no_list_and_leaves_an_open_slate_alone(app_chain):
    chain = app_chain
    chain.scenario["stats"] = FakeStats()
    chain.reader.result = [s for s in selections() if s.home == "Chelsea"]      # below the floor
    chain.text(OWNER_ID, "Review ABC123")
    assert len(chain.sent) == 1 and chain.sent[0][1].get("reply_markup") is None and bot.SESSIONS.active is None
    chain.scenario["gate"] = gate_of(chain.evals)
    chain.text(OWNER_ID, "Matchday 10 October")
    chain.text(OWNER_ID, "Review ABC123")
    assert bot.SESSIONS.active.origin == "matchday" and len(bot.SESSIONS.active.slip) == 2


def test_two_approved_picks_on_one_game_make_the_owner_choose_one(app_chain):
    chain = app_chain
    extra = replace(selections()[0], market_id="18", specifier="total=1.5", outcome_name="Over 1.5", odds=Decimal("1.20"))
    chain.reader.result = [selections()[0], extra]
    chain.text(OWNER_ID, "Review ABC123")
    labels = [b.text for row in chain.sent[-1][1]["reply_markup"].inline_keyboard for b in row]
    assert labels == ["1a Keep", "1a Delete", "1b Keep", "1b Delete", "Final confirm"]   # one game, two plays, each its own row
    text = chain.sent[-1][0]
    assert "Kept 1 of 1 game(s)" in text and f"1a {slate.MARK_KEPT}" in text and "(advised)" in text and f"1b {slate.MARK_REMOVED}" in text
    assert len(bot.SESSIONS.active.slip) == 1                          # a game contributes exactly one play
    advised = bot.SESSIONS.active.slip.picks[0].pick_id
    press_by_label(chain, "1b Keep")
    assert bot.SESSIONS.active.slip.picks[0].pick_id != advised and len(bot.SESSIONS.active.slip) == 1
    press_by_label(chain, "Final confirm")
    assert bot.SESSIONS.active.state is State.REVIEWING_FINAL and chain.edits[-1][0].startswith("Final edit: 1 game(s) kept")


# ================================================================ 5.7: one slate at a time

def picks(*names):
    return [pick_evaluation(n, "Beta", "Over/Under 2.5", "Over 2.5", "1.85") for n in names]


def test_a_trigger_with_no_open_slate_starts_at_once_and_with_one_open_it_asks_first():
    manager = SessionManager(clock=lambda: NOW)
    first = manager.request_start(OWNER, picks("Alpha"), "matchday", "aaaa1111")
    assert first.status is Status.APPLIED and manager.active.session_id == "aaaa1111"
    second = manager.request_start(OWNER, picks("Gamma"), "review:XYZ", "bbbb2222")
    assert second.status is Status.ASK_REPLACE and second.pending.session_id == "bbbb2222"
    assert manager.active.session_id == "aaaa1111" and manager.active.state_version == 1   # nothing replaced yet
    screen = slate.start_screen(second)
    assert [label for label, _ in screen.buttons[0]] == ["Replace the open slate", "Keep the open slate"]
    assert "from Matchday" in screen.text and "from Review XYZ" in screen.text and "buttons stop working" in screen.text


def test_replace_needs_the_explicit_button_and_kills_the_old_buttons():
    manager = SessionManager(clock=lambda: NOW)
    manager.request_start(OWNER, picks("Alpha", "Delta"), "matchday", "aaaa1111")
    old_delete = encode("aaaa1111", 1, Action.DELETE, manager.active.slip.picks[0].pick_id)
    ask = manager.request_start(OWNER, picks("Gamma"), "matchday", "bbbb2222")
    replace_button = slate.start_screen(ask).buttons[0][0][1]
    reply = handle_callback(replace_button, OWNER, manager)
    assert reply.answer == "Replaced." and reply.screen.text.startswith("Final list: Kept 1 of 1 game(s)")
    assert manager.active.session_id == "bbbb2222" and manager.pending is None
    assert handle_callback(old_delete, OWNER, manager).answer.startswith("This list changed")   # the old button is dead
    assert manager.active.session_id == "bbbb2222" and len(manager.active.slip) == 1


def test_a_repeated_tap_on_replace_or_keep_has_no_extra_effect():
    manager = SessionManager(clock=lambda: NOW)
    manager.request_start(OWNER, picks("Alpha"), "matchday", "aaaa1111")
    ask = manager.request_start(OWNER, picks("Gamma"), "matchday", "bbbb2222")
    replace_button, keep_button = (b for _, b in slate.start_screen(ask).buttons[0])
    first = handle_callback(replace_button, OWNER, manager)
    session = manager.active
    again = handle_callback(replace_button, OWNER, manager)
    assert first.answer == "Replaced." and again.answer == "Already done." and again.screen is None
    assert manager.active is session
    assert handle_callback(keep_button, OWNER, manager).answer in ("This list changed. Please use the newest list.",
                                                                    "Already done.")
    assert manager.active is session


def test_keep_drops_the_new_request_and_the_open_slate_carries_on():
    manager = SessionManager(clock=lambda: NOW)
    manager.request_start(OWNER, picks("Alpha", "Delta"), "matchday", "aaaa1111")
    ask = manager.request_start(OWNER, picks("Gamma"), "matchday", "bbbb2222")
    replace_button, keep_button = (b for _, b in slate.start_screen(ask).buttons[0])
    reply = handle_callback(keep_button, OWNER, manager)
    assert reply.answer == "Kept." and "Kept the open slate" in reply.screen.text
    assert manager.active.session_id == "aaaa1111" and manager.pending is None
    assert handle_callback(replace_button, OWNER, manager).answer.startswith("This list changed")   # the prompt is dead
    assert handle_callback(keep_button, OWNER, manager).answer == "Already done."
    delete = encode("aaaa1111", 1, Action.DELETE, manager.active.slip.picks[0].pick_id)
    assert handle_callback(delete, OWNER, manager).answer == "Deleted."                     # the open slate still works


def test_a_prompt_dies_when_a_newer_trigger_arrives_or_the_open_slate_changes():
    manager = SessionManager(clock=lambda: NOW)
    manager.request_start(OWNER, picks("Alpha", "Delta"), "matchday", "aaaa1111")
    old_ask = manager.request_start(OWNER, picks("Gamma"), "matchday", "bbbb2222")
    new_ask = manager.request_start(OWNER, picks("Echo"), "matchday", "cccc3333")
    assert handle_callback(slate.start_screen(old_ask).buttons[0][0][1], OWNER, manager).answer.startswith("This list changed")
    assert manager.active.session_id == "aaaa1111"
    delete = encode("aaaa1111", 1, Action.DELETE, manager.active.slip.picks[0].pick_id)
    handle_callback(delete, OWNER, manager)                                # the open slate changed: version 2
    assert handle_callback(slate.start_screen(new_ask).buttons[0][0][1], OWNER, manager).answer.startswith("This list changed")
    assert manager.active.session_id == "aaaa1111"


@pytest.mark.parametrize("state", [State.DONE, State.FAILED, State.CANCELLED, State.PLACEMENT_UNKNOWN])
def test_a_finished_or_cancelled_slate_does_not_need_asking(state):
    manager = SessionManager(clock=lambda: NOW)
    manager.start(picks("Alpha"), "aaaa1111")
    manager.restore(replace(manager.active, state=state))
    result = manager.request_start(OWNER, picks("Gamma"), "matchday", "bbbb2222")
    assert result.status is Status.APPLIED and manager.active.session_id == "bbbb2222"


@pytest.mark.parametrize("state", [State.PLACING, State.RECONCILING])
def test_a_slate_with_a_bet_in_flight_is_never_replaced(state):
    manager = SessionManager(clock=lambda: NOW)
    manager.start(picks("Alpha"), "aaaa1111")
    ask = manager.request_start(OWNER, picks("Gamma"), "matchday", "bbbb2222")
    manager.restore(replace(manager.active, state=state))
    blocked = manager.request_start(OWNER, picks("Echo"), "matchday", "cccc3333")
    assert blocked.status is Status.BLOCKED and "bet is being placed" in blocked.notice
    stale = manager.act(OWNER, "aaaa1111", 1, Action.REPLACE, ask.pending.pending_id)
    assert stale.status in (Status.STALE, Status.BLOCKED) and manager.active.session_id == "aaaa1111"
    manager.end()
    assert manager.active.state is state                                # end() never cancels a slate with a bet in flight
    assert manager.active.session_id == "aaaa1111"


def test_replace_is_refused_if_a_bet_started_between_the_question_and_the_answer():
    manager = SessionManager(clock=lambda: NOW)
    manager.start(picks("Alpha"), "aaaa1111")
    ask = manager.request_start(OWNER, picks("Gamma"), "matchday", "bbbb2222")
    busy = replace(manager.active, state=State.PLACING)                # same version: the buttons still "match"
    manager._active = busy
    result = manager.act(OWNER, "aaaa1111", 1, Action.REPLACE, ask.pending.pending_id)
    assert result.status is Status.BLOCKED and manager.active.session_id == "aaaa1111"


def test_matchday_and_review_both_ask_before_replacing_and_old_buttons_die_after(app_chain):
    chain = app_chain
    chain.scenario["gate"] = gate_of(chain.evals)
    chain.text(OWNER_ID, "Matchday 10 October")
    matchday_buttons = chain.buttons(-1)
    assert bot.SESSIONS.active.origin == "matchday"
    chain.text(OWNER_ID, "Review ABC123")                                 # a Review while a Matchday slate is open
    assert chain.sent[-1][0].startswith("You already have a slate from Matchday with 2 game(s)")
    assert "from Review ABC123" in chain.sent[-1][0]
    assert bot.SESSIONS.active.origin == "matchday" and bot.SESSIONS.active.state_version == 1
    labels = [t for t, _ in chain.buttons(-1)]
    assert labels == ["Replace the open slate", "Keep the open slate"]
    chain.press(STRANGER, find(chain.buttons(-1), "Replace the open slate"))        # the owner gate drops a stranger
    assert bot.SESSIONS.active.origin == "matchday"
    chain.press(OWNER_ID, find(chain.buttons(-1), "Replace the open slate"))
    assert bot.SESSIONS.active.origin == "review:ABC123"
    assert chain.edits[-1][0].startswith("Approved selections from Review ABC123")
    chain.press(OWNER_ID, find(matchday_buttons, "1 Delete"))              # a button of the old Matchday slate
    assert chain.answers[-1][0].startswith("This list changed") and len(bot.SESSIONS.active.slip) == 2
    # and now a Matchday while the Review slate is open asks as well
    chain.text(OWNER_ID, "Matchday 10 October")
    assert chain.sent[-1][0].startswith("You already have a slate from Review ABC123")
    chain.press(OWNER_ID, find(chain.buttons(-1), "Keep the open slate"))
    assert bot.SESSIONS.active.origin == "review:ABC123" and "Kept the open slate" in chain.edits[-1][0]


def test_a_matchday_after_a_finished_slate_starts_without_asking(app_chain):
    chain = app_chain
    chain.scenario["gate"] = gate_of(chain.evals)
    chain.text(OWNER_ID, "Matchday 10 October")
    sid = bot.SESSIONS.active.session_id
    bot.SESSIONS.restore(replace(bot.SESSIONS.active, state=State.DONE))
    chain.text(OWNER_ID, "Matchday 10 October")
    assert chain.sent[-1][0].startswith("Final list") and bot.SESSIONS.active.session_id != sid


def test_the_replaced_slate_is_saved_as_cancelled_so_a_restart_never_revives_it(app_chain):
    chain = app_chain
    chain.scenario["gate"] = gate_of(chain.evals)
    chain.text(OWNER_ID, "Matchday 10 October")
    old = bot.SESSIONS.active.session_id
    chain.text(OWNER_ID, "Review ABC123")
    chain.press(OWNER_ID, find(chain.buttons(-1), "Replace the open slate"))
    states = {r["session_id"]: r["state"] for r in chain.store.list_sessions()}
    assert states[old] == "CANCELLED" and states[bot.SESSIONS.active.session_id] == "REVIEWING_LIST"


def test_the_callback_encoding_covers_the_new_actions_within_telegrams_limit():
    for action in (Action.REPLACE, Action.KEEP, Action.OVERRIDE):
        data = encode("aaaa1111", 123456789, action, "0123456789ab")
        assert len(data.encode()) <= 64 and decode(data) == ("aaaa1111", 123456789, action, "0123456789ab")
