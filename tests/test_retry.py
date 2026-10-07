"""Owner-pressed retry after a failure (owner request, FR-10, AD-6, AD-14, AD-19, AD-22, AD-24).

A retry is ALWAYS pressed by the owner and NEVER automatic. A failure before the click offers 'Try again'; an unknown
result (Place was clicked) offers 'Check history again' (read-only) and, only after a check found nothing, 'Try again, I
checked' behind a loud warning. Every retry reloads the betslip, shows a NEW summary and needs a NEW confirmation.

Everything runs against the FakeBookmaker and a real SQLite store; nothing is placed and nothing logs in anywhere."""
import ast
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from telegram import User
from telegram.ext import ApplicationBuilder, ExtBot

import app as bot
from fakes_bookmaker import NOW, FakeBookmaker, two_pick_setup
from predictions.core.ports import BookmakerError, PlacementUnknown
from predictions.core.types import Owner
from predictions.flow.loading import KIND_FAILED, KIND_LOADED, run_load
from predictions.flow.placement import OUTCOME_FAILED, OUTCOME_PLACED, OUTCOME_UNKNOWN, run_placement
from predictions.flow.recovery import attach_store, recover_on_startup
from predictions.flow.retry import run_history_check
from predictions.flow.session import (MAX_RETRIES, Action, Session, SessionManager, State, Status)
from predictions.flow.snapshot import from_json, session_to_data, to_json
from predictions.flow.store import EvaluationStore
from predictions.telegram.slate import (CALLBACK_LIMIT, CHECK_LABEL, RETRY_CHECKED_LABEL, TRY_AGAIN_LABEL, YES_AGAIN_LABEL,
                                        decode, encode, failure_screen, handle_callback, slate_screen)
from test_epic5_app import live_chain, to_stake  # noqa: F401  (the fixture is used by name)
from test_epic5_recovery import Rig, history_bet
from test_final_list import find

OWNER = Owner(111222333)
OWNER_ID = 111222333
REPO = Path(__file__).resolve().parent.parent


def run(coro):
    return asyncio.run(coro)


class RetryRig(Rig):
    """The state machine attached to a real store (registry and audit included), with buttons pressed like Telegram does."""

    def tick(self, seconds=2):
        self.now = self.now + timedelta(seconds=seconds)

    def screen(self):
        return slate_screen(self.session, False, self.now)

    def labels(self):
        return [label for row in self.screen().buttons for label, _ in row]

    def data_of(self, label):
        return next(data for row in self.screen().buttons for text, data in row if text == label)

    def press(self, label):
        self.tick()
        return handle_callback(self.data_of(label), OWNER, self.manager, dry_run=False)

    def drive_load(self):
        """What app.py does after a load job: the odds, then (for a retry) the carried stake into a NEW summary."""
        assert self.load().kind == KIND_LOADED
        self.manager.mark_odds_shown(self.sid, self.session.state_version)
        if self.session.retry_stake is not None:
            assert self.manager.apply_retry_stake(self.sid, self.session.state_version).status is Status.APPLIED
        return self.session

    def check(self):
        self.tick()
        reply = self.press_label_unchecked(CHECK_LABEL)
        assert reply.job is not None and reply.job.kind == "check"
        return run(run_history_check(self.manager, self.store, self.bm, self.sid, str))

    def press_label_unchecked(self, label):
        return handle_callback(self.data_of(label), OWNER, self.manager, dry_run=False)

    def places(self):
        return [c for c in self.bm.calls if c[0] == "place"]

    def fail_at_place(self, error=None, stake="500"):
        """To the summary, confirm, and a placement that fails BEFORE the click."""
        self.to_summary(stake)
        token = self.press_final().token
        if error is not None:
            async def refuse(_token):
                self.bm.calls.append(("place", _token.token_hash))
                raise error
            self.bm.place = refuse
        out = self.place(token)
        assert out.kind == OUTCOME_FAILED and self.session.state is State.FAILED, out
        return token

    def make_unknown(self, stake="500"):
        self.bm.place_mode = "unknown"
        outcome = self.confirm_and_place(stake)
        assert outcome.kind == OUTCOME_UNKNOWN and self.session.state is State.PLACEMENT_UNKNOWN
        return outcome


@pytest.fixture
def rig(tmp_path):
    r = RetryRig(tmp_path, dry_run=False)
    yield r
    r.close()


# ================================================================ 1. FAILED before the click -> Try again

def check_error(rig):
    rig.bm.check_error = BookmakerError("The bet history could not be read (no entries and no empty-list marker).")


def login_error(rig):
    rig.bm.check_error = BookmakerError("SportyBet asked for a login and no login details are available.")


def load_error(rig):
    rig.bm.load_error = BookmakerError("The betslip could not be emptied.")


def mismatch_at_load(rig):
    original = rig.bm.load_slip

    async def wrong(legs):
        slip = await original(legs)
        return replace(slip, legs=slip.legs[:1], total_odds=slip.legs[0].odds)

    rig.bm.load_slip = wrong


@pytest.mark.parametrize("setup", [check_error, login_error, load_error, mismatch_at_load],
                         ids=["history unreadable", "login problem", "betslip cannot be emptied", "betslip mismatch"])
def test_try_again_after_a_load_failure_reloads_the_same_plays_and_asks_for_the_stake(rig, setup):
    setup(rig)
    rig.confirm()
    assert rig.load().kind == KIND_FAILED
    session = rig.session
    assert session.state is State.FAILED and session.retryable
    screen = rig.screen()
    assert screen.text.startswith("No bet was placed.") and rig.labels() == [TRY_AGAIN_LABEL]
    assert all(len(data.encode()) <= CALLBACK_LIMIT for row in screen.buttons for _, data in row)
    calls_before = list(rig.bm.calls)
    kept = session.slip.picks
    rig.bm.check_error = rig.bm.load_error = None
    rig.bm.load_slip = FakeBookmaker.load_slip.__get__(rig.bm)   # the problem is gone
    reply = rig.press(TRY_AGAIN_LABEL)
    assert reply.job.kind == "load" and rig.bm.calls == calls_before          # pressing only moves the state
    assert rig.session.state is State.LOADING_SLIP and rig.session.retries_used == 1 and rig.session.slip.picks == kept
    rig.drive_load()
    assert rig.session.state is State.AWAITING_STAKE and rig.session.stake is None   # no stake had been given
    assert not rig.places() and rig.bm.clicks == 0


@pytest.mark.parametrize("error", [
    BookmakerError("The stake box was not found on the betslip."),
    BookmakerError("The SportyBet betslip no longer matches what you confirmed: a selection is missing. Nothing was placed."),
    BookmakerError("SportyBet is not logged in. Nothing was placed."),
    BookmakerError("Placing took too long before anything was clicked. Nothing was placed."),
], ids=["stake box", "betslip mismatch", "login", "timeout before the click"])
def test_try_again_after_a_placement_failure_before_the_click_gives_a_new_summary_with_the_same_stake(rig, error):
    old_token = rig.fail_at_place(error, "750")
    old_summary_id = rig.session.summary.summary_id
    old_version = rig.session.state_version
    screen = rig.screen()
    assert screen.text.startswith("No bet was placed.") and rig.labels() == [TRY_AGAIN_LABEL]
    assert "Nothing is retried on its own" in screen.text or "Nothing is placed until you confirm" in screen.text
    placed_before = len(rig.places())
    rig.bm.place = FakeBookmaker.place.__get__(rig.bm)
    rig.press(TRY_AGAIN_LABEL)
    assert rig.session.state is State.LOADING_SLIP and rig.session.retry_stake == Decimal("750")
    assert rig.session.summary is None and rig.session.token_hash == ""
    rig.drive_load()
    new = rig.session
    assert new.state is State.AWAITING_FINAL_CONFIRM and new.stake == Decimal("750") and new.retries_used == 1
    assert new.summary.summary_id != old_summary_id and len(rig.places()) == placed_before   # nothing placed yet
    # the old confirm button (old version, old summary) is dead
    assert rig.manager.act(OWNER, rig.sid, old_version, Action.FINAL, old_summary_id).status is Status.STALE
    assert len(rig.places()) == placed_before
    # the owner confirms the NEW summary: a NEW token places
    result = rig.press_final()
    assert result.status is Status.APPLIED and result.token.token_hash != old_token.token_hash
    assert result.token.stake == Decimal("750")
    out = rig.place(result.token)
    assert out.kind == OUTCOME_PLACED and rig.bm.clicks == 1 and len(rig.places()) == placed_before + 1


def test_the_retry_summary_shows_the_winnings_banner_and_change_amount(rig):
    rig.fail_at_place(BookmakerError("The stake box was not found on the betslip."))
    rig.bm.place = FakeBookmaker.place.__get__(rig.bm)
    rig.press(TRY_AGAIN_LABEL)
    rig.drive_load()
    from predictions.telegram.slate import LIVE_BANNER, retry_notice, summary_screen
    screen = summary_screen(rig.session, dry_run=False, notice=retry_notice(rig.session))
    assert screen.text.splitlines()[0] == LIVE_BANNER and "Potential winnings: NGN" in screen.text
    assert "Retry 1 of 3" in screen.text and "Nothing is placed until you confirm" in screen.text
    assert [label for row in screen.buttons for label, _ in row] == ["Confirm and place bet", "Change amount"]


def test_a_retry_never_places_by_itself_and_a_reload_alone_never_confirms(rig):
    rig.fail_at_place(BookmakerError("The stake box was not found on the betslip."))
    rig.bm.place = FakeBookmaker.place.__get__(rig.bm)
    rig.press(TRY_AGAIN_LABEL)
    rig.drive_load()
    assert rig.session.state is State.AWAITING_FINAL_CONFIRM and rig.session.token_hash == ""
    assert rig.bm.clicks == 0 and len(rig.places()) == 1                  # only the first (failed) attempt ever reached place


def test_the_cap_of_three_retries_per_slate(rig):
    rig.bm.place_mode = "error"
    rig.to_summary("500")
    out = rig.place(rig.press_final().token)
    assert out.kind == OUTCOME_FAILED
    for attempt in range(1, MAX_RETRIES + 1):
        assert rig.labels() == [TRY_AGAIN_LABEL] and f"{MAX_RETRIES - attempt + 1} of {MAX_RETRIES} retries left" in rig.screen().text
        rig.press(TRY_AGAIN_LABEL)
        assert rig.session.retries_used == attempt
        rig.drive_load()
        out = rig.place(rig.press_final().token)
        assert out.kind == OUTCOME_FAILED
    assert rig.labels() == [] and "start again" in rig.screen().text and f"All {MAX_RETRIES} retries" in rig.screen().text
    stale_button = encode(rig.sid, rig.session.state_version, Action.RETRY)
    reply = handle_callback(stale_button, OWNER, rig.manager, dry_run=False)
    assert rig.session.state is State.FAILED and rig.session.retries_used == MAX_RETRIES
    assert "start again" in reply.answer and reply.alert and reply.job is None


def test_repeat_and_stale_taps_on_try_again(rig):
    rig.fail_at_place(BookmakerError("The stake box was not found on the betslip."))
    data = rig.data_of(TRY_AGAIN_LABEL)
    rig.tick()
    first = handle_callback(data, OWNER, rig.manager, dry_run=False)
    assert first.job is not None and rig.session.retries_used == 1
    version = rig.session.state_version
    again = handle_callback(data, OWNER, rig.manager, dry_run=False)     # the same tap again: idempotent
    assert again.job is None and again.answer == "Already done." and rig.session.state_version == version
    assert rig.session.retries_used == 1
    rig.bm.place = FakeBookmaker.place.__get__(rig.bm)
    rig.drive_load()
    late = handle_callback(data, OWNER, rig.manager, dry_run=False)      # long after: stale, nothing happens
    assert late.job is None and rig.session.retries_used == 1 and rig.session.state is State.AWAITING_FINAL_CONFIRM
    other = handle_callback(encode("ffffffff", 1, Action.RETRY), OWNER, rig.manager, dry_run=False)
    assert other.job is None and other.answer


def test_a_failure_that_is_not_certainly_before_the_click_has_no_try_again_button(rig):
    # a refusal by the placed-leg registry and the 'Under' / whole-line safety nets are not retryable
    rig.fail_at_place(BookmakerError("x"))
    not_retryable = replace(rig.session, retryable=False, state_version=rig.session.state_version + 1)
    rig.manager.restore(not_retryable)
    assert rig.labels() == [] and "Send a new request to start again." in rig.screen().text
    assert rig.manager.act(OWNER, rig.sid, not_retryable.state_version, Action.RETRY).status is Status.STALE


def test_a_real_refusal_shows_its_reason_first_and_try_again_is_only_a_button(rig):
    rig.bm.check_error = BookmakerError("A confirmed game or option is no longer available. Nothing was loaded.")
    rig.confirm()
    rig.load()
    text = rig.screen().text
    assert text.splitlines()[0] == "No bet was placed." and "no longer available" in text.splitlines()[2]
    assert "real refusal by SportyBet" in text and rig.labels() == [TRY_AGAIN_LABEL]
    assert rig.session.state is State.FAILED and [c[0] for c in rig.bm.calls] == ["check"]   # nothing happened by itself


def test_odds_changed_still_needs_a_fresh_confirmation_and_never_places_by_itself(rig):
    rig.bm.place_mode = "odds_changed"
    rig.to_summary("500")
    out = rig.place(rig.press_final().token)
    assert out.kind == "odds_changed" and rig.session.state is State.AWAITING_FINAL_CONFIRM and rig.bm.clicks == 0
    assert len(rig.places()) == 1


# ================================================================ 2. PLACEMENT_UNKNOWN -> Check history again

def test_unknown_screen_says_a_bet_may_exist_and_offers_only_the_read_only_check(rig):
    async def unknown(token):
        rig.bm.calls.append(("place", token.token_hash))
        rig.bm.clicks += 1
        raise PlacementUnknown('Place was clicked but the bet was not found. After Place the page showed: "Insufficient '
                               'balance" || server: code 11000.')

    rig.bm.place = unknown
    rig.to_summary("500")
    out = rig.place(rig.press_final().token)
    assert out.kind == OUTCOME_UNKNOWN
    rig.tick(180)
    text = rig.screen().text
    assert text.startswith("A bet MAY exist.") and "No bet was placed" not in text
    assert "The Place button was clicked about 3 minutes ago." in text
    assert 'After Place the page showed: "Insufficient balance"' in text and "server: code 11000" in text
    assert rig.labels() == [CHECK_LABEL]                                  # no second attempt before a check
    assert "appears only after a history check" in text
    s = rig.session
    assert rig.manager.act(OWNER, rig.sid, s.state_version, Action.RETRY_ASK).status is Status.BLOCKED
    assert rig.manager.act(OWNER, rig.sid, s.state_version, Action.RETRY_YES).status is Status.BLOCKED
    assert rig.manager.act(OWNER, rig.sid, s.state_version, Action.RETRY).status is Status.STALE
    assert rig.session.state is State.PLACEMENT_UNKNOWN


def test_check_history_again_finds_the_bet_and_registers_its_legs(rig):
    rig.make_unknown()
    assert rig.store.list_placed_legs() == [] and rig.labels() == [CHECK_LABEL]
    rig.bm.history = (history_bet("B4242"),)
    placed_before = len(rig.places())
    session = rig.check()
    assert session.state is State.DONE and "EXISTS" in session.message and "B4242" in session.message
    (row,) = rig.store.list_placements()
    assert row["state"] == "PLACED" and row["bet_id"] == "B4242" and len(rig.store.list_placed_legs()) == 2
    assert len(rig.places()) == placed_before and rig.bm.clicks == 1      # read-only: nothing was placed
    assert rig.labels() == []                                             # nothing left to retry


def test_check_history_again_not_found_says_so_and_unlocks_the_second_attempt(rig):
    rig.make_unknown()
    rig.press(CHECK_LABEL)
    assert rig.session.state is State.RECONCILING
    session = run(run_history_check(rig.manager, rig.store, rig.bm, rig.sid, str))
    assert session.state is State.PLACEMENT_UNKNOWN and session.history_checked
    text = rig.screen().text
    assert "NOT found" in text and "can lag by a minute" in text and text.startswith("A bet MAY exist.")
    assert rig.labels() == [CHECK_LABEL, RETRY_CHECKED_LABEL]
    assert rig.bm.clicks == 1 and len(rig.places()) == 1
    (row,) = rig.store.list_placements()
    assert row["state"] == "PLACEMENT_UNKNOWN"


def test_check_history_again_unreadable_says_so_and_does_not_unlock(rig):
    rig.make_unknown()

    async def unreadable():
        raise BookmakerError("The bet history could not be read (no entries and no empty-list marker).")

    rig.bm.bet_history = unreadable
    session = rig.check()
    assert session.state is State.PLACEMENT_UNKNOWN and not session.history_checked
    text = rig.screen().text
    assert "could not read your SportyBet bet history" in text and rig.labels() == [CHECK_LABEL]


def test_check_history_again_with_two_look_alikes_does_not_unlock(rig):
    rig.make_unknown()
    rig.bm.history = (history_bet("B1"), history_bet("B2"))
    session = rig.check()
    assert session.state is State.PLACEMENT_UNKNOWN and not session.history_checked
    assert "WARNING" in rig.screen().text and rig.labels() == [CHECK_LABEL]


def test_check_history_is_idempotent_and_blocks_other_work_while_it_runs(rig):
    rig.make_unknown()
    data = rig.data_of(CHECK_LABEL)
    first = handle_callback(data, OWNER, rig.manager, dry_run=False)
    assert first.job.kind == "check" and rig.session.state is State.RECONCILING
    second = handle_callback(data, OWNER, rig.manager, dry_run=False)
    assert second.job is None and second.answer == "Already done."
    result = rig.manager.request_start(OWNER, rig.evals)
    assert result.status is Status.BLOCKED                                # a bet may be in flight: nothing new starts


def test_try_again_i_checked_shows_the_loud_warning_cancel_changes_nothing(rig):
    rig.make_unknown()
    rig.check()
    version = rig.session.state_version
    reply = rig.press(RETRY_CHECKED_LABEL)
    text = reply.screen.text
    assert "If the first attempt actually went through, this will place a SECOND bet." in text
    assert "Check your SportyBet bet history first." in text and "Press 'Yes, place again' only if you have checked." in text
    assert [label for row in reply.screen.buttons for label, _ in row] == [YES_AGAIN_LABEL, "Cancel"]
    assert rig.session.state_version == version and rig.session.retries_used == 0   # a view: nothing changed
    cancel = next(data for row in reply.screen.buttons for label, data in row if label == "Cancel")
    back = handle_callback(cancel, OWNER, rig.manager, dry_run=False)
    assert back.screen.text.startswith("A bet MAY exist.") and rig.session.state is State.PLACEMENT_UNKNOWN
    assert rig.session.state_version == version and back.job is None


def test_yes_place_again_records_the_override_then_reloads_and_needs_a_new_confirmation(rig):
    rig.make_unknown("640")
    rig.check()
    (old,) = rig.store.list_placements()
    old_hash = old["token_hash"]
    ask = rig.press(RETRY_CHECKED_LABEL)
    yes = next(data for row in ask.screen.buttons for label, data in row if label == YES_AGAIN_LABEL)
    rig.tick()
    reply = handle_callback(yes, OWNER, rig.manager, dry_run=False)
    assert reply.job.kind == "load" and rig.session.state is State.LOADING_SLIP and rig.session.retry_override
    assert rig.session.retries_used == 1 and rig.session.retry_stake == Decimal("640")
    # the audit record: that the owner chose it, when, and for which placement id
    (row,) = rig.store.list_placements()
    assert row["state"] == "PLACEMENT_UNKNOWN" and "Yes, place again" in row["detail"]
    assert f"placement #{row['id']}" in row["detail"] and "20" in row["detail"]
    # the old unknown row still blocks its legs
    from predictions.core.betting import leg_identity
    identities = [leg_identity(l.leg_key) for l in rig.bm.loaded.legs]
    assert len(rig.store.find_blocking_legs(identities)) == 2
    rig.bm.place_mode = "ok"
    clicks_before, places_before = rig.bm.clicks, len(rig.places())
    rig.drive_load()
    new = rig.session
    assert new.state is State.AWAITING_FINAL_CONFIRM and new.stake == Decimal("640") and new.overrides
    assert new.override_summary == new.summary.summary_id and not new.retry_override
    assert rig.bm.clicks == clicks_before and len(rig.places()) == places_before   # nothing placed by the reload
    token = rig.press_final().token
    assert token.token_hash != old_hash
    out = rig.place(token)
    assert out.kind == OUTCOME_PLACED and rig.bm.clicks == clicks_before + 1
    rows = rig.store.list_placements()
    assert len(rows) == 2 and {r["state"] for r in rows} == {"PLACEMENT_UNKNOWN", "PLACED"}   # the old row stays UNKNOWN
    unknown = next(r for r in rows if r["state"] == "PLACEMENT_UNKNOWN")
    assert unknown["token_hash"] == old_hash


def test_without_the_explicit_yes_the_old_unknown_legs_still_block_a_new_slate_of_the_same_games(rig):
    rig.make_unknown()
    from predictions.core.betting import leg_identity
    blocked = rig.store.find_blocking_legs([leg_identity(l.leg_key) for l in rig.bm.loaded.legs])
    assert len(blocked) == 2 and all(not d.proven for d in blocked)
    rig.sid = "newslate"                           # the owner starts the same games again, with no override
    rig.manager.start(rig.evals, "newslate", "matchday")
    rig.bm.place_mode = "ok"
    rig.to_summary("500")
    assert rig.press_final().status is Status.DUPLICATE and rig.bm.clicks == 1


def test_the_second_attempt_is_refused_when_the_legs_are_already_proven_placed(rig):
    rig.make_unknown()
    rig.check()
    row = rig.store.list_placements()[0]
    from predictions.flow.reconcile import legs_from_row
    rig.store.resolve_placement_found(row["token_hash"], "B9", "", "found elsewhere", legs_from_row(row))
    result = rig.manager.act(OWNER, rig.sid, rig.session.state_version, Action.RETRY_ASK)
    assert result.status is Status.BLOCKED and "already placed" in result.notice
    assert rig.session.state is State.PLACEMENT_UNKNOWN


def test_the_second_attempt_is_refused_when_the_choice_cannot_be_recorded_or_the_registry_is_down(rig):
    rig.make_unknown()
    rig.check()
    audit = rig.manager._audit
    rig.manager._audit = None
    assert rig.manager.act(OWNER, rig.sid, rig.session.state_version, Action.RETRY_YES).status is Status.BLOCKED
    rig.manager._audit = lambda *a: (_ for _ in ()).throw(RuntimeError("disk"))
    assert rig.manager.act(OWNER, rig.sid, rig.session.state_version, Action.RETRY_YES).status is Status.BLOCKED
    rig.manager._audit = audit
    registry = rig.manager._registry
    rig.manager._registry = lambda ids: (_ for _ in ()).throw(RuntimeError("db"))
    assert rig.manager.act(OWNER, rig.sid, rig.session.state_version, Action.RETRY_ASK).status is Status.BLOCKED
    assert rig.session.state is State.PLACEMENT_UNKNOWN and rig.session.retries_used == 0
    rig.manager._registry = registry


def test_the_cap_counts_unknown_overrides_too(rig):
    rig.make_unknown()
    rig.check()
    used = replace(rig.session, retries_used=MAX_RETRIES)
    rig.manager.restore(used)
    assert rig.labels() == [CHECK_LABEL] and f"All {MAX_RETRIES} retries" in rig.screen().text
    assert rig.manager.act(OWNER, rig.sid, rig.session.state_version, Action.RETRY_YES).status is Status.BLOCKED


# ================================================================ 4. persistence and recovery

def test_the_new_fields_round_trip_and_old_snapshots_still_load(rig):
    rig.make_unknown()
    session = replace(rig.session, retries_used=2, history_checked=True, retry_stake=Decimal("123.50"),
                      retry_override=True, check_text="checked", retryable=True)
    assert from_json(to_json(session)) == session
    data = session_to_data(session)
    for key in ("retries_used", "retryable", "history_checked", "retry_stake", "retry_override", "unknown_at", "check_text"):
        del data[key]
    from predictions.flow.snapshot import session_from_data
    old = session_from_data(data)
    assert (old.retries_used, old.retryable, old.history_checked, old.retry_stake, old.retry_override,
            old.unknown_at, old.check_text) == (0, False, False, None, False, None, "")


def test_after_a_restart_the_unknown_slate_keeps_its_retry_count_and_the_checked_marker(rig):
    rig.make_unknown()
    rig.check()
    rig.manager.restore(replace(rig.session, retries_used=2))
    reports = rig.restart()
    session = rig.manager.active
    assert session.state is State.PLACEMENT_UNKNOWN and session.retries_used == 2 and session.history_checked
    assert any("did not find" in r.text for r in reports) and rig.bm.clicks == 1


def test_a_restored_slate_after_loading_returns_to_loading_with_a_fresh_stake_request(rig):
    rig.fail_at_place(BookmakerError("The stake box was not found on the betslip."))
    rig.press(TRY_AGAIN_LABEL)
    rig.drive_load()
    assert rig.session.state is State.AWAITING_FINAL_CONFIRM and rig.session.retries_used == 1
    rig.manager.restore(replace(rig.session, retry_override=True, retry_stake=Decimal("500"), history_checked=True))
    rig.restart()
    session = rig.manager.active
    assert session.state is State.LOADING_SLIP and session.retries_used == 1            # the cap cannot be dodged
    assert session.retry_stake is None and not session.retry_override and not session.history_checked
    assert session.stake is None and session.summary is None and session.token_hash == ""


# ================================================================ 5. never automatic

def _calls_in(path, attr):
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == attr]


def test_bookmaker_place_is_only_called_with_a_token_the_state_machine_minted():
    callers = sorted(str(p.relative_to(REPO)).replace("\\", "/") for p in [*REPO.glob("predictions/**/*.py"), REPO / "app.py"]
                     if _calls_in(p.relative_to(REPO), "place"))
    assert callers == ["predictions/flow/placement.py", "predictions/tools/betslip_smoke.py"]
    # a token is minted in exactly one place: flow/session.py, inside the final confirmation
    minters = [p for p in [*REPO.glob("predictions/**/*.py"), REPO / "app.py"]
               if "mint_token(" in p.read_text(encoding="utf-8") and p.name not in ("placement.py",)]
    assert [m.name for m in minters if m.name != "session.py"] == []
    source = (REPO / "predictions/flow/session.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    mint_in = [f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
               and any(isinstance(c, ast.Call) and getattr(c.func, "id", "") == "mint_token" for c in ast.walk(f))]
    assert mint_in == ["_final_confirm"]
    # _final_confirm is reached only from act(), for the owner's final-confirm tap
    reached = [f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
               and any(isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "_final_confirm" for c in ast.walk(f))]
    assert reached == ["act"]
    # the retry, check and reconcile modules can never place or mint
    for module in ("predictions/flow/retry.py", "predictions/flow/reconcile.py", "predictions/flow/recovery.py",
                   "predictions/flow/loading.py"):
        text = (REPO / module).read_text(encoding="utf-8")
        assert not _calls_in(module, "place") and "mint_token" not in text and "run_placement" not in text


def test_a_load_is_only_started_by_an_owner_press_or_by_recovery_and_run_placement_only_with_a_job_token():
    tree = ast.parse((REPO / "predictions/flow/session.py").read_text(encoding="utf-8"))
    loaders = sorted(f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
                     and any(isinstance(k, ast.keyword) and k.arg == "load_next" for c in ast.walk(f)
                             if isinstance(c, ast.Call) for k in c.keywords))
    assert loaders == ["_act_on_final", "_act_on_partial", "_reload", "_resolve"]   # confirm games, owner skip/accept, owner retry, owner Retry the failed picks (AD-26)
    retry = next(f for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) and f.name == "_retry")
    assert "_reload" in ast.unparse(retry)
    app_tree = ast.parse((REPO / "app.py").read_text(encoding="utf-8"))
    run_placements = [n for n in ast.walk(app_tree) if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "run_placement"]
    assert len(run_placements) == 1 and ast.unparse(run_placements[0]).endswith("job.token, sanitize_error)")
    # the failure path in app.py only reports: it never schedules a load or a place
    report = next(f for f in ast.walk(app_tree) if isinstance(f, ast.AsyncFunctionDef) and f.name == "_report_failure")
    text = ast.unparse(report)
    assert "Job(" not in text and "run_load" not in text and "run_placement" not in text and "_start_slate_job" not in text


def test_a_before_click_failure_triggers_nothing_by_itself(rig):
    rig.bm.place_mode = "error"
    rig.to_summary("500")
    token = rig.press_final().token
    rig.place(token)
    snapshot = (list(rig.bm.calls), rig.session)
    for _ in range(3):
        rig.tick(600)                       # time passing triggers nothing
        slate_screen(rig.session, False, rig.now)
        failure_screen(rig.session, now=rig.now)
    assert (list(rig.bm.calls), rig.session) == snapshot and rig.session.state is State.FAILED
    assert rig.manager.act(OWNER, rig.sid, rig.session.state_version, Action.FINAL, "0" * 12).status is Status.STALE
    assert rig.bm.clicks == 0 and len(rig.places()) == 1


def test_a_new_trigger_starts_without_asking_after_a_failure_but_not_while_checking(rig):
    rig.fail_at_place(BookmakerError("The stake box was not found on the betslip."))
    assert rig.manager.request_start(OWNER, rig.evals).status is Status.APPLIED       # FAILED is over: nothing is open
    old_button = encode(rig.sid, 5, Action.RETRY)
    assert handle_callback(old_button, OWNER, rig.manager, dry_run=False).job is None  # the old slate's button is stale


# ================================================================ the whole thing through the Telegram handlers

@pytest.fixture
def app_chain(live_chain, monkeypatch):
    """The real handler chain with a live-mode fake bookmaker, the store attached, and a clock that moves on."""
    chain = live_chain
    ticks = [NOW]

    def clock():
        ticks[0] = ticks[0] + timedelta(seconds=1)
        return ticks[0]

    manager = SessionManager(clock=clock)
    attach_store(manager, chain.store)
    live = FakeBookmaker(chain.fake.listing, dry_run=False)
    monkeypatch.setattr(bot, "SESSIONS", manager)
    monkeypatch.setattr(bot, "BOOKMAKER", live)
    monkeypatch.setattr(bot, "DRY_RUN", False)
    chain.live = live
    return chain


def labels(chain):
    return [text for text, _ in chain.buttons(-1)]


def to_summary(chain, stake="500"):
    to_stake(chain)
    chain.text(OWNER_ID, stake)
    assert "Final summary" in chain.sent[-1][0]


def test_app_try_again_after_a_failed_placement_reloads_and_asks_for_the_new_confirmation(app_chain):
    chain = app_chain
    chain.live.place_mode = "error"
    to_summary(chain, "500")
    chain.press(OWNER_ID, find(chain.buttons(-1), "Confirm and place bet"))
    text = chain.sent[-1][0]
    assert text.startswith("No bet was placed.") and "The stake box was not found" in text
    assert labels(chain) == [TRY_AGAIN_LABEL] and bot.SESSIONS.active.state is State.FAILED
    places = [c for c in chain.live.calls if c[0] == "place"]
    assert len(places) == 1
    chain.live.place_mode = "ok"
    chain.press(OWNER_ID, find(chain.buttons(-1), TRY_AGAIN_LABEL))
    texts = [t for t, _ in chain.sent]
    assert any(t.startswith("Loaded on SportyBet") for t in texts[-3:])           # the odds are shown again
    summary = chain.sent[-1][0]
    assert "Final summary" in summary and "Stake: NGN 500.00" in summary and "Retry 1 of 3" in summary
    assert bot.SESSIONS.active.state is State.AWAITING_FINAL_CONFIRM and chain.live.clicks == 0
    assert len([c for c in chain.live.calls if c[0] == "place"]) == 1             # nothing placed by the reload
    chain.press(OWNER_ID, find(chain.buttons(-1), "Confirm and place bet"))
    assert "Bet placed" in chain.sent[-1][0] and chain.live.clicks == 1 and bot.SESSIONS.active.state is State.DONE


def test_app_unknown_result_check_then_warning_then_second_attempt(app_chain):
    chain = app_chain
    chain.live.place_mode = "unknown"
    to_summary(chain, "500")
    chain.press(OWNER_ID, find(chain.buttons(-1), "Confirm and place bet"))
    assert chain.sent[-1][0].startswith("A bet MAY exist.") and labels(chain) == [CHECK_LABEL]
    chain.press(OWNER_ID, find(chain.buttons(-1), CHECK_LABEL))
    assert "NOT found" in chain.sent[-1][0] and labels(chain) == [CHECK_LABEL, RETRY_CHECKED_LABEL]
    chain.press(OWNER_ID, find(chain.buttons(-1), RETRY_CHECKED_LABEL))
    assert "SECOND bet" in chain.events[-1][0] and [t for t, _ in chain.buttons(-1)] == [YES_AGAIN_LABEL, "Cancel"]
    chain.live.place_mode = "ok"
    chain.press(OWNER_ID, find(chain.buttons(-1), YES_AGAIN_LABEL))
    summary = chain.sent[-1][0]
    assert "Final summary" in summary and "You chose to place again" in summary and "Stake: NGN 500.00" in summary
    assert chain.live.clicks == 1                                                   # still only the first (unknown) click
    chain.press(OWNER_ID, find(chain.buttons(-1), "Confirm and place bet"))
    assert "Bet placed" in chain.sent[-1][0] and chain.live.clicks == 2
    assert {r["state"] for r in chain.store.list_placements()} == {"PLACEMENT_UNKNOWN", "PLACED"}


def test_app_check_history_finds_the_bet(app_chain):
    chain = app_chain
    chain.live.place_mode = "unknown"
    to_summary(chain, "500")
    chain.press(OWNER_ID, find(chain.buttons(-1), "Confirm and place bet"))
    chain.live.history = (history_bet("B5150"),)
    chain.press(OWNER_ID, find(chain.buttons(-1), CHECK_LABEL))
    assert "EXISTS" in chain.sent[-1][0] and "B5150" in chain.sent[-1][0]
    assert bot.SESSIONS.active.state is State.DONE and len(chain.store.list_placed_legs()) == 2
    assert chain.live.clicks == 1
