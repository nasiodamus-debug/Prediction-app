"""AD-26: a pick that cannot be added to the betslip is SKIPPED, the others still load, and the OWNER decides (Go ahead with
the loaded / Retry the failed / Cancel). Nothing is dropped silently and nothing is retried or placed by itself.

Everything runs on fakes: no browser, no network, no login, nothing is ever placed or booked."""
import asyncio
import json
from dataclasses import replace
from decimal import Decimal

import pytest

import app as bot
from fakes_bookmaker import NOW, FakeBookmaker, make_fixture, pick_evaluation, totals_offer, two_pick_setup
from predictions.adapters.sportybet.betslip import SportyBetBookmaker
from predictions.core.betting import BetslipLine, LoadAttempt, leg_from_evaluation, total_odds
from predictions.core.ports import BookmakerError, PickNotAdded
from predictions.core.types import Owner
from predictions.flow.loading import KIND_FAILED, KIND_LOADED, KIND_PARTIAL, run_load
from predictions.flow.recovery import attach_store, recover_on_startup
from predictions.flow.session import MAX_RETRIES, Action, SessionManager, State, Status
from predictions.flow.snapshot import from_json, session_to_data, to_json
from predictions.flow.store import EvaluationStore
from predictions.telegram.slate import (CALLBACK_LIMIT, decode, encode, handle_callback, handle_stake_text, odds_screen,
                                        slate_screen, summary_screen)
from test_epic5_app import live_chain, to_stake  # noqa: F401  (the fixture is used by name)
from test_epic5_recovery import Rig
from test_final_list import find
from test_retry import RetryRig
from test_sportybet_betslip import FakeFixtures, FakeSite, catalog_for, make_site, add_event

OWNER = Owner(111222333)
GO, RETRY, CANCEL = "Go ahead with the 4 loaded", "Retry the failed picks", "Cancel"
REASON = ("A selection did not appear on the betslip after it was clicked (Team A +1.5 in Handicap; the option's state: "
          "unknown)")


def run(coro):
    return asyncio.run(coro)


GAMES = [("Alpha", "Beta", "2.5", "1.85", 14), ("Gamma", "Delta", "1.5", "1.30", 15), ("Epsilon", "Zeta", "2.5", "1.90", 16),
         ("Eta", "Theta", "1.5", "1.40", 17), ("Iota", "Kappa", "2.5", "1.95", 18)]


def five():
    evals = [pick_evaluation(h, a, f"Over/Under {line}", f"Over {line}", odds, hour=hour) for h, a, line, odds, hour in GAMES]
    listing = [make_fixture(h, a, [totals_offer(line, odds, "2.00")], hour) for h, a, line, odds, hour in GAMES]
    return evals, listing


def key_of(evaluation):
    return leg_from_evaluation(evaluation).key


class PartialRig(RetryRig):
    """Five picks on the fake bookmaker; the 4th (Eta v Theta) cannot be added unless a test says otherwise."""

    def __init__(self, tmp_path, book=False, fail=(3,), **kw):
        evals, listing = five()
        super().__init__(tmp_path, evals=evals, listing=listing, **kw)
        if book:
            self.manager = SessionManager(clock=lambda: self.now, book_mode=True)
            attach_store(self.manager, self.store)
            self.manager.start(self.evals, self.sid, "matchday")
        for index in fail:
            self.bm.pick_failures[key_of(self.evals[index])] = REASON

    def load_calls(self):
        return [c for c in self.bm.calls if c[0] == "load"]

    def to_partial(self):
        self.confirm()
        out = self.load()
        assert out.kind == KIND_PARTIAL and self.session.state is State.LOAD_PARTIAL, out
        return out

    def go_ahead(self):
        reply = self.press(GO)
        assert reply.job is not None and reply.job.kind == "go"
        return reply

    def after_go(self):
        """What app.py does after a Go ahead job: the odds, then the summary (book mode) or the stake question."""
        self.manager.mark_odds_shown(self.sid, self.session.state_version)
        if self.manager.book_mode:
            assert self.manager.apply_booking_summary(self.sid, self.session.state_version).status is Status.APPLIED


@pytest.fixture
def rig(tmp_path):
    r = PartialRig(tmp_path, dry_run=False)
    yield r
    r.close()


@pytest.fixture
def book(tmp_path):
    r = PartialRig(tmp_path, book=True, dry_run=False)
    yield r
    r.close()


# ================================================================ the partial screen
def test_one_pick_failing_mid_list_still_loads_the_rest_and_shows_the_partial_screen(rig):
    rig.to_partial()
    s = rig.session
    assert [c[1] for c in rig.load_calls()] == [tuple(key_of(e) for e in rig.evals)]       # all five were tried, once
    assert len(s.loaded.legs) == 4 and len(s.failed_picks) == 1 and len(s.slip) == 5        # nothing dropped yet
    assert rig.bm.clicks == 0 and not rig.places()
    text = rig.screen().text
    assert text.startswith("Loaded 4 of 5 picks on SportyBet. 1 could not be added. No bet has been placed.")
    for game in ("Alpha v Beta", "Gamma v Delta", "Epsilon v Zeta", "Iota v Kappa"):
        assert game in text
    assert "@ 1.85" in text and "@ 1.30" in text and "@ 1.90" in text and "@ 1.95" in text          # the real odds
    assert "Eta vs Theta" in text and "Why: A selection did not appear on the betslip" in text      # the failed one + reason
    expected_total = total_odds([Decimal("1.85"), Decimal("1.30"), Decimal("1.90"), Decimal("1.95")])
    assert f"Total odds of the 4 loaded picks only: {expected_total:.2f}" in text
    assert rig.labels() == [GO, RETRY, CANCEL]
    assert "failed twice" not in text
    for row in rig.screen().buttons:
        for _, data in row:
            assert len(data.encode()) <= CALLBACK_LIMIT and decode(data) is not None


def test_the_locked_or_suspended_hint_reaches_the_screen(rig):
    rig.bm.pick_failures[key_of(rig.evals[3])] = REASON + "; it looks locked or suspended on SportyBet right now)"
    rig.to_partial()
    assert "locked or suspended" in rig.screen().text


def test_nothing_loaded_is_the_existing_failure_screen_with_try_again(rig):
    for e in rig.evals:
        rig.bm.pick_failures[key_of(e)] = REASON
    rig.confirm()
    assert rig.load().kind == KIND_FAILED
    s = rig.session
    assert s.state is State.FAILED and s.retryable
    text = rig.screen().text
    assert text.startswith("No bet was placed.") and "none of your picks could be added" in text
    assert rig.labels() == ["Try again"]


def test_a_failure_that_is_not_about_one_pick_still_fails_the_whole_load(rig):
    rig.bm.load_error = BookmakerError("The betslip could not be emptied.")
    rig.confirm()
    assert rig.load().kind == KIND_FAILED and rig.session.state is State.FAILED
    assert "could not be emptied" in rig.session.message


def test_a_betslip_that_holds_something_else_fails_closed(rig):
    original = rig.bm.load_legs

    async def wrong(legs):
        got = await original(legs)
        return LoadAttempt(replace(got.loaded, legs=got.loaded.legs[:2], total_odds=got.loaded.legs[0].odds), got.failed)

    rig.bm.load_legs = wrong
    rig.confirm()
    assert rig.load().kind == KIND_FAILED and rig.session.state is State.FAILED
    assert "not exactly the games" in rig.session.message


def test_a_failed_leg_that_was_never_asked_for_fails_closed(rig):
    original = rig.bm.load_legs
    stranger = leg_from_evaluation(pick_evaluation("Nobody", "Home", hour=19)).key

    async def odd(legs):
        got = await original(legs)
        return LoadAttempt(got.loaded, (replace(got.failed[0], leg_key=stranger),))

    rig.bm.load_legs = odd
    rig.confirm()
    assert rig.load().kind == KIND_FAILED


def test_a_price_change_stays_a_price_note_not_a_pick_failure(rig):
    rig.bm.odds_override[key_of(rig.evals[0])] = Decimal("1.70")
    rig.to_partial()
    text = rig.screen().text
    assert "evaluation used 1.85, SportyBet now shows 1.70" in text and len(rig.session.failed_picks) == 1


# ================================================================ Go ahead
def test_go_ahead_reduces_the_kept_list_and_asks_for_the_stake_in_place_mode(rig):
    rig.to_partial()
    reply = rig.go_ahead()
    s = rig.session
    assert s.state is State.AWAITING_STAKE and len(s.slip) == 4 and len(s.loaded.legs) == 4
    assert [p.evaluation.selection for p in s.slip.picks].count("Over 1.5") == 1     # Gamma v Delta stays, Eta v Theta is gone
    assert all(p.match_id.home != "Eta" for p in s.slip.picks)
    assert len(s.dropped) == 1 and "Eta vs Theta" in s.dropped[0].label and "did not appear" in s.dropped[0].reason
    assert not s.failed_picks and "Left out (1)" in reply.screen.text and "Eta vs Theta" in reply.screen.text
    assert rig.bm.clicks == 0 and not rig.places()
    rig.after_go()
    assert rig.session.odds_shown and rig.session.stake is None                       # the stake is asked for, as today
    odds = odds_screen(rig.session).text
    assert "Loaded on SportyBet: 4 game(s)" in odds and "Left out (1)" in odds and "Eta vs Theta" in odds


def test_the_token_and_summary_bind_only_the_loaded_legs_and_the_record_names_the_dropped_pick(rig):
    rig.to_partial()
    rig.go_ahead()
    rig.after_go()
    reply = handle_stake_text("500", OWNER, rig.manager, False)
    assert reply is not None and "Left out (1)" in reply.screen.text and "Eta vs Theta" in reply.screen.text
    new_total = rig.session.loaded.total_odds
    assert f"Total odds: {new_total:.2f}" in reply.screen.text and rig.session.summary.legs == rig.session.loaded.legs
    assert len(rig.session.summary.legs) == 4
    token = rig.press_final().token
    dropped_key = key_of(rig.evals[3])
    assert len(token.legs) == 4 and dropped_key not in {l.leg_key for l in token.legs} and token.total_odds == new_total
    assert {l.leg_key for l in token.legs} == {l.leg_key for l in rig.bm.loaded.legs}      # the browser slip, exactly
    out = rig.place(token)
    assert out.kind in ("dry_run", "placed", "unknown", "failed")
    (row,) = rig.store.list_placements()
    assert "went ahead without" in row["detail"] and "Eta vs Theta" in row["detail"]
    assert "Eta" not in row["legs"]


def test_the_registry_check_applies_to_the_loaded_legs_only(rig):
    """A leg already placed that is NOT on the bet (it was left out) must not block it; one that IS on the bet still does."""
    rig.to_partial()
    rig.go_ahead()
    rig.after_go()
    handle_stake_text("500", OWNER, rig.manager, False)
    s = rig.session
    asked = []
    rig.manager.attach(lambda _s: None, lambda identities: asked.append(list(identities)) or [])
    rig.press_final()
    assert len(asked[0]) == 4 and all("Eta" not in str(i) for i in asked[0])
    assert s.summary is not None


def test_book_mode_go_ahead_shows_the_booking_summary_with_the_new_total_and_the_left_out_pick(book):
    book.to_partial()
    reply = book.go_ahead()
    assert reply.job.kind == "go"
    book.after_go()
    s = book.session
    assert s.state is State.AWAITING_FINAL_CONFIRM and s.summary.booking and len(s.summary.legs) == 4
    text = summary_screen(s, dry_run=False).text
    assert "Left out (1)" in text and "Eta vs Theta" in text and f"Total odds: {s.loaded.total_odds:.2f}" in text
    assert "Stake" not in text and [b[0] for r in slate_screen(s, False).buttons for b in r] == ["Get booking code"]
    token = book.press_final().token
    assert token.booking and len(token.legs) == 4 and token.total_odds == s.loaded.total_odds


def test_go_ahead_taps_are_idempotent_and_old_buttons_are_stale(rig):
    rig.to_partial()
    data = rig.data_of(GO)
    retry_data = rig.data_of(RETRY)
    first = handle_callback(data, OWNER, rig.manager, dry_run=False)
    assert first.job is not None
    again = handle_callback(data, OWNER, rig.manager, dry_run=False)
    assert again.job is None and again.answer == "Already done." and len(rig.session.slip) == 4
    stale = handle_callback(retry_data, OWNER, rig.manager, dry_run=False)               # another button of the old screen
    assert stale.job is None and stale.answer.startswith("This list changed") and rig.session.state is State.AWAITING_STAKE
    assert len(rig.session.dropped) == 1


def test_go_ahead_fails_closed_when_the_loaded_legs_do_not_match_the_kept_list(rig):
    rig.to_partial()
    s = rig.session
    data = rig.data_of(GO)
    odd = replace(s.loaded, legs=s.loaded.legs[:3])
    rig.manager._active = replace(s, loaded=odd)
    reply = handle_callback(data, OWNER, rig.manager, dry_run=False)
    assert reply.job is None and "no longer match" in reply.answer and rig.session.state is State.LOAD_PARTIAL
    assert not rig.session.dropped


# ================================================================ Retry
def test_retry_reloads_every_kept_pick_counts_and_returns_to_the_same_screen_saying_failed_twice(rig):
    rig.to_partial()
    reply = rig.press(RETRY)
    assert reply.job is not None and reply.job.kind == "load"
    s = rig.session
    assert s.state is State.LOADING_SLIP and s.retries_used == 1 and len(s.slip) == 5
    assert "ALL 5 picks are added again" in reply.screen.text
    assert rig.load().kind == KIND_PARTIAL                                                # fails again
    assert [c[1] for c in rig.load_calls()] == [tuple(key_of(e) for e in rig.evals)] * 2  # all five, both times
    text = rig.screen().text
    assert "This pick has failed twice now." in text
    assert "(2 of 3 retries left.)" in text and rig.session.retries_used == 1
    assert rig.session.failed_picks[0].times == 2


def test_retry_cap_leaves_only_go_ahead_and_cancel(rig):
    rig.to_partial()
    for used in range(1, MAX_RETRIES + 1):
        assert rig.press(RETRY).job is not None and rig.session.retries_used == used
        assert rig.load().kind == KIND_PARTIAL
    text = rig.screen().text
    assert rig.labels() == [GO, CANCEL] and f"All {MAX_RETRIES} retries for this slate are used" in text
    assert "failed 4 times" in text
    forced = handle_callback(encode(rig.sid, rig.session.state_version, Action.RETRY_PARTIAL), OWNER, rig.manager, dry_run=False)
    assert forced.job is None and forced.alert and "retries for this slate are used" in forced.answer
    assert rig.session.state is State.LOAD_PARTIAL and rig.session.retries_used == MAX_RETRIES
    assert rig.go_ahead().job.kind == "go"                                                 # the owner can still go ahead


def test_retry_then_success_is_the_normal_flow_with_all_picks(rig):
    rig.to_partial()
    rig.press(RETRY)
    rig.bm.pick_failures.clear()
    assert rig.load().kind == KIND_LOADED
    s = rig.session
    assert s.state is State.AWAITING_STAKE and len(s.slip) == 5 and len(s.loaded.legs) == 5 and not s.dropped
    assert not s.failed_picks and s.retries_used == 1
    assert "Left out" not in odds_screen(s).text


def test_retry_with_nothing_loading_is_the_failure_screen_and_try_again_still_counts(rig):
    rig.to_partial()
    rig.press(RETRY)
    for e in rig.evals:
        rig.bm.pick_failures[key_of(e)] = REASON
    assert rig.load().kind == KIND_FAILED
    assert rig.session.state is State.FAILED and rig.session.retries_used == 1 and rig.session.retryable
    assert rig.press("Try again").job is not None and rig.session.retries_used == 2        # the same counter


def test_retry_that_finds_a_missing_option_goes_to_the_availability_screen_not_a_silent_drop(rig):
    rig.to_partial()
    rig.press(RETRY)
    rig.bm.listing = rig.bm.listing[:4]                                                    # the 5th game is gone now
    out = rig.load()
    assert out.kind == "issues" and rig.session.state is State.RESOLVING_AVAILABILITY and len(rig.session.slip) == 5


# ================================================================ Cancel
def test_cancel_ends_the_slate_and_nothing_is_placed(rig):
    rig.to_partial()
    data = rig.data_of(CANCEL)
    reply = handle_callback(data, OWNER, rig.manager, dry_run=False)
    assert reply.job is None and "Nothing was placed" in reply.screen.text
    assert rig.session.state is State.CANCELLED and not rig.places()
    assert handle_callback(data, OWNER, rig.manager, dry_run=False).answer == "Already done."
    assert handle_callback(encode(rig.sid, 2, Action.GO_AHEAD), OWNER, rig.manager).job is None
    assert not rig.manager.is_open(rig.session)


def test_the_partial_buttons_do_nothing_in_any_other_state(rig):
    rig.confirm()
    for action in (Action.GO_AHEAD, Action.RETRY_PARTIAL, Action.CANCEL):
        result = rig.manager.act(OWNER, rig.sid, rig.session.state_version, action)
        assert result.status is Status.STALE and rig.session.state is State.LOADING_SLIP


# ================================================================ snapshot and recovery
def test_the_snapshot_round_trips_the_partial_fields_and_old_snapshots_still_load(rig):
    rig.to_partial()
    s = rig.session
    assert from_json(to_json(s)) == s
    data = session_to_data(s)
    data.pop("failed_picks"), data.pop("dropped")
    old = from_json(json.dumps(data))
    assert old.failed_picks == () and old.dropped == () and old.state is State.LOAD_PARTIAL
    rig.go_ahead()
    assert from_json(to_json(rig.session)) == rig.session and len(rig.session.dropped) == 1


def test_a_restart_in_load_partial_reloads_everything_with_a_fresh_start(rig):
    rig.to_partial()
    rig.press(RETRY)
    rig.load()
    assert rig.session.state is State.LOAD_PARTIAL and rig.session.retries_used == 1
    reports = rig.restart()
    (report,) = [r for r in reports if r.counts]
    assert report.reload and rig.session.state is State.LOADING_SLIP
    s = rig.session
    assert s.loaded is None and not s.failed_picks and not s.token_hash and len(s.slip) == 5 and s.retries_used == 1
    rig.bm.pick_failures.clear()
    assert rig.load().kind == KIND_LOADED and len(rig.session.slip) == 5


def test_a_restart_after_go_ahead_keeps_the_left_out_record(rig):
    rig.to_partial()
    rig.go_ahead()
    rig.after_go()
    rig.restart()
    s = rig.session
    assert s.state is State.LOADING_SLIP and len(s.slip) == 4 and len(s.dropped) == 1
    rig.bm.pick_failures.clear()
    assert rig.load().kind == KIND_LOADED and len(rig.session.loaded.legs) == 4


# ================================================================ other origins
@pytest.mark.parametrize("origin", ["matchday", "gameday", "america", "review:AB12CD"])
def test_every_kind_of_slate_gets_the_same_partial_flow(tmp_path, origin):
    r = PartialRig(tmp_path, dry_run=False)
    try:
        r.manager.start(r.evals, r.sid, origin)
        r.to_partial()
        assert r.session.origin == origin and r.labels() == [GO, RETRY, CANCEL]
        r.go_ahead()
        assert r.session.state is State.AWAITING_STAKE and len(r.session.slip) == 4
    finally:
        r.close()


def test_a_full_load_is_unchanged(tmp_path):
    r = PartialRig(tmp_path, fail=(), dry_run=False)
    try:
        r.confirm()
        assert r.load().kind == KIND_LOADED and r.session.state is State.AWAITING_STAKE
        assert len(r.session.loaded.legs) == 5 and not r.session.dropped and not r.session.failed_picks
    finally:
        r.close()


def test_a_bookmaker_without_load_legs_still_works(tmp_path):
    class Old:
        def __init__(self, inner):
            self.inner = inner
            self.dry_run = True

        async def check_availability(self, legs):
            return await self.inner.check_availability(legs)

        async def load_slip(self, legs):
            return await self.inner.load_slip(legs)

    r = Rig(tmp_path)
    try:
        r.bm = Old(r.bm)
        r.confirm()
        assert r.load().kind == KIND_LOADED
    finally:
        r.close()


# ================================================================ the app: screens and the Go ahead job
def test_the_app_shows_the_partial_screen_then_go_ahead_shows_the_odds_and_asks_for_the_stake(live_chain):
    chain = live_chain
    second = leg_from_evaluation(chain.evals[1]).key
    first = leg_from_evaluation(chain.evals[0]).key
    chain.fake.pick_failures[second if second != first else first] = REASON
    to_stake(chain)
    text, kwargs = chain.sent[-1]
    assert text.startswith("Loaded 1 of 2 picks") and bot.SESSIONS.active.state is State.LOAD_PARTIAL
    labels = [b.text for row in kwargs["reply_markup"].inline_keyboard for b in row]
    assert labels == ["Go ahead with the 1 loaded", "Retry the failed picks", "Cancel"]
    go = find(chain.buttons(-1), "Go ahead with the 1 loaded")
    chain.press(111222333, go)
    sent = [t for t, _ in chain.sent]
    odds_at = next(i for i, t in enumerate(sent) if t.startswith("Loaded on SportyBet: 1 game(s)"))
    assert "Left out (1)" in sent[odds_at] and any("How much do you want to stake" in t for t in sent[odds_at:])
    s = bot.SESSIONS.active
    assert s.state is State.AWAITING_STAKE and s.odds_shown and len(s.slip) == 1
    chain.press(111222333, go)                                                              # a second tap does nothing
    assert [t for t, _ in chain.sent].count(sent[odds_at]) == 1
    assert chain.fake.clicks == 0 and not any(c[0] == "place" for c in chain.fake.calls)


def test_the_app_retry_button_loads_again_and_cancel_ends(live_chain):
    chain = live_chain
    key = leg_from_evaluation(chain.evals[0]).key
    chain.fake.pick_failures[key] = REASON
    to_stake(chain)
    chain.press(111222333, find(chain.buttons(-1), "Retry the failed picks"))
    assert [c[0] for c in chain.fake.calls].count("load") == 2
    assert bot.SESSIONS.active.state is State.LOAD_PARTIAL and bot.SESSIONS.active.retries_used == 1
    assert "failed twice" in chain.sent[-1][0]
    chain.press(111222333, find(chain.buttons(-1), "Cancel"))
    assert bot.SESSIONS.active.state is State.CANCELLED


# ================================================================ the adapter
class SkipRig:
    """A SportyBetBookmaker over a FakeSite whose add_outcome can refuse one game with PickNotAdded."""

    def __init__(self, tmp_path, refuse=("Gamma v Delta",), error=PickNotAdded("The selected option was not found on the "
                                                                                 "game's page.")):
        self.evals, self.listing = two_pick_setup()
        self.fixtures = FakeFixtures(self.listing)
        self.site = FakeSite(catalog_for(self.evals, self.listing))
        original = self.site.add_outcome
        names = {f.bookmaker_ref: f"{f.home} v {f.away}" for f in self.listing}

        async def add(event_ref, titles, outcome):
            if names[event_ref] in refuse:
                self.site.log.append(f"refused:{event_ref}")
                raise error
            await original(event_ref, titles, outcome)

        self.site.add_outcome = add

        async def factory():
            return self.site

        self.bm = SportyBetBookmaker(None, True, self.fixtures, factory, screenshot_dir=tmp_path / "s", clock=lambda: NOW)
        self.legs = [leg_from_evaluation(e) for e in self.evals]


def test_the_adapter_skips_a_pick_that_cannot_be_added_and_verifies_the_rest(tmp_path):
    rig = SkipRig(tmp_path)
    attempt = run(rig.bm.load_legs(rig.legs))
    assert [l.match_text for l in attempt.loaded.legs] == ["Alpha v Beta"] and attempt.loaded.total_odds == Decimal("1.85")
    assert [f.leg_key for f in attempt.failed] == [rig.legs[1].key] and "not found on the game's page" in attempt.failed[0].reason
    assert rig.site.log[:2] == ["login", "clear"] and rig.site.log[-1] == "read" and not rig.site.closed
    assert [r.match_text for r in rig.site.rows] == ["Alpha v Beta"]


def test_the_strict_load_still_fails_on_the_same_pick(tmp_path):
    rig = SkipRig(tmp_path)
    with pytest.raises(BookmakerError, match="not found on the game's page"):
        run(rig.bm.load_slip(rig.legs))
    assert rig.site.closed


def test_nothing_added_returns_no_loaded_slip_and_reads_nothing(tmp_path):
    rig = SkipRig(tmp_path, refuse=("Alpha v Beta", "Gamma v Delta"))
    attempt = run(rig.bm.load_legs(rig.legs))
    assert attempt.loaded is None and len(attempt.failed) == 2 and "read" not in rig.site.log


def test_a_driver_crash_is_not_a_pick_failure_and_fails_the_whole_load(tmp_path):
    rig = SkipRig(tmp_path, error=RuntimeError("boom"))
    with pytest.raises(BookmakerError, match="failed unexpectedly"):
        run(rig.bm.load_legs(rig.legs))
    assert rig.site.closed


def test_a_stray_selection_on_the_betslip_fails_closed_even_after_a_skipped_pick(tmp_path):
    rig = SkipRig(tmp_path)
    rig.site.extra_stray = True
    from predictions.core.ports import BetslipMismatch
    with pytest.raises(BetslipMismatch):
        run(rig.bm.load_legs(rig.legs))


def test_the_page_driver_raises_pick_not_added_for_the_three_pick_problems():
    site, dom = make_site()
    add_event(dom, "sr:match:1", "Alpha v Beta", "Over/Under", {"Over 2.5": "1.85"})
    with pytest.raises(PickNotAdded):
        run(site.add_outcome("sr:match:1", ["Over/Under"], "Over 9.5"))                    # option not found
    dom.growth_broken = True
    with pytest.raises(PickNotAdded, match="did not appear"):
        run(site.add_outcome("sr:match:1", ["Over/Under"], "Over 2.5"))                    # did not appear after the re-click

    class TimeoutError(Exception):                                                          # Playwright's class name
        pass

    async def slow(url, **kw):
        raise TimeoutError("Timeout 60000ms exceeded")

    site.page.goto = slow
    with pytest.raises(PickNotAdded, match="did not load in time"):
        run(site.add_outcome("sr:match:1", ["Over/Under"], "Over 2.5"))

    async def crashed(url, **kw):
        raise ConnectionError("Target closed")

    site.page.goto = crashed
    with pytest.raises(ConnectionError):                                                    # a crash is NOT a pick problem
        run(site.add_outcome("sr:match:1", ["Over/Under"], "Over 2.5"))
