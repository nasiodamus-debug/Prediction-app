"""AD-25: PLACE_MODE=book. The final confirmation only generates a booking code (Book Bet is the only thing pressed), the code and
the total odds are shown apart in ONE message, and the owner can say whether he placed it ('I placed it' / 'I did not place it',
Mybets). Everything runs on fakes: no browser, no network, no login, nothing is ever placed."""
import asyncio
import itertools
import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from telegram import User
from telegram.error import BadRequest
from telegram.ext import ApplicationBuilder, ExtBot

import app as bot
from fakes_bookmaker import NOW, FakeBookmaker, history_entry, pick_evaluation, two_pick_setup
from predictions.adapters.sportybet.betslip import SportyBetBookmaker, parse_place_mode
from predictions.core.betting import LoadedLeg, leg_from_evaluation, leg_identity
from predictions.core.placement import (BOOKING_STAKE, MINT_AUTHORITY, PlacementKind, PlacementToken, Summary, TokenNotMinted,
                                        booking_lines, mint_token, summary_digest)
from predictions.core.ports import BetslipMismatch, BookmakerError, OddsChanged
from predictions.core.types import Owner
from predictions.flow.booking import (ALREADY_TEXT, ASK_TEXT, NOT_PLACED_TEXT, RECORDED_TEXT, PendingStake, press, stake_reply)
from predictions.flow.evaluate import GateResult
from predictions.flow.loading import KIND_LOADED
from predictions.flow.placement import OUTCOME_BOOKED, OUTCOME_DRY_RUN, OUTCOME_FAILED, run_placement
from predictions.flow.results import calibration_lines, check_results
from predictions.flow.recovery import recover_on_startup
from predictions.flow.session import Action, SessionManager, State, Status
from predictions.flow.snapshot import from_json, to_json
from predictions.flow.store import (MIGRATIONS, P_BOOKED, P_FAILED, P_PLACED, P_PLACED_BY_OWNER, P_PLACING, EvaluationStore)
from predictions.telegram.booked import (INSTRUCTION, answer_buttons, booked_message, decode_booking, encode_booking,
                                         mybets_screens)
from predictions.telegram.slate import handle_callback, handle_stake_text, summary_screen
from test_epic5_flow import Env, SID, run
from test_final_list import Chain, find
from test_sportybet_betslip import FakeFixtures, FakeSite, catalog_for

OWNER = Owner(111222333)
OWNER_ID = 111222333
STRANGER = 999888777
EXPIRES = datetime(2026, 12, 1, tzinfo=timezone.utc)


# ================================================================ the setting
@pytest.mark.parametrize("raw, expected", [(None, "book"), ("", "book"), ("  ", "book"), ("book", "book"), ("BOOK", "book"),
                                           (" Book ", "book"), ("place", "place"), ("PLACE", "place"), (" Place", "place")])
def test_place_mode_parses(raw, expected):
    assert parse_place_mode(raw) == expected


@pytest.mark.parametrize("raw", ["yes", "true", "booking", "placed", "0", "book place", "pl ace"])
def test_a_bad_place_mode_refuses_to_start_and_never_echoes_the_value(raw):
    with pytest.raises(ValueError) as caught:
        parse_place_mode(raw)
    assert raw not in str(caught.value) and "book or place" in str(caught.value)


def test_the_bot_module_reads_the_setting_once_and_builds_the_manager_from_it():
    src = open(bot.__file__, encoding="utf-8").read()
    assert 'parse_place_mode(os.getenv("PLACE_MODE"))' in src and 'book_mode=PLACE_MODE == "book"' in src
    assert bot.PLACE_MODE == "place"   # tests/conftest.py pins the older tests to place mode


# ================================================================ the summary and the token
def two_legs():
    evals, listing = two_pick_setup()
    from predictions.core.betting import LoadedLeg
    return tuple(LoadedLeg(leg_from_evaluation(e).key, f"{e.match_id.home} v {e.match_id.away}", "Over/Under", e.selection,
                           e.odds, e.selection, e.odds) for e in evals)


def test_a_booking_summary_has_no_stake_and_a_placing_one_must_have_one():
    legs = two_legs()
    booking = Summary("abcd1234", legs, Decimal("2.41"), BOOKING_STAKE, EXPIRES, booking=True)
    assert booking.booking and booking.stake == 0
    with pytest.raises(ValueError):
        Summary("abcd1234", legs, Decimal("2.41"), Decimal("500"), EXPIRES, booking=True)   # a booking summary has no stake
    for bad in (Decimal(0), Decimal("-5")):
        with pytest.raises(ValueError):
            Summary("abcd1234", legs, Decimal("2.41"), bad, EXPIRES)                         # place mode still needs stake > 0


def test_a_booking_token_is_bound_to_its_mode_and_cannot_be_forged_into_a_placing_one():
    legs = two_legs()
    booking = mint_token(MINT_AUTHORITY, Summary("abcd1234", legs, Decimal("2.41"), BOOKING_STAKE, EXPIRES, booking=True))
    placing = mint_token(MINT_AUTHORITY, Summary("abcd1234", legs, Decimal("2.41"), Decimal("500"), EXPIRES))
    assert booking.booking and not placing.booking and booking.token_hash != placing.token_hash
    assert booking.matches_contents() and placing.matches_contents()
    with pytest.raises(TokenNotMinted):   # a zero-stake token that claims to be a placing one
        PlacementToken("abcd1234", legs, Decimal("2.41"), BOOKING_STAKE, EXPIRES,
                       summary_digest("abcd1234", legs, Decimal("2.41"), BOOKING_STAKE, EXPIRES), MINT_AUTHORITY)
    with pytest.raises(TokenNotMinted):   # a staked token that claims to be a booking one
        PlacementToken("abcd1234", legs, Decimal("2.41"), Decimal("500"), EXPIRES,
                       summary_digest("abcd1234", legs, Decimal("2.41"), Decimal("500"), EXPIRES, True), MINT_AUTHORITY, True)
    with pytest.raises(TokenNotMinted):   # and nobody without the mint authority can build one
        PlacementToken("abcd1234", legs, Decimal("2.41"), BOOKING_STAKE, EXPIRES, booking.token_hash, None, True)


# ================================================================ the session: no stake step in book mode
class BookEnv(Env):
    def __init__(self, tmp_path, dry_run=False):
        super().__init__(tmp_path, dry_run)
        self.manager = SessionManager(clock=lambda: self.now, book_mode=True)
        from predictions.flow.recovery import attach_store
        attach_store(self.manager, self.store)      # saved after every step, like the real bot
        self.manager.start(self.evals, SID)

    def to_summary(self, stake=None):
        self.confirm()
        assert self.load().kind == KIND_LOADED
        self.manager.mark_odds_shown(SID, self.session.state_version)
        result = self.manager.apply_booking_summary(SID, self.session.state_version)
        assert result.status is Status.APPLIED
        return self.session


@pytest.fixture
def benv(tmp_path):
    e = BookEnv(tmp_path)
    yield e
    e.close()


@pytest.fixture
def bdry(tmp_path):
    e = BookEnv(tmp_path, dry_run=True)
    yield e
    e.close()


def test_book_mode_goes_from_the_odds_straight_to_the_final_screen_with_no_stake(benv):
    session = benv.to_summary()
    assert session.state is State.AWAITING_FINAL_CONFIRM and session.summary.booking and session.summary.stake == 0
    assert session.stake is None and session.summary.total_odds == Decimal("2.41")
    # typed text is never a stake, Change amount does not exist, MAX_STAKE is never consulted
    assert benv.manager.submit_stake(OWNER, "500").status is Status.STALE
    assert benv.manager.act(OWNER, SID, session.state_version, Action.CHANGE).status is Status.STALE
    assert handle_stake_text("500", OWNER, benv.manager, False) is None
    assert benv.session.state is State.AWAITING_FINAL_CONFIRM


def test_the_book_summary_screen_says_confirm_to_get_your_booking_code(benv):
    screen = summary_screen(benv.to_summary(), dry_run=False)
    assert "Confirm to get your booking code" in screen.text and "Total odds: 2.41" in screen.text
    assert [label for row in screen.buttons for label, _ in row] == ["Get booking code"]      # no Change amount
    for word in ("Stake:", "Potential winnings", "Profit", "Confirm and place bet", "REAL bet"):
        assert word not in screen.text
    callback = screen.buttons[0][0][1]
    assert callback.startswith("s:") and len(callback.encode()) <= 64 and callback.split(":")[3] == Action.FINAL.value


def test_the_book_summary_in_dry_run_says_nothing_is_made(bdry):
    screen = summary_screen(bdry.to_summary(), dry_run=True)
    assert "DRY RUN" in screen.text and "No booking code is made" in screen.text
    assert "Get booking code" not in [label for row in screen.buttons for label, _ in row]


def test_place_mode_is_unchanged_it_still_asks_for_a_stake_and_enforces_max_stake_and_positive(tmp_path):
    env = Env(tmp_path, dry_run=False)
    try:
        env.to_stake()
        assert env.manager.apply_booking_summary(SID, env.session.state_version).status is Status.STALE
        for bad in ("0", "-5", "abc", "99999999999"):
            assert env.manager.submit_stake(OWNER, bad).status is Status.BLOCKED
        env.manager.submit_stake(OWNER, "500")
        screen = summary_screen(env.session, dry_run=False)
        labels = [label for row in screen.buttons for label, _ in row]
        assert labels == ["Confirm and place bet", "Change amount"] and "Stake: NGN 500.00" in screen.text
        assert "Potential winnings" in screen.text and "REAL bet" in screen.text
        assert not env.session.summary.booking and env.press_final().token.booking is False
    finally:
        env.close()


def test_the_final_confirmation_mints_one_booking_token_once(benv):
    session = benv.to_summary()
    first = benv.press_final(session)
    assert first.status is Status.APPLIED and first.token.booking and first.token.stake == 0
    assert benv.session.state is State.PLACING
    again = benv.manager.act(OWNER, SID, session.state_version, Action.FINAL, session.summary.summary_id)
    assert again.token is None and again.status in (Status.REPEAT, Status.STALE)


def test_a_summary_made_under_the_other_mode_never_mints_a_token(tmp_path):
    env = Env(tmp_path, dry_run=False)
    try:
        env.to_summary()
        placing_session = env.session
        env.manager.book_mode = True    # the same slate meets a book-mode manager
        result = env.manager.act(OWNER, SID, placing_session.state_version, Action.FINAL, placing_session.summary.summary_id)
        assert result.status is Status.STALE and result.token is None
    finally:
        env.close()


def test_odds_moving_in_book_mode_gives_a_new_booking_summary(benv):
    benv.bm.place_mode = "odds_changed"
    benv.to_summary()
    out = benv.place(benv.press_final().token)
    assert out.kind == "odds_changed" and benv.session.state is State.AWAITING_FINAL_CONFIRM
    assert benv.session.summary.booking and benv.session.summary.total_odds > Decimal("2.41") and benv.bm.book_clicks == 0


# ================================================================ the placement flow (fake bookmaker)
def test_a_booking_ends_done_with_a_booked_row_and_nothing_is_placed_or_blocked(benv):
    benv.to_summary()
    token = benv.press_final().token
    out = benv.place(token)
    assert out.kind == OUTCOME_BOOKED and out.result.kind is PlacementKind.BOOKED and out.result.booking_code == "HW63TV"
    assert benv.session.state is State.DONE and out.token is token
    assert benv.bm.book_clicks == 1 and benv.bm.clicks == 0           # Book Bet pressed once, Place never
    (row,) = benv.store.list_placements()
    assert row["state"] == P_BOOKED and row["booking_code"] == "HW63TV" and row["dry_run"] == 0 and row["bet_id"] == ""
    assert json.loads(row["legs"]) and benv.store.list_placed_legs() == [] and benv.store.list_booked() == [row]
    # BOOKED never blocks a later list, never counts for results, calibration or the open-bets query
    assert benv.store.find_blocking_legs([leg_identity(l.leg_key) for l in token.legs]) == []
    assert benv.store.open_placements_for_results() == [] and benv.store.graded_legs() == []


def test_a_booking_never_changes_what_placed_and_unknown_rows_block(benv):
    benv.to_summary()
    token = benv.press_final().token
    benv.place(token)
    identities = [leg_identity(l.leg_key) for l in token.legs]
    assert benv.store.find_blocking_legs(identities) == []
    for state in (P_PLACED, "PLACEMENT_UNKNOWN", P_PLACING):
        benv.store.update_placement(token.token_hash, state, bet_id="B1" if state == P_PLACED else "")
        if state == P_PLACED:
            benv.store.record_placed_legs(token.token_hash, token.legs)
        assert len(benv.store.find_blocking_legs(identities)) == 2, state
        benv.store._conn.execute("DELETE FROM placed_legs")
    benv.store.update_placement(token.token_hash, P_BOOKED, booking_code="HW63TV")
    assert benv.store.find_blocking_legs(identities) == []


def test_no_readable_code_is_a_retryable_failure_with_no_bet(benv):
    benv.bm.book_code = ""
    benv.to_summary()
    out = benv.place(benv.press_final().token)
    assert out.kind == OUTCOME_FAILED and benv.session.state is State.FAILED and benv.session.retryable
    assert "No bet was placed. The booking code could not be read." in out.message
    (row,) = benv.store.list_placements()
    assert row["state"] == P_FAILED and benv.store.list_placed_legs() == []
    # the owner-pressed Try again counts toward the cap and goes back to the same final screen
    again = benv.manager.act(OWNER, SID, benv.session.state_version, Action.RETRY)
    assert again.status is Status.APPLIED and again.session.retries_used == 1 and again.load_next
    benv.bm.book_code = "ZZ99AA"
    assert benv.load().kind == KIND_LOADED
    benv.manager.mark_odds_shown(SID, benv.session.state_version)
    assert benv.manager.apply_booking_summary(SID, benv.session.state_version).status is Status.APPLIED
    assert benv.session.summary.booking and benv.session.retries_used == 1


def test_the_retry_cap_applies_to_booking_failures(benv):
    benv.bm.book_code = ""
    benv.to_summary()
    for used in range(3):
        benv.place(benv.press_final().token)
        assert benv.manager.act(OWNER, SID, benv.session.state_version, Action.RETRY).status is Status.APPLIED
        benv.load()
        benv.manager.mark_odds_shown(SID, benv.session.state_version)
        benv.manager.apply_booking_summary(SID, benv.session.state_version)
    benv.place(benv.press_final().token)
    assert benv.manager.act(OWNER, SID, benv.session.state_version, Action.RETRY).status is Status.BLOCKED


@pytest.mark.parametrize("mode", ["error", "unexpected"])
def test_a_bookmaker_error_or_a_crash_while_booking_is_no_bet_not_unknown(tmp_path, mode):
    env = BookEnv(tmp_path)
    try:
        env.bm.place_mode = mode
        env.to_summary()
        out = env.place(env.press_final().token)
        assert out.kind == OUTCOME_FAILED and env.session.state is State.FAILED and env.session.retryable
        assert "No bet was placed." in out.message and env.store.list_placements()[0]["state"] == P_FAILED
    finally:
        env.close()


def test_dry_run_in_book_mode_books_nothing(bdry):
    bdry.to_summary()
    out = bdry.place(bdry.press_final().token)
    assert out.kind == OUTCOME_DRY_RUN and "no booking code was made" in out.message and bdry.bm.book_clicks == 0
    assert bdry.bm.clicks == 0 and not any(c[0] == "book" for c in bdry.bm.calls)
    (row,) = bdry.store.list_placements()
    assert row["state"] == "DRY_RUN" and row["booking_code"] == ""


def test_the_token_is_single_use_in_book_mode(benv):
    benv.to_summary()
    token = benv.press_final().token
    assert benv.place(token).kind == OUTCOME_BOOKED
    second = benv.place(token)
    assert second.kind == OUTCOME_FAILED and "no longer valid" in second.message
    assert benv.bm.book_clicks == 1 and [c[0] for c in benv.bm.calls].count("place") == 1


# ================================================================ the adapter: only Book Bet is ever pressed
class BookRig:
    """The REAL SportyBetBookmaker over a fake page driver that records every action (PLACE, confirm, stake, BOOK)."""

    def __init__(self, tmp_path, dry_run=False, place_mode="book", code="HW63TV"):
        self.evals, self.listing = two_pick_setup()
        self.site = FakeSite(catalog_for(self.evals, self.listing), allow_place=not dry_run and place_mode == "place")
        self.code, self.books = code, 0

        async def book_for_code():
            self.site.log.append("BOOK")
            self.books += 1
            return self.code

        self.site.book_for_code = book_for_code

        async def factory():
            return self.site

        self.bm = SportyBetBookmaker(None, dry_run, FakeFixtures(self.listing), factory, screenshot_dir=tmp_path / "shots",
                                     clock=lambda: NOW, place_mode=place_mode)
        self.legs = [leg_from_evaluation(e) for e in self.evals]

    def token(self, booking=True, odds=None):
        loaded = run(self.bm.load_slip(self.legs))
        legs = loaded.legs if odds is None else tuple(
            LoadedLeg(l.leg_key, l.match_text, l.market_name, l.outcome_name, o, l.selection, l.evaluation_odds)
            for l, o in zip(loaded.legs, odds))
        stake = BOOKING_STAKE if booking else Decimal("500")
        return mint_token(MINT_AUTHORITY, Summary("abcd1234", legs, loaded.total_odds, stake, NOW + timedelta(minutes=5),
                                                  booking=booking))


def test_book_mode_presses_only_book_bet_never_place_confirm_or_a_stake(tmp_path):
    rig = BookRig(tmp_path)
    result = run(rig.bm.place(rig.token()))
    assert result.kind is PlacementKind.BOOKED and result.booking_code == "HW63TV" and result.bet_id == ""
    log = rig.site.log
    assert log.count("BOOK") == 1 and rig.books == 1
    assert "PLACE" not in log and "confirm" not in log and rig.site.place_clicks == 0
    assert not any(entry.startswith("stake:") for entry in log) and rig.site.stake is None
    assert "history" not in log[log.index("read"):]                  # no bet-history snapshot or proof either
    assert log.index("BOOK") > max(i for i, e in enumerate(log) if e == "read")   # the betslip was verified first
    assert rig.site.closed


def test_book_mode_never_lets_the_driver_click_place_even_if_asked(tmp_path):
    rig = BookRig(tmp_path)
    assert rig.bm.place_mode == "book" and rig.site.allow_place is False
    with pytest.raises(BookmakerError):
        run(rig.site.click_place())


def test_book_mode_dry_run_never_presses_book_bet(tmp_path):
    rig = BookRig(tmp_path, dry_run=True)
    result = run(rig.bm.place(rig.token()))
    assert result.kind is PlacementKind.DRY_RUN and rig.books == 0 and "BOOK" not in rig.site.log and "PLACE" not in rig.site.log
    assert any("Book Bet would be pressed once" in line for line in result.report) and all("stake NGN" not in l for l in result.report)


def test_book_mode_refuses_a_changed_betslip_or_odds_without_pressing_anything(tmp_path):
    rig = BookRig(tmp_path)
    token = rig.token()
    rig.site.odds_shift = Decimal("0.10")
    with pytest.raises(OddsChanged):
        run(rig.bm.place(token))
    assert rig.books == 0 and "BOOK" not in rig.site.log
    rig2 = BookRig(tmp_path)
    token2 = rig2.token()
    rig2.site.rows.pop()
    with pytest.raises(BetslipMismatch):
        run(rig2.bm.place(token2))
    assert rig2.books == 0


@pytest.mark.parametrize("code", ["", "<script>x</script>", "ab"])
def test_book_mode_without_a_readable_code_is_a_plain_failure(tmp_path, code):
    rig = BookRig(tmp_path, code=code)
    with pytest.raises(BookmakerError) as caught:
        run(rig.bm.place(rig.token()))
    assert "booking code could not be read" in str(caught.value) and rig.books == 1 and "PLACE" not in rig.site.log


def test_the_adapter_token_is_single_use_in_book_mode(tmp_path):
    rig = BookRig(tmp_path)
    token = rig.token()
    run(rig.bm.place(token))
    with pytest.raises(BookmakerError):
        run(rig.bm.place(token))
    assert rig.books == 1


def test_a_booking_token_never_reaches_a_place_mode_bookmaker_and_vice_versa(tmp_path):
    placer = BookRig(tmp_path, place_mode="place")
    with pytest.raises(BookmakerError) as caught:
        run(placer.bm.place(placer.token(booking=True)))
    assert "PLACE_MODE" in str(caught.value) and placer.site.place_clicks == 0 and "BOOK" not in placer.site.log
    booker = BookRig(tmp_path, place_mode="book")
    with pytest.raises(BookmakerError) as caught:
        run(booker.bm.place(booker.token(booking=False)))
    assert "PLACE_MODE" in str(caught.value) and booker.books == 0 and "PLACE" not in booker.site.log


def test_place_mode_adapter_is_unchanged_it_still_types_the_stake_and_clicks_place(tmp_path):
    rig = BookRig(tmp_path, place_mode="place")
    token = rig.token(booking=False)
    rig.site.history_after_click = [history_entry(("Alpha v Beta", "Gamma v Delta"), "500", "Running", 0, NOW)]
    result = run(rig.bm.place(token))
    assert result.kind is PlacementKind.PLACED and "PLACE" in rig.site.log and "stake:500" in rig.site.log
    assert "BOOK" not in rig.site.log


def test_the_site_factory_is_built_without_the_right_to_place_in_book_mode(tmp_path):
    import inspect
    from predictions.adapters.sportybet import betslip
    source = inspect.getsource(betslip.SportyBetBookmaker.__init__)
    assert 'allow_place=not self._dry_run and place_mode == "place"' in source


# ================================================================ the messages
def test_booking_lines_put_the_code_and_the_odds_on_their_own_lines():
    assert booking_lines("HW63TV", Decimal("2.65")) == ["Booking code: HW63TV", "Total odds: 2.65"]
    assert booking_lines("HW63TV", Decimal("2.41"), Decimal("500")) == ["Booking code: HW63TV", "Total odds: 2.41", "Stake: NGN 500"]


def test_the_booked_message_layout(benv):
    benv.to_summary()
    token = benv.press_final().token
    plain, html_text = booked_message(token, "HW63TV")
    lines = plain.split("\n")
    assert lines[0] == "Booking code: HW63TV" and lines[1] == "Total odds: 2.41" and lines[2] == ""
    assert lines[3] == "1. Alpha v Beta: Over 2.5 (Over/Under) @ 1.85" and lines[4].startswith("2. Gamma v Delta: Over 1.5")
    assert lines[-1] == INSTRUCTION and INSTRUCTION == ("Open SportyBet, enter the code, choose your stake and press Place Bet "
                                                      "yourself. Nothing was placed by the bot.")
    assert html_text.split("\n")[0] == "Booking code: <code>HW63TV</code>" and "<code>" not in html_text.split("\n", 1)[1]
    for text in (plain, html_text):
        assert "Stake" not in text and "winnings" not in text.lower() and "bet placed" not in text.lower()
        assert "placed" not in text.replace("Nothing was placed by the bot", "").replace("Place Bet", "").lower()


def test_the_html_message_escapes_game_names(benv):
    benv.to_summary()
    token = benv.press_final().token
    legs = (replace(token.legs[0], match_text="A <b>&</b> v B"),) + token.legs[1:]
    evil = mint_token(MINT_AUTHORITY, Summary("abcd1234", legs, token.total_odds, BOOKING_STAKE, EXPIRES, booking=True))
    _, html_text = booked_message(evil, "HW63TV")
    assert "A &lt;b&gt;&amp;&lt;/b&gt; v B" in html_text and "<b>" not in html_text


def test_the_callback_data_is_versioned_short_and_decodes_only_valid_actions():
    data = encode_booking("a" * 64, "p")
    assert data == "b:" + "a" * 12 + ":p" and len(data.encode()) <= 64
    assert decode_booking(data) == ("a" * 12, "p")
    for bad in ("b:xyz:p", "b:" + "a" * 12 + ":z", "s:" + "a" * 12 + ":p", None, 5, "b:" + "a" * 12 + ":p:extra"):
        assert decode_booking(bad) is None


# ================================================================ the owner's word: I placed it / I did not
def booked_row(store, code="HW63TV", legs=None, state=P_BOOKED, dry_run=False, session="s1", stake="0"):
    evals = two_pick_setup()[0]
    loaded = tuple(LoadedLeg(leg_from_evaluation(e).key, f"{e.match_id.home} v {e.match_id.away}", "Over/Under", e.selection,
                             e.odds, e.selection, e.odds) for e in evals) if legs is None else legs
    total = Decimal("2.41")
    summary = Summary(session, loaded, total, BOOKING_STAKE, EXPIRES, booking=True)
    token = mint_token(MINT_AUTHORITY, summary)
    store.record_placing(token, dry_run, NOW)
    store.update_placement(token.token_hash, state, booking_code=code, detail="booking code only; no bet was placed", now=NOW)
    return token


@pytest.fixture
def store(tmp_path):
    s = EvaluationStore.open(tmp_path / "b.db", str)
    yield s
    s.close()


def test_i_placed_it_asks_for_the_stake_then_records_it_and_registers_the_legs(store):
    token = booked_row(store)
    desk, key = PendingStake(), token.token_hash[:12]
    ask = press(store, desk, key, "p")
    assert ask.status == "ask" and ask.text == ASK_TEXT == "How much did you stake? Reply with the amount, or press Skip."
    assert store.get_placement(token.token_hash)["state"] == P_BOOKED and store.list_placed_legs() == []   # nothing yet
    for bad in ("abc", "0", "-5", "1.234", ""):
        assert stake_reply(store, desk, bad).status == "invalid" and desk.key == key
    done = stake_reply(store, desk, "1,250.50")
    assert done.status == "recorded" and done.text.startswith(RECORDED_TEXT) and RECORDED_TEXT == "Recorded as placed (confirmed by you)."
    row = store.get_placement(token.token_hash)
    assert row["state"] == P_PLACED_BY_OWNER and row["source"] == "owner-confirmed" and Decimal(row["owner_stake"]) == Decimal("1250.50")
    assert row["booking_code"] == "HW63TV" and row["total_odds"] == "2.41" and len(json.loads(row["legs"])) == 2
    assert len(store.list_placed_legs()) == 2 and desk.key is None
    blocked = store.find_blocking_legs([leg_identity(l.leg_key) for l in token.legs])
    assert len(blocked) == 2 and all(d.proven for d in blocked)          # blocks exactly like PLACED


def test_no_max_stake_refusal_for_a_record_but_nothing_unreadable_is_accepted(store):
    token = booked_row(store)
    desk = PendingStake()
    press(store, desk, token.token_hash[:12], "p")
    assert stake_reply(store, desk, "99999999").status == "recorded"


def test_skip_records_it_without_a_stake(store):
    token = booked_row(store)
    desk = PendingStake()
    press(store, desk, token.token_hash[:12], "p")
    done = press(store, desk, token.token_hash[:12], "k")
    assert done.status == "recorded" and "No stake was recorded" in done.text
    row = store.get_placement(token.token_hash)
    assert row["state"] == P_PLACED_BY_OWNER and row["owner_stake"] is None and len(store.list_placed_legs()) == 2


def test_i_did_not_place_it_changes_nothing_and_it_can_be_answered_later(store):
    token = booked_row(store)
    desk, key = PendingStake(), token.token_hash[:12]
    press(store, desk, key, "p")
    noted = press(store, desk, key, "n")
    assert noted.status == "noted" and noted.text == NOT_PLACED_TEXT == "Noted: not placed." and desk.key is None
    assert press(store, desk, key, "n").status == "noted"                        # idempotent
    row = store.get_placement(token.token_hash)
    assert row["state"] == P_BOOKED and store.list_placed_legs() == []
    assert store.find_blocking_legs([leg_identity(l.leg_key) for l in token.legs]) == []
    press(store, desk, key, "p")
    assert press(store, desk, key, "k").status == "recorded"                    # later, from Mybets or the old message


def test_repeat_and_stale_presses_are_answered_and_change_nothing(store):
    token = booked_row(store)
    desk, key = PendingStake(), token.token_hash[:12]
    press(store, desk, key, "k")
    again = press(store, desk, key, "k")
    assert again.status == "already" and again.text == ALREADY_TEXT and len(store.list_placed_legs()) == 2   # not registered twice
    assert press(store, desk, key, "p").status == "already"
    assert press(store, desk, key, "n").status == "already"
    assert store.get_placement(token.token_hash)["state"] == P_PLACED_BY_OWNER
    assert press(store, desk, "0" * 12, "p").status == "unknown"
    other = booked_row(store, session="s2", state=P_PLACED)                      # a bet the bot proved: not for these buttons
    assert press(store, desk, other.token_hash[:12], "k").status == "stale"
    dry = booked_row(store, session="s3", dry_run=True)
    assert press(store, desk, dry.token_hash[:12], "k").status == "unknown"       # a dry-run row is never a booking
    assert stake_reply(store, PendingStake(), "500") is None                       # no open question: not a stake


def test_a_restart_forgets_the_question_but_the_booking_stays_booked_and_answerable(store):
    token = booked_row(store)
    key = token.token_hash[:12]
    press(store, PendingStake(), key, "p")                                         # the question was open ...
    fresh = PendingStake()                                                         # ... and the bot restarted
    assert stake_reply(store, fresh, "500") is None
    assert store.get_placement(token.token_hash)["state"] == P_BOOKED
    assert press(store, fresh, key, "p").status == "ask"                           # the old button still works
    assert stake_reply(store, fresh, "500").status == "recorded"


def test_a_failed_save_changes_nothing_and_can_be_pressed_again(store, monkeypatch):
    token = booked_row(store)
    desk = PendingStake()

    def broken(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(store, "resolve_owner_placed", broken)
    assert press(store, desk, token.token_hash[:12], "k").status == "error"
    assert store.get_placement(token.token_hash)["state"] == P_BOOKED


def test_the_state_change_is_a_compare_and_swap_in_the_database_itself(store):
    token = booked_row(store)
    assert store.resolve_owner_placed(token.token_hash, Decimal("500")) is True
    assert store.resolve_owner_placed(token.token_hash, Decimal("999")) is False
    assert Decimal(store.get_placement(token.token_hash)["owner_stake"]) == Decimal("500") and len(store.list_placed_legs()) == 2
    failed = booked_row(store, session="s9", state=P_FAILED)
    assert store.resolve_owner_placed(failed.token_hash, None) is False and store.get_placement(failed.token_hash)["state"] == P_FAILED


def test_mybets_lists_records_with_buttons_only_for_booked_rows(store):
    booked = booked_row(store, code="AAAA11", session="s1")
    mine = booked_row(store, code="BBBB22", session="s2")
    store.resolve_owner_placed(mine.token_hash, Decimal("500"))
    placed = booked_row(store, code="CCCC33", session="s3", state=P_PLACED)
    unknown = booked_row(store, code="", session="s4", state="PLACEMENT_UNKNOWN")
    dry = booked_row(store, code="DDDD44", session="s5", dry_run=True)
    screens = mybets_screens(store.recent_records(10))
    texts = "\n".join(s.text for s in screens)
    assert "BOOKED" in texts and "PLACED BY YOU" in texts and "UNKNOWN" in texts and "DDDD44" not in texts
    assert "Booking code: AAAA11" in texts and "Total odds: 2.41" in texts and "Stake: NGN 500" in texts
    with_buttons = [s for s in screens if s.buttons]
    assert len(with_buttons) == 1 and "AAAA11" in with_buttons[0].text
    assert [label for row in with_buttons[0].buttons for label, _ in row] == ["I placed it", "I did not place it"]
    assert mybets_screens([])[0].text.startswith("No booking codes")


# ================================================================ Results and Calibration
def graded_setup(store, stake="500", code="HW63TV"):
    evals = two_pick_setup()[0]
    store.save_evaluations("s1", evals)
    token = booked_row(store, code=code)
    return token, evals


def entry(status="Running", stake="500", bet_id="K1", placed_at=None):
    return history_entry(("Alpha v Beta", "Gamma v Delta"), stake, status, 0, placed_at or NOW + timedelta(minutes=30),
                         bet_id=bet_id)


def bm_with(*bets):
    fake = FakeBookmaker()
    fake.history = tuple(bets)
    return fake


def test_results_grade_an_owner_confirmed_bet_only_when_the_history_shows_it(store):
    token, _ = graded_setup(store)
    store.resolve_owner_placed(token.token_hash, Decimal("500"))
    report = run(check_results(store, bm_with(entry("Won")), str))
    assert report.checked == 1 and report.changes and "WON" in report.changes[0] and not report.problems
    row = store.get_placement(token.token_hash)
    assert row["bet_id"] == "K1" and row["state"] == P_PLACED_BY_OWNER and store.list_bet_outcomes()[0]["result"] == "win"
    assert len(store.list_leg_outcomes()) == 2 and any("graded leg" in l for l in calibration_lines(store))


def test_an_owner_confirmed_bet_without_a_match_is_not_graded_and_nothing_is_invented(store):
    token, _ = graded_setup(store)
    store.resolve_owner_placed(token.token_hash, Decimal("500"))
    report = run(check_results(store, bm_with(entry("Won", stake="999")), str))     # a different stake: not this bet
    assert report.changes == [] and "not graded: not found in the bet history" in report.problems[0]
    assert store.list_bet_outcomes() == [] and store.get_placement(token.token_hash)["bet_id"] == ""
    two = run(check_results(store, bm_with(entry("Won", bet_id="K1"), entry("Won", bet_id="K2")), str))
    assert "more than one history entry" in two.problems[0] and store.list_bet_outcomes() == []


def test_an_owner_confirmed_bet_without_a_stake_is_not_graded(store):
    token, _ = graded_setup(store)
    store.resolve_owner_placed(token.token_hash, None)
    report = run(check_results(store, bm_with(entry("Won")), str))
    assert "not graded: you did not give the stake" in report.problems[0] and store.list_bet_outcomes() == []


def test_a_booked_row_is_never_graded_even_when_a_matching_bet_exists(store):
    graded_setup(store)
    report = run(check_results(store, bm_with(entry("Won")), str))
    assert report.checked == 0 and store.list_bet_outcomes() == [] and store.graded_legs() == []


def test_a_history_entry_older_than_the_booking_is_not_the_owners_bet(store):
    token, _ = graded_setup(store)
    store.resolve_owner_placed(token.token_hash, Decimal("500"))
    old = entry("Won", placed_at=NOW - timedelta(days=3))
    report = run(check_results(store, bm_with(old), str))
    assert report.changes == [] and store.list_bet_outcomes() == []


# ================================================================ snapshot, migration, recovery
def test_the_snapshot_round_trips_the_booking_summary_and_reads_older_snapshots(benv, tmp_path):
    session = benv.to_summary()
    again = from_json(to_json(session))
    assert again == session and again.booking and again.summary.booking and again.summary.stake == 0
    placing = Env(tmp_path, dry_run=False)
    try:
        placing.to_summary()
        data = json.loads(to_json(placing.session))
    finally:
        placing.close()
    del data["booking"], data["summary"]["booking"]                       # a snapshot written before AD-25
    old = from_json(json.dumps(data))
    assert old.booking is False and old.summary.booking is False and old.summary.stake == 500


def test_the_v6_migration_keeps_every_older_row_and_adds_the_owner_columns(tmp_path):
    db = tmp_path / "old.db"
    with EvaluationStore.open(db, str, migrations=MIGRATIONS[:5]) as old:
        token = booked_row(old, state=P_PLACED)
        assert old.schema_version == 5
    with EvaluationStore.open(db, str, backup_dir=tmp_path / "bk") as new:
        assert new.schema_version == 7 == len(MIGRATIONS)
        row = new.get_placement(token.token_hash)
        assert row["state"] == P_PLACED and row["source"] == "" and row["owner_stake"] is None


class _NoHistory:
    dry_run = False

    async def bet_history(self):
        raise AssertionError("a booking has no bet to look for: the history must not be read")


def test_a_restart_during_a_booking_is_no_bet_not_unknown(tmp_path, benv):
    benv.to_summary()
    token = benv.press_final().token                                          # PLACING is saved; the bot "dies" here
    row = benv.store.record_placing(token, False)
    fresh = SessionManager(clock=lambda: NOW, book_mode=True)
    from predictions.flow.recovery import attach_store
    attach_store(fresh, benv.store)
    reports = run(recover_on_startup(fresh, benv.store, lambda: _NoHistory(), str))
    assert reports and "no bet was placed" in reports[0].text.lower()
    assert benv.store.get_placement(token.token_hash)["state"] == P_FAILED
    assert benv.store.find_blocking_legs([leg_identity(l.leg_key) for l in token.legs]) == []


def test_a_restart_after_a_booking_keeps_it_booked_and_the_slate_over(benv):
    benv.to_summary()
    token = benv.press_final().token
    benv.place(token)
    saved = benv.store.load_latest_session()
    assert saved["state"] == "DONE" and benv.store.get_placement(token.token_hash)["state"] == P_BOOKED
    fresh = SessionManager(clock=lambda: NOW, book_mode=True)
    reports = run(recover_on_startup(fresh, benv.store, lambda: _NoHistory(), str))
    assert reports == [] and benv.store.get_placement(token.token_hash)["state"] == P_BOOKED


# ================================================================ place mode's unknown message: code and odds apart
def test_place_mode_unknown_message_shows_the_code_the_odds_and_the_stake_on_separate_lines(tmp_path):
    env = Env(tmp_path, dry_run=False)
    try:
        env.bm.place_mode = "unknown"
        env.bm.unknown_code, env.bm.unknown_tried = "ABC12D", True
        env.to_summary()
        out = env.place(env.press_final().token)
        lines = out.message.split("\n")
        assert lines[-3:] == ["Booking code: ABC12D", "Total odds: 2.41", "Stake: NGN 500"]
        assert "not proof of a bet" in lines[-4] and "may or may not exist" in lines[-4]
        from predictions.telegram.slate import failure_screen
        screen = failure_screen(env.session, now=NOW)
        assert "\nBooking code: ABC12D\nTotal odds: 2.41\nStake: NGN 500" in screen.text
    finally:
        env.close()


# ================================================================ end to end through the real Telegram handlers
@pytest.fixture
def book_chain(monkeypatch, tmp_path):
    sent, edits, answers, events, reply_markup_edits = [], [], [], [], []
    scenario = {"html_error": False}
    evals, listing = two_pick_setup()
    fake = FakeBookmaker(listing, dry_run=False)

    async def no_network(self):
        self._bot_user = User(id=1, first_name="TestBot", is_bot=True, username="test_bot")

    async def send_message(self, chat_id, text, **kwargs):
        if scenario["html_error"] and kwargs.get("parse_mode") == "HTML":
            raise BadRequest("can't parse entities")
        sent.append((text, kwargs))
        events.append((text, kwargs))

    async def edit_message_text(self, text, *args, **kwargs):
        edits.append((text, kwargs))
        events.append((text, kwargs))

    async def edit_message_reply_markup(self, *args, **kwargs):
        reply_markup_edits.append(kwargs)

    async def answer_callback_query(self, callback_query_id, text=None, show_alert=None, **kwargs):
        answers.append((text, show_alert))

    async def fake_run(argument, source, sanitize, **kwargs):
        kwargs["on_result"](GateResult(tuple(evals), (), 2, 0, {}))
        return ["Matchday fixtures ..."]

    monkeypatch.setattr(ExtBot, "initialize", no_network)
    monkeypatch.setattr(ExtBot, "send_message", send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", edit_message_text)
    monkeypatch.setattr(ExtBot, "edit_message_reply_markup", edit_message_reply_markup)
    monkeypatch.setattr(ExtBot, "answer_callback_query", answer_callback_query)
    monkeypatch.setattr(bot, "run_matchday", fake_run)
    ticks = itertools.count()    # a clock that moves a little on every look, so a retry's summary has a new expiry (a new token)
    monkeypatch.setattr(bot, "SESSIONS", SessionManager(clock=lambda: NOW + timedelta(milliseconds=next(ticks)), book_mode=True))
    monkeypatch.setattr(bot, "BOOKMAKER", fake)
    monkeypatch.setattr(bot, "DRY_RUN", False)
    monkeypatch.setattr(bot, "BOOKING_DESK", PendingStake())
    store = EvaluationStore.open(tmp_path / "t.db", bot.redact)
    monkeypatch.setattr(bot, "STORE", store)
    application = ApplicationBuilder().token("123456:TEST-TOKEN-NOT-REAL").build()
    bot.register_handlers(application)
    chain = Chain(application, sent, edits, answers)
    chain.scenario, chain.store, chain.events, chain.fake, chain.evals = scenario, store, events, fake, evals
    chain.markup_edits = reply_markup_edits
    yield chain
    store.close()


def to_final(chain):
    chain.text(OWNER_ID, "Matchday 10 October")
    chain.press(OWNER_ID, find(chain.buttons(-1), "Final confirm"))
    chain.press(OWNER_ID, find(chain.buttons(-1), "Confirm games"))


def texts(chain):
    return [t for t, _ in chain.sent]


def test_the_whole_book_journey_one_message_with_the_code_then_the_owner_says_he_placed_it(book_chain):
    chain = book_chain
    to_final(chain)
    shown = texts(chain)
    assert any(t.startswith("Loaded on SportyBet") for t in shown)
    assert not any("How much do you want to stake" in t for t in shown)               # no stake question at all
    summary, kwargs = chain.sent[-1]
    assert "Confirm to get your booking code" in summary and "Total odds: 2.41" in summary
    assert [b.text for row in kwargs["reply_markup"].inline_keyboard for b in row] == ["Get booking code"]
    assert bot.SESSIONS.active.state is State.AWAITING_FINAL_CONFIRM
    chain.text(OWNER_ID, "500")                                                        # a typed number is not a stake here
    assert "Unrecognized trigger" in chain.sent[-1][0] and bot.SESSIONS.active.state is State.AWAITING_FINAL_CONFIRM
    before = len(chain.sent)
    chain.press(OWNER_ID, kwargs["reply_markup"].inline_keyboard[0][0].callback_data)
    new = chain.sent[before:]
    assert "getting your booking code" in chain.edits[-1][0]                           # the summary message became the working screen
    assert len(new) == 1                                                               # and exactly ONE new message: the result
    text, extra = new[0]
    assert extra["parse_mode"] == "HTML"
    lines = text.split("\n")
    assert lines[0] == "Booking code: <code>HW63TV</code>" and lines[1] == "Total odds: 2.41"
    assert lines[3].startswith("1. Alpha v Beta") and lines[-1].endswith("Nothing was placed by the bot.")
    assert "Open SportyBet, enter the code, choose your stake and press Place Bet yourself." in text
    assert [b.text for row in extra["reply_markup"].inline_keyboard for b in row] == ["I placed it", "I did not place it"]
    assert chain.fake.book_clicks == 1 and chain.fake.clicks == 0 and bot.SESSIONS.active.state is State.DONE
    (row,) = chain.store.list_placements()
    assert row["state"] == P_BOOKED and row["booking_code"] == "HW63TV" and chain.store.list_placed_legs() == []
    # a second tap places/books nothing more
    chain.press(OWNER_ID, kwargs["reply_markup"].inline_keyboard[0][0].callback_data)
    assert chain.fake.book_clicks == 1
    # 'I placed it' -> the stake question with Skip -> the amount
    placed = extra["reply_markup"].inline_keyboard[0][0].callback_data
    chain.press(OWNER_ID, placed)
    assert chain.sent[-1][0] == ASK_TEXT and [b.text for row in chain.sent[-1][1]["reply_markup"].inline_keyboard for b in row] == ["Skip"]
    chain.text(OWNER_ID, "nonsense")
    assert "not a valid stake" in chain.sent[-1][0] or "stake" in chain.sent[-1][0].lower()
    chain.text(OWNER_ID, "500")
    assert chain.sent[-1][0].startswith("Recorded as placed (confirmed by you).") and "NGN 500.00" in chain.sent[-1][0]
    row = chain.store.list_placements()[0]
    assert row["state"] == P_PLACED_BY_OWNER and Decimal(row["owner_stake"]) == 500 and len(chain.store.list_placed_legs()) == 2
    assert chain.markup_edits and chain.markup_edits[-1]["reply_markup"] is None      # the two buttons are gone from that message
    chain.press(OWNER_ID, placed)                                                      # repeat: nothing changes
    assert len(chain.store.list_placed_legs()) == 2


def test_i_did_not_place_it_through_the_handlers(book_chain):
    chain = book_chain
    to_final(chain)
    chain.press(OWNER_ID, chain.buttons(-1)[0][1])
    answer_buttons_row = chain.buttons(-1)
    chain.press(OWNER_ID, find(answer_buttons_row, "I did not place it"))
    assert chain.sent[-1][0] == "Noted: not placed." and chain.store.list_placements()[0]["state"] == P_BOOKED
    assert chain.store.list_placed_legs() == []


def test_a_stranger_cannot_press_the_booking_buttons(book_chain):
    chain = book_chain
    to_final(chain)
    chain.press(OWNER_ID, chain.buttons(-1)[0][1])
    placed = find(chain.buttons(-1), "I placed it")
    count = len(chain.sent)
    chain.press(STRANGER, placed)
    chain.text(STRANGER, "Mybets")
    assert len(chain.sent) == count and chain.store.list_placements()[0]["state"] == P_BOOKED


def test_mybets_through_the_handlers_offers_the_buttons_for_booked_rows_and_skip_works(book_chain):
    chain = book_chain
    to_final(chain)
    chain.press(OWNER_ID, chain.buttons(-1)[0][1])
    chain.text(OWNER_ID, "Mybets")
    listing = chain.sent[-2:]
    assert "most recent records" in chain.sent[-2][0] and "BOOKED" in chain.sent[-1][0] and "Booking code: HW63TV" in chain.sent[-1][0]
    chain.press(OWNER_ID, find(chain.buttons(-1), "I placed it"))
    chain.press(OWNER_ID, find(chain.buttons(-1), "Skip"))
    assert chain.sent[-1][0].startswith("Recorded as placed (confirmed by you).") and "No stake was recorded" in chain.sent[-1][0]
    chain.text(OWNER_ID, "mybets")
    assert "PLACED BY YOU" in chain.sent[-1][0] and "Stake: not recorded" in chain.sent[-1][0]


def test_when_telegram_refuses_html_the_same_message_goes_out_plain(book_chain):
    chain = book_chain
    chain.scenario["html_error"] = True
    to_final(chain)
    chain.press(OWNER_ID, chain.buttons(-1)[0][1])
    text, extra = chain.sent[-1]
    assert extra.get("parse_mode") != "HTML" and text.split("\n")[0] == "Booking code: HW63TV" and "<code>" not in text
    assert [b.text for row in extra["reply_markup"].inline_keyboard for b in row] == ["I placed it", "I did not place it"]


def test_a_booking_without_a_readable_code_shows_the_failure_screen_and_try_again_returns_to_the_final_screen(book_chain):
    chain = book_chain
    chain.fake.book_code = ""
    to_final(chain)
    chain.press(OWNER_ID, chain.buttons(-1)[0][1])
    text, extra = chain.sent[-1]
    assert "No bet was placed." in text and "The booking code could not be read." in text and "Try again" in text
    assert "same stake" not in text and "Nothing is booked until you confirm" in text
    assert chain.store.list_booked() == []
    chain.fake.book_code = "ZZ99AA"
    chain.press(OWNER_ID, find(chain.buttons(-1), "Try again"))
    summary, kwargs = chain.sent[-1]
    assert "Retry 1 of 3" in summary and "Confirm to get your booking code" in summary
    assert not any("How much do you want to stake" in t for t in texts(chain))
    chain.press(OWNER_ID, kwargs["reply_markup"].inline_keyboard[0][0].callback_data)
    assert chain.sent[-1][0].split("\n")[0] == "Booking code: <code>ZZ99AA</code>"


def test_book_dry_run_through_the_handlers_books_nothing(book_chain, monkeypatch):
    chain = book_chain
    dry = FakeBookmaker(chain.fake.listing, dry_run=True)
    monkeypatch.setattr(bot, "BOOKMAKER", dry)
    monkeypatch.setattr(bot, "DRY_RUN", True)
    to_final(chain)
    summary, kwargs = chain.sent[-1]
    assert "DRY RUN" in summary and "No booking code is made" in summary
    chain.press(OWNER_ID, kwargs["reply_markup"].inline_keyboard[0][0].callback_data)
    assert "no booking code was made" in chain.sent[-1][0] and dry.book_clicks == 0 and dry.clicks == 0
    assert "Booking code" not in chain.sent[-1][0] and chain.store.list_booked() == []


def test_the_startup_message_and_help_name_the_mode_and_mybets(book_chain, monkeypatch):
    monkeypatch.setattr(bot, "PLACE_MODE", "book")
    sent = []

    async def fake_send(bot_, text, markup=None, parse_mode=None):
        sent.append(text)

    monkeypatch.setattr(bot, "send_to_owner", fake_send)

    async def no_recover(b):
        return []

    monkeypatch.setattr(bot, "recover_after_start", no_recover)
    monkeypatch.setattr(bot, "start_day_cache_build", lambda: None)
    asyncio.run(bot.announce_startup(type("A", (), {"bot": object(), "create_task": lambda self, c: None})()))
    assert "PLACE_MODE=book" in sent[0] and "never place a bet" in sent[0]
    src = open(bot.__file__, encoding="utf-8").read()
    assert "`Mybets`" in src
