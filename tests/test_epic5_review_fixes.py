"""Epic 5 review fixes: a bet is recognised in the history LIST only by what that page shows (games, stake, leg count,
status, time; never odds or picks), reconciliation cannot resolve an
unknown placement with an unrelated bet, PLACED and its legs are one write, the stake has an upper bound, a Telegram
failure cannot leave a slate stuck, and unsaved slates are reported. No browser, no network, no real bet."""
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest
from telegram.error import TimedOut
from telegram.ext import ExtBot

import app as bot
from fakes_bookmaker import NOW, FakeBookmaker, history_entry
from predictions.adapters.sportybet import betslip as bs
from predictions.adapters.sportybet.betslip import SELECTORS
from predictions.core import staking
from predictions.core.betting import HistoryBet, LoadedLeg, history_bet_matches
from predictions.core.placement import PlacementKind
from predictions.core.ports import PlacementUnknown
from predictions.core.staking import StakeError, parse_max_stake, parse_stake
from predictions.flow.placement import OUTCOME_PLACED, OUTCOME_UNKNOWN
from predictions.flow.reconcile import reconcile_placement
from predictions.flow.session import SessionManager, State
from predictions.flow.snapshot import to_json
from predictions.flow.store import EvaluationStore
from test_epic5_app import live_chain, to_stake, OWNER_ID, texts  # noqa: F401  (the fixture)
from test_epic5_recovery import OWNER, Rig as FlowRig, history_bet, run as flow_run
from test_sportybet_betslip import Rig as SiteRig, new_bet, run

OVER = LoadedLeg(("m", "18", "total=2.5", "over"), "Arsenal v Chelsea", "Over/Under", "Over 2.5", Decimal("2.00"), "Over 2.5")
T0 = datetime(2026, 10, 10, 12, 0, tzinfo=timezone.utc)


def entry(matches=("Arsenal v Chelsea",), stake="1000", placed_at=None, others=0, status="Running", **kw):
    return history_entry(matches, stake, status, others, placed_at, **kw)


# ================================================================ 1. the list shows no odds or picks: they are never compared

def test_the_entry_proves_the_games_stake_and_leg_count_and_says_nothing_about_odds_or_picks():
    assert history_bet_matches(entry(), [OVER], Decimal("1000"))
    assert history_bet_matches(replace(entry(), total_odds=Decimal("1.10")), [OVER], Decimal("1000"))   # odds are not read
    assert not history_bet_matches(entry(stake="999"), [OVER], Decimal("1000"))
    assert not history_bet_matches(entry(("Arsenal v Fulham",)), [OVER], Decimal("1000"))
    assert not history_bet_matches(entry(("Chelsea v Arsenal",)), [OVER], Decimal("1000"))              # the reverse fixture


def test_the_picked_side_cannot_be_checked_on_the_list_so_the_message_says_so():
    from predictions.core.betting import HISTORY_LIMITS_NOTE, history_limits_note
    assert "does not show odds or selections" in HISTORY_LIMITS_NOTE
    assert "only the games, stake and number of legs were verified" in HISTORY_LIMITS_NOTE
    assert history_limits_note(entry()) == HISTORY_LIMITS_NOTE


# ================================================================ 2. leg count and time

def test_an_entry_with_another_leg_count_or_an_unknown_one_never_matches():
    assert not history_bet_matches(entry(("Arsenal v Chelsea", "Other v Team")), [OVER], Decimal("1000"))   # larger
    assert not history_bet_matches(replace(entry(), leg_count=None), [OVER], Decimal("1000"))
    assert not history_bet_matches(entry(others=2), [OVER], Decimal("1000"))      # 1 shown + 2 other = 3 games, 1 leg
    assert history_bet_matches(entry(), [OVER], Decimal("1000"))


def test_an_entry_older_than_the_placement_never_matches():
    assert not history_bet_matches(entry(placed_at=T0 - timedelta(days=3)), [OVER], Decimal("1000"), T0)
    assert history_bet_matches(entry(placed_at=T0 + timedelta(minutes=1)), [OVER], Decimal("1000"), T0)
    assert history_bet_matches(entry(placed_at=T0 - timedelta(seconds=40)), [OVER], Decimal("1000"), T0)
    assert not history_bet_matches(replace(entry(), placed_at=None), [OVER], Decimal("1000"), T0)   # unreadable time: no proof


# ================================================================ 1. the adapter

@pytest.fixture
def live(tmp_path):
    return SiteRig(tmp_path, dry_run=False)


def test_a_proven_bet_says_plainly_that_odds_and_selections_were_not_verified_on_the_list(live):
    token = live.token("500")
    live.site.history_after_click = [new_bet(live)]
    result = run(live.bm.place(token))
    assert result.kind is PlacementKind.PLACED and result.bet_id == new_bet(live).bet_id
    assert ("the history list does not show odds or selections, so only the games, stake and number of legs were "
            "verified") in result.note.lower().replace("the history list", "the history list")
    assert live.site.place_clicks == 1


def test_a_proven_bet_with_more_than_three_legs_says_only_three_could_be_compared(live):
    from predictions.core.betting import history_limits_note
    long_bet = entry(("A v B", "C v D", "E v F"), others=2, bet_id="K")
    assert "first 3 of 5 games" in history_limits_note(long_bet) and "other 2 could not be compared" in history_limits_note(long_bet)


def test_a_new_entry_naming_other_games_is_not_proof(live, monkeypatch):
    monkeypatch.setattr(bs, "HISTORY_POLLS", 1)
    token = live.token("500")
    live.site.history_after_click = [history_entry(("Alpha v Beta", "Zulu v Yankee"), "500")]
    with pytest.raises(PlacementUnknown):
        run(live.bm.place(token))


def test_the_history_is_read_after_the_stake_and_just_before_the_last_betslip_check(live):
    token = live.token("500")
    live.site.history_after_click = [new_bet(live)]
    run(live.bm.place(token))
    log = live.site.log[live.site.log.index("stake:500"):]
    assert log[:4] == ["stake:500", "history", "read", "PLACE"]          # nothing slow between the last check and the click


def test_the_refusal_words_after_place_include_the_new_ones():
    for word in ("updated", "moved", "increase", "decrease", "reject", "limit", "expired", "changed", "fail"):
        assert word in SELECTORS.dialog_refusal_words


@pytest.mark.parametrize("dialog", ["Odds updated", "Stake limit exceeded", "Bet rejected", "Selection expired"])
def test_a_refusal_dialog_after_place_is_never_confirmed(dialog):
    from test_sportybet_betslip import make_site
    site, dom = make_site(allow_place=True)
    clicked = []

    class Loc:
        def __init__(self, sel):
            self.sel = sel

        @property
        def first(self):
            return self

        async def count(self):
            return 1

        async def is_visible(self):
            return True

        async def inner_text(self):
            return dialog

        async def click(self):
            clicked.append(self.sel)

    site.page = SimpleNamespace(locator=Loc, wait_for_timeout=lambda ms: asyncio.sleep(0))
    with pytest.raises(bs.BookmakerError):
        run(site.confirm_placement())
    assert clicked == []


# ================================================================ 2. reconciliation

def unknown_row(rig):
    rig.bm.place_mode = "unknown"
    rig.to_summary()
    token = rig.press_final().token
    outcome = rig.place(token)
    assert outcome.kind == OUTCOME_UNKNOWN
    return token


def reconcile(rig, token):
    return flow_run(reconcile_placement(rig.store, rig.bm, token.token_hash, str))


@pytest.fixture
def rig(tmp_path):
    r = FlowRig(tmp_path)
    yield r
    r.close()


def test_reconciliation_ignores_an_older_bet_a_larger_accumulator_and_an_unreadable_leg_count(rig):
    rig.bm.place_mode = "unknown"
    rig.bm.history = ()
    token = unknown_row(rig)
    created = datetime.fromisoformat(rig.store.get_placement(token.token_hash)["created_at_utc"])
    for other in (history_bet("B1", placed_at=created - timedelta(days=2)),
                  history_bet("B2", leg_count=3, shown=("Alpha v Beta", "Gamma v Delta", "Foo v Bar")),
                  history_bet("B3", leg_count=None)):
        rig.bm.history = (other,)
        outcome = reconcile(rig, token)
        assert not outcome.found and outcome.kind == "not_found", other.bet_id
    assert rig.store.get_placement(token.token_hash)["state"] == "PLACEMENT_UNKNOWN"
    assert rig.store.list_placed_legs() == []
    rig.bm.history = (history_bet("B4", placed_at=created + timedelta(minutes=1)),)
    assert reconcile(rig, token).found


def test_reconciliation_says_the_list_showed_no_odds_or_selections(rig):
    token = unknown_row(rig)
    rig.bm.history = (history_bet("B8"),)
    outcome = reconcile(rig, token)
    assert outcome.found and "does not show odds or selections" in outcome.text
    assert rig.store.get_placement(token.token_hash)["state"] == "PLACED" and len(rig.store.list_placed_legs()) == 2


def test_a_placed_result_with_a_note_carries_it_to_the_owner(rig):
    rig.bm.place_mode = "ok"
    rig.bm.dry = False
    rig.to_summary()
    token = rig.press_final().token
    real = rig.bm.place

    async def with_note(t):
        result = await real(t)
        return replace(result, note="The history list does not show odds or selections.")

    rig.bm.place = with_note
    outcome = rig.place(token)
    assert outcome.kind == OUTCOME_PLACED and "does not show odds or selections" in outcome.message


# ================================================================ 3. PLACED and its legs are one write

def test_a_failure_between_the_two_writes_leaves_nothing_half_done(rig):
    rig.to_summary()
    token = rig.press_final().token
    rig.store.record_placing(token, False)
    real = rig.store._text
    seen = []

    def failing(value):
        seen.append(value)
        if value == "Over 1.5":           # the second leg's selection: the first leg row is already written
            raise RuntimeError("disk full")
        return real(value)

    rig.store._text = failing
    with pytest.raises(RuntimeError):
        rig.store.resolve_placement_found(token.token_hash, "B1", "C", "d", token.legs)
    rig.store._text = real
    assert rig.store.get_placement(token.token_hash)["state"] == "PLACING" and rig.store.list_placed_legs() == []


def test_run_placement_registers_legs_and_state_together(rig):
    rig.bm._dry_run = False
    rig.to_summary()
    outcome = rig.place(rig.press_final().token)
    assert outcome.kind == OUTCOME_PLACED
    assert rig.store.list_placements("PLACED") and len(rig.store.list_placed_legs()) == 2


def test_if_the_joint_write_fails_the_bet_is_still_blocked_and_the_owner_told(rig, monkeypatch):
    rig.bm._dry_run = False
    rig.to_summary()
    token = rig.press_final().token

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(rig.store, "resolve_placement_found", boom)
    outcome = rig.place(token)
    assert outcome.kind == OUTCOME_PLACED and "could not be registered" in outcome.warning
    identities = [leg.leg_key[:3] for leg in token.legs]
    assert len(rig.store.find_blocking_legs(identities)) == 2       # PLACED without legs rows still blocks


def test_a_placed_row_with_no_leg_rows_blocks_and_recovery_registers_the_legs(rig):
    rig.bm._dry_run = False
    rig.to_summary()
    token = rig.press_final().token
    rig.store.record_placing(token, False)
    rig.store.update_placement(token.token_hash, "PLACED", bet_id="B1")          # the crash: legs never written
    identities = [leg.leg_key[:3] for leg in token.legs]
    blocking = rig.store.find_blocking_legs(identities)
    assert len(blocking) == 2 and all(d.source == "placed" for d in blocking)
    assert rig.store.list_placed_legs() == []
    reports = rig.restart()
    assert len(rig.store.list_placed_legs()) == 2
    assert reports == [] or all("could not" not in r.text for r in reports)
    rig.restart()                                                                # idempotent
    assert len(rig.store.list_placed_legs()) == 2


def test_a_saved_slate_whose_bet_was_placed_is_never_told_nothing_was_placed(rig):
    rig.bm._dry_run = False
    rig.to_summary()
    outcome = rig.place(rig.press_final().token)
    assert outcome.kind == OUTCOME_PLACED
    stale = replace(rig.session, state=State.AWAITING_FINAL_CONFIRM)             # an old snapshot with the token set
    rig.store.save_session(stale.session_id, stale.state.value, stale.state_version + 5, to_json(stale), stale.origin)
    rig.store._conn.execute("DELETE FROM placed_legs")
    reports = rig.restart()
    text = " ".join(r.text for r in reports)
    assert "was placed" in text and "Nothing was placed" not in text and "bet id" in text
    assert len(rig.store.list_placed_legs()) == 2


# ================================================================ 4. the stake limit

def test_a_stake_above_the_limit_is_refused_with_how_to_raise_it():
    assert parse_stake("100000") == Decimal("100000")
    with pytest.raises(StakeError) as caught:
        parse_stake("100000.01")
    assert "100,000" in str(caught.value) and "MAX_STAKE" in str(caught.value) and "config.env" in str(caught.value)
    with pytest.raises(StakeError):
        parse_stake("99999999999999.99")
    assert parse_stake("250000", Decimal("300000")) == Decimal("250000")
    with pytest.raises(StakeError):
        parse_stake("500.01", Decimal("500"))


def test_max_stake_reads_the_setting_and_falls_back_to_the_default():
    assert parse_max_stake("2500") == Decimal("2500") and parse_max_stake(" 1,000 ") == Decimal("1000")
    for unset in (None, "", "   "):
        assert parse_max_stake(unset) == staking.DEFAULT_MAX_STAKE == Decimal("100000")
    for bad in ("abc", "5k", "0", "-5", "NaN", "Infinity"):
        with pytest.raises(ValueError) as caught:
            parse_max_stake(bad)
        assert "MAX_STAKE" in str(caught.value)
    with pytest.raises(ValueError) as caught:
        parse_max_stake("zzsecret9")
    assert "zzsecret9" not in str(caught.value)


def test_the_session_uses_the_limit_it_was_given(rig):
    rig.to_stake()
    manager = SessionManager(clock=lambda: NOW, max_stake=Decimal("400"))
    assert manager.max_stake == Decimal("400") and rig.manager.max_stake == Decimal("100000")
    assert rig.manager.submit_stake(OWNER, "100001").notice.startswith("That stake is above the limit")
    assert rig.session.state is State.AWAITING_STAKE
    assert rig.manager.submit_stake(OWNER, "100000").status.name == "APPLIED"


def test_the_app_reads_max_stake_at_startup_and_hands_it_down():
    assert bot.MAX_STAKE == parse_max_stake(__import__("os").getenv("MAX_STAKE"))
    assert "MAX_STAKE" in (bot.Path(bot.__file__).read_text(encoding="utf-8"))


def test_an_over_limit_stake_typed_in_the_chat_is_refused(live_chain):
    chain = live_chain
    to_stake(chain)
    chain.text(OWNER_ID, "1000000")
    assert "above the limit" in chain.sent[-1][0] and "MAX_STAKE" in chain.sent[-1][0]
    assert bot.SESSIONS.active.state is State.AWAITING_STAKE


# ================================================================ 5. Telegram failures cannot strand a slate

def test_a_timeout_answering_the_button_still_runs_the_job(live_chain, monkeypatch):
    chain = live_chain

    async def timed_out(self, *a, **k):
        raise TimedOut("timed out")

    monkeypatch.setattr(ExtBot, "answer_callback_query", timed_out)
    to_stake(chain)
    assert bot.SESSIONS.active.state is State.AWAITING_STAKE and [c[0] for c in chain.fake.calls] == ["check", "load"]
    assert any("How much do you want to stake" in t for t in texts(chain))


def test_a_timeout_editing_the_message_still_runs_the_job(live_chain, monkeypatch):
    chain = live_chain

    async def timed_out(self, *a, **k):
        raise TimedOut("timed out")

    monkeypatch.setattr(ExtBot, "edit_message_text", timed_out)
    to_stake(chain)
    assert bot.SESSIONS.active.state is State.AWAITING_STAKE and [c[0] for c in chain.fake.calls] == ["check", "load"]


def test_the_odds_screen_failing_once_is_resent_to_the_owner_and_the_stake_is_still_asked(live_chain, monkeypatch):
    chain = live_chain
    real = ExtBot.send_message
    state = {"failed": 0}

    async def flaky(self, chat_id, text, **kwargs):
        if text.startswith("Loaded on SportyBet") and state["failed"] == 0:
            state["failed"] += 1
            raise TimedOut("timed out")
        return await real(self, chat_id, text, **kwargs)

    monkeypatch.setattr(ExtBot, "send_message", flaky)
    to_stake(chain)
    assert state["failed"] == 1 and bot.SESSIONS.active.state is State.AWAITING_STAKE and bot.SESSIONS.active.odds_shown
    shown = texts(chain)
    assert (next(i for i, t in enumerate(shown) if t.startswith("Loaded on SportyBet"))
            < next(i for i, t in enumerate(shown) if "How much do you want to stake" in t))


def test_when_the_odds_can_never_be_shown_no_stake_is_asked_and_the_slate_ends(live_chain, monkeypatch):
    chain = live_chain
    real = ExtBot.send_message

    async def deaf(self, chat_id, text, **kwargs):
        if text.startswith("Loaded on SportyBet"):
            raise TimedOut("timed out")
        return await real(self, chat_id, text, **kwargs)

    monkeypatch.setattr(ExtBot, "send_message", deaf)
    to_stake(chain)
    assert bot.SESSIONS.active.state is State.FAILED and not bot.SESSIONS.active.odds_shown
    assert not any("How much do you want to stake" in t for t in texts(chain))
    assert chain.fake.clicks == 0


def test_a_timeout_sending_the_placement_result_does_not_lose_it(live_chain, monkeypatch):
    chain = live_chain
    to_stake(chain)
    chain.text(OWNER_ID, "500")
    button = chain.buttons(-1)[0][1]
    real = ExtBot.send_message
    state = {"failed": 0}

    async def flaky(self, chat_id, text, **kwargs):
        if "DRY RUN: nothing was placed" in text and state["failed"] == 0:
            state["failed"] += 1
            raise TimedOut("timed out")
        return await real(self, chat_id, text, **kwargs)

    monkeypatch.setattr(ExtBot, "send_message", flaky)
    chain.press(OWNER_ID, button)
    assert state["failed"] == 1 and "DRY RUN: nothing was placed" in chain.sent[-1][0]
    assert bot.SESSIONS.active.state is State.DONE


# ================================================================ 6. persist errors are reported

def test_the_session_manager_hands_out_persist_errors_once():
    manager = SessionManager(clock=lambda: NOW)

    def broken(session):
        raise OSError("disk 08031234567")

    manager.attach(broken, None)
    from fakes_bookmaker import two_pick_setup
    manager.start(two_pick_setup()[0], "s1", "matchday")
    assert manager.take_persist_errors() == ["OSError"] and manager.take_persist_errors() == []


def test_a_slate_that_could_not_be_saved_is_reported_to_the_owner(live_chain, monkeypatch):
    chain = live_chain

    def broken(*a, **k):
        raise OSError("disk full")

    from predictions.flow.recovery import attach_store
    attach_store(bot.SESSIONS, chain.store)
    to_stake(chain)
    monkeypatch.setattr(chain.store, "save_session", broken)
    chain.text(OWNER_ID, "500")
    assert any("could not be saved" in t and "OSError" in t for t in texts(chain))
    assert bot.SESSIONS.take_persist_errors() == []                       # reported once, then cleared
