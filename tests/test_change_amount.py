"""Potential winnings and the Change amount button on the final summary (owner request, FR-9/FR-10, AD-6, AD-19, AD-22).

Everything runs against the FakeBookmaker; nothing is ever placed and nothing logs in anywhere."""
from decimal import Decimal

import pytest

from fakes_bookmaker import make_fixture, pick_evaluation, totals_offer
from predictions.core.staking import DEFAULT_MAX_STAKE, format_money, potential_profit, potential_winnings
from predictions.flow.loading import KIND_LOADED
from predictions.flow.recovery import KIND_RELOAD
from predictions.flow.session import Action, State, Status
from predictions.flow.snapshot import from_json, to_json
from predictions.telegram.slate import (DRY_BANNER, LIVE_BANNER, decode, encode, handle_callback, handle_stake_text,
                                        stake_prompt, summary_screen)
from test_epic5_flow import Env, OWNER, SID
from test_epic5_recovery import Rig


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path, dry_run=False)
    yield e
    e.close()


# ---------------------------------------------------------------- the maths (core)

@pytest.mark.parametrize("odds,stake,expected", [
    ("3.50", "1000", "3500.00"),
    ("2.41", "500", "1205.00"),
    ("1.01", "0.01", "0.01"),                        # 0.0101 rounds DOWN
    ("2.33", "1.99", "4.63"),                        # 4.6367 rounds DOWN, never up to 4.64
    ("1.005", "100", "100.50"),
    ("3.333333", "3", "9.99"),                       # many decimals: 9.999999 rounds down to 9.99
    ("1234.56", "100000", "123456000.00"),           # a large stake
    ("999999.99", "99999999.99", "99999998990000.00"),
])
def test_potential_winnings_round_down_and_never_overstate(odds, stake, expected):
    value = potential_winnings(Decimal(odds), Decimal(stake))
    exact = Decimal(odds) * Decimal(stake)
    assert value <= exact and exact - value < Decimal("0.01")
    if expected != "99999998990000.00":
        assert str(value) == expected


def test_potential_profit_is_winnings_minus_stake_and_the_inputs_are_checked():
    assert potential_profit(Decimal("3.50"), Decimal("1000")) == Decimal("2500.00")
    assert potential_profit(Decimal("2.33"), Decimal("1.99")) == Decimal("2.64")
    for odds, stake in (("0", "10"), ("-1", "10"), ("2", "0"), ("2", "-5"), ("NaN", "10"), ("2", "Infinity")):
        with pytest.raises(ValueError):
            potential_winnings(Decimal(odds), Decimal(stake))


def test_money_is_formatted_with_thousands_separators():
    assert format_money(Decimal("3500")) == "3,500.00" and format_money(Decimal("1234567.5")) == "1,234,567.50"
    assert format_money(Decimal("0.5")) == "0.50"


# ---------------------------------------------------------------- the summary text

def test_the_summary_shows_winnings_with_the_working_profit_and_the_estimate_note(env):
    session = env.to_summary("1000")
    text = summary_screen(session, dry_run=False).text
    assert "Potential winnings: NGN 2,410.00 (total odds 2.41 x stake 1,000.00; includes your stake back)" in text
    assert "Profit if it wins: NGN 1,410.00" in text
    assert "estimate" in text and "bonuses or limits" in text
    assert "Stake: NGN 1,000.00" in text


def test_the_winnings_use_the_exact_total_odds_the_token_is_bound_to(env):
    session = env.to_summary("500")
    token = env.press_final().token
    assert token.total_odds == session.summary.total_odds == session.loaded.total_odds
    expected = f"NGN {format_money(potential_winnings(token.total_odds, token.stake))}"
    assert expected in summary_screen(session, dry_run=False).text


def test_live_mode_banner_is_first_plain_text_and_dry_run_has_its_own(env):
    session = env.to_summary()
    live = summary_screen(session, dry_run=False).text
    dry = summary_screen(session, dry_run=True).text
    assert live.splitlines()[0] == LIVE_BANNER == "*** LIVE: this will place a REAL bet on SportyBet ***"
    assert dry.splitlines()[0] == DRY_BANNER == "DRY RUN: nothing will be placed"
    assert "LIVE" not in dry and "DRY RUN: nothing" not in live
    for text in (live, dry):
        assert "`" not in text


def test_a_very_long_summary_never_loses_the_winnings(tmp_path):
    evals, listing = [], []
    for i in range(120):
        evals.append(pick_evaluation(f"Home{i}", f"Away{i}", "Over/Under 2.5", "Over 2.5", "1.05", hour=14))
        listing.append(make_fixture(f"Home{i}", f"Away{i}", [totals_offer("2.5", "1.05", "9.00")], 14))
    e = Env(tmp_path, evals=evals, listing=listing, dry_run=False)
    try:
        text = summary_screen(e.to_summary("100"), dry_run=False).text
        assert len(text) <= 4096 and "Potential winnings: NGN" in text and "Profit if it wins" in text and "more" in text
    finally:
        e.close()


def test_the_typed_stake_gives_the_new_summary_with_winnings_at_once(env):
    env.to_stake()
    reply = handle_stake_text("2,000", OWNER, env.manager, dry_run=False)
    assert "Potential winnings: NGN 4,820.00" in reply.screen.text and "Stake: NGN 2,000.00" in reply.screen.text


# ---------------------------------------------------------------- Change amount

def press(holder, data, dry_run=False):
    return handle_callback(data, OWNER, holder.manager, dry_run=dry_run)


def change_button(session, dry_run=False):
    (_, row2) = summary_screen(session, dry_run=dry_run).buttons
    return row2[0]


def test_the_change_amount_button_is_versioned_and_fits_64_bytes(env):
    session = env.to_summary("500")
    label, data = change_button(session)
    assert label == "Change amount" and len(data.encode()) <= 64
    assert decode(data) == (SID, session.state_version, Action.CHANGE, "")
    assert len({a.value for a in Action}) == len(list(Action))


def test_change_amount_round_trip_new_stake_new_summary_new_token_old_confirm_dead(env):
    first = env.to_summary("500")
    old_confirm = summary_screen(first, dry_run=False).buttons[0][0][1]
    old_hash, old_summary_id = first.summary.digest, first.summary.summary_id
    calls_before = list(env.bm.calls)
    reply = press(env, change_button(first)[1])
    session = env.session
    assert reply.answer == "Change amount." and reply.job is None          # no reload, no job
    assert session.state is State.AWAITING_STAKE and session.state_version == first.state_version + 1
    assert session.loaded == first.loaded and session.odds_shown and session.summary is None and session.token_hash == ""
    assert "Current stake: NGN 500.00" in reply.screen.text and "How much do you want to stake" in reply.screen.text
    assert "Suggestion only" in reply.screen.text and reply.screen.buttons == ()
    assert reply.screen.text == stake_prompt(session).text
    # the old confirm button is dead: stale, nothing minted, nothing placed
    stale = press(env, old_confirm)
    assert stale.job is None and stale.answer.startswith("This list changed")
    assert env.session.state is State.AWAITING_STAKE
    # a new valid amount gives a NEW summary and a NEW token
    new_reply = handle_stake_text("1,500.50", OWNER, env.manager, dry_run=False)
    second = env.session
    assert second.state is State.AWAITING_FINAL_CONFIRM and second.stake == Decimal("1500.50")
    assert "Stake: NGN 1,500.50" in new_reply.screen.text and "Potential winnings: NGN 3,616.20" in new_reply.screen.text
    assert second.summary.digest != old_hash and second.summary.summary_id != old_summary_id
    assert press(env, old_confirm).job is None and env.session.state is State.AWAITING_FINAL_CONFIRM
    final = press(env, new_reply.screen.buttons[0][0][1])
    assert final.job.kind == "place" and final.job.token.stake == Decimal("1500.50")
    assert final.job.token.token_hash == second.summary.digest != old_hash and env.session.state is State.PLACING
    assert env.bm.calls == calls_before and env.bm.clicks == 0              # no bookmaker call at all so far


def test_the_token_hash_changes_when_the_stake_changes_and_is_bound_to_the_new_stake(env):
    env.to_summary("500")
    first_hash = env.session.summary.digest
    press(env, change_button(env.session)[1])
    handle_stake_text("501", OWNER, env.manager, dry_run=False)
    token = env.press_final().token
    assert token.token_hash != first_hash and token.stake == Decimal("501") and token.matches_contents()


def test_an_unchanged_amount_still_gets_a_new_summary_and_never_the_old_button(env):
    first = env.to_summary("500")
    press(env, change_button(first)[1])
    old = first.summary.summary_id
    handle_stake_text("500", OWNER, env.manager, dry_run=False)
    assert env.session.state_version > first.state_version + 1
    assert env.manager.act(OWNER, SID, first.state_version, Action.FINAL, old).status is not Status.APPLIED
    assert env.session.state is State.AWAITING_FINAL_CONFIRM


@pytest.mark.parametrize("bad", ["abc", "0", "-5", "10.123", "1e3", "", "100001"])
def test_the_new_amount_has_the_same_validation_and_max_stake_limit(env, bad):
    first = env.to_summary("500")
    press(env, change_button(first)[1])
    version = env.session.state_version
    reply = handle_stake_text(bad, OWNER, env.manager, dry_run=False)
    assert reply is not None and reply.screen is None and reply.answer
    if bad == "100001":
        assert "limit" in reply.answer
    assert env.session.state is State.AWAITING_STAKE and env.session.state_version == version and env.session.summary is None
    assert handle_stake_text("100000", OWNER, env.manager, dry_run=False).screen is not None   # the limit itself is fine
    assert env.session.stake == DEFAULT_MAX_STAKE


def test_a_repeat_tap_is_idempotent_and_a_stale_tap_changes_nothing(env):
    first = env.to_summary("500")
    data = change_button(first)[1]
    assert press(env, data).answer == "Change amount."
    version = env.session.state_version
    again = press(env, data)
    assert again.answer == "Already done." and again.screen is None and env.session.state_version == version
    handle_stake_text("700", OWNER, env.manager, dry_run=False)
    version = env.session.state_version
    late = press(env, data)                                     # the old button after a new summary exists
    assert late.answer.startswith("This list changed") and env.session.state_version == version
    assert env.session.state is State.AWAITING_FINAL_CONFIRM and env.session.stake == Decimal("700")


def test_a_forged_change_button_before_the_summary_does_nothing(env):
    env.to_stake()
    version = env.session.state_version
    reply = press(env, encode(SID, version, Action.CHANGE))
    assert env.session.state is State.AWAITING_STAKE and env.session.state_version == version
    assert reply.answer.startswith("This list changed")


def test_the_button_does_not_work_once_placing_has_started_or_later(env):
    session = env.to_summary("500")
    data = change_button(session)[1]
    env.press_final()
    assert env.session.state is State.PLACING
    version = env.session.state_version
    reply = press(env, data)
    assert env.session.state is State.PLACING and env.session.state_version == version and reply.job is None
    forged = encode(SID, version, Action.CHANGE)                # even a button made for the PLACING version
    assert press(env, forged).job is None and env.session.state is State.PLACING and env.session.stake == Decimal("500")
    env.manager.finish(SID, State.DONE, "done")
    assert press(env, encode(SID, env.session.state_version, Action.CHANGE)).job is None
    assert env.session.state is State.DONE


def test_a_summary_after_the_odds_moved_still_offers_the_button(env):
    session = env.to_summary("500")
    env.press_final()
    env.manager.odds_changed(SID, session.loaded)
    assert env.session.state is State.AWAITING_FINAL_CONFIRM and change_button(env.session)[0] == "Change amount"


def test_an_override_does_not_survive_a_change_of_amount(env):
    session = env.to_summary("500")
    press(env, change_button(session)[1])
    assert env.session.overrides == () and env.session.override_summary == ""


# ---------------------------------------------------------------- persistence and recovery

def test_the_new_transitions_round_trip_through_the_snapshot(env):
    first = env.to_summary("500")
    press(env, change_button(first)[1])
    changing = env.session
    assert changing.state is State.AWAITING_STAKE and changing.stake == Decimal("500")
    assert from_json(to_json(changing)) == changing and to_json(from_json(to_json(changing))) == to_json(changing)
    assert changing.last == (first.state_version, Action.CHANGE, "")
    handle_stake_text("800", OWNER, env.manager, dry_run=False)
    assert from_json(to_json(env.session)) == env.session


@pytest.mark.parametrize("stop", ["changing", "second_summary"])
def test_a_restart_on_the_stake_step_or_the_new_summary_reloads_and_asks_again(tmp_path, stop):
    r = Rig(tmp_path)
    first = r.to_summary("500")
    press(r, summary_screen(first, dry_run=False).buttons[1][0][1])
    if stop == "second_summary":
        handle_stake_text("900", OWNER, r.manager, dry_run=False)
    before = r.session
    r.bm.calls.clear()
    (report,) = r.restart()
    assert report.kind == KIND_RELOAD and report.reload and r.session.state is State.LOADING_SLIP
    assert r.session.stake is None and r.session.summary is None and r.session.loaded is None and not r.session.odds_shown
    assert r.session.token_hash == "" and r.session.state_version == before.state_version + 1 and r.bm.calls == []
    assert press(r, encode(SID, before.state_version, Action.CHANGE)).job is None
    assert r.load().kind == KIND_LOADED and r.session.state is State.AWAITING_STAKE and r.session.summary is None
    r.close()
