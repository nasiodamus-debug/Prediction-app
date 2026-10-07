"""Stories 5.1 to 5.4: loading, missing games and options, odds and stake, final confirmation and placement.

Everything runs against a FakeBookmaker (the Bookmaker port) and a real SQLite store in a temp folder."""
import asyncio
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal

import pytest

from fakes_bookmaker import (KICKOFF, NOW, FakeBookmaker, make_fixture, pick_evaluation, totals_offer, two_pick_setup,
                             xg_inputs)
from predictions.core.betting import Availability, leg_from_evaluation
from predictions.core.model import PROBABILITY_FLOOR
from predictions.core.placement import PlacementToken
from predictions.core.ports import BookmakerError
from predictions.core.slip import make_pick_id
from predictions.core.types import Owner
from predictions.flow.loading import (KIND_FAILED, KIND_ISSUES, KIND_LOADED, KIND_STALE, run_load)
from predictions.flow.placement import (OUTCOME_DRY_RUN, OUTCOME_FAILED, OUTCOME_ODDS_CHANGED, OUTCOME_PLACED,
                                        OUTCOME_UNKNOWN, run_placement)
from predictions.flow.session import (ISSUE_GAME, ISSUE_OPTION, Action, Issue, SessionManager, State, Status)
from predictions.flow.store import (P_DRY_RUN, P_FAILED, P_PLACED, P_PLACING, P_UNKNOWN, P_VOIDED, EvaluationStore)
from predictions.telegram import slate
from predictions.telegram.slate import (Job, OddsNotShown, encode, handle_callback, handle_stake_text, issues_screen,
                                        odds_screen, placement_text, stake_prompt, summary_screen)

OWNER = Owner(111222333)
SID = "abcd1234"


def run(coro):
    return asyncio.run(coro)


class Env:
    def __init__(self, tmp_path, dry_run=True, evals=None, listing=None):
        default_evals, default_listing = two_pick_setup()
        self.evals = evals if evals is not None else default_evals
        self.listing = listing if listing is not None else default_listing
        self.manager = SessionManager(clock=lambda: self.now)
        self.now = NOW
        self.manager.start(self.evals, SID)
        self.bm = FakeBookmaker(self.listing, dry_run)
        self.store = EvaluationStore.open(tmp_path / "e5.db", str)

    @property
    def session(self):
        return self.manager.active

    def confirm(self):
        assert self.manager.act(OWNER, SID, self.session.state_version, Action.DONE).status is Status.APPLIED
        result = self.manager.act(OWNER, SID, self.session.state_version, Action.CONFIRM)
        assert result.status is Status.APPLIED and result.load_next
        return result.session

    def load(self):
        session = self.session
        return run(run_load(self.manager, self.bm, SID, session.state_version, str))

    def to_stake(self):
        self.confirm()
        out = self.load()
        assert out.kind == KIND_LOADED
        self.manager.mark_odds_shown(SID, self.session.state_version)
        return out

    def to_summary(self, stake="500"):
        self.to_stake()
        result = self.manager.submit_stake(OWNER, stake)
        assert result.status is Status.APPLIED
        return self.session

    def press_final(self, session=None):
        session = session or self.session
        return self.manager.act(OWNER, SID, session.state_version, Action.FINAL, session.summary.summary_id)

    def place(self, token):
        return run(run_placement(self.manager, self.store, self.bm, token, str))

    def close(self):
        self.store.close()


@pytest.fixture
def env(tmp_path):
    e = Env(tmp_path)
    yield e
    e.close()


@pytest.fixture
def live_env(tmp_path):
    e = Env(tmp_path, dry_run=False)
    yield e
    e.close()


# ================================================================ 5.1 confirm and load

def test_confirm_loads_exactly_the_confirmed_games_and_places_nothing(env):
    env.confirm()
    assert env.session.state is State.LOADING_SLIP
    out = env.load()
    assert out.kind == KIND_LOADED and env.session.state is State.AWAITING_STAKE
    assert [c[0] for c in env.bm.calls] == ["check", "load"]        # checked first, then loaded, never placed
    assert not any(c[0] == "place" for c in env.bm.calls) and env.bm.clicks == 0
    loaded = env.session.loaded
    assert len(loaded.legs) == 2 and loaded.total_odds == Decimal("2.41")   # 1.85 * 1.30 = 2.405 -> 2.41
    assert env.store.list_placements() == []


def test_a_failed_check_is_reported_sanitized_and_never_retried(env):
    env.bm.check_error = BookmakerError("could not reach C:\\secret\\path 08031234567")
    env.confirm()
    out = env.load()
    assert out.kind == KIND_FAILED and env.session.state is State.FAILED
    assert "Nothing was placed" in out.message and "nothing will be retried" in out.message
    assert [c[0] for c in env.bm.calls] == ["check"]      # one attempt only


def test_a_failed_load_ends_in_failed_with_no_retry_and_no_place(env):
    env.bm.load_error = BookmakerError("The betslip could not be emptied.")
    env.confirm()
    out = env.load()
    assert out.kind == KIND_FAILED and env.session.state is State.FAILED
    assert [c[0] for c in env.bm.calls] == ["check", "load"] and env.bm.clicks == 0
    assert env.session.loaded is None


def test_a_betslip_that_differs_from_the_confirmed_games_aborts_the_load(env):
    original_load = env.bm.load_slip

    async def wrong(legs):
        slip = await original_load(legs)
        return replace(slip, legs=slip.legs[:1], total_odds=slip.legs[0].odds)   # one game went missing

    env.bm.load_slip = wrong
    env.confirm()
    out = env.load()
    assert out.kind == KIND_FAILED and "not exactly the games you confirmed" in out.message
    assert env.session.state is State.FAILED and env.session.loaded is None


def test_loading_a_session_that_moved_on_does_nothing(env):
    env.confirm()
    stale = run(run_load(env.manager, env.bm, SID, 999, str))
    other = run(run_load(env.manager, env.bm, "zzzzzzzz", env.session.state_version, str))
    assert stale.kind == other.kind == KIND_STALE and env.bm.calls == []


def test_an_unsupported_kind_of_pick_is_offered_for_skipping_never_loaded(tmp_path):
    weird = pick_evaluation("Alpha", "Beta", "Corners 9.5", "Over 9.5")
    e = Env(tmp_path, evals=[weird])
    try:
        e.confirm()
        out = e.load()
        assert out.kind == KIND_ISSUES and out.session.issues[0].kind == ISSUE_GAME
        assert e.bm.calls == []
    finally:
        e.close()


# ================================================================ 5.2 missing game and option

def test_a_missing_game_asks_the_owner_to_skip_and_does_nothing_on_its_own(tmp_path):
    evals, listing = two_pick_setup()
    e = Env(tmp_path, evals=evals, listing=listing[:1])          # the second game is not on SportyBet
    try:
        e.confirm()
        out = e.load()
        assert out.kind == KIND_ISSUES and e.session.state is State.RESOLVING_AVAILABILITY
        (issue,) = e.session.issues
        assert issue.kind == ISSUE_GAME and issue.alternative is None
        assert len(e.session.slip) == 2                          # nothing was skipped by itself
        assert [c[0] for c in e.bm.calls] == ["check"]           # and nothing was loaded
        text = issues_screen(e.session).text
        assert "Gamma vs Delta" in text and "Please skip this game" in text
        labels = [b[0] for row in issues_screen(e.session).buttons for b in row]
        assert labels == ["Skip 1"]
    finally:
        e.close()


def test_a_game_that_started_after_the_request_is_treated_as_missing(tmp_path):
    evals, listing = two_pick_setup()
    listing[0] = make_fixture("Alpha", "Beta", [totals_offer("2.5", "1.85", "2.00")], 14, started=True)
    e = Env(tmp_path, evals=evals, listing=listing)
    try:
        e.confirm()
        assert e.load().kind == KIND_ISSUES and e.session.issues[0].kind == ISSUE_GAME
    finally:
        e.close()


def test_skipping_removes_the_game_and_the_load_resumes_with_the_rest(tmp_path):
    evals, listing = two_pick_setup()
    e = Env(tmp_path, evals=evals, listing=listing[:1])
    try:
        e.confirm()
        e.load()
        issue = e.session.issues[0]
        skipped = e.manager.act(OWNER, SID, e.session.state_version, Action.SKIP, issue.pick_id)
        assert skipped.status is Status.APPLIED and skipped.load_next
        assert skipped.session.state is State.LOADING_SLIP and len(skipped.session.slip) == 1
        assert skipped.session.slip.one_per_match
        assert e.load().kind == KIND_LOADED
        assert len(e.session.loaded.legs) == 1
    finally:
        e.close()


def test_skipping_every_game_ends_the_session_with_nothing_placed(tmp_path):
    e = Env(tmp_path, listing=[])
    try:
        e.confirm()
        e.load()
        for issue in list(e.session.issues):
            e.manager.act(OWNER, SID, e.session.state_version, Action.SKIP, issue.pick_id)
        assert e.session.state is State.DONE and "Nothing was placed" in e.session.message
        assert len(e.session.slip) == 0 and e.bm.clicks == 0
    finally:
        e.close()


def missing_option_env(tmp_path, offers):
    pick = pick_evaluation()          # Over 2.5, xG 2.5 / 1.5, a strong pick
    listing = [make_fixture("Alpha", "Beta", offers, 14)]
    return Env(tmp_path, evals=[pick], listing=listing)


def test_a_missing_option_offers_the_closest_one_that_clears_the_floor_with_its_estimate(tmp_path):
    e = missing_option_env(tmp_path, [totals_offer("1.5", "1.30", "3.20"), totals_offer("2", "1.60", "2.30"),
                                      totals_offer("3.5", "2.30", "1.60")])   # the nearer whole line 2.0 is never offered (AD-21)
    try:
        e.confirm()
        out = e.load()
        (issue,) = out.session.issues
        assert issue.kind == ISSUE_OPTION and issue.alternative.selection == "Over 1.5"
        assert issue.alternative.estimate.probability >= PROBABILITY_FLOOR
        screen = issues_screen(e.session)
        assert "Closest option that still clears the 70% floor: Over 1.5 goals @ 1.30" in screen.text and "success " in screen.text
        assert [b[0] for row in screen.buttons for b in row] == ["Accept 1", "Skip 1"]
        # nothing changed until the owner decides
        assert e.session.slip.picks[0].evaluation.selection == "Over 2.5"
    finally:
        e.close()


def test_accepting_the_closest_option_substitutes_it_and_keeps_one_pick_per_match(tmp_path):
    e = missing_option_env(tmp_path, [totals_offer("1.5", "1.30", "3.20"), totals_offer("2", "1.60", "2.30")])
    try:
        e.confirm()
        e.load()
        issue = e.session.issues[0]
        accepted = e.manager.act(OWNER, SID, e.session.state_version, Action.ACCEPT, issue.pick_id)
        assert accepted.status is Status.APPLIED and accepted.load_next
        slip = accepted.session.slip
        assert len(slip) == 1 and slip.one_per_match and slip.picks[0].evaluation.selection == "Over 1.5"
        assert "Closest option" in slip.picks[0].evaluation.rule and slip.picks[0].evaluation.original.startswith("Over 2.5")
        assert e.load().kind == KIND_LOADED
        assert e.session.loaded.legs[0].outcome_name == "Over 1.5"       # the substituted leg is what got loaded
    finally:
        e.close()


def test_no_option_clearing_the_floor_means_skip_only_and_a_below_floor_option_is_never_offered(tmp_path):
    e = missing_option_env(tmp_path, [totals_offer("3.5", "2.30", "1.60"), totals_offer("4.5", "4.50", "1.20")])
    try:
        e.confirm()
        e.load()
        (issue,) = e.session.issues
        assert issue.kind == ISSUE_OPTION and issue.alternative is None
        screen = issues_screen(e.session)
        assert "No available option clears the 70% floor. Please skip this game." in screen.text
        assert [b[0] for row in screen.buttons for b in row] == ["Skip 1"]
        refused = e.manager.act(OWNER, SID, e.session.state_version, Action.ACCEPT, issue.pick_id)
        assert refused.status is Status.BLOCKED and e.session.state is State.RESOLVING_AVAILABILITY
    finally:
        e.close()


def test_the_screen_refuses_to_show_a_below_floor_offer_even_if_one_were_handed_to_it(tmp_path):
    e = missing_option_env(tmp_path, [totals_offer("3.5", "2.30", "1.60")])
    try:
        e.confirm()
        e.load()
        weak = pick_evaluation(market="Over/Under 3.5", selection="Over 3.5", odds="2.30", prob=0.60)
        forged = replace(e.session, issues=(Issue(e.session.slip.picks[0].pick_id, ISSUE_OPTION, "", replace(weak, passed=True)),))
        screen = issues_screen(forged)
        assert "Accept" not in [b[0] for row in screen.buttons for b in row][0] and "Over 3.5" not in screen.text
    finally:
        e.close()


def test_resolution_buttons_are_versioned_and_double_taps_do_nothing_extra(tmp_path):
    evals, listing = two_pick_setup()
    e = Env(tmp_path, evals=evals, listing=[])
    try:
        e.confirm()
        e.load()
        first, second = e.session.issues
        v = e.session.state_version
        skip_first = encode(SID, v, Action.SKIP, first.pick_id)
        skip_second = encode(SID, v, Action.SKIP, second.pick_id)
        assert handle_callback(skip_first, OWNER, e.manager, dry_run=True).answer == "Skipped."
        assert handle_callback(skip_first, OWNER, e.manager, dry_run=True).answer == "Already done."   # a double tap
        assert handle_callback(skip_second, OWNER, e.manager, dry_run=True).answer.startswith("This list changed")  # stale
        assert len(e.session.slip) == 1
    finally:
        e.close()


# ================================================================ 5.3 odds, then stake

def test_odds_and_total_are_shown_and_both_odds_when_sportybet_differs(tmp_path):
    e = Env(tmp_path)
    try:
        leg = leg_from_evaluation(e.evals[0])
        e.bm.odds_override[leg.key] = Decimal("1.90")            # SportyBet says 1.90, evaluation used 1.85
        e.confirm()
        assert e.load().kind == KIND_LOADED
        text = odds_screen(e.session).text
        assert "1.90" in text and "evaluation used 1.85, SportyBet now shows 1.90" in text
        assert "Gamma v Delta" in text and "@ 1.30" in text and "evaluation used 1.30" not in text
        assert "Total odds: 2.47" in text and "No bet has been placed" in text   # 1.90 * 1.30 = 2.47
        assert "stake" not in text.lower()                                       # the odds message never asks for it
    finally:
        e.close()


def test_the_stake_is_never_asked_for_before_the_odds_were_shown(env):
    env.confirm()
    env.load()
    assert env.session.state is State.AWAITING_STAKE and not env.session.odds_shown
    with pytest.raises(OddsNotShown):
        stake_prompt(env.session)
    early = env.manager.submit_stake(OWNER, "500")
    assert early.status is Status.BLOCKED and "odds have not been shown" in early.notice
    assert env.session.stake is None and env.session.state is State.AWAITING_STAKE
    assert handle_stake_text("500", OWNER, env.manager, True).answer.startswith("The odds have not been shown")
    env.manager.mark_odds_shown(SID, env.session.state_version)
    text = stake_prompt(env.session).text
    assert "How much do you want to stake" in text


def test_no_stake_prompt_in_any_earlier_or_later_state(env):
    for state_setup in (lambda: None, lambda: env.confirm()):
        state_setup()
        with pytest.raises(OddsNotShown):
            stake_prompt(env.session)
    with pytest.raises(OddsNotShown):
        odds_screen(env.session)          # nothing loaded, so nothing to show


def test_the_prompt_shows_the_kelly_suggestion_in_units_and_says_the_stake_is_final(env):
    env.to_stake()
    text = stake_prompt(env.session).text
    assert "Suggestion only:" in text and "units" in text and "between 1.5 and 3.5" in text
    assert "own standard stake size" in text and "bankroll" in text and "final" in text
    import re
    units = Decimal(re.search(r"Suggestion only: ([\d.]+) units", text).group(1))
    assert Decimal("1.5") <= units <= Decimal("3.5")


@pytest.mark.parametrize("bad", ["", "abc", "0", "-10", "10.999", "1e3", "5,5", "10 000", "NaN", "500NGN"])
def test_an_invalid_stake_is_rejected_and_asked_again(env, bad):
    env.to_stake()
    version = env.session.state_version
    result = env.manager.submit_stake(OWNER, bad)
    assert result.status is Status.BLOCKED and "not a valid stake" in result.notice
    assert env.session.state is State.AWAITING_STAKE and env.session.state_version == version and env.session.stake is None
    reply = handle_stake_text(bad, OWNER, env.manager, True)
    assert reply.screen is None and "not a valid stake" in reply.answer


def test_a_valid_stake_is_parsed_once_into_a_decimal_and_leads_to_the_summary(env):
    env.to_stake()
    reply = handle_stake_text("1250.50", OWNER, env.manager, True)
    assert reply.screen is not None and env.session.stake == Decimal("1250.50") and isinstance(env.session.stake, Decimal)
    assert env.session.state is State.AWAITING_FINAL_CONFIRM and env.session.summary.stake == Decimal("1250.50")
    assert "Stake: NGN 1,250.50" in reply.screen.text


def test_typed_text_is_not_a_stake_when_no_stake_is_awaited(env):
    assert handle_stake_text("500", OWNER, env.manager, True) is None      # still reviewing the list
    env.confirm()
    assert handle_stake_text("500", OWNER, env.manager, True) is None      # loading


def test_the_entered_stake_is_final_the_bot_never_changes_it(env):
    env.to_summary("37")
    assert env.session.stake == Decimal("37") and env.session.summary.stake == Decimal("37")   # not 1.5 units, not rounded


# ================================================================ 5.4 final confirmation and placement

def test_the_summary_lists_games_selections_odds_total_and_stake_with_one_confirm_button(env):
    session = env.to_summary("500")
    screen = summary_screen(session, dry_run=True)
    text = screen.text
    for wanted in ("Alpha v Beta", "Over 2.5", "@ 1.85", "Gamma v Delta", "@ 1.30", "Total odds: 2.41", "Stake: NGN 500.00"):
        assert wanted in text
    assert "DRY RUN is ON" in text and "Nothing will be placed" in text
    confirm_row, change_row = screen.buttons      # the one confirm button, then Change amount on its own row
    assert len(confirm_row) == 1 and confirm_row[0][1] == encode(SID, session.state_version, Action.FINAL,
                                                                 session.summary.summary_id)
    assert change_row == (("Change amount", encode(SID, session.state_version, Action.CHANGE)),)
    assert "Potential winnings: NGN 1,205.00" in text      # 2.41 x 500
    assert "WILL place" in summary_screen(session, dry_run=False).text
    assert "WILL place" not in text


def test_confirming_mints_one_token_bound_to_the_exact_summary_and_moves_to_placing(env):
    session = env.to_summary()
    summary = session.summary
    result = env.press_final()
    assert result.status is Status.APPLIED and isinstance(result.token, PlacementToken)
    token = result.token
    assert (token.stake, token.total_odds, token.legs, token.expires_at) == (
        summary.stake, summary.total_odds, summary.legs, summary.expires_at)
    assert token.token_hash == summary.digest and env.session.state is State.PLACING
    assert env.session.token_hash == token.token_hash


def test_the_confirmation_is_single_use_a_second_tap_mints_nothing(env):
    session = env.to_summary()
    button_version = session.state_version
    first = env.press_final(session)
    again = env.manager.act(OWNER, SID, button_version, Action.FINAL, session.summary.summary_id)
    assert first.token is not None and again.status is Status.REPEAT and again.token is None
    later = env.manager.act(OWNER, SID, env.session.state_version, Action.FINAL, session.summary.summary_id)
    assert later.status is Status.STALE and later.token is None                # PLACING is not a confirmable state


def test_a_wrong_or_forged_summary_id_confirms_nothing(env):
    session = env.to_summary()
    result = env.manager.act(OWNER, SID, session.state_version, Action.FINAL, "0" * 12)
    assert result.status is Status.STALE and result.token is None and env.session.state is State.AWAITING_FINAL_CONFIRM


def test_a_changed_stake_voids_the_old_confirmation_and_issues_a_new_summary(env):
    old = env.to_summary("500")
    old_button = encode(SID, old.state_version, Action.FINAL, old.summary.summary_id)
    assert handle_stake_text("600", OWNER, env.manager, True).screen is not None
    new = env.session
    assert new.state_version == old.state_version + 1 and new.summary.stake == Decimal("600")
    assert new.summary.summary_id != old.summary.summary_id
    reply = handle_callback(old_button, OWNER, env.manager, dry_run=True)      # the old button
    assert reply.answer.startswith("This list changed") and reply.job is None and env.session.state is State.AWAITING_FINAL_CONFIRM


def test_an_expired_summary_is_replaced_never_confirmed(env):
    session = env.to_summary()
    env.now = env.now + timedelta(minutes=6)
    result = env.press_final(session)
    assert result.status is Status.EXPIRED and result.token is None
    assert env.session.state is State.AWAITING_FINAL_CONFIRM and env.session.state_version == session.state_version + 1
    assert env.session.summary.summary_id != session.summary.summary_id and env.session.summary.expires_at > env.now
    reply = handle_callback(encode(SID, session.state_version, Action.FINAL, session.summary.summary_id), OWNER,
                            env.manager, dry_run=True)
    assert reply.answer.startswith("This list changed") and reply.job is None


def test_the_final_button_hands_app_a_place_job_with_the_token(env):
    session = env.to_summary()
    reply = handle_callback(encode(SID, session.state_version, Action.FINAL, session.summary.summary_id), OWNER,
                            env.manager, dry_run=True)
    assert isinstance(reply.job, Job) and reply.job.kind == "place" and isinstance(reply.job.token, PlacementToken)
    assert "dry run" in reply.screen.text.lower()
    again = handle_callback(encode(SID, session.state_version, Action.FINAL, session.summary.summary_id), OWNER,
                            env.manager, dry_run=True)
    assert again.job is None and again.answer == "Already done."


# ---------------------------------------------------------------- dry run

def test_dry_run_reports_what_would_be_placed_and_never_clicks(env):
    env.to_summary("500")
    token = env.press_final().token
    out = env.place(token)
    assert out.kind == OUTCOME_DRY_RUN and env.bm.clicks == 0
    assert env.session.state is State.DONE and "DRY RUN" in out.message
    text = placement_text(out)
    assert "What would have been placed" in text and "Alpha" in text
    (row,) = env.store.list_placements()
    assert row["state"] == P_DRY_RUN and row["dry_run"] == 1 and env.store.list_placed_legs() == []


def test_dry_run_is_the_default_of_the_bookmaker_and_of_the_app():
    import app as bot
    assert bot.parse_dry_run(None) is True and bot.DRY_RUN is True


# ---------------------------------------------------------------- real placement

def test_with_dry_run_off_success_needs_a_bet_id_and_returns_the_booking_code(live_env):
    env = live_env
    env.to_summary("500")
    token = env.press_final().token
    out = env.place(token)
    assert out.kind == OUTCOME_PLACED and env.bm.clicks == 1 and env.session.state is State.DONE
    assert out.result.bet_id == "B1234567" and "Booking code: ABC123" in placement_text(out)
    (row,) = env.store.list_placements()
    assert row["state"] == P_PLACED and row["bet_id"] == "B1234567" and row["booking_code"] == "ABC123"
    keys = {(r["match_id"], r["market_id"], r["specifier"], r["side"]) for r in env.store.list_placed_legs()}
    assert keys == {leg.leg_key for leg in token.legs}


def test_a_success_without_a_bet_id_is_not_proof(live_env):
    env = live_env
    env.bm.place_mode = "no_proof"
    env.to_summary()
    out = env.place(env.press_final().token)
    assert out.kind == OUTCOME_UNKNOWN and env.session.state is State.PLACEMENT_UNKNOWN
    (row,) = env.store.list_placements()
    assert row["state"] == P_UNKNOWN and env.store.list_placed_legs() == []


def test_placing_is_persisted_before_the_bookmaker_is_asked_to_place(live_env):
    env = live_env
    seen = {}
    real = env.bm.place

    async def spy(token):
        seen["row"] = env.store.get_placement(token.token_hash)
        return await real(token)

    env.bm.place = spy
    env.to_summary()
    token = env.press_final().token
    env.place(token)
    row = seen["row"]
    assert row is not None and row["state"] == P_PLACING and row["stake"] == "500" and row["session_id"] == SID
    assert row["total_odds"] == "2.41" and row["dry_run"] == 0 and row["expires_at_utc"] and row["created_at_utc"]
    import json
    legs = json.loads(row["legs"])
    assert len(legs) == 2 and legs[0]["odds"] and legs[0]["leg_key"]


def test_if_placing_cannot_be_persisted_nothing_is_clicked(live_env):
    env = live_env
    env.to_summary()
    token = env.press_final().token
    env.store.record_placing = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full"))
    out = env.place(token)
    assert out.kind == OUTCOME_FAILED and "No bet was placed" in out.message and env.bm.clicks == 0
    assert not any(c[0] == "place" for c in env.bm.calls) and env.session.state is State.FAILED


def test_without_an_open_database_nothing_is_clicked(live_env):
    env = live_env
    env.to_summary()
    token = env.press_final().token
    out = run(run_placement(env.manager, None, env.bm, token, str))
    assert out.kind == OUTCOME_FAILED and env.bm.clicks == 0 and not any(c[0] == "place" for c in env.bm.calls)


def test_the_same_token_recorded_twice_is_refused_by_the_database(live_env):
    env = live_env
    env.to_summary()
    token = env.press_final().token
    env.store.record_placing(token, False)
    out = env.place(token)                                 # its row already exists: the unique hash stops it
    assert out.kind == OUTCOME_FAILED and env.bm.clicks == 0 and not any(c[0] == "place" for c in env.bm.calls)


def test_a_failure_before_the_click_means_no_bet_and_no_retry(live_env):
    env = live_env
    env.bm.place_mode = "error"
    env.to_summary()
    out = env.place(env.press_final().token)
    assert out.kind == OUTCOME_FAILED and env.bm.clicks == 0 and env.session.state is State.FAILED
    assert "No bet was placed" in out.message and "Nothing will be retried" in out.message
    assert [c[0] for c in env.bm.calls].count("place") == 1
    assert env.store.list_placements()[0]["state"] == P_FAILED


def test_a_betslip_that_no_longer_matches_aborts_with_no_bet(live_env):
    env = live_env
    env.bm.place_mode = "mismatch"
    env.to_summary()
    out = env.place(env.press_final().token)
    assert out.kind == OUTCOME_FAILED and env.bm.clicks == 0 and "no longer matches" in out.message
    assert env.store.list_placed_legs() == []


def test_an_error_after_the_click_is_unknown_and_is_never_placed_again(live_env):
    env = live_env
    env.bm.place_mode = "unknown"
    env.to_summary()
    token = env.press_final().token
    out = env.place(token)
    assert out.kind == OUTCOME_UNKNOWN and env.session.state is State.PLACEMENT_UNKNOWN
    assert "UNKNOWN" in out.message and "will not place again" in out.message and env.bm.clicks == 1
    assert env.store.list_placements()[0]["state"] == P_UNKNOWN
    again = env.manager.act(OWNER, SID, env.session.state_version, Action.FINAL, token.token_hash[:12])
    assert again.status is Status.STALE and again.token is None            # the session cannot confirm again
    assert env.bm.clicks == 1


def test_an_unexpected_exception_from_the_bookmaker_is_unknown_not_no_bet(live_env):
    env = live_env
    env.bm.place_mode = "unexpected"
    env.to_summary()
    out = env.place(env.press_final().token)
    assert out.kind == OUTCOME_UNKNOWN and env.session.state is State.PLACEMENT_UNKNOWN and "UNKNOWN" in out.message


# ---------------------------------------------------------------- odds move

def test_changed_odds_void_the_confirmation_and_a_new_summary_needs_a_new_confirmation(live_env):
    env = live_env
    env.bm.place_mode = "odds_changed"
    old = env.to_summary("500")
    old_token = env.press_final(old).token
    out = env.place(old_token)
    assert out.kind == OUTCOME_ODDS_CHANGED and env.bm.clicks == 0
    session = env.session
    assert session.state is State.AWAITING_FINAL_CONFIRM and session.token_hash == ""
    assert session.summary.summary_id != old.summary.summary_id and session.summary.total_odds != old.summary.total_odds
    assert session.summary.stake == Decimal("500") and session.summary.legs[0].odds == Decimal("1.95")
    assert env.store.list_placements()[0]["state"] == P_VOIDED
    # the old button and the old token both stopped working
    stale = handle_callback(encode(SID, old.state_version, Action.FINAL, old.summary.summary_id), OWNER, env.manager, dry_run=False)
    assert stale.job is None and stale.answer.startswith("This list changed")
    env.bm.place_mode = "ok"
    replay = env.place(old_token)                                        # the old token cannot place either
    assert replay.kind == OUTCOME_FAILED and "no longer valid" in replay.message and env.bm.clicks == 0
    assert env.session.state is State.AWAITING_FINAL_CONFIRM             # and it did not disturb the new summary
    assert [c[0] for c in env.bm.calls].count("place") == 1


def test_a_new_confirmation_after_an_odds_move_works_and_uses_a_different_token(live_env):
    env = live_env
    env.bm.place_mode = "odds_changed"
    old = env.to_summary("500")
    old_token = env.press_final(old).token
    env.place(old_token)
    env.bm.place_mode = "ok"
    new_token = env.press_final().token
    assert new_token is not None and new_token.token_hash != old_token.token_hash
    assert new_token.total_odds != old_token.total_odds
    out = env.place(new_token)
    assert out.kind == OUTCOME_PLACED and env.bm.clicks == 1
    states = [r["state"] for r in env.store.list_placements()]
    assert states == [P_VOIDED, P_PLACED]


def test_the_odds_changed_message_is_followed_by_the_new_summary_text(live_env):
    env = live_env
    env.bm.place_mode = "odds_changed"
    env.to_summary()
    out = env.place(env.press_final().token)
    assert "odds changed" in out.message and "confirm again" in out.message
    assert "Total odds: 2.73" in summary_screen(env.session, dry_run=False).text     # 1.95 * 1.40
