"""The games list: every game with its valid plays, a Keep/Delete pair PER PLAY (1, or 1a/1b/1c), at most one kept play
per game (the advised one by default), and the Final confirm -> review -> Confirm games / Back to adjust path."""
import asyncio
import json
import re

import pytest

from predictions.core.slip import advised_order, build_games, make_pick_id
from predictions.flow.session import Action, SessionManager, State, Status
from predictions.flow.snapshot import SnapshotError, from_json, to_json
from predictions.telegram import slate
from predictions.telegram.slate import CALLBACK_LIMIT, PAGE_SIZE, encode, handle_callback
from test_epic5_recovery import Rig, rig  # noqa: F401  (rig is a fixture)
from test_final_list import NOW, OWNER, ev, two_matches, with_alternatives

SID = "abcd1234"


def started(evals=None, origin="matchday"):
    manager = SessionManager(clock=lambda: NOW)
    return manager, manager.start(evals or two_matches(), SID, origin)


def press(manager, data):
    return handle_callback(data, OWNER, manager)


def button(manager, label, page=0):
    session = manager.active
    screen = slate.final_screen(session) if session.state is State.REVIEWING_FINAL else slate.list_screen(session, page)
    return next(d for row in screen.buttons for t, d in row if t == label)


def labels(manager, page=0):
    return [t for row in slate.list_screen(manager.active, page).buttons for t, _ in row]


def pid(evaluation):
    return make_pick_id(evaluation)


def act(manager, action, pick_id="", version=None):
    return manager.act(OWNER, SID, version or manager.active.state_version, action, pick_id)


def options_game(count=3):
    """One game (Alpha vs Beta) with `count` valid options of falling probability, plus a single-option game."""
    plays = [ev("Alpha", "Beta", f"Over {i}.5", 0.90 - i * 0.05, "1.80") for i in range(count)]
    return plays + [ev("Gamma", "Delta", "Over 1.5", 0.88, "1.30", hour=16)]


# ---------- the list: a game per line, state visible, Keep/Delete beside it ----------

def test_the_list_is_one_numbered_line_per_game_with_a_state_mark_and_a_button_row_each():
    manager, _ = started()
    screen = slate.list_screen(manager.active)
    assert screen.text.startswith("Final list: Kept 2 of 2 game(s).")
    assert f"1) {slate.MARK_KEPT} 10 Oct 15:00 Alpha vs Beta\n   {slate.MARK_KEPT} Over 2.5 @ 1.85 - 72.0%" in screen.text
    assert f"2) {slate.MARK_KEPT} 10 Oct 17:00 Gamma vs Delta\n   {slate.MARK_KEPT} Over 1.5 @ 1.30 - 88.0%" in screen.text
    assert [[t for t, _ in row] for row in screen.buttons] == [["1 Keep", "1 Delete"], ["2 Keep", "2 Delete"],
                                                               ["Final confirm"]]
    for noise in ("EV", "inputs", "source", "success", "skipped", "not covered", "rejected"):
        assert noise not in screen.text


def test_delete_then_keep_toggles_a_game_and_bumps_the_version_each_time():
    manager, session = started()
    a, b = two_matches()
    r1 = act(manager, Action.DELETE, pid(a))
    assert r1.status is Status.APPLIED and r1.session.state_version == 2
    assert [p.pick_id for p in r1.session.slip.picks] == [pid(b)] and r1.session.removed_games == (str(a.match_id),)
    text = slate.list_screen(r1.session).text
    assert f"1) {slate.MARK_REMOVED} 10 Oct 15:00 Alpha vs Beta (removed)" in text and "Kept 1 of 2 game(s)" in text
    assert labels(manager) == ["1 Keep", "1 Delete", "2 Keep", "2 Delete", "Final confirm"]   # same pairs, state in the text
    r2 = act(manager, Action.RESTORE, pid(a))
    assert r2.status is Status.APPLIED and r2.session.state_version == 3 and r2.session.removed_games == ()
    assert [p.pick_id for p in r2.session.slip.picks] == [pid(a), pid(b)]
    assert f"1) {slate.MARK_KEPT} " in slate.list_screen(r2.session).text and "Kept 2 of 2 game(s)" in slate.list_screen(r2.session).text


def test_the_button_replies_name_the_action_and_stay_on_the_same_page():
    evals = [ev(f"Home{i:02d}", f"Away{i:02d}", hour=8 + i % 12, prob=0.7 + i / 1000) for i in range(20)]
    manager, _ = started(evals)
    page2 = slate.list_screen(manager.active, 1)
    delete = next(d for row in page2.buttons for t, d in row if t.endswith("Delete"))
    reply = press(manager, delete)
    assert reply.answer == "Deleted." and "Page 2 of 3" in reply.screen.text and "(removed)" in reply.screen.text
    keep = next(d for row in reply.screen.buttons for t, d in row if t.endswith("Keep"))
    reply = press(manager, keep)
    assert reply.answer == "Kept." and "Page 2 of 3" in reply.screen.text and "(removed)" not in reply.screen.text


def test_open_is_gone_there_is_no_open_button_no_open_action_and_no_option_selector():
    assert "o" not in {a.value for a in Action} and "c" not in {a.value for a in Action}
    assert not hasattr(Action, "OPEN") and not hasattr(Action, "CHOOSE") and not hasattr(slate, "detail_screen")
    assert slate.decode("s:abcd1234:1:o:" + "a" * 12) is None and slate.decode("s:abcd1234:1:c:" + "a" * 12) is None
    manager, session = started(options_game(3))
    for screen in (slate.list_screen(session), slate.final_screen(session)):
        for row in screen.buttons:
            for label, data in row:
                assert "Open" not in label and label != "Done" and slate.decode(data) is not None
    assert "Open" not in slate.list_screen(session).text and "Press Done" not in slate.list_screen(session).text


# ---------- several valid options: listed together, advised first, exactly one chosen ----------

def test_a_single_play_game_is_numbered_1_with_one_row_of_keep_and_delete():
    manager, session = started([ev("Alpha", "Beta", "Over 1.5", 0.80)])
    screen = slate.list_screen(session)
    assert [[t for t, _ in row] for row in screen.buttons] == [["1 Keep", "1 Delete"], ["Final confirm"]]
    assert "1a" not in screen.text and "Kept 1 of 1 game(s)" in screen.text


def test_plays_are_lettered_under_their_game_advised_first_each_with_its_own_keep_delete_row():
    manager, session = started(options_game(3))
    text = slate.list_screen(session).text
    assert text.count("Alpha vs Beta") == 1
    assert f"   1a {slate.MARK_KEPT} Over 0.5 @ 1.80 - 90.0% (advised)" in text
    assert f"   1b {slate.MARK_REMOVED} Over 1.5 @ 1.80 - 85.0%" in text
    assert f"   1c {slate.MARK_REMOVED} Over 2.5 @ 1.80 - 80.0%" in text
    assert [[t for t, _ in row] for row in slate.list_screen(session).buttons] == [
        ["1a Keep", "1a Delete"], ["1b Keep", "1b Delete"], ["1c Keep", "1c Delete"], ["2 Keep", "2 Delete"],
        ["Final confirm"]]
    assert len(session.slip) == 2 and session.slip.one_per_match and "Kept 2 of 2 game(s)" in text


def test_the_final_confirm_button_is_one_full_width_row_at_the_very_bottom_of_every_page():
    manager, _ = started(many(PAGE_SIZE + 4))
    for page in range(slate.page_count(manager.active)):
        rows = slate.list_screen(manager.active, page).buttons
        assert [t for t, _ in rows[-1]] == ["Final confirm"] and len(rows[-1]) == 1
        assert slate.decode(rows[-1][0][1])[2] is Action.DONE
        assert all("Final confirm" != t for row in rows[:-1] for t, _ in row)


def test_the_advised_option_is_the_highest_estimate_then_the_highest_ev_then_the_text():
    low, high = ev("Alpha", "Beta", "Over 1.5", 0.70), ev("Alpha", "Beta", "Over 2.5", 0.91)
    assert advised_order([make_pick_of(low), make_pick_of(high)])[0].evaluation is high
    a, b = ev("Alpha", "Beta", "Over 2.5", 0.80, "1.90"), ev("Alpha", "Beta", "Over 1.5", 0.80, "1.50")
    from dataclasses import replace
    better = replace(b, ev=b.ev + 1)
    assert advised_order([make_pick_of(a), make_pick_of(better)])[0].evaluation is better       # equal estimate: EV
    tie = replace(b, ev=a.ev)
    assert advised_order([make_pick_of(a), make_pick_of(tie)])[0].evaluation.selection == "Over 1.5"   # then the text
    games = build_games([make_pick_of(low), make_pick_of(high)])
    assert games[0].chosen.evaluation is high and games[0].kept          # the advised one is chosen by default


def make_pick_of(evaluation):
    from predictions.core.slip import make_pick
    return make_pick(evaluation)


def test_keep_on_a_play_makes_it_the_kept_play_and_unkeeps_its_sibling():
    manager, session = started(options_game(3))
    plays = options_game(3)
    r = act(manager, Action.RESTORE, pid(plays[2]))
    assert r.status is Status.APPLIED and r.session.state_version == 2
    assert {p.pick_id for p in r.session.slip.picks} == {pid(plays[2]), pid(plays[3])}   # exactly one play per game
    text = slate.list_screen(r.session).text
    assert f"1c {slate.MARK_KEPT} Over 2.5" in text and f"1a {slate.MARK_REMOVED}" in text and "(advised)" in text
    assert text.count(slate.MARK_KEPT) == 4 and "Kept 2 of 2 game(s)" in text      # 2 game lines + 1 kept play per game
    assert act(manager, Action.RESTORE, pid(plays[2])).status is Status.REPEAT      # Keep on the kept play: no-op
    assert manager.active.state_version == 2


def test_delete_on_the_kept_play_takes_the_game_out_of_the_slip_and_keep_brings_it_back():
    manager, _ = started(options_game(3))
    plays = options_game(3)
    r = act(manager, Action.DELETE, pid(plays[0]))
    assert r.status is Status.APPLIED and r.session.state_version == 2
    assert [p.pick_id for p in r.session.slip.picks] == [pid(plays[3])]
    text = slate.list_screen(r.session).text
    assert f"1) {slate.MARK_REMOVED} 10 Oct 15:00 Alpha vs Beta (removed)" in text and "Kept 1 of 2 game(s)" in text
    assert f"1a {slate.MARK_KEPT}" not in text and text.count(slate.MARK_KEPT) == 2     # no kept play in game 1
    r = act(manager, Action.RESTORE, pid(plays[1]))                                    # any play can bring it back
    assert r.status is Status.APPLIED and r.session.state_version == 3
    assert {p.pick_id for p in r.session.slip.picks} == {pid(plays[1]), pid(plays[3])}
    assert "Kept 2 of 2 game(s)" in slate.list_screen(r.session).text and r.session.removed_games == ()


def test_delete_on_a_play_that_is_not_kept_and_keep_on_the_kept_play_change_nothing():
    manager, session = started(options_game(3))
    plays = options_game(3)
    for action, play in ((Action.DELETE, plays[1]), (Action.DELETE, plays[2]), (Action.RESTORE, plays[0]),
                         (Action.RESTORE, plays[3])):
        r = act(manager, action, pid(play))
        assert r.status is Status.REPEAT
        assert press(manager, encode(SID, 1, action, pid(play))).answer == "Already done."
    assert manager.active == session and manager.active.state_version == 1
    act(manager, Action.DELETE, pid(plays[0]))                                          # game 1 now has no kept play
    version = manager.active.state_version
    assert act(manager, Action.DELETE, pid(plays[0])).status is Status.REPEAT           # a removed game's play: no-op
    assert act(manager, Action.DELETE, pid(plays[1])).status is Status.REPEAT
    assert manager.active.state_version == version


def test_a_play_choice_survives_a_delete_and_a_keep_of_the_same_game():
    manager, session = started(options_game(3))
    plays = options_game(3)
    act(manager, Action.RESTORE, pid(plays[1]))
    act(manager, Action.DELETE, pid(plays[1]))
    assert pid(plays[1]) not in manager.active.slip and len(manager.active.slip) == 1
    act(manager, Action.RESTORE, pid(plays[0]))
    assert pid(plays[0]) in manager.active.slip and pid(plays[1]) not in manager.active.slip


def test_keeping_a_play_of_a_removed_game_puts_the_game_back_in_the_slip():
    manager, _ = started(options_game(2))
    plays = options_game(2)
    act(manager, Action.DELETE, pid(plays[0]))
    r = act(manager, Action.RESTORE, pid(plays[1]))
    assert r.status is Status.APPLIED and pid(plays[1]) in r.session.slip and r.session.removed_games == ()


def test_the_slip_is_structurally_one_play_per_game_whatever_is_done():
    manager, _ = started(options_game(4))
    plays = options_game(4)
    steps = [(Action.RESTORE, 3), (Action.RESTORE, 1), (Action.DELETE, 0), (Action.DELETE, 1), (Action.RESTORE, 2),
             (Action.RESTORE, 0), (Action.DELETE, 0), (Action.RESTORE, 3), (Action.RESTORE, 3)]
    for action, index in steps:
        act(manager, action, pid(plays[index]))
        assert manager.active.slip.one_per_match
        assert len({p.key for p in manager.active.slip.picks}) == len(manager.active.slip)
        game_one = [p for p in manager.active.slip.picks if p.key == plays[0].match_id.__str__()]
        assert len(game_one) <= 1


def test_an_unknown_pick_for_any_game_action_is_refused_and_changes_nothing():
    manager, _ = started()
    for action in (Action.DELETE, Action.RESTORE):
        assert act(manager, action, "0" * 12).status is Status.UNKNOWN_PICK
    assert manager.active.state_version == 1


# ---------- Final confirm, the review (final edit) screen, Back to adjust ----------

def test_final_confirm_shows_only_the_kept_games_and_their_chosen_plays_with_two_buttons():
    manager, _ = started(options_game(3))
    plays = options_game(3)
    act(manager, Action.RESTORE, pid(plays[1]))
    r = press(manager, button(manager, "Final confirm"))
    assert r.answer == "Final confirm." and manager.active.state is State.REVIEWING_FINAL and manager.active.state_version == 3
    text = r.screen.text
    assert text.startswith("Final edit: 2 game(s) kept.")
    assert "1) 10 Oct 15:00 Alpha vs Beta: Over 1.5 @ 1.80 - 85.0%" in text and "2) 10 Oct 17:00 Gamma vs Delta" in text
    assert "Over 0.5" not in text and "Over 2.5" not in text               # only the chosen play
    assert [t for row in r.screen.buttons for t, _ in row] == ["Confirm games", "Back to adjust"]
    act(manager, Action.ADJUST)
    act(manager, Action.DELETE, pid(plays[3]))
    r = press(manager, button(manager, "Final confirm"))
    assert r.screen.text.startswith("Final edit: 1 game(s) kept, 1 removed.") and "Gamma" not in r.screen.text


def test_final_confirm_is_refused_with_zero_kept_games_and_works_again_after_a_keep():
    manager, _ = started()
    a, b = two_matches()
    act(manager, Action.DELETE, pid(a))
    act(manager, Action.DELETE, pid(b))
    version = manager.active.state_version
    reply = press(manager, button(manager, "Final confirm"))
    assert reply.alert and "no play is kept" in reply.answer.lower() and manager.active.state is State.REVIEWING_LIST
    assert manager.active.state_version == version and "Kept 0 of 2 game(s)" in reply.screen.text
    assert "No play is kept" in reply.screen.text and [t for t, _ in reply.screen.buttons[-1]] == ["Final confirm"]
    act(manager, Action.RESTORE, pid(b))
    assert act(manager, Action.DONE).status is Status.APPLIED


def test_confirm_is_only_possible_from_the_final_screen():
    manager, _ = started()
    blocked = act(manager, Action.CONFIRM)
    assert blocked.status is Status.BLOCKED and manager.active.state is State.REVIEWING_LIST
    act(manager, Action.DONE)
    r = act(manager, Action.CONFIRM)
    assert r.status is Status.APPLIED and r.load_next and r.session.state is State.LOADING_SLIP
    assert r.session.state_version == 3


def test_back_to_adjust_returns_to_the_editable_list_with_every_decision_kept():
    manager, _ = started(options_game(3))
    plays = options_game(3)
    act(manager, Action.RESTORE, pid(plays[2]))
    act(manager, Action.DELETE, pid(plays[3]))
    before = manager.active
    act(manager, Action.DONE)
    r = press(manager, button(manager, "Back to adjust"))
    assert r.answer == "Back to adjust." and manager.active.state is State.REVIEWING_LIST
    after = manager.active
    assert after.state_version == before.state_version + 2
    assert (after.slip, after.chosen, after.removed_games, after.menu) == (before.slip, before.chosen,
                                                                           before.removed_games, before.menu)
    assert f"1c {slate.MARK_KEPT} Over 2.5" in r.screen.text and "(removed)" in r.screen.text
    assert act(manager, Action.RESTORE, pid(plays[3])).status is Status.APPLIED       # and it can be edited again


def test_every_transition_bumps_the_version_and_voids_the_previous_screens_buttons():
    manager, _ = started()
    a, _b = two_matches()
    list_done = button(manager, "Final confirm")
    list_delete = button(manager, "1 Delete")
    press(manager, list_done)                                              # -> final edit, version 2
    assert manager.active.state_version == 2
    assert press(manager, list_delete).answer.startswith("This list changed")      # the list's buttons are dead
    final_confirm, final_back = button(manager, "Confirm games"), button(manager, "Back to adjust")
    press(manager, final_back)                                              # -> list, version 3
    assert manager.active.state_version == 3
    assert press(manager, final_confirm).answer.startswith("This list changed")     # a change voided the final screen
    assert manager.active.state is State.REVIEWING_LIST
    press(manager, button(manager, "1 Delete"))                             # version 4
    press(manager, button(manager, "Final confirm"))                       # version 5
    assert manager.active.state_version == 5
    assert press(manager, final_confirm).answer.startswith("This list changed")     # an old final screen never confirms


def test_stale_and_repeat_taps_on_every_action():
    manager, _ = started()
    a, b = two_matches()
    delete_a = encode(SID, 1, Action.DELETE, pid(a))
    assert press(manager, delete_a).answer == "Deleted."
    assert press(manager, delete_a).answer == "Already done."                      # exact repeat: no second effect
    assert manager.active.state_version == 2 and len(manager.active.slip) == 1
    keep_a = encode(SID, 2, Action.RESTORE, pid(a))
    other = encode(SID, 2, Action.DELETE, pid(b))
    assert press(manager, keep_a).answer == "Kept."
    assert press(manager, keep_a).answer == "Already done."
    assert press(manager, other).answer.startswith("This list changed")             # made for version 2
    assert press(manager, delete_a).answer.startswith("This list changed")
    done = encode(SID, 3, Action.DONE)
    assert press(manager, done).answer == "Final confirm." and press(manager, done).answer == "Already done."
    adjust = encode(SID, 4, Action.ADJUST)
    assert press(manager, adjust).answer == "Back to adjust." and press(manager, adjust).answer == "Already done."
    assert manager.active.state_version == 5


def test_toggling_at_the_current_version_when_already_in_that_state_changes_nothing():
    manager, _ = started()
    a, _b = two_matches()
    assert act(manager, Action.RESTORE, pid(a)).status is Status.REPEAT
    act(manager, Action.DELETE, pid(a))
    version = manager.active.state_version
    assert act(manager, Action.DELETE, pid(a)).status is Status.REPEAT
    assert manager.active.state_version == version


def test_game_edits_are_refused_on_the_final_screen_and_after_loading():
    manager, _ = started()
    a, _b = two_matches()
    act(manager, Action.DONE)
    version = manager.active.state_version
    assert act(manager, Action.DELETE, pid(a)).status is Status.STALE and manager.active.state_version == version
    act(manager, Action.CONFIRM)
    assert act(manager, Action.RESTORE, pid(a)).status is Status.STALE


def test_a_final_screen_too_old_cannot_be_confirmed():
    from datetime import timedelta
    clock = {"now": NOW}
    manager = SessionManager(clock=lambda: clock["now"])
    manager.start(two_matches(), SID)
    act(manager, Action.DONE)
    clock["now"] = NOW + timedelta(hours=6, seconds=1)
    assert act(manager, Action.CONFIRM).status is Status.STALE and manager.active.state is State.REVIEWING_FINAL
    assert act(manager, Action.ADJUST).status is Status.APPLIED                  # going back is always allowed


def test_a_long_final_list_keeps_its_buttons_and_ends_with_and_more():
    long_name = "Wolverhampton Wanderers Football Club "
    evals = [ev(f"{long_name}{i:02d}", f"{long_name}Away {i:02d}", "Over 2.5", 0.72, "1.85") for i in range(60)]
    manager, _ = started(evals)
    act(manager, Action.DONE)
    screen = slate.final_screen(manager.active)
    assert len(screen.text) <= slate.MESSAGE_LIMIT and re.search(r"\nand \d+ more\n", screen.text)
    assert [t for row in screen.buttons for t, _ in row] == ["Confirm games", "Back to adjust"]


# ---------- the Slip used by loading, odds, stake and placement ----------

def test_the_slip_holds_only_the_chosen_play_of_each_kept_game():
    manager, _ = started(options_game(3))
    plays = options_game(3)
    act(manager, Action.RESTORE, pid(plays[1]))
    act(manager, Action.DELETE, pid(plays[3]))
    act(manager, Action.DONE)
    result = act(manager, Action.CONFIRM)
    assert [p.pick_id for p in result.session.slip.picks] == [pid(plays[1])]
    assert slate.confirmed_screen(result.session).text.count("Over") == 1


# ---------- snapshot and recovery ----------

def test_the_decisions_round_trip_through_the_snapshot_in_every_list_state():
    manager, _ = started(options_game(3))
    plays = options_game(3)
    act(manager, Action.RESTORE, pid(plays[2]))
    act(manager, Action.DELETE, pid(plays[3]))
    for state_action in (None, Action.DONE):
        if state_action:
            act(manager, state_action)
        session = manager.active
        text = to_json(session)
        assert text == to_json(session) and '"menu"' in text and '"chosen"' in text and '"removed_games"' in text
        back = from_json(text)
        assert back == session and back.slip == session.slip
        assert back.chosen == session.chosen and back.removed_games == session.removed_games
        assert [p.pick_id for p in back.menu] == [p.pick_id for p in session.menu]


def test_an_old_snapshot_without_the_new_fields_loads_as_all_kept_with_the_advised_option_chosen():
    manager, session = started(with_alternatives())
    data = json.loads(to_json(session))
    for key in ("menu", "chosen", "removed_games"):
        del data[key]
    data["picks"] = [{"pick_id": p.pick_id, "evaluation": json.loads(to_json(session))["menu"][i]["evaluation"]}
                     for i, p in enumerate(session.menu)]          # an old slip held every alternative
    back = from_json(json.dumps(data))
    assert back.removed_games == () and len(back.games()) == 2 and all(g.kept for g in back.games())
    assert len(back.slip) == 2 and back.slip.one_per_match
    assert back.games()[0].chosen_id == back.games()[0].advised.pick_id


def test_a_saved_pick_that_is_not_an_option_is_refused():
    manager, session = started()
    data = json.loads(to_json(session))
    data["menu"] = data["menu"][:1]
    with pytest.raises(SnapshotError):
        from_json(json.dumps(data))


def test_recovery_restores_the_same_decisions_and_the_buttons_still_work(rig):  # noqa: F811
    plays = list(rig.session.menu)
    rig.manager.act(OWNER, rig.sid, rig.session.state_version, Action.DELETE, plays[0].pick_id)
    version = rig.session.state_version
    (report,) = rig.restart()
    assert report.show_list and "1 of 2 games kept" in report.text
    restored = rig.manager.active
    assert restored.state is State.REVIEWING_LIST and restored.state_version == version
    assert [p.pick_id for p in restored.slip.picks] == [plays[1].pick_id] and len(restored.removed_games) == 1
    assert "(removed)" in slate.list_screen(restored).text
    keep = next(d for row in slate.list_screen(restored).buttons for t, d in row if t.endswith("Keep"))
    assert handle_callback(keep, OWNER, rig.manager).answer == "Kept." and len(rig.manager.active.slip) == 2


def test_a_session_saved_on_the_final_screen_returns_to_the_editable_list_after_a_restart(rig):  # noqa: F811
    first = rig.session.menu[0]
    rig.manager.act(OWNER, rig.sid, rig.session.state_version, Action.DELETE, first.pick_id)
    rig.manager.act(OWNER, rig.sid, rig.session.state_version, Action.DONE)
    saved_version = rig.session.state_version
    final_confirm = next(d for row in slate.final_screen(rig.session).buttons for t, d in row if t == "Confirm games")
    assert rig.session.state is State.REVIEWING_FINAL
    (report,) = rig.restart()
    restored = rig.manager.active
    assert report.show_list and restored.state is State.REVIEWING_LIST and restored.state_version == saved_version + 1
    assert len(restored.slip) == 1 and len(restored.removed_games) == 1            # decisions kept
    assert handle_callback(final_confirm, OWNER, rig.manager).answer.startswith("This list changed")   # old Confirm is dead
    assert rig.manager.active.state is State.REVIEWING_LIST


def test_a_restart_after_confirm_reloads_only_the_kept_chosen_plays(rig):  # noqa: F811
    first, second = [p.pick_id for p in rig.session.slip.picks]
    rig.manager.act(OWNER, rig.sid, rig.session.state_version, Action.DELETE, first)
    rig.confirm()
    (report,) = rig.restart()
    assert report.reload and [p.pick_id for p in rig.manager.active.slip.picks] == [second]


# ---------- Review-continue lists ----------

def test_a_review_list_has_the_same_games_toggles_and_final_screen_and_never_a_disqualified_selection():
    approved = two_matches()
    manager, _ = started(approved, origin="review:ABC123")
    text = slate.list_screen(manager.active).text
    assert text.startswith("Approved selections from Review ABC123: Kept 2 of 2 game(s).")
    press(manager, button(manager, "1 Delete"))
    assert "(removed)" in slate.list_screen(manager.active).text and "1 Keep" in labels(manager)
    bad = ev("Eta", "Theta", "Over 3.5", 0.40, passed=False)
    with pytest.raises(ValueError):
        manager.request_start(OWNER, approved + [bad], "review:ZZZ")
    with pytest.raises(ValueError):
        manager.start([bad], "ffff0000")
    assert act(manager, Action.RESTORE, pid(bad)).status is Status.UNKNOWN_PICK     # never addable, never restorable
    assert act(manager, Action.DELETE, pid(bad)).status is Status.UNKNOWN_PICK
    final = press(manager, button(manager, "Final confirm")).screen
    assert final.text.startswith("Final edit: 1 game(s) kept, 1 removed.") and "Eta" not in final.text


def test_two_review_selections_on_one_match_are_options_of_one_game():
    manager, _ = started(with_alternatives(), origin="review:ABC123")
    assert len(manager.active.games()) == 2 and len(manager.active.slip) == 2
    assert f"1a {slate.MARK_KEPT}" in slate.list_screen(manager.active).text and "1b" in labels(manager)[2]


# ---------- paging, message size, callback size ----------

def many(n, options=1, name="Home"):
    out = []
    for i in range(n):
        for k in range(options):
            out.append(ev(f"{name}{i:02d}", f"Away{i:02d}", f"Over {k}.5 goals in the match", 0.90 - k * 0.01,
                          hour=8 + (i % 12)))
    return out


def test_removed_games_stay_on_their_page_and_every_page_fits_telegram():
    evals = many(PAGE_SIZE * 2 + 3)
    manager, _ = started(evals)
    for e in evals[::2]:
        act(manager, Action.DELETE, pid(e))
    games = manager.active.games()
    assert len(games) == len(evals) and sum(not g.kept for g in games) == len(manager.active.removed_games)
    pages = slate.page_count(manager.active)
    assert pages == 3
    seen_removed = seen_kept = 0   # a removed single-play game shows two crosses (game line and play), a kept one two checks
    for page in range(pages):
        screen = slate.list_screen(manager.active, page)
        assert len(screen.text) <= 4000 and f"Page {page + 1} of 3" in screen.text
        seen_removed += screen.text.count(slate.MARK_REMOVED) // 2
        seen_kept += screen.text.count(slate.MARK_KEPT) // 2
        assert sum(t.endswith(" Keep") for row in screen.buttons for t, _ in row) <= PAGE_SIZE   # one pair per play (1 play each)
        assert sum(t.endswith(" Delete") for row in screen.buttons for t, _ in row) == sum(
            t.endswith(" Keep") for row in screen.buttons for t, _ in row)
        for row in screen.buttons:
            for _, data in row:
                assert len(data.encode()) <= CALLBACK_LIMIT
    assert seen_removed == len(manager.active.removed_games) and seen_kept == len(manager.active.slip)
    assert slate.list_screen(manager.active, 1).text.splitlines()[0].startswith("Final list: Kept ")


def test_games_with_many_options_page_by_size_so_a_message_never_overflows():
    evals = many(PAGE_SIZE, options=8, name="Wolverhampton Wanderers Football Club ")
    manager, _ = started(evals)
    pages = slate.page_count(manager.active)
    assert pages >= 2                                               # 8 games x 8 options do not fit one message
    numbers = []
    for page in range(pages):
        screen = slate.list_screen(manager.active, page)
        assert len(screen.text) <= 4000 and "…" not in screen.text
        numbers += [int(m) for m in re.findall(r"^(\d+)\) ", screen.text, re.M)]
        assert len([b for row in screen.buttons for b in row]) <= 100            # Telegram's button limit
        assert all(len(row) <= 2 for row in screen.buttons)
        assert len(screen.buttons) - 1 <= slate.MAX_PLAY_ROWS_PER_PAGE + 1                # play rows + nav + Final confirm
    assert numbers == list(range(1, PAGE_SIZE + 1))                 # every game appears exactly once, in order


def test_callback_data_for_every_action_is_short_and_round_trips():
    for action in (Action.DELETE, Action.RESTORE):
        data = encode("a" * 8, 999999999, action, "f" * 12)
        assert len(data.encode()) <= CALLBACK_LIMIT and slate.decode(data) == ("a" * 8, 999999999, action, "f" * 12)
    for action in (Action.DONE, Action.ADJUST, Action.CONFIRM):
        assert len(encode("a" * 8, 999999999, action).encode()) <= CALLBACK_LIMIT
    values = [a.value for a in Action]
    assert len(values) == len(set(values))                          # no two actions share a letter


def test_every_button_of_the_list_and_final_screens_fits_64_bytes():
    manager, session = started(options_game(5))
    screens = [slate.list_screen(session)]
    act(manager, Action.DONE)
    screens.append(slate.final_screen(manager.active))
    for screen in screens:
        assert all(len(d.encode()) <= CALLBACK_LIMIT for row in screen.buttons for _, d in row)


# ---------- the reply: only games with a play when something passes; NO VALID PLAY keeps its WHY block ----------

def test_something_passing_gives_no_skipped_game_noise_and_nothing_passing_keeps_the_why_block():
    from test_clear_reply import chunks_for, game, many as fixtures_many, not_covered, ready_inputs, totals
    from predictions.core.stats import READY, FixtureStats
    ok = game("Alpha", "Beta", "Testland - League", markets=[totals("1.5", "1.60", "2.20")])
    off = fixtures_many("Other - League", 3, 100)

    def answer(f):
        return FixtureStats(f.match_id, READY, tuple(ready_inputs(2.5, 1.5))) if f is ok else not_covered(f)
    assert chunks_for([ok, *off], answer) == []                                    # the list is the whole reply
    chunks = chunks_for(off, not_covered)
    assert chunks[0] == "NO VALID PLAY - PASS" and "Why there is no valid play" in chunks[1]


def test_the_main_list_marks_a_goals_only_estimate_and_a_converted_play():
    from dataclasses import replace
    from predictions.core.engines.soccer import GOALS_ONLY_MODEL_NAME
    plain = ev("Alpha", "Beta", "Over 1.5", 0.80)
    goals_only = replace(ev("Gamma", "Delta", "Over 2.5", 0.75, hour=16),
                         estimate=replace(ev().estimate, probability=0.75, model=GOALS_ONLY_MODEL_NAME))
    shielded = ev("Echo", "Foxtrot", "Over 3.5", 0.85, rule="1X2 swap: tried as a goal line", original="Home @ 2.10", hour=18)
    manager, _ = started([plain, goals_only, shielded])
    text = slate.list_screen(manager.active).text
    assert "Over 2.5 @ 1.85 - 75.0% [goals-only]" in text and "Over 3.5 @ 1.85 - 85.0% (1X2 swap)" in text
    assert "[goals-only]" not in text.split("Alpha vs Beta")[1].split("Gamma")[0]
    assert "[goals-only]" in slate.final_screen(act(manager, Action.DONE).session).text


# ---------- more per-play coverage ----------

def test_header_counts_games_with_a_kept_play_not_plays():
    manager, session = started(options_game(3))
    plays = options_game(3)
    assert slate.list_screen(session).text.startswith("Final list: Kept 2 of 2 game(s).")
    act(manager, Action.RESTORE, pid(plays[2]))                      # switching the play keeps the count
    assert slate.list_screen(manager.active).text.startswith("Final list: Kept 2 of 2 game(s).")
    act(manager, Action.DELETE, pid(plays[2]))
    assert slate.list_screen(manager.active).text.startswith("Final list: Kept 1 of 2 game(s).")
    act(manager, Action.DELETE, pid(plays[3]))
    assert slate.list_screen(manager.active).text.startswith("Final list: Kept 0 of 2 game(s).")


def test_every_keep_or_delete_that_changes_state_bumps_the_version_and_a_no_op_does_not():
    manager, _ = started(options_game(3))
    plays = options_game(3)
    seen = [manager.active.state_version]
    for action, index in ((Action.RESTORE, 1), (Action.RESTORE, 2), (Action.DELETE, 2), (Action.RESTORE, 0)):
        assert act(manager, action, pid(plays[index])).status is Status.APPLIED
        seen.append(manager.active.state_version)
    assert seen == [1, 2, 3, 4, 5]
    for action, index in ((Action.RESTORE, 0), (Action.DELETE, 1)):
        assert act(manager, action, pid(plays[index])).status is Status.REPEAT
    assert manager.active.state_version == 5


def test_a_stale_play_button_answers_this_list_changed_and_changes_nothing():
    manager, _ = started(options_game(3))
    plays = options_game(3)
    old_keep_b = encode(SID, 1, Action.RESTORE, pid(plays[1]))
    old_delete_a = encode(SID, 1, Action.DELETE, pid(plays[0]))
    assert press(manager, old_keep_b).answer == "Kept."
    after = manager.active
    assert press(manager, old_delete_a).answer.startswith("This list changed") and manager.active == after
    assert press(manager, old_keep_b).answer == "Already done." and manager.active == after


def test_keep_and_delete_on_every_play_of_a_multi_play_review_list_work_the_same_way():
    manager, _ = started(options_game(2), origin="review:ABC123")
    plays = options_game(2)
    assert press(manager, button(manager, "1b Keep")).answer == "Kept."
    reply = press(manager, button(manager, "1b Delete"))
    assert reply.answer == "Deleted." and "(removed)" in reply.screen.text and "Approved selections from Review" in reply.screen.text
    assert pid(plays[0]) not in manager.active.slip and pid(plays[1]) not in manager.active.slip


def test_a_snapshot_stores_which_play_is_kept_per_game_and_an_old_last_action_is_dropped():
    manager, _ = started(options_game(3))
    plays = options_game(3)
    act(manager, Action.RESTORE, pid(plays[1]))
    data = json.loads(to_json(manager.active))
    assert data["chosen"] and pid(plays[1]) in data["chosen"] and data["removed_games"] == []
    data["last"] = [2, "c", pid(plays[1])]                       # the retired option-select action
    back = from_json(json.dumps(data))
    assert back.last is None and back.slip == manager.active.slip


def test_many_multi_play_games_page_with_a_bounded_keyboard_and_every_play_reachable():
    evals = many(12, options=6)
    manager, _ = started(evals)
    pages = slate.page_count(manager.active)
    assert pages >= 2
    keeps, count = set(), 0
    for page in range(pages):
        screen = slate.list_screen(manager.active, page)
        buttons = [b for row in screen.buttons for b in row]
        assert len(buttons) <= 100 and len(screen.text) <= 4000
        keeps |= {slate.decode(d)[3] for t, d in buttons if t.endswith(" Keep")}
        count += sum(t.endswith(" Keep") for t, _ in buttons)
    assert keeps == {p.pick_id for p in manager.active.menu} and count == len(manager.active.menu) == 72
    # a Keep pressed on the last page edits that page again
    last = slate.list_screen(manager.active, pages - 1)
    keep = next(d for row in last.buttons for t, d in row if t.endswith("b Keep"))
    reply = press(manager, keep)
    assert reply.answer == "Kept." and f"Page {pages} of {pages}" in reply.screen.text
