"""Story 2.6: the final list (Slip, session state machine, buttons, and the real handler chain)."""
import asyncio
import re
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest
from telegram import Update, User
from telegram.ext import ApplicationBuilder, ExtBot

import app as bot
from handled import drop_ack
from predictions.core.inputs import Input
from predictions.core.model import Estimate, Evaluation
from predictions.core.slip import Slip, UnknownPick, make_pick, make_pick_id
from predictions.core.types import MatchId, Owner
from predictions.flow.evaluate import GateResult
from predictions.flow.session import Action, SessionManager, State, Status
from predictions.flow.store import EvaluationStore
from predictions.telegram import slate
from listview import detail_of
from predictions.telegram.slate import CALLBACK_LIMIT, PAGE_SIZE, decode, encode, handle_callback

OWNER_ID = 111222333
STRANGER = 999888777
OWNER = Owner(OWNER_ID)
KICKOFF = datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc)
INPUTS = (Input("home.xg_for", 1.8, "xG per match", "understat"),
          Input("away.xg_against", 1.4, "xG per match", "understat"))


def mid(home="Alpha", away="Beta", hour=14):
    return MatchId("soccer", "Testland - League", KICKOFF.replace(hour=hour), home, away)


def ev(home="Alpha", away="Beta", selection="Over 2.5", prob=0.72, odds="1.85", passed=True, rule="", original="",
       hour=14, push=0.0):
    est = Estimate(prob, INPUTS, "poisson", push)
    return Evaluation(mid(home, away, hour), "Over/Under", selection, Decimal(odds), est, Decimal("0.10"), passed,
                      "passed" if passed else "below the floor", rule, original, ("inputs: a, b",))


def two_matches():
    return [ev("Alpha", "Beta", "Over 2.5", 0.72), ev("Gamma", "Delta", "Over 1.5", 0.88, "1.30", hour=16)]


def with_alternatives():
    return [ev("Alpha", "Beta", "Over 2.5", 0.72, "1.85"), ev("Alpha", "Beta", "Over 3.5", 0.81, "1.50",
                                                             rule="1X2 swap", original="Over 2.5"),
            ev("Gamma", "Delta", "Over 1.5", 0.88, "1.30", hour=16)]


# ---------- Slip (pure core) ----------

def test_pick_ids_are_stable_short_and_not_positions():
    a, b = two_matches()
    slip = Slip.from_evaluations([a, b])
    id_a, id_b = make_pick_id(a), make_pick_id(b)
    assert id_a != id_b and re.fullmatch(r"[0-9a-f]{12}", id_a)
    assert [p.pick_id for p in slip.picks] == [id_a, id_b]
    after = slip.delete(id_a)
    assert after.picks[0].pick_id == id_b  # b keeps its id though its position changed
    assert Slip.from_evaluations([b, a]).get(id_a).evaluation == a  # order does not matter


def test_delete_removes_only_that_pick_and_the_slip_is_immutable():
    a, b = two_matches()
    slip = Slip.from_evaluations([a, b])
    smaller = slip.delete(make_pick_id(a))
    assert len(smaller) == 1 and len(slip) == 2
    with pytest.raises(UnknownPick):
        smaller.delete(make_pick_id(a))


def test_all_alternatives_are_kept_until_one_is_chosen():
    slip = Slip.from_evaluations(with_alternatives())
    assert len(slip) == 3 and not slip.one_per_match
    (key, picks), = slip.conflicts().items()
    assert len(picks) == 2 and key == str(mid())


def test_choose_keeps_one_pick_for_the_match_and_leaves_other_matches_alone():
    alts = with_alternatives()
    slip = Slip.from_evaluations(alts).choose(make_pick_id(alts[1]))
    assert slip.one_per_match and len(slip) == 2
    assert {p.evaluation.selection for p in slip.picks} == {"Over 3.5", "Over 1.5"}


def test_one_per_match_is_keyed_by_the_canonical_match_id():
    a = ev("Alpha FC", "Beta")
    b = ev("alpha  fc", "BETA", "Over 3.5", 0.8, "1.5")  # same match once names are normalized
    assert not Slip.from_evaluations([a, b]).one_per_match


def test_only_passed_picks_enter_a_slip_and_a_repeat_counts_once():
    with pytest.raises(ValueError):
        make_pick(ev(passed=False, prob=0.6))
    a = ev()
    assert len(Slip.from_evaluations([a, a])) == 1


# ---------- session state machine ----------

NOW = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)  # before KICKOFF, so the tests do not age


def started(evals=None):
    manager = SessionManager(clock=lambda: NOW)
    return manager, manager.start(evals or two_matches(), "abcd1234")


def test_a_session_starts_in_reviewing_list_at_version_one():
    manager, session = started()
    assert session.state is State.REVIEWING_LIST and session.state_version == 1 and manager.active == session


def test_delete_bumps_the_version_and_removes_the_pick():
    manager, session = started()
    pid = session.slip.picks[0].pick_id
    result = manager.act(OWNER, "abcd1234", 1, Action.DELETE, pid)
    assert result.status is Status.APPLIED and result.session.state_version == 2
    assert pid not in result.session.slip and manager.active == result.session


def test_a_button_from_an_older_version_does_nothing():
    manager, session = started()
    first, second = (p.pick_id for p in session.slip.picks)
    manager.act(OWNER, "abcd1234", 1, Action.DELETE, first)
    before = manager.active
    result = manager.act(OWNER, "abcd1234", 1, Action.DELETE, second)  # made for version 1
    assert result.status is Status.STALE and manager.active == before and second in manager.active.slip


def test_a_double_tap_has_no_extra_effect():
    manager, session = started(with_alternatives())
    pid = session.slip.picks[0].pick_id
    manager.act(OWNER, "abcd1234", 1, Action.DELETE, pid)
    after_first = manager.active
    again = manager.act(OWNER, "abcd1234", 1, Action.DELETE, pid)
    assert again.status is Status.REPEAT and manager.active == after_first


def test_a_wrong_session_id_is_stale():
    manager, _ = started()
    assert manager.act(OWNER, "ffffffff", 1, Action.CONFIRM).status is Status.STALE


def test_a_new_session_voids_the_old_buttons():
    manager, old = started()
    manager.start(two_matches(), "11112222")
    result = manager.act(OWNER, "abcd1234", 1, Action.DELETE, old.slip.picks[0].pick_id)
    assert result.status is Status.STALE and len(manager.active.slip) == 2


def test_ending_a_session_voids_its_buttons():
    manager, _ = started()
    manager.end()
    assert manager.act(OWNER, "abcd1234", 1, Action.CONFIRM).status is Status.STALE


def test_several_options_for_a_match_are_one_game_with_exactly_one_chosen_play():
    alts = with_alternatives()
    manager, session = started(alts)
    assert len(session.games()) == 2 and len(session.menu) == 3
    assert session.slip.one_per_match and len(session.slip) == 2        # structural: one play per game
    game = session.games()[0]
    assert [p.evaluation.selection for p in game.options] == ["Over 3.5", "Over 2.5"]   # advised (81%) first
    assert game.chosen_id == make_pick_id(alts[1]) and game.advised.pick_id == game.chosen_id


def test_confirm_straight_from_the_list_is_refused_the_final_screen_comes_first():
    manager, session = started()
    result = manager.act(OWNER, "abcd1234", 1, Action.CONFIRM)
    assert result.status is Status.BLOCKED and "Final confirm" in result.notice
    assert manager.active == session  # nothing changed, not even the version


def test_choose_then_confirm_moves_to_the_confirmed_state():
    alts = with_alternatives()
    manager, session = started(alts)
    manager.act(OWNER, "abcd1234", 1, Action.RESTORE, make_pick_id(alts[0]))
    assert manager.act(OWNER, "abcd1234", 2, Action.DONE).status is Status.APPLIED
    result = manager.act(OWNER, "abcd1234", 3, Action.CONFIRM)
    assert result.status is Status.APPLIED and result.session.state is State.LOADING_SLIP
    assert result.session.state_version == 4 and len(result.session.slip) == 2
    assert make_pick_id(alts[0]) in result.session.slip and make_pick_id(alts[1]) not in result.session.slip


def test_after_confirming_every_old_button_is_dead_and_a_repeat_is_harmless():
    manager, session = started()
    manager.act(OWNER, "abcd1234", 1, Action.DONE)
    manager.act(OWNER, "abcd1234", 2, Action.CONFIRM)
    confirmed = manager.active
    assert manager.act(OWNER, "abcd1234", 2, Action.CONFIRM).status is Status.REPEAT
    assert manager.act(OWNER, "abcd1234", 1, Action.DELETE, session.slip.picks[0].pick_id).status is Status.STALE
    assert manager.act(OWNER, "abcd1234", 2, Action.DELETE, session.slip.picks[0].pick_id).status is Status.STALE
    assert manager.active == confirmed


def test_an_empty_list_cannot_be_confirmed():
    manager, session = started([ev()])
    manager.act(OWNER, "abcd1234", 1, Action.DELETE, session.slip.picks[0].pick_id)
    assert manager.act(OWNER, "abcd1234", 2, Action.DONE).status is Status.BLOCKED


def test_views_do_not_change_the_version():
    manager, session = started()
    pid = session.slip.picks[0].pick_id
    assert manager.act(OWNER, "abcd1234", 1, Action.PAGE, "0").status is Status.VIEW and manager.active == session
    assert manager.act(OWNER, "abcd1234", 1, Action.BACK, "0").status is Status.VIEW and manager.active == session
    assert manager.act(OWNER, "abcd1234", 1, Action.DELETE, "0" * 12).status is Status.UNKNOWN_PICK and manager.active == session


def test_session_actions_need_a_typed_owner():
    manager, _ = started()
    with pytest.raises(TypeError):
        manager.act(OWNER_ID, "abcd1234", 1, Action.CONFIRM)


# ---------- callback data ----------

def all_buttons(screen):
    return [b for row in screen.buttons for b in row]


def test_callback_data_carries_session_version_and_pick_id_and_fits_64_bytes():
    manager, session = started(with_alternatives())
    screen = slate.list_screen(session)
    pick_ids = {p.pick_id for p in session.menu}
    for label, data in all_buttons(screen):
        assert len(data.encode()) <= CALLBACK_LIMIT
        sid, version, action, arg = decode(data)
        assert sid == "abcd1234" and version == 1
        if action in (Action.DELETE, Action.RESTORE):
            assert arg in pick_ids  # a stable pick id, never a list index
    worst = encode("abcd1234", 999999999, Action.RESTORE, "f" * 12)
    assert len(worst.encode()) <= CALLBACK_LIMIT


@pytest.mark.parametrize("data", [None, "", "TOGGLE_0", "s:zzzz:1:d:", "s:abcd1234:x:d:", "s:abcd1234:1:?:",
                                  "s:abcd1234:1:d:" + "a" * 60])
def test_bad_callback_data_is_rejected_with_a_notice(data):
    manager, _ = started()
    reply = handle_callback(data, OWNER, manager)
    assert reply.answer == slate.NOTICE_UNKNOWN and reply.screen is None


# ---------- screens ----------

def test_the_list_shows_game_selection_and_percentage_for_every_pick():
    _, session = started(two_matches())
    text = slate.list_screen(session).text
    assert "Alpha vs Beta" in text and "Over 2.5 @ 1.85 - 72.0%" in text
    assert "Gamma vs Delta" in text and "Over 1.5 @ 1.30 - 88.0%" in text
    assert "EV" not in text and "understat" not in text and "success" not in text   # no extra detail on the main list


def test_the_list_shows_the_optimized_selection_and_the_original():
    _, session = started(with_alternatives())
    text = slate.list_screen(session).text
    assert f"1a {slate.MARK_KEPT} Over 3.5 @ 1.50 - 81.0% (1X2 swap) (advised)" in text
    assert f"1b {slate.MARK_REMOVED} Over 2.5 @ 1.85 - 72.0%" in text
    assert "Original pick: Over 2.5" in detail_of(session.find_pick(make_pick_id(with_alternatives()[1])))


def test_a_refund_split_is_shown_for_push_protected_picks():
    _, session = started([ev(prob=0.80, push=0.10)])
    assert "Over 2.5 @ 1.85 - 80.0%" in slate.list_screen(session).text
    assert "success 80.0% (win 70.0% + refund 10.0%)" in detail_of(session.menu[0])


def test_all_plays_for_a_match_are_listed_together_each_with_its_own_keep_and_delete():
    _, session = started(with_alternatives())
    screen = slate.list_screen(session)
    assert screen.text.count("Alpha vs Beta") == 1   # listed together, once
    labels = [[t for t, _ in row] for row in screen.buttons]
    assert labels == [["1a Keep", "1a Delete"], ["1b Keep", "1b Delete"], ["2 Keep", "2 Delete"], ["Final confirm"]]
    assert not [1 for _, d in all_buttons(screen) if decode(d)[2].value in ("o", "c")]   # no Open, no option selectors


def test_the_inputs_sources_and_rule_are_part_of_the_play_record():
    manager, session = started(with_alternatives())
    alt = session.find_pick(make_pick_id(with_alternatives()[1]))
    text = detail_of(alt)
    assert "home.xg_for: 1.8 (xG per match) from understat" in text
    assert "Rule applied: 1X2 swap" in text and "Original pick: Over 2.5" in text


def test_a_confirm_from_the_list_is_refused_with_a_notice_and_the_list_stays():
    manager, session = started(with_alternatives())
    reply = handle_callback(encode("abcd1234", 1, Action.CONFIRM), OWNER, manager)
    assert reply.alert and "Press Final confirm first" in reply.answer
    assert reply.screen is not None and "Press Final confirm first" in reply.screen.text
    assert manager.active.state is State.REVIEWING_LIST and manager.active.state_version == 1


def test_confirmed_screen_lists_kept_picks_and_says_the_betslip_is_being_loaded():
    manager, session = started(two_matches())
    handle_callback(encode("abcd1234", 1, Action.DONE), OWNER, manager)
    reply = handle_callback(encode("abcd1234", 2, Action.CONFIRM), OWNER, manager)
    text = reply.screen.text
    assert "Games confirmed: 2" in text and "Over 2.5 @ 1.85" in text and "Over 1.5 @ 1.30" in text
    assert "Loading them onto SportyBet now" in text and "No bet is placed" in text
    assert reply.screen.buttons == ()


def test_deleting_every_game_keeps_it_visible_with_a_keep_button_and_done_is_refused():
    manager, session = started([ev()])
    reply = handle_callback(encode("abcd1234", 1, Action.DELETE, session.slip.picks[0].pick_id), OWNER, manager)
    text = reply.screen.text
    assert "Kept 0 of 1 game(s)" in text and "(removed)" in text and "No play is kept" in text
    assert [t for r in reply.screen.buttons for t, _ in r] == ["1 Keep", "1 Delete", "Final confirm"]
    refused = handle_callback(encode("abcd1234", 2, Action.DONE), OWNER, manager)
    assert refused.alert and "no play is kept" in refused.answer.lower() and manager.active.state is State.REVIEWING_LIST


def test_a_long_list_is_paged_and_every_pick_stays_reachable():
    evals = [ev(f"Home{i}", f"Away{i}", hour=8 + i % 12, prob=0.7 + i / 1000) for i in range(25)]
    manager, session = started(evals)
    seen, page = set(), 0
    while True:
        screen = slate.list_screen(session, page)
        assert len(screen.text) <= 4000
        seen |= {decode(d)[3] for _, d in all_buttons(screen) if decode(d)[2] is Action.DELETE}
        assert sum(decode(d)[2] is Action.DELETE for _, d in all_buttons(screen)) <= PAGE_SIZE
        nxt = [d for label, d in all_buttons(screen) if label == "Next"]
        if not nxt:
            break
        reply = handle_callback(nxt[0], OWNER, manager)
        assert reply.screen is not None
        page += 1
    assert seen == {p.pick_id for p in session.slip.picks} and page == 3


def test_handle_callback_keep_delete_flow():
    alts = with_alternatives()
    manager, session = started(alts)
    keep = handle_callback(encode("abcd1234", 1, Action.RESTORE, make_pick_id(alts[0])), OWNER, manager)
    assert keep.answer == "Kept." and f"1b {slate.MARK_KEPT} Over 2.5" in keep.screen.text
    stale = handle_callback(encode("abcd1234", 1, Action.DELETE, make_pick_id(alts[2])), OWNER, manager)
    assert stale.answer == slate.NOTICE_STALE and stale.screen.text == slate.list_screen(manager.active).text  # shows the CURRENT list
    assert make_pick_id(alts[2]) in manager.active.slip


# ---------- the flow hands over the evaluation ----------

def test_run_matchday_hands_the_gate_to_on_result_only_when_games_were_evaluated():
    from predictions.core.model import validate  # noqa: F401  (import check only)
    from test_gate import FakeFixtures, FakeStats, fixture, inputs_for, ready, totals  # noqa
    from predictions.flow.matchday import run_matchday

    NOW = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    f = fixture(markets=[totals("1.5", "1.30", "3.20")])
    seen = []
    stats = FakeStats(lambda fx: ready(fx, inputs_for(2.5, 1.5)))
    asyncio.run(run_matchday("10 October", FakeFixtures([f]), str, NOW, stats_source=stats, on_result=seen.append))
    assert len(seen) == 1 and isinstance(seen[0], GateResult) and seen[0].passed
    seen.clear()
    asyncio.run(run_matchday("10 October", FakeFixtures([f]), str, NOW, on_result=seen.append))  # no stats source
    asyncio.run(run_matchday("banana", FakeFixtures([f]), str, NOW, stats_source=stats, on_result=seen.append))
    assert seen == []


# ---------- wiring through the real python-telegram-bot handler chain ----------

def message(user_id, chat_id, text, chat_type="private"):
    return {"message_id": 7, "date": 1700000000, "chat": {"id": chat_id, "type": chat_type},
            "from": {"id": user_id, "is_bot": False, "first_name": "T"}, "text": text}


class Chain:
    def __init__(self, application, sent, edits, answers):
        self.application, self.sent, self.edits, self.answers = application, sent, edits, answers
        self._n = 100

    def feed(self, payload):
        self._n += 1
        update = Update.de_json({"update_id": self._n, **payload}, self.application.bot)

        async def go():
            await self.application.initialize()
            await self.application.process_update(update)
            await bot.JOBS.wait()  # background requests and slate jobs (AD-22)
            drop_ack(self.sent)
            drop_ack(getattr(self, 'events', []))

        asyncio.run(go())

    def text(self, user_id, text):
        self.feed({"message": message(user_id, user_id, text)})

    def press(self, user_id, data, chat_id=None):
        chat_id = user_id if chat_id is None else chat_id
        self.feed({"callback_query": {"id": str(self._n), "chat_instance": "x", "data": data,
                                      "from": {"id": user_id, "is_bot": False, "first_name": "T"},
                                      "message": message(user_id, chat_id, "list")}})

    def buttons(self, which=-1):
        markup = self.events[which][1].get("reply_markup")
        return [(b.text, b.callback_data) for row in markup.inline_keyboard for b in row]


@pytest.fixture
def chain(monkeypatch, tmp_path):
    sent, edits, answers, events = [], [], [], []
    scenario = {"gate": None}

    async def no_network(self):
        self._bot_user = User(id=1, first_name="TestBot", is_bot=True, username="test_bot")

    async def send_message(self, chat_id, text, **kwargs):
        sent.append((text, kwargs))
        events.append((text, kwargs))

    async def edit_message_text(self, text, *args, **kwargs):
        if scenario.get("edit_error") is not None:
            raise scenario["edit_error"]
        edits.append((text, kwargs))
        events.append((text, kwargs))

    async def answer_callback_query(self, callback_query_id, text=None, show_alert=None, **kwargs):
        answers.append((text, show_alert))

    async def fake_run(argument, source, sanitize, **kwargs):
        if scenario["gate"] is not None:
            kwargs["on_result"](scenario["gate"])
        return ["Matchday fixtures ..."]

    monkeypatch.setattr(ExtBot, "initialize", no_network)
    monkeypatch.setattr(ExtBot, "send_message", send_message)
    monkeypatch.setattr(ExtBot, "edit_message_text", edit_message_text)
    monkeypatch.setattr(ExtBot, "answer_callback_query", answer_callback_query)
    jobs = []

    async def no_job(job, message):  # Epic 5's loading and placing are tested in test_epic5_flow.py
        jobs.append(job)

    monkeypatch.setattr(bot, "run_matchday", fake_run)
    monkeypatch.setattr(bot, "run_slate_job", no_job)
    monkeypatch.setattr(bot, "SESSIONS", SessionManager(clock=lambda: NOW))
    store = EvaluationStore.open(tmp_path / "t.db", bot.redact)
    monkeypatch.setattr(bot, "STORE", store)
    application = ApplicationBuilder().token("123456:TEST-TOKEN-NOT-REAL").build()
    bot.register_handlers(application)
    c = Chain(application, sent, edits, answers)
    c.scenario, c.store, c.events, c.jobs = scenario, store, events, jobs
    yield c
    store.close()


def gate_of(passed, rejected=()):
    return GateResult(tuple(passed), tuple(rejected), 2, 0, {})


def open_list(chain, evals):
    chain.scenario["gate"] = gate_of(evals)
    chain.text(OWNER_ID, "Matchday 10 October")
    return chain.buttons(-1)


def find(buttons, label):
    return next(data for text, data in buttons if text == label)


def test_matchday_shows_the_list_as_buttons_after_the_normal_reply(chain):
    open_list(chain, two_matches())
    assert chain.sent[0][0] == "Matchday fixtures ..."
    text, kwargs = chain.sent[1]
    assert "Final list: Kept 2 of 2 game(s)" in text and "Over 2.5 @ 1.85 - 72.0%" in text
    labels = [t for t, _ in chain.buttons(-1)]
    assert labels == ["1 Keep", "1 Delete", "2 Keep", "2 Delete", "Final confirm"]


def test_delete_button_removes_the_pick_and_edits_the_message(chain):
    buttons = open_list(chain, two_matches())
    chain.press(OWNER_ID, find(buttons, "1 Delete"))
    assert chain.answers[-1] == ("Deleted.", False)
    text = chain.edits[-1][0]
    assert f"1) {slate.MARK_REMOVED} " in text and "(removed)" in text and f"2) {slate.MARK_KEPT} " in text
    assert "Kept 1 of 2 game(s)" in text and "Over 2.5 @ 1.85" in text      # the removed game stays visible
    assert [t for t, _ in chain.buttons(-1)] == ["1 Keep", "1 Delete", "2 Keep", "2 Delete", "Final confirm"]


def test_stale_buttons_answer_with_a_notice_and_change_nothing(chain):
    buttons = open_list(chain, two_matches())
    delete_first, delete_second = find(buttons, "1 Delete"), find(buttons, "2 Delete")
    chain.press(OWNER_ID, delete_first)
    edits_before = len(chain.edits)
    chain.press(OWNER_ID, delete_second)  # from the version before the first delete
    assert chain.answers[-1] == ("This list changed. Please use the newest list.", False)
    assert len(chain.edits) == edits_before + 1 and "Kept 1 of 2" in chain.edits[-1][0]   # the CURRENT list is shown again
    assert len(bot.SESSIONS.active.slip) == 1


def test_a_double_tap_deletes_only_once(chain):
    buttons = open_list(chain, two_matches())
    data = find(buttons, "1 Delete")
    chain.press(OWNER_ID, data)
    chain.press(OWNER_ID, data)
    assert chain.answers[-1] == ("Already done.", False)
    assert len(chain.edits) == 1 and len(bot.SESSIONS.active.slip) == 1


def test_a_new_matchday_asks_first_and_the_old_buttons_die_only_after_replace(chain):
    old = open_list(chain, two_matches())
    ask = open_list(chain, [ev("Echo", "Foxtrot", "Over 1.5", 0.9, "1.3")])   # Story 5.7: asks, does not replace
    assert [t for t, _ in ask] == ["Replace the open slate", "Keep the open slate"]
    assert len(bot.SESSIONS.active.slip) == 2                                   # the open slate is untouched
    chain.press(OWNER_ID, find(ask, "Replace the open slate"))
    chain.press(OWNER_ID, find(old, "1 Delete"))
    assert chain.answers[-1][0] == "This list changed. Please use the newest list."
    assert len(bot.SESSIONS.active.slip) == 1


def test_a_matchday_with_no_valid_play_shows_no_list_and_leaves_the_open_slate_alone(chain):
    old = open_list(chain, two_matches())
    count = len(chain.sent)
    chain.scenario["gate"] = gate_of([], [ev(passed=False, prob=0.6)])
    chain.text(OWNER_ID, "Matchday 10 October")
    assert len(chain.sent) == count + 1  # only the normal reply: no list, no question
    chain.press(OWNER_ID, find(old, "Final confirm"))  # Story 5.7: nothing replaced it, so it still works
    assert bot.SESSIONS.active.state is State.REVIEWING_FINAL


def test_options_are_chosen_then_done_then_the_final_screen_then_confirm(chain):
    alts = with_alternatives()
    buttons = open_list(chain, alts)
    assert [t for t, _ in buttons] == ["1a Keep", "1a Delete", "1b Keep", "1b Delete", "2 Keep", "2 Delete", "Final confirm"]
    assert bot.SESSIONS.active.slip.one_per_match and len(bot.SESSIONS.active.slip) == 2
    chain.press(OWNER_ID, find(chain.buttons(-1), "1b Keep"))
    assert chain.answers[-1] == ("Kept.", False) and f"1b {slate.MARK_KEPT} Over 2.5" in chain.edits[-1][0]
    chain.press(OWNER_ID, find(chain.buttons(-1), "Final confirm"))
    assert bot.SESSIONS.active.state is State.REVIEWING_FINAL
    final_text = chain.edits[-1][0]
    assert final_text.startswith("Final edit: 2 game(s) kept.") and "Over 2.5 @ 1.85 - 72.0%" in final_text
    assert [t for t, _ in chain.buttons(-1)] == ["Confirm games", "Back to adjust"]
    chain.press(OWNER_ID, find(chain.buttons(-1), "Back to adjust"))
    assert bot.SESSIONS.active.state is State.REVIEWING_LIST and f"1b {slate.MARK_KEPT} Over 2.5" in chain.edits[-1][0]
    chain.press(OWNER_ID, find(chain.buttons(-1), "Final confirm"))
    chain.press(OWNER_ID, find(chain.buttons(-1), "Confirm games"))
    assert bot.SESSIONS.active.state is State.LOADING_SLIP
    final = chain.edits[-1]
    assert "Games confirmed: 2" in final[0] and "Loading them onto SportyBet now" in final[0]
    assert final[1].get("reply_markup") is None


def test_a_repeated_confirm_tap_is_harmless(chain):
    buttons = open_list(chain, two_matches())
    chain.press(OWNER_ID, find(buttons, "Final confirm"))
    confirm = find(chain.buttons(-1), "Confirm games")
    chain.press(OWNER_ID, confirm)
    edits = len(chain.edits)
    chain.press(OWNER_ID, confirm)
    assert chain.answers[-1] == ("Already done.", False) and len(chain.edits) == edits


def test_a_stranger_cannot_press_the_owners_buttons(chain):
    buttons = open_list(chain, two_matches())
    chain.press(STRANGER, find(buttons, "1 Delete"))
    chain.press(OWNER_ID, find(buttons, "1 Delete"), chat_id=-100123)  # the owner in a group
    assert chain.answers == [] and chain.edits == []
    assert len(bot.SESSIONS.active.slip) == 2 and bot.SESSIONS.active.state_version == 1


def test_a_stranger_matchday_starts_nothing(chain):
    chain.scenario["gate"] = gate_of(two_matches())
    chain.text(STRANGER, "Matchday 10 October")
    assert chain.sent == [] and bot.SESSIONS.active is None and chain.store.list_evaluations() == []


# ---------- Story 2.7 wiring: every evaluation shown is saved ----------

def test_matchday_saves_every_evaluation_under_the_session_id(chain):
    rejected = ev("Echo", "Foxtrot", "Under 2.5", 0.55, "1.9", passed=False)
    chain.scenario["gate"] = gate_of(two_matches(), [rejected])
    chain.text(OWNER_ID, "Matchday 10 October")
    rows = chain.store.list_evaluations()
    assert len(rows) == 3 and {r["session_id"] for r in rows} == {bot.SESSIONS.active.session_id}
    assert sorted(r["passed"] for r in rows) == [0, 1, 1] and all(r["placed"] == 0 for r in rows)
    assert all(r["inputs"] for r in rows)


def test_no_valid_play_still_saves_the_rejections(chain):
    chain.scenario["gate"] = gate_of([], [ev(passed=False, prob=0.6)])
    chain.text(OWNER_ID, "Matchday 10 October")
    assert len(chain.store.list_evaluations()) == 1 and bot.SESSIONS.active is None


def test_a_failed_save_is_reported_and_the_list_is_still_shown(chain, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("disk full at C:\\somewhere\\predictions.db")

    monkeypatch.setattr(chain.store, "save_evaluations", broken)
    chain.scenario["gate"] = gate_of(two_matches())
    chain.text(OWNER_ID, "Matchday 10 October")
    texts = [t for t, _ in chain.sent]
    assert any("Could not save this evaluation" in t and "somewhere" not in t for t in texts)
    assert any("Final list" in t for t in texts)


# ---------- review fixes: expiry, stale lists, long confirmed list, failed edits ----------

def test_a_session_records_when_it_started_and_the_earliest_kickoff():
    manager, session = started(with_alternatives())
    assert session.created_at == NOW and session.earliest_kickoff == KICKOFF


def test_confirm_is_stale_after_six_hours():
    from datetime import timedelta
    clock = {"now": NOW}
    manager = SessionManager(clock=lambda: clock["now"])
    manager.start(two_matches(), "abcd1234")
    clock["now"] = NOW + timedelta(hours=6, seconds=1)
    result = manager.act(OWNER, "abcd1234", 1, Action.DONE)
    assert result.status is Status.STALE and manager.active.state is State.REVIEWING_LIST
    # viewing and deleting are not confirming: they still work
    assert manager.act(OWNER, "abcd1234", 1, Action.PAGE, "0").status is Status.VIEW


def test_confirm_is_allowed_just_inside_six_hours():
    from datetime import timedelta
    clock = {"now": NOW}
    manager = SessionManager(clock=lambda: clock["now"])
    manager.start(two_matches(), "abcd1234")
    clock["now"] = NOW + timedelta(hours=6)
    assert manager.act(OWNER, "abcd1234", 1, Action.DONE).status is Status.APPLIED
    assert manager.act(OWNER, "abcd1234", 2, Action.CONFIRM).status is Status.APPLIED


def test_confirm_is_stale_once_the_earliest_kickoff_has_passed():
    clock = {"now": KICKOFF.replace(hour=13, minute=59)}
    manager = SessionManager(clock=lambda: clock["now"])
    manager.start(two_matches(), "abcd1234")  # earliest kickoff 14:00
    clock["now"] = KICKOFF
    assert manager.act(OWNER, "abcd1234", 1, Action.DONE).status is Status.STALE


def test_the_earliest_kickoff_follows_the_picks_not_the_first_one_listed():
    manager, session = started([ev("Gamma", "Delta", hour=16), ev("Alpha", "Beta", hour=12)])
    assert session.earliest_kickoff == KICKOFF.replace(hour=12)


def test_a_matchday_that_finds_no_games_or_fails_leaves_the_open_slate_alone(chain):
    old = open_list(chain, two_matches())
    chain.scenario["gate"] = None  # the request produces no evaluation (no games, or a fetch error)
    chain.text(OWNER_ID, "Matchday 10 October")
    assert bot.SESSIONS.active is not None and len(bot.SESSIONS.active.slip) == 2  # a failed request kills nothing
    chain.press(OWNER_ID, find(old, "1 Delete"))
    assert chain.answers[-1][0] == "Deleted." and len(bot.SESSIONS.active.slip) == 1


def test_the_confirmed_screen_of_40_picks_keeps_the_notice_and_ends_with_and_more():
    long_name = "Wolverhampton Wanderers Football Club "
    evals = [ev(f"{long_name}{i:02d}", f"{long_name}Away {i:02d}", "Over 2.5", 0.72, "1.85") for i in range(40)]
    manager, session = started(evals)
    manager.act(OWNER, "abcd1234", 1, Action.DONE)
    manager.act(OWNER, "abcd1234", 2, Action.CONFIRM)
    text = slate.confirmed_screen(manager.active).text
    assert len(text) <= slate.MESSAGE_LIMIT
    assert "Games confirmed: 40 pick(s)" in text and slate.LOADING_NOTE in text
    assert "No bet is placed" in text
    lines = text.splitlines()
    assert re.fullmatch(r"and \d+ more", lines[-1])
    shown = [l for l in lines if re.match(r"\d+\. ", l)]
    assert 0 < len(shown) < 40 and len(shown) + int(lines[-1].split()[1]) == 40
    assert all(l.endswith("%") for l in shown)  # no pick is cut mid-word


def test_forty_short_picks_all_fit_on_one_line_each():
    evals = [ev(f"Home{i:02d}", f"Away{i:02d}") for i in range(40)]
    manager, session = started(evals)
    manager.act(OWNER, "abcd1234", 1, Action.DONE)
    manager.act(OWNER, "abcd1234", 2, Action.CONFIRM)
    text = slate.confirmed_screen(manager.active).text
    assert len([l for l in text.splitlines() if re.match(r"\d+\. ", l)]) == 40
    assert "more" not in text.splitlines()[-1] and slate.LOADING_NOTE in text


def test_a_short_confirmed_list_has_no_and_more_line():
    manager, session = started(two_matches())
    manager.act(OWNER, "abcd1234", 1, Action.CONFIRM)
    text = slate.confirmed_screen(manager.active).text
    assert "more" not in text.splitlines()[-1] and len([l for l in text.splitlines() if re.match(r"\d+\. ", l)]) == 2


def test_a_failed_edit_after_a_change_sends_the_screen_as_a_new_message(chain):
    from telegram.error import NetworkError, TimedOut
    for error in (TimedOut(), NetworkError("connection reset")):
        chain.scenario["edit_error"] = None
        bot.SESSIONS.end()
        buttons = open_list(chain, two_matches())
        chain.scenario["edit_error"] = error
        sent_before = len(chain.sent)
        chain.press(OWNER_ID, find(buttons, "1 Delete"))
        assert len(bot.SESSIONS.active.slip) == 1  # the change was applied
        text, kwargs = chain.sent[-1]
        assert len(chain.sent) == sent_before + 1 and "Kept 1 of 2 game(s)" in text and "(removed)" in text
        assert [t for t, _ in chain.buttons(-1)] == ["1 Keep", "1 Delete", "2 Keep", "2 Delete", "Final confirm"]
