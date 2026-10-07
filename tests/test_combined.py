"""Combined requests: several plays merged into one list. Parsing, then the app flow with the parts faked (no network)."""
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

import app as bot
from predictions.core.model import Estimate, Evaluation
from predictions.core.types import MatchId
from predictions.flow.combined import SubRequest, parse_combined


# ---------------------------------------------------------------- parsing
def test_the_two_examples_are_read_as_the_owner_meant_them():
    assert parse_combined("Review 3ercer and America 6 October") == [SubRequest("REVIEW", "Review", "3ercer"),
                                                                       SubRequest("AMERICA", "America", "6 October")]
    assert parse_combined("Gameday, Matchday and America 5 and 6 October") == [
        SubRequest("GAMEDAY", "Gameday", "5 and 6 October"), SubRequest("MATCHDAY", "Matchday", "5 and 6 October"),
        SubRequest("AMERICA", "America", "5 and 6 October")]


def test_case_connectors_and_order_do_not_matter_and_each_part_may_have_its_own_date():
    parts = parse_combined("matchday 5 october + AMERICA 6 October & review ABC123")
    assert [(p.kind, p.argument) for p in parts] == [("MATCHDAY", "5 october"), ("AMERICA", "6 October"), ("REVIEW", "ABC123")]
    parts = parse_combined("America 5 and 6 October, Review ABC123")
    assert [(p.kind, p.argument) for p in parts] == [("AMERICA", "5 and 6 October"), ("REVIEW", "ABC123")]


def test_a_date_borrows_from_the_next_dated_play_else_the_previous_one():
    parts = parse_combined("Matchday and America 5 October and Gameday")
    assert [p.argument for p in parts] == ["5 October", "5 October", "5 October"]


def test_a_single_play_or_a_text_not_starting_with_a_trigger_is_not_combined():
    for text in ("Matchday today", "Review ABC123", "Results", "hello Matchday and Gameday today", "", None):
        assert parse_combined(text) is None


def test_unreadable_combinations_get_a_helpful_message_not_a_guess():
    for text in ("Review ABC123 and Review XYZ789", "Matchday today and Matchday tomorrow", "Review one two and Matchday today",
                 "Review and Matchday today", "Matchday and Gameday", "Matchday 5 October and Gameday 6 October and America 7 October "
                 "and Review ABC123 and Review DEF456"):
        result = parse_combined(text)
        assert isinstance(result, str) and "combined" in result.lower() or "needs a date" in str(result), text


# ---------------------------------------------------------------- the flow
def pick(home, away, selection="Over 2.5", market="Over/Under 2.5", hour=15):
    kickoff = datetime(2026, 10, 6, hour, 0, tzinfo=timezone.utc)
    mid = MatchId("soccer", "l", kickoff, home, away)
    return Evaluation(mid, market, selection, Decimal("1.80"), Estimate(0.75), Decimal("0.35"), True, "passed")


class FakeMessage:
    def __init__(self):
        self.sent = []

    async def reply_text(self, text, **kw):
        self.sent.append(text)


@pytest.fixture
def rig(monkeypatch):
    offered, saved = [], []

    async def offer_slate(update, evaluations, origin, session_id, footer=()):
        offered.append((list(evaluations), origin, session_id, tuple(footer)))

    async def save_review(update, result, session_id=None):
        saved.append(("review", session_id))

    monkeypatch.setattr(bot, "offer_slate", offer_slate)
    monkeypatch.setattr(bot, "save_review", save_review)
    monkeypatch.setattr(bot, "STORE", SimpleNamespace(save_evaluations=lambda sid, evs: saved.append(("slate", sid, len(evs)))))
    monkeypatch.setattr(bot, "approved_evaluations", lambda result: result.approved)
    message = FakeMessage()
    return SimpleNamespace(offered=offered, saved=saved, update=SimpleNamespace(message=message), message=message, monkeypatch=monkeypatch)


def gate(*picks, footer=()):
    return SimpleNamespace(passed=tuple(picks), rejected=(pick("x", "y", "Home", "1X2"),), footer=tuple(footer))


def fake(rig, name, chunks, results):
    async def collect(argument):
        collect.arguments.append(argument)
        if isinstance(results, Exception):
            raise results
        return chunks, results
    collect.arguments = []
    rig.monkeypatch.setattr(bot, name, collect)
    return collect


def test_the_parts_run_as_on_their_own_and_ONE_merged_list_is_offered(rig):
    football = fake(rig, "_matchday_collect", ["Matchday report"], [gate(pick("arsenal", "chelsea", hour=17))])
    america = fake(rig, "_america_collect", ["America report"], [gate(pick("a", "b", "Home", "Winner", hour=19), footer=["2 games skipped."])])
    review = fake(rig, "_review_collect", ["Review report"], [SimpleNamespace(approved=[pick("c", "d", hour=13)])])
    parts = parse_combined("Review ABC123 and Matchday and America 6 October")
    asyncio.run(bot._combined_request(rig.update, parts))
    assert review.arguments == ["ABC123"] and football.arguments == ["6 October"] and america.arguments == ["6 October"]
    assert len(rig.offered) == 1                                              # ONE list, not three
    evaluations, origin, session_id, footer = rig.offered[0]
    assert origin == "combined" and len(evaluations) == 3 and footer == ("2 games skipped.",)
    assert any(t.startswith("Part 1 of 3: Review ABC123") for t in rig.message.sent) and "Matchday report" in rig.message.sent
    assert ("review", session_id) in rig.saved and ("slate", session_id, 2) in rig.saved   # everything saved under the one session


def test_the_same_pick_from_two_parts_is_listed_once(rig):
    same = pick("arsenal", "chelsea")
    fake(rig, "_matchday_collect", [], [gate(same)])
    fake(rig, "_review_collect", [], [SimpleNamespace(approved=[same, pick("e", "f", "Home", "1X2", hour=18)])])
    asyncio.run(bot._combined_request(rig.update, parse_combined("Review ABC123 and Matchday today")))
    assert len(rig.offered[0][0]) == 2


def test_a_failing_part_is_reported_and_left_out_while_the_others_still_count(rig):
    fake(rig, "_gameday_collect", [], RuntimeError("ESPN is down"))
    fake(rig, "_matchday_collect", ["ok"], [gate(pick("a", "b"))])
    asyncio.run(bot._combined_request(rig.update, parse_combined("Gameday and Matchday today")))
    assert any("Gameday failed" in t and "left out of the combined list" in t for t in rig.message.sent)
    assert len(rig.offered) == 1 and len(rig.offered[0][0]) == 1


def test_when_nothing_passes_anywhere_no_list_is_offered_and_the_owner_is_told(rig):
    fake(rig, "_matchday_collect", ["none"], [gate()])
    fake(rig, "_america_collect", ["none"], [])
    asyncio.run(bot._combined_request(rig.update, parse_combined("Matchday and America today")))
    assert rig.offered == [] and "Nothing passed in the combined request" in rig.message.sent[-1]


def test_a_combined_slate_has_its_own_origin_name_for_the_replace_screen():
    from predictions.telegram.slate import ORIGIN_NAMES, _slate_words
    assert ORIGIN_NAMES["combined"] == "a combined request" and "combined request" in _slate_words("combined", 5)
