"""Story 2.1 review fixes: dates, adapter guards, flow errors, odds display, ordering, app wiring."""
import asyncio
import json
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import app as bot
from handled import drop_ack, handled
from predictions.adapters.sportybet import fixtures as sb
from predictions.adapters.sportybet.fixtures import SportyBetFixtures, parse_page
from predictions.core.dates import parse_dates
from predictions.core.ports import FixtureFetchError
from predictions.core.schedule import select_fixtures
from predictions.core.types import Fixture, MarketOffer, MatchId, Outcome
from predictions.flow.matchday import run_matchday
from predictions.telegram.render import main_odds, render_matchday

NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)
SAMPLE = Path(__file__).parent / "data" / "sportybet_upcoming_sample.json"


def dates(text):
    return parse_dates(text, NOW)


# ---------- dates ----------

def test_a_day_after_a_month_uses_that_month():
    result = dates("20 September and 22")
    assert result.ok and result.dates == [date(2026, 9, 20), date(2026, 9, 22)]
    result = dates("5 Oct, 6")
    assert result.ok and result.dates == [date(2026, 10, 5), date(2026, 10, 6)]


def test_month_and_year_carry_to_later_days():
    result = dates("20 and 22 September 2027 and 5")
    assert result.ok and result.dates == [date(2027, 9, 5), date(2027, 9, 20), date(2027, 9, 22)]


def test_month_first_still_works():
    assert dates("October 5th and 6").dates == [date(2026, 10, 5), date(2026, 10, 6)]


def test_a_month_with_no_day_is_reported():
    result = dates("September 20 and 22 October")
    assert not result.ok and any('"October"' in p for p in result.problems)


@pytest.mark.parametrize("text", ["20/09", "20-09-2026", "20-22 September", "20 - 22 September", "5/10/2026"])
def test_numeric_dates_and_ranges_are_never_guessed(text):
    result = dates(text)
    assert not result.ok and result.dates == []
    assert any("not supported" in p for p in result.problems)


def test_huge_day_number_is_a_problem_not_a_crash():
    result = dates("99999999999999999999 September")
    assert not result.ok and result.problems


def test_year_zero_is_rejected_not_replaced_by_this_year():
    result = dates("5 September 0000")
    assert not result.ok and any("not a real date" in p for p in result.problems)


def test_ordinals_in_capitals():
    assert dates("5TH October").dates == [date(2026, 10, 5)]
    assert dates("1ST and 2ND October").dates == [date(2026, 10, 1), date(2026, 10, 2)]


# ---------- adapter guards ----------

def event(i, **extra):
    base = {"eventId": f"e{i}", "estimateStartTime": 1790880300000 + i * 60000, "status": 0,
            "homeTeamName": f"Home {i}", "awayTeamName": f"Away {i}", "markets": []}
    base.update(extra)
    return base


def page(events, total, **tournament_extra):
    tournament = {"id": "t", "name": "League", "categoryName": "Country", "events": events}
    tournament.update(tournament_extra)
    return {"bizCode": 10000, "data": {"totalNum": total, "tournaments": [tournament]}}


@pytest.mark.parametrize("total", [True, False, -1, "5", 3.0, None])
def test_total_must_be_a_real_non_negative_integer(total):
    with pytest.raises(FixtureFetchError):
        parse_page(page([event(1)], total))


def test_events_must_be_a_list():
    with pytest.raises(FixtureFetchError):
        parse_page(page({"a": 1}, 1))


@pytest.mark.parametrize("extra,started", [
    ({"status": "0"}, False),           # a string zero still means not started
    ({"status": 1}, True),
    ({"status": "Ended"}, True),        # an unknown status is never offered
    ({"status": None, "matchStatus": "Ended"}, True),
    ({"status": None, "matchStatus": "Not start"}, False),
])
def test_started_detection(extra, started):
    fixtures, _, _ = parse_page(page([event(1, **extra)], 1))
    assert fixtures[0].started is started


def test_a_missing_status_and_match_status_is_not_started():
    ev = event(1)
    del ev["status"]
    fixtures, _, _ = parse_page(page([ev], 1))
    assert fixtures[0].started is False


def client_for(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_a_listing_that_changes_between_pages_is_an_error():
    calls = []

    async def handler(request):
        number = int(request.url.params["pageNum"])
        calls.append(number)
        body = page([event(number)], 3 if number == 1 else 40)  # a real change, not the usual flicker
        return httpx.Response(200, json=body)

    async def run():
        async with client_for(handler) as client:
            return await SportyBetFixtures(client).fetch_fixtures()

    with pytest.raises(FixtureFetchError) as err:
        asyncio.run(run())
    assert "changed while" in str(err.value)


def test_the_whole_listing_has_an_overall_deadline(monkeypatch):
    monkeypatch.setattr(sb, "OVERALL_TIMEOUT_SECONDS", 0.05)

    async def slow(request):
        await asyncio.sleep(2)
        return httpx.Response(200, json=page([event(1)], 1))

    async def run():
        async with client_for(slow) as client:
            return await SportyBetFixtures(client).fetch_fixtures()

    with pytest.raises(FixtureFetchError) as err:
        asyncio.run(run())
    assert "too long" in str(err.value)


def test_a_wrong_biz_code_is_an_error():
    body = page([event(1)], 1)
    body["bizCode"] = 10001
    with pytest.raises(FixtureFetchError):
        parse_page(body)


# ---------- flow ----------

class BrokenSource:
    def __init__(self, error):
        self.error = error

    async def fetch_fixtures(self):
        raise self.error


def test_a_bug_is_not_reported_as_a_sportybet_problem():
    """Only fetch failures are explained to the owner; anything else reaches the bot's error handler."""
    with pytest.raises(TypeError):
        asyncio.run(run_matchday(" 2 October", BrokenSource(TypeError("boom")), str, now=NOW))


def test_a_fetch_failure_is_explained_and_nothing_is_listed():
    replies = asyncio.run(run_matchday(" 2 October", BrokenSource(FixtureFetchError("SportyBet returned HTTP 503")),
                                       str, now=NOW))
    assert len(replies) == 1 and "HTTP 503" in replies[0] and "Nothing was listed" in replies[0]


def test_many_unreadable_words_give_a_short_reply():
    replies = asyncio.run(run_matchday(" a b c d e f g h i j", BrokenSource(RuntimeError()), str, now=NOW))
    assert len(replies) == 1 and len(replies[0]) < 600 and "more" in replies[0]


# ---------- odds display ----------

def offer(market_id, specifier, *pairs):
    return MarketOffer(market_id, "m", specifier, tuple(Outcome(name, Decimal(odds)) for name, odds in pairs))


def make_fixture(*markets, comp="England - Premier League", home="Arsenal", away="Leeds", hour=10, day=2):
    kickoff = datetime(2026, 10, day, hour, 0, tzinfo=timezone.utc)
    return Fixture(MatchId("soccer", comp, kickoff, home, away), comp, home, away, kickoff, False, tuple(markets))


def test_the_recorded_sample_shows_the_exact_odds_line():
    fixtures, _, _ = parse_page(json.loads(SAMPLE.read_text(encoding="utf-8")))
    arsenal = next(f for f in fixtures if f.home == "Arsenal")
    assert main_odds(arsenal) == "1 1.43 X 4.88 2 7.36 | O/U 2.5: 1.68/2.15"


def test_odds_follow_the_names_not_the_order():
    reordered = offer("1", "", ("Away", "7.36"), ("Draw", "4.88"), ("Home", "1.43"))
    assert main_odds(make_fixture(reordered)) == "1 1.43 X 4.88 2 7.36"


def test_only_the_2_5_line_is_shown():
    other = offer("18", "total=1.5", ("Over 1.5", "1.20"), ("Under 1.5", "4.00"))
    assert main_odds(make_fixture(other)) == "odds not listed"
    numeric = offer("18", "total=2.50", ("Over 2.5", "1.68"), ("Under 2.5", "2.15"))
    assert main_odds(make_fixture(numeric)) == "O/U 2.5: 1.68/2.15"


def test_a_market_missing_an_outcome_is_not_shown():
    partial = offer("1", "", ("Home", "1.43"), ("Away", "7.36"))
    assert main_odds(make_fixture(partial)) == "odds not listed"


# ---------- selection and layout ----------

def test_a_repeated_game_is_listed_once():
    game = make_fixture()
    selection = select_fixtures([game, game], [date(2026, 10, 2)], NOW)
    assert len(selection.fixtures) == 1


def test_games_are_grouped_by_competition_in_order_of_first_kickoff():
    fixtures = [
        make_fixture(comp="A League", home="A1", away="A2", hour=9),
        make_fixture(comp="B League", home="B1", away="B2", hour=10),
        make_fixture(comp="A League", home="A3", away="A4", hour=12),
    ]
    text = "\n".join(render_matchday(select_fixtures(fixtures, [date(2026, 10, 2)], NOW), [date(2026, 10, 2)]))
    assert text.count("A League") == 1 and text.count("B League") == 1
    assert text.index("A League") < text.index("A1 vs A2") < text.index("A3 vs A4") < text.index("B League")
    assert text.index("B League") < text.index("B1 vs B2")


# ---------- the Telegram handler ----------

def owner_message(text):
    sent = []

    async def reply_text(message, **kwargs):
        sent.append((message, kwargs))

    update = SimpleNamespace(effective_user=SimpleNamespace(id=bot.OWNER_ID), callback_query=None,
                             message=SimpleNamespace(text=text, reply_text=reply_text))
    return update, sent


@pytest.mark.parametrize("text,expected", [
    ("Matchday 20 and 22 September", " 20 and 22 September"),
    ("matchday 5", " 5"),
    ("MATCHDAY", ""),
])
def test_matchday_hands_over_only_the_date_and_sends_plain_text(monkeypatch, text, expected):
    seen = []

    async def fake_run(argument, source, sanitize, **kwargs):
        seen.append(argument)
        return ["first chunk", "second chunk"]

    monkeypatch.setattr(bot, "run_matchday", fake_run)
    update, sent = owner_message(text)
    asyncio.run(handled(update))
    drop_ack(sent)
    assert seen == [expected]
    assert [m for m, _ in sent] == ["first chunk", "second chunk"]  # no "Processing" message, no legacy engine
    assert all("parse_mode" not in kwargs for _, kwargs in sent)



def test_a_flicker_of_one_game_in_the_announced_total_is_tolerated():
    """The live listing announces 1472 on one page and 1473 on the next; that must not fail the read."""
    async def handler(request):
        number = int(request.url.params["pageNum"])
        events = [event(number * 10 + i) for i in range(2)]
        return httpx.Response(200, json=page(events, 4 if number == 1 else 5))

    async def run():
        async with client_for(handler) as client:
            return await SportyBetFixtures(client).fetch_fixtures()

    assert len(asyncio.run(run())) == 4
