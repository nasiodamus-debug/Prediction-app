"""Story 2.1: Matchday fixtures listing (dates, selection, adapter, flow, rendering)."""
import asyncio
import json
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from openmarket import open_markets
from predictions.adapters.sportybet.fixtures import FixtureFetchError, SportyBetFixtures, parse_page
from predictions.core.dates import LAGOS, parse_dates
from predictions.core.schedule import select_fixtures
from predictions.core.types import Fixture, MatchId, normalize_team
from predictions.flow.matchday import run_matchday
from predictions.telegram.render import clean_text, MESSAGE_LIMIT

SAMPLE = json.loads((Path(__file__).parent / "data" / "sportybet_upcoming_sample.json").read_text(encoding="utf-8"))
NOW = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)  # 10:00 in Lagos


def make_fixture(kickoff, home="Home FC", away="Away FC", competition="Testland - League", started=False):
    return Fixture(MatchId("soccer", competition, kickoff, home, away), competition, home, away, kickoff, started,
                   open_markets())


def sanitize(error):
    return str(error)


class FakeSource:
    def __init__(self, fixtures=None, error=None):
        self.fixtures, self.error, self.calls = fixtures or [], error, 0

    async def fetch_fixtures(self):
        self.calls += 1
        if self.error:
            raise self.error
        return self.fixtures


def run(argument, source, now=NOW):
    return asyncio.run(run_matchday(argument, source, sanitize, now))


# ---- dates ----
def test_two_dates_with_month():
    r = parse_dates("20 and 22 September", NOW)
    assert r.ok and r.dates == [date(2026, 9, 20), date(2026, 9, 22)]


def test_comma_list_uses_current_month_and_year():
    assert parse_dates("20, 22", NOW).dates == [date(2026, 10, 20), date(2026, 10, 22)]


def test_explicit_year_and_short_month():
    assert parse_dates("5 Oct 2027", NOW).dates == [date(2027, 10, 5)]


def test_date_only_defaults_month_year():
    assert parse_dates("5", NOW).dates == [date(2026, 10, 5)]


def test_month_first_and_ordinal():
    assert parse_dates("October 5th and 6", NOW).dates == [date(2026, 10, 5), date(2026, 10, 6)]


def test_uses_lagos_date_not_utc():
    late_utc = datetime(2026, 10, 1, 23, 30, tzinfo=timezone.utc)  # already 2 Oct in Lagos
    assert parse_dates("3", late_utc).dates == [date(2026, 10, 3)]
    assert parse_dates("3 November", late_utc).dates == [date(2026, 11, 3)]


@pytest.mark.parametrize("text", ["", "   "])
def test_no_date_is_not_defaulted(text):
    r = parse_dates(text, NOW)
    assert not r.ok and not r.dates and not r.problems


@pytest.mark.parametrize("text", ["soon", "32 March", "20 and soon", "30 February"])
def test_unreadable_date_names_problem(text):
    r = parse_dates(text, NOW)
    assert not r.ok and r.problems


# ---- selection ----
def test_selects_lagos_dates_and_counts_started():
    # 23:30 UTC on 4 Oct is 00:30 on 5 Oct in Lagos.
    on_5th_lagos = make_fixture(datetime(2026, 10, 4, 23, 30, tzinfo=timezone.utc), "A", "B")
    on_4th_lagos = make_fixture(datetime(2026, 10, 4, 22, 30, tzinfo=timezone.utc), "C", "D")
    started = make_fixture(datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc), "E", "F", started=True)
    sel = select_fixtures([on_5th_lagos, on_4th_lagos, started], [date(2026, 10, 5)], NOW)
    assert sel.fixtures == [on_5th_lagos]
    assert sel.excluded_started == 1


def test_kickoff_at_or_before_now_excluded():
    at_now = make_fixture(NOW, "A", "B")
    before = make_fixture(NOW - timedelta(minutes=5), "C", "D")
    after = make_fixture(NOW + timedelta(minutes=5), "E", "F")
    sel = select_fixtures([at_now, before, after], [date(2026, 10, 1)], NOW)
    assert sel.fixtures == [after] and sel.excluded_started == 2


def test_past_date_reported_and_others_kept():
    future = make_fixture(datetime(2026, 10, 3, 15, 0, tzinfo=timezone.utc))
    sel = select_fixtures([future], [date(2026, 9, 20), date(2026, 10, 3)], NOW)
    assert sel.passed_dates == [date(2026, 9, 20)] and sel.fixtures == [future]


# ---- adapter ----
def test_parse_sample_page():
    fixtures, raw, total = parse_page(SAMPLE)
    assert (len(fixtures), raw, total) == (4, 4, 4)
    first = fixtures[0]
    assert first.home == "Arsenal" and first.kickoff_utc.tzinfo is not None and not first.started
    assert first.competition == "England - Premier League"
    one_x_two = next(m for m in first.markets if m.market_id == "1")
    assert isinstance(one_x_two.outcomes[0].odds, Decimal)


def test_started_status_detected_and_missing_fields_tolerated():
    payload = json.loads(json.dumps(SAMPLE))
    events = payload["data"]["tournaments"][0]["events"]
    events[0]["status"] = 1
    del events[1]["markets"]
    payload["data"]["tournaments"][1]["events"][0].pop("estimateStartTime")
    fixtures, raw, _ = parse_page(payload)
    assert raw == 4 and len(fixtures) == 3
    assert fixtures[0].started and fixtures[1].markets == ()


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_fetches_every_page():
    def page(n, count, total):
        events = [{"estimateStartTime": 1791631800000 + i, "homeTeamName": f"H{n}{i}", "awayTeamName": "A",
                   "status": 0} for i in range(count)]
        return {"bizCode": 10000, "data": {"totalNum": total, "tournaments": [{"name": "L", "events": events}]}}

    seen = []

    def handler(request):
        n = int(request.url.params["pageNum"])
        seen.append(n)
        return httpx.Response(200, json=page(n, 100 if n < 3 else 5, 205))

    async def go():
        async with _client(handler) as client:
            return await SportyBetFixtures(client).fetch_fixtures()

    fixtures = asyncio.run(go())
    assert seen == [1, 2, 3] and len(fixtures) == 205


@pytest.mark.parametrize("failing_page", [1, 2])
def test_http_error_or_timeout_raises_without_partial(failing_page):
    calls = []

    def handler(request):
        n = int(request.url.params["pageNum"])
        calls.append(n)
        if n == failing_page:
            if failing_page == 1:
                return httpx.Response(500)
            raise httpx.ReadTimeout("slow", request=request)
        events = [{"estimateStartTime": 1791631800000, "homeTeamName": "H", "awayTeamName": "A", "status": 0}]
        return httpx.Response(200, json={"bizCode": 10000, "data": {"totalNum": 300,
                                         "tournaments": [{"name": "L", "events": events}]}})

    async def go():
        async with _client(handler) as client:
            await SportyBetFixtures(client).fetch_fixtures()

    with pytest.raises(FixtureFetchError):
        asyncio.run(go())
    assert 1 in calls and len(calls) == len(set(calls))  # no retry: no page was asked twice (later pages are read together)


def test_listing_ending_early_is_an_error():
    def handler(request):
        n = int(request.url.params["pageNum"])
        events = [] if n > 1 else [{"estimateStartTime": 1, "homeTeamName": "H", "awayTeamName": "A", "status": 0}]
        return httpx.Response(200, json={"bizCode": 10000, "data": {"totalNum": 50,
                                         "tournaments": [{"name": "L", "events": events}]}})

    async def go():
        async with _client(handler) as client:
            await SportyBetFixtures(client).fetch_fixtures()

    with pytest.raises(FixtureFetchError):
        asyncio.run(go())


def test_adapter_error_message_hides_url():
    def handler(request):
        raise httpx.ConnectError("boom https://www.sportybet.com/secret", request=request)

    async def go():
        async with _client(handler) as client:
            await SportyBetFixtures(client).fetch_fixtures()

    with pytest.raises(FixtureFetchError) as info:
        asyncio.run(go())
    assert "http" not in str(info.value)


# ---- flow and rendering ----
def _sample_source():
    fixtures, _, _ = parse_page(SAMPLE)
    return FakeSource(fixtures)


def test_flow_lists_games_grouped_by_competition():
    fixtures, _, _ = parse_page(SAMPLE)
    lagos_date = fixtures[0].kickoff_utc.astimezone(LAGOS).date()
    now = datetime.combine(lagos_date, datetime.min.time(), tzinfo=timezone.utc) - timedelta(days=1)
    chunks = run(f"{lagos_date.day} {lagos_date.strftime('%B')} {lagos_date.year}", FakeSource(fixtures), now)
    text = "\n".join(chunks)
    assert "Arsenal vs Leeds United" in text and "England - Premier League" in text and "1 " in text
    assert "listing only" in text


def test_no_date_asks_and_does_not_fetch():
    source = _sample_source()
    for arg in ("", "   "):
        assert "Which date" in run(arg, source)[0]
    assert source.calls == 0


def test_unreadable_date_asks_and_does_not_fetch():
    source = _sample_source()
    for arg in ("soon", "32 March"):
        reply = run(arg, source)[0]
        assert "could not read" in reply or "not a real date" in reply
    assert source.calls == 0


def test_past_date_says_passed_and_lists_nothing():
    source = _sample_source()
    reply = "\n".join(run("20 September", source))
    assert "passed" in reply.lower() and source.calls == 0


def test_past_and_future_dates_together():
    future = make_fixture(datetime(2026, 10, 3, 15, 0, tzinfo=timezone.utc), "Fut", "Ure")
    reply = "\n".join(run("20 September and 3 October", FakeSource([future])))
    assert "passed" in reply.lower() and "Fut vs Ure" in reply


def test_started_games_counted_in_reply():
    future = make_fixture(datetime(2026, 10, 3, 15, 0, tzinfo=timezone.utc), "Fut", "Ure")
    started = make_fixture(datetime(2026, 10, 3, 9, 0, tzinfo=timezone.utc), "Old", "Game", started=True)
    reply = "\n".join(run("3", FakeSource([future, started])))
    assert "1 game(s) excluded" in reply and "Old vs Game" not in reply


def test_nothing_found():
    assert "No games found" in "\n".join(run("3", FakeSource([])))


def test_source_failure_gives_sanitized_error_no_partial_no_retry():
    source = FakeSource(error=FixtureFetchError("SportyBet returned HTTP 503"))
    chunks = run("3", source)
    assert len(chunks) == 1 and "HTTP 503" in chunks[0] and "Nothing was listed" in chunks[0]
    assert source.calls == 1


def test_odd_characters_cleaned_and_plain():
    kickoff = datetime(2026, 10, 3, 15, 0, tzinfo=timezone.utc)
    nasty = make_fixture(kickoff, "Bad*_[Team](x)\x00\x1b‮\nline", "Café `A`" + "z" * 200)
    text = "\n".join(run("3", FakeSource([nasty])))
    assert "\x00" not in text and "\x1b" not in text and "‮" not in text
    assert "Bad*_[Team](x)" in text and "Café" in text and "z" * 100 not in text


def test_clean_text_caps_length():
    assert len(clean_text("x" * 500)) == 60


def test_long_lists_are_chunked_under_limit():
    kickoff = datetime(2026, 10, 3, 15, 0, tzinfo=timezone.utc)
    games = [make_fixture(kickoff + timedelta(minutes=i), f"Home {i}", f"Away {i}", f"Country - League {i % 7}")
             for i in range(300)]
    chunks = run("3", FakeSource(games))
    assert len(chunks) > 1 and all(len(c) <= MESSAGE_LIMIT for c in chunks)
    assert sum(c.count(" vs ") for c in chunks) == 300


def test_normalize_team():
    assert normalize_team("  Atlético  Madrid! ") == "atletico madrid"
