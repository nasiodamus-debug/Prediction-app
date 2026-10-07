"""Spec 14-1, item 1: SportyBet's game list is read with page 1 first and the other pages together, with every integrity rule
of the one-by-one reader kept. Fake HTTP transports only (no network); the timing test uses fake delays of a few milliseconds."""
import asyncio
import time

import httpx
import pytest

from predictions.adapters.sportybet import fixtures as sb
from predictions.adapters.sportybet.fixtures import PAGE_SIZE, PARALLEL_PAGES, LISTINGS, SportyBetFixtures
from predictions.core.ports import FixtureFetchError

run = asyncio.run


def event(n, ref=None):
    return {"eventId": ref or f"sr:match:{n}", "estimateStartTime": 1791631800000 + n * 1000, "homeTeamName": f"H{n}",
            "awayTeamName": f"A{n}", "status": 0}


def listing(total, per_page=PAGE_SIZE, announced=None, make=event):
    """A handler-ready function: page number -> payload. Page n holds events (n-1)*per_page ... ; the last page is shorter."""
    def payload(number):
        first = (number - 1) * per_page
        events = [make(i) for i in range(first, min(total, first + per_page))]
        return {"bizCode": 10000, "data": {"totalNum": total if announced is None else announced(number),
                                           "tournaments": [{"name": "L", "categoryName": "C", "events": events}]}}
    return payload


def client_with(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def fetch(handler, **kw):
    async def go():
        async with client_with(handler) as client:
            return await SportyBetFixtures(client, **kw).fetch_fixtures()
    return run(go())


def test_the_default_is_six_pages_at_a_time_and_a_named_constant():
    assert PARALLEL_PAGES == 6 and SportyBetFixtures()._concurrency == 6


def test_every_page_is_read_once_and_the_result_keeps_page_order_whatever_finishes_first():
    payload, seen = listing(1050), []

    async def handler(request):
        n = int(request.url.params["pageNum"])
        seen.append(n)
        await asyncio.sleep(0.03 if n == 2 else 0.0)        # page 2 answers last
        return httpx.Response(200, json=payload(n))

    fixtures = fetch(handler)
    assert sorted(seen) == list(range(1, 12)) and seen[0] == 1                 # page 1 first, then the rest, each once
    assert [f.home for f in fixtures] == [f"H{i}" for i in range(1050)]        # listing order, not completion order


def test_concurrency_never_exceeds_the_bound_and_is_really_used():
    payload, state = listing(2000), {"now": 0, "max": 0}

    async def handler(request):
        state["now"] += 1
        state["max"] = max(state["max"], state["now"])
        await asyncio.sleep(0.01)
        state["now"] -= 1
        return httpx.Response(200, json=payload(int(request.url.params["pageNum"])))

    assert len(fetch(handler)) == 2000 and state["max"] == PARALLEL_PAGES
    state.update(now=0, max=0)
    fetch(handler, concurrency=2)
    assert state["max"] == 2
    state.update(now=0, max=0)
    fetch(handler, concurrency=1)
    assert state["max"] == 1


def test_parallel_is_faster_than_sequential_with_fake_delays():
    payload = listing(900)

    async def handler(request):
        await asyncio.sleep(0.05)
        return httpx.Response(200, json=payload(int(request.url.params["pageNum"])))

    started = time.perf_counter()
    fetch(handler, concurrency=1)
    sequential = time.perf_counter() - started
    started = time.perf_counter()
    fetch(handler)
    parallel = time.perf_counter() - started
    assert sequential >= 0.45 and parallel < sequential * 0.6        # 9 pages: 9 x 0.05 s against 0.05 + 2 x 0.05 s


@pytest.mark.parametrize("failing", [1, 2, 5, 9])
def test_one_failing_page_fails_the_whole_read_with_no_partial_list_and_no_retry(failing):
    payload, calls = listing(900), []

    async def handler(request):
        n = int(request.url.params["pageNum"])
        calls.append(n)
        if n == failing:
            if failing % 2:
                return httpx.Response(503)
            raise httpx.ReadTimeout("slow https://www.sportybet.com/secret", request=request)
        return httpx.Response(200, json=payload(n))

    with pytest.raises(FixtureFetchError) as info:
        fetch(handler)
    assert len(calls) == len(set(calls))                         # nothing was asked twice
    assert "http" not in str(info.value) and "sportybet.com" not in str(info.value)


def test_an_unreadable_page_in_the_middle_fails_the_whole_read():
    payload = listing(500)

    def handler(request):
        n = int(request.url.params["pageNum"])
        if n == 3:
            return httpx.Response(200, text="<html>not json</html>")
        return httpx.Response(200, json=payload(n))

    with pytest.raises(FixtureFetchError):
        fetch(handler)


def test_the_announced_total_may_drift_by_the_tolerated_amount_only():
    def handler_for(second_total):
        payload = listing(300, announced=lambda n: 300 if n != 2 else second_total)
        return lambda request: httpx.Response(200, json=payload(int(request.url.params["pageNum"])))

    assert len(fetch(handler_for(299))) == 300 and len(fetch(handler_for(302))) == 300   # drift of up to 2 games is accepted
    with pytest.raises(FixtureFetchError) as info:
        fetch(handler_for(320))
    assert "changed while" in str(info.value)


def test_a_big_listing_tolerates_half_a_percent_of_drift():
    payload = listing(1700, announced=lambda n: 1700 if n != 4 else 1707)
    assert len(fetch(lambda r: httpx.Response(200, json=payload(int(r.url.params["pageNum"]))))) == 1700


def test_games_listed_twice_are_kept_once():
    def make(i):
        return event(i, ref="sr:match:dup" if i in (99, 100) else None)       # the last game of page 1 reappears on page 2

    payload = listing(250, make=make)
    fixtures = fetch(lambda r: httpx.Response(200, json=payload(int(r.url.params["pageNum"]))))
    assert len(fixtures) == 249 and sum(1 for f in fixtures if f.bookmaker_ref == "sr:match:dup") == 1


def test_a_listing_that_ends_early_is_an_error_even_when_the_pages_are_read_together():
    def handler(request):
        n = int(request.url.params["pageNum"])
        events = [] if n > 1 else [event(1)]
        return httpx.Response(200, json={"bizCode": 10000, "data": {"totalNum": 250,
                                                                      "tournaments": [{"name": "L", "events": events}]}})

    with pytest.raises(FixtureFetchError):
        fetch(handler)


def test_a_listing_with_more_pages_than_the_safety_stop_is_refused():
    payload = listing(PAGE_SIZE * (sb.MAX_PAGES + 1), per_page=1)

    with pytest.raises(FixtureFetchError) as info:
        fetch(lambda r: httpx.Response(200, json=payload(int(r.url.params["pageNum"]))))
    assert "more pages" in str(info.value)


def test_short_pages_still_read_until_the_announced_total_one_page_at_a_time():
    payload = listing(101, per_page=2)       # pages smaller than PAGE_SIZE: the old one-by-one rule still applies
    assert len(fetch(lambda r: httpx.Response(200, json=payload(int(r.url.params["pageNum"]))))) == 101


def test_the_overall_deadline_still_applies(monkeypatch):
    monkeypatch.setattr(sb, "OVERALL_TIMEOUT_SECONDS", 0.05)
    payload = listing(500)

    async def slow(request):
        await asyncio.sleep(2)
        return httpx.Response(200, json=payload(1))

    with pytest.raises(FixtureFetchError) as info:
        fetch(slow)
    assert "too long" in str(info.value)


def test_a_request_without_a_client_sends_the_browser_user_agent_and_json_accept(monkeypatch):
    seen = {}
    payload = listing(3)

    def handler(request):
        seen.update(request.headers)
        return httpx.Response(200, json=payload(1))

    real = httpx.AsyncClient
    monkeypatch.setattr(sb.httpx, "AsyncClient",
                        lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    assert len(run(SportyBetFixtures().fetch_fixtures())) == 3
    assert "Mozilla/5.0" in seen["user-agent"] and seen["accept"] == "application/json"


@pytest.mark.parametrize("sport", sorted(LISTINGS))
def test_every_sport_that_uses_the_adapter_reads_in_parallel(sport):
    payload, state = listing(450), {"now": 0, "max": 0}

    async def handler(request):
        assert request.url.params["sportId"] == LISTINGS[sport].sport_id
        state["now"] += 1
        state["max"] = max(state["max"], state["now"])
        await asyncio.sleep(0.01)
        state["now"] -= 1
        return httpx.Response(200, json=payload(int(request.url.params["pageNum"])))

    async def go():
        async with client_with(handler) as client:
            return await SportyBetFixtures(client, sport=sport).fetch_fixtures()

    assert len(run(go())) == 450 and state["max"] > 1


def test_the_page_size_stays_within_what_sportybet_accepts():
    assert PAGE_SIZE == 100
