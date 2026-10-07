"""Spec 14-1, item 2: the stats cache (core/ttl_cache.py) and its use by the stats adapters; the analyst cache. Fake clocks and
fake HTTP transports only: nothing here touches the network and nothing sleeps."""
import asyncio
import logging
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from predictions.adapters.espn.football import EspnFootballStats
from predictions.adapters.espn.soccer import EspnSoccerStats
from predictions.adapters.espn.stats import EspnBasketballStats
from predictions.adapters.footballdata.stats import FootballDataInputs
from predictions.adapters.mlb.stats import MlbStats
from predictions.adapters.nhl.stats import NhlStats
from predictions.adapters.understat import stats as understat
from predictions.adapters.weather import openmeteo
from predictions.adapters.weather.stadiums import Stadium
from predictions.core.engines.american_football import NFL
from predictions.core.analyst import AnalystReport, AnalystUsage, parse_report
from predictions.core.ports import AnalystUnavailable, StatsFetchError
from predictions.core.ttl_cache import (MAX_TTL_SCALE, STATS_CACHE, TTL_FORM_SECONDS, TTL_WEATHER_SECONDS, TtlCache,
                                        parse_cache_ttl_scale, parse_listing_ttl_seconds, request_key)
from predictions.flow.analyst import ANALYST_CACHE, AnalystCache, AnalystRun, DailyBudget
from test_competition_limits import FakeAnalyst, game

run = asyncio.run


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


# ------------------------------------------------------------------ the cache itself
def test_a_hit_returns_exactly_what_the_miss_returned_and_does_not_fetch_again():
    cache, calls = TtlCache(Clock()), []

    async def fetch():
        calls.append(1)
        return {"a": [1, 2]}

    first = run(cache.get_or_fetch("k", 60, fetch))
    second = run(cache.get_or_fetch("k", 60, fetch))
    assert first == second == {"a": [1, 2]} and second is first and calls == [1]
    assert cache.counts() == (1, 1)


def test_an_entry_expires_after_its_time_to_live_with_a_fake_clock():
    clock, cache, calls = Clock(), None, []
    cache = TtlCache(clock)

    async def fetch():
        calls.append(clock.now)
        return len(calls)

    assert run(cache.get_or_fetch("k", 100, fetch)) == 1
    clock.now += 99
    assert run(cache.get_or_fetch("k", 100, fetch)) == 1          # still fresh
    clock.now += 1
    assert run(cache.get_or_fetch("k", 100, fetch)) == 2          # expired exactly at the time to live: fetched again
    assert len(calls) == 2


def test_an_error_is_never_cached_and_the_next_call_fetches_again():
    cache, calls = TtlCache(Clock()), []

    async def fetch():
        calls.append(1)
        if len(calls) == 1:
            raise StatsFetchError("down")
        return "fine"

    with pytest.raises(StatsFetchError):
        run(cache.get_or_fetch("k", 60, fetch))
    assert len(cache) == 0
    assert run(cache.get_or_fetch("k", 60, fetch)) == "fine" and calls == [1, 1]


def test_a_none_answer_is_not_stored():
    cache, calls = TtlCache(Clock()), []

    async def fetch():
        calls.append(1)
        return None

    run(cache.get_or_fetch("k", 60, fetch))
    run(cache.get_or_fetch("k", 60, fetch))
    assert calls == [1, 1] and len(cache) == 0


def test_scale_scales_the_time_to_live_and_zero_switches_the_cache_off():
    clock = Clock()
    half = TtlCache(clock, scale=0.5)
    run(half.get_or_fetch("k", 100, lambda: _value(1)))
    clock.now += 49
    assert run(half.get_or_fetch("k", 100, lambda: _value(2))) == 1
    clock.now += 2
    assert run(half.get_or_fetch("k", 100, lambda: _value(3))) == 3
    off = TtlCache(clock, scale=0)
    assert not off.enabled
    assert run(off.get_or_fetch("k", 100, lambda: _value(1))) == 1
    assert run(off.get_or_fetch("k", 100, lambda: _value(2))) == 2 and len(off) == 0


async def _value(v):
    return v


def test_the_key_is_the_exact_request_in_any_parameter_order():
    assert request_key("espn", "/x", {"a": 1, "b": 2}) == request_key("espn", "/x", {"b": 2, "a": 1})
    assert request_key("espn", "/x", {"a": 1}) != request_key("espn", "/x", {"a": 2})
    assert request_key("espn", "/x") != request_key("mlb", "/x")


@pytest.mark.parametrize("raw, expected", [(None, 1.0), ("", 1.0), ("  ", 1.0), ("1", 1.0), ("0", 0.0), ("0.5", 0.5),
                                           (" 2 ", 2.0), (".25", 0.25), ("24", 24.0)])
def test_cache_ttl_scale_is_read_like_max_stake(raw, expected):
    assert parse_cache_ttl_scale(raw) == expected


@pytest.mark.parametrize("raw", ["abc", "-1", "1e3", "nan", "inf", "25", "1,5", "1.2.3", ".", "١", "0x10", "yes"])
def test_an_unparseable_cache_ttl_scale_refuses_to_start_without_echoing_it(raw):
    with pytest.raises(ValueError) as info:
        parse_cache_ttl_scale(raw)
    assert "CACHE_TTL_SCALE" in str(info.value) and (len(raw) < 3 or raw not in str(info.value))


@pytest.mark.parametrize("raw, expected", [(None, 600), ("", 600), ("0", 0), ("90", 90), (" 3600 ", 3600)])
def test_listing_ttl_seconds_is_read_strictly(raw, expected):
    assert parse_listing_ttl_seconds(raw) == expected


@pytest.mark.parametrize("raw", ["-5", "1.5", "ten", "3601", "1e2", "999999"])
def test_an_unparseable_listing_ttl_refuses_to_start(raw):
    with pytest.raises(ValueError) as info:
        parse_listing_ttl_seconds(raw)
    assert "LISTING_TTL_SECONDS" in str(info.value)


def test_the_app_refuses_to_start_on_an_unparseable_setting(monkeypatch):
    """app.py parses all three settings at import, like MAX_STAKE: a bad value stops the bot from starting."""
    import importlib
    import sys
    for name, bad in (("CACHE_TTL_SCALE", "fast"), ("LISTING_TTL_SECONDS", "soon"), ("FAST_BROWSER", "maybe")):
        monkeypatch.setenv(name, bad)
        saved = sys.modules.pop("app")
        try:
            with pytest.raises(ValueError):
                importlib.import_module("app")
        finally:
            sys.modules["app"] = saved
            monkeypatch.delenv(name)


# ------------------------------------------------------------------ the adapters use it
@pytest.fixture
def cache_on(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(STATS_CACHE, "_clock", clock)
    STATS_CACHE.set_scale(1.0)
    return clock


def counting_client(payload=None, status=200, text=None):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if text is not None:
            return httpx.Response(status, text=text)
        return httpx.Response(status, json=payload if payload is not None else {"ok": True})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), calls


GATE = lambda: asyncio.Semaphore(4)


ADAPTER_READS = [
    ("espn soccer", lambda c: EspnSoccerStats()._get(c, GATE(), "uefa.champions/scoreboard", dates="20261001")),
    ("espn basketball", lambda c: EspnBasketballStats()._get(c, GATE(), "nba", "/teams")),
    ("espn football", lambda c: EspnFootballStats()._get(c, GATE(), NFL, "/teams/1/statistics")),
    ("mlb", lambda c: MlbStats()._get(c, "/teams", sportId=1)),
    ("nhl", lambda c: NhlStats()._get(c, "/schedule/2026-10-01")),
]


@pytest.mark.parametrize("name, read", ADAPTER_READS, ids=[n for n, _ in ADAPTER_READS])
def test_a_repeated_team_stats_read_is_one_request_and_equals_the_miss(cache_on, name, read):
    async def go():
        client, calls = counting_client({"events": [], "n": name})
        async with client:
            first, second = await read(client), await read(client)
        return first, second, calls

    first, second, calls = run(go())
    assert first == second == {"events": [], "n": name} and len(calls) == 1


@pytest.mark.parametrize("name, read", ADAPTER_READS, ids=[n for n, _ in ADAPTER_READS])
def test_a_failed_stats_read_is_not_cached(cache_on, name, read):
    responses = [httpx.Response(500), httpx.Response(200, json={"events": []})]
    calls = []

    def handler(request):
        calls.append(1)
        return responses[min(len(calls), 2) - 1]

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(StatsFetchError):
                await read(client)
            return await read(client)

    assert run(go()) == {"events": []} and len(calls) == 2


@pytest.mark.parametrize("name, read", ADAPTER_READS, ids=[n for n, _ in ADAPTER_READS])
def test_cache_scale_zero_means_every_read_is_a_real_request(name, read):
    STATS_CACHE.set_scale(0)

    async def go():
        client, calls = counting_client({"events": []})
        async with client:
            await read(client)
            await read(client)
        return calls

    assert len(run(go())) == 2


def test_a_different_request_is_a_different_entry(cache_on):
    async def go():
        client, calls = counting_client()
        async with client:
            soccer = EspnSoccerStats()
            await soccer._get(client, GATE(), "uefa.champions/scoreboard", dates="20261001")
            await soccer._get(client, GATE(), "uefa.champions/scoreboard", dates="20261002")
            await soccer._get(client, GATE(), "uefa.champions/scoreboard", dates="20261001")
        return calls

    assert len(run(go())) == 2


def test_team_form_lives_three_hours_and_wind_one_hour(cache_on):
    assert TTL_FORM_SECONDS == 3 * 3600 and TTL_WEATHER_SECONDS == 3600

    async def go():
        client, calls = counting_client({"events": []})
        async with client:
            await NhlStats()._get(client, "/schedule/2026-10-01")
            cache_on.now += TTL_FORM_SECONDS - 1
            await NhlStats()._get(client, "/schedule/2026-10-01")
            assert len(calls) == 1
            cache_on.now += 1
            await NhlStats()._get(client, "/schedule/2026-10-01")
        return calls

    assert len(run(go())) == 2


def test_announced_starting_pitchers_are_remembered_for_an_hour_only(cache_on):
    async def go():
        client, calls = counting_client({"dates": []})
        async with client:
            mlb = MlbStats()
            await mlb._get(client, "/schedule", sportId=1, hydrate="probablePitcher")
            await mlb._get(client, "/schedule", sportId=1, hydrate="linescore")
            cache_on.now += TTL_WEATHER_SECONDS
            await mlb._get(client, "/schedule", sportId=1, hydrate="probablePitcher")   # expired after an hour
            await mlb._get(client, "/schedule", sportId=1, hydrate="linescore")         # form: still fresh
        return calls

    assert len(run(go())) == 3


def test_football_data_and_understat_are_cached(cache_on, monkeypatch):
    async def go():
        client, calls = counting_client(text="Date,HomeTeam\n")
        async with client:
            fd = FootballDataInputs()
            a, b = await fd._get(client, "https://example.test/E0.csv"), await fd._get(client, "https://example.test/E0.csv")
        assert a == b and len(calls) == 1
        client, calls = counting_client({"x": 1})
        monkeypatch.setattr(understat, "parse_league", lambda payload: ("parsed", payload["x"]))
        async with client:
            source = understat.UnderstatStats()
            one = await source._league(client, {}, "EPL", 2026)
            two = await source._league(client, {}, "EPL", 2026)    # a different request-local dict: the shared cache answers
        assert one == two == ("parsed", 1) and len(calls) == 1

    run(go())


def wind_payload(kickoff):
    first = kickoff.replace(minute=0, second=0, microsecond=0)
    times = [(first + timedelta(hours=h)).strftime("%Y-%m-%dT%H:00") for h in range(4)]
    return {"hourly": {"time": times, "wind_speed_10m": [5.0, 7.0, 9.0, 6.0]}}


def some_stadium():
    return Stadium("Test Field", 40.0, -75.0, "open")


def test_the_wind_forecast_is_cached_for_an_hour_and_a_failed_read_is_not(cache_on):
    now = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    kickoff = now + timedelta(days=1)
    stadium = some_stadium()
    state = {"fail": True, "calls": 0}

    def handler(request):
        state["calls"] += 1
        if state["fail"]:
            return httpx.Response(500)
        return httpx.Response(200, json=wind_payload(kickoff))

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            assert await openmeteo.fetch_wind(client, stadium, kickoff, now) is None       # failed: not remembered
            state["fail"] = False
            first = await openmeteo.fetch_wind(client, stadium, kickoff, now)
            second = await openmeteo.fetch_wind(client, stadium, kickoff, now)
            return first, second

    first, second = run(go())
    assert first == second == 9.0 and state["calls"] == 2


def test_the_per_request_hit_and_miss_counts_are_logged_at_info(cache_on, caplog):
    from test_competition_limits import Listing, Table, good_stats, NOW
    from predictions.flow.matchday import run_matchday

    class CachedStats:
        async def fetch_stats(self, fixtures):
            async def fetch():
                return 1
            await STATS_CACHE.get_or_fetch("k", 60, fetch)
            return {f.match_id: good_stats(f) for f in fixtures}

    fixtures = [game("Alpha", "Beta", "England - Premier League")]
    with caplog.at_level(logging.INFO):
        run(run_matchday("1 October", Listing(fixtures), str, NOW, stats_source=CachedStats()))
        run(run_matchday("1 October", Listing(fixtures), str, NOW, stats_source=CachedStats()))
    lines = [r.getMessage() for r in caplog.records if "stats cache" in r.getMessage()]
    assert lines[0].startswith("Matchday request: stats cache 0 hit(s), 1 miss(es)")
    assert lines[1].startswith("Matchday request: stats cache 1 hit(s), 0 miss(es)")


# ------------------------------------------------------------------ the analyst cache
class CountingAnalyst(FakeAnalyst):
    pass


def ask(run_, fixture, wanted=("home.matches",)):
    return asyncio.run(run_.ask(fixture, wanted))


def test_a_repeated_fixture_and_inputs_cost_nothing_and_use_no_limit(tmp_path):
    from predictions.flow.store import EvaluationStore
    now = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    with EvaluationStore.open(tmp_path / "a.db", str) as store:
        analyst = FakeAnalyst()
        budget = DailyBudget(store, 5, now)
        first = AnalystRun(analyst, limit=3, budget=budget, clock=lambda: now)
        f = game("Alpha", "Beta", "Netherlands - Eredivisie")
        report = ask(first, f)
        assert first.calls == 1 and budget.used() == 1 and first.cost > 0
        second = AnalystRun(analyst, limit=3, budget=DailyBudget(store, 5, now), clock=lambda: now)
        again = ask(second, f)
        assert again is report and len(analyst.queries) == 1                 # no second call
        assert second.calls == 0 and second.cost == 0 and second.cache_hits == 1
        assert DailyBudget(store, 5, now).used() == 1                        # the daily count did not move
        assert not second.limit_hit and "reused from today's cache at no cost" in second.short_summary()
        assert "no cost" in second.summary() and "0 of 3 calls" in second.summary()


def test_the_cache_key_is_the_fixture_and_the_requested_inputs():
    now = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    analyst = FakeAnalyst()
    f, g = game("Alpha", "Beta", "Netherlands - Eredivisie"), game("Gamma", "Delta", "Netherlands - Eredivisie")
    run_ = AnalystRun(analyst, limit=10, clock=lambda: now)
    ask(run_, f)
    ask(run_, f, ("home.matches", "away.matches"))   # other inputs: a new call
    ask(run_, f, ())                                  # flags only: a new call
    ask(run_, g)                                      # another fixture: a new call
    assert run_.calls == 4 and run_.cache_hits == 0
    ask(run_, f)
    ask(run_, g)
    assert run_.calls == 4 and run_.cache_hits == 2


def test_the_analyst_cache_lasts_the_lagos_day_only():
    cache = AnalystCache()
    now = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    report = AnalystReport()
    cache.put("k", report, now)
    assert cache.get("k", now + timedelta(hours=13, minutes=59)) is report      # 23:59 Lagos, same day
    assert cache.get("k", now + timedelta(hours=14)) is None                    # midnight in Lagos: a new day


def test_a_malformed_answer_and_an_outage_are_never_cached():
    now = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    f = game("Alpha", "Beta", "Netherlands - Eredivisie")

    class Bad:
        calls = 0

        async def research(self, query):
            Bad.calls += 1
            return parse_report("not json", query, AnalystUsage(10, 1, 0))

    run_ = AnalystRun(Bad(), clock=lambda: now)
    assert ask(run_, f).rejected and ask(run_, f).rejected and Bad.calls == 2 and len(ANALYST_CACHE) == 0

    class Down:
        async def research(self, query):
            raise AnalystUnavailable("down")

    out = AnalystRun(Down(), clock=lambda: now)
    assert ask(out, f) is None and out.unavailable and len(ANALYST_CACHE) == 0


def test_cache_scale_zero_switches_the_analyst_cache_off():
    now = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    ANALYST_CACHE.enabled = False
    analyst = FakeAnalyst()
    f = game("Alpha", "Beta", "Netherlands - Eredivisie")
    run_ = AnalystRun(analyst, clock=lambda: now)
    ask(run_, f)
    ask(run_, f)
    assert run_.calls == 2 and run_.cache_hits == 0 and len(analyst.queries) == 2


def test_the_analyst_cache_holds_only_schema_validated_reports_and_nothing_from_the_account():
    """What is kept is the AnalystReport the analyst port returned (stats and flags), keyed by fixture identity."""
    now = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
    f = game("Alpha", "Beta", "Netherlands - Eredivisie")
    ask(AnalystRun(FakeAnalyst(), clock=lambda: now), f)
    values = list(ANALYST_CACHE._items.values())
    assert len(values) == 1 and isinstance(values[0], AnalystReport)
