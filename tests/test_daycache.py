"""Spec 14-1: the daily games cache (flow/daycache.py): what it holds, when it refreshes, how a request uses it, the 06:00 job, the
Cache trigger. Fake listings and fake clocks only: nothing touches SportyBet and nothing sleeps for long."""
import asyncio
import inspect
import re
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from telegram import Update, User
from telegram.ext import ApplicationBuilder, ExtBot

import app as bot
from handled import handled
from openmarket import open_markets
from predictions.core.dates import LAGOS
from predictions.core.ports import FixtureFetchError
from predictions.core.ttl_cache import STATS_CACHE
from predictions.core.types import Fixture, MatchId
from predictions.flow import daycache
from predictions.flow.america import MultiSportFixtures, run_america
from predictions.flow.daycache import (DAY_CACHE_DAYS, DAY_CACHE_JOB_NAME, DAY_CACHE_TIME, PREWARM_CONCURRENCY, DayCache,
                                       odds_as_of_line, window_days)
from predictions.flow.gameday import run_gameday
from predictions.flow.matchday import run_matchday
from predictions.flow.review import run_review
from predictions.flow.stats import gather_stats
from test_competition_limits import Table, good_stats, game
from test_owner_gate import feed, message as telegram_message, wired  # noqa: F401  (wired is a fixture)

run = asyncio.run
NOW = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)      # 10:00 in Lagos
PL, FA_CUP = "England - Premier League", "England - FA Cup"


class Listing:
    """A fake SportyBet listing that counts reads, can be slow or fail, and tracks how many reads are in flight."""

    peak, now = 0, 0

    def __init__(self, fixtures, delay=0.0, error=None):
        self.fixtures, self.delay, self.error, self.calls = list(fixtures), delay, error, 0

    async def fetch_fixtures(self):
        self.calls += 1
        Listing.now += 1
        Listing.peak = max(Listing.peak, Listing.now)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.error:
                raise self.error
            return list(self.fixtures)
        finally:
            Listing.now -= 1


@pytest.fixture(autouse=True)
def _reset_listing_counters():
    Listing.peak = Listing.now = 0


class Clock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now


def soccer_day():
    """Today (1 Oct Lagos): two allowed games, one cup game, one that already started, one not open for betting, and one
    allowed game on 3 Oct that is outside the two days."""
    started = game("Old", "Game", PL, hour=8)                        # 09:00 Lagos: before NOW
    return [game("Alpha", "Beta", PL, hour=16), game("Gamma", "Delta", PL, hour=18),
            game("Cup", "Tie", FA_CUP, hour=17), started,
            game("Shut", "Door", PL, hour=19, markets=()),
            game("Late", "Night", PL, hour=23),                      # 00:00 on 2 Oct in Lagos: tomorrow
            game("Fri", "Day", PL, hour=16, day=2),                  # 2 Oct 17:00 Lagos: tomorrow
            game("Far", "Away", PL, hour=16, day=3)]                 # 3 Oct: outside the window


def make_cache(soccer=None, basketball=None, america=None, clock=None, ttl=600, lock=None, warmers=None):
    sources = {"Matchday": soccer or Listing(soccer_day()), "Gameday": basketball or Listing([]),
               "America": america or Listing([])}
    clock = clock or Clock()
    return DayCache(sources, clock, ttl, lock, warmers), sources, clock


# ------------------------------------------------------------------ what is held
def test_two_lagos_days_is_one_named_constant_used_everywhere():
    assert DAY_CACHE_DAYS == 2 and window_days(date(2026, 10, 1)) == (date(2026, 10, 1), date(2026, 10, 2))
    cache, _, _ = make_cache()
    run(cache.build_all())
    assert f"{DAY_CACHE_DAYS} days" in cache.describe()


def test_the_cache_holds_only_allowed_open_not_started_games_of_today_and_tomorrow():
    cache, sources, _ = make_cache()
    report = run(cache.build_all())
    snap = cache.snapshot("Matchday")
    assert [f.home for f in snap.fixtures] == ["Alpha", "Gamma", "Late", "Fri"]         # by kickoff
    names = {f.home for f in snap.fixtures}
    assert not names & {"Cup", "Old", "Shut", "Far"}              # not allowed / started / not open / a third day
    assert report.built == {"Matchday": 4, "Gameday": 0, "America": 0} and not report.failed
    assert snap.ignored == {date(2026, 10, 1): 1} and snap.not_open == {date(2026, 10, 1): 1}
    assert snap.snapshot_at == NOW and all(f.markets for f in snap.fixtures)           # the odds snapshot is kept


def test_the_two_days_are_lagos_calendar_days_not_utc_days():
    late = game("Late", "Night", PL, hour=23)                       # 23:00 UTC on 1 Oct = 00:00 on 2 Oct in Lagos
    later = game("Later", "Night", PL, hour=23, day=2)              # 00:00 on 3 Oct in Lagos: outside
    cache, _, _ = make_cache(Listing([late, later]))
    run(cache.build_all())
    assert [f.home for f in cache.snapshot("Matchday").fixtures] == ["Late"]


def test_each_trigger_keeps_only_its_own_allowed_competitions():
    nba = game("Lakers", "Celtics", "USA - NBA", sport="basketball", hour=22)
    euro_cup = game("X", "Y", "Region - Basketball League", sport="basketball", hour=22)   # a test-only allowed name
    junk = game("Z", "W", "Australia - NBL", sport="basketball", hour=22)
    mlb = game("Yankees", "Mets", "USA - MLB", sport="baseball", hour=22)
    nfl = game("Chiefs", "Bills", "USA - NFL", sport="american_football", hour=22)
    cfl = game("Argos", "Lions", "Canada - CFL", sport="american_football", hour=22)
    america = MultiSportFixtures({"baseball": Listing([mlb]), "american_football": Listing([nfl, cfl])})
    cache, _, _ = make_cache(basketball=Listing([nba, euro_cup, junk]), america=america)
    run(cache.build_all())
    assert {f.home for f in cache.snapshot("Gameday").fixtures} == {"Lakers", "X"}
    assert {f.home for f in cache.snapshot("America").fixtures} == {"Yankees", "Chiefs"}


def test_the_three_listings_are_read_at_the_same_time_under_the_one_sportybet_lock():
    async def go():
        lock = asyncio.Lock()
        sources = {t: Listing([], delay=0.02) for t in daycache.TRIGGERS}
        cache = DayCache(sources, Clock(), 600, lock)
        held = []
        for source in sources.values():
            real = source.fetch_fixtures

            async def watched(real=real):
                held.append(lock.locked())
                return await real()
            source.fetch_fixtures = watched
        await cache.build_all()
        return held

    held = run(go())
    assert Listing.peak == 3 and held == [True, True, True]          # concurrent, and inside the one lock (AD-17)


def test_a_listing_that_fails_leaves_that_trigger_empty_and_keeps_the_others():
    cache, _, _ = make_cache(basketball=Listing([], error=FixtureFetchError("SportyBet returned HTTP 503")))
    report = run(cache.build_all())
    assert report.failed == {"Gameday": "SportyBet returned HTTP 503"} and "Matchday" in report.built
    assert cache.snapshot("Gameday") is None and cache.snapshot("Matchday") is not None
    assert "Gameday: nothing held (SportyBet returned HTTP 503)" in cache.describe()


def test_nothing_is_written_to_sqlite_and_the_flow_module_does_not_import_it():
    source = Path(daycache.__file__).read_text(encoding="utf-8")
    assert "sqlite3" not in source and "store" not in source.replace("restore", "")


# ------------------------------------------------------------------ pre-warming the stats
def test_the_prewarm_covers_only_the_cached_games_with_bounded_concurrency_and_never_the_analyst():
    seen, state = {}, {"now": 0, "peak": 0}

    def warmer(trigger):
        async def warm(fixtures):
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
            seen[trigger] = [f.home for f in fixtures]
            await asyncio.sleep(0.01)
            state["now"] -= 1
        return warm

    america = MultiSportFixtures({"baseball": Listing([game("Yankees", "Mets", "USA - MLB", sport="baseball", hour=22)])})
    cache, _, _ = make_cache(america=america, basketball=Listing([game("L", "C", "USA - NBA", sport="basketball", hour=22)]),
                             warmers={t: warmer(t) for t in daycache.TRIGGERS})
    run(cache.build_all())
    assert sorted(seen["Matchday"]) == ["Alpha", "Fri", "Gamma", "Late"]            # not the cup game, the started one, ...
    assert seen["Gameday"] == ["L"] and seen["America"] == ["Yankees"]
    assert state["peak"] <= PREWARM_CONCURRENCY and cache.warm_state == "done"
    source = inspect.getsource(daycache)
    assert "analyst" not in source.lower().replace("never calls the analyst", "").replace("the analyst is never called", "")


def test_a_failing_prewarm_costs_only_the_head_start():
    async def boom(fixtures):
        raise RuntimeError("down")

    cache, _, _ = make_cache(warmers={"Matchday": boom})
    run(cache.build_all())
    assert cache.snapshot("Matchday") is not None and "failed" in cache.warm_state


def test_the_app_warmers_fetch_stats_of_the_cached_games_and_have_no_analyst(monkeypatch):
    asked = []

    class Source:
        async def fetch_stats(self, fixtures):
            asked.append([f.home for f in fixtures])
            return {f.match_id: good_stats(f) for f in fixtures}

    class Extra:
        async def fetch_inputs(self, fixtures):
            asked.append(("extra", [f.home for f in fixtures]))
            return {}

    monkeypatch.setattr(bot, "soccer_stats_source", lambda: Source())
    monkeypatch.setattr(bot, "FootballDataInputs", lambda: Extra())
    monkeypatch.setattr(bot, "EspnBasketballStats", lambda: Source())
    run(bot._warm_soccer([game("Alpha", "Beta", PL)]))
    run(bot._warm_basketball([game("L", "C", "USA - NBA", sport="basketball")]))
    assert asked[0] == ["Alpha"] and asked[1] == ("extra", ["Alpha"]) and asked[2] == ["L"]
    for warm in (bot._warm_soccer, bot._warm_basketball, bot._warm_america):
        assert "analyst" not in inspect.getsource(warm).lower()


def test_prewarmed_stats_make_the_first_request_a_cache_hit_not_a_miss(caplog):
    """Warm the stats cache through the daily build, then run a real request: its reads are hits."""
    import logging
    STATS_CACHE.set_scale(1.0)
    reads = []

    class CachedSource:
        async def fetch_stats(self, fixtures):
            for f in fixtures:
                async def fetch(f=f):
                    reads.append(f.home)
                    return 1
                await STATS_CACHE.get_or_fetch(("team", f.home), 3600, fetch)
            return {f.match_id: good_stats(f) for f in fixtures}

    async def warm(fixtures):
        await gather_stats(fixtures, CachedSource())

    cache, _, _ = make_cache(warmers={"Matchday": warm})
    run(cache.build_all())
    warmed = list(reads)
    assert sorted(warmed) == ["Alpha", "Fri", "Gamma", "Late"]
    with caplog.at_level(logging.INFO):
        run(run_matchday("1 October", Listing([]), str, NOW, stats_source=CachedSource(), day_cache=cache))
    assert reads == warmed                                           # the request fetched nothing new
    assert any("stats cache 2 hit(s), 0 miss(es); games from the daily games cache" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------ how a request uses it
def request(cache, source, argument="1 October", now=NOW, **kw):
    gates = []
    stats = Table(good_stats)
    chunks = run(run_matchday(argument, source, str, now, stats_source=stats, on_result=gates.append, day_cache=cache, **kw))
    return chunks, (gates[0] if gates else None), stats


def test_a_fresh_request_makes_no_full_listing_read_and_lists_only_the_cached_allowed_games():
    cache, sources, clock = make_cache()
    run(cache.build_all())
    live = Listing(soccer_day())
    clock.now = NOW + timedelta(minutes=5)
    chunks, gate, stats = request(cache, live, now=clock.now)
    assert live.calls == 0 and sources["Matchday"].calls == 1          # only the build read the listing
    assert set(stats.asked) == {"Alpha", "Gamma"}          # 1 October only: Late (00:00) and Fri are 2 Oct in Lagos


def test_the_list_says_when_its_odds_were_read_in_one_short_footer_line_only_for_cached_lists():
    cache, _, clock = make_cache()
    run(cache.build_all())
    clock.now = NOW + timedelta(minutes=3)
    _, gate, _ = request(cache, Listing(soccer_day()), now=clock.now)
    lines = [l for l in gate.footer if l.startswith("Odds as of")]
    assert lines == ["Odds as of 10:00 (Lagos)"] == [odds_as_of_line(NOW)]
    assert any("ignored" in l for l in gate.footer)                      # the cache carries the ignored-games count too
    live = Listing(soccer_day())
    _, gate, _ = request(None, live)                                      # the normal path: live odds, no such line
    assert live.calls == 1 and not any(l.startswith("Odds as of") for l in gate.footer)


def test_the_footer_time_is_lagos_time():
    assert odds_as_of_line(datetime(2026, 10, 1, 13, 5, tzinfo=timezone.utc)) == "Odds as of 14:05 (Lagos)"


def test_refresh_policy_by_age_of_the_odds_snapshot():
    cache, sources, clock = make_cache(ttl=600)
    run(cache.build_all())
    assert sources["Matchday"].calls == 1
    live = Listing(soccer_day())
    for minutes, expected_reads in ((1, 1), (9, 1), (10, 2), (15, 2), (21, 3)):
        clock.now = NOW + timedelta(minutes=minutes)
        request(cache, live, now=clock.now)
        assert sources["Matchday"].calls == expected_reads, minutes   # a snapshot is refreshed once it is 10 minutes old
    assert live.calls == 0                                            # and the request never read the listing itself


def test_an_older_snapshot_is_refreshed_from_the_live_listing_and_the_odds_are_updated():
    first = [game("Alpha", "Beta", PL, hour=16)]
    cache, sources, clock = make_cache(Listing(first))
    run(cache.build_all())
    new_markets = tuple(replace(m, outcomes=tuple(replace(o, odds=o.odds + 1) for o in m.outcomes)) for m in open_markets("soccer"))
    sources["Matchday"].fixtures = [replace(first[0], markets=new_markets), game("Newly", "Listed", PL, hour=17)]
    clock.now = NOW + timedelta(minutes=11)
    got = run(cache.read("Matchday", [date(2026, 10, 1)], clock.now))
    assert [f.home for f in got.fixtures] == ["Alpha", "Newly"]          # games added since are picked up
    assert got.fixtures[0].markets == new_markets and got.snapshot_at == clock.now


def test_a_game_that_has_started_is_dropped_even_though_it_is_cached():
    cache, _, clock = make_cache(Listing([game("Alpha", "Beta", PL, hour=10, day=1)]), ttl=7200)  # 10:00 UTC, 11:00 Lagos
    run(cache.build_all())
    assert [f.home for f in cache.snapshot("Matchday").fixtures] == ["Alpha"]
    clock.now = NOW + timedelta(hours=1, minutes=30)                      # 10:30 UTC: kicked off; the snapshot is still fresh
    live = Listing([])
    chunks, gate, stats = request(cache, live, now=clock.now)
    assert live.calls == 0 and stats.asked == [] and "Alpha" not in "\n".join(chunks)


def test_a_cached_game_is_rechecked_for_betting_status_on_the_snapshot_a_request_uses():
    fixture = game("Alpha", "Beta", PL, hour=16)
    cache, sources, clock = make_cache(Listing([fixture]))
    run(cache.build_all())
    # (a) the snapshot is old: it is refreshed first, SportyBet now lists the game with no open market, and it is not analysed
    sources["Matchday"].fixtures = [replace(fixture, markets=())]
    clock.now = NOW + timedelta(minutes=30)
    _, _, stats = request(cache, Listing([]), now=clock.now)
    assert stats.asked == [] and cache.snapshot("Matchday").fixtures == () and cache.snapshot("Matchday").not_open
    # (b) even a snapshot young enough to be used as it is goes through the same open-for-betting check in the request
    sources["Matchday"].fixtures = [fixture]
    run(cache.refresh("Matchday", clock.now, force=True))
    snap = cache.snapshot("Matchday")
    assert [f.home for f in snap.fixtures] == ["Alpha"]
    snap.fixtures = (replace(snap.fixtures[0], markets=()),)
    _, _, stats = request(cache, Listing([]), now=clock.now)
    assert stats.asked == []


def test_a_date_outside_the_cached_two_days_falls_back_to_the_normal_path():
    cache, sources, _ = make_cache()
    run(cache.build_all())
    live = Listing(soccer_day())
    request(cache, live, "3 October")
    assert live.calls == 1 and sources["Matchday"].calls == 1
    request(cache, live, "1 October and 3 October")                      # one date outside: the whole request is normal
    assert live.calls == 2
    request(cache, live, "tomorrow")
    assert live.calls == 2                                               # inside: from the cache
    request(cache, live, "20 September and 2 October")                   # a passed date does not disqualify the rest
    assert live.calls == 2


def test_an_empty_cache_is_built_lazily_by_the_first_request_for_its_own_trigger_only():
    cache, sources, _ = make_cache()
    live = Listing(soccer_day())
    chunks, gate, stats = request(cache, live)
    assert sources["Matchday"].calls == 1 and live.calls == 0 and sources["Gameday"].calls == sources["America"].calls == 0
    assert cache.snapshot("Matchday") is not None and set(stats.asked) == {"Alpha", "Gamma"}


def test_a_snapshot_from_a_previous_lagos_day_is_rebuilt_even_when_young():
    clock = Clock(datetime(2026, 10, 1, 22, 55, tzinfo=timezone.utc))      # 23:55 Lagos
    cache, sources, _ = make_cache(clock=clock, ttl=3600)
    run(cache.build_all())
    clock.now = datetime(2026, 10, 1, 23, 5, tzinfo=timezone.utc)          # 00:05 on 2 Oct in Lagos: 10 minutes old, new day
    got = run(cache.read("Matchday", [date(2026, 10, 2)], clock.now))
    assert sources["Matchday"].calls == 2 and got.snapshot_at == clock.now
    assert cache.snapshot("Matchday").day == date(2026, 10, 2)


def test_if_the_refresh_fails_the_request_takes_the_normal_path_and_reports_the_real_problem():
    cache, sources, _ = make_cache(Listing([], error=FixtureFetchError("SportyBet returned HTTP 503")))
    live = Listing([], error=FixtureFetchError("SportyBet returned HTTP 503"))
    chunks, gate, _ = request(cache, live)
    assert live.calls == 1 and "Could not get fixtures from SportyBet: SportyBet returned HTTP 503" in chunks[0]


def test_two_requests_at_once_share_one_listing_read():
    async def go():
        cache, sources, clock = make_cache(Listing(soccer_day(), delay=0.03))
        a, b = await asyncio.gather(cache.read("Matchday", [date(2026, 10, 1)], NOW), cache.read("Matchday", [date(2026, 10, 1)], NOW))
        return sources["Matchday"].calls, a, b

    calls, a, b = run(go())
    assert calls == 1 and a is not None and b is not None


def test_only_the_requested_trigger_is_refreshed():
    cache, sources, clock = make_cache()
    run(cache.build_all())
    clock.now = NOW + timedelta(minutes=30)
    run(cache.read("Gameday", [date(2026, 10, 1)], clock.now))
    assert [sources[t].calls for t in ("Matchday", "Gameday", "America")] == [1, 2, 1]


def test_gameday_and_america_requests_use_the_cache_too():
    nba = game("Lakers", "Celtics", "USA - NBA", sport="basketball", hour=22)
    mlb = game("Yankees", "Mets", "USA - MLB", sport="baseball", hour=22)
    america = MultiSportFixtures({"baseball": Listing([mlb])})
    cache, sources, _ = make_cache(basketball=Listing([nba]), america=america)
    run(cache.build_all())
    gates, stats = [], Table(good_stats)
    run(run_gameday("1 October", Listing([]), str, NOW, stats_source=stats, on_result=gates.append, day_cache=cache))
    assert stats.asked == ["Lakers"] and any(l.startswith("Odds as of") for l in gates[0].footer)
    stats2, gates2 = Table(good_stats), []
    run(run_america("1 October", Listing([]), str, NOW, stats_source=stats2, on_result=gates2.append, day_cache=cache))
    assert stats2.asked == ["Yankees"]


def test_review_is_unaffected():
    assert "day_cache" not in inspect.signature(run_review).parameters


def test_listing_ttl_zero_switches_the_cache_off():
    cache, sources, _ = make_cache(ttl=0)
    assert not cache.enabled and run(cache.build_all()).built == {}
    live = Listing(soccer_day())
    request(cache, live)
    assert live.calls == 1 and sources["Matchday"].calls == 0
    assert "off" in cache.describe() and "LISTING_TTL_SECONDS" in cache.describe()


# ------------------------------------------------------------------ safety: it never feeds placement
def test_the_cache_is_never_imported_by_anything_that_places_or_loads_a_bet():
    root = Path(daycache.__file__).resolve().parents[1]
    for path in [*(root / "adapters" / "sportybet").glob("*.py"), root / "flow" / "placement.py", root / "flow" / "loading.py",
                 root / "flow" / "session.py", root / "core" / "placement.py", root / "core" / "betting.py"]:
        assert "daycache" not in path.read_text(encoding="utf-8"), path.name


def test_the_summary_the_owner_confirms_is_built_from_the_betslip_reading_not_from_the_cache():
    """flow/loading.py verifies the real betslip against the confirmed legs (load_slip), and placement re-reads it again."""
    loading = Path(daycache.__file__).with_name("loading.py").read_text(encoding="utf-8")
    assert "load_slip" in loading and "daycache" not in loading and "day_cache" not in loading


# ------------------------------------------------------------------ the scheduled job and the non-overlap guard
def test_the_build_is_scheduled_daily_at_0600_lagos_without_overlap_or_catch_up():
    application = ApplicationBuilder().token("123456:TEST-TOKEN-NOT-REAL").build()
    job = bot.schedule_daily_cache(application)
    assert job is not None and job.name == DAY_CACHE_JOB_NAME == "daily-games-cache"
    assert application.job_queue.get_jobs_by_name(DAY_CACHE_JOB_NAME) == (job,)
    assert (DAY_CACHE_TIME.hour, DAY_CACHE_TIME.minute, str(DAY_CACHE_TIME.tzinfo)) == (6, 0, "Africa/Lagos")
    trigger = job.job.trigger
    fields = {f.name: str(f) for f in trigger.fields}
    assert fields["hour"] == "6" and fields["minute"] == "0" and str(trigger.timezone) == "Africa/Lagos"
    assert job.job.max_instances == 1 and job.job.coalesce and job.job.misfire_grace_time <= 60
    assert job.callback is bot.daily_cache_job


def test_the_job_is_registered_on_the_real_entry_point():
    assert "schedule_daily_cache(app)" in Path(bot.__file__).read_text(encoding="utf-8")


@pytest.fixture
def live_cache(monkeypatch):
    """The bot's own DAY_CACHE, replaced by one built on fake listings and switched on."""
    clock = Clock()
    cache, sources, _ = make_cache(clock=clock)
    monkeypatch.setattr(bot, "DAY_CACHE", cache)
    monkeypatch.setattr(bot, "DAY_CACHE_GUARD", bot.ResultsGuard())
    return cache, sources


def test_overlapping_builds_never_stack_or_double_run(live_cache):
    cache, sources = live_cache
    sources["Matchday"].delay = 0.05

    async def go():
        first = asyncio.create_task(bot.run_day_cache_build())
        await asyncio.sleep(0.01)
        second = await bot.run_day_cache_build()                          # while the first is reading
        await bot.daily_cache_job(SimpleNamespace(bot=None))              # the 06:00 job is skipped too
        report = await first
        return first, second, report

    _, second, report = run(go())
    assert second is None and report.built["Matchday"] == 4 and sources["Matchday"].calls == 1
    assert not bot.DAY_CACHE_GUARD.busy                                   # released: the next build can run


def test_the_guard_is_released_after_an_error(live_cache):
    cache, sources = live_cache
    sources["Matchday"].error = RuntimeError("bug")
    report = run(bot.run_day_cache_build())
    assert report.failed and not bot.DAY_CACHE_GUARD.busy


# ------------------------------------------------------------------ at start-up
def test_the_start_up_build_runs_in_the_background_and_never_delays_start_up(live_cache, monkeypatch):
    cache, sources = live_cache
    sources["Matchday"].delay = 0.1
    sent = []

    async def send_message(chat_id, text, **kw):
        sent.append(text)

    application = SimpleNamespace(bot=SimpleNamespace(send_message=send_message), create_task=lambda c: c.close())

    async def go():
        started = asyncio.get_running_loop().time()
        await bot.announce_startup(application)
        returned = asyncio.get_running_loop().time() - started
        still_empty = cache.snapshot("Matchday") is None
        running = bot.JOBS.running(bot.DAY_CACHE_JOB_KEY)
        await bot.JOBS.wait()
        return returned, still_empty, running

    returned, still_empty, running = run(go())
    assert still_empty and running and returned < 0.09                    # start-up returned before the listing was read
    assert cache.snapshot("Matchday") is not None


def test_a_failing_start_up_build_is_logged_and_does_not_stop_the_bot(live_cache):
    cache, sources = live_cache
    sources["Matchday"].error = RuntimeError("boom")

    async def send_message(chat_id, text, **kw):
        pass

    application = SimpleNamespace(bot=SimpleNamespace(send_message=send_message), create_task=lambda c: c.close())

    async def go():
        await bot.announce_startup(application)
        await bot.JOBS.wait()

    run(go())                                                             # must not raise


# ------------------------------------------------------------------ the Cache trigger
def owner_text(text):
    sent = []

    async def reply_text(message, **kwargs):
        sent.append(message)

    update = SimpleNamespace(effective_user=SimpleNamespace(id=bot.OWNER_ID), callback_query=None,
                             message=SimpleNamespace(text=text, reply_text=reply_text))
    return update, sent


def test_cache_on_an_empty_cache_says_so_and_how_to_build_it(live_cache):
    update, sent = owner_text("Cache")
    run(handled(update))
    assert len(sent) == 1 and "empty" in sent[0] and "Cache refresh" in sent[0]


def test_cache_reports_when_built_the_window_games_per_trigger_and_snapshot_ages(live_cache):
    cache, sources = live_cache
    run(cache.build_all())
    cache._clock.now = NOW + timedelta(minutes=7)
    update, sent = owner_text("cache")                                    # any letter case
    run(handled(update))
    text = sent[0]
    assert "Built: 01 Oct 10:00 (Lagos)" in text and "Window: 01 Oct and 02 Oct (Lagos), 2 days" in text
    assert "Matchday: 4 game(s), odds read 7 min ago." in text and "Gameday: 0 game(s)" in text and "America: 0 game(s)" in text
    assert "Oldest snapshot: 7 min. Newest: 7 min." in text and "memory only" in text
    assert "Stats warm-up:" in text


def test_cache_shows_different_ages_per_trigger(live_cache):
    cache, sources = live_cache
    run(cache.build_all())
    cache._clock.now = NOW + timedelta(minutes=5)
    run(cache.refresh("Gameday", cache._clock.now, force=True))
    cache._clock.now = NOW + timedelta(minutes=9)
    text = cache.describe(cache._clock.now)
    assert "Matchday: 4 game(s), odds read 9 min ago" in text and "Gameday: 0 game(s), odds read 4 min ago" in text
    assert "Oldest snapshot: 9 min. Newest: 4 min." in text


def test_cache_refresh_rebuilds_in_the_background_and_answers_at_once(live_cache):
    cache, sources = live_cache
    sources["Matchday"].delay = 0.05

    async def go():
        update, sent = owner_text("Cache refresh")
        await bot.handle_message(update, None)
        immediately = list(sent)
        building = bot.JOBS.running(bot.DAY_CACHE_JOB_KEY)
        again_update, again_sent = owner_text("Cache refresh")
        await bot.handle_message(again_update, None)                       # a second one while it runs
        await bot.JOBS.wait()
        return immediately, building, again_sent

    immediately, building, again = run(go())
    assert "Rebuilding the games cache in the background" in immediately[0] and building
    assert "already being built" in again[0] and sources["Matchday"].calls == 1
    assert cache.snapshot("Matchday") is not None


def test_cache_with_something_else_after_it_explains_the_two_forms(live_cache):
    update, sent = owner_text("Cache please")
    run(handled(update))
    assert "Cache refresh" in sent[0]


def test_cache_is_in_the_help_the_unrecognised_text_and_the_readme(live_cache):
    shown = []

    async def reply_text(text, **kw):
        shown.append(text)

    run(bot.start_cmd(SimpleNamespace(message=SimpleNamespace(reply_text=reply_text)), None))
    assert "`Cache`" in shown[0] and "Cache refresh" in shown[0]
    update, sent = owner_text("what is this")
    run(handled(update))
    assert "`Cache`" in sent[0]
    readme = (Path(bot.__file__).parent / "Readme.md").read_text(encoding="utf-8")
    for needle in ("Cache refresh", "CACHE_TTL_SCALE", "LISTING_TTL_SECONDS", "FAST_BROWSER", "06:00"):
        assert needle in readme


def test_cache_is_owner_only_a_stranger_never_reaches_the_handler(wired):
    application, calls = wired
    stranger = 999888777
    feed(application, {"message": telegram_message(stranger, stranger, "private", "Cache refresh")})
    feed(application, {"message": telegram_message(bot.OWNER_ID, bot.OWNER_ID, "private", "Cache")})
    assert calls == ["text"]                                              # the owner's got through, the stranger's did not


def test_cache_refresh_by_a_stranger_in_a_group_is_dropped_too(wired):
    application, calls = wired
    feed(application, {"message": telegram_message(bot.OWNER_ID, -1001, "group", "Cache refresh")})
    assert calls == []
