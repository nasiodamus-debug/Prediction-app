"""The daily games cache (spec-14-1, AD-23): the allowed games of the next two Lagos days, kept IN MEMORY ONLY.

WHAT IT HOLDS: for each trigger (Matchday = soccer, Gameday = basketball, America = MLB, NHL and American football) the games
in the competitions the owner allows (core/competitions.py), open for betting and not yet started, that kick off today or
tomorrow (Africa/Lagos calendar days, DAY_CACHE_DAYS), exactly as SportyBet's listing gave them (teams, competition, kickoff,
markets and odds) with the time the snapshot was read. Nothing is written to SQLite.

HOW IT IS BUILT: at the start of the day (a scheduled job in app.py, 06:00 Lagos), in the background at start-up, on demand
(`Cache refresh`) and lazily when a request finds it empty or from an earlier Lagos day. The three listings are read at the same
time (the SportyBet adapter already reads its pages in parallel), filtered with the same rules as the live flow, and then the
stats of those games only are fetched ahead of time so the stats cache (core/ttl_cache.py) is warm. The analyst is never called.

HOW IT IS USED: a Matchday, Gameday or America request whose dates all lie inside the cached days reads its games from here when
the odds snapshot is younger than LISTING_TTL_SECONDS (default 10 minutes), so NO listing is read. An older snapshot is refreshed
first: the live listing is read again in parallel (measured: the whole soccer list in about 7 to 10 s) and the cache is rebuilt
from it, which also picks up games SportyBet added since. The listing beat one lookup per game (about 300 to 700 markets each,
about one second each, four at a time): see the spec. Any date outside the cached days, a failed refresh, or LISTING_TTL_SECONDS=0
sends the request down the normal path, which reads the live listing itself.

SAFETY: the cache NEVER feeds placement. Every request still re-checks that each game has not started and is open for betting
(flow/slate_run.py), and placing still reads the real betslip and verifies the odds against the confirmed token
(adapters/sportybet/betslip.py): the odds in this cache are only for the list the owner reads, which shows when they were read.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Awaitable, Callable, Mapping

from predictions.core.availability import is_open_for_betting
from predictions.core.competitions import is_allowed
from predictions.core.dates import LAGOS, lagos_today
from predictions.core.ports import FixtureFetchError, FixtureSource
from predictions.core.types import Fixture
from predictions.core.ttl_cache import DEFAULT_LISTING_TTL_SECONDS

log = logging.getLogger(__name__)

DAY_CACHE_DAYS = 2                                  # Lagos calendar days held: today and tomorrow (used everywhere)
DAY_CACHE_TIME = time(6, 0, tzinfo=LAGOS)           # the scheduled build: 06:00 Africa/Lagos
DAY_CACHE_JOB_NAME = "daily-games-cache"
PREWARM_CONCURRENCY = 3                             # triggers whose stats are warmed at the same time
TRIGGERS = ("Matchday", "Gameday", "America")       # the names core/competitions.py and the slate profiles use
Warmer = Callable[[list[Fixture]], Awaitable[None]]


def window_days(today: date) -> tuple[date, ...]:
    """The Lagos calendar days the cache covers when today is `today`."""
    return tuple(today + timedelta(days=i) for i in range(DAY_CACHE_DAYS))


def lagos_clock(moment: datetime) -> str:
    return moment.astimezone(LAGOS).strftime("%H:%M")


def odds_as_of_line(snapshot_at: datetime) -> str:
    """The ONE footer line of a list built from the cache."""
    return f"Odds as of {lagos_clock(snapshot_at)} (Lagos)"


@dataclass
class TriggerSnapshot:
    trigger: str
    snapshot_at: datetime                    # when SportyBet's listing was read (UTC)
    fixtures: tuple[Fixture, ...]            # allowed, open for betting, not started, kicking off inside the window
    ignored: dict[date, int] = field(default_factory=dict)    # not-started games outside the allowed competitions, per day
    not_open: dict[date, int] = field(default_factory=dict)   # allowed games listed but not open for betting, per day

    @property
    def day(self) -> date:
        return lagos_today(self.snapshot_at)


@dataclass
class CachedListing:
    """What a request gets from the cache."""
    fixtures: list[Fixture]
    snapshot_at: datetime
    ignored: int
    not_open: int


@dataclass
class BuildReport:
    built: dict[str, int] = field(default_factory=dict)      # trigger -> games held
    failed: dict[str, str] = field(default_factory=dict)     # trigger -> safe reason


def take_snapshot(trigger: str, listed: list[Fixture], now: datetime, snapshot_at: datetime) -> TriggerSnapshot:
    """The allowed, open, not-started games of the window, from one raw listing. Same rules as the live flow (flow/slate_run.py)."""
    window = set(window_days(lagos_today(now)))
    kept: dict = {}
    ignored: dict[date, int] = {}
    not_open: dict[date, int] = {}
    seen = set()
    for fixture in listed:
        day = fixture.kickoff_utc.astimezone(LAGOS).date()
        if day not in window or fixture.match_id in seen:
            continue
        seen.add(fixture.match_id)
        if fixture.started or fixture.kickoff_utc <= now:
            continue
        if not is_allowed(trigger, fixture.match_id.sport, fixture.competition)[0]:
            ignored[day] = ignored.get(day, 0) + 1
            continue
        if not is_open_for_betting(fixture, now):
            not_open[day] = not_open.get(day, 0) + 1
            continue
        kept[fixture.match_id] = fixture
    ordered = tuple(sorted(kept.values(), key=lambda f: (f.kickoff_utc, f.competition, f.home)))
    return TriggerSnapshot(trigger, snapshot_at, ordered, ignored, not_open)


class DayCache:
    """In-memory, per trigger. `sources` maps a trigger to the (unlocked) fixture source that reads its listing; `lock` is the
    one SportyBet lock (AD-17), taken once around a read. `warmers` maps a trigger to the stats pre-warm for its games."""

    def __init__(self, sources: Mapping[str, FixtureSource], clock: Callable[[], datetime] | None = None,
                 listing_ttl_seconds: int = DEFAULT_LISTING_TTL_SECONDS, lock: asyncio.Lock | None = None,
                 warmers: Mapping[str, Warmer] | None = None):
        self._sources = dict(sources)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self.ttl = int(listing_ttl_seconds)
        self._lock = lock
        self._warmers = dict(warmers or {})
        self._snapshots: dict[str, TriggerSnapshot] = {}
        self._trigger_locks = {t: asyncio.Lock() for t in self._sources}
        self.built_at: datetime | None = None
        self.warm_state = "not started"
        self.last_errors: dict[str, str] = {}

    # ------------------------------------------------------------------ state
    @property
    def enabled(self) -> bool:
        return self.ttl > 0

    def snapshot(self, trigger: str) -> TriggerSnapshot | None:
        return self._snapshots.get(trigger)

    def clear(self) -> None:
        self._snapshots.clear()
        self.built_at = None

    def _age(self, snap: TriggerSnapshot, now: datetime) -> float:
        return (now - snap.snapshot_at).total_seconds()

    def _usable(self, snap: TriggerSnapshot | None, now: datetime) -> bool:
        """Built today (Lagos) and younger than the listing time to live."""
        return snap is not None and snap.day == lagos_today(now) and self._age(snap, now) < self.ttl

    # ------------------------------------------------------------------ reading the listings
    async def _read(self, trigger: str) -> TriggerSnapshot:
        listed = await self._sources[trigger].fetch_fixtures()
        stamped = self._clock()
        snap = take_snapshot(trigger, listed, stamped, stamped)
        self._snapshots[trigger] = snap
        return snap

    async def _refresh_one(self, trigger: str, now: datetime, force: bool) -> TriggerSnapshot:
        async with self._trigger_locks[trigger]:          # one read per trigger at a time
            snap = self._snapshots.get(trigger)
            if not force and self._usable(snap, self._clock()):
                return snap                                # another task just refreshed it
            return await self._read(trigger)

    async def refresh(self, trigger: str, now: datetime | None = None, force: bool = False) -> TriggerSnapshot:
        """Read one trigger's listing now (unless a fresh snapshot is already there and `force` is off). Raises
        FixtureFetchError like the listing does."""
        now = now or self._clock()
        if self._lock is None:
            return await self._refresh_one(trigger, now, force)
        async with self._lock:
            return await self._refresh_one(trigger, now, force)

    # ------------------------------------------------------------------ the scheduled / startup / manual build
    async def build_all(self, now: datetime | None = None) -> BuildReport:
        """Read the three listings at the same time, keep the allowed games of the two days, then warm the stats of those
        games only. A listing that cannot be read is reported and leaves that trigger as it was; the others are kept."""
        report = BuildReport()
        if not self.enabled:
            return report
        triggers = list(self._sources)

        async def read_all():
            return await asyncio.gather(*(self._refresh_one(t, now or self._clock(), True) for t in triggers),
                                        return_exceptions=True)

        if self._lock is None:
            outcomes = await read_all()
        else:
            async with self._lock:
                outcomes = await read_all()
        for trigger, outcome in zip(triggers, outcomes):
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome
            if isinstance(outcome, BaseException):
                report.failed[trigger] = str(outcome) if isinstance(outcome, FixtureFetchError) else type(outcome).__name__
                log.warning("Games cache: the %s listing could not be read (%s)", trigger, type(outcome).__name__)
            else:
                report.built[trigger] = len(outcome.fixtures)
        self.last_errors = dict(report.failed)
        if report.built:
            self.built_at = self._clock()
        log.info("Games cache built: %s", ", ".join(f"{t} {n}" for t, n in report.built.items()) or "nothing")
        await self.prewarm(list(report.built))
        return report

    async def prewarm(self, triggers: list[str]) -> None:
        """Fetch the stats of the cached games (only those) ahead of time, a few triggers at a time. Never calls the analyst;
        a failure here is logged and costs nothing but the head start."""
        work = [t for t in triggers if t in self._warmers and self._snapshots.get(t) and self._snapshots[t].fixtures]
        if not work:
            self.warm_state = "nothing to warm"
            return
        self.warm_state = "running"
        gate = asyncio.Semaphore(PREWARM_CONCURRENCY)
        failed = 0

        async def one(trigger: str) -> None:
            nonlocal failed
            async with gate:
                try:
                    await self._warmers[trigger](list(self._snapshots[trigger].fixtures))
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    failed += 1
                    log.warning("Games cache: warming the %s stats failed (%s)", trigger, type(error).__name__)

        try:
            await asyncio.gather(*(one(t) for t in work))
        except asyncio.CancelledError:
            self.warm_state = "stopped"
            raise
        self.warm_state = "done" if not failed else f"done, {failed} trigger(s) failed"

    # ------------------------------------------------------------------ used by a request
    async def read(self, trigger: str, dates: list[date], now: datetime) -> CachedListing | None:
        """The cached allowed games for a request, or None when the request must take the normal path (cache off, a date
        outside the cached days, nothing usable and the refresh failed). Refreshes the snapshot first when it is missing,
        from an earlier Lagos day or older than the listing time to live."""
        if not self.enabled or trigger not in self._sources:
            return None
        today = lagos_today(now)
        future = [d for d in dates if d >= today]
        window = set(window_days(today))
        if not future or any(d not in window for d in future):
            return None
        snap = self._snapshots.get(trigger)
        if not self._usable(snap, now):
            try:
                snap = await self.refresh(trigger, now)
            except Exception as error:   # the normal path reads the listing itself and reports a real failure properly
                log.warning("Games cache: refreshing the %s listing failed (%s); using the normal path", trigger,
                            type(error).__name__)
                return None
            if snap.day != today:       # the snapshot is not from today (Lagos): never serve it
                return None
        wanted = set(future)
        fixtures = [f for f in snap.fixtures if f.kickoff_utc.astimezone(LAGOS).date() in wanted]
        return CachedListing(fixtures, snap.snapshot_at, sum(snap.ignored.get(d, 0) for d in wanted),
                             sum(snap.not_open.get(d, 0) for d in wanted))

    # ------------------------------------------------------------------ the `Cache` trigger
    def describe(self, now: datetime | None = None, building: bool = False) -> str:
        now = now or self._clock()
        if not self.enabled:
            return ("The games cache is off (LISTING_TTL_SECONDS is 0). Every Matchday, Gameday and America request reads "
                    "SportyBet's listing itself.")
        lines = ["Games cache (kept in memory only, never used for placing a bet)."]
        if building:
            lines.append("A build is running now.")
        if self.built_at is None and not self._snapshots:
            lines.append("It is empty: it has not been built yet. Send Cache refresh to build it now.")
            return "\n".join(lines)
        if self.built_at is not None:
            lines.append(f"Built: {self.built_at.astimezone(LAGOS).strftime('%d %b %H:%M')} (Lagos). "
                         f"Stats warm-up: {self.warm_state}.")
        first = min((s.day for s in self._snapshots.values()), default=lagos_today(now))
        days = window_days(first)
        lines.append("Window: " + " and ".join(d.strftime("%d %b") for d in days) + f" (Lagos), {DAY_CACHE_DAYS} days.")
        ages = []
        for trigger in TRIGGERS:
            snap = self._snapshots.get(trigger)
            if snap is None:
                reason = f" ({self.last_errors[trigger]})" if trigger in self.last_errors else ""
                lines.append(f"{trigger}: nothing held{reason}.")
                continue
            age = max(0, int(self._age(snap, now) // 60))
            ages.append(age)
            lines.append(f"{trigger}: {len(snap.fixtures)} game(s), odds read {age} min ago.")
        if ages:
            lines.append(f"Oldest snapshot: {max(ages)} min. Newest: {min(ages)} min. A request reads a snapshot younger than "
                         f"{self.ttl / 60:g} min as it is and refreshes an older one.")
        return "\n".join(lines)
