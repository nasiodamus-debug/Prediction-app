"""Use the AI analyst during a Matchday run (Epic 3). The flow owns the call budget and the outage stance (AD-16);
the analyst itself is behind the `Analyst` port and only ever sees fixture identifiers (AD-10)."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Awaitable, Callable

from predictions.core.analyst import (DEFAULT_DAILY_CALL_LIMIT, RUN_LIMIT, AnalystFlag, AnalystReport, AnalystUsage, query_for)
from predictions.core.dates import lagos_today
from predictions.core.ports import Analyst, AnalystUnavailable
from predictions.core.stats import (CAT_HISTORY, INSUFFICIENT, FixtureStats, add_reason, assess_goals_only, merge_analyst, missing_for,
                                    skip_category)
from predictions.core.types import Fixture, MatchId

# Approximate prices (USD) for the cost estimate, from Anthropic's published pricing for Claude Sonnet 5.5
# ($2 / $10 per million tokens) and web search ($10 per 1,000 searches). An estimate, not a bill.
USD_PER_INPUT_TOKEN = 2.0 / 1_000_000
USD_PER_OUTPUT_TOKEN = 10.0 / 1_000_000
USD_PER_SEARCH = 10.0 / 1_000

ANALYST_CONCURRENCY = 3   # per-fixture analyst calls in flight at once (spec-14-1); the call caps still hold exactly


class AnalystCache:
    """What the analyst answered, kept in memory for the rest of the Lagos day, per (fixture identity, requested inputs)
    (spec-14-1, AD-23). A repeated request for the same fixture and inputs is served from here: it makes no call, counts
    against neither the per-request nor the daily limit and costs nothing. Only a valid answer is kept (an unavailable
    service or a discarded malformed answer is never kept). It holds nothing but the schema-validated report. `enabled` is
    False when CACHE_TTL_SCALE is 0. The day rolls over by Lagos date."""

    def __init__(self):
        self.enabled = True
        self._day = None
        self._items: dict = {}

    def _roll(self, now: datetime) -> None:
        day = lagos_today(now)
        if day != self._day:
            self._day, self._items = day, {}

    def get(self, key, now: datetime):
        if not self.enabled:
            return None
        self._roll(now)
        return self._items.get(key)

    def put(self, key, report, now: datetime) -> None:
        if self.enabled and not report.rejected:
            self._roll(now)
            self._items[key] = report

    def clear(self) -> None:
        self._day, self._items = None, {}

    def __len__(self) -> int:
        return len(self._items)


ANALYST_CACHE = AnalystCache()   # shared by every request of this process


async def gather_in_order(jobs: list[Callable[[], Awaitable]], concurrency: int) -> list:
    """Run the jobs with at most `concurrency` in flight and return their results in the ORDER OF THE JOBS (never completion
    order). Jobs start in list order (a FIFO semaphore), so the earliest job reaches any shared limit first. If one raises,
    the others are cancelled and the error goes on."""
    gate = asyncio.Semaphore(max(1, int(concurrency)))

    async def run(job):
        async with gate:
            return await job()

    tasks = [asyncio.ensure_future(run(job)) for job in jobs]
    try:
        return list(await asyncio.gather(*tasks))
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class DailyBudget:
    """The analyst calls allowed per Lagos day, counted across requests (Story 10.1). The count lives in the evaluation
    store (flow/store.py), so a restart does not reset it. `store` needs analyst_calls_on(day) and add_analyst_calls(day, n);
    a store that cannot be read or written never blocks a request (the per-request cap still applies) but is said once."""

    def __init__(self, store, limit: int = DEFAULT_DAILY_CALL_LIMIT, now: datetime | None = None):
        self.store, self.limit = store, limit
        self.day = lagos_today(now or datetime.now(timezone.utc)).isoformat()
        self.broken = False

    def used(self) -> int:
        try:
            return self.store.analyst_calls_on(self.day)
        except Exception:
            self.broken = True
            return 0

    def remaining(self) -> int:
        return max(0, self.limit - self.used())

    def record(self, count: int = 1) -> None:
        try:
            self.store.add_analyst_calls(self.day, count)
        except Exception:
            self.broken = True


class AnalystRun:
    """One request's use of the analyst: counts calls, enforces the per-request cap and the daily budget, remembers an outage."""

    def __init__(self, analyst: Analyst, limit: int | None = None, budget: DailyBudget | None = None,
                 cache: AnalystCache | None = ANALYST_CACHE, clock: Callable[[], datetime] | None = None):
        self.analyst, self.limit, self.budget = analyst, RUN_LIMIT.value if limit is None else limit, budget
        self.cache, self._clock = cache, clock or (lambda: datetime.now(timezone.utc))
        self.cache_hits = 0          # answers reused from today's cache: no call, no cost, no limit used
        self.daily_hit = False       # the day's budget is used up: no call is made
        self.calls = 0
        self.usage = AnalystUsage()
        self.unavailable = ""        # the safe message of the outage, once seen; no more calls are made then
        self.limit_hit = False
        self.discarded = 0           # responses thrown away for breaking the schema
        self.asked: set[MatchId] = set()
        self.last_resort = bool(getattr(analyst, "last_resort", False))   # ANALYST_MODE=last_resort: see core/analyst.py

    @property
    def cost(self) -> float:
        u = self.usage
        return u.input_tokens * USD_PER_INPUT_TOKEN + u.output_tokens * USD_PER_OUTPUT_TOKEN + u.searches * USD_PER_SEARCH

    async def ask(self, fixture: Fixture, wanted: tuple[str, ...]) -> AnalystReport | None:
        """None when no call was made (cap reached, outage) or the service failed; else the validated report.

        The checks and the counting below happen with NO await in between, so concurrent asks (flow/analyst.py runs a few
        at once) can never take more than the per-request cap or the daily limit allows."""
        key = (fixture.match_id, tuple(wanted))
        if self.cache is not None:
            cached = self.cache.get(key, self._clock())
            if cached is not None:
                self.cache_hits += 1
                self.asked.add(fixture.match_id)
                return cached
        if self.unavailable:
            return None
        if self.budget is not None and self.budget.remaining() <= 0:
            self.daily_hit = True
            return None
        if self.calls >= self.limit:
            self.limit_hit = True
            return None
        self.calls += 1
        if self.budget is not None:
            self.budget.record(1)  # counted before the call: a call that fails still cost a request
        self.asked.add(fixture.match_id)
        try:
            report = await self.analyst.research(query_for(fixture, wanted))
        except AnalystUnavailable as error:
            self.unavailable = str(error) or "the Claude service is unavailable"
            return None
        u = report.usage
        self.usage = AnalystUsage(self.usage.input_tokens + u.input_tokens, self.usage.output_tokens + u.output_tokens,
                                  self.usage.searches + u.searches)
        if report.rejected:
            self.discarded += 1
        elif self.cache is not None:
            self.cache.put(key, report, self._clock())
        return report

    def daily_note(self) -> str:
        return (f" Daily analyst limit of {self.budget.limit} calls reached: the analyst is skipped until tomorrow "
                "(Lagos time).")

    def short_summary(self) -> str:
        """ONE short line for the end of a passing list: the calls made and the estimated cost ('' when none were made)."""
        if not self.calls and not self.cache_hits:
            return ""
        text = f"AI analyst: {self.calls} call(s) used, about ${self.cost:.2f} (estimate, not a bill)."
        if self.cache_hits:
            text += f" {self.cache_hits} answer(s) reused from today's cache at no cost."
        if self.daily_hit:
            text += f" Daily limit of {self.budget.limit} calls reached."
        return text

    def summary(self, skipped_for_limit: int = 0) -> str:
        """The line telling the owner what the analyst cost this run (shown every run it is used)."""
        text = (f"AI analyst: used {self.calls} of {self.limit} calls, about ${self.cost:.2f} "
                f"({self.usage.searches} web searches; estimate, not a bill).")
        if self.cache_hits:
            text += f" {self.cache_hits} answer(s) reused from today's cache at no cost (not counted in the limits)."
        if self.daily_hit:
            text += self.daily_note()
        if self.limit_hit:
            text += (f" Call limit reached: {skipped_for_limit} game(s) not looked up and left as insufficient data."
                     if skipped_for_limit else " Call limit reached.")
        if self.unavailable:
            text += f" The analyst became unavailable ({self.unavailable}); I continued without it."
        if self.discarded:
            text += f" {self.discarded} malformed answer(s) were discarded."
        return text


def season_not_started(entry: FixtureStats) -> bool:
    """True when the free source found no completed games for a team in its window (a season that has not started or is over):
    the analyst cannot invent the history, so it is not paid to look (ANALYST_MODE=last_resort)."""
    return skip_category(entry) == CAT_HISTORY and any("has no completed" in r for r in entry.reasons)


async def fill_missing_inputs(fixtures: list[Fixture], stats: dict[MatchId, FixtureStats], run: AnalystRun,
                              concurrency: int = ANALYST_CONCURRENCY, goals_only_fallback: bool = False
                              ) -> tuple[dict[MatchId, FixtureStats], dict[MatchId, tuple[AnalystFlag, ...]], int]:
    """Ask the analyst for the inputs the free sources left missing, one call per fixture, earliest kickoff first, a few at a
    time (`concurrency`), until the call cap. The earliest games are launched first, so they get the calls when the cap
    bites; after an outage no new call is launched. The answers are applied in kickoff order whatever order they finished
    in. Returns (stats, flags by fixture, games left unlooked-up because of the cap)."""
    stats, flags, skipped = dict(stats), {}, 0
    work: list[tuple[Fixture, FixtureStats, tuple[str, ...]]] = []
    for fixture in sorted(fixtures, key=lambda f: f.kickoff_utc):
        entry = stats.get(fixture.match_id)
        if entry is None or entry.status != INSUFFICIENT:
            continue
        wanted = missing_for(entry)
        if not wanted:
            continue
        if run.last_resort and season_not_started(entry):
            continue   # nothing to find: the free source says the team has no recent games (preseason or offseason)
        work.append((fixture, entry, wanted))
    reports = await gather_in_order([(lambda f=f, w=w: run.ask(f, w)) for f, _, w in work], concurrency)
    for (fixture, entry, _), report in zip(work, reports):
        if report is None:
            if run.unavailable:
                stats[fixture.match_id] = add_reason(entry, "the AI analyst was unavailable")
            elif run.daily_hit:
                skipped += 1
                stats[fixture.match_id] = add_reason(
                    entry, f"the AI analyst daily limit ({run.budget.limit} calls) was reached")
            else:
                skipped += 1
                stats[fixture.match_id] = add_reason(entry, f"the AI analyst call limit ({run.limit}) was reached")
            continue
        if report.rejected:
            stats[fixture.match_id] = add_reason(entry, "the analyst's answer was discarded (" + report.rejected + ")")
            continue
        merged = merge_analyst(entry, report.stats)
        if goals_only_fallback and merged.status == INSUFFICIENT and fixture.match_id.sport == "soccer":
            # Soccer with no xG anywhere (lower leagues): goals scored and conceded over at least GOALS_ONLY_MIN_MATCHES matches
            # price the game with the goals-only model, labelled as such on every pick (Story 9.3). xG inputs are dropped there.
            goals_only = assess_goals_only(fixture.match_id, [*entry.partial, *report.stats])
            if goals_only.ready:
                merged = goals_only
        stats[fixture.match_id] = merged
        if report.flags:
            flags[fixture.match_id] = report.flags
    return stats, flags, skipped


async def check_flags(fixtures: list[Fixture], match_ids: set[MatchId], run: AnalystRun,
                      concurrency: int = ANALYST_CONCURRENCY) -> dict[MatchId, tuple[AnalystFlag, ...]]:
    """A flags-only lookup for fixtures that still have a passing pick, within the call cap, a few at a time, earliest
    kickoff first. Fixtures the analyst was already asked about this run are not asked again. In last-resort mode no
    flags-only call is made at all (the free injury and weather data are not a reason to pay)."""
    if run.last_resort:
        return {}
    wanted = [f for f in sorted(fixtures, key=lambda f: f.kickoff_utc)
              if f.match_id in match_ids and f.match_id not in run.asked]
    reports = await gather_in_order([(lambda f=f: run.ask(f, ())) for f in wanted], concurrency)
    found: dict[MatchId, tuple[AnalystFlag, ...]] = {}
    for fixture, report in zip(wanted, reports):
        if report is not None and report.flags:
            found[fixture.match_id] = report.flags
    return found
