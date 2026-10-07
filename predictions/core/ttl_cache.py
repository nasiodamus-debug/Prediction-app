"""A small in-memory cache with a time to live (spec-14-1, AD-23). Standard library only.

WHAT MAY BE CACHED: only the slow, public, owner-independent reads that give a team's recent form or a schedule (ESPN, MLB, NHL,
Understat, football-data) and the wind forecast. NEVER odds, markets, whether a game is open for betting, anything used to price
a bet for placement, the bet history, the betslip, or anything about the owner's account. The callers decide the key; this
module only keeps what it is given.

RULES: an error is never cached (the fetch raises, nothing is stored); a hit returns exactly the object a miss returned; an entry
older than its time to live is dropped and fetched again; `CACHE_TTL_SCALE` multiplies every time to live (1 = as coded,
0 = the cache is off, so every call is a real read). Time comes from an injectable clock so tests never sleep.
"""
from __future__ import annotations

import math
import time
from typing import Awaitable, Callable, Hashable

TTL_FORM_SECONDS = 3 * 3600      # a team's recent form and schedules: 3 hours
TTL_WEATHER_SECONDS = 1 * 3600   # wind forecasts (and announced starting pitchers): 1 hour
MAX_TTL_SCALE = 24.0             # a scale above this would keep yesterday's form: refused at startup


def parse_cache_ttl_scale(raw) -> float:
    """The CACHE_TTL_SCALE setting, read once at startup. Unset or empty = 1. '0' = cache off. Otherwise a finite number from 0
    to MAX_TTL_SCALE written plainly (digits and one dot); anything else raises ValueError. The message never echoes the value."""
    if raw is None or not str(raw).strip():
        return 1.0
    text = str(raw).strip()
    plain = text.replace(".", "", 1)
    if not text.isascii() or not plain.isdigit():
        value = None
    else:
        value = float(text)
    if value is None or not math.isfinite(value) or value > MAX_TTL_SCALE:
        raise ValueError("CACHE_TTL_SCALE in config.env is not a number from 0 to 24 (for example 1 for the normal time, 0.5 "
                         "for half of it, 0 to switch the cache off). Fix it or remove the line to use 1, then start the bot "
                         "again.")
    return value


DEFAULT_LISTING_TTL_SECONDS = 600   # how old the daily games cache's odds snapshot may be before a request refreshes it
MAX_LISTING_TTL_SECONDS = 3600


def parse_listing_ttl_seconds(raw) -> int:
    """The LISTING_TTL_SECONDS setting, read once at startup. Unset or empty = 600 (10 minutes). '0' switches the daily games
    cache off. Otherwise a whole number of seconds up to 3600; anything else raises ValueError (never echoed)."""
    if raw is None or not str(raw).strip():
        return DEFAULT_LISTING_TTL_SECONDS
    text = str(raw).strip()
    if not text.isascii() or not text.isdigit() or len(text) > 5 or int(text) > MAX_LISTING_TTL_SECONDS:
        raise ValueError("LISTING_TTL_SECONDS in config.env is not a whole number of seconds from 0 to 3600 (for example 600; "
                         "0 switches the daily games cache off). Fix it or remove the line to use 600, then start the bot "
                         "again.")
    return int(text)


class TtlCache:
    """Key -> value with a time to live. One instance is shared by the stats adapters (STATS_CACHE below)."""

    def __init__(self, clock: Callable[[], float] = time.monotonic, scale: float = 1.0):
        self._clock = clock
        self.scale = float(scale)
        self._items: dict[Hashable, tuple[float, object]] = {}
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self.scale > 0

    def set_scale(self, scale: float) -> None:
        self.scale = float(scale)
        if not self.enabled:
            self._items.clear()

    def clear(self) -> None:
        self._items.clear()

    def counts(self) -> tuple[int, int]:
        """(hits, misses) so far; a request logs the difference between two readings."""
        return self.hits, self.misses

    def __len__(self) -> int:
        return len(self._items)

    def _fresh(self, key: Hashable):
        entry = self._items.get(key)
        if entry is None:
            return None
        if self._clock() >= entry[0]:
            del self._items[key]
            return None
        return entry

    async def get_or_fetch(self, key: Hashable, ttl_seconds: float, fetch: Callable[[], Awaitable[object]]):
        """The cached value for `key`, or `await fetch()` (stored for ttl_seconds x scale). A fetch that raises stores nothing
        and the error goes on to the caller. A fetch that returns None is not stored either (it means 'not there')."""
        if not self.enabled:
            self.misses += 1
            return await fetch()
        entry = self._fresh(key)
        if entry is not None:
            self.hits += 1
            return entry[1]
        self.misses += 1
        value = await fetch()
        if value is not None:
            self._items[key] = (self._clock() + ttl_seconds * self.scale, value)
        return value

    def put(self, key: Hashable, value: object, ttl_seconds: float) -> None:
        if self.enabled and value is not None:
            self._items[key] = (self._clock() + ttl_seconds * self.scale, value)


def request_key(source: str, path: str, params: dict | None = None) -> tuple:
    """The key of one exact request: source, path and every parameter (order does not matter)."""
    return (source, path, tuple(sorted((str(k), str(v)) for k, v in (params or {}).items())))


STATS_CACHE = TtlCache()   # the one cache the stats adapters share (scale set at startup from CACHE_TTL_SCALE)
