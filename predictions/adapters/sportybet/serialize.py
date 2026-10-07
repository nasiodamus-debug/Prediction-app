"""One lock around every SportyBet operation (AD-17).

The bookmaker adapter (betslip.py) holds this lock for each of its operations. The two read-only SportyBet readers, the
game list used by `Matchday` and the booking-code reader used by `Review`, take the SAME lock through these wrappers, so a
Review or a game-list read never runs in the middle of a betslip load or a placement, and a load waits for them. They use
throwaway anonymous connections and never touch the owner's betslip; the lock only keeps the operations one at a time.
"""
from __future__ import annotations

import asyncio

from predictions.core.ports import CodeReader, EventMarketsSource, FixtureSource


class SerializedFixtures:
    """A FixtureSource that waits for the SportyBet lock before reading."""

    def __init__(self, inner: FixtureSource, lock: asyncio.Lock):
        self._inner, self._lock = inner, lock

    async def fetch_fixtures(self):
        async with self._lock:
            return await self._inner.fetch_fixtures()


class SerializedEventMarkets:
    """An EventMarketsSource (the per-game corners and cards lookup, spec 11-1) that waits for the SportyBet lock."""

    def __init__(self, inner: EventMarketsSource, lock: asyncio.Lock):
        self._inner, self._lock = inner, lock

    async def fetch_event_markets(self, fixtures):
        async with self._lock:
            return await self._inner.fetch_event_markets(fixtures)


class SerializedCodeReader:
    """A CodeReader that waits for the SportyBet lock before reading."""

    def __init__(self, inner: CodeReader, lock: asyncio.Lock):
        self._inner, self._lock = inner, lock

    async def read_code(self, code: str):
        async with self._lock:
            return await self._inner.read_code(code)
