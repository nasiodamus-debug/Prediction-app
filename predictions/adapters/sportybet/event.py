"""SportyBet's public per-event lookup, behind the EventMarketsSource port (spec 11-1).

The upcoming-games listing carries only the markets it is asked for (fixtures.MARKET_IDS) and SportyBet does not put total
corners or total bookings (cards) into it. Those two markets exist only on the per-event endpoint, and only for some games
(probed 2026-10-01, see spec-11-1): `GET /api/ng/factsCenter/event?eventId=sr:match:...&productId=3` returns ALL markets of one
game (about 300 to 700), no login needed. This adapter asks it once per game, keeps only the two markets the bot plays
(166 "Corners - Over/Under", 139 "Bookings - Over/Under") and hands them to the engine as ordinary MarketOffers, with the real
listed lines and odds (suspended markets and inactive outcomes are dropped exactly as the listing does). Nothing is cached
across requests (odds change) and nothing is retried (AD-14). All knowledge of this unofficial endpoint lives here (AD-11).
"""
from __future__ import annotations

import asyncio
import time

import httpx

from predictions.adapters.sportybet.fixtures import _HEADERS, _markets
from predictions.core.types import Fixture, MarketOffer, MatchId

EVENT_URL = "https://www.sportybet.com/api/ng/factsCenter/event"
PRODUCT_ID = "3"
WANTED_MARKET_IDS = frozenset({"166", "139"})   # total corners, total bookings (cards)
CONCURRENCY = 4
TIMEOUT_SECONDS = 15.0       # per request
OVERALL_TIMEOUT_SECONDS = 60.0


def parse_event_markets(payload) -> tuple[MarketOffer, ...] | None:
    """The corners and cards offers of one event response, () when it lists none, None when the answer is unreadable."""
    if not isinstance(payload, dict) or payload.get("bizCode") != 10000 or not isinstance(payload.get("data"), dict):
        return None
    markets = payload["data"].get("markets")
    if markets is None:
        return ()
    if not isinstance(markets, list):
        return None
    wanted = [m for m in markets if isinstance(m, dict) and str(m.get("id")) in WANTED_MARKET_IDS]
    return _markets(wanted)


class SportyBetEventMarkets:
    """EventMarketsSource backed by SportyBet's public event endpoint."""

    def __init__(self, client: httpx.AsyncClient | None = None, concurrency: int = CONCURRENCY,
                 overall_timeout: float = OVERALL_TIMEOUT_SECONDS):
        self._client = client
        self._concurrency = max(1, int(concurrency))
        self._overall = overall_timeout

    async def fetch_event_markets(self, fixtures: list[Fixture]) -> dict[MatchId, tuple[MarketOffer, ...] | None]:
        results: dict[MatchId, tuple[MarketOffer, ...] | None] = {f.match_id: None for f in fixtures}
        if not fixtures:
            return results
        gate = asyncio.Semaphore(self._concurrency)

        async def one(client, fixture: Fixture) -> None:
            if not fixture.bookmaker_ref:
                return
            async with gate:
                try:
                    response = await client.get(EVENT_URL, params={"eventId": fixture.bookmaker_ref, "productId": PRODUCT_ID,
                                                                   "_t": int(time.time() * 1000)})
                    response.raise_for_status()
                    results[fixture.match_id] = parse_event_markets(response.json())
                except (httpx.HTTPError, ValueError):
                    results[fixture.match_id] = None   # this game keeps the markets it has; the caller says so

        async def run(client) -> None:
            tasks = [asyncio.ensure_future(one(client, f)) for f in fixtures]
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), self._overall)
            except asyncio.TimeoutError:
                for task in tasks:
                    task.cancel()   # the games not reached stay None (failed)
                await asyncio.gather(*tasks, return_exceptions=True)

        if self._client is not None:
            await run(self._client)
        else:
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, headers=_HEADERS) as client:
                await run(client)
        return results
