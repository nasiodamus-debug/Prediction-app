"""SportyBet's public upcoming-games listing (soccer, basketball, ...), behind the FixtureSource port.

All knowledge of this unofficial endpoint lives here (AD-11). It needs no login and never uses the
owner's credentials. The listing has no date filter and is not sorted by time, so every page is
fetched and the caller filters. One class serves every sport: `SportyBetFixtures(sport="basketball")` asks for that
sport's id and markets (see LISTINGS); a sport is added by adding one row there.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

import httpx

from predictions.core.engines.american_football import LIST_MARKET_IDS as FOOTBALL_MARKET_IDS
from predictions.core.engines.basketball import LIST_MARKET_IDS as BASKETBALL_MARKET_IDS
from predictions.core.engines.mlb import LIST_MARKET_IDS as BASEBALL_MARKET_IDS
from predictions.core.engines.nhl import LIST_MARKET_IDS as HOCKEY_MARKET_IDS
from predictions.core.ports import FixtureFetchError
from predictions.core.types import Fixture, MarketOffer, MatchId, Outcome

URL = "https://www.sportybet.com/api/ng/factsCenter/pcUpcomingEvents"
SPORT_ID = "sr:sport:1"
MARKET_IDS = "1,10,18,19,20"  # 1 = 1X2, 10 = Double Chance (spec 11-1), 18 = Over/Under (half and whole lines), 19/20 = Home/Away team totals


@dataclass(frozen=True)
class Listing:
    """What to ask SportyBet for one sport: its id and the markets the engine understands."""
    sport: str       # the canonical name engines are registered under
    sport_id: str    # SportyBet's (Sportradar's) sport id
    market_ids: str  # comma separated; the listing answers 19000 "Invalid" when this is empty


LISTINGS: dict[str, Listing] = {
    "soccer": Listing("soccer", SPORT_ID, MARKET_IDS),
    # 219 winner, 223 handicap, 225 total, 227/228 team totals, 68 1st-half total (probed 2026-09-30, spec-6-1).
    "basketball": Listing("basketball", "sr:sport:2", BASKETBALL_MARKET_IDS),
    # MLB: 251 winner, 256 run line, 258 total, 260/261 team totals, 274/275/276 first five innings (probed 2026-09-30,
    # spec-6-3). NHL: 406 winner, 410/16 puck line and Asian handicap, 412/18 totals, 19/20 team totals, 446 period total
    # (spec-6-4). The listing carries every league of the sport; the America run keeps MLB and NHL only.
    "baseball": Listing("baseball", "sr:sport:3", BASEBALL_MARKET_IDS),
    "hockey": Listing("hockey", "sr:sport:4", HOCKEY_MARKET_IDS),
    # American football (Epic 7): 219 winner, 223 handicap, 225 total, 227/228 team totals (a few games), 68 1st-half total
    # (probed 2026-09-30, sr:sport:16, spec-7-1). Competitions: "USA - NFL", "USA - NCAA, Regular Season", "Canada - CFL"
    # (the CFL is listed but has no rules or stats). SportyBet NG lists no 1st-half team totals.
    "american_football": Listing("american_football", "sr:sport:16", FOOTBALL_MARKET_IDS),
}
PAGE_SIZE = 100
MAX_PAGES = 60  # safety stop; about 17 pages of 100 exist today for soccer
PARALLEL_PAGES = 6  # pages read at the same time after page 1 (measured: 3 -> 12.9 s, 5 -> 10.4 s, 8 -> 7.0 s; was 28.7 s one by one)
TIMEOUT_SECONDS = 20.0  # per request
OVERALL_TIMEOUT_SECONDS = 120.0  # for the whole listing


def TOTAL_DRIFT_ALLOWED(total: int) -> int:
    """How far the announced total may differ between pages: 2 games, or 0.5% of a big listing."""
    return max(2, total // 200)
_HEADERS = {
    # The endpoint refuses requests without a browser-like user agent.
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "application/json",
}
_NOT_STARTED_STATUS = 0


def _decimal(value) -> Decimal | None:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None
    return number if number.is_finite() and number > 0 else None


def _market_open(market: dict) -> bool:
    """Only market status 0 (or absent) is bettable; the live listing shows suspended markets as status 2
    (probed 2026-09-30: 3691 markets at 0, 84 at 2, every outcome isActive 1). Banned markets are skipped too."""
    status = market.get("status")
    if status is not None:
        try:
            if int(status) != 0:
                return False
        except (TypeError, ValueError):
            return False
    return market.get("banned") not in (True, 1, "true")


def _outcome_active(outcome: dict) -> bool:
    active = outcome.get("isActive")
    return not (active is not None and str(active).strip().lower() in ("0", "false"))


def _markets(raw_markets) -> tuple[MarketOffer, ...]:
    offers = []
    for market in raw_markets or []:
        if not isinstance(market, dict) or not _market_open(market):
            continue
        outcomes = []
        for outcome in market.get("outcomes") or []:
            if not isinstance(outcome, dict) or not _outcome_active(outcome):
                continue
            odds = _decimal(outcome.get("odds"))
            name = outcome.get("desc")
            if odds is not None and name:
                outcome_id = outcome.get("id")
                outcomes.append(Outcome(str(name), odds, "" if outcome_id is None else str(outcome_id)))
        if outcomes and market.get("id") is not None:
            offers.append(MarketOffer(str(market["id"]), str(market.get("name") or market.get("desc") or ""),
                                      str(market.get("specifier") or ""), tuple(outcomes)))
    return tuple(offers)


def _started(event: dict) -> bool:
    status = event.get("status")
    if status is not None:
        try:
            status = int(status)  # "0" and 0 both mean not started
        except (TypeError, ValueError):
            return True  # an unknown status: never offer the game
        if status != _NOT_STARTED_STATUS:
            return True
    match_status = event.get("matchStatus")
    return bool(match_status) and str(match_status).strip().lower() != "not start"


def parse_event(event, tournament: dict, sport: str = "soccer") -> Fixture | None:
    """One raw event to a Fixture, or None when it lacks teams or a kickoff time."""
    if not isinstance(event, dict):
        return None
    home, away = event.get("homeTeamName"), event.get("awayTeamName")
    start = event.get("estimateStartTime")
    if not home or not away or not isinstance(start, (int, float)) or isinstance(start, bool):
        return None
    try:
        kickoff = datetime.fromtimestamp(start / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    country = tournament.get("categoryName")
    league = tournament.get("name") or "Unknown competition"
    competition = f"{country} - {league}" if country else str(league)
    match_id = MatchId(sport, competition, kickoff, str(home), str(away))
    event_id = event.get("eventId")
    return Fixture(match_id, competition, str(home), str(away), kickoff, _started(event),
                   _markets(event.get("markets")), "" if event_id is None else str(event_id))


def parse_page(payload, sport: str = "soccer") -> tuple[list[Fixture], int, int]:
    """(fixtures, raw event count, totalNum) for one response page."""
    if not isinstance(payload, dict) or payload.get("bizCode") != 10000 or not isinstance(payload.get("data"), dict):
        raise FixtureFetchError("SportyBet returned an unexpected response")
    data = payload["data"]
    total = data.get("totalNum")
    tournaments = data.get("tournaments")
    if type(total) is not int or total < 0 or not isinstance(tournaments, list):
        raise FixtureFetchError("SportyBet returned an unexpected response")
    fixtures, raw_count = [], 0
    for tournament in tournaments:
        if not isinstance(tournament, dict):
            continue
        events = tournament.get("events")
        if events is None:
            events = []
        if not isinstance(events, list):
            raise FixtureFetchError("SportyBet returned an unexpected response")
        raw_count += len(events)
        for event in events:
            fixture = parse_event(event, tournament, sport)
            if fixture is not None:
                fixtures.append(fixture)
    return fixtures, raw_count, total


class SportyBetFixtures:
    """FixtureSource backed by SportyBet's public listing. No retries. Page 1 is read first (it announces the total), then
    the remaining pages are read concurrently, at most `concurrency` at a time (measured 2026-10-01 on the real soccer
    list: 1 at a time 28.7 s, 3 at a time 12.9 s, 5 at a time 10.4 s, 8 at a time 7.0 s, every event retrieved)."""

    def __init__(self, client: httpx.AsyncClient | None = None, sport: str = "soccer",
                 concurrency: int = PARALLEL_PAGES):
        if sport not in LISTINGS:
            raise ValueError(f"no SportyBet listing is defined for {sport!r}")
        self._client = client
        self._listing = LISTINGS[sport]
        self._concurrency = max(1, int(concurrency))

    async def fetch_fixtures(self) -> list[Fixture]:
        try:
            if self._client is not None:
                return await asyncio.wait_for(self._fetch_all(self._client), OVERALL_TIMEOUT_SECONDS)
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, headers=_HEADERS) as client:
                return await asyncio.wait_for(self._fetch_all(client), OVERALL_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raise FixtureFetchError("SportyBet took too long to answer") from None

    async def _page(self, client: httpx.AsyncClient, page: int) -> tuple[list[Fixture], int, int]:
        """One page: (fixtures, raw event count, announced total). Any failure is a FixtureFetchError with no URL in it."""
        params = {"sportId": self._listing.sport_id, "marketId": self._listing.market_ids, "pageSize": PAGE_SIZE, "pageNum": page,
                  "todayGames": "false", "timeline": "", "_t": int(time.time() * 1000)}
        try:
            response = await client.get(URL, params=params)
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException:
            raise FixtureFetchError("SportyBet did not answer in time") from None
        except httpx.HTTPStatusError as error:
            raise FixtureFetchError(f"SportyBet returned HTTP {error.response.status_code}") from None
        except (httpx.HTTPError, ValueError):
            raise FixtureFetchError("SportyBet could not be reached or gave unreadable data") from None
        return parse_page(payload, self._listing.sport)

    async def _fetch_all(self, client: httpx.AsyncClient) -> list[Fixture]:
        first = await self._page(client, 1)                     # page 1 announces the total
        count = max(1, -(-first[2] // PAGE_SIZE))               # pages that total needs when every page is full
        if count > MAX_PAGES:
            raise FixtureFetchError("SportyBet's listing had more pages than expected")
        pages = [first]
        if count > 1:
            gate = asyncio.Semaphore(self._concurrency)

            async def one(number: int):
                async with gate:
                    return await self._page(client, number)

            tasks = [asyncio.ensure_future(one(n)) for n in range(2, count + 1)]
            try:
                pages.extend(await asyncio.gather(*tasks))      # ANY failure fails the whole read: no partial list
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        fixtures: list[Fixture] = []
        state = {"seen_raw": 0, "total": None}

        def absorb(result) -> bool:
            """Take one page, in page order, with the integrity rules of the one-by-one reader. True when the listing is complete."""
            page_fixtures, raw_count, page_total = result
            total = state["total"]
            if total is None:
                total = page_total
            elif page_total != total:
                # The announced total flickers by a game or two between pages (different servers answer
                # different pages). A small drift is accepted; a large one means the listing really moved
                # and pages read now would skip or repeat games.
                if abs(page_total - total) > TOTAL_DRIFT_ALLOWED(total):
                    raise FixtureFetchError("SportyBet's listing changed while it was being read. Please try again.")
                total = min(total, page_total)  # stop on the smaller figure: every page read is complete
            state["total"] = total
            fixtures.extend(page_fixtures)
            state["seen_raw"] += raw_count
            if state["seen_raw"] >= total:
                return True
            if raw_count == 0:
                raise FixtureFetchError("SportyBet's listing ended before all games were received")
            return False

        for result in pages:
            if absorb(result):
                return _deduplicated(fixtures)
        # Pages held fewer games than PAGE_SIZE (not the case on the real listing): carry on one page at a time.
        for number in range(len(pages) + 1, MAX_PAGES + 1):
            if absorb(await self._page(client, number)):
                return _deduplicated(fixtures)
        raise FixtureFetchError("SportyBet's listing had more pages than expected")


def _deduplicated(fixtures: list[Fixture]) -> list[Fixture]:
    """Each game once (a game can appear on two pages when the listing shifts while it is read), in listing order."""
    seen, unique = set(), []
    for fixture in fixtures:
        key = fixture.bookmaker_ref or fixture.match_id
        if key in seen:
            continue
        seen.add(key)
        unique.append(fixture)
    return unique
