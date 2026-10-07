"""SportyBet booking-code reader, behind the CodeReader port (Story 4.1).

All knowledge of this unofficial endpoint lives here (AD-11). It is a plain anonymous HTTP GET of the public
share-code endpoint: no browser, no cookies, no login, and nothing that can reach a betslip (AD-17). A new
client is created for every read, so no state is shared between reads. No retries; any failure is a
CodeReadError with a safe message, and a code is either read completely or not at all.

Probed on 2026-09-30 with made-up codes only (no real code was available): an unknown code answers HTTP 200 with
`{"bizCode":19000,"message":"The code is invalid."}`, and the request is refused (403) without a browser-like
user agent. The SUCCESS shape below (`bizCode` 10000, `data.outcomes[]` each an event with `markets[].outcomes[]`)
is from memory of how the SportyBet web app displays a shared code and is UNVERIFIED; tests use a fabricated
sample. If it is wrong the read fails with a safe message ("unexpected answer"), it never guesses.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx

from predictions.adapters.sportybet.fixtures import _decimal, _market_open, _outcome_active, _started
from predictions.core.ports import CodeReadError
from predictions.core.review import CodeSelection, valid_code_format

URL = "https://www.sportybet.com/api/ng/orders/share/{code}"
TIMEOUT_SECONDS = 20.0
OVERALL_TIMEOUT_SECONDS = 30.0
_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    "Accept": "application/json",
}
OK_CODE = 10000
INVALID_CODE = 19000

INVALID_MESSAGE = "SportyBet says this booking code is invalid or has expired. Please check the code."
UNEXPECTED_MESSAGE = "SportyBet gave an answer for this code that I could not read. Nothing was reviewed."
EMPTY_MESSAGE = "This booking code has no selections."

# SportyBet sport ids (Sportradar) to the canonical names engines are registered under.
SPORTS = {
    "sr:sport:1": "soccer",
    "sr:sport:2": "basketball",
    "sr:sport:3": "baseball",
    "sr:sport:4": "hockey",
    "sr:sport:5": "tennis",
    "sr:sport:16": "american_football",
    "sr:sport:20": "table_tennis",
    "sr:sport:23": "volleyball",
    "sr:sport:12": "rugby",
    "sr:sport:31": "badminton",
    "sr:sport:117": "mma",
    "sr:sport:21": "cricket",
}


def _fail() -> CodeReadError:
    return CodeReadError(UNEXPECTED_MESSAGE)


def _canonical_sport(sport: dict) -> tuple[str, str]:
    sport_id, name = sport.get("id"), sport.get("name")
    label = str(name).strip() if name else ""
    if sport_id in SPORTS:
        return SPORTS[sport_id], label or SPORTS[sport_id]
    if label:  # a sport with no engine anyway: keep a stable slug so it is listed and disqualified by default
        return "_".join(label.lower().split()), label
    raise _fail()


def parse_selections(payload) -> list[CodeSelection]:
    """The selections on a share-code response, or CodeReadError. Complete or nothing."""
    if not isinstance(payload, dict):
        raise _fail()
    biz = payload.get("bizCode")
    if biz == INVALID_CODE:
        raise CodeReadError(INVALID_MESSAGE)
    if biz != OK_CODE:
        raise CodeReadError("SportyBet could not read this booking code (it may be invalid or expired).")
    data = payload.get("data")
    events = data.get("outcomes") if isinstance(data, dict) else None
    if not isinstance(events, list):
        raise _fail()
    if not events:
        raise CodeReadError(EMPTY_MESSAGE)
    selections: list[CodeSelection] = []
    for event in events:
        selections.extend(_event_selections(event))
    if not selections:
        raise CodeReadError(EMPTY_MESSAGE)
    return selections


def _event_selections(event) -> list[CodeSelection]:
    if not isinstance(event, dict):
        raise _fail()
    home, away, start = event.get("homeTeamName"), event.get("awayTeamName"), event.get("estimateStartTime")
    sport = event.get("sport")
    if (not home or not away or not isinstance(start, (int, float)) or isinstance(start, bool)
            or not isinstance(sport, dict)):
        raise _fail()
    try:
        kickoff = datetime.fromtimestamp(start / 1000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        raise _fail() from None
    category = sport.get("category") if isinstance(sport.get("category"), dict) else {}
    tournament = category.get("tournament") if isinstance(category.get("tournament"), dict) else {}
    country, league = category.get("name"), tournament.get("name") or "Unknown competition"
    competition = f"{country} - {league}" if country else str(league)  # the same form the fixture listing uses
    canonical, label = _canonical_sport(sport)
    started = _started(event)
    markets = event.get("markets")
    if not isinstance(markets, list) or not markets:
        raise _fail()
    found = []
    for market in markets:
        if not isinstance(market, dict) or market.get("id") is None or not isinstance(market.get("outcomes"), list):
            raise _fail()
        market_open = _market_open(market)
        for outcome in market["outcomes"]:
            if not isinstance(outcome, dict):
                raise _fail()
            odds, name = _decimal(outcome.get("odds")), outcome.get("desc")
            if odds is None or not name:
                raise _fail()
            found.append(CodeSelection(
                sport=canonical, sport_label=label, competition=competition, home=str(home), away=str(away),
                kickoff_utc=kickoff, started=started, market_id=str(market["id"]),
                market_name=str(market.get("desc") or market.get("name") or ""),
                specifier=str(market.get("specifier") or ""), outcome_name=str(name), odds=odds,
                available=market_open and _outcome_active(outcome)))
    return found


class SportyBetCodeReader:
    """CodeReader backed by SportyBet's public share-code endpoint. Anonymous, one GET, no retries."""

    def __init__(self, client: httpx.AsyncClient | None = None):
        self._client = client  # tests inject a mock transport; production makes a fresh client per read

    async def read_code(self, code: str) -> list[CodeSelection]:
        if not valid_code_format(code):
            raise CodeReadError(INVALID_MESSAGE)
        try:
            if self._client is not None:
                return await asyncio.wait_for(self._get(self._client, code), OVERALL_TIMEOUT_SECONDS)
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, headers=_HEADERS) as client:
                return await asyncio.wait_for(self._get(client, code), OVERALL_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raise CodeReadError("SportyBet took too long to answer. Nothing was reviewed.") from None

    async def _get(self, client: httpx.AsyncClient, code: str) -> list[CodeSelection]:
        try:
            response = await client.get(URL.format(code=code))
            response.raise_for_status()
            payload = response.json()
        except httpx.TimeoutException:
            raise CodeReadError("SportyBet did not answer in time. Nothing was reviewed.") from None
        except httpx.HTTPStatusError as error:
            raise CodeReadError(f"SportyBet returned HTTP {error.response.status_code}. "
                                "Nothing was reviewed.") from None
        except (httpx.HTTPError, ValueError):
            raise CodeReadError("SportyBet could not be reached or gave unreadable data. "
                                "Nothing was reviewed.") from None
        return parse_selections(payload)
