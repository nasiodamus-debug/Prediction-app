"""API-Football (api-sports.io) reads for soccer: the day's fixtures, injuries, lineups and the vendor's win prediction.

Free-plan facts probed live on 2026-10-03 (see the spec notes): a date query works only for yesterday and today
(UTC) and the current season is refused, `last=N` is refused, and the allowance is 100 requests a day. So this adapter
asks about TODAY'S games only, by date (one fixtures call and one injuries call cover every league), and caches
every answer (core/ttl_cache.py). It reads public data only and never touches the betslip, the database or the
analyst. Any failure (an error answer, the plan refusing a date, a timeout) gives "no data", never a failed slate.
Written from the live probe; the lineups answer was only seen empty or for one women's game.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime, timezone

import httpx

from predictions.core.ttl_cache import STATS_CACHE, request_key

API_KEY_ENV = "API_FOOTBALL_KEY"
BASE_URL = "https://v3.football.api-sports.io"
TIMEOUT_SECONDS = 20.0
TTL_FIXTURES_SECONDS = 3600
TTL_INJURIES_SECONDS = 1800
TTL_LINEUPS_SECONDS = 300
TTL_PREDICTION_SECONDS = 3 * 3600


@dataclass(frozen=True)
class ApiFixture:
    id: int
    home: str
    away: str
    kickoff_utc: datetime
    league: str


@dataclass(frozen=True)
class Missing:
    """A player the service lists as out for a fixture (an injury, a suspension, a doubt)."""
    team: str
    player: str
    kind: str     # for example "Missing Fixture" or "Questionable"
    reason: str   # for example "Muscle Injury" or "Yellow Cards"


@dataclass(frozen=True)
class Prediction:
    """The vendor's win probabilities (0 to 1) for one fixture. Used only to warn the owner, never to pick."""
    home: float
    draw: float
    away: float


def _check(payload) -> list:
    """The `response` list of an answer, or ValueError when the service reported errors (for example the plan refusing a
    date): such an answer is not cached and the caller treats it as no data."""
    if not isinstance(payload, dict) or payload.get("errors"):
        raise ValueError("API-Football refused the request")
    response = payload.get("response")
    if not isinstance(response, list):
        raise ValueError("API-Football gave no response list")
    return response


def parse_fixtures(response: list) -> list[ApiFixture]:
    found = []
    for item in response:
        try:
            fixture, teams = item["fixture"], item["teams"]
            kickoff = datetime.fromisoformat(fixture["date"]).astimezone(timezone.utc)
            found.append(ApiFixture(int(fixture["id"]), str(teams["home"]["name"]), str(teams["away"]["name"]), kickoff,
                                    str(item["league"]["name"])))
        except (KeyError, TypeError, ValueError):
            continue
    return found


def parse_injuries(response: list) -> dict[int, tuple[Missing, ...]]:
    by_fixture: dict[int, list[Missing]] = {}
    for item in response:
        try:
            player = item["player"]
            by_fixture.setdefault(int(item["fixture"]["id"]), []).append(
                Missing(str(item["team"]["name"]), str(player["name"]), str(player.get("type") or ""),
                        str(player.get("reason") or "")))
        except (KeyError, TypeError, ValueError):
            continue
    return {k: tuple(v) for k, v in by_fixture.items()}


def parse_lineups_published(response: list) -> bool:
    """True when both teams' starting XI are listed."""
    teams = [t for t in response if isinstance(t, dict) and t.get("startXI")]
    return len(teams) >= 2


def parse_prediction(response: list) -> Prediction | None:
    try:
        percent = response[0]["predictions"]["percent"]
        values = [float(str(percent[k]).strip().rstrip("%")) / 100 for k in ("home", "draw", "away")]
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    if any(not 0.0 <= v <= 1.0 for v in values) or not 0.95 <= sum(values) <= 1.05:
        return None
    return Prediction(*values)


class ApiFootball:
    def __init__(self, client=None, api_key: str | None = None):
        if client is None:
            if not api_key:
                raise ValueError("an API key or a client is required")
            client = httpx.AsyncClient(timeout=TIMEOUT_SECONDS, headers={"x-apisports-key": api_key})
        self._client = client

    @classmethod
    def from_environment(cls) -> "ApiFootball | None":
        """The reader when API_FOOTBALL_KEY is set, else None (the free flags and the model check are then simply off)."""
        key = (os.environ.get(API_KEY_ENV) or "").strip()
        return cls(api_key=key) if key else None

    async def _get(self, path: str, params: dict, ttl: float, parse):
        async def read():
            response = await self._client.get(BASE_URL + path, params=params)
            response.raise_for_status()
            return parse(_check(response.json()))

        try:
            return await STATS_CACHE.get_or_fetch(request_key("api-football", path, params), ttl, read)
        except (httpx.HTTPError, OSError, ValueError):
            return None

    async def fixtures(self, day: date) -> list[ApiFixture]:
        return await self._get("/fixtures", {"date": day.isoformat()}, TTL_FIXTURES_SECONDS, parse_fixtures) or []

    async def injuries(self, day: date) -> dict[int, tuple[Missing, ...]]:
        return await self._get("/injuries", {"date": day.isoformat()}, TTL_INJURIES_SECONDS, parse_injuries) or {}

    async def lineups_published(self, fixture_id: int) -> bool | None:
        """True or False, or None when it could not be read (never guessed as 'not published')."""
        return await self._get("/fixtures/lineups", {"fixture": fixture_id}, TTL_LINEUPS_SECONDS, parse_lineups_published)

    async def prediction(self, fixture_id: int) -> Prediction | None:
        return await self._get("/predictions", {"fixture": fixture_id}, TTL_PREDICTION_SECONDS, parse_prediction)
