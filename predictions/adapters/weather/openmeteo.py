"""Projected wind at a stadium from Open-Meteo's public forecast API (Story 7.4). Keyless, no login, read-only GET.

Probed 2026-09-30: `https://api.open-meteo.com/v1/forecast?latitude=..&longitude=..&hourly=wind_speed_10m&
wind_speed_unit=mph&timezone=UTC&start_date=..&end_date=..` answers hourly 10 m wind speeds (the mean speed over the
hour, i.e. SUSTAINED wind, not gusts) and covers about 16 days ahead. api.weather.gov (US only, needs a User-Agent and a
two-step lookup) was probed too and not used because Open-Meteo also covers the international NFL venues.

What is derived: the projected sustained wind of a game = the HIGHEST hourly mean wind speed from the kickoff hour to
WINDOW_HOURS hours later (a game lasts about three hours; the worst hour is the cautious reading for an Over).
Anything unreadable or out of the forecast horizon is None, which the engine treats as "no wind data" (an Over is then
"insufficient data"): a wind figure is never guessed and a weather outage never fails the slate.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import httpx

from predictions.adapters.weather.stadiums import Stadium
from predictions.core.ttl_cache import STATS_CACHE, TTL_WEATHER_SECONDS, request_key

NAME = "open-meteo"
URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 20.0
WINDOW_HOURS = 3            # hourly readings from the kickoff hour to this many hours later, inclusive
HORIZON_DAYS = 15           # the forecast reaches about 16 days; beyond this no figure is asked for
MAX_PLAUSIBLE_MPH = 150.0


def parse_wind(payload, kickoff: datetime) -> float | None:
    """The highest hourly wind (mph) from the kickoff hour through WINDOW_HOURS later, or None when the response does not
    hold every one of those hours as a finite number."""
    try:
        hourly = payload["hourly"]
        times, speeds = hourly["time"], hourly["wind_speed_10m"]
        if not isinstance(times, list) or not isinstance(speeds, list) or len(times) != len(speeds):
            return None
        first = kickoff.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
        wanted = [(first + timedelta(hours=h)).strftime("%Y-%m-%dT%H:00") for h in range(WINDOW_HOURS + 1)]
        by_time = dict(zip(times, speeds))
        values = []
        for stamp in wanted:
            value = by_time.get(stamp)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                return None
            if not 0.0 <= value <= MAX_PLAUSIBLE_MPH:
                return None
            values.append(float(value))
        return max(values)
    except (KeyError, TypeError, AttributeError, ValueError):
        return None


async def fetch_wind(client: httpx.AsyncClient, stadium: Stadium, kickoff: datetime, now: datetime) -> float | None:
    """Projected sustained wind in mph at this stadium for a game starting at `kickoff`, or None."""
    kickoff = kickoff.astimezone(timezone.utc)
    now = now.astimezone(timezone.utc)
    if kickoff < now - timedelta(hours=1) or kickoff > now + timedelta(days=HORIZON_DAYS):
        return None
    params = {"latitude": f"{stadium.latitude:.4f}", "longitude": f"{stadium.longitude:.4f}", "hourly": "wind_speed_10m",
              "wind_speed_unit": "mph", "timezone": "UTC", "start_date": kickoff.date().isoformat(),
              "end_date": (kickoff + timedelta(hours=WINDOW_HOURS)).date().isoformat()}
    async def read():
        response = await client.get(URL, params=params, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()

    try:   # the forecast is remembered for an hour (core/ttl_cache.py); a failed read is never remembered
        payload = await STATS_CACHE.get_or_fetch(request_key("open-meteo", "forecast", params), TTL_WEATHER_SECONDS, read)
        return parse_wind(payload, kickoff)
    except (httpx.HTTPError, ValueError):
        return None
