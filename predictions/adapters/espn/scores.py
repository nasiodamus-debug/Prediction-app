"""Final scores from ESPN's public scoreboard, for the Monday post-game review (diagnosis only; the bookmaker decides every
official result). It reads public data and never touches the betslip, the database or the analyst.

ESPN's `site.api.espn.com` needs no key and answers 403 to a custom User-Agent, so requests carry httpx's own default header. It
is an unofficial endpoint: every field is validated, a missing or odd one means "no score" (the leg is then UNVERIFIABLE), never a
guess. A scoreboard of a finished day does not change, so each (league, date) is remembered for a day (core/ttl_cache.py).

Covered: NBA and WNBA; NFL and NCAA football (FBS group 80, FCS group 81); MLB; NHL; and every soccer league ESPN has a slug for
(adapters/espn/soccer.py `slug_for`). Everything else returns None. Games are found by date (the kickoff day and the day before,
because ESPN files late-evening games under the US date) and by BOTH team names through the project's identity resolver.

Written from the public scoreboard shape (events -> competitions -> competitors with `homeAway`, `score`, `linescores`, and
`status.type.name`); the live shape was spot-checked on 2026-10-03.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx

from predictions.adapters.espn.soccer import ALIASES as SOCCER_ALIASES
from predictions.adapters.espn.soccer import slug_for
from predictions.core.identity import resolve_team
from predictions.core.settle import FinalScore
from predictions.core.ttl_cache import STATS_CACHE, request_key

log = logging.getLogger("scores")

BASE = "https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"
TIMEOUT_SECONDS = 20.0
TTL_SCOREBOARD_SECONDS = 24 * 3600

# (our sport, normalized league) -> (ESPN sport, ESPN league, extra query groups)
BASKETBALL = {"usa nba": "nba", "usa wnba": "wnba"}
HOCKEY = {"usa nhl": "nhl"}
BASEBALL = {"usa mlb": "mlb"}

# ESPN event status names that mean the game will not produce a result.
VOID_STATUSES = {"STATUS_POSTPONED": "postponed", "STATUS_CANCELED": "abandoned", "STATUS_CANCELLED": "abandoned",
                 "STATUS_ABANDONED": "abandoned", "STATUS_FORFEIT": "abandoned"}
SUSPENDED_STATUSES = {"STATUS_SUSPENDED", "STATUS_DELAYED", "STATUS_RAIN_DELAY"}
EXTRA_TIME_STATUSES = {"STATUS_FINAL_AET", "STATUS_FINAL_PEN", "STATUS_FINAL_ET"}


# Soccer leagues the xG / goals-only stats adapters do not need a slug for (Understat or football-data cover them there), so
# `slug_for` does not know them: the same ESPN slugs, checked to answer HTTP 200 on 2026-10-03. A name that does not match
# simply gives no score (the leg is then unverifiable).
EXTRA_SOCCER_SLUGS = {
    "england premier league": "eng.1", "england championship": "eng.2", "england league one": "eng.3", "england league two": "eng.4",
    "england national league": "eng.5", "england fa cup": "eng.fa", "england efl cup": "eng.league_cup",
    "germany bundesliga": "ger.1", "germany 2 bundesliga": "ger.2", "spain laliga": "esp.1", "spain laliga 2": "esp.2",
    "italy serie a": "ita.1", "italy serie b": "ita.2", "france ligue 1": "fra.1", "france ligue 2": "fra.2",
    "portugal liga portugal": "por.1", "turkiye super lig": "tur.1", "belgium jupiler pro league": "bel.1",
    "mexico liga mx": "mex.1", "mexico liga de expansion mx": "mex.2", "argentina primera lpf": "arg.1",
    "argentina liga profesional": "arg.1", "brazil serie a": "bra.1", "brazil serie b": "bra.2", "japan j1 league": "jpn.1",
    "greece super league": "gre.1", "denmark superliga": "den.1", "sweden allsvenskan": "swe.1", "norway eliteserien": "nor.1",
}


def parse_match_id(text: str):
    """'basketball:usa wnba:20261003T0100Z:golden state valkyries:dallas wings' -> (sport, league, kickoff UTC, home, away)."""
    parts = str(text).split(":")
    if len(parts) != 5:
        return None
    try:
        kickoff = datetime.strptime(parts[2], "%Y%m%dT%H%MZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return parts[0], parts[1], kickoff, parts[3], parts[4]


def endpoints(sport: str, league: str) -> list[tuple[str, str, dict]]:
    """The scoreboard URLs (ESPN sport, league, query extras) that can hold a game of this sport and normalized league."""
    if sport == "basketball" and league in BASKETBALL:
        return [("basketball", BASKETBALL[league], {})]
    if sport == "hockey" and league in HOCKEY:
        return [("hockey", HOCKEY[league], {})]
    if sport == "baseball" and league in BASEBALL:
        return [("baseball", BASEBALL[league], {})]
    if sport == "american_football":
        if league == "usa nfl":
            return [("football", "nfl", {})]
        if league.startswith("usa ncaa"):
            return [("football", "college-football", {"groups": "80"}), ("football", "college-football", {"groups": "81"})]
    if sport == "soccer":
        slug = slug_for(league) or EXTRA_SOCCER_SLUGS.get(league)
        if slug:
            return [("soccer", slug, {})]
    return []


def _int(value) -> int | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(number) if number == int(number) and number >= 0 else None


def parse_event(event: dict) -> tuple[str, str, FinalScore] | None:
    """(home name, away name, score) of one scoreboard event, or None when its shape is not the expected one."""
    try:
        competition = event["competitions"][0]
        sides = {c["homeAway"]: c for c in competition["competitors"]}
        home, away = sides["home"], sides["away"]
        names = (str(home["team"]["displayName"]), str(away["team"]["displayName"]))
        status = str((event.get("status") or competition.get("status") or {}).get("type", {}).get("name", ""))
    except (KeyError, IndexError, TypeError):
        return None
    if status in VOID_STATUSES:
        return names[0], names[1], FinalScore(0, 0, status=VOID_STATUSES[status])
    if status in SUSPENDED_STATUSES:
        return names[0], names[1], FinalScore(0, 0, status="suspended")
    if not status.startswith("STATUS_FINAL"):
        return None   # scheduled or in progress: no final score yet
    h, a = _int(home.get("score")), _int(away.get("score"))
    if h is None or a is None:
        return None
    periods = []
    home_lines, away_lines = home.get("linescores") or [], away.get("linescores") or []
    for hl, al in zip(home_lines, away_lines):
        hv, av = _int(hl.get("value")), _int(al.get("value"))
        if hv is None or av is None:
            periods = []
            break
        periods.append((hv, av))
    return names[0], names[1], FinalScore(h, a, "final", tuple(periods), None, status in EXTRA_TIME_STATUSES)


class EspnScores:
    def __init__(self, client: httpx.AsyncClient | None = None):
        self._client = client

    async def _scoreboard(self, client: httpx.AsyncClient, espn_sport: str, league: str, day: str, extra: dict) -> list[dict] | None:
        params = {"dates": day, "limit": "400", **extra}

        async def read():
            response = await client.get(BASE.format(sport=espn_sport, league=league), params=params, timeout=TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
            events = payload.get("events") if isinstance(payload, dict) else None
            if not isinstance(events, list):
                raise ValueError("no events list")
            return events

        try:
            return await STATS_CACHE.get_or_fetch(request_key("espn-scores", f"{espn_sport}/{league}", params),
                                                  TTL_SCOREBOARD_SECONDS, read)
        except (httpx.HTTPError, ValueError, OSError):
            return None

    async def final_score(self, match_id: str) -> tuple[FinalScore | None, str]:
        """(the score or None, a short note on why not). `match_id` is the stored match id text."""
        parsed = parse_match_id(match_id)
        if parsed is None:
            return None, "the match id could not be read"
        sport, league, kickoff, home, away = parsed
        targets = endpoints(sport, league)
        if not targets:
            return None, f"no free score source for {sport} league {league!r}"
        own = self._client is None
        client = self._client or httpx.AsyncClient(timeout=TIMEOUT_SECONDS)
        try:
            reachable = False
            for espn_sport, slug, extra in targets:
                for day in {(kickoff - timedelta(days=1)).strftime("%Y%m%d"), kickoff.strftime("%Y%m%d")}:
                    events = await self._scoreboard(client, espn_sport, slug, day, extra)
                    if events is None:
                        continue
                    reachable = True
                    found = self._pick(events, home, away)
                    if found is not None:
                        return found, ""
            return None, ("the game was not on the scoreboard (or not finished)" if reachable else "the score source could not be reached")
        finally:
            if own:
                await client.aclose()

    @staticmethod
    def _pick(events: list[dict], home: str, away: str) -> FinalScore | None:
        parsed = [p for p in (parse_event(e) for e in events) if p is not None]
        for espn_home, espn_away, score in parsed:
            h = resolve_team(home, [espn_home], SOCCER_ALIASES)
            a = resolve_team(away, [espn_away], SOCCER_ALIASES)
            if h.status == "matched" and a.status == "matched":
                return score
        return None
