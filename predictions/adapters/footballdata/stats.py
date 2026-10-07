"""Corner, card and referee inputs from football-data.co.uk's public CSV files, behind the InputsSource port.

Free, no key, no login: one CSV per league and season (`mmz4281/<yyzz>/<code>.csv`, one row per played game,
with the referee and corners / yellow / red cards for both teams) and one `fixtures.csv` for the games in the
next day or two, which carries the ASSIGNED REFEREE for those games only. So:
- team corner and card rates are available for every covered league (last LAST_N games before kickoff);
- the referee's cards-per-game average exists only for games that appear in `fixtures.csv`, that is only
  shortly before kickoff. Any other fixture simply gets no referee input and its cards picks are rejected as
  "insufficient data". Nothing is guessed or filled in.
Not covered: Russia and every league not in LEAGUES. The site is an unofficial hobby source; every field is
validated, a change in shape fails loudly. No retries (AD-14). Wired into Matchday (app.py) by spec 11-1: SportyBet lists
corners and cards only for some games, found per game on its event endpoint; the cards here are YELLOW cards only (HY/AY),
which is what the owner plays ("yellow cards"; SportyBet's "Bookings" market).
"""
from __future__ import annotations

import asyncio
import csv
import io
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import httpx

from predictions.core.identity import ALIASES, resolve_team
from predictions.core.inputs import MIN_MATCHES, Input
from predictions.core.ports import StatsFetchError
from predictions.core.ttl_cache import STATS_CACHE, TTL_FORM_SECONDS, request_key
from predictions.core.types import Fixture, MatchId, normalize_team

NAME = "footballdata"
SEASON_URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"
TIMEOUT_SECONDS = 20.0
OVERALL_TIMEOUT_SECONDS = 90.0
LAST_N = 10
GAME_WINDOW = timedelta(days=1)  # the CSV date is local; the fixture's kickoff is UTC
_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}

# Normalized SportyBet "<country> - <league>" -> football-data code.
LEAGUES: dict[str, str] = {
    "england premier league": "E0",
    "england championship": "E1",
    "germany bundesliga": "D1",
    "germany 2 bundesliga": "D2",
    "spain laliga": "SP1",
    "spain la liga": "SP1",
    "italy serie a": "I1",
    "france ligue 1": "F1",
    "netherlands eredivisie": "N1",
    "belgium jupiler pro league": "B1",
    "portugal liga portugal": "P1",
    "portugal primeira liga": "P1",
    "turkey super lig": "T1",
    "turkiye super lig": "T1",             # SportyBet's spelling (spec 11-1 alias probe, 2026-10-01)
    "belgium pro league": "B1",
    "scotland premiership": "SC0",
    "greece super league": "G1",
}

# Normalized bookmaker spellings -> normalized football-data spellings, on top of core/identity's table.
# Not exhaustive: a club that does not resolve simply gets no inputs (its corners and cards picks are then
# "insufficient data"), it is never guessed.
_FD_SPELLINGS: dict[str, str] = {
    "manchester united": "man united", "manchester city": "man city", "nottingham forest": "nott m forest",
    "tottenham hotspur": "tottenham", "newcastle united": "newcastle", "leeds united": "leeds",
    "west ham united": "west ham", "wolverhampton wanderers": "wolves", "wolverhampton": "wolves",
    "atletico madrid": "ath madrid", "athletic club": "ath bilbao", "real sociedad": "sociedad",
    "real betis": "betis", "rayo vallecano": "vallecano", "celta vigo": "celta",
    "borussia dortmund": "dortmund", "bayer leverkusen": "leverkusen", "eintracht frankfurt": "ein frankfurt",
    "borussia monchengladbach": "m gladbach", "borussia m gladbach": "m gladbach",
    "ac milan": "milan", "paris saint germain": "paris sg", "olympique marseille": "marseille",
    "olympique lyonnais": "lyon",
    # Added by spec 11-1 after one read-only probe of today's real SportyBet listing against football-data's 2026/27 and
    # 2025/26 team lists (every club of the covered leagues now resolves; Understat-spelling targets are re-pointed too).
    "fc cologne": "fc koln", "cologne": "fc koln", "hamburger sv": "hamburg", "sv 07 elversberg": "elversberg",
    "schalke": "schalke 04", "rasenballsport leipzig": "rb leipzig",
    "coventry city": "coventry", "hull city": "hull", "ipswich town": "ipswich",
    "espanyol": "espanol", "rc deportivo de a coruna": "la coruna", "racing santander": "santander",
    "como 1907": "como", "hellas verona": "verona", "aj auxerre": "auxerre",
    "ado den haag": "den haag", "alkmaar": "az alkmaar", "excelsior rotterdam": "excelsior", "fc twente enschede": "twente",
    "fortuna sittard": "for sittard", "nec nijmegen": "nijmegen", "pec zwolle": "zwolle", "willem ii tilburg": "willem ii",
    "academico de viseu fc": "academico viseu", "alverca futebol": "alverca", "braga": "sp braga",
    "casa pia lisbon": "casa pia", "estoril praia": "estoril", "estrela amadora": "estrela",
    "gil vicente barcelos": "gil vicente", "maritimo madeira": "maritimo", "nacional da madeira": "nacional",
    "santa clara azores": "santa clara", "sporting": "sp lisbon", "sporting cp": "sp lisbon", "sporting lisbon": "sp lisbon",
    "vitoria sc guimaraes": "guimaraes",
    "amed sportif faaliyetler": "amedspor", "besiktas istanbul": "besiktas", "caykur rizespor": "rizespor",
    "fenerbahce istanbul": "fenerbahce", "galatasaray istanbul": "galatasaray", "genclerbirligi sk": "genclerbirligi",
    "goztepe izmir": "goztep", "istanbul basaksehir": "buyuksehyr", "gaziantep fk": "gaziantep",
    "eyupspor istanbul": "eyupspor", "kasimpasa sk": "kasimpasa", "kv kortrijk": "kortrijk", "kv waasland beveren": "beveren",
    "kvc westerlo": "westerlo", "rsc anderlecht": "anderlecht", "royal antwerp fc": "antwerp", "royal charleroi sc": "charleroi",
    "sv zulte waregem": "waregem", "st truiden": "st truiden", "royal union saint gilloise": "st gilloise",
    "krc genk": "genk", "kaa gent": "gent", "kv mechelen": "mechelen", "ou heverlee leuven": "oud heverlee leuven",
    "standard liege": "standard", "lommel sk": "lommel sk", "kasimpasa istanbul": "kasimpasa", "istanbul bb": "buyuksehyr",
    "union gilloise": "st gilloise", "yellow red kv mechelen": "mechelen", "st truidense vv": "st truiden",
    "dundee fc": "dundee", "heart of midlothian fc": "hearts", "heart of midlothian": "hearts",
    "ae kifisia fc": "kifisia", "aek athens": "aek", "apo levadiakos fc": "levadeiakos", "aris thessaloniki": "aris",
    "atromitos athens": "atromitos", "atromitos athinon": "atromitos", "olympiacos": "olympiakos",
    "pae ps kalamata": "kalamata", "pot iraklis": "iraklis", "panaitolikos agrinio": "panetolikos", "volos nps": "volos nfc",
}
# The shared table's targets (Understat spellings) are re-pointed to football-data spellings where they differ.
FD_ALIASES: dict[str, str] = {**{k: _FD_SPELLINGS.get(v, v) for k, v in ALIASES.items()}, **_FD_SPELLINGS}


@dataclass(frozen=True)
class PlayedGame:
    day: date
    home: str
    away: str
    referee: str
    corners_home: int
    corners_away: int
    cards_home: int   # yellow cards (spec 11-1: reds are not counted)
    cards_away: int


@dataclass(frozen=True)
class ScheduledGame:
    day: date
    home: str
    away: str
    referee: str


def _bad() -> StatsFetchError:
    return StatsFetchError("football-data.co.uk returned data in an unexpected format")


def _day(text) -> date:
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(str(text).strip(), fmt).date()
        except ValueError:
            continue
    raise _bad()


def _int(value) -> int | None:
    try:
        number = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if 0 <= number <= 100 else None


def _rows(text: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    if not reader.fieldnames or not {"Date", "HomeTeam", "AwayTeam"} <= set(reader.fieldnames):
        raise _bad()
    return [row for row in reader if row.get("HomeTeam") and row.get("AwayTeam") and row.get("Date")]


def parse_season(text: str) -> list[PlayedGame]:
    """Played games only: a row without corners and card counts is skipped, never treated as zero."""
    games = []
    for row in _rows(text):
        counts = [_int(row.get(k)) for k in ("HC", "AC", "HY", "AY")]
        if any(c is None for c in counts):
            continue
        hc, ac, hy, ay = counts
        games.append(PlayedGame(_day(row["Date"]), row["HomeTeam"].strip(), row["AwayTeam"].strip(),
                                (row.get("Referee") or "").strip(), hc, ac, hy, ay))
    return sorted(games, key=lambda g: g.day)


def parse_fixtures(text: str) -> list[ScheduledGame]:
    return [ScheduledGame(_day(row["Date"]), row["HomeTeam"].strip(), row["AwayTeam"].strip(),
                          (row.get("Referee") or "").strip()) for row in _rows(text)]


def season_code(kickoff: datetime) -> str:
    start = kickoff.year if kickoff.month >= 7 else kickoff.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


def previous_code(code: str) -> str:
    a = (int(code[:2]) - 1) % 100
    return f"{a:02d}{int(code[:2]):02d}"


def team_inputs(side: str, team: str, games: list[PlayedGame], source: str) -> list[Input]:
    """Averages over the team's `games` (already the last N before kickoff)."""
    own_corners, opp_corners, own_cards = [], [], []
    for g in games:
        home = g.home == team
        own_corners.append(g.corners_home if home else g.corners_away)
        opp_corners.append(g.corners_away if home else g.corners_home)
        own_cards.append(g.cards_home if home else g.cards_away)
    n = len(games)
    if n < MIN_MATCHES:
        return []

    def mean(values) -> float:
        return round(sum(values) / n, 4)

    return [Input(f"{side}.corners_for", mean(own_corners), "corners per match", source),
            Input(f"{side}.corners_against", mean(opp_corners), "corners per match", source),
            Input(f"{side}.cards_for", mean(own_cards), "cards per match", source)]  # yellow cards


def referee_inputs(referee: str, games: list[PlayedGame], source: str) -> list[Input]:
    """The referee's cards per game over the games listed under his name (both teams' cards)."""
    mine = [g for g in games if g.referee == referee]
    if not mine:
        return []
    avg = round(sum(g.cards_home + g.cards_away for g in mine) / len(mine), 4)
    return [Input("match.referee_cards_avg", avg, "cards per game", f"{source}:referee {referee}"),
            Input("match.referee_games", float(len(mine)), "games", f"{source}:referee {referee}")]


class FootballDataInputs:
    """InputsSource backed by football-data.co.uk. At most two season files per league, plus fixtures.csv."""

    def __init__(self, client: httpx.AsyncClient | None = None):
        self._client = client

    async def fetch_inputs(self, fixtures: list[Fixture]) -> dict[MatchId, list[Input]]:
        try:
            if self._client is not None:
                return await asyncio.wait_for(self._run(self._client, fixtures), OVERALL_TIMEOUT_SECONDS)
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS, headers=_HEADERS,
                                         follow_redirects=True) as client:
                return await asyncio.wait_for(self._run(client, fixtures), OVERALL_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raise StatsFetchError("football-data.co.uk took too long to answer") from None

    async def _get(self, client, url: str) -> str:
        """One GET, remembered for a few hours (core/ttl_cache.py): season files and the fixtures list, never an error."""
        return await STATS_CACHE.get_or_fetch(request_key("football-data", url), TTL_FORM_SECONDS,
                                              lambda: self._read(client, url))

    async def _read(self, client, url: str) -> str:
        try:
            response = await client.get(url)
            response.raise_for_status()
            return response.text
        except httpx.TimeoutException:
            raise StatsFetchError("football-data.co.uk did not answer in time") from None
        except httpx.HTTPStatusError as error:
            raise StatsFetchError(f"football-data.co.uk returned HTTP {error.response.status_code}") from None
        except httpx.HTTPError:
            raise StatsFetchError("football-data.co.uk could not be reached") from None

    async def _run(self, client, fixtures: list[Fixture]) -> dict[MatchId, list[Input]]:
        seasons: dict[tuple[str, str], list[PlayedGame]] = {}
        schedule: list[ScheduledGame] | None = None
        results: dict[MatchId, list[Input]] = {}
        for fixture in fixtures:
            code = LEAGUES.get(normalize_team(fixture.competition))
            if code is None:
                continue
            season = season_code(fixture.kickoff_utc)
            games = []
            for s in (previous_code(season), season):
                if (code, s) not in seasons:
                    seasons[(code, s)] = parse_season(await self._get(client, SEASON_URL.format(season=s, code=code)))
                games += seasons[(code, s)]
            teams = {g.home for g in games} | {g.away for g in games}
            resolved = {}
            for side, name in (("home", fixture.home), ("away", fixture.away)):
                match = resolve_team(name, teams, FD_ALIASES)
                if match.status != "matched":
                    break  # unknown or ambiguous: no inputs (the stats source is the one that asks the owner)
                resolved[side] = match.team
            if len(resolved) < 2 or resolved["home"] == resolved["away"]:
                continue
            day = fixture.kickoff_utc.date()
            source = f"{NAME}:{code}:{season}"
            inputs: list[Input] = []
            for side, team in resolved.items():
                played = [g for g in games if team in (g.home, g.away) and g.day < day]
                inputs += team_inputs(side, team, played[-LAST_N:], source)
            if schedule is None:
                schedule = parse_fixtures(await self._get(client, FIXTURES_URL))
            listed = [g for g in schedule if g.referee and abs(g.day - day) <= GAME_WINDOW
                      and g.home == resolved["home"] and g.away == resolved["away"]]
            if len(listed) == 1:
                inputs += referee_inputs(listed[0].referee, games, source)
            if inputs:
                results[fixture.match_id] = inputs
        return results
