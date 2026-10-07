"""Soccer team stats from ESPN's public site API, GOALS ONLY (no xG), behind the StatsSource port (Story 9.3).

ESPN's `site.api.espn.com` needs no key and no login and answers 403 to a custom User-Agent, so requests carry httpx's own
default header and nothing else (as for basketball, adapters/espn/stats.py). It is unofficial: every field is validated and
a change in its shape fails loudly (StatsFetchError). Probed 2026-10-01, see spec-9-3.

What it covers: only the competitions in COMPETITIONS (SportyBet's name, normalized -> ESPN league slug) and the national-team
rules in `national_slug`: since Story 10.1 also the top divisions Understat lacks (DOMESTIC), the continental club cups
(CONTINENTAL) and senior World Cup / European Championship / AFCON / Copa America / Asian Cup games with their qualifiers. Earlier: the UEFA Nations
League, UEFA Champions League (men and women), Europa League, Conference League, U21 European Championship qualification,
FIFA international friendlies, club friendlies, the CONCACAF Nations League and Africa Cup of Nations qualification. Every
other competition is "not covered by the stats source" (Understat covers six domestic leagues; see flow/stats.py).

How a fixture is answered:
  1. The league scoreboard of the kickoff's date(s) lists the day's games; the two team names are matched to the teams
     playing then with the ONE identity resolver (ambiguous = ask, never guess) and the pair must be a listed game near the
     kickoff. The competition fixes the squad type: in a women's competition "W"/"WFC"/"Women" is dropped from both names,
     in an under-21 competition "U21" is, so a women's or youth side is never matched to a senior side (a name with an
     extra W/U21 in any other competition stays unmatched).
  2. Each team's `all/teams/<id>/schedule` (every competition, so recent form comes from friendlies and qualifiers too) gives
     its completed matches before kickoff; when fewer than LAST_N are found the previous season is read as well.
  3. The inputs are matches, goals_for and goals_against per match over the last LAST_N matches. Fewer than the
     goals-only minimum (8, core/inputs.py) is "insufficient data". The soccer engine prices such a fixture with its
     separate goals-only model (core/engines/soccer.py) and labels every pick "goals-only estimate (no xG)".
No retries: a failed read is reported (AD-14).
"""
from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx

from predictions.core.identity import ALIASES as BASE_ALIASES, resolve_team
from predictions.core.inputs import GOALS_ONLY_MIN_MATCHES, Input
from predictions.core.ports import StatsFetchError
from predictions.core.stats import FixtureStats, ambiguous, assess_goals_only, insufficient
from predictions.core.ttl_cache import STATS_CACHE, TTL_FORM_SECONDS, request_key
from predictions.core.types import Fixture, MatchId, normalize_team

NAME = "espn"
BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer/"
TIMEOUT_SECONDS = 20.0
OVERALL_TIMEOUT_SECONDS = 240.0
CONCURRENCY = 4           # polite: at most this many requests at once
LAST_N = 10               # a team's inputs come from its last LAST_N completed matches before kickoff
GAME_WINDOW = timedelta(hours=24)  # ESPN's kickoff may differ from the bookmaker's by this much
EASTERN = ZoneInfo("America/New_York")  # ESPN's scoreboard days may follow US time: the day in UTC and in ET are both read
NOT_COVERED = "this competition is not covered by the stats source"

# SportyBet "<region> - <competition>" (normalized) -> ESPN league slug. Anything else is not covered. Only names SportyBet
# was seen to use are listed (2026-10-01); a name that differs is simply "not covered", never guessed.
COMPETITIONS: dict[str, str] = {
    "international uefa nations league": "uefa.nations",
    "international clubs uefa champions league": "uefa.champions",
    "international clubs uefa champions league women": "uefa.wchampions",
    "international clubs uefa europa league": "uefa.europa",
    "international clubs uefa europa conference league": "uefa.europa.conf",
    "international clubs uefa conference league": "uefa.europa.conf",
    "international youth u21 uefa european championship qualification": "uefa.euro_u21_qual",
    "international int friendly games": "fifa.friendly",
    "international clubs club friendly games": "club.friendly",
    "international concacaf nations league": "concacaf.nations.league",
    "international africa cup of nations qualification": "caf.nations_qual",
}
# Story 10.1: the top divisions Understat does not cover (it has England, Spain, Germany, Italy, France and Russia, which are
# answered first), as SportyBet spells them (normalized) -> ESPN slug. Slugs probed read-only 2026-10-01 (each answered 200);
# these have NO ESPN slug (400): Ukraine, Poland, Croatia, Serbia, Bulgaria, Slovakia, Slovenia, Hungary, Canada, Panama,
# Iceland, Estonia. A league without a slug is "not covered" and costs no request.
DOMESTIC: dict[str, str] = {
    "netherlands eredivisie": "ned.1", "portugal liga portugal": "por.1", "turkiye super lig": "tur.1",
    "scotland premiership": "sco.1", "belgium pro league": "bel.1", "greece super league": "gre.1",
    "austria bundesliga": "aut.1", "switzerland super league": "sui.1", "denmark superliga": "den.1",
    "norway eliteserien": "nor.1", "sweden allsvenskan": "swe.1", "finland veikkausliiga": "fin.1",
    "czechia 1 liga": "cze.1", "romania superliga": "rou.1", "cyprus 1st division": "cyp.1",
    "ireland premier division": "irl.1", "northern ireland premiership": "nir.1", "wales cymru premier": "wal.1",
    "israel ligat haal": "isr.1", "brazil brasileiro serie a": "bra.1", "argentina primera lpf": "arg.1",
    "argentina primera division": "arg.1", "colombia liga dimayor": "col.1", "chile primera division": "chi.1",
    "uruguay primera division": "uru.1", "paraguay division de honor": "par.1", "ecuador ligapro primera a": "ecu.1",
    "peru liga 1": "per.1", "bolivia division profesional": "bol.1", "venezuela liga futve": "ven.1",
    "usa mls": "usa.1", "mexico liga mx": "mex.1", "costa rica primera division": "crc.1",
    "costa rica liga promerica": "crc.1", "el salvador primera division": "slv.1", "honduras liga nacional": "hon.1",
    "guatemala liga nacional": "gua.1", "jamaica premier league": "jam.1",
}
CONTINENTAL: dict[str, str] = {
    "international clubs conmebol libertadores": "conmebol.libertadores", "international clubs copa libertadores": "conmebol.libertadores",
    "international clubs conmebol sudamericana": "conmebol.sudamericana", "international clubs copa sudamericana": "conmebol.sudamericana",
    "international clubs concacaf champions cup": "concacaf.champions", "international clubs concacaf champions league": "concacaf.champions",
    "international clubs caf champions league": "caf.champions",
}
COMPETITIONS.update(DOMESTIC)
COMPETITIONS.update(CONTINENTAL)
WOMENS_SLUGS = frozenset({"uefa.wchampions", "fifa.friendly.w", "fifa.wwc", "fifa.wworldq.uefa", "uefa.weuro",
                          "concacaf.womens.championship", "uefa.w.nations"})
YOUTH_SLUGS = frozenset({"uefa.euro_u21_qual"})
_WOMEN_TOKENS = frozenset({"w", "wfc", "women", "womens", "ladies"})
_YOUTH_TOKEN = re.compile(r"^u\d{2}$")

_REGION_SLUGS = (("uefa", "uefa"), ("europe", "uefa"), ("caf", "caf"), ("africa", "caf"), ("conmebol", "conmebol"),
                 ("south", "conmebol"), ("concacaf", "concacaf"), ("north", "concacaf"), ("central", "concacaf"),
                 ("afc", "afc"), ("asia", "afc"), ("ofc", "ofc"), ("oceania", "ofc"))


def national_slug(competition: str) -> str | None:
    """The ESPN slug of a senior national-team competition SportyBet files under "International" (World Cup and its
    qualifiers, European Championship, Nations Leagues, AFCON, Copa America, Gold Cup, Asian Cup, friendlies), men or women.
    Read by rule because SportyBet's qualifier names carry a region word that varies; youth sides never match. None when
    ESPN has no slug for it (or the name is something else)."""
    country, sep, league = str(competition).partition(" - ")
    if not sep or normalize_team(country) != "international":
        return None
    words = normalize_team(league).split()
    text = " ".join(words)
    if any(_YOUTH_TOKEN.match(w) for w in words) or {"youth", "reserve", "reserves", "srl", "simulated"} & set(words):
        return None
    women = bool(_WOMEN_TOKENS & set(words))
    qual = any(w.startswith("qualif") for w in words)
    region = next((r for w in words for k, r in _REGION_SLUGS if w == k), "")
    if "world cup" in text:
        if women:
            return "fifa.wworldq.uefa" if qual and region == "uefa" else None if qual else "fifa.wwc"
        if not qual:
            return "fifa.world"
        return f"fifa.worldq.{region}" if region else "fifa.worldq"
    if "nations league" in text:
        if "concacaf" in words:
            return "concacaf.nations.league"
        if women:
            return "uefa.w.nations" if region in ("", "uefa") else None
        return "uefa.nations" if region in ("", "uefa") else None
    if "european championship" in text or "uefa euro" in text:
        if women:
            return None if qual else "uefa.weuro"
        return "uefa.euroq" if qual else "uefa.euro"
    if "africa cup of nations" in text or "african cup of nations" in text or "afcon" in words:
        return None if women else "caf.nations_qual" if qual else "caf.nations"
    if "copa america" in text:
        return None if women or qual else "conmebol.america"
    if "gold cup" in text:
        return None if women else "concacaf.gold"
    if "concacaf" in words and "championship" in words:
        return "concacaf.womens.championship" if women else None
    if "asian cup" in text:
        return None if women or qual else "afc.asian.cup"
    if "friendly" in text or "friendlies" in text:
        return "fifa.friendly.w" if women else "fifa.friendly"
    return None


def slug_for(competition: str) -> str | None:
    """ESPN's league slug for SportyBet's competition text: the exact table first, then the national-team rules."""
    return COMPETITIONS.get(normalize_team(competition)) or national_slug(competition)


# SportyBet's spelling -> ESPN's (both normalized). Each is an identity statement about ONE team; the target must be in
# the day's list for it to apply. Everything else goes through the identity resolver, which asks when unsure.
ALIASES: dict[str, str] = {
    **BASE_ALIASES,
    "ireland": "republic of ireland",
    "virgin islands british": "british virgin islands",
    "virgin islands u s": "us virgin islands",
    "fk austria wien": "austria vienna",
    "hb koege": "hb køge",
    "oud heverlee leuven": "oh leuven",
    "inter milano": "internazionale",
    "jahn regensburg": "ssv jahn regensburg",
    "wsg tirol": "wsg swarovski tirol",
    "wolfsburg": "vfl wolfsburg",
    "stuttgart": "vfb stuttgart",
    "dac 1904 dunajska streda": "dunajska streda",
    "puskas akademia fc felcsut": "puskas akademia",
    "us palmese 1914": "palmese",
}


@dataclass(frozen=True)
class ListedGame:
    when: datetime  # UTC
    teams: tuple[tuple[str, str], ...]  # (team id, display name) of the two sides


@dataclass(frozen=True)
class PlayedMatch:
    event_id: str
    when: datetime  # UTC
    goals_for: int
    goals_against: int


def _bad() -> StatsFetchError:
    return StatsFetchError("ESPN returned an unexpected response")


def _when(text) -> datetime:
    try:
        return datetime.strptime(str(text), "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
    except ValueError:
        raise _bad() from None


def _goals(value) -> int | None:
    if isinstance(value, dict):
        value = value.get("value", value.get("displayValue"))
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if number == int(number) and 0 <= number <= 30 else None


def parse_scoreboard(payload) -> list[ListedGame]:
    """The games of one scoreboard day (any status), each with its two teams. A payload that is not a scoreboard is an error."""
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise _bad()
    games = []
    for event in payload["events"]:
        if not isinstance(event, dict):
            raise _bad()
        competitions = event.get("competitions")
        if not isinstance(competitions, list) or len(competitions) != 1:
            raise _bad()
        teams = []
        for competitor in competitions[0].get("competitors") or []:
            team = (competitor or {}).get("team") or {}
            if team.get("id") and team.get("displayName"):
                teams.append((str(team["id"]), str(team["displayName"])))
        if len(teams) == 2 and teams[0][0] != teams[1][0]:
            games.append(ListedGame(_when(event.get("date")), (teams[0], teams[1])))
    return games


def parse_schedule(payload, team_id: str) -> tuple[list[PlayedMatch], int | None]:
    """(the team's completed matches with a score, the season year ESPN reports). Not-finished events are left out;
    a payload that is not a schedule is an error."""
    if not isinstance(payload, dict) or not isinstance(payload.get("events"), list):
        raise _bad()
    year = (payload.get("requestedSeason") or {}).get("year")
    played = []
    for event in payload["events"]:
        if not isinstance(event, dict):
            raise _bad()
        competitions = event.get("competitions")
        if not isinstance(competitions, list) or len(competitions) != 1 or not event.get("id"):
            raise _bad()
        competition = competitions[0]
        if (((competition.get("status") or {}).get("type") or {}).get("completed")) is not True:
            continue
        sides = {}
        for competitor in competition.get("competitors") or []:
            tid = ((competitor or {}).get("team") or {}).get("id")
            goals = _goals((competitor or {}).get("score"))
            if tid is not None and goals is not None:
                sides[str(tid)] = goals
        if str(team_id) in sides and len(sides) == 2:
            opponent = next(g for t, g in sides.items() if t != str(team_id))
            played.append(PlayedMatch(str(event["id"]), _when(event.get("date")), sides[str(team_id)], opponent))
    return played, year if type(year) is int else None


def team_inputs(side: str, matches: list[PlayedMatch], source: str) -> list[Input]:
    """matches, goals_for and goals_against per match over `matches` (already the last N before kickoff)."""
    n = len(matches)
    if n == 0:
        return []
    return [Input(f"{side}.matches", float(n), "matches", source),
            Input(f"{side}.goals_for", round(sum(m.goals_for for m in matches) / n, 4), "goals per match", source),
            Input(f"{side}.goals_against", round(sum(m.goals_against for m in matches) / n, 4), "goals per match",
                  source)]


def _canon(name: str, slug: str) -> str:
    """The name as compared inside one competition: the squad type is fixed by the competition, so a women's or under-21
    marker is dropped (and only there)."""
    tokens = normalize_team(name).split()
    if slug in WOMENS_SLUGS:
        tokens = [t for t in tokens if t not in _WOMEN_TOKENS]
    if slug in YOUTH_SLUGS:
        tokens = [t for t in tokens if not _YOUTH_TOKEN.match(t)]
    return " ".join(tokens)


class EspnSoccerStats:
    """StatsSource for soccer competitions Understat does not cover, backed by ESPN (goals only)."""

    def __init__(self, client: httpx.AsyncClient | None = None):
        self._client = client

    async def fetch_stats(self, fixtures: list[Fixture]) -> dict[MatchId, FixtureStats]:
        try:
            if self._client is not None:
                return await asyncio.wait_for(self._run(self._client, fixtures), OVERALL_TIMEOUT_SECONDS)
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:  # httpx's own headers: see the module note
                return await asyncio.wait_for(self._run(client, fixtures), OVERALL_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            raise StatsFetchError("ESPN took too long to answer") from None

    async def _get(self, client, gate: asyncio.Semaphore, path: str, **params):
        """One GET, remembered for a few hours (core/ttl_cache.py): team form and schedules only, never an error."""
        return await STATS_CACHE.get_or_fetch(request_key("espn-soccer", path, params), TTL_FORM_SECONDS,
                                              lambda: self._read(client, gate, path, **params))

    async def _read(self, client, gate: asyncio.Semaphore, path: str, **params):
        """One GET: the JSON, or None for a 404; anything else that goes wrong is a StatsFetchError. No retries."""
        async with gate:
            try:
                response = await client.get(BASE_URL + path, params=params)
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return response.json()
            except httpx.TimeoutException:
                raise StatsFetchError("ESPN did not answer in time") from None
            except httpx.HTTPStatusError as error:
                raise StatsFetchError(f"ESPN returned HTTP {error.response.status_code}") from None
            except (httpx.HTTPError, ValueError):
                raise StatsFetchError("ESPN could not be reached or gave unreadable data") from None

    async def _run(self, client, fixtures: list[Fixture]) -> dict[MatchId, FixtureStats]:
        results: dict[MatchId, FixtureStats] = {}
        wanted: list[tuple[Fixture, str]] = []
        for fixture in fixtures:
            slug = slug_for(fixture.competition)
            if fixture.match_id.sport != "soccer" or slug is None:
                results[fixture.match_id] = insufficient(fixture.match_id, NOT_COVERED)
            else:
                wanted.append((fixture, slug))
        if not wanted:
            return results
        gate = asyncio.Semaphore(CONCURRENCY)
        # 1. the scoreboards of every (competition, day) needed
        days: dict[tuple[str, str], list[ListedGame]] = {}
        keys = sorted({(slug, d) for f, slug in wanted for d in self._days(f.kickoff_utc)})
        boards = await asyncio.gather(*(self._get(client, gate, f"{slug}/scoreboard", dates=day) for slug, day in keys))
        for key, payload in zip(keys, boards):
            days[key] = parse_scoreboard(payload) if payload is not None else []
        # 2. identify the two teams of each fixture on those boards
        found: dict[MatchId, tuple[str, str, str, str]] = {}  # match -> (home id, away id, home name, away name)
        for fixture, slug in wanted:
            listed = [g for d in self._days(fixture.kickoff_utc) for g in days[(slug, d)]
                      if abs(g.when - fixture.kickoff_utc) <= GAME_WINDOW]
            outcome = self._identify(fixture, slug, listed)
            if isinstance(outcome, FixtureStats):
                results[fixture.match_id] = outcome
            else:
                found[fixture.match_id] = outcome
        # 3. each needed team's history (once per team and kickoff season)
        histories: dict[tuple[str, int], list[PlayedMatch]] = {}
        need = {(tid, self._season(f.kickoff_utc)) for f, _ in wanted if f.match_id in found
                for tid in found[f.match_id][:2]}
        loaded = await asyncio.gather(*(self._history(client, gate, tid, season) for tid, season in sorted(need)))
        for key, matches in zip(sorted(need), loaded):
            histories[key] = matches
        for fixture, slug in wanted:
            if fixture.match_id in found:
                results[fixture.match_id] = self._assess(fixture, slug, found[fixture.match_id], histories)
        return results

    @staticmethod
    def _days(kickoff: datetime) -> list[str]:
        utc = kickoff.astimezone(timezone.utc).strftime("%Y%m%d")
        eastern = kickoff.astimezone(EASTERN).strftime("%Y%m%d")
        return sorted({utc, eastern})

    @staticmethod
    def _season(kickoff: datetime) -> int:
        return kickoff.year if kickoff.month >= 7 else kickoff.year - 1

    @staticmethod
    def _identify(fixture: Fixture, slug: str, listed: list[ListedGame]):
        """(home id, away id, home name, away name) of the listed game, or a FixtureStats saying why not."""
        mid = fixture.match_id
        teams: dict[str, tuple[str, str]] = {}  # canonical name -> (id, ESPN name)
        for game in listed:
            for tid, name in game.teams:
                teams.setdefault(_canon(name, slug), (tid, name))
        aliases = {_canon(k, slug): _canon(v, slug) for k, v in ALIASES.items()}
        resolved: dict[str, tuple[str, str]] = {}
        for side, name in (("home", fixture.home), ("away", fixture.away)):
            match = resolve_team(_canon(name, slug), list(teams), aliases)
            if match.status == "ambiguous":
                names = [teams[c][1] for c in match.matches]
                return ambiguous(mid, f"Which team is '{name}' ({side})? It could be: {', '.join(names)}.", names)
            if match.status == "none":
                return insufficient(mid, f"the {side} team '{name}' is not in the stats source's {slug} list for this date")
            resolved[side] = teams[match.team]
        home, away = resolved["home"], resolved["away"]
        if home[0] == away[0]:
            return ambiguous(mid, f"'{fixture.home}' and '{fixture.away}' both match {home[1]}; which is which?", (home[1],))
        pair = {home[0], away[0]}
        if not any({t for t, _ in g.teams} == pair for g in listed):
            return insufficient(mid, f"the stats source lists no {home[1]} v {away[1]} game near this kickoff")
        return home[0], away[0], home[1], away[1]

    async def _history(self, client, gate, team_id: str, season: int) -> list[PlayedMatch]:
        """Completed matches of the team across competitions: the current schedule, and the previous season's when the
        current one holds fewer than LAST_N."""
        payload = await self._get(client, gate, f"all/teams/{team_id}/schedule")
        matches, year = parse_schedule(payload, team_id) if payload is not None else ([], None)
        if len(matches) < LAST_N:
            earlier = await self._get(client, gate, f"all/teams/{team_id}/schedule", season=(year or season) - 1)
            if earlier is not None:
                matches = matches + parse_schedule(earlier, team_id)[0]
        unique = {m.event_id: m for m in matches}
        return sorted(unique.values(), key=lambda m: m.when)

    @staticmethod
    def _assess(fixture: Fixture, slug: str, found, histories) -> FixtureStats:
        home_id, away_id, home_name, away_name = found
        season = EspnSoccerStats._season(fixture.kickoff_utc)
        source = f"{NAME}:goals-only:{slug}"
        raw: list[Input] = []
        problems = []
        for side, tid, name in (("home", home_id, home_name), ("away", away_id, away_name)):
            before = [m for m in histories[(tid, season)] if m.when < fixture.kickoff_utc][-LAST_N:]
            if len(before) < GOALS_ONLY_MIN_MATCHES:
                problems.append(f"too little history: {name} has {len(before)} completed match(es) before this game "
                                f"on file, at least {GOALS_ONLY_MIN_MATCHES} are needed (goals only, no xG)")
            else:
                raw.extend(team_inputs(side, before, source))
        if problems:
            return insufficient(fixture.match_id, *problems)
        return assess_goals_only(fixture.match_id, raw)
