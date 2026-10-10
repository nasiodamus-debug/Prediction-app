"""Free context flags and the model check, from API-Football (soccer, games of TODAY only).

Flags (cost nothing, no AI call): a team with KEY_PLAYER_OUT_MIN or more DISTINCT players ruled out (doubts do not count, and
players under YOUTH_AGE by the squad list do not either) for the fixture raises
`key_player_out`; a game that kicks off within LINEUP_DUE_MINUTES with no published line-ups raises `lineup_uncertain`.
They are the same AnalystFlag values the AI analyst produces, so the gate treats them the same way: a flagged game can only
become NO VALID PLAY - PASS, an estimate is never touched. The service lists every absent player, not only key ones, so
the minimum count is a deliberate, adjustable rule.

Model check: for a PASSING soccer pick on the 1X2 or double-chance market, the vendor's own win percentages are read and
a one-line warning is added to the list footer when they put the pick's chance PREDICTION_GAP or more BELOW ours. It
never changes a pick, a probability or an order: it is a prompt for the owner to look again.

`SOURCE` is set once at startup (app.py) and is None when API_FOOTBALL_KEY is not set: then none of this runs.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher

from predictions.adapters.apifootball.client import ApiFixture, ApiFootball, Prediction
from predictions.core.analyst import AnalystFlag, ContextFlag
from predictions.core.identity import _NOT_SAME_TEAM, _YOUTH
from predictions.core.model import Evaluation, percent
from predictions.core.types import Fixture, MatchId, normalize_team

log = logging.getLogger("freeflags")

SOURCE: ApiFootball | None = None
DEFAULT_KEY_PLAYER_OUT_MIN = 5
KEY_PLAYER_OUT_MIN = DEFAULT_KEY_PLAYER_OUT_MIN   # DISTINCT players listed out for one team before `key_player_out` is raised (KEY_PLAYERS_OUT_MIN)
MAX_KEY_PLAYER_OUT_MIN = 30
YOUTH_AGE = 21                # a listed player younger than this (by the team's squad list) is not counted as a key player out
DOUBT_KINDS = frozenset({"questionable", "doubtful"})   # listed as a doubt, not ruled out: not counted
LINEUP_DUE_MINUTES = 40       # a game this close to kickoff with no line-ups raises `lineup_uncertain`
PREDICTION_GAP = 0.25         # the vendor puts the pick this far (0 to 1) below our estimate: warn
MATCH_WINDOW = timedelta(minutes=45)
SOURCE_NAME = "API-Football"


async def _without_youth(source, players: list) -> list:
    """Drop players the team's squad list gives as younger than YOUTH_AGE. A squad that cannot be read (or a source with none), or
    a player who is not in it, filters nothing: the count is only ever lowered by a known age."""
    squad = getattr(source, "squad", None)
    team_id = next((p.team_id for p in players if p.team_id is not None), None)
    if squad is None or team_id is None:
        return players
    try:
        ages = await squad(team_id)
    except Exception:
        return players
    return [p for p in players if p.player_id is None or ages.get(p.player_id, YOUTH_AGE) >= YOUTH_AGE]


def parse_key_player_min(raw) -> int:
    """KEY_PLAYERS_OUT_MIN: distinct players out before a team's game is flagged. Unset or empty keeps the default; a value that is
    not a whole number from 1 to 30 raises ValueError (the message never echoes the value)."""
    if raw is None or not str(raw).strip():
        return DEFAULT_KEY_PLAYER_OUT_MIN
    text = str(raw).strip()
    if not text.isascii() or not text.isdigit() or not 1 <= int(text) <= MAX_KEY_PLAYER_OUT_MIN:
        raise ValueError("KEY_PLAYERS_OUT_MIN in config.env is not a whole number from 1 to 30. Fix it or remove the line to use "
                         f"the default of {DEFAULT_KEY_PLAYER_OUT_MIN}, then start the bot again.")
    return int(text)


def _now() -> datetime:
    return datetime.now(timezone.utc)


_FILLER = frozenset({"fc", "cf", "ca", "cd", "csd", "sc", "ac", "as", "club", "de", "da", "do", "del", "la", "el", "fk", "sv", "afc"})


def _words(name: str) -> list[str]:
    return [w for w in normalize_team(name).split() if w not in _FILLER]


def _prefix_cover(small: list[str], large: list[str]) -> bool:
    """Every word of `small` is a word of `large` or an abbreviation of one ("independ" for "independiente")."""
    return all(any(w == v or (min(len(w), len(v)) >= 4 and (v.startswith(w) or w.startswith(v))) for v in large) for w in small)


def closeness(a: str, b: str) -> int:
    """How alike two spellings of a team are: 2 = the same club (one name contains the other, abbreviations, or very similar),
    1 = partly alike (a shared name word or a similar spelling), 0 = different or a different side (reserves, women, youth)."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0
    sa, sb = set(wa), set(wb)
    if any(w in _NOT_SAME_TEAM or _YOUTH.match(w) for w in sa ^ sb):
        return 0
    if sa <= sb or sb <= sa:   # one name inside the other: the same club only when the extra words are tags ("MG", "SP")
        return 2 if all(len(w) <= 2 for w in sa ^ sb) else 1   # "Real Madrid Castilla" is only partly "Real Madrid"
    if _prefix_cover(wa, wb) or _prefix_cover(wb, wa):
        return 2
    ratio = SequenceMatcher(None, " ".join(wa), " ".join(wb)).ratio()
    if ratio >= 0.9:
        return 2
    if ratio >= 0.6 or any(len(w) >= 4 for w in sa & sb):
        return 1
    return 0


def match_fixture(fixture: Fixture, candidates: list[ApiFixture]) -> ApiFixture | None:
    """The API fixture for a SportyBet game. The kickoffs must be within MATCH_WINDOW, both teams must be at least partly alike
    and one of them clearly the same club, and exactly ONE candidate may fit best; with two equally good ones nothing is
    matched (a wrong match could flag or warn about the wrong game)."""
    scored = []
    for c in candidates:
        if abs(c.kickoff_utc - fixture.kickoff_utc) > MATCH_WINDOW:
            continue
        home, away = closeness(fixture.home, c.home), closeness(fixture.away, c.away)
        if home >= 1 and away >= 1 and home + away >= 3:
            scored.append((home + away, c))
    if not scored:
        return None
    best = max(score for score, _ in scored)
    top = [c for score, c in scored if score == best]
    return top[0] if len(top) == 1 else None


def _today_soccer(fixtures: list[Fixture], now: datetime) -> list[Fixture]:
    return [f for f in fixtures if f.match_id.sport == "soccer" and not f.started and f.kickoff_utc > now
            and f.kickoff_utc.date() == now.date()]


def merge_flags(*maps: dict) -> dict[MatchId, tuple[AnalystFlag, ...]]:
    merged: dict[MatchId, tuple[AnalystFlag, ...]] = {}
    for flags in maps:
        for match_id, items in (flags or {}).items():
            merged[match_id] = merged.get(match_id, ()) + tuple(items)
    return merged


async def free_flags(fixtures: list[Fixture], now: datetime | None = None,
                     source: ApiFootball | None = None) -> dict[MatchId, tuple[AnalystFlag, ...]]:
    source = source or SOURCE
    now = now or _now()
    games = _today_soccer(fixtures, now)
    if source is None or not games:
        return {}
    candidates = await source.fixtures(now.date())
    if not candidates:
        return {}
    injuries = await source.injuries(now.date())
    found: dict[MatchId, tuple[AnalystFlag, ...]] = {}
    for game in games:
        api = match_fixture(game, candidates)
        if api is None:
            log.info("no API-Football fixture matched %s v %s (names or kickoff differ): no free flags for it", game.home, game.away)
            continue
        flags: list[AnalystFlag] = []
        by_team: dict[str, list] = {}
        for missing in injuries.get(api.id, ()):
            by_team.setdefault(missing.team, []).append(missing)
        for team, listed in by_team.items():
            # only players ruled out (not doubts), each counted once (the service repeats a player once per listed issue)
            out = [p for p in listed if p.kind.strip().lower() not in DOUBT_KINDS]
            players = list({normalize_team(p.player): p for p in out}.values())
            if len(players) >= KEY_PLAYER_OUT_MIN:
                players = await _without_youth(source, players)   # one squad request, only for a team already at the threshold
            if len(players) >= KEY_PLAYER_OUT_MIN:
                names = ", ".join(f"{p.player} ({p.reason or p.kind})" for p in players[:4])
                more = f" and {len(players) - 4} more" if len(players) > 4 else ""
                flags.append(AnalystFlag(ContextFlag.KEY_PLAYER_OUT,
                                         f"{SOURCE_NAME} injuries: {team} without {len(players)}: {names}{more}"))
        if game.kickoff_utc - now <= timedelta(minutes=LINEUP_DUE_MINUTES):
            if await source.lineups_published(api.id) is False:
                flags.append(AnalystFlag(ContextFlag.LINEUP_UNCERTAIN,
                                         f"{SOURCE_NAME} lineups: none published {LINEUP_DUE_MINUTES} minutes before kickoff"))
        if flags:
            found[game.match_id] = tuple(flags)
    if found:
        log.info("free flags raised for %d game(s)", len(found))
    return found


def vendor_chance(selection: str, market: str, prediction: Prediction) -> float | None:
    """The vendor's probability for a 1X2 or double-chance pick, or None for any other market."""
    if market == "1X2":
        return {"Home": prediction.home, "Away": prediction.away}.get(selection)
    if market == "Double Chance":
        return {"Home or Draw": prediction.home + prediction.draw, "Away or Draw": prediction.away + prediction.draw,
                "Home or Away": prediction.home + prediction.away}.get(selection)
    return None


async def disagreement_lines(passed: tuple[Evaluation, ...], fixtures: dict[MatchId, Fixture],
                             now: datetime | None = None, source: ApiFootball | None = None) -> tuple[str, ...]:
    """One warning line per passing pick the vendor rates far below our estimate (see the module note)."""
    source = source or SOURCE
    now = now or _now()
    plays = [ev for ev in passed if ev.market in ("1X2", "Double Chance") and ev.match_id in fixtures]
    games = _today_soccer([fixtures[ev.match_id] for ev in plays], now)
    if source is None or not games:
        return ()
    candidates = await source.fixtures(now.date())
    lines = []
    for ev in plays:
        game = fixtures[ev.match_id]
        if game not in games:
            continue
        api = match_fixture(game, candidates)
        prediction = await source.prediction(api.id) if api is not None else None
        theirs = vendor_chance(ev.selection, ev.market, prediction) if prediction is not None else None
        if theirs is not None and ev.estimate.probability - theirs >= PREDICTION_GAP:
            lines.append(f"Warning: {game.home} v {game.away}, {ev.selection}: we estimate "
                         f"{percent(ev.estimate.probability)}, {SOURCE_NAME}'s own model gives {percent(theirs)}. "
                         "Look again before betting.")
    return tuple(lines)
