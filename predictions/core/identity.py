"""The one team-identity resolver (AD-5). Standard library only.

A source's team name is matched to one of the names a league lists. Exact match after normalization comes
first, then a short alias table, then a conservative partial match. If more than one team fits, the result
is "ambiguous" and the caller must ask the owner; nothing here ever picks one of several.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Mapping

from predictions.core.types import normalize_team

# Normalized bookmaker/other spellings -> normalized name used by the stats source.
ALIASES: Mapping[str, str] = {
    "bayern munchen": "bayern munich",
    "fc bayern munchen": "bayern munich",
    "man utd": "manchester united",
    "man united": "manchester united",
    "manchester utd": "manchester united",
    "man city": "manchester city",
    "spurs": "tottenham",
    "tottenham hotspur": "tottenham",
    "newcastle": "newcastle united",
    "inter milan": "inter",
    "internazionale": "inter",
    "internazionale milano": "inter",
    "milan": "ac milan",
    "psg": "paris saint germain",
    "paris sg": "paris saint germain",
    "paris saint germain fc": "paris saint germain",
    "borussia monchengladbach": "borussia m gladbach",
    "monchengladbach": "borussia m gladbach",
    "gladbach": "borussia m gladbach",
    "rb leipzig": "rasenballsport leipzig",
    "leipzig": "rasenballsport leipzig",
    "1 fc koln": "fc cologne",
    "fc koln": "fc cologne",
    "koln": "fc cologne",
    "cologne": "fc cologne",
    "hamburg": "hamburger sv",
    "olympique marseille": "marseille",
    "olympique lyonnais": "lyon",
    "olympique lyon": "lyon",
    "atletico de madrid": "atletico madrid",
    "atl madrid": "atletico madrid",
    "athletic bilbao": "athletic club",
    "deportivo alaves": "alaves",
    "real betis balompie": "real betis",
    "wolverhampton wanderers": "wolverhampton",
    "wolves": "wolverhampton",
    "nottm forest": "nottingham forest",
    "leeds united": "leeds",
    "west ham united": "west ham",
    "los chankas cyc": "los chankas",
}

# Words that carry no identity ("Arsenal FC" is "Arsenal"). Used only in the partial match.
_GENERIC = frozenset({"fc", "cf", "afc", "sc", "ssc", "ac", "as", "us", "ss", "fk", "sv", "cd", "ud", "rc",
                      "calcio", "1913"})
# An extra word like these means a different team (reserves, youth, women), never the same club.
_NOT_SAME_TEAM = frozenset({"ii", "iii", "b", "w", "women", "womens", "ladies", "reserves", "res", "youth",
                            "academy"})
_YOUTH = re.compile(r"^u\d{2}$")


@dataclass(frozen=True)
class TeamMatch:
    status: str  # "matched", "ambiguous" or "none"
    matches: tuple[str, ...] = ()  # candidate names as the source spells them
    method: str = ""  # "exact", "alias" or "partial" when matched

    @property
    def team(self) -> str | None:
        return self.matches[0] if self.status == "matched" else None


def _tokens(normalized: str) -> frozenset[str]:
    return frozenset(normalized.split()) - _GENERIC


def _same_club(a: frozenset[str], b: frozenset[str]) -> bool:
    if not a or not b:
        return False
    small, large = (a, b) if len(a) <= len(b) else (b, a)
    if not small <= large:
        return False
    extra = large - small
    if any(t in _NOT_SAME_TEAM or _YOUTH.match(t) for t in extra):
        return False  # reserves, youth, women: a different team
    return True


def resolve_team(name: str, candidates: Iterable[str], aliases: Mapping[str, str] | None = None) -> TeamMatch:
    """Match `name` to exactly one candidate, or report that it is ambiguous or unknown."""
    aliases = ALIASES if aliases is None else aliases
    pool: dict[str, str] = {}
    for candidate in candidates:
        pool.setdefault(normalize_team(candidate), candidate)
    wanted = normalize_team(name)
    if not wanted:
        return TeamMatch("none")
    if wanted in pool:
        return TeamMatch("matched", (pool[wanted],), "exact")
    target = aliases.get(wanted)
    if target is not None and normalize_team(target) in pool:
        return TeamMatch("matched", (pool[normalize_team(target)],), "alias")
    wanted_tokens = _tokens(wanted)
    found = [original for norm, original in pool.items() if _same_club(wanted_tokens, _tokens(norm))]
    # A partial match is only safe when the two names differ by generic words alone (those are already stripped,
    # so the token sets are equal). An extra real name part on either side ("Paris FC" vs "Paris Saint
    # Germain", "Real Madrid" vs "Real Madrid Castilla") never counts as the same club: the owner is asked.
    if len(found) == 1 and _tokens(normalize_team(found[0])) == wanted_tokens:
        return TeamMatch("matched", (found[0],), "partial")
    if found:
        return TeamMatch("ambiguous", tuple(sorted(found)))
    return TeamMatch("none")
