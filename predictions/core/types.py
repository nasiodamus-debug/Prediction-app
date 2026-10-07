"""Shared vocabulary. Standard library only."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal


def normalize_team(name: str) -> str:
    """The one identity resolver for team names: case, accents, punctuation and spacing removed."""
    decomposed = unicodedata.normalize("NFKD", str(name))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    kept = "".join(c if c.isalnum() else " " for c in stripped.casefold())
    return " ".join(kept.split())


def _require_utc(value: datetime, what: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{what} must be a UTC-aware datetime")


@dataclass(frozen=True)
class MatchId:
    """sport:league:kickoff_utc:home:away, with normalized names."""
    sport: str
    league: str
    kickoff_utc: datetime
    home: str
    away: str

    def __post_init__(self):
        _require_utc(self.kickoff_utc, "MatchId.kickoff_utc")

    def __str__(self) -> str:
        stamp = self.kickoff_utc.strftime("%Y%m%dT%H%MZ")
        return ":".join((self.sport, normalize_team(self.league), stamp,
                         normalize_team(self.home), normalize_team(self.away)))


@dataclass(frozen=True)
class Outcome:
    name: str
    odds: Decimal
    outcome_id: str = ""  # the bookmaker's own id for this outcome, when it gave one (used to find it on the site)


@dataclass(frozen=True)
class MarketOffer:
    """One market a bookmaker offers on a fixture. `specifier` is e.g. "total=2.5" (empty if none)."""
    market_id: str
    name: str
    specifier: str
    outcomes: tuple[Outcome, ...]


@dataclass(frozen=True)
class Fixture:
    match_id: MatchId
    competition: str
    home: str
    away: str
    kickoff_utc: datetime  # UTC-aware
    started: bool  # the source says the game has started or finished
    markets: tuple[MarketOffer, ...] = ()
    bookmaker_ref: str = ""  # the bookmaker's own id for the game (SportyBet: the sr:match id), "" when unknown

    def __post_init__(self):
        _require_utc(self.kickoff_utc, "Fixture.kickoff_utc")


@dataclass(frozen=True)
class Owner:
    """The owner of the bot. Only the Telegram gate (or the scheduler acting as the owner) creates one, and
    session-flow entry points accept nothing else (AD-8)."""
    user_id: int
