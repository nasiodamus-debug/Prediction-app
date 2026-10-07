"""The final summary and the single-use PlacementToken (AD-6). Standard library only.

A `Summary` is exactly what the owner is shown before the one final confirmation: the legs as the betslip holds them,
the odds per leg, the total odds, the stake and an expiry. Its digest identifies that summary. A `PlacementToken` is
minted from a summary only by the session state machine (flow/session.py) at the moment the owner confirms; it carries
a hash of everything the owner saw, and `Bookmaker.place(token)` refuses to click unless the real betslip matches it.

Only the state machine may mint. `PlacementToken` refuses to be built without the mint authority, which only
flow/session.py imports (a test scans the source tree for any other user). Single use is enforced by the state
machine (PLACING is reached once), by the database (the token hash is unique) and by the adapter (it remembers the
hashes it has acted on).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum

from predictions.core.betting import LoadedLeg, quantize_odds

SUMMARY_TTL = timedelta(minutes=5)   # how long one final confirmation stays valid
SUMMARY_ID_LENGTH = 12               # hex characters carried in the confirm button

# The mint authority. Nothing but flow/session.py may import this name.
MINT_AUTHORITY = object()

# AD-25 (PLACE_MODE=book): a booking summary carries NO stake (the owner sets it in SportyBet). It is the one named value
# a booking token's stake can have; a stake of zero is refused everywhere else.
BOOKING_STAKE = Decimal(0)


def _stake_text(stake: Decimal) -> str:
    return format(Decimal(stake).normalize(), "f")


def summary_digest(session_id: str, legs: tuple[LoadedLeg, ...], total: Decimal, stake: Decimal,
                   expires_at: datetime, booking: bool = False) -> str:
    """SHA-256 over the legs (match, market, line, selection), the odds per leg, the total odds, the stake and the
    expiry. Legs are sorted, so the order the betslip lists them in does not matter. A booking summary (AD-25) also
    hashes its mode, so it can never be mistaken for a placing one."""
    rows = sorted([*leg.leg_key, leg.match_text.strip().lower(), leg.market_name.strip().lower(),
                   leg.outcome_name.strip().lower(), str(quantize_odds(leg.odds))] for leg in legs)
    payload = {"session": session_id, "legs": rows, "total": str(quantize_odds(total)), "stake": _stake_text(stake),
               "expires": expires_at.astimezone(timezone.utc).isoformat()}
    if booking:
        payload["mode"] = "book"
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Summary:
    session_id: str
    legs: tuple[LoadedLeg, ...]
    total_odds: Decimal
    stake: Decimal
    expires_at: datetime
    booking: bool = False   # AD-25: PLACE_MODE=book. No stake (it is BOOKING_STAKE); the final action is a booking code

    def __post_init__(self):
        if self.expires_at.tzinfo is None:
            raise ValueError("a summary expiry must be UTC-aware")
        if not self.legs:
            raise ValueError("a summary needs at least one leg")
        if self.booking:
            if Decimal(self.stake) != BOOKING_STAKE:
                raise ValueError("a booking summary carries no stake")
        elif Decimal(self.stake) <= 0:
            raise ValueError("a summary needs a positive stake")

    @property
    def digest(self) -> str:
        return summary_digest(self.session_id, self.legs, self.total_odds, self.stake, self.expires_at, self.booking)

    @property
    def summary_id(self) -> str:
        return self.digest[:SUMMARY_ID_LENGTH]

    def expired(self, now: datetime) -> bool:
        return now >= self.expires_at


class TokenNotMinted(Exception):
    """A PlacementToken was built without the mint authority, or its hash does not match its contents."""


@dataclass(frozen=True)
class PlacementToken:
    session_id: str
    legs: tuple[LoadedLeg, ...]
    total_odds: Decimal
    stake: Decimal
    expires_at: datetime
    token_hash: str
    authority: object = field(default=None, repr=False, compare=False)
    booking: bool = False   # AD-25: this token may only ever be used to read a booking code, never to place a bet

    def __post_init__(self):
        if self.authority is not MINT_AUTHORITY:
            raise TokenNotMinted("a PlacementToken can only be minted by the session state machine")
        if not self.matches_contents():
            raise TokenNotMinted("the token hash does not match its contents")

    def matches_contents(self) -> bool:
        if self.booking != (Decimal(self.stake) == BOOKING_STAKE):
            return False   # a booking token has exactly no stake; a placing token never has none
        return self.token_hash == summary_digest(self.session_id, self.legs, self.total_odds, self.stake,
                                                 self.expires_at, self.booking)

    def expired(self, now: datetime) -> bool:
        return now >= self.expires_at


def mint_token(authority: object, summary: Summary) -> PlacementToken:
    """Only the state machine calls this (with MINT_AUTHORITY), at the moment of the final confirmation."""
    if authority is not MINT_AUTHORITY:
        raise TokenNotMinted("a PlacementToken can only be minted by the session state machine")
    return PlacementToken(summary.session_id, summary.legs, summary.total_odds, summary.stake, summary.expires_at,
                          summary.digest, MINT_AUTHORITY, summary.booking)


class PlacementKind(Enum):
    DRY_RUN = "dry_run"   # dry run: everything was checked and NOTHING was clicked
    PLACED = "placed"     # a bet id was read back from the bookmaker's bet history
    BOOKED = "booked"     # AD-25 (PLACE_MODE=book): a NEW booking code was read; NO bet was placed, nothing was confirmed


@dataclass(frozen=True)
class PlacementResult:
    kind: PlacementKind
    report: tuple[str, ...] = ()    # dry run: what would have been placed, one line per fact
    bet_id: str = ""                # PLACED: the id read from the bet history (the proof)
    booking_code: str = ""          # BOOKED: the code (the whole result); PLACED: the code when it could be read (never proof)
    note: str = ""                  # PLACED: a loud warning, e.g. the bet exists but its odds differ from the confirmed ones


@dataclass(frozen=True)
class DuplicateLeg:
    """A leg (match, market, line) the placed-leg registry already knows (Story 5.5). `source` is "placed" (a bet id
    was read back) or "unknown" (a placement whose result was never proven: the bet may exist)."""
    identity: tuple[str, str, str]
    source: str
    bet_id: str = ""
    match_text: str = ""
    selection: str = ""

    @property
    def proven(self) -> bool:
        return self.source == "placed"


def booking_lines(code: str, total_odds: Decimal, stake: Decimal | None = None) -> list[str]:
    """The booking code, the total odds and (place mode only) the stake, each on its own line. One formatter for the
    BOOKED message and for the unknown-after-click message."""
    lines = [f"Booking code: {code}", f"Total odds: {Decimal(total_odds):.2f}"]
    if stake is not None and Decimal(stake) > 0:
        lines.append(f"Stake: NGN {format(Decimal(stake).normalize(), 'f')}")
    return lines
