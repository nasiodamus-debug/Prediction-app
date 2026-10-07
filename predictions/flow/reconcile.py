"""Reconciling a placement whose result is not proven against SportyBet's bet history (Story 5.5, AD-6).

This is READ-ONLY toward the bookmaker: it calls `Bookmaker.bet_history()` and nothing else, so it can never place,
click or change a betslip. It is the only thing that resolves an unclear result (PLACEMENT_UNKNOWN, or a restart with a
bet in PLACING): never a second placement.

  found       exactly one entry in the history (first page: the 5 newest bets) matches the placement's games, stake and
              total number of legs, is not older than the placement and is not already known as another placement: the
              bet EXISTS. The list shows no odds, picks or bet id (the id is a synthetic key from the visible fields),
              so only those were verified, and the message says so. The placement becomes PLACED and its legs go into the placed-leg registry.
  not found   the history was read and holds no such bet. That does not prove the bet does not exist (history can lag),
              so the placement stays PLACEMENT_UNKNOWN and its legs stay blocked by the registry until the owner
              explicitly overrides.
  unreadable  the history could not be read: the same, with the reason.
  multiple    more than one entry looks like it (or two entries with an identical key, one of which would match): a possible double placement, reported loudly, never resolved silently.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Callable, Sequence

from predictions.core.betting import (SETTLED_OR_RUNNING, HistoryBet, LoadedLeg, history_bet_matches,
                                      history_limits_note)
from predictions.core.ports import Bookmaker
from predictions.flow.store import P_PLACED, P_PLACING, P_UNKNOWN, EvaluationStore

KIND_FOUND = "found"
KIND_NOT_FOUND = "not_found"
KIND_UNREADABLE = "unreadable"
KIND_MULTIPLE = "multiple"

WONT_PLACE_AGAIN = "I will not place it again on my own."


@dataclass(frozen=True)
class ReconcileOutcome:
    kind: str
    text: str
    bet: HistoryBet | None = None
    warning: str = ""

    @property
    def found(self) -> bool:
        return self.kind == KIND_FOUND


def legs_from_row(row: dict) -> tuple[LoadedLeg, ...]:
    """The legs of a saved placement, exactly as they were confirmed (raises ValueError when they cannot be read)."""
    legs = []
    for item in json.loads(row["legs"]):
        key = tuple(item["leg_key"])
        if len(key) != 4:
            raise ValueError("a saved leg key is malformed")
        evaluation_odds = Decimal(item["evaluation_odds"]) if item.get("evaluation_odds") else None
        legs.append(LoadedLeg(key, item["match"], item["market"], item["outcome"], Decimal(item["odds"]),
                              item.get("selection", ""), evaluation_odds))
    return tuple(legs)


def parse_created(value) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


async def read_matches(bookmaker: Bookmaker, legs: Sequence[LoadedLeg], stake: Decimal, exclude_ids: set[str],
                       sanitize: Callable[[object], str], total: Decimal | None = None,
                       created_at: datetime | None = None) -> tuple[str, list[HistoryBet], str]:
    """(KIND_FOUND|KIND_NOT_FOUND|KIND_MULTIPLE|KIND_UNREADABLE, the matching bets, the reason when unreadable).

    The history LIST shows no odds and no picks (VERIFIED-LIVE 2026-10-01), so a match is: the same stake, the same TOTAL
    number of games, every shown game one of the legs, a recognised status (Running, or Won/Lost when the bet has finished
    since) and, when `created_at` is given, an entry time not before it (2 minutes of slack). `total` is accepted for the
    callers' sake and ignored: odds cannot be compared there. Only the first page (5 newest bets) is read: an older bet is
    simply not found. An entry whose synthetic key is ambiguous (an identical key twice) never matches, but when one WOULD
    match it is reported as KIND_MULTIPLE: a possible double placement, never resolved silently."""
    try:
        history = await bookmaker.bet_history()
    except Exception as error:  # any failure to read is "unreadable", never "no bet"
        return KIND_UNREADABLE, [], sanitize(error)
    candidates = [b for b in history if b.bet_id and b.bet_id not in exclude_ids]
    matches = [b for b in candidates if history_bet_matches(b, legs, stake, created_at, SETTLED_OR_RUNNING)]
    if not matches:
        shadowed = [b for b in candidates if b.ambiguous
                    and history_bet_matches(b, legs, stake, created_at, SETTLED_OR_RUNNING, ignore_ambiguity=True)]
        if shadowed:
            return KIND_MULTIPLE, shadowed, ""
        return KIND_NOT_FOUND, [], ""
    return (KIND_FOUND if len(matches) == 1 else KIND_MULTIPLE), matches, ""


def text_found(bet: HistoryBet) -> str:
    code = f" Booking code: {bet.booking_code}." if bet.booking_code else ""
    return (f"I checked your SportyBet bet history: the bet EXISTS (history entry {bet.bet_id}).{code} "
            f"{history_limits_note(bet)} {WONT_PLACE_AGAIN}")


def text_not_found() -> str:
    return ("I checked your SportyBet bet history and did not find this bet. That does not prove it was not placed "
            "(the history can lag), so its games stay blocked. " + WONT_PLACE_AGAIN + " Please look in the SportyBet app.")


def text_unreadable(reason: str) -> str:
    return (f"I could not read your SportyBet bet history ({reason}), so I cannot tell whether the bet exists. Its games "
            f"stay blocked. {WONT_PLACE_AGAIN} Please look in the SportyBet app.")


def text_multiple(matches: Sequence[HistoryBet]) -> str:
    ids = ", ".join(b.bet_id for b in matches[:5])
    return (f"WARNING: {len(matches)} entries in your SportyBet bet history look like this one (history entries {ids}; "
            f"two with the very same time, stake and games cannot be told apart). It may have "
            f"been placed more than once. Check the SportyBet app right away. {WONT_PLACE_AGAIN}")


async def reconcile_placement(store: EvaluationStore, bookmaker: Bookmaker, token_hash: str,
                              sanitize: Callable[[object], str], now: datetime | None = None) -> ReconcileOutcome:
    """Check the bet history for one recorded placement (state PLACING or PLACEMENT_UNKNOWN) and record what it shows."""
    row = store.get_placement(token_hash)
    if row is None:
        return ReconcileOutcome(KIND_UNREADABLE, text_unreadable("the placement was never recorded"))
    if row["state"] == P_PLACED:
        bet = HistoryBet(row["bet_id"], row["booking_code"])
        return ReconcileOutcome(KIND_FOUND, text_found(bet), bet)
    if row["state"] not in (P_PLACING, P_UNKNOWN):
        return ReconcileOutcome(KIND_UNREADABLE, text_unreadable(f"the placement is already {row['state']}"))
    try:
        legs, stake = legs_from_row(row), Decimal(row["stake"])
    except Exception:
        return ReconcileOutcome(KIND_UNREADABLE, text_unreadable("the saved placement could not be read"))
    try:
        known = {r["bet_id"] for r in store.list_placements() if r["bet_id"] and r["token_hash"] != token_hash}
    except Exception:
        known = set()
    try:
        total = Decimal(row["total_odds"])
    except Exception:
        return ReconcileOutcome(KIND_UNREADABLE, text_unreadable("the saved placement could not be read"))
    kind, matches, reason = await read_matches(bookmaker, legs, stake, known, sanitize, total,
                                               parse_created(row["created_at_utc"]))

    def note(state: str, detail: str, **fields) -> str:
        try:
            store.update_placement(token_hash, state, detail=detail, now=now, **fields)
            return ""
        except Exception as error:
            return f"The local record could not be updated ({sanitize(error)})."

    if kind == KIND_FOUND:
        bet = matches[0]
        try:
            store.resolve_placement_found(token_hash, bet.bet_id, bet.booking_code,
                                          "found in bet history by reconciliation", legs, now)
            warning = ""
        except Exception as error:
            warning = f"The placed legs could not be registered ({sanitize(error)})."
        return ReconcileOutcome(KIND_FOUND, text_found(bet), bet, warning)
    if kind == KIND_MULTIPLE:
        return ReconcileOutcome(KIND_MULTIPLE, text_multiple(matches), matches[0],
                                note(P_UNKNOWN, f"{len(matches)} matching bets found in bet history"))
    if kind == KIND_NOT_FOUND:
        return ReconcileOutcome(KIND_NOT_FOUND, text_not_found(),
                                warning=note(P_UNKNOWN, "checked bet history: not found"))
    return ReconcileOutcome(KIND_UNREADABLE, text_unreadable(reason),
                            warning=note(P_UNKNOWN, "bet history could not be read"))
