"""The owner's own word about a booking code (AD-25). Standard library plus the store only; the bot never places anything here.

A BOOKED row is a booking code the bot generated (no bet was placed). The owner can say:
  'I placed it'        -> the bot asks for the stake (a reply with the amount, or Skip); the row becomes PLACED_BY_OWNER
                          (source 'owner-confirmed', his stake or NULL) and its legs go into the placed-leg registry, so they
                          block future lists exactly like a PLACED bet. It is his word: Results grades it only when the bet
                          history shows a matching bet, and never invents an outcome.
  'I did not place it' -> the row stays BOOKED (it never blocks anything); he may answer 'I placed it' later (Mybets).

Every press is a compare-and-swap on the row's state (the UPDATE itself only moves a BOOKED row), so a repeated tap changes
nothing and a stale tap is answered as stale. The pending 'how much did you stake?' question lives in memory only: a restart
forgets it, the row is simply still BOOKED, and its buttons can be pressed again.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from predictions.core.staking import StakeError, format_money, parse_stake
from predictions.flow.store import P_BOOKED, P_PLACED_BY_OWNER, EvaluationStore

ACT_PLACED = "p"
ACT_NOT_PLACED = "n"
ACT_SKIP = "k"

ASK_TEXT = "How much did you stake? Reply with the amount, or press Skip."
RECORDED_TEXT = "Recorded as placed (confirmed by you)."
NOT_PLACED_TEXT = "Noted: not placed."
ALREADY_TEXT = "Already recorded as placed (confirmed by you)."
UNKNOWN_TEXT = "That booking is not known any more."

ST_ASK, ST_RECORDED, ST_NOTED, ST_ALREADY, ST_UNKNOWN, ST_STALE, ST_INVALID, ST_ERROR = (
    "ask", "recorded", "noted", "already", "unknown", "stale", "invalid", "error")


@dataclass(frozen=True)
class BookingAnswer:
    status: str
    text: str
    key: str = ""      # the booking (12 characters of its token hash) the answer is about


class PendingStake:
    """The one 'how much did you stake?' question that may be open (in memory only)."""

    def __init__(self) -> None:
        self.key: str | None = None
        self.origin: tuple | None = None   # (chat id, message id) of the message whose button opened the question

    def clear(self, key: str | None = None) -> None:
        if key is None or self.key == key:
            self.key = None
            self.origin = None


def _stale_text(state: str) -> str:
    return f"This record is {state} now, so that button no longer applies."


def press(store: EvaluationStore | None, desk: PendingStake, key: str, action: str,
          now: datetime | None = None) -> BookingAnswer:
    if store is None:
        return BookingAnswer(ST_ERROR, "The database is not open, so nothing was changed.", key)
    try:
        row = store.find_placement_by_prefix(key)
    except Exception:
        row = None
    if row is None or row["dry_run"]:
        desk.clear(key)
        return BookingAnswer(ST_UNKNOWN, UNKNOWN_TEXT, key)
    state = row["state"]
    if state == P_PLACED_BY_OWNER:
        desk.clear(key)
        return BookingAnswer(ST_ALREADY, ALREADY_TEXT if action != ACT_NOT_PLACED
                             else "You already recorded this as placed, so it cannot be marked not placed here.", key)
    if state != P_BOOKED:
        desk.clear(key)
        return BookingAnswer(ST_STALE, _stale_text(state), key)
    if action == ACT_NOT_PLACED:
        desk.clear(key)
        return BookingAnswer(ST_NOTED, NOT_PLACED_TEXT, key)
    if action == ACT_PLACED:
        desk.key = key
        return BookingAnswer(ST_ASK, ASK_TEXT, key)
    if action == ACT_SKIP:
        return _record(store, desk, row, None, now)
    return BookingAnswer(ST_STALE, "That button is not valid.", key)


def stake_reply(store: EvaluationStore | None, desk: PendingStake, text,
                now: datetime | None = None) -> BookingAnswer | None:
    """A typed message while the question is open. None when no question is open (the text is not for this)."""
    key = desk.key
    if key is None:
        return None
    try:
        stake = parse_stake(text, None)   # positive, at most two decimals; no MAX_STAKE: this is a record, not a bet
    except StakeError as error:
        return BookingAnswer(ST_INVALID, str(error) + " Or press Skip.", key)
    if store is None:
        return BookingAnswer(ST_ERROR, "The database is not open, so nothing was changed.", key)
    try:
        row = store.find_placement_by_prefix(key)
    except Exception:
        row = None
    if row is None:
        desk.clear(key)
        return BookingAnswer(ST_UNKNOWN, UNKNOWN_TEXT, key)
    return _record(store, desk, row, stake, now)


def _record(store: EvaluationStore, desk: PendingStake, row: dict, stake, now: datetime | None) -> BookingAnswer:
    key = row["token_hash"][:12]
    try:
        moved = store.resolve_owner_placed(row["token_hash"], stake, now)
    except Exception:
        return BookingAnswer(ST_ERROR, "I could not save that, so nothing was changed. Press the button again.", key)
    desk.clear(key)
    if not moved:
        return BookingAnswer(ST_ALREADY, ALREADY_TEXT, key)
    text = RECORDED_TEXT + (f" Stake NGN {format_money(stake)}." if stake is not None else " No stake was recorded.")
    return BookingAnswer(ST_RECORDED, text, key)
