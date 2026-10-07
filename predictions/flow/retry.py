"""'Check history again' (owner request, AD-6, AD-14, AD-24): a READ-ONLY look at SportyBet's bet history for a placement
whose result is unknown. It is the same machinery as startup recovery (`reconcile_placement`) and it can only ever read:
it never calls `Bookmaker.place`, never touches the betslip and never mints a token.

The press moves the session PLACEMENT_UNKNOWN -> RECONCILING (busy, so a second tap is a repeat and nothing else can
start); this job reads the history and moves it on:
  found        exactly one matching new bet: PLACED with its legs registered atomically (resolve_placement_found), DONE.
  not found    the history was READ and holds no such bet: back to PLACEMENT_UNKNOWN with the 'history checked' marker,
               which is what unlocks "Try again, I checked". History can lag by a minute, so it is said again.
  unreadable   the history could not be read: back to PLACEMENT_UNKNOWN WITHOUT the marker (nothing was learned).
  multiple     more than one entry looks like it: back to PLACEMENT_UNKNOWN WITHOUT the marker (a possible double bet).
"""
from __future__ import annotations

from datetime import datetime
from typing import Callable

from predictions.core.ports import Bookmaker
from predictions.flow.reconcile import KIND_NOT_FOUND, reconcile_placement
from predictions.flow.session import SessionManager, State
from predictions.flow.store import EvaluationStore

TEXT_NOT_FOUND = ("I checked your SportyBet bet history again and the bet was NOT found. Bet history can lag by a minute "
                  "or more, so it may still appear: check again in a minute, and look in the SportyBet app too.")


async def run_history_check(manager: SessionManager, store: EvaluationStore | None, bookmaker: Bookmaker,
                            session_id: str, sanitize: Callable[[object], str], now: datetime | None = None):
    """Run the check for the session that is RECONCILING after the owner pressed Check history again. Returns the
    session after it (None when the session is not the active one any more)."""
    active = manager.active
    if active is None or active.session_id != session_id or active.state is not State.RECONCILING or not active.token_hash:
        return None
    if store is None:
        return manager.resolve(session_id, State.PLACEMENT_UNKNOWN, active.message,
                               check_text="I could not check the bet history (the database is not open). Look in the "
                                          "SportyBet app.").session
    try:
        found = await reconcile_placement(store, bookmaker, active.token_hash, sanitize, now)
    except Exception as error:   # a failed check never turns an unknown into anything else
        text = (f"I could not read your SportyBet bet history ({type(error).__name__}), so I cannot tell whether the "
                "bet exists. Look in the SportyBet app.")
        return manager.resolve(session_id, State.PLACEMENT_UNKNOWN, active.message, check_text=text).session
    warning = f" ({found.warning})" if found.warning else ""
    if found.found:
        return manager.resolve(session_id, State.DONE, found.text + warning).session
    if found.kind == KIND_NOT_FOUND:
        return manager.resolve(session_id, State.PLACEMENT_UNKNOWN, active.message, checked=True,
                               check_text=TEXT_NOT_FOUND + warning).session
    return manager.resolve(session_id, State.PLACEMENT_UNKNOWN, active.message, check_text=found.text + warning).session
