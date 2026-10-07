"""Placing the bet (Story 5.4), with PLACING persisted before anything is clicked (AD-6, AD-14, AD-16).

`run_placement` receives a PlacementToken that only the session state machine could mint, at the owner's one final
confirmation. In order:

1. Write the PLACING row (token hash, legs, stake, total odds, expiry, timestamps) to SQLite. If that fails, NOTHING is
   clicked: the session fails and the owner is told no bet was placed.
2. Ask the bookmaker to place. It re-reads the real betslip and refuses unless it matches the token exactly. In dry run
   (the bookmaker's own `dry_run`, true by default) it never clicks and reports what it would have placed.
3. Record the result:
     dry run        -> DRY_RUN     (nothing was clicked, no legs are registered as placed)
     bet id proven  -> PLACED      (legs go into the placed-leg registry)
     odds moved     -> VOIDED_ODDS_CHANGED and a NEW summary is shown (this token is dead)
     error, nothing clicked -> FAILED (no bet; never retried automatically: the owner may press Try again)
     error after / unprovable -> PLACEMENT_UNKNOWN, then the bet history is read (read-only, flow/reconcile.py): a
                     matching bet found there means the bet EXISTS (PLACED, legs registered); otherwise the result stays
                     unknown and the legs stay blocked. Never a second placement (Story 5.5).

PLACE_MODE=book (AD-25): the token is a booking token. The bookmaker re-reads and verifies the betslip exactly the same way,
then presses ONLY Book Bet and returns BOOKED with a new booking code. That is a success of this mode (the session goes
to DONE, the row is BOOKED) and NOT a placed bet: nothing is registered as placed, nothing counts for Results, Calibration
or MAX_STAKE. If no code can be read, no bet exists: a plain retryable failure ('Try again', owner-pressed).

Anything unexpected is treated as PLACEMENT_UNKNOWN, never as "nothing happened": only a bookmaker that says it failed
BEFORE the click (BookmakerError) is trusted to mean no bet.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable

from predictions.core.betting import leg_identity
from predictions.core.placement import (SUMMARY_ID_LENGTH, PlacementKind, PlacementResult, PlacementToken,
                                         booking_lines)
from predictions.core.ports import Bookmaker, BookmakerError, OddsChanged, PlacementUnknown
from predictions.flow.reconcile import reconcile_placement
from predictions.flow.session import Session, SessionManager, State
from predictions.flow.store import (P_BOOKED, P_DRY_RUN, P_FAILED, P_PLACED, P_UNKNOWN, P_VOIDED, EvaluationStore)

OUTCOME_DRY_RUN = "dry_run"
OUTCOME_PLACED = "placed"
OUTCOME_BOOKED = "booked"
OUTCOME_ODDS_CHANGED = "odds_changed"
OUTCOME_FAILED = "failed"
OUTCOME_UNKNOWN = "unknown"


@dataclass(frozen=True)
class PlacementOutcome:
    kind: str
    session: Session | None = None
    result: PlacementResult | None = None
    message: str = ""
    warning: str = ""   # e.g. the bet is proven but the local record could not be updated (Story 5.5 reconciles it)
    token: PlacementToken | None = None   # the token this outcome belongs to (its legs and total odds feed the messages)


def booking_text(error: PlacementUnknown, token: PlacementToken) -> str:
    """What to tell the owner about the Book Bet fallback ('' when it was not tried). A booking code is NEVER proof of a bet."""
    code = str(getattr(error, "booking_code", "") or "")
    if code and re.fullmatch(r"[0-9A-Za-z]{4,10}", code):
        # the code and the odds each on their own line (the same formatter as the BOOKED message)
        return ("No success message was seen. The bet may or may not exist (check history). To place it yourself, open "
                "SportyBet and enter this booking code (loading it onto your slip is safe; a booking code is not proof of "
                "a bet):\n" + "\n".join(booking_lines(code, token.total_odds, token.stake)))
    if getattr(error, "booking_tried", False):
        return "I also tried to make a booking code from the slip but could not read one."
    return ""


async def run_placement(manager: SessionManager, store: EvaluationStore | None, bookmaker: Bookmaker,
                        token: PlacementToken, sanitize: Callable[[object], str],
                        now: datetime | None = None) -> PlacementOutcome:
    outcome = await _run_placement(manager, store, bookmaker, token, sanitize, now)
    return replace(outcome, token=token)


async def _run_placement(manager: SessionManager, store: EvaluationStore | None, bookmaker: Bookmaker,
                         token: PlacementToken, sanitize: Callable[[object], str],
                         now: datetime | None = None) -> PlacementOutcome:
    session_id, dry = token.session_id, bookmaker.dry_run
    book = token.booking
    active = manager.active
    if (active is None or active.session_id != session_id or active.state is not State.PLACING
            or active.token_hash != token.token_hash):
        # Only the token the state machine minted for the session's CURRENT PLACING state may place. An old, voided or
        # repeated token is refused here without touching the session or the bookmaker.
        return PlacementOutcome(OUTCOME_FAILED, active, None, "That confirmation is no longer valid. Nothing was placed.")

    def fail(text: str, state: State = State.FAILED, retryable: bool = False) -> PlacementOutcome:
        """`retryable` only for a failure that is CERTAINLY before the click: it lets the owner press Try again."""
        session = manager.finish(session_id, state, text, retryable=retryable).session
        return PlacementOutcome(OUTCOME_FAILED if state is State.FAILED else OUTCOME_UNKNOWN, session, None, text)

    # 1. PLACING is persisted BEFORE the bookmaker is asked to do anything.
    if store is None:
        return fail("The placement could not be recorded (the database is not open), so nothing was clicked. "
                    "No bet was placed. Nothing will be retried automatically.", retryable=True)
    # 0. The placed-leg registry (AD-6): a leg already placed (or possibly placed) is never placed again unless the owner
    #    explicitly overrode it for exactly this summary. The state machine checked this at the confirmation; this is the
    #    second, independent check, made before anything is recorded or clicked.
    try:
        known = store.find_blocking_legs([leg_identity(leg.leg_key) for leg in token.legs])
    except Exception as error:
        return fail(f"The list of legs already placed could not be read ({sanitize(error)}), so nothing was clicked. "
                    "No bet was placed. Nothing will be retried automatically.", retryable=True)
    allowed = set(active.overrides) if active.override_summary == token.token_hash[:SUMMARY_ID_LENGTH] else set()
    if [d for d in known if d.identity not in allowed]:
        return fail("A game in this bet was already placed (or may have been), so nothing was clicked. No bet was "
                    "placed. Nothing will be retried automatically. To place it anyway you must use the override button.")
    try:
        store.record_placing(token, dry, now)
    except Exception as error:
        return fail(f"The placement could not be recorded ({sanitize(error)}), so nothing was clicked. "
                    "No bet was placed. Nothing will be retried automatically.", retryable=True)
    left_out = ""
    if active.dropped:   # AD-26: the record says which picks the owner went ahead without, and why
        left_out = ("the owner went ahead without these picks, which could not be added: "
                    + "; ".join(f"{d.label} ({d.reason})" for d in active.dropped))
        try:
            store.annotate_placement(token.token_hash, left_out, now)
        except Exception:
            pass   # the note is a courtesy to the audit trail; it never changes what is placed

    def record(state: str, **fields) -> str:
        """Update the placement row; a failure here never changes what happened, it is only reported."""
        if left_out:   # every later update keeps the note (an update replaces the row's detail text)
            fields["detail"] = " | ".join(x for x in (left_out, fields.get("detail", "")) if x)
        try:
            store.update_placement(token.token_hash, state, now=now, **fields)
            return ""
        except Exception as error:
            return f"The local record could not be updated ({sanitize(error)})."

    async def unknown(base: str, detail: str, booking: str = "") -> PlacementOutcome:
        """The result after the click is unclear. Record it, then read the bet history (read-only) to see whether the
        bet exists. It is NEVER placed again: a found bet is PLACED, anything else stays unknown and its legs blocked."""
        warning = record(P_UNKNOWN, detail=(detail + " " + booking).strip())
        if dry:  # dry run never clicks (two guards), so there is nothing in the bet history to look for
            text = base + " Check your SportyBet bet history."
            outcome = fail(text, State.PLACEMENT_UNKNOWN)
            return PlacementOutcome(OUTCOME_UNKNOWN, outcome.session, None, text, warning)
        try:
            found = await reconcile_placement(store, bookmaker, token.token_hash, sanitize, now)
            checked, proven, extra = found.text, found.bet if found.found else None, found.warning
        except Exception as error:  # reconciliation must never turn an unknown into anything else
            checked, proven, extra = f"The bet history could not be checked ({type(error).__name__}).", None, ""
        warning = " ".join(w for w in (warning, extra) if w)
        text = f"{base} {checked}"
        if proven is not None and proven.bet_id:
            session = manager.finish(session_id, State.DONE, text).session
            result = PlacementResult(PlacementKind.PLACED, (), proven.bet_id, proven.booking_code)
            return PlacementOutcome(OUTCOME_PLACED, session, result, text, warning)
        if booking:   # the booking code goes into the record detail too (never into bet_id/booking_code: it proves nothing)
            try:
                store.annotate_placement(token.token_hash, booking, now)
            except Exception:
                pass
        text = text + " Check your SportyBet bet history." + ("\n" + booking if booking else "")
        outcome = fail(text, State.PLACEMENT_UNKNOWN)
        return PlacementOutcome(OUTCOME_UNKNOWN, outcome.session, None, text, warning)

    def booking_failed(reason: str) -> PlacementOutcome:
        """A booking that did not give a code. Booking places nothing, so there is no bet and nothing is unknown: a normal,
        retryable failure (the owner may press Try again; it counts toward the 3-retry cap)."""
        reason = reason.strip()
        if reason and reason[-1] not in ".!?":
            reason += "."
        text = f"No bet was placed. {reason} Nothing will be retried automatically."
        warning = record(P_FAILED, detail="booking failed: " + reason)
        outcome = fail(text, retryable=True)
        return PlacementOutcome(OUTCOME_FAILED, outcome.session, None, text, warning)

    # 2. The bookmaker re-reads the real betslip, refuses on any mismatch, and never clicks in dry run.
    try:
        result = await bookmaker.place(token)
    except OddsChanged as changed:
        warning = record(P_VOIDED, detail="the odds changed before the click; nothing was clicked")
        session = manager.odds_changed(session_id, changed.loaded).session
        return PlacementOutcome(OUTCOME_ODDS_CHANGED, session, None,
                                "SportyBet's odds changed before anything was placed. Nothing was placed. "
                                "Here is a new summary: check it and confirm again.", warning)
    except PlacementUnknown as error:
        if book:   # booking places nothing, so nothing can be unknown
            return booking_failed(sanitize(error))
        return await unknown(f"{sanitize(error)} The result is UNKNOWN. I will not place again on my own.", sanitize(error),
                             booking_text(error, token))
    except BookmakerError as error:
        if book:
            return booking_failed(sanitize(error))
        text = (f"Placing failed before anything was clicked: {sanitize(error)}. No bet was placed. Nothing will be "
                "retried automatically.")
        warning = record(P_FAILED, detail=sanitize(error))
        outcome = fail(text, retryable=True)
        return PlacementOutcome(OUTCOME_FAILED, outcome.session, None, text, warning)
    except Exception as error:  # cannot prove nothing was clicked: unknown, never "no bet"
        if book:
            return booking_failed(f"Something unexpected happened while getting the booking code ({type(error).__name__}).")
        return await unknown(f"Something unexpected happened while placing ({type(error).__name__}). The result is "
                             "UNKNOWN. I will not place again on my own.", type(error).__name__)

    # 3. Record what really happened.
    if result.kind is PlacementKind.BOOKED:
        code = str(result.booking_code or "")
        if not (book and re.fullmatch(r"[0-9A-Za-z]{4,10}", code)):
            if book:
                return booking_failed("The booking code could not be read.")
            text = ("An unexpected booking result came back for a placing confirmation. No bet was placed and nothing was "
                    "recorded as placed.")
            record(P_FAILED, detail="unexpected booking result")
            return PlacementOutcome(OUTCOME_FAILED, fail(text).session, None, text)
        warning = record(P_BOOKED, booking_code=code, detail="booking code only; no bet was placed")
        text = f"Booking code {code} generated. No bet was placed."
        session = manager.finish(session_id, State.DONE, text).session
        return PlacementOutcome(OUTCOME_BOOKED, session, result, text, warning)
    if result.kind is PlacementKind.DRY_RUN:
        text = ("DRY RUN: no booking code was made and nothing was placed. Book Bet, Place and Confirm were never pressed."
                if book else "DRY RUN: nothing was placed and the Place button was never clicked.")
        warning = record(P_DRY_RUN, detail="dry run; nothing clicked")
        session = manager.finish(session_id, State.DONE, text).session
        return PlacementOutcome(OUTCOME_DRY_RUN, session, result, text, warning)
    if book:   # a booking token can only end as BOOKED, DRY_RUN or a failure: anything else is not a bet
        return booking_failed("The booking code could not be read.")
    if result.kind is PlacementKind.PLACED and result.bet_id:
        try:  # the PLACED state and the legs registry are written in ONE transaction (all or nothing)
            store.resolve_placement_found(token.token_hash, result.bet_id, result.booking_code, (left_out + " | " if left_out else "") + "bet id read back",
                                          token.legs, now)
            warning = ""
        except Exception as error:
            # Neither write happened. Try the state alone: a PLACED row without legs still blocks (its saved legs
            # count); if even that fails the row stays PLACING, which blocks too, and reconciliation resolves it.
            warning = (f"The placed legs could not be registered ({sanitize(error)}). "
                       + (record(P_PLACED, bet_id=result.bet_id, booking_code=result.booking_code,
                                 detail="bet id read back; legs not registered")
                          or "The placement is recorded as placed; its games stay blocked."))
        text = ("Bet placed. SportyBet confirmed it with its Submission Successful message."
                if str(result.bet_id).startswith("booking:")
                else f"Bet placed. It was found in your SportyBet bet history (bet id {result.bet_id}).")
        if result.note:
            text += " " + result.note
        session = manager.finish(session_id, State.DONE, text).session
        return PlacementOutcome(OUTCOME_PLACED, session, result, text, warning)
    # A "success" without a bet id is not proof of anything.
    text = ("SportyBet did not return proof of a bet, so the result is UNKNOWN. I will not place again on my own. "
            "Check your SportyBet bet history.")
    warning = record(P_UNKNOWN, detail="no bet id was returned")
    outcome = fail(text, State.PLACEMENT_UNKNOWN)
    return PlacementOutcome(OUTCOME_UNKNOWN, outcome.session, None, text, warning)
