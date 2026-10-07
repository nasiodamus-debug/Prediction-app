"""Recovering after a crash or restart (Story 5.5, AD-6, AD-12, AD-16). Everything here fails closed: when anything is
unexpected it does nothing risky and says so, and nothing in this module can place a bet (it never calls `place`).

`attach_store` wires the flow layer's SQLite storage into the SessionManager: the slate is saved after every step, and the
placed-leg registry is consulted before a bet is confirmed.

`recover_on_startup` runs once at start-up and returns one `Recovered` per thing it did, for the owner:

  * a placement still recorded as PLACING (the bot died around the click), or a session left in PLACING / RECONCILING /
    PLACEMENT_UNKNOWN: the bet history is read FIRST, before anything at all is offered. Found: the bet exists. Not
    found or unreadable: it stays unknown, its legs stay blocked, and it is never placed again.
  * a slate saved before the betslip was touched (EVALUATING, REVIEWING_LIST): resumed exactly where it was, same version,
    so the buttons already on the owner's screen keep working.
  * a slate at or after LOADING_SLIP (loading, resolving availability, awaiting stake or final confirmation): resumed AT
    LOADING_SLIP: the betslip is reloaded, the odds are re-read, and the stake and a fresh confirmation are asked for again
    (nothing that was confirmed before the restart is trusted). A slate that is too old, or whose first game has
    started, is ended instead.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
from typing import Callable

from predictions.core.ports import BookmakerError
from predictions.flow.reconcile import (legs_from_row, parse_created, KIND_FOUND, KIND_NOT_FOUND, ReconcileOutcome, read_matches, reconcile_placement, text_found, text_multiple, text_not_found,
                                        text_unreadable)
from predictions.flow.session import PRE_LOADING_STATES, TERMINAL_STATES, Session, SessionManager, State
from predictions.flow.snapshot import SnapshotError, from_json, to_json
from predictions.flow.store import (P_BOOKED, P_DRY_RUN, P_FAILED, P_PLACED, P_PLACED_BY_OWNER, P_PLACING, P_UNKNOWN,
                                    P_VOIDED, EvaluationStore)

KIND_RESUMED = "resumed"          # a slate before loading: restored as it was (show the list again)
KIND_RELOAD = "reload"            # a slate at or after loading: back at LOADING_SLIP, the caller reloads the betslip
KIND_RECONCILED = "reconciled"    # a bet in PLACING or unclear: the bet history was checked
KIND_ENDED = "ended"              # a slate too old to resume, or one that cannot be resumed safely
KIND_UNREADABLE = "unreadable"    # the saved state could not be read: nothing was done


@dataclass(frozen=True)
class Recovered:
    session_id: str
    kind: str
    text: str
    session: Session | None = None
    reload: bool = False       # run the load job for `session` now
    show_list: bool = False    # show the restored list with its buttons

    @property
    def counts(self) -> bool:
        """Whether this is a session the owner is told was recovered or reconciled."""
        return self.kind in (KIND_RESUMED, KIND_RELOAD, KIND_RECONCILED)


def make_persister(store: EvaluationStore) -> Callable[[Session], None]:
    def persist(session: Session) -> None:
        store.save_session(session.session_id, session.state.value, session.state_version, to_json(session),
                           session.origin)
    return persist


def attach_store(manager: SessionManager, store: EvaluationStore) -> None:
    """The slate is saved after every step; a confirmation asks the placed-leg registry first."""
    manager.attach(make_persister(store),
                   lambda identities: store.find_blocking_legs(identities),
                   lambda token_hash, note: store.annotate_placement(token_hash, note))


class _NoBookmaker:
    """Stands in when the bookmaker cannot even be built (no login details): every history read fails, honestly."""

    def __init__(self, error: Exception):
        self._error = error
        self.dry_run = True

    async def bet_history(self):
        raise BookmakerError("the bookmaker is not available")


def _bookmaker(get_bookmaker: Callable[[], object]):
    try:
        return get_bookmaker()
    except Exception as error:
        return _NoBookmaker(error)


def _count(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


async def recover_on_startup(manager: SessionManager, store: EvaluationStore, get_bookmaker: Callable[[], object],
                             sanitize: Callable[[object], str], now: datetime | None = None) -> list[Recovered]:
    reports: list[Recovered] = []
    try:
        row = store.load_latest_session()
        placing = store.list_placements(P_PLACING)
    except Exception as error:
        return [Recovered("", KIND_UNREADABLE, f"I could not read my saved state ({sanitize(error)}). Nothing was "
                          "resumed and nothing was placed. Send a new request to start again.")]

    session: Session | None = None
    if row is not None:
        try:
            session = from_json(row["payload"])
        except SnapshotError as error:
            reports.append(Recovered(str(row.get("session_id", "")), KIND_UNREADABLE,
                                     f"My saved slate could not be restored ({error}). Nothing was resumed and nothing "
                                     "was placed. Send a new request to start again."))
    bookmaker = _bookmaker(get_bookmaker)
    handled: set[str] = set()

    # 0. A PLACED bet whose legs never reached the registry (a crash between the two writes) gets them now.
    try:
        for prow in store.list_placements(P_PLACED):
            if not prow["dry_run"]:
                warn = _register_missing_legs(store, prow, sanitize, now)
                if warn:
                    reports.append(Recovered(prow["session_id"], KIND_UNREADABLE,
                                             f"Bet {prow['bet_id']} was placed earlier.{warn}"))
    except Exception as error:
        reports.append(Recovered("", KIND_UNREADABLE, f"I could not check the placed list ({sanitize(error)}). Games "
                                 "already placed stay blocked. Nothing was placed."))

    # 1. A bet recorded as PLACING: the bet history is read BEFORE anything is offered.
    for prow in placing:
        own = session is not None and session.session_id == prow["session_id"] and session.state not in TERMINAL_STATES
        if own:
            session = _enter_reconciling(manager, session)
            handled.add(session.session_id)
        outcome = None
        if _is_booking_row(prow):   # AD-25: booking places nothing, so no bet can exist: there is no history to look for
            store.update_placement(prow["token_hash"], P_FAILED,
                                   detail="interrupted by a restart while a booking code was being made; no bet exists", now=now)
            text = ("The bot restarted while a booking code was being made. Booking places nothing, so no bet was placed. "
                    "A code may or may not have been generated: send a new request to start again.")
            final = State.FAILED
        elif prow["dry_run"]:  # a dry run can never click: nothing was placed
            store.update_placement(prow["token_hash"], P_DRY_RUN,
                                   detail="interrupted by a restart; dry run, nothing clicked", now=now)
            text = "A dry-run confirmation was interrupted by the restart. Dry run never clicks Place: nothing was placed."
            final = State.DONE
        else:
            outcome = await reconcile_placement(store, bookmaker, prow["token_hash"], sanitize, now)
            text = "The bot restarted while a bet was being placed. " + outcome.text
            final = State.DONE if outcome.found else State.PLACEMENT_UNKNOWN
        if own:
            session = manager.resolve(session.session_id, final, text, since=parse_created(prow["created_at_utc"])).session or session
        reports.append(Recovered(prow["session_id"], KIND_RECONCILED, text + _warn(outcome), session if own else None))

    if session is None or session.session_id in handled:
        return reports

    # 2. The saved slate itself.
    state = session.state
    # A slate that FAILED before the click (a lost network, a login problem) was never placed: it is offered again like the
    # others below, so the owner always sees the last confirmed games that were not played and decides.
    unplaced_failure = state is State.FAILED and bool(session.retryable) and len(session.slip) > 0
    if state in TERMINAL_STATES and state is not State.PLACEMENT_UNKNOWN and not unplaced_failure:
        return reports  # over: nothing to recover
    if state in (State.PLACING, State.RECONCILING, State.PLACEMENT_UNKNOWN):
        reports.append(await _recover_unclear(manager, store, bookmaker, session, sanitize, now))
        return reports
    if session.token_hash:
        # Whatever state the slate was saved in, a placement row for its token is consulted BEFORE anything is said:
        # a bet that was placed (or may have been) is never described as "nothing was placed".
        try:
            trow = store.get_placement(session.token_hash)
        except Exception:
            trow = None
        if trow is not None and not trow["dry_run"] and trow["state"] in (P_PLACED, P_PLACING, P_UNKNOWN, P_PLACED_BY_OWNER):
            reports.append(await _recover_unclear(manager, store, bookmaker, session, sanitize, now))
            return reports
    if state in PRE_LOADING_STATES:
        if state is State.REVIEWING_FINAL:
            # The final edit screen is not resumed: the owner is returned to the editable list, decisions intact. The
            # version moves on, so the old final screen's Confirm button can no longer work.
            session = replace(session, state=State.REVIEWING_LIST, state_version=session.state_version + 1, last=None)
        manager.restore(session)
        reports.append(Recovered(session.session_id, KIND_RESUMED,
                                 f"The bot restarted. Your open list ({len(session.slip)} of "
                                 f"{_count(len(session.games()), 'game')} kept) was restored "
                                 "with the same Keep and Delete decisions; its buttons still work. Press Final confirm "
                                 "when you are finished.", session, show_list=True))
        return reports
    if manager.is_expired(session) or not len(session.slip):
        ended = replace(session, state=State.CANCELLED, state_version=session.state_version + 1, last=None,
                        message="This slate was too old to resume after the restart.")
        manager.restore(ended)
        reports.append(Recovered(session.session_id, KIND_ENDED,
                                 "The bot restarted while a slate was in progress, but it is too old to resume (or a "
                                 "game has started). It was ended and nothing was placed. Send a new request to start "
                                 "again.", ended))
        return reports
    reloaded = replace(session, state=State.LOADING_SLIP, state_version=session.state_version + 1, last=None,
                       issues=(), loaded=None, odds_shown=False, stake=None, summary=None, token_hash="",
                       overrides=(), override_summary="", message="", retry_stake=None, retry_override=False,
                       history_checked=False, retryable=False, unknown_at=None, check_text="", failed_picks=())
    manager.restore(reloaded)
    where = "failed before anything was placed" if unplaced_failure else f"was at {state.value}"
    reports.append(Recovered(session.session_id, KIND_RELOAD,
                             f"The bot restarted and your last confirmed slate ({_count(len(session.slip), 'game')}) {where}. "
                             "NOTHING was placed. The games are listed below. Press Continue to reload them on SportyBet and "
                             "re-read the odds, then "
                             + ("I will show you a fresh summary to confirm." if manager.book_mode else
                                "I will ask for your stake and a fresh confirmation again.")
                             + " Press Discard to drop this slate.", reloaded, reload=True))
    return reports


def _is_booking_row(row: dict) -> bool:
    """A placement made by a booking token: its stake is exactly zero (nothing in place mode ever has none)."""
    try:
        return Decimal(row["stake"]) == 0 and not row["dry_run"] and row.get("source", "") == ""
    except Exception:
        return False


def _register_missing_legs(store: EvaluationStore, row: dict, sanitize, now: datetime | None) -> str:
    """A PLACED row whose legs never reached the registry (a crash between the two writes) gets them now, in one
    transaction. Returns a sentence for the owner when that could not be done ('' otherwise)."""
    try:
        if any(l["placement_id"] == row["id"] for l in store.list_placed_legs()):
            return ""
        store.resolve_placement_found(row["token_hash"], row["bet_id"], row["booking_code"],
                                      row["detail"] or "legs registered by recovery", legs_from_row(row), now)
        return ""
    except Exception as error:
        return f" (Its games could not be added to the placed list: {sanitize(error)}; they stay blocked anyway.)"


def _warn(outcome: ReconcileOutcome | None) -> str:
    return f" ({outcome.warning})" if outcome is not None and outcome.warning else ""


def _enter_reconciling(manager: SessionManager, session: Session) -> Session:
    """The session is saved as RECONCILING before the history is read, so a crash during the check is recovered again."""
    if session.state is State.RECONCILING:
        return session
    return manager.restore(replace(session, state=State.RECONCILING, state_version=session.state_version + 1, last=None))


async def _recover_unclear(manager: SessionManager, store: EvaluationStore, bookmaker, session: Session,
                           sanitize, now: datetime | None) -> Recovered:
    """A saved session in PLACING, RECONCILING or PLACEMENT_UNKNOWN whose PLACING row is not (or no longer) there."""
    sid = session.session_id
    row = store.get_placement(session.token_hash) if session.token_hash else None
    session = _enter_reconciling(manager, session)
    if row is None:
        # No placement was ever recorded, and it is recorded BEFORE the click: nothing could have been clicked. The
        # history is still read, so a bet that exists anyway is never called "no bet".
        summary = session.summary
        if summary is None:
            text = ("The bot restarted while a bet was being confirmed and nothing about it was recorded, so nothing was "
                    "clicked. No bet was placed.")
            manager.resolve(sid, State.FAILED, text)
            return Recovered(sid, KIND_RECONCILED, text, manager.active)
        kind, matches, reason = await read_matches(bookmaker, summary.legs, summary.stake, set(), sanitize,
                                                   summary.total_odds)
        if kind == KIND_NOT_FOUND:
            text = ("The bot restarted while a bet was being confirmed. Nothing had been recorded as clicked, and your "
                    "bet history shows no such bet. No bet was placed.")
            manager.resolve(sid, State.FAILED, text)
        else:
            detail = (text_multiple(matches) if matches and kind != KIND_FOUND else
                      text_found(matches[0]) if matches else text_unreadable(reason))
            text = "The bot restarted while a bet was being confirmed. " + detail
            manager.resolve(sid, State.PLACEMENT_UNKNOWN, text)
        return Recovered(sid, KIND_RECONCILED, text, manager.active)
    if row["state"] in (P_BOOKED, P_PLACED_BY_OWNER):
        text = (f"The bot restarted after booking code {row['booking_code']} was made. No bet was placed by the bot."
                if row["state"] == P_BOOKED else
                f"The bot restarted after you confirmed that you placed booking code {row['booking_code']}.")
        manager.resolve(sid, State.DONE, text)
        return Recovered(sid, KIND_RECONCILED, text, manager.active)
    if _is_booking_row(row) and row["state"] in (P_PLACING, P_UNKNOWN):
        store.update_placement(row["token_hash"], P_FAILED, detail="interrupted by a restart; booking places nothing", now=now)
        text = ("The bot restarted while a booking code was being made. Booking places nothing, so no bet was placed. Send "
                "a new request to start again.")
        manager.resolve(sid, State.FAILED, text)
        return Recovered(sid, KIND_RECONCILED, text, manager.active)
    if row["state"] == P_PLACED:
        text = f"The bot restarted after your bet was placed (bet id {row['bet_id']}). I will not place it again."
        text += _register_missing_legs(store, row, sanitize, now)
        manager.resolve(sid, State.DONE, text)
        return Recovered(sid, KIND_RECONCILED, text, manager.active)
    if row["state"] in (P_DRY_RUN, P_FAILED):
        text = ("The bot restarted after a placement attempt that did not place anything "
                f"({'dry run' if row['state'] == P_DRY_RUN else 'it failed before the click'}). No bet was placed.")
        manager.resolve(sid, State.DONE if row["state"] == P_DRY_RUN else State.FAILED, text)
        return Recovered(sid, KIND_RECONCILED, text, manager.active)
    if row["state"] == P_VOIDED:
        text = ("The bot restarted after SportyBet's odds changed before a click. Nothing was placed. Send a new request "
                "to start again.")
        manager.resolve(sid, State.FAILED, text)
        return Recovered(sid, KIND_RECONCILED, text, manager.active)
    outcome = await reconcile_placement(store, bookmaker, row["token_hash"], sanitize, now)
    text = "The bot restarted while a bet's result was unclear. " + outcome.text
    manager.resolve(sid, State.DONE if outcome.found else State.PLACEMENT_UNKNOWN, text,
                    since=parse_created(row["created_at_utc"]))
    return Recovered(sid, KIND_RECONCILED, text + _warn(outcome), manager.active)
