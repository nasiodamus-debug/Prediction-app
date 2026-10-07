"""The session state machine (AD-6, AD-18, AD-19). In memory; PLACING is also persisted by flow/placement.py.

    (PLACE_MODE=book, AD-25: no AWAITING_STAKE question; the summary has no stake and PLACING reads a booking code)
    REVIEWING_LIST -> LOADING_SLIP -> (RESOLVING_AVAILABILITY -> LOADING_SLIP)* -> AWAITING_STAKE
        -> AWAITING_FINAL_CONFIRM -> PLACING -> DONE | PLACEMENT_UNKNOWN | FAILED
        and PLACING -> AWAITING_FINAL_CONFIRM when the odds moved.

    (AD-26) LOADING_SLIP -> LOAD_PARTIAL when some picks could not be added to the betslip and others could. The owner
    presses one of: Go ahead with the loaded picks (-> AWAITING_STAKE with the kept list reduced to them, the left-out picks
    recorded in `dropped`), Retry the failed picks (-> LOADING_SLIP, counted in the 3-retry cap) or Cancel (-> CANCELLED).
    No token exists in LOAD_PARTIAL, so a restart simply reloads (like any other pre-final state).

Every change goes through this class and is a compare-and-swap on `state_version`: a button made for an older
version does nothing, a repeated tap is idempotent, and any change to the slip, the odds or the stake bumps the
version so earlier confirmations stop working (AD-19). Views (open a pick, back, page) never change the version.
Only one session is open at a time (AD-18, Story 5.7): a new trigger while one is open does not replace it until the
owner presses the explicit Replace button (`request_start`); the old slate's buttons then stop working.

Every change is handed to an optional `persist` callable (flow/recovery.py saves it to SQLite, Story 5.5), and the final
confirmation asks an optional `registry` callable whether a leg was already placed (AD-6). Both are set by `attach`.

This module is the ONLY place a PlacementToken is minted (AD-6): the token is created inside `act`, at the moment
of the final confirmation, from the summary the owner was shown.
"""
from __future__ import annotations

import threading
import uuid
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, replace
from decimal import Decimal
from enum import Enum
from typing import Callable, Iterable

from predictions.core.betting import LoadedSlip, UnsupportedLeg, leg_from_evaluation
from predictions.core.play_labels import play_label
from predictions.core.model import Evaluation
from predictions.core.options import as_substitute
from predictions.core.placement import (BOOKING_STAKE, MINT_AUTHORITY, SUMMARY_TTL, DuplicateLeg, PlacementToken,
                                         Summary, mint_token)
from predictions.core.slip import Game, Pick, Slip, SlipError, UnknownPick, build_games, slip_of
from predictions.core.staking import DEFAULT_MAX_STAKE, StakeError, parse_stake
from predictions.core.types import Owner
from predictions.core.betting import leg_identity


class State(Enum):
    EVALUATING = "EVALUATING"
    REVIEWING_LIST = "REVIEWING_LIST"
    REVIEWING_FINAL = "REVIEWING_FINAL"  # the owner pressed Final confirm: the final edit screen (Confirm games / Back to adjust)
    LOADING_SLIP = "LOADING_SLIP"  # games confirmed; the bookmaker's betslip is being cleared and loaded
    RESOLVING_AVAILABILITY = "RESOLVING_AVAILABILITY"  # a game or option is missing: the owner skips or accepts
    LOAD_PARTIAL = "LOAD_PARTIAL"                      # AD-26: some picks loaded, some could not be added: the owner decides
    AWAITING_STAKE = "AWAITING_STAKE"                  # odds and total odds are on screen; the owner types a stake
    AWAITING_FINAL_CONFIRM = "AWAITING_FINAL_CONFIRM"  # the summary is on screen; ONE explicit confirmation
    PLACING = "PLACING"                                # token minted, persisted, the adapter is working
    PLACEMENT_UNKNOWN = "PLACEMENT_UNKNOWN"            # an error after the click: resolved from bet history only
    FAILED = "FAILED"                                  # an error before the click: no bet, reported, no retry
    DONE = "DONE"                                      # finished: dry run reported, bet proven, or nothing left
    RECONCILING = "RECONCILING"                        # restart with a bet in PLACING: bet history is being checked
    CANCELLED = "CANCELLED"                            # replaced by a newer slate, or ended with no list


# A slate in one of these is over: it is not "open", so a new trigger starts without asking (Story 5.7).
# PLACEMENT_UNKNOWN is over as a slate; its legs stay blocked by the placed-leg registry until the owner overrides.
TERMINAL_STATES = frozenset({State.DONE, State.FAILED, State.PLACEMENT_UNKNOWN, State.CANCELLED})
# A bet may be in flight: the slate can neither be replaced nor cancelled.
BUSY_STATES = frozenset({State.PLACING, State.RECONCILING})
# Saved before the betslip was touched: a restart resumes exactly here (AD-6). Everything later resumes at LOADING_SLIP.
PRE_LOADING_STATES = frozenset({State.EVALUATING, State.REVIEWING_LIST, State.REVIEWING_FINAL})
LIST_STATES = frozenset({State.REVIEWING_LIST, State.REVIEWING_FINAL})


class Action(Enum):
    DELETE = "d"    # delete a PLAY: deleting the kept play takes its game out of the slip; the argument is the play's pick id
    CONFIRM = "k"   # only from the final edit screen
    BACK = "b"      # view: back to the list
    PAGE = "g"      # view: another page of the list
    SKIP = "s"      # Story 5.2: drop the game with a missing game or option
    ACCEPT = "a"    # Story 5.2: take the offered closest option
    FINAL = "f"     # Story 5.4: the one final confirmation; the argument is the summary id
    OVERRIDE = "v"  # Story 5.5: place a leg that was already placed anyway; the argument is the summary id
    REPLACE = "r"   # Story 5.7: replace the open slate with the new one; the argument is the pending id
    KEEP = "x"      # Story 5.7: keep the open slate and drop the new request; the argument is the pending id
    RESTORE = "u"   # KEEP a play: it becomes THE kept play of its game (any other play of it is un-kept); argument: pick id
    DONE = "n"      # "Final confirm": the owner is finished editing, show the review (final edit) screen
    ADJUST = "j"    # "Back to adjust": from the final edit screen back to the editable list
    CHANGE = "m"    # "Change amount": from the final summary back to the stake step (loaded slip and odds kept)
    RETRY = "t"     # "Try again": FAILED before anything was clicked -> reload the same plays with the same stake
    CHECK = "e"     # "Check history again": PLACEMENT_UNKNOWN -> a read-only look at the bet history (never places)
    RETRY_ASK = "w" # "Try again, I checked": view of the loud warning (only after a check found nothing)
    RETRY_YES = "y" # "Yes, place again": the explicit override, then the same reload as RETRY
    GO_AHEAD = "h"  # AD-26: LOAD_PARTIAL -> go ahead with only the loaded picks
    RETRY_PARTIAL = "p"  # AD-26: LOAD_PARTIAL -> clear the slip and add ALL kept picks again (counts as a retry)
    CANCEL = "q"    # AD-26: LOAD_PARTIAL -> end the slate; nothing is booked or placed


VIEWS = frozenset({Action.BACK, Action.PAGE})
CHANGES = frozenset({Action.DELETE, Action.RESTORE, Action.DONE, Action.ADJUST, Action.CONFIRM, Action.SKIP, Action.ACCEPT, Action.FINAL,
                     Action.OVERRIDE, Action.CHANGE, Action.RETRY, Action.CHECK, Action.RETRY_YES, Action.GO_AHEAD,
                     Action.RETRY_PARTIAL, Action.CANCEL})
MAX_RETRIES = 3  # owner-pressed retries per slate (before-click Try again and the unknown-result override together)


class Status(Enum):
    APPLIED = "applied"        # the list changed; `session` is the new version
    VIEW = "view"              # nothing changed; show what was asked for
    STALE = "stale"            # the button is from an older list (or another session): nothing happened
    REPEAT = "repeat"          # the same tap again after it was applied: no extra effect
    BLOCKED = "blocked"        # the change is not allowed yet (see `notice`)
    UNKNOWN_PICK = "unknown"   # the pick is not in this list
    EXPIRED = "expired"        # the summary timed out: a NEW summary was issued, nothing was confirmed
    DUPLICATE = "duplicate"    # Story 5.5: a leg was already placed; only an explicit override can go on
    ASK_REPLACE = "ask_replace"  # Story 5.7: a slate is open; the owner must confirm replacing it


ISSUE_GAME = "game"      # the game is not on SportyBet (or has started): the only choice is to skip it
ISSUE_OPTION = "option"  # the game is there but the selected option is not


@dataclass(frozen=True)
class Issue:
    """One confirmed pick that cannot be loaded as it is (Story 5.2). Nothing happens until the owner decides."""
    pick_id: str
    kind: str                            # ISSUE_GAME or ISSUE_OPTION
    detail: str = ""
    alternative: Evaluation | None = None  # the closest option that clears the floor, or None (then: skip)


@dataclass(frozen=True)
class PickFailure:
    """AD-26: a kept pick the bookmaker could not add to the betslip. `times` counts the loads (this one included) in which
    it failed, so the screen can say 'failed twice'. `reason` is the adapter's plain reason, already sanitized."""
    pick_id: str
    reason: str
    times: int = 1


@dataclass(frozen=True)
class DroppedPick:
    """AD-26: a pick the owner chose to go ahead WITHOUT (it could not be added). It is no longer in the kept list; this
    record keeps what it was and why, for the odds screen, the summary and the placement record."""
    pick_id: str
    label: str      # "Alpha vs Beta: Over 2.5 @ 1.85"
    reason: str


@dataclass(frozen=True)
class Session:
    session_id: str
    state: State
    state_version: int
    slip: Slip
    last: tuple | None = None  # (version it was applied to, action, arg) of the latest change
    created_at: datetime | None = None  # UTC; CONFIRM is refused (STALE) once the list is too old
    earliest_kickoff: datetime | None = None  # UTC; CONFIRM is refused once the first game has kicked off
    issues: tuple[Issue, ...] = ()       # RESOLVING_AVAILABILITY: what still needs the owner
    loaded: LoadedSlip | None = None     # the betslip as SportyBet holds it (odds are final only once shown)
    odds_shown: bool = False             # the odds message reached the owner: only then may a stake be asked for
    stake: Decimal | None = None
    summary: Summary | None = None       # what the owner is asked to confirm
    token_hash: str = ""                 # PLACING and after: the one token that was minted
    message: str = ""                    # FAILED / PLACEMENT_UNKNOWN / DONE: what was reported
    origin: str = ""                     # where the slate came from: "matchday" or "review:<code>"
    overrides: tuple[tuple[str, str, str], ...] = ()  # legs (match, market, line) the owner said to place again ...
    override_summary: str = ""           # ... valid only for THIS summary id: any change voids it
    menu: tuple[Pick, ...] = ()          # every offered play (all valid options of every game); empty: the slip itself
    chosen: tuple[str, ...] = ()         # the chosen option (pick id) per game; a game not named here uses the advised one
    removed_games: tuple[str, ...] = ()  # match keys of the games the owner deleted (kept in the list, shown as removed)
    retries_used: int = 0                # owner-pressed retries of this slate so far (capped by MAX_RETRIES, survives a restart)
    retryable: bool = False              # FAILED: certainly before the click, so a Try again button is safe
    history_checked: bool = False        # PLACEMENT_UNKNOWN: an owner-pressed history check ran and found nothing
    retry_stake: Decimal | None = None   # a retry's reload: the stake carried over to the NEW summary (never a token)
    retry_override: bool = False         # the owner chose "Yes, place again": applied to the NEW summary only
    unknown_at: datetime | None = None   # PLACEMENT_UNKNOWN: when the Place click happened (UTC), for "N minutes ago"
    check_text: str = ""                 # PLACEMENT_UNKNOWN: what the latest owner-pressed history check said
    booking: bool = False                # AD-25: PLACE_MODE=book. No stake step; the final action is a booking code
    failed_picks: tuple[PickFailure, ...] = ()   # AD-26 LOAD_PARTIAL (and the reload after Retry): picks that could not be added
    dropped: tuple[DroppedPick, ...] = ()        # AD-26: picks the owner went ahead without (never silently: always listed)

    def games(self) -> tuple[Game, ...]:
        """The list the owner edits: every game with its options, its chosen play and whether it is kept."""
        return build_games(self.menu or self.slip.picks, self.chosen, self.removed_games)

    def game_of(self, pick_id: str) -> Game:
        for game in self.games():
            if any(p.pick_id == pick_id for p in game.options):
                return game
        raise UnknownPick(pick_id)

    def find_pick(self, pick_id: str) -> Pick:
        return next(p for p in self.game_of(pick_id).options if p.pick_id == pick_id)

    def with_decisions(self, chosen: tuple[str, ...] | None = None, removed_games: tuple[str, ...] | None = None,
                       menu: tuple[Pick, ...] | None = None, **changes) -> "Session":
        """A copy with new decisions and the Slip rebuilt from them (the chosen option of every kept game)."""
        base = replace(self, menu=menu if menu is not None else (self.menu or self.slip.picks),
                       chosen=self.chosen if chosen is None else chosen,
                       removed_games=self.removed_games if removed_games is None else removed_games)
        games = base.games()
        return replace(base, chosen=tuple(g.chosen_id for g in games),
                       removed_games=tuple(sorted(g.key for g in games if not g.kept)), slip=slip_of(games), **changes)


@dataclass(frozen=True)
class Result:
    status: Status
    session: Session | None = None
    pick: Pick | None = None
    arg: str = ""
    notice: str = ""
    token: PlacementToken | None = None  # only on the final confirmation: the single-use token, minted here
    load_next: bool = False              # the caller must (re)load the betslip now
    duplicates: tuple[DuplicateLeg, ...] = ()  # DUPLICATE: the legs the registry already knows
    pending: "Pending | None" = None     # ASK_REPLACE: the new slate waiting for the owner's decision


@dataclass(frozen=True)
class Pending:
    """A new slate waiting for the owner to confirm replacing the open one (Story 5.7). The Replace and Keep buttons
    carry `pending_id`, are bound to the open session's id and version, and die if either changes."""
    pending_id: str
    session_id: str                      # the id the new slate will have
    evaluations: tuple[Evaluation, ...]
    origin: str
    old_session_id: str
    old_version: int


MAX_AGE = timedelta(hours=6)  # a list older than this can no longer be confirmed
NOTICE_CHOOSE_FIRST = "Choose one pick for each match that has more than one before you confirm."
NOTICE_EMPTY = "There is nothing to confirm: no play is kept. Keep at least one play first."
NOTICE_DONE_FIRST = "Press Final confirm first: the final list is checked before games are confirmed."
NOTICE_NO_ALTERNATIVE = "There is no option to accept for this game. Please skip it."
NOTICE_ODDS_FIRST = "The odds have not been shown yet, so no stake is asked for. Please wait."
NOTICE_EXPIRED = "That confirmation expired. Here is a new summary: check it and confirm again."
NOTICE_BUSY = "A bet is being placed or checked right now. Wait for that to finish before starting something new."
NOTICE_REGISTRY_DOWN = ("I could not check the list of legs already placed, so nothing was placed. "
                        "Nothing will be retried; send a new request to start again.")
NOTICE_RETRY_CAP = ("All {n} retries for this slate are used. Send a new request to start again.")
NOTICE_CHECK_FIRST = "Press Check history again first: a second attempt is offered only after a check found nothing."
NOTICE_ALREADY_PLACED = "Your SportyBet history shows this bet was already placed, so it will not be placed again."
NOTICE_AUDIT_DOWN = "I could not record your choice, so nothing was reloaded and nothing was placed. Nothing is retried."
NOTICE_TOO_OLD = "This list is more than 6 hours old, so it cannot be confirmed. Send a new request to start again."
NOTICE_STARTED = ("{game} has already kicked off, so it cannot go in the bet. Delete it (or any other game that has "
                  "started), then press Final confirm again.")
MSG_PARTIAL_CANCELLED = ("You cancelled. Nothing was placed and no booking code was made. Your SportyBet betslip may still "
                         "hold the picks that were loaded: the next load clears it first.")
NOTICE_PARTIAL_INCONSISTENT = ("The loaded picks no longer match the list, so I did not go ahead. Nothing was placed. "
                               "Press Retry the failed picks or Cancel.")
MSG_NOTHING_LEFT = "Every game was skipped, so there is nothing to bet on. Nothing was placed."


def _pick_label(pick: Pick) -> str:
    mid, ev = pick.match_id, pick.evaluation
    return f"{mid.home} vs {mid.away}: {play_label(ev)} @ {ev.odds:.2f}"


def new_session_id() -> str:
    return uuid.uuid4().hex[:8]


class SessionManager:
    def __init__(self, clock: Callable[[], datetime] | None = None, max_stake: Decimal = DEFAULT_MAX_STAKE,
                 book_mode: bool = False):
        self.max_stake = Decimal(max_stake)   # the largest stake accepted (app.py reads MAX_STAKE and passes it here)
        # AD-25 (PLACE_MODE=book): no stake is asked for or typed; the summary carries BOOKING_STAKE and its token can only
        # read a booking code. MAX_STAKE and the positive-stake rule are skipped here, and ONLY here: place mode keeps both.
        self.book_mode = bool(book_mode)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()
        self._active: Session | None = None
        self._pending: Pending | None = None
        self._last_replace: tuple | None = None   # (old session id, version, pending id) of the latest Replace tap
        self._persist: Callable[[Session], None] | None = None
        self._deferring = False                   # AD-22: saves are held back while a button press is being answered
        self._deferred: list[Session] = []
        self._registry: Callable[[list[tuple[str, str, str]]], list[DuplicateLeg]] | None = None
        self._audit: Callable[[str, str], None] | None = None   # (placement token hash, note): the override audit record
        self.persist_errors: list[str] = []       # what could not be saved (never secrets); the caller may report it

    def take_persist_errors(self) -> list[str]:
        """What could not be saved since the last call (error names only, never secrets); the list is cleared, so
        each failure is reported once. The caller tells the owner."""
        with self._lock:
            taken, self.persist_errors = self.persist_errors, []
        return taken

    @property
    def active(self) -> Session | None:
        return self._active

    @property
    def pending(self) -> Pending | None:
        return self._pending

    def attach(self, persist: Callable[[Session], None] | None = None,
               registry: Callable[[list[tuple[str, str, str]]], list[DuplicateLeg]] | None = None,
               audit: Callable[[str, str], None] | None = None) -> None:
        """Wire in the flow layer's storage: `persist(session)` after every change, `registry(identities)` that
        returns the legs already placed, and `audit(token_hash, note)` that records on the placement row that the owner
        chose a second attempt after an unknown result. Without a registry no duplicate check runs (tests); app.py
        always attaches. Without an audit callable the second attempt after an unknown result is refused."""
        self._persist, self._registry, self._audit = persist, registry, audit

    def _set(self, session: Session | None) -> None:
        """The one place the active session changes: it is saved after each step (Story 5.5)."""
        self._active = session
        if session is not None and self._persist is not None:
            if self._deferring:
                self._deferred.append(session)   # saved, in order, by flush_deferred_persist() right after the answer
                return
            self._save(session)

    def _save(self, session: Session) -> None:
        try:
            self._persist(session)
        except Exception as error:  # a failed save never changes what happens next; it is recorded
            self.persist_errors.append(type(error).__name__)
            del self.persist_errors[:-50]   # bounded

    def defer_persist(self) -> None:
        """Hold back the save of the next synchronous state change(s) so the caller can answer the Telegram button first
        (AD-22). The caller MUST call `flush_deferred_persist()` afterwards, before it awaits any slow work."""
        with self._lock:
            self._deferring = True
            self._deferred = []

    def flush_deferred_persist(self) -> None:
        """Save, in order, every change that was held back (the same saves as without the delay, just after the answer)."""
        with self._lock:
            waiting, self._deferred, self._deferring = self._deferred, [], False
            if self._persist is not None:
                for session in waiting:
                    self._save(session)

    @staticmethod
    def is_open(session: Session | None) -> bool:
        return session is not None and session.state not in TERMINAL_STATES

    def start(self, evaluations: Iterable[Evaluation], session_id: str | None = None, origin: str = "") -> Session:
        """A new session in REVIEWING_LIST holding the passed evaluations. It replaces any active session, whose
        buttons stop working. Triggers do NOT call this directly while a slate is open: they use `request_start`,
        which asks the owner first (Story 5.7)."""
        with self._lock:
            return self._install(evaluations, session_id, origin)

    def _install(self, evaluations: Iterable[Evaluation], session_id: str | None, origin: str) -> Session:
        menu = Slip.from_evaluations(evaluations).picks      # every valid option of every game
        kickoffs = [p.match_id.kickoff_utc for p in menu]
        session = Session(session_id or new_session_id(), State.REVIEWING_LIST, 1, Slip(),
                          created_at=self._clock(), earliest_kickoff=min(kickoffs) if kickoffs else None,
                          origin=origin, booking=self.book_mode).with_decisions(chosen=(), removed_games=(), menu=menu)
        self._retire_active()
        self._pending = None
        self._set(session)
        return session

    def _retire_active(self) -> None:
        """The open session (if any) is over: it is saved as CANCELLED, so a restart never brings it back."""
        old = self._active
        if old is not None and old.state not in TERMINAL_STATES:
            self._set(replace(old, state=State.CANCELLED, state_version=old.state_version + 1, last=None,
                              message="This slate was replaced or ended."))

    def request_start(self, owner: Owner, evaluations: Iterable[Evaluation], origin: str = "",
                      session_id: str | None = None) -> Result:
        """A new trigger produced a slate. With no open slate it starts at once (APPLIED). With one open, NOTHING
        changes yet: the result is ASK_REPLACE carrying the pending slate, and only the owner's Replace button
        (`Action.REPLACE`) starts it. While a bet is being placed or checked the answer is BLOCKED."""
        if not isinstance(owner, Owner):
            raise TypeError("session actions need an Owner made by the owner gate")
        evaluations = tuple(evaluations)
        with self._lock:
            Slip.from_evaluations(evaluations)  # refuses anything that did not pass (a disqualified pick never enters)
            active = self._active
            if not self.is_open(active):
                return Result(Status.APPLIED, self._install(evaluations, session_id, origin))
            if active.state in BUSY_STATES:
                return Result(Status.BLOCKED, active, notice=NOTICE_BUSY)
            pending = Pending(uuid.uuid4().hex[:8], session_id or new_session_id(), evaluations, origin,
                              active.session_id, active.state_version)
            self._pending = pending
            return Result(Status.ASK_REPLACE, active, pending=pending)

    def end(self) -> None:
        """No list to review and nothing should stay live: the open session (if any) is cancelled."""
        with self._lock:
            if self._active is not None and self._active.state in BUSY_STATES:
                return  # a bet may be in flight: never cancel that
            self._retire_active()
            self._active = None
            self._pending = None

    def restore(self, session: Session) -> Session:
        """Install a session recovered after a restart (flow/recovery.py). Saved like any other change."""
        with self._lock:
            self._pending = None
            session = replace(session, booking=self.book_mode)   # a slate restored under another PLACE_MODE adopts the current one
            self._set(session)
            return session

    def now(self) -> datetime:
        """The manager's clock (UTC), so screens can say how long ago something happened."""
        return self._clock()

    def is_expired(self, session: Session) -> bool:
        return self._expired(session)

    def _expired(self, session: Session) -> bool:
        return self._expiry(session) != ""

    def _expiry(self, session: Session) -> str:
        """Why this list can no longer be confirmed ('' when it can). Only the KEPT plays count for kickoff: a game the
        owner deleted is not in the bet, so its kickoff must never block the rest (the reported bug)."""
        now = self._clock()
        if session.created_at is not None and now - session.created_at > MAX_AGE:
            return NOTICE_TOO_OLD
        started = [p for p in session.slip.picks if now >= p.match_id.kickoff_utc]
        if started:
            first = min(started, key=lambda p: p.match_id.kickoff_utc)
            return NOTICE_STARTED.format(game=f"{first.match_id.home} vs {first.match_id.away}"[:80])
        return ""

    def act(self, owner: Owner, session_id: str, version: int, action: Action, arg: str = "") -> Result:
        if not isinstance(owner, Owner):
            raise TypeError("session actions need an Owner made by the owner gate")
        with self._lock:
            if action in (Action.REPLACE, Action.KEEP):
                return self._decide_replace(session_id, version, action, arg)
            session = self._active
            if session is None or session.session_id != session_id:
                return Result(Status.STALE)
            if version != session.state_version:
                if action in CHANGES and session.last == (version, action, arg):
                    return Result(Status.REPEAT, session)
                return Result(Status.STALE, session)
            if session.state is State.REVIEWING_LIST:
                return self._act_on_list(session, version, action, arg)
            if session.state is State.REVIEWING_FINAL:
                return self._act_on_final(session, version, action, arg)
            if session.state is State.RESOLVING_AVAILABILITY and action in (Action.SKIP, Action.ACCEPT):
                return self._resolve(session, version, action, arg)
            if session.state is State.LOAD_PARTIAL and action in (Action.GO_AHEAD, Action.RETRY_PARTIAL, Action.CANCEL):
                return self._act_on_partial(session, version, action, arg)
            if session.state is State.AWAITING_FINAL_CONFIRM and action is Action.FINAL:
                return self._final_confirm(session, version, arg)
            if session.state is State.AWAITING_FINAL_CONFIRM and action is Action.OVERRIDE:
                return self._override(session, version, arg)
            if session.state is State.AWAITING_FINAL_CONFIRM and action is Action.CHANGE:
                return self._change_amount(session, version, arg)
            if session.state is State.FAILED and action is Action.RETRY:
                return self._retry(session, version, arg)
            if session.state is State.PLACEMENT_UNKNOWN and action in (Action.CHECK, Action.RETRY_ASK, Action.RETRY_YES,
                                                                       Action.BACK):
                return self._act_on_unknown(session, version, action, arg)
            return Result(Status.STALE, session)

    # ---- Story 5.7: replacing the open slate needs the owner's explicit button ----
    def _decide_replace(self, session_id: str, version: int, action: Action, arg: str) -> Result:
        session, pending = self._active, self._pending
        matches = (pending is not None and session is not None and pending.pending_id == arg
                   and pending.old_session_id == session_id == session.session_id
                   and pending.old_version == version == session.state_version)
        if action is Action.KEEP:
            if matches:
                self._pending = None
                return Result(Status.VIEW, session, arg=arg, notice="kept")
            return Result(Status.REPEAT if pending is None else Status.STALE, session)
        if not matches:
            if self._last_replace == (session_id, version, arg):
                return Result(Status.REPEAT, session)  # the same tap again: no extra effect
            return Result(Status.STALE, session)
        if session.state in BUSY_STATES:
            return Result(Status.BLOCKED, session, notice=NOTICE_BUSY)
        new = self._install(pending.evaluations, pending.session_id, pending.origin)  # retires the old slate
        self._last_replace = (session_id, version, arg)
        return Result(Status.APPLIED, new, arg=arg)

    # ---- REVIEWING_LIST (Story 2.6, unchanged) ----
    def _act_on_list(self, session: Session, version: int, action: Action, arg: str) -> Result:
        if action in VIEWS:
            return Result(Status.VIEW, session, arg=arg)
        if action is Action.CONFIRM:
            return Result(Status.BLOCKED, session, notice=NOTICE_DONE_FIRST)
        if action is Action.DONE:
            if self._expiry(session):
                return Result(Status.STALE, session, notice=self._expiry(session))
            if not len(session.slip):
                return Result(Status.BLOCKED, session, notice=NOTICE_EMPTY)
            new = replace(session, state=State.REVIEWING_FINAL, state_version=version + 1, last=(version, action, arg))
            self._set(new)
            return Result(Status.APPLIED, new, arg=arg)
        if action in (Action.DELETE, Action.RESTORE):
            try:
                game = session.game_of(arg)
            except UnknownPick:
                return Result(Status.UNKNOWN_PICK, session)
            games = session.games()
            removed = set(session.removed_games)
            chosen = tuple(g.chosen_id for g in games)
            is_kept_play = game.kept and game.chosen_id == arg
            if action is Action.DELETE:
                if not is_kept_play:
                    return Result(Status.REPEAT, session)  # this play is not kept: nothing to delete
                removed.add(game.key)                      # the game now has no kept play: it is out of the slip
            else:  # KEEP: this play becomes THE kept play of its game; any other play of it is un-kept
                if is_kept_play:
                    return Result(Status.REPEAT, session)
                chosen = tuple(arg if g.key == game.key else g.chosen_id for g in games)
                removed.discard(game.key)
            new = session.with_decisions(chosen=chosen, removed_games=tuple(sorted(removed)),
                                         state_version=version + 1, last=(version, action, arg))
            self._set(new)
            return Result(Status.APPLIED, new, arg=arg)
        return Result(Status.STALE, session)

    # ---- REVIEWING_FINAL: the compact final edit screen; Confirm games moves on, Back to adjust returns ----
    def _act_on_final(self, session: Session, version: int, action: Action, arg: str) -> Result:
        if action is Action.ADJUST:
            new = replace(session, state=State.REVIEWING_LIST, state_version=version + 1, last=(version, action, arg))
            self._set(new)  # every decision is kept; the final screen's buttons stop working (new version)
            return Result(Status.APPLIED, new, arg=arg)
        if action is Action.CONFIRM:
            if self._expiry(session):
                return Result(Status.STALE, session, notice=self._expiry(session))
            if not len(session.slip):
                return Result(Status.BLOCKED, session, notice=NOTICE_EMPTY)
            new = replace(session, state=State.LOADING_SLIP, state_version=version + 1, last=(version, action, arg))
            self._set(new)  # compare-and-swap: the version was checked under the same lock
            return Result(Status.APPLIED, new, arg=arg, load_next=True)
        return Result(Status.STALE, session)

    # ---- RESOLVING_AVAILABILITY (Story 5.2): the owner decides, nothing happens on its own ----
    def _resolve(self, session: Session, version: int, action: Action, arg: str) -> Result:
        issue = next((i for i in session.issues if i.pick_id == arg), None)
        if issue is None:
            return Result(Status.UNKNOWN_PICK, session)
        try:
            if action is Action.SKIP:
                slip = session.slip.delete(arg)
            else:
                if issue.alternative is None:
                    return Result(Status.BLOCKED, session, notice=NOTICE_NO_ALTERNATIVE)
                slip = session.slip.substitute(arg, as_substitute(session.slip.get(arg).evaluation, issue.alternative))
        except (UnknownPick, SlipError):
            return Result(Status.UNKNOWN_PICK, session)
        if not slip.one_per_match:  # cannot happen (a skip removes, a substitute keeps the match); refuse if it does
            return Result(Status.BLOCKED, session, notice=NOTICE_CHOOSE_FIRST)
        issues = tuple(i for i in session.issues if i.pick_id != arg)
        last = (version, action, arg)
        if issues:
            new = replace(session, slip=slip, issues=issues, state_version=version + 1, last=last)
            self._set(new)
            return Result(Status.APPLIED, new, arg=arg)
        if not len(slip):
            new = replace(session, slip=slip, issues=(), state=State.DONE, state_version=version + 1, last=last,
                          message=MSG_NOTHING_LEFT)
            self._set(new)
            return Result(Status.APPLIED, new, arg=arg)
        new = replace(session, slip=slip, issues=(), state=State.LOADING_SLIP, state_version=version + 1, last=last)
        self._set(new)
        return Result(Status.APPLIED, new, arg=arg, load_next=True)  # the betslip is checked and loaded again

    # ---- the final confirmation (Story 5.4): the ONLY place a token is minted ----
    def _final_confirm(self, session: Session, version: int, summary_id: str) -> Result:
        summary = session.summary
        if summary is None or session.loaded is None or summary_id != summary.summary_id:
            return Result(Status.STALE, session)
        if summary.booking != self.book_mode:   # a summary made under the other PLACE_MODE never mints a token
            return Result(Status.STALE, session)
        if summary.expired(self._clock()):
            fresh = self._new_summary(session, session.loaded, summary.stake)
            new = replace(session, summary=fresh, state_version=version + 1, last=None)
            self._set(new)
            return Result(Status.EXPIRED, new, notice=NOTICE_EXPIRED)
        blocked = self._blocked_legs(session, summary)
        if isinstance(blocked, Result):
            return blocked
        if blocked:
            return Result(Status.DUPLICATE, session, duplicates=tuple(blocked))
        token = mint_token(MINT_AUTHORITY, summary)
        new = replace(session, state=State.PLACING, state_version=version + 1,
                      last=(version, Action.FINAL, summary_id), token_hash=token.token_hash)
        self._set(new)
        return Result(Status.APPLIED, new, arg=summary_id, token=token)

    def _known_duplicates(self, summary: Summary) -> list[DuplicateLeg] | Result:
        """Legs of this summary the placed-leg registry already knows. A registry that cannot be read is a refusal."""
        if self._registry is None:
            return []
        try:
            return list(self._registry([leg_identity(leg.leg_key) for leg in summary.legs]))
        except Exception:
            return Result(Status.BLOCKED, self._active, notice=NOTICE_REGISTRY_DOWN)

    def _blocked_legs(self, session: Session, summary: Summary) -> list[DuplicateLeg] | Result:
        """Known duplicates minus the ones the owner explicitly overrode FOR THIS EXACT SUMMARY."""
        found = self._known_duplicates(summary)
        if isinstance(found, Result):
            return found
        allowed = set(session.overrides) if session.override_summary == summary.summary_id else set()
        return [d for d in found if d.identity not in allowed]

    # ---- Change amount: the final summary goes back to the stake step; the betslip and odds are kept ----
    def _change_amount(self, session: Session, version: int, arg: str) -> Result:
        """AWAITING_FINAL_CONFIRM -> AWAITING_STAKE. Pure in-memory change: no betslip reload, no bookmaker call. The
        summary is dropped and the version moves on, so the old confirm button (and any override) is dead; the current
        stake is kept only to be shown. The next valid stake makes a NEW summary and, on confirmation, a NEW token."""
        if session.loaded is None or self.book_mode:   # booking mode has no stake to change
            return Result(Status.STALE, session)
        new = replace(session, state=State.AWAITING_STAKE, summary=None, token_hash="", overrides=(),
                      override_summary="", odds_shown=True, state_version=version + 1,
                      last=(version, Action.CHANGE, arg))
        self._set(new)
        return Result(Status.APPLIED, new, arg=arg)

    # ---- owner-pressed retries (never automatic): the new attempt is a NEW load, a NEW summary and a NEW token ----
    def _retry_refusal(self, session: Session) -> Result | None:
        if session.retries_used >= MAX_RETRIES:
            return Result(Status.BLOCKED, session, notice=NOTICE_RETRY_CAP.format(n=MAX_RETRIES))
        if self._expiry(session):
            return Result(Status.STALE, session, notice=self._expiry(session))
        if not len(session.slip):
            return Result(Status.BLOCKED, session, notice=NOTICE_EMPTY)
        return None

    def _reload(self, session: Session, version: int, action: Action, arg: str, override: bool = False) -> Result:
        """Back to LOADING_SLIP for the SAME plays and the SAME stake: the betslip is cleared and reloaded and verified
        exactly as always, nothing is confirmed (summary, token and overrides are gone) and nothing is placed."""
        new = replace(session, state=State.LOADING_SLIP, state_version=version + 1, last=(version, action, arg),
                      issues=(), loaded=None, odds_shown=False, stake=None, summary=None, token_hash="", overrides=(),
                      override_summary="", message="", retryable=False, history_checked=False, unknown_at=None,
                      check_text="", retries_used=session.retries_used + 1, retry_stake=session.stake, retry_override=override,
                      failed_picks=())
        self._set(new)
        return Result(Status.APPLIED, new, arg=arg, load_next=True)

    def _retry(self, session: Session, version: int, arg: str) -> Result:
        """FAILED -> LOADING_SLIP. Only a failure that was certainly before the click carries `retryable`."""
        if not session.retryable:
            return Result(Status.STALE, session)
        refusal = self._retry_refusal(session)
        if refusal is not None:
            return refusal
        return self._reload(session, version, Action.RETRY, arg)

    # ---- AD-26 LOAD_PARTIAL: some picks loaded, some could not be added. Only the owner's buttons move on. ----
    def _act_on_partial(self, session: Session, version: int, action: Action, arg: str) -> Result:
        last = (version, action, arg)
        if action is Action.CANCEL:
            new = replace(session, state=State.CANCELLED, state_version=version + 1, last=last, issues=(), loaded=None,
                          failed_picks=(), odds_shown=False, message=MSG_PARTIAL_CANCELLED)
            self._set(new)
            return Result(Status.APPLIED, new, arg=arg)
        if action is Action.RETRY_PARTIAL:
            refusal = self._retry_refusal(session)
            if refusal is not None:
                return refusal
            # ALL kept picks (loaded and failed) are added again after the betslip is cleared: the slip is verified exactly
            # as always. `failed_picks` stays, so a pick that fails again is shown as failed more than once.
            new = replace(session, state=State.LOADING_SLIP, state_version=version + 1, last=last, issues=(), loaded=None,
                          odds_shown=False, stake=None, summary=None, token_hash="", overrides=(), override_summary="",
                          message="", retries_used=session.retries_used + 1)
            self._set(new)
            return Result(Status.APPLIED, new, arg=arg, load_next=True)
        # GO_AHEAD: the kept list is reduced to the loaded picks. The slip must hold exactly those legs (fail closed).
        if self._expiry(session):
            return Result(Status.STALE, session, notice=self._expiry(session))
        failed_ids = {f.pick_id for f in session.failed_picks}
        remaining = tuple(p for p in session.slip.picks if p.pick_id not in failed_ids)
        loaded = session.loaded
        if loaded is None or not remaining or not failed_ids <= {p.pick_id for p in session.slip.picks}:
            return Result(Status.BLOCKED, session, notice=NOTICE_PARTIAL_INCONSISTENT)
        try:
            wanted = sorted(leg_from_evaluation(p.evaluation).key for p in remaining)
        except UnsupportedLeg:
            return Result(Status.BLOCKED, session, notice=NOTICE_PARTIAL_INCONSISTENT)
        if wanted != sorted(l.leg_key for l in loaded.legs):
            return Result(Status.BLOCKED, session, notice=NOTICE_PARTIAL_INCONSISTENT)
        slip = session.slip
        dropped = list(session.dropped)
        for failure in session.failed_picks:
            pick = slip.get(failure.pick_id)
            slip = slip.delete(failure.pick_id)
            dropped.append(DroppedPick(failure.pick_id, _pick_label(pick), failure.reason))
        new = replace(session, state=State.AWAITING_STAKE, state_version=version + 1, last=last, slip=slip,
                      failed_picks=(), dropped=tuple(dropped), odds_shown=False, stake=None, summary=None,
                      retry_stake=None, retry_override=False)
        self._set(new)
        return Result(Status.APPLIED, new, arg=arg)

    def _act_on_unknown(self, session: Session, version: int, action: Action, arg: str) -> Result:
        if action is Action.BACK:   # Cancel on the warning screen: nothing changes
            return Result(Status.VIEW, session, arg=arg)
        if not session.token_hash:
            return Result(Status.STALE, session)
        if action is Action.CHECK:
            new = replace(session, state=State.RECONCILING, state_version=version + 1, last=(version, action, arg))
            self._set(new)   # RECONCILING is busy: a second tap is a REPEAT, and nothing new can start meanwhile
            return Result(Status.APPLIED, new, arg=arg)
        if not session.history_checked:
            return Result(Status.BLOCKED, session, notice=NOTICE_CHECK_FIRST)
        refusal = self._retry_refusal(session)
        if refusal is not None:
            return refusal
        summary = session.summary
        if summary is None:
            return Result(Status.STALE, session)
        found = self._known_duplicates(summary)
        if isinstance(found, Result):
            return found   # the registry cannot be read: fail closed
        if any(d.proven for d in found):
            return Result(Status.BLOCKED, session, notice=NOTICE_ALREADY_PLACED)
        if action is Action.RETRY_ASK:   # a view: the loud warning
            return Result(Status.VIEW, session, arg=arg, notice="warn")
        if self._audit is None:
            return Result(Status.BLOCKED, session, notice=NOTICE_AUDIT_DOWN)
        try:
            self._audit(session.token_hash, "the owner chose 'Yes, place again' after an unknown result")
        except Exception:
            return Result(Status.BLOCKED, session, notice=NOTICE_AUDIT_DOWN)   # no record, no second attempt
        return self._reload(session, version, Action.RETRY_YES, arg, override=True)

    def apply_retry_stake(self, session_id: str, version: int) -> Result:
        """AWAITING_STAKE -> AWAITING_FINAL_CONFIRM for a retry: the reloaded betslip's odds were shown, and the stake of
        the failed attempt is carried over into a NEW summary. This mints nothing: the owner must press confirm."""
        with self._lock:
            session = self._internal(session_id, version, (State.AWAITING_STAKE,))
            if session is None or session.loaded is None or not session.odds_shown or session.retry_stake is None:
                return Result(Status.STALE, session)
            try:
                stake = parse_stake(str(session.retry_stake), self.max_stake)
            except StakeError as error:
                self._set(replace(session, retry_stake=None))
                return Result(Status.BLOCKED, self._active, notice=str(error))   # the stake is asked for again
            return self._to_summary(session, stake)

    def apply_booking_summary(self, session_id: str, version: int) -> Result:
        """AD-25, book mode only: AWAITING_STAKE -> AWAITING_FINAL_CONFIRM once the real odds were shown. There is no stake
        question: the summary (legs, odds, total odds) is made at once and the owner must still press the one confirmation.
        This mints nothing."""
        with self._lock:
            session = self._internal(session_id, version, (State.AWAITING_STAKE,))
            if not self.book_mode or session is None or session.loaded is None or not session.odds_shown:
                return Result(Status.STALE, session)
            summary = self._new_summary(session, session.loaded, BOOKING_STAKE)
            new = replace(session, state=State.AWAITING_FINAL_CONFIRM, stake=None, summary=summary, retry_stake=None,
                          retry_override=False, overrides=(), override_summary="",
                          state_version=session.state_version + 1, last=None)
            self._set(new)
            return Result(Status.APPLIED, new)

    def _to_summary(self, session: Session, stake: Decimal) -> Result:
        """The new summary for a typed or carried-over stake. After "Yes, place again" the legs the registry holds as
        UNKNOWN (never proven placed ones) are overridden for THIS summary only; the owner still confirms."""
        summary = self._new_summary(session, session.loaded, stake)
        overrides, override_summary = (), ""
        if session.retry_override:
            found = self._known_duplicates(summary)
            if not isinstance(found, Result) and not any(d.proven for d in found):
                overrides, override_summary = tuple(sorted({d.identity for d in found})), summary.summary_id
        new = replace(session, state=State.AWAITING_FINAL_CONFIRM, stake=stake, summary=summary, retry_stake=None,
                      retry_override=False, overrides=overrides, override_summary=override_summary,
                      state_version=session.state_version + 1, last=None)
        self._set(new)
        return Result(Status.APPLIED, new)

    # ---- Story 5.5: the explicit override of a leg that was already placed ----
    def _override(self, session: Session, version: int, summary_id: str) -> Result:
        summary = session.summary
        if summary is None or summary_id != summary.summary_id:
            return Result(Status.STALE, session)
        found = self._known_duplicates(summary)
        if isinstance(found, Result):
            return found
        if not found:
            return Result(Status.STALE, session)  # nothing to override: the confirm button is the way on
        new = replace(session, overrides=tuple(sorted({d.identity for d in found})), override_summary=summary_id,
                      state_version=version + 1, last=(version, Action.OVERRIDE, summary_id))
        self._set(new)
        return Result(Status.APPLIED, new, arg=summary_id, duplicates=tuple(found))

    def _new_summary(self, session: Session, loaded: LoadedSlip, stake: Decimal) -> Summary:
        if self.book_mode:   # no stake exists in booking mode
            return Summary(session.session_id, loaded.legs, loaded.total_odds, BOOKING_STAKE, self._clock() + SUMMARY_TTL,
                           booking=True)
        return Summary(session.session_id, loaded.legs, loaded.total_odds, stake, self._clock() + SUMMARY_TTL)

    # ---- transitions driven by the loading and placement code (not by buttons) ----
    def _internal(self, session_id: str, version: int | None, states: tuple[State, ...]) -> Session | None:
        session = self._active
        if session is None or session.session_id != session_id or session.state not in states:
            return None
        if version is not None and version != session.state_version:
            return None
        return session

    def begin_resolution(self, session_id: str, version: int, issues: Iterable[Issue]) -> Result:
        """LOADING_SLIP -> RESOLVING_AVAILABILITY: some confirmed picks cannot be loaded as they are."""
        with self._lock:
            session = self._internal(session_id, version, (State.LOADING_SLIP,))
            issues = tuple(issues)
            if session is None or not issues:
                return Result(Status.STALE, session)
            new = replace(session, state=State.RESOLVING_AVAILABILITY, issues=issues,
                          state_version=session.state_version + 1, last=None)
            self._set(new)
            return Result(Status.APPLIED, new)

    def mark_partial(self, session_id: str, version: int, loaded: LoadedSlip,
                     failures: Iterable[tuple[str, str]]) -> Result:
        """LOADING_SLIP -> LOAD_PARTIAL (AD-26): `loaded` is the verified betslip of the picks that WERE added, `failures`
        are (pick id, plain reason) of the ones that were not. Nothing is skipped for good: the owner decides."""
        with self._lock:
            session = self._internal(session_id, version, (State.LOADING_SLIP,))
            failures = tuple(failures)
            if session is None or not failures or not loaded.legs:
                return Result(Status.STALE, session)
            before = {f.pick_id: f.times for f in session.failed_picks}
            kept = {p.pick_id for p in session.slip.picks}
            failed = tuple(PickFailure(pid, reason, before.get(pid, 0) + 1) for pid, reason in failures if pid in kept)
            if not failed:
                return Result(Status.STALE, session)
            new = replace(session, state=State.LOAD_PARTIAL, loaded=loaded, failed_picks=failed, odds_shown=False,
                          stake=None, summary=None, state_version=session.state_version + 1, last=None)
            self._set(new)
            return Result(Status.APPLIED, new)

    def mark_loaded(self, session_id: str, version: int, loaded: LoadedSlip) -> Result:
        """LOADING_SLIP -> AWAITING_STAKE with the betslip as SportyBet holds it. The stake is not asked for until
        `mark_odds_shown` (the odds message reached the owner)."""
        with self._lock:
            session = self._internal(session_id, version, (State.LOADING_SLIP,))
            if session is None:
                return Result(Status.STALE, session)
            new = replace(session, state=State.AWAITING_STAKE, loaded=loaded, odds_shown=False, stake=None,
                          summary=None, failed_picks=(), state_version=session.state_version + 1, last=None)
            self._set(new)
            return Result(Status.APPLIED, new)

    def mark_odds_shown(self, session_id: str, version: int) -> Result:
        """The final and total odds were shown: from now on (and never before) a stake may be asked for and taken."""
        with self._lock:
            session = self._internal(session_id, version, (State.AWAITING_STAKE,))
            if session is None or session.loaded is None:
                return Result(Status.STALE, session)
            new = replace(session, odds_shown=True)  # a view: the version does not change
            self._set(new)
            return Result(Status.VIEW, new)

    def submit_stake(self, owner: Owner, text) -> Result:
        """The owner typed a stake. Valid only once the odds were shown, in AWAITING_STAKE (first stake) or
        AWAITING_FINAL_CONFIRM (a changed stake: the old confirmation stops working and a new summary is issued).
        STALE means there is nothing waiting for a stake; BLOCKED carries the reason and asks again."""
        if not isinstance(owner, Owner):
            raise TypeError("session actions need an Owner made by the owner gate")
        if self.book_mode:   # AD-25: nothing in booking mode ever asks for, or accepts, a stake
            return Result(Status.STALE)
        with self._lock:
            active = self._active
            session = None
            if active is not None:
                session = self._internal(active.session_id, None, (State.AWAITING_STAKE, State.AWAITING_FINAL_CONFIRM))
            if session is None:
                return Result(Status.STALE)
            if session.loaded is None or not session.odds_shown:
                return Result(Status.BLOCKED, session, notice=NOTICE_ODDS_FIRST)
            try:
                stake = parse_stake(text, self.max_stake)
            except StakeError as error:
                return Result(Status.BLOCKED, session, notice=str(error))
            return self._to_summary(session, stake)

    def odds_changed(self, session_id: str, loaded: LoadedSlip) -> Result:
        """PLACING -> AWAITING_FINAL_CONFIRM: the betslip's odds moved before the click. The old token is dead
        (PLACING was already left), a new summary at the new odds is issued and needs a NEW confirmation."""
        with self._lock:
            session = self._internal(session_id, None, (State.PLACING,))
            if session is None or (session.stake is None and not self.book_mode):
                return Result(Status.STALE, session)
            new = replace(session, state=State.AWAITING_FINAL_CONFIRM, loaded=loaded, odds_shown=True,
                          summary=self._new_summary(session, loaded, session.stake or BOOKING_STAKE), token_hash="",
                          state_version=session.state_version + 1, last=None)
            self._set(new)
            return Result(Status.APPLIED, new)

    def finish(self, session_id: str, state: State, message: str, version: int | None = None,
               retryable: bool = False) -> Result:
        """Move to a terminal state: DONE, FAILED or PLACEMENT_UNKNOWN. Loading failures come from LOADING_SLIP
        (or RESOLVING_AVAILABILITY), placement outcomes from PLACING. `retryable` (FAILED only) says the failure was
        certainly before the click, so the owner may press Try again; it is never set for anything after the click."""
        if state not in (State.DONE, State.FAILED, State.PLACEMENT_UNKNOWN):
            raise ValueError("not a terminal state")
        with self._lock:
            session = self._internal(session_id, version, (State.LOADING_SLIP, State.RESOLVING_AVAILABILITY,
                                                          State.AWAITING_STAKE, State.AWAITING_FINAL_CONFIRM,
                                                          State.PLACING))
            if session is None:
                return Result(Status.STALE, session)
            new = replace(session, state=state, message=message, state_version=session.state_version + 1, last=None,
                          retryable=retryable and state is State.FAILED, history_checked=False, check_text="",
                          unknown_at=self._clock() if state is State.PLACEMENT_UNKNOWN else None)
            self._set(new)
            return Result(Status.APPLIED, new)

    def resolve(self, session_id: str, state: State, message: str, checked: bool = False,
                since: datetime | None = None, check_text: str = "") -> Result:
        """Story 5.5: a placement whose result was unclear (PLACEMENT_UNKNOWN, or RECONCILING after a restart) has been
        checked against the bet history: move to DONE (the bet was found), or stay/move to PLACEMENT_UNKNOWN or FAILED.
        `checked` (an owner-pressed check that READ the history and found nothing) unlocks "Try again, I checked";
        `since` is when the Place click happened, when known."""
        if state not in (State.DONE, State.FAILED, State.PLACEMENT_UNKNOWN):
            raise ValueError("not a resolved state")
        with self._lock:
            session = self._internal(session_id, None, (State.RECONCILING, State.PLACEMENT_UNKNOWN, State.PLACING))
            if session is None:
                return Result(Status.STALE, session)
            new = replace(session, state=state, message=message, state_version=session.state_version + 1, last=None,
                          retryable=False, history_checked=state is State.PLACEMENT_UNKNOWN
                          and (session.history_checked or checked),
                          unknown_at=(since or session.unknown_at or self._clock())
                          if state is State.PLACEMENT_UNKNOWN else None,
                          check_text=check_text if state is State.PLACEMENT_UNKNOWN else "")
            self._set(new)
            return Result(Status.APPLIED, new)
