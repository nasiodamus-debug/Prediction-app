"""Loading the confirmed games onto SportyBet and handling what is missing (Stories 5.1 and 5.2).

`run_load` is called once each time the session enters LOADING_SLIP (games confirmed, or every open issue resolved):

1. Ask the bookmaker which confirmed picks are available. Nothing is touched yet.
2. If any game or option is missing, move to RESOLVING_AVAILABILITY with one `Issue` per pick and STOP. Nothing is
   skipped or swapped on its own (AD-14): the owner presses Skip or Accept. For a missing option the closest available
   option whose own estimate still clears the floor is offered with that estimate; when none clears it, the only
   choice offered is to skip. A below-floor option is never offered.
3. If everything is available, have the bookmaker clear the betslip, add exactly these legs and verify it. The result is
   checked AGAIN here against the confirmed legs (defence in depth) before the session moves to AWAITING_STAKE.

4. (AD-26) A pick that cannot be added (option not found, locked or suspended, did not appear after the one retry-click, or a
   page timeout) is SKIPPED by the bookmaker and the others are still loaded. When some picks loaded and some did not, the
   session goes to LOAD_PARTIAL and the owner decides (Go ahead with the loaded, Retry the failed, Cancel): nothing is dropped on
   its own. When NO pick loaded, it is the plain failure below. A failure that is not about one pick (login, the betslip cannot
   be emptied, a browser crash, the overall timeout, a betslip holding anything else) still fails the whole load.

Any failure ends in FAILED with a sanitized message. Nothing is retried automatically, and nothing here can place a
bet. A failure at the bookmaker is `retryable`: loading never clicks Place, so no bet can exist and the owner may press
Try again (the state machine reloads the same plays; the safety-net refusals below are not retryable).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from predictions.core.betting import (Availability, Leg, LegAvailability, LoadAttempt, LoadedSlip, UnsupportedLeg,
                                      leg_from_evaluation, total_is_consistent)
from predictions.core.engines import ENGINES
from predictions.core.market_rules import UNDER_REASON, WHOLE_LINE_REASON, is_under, is_whole_or_quarter_line
from predictions.core.model import Evaluation
from predictions.core.options import closest_option
from predictions.core.ports import Bookmaker, BookmakerError
from predictions.core.slip import Pick
from predictions.flow.session import ISSUE_GAME, ISSUE_OPTION, Issue, Session, SessionManager, State

KIND_ISSUES = "issues"     # the owner must decide (skip / accept)
KIND_LOADED = "loaded"     # the betslip holds exactly the confirmed games; the odds are ready to show
KIND_PARTIAL = "partial"   # some picks loaded and some could not be added: the owner decides (AD-26)
KIND_FAILED = "failed"     # reported; nothing placed; never retried automatically
KIND_STALE = "stale"       # the session moved on (a newer list or a repeated trigger): do nothing

MSG_UNSUPPORTED = "This kind of selection cannot be put on the SportyBet betslip by this bot."


@dataclass(frozen=True)
class LoadOutcome:
    kind: str
    session: Session | None = None
    message: str = ""


def alternative_for(pick: Pick, found: LegAvailability) -> Evaluation | None:
    """The closest option worth offering for a pick whose option is missing: judged by the sport's own engine with the
    pick's own inputs, against the game's live listing. None when nothing clears the floor (then: skip)."""
    engine = ENGINES.get(pick.match_id.sport)
    if engine is None or found.fixture is None or is_under(pick) or is_under(found.leg):
        return None
    if is_whole_or_quarter_line(pick) or is_whole_or_quarter_line(found.leg):   # AD-21: no whole-line alternative either
        return None
    candidates: list[Evaluation] = []
    for offer in found.fixture.markets:
        if offer.market_id != found.leg.market_id:
            continue
        try:
            candidates.extend(engine.evaluate_exact(found.fixture, pick.evaluation.estimate.inputs, offer))
        except Exception:  # an engine that cannot price this offer simply offers nothing
            continue
    return closest_option(found.leg, candidates)


def build_issues(picks: list[Pick], legs: dict[str, Leg], unsupported: dict[str, str],
                 checked: dict[str, LegAvailability]) -> list[Issue]:
    issues: list[Issue] = []
    for pick in picks:
        if pick.pick_id in unsupported:
            issues.append(Issue(pick.pick_id, ISSUE_GAME, unsupported[pick.pick_id]))
            continue
        found = checked[pick.pick_id]
        if found.status is Availability.GAME_UNAVAILABLE:
            issues.append(Issue(pick.pick_id, ISSUE_GAME, found.detail))
        elif found.status is Availability.OPTION_MISSING:
            issues.append(Issue(pick.pick_id, ISSUE_OPTION, found.detail, alternative_for(pick, found)))
    return issues


def loaded_matches(legs: list[Leg], loaded: LoadedSlip) -> bool:
    """The loaded slip holds exactly the confirmed legs: same set of leg keys, each once, and a sane total."""
    wanted = sorted(leg.key for leg in legs)
    got = sorted(leg.leg_key for leg in loaded.legs)
    return bool(legs) and wanted == got and total_is_consistent(loaded.legs, loaded.total_odds)


async def run_load(manager: SessionManager, bookmaker: Bookmaker, session_id: str, version: int,
                   sanitize: Callable[[object], str]) -> LoadOutcome:
    session = manager.active
    if (session is None or session.session_id != session_id or session.state_version != version
            or session.state is not State.LOADING_SLIP):
        return LoadOutcome(KIND_STALE, session)
    picks = sorted(session.slip.picks, key=lambda p: (p.match_id.kickoff_utc, p.key))

    def fail(error, retryable: bool = True) -> LoadOutcome:
        text = (f"Loading your games onto SportyBet failed: {sanitize(error)}. "
                "Nothing was placed and nothing will be retried automatically.")
        result = manager.finish(session_id, State.FAILED, text, version, retryable=retryable)
        return LoadOutcome(KIND_FAILED if result.session else KIND_STALE, result.session, text)

    legs: dict[str, Leg] = {}
    unsupported: dict[str, str] = {}
    for pick in picks:
        if is_under(pick):   # AD-20: the last safety net, before anything is asked of SportyBet
            return fail(f"{UNDER_REASON} ({pick.evaluation.selection})", False)
        if is_whole_or_quarter_line(pick):   # AD-21: the same safety net for a whole or quarter line
            return fail(f"{WHOLE_LINE_REASON} ({pick.evaluation.selection})", False)
        try:
            legs[pick.pick_id] = leg_from_evaluation(pick.evaluation)
        except UnsupportedLeg:
            unsupported[pick.pick_id] = MSG_UNSUPPORTED
        else:
            if is_under(legs[pick.pick_id]):
                return fail(f"{UNDER_REASON} ({pick.evaluation.selection})", False)
            if is_whole_or_quarter_line(legs[pick.pick_id]):
                return fail(f"{WHOLE_LINE_REASON} ({pick.evaluation.selection})", False)
    try:
        supported = [legs[p.pick_id] for p in picks if p.pick_id in legs]
        results = await bookmaker.check_availability(supported) if supported else ()
    except BookmakerError as error:
        return fail(error)
    if len(results) != len(supported):
        return fail("SportyBet's answer did not cover every game")
    by_pick = {p.pick_id: r for p, r in zip([p for p in picks if p.pick_id in legs], results)}
    if any(by_pick[pid].leg != legs[pid] for pid in by_pick):
        return fail("SportyBet's answer did not match the confirmed games")

    issues = build_issues(picks, legs, unsupported, by_pick)
    if issues:
        result = manager.begin_resolution(session_id, version, issues)
        if result.session is None or result.session.state is not State.RESOLVING_AVAILABILITY:
            return LoadOutcome(KIND_STALE, result.session)
        return LoadOutcome(KIND_ISSUES, result.session)

    ordered = [legs[p.pick_id] for p in picks]
    try:
        load_legs = getattr(bookmaker, "load_legs", None)
        attempt = await load_legs(ordered) if load_legs is not None else LoadAttempt(await bookmaker.load_slip(ordered))
    except BookmakerError as error:
        return fail(error)
    if not isinstance(attempt, LoadAttempt):
        return fail("SportyBet's answer to the load was not understood")
    if attempt.failed:
        return _partial(manager, session_id, version, picks, ordered, attempt, sanitize, fail)
    loaded = attempt.loaded
    if loaded is None or not loaded_matches(ordered, loaded):
        return fail("the betslip SportyBet holds is not exactly the games you confirmed")
    result = manager.mark_loaded(session_id, version, loaded)
    if result.session is None or result.session.state is not State.AWAITING_STAKE:
        return LoadOutcome(KIND_STALE, result.session)
    return LoadOutcome(KIND_LOADED, result.session)


REASON_LIMIT = 300


def _partial(manager, session_id, version, picks, ordered, attempt: LoadAttempt, sanitize, fail) -> LoadOutcome:
    """Some picks could not be added (AD-26). Everything is checked again here: the failed legs must be some of the confirmed
    legs, each once, and the betslip must hold EXACTLY the other legs. Anything else fails the whole load (fail closed)."""
    by_key = {leg.key: pick for pick, leg in zip(picks, ordered)}
    failed_keys = [f.leg_key for f in attempt.failed]
    if len(set(failed_keys)) != len(failed_keys) or any(k not in by_key for k in failed_keys):
        return fail("SportyBet's answer did not match the confirmed games")
    failures = [(by_key[f.leg_key].pick_id, " ".join(str(sanitize(f.reason)).split())[:REASON_LIMIT]) for f in attempt.failed]
    if attempt.loaded is None:
        lines = "; ".join(f"{by_key[f.leg_key].match_id.home} vs {by_key[f.leg_key].match_id.away}: "
                          f"{by_key[f.leg_key].evaluation.selection} ({reason})"
                          for f, (_, reason) in zip(attempt.failed, failures))
        return fail(f"none of your picks could be added to the betslip ({lines})")
    rest = [leg for leg in ordered if leg.key not in set(failed_keys)]
    if not loaded_matches(rest, attempt.loaded):
        return fail("the betslip SportyBet holds is not exactly the games that loaded")
    result = manager.mark_partial(session_id, version, attempt.loaded, failures)
    if result.session is None or result.session.state is not State.LOAD_PARTIAL:
        return LoadOutcome(KIND_STALE, result.session)
    return LoadOutcome(KIND_PARTIAL, result.session)
