"""Results and calibration (Epic 8: Stories 8.1 to 8.3). The flow layer is the only writer (AD-12).

The BOOKMAKER decides every outcome (AD-13): a bet's result is read from SportyBet's own bet history and saved as it is, with
the source 'sportybet'. Nothing here compares a scoreline to a line, so a pick that was evaluated but never placed stays
ungraded, and a dry-run bet (never placed) is never graded. Reading is READ-ONLY toward the bookmaker:
`Bookmaker.bet_history()` (a second page, under the one SportyBet lock, never the betslip).

WHAT THE LIST PAGE GIVES (VERIFIED-LIVE 2026-10-01): one status per BET ('Won', 'Lost', 'Running') and the bet type, no
per-leg result and no bet id (the id is a synthetic key). So, see core/results.py:
  * Singles: Won / Lost is the single leg's result.
  * Multiple Won: every leg is recorded as a win, flagged INFERRED from the bet result (a won multiple can hide a void leg).
  * Multiple Lost: the bet is recorded as lost, its legs stay UNGRADED (which leg lost is only on the bet details page).
  * Running: pending. Any other wording: unknown, reported, left pending (fail closed, AD-14: never retried silently).
Only the first page of the history (the 5 newest bets) is read: a bet that has dropped off it is reported as not found.

AD-25: a bet the OWNER says he placed from a booking code (state PLACED_BY_OWNER, source 'owner-confirmed') has no bet id yet.
It is graded only when the history shows exactly one matching bet (same games, his stated stake, number of legs, not older than
the booking); that entry's key is then saved and the grading is the one above. Without a stake, or with no (or more than one)
matching entry, it is listed as 'not graded' and nothing is invented. A BOOKED row (no bet) is never graded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from decimal import Decimal
from typing import Callable

from predictions.core.betting import (SETTLED_OR_RUNNING, HistoryBet, entry_matches_legs, history_bet_matches,
                                      leg_from_evaluation, normalized_specifier)
from predictions.core.calibration import MIN_BAND_SAMPLE, GradedLeg, summarise
from predictions.core.dates import LAGOS
from predictions.core.model import Estimate, Evaluation
from predictions.core.ports import Bookmaker
from predictions.core.results import (INFERRED_WIN_TEXT, LOSS, MULTIPLE_LOST_REASON, PENDING, PUSH, UNKNOWN, VOID, WIN,
                                      grade_entry)
from predictions.core.types import MatchId
from predictions.flow.reconcile import legs_from_row, parse_created
from predictions.flow.store import P_PLACED_BY_OWNER, EvaluationStore

DAILY_CHECK_TIME = time(9, 0, tzinfo=LAGOS)   # the automatic daily check (Story 8.2): 09:00 Africa/Lagos
DAILY_JOB_NAME = "daily-results-check"

SHOWN = {WIN: "WON", LOSS: "LOST", VOID: "VOID", PUSH: "PUSH"}


@dataclass
class ResultsReport:
    checked: int = 0                                   # placed bets that were still open and looked at
    changes: list[str] = field(default_factory=list)   # one line per bet whose result (or a leg's) was newly saved
    pending: int = 0                                   # placed bets still waiting for SportyBet to settle
    problems: list[str] = field(default_factory=list)  # unreadable wording, not found in history, save failures
    ungraded: int = 0                                  # legs left ungraded in this check (lost multiples)
    error: str = ""                                    # the whole check failed (sanitized): nothing was saved

    @property
    def has_news(self) -> bool:
        return bool(self.changes or self.problems or self.error)

    def lines(self) -> list[str]:
        if self.error:
            return [f"The results check failed: {self.error}. Nothing was changed and it will not be retried "
                    "automatically. Send Results to try again."]
        if not self.checked:
            return ["No placed bet is waiting for a result. (Only bets that were really placed are graded; picks that "
                    "were never placed stay ungraded.)"]
        out = [f"Results check: {self.checked} placed bet(s) looked at, {len(self.changes)} with a new result, "
               f"{self.pending} still pending."]
        out += self.changes
        if self.problems:
            out.append("Needs a look (left pending, nothing guessed):")
            out += [f"- {p}" for p in self.problems]
        return out

    @property
    def text(self) -> str:
        return "\n".join(self.lines())


class ResultsGuard:
    """Only one results check runs at a time. `try_begin` never waits: a second request while one runs is refused, so
    checks (manual or daily) can never stack up or run twice at once."""

    def __init__(self) -> None:
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def try_begin(self) -> bool:
        if self._busy:
            return False
        self._busy = True
        return True

    def end(self) -> None:
        self._busy = False


def _norm_specifier(value: str) -> str:
    return normalized_specifier(value)


def _evaluation_from_row(row: dict) -> Evaluation | None:
    try:
        kickoff = datetime.fromisoformat(row["kickoff_utc"])
        mid = MatchId(row["sport"], row["league"], kickoff, row["home"], row["away"])
        return Evaluation(mid, row["market"], row["selection"], Decimal(row["odds"]),
                          Estimate(float(row["probability"]), (), row["model"], float(row["push"])),
                          Decimal(row["ev"]), bool(row["passed"]), row["reason"])
    except Exception:
        return None


def find_evaluation_id(store: EvaluationStore, leg: dict, placement: dict) -> int | None:
    """The saved evaluation a placed leg was placed on: same game and selection, and (when the bot can rebuild its bet) the
    same market and line. The one from the same session wins, else the latest made before the bet was placed. None when
    it cannot be told (the leg is still graded; it just has no stored estimate for the calibration)."""
    matches = []
    for row in store.evaluation_candidates(leg["match_id"], leg["selection"]):
        if not row["passed"] or row["insufficient"]:
            continue
        ev = _evaluation_from_row(row)
        if ev is None:
            continue
        try:
            built = leg_from_evaluation(ev)
        except Exception:
            continue
        if built.market_id == leg["market_id"] and _norm_specifier(built.specifier) == _norm_specifier(leg["specifier"]):
            matches.append(row)
    if not matches:
        return None
    same = [r for r in matches if r["session_id"] == placement["session_id"]]
    pool = same or [r for r in matches if r["evaluated_at_utc"] <= leg["placed_at_utc"]] or matches
    return pool[-1]["id"]


def _describe(placed: dict, result: str) -> str:
    return f"{placed['match']} {placed['outcome']} = {SHOWN[result]}"


async def check_results(store: EvaluationStore, bookmaker: Bookmaker, sanitize: Callable[[object], str],
                        now: datetime | None = None) -> ResultsReport:
    """Read the bet history once and save every FINAL result it shows for the owner's placed, still-open bets. Pending
    games are left alone (checked again next time); unknown wording is reported, never guessed."""
    report = ResultsReport()
    try:
        open_bets = store.open_placements_for_results()
    except Exception as error:
        report.error = sanitize(error)
        return report
    report.checked = len(open_bets)
    if not open_bets:
        return report
    try:
        history = await bookmaker.bet_history()
    except Exception as error:   # an unreadable history is reported, never "no result", never retried silently
        report.error = sanitize(error)
        report.checked = 0
        return report
    by_id = {bet.bet_id: bet for bet in history if bet.bet_id}
    for row in open_bets:
        if row["state"] == P_PLACED_BY_OWNER and not row["bet_id"]:
            if not _link_owner_bet(store, row, history, open_bets, report, sanitize, now):
                continue
        short = row["bet_id"]
        try:
            _check_one(store, row, by_id.get(row["bet_id"]), report, now)
        except Exception as error:
            report.problems.append(f"Bet {short}: could not be saved ({sanitize(error)}).")
    return report


def _link_owner_bet(store: EvaluationStore, row: dict, history, open_bets, report: ResultsReport,
                    sanitize: Callable[[object], str], now: datetime | None) -> bool:
    """An owner-confirmed bet has no bet id: find ITS entry in the history (the same rule as for any placed bet: games, stake,
    number of legs, status, not older than the booking) and remember the entry's key. True when the row now has its bet id."""
    code = row["booking_code"] or "?"

    def not_graded(why: str) -> bool:
        report.pending += 1
        report.problems.append(f"Bet you placed from booking code {code}: not graded: {why} Left pending.")
        return False

    try:
        legs = legs_from_row(row)
        stake = Decimal(row["owner_stake"]) if row.get("owner_stake") else None
    except Exception:
        return not_graded("the saved record could not be read.")
    if stake is None:
        return not_graded("you did not give the stake, so it cannot be matched to the bet history.")
    taken = {r["bet_id"] for r in store.list_placements() if r["bet_id"] and r["id"] != row["id"]}
    found = [b for b in history if b.bet_id and b.bet_id not in taken
             and history_bet_matches(b, legs, stake, parse_created(row["created_at_utc"]), SETTLED_OR_RUNNING)]
    if not found:
        return not_graded("not found in the bet history (only the 5 newest bets are read, so it may be older, or you did "
                          "not place it).")
    if len(found) > 1:
        return not_graded("more than one history entry looks like it, so it cannot be told which is yours.")
    try:
        store.link_owner_bet(row["id"], found[0].bet_id, now)
    except Exception as error:
        return not_graded(f"its history entry could not be saved ({sanitize(error)}).")
    row["bet_id"] = found[0].bet_id
    return True


def _check_one(store: EvaluationStore, row: dict, bet: HistoryBet | None, report: ResultsReport,
               now: datetime | None) -> None:
    bet_id = row["bet_id"]

    def leave_pending(why: str) -> None:
        report.pending += 1
        report.problems.append(f"Bet {bet_id}: {why} Left pending.")

    if bet is None:
        return leave_pending("not found on the first page of the bet history (only the 5 newest bets are read, so it may "
                             "be older, or the history may lag).")
    if bet.ambiguous:
        return leave_pending("another history entry has the very same time, stake and games, so the two cannot be told "
                             "apart.")
    try:
        legs, stake = legs_from_row(row), Decimal(row["stake"])
    except Exception:
        return leave_pending("the saved placement could not be read.")
    if not entry_matches_legs(bet, legs, stake):
        return leave_pending("its history entry no longer matches the games, stake or number of legs that were placed.")
    grade = grade_entry(bet.bet_type, bet.status, bet.leg_count)
    if grade.bet == PENDING:
        report.pending += 1
        return
    if grade.bet == UNKNOWN:
        return leave_pending(f"SportyBet shows '{bet.status}' ({grade.note}); I do not guess a result.")
    placed = row["placed_legs"]
    if grade.legs is not None and len(placed) != len(legs):
        return leave_pending("the games recorded as placed do not match the bet; nothing was graded.")
    try:
        json_legs = {tuple(item["leg_key"]): item for item in json.loads(row["legs"])}
    except Exception:
        json_legs = {}
    saved_legs, lines = [], []
    if grade.legs is not None:
        raw = INFERRED_WIN_TEXT if grade.inferred else bet.status
        for leg in placed:
            if leg["decided"]:
                continue
            info = json_legs.get((leg["match_id"], leg["market_id"], leg["specifier"], leg["side"]))
            if info is None:
                return leave_pending("a placed game could not be matched to the saved bet; nothing was graded.")
            saved_legs.append((leg["id"], find_evaluation_id(store, leg, row), grade.legs, raw))
            lines.append(_describe(info, grade.legs))
    store.record_results(row["id"], grade.bet, saved_legs, raw_bet=bet.status, now=now)
    head = f"Bet {bet_id}: {SHOWN[grade.bet]}"
    if grade.legs is None:
        undecided = sum(1 for leg in placed if not leg["decided"])
        report.ungraded += undecided
        head += f" ({MULTIPLE_LOST_REASON}; {undecided} leg(s) left ungraded)"
    elif grade.inferred:
        head += " (" + "; ".join(lines) + "; legs inferred from the bet result)"
    elif lines:
        head += " (" + "; ".join(lines) + ")"
    report.changes.append(head)


CALIBRATION_NOTES = ("Only bets that were really placed and that SportyBet has graded are counted. SportyBet's list shows "
                     "only the result of a whole bet: the legs of a lost multiple are left ungraded (which leg lost is only "
                     "on the bet details page) and the legs of a won multiple are counted as wins, inferred from the bet. "
                     "Voids and pushes are listed apart: they are neither a win nor a loss, and the win rate is wins / "
                     "(wins + losses). For push-protected picks (a whole line where an exact result refunds the stake) the "
                     "stored estimate counts a refund as success, so a pushed leg shows as a push, not a win, and their win "
                     "rate can sit below the estimate.")


def calibration_lines(store: EvaluationStore) -> list[str]:
    rows = store.graded_legs()
    ungraded = store.ungraded_legs_of_settled_bets()
    ungraded_line = (f"{ungraded} leg(s) of lost multiples are ungraded (which leg lost is only on the bet details page); "
                     "they are never counted as wins or losses." if ungraded else "")
    if not rows:
        out = ["No placed bet has a graded leg yet, so there is nothing to calibrate. Send Results after your placed bets "
               "have finished."]
        if ungraded_line:
            out.append(ungraded_line)
        return out
    linked = [GradedLeg(float(r["probability"]), r["result"]) for r in rows if r["probability"] is not None]
    unlinked = len(rows) - len(linked)
    out = [f"Calibration: {len(rows)} graded leg(s) from placed bets, by the model's stored estimate."]
    for band in summarise(linked):
        rate = "n/a" if band.win_rate is None else f"{band.win_rate * 100:.0f}%"
        line = (f"{band.label}: {band.legs} leg(s): {band.wins} won, {band.losses} lost, {band.voids} void, "
                f"{band.pushes} push. Win rate {rate} (estimated {band.average_estimate * 100:.0f}% on average).")
        if band.too_small:
            line += f" Only {band.decided} decided leg(s) (under {MIN_BAND_SAMPLE}): too small a sample to judge."
        out.append(line)
    if unlinked:
        out.append(f"{unlinked} graded leg(s) could not be linked to a saved estimate and are left out of the bands.")
    if not linked:
        out.append("None of the graded legs has a stored estimate, so there are no bands to show.")
    if ungraded_line:
        out.append(ungraded_line)
    out.append(CALIBRATION_NOTES)
    return out
