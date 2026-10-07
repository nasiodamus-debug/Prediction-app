"""Reading the RESULT of a bet from the bet-level status SportyBet's bet-history LIST page shows (AD-13).

The bookmaker decides every outcome. Nothing here compares a scoreline to a line: it only recognises the bookmaker's own
status word of a whole bet. What the list page really shows (VERIFIED-LIVE 2026-10-01): one status per BET ('Won', 'Lost',
'Running'), a bet type ('Multiple' or 'Singles') and NO per-leg result. So:

  * Singles, 'Won' / 'Lost': the single leg's result is the bet's result.
  * Multiple, 'Won': every leg is recorded as a win, INFERRED from the bet result. This is an approximation: a won
    multiple can contain a void leg (its odds count as 1.00), and that leg would be shown here as a win.
  * Multiple, 'Lost': it is NOT known which leg lost. The bet is recorded as lost and its legs are left UNGRADED (never a
    win, never a loss). Which leg lost is only on the bet's details page (not read by this bot yet).
  * 'Running': pending. Any other word (cashed out, void, cancelled, half won, ...): UNKNOWN, reported, never guessed.
    Void and cancelled wording has not been seen on the page, so it is deliberately NOT recognised.

Standard library only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

WIN, LOSS, VOID, PUSH = "win", "loss", "void", "push"
FINAL_RESULTS = (WIN, LOSS, VOID, PUSH)
PENDING = "pending"     # not decided yet: checked again next time
UNKNOWN = "unknown"     # a status was shown but not understood: reported, never guessed
SOURCE_SPORTYBET = "sportybet"   # the only source a result may have

STATUS_RUNNING = "running"                   # the only status a brand-new bet can show (VERIFIED-LIVE)
_STATUS_WORDS = {"won": WIN, "lost": LOSS, STATUS_RUNNING: PENDING}   # everything else is UNKNOWN (fail closed)
TYPE_SINGLES, TYPE_MULTIPLE = "singles", "multiple"

INFERRED_WIN_TEXT = "won (inferred from the bet result)"
MULTIPLE_LOST_REASON = "multiple lost: which leg lost is only on the bet details page"


def normalise_word(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def classify_bet_status(word: str) -> str:
    """The bet-level status word as WIN, LOSS, PENDING or UNKNOWN. Only the exact words 'Won', 'Lost' and 'Running' are
    understood; anything else is UNKNOWN (never a win or a loss)."""
    return _STATUS_WORDS.get(normalise_word(word), UNKNOWN)


@dataclass(frozen=True)
class EntryGrade:
    """What a list-page entry (type, status, leg count) lets us record.

    bet:      WIN, LOSS, PENDING or UNKNOWN: the bet-level result.
    legs:     WIN when every leg may be recorded as a win (a single leg's own result, or INFERRED for a won multiple),
              LOSS for a lost single, None when the legs must stay UNGRADED.
    inferred: the legs' result is inferred from the bet result (won multiple), not shown per leg.
    note:     why the legs stay ungraded or why the bet is unknown, '' otherwise."""
    bet: str
    legs: str | None = None
    inferred: bool = False
    note: str = ""


def grade_entry(bet_type: str, status: str, leg_count: int | None) -> EntryGrade:
    """Fail closed: anything not understood is UNKNOWN or PENDING, never a win or a loss."""
    result = classify_bet_status(status)
    if result == PENDING:
        return EntryGrade(PENDING)
    if result == UNKNOWN:
        return EntryGrade(UNKNOWN, None, False, f"the status {status!r} is not understood")
    kind = normalise_word(bet_type)
    if kind == TYPE_SINGLES and leg_count == 1:
        return EntryGrade(result, result)   # one selection: the bet's result is that selection's result
    if kind == TYPE_MULTIPLE and leg_count is not None and leg_count >= 2:
        if result == WIN:
            return EntryGrade(WIN, WIN, True, "every leg recorded as won, inferred from the bet result (a won multiple "
                                              "can contain a void leg)")
        return EntryGrade(LOSS, None, False, MULTIPLE_LOST_REASON)
    return EntryGrade(UNKNOWN, None, False, "the bet type or number of games does not fit a single or a multiple")
