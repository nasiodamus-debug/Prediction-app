"""The final list, and everything after it, as Telegram text and buttons (Stories 2.6 and 5.1 to 5.4). Plain text only.

Buttons carry `s:<session_id>:<state_version>:<action>:<arg>` where `arg` is a stable pick id (or a page
number for paging), never a list index (AD-19). The whole string stays under Telegram's 64-byte limit.
Nothing here imports the telegram library: `Screen` is text plus rows of (label, callback data), and app.py
turns it into a message.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from predictions.core.dates import LAGOS
from predictions.core.engines.soccer import GOALS_ONLY_LABEL, GOALS_ONLY_MODEL_NAME
from predictions.core.model import PROBABILITY_FLOOR, Estimate, percent
from predictions.core.play_labels import play_label
from predictions.core.betting import leg_identity
from predictions.core.placement import DuplicateLeg, PlacementToken
from predictions.core.slip import Game, Pick
from predictions.core.staking import (MAX_UNITS, MIN_UNITS, format_money, potential_profit, potential_winnings,
                                      suggest_stake)
from predictions.core.types import Owner
from predictions.flow.session import (ISSUE_OPTION, MAX_RETRIES, Action, Issue, Pending, Result, Session,
                                      SessionManager, State, Status)
from predictions.telegram.render import MESSAGE_LIMIT, clean_text

ORIGIN_NAMES = {"matchday": "Matchday", "gameday": "Gameday", "america": "America", "combined": "a combined request"}  # Session.origin of a slate that came from a trigger
REVIEW_ORIGIN = "review:"  # Session.origin of a slate that came from `Review <code>` (Story 5.6)
CALLBACK_PREFIX = "s"
CALLBACK_LIMIT = 64  # bytes, Telegram's limit for callback data
PAGE_SIZE = 8
_DATA = re.compile(r"^s:([0-9a-f]{8}):(\d{1,9}):([a-z]):([0-9a-f]{0,12})$")

NOTICE_STALE = "This list changed. Please use the newest list."
NOTICE_REPEAT = "Already done."
NOTICE_GONE = "That pick is no longer in the list."
NOTICE_UNKNOWN = "That button is not valid."
LOADING_NOTE = ("Loading them onto SportyBet now: I clear your betslip first, add exactly these games and check it. "
                "No bet is placed at this step.")


class OddsNotShown(RuntimeError):
    """A stake was about to be asked for before the final and total odds were shown. It never is (Story 5.3)."""


@dataclass(frozen=True)
class Job:
    """Work app.py must do after showing a reply: load the betslip, or place with a freshly minted token."""
    kind: str                       # "load", "go" (AD-26: carry on with the loaded picks), "place" (a freshly minted token) or "check"
    session_id: str
    version: int
    token: PlacementToken | None = None


def encode(session_id: str, version: int, action: Action, arg: str = "") -> str:
    data = f"{CALLBACK_PREFIX}:{session_id}:{version}:{action.value}:{arg}"
    if len(data.encode("utf-8")) > CALLBACK_LIMIT:
        raise ValueError("callback data is longer than Telegram allows")
    return data


def decode(data) -> tuple[str, int, Action, str] | None:
    match = _DATA.match(data) if isinstance(data, str) else None
    if match is None:
        return None
    try:
        action = Action(match.group(3))
    except ValueError:
        return None
    return match.group(1), int(match.group(2)), action, match.group(4)


@dataclass(frozen=True)
class Screen:
    text: str
    buttons: tuple[tuple[tuple[str, str], ...], ...] = ()  # rows of (label, callback data)


@dataclass(frozen=True)
class Reply:
    answer: str = ""          # the short notice shown when the button is pressed
    alert: bool = False
    screen: Screen | None = None  # the message to show instead of the current one (None: leave it alone)
    job: Job | None = None        # what to do next (load the betslip / place), after the screen is shown


# ---------- rendering ----------

def _ordered(session: Session) -> list[Pick]:
    """The KEPT chosen plays only (the Slip), in list order."""
    return sorted(session.slip.picks, key=lambda p: (p.match_id.kickoff_utc, p.key))


def _who(pick: Pick) -> str:
    mid = pick.match_id
    local = mid.kickoff_utc.astimezone(LAGOS)
    return f"{local.day} {local.strftime('%b %H:%M')} {clean_text(mid.home)} vs {clean_text(mid.away)}"


def _success(pick: Pick) -> str:
    est = pick.evaluation.estimate
    text = f"success {percent(est.probability)}"
    if est.push:
        text += f" (win {percent(est.win)} + refund {percent(est.push)})"
    if est.model == GOALS_ONLY_MODEL_NAME:
        text += f" [{GOALS_ONLY_LABEL}]"  # Story 9.3: no xG behind this estimate
    return text


MARK_KEPT = "\u2713"      # check mark: this play is the kept play of its game
MARK_REMOVED = "\u2717"   # cross: this play is not kept (a game with no kept play is out of the slip)
MAX_PLAY_ROWS_PER_PAGE = 40  # two buttons per play row: with Previous/Next and Final confirm this stays under Telegram's 100


def _tag(pick: Pick) -> str:
    """The short labels the owner needs to stay safe: a goals-only estimate and a shielded or converted play."""
    ev, text = pick.evaluation, ""
    if ev.estimate.model == GOALS_ONLY_MODEL_NAME:
        text += " [goals-only]"
    if ev.rule:
        text += f" ({clean_text(re.split(r'[:;]', ev.rule)[0].strip(), 24)})"
    return text


def _play(pick: Pick) -> str:
    """One play, compact: 'Over 2.5 @ 1.85 - 72.0%' plus its safety tags."""
    ev = pick.evaluation
    return f"{clean_text(play_label(ev))} @ {ev.odds:.2f} - {percent(ev.estimate.probability)}{_tag(pick)}"


def _letter(index: int) -> str:
    return chr(ord("a") + index)


def _play_label(game: Game, number: int, index: int) -> str:
    """'1' for a game with one play; '1a', '1b' ... for a game with several."""
    return str(number) if len(game.options) == 1 else f"{number}{_letter(index)}"


def _game_lines(game: Game, number: int) -> list[str]:
    head = f"{number}) {MARK_KEPT if game.kept else MARK_REMOVED} {_who(game.advised)}" + ("" if game.kept else " (removed)")
    lines = [head]
    for index, option in enumerate(game.options):
        mark = MARK_KEPT if game.kept and option.pick_id == game.chosen_id else MARK_REMOVED
        advised = " (advised)" if index == 0 and len(game.options) > 1 else ""
        label = "" if len(game.options) == 1 else f"{_play_label(game, number, index)} "
        lines.append(f"   {label}{mark} {_play(option)}{advised}")
    return lines


LIST_TEXT_BUDGET = 3000  # characters of game lines per page, so a page of options never passes Telegram's limit


def _pages(session: Session) -> list[list[Game]]:
    """The games split into pages: at most PAGE_SIZE games, few enough characters to fit one message, and few enough
    play rows (two buttons each) to keep the keyboard within Telegram's button limit."""
    pages: list[list[Game]] = []
    current: list[Game] = []
    used = rows = 0
    for game in session.games():
        size = sum(len(line) + 1 for line in _game_lines(game, 99))
        if current and (len(current) >= PAGE_SIZE or used + size > LIST_TEXT_BUDGET
                        or rows + len(game.options) > MAX_PLAY_ROWS_PER_PAGE):
            pages.append(current)
            current, used, rows = [], 0, 0
        current.append(game)
        used += size
        rows += len(game.options)
    if current:
        pages.append(current)
    return pages or [[]]


def page_count(session: Session) -> int:
    return len(_pages(session))


def _page_of(session: Session, pick_id: str) -> int:
    for number, page in enumerate(_pages(session)):
        if any(p.pick_id == pick_id for g in page for p in g.options):
            return number
    return 0


def list_screen(session: Session, page: int = 0, notice: str = "") -> Screen:
    games = session.games()
    if not games:
        return Screen("The list is empty: there is nothing to confirm. Send a new request to start again.")
    kept = len(session.slip)
    paged = _pages(session)
    pages = len(paged)
    page = min(max(page, 0), pages - 1)
    count = f"Kept {kept} of {len(games)} game(s)"
    if session.origin.startswith(REVIEW_ORIGIN):
        code = clean_text(session.origin[len(REVIEW_ORIGIN):], 30)
        lines = [f"Approved selections from Review {code}: {count}. Disqualified selections are not here and cannot be "
                 "added. I load what you keep fresh onto SportyBet (never the code's own slip) and re-read the odds."]
    else:
        lines = [f"Final list: {count}."]
    if not kept:
        lines.append("No play is kept. Press Keep on at least one play, then Final confirm.")
    else:
        lines.append("Each play has its own Keep and Delete. A game keeps one play at most: Keep on another play "
                     "replaces it, Delete removes the game (Keep brings it back). The advised play is kept first. "
                     "Press Final confirm when finished.")
    if notice:
        lines += ["", clean_text(notice, 200)]
    sid, version = session.session_id, session.state_version
    rows: list[tuple[tuple[str, str], ...]] = []
    start = sum(len(p) for p in paged[:page])
    lines.append("")
    for offset, game in enumerate(paged[page]):
        number = start + offset + 1
        lines += _game_lines(game, number)
        for index, option in enumerate(game.options):
            label = _play_label(game, number, index)
            rows.append(((f"{label} Keep", encode(sid, version, Action.RESTORE, option.pick_id)),
                         (f"{label} Delete", encode(sid, version, Action.DELETE, option.pick_id))))
    if pages > 1:
        lines.append(f"Page {page + 1} of {pages}")
        nav = []
        if page > 0:
            nav.append(("Previous", encode(sid, version, Action.PAGE, str(page - 1))))
        if page < pages - 1:
            nav.append(("Next", encode(sid, version, Action.PAGE, str(page + 1))))
        rows.append(tuple(nav))
    rows.append((("Final confirm", encode(sid, version, Action.DONE)),))
    return Screen(_fit("\n".join(lines).rstrip()), tuple(rows))


def final_screen(session: Session, notice: str = "") -> Screen:
    """The final edit screen: only the kept games and their chosen plays, compact, with Confirm and Back to adjust."""
    games = session.games()
    kept = [g for g in games if g.kept]
    removed = len(games) - len(kept)
    sid, version = session.session_id, session.state_version
    head = [f"Final edit: {len(kept)} game(s) kept" + (f", {removed} removed." if removed else ".")]
    if notice:
        head += ["", clean_text(notice, 200)]
    head.append("")
    tail = ["", "Confirm games to load them onto SportyBet (nothing is bet yet), or go back to adjust."]
    lines, used = list(head), len("\n".join(head + tail))
    shown = 0
    for number, game in enumerate(kept, start=1):
        line = f"{number}) {_who(game.advised)}: {_play(game.chosen)}"
        reserve = len(f"\nand {len(kept) - number} more") if len(kept) > number else 0
        if used + 1 + len(line) + reserve > MESSAGE_LIMIT - 1:
            break
        lines.append(line)
        used += 1 + len(line)
        shown += 1
    if shown < len(kept):
        lines.append(f"and {len(kept) - shown} more")
    rows = ((("Confirm games", encode(sid, version, Action.CONFIRM)),),
            (("Back to adjust", encode(sid, version, Action.ADJUST)),))
    return Screen("\n".join(lines + tail), rows)


def _compact_line(pick: Pick, number: int) -> str:
    ev = pick.evaluation
    return f"{number}. {_who(pick)}: {clean_text(play_label(ev))} @ {ev.odds:.2f}, {_success(pick)}"


def confirmed_screen(session: Session) -> Screen:
    """The notice lines come first so they can never be cut; picks follow, one line each, and a long list
    ends with "and N more" instead of being cut mid-word."""
    picks = _ordered(session)
    lines = [f"Games confirmed: {len(picks)} pick(s) on {session.slip.matches} match(es).", "", LOADING_NOTE, ""]
    if session.retries_used and session.failed_picks:
        lines[2:2] = [f"Retry {session.retries_used} of {MAX_RETRIES}: the betslip is cleared and ALL {len(picks)} picks are "
                      "added again. Nothing is skipped without your choice.", ""]
    elif session.retries_used:
        lines[2:2] = [f"Retry {session.retries_used} of {MAX_RETRIES}: the same games and the same stake. You confirm "
                      "again on a new summary before anything is placed.", ""]
    used = len("\n".join(lines))
    shown = 0
    for number, pick in enumerate(picks, start=1):
        line = _compact_line(pick, number)
        remaining = len(picks) - number
        reserve = len(f"\nand {remaining} more") if remaining else 0
        if used + 1 + len(line) + reserve > MESSAGE_LIMIT - 1:
            break
        lines.append(line)
        used += 1 + len(line)
        shown += 1
    if shown < len(picks):
        lines.append(f"and {len(picks) - shown} more")
    return Screen("\n".join(lines))


def _fit(text: str) -> str:
    return text if len(text) <= MESSAGE_LIMIT else text[: MESSAGE_LIMIT - 1] + "…"


# ---------- Story 5.2: a missing game or option ----------

def _estimate_text(estimate: Estimate) -> str:
    text = f"success {percent(estimate.probability)}"
    if estimate.push:
        text += f" (win {percent(estimate.win)} + refund {percent(estimate.push)})"
    return text


def _offered(issue: Issue) -> bool:
    """Only an option whose own estimate clears the floor is ever shown as an offer (checked again here)."""
    alt = issue.alternative
    return alt is not None and alt.passed and not alt.insufficient and alt.estimate.probability >= PROBABILITY_FLOOR


def issues_screen(session: Session, notice: str = "") -> Screen:
    lines = [f"{len(session.issues)} game(s) cannot be loaded as they are. Nothing has been skipped or changed: "
             "you decide for each one."]
    if notice:
        lines += ["", clean_text(notice, 200)]
    lines.append("")
    rows: list[tuple[tuple[str, str], ...]] = []
    for number, issue in enumerate(session.issues, start=1):
        try:
            pick = session.slip.get(issue.pick_id)
        except Exception:
            continue
        ev = pick.evaluation
        lines.append(f"{number}. {_who(pick)}: {clean_text(play_label(ev))} @ {ev.odds:.2f}")
        buttons = []
        if issue.kind == ISSUE_OPTION:
            lines.append("   The game is on SportyBet but this option is not offered.")
            if _offered(issue):
                alt = issue.alternative
                lines.append(f"   Closest option that still clears the {PROBABILITY_FLOOR:.0%} floor: "
                             f"{clean_text(play_label(alt))} @ {alt.odds:.2f}, {_estimate_text(alt.estimate)}.")
                lines.append("   Accept it, or skip this game.")
                buttons.append((f"Accept {number}", encode(session.session_id, session.state_version, Action.ACCEPT,
                                                         issue.pick_id)))
            else:
                lines.append(f"   No available option clears the {PROBABILITY_FLOOR:.0%} floor. Please skip this game.")
        else:
            lines.append(f"   {clean_text(issue.detail or 'This game is not available on SportyBet')}. "
                         "It cannot be loaded. Please skip this game.")
        buttons.append((f"Skip {number}", encode(session.session_id, session.state_version, Action.SKIP, issue.pick_id)))
        rows.append(tuple(buttons))
    return Screen(_fit("\n".join(lines)), tuple(rows))


# ---------- Story 5.3: final odds, then the stake ----------

def odds_screen(session: Session) -> Screen:
    """The final odds of each game and the total odds. It never asks for a stake (that is `stake_prompt`)."""
    loaded = session.loaded
    if loaded is None:
        raise OddsNotShown("there is no loaded betslip to show")
    lines = [f"Loaded on SportyBet: {len(loaded.legs)} game(s). Final odds:", ""]
    for number, leg in enumerate(loaded.legs, start=1):
        lines.append(f"{number}. {clean_text(leg.match_text)}: {clean_text(leg.selection or leg.outcome_name)} "
                     f"@ {leg.odds:.2f}")
        if leg.evaluation_odds is not None and leg.evaluation_odds != leg.odds:
            lines.append(f"   evaluation used {leg.evaluation_odds:.2f}, SportyBet now shows {leg.odds:.2f}")
    lines += left_out_lines(session)
    lines += ["", f"Total odds: {loaded.total_odds:.2f}", "", "No bet has been placed."]
    return Screen(_fit("\n".join(lines)))


def left_out_lines(session: Session) -> list[str]:
    """AD-26: the picks the owner went ahead WITHOUT, listed plainly (never silent). Empty when none was left out."""
    if not session.dropped:
        return []
    lines = ["", f"Left out ({len(session.dropped)}): not on your slip and not in this bet. You chose to go ahead without:"]
    for drop in session.dropped:
        lines.append(f"- {clean_text(drop.label, 120)}: {clean_text(drop.reason, 160)}")
    return lines


def stake_prompt(session: Session) -> Screen:
    """Asks for the stake. Refuses (OddsNotShown) unless the odds were already shown to the owner."""
    if session.state is not State.AWAITING_STAKE or session.loaded is None or not session.odds_shown:
        raise OddsNotShown("the stake is asked for only after the final and total odds were shown")
    suggestion = suggest_stake([p.evaluation.estimate.win for p in session.slip.picks], session.loaded.total_odds)
    lines = []
    if session.stake is not None:   # reached with Change amount: the odds and the betslip are kept as they were
        lines += [f"Current stake: NGN {format_money(session.stake)}. Send the new amount as a normal message. The earlier "
                  "confirmation no longer works: you get a new summary to confirm.", ""]
    lines += ["How much do you want to stake (NGN)? Send a number, for example 500 or 1250.50: positive, at most two "
              "decimals.", "",
             f"Suggestion only: {suggestion.units} units (always between {MIN_UNITS} and {MAX_UNITS}). A unit is your "
             "own standard stake size; I do not know your bankroll, so I never turn units into money. It is a "
             "quarter-Kelly figure from the estimates and the total odds."]
    if not suggestion.edge:
        lines.append("The model sees no positive edge on this combination, so only the minimum is shown.")
    lines.append("The stake you send is final: I will not change it.")
    return Screen(_fit("\n".join(lines)))


# ---------- Story 5.4: the summary and the one final confirmation ----------

LIVE_BANNER = "*** LIVE: this will place a REAL bet on SportyBet ***"
DRY_BANNER = "DRY RUN: nothing will be placed"
CHANGE_LABEL = "Change amount"
# AD-25 (PLACE_MODE=book): no stake, no winnings, no Change amount; the one confirmation only gets a booking code.
BOOK_BANNER = "BOOKING CODE: no bet will be placed by me"
BOOK_LABEL = "Get booking code"
BOOK_DRY_LABEL = "Confirm (dry run: no code is made)"


def winnings_lines(total_odds, stake) -> list[str]:
    """The potential winnings (total odds x stake, rounded down, stake included) with the working, the profit part and
    the one-phrase estimate note. The odds are the exact total the confirmation is bound to."""
    return [f"Potential winnings: NGN {format_money(potential_winnings(total_odds, stake))} (total odds "
            f"{total_odds:.2f} x stake {format_money(stake)}; includes your stake back)",
            f"Profit if it wins: NGN {format_money(potential_profit(total_odds, stake))}",
            "This is an estimate: SportyBet may adjust for bonuses or limits."]


def _booking_summary_screen(session: Session, summary, dry_run: bool, notice: str) -> Screen:
    """AD-25: the final screen of PLACE_MODE=book. The games, their odds and the total odds only: no stake, no winnings."""
    head = [DRY_BANNER if dry_run else BOOK_BANNER, "",
            "Final summary. Check every line, then confirm once to get your booking code." if not dry_run
            else "Final summary. Check every line, then confirm once."]
    if notice:
        head += ["", clean_text(notice, 200)]
    head.append("")
    tail = [*left_out_lines(session), "", f"Total odds: {summary.total_odds:.2f}", ""]
    if dry_run:
        tail.append("DRY RUN is ON: confirming re-checks the betslip and reports what would be done. No booking code is "
                    "made and nothing is placed.")
        label = BOOK_DRY_LABEL
    else:
        tail.append("Confirm to get your booking code. I check the betslip once more and press only Book Bet on it. "
                    "I never press Place Bet and I type no stake: you choose the stake in SportyBet and place it yourself.")
        label = BOOK_LABEL
    tail.append("This confirmation is only for exactly this summary and stops working after 5 minutes. If the odds or a game "
                "change you get a new summary.")
    lines, used = list(head), len("\n".join(head + tail))
    shown = 0
    for number, leg in enumerate(summary.legs, start=1):
        line = (f"{number}. {clean_text(leg.match_text)}: {clean_text(leg.selection or leg.outcome_name)} "
                f"({clean_text(leg.market_name, 40)}) @ {leg.odds:.2f}")
        reserve = len(f"\nand {len(summary.legs) - number} more") if len(summary.legs) > number else 0
        if used + 1 + len(line) + reserve > MESSAGE_LIMIT - 1:
            break
        lines.append(line)
        used += 1 + len(line)
        shown += 1
    if shown < len(summary.legs):
        lines.append(f"and {len(summary.legs) - shown} more")
    sid, version = session.session_id, session.state_version
    return Screen("\n".join(lines + tail), (((label, encode(sid, version, Action.FINAL, summary.summary_id)),),))


def summary_screen(session: Session, *, dry_run: bool, notice: str = "") -> Screen:
    summary = session.summary
    if session.state is not State.AWAITING_FINAL_CONFIRM or summary is None:
        raise ValueError("there is no summary to show")
    if summary.booking:
        return _booking_summary_screen(session, summary, dry_run, notice)
    head = [DRY_BANNER if dry_run else LIVE_BANNER, "", "Final summary. Check every line, then confirm once."]
    if notice:
        head += ["", clean_text(notice, 200)]
    head.append("")
    tail = [*left_out_lines(session), "", f"Total odds: {summary.total_odds:.2f}", f"Stake: NGN {format_money(summary.stake)}",
            *winnings_lines(summary.total_odds, summary.stake), ""]
    if dry_run:
        tail.append("DRY RUN is ON: confirming re-checks the betslip and reports what would be placed. "
                    "Nothing will be placed.")
        label = "Confirm (dry run: nothing is placed)"
    else:
        tail.append("DRY RUN is OFF: confirming WILL place this bet with real money.")
        label = "Confirm and place bet"
    tail.append("This confirmation is only for exactly this summary and stops working after 5 minutes. If the odds, "
                "a game or the stake change you get a new summary. To change the stake, press Change amount.")
    lines, used = list(head), len("\n".join(head + tail))
    shown = 0
    for number, leg in enumerate(summary.legs, start=1):   # the winnings are never cut: the legs yield first
        line = (f"{number}. {clean_text(leg.match_text)}: {clean_text(leg.selection or leg.outcome_name)} "
                f"({clean_text(leg.market_name, 40)}) @ {leg.odds:.2f}")
        reserve = len(f"\nand {len(summary.legs) - number} more") if len(summary.legs) > number else 0
        if used + 1 + len(line) + reserve > MESSAGE_LIMIT - 1:
            break
        lines.append(line)
        used += 1 + len(line)
        shown += 1
    if shown < len(summary.legs):
        lines.append(f"and {len(summary.legs) - shown} more")
    sid, version = session.session_id, session.state_version
    rows = (((label, encode(sid, version, Action.FINAL, summary.summary_id)),),
            ((CHANGE_LABEL, encode(sid, version, Action.CHANGE)),))
    return Screen("\n".join(lines + tail), rows)


def placing_screen(dry_run: bool, book: bool = False) -> Screen:
    if dry_run:
        return Screen("Confirmed. Checking the SportyBet betslip against your summary (dry run: nothing will be placed).")
    if book:
        return Screen("Confirmed. Checking the SportyBet betslip against your summary, then getting your booking code. "
                      "No bet is placed. Please wait.")
    return Screen("Confirmed. Checking the SportyBet betslip against your summary, then placing the bet. Please wait.")


def placement_text(outcome) -> str:
    """The message for the end of a placement attempt (a flow.placement.PlacementOutcome)."""
    lines = [outcome.message]
    result = outcome.result
    if outcome.kind == "booked" and result is not None and outcome.token is not None:
        from predictions.telegram.booked import booked_message   # the one BOOKED message (the app sends it with its buttons)
        return booked_message(outcome.token, result.booking_code)[0]
    if outcome.kind == "dry_run" and result is not None:
        lines += ["", "What would have been placed:", *result.report]
    elif outcome.kind == "placed" and result is not None and result.booking_code:
        lines += [f"Booking code: {result.booking_code}"]
    elif outcome.kind == "placed":
        lines += ["SportyBet's bet history did not show a booking code; look it up in the app."]
    if outcome.warning:
        lines += ["", clean_text(outcome.warning, 300)]
    return "\n".join(lines)


# ---------- Story 5.5: a leg that was already placed ----------

def duplicate_screen(session: Session, duplicates: tuple[DuplicateLeg, ...]) -> Screen:
    summary = session.summary
    lines = ["NOT PLACED. A game in this bet was already placed, or may have been. Nothing was placed.", ""]
    for number, dup in enumerate(duplicates, start=1):
        leg = next((l for l in (summary.legs if summary else ()) if leg_identity(l.leg_key) == dup.identity), None)
        who = clean_text(leg.match_text) if leg else clean_text(dup.match_text or "a game in this bet")
        what = clean_text((leg.selection or leg.outcome_name) if leg else dup.selection)
        state = (f"already placed (bet id {clean_text(dup.bet_id, 40)})" if dup.proven and dup.bet_id else
                 "already placed" if dup.proven else "may already exist: its result was never confirmed")
        lines.append(f"{number}. {who}: {what} - {state}")
    lines += ["", "Only place it again if you are sure. The override applies to this exact summary only, and you must "
                  "confirm once more after it."]
    button = ("Override: place it anyway",
              encode(session.session_id, session.state_version, Action.OVERRIDE, summary.summary_id if summary else ""))
    return Screen(_fit("\n".join(lines)), ((button,),))


# ---------- Story 5.7: one slate at a time ----------

_STATE_WORDS = {State.EVALUATING: "being evaluated", State.REVIEWING_LIST: "waiting for you to review the list",
                State.REVIEWING_FINAL: "waiting for you to confirm the final edit",
                State.LOADING_SLIP: "being loaded onto SportyBet",
                State.RESOLVING_AVAILABILITY: "waiting for you to skip or accept a missing game or option",
                State.LOAD_PARTIAL: "waiting for you to decide about picks that could not be loaded",
                State.AWAITING_STAKE: "waiting for your stake", State.AWAITING_FINAL_CONFIRM: "waiting for your final "
                "confirmation", State.PLACING: "placing a bet", State.RECONCILING: "checking the bet history"}


def _slate_words(origin: str, picks: int, state: State | None = None) -> str:
    where = (f"from Review {clean_text(origin[len(REVIEW_ORIGIN):], 30)}" if origin.startswith(REVIEW_ORIGIN)
             else f"from {ORIGIN_NAMES.get(origin, 'Matchday')}")
    text = f"a slate {where} with {picks} game(s)"
    return text + (f", {_STATE_WORDS.get(state, state.value)}" if state is not None else "")


def replace_screen(open_session: Session, pending: Pending) -> Screen:
    lines = [f"You already have {_slate_words(open_session.origin, len(open_session.games()), open_session.state)}.",
             f"The new request made {_slate_words(pending.origin, len({e.match_id for e in pending.evaluations}))}.", "",
             "Replace the open slate with the new one? If you replace it, the old slate's buttons stop working. "
             "If you keep it, the new request is dropped and nothing else changes."]
    sid, version = open_session.session_id, open_session.state_version
    rows = ((("Replace the open slate", encode(sid, version, Action.REPLACE, pending.pending_id)),
             ("Keep the open slate", encode(sid, version, Action.KEEP, pending.pending_id))),)
    return Screen(_fit("\n".join(lines)), rows)


def start_screen(result: Result) -> Screen:
    """What to show after a trigger produced a slate: the list, or the question whether to replace the open slate."""
    if result.status is Status.APPLIED:
        return list_screen(result.session)
    if result.status is Status.ASK_REPLACE:
        return replace_screen(result.session, result.pending)
    return Screen(clean_text(result.notice or "A slate is already open.", 300))


# ---------- choosing the screen for a session ----------

def slate_screen(session: Session, dry_run: bool | None = None, now: datetime | None = None) -> Screen:
    state = session.state
    if state is State.REVIEWING_LIST:
        return list_screen(session)
    if state is State.REVIEWING_FINAL:
        return final_screen(session)
    if state is State.LOADING_SLIP:
        return confirmed_screen(session)
    if state is State.RESOLVING_AVAILABILITY:
        return issues_screen(session)
    if state is State.LOAD_PARTIAL:
        return partial_screen(session)
    if state is State.AWAITING_STAKE:
        return odds_screen(session)
    if state is State.AWAITING_FINAL_CONFIRM:
        if dry_run is None:
            raise ValueError("the summary needs to know whether dry run is on")
        return summary_screen(session, dry_run=dry_run)
    if state is State.PLACING:
        return placing_screen(True if dry_run is None else dry_run, session.booking)
    if state is State.RECONCILING:
        return Screen("Checking your SportyBet bet history to see whether a bet exists. Nothing will be placed.")
    if state in (State.FAILED, State.PLACEMENT_UNKNOWN):
        return failure_screen(session, now=now)
    return Screen(clean_text(session.message or "This session has ended.", 1000))


# ---------- AD-26: some picks could not be added to the betslip ----------

GO_AHEAD_LABEL = "Go ahead with the {n} loaded"
RETRY_PARTIAL_LABEL = "Retry the failed picks"


def partial_screen(session: Session, notice: str = "") -> Screen:
    """Plain: how many loaded, which (pick and the real odds), which failed and why, the total odds of the LOADED picks only,
    and three owner-pressed buttons. Nothing is dropped until the owner presses Go ahead."""
    loaded = session.loaded
    if loaded is None or not session.failed_picks:
        raise ValueError("there is no partial load to show")
    sid, version = session.session_id, session.state_version
    n, total = len(loaded.legs), len(loaded.legs) + len(session.failed_picks)
    left = MAX_RETRIES - session.retries_used
    lines = [f"Loaded {n} of {total} picks on SportyBet. {total - n} could not be added. No bet has been placed.", ""]
    if notice:
        lines += [clean_text(notice, 200), ""]
    lines.append("Loaded:")
    for number, leg in enumerate(loaded.legs, start=1):
        lines.append(f"{number}. {clean_text(leg.match_text)}: {clean_text(leg.selection or leg.outcome_name)} @ {leg.odds:.2f}")
        if leg.evaluation_odds is not None and leg.evaluation_odds != leg.odds:
            lines.append(f"   evaluation used {leg.evaluation_odds:.2f}, SportyBet now shows {leg.odds:.2f}")
    lines += ["", "Could not be added:"]
    for number, failure in enumerate(session.failed_picks, start=1):
        try:
            pick = session.slip.get(failure.pick_id)
        except Exception:
            continue
        ev = pick.evaluation
        lines.append(f"{number}. {_who(pick)}: {clean_text(play_label(ev))} @ {ev.odds:.2f}")
        lines.append(f"   Why: {clean_text(failure.reason, 300)}")
        if failure.times >= 2:
            times = "twice" if failure.times == 2 else f"{failure.times} times"
            lines.append(f"   This pick has failed {times} now.")
    lines += ["", f"Total odds of the {n} loaded pick{'' if n == 1 else 's'} only: {loaded.total_odds:.2f}", "",
              "Nothing is skipped or dropped on its own: you decide.",
              f"Go ahead: continue with only the {n} loaded; the {total - n} that failed are left out and listed on the next "
              "screens."]
    rows = [((GO_AHEAD_LABEL.format(n=n), encode(sid, version, Action.GO_AHEAD)),)]
    if left > 0:
        lines.append(f"{RETRY_PARTIAL_LABEL}: clears your betslip and adds ALL {total} picks again, then shows this screen "
                     f"with fresh results. ({left} of {MAX_RETRIES} retries left.)")
        rows.append(((RETRY_PARTIAL_LABEL, encode(sid, version, Action.RETRY_PARTIAL)),))
    else:
        lines.append(f"All {MAX_RETRIES} retries for this slate are used, so only Go ahead or Cancel are offered.")
    lines.append(f"{CANCEL_LABEL}: ends this slate. Nothing is booked or placed.")
    rows.append(((CANCEL_LABEL, encode(sid, version, Action.CANCEL)),))
    return Screen(_fit("\n".join(lines)), tuple(rows))


# ---------- failures: owner-pressed retry, never automatic (AD-24) ----------

TRY_AGAIN_LABEL = "Try again"
CHECK_LABEL = "Check history again"
RETRY_CHECKED_LABEL = "Try again, I checked"
YES_AGAIN_LABEL = "Yes, place again"
CANCEL_LABEL = "Cancel"
# Words in a reason that mean SportyBet itself refused the bet (a suspended market, an unavailable selection, no money).
_REFUSAL_WORDS = ("suspend", "unavailable", "no longer available", "insufficient", "balance", "refus", "not accepted")
MESSAGE_CAP = 1800   # characters of the reported reason (it keeps the adapter's own diagnostic text)


def clean_lines(text, limit: int) -> str:
    """Like clean_text, but the line breaks of the text are kept (the booking code and the odds sit on their own lines)."""
    out = "\n".join(clean_text(line, limit) for line in str(text).splitlines() if line.strip())
    return out if len(out) <= limit else out[: limit - 1] + "…"


def looks_like_refusal(message: str) -> bool:
    lowered = str(message).lower()
    return any(word in lowered for word in _REFUSAL_WORDS)


def minutes_ago(since: datetime | None, now: datetime | None) -> int | None:
    if since is None:
        return None
    return max(0, int(((now or datetime.now(timezone.utc)) - since).total_seconds() // 60))


def retry_notice(session: Session) -> str:
    """The short line on the NEW summary a retry leads to (the summary itself shows the stake, odds and winnings)."""
    if session.booking:
        return (f"Retry {session.retries_used} of {MAX_RETRIES}: same games, odds freshly read. "
                "Nothing is booked until you confirm it.")
    text = f"Retry {session.retries_used} of {MAX_RETRIES}: same games, same stake, odds freshly read. "
    if session.overrides and session.override_summary == (session.summary.summary_id if session.summary else ""):
        text += "You chose to place again, so a second bet may be placed. "
    return text + "Nothing is placed until you confirm."


def failure_screen(session: Session, notice: str = "", now: datetime | None = None, extra: str = "") -> Screen:
    """FAILED or PLACEMENT_UNKNOWN: whether a bet could exist, the reason (sanitized, with the adapter's own diagnostic
    text), and what each button does. The buttons are bound to this version; they are owner-pressed only."""
    sid, version = session.session_id, session.state_version
    left = MAX_RETRIES - session.retries_used
    reason = clean_lines(session.message or "No reason was given.", MESSAGE_CAP)
    lines: list[str] = []
    rows: list[tuple[tuple[str, str], ...]] = []
    if session.state is State.PLACEMENT_UNKNOWN:
        ago = minutes_ago(session.unknown_at, now)
        lines += ["A bet MAY exist. Do not assume it failed.", ""]
        if ago is not None:
            lines.append(f"The Place button was clicked about {ago} minute{'' if ago == 1 else 's'} ago.")
        lines.append(reason)
        if session.check_text:
            lines += ["", clean_text(session.check_text, 600)]
        if extra:
            lines.append(clean_text(extra, 300))
        if notice:
            lines += ["", clean_text(notice, 200)]
        lines += ["", f"{CHECK_LABEL}: reads your SportyBet bet history again. It only reads: it never places anything "
                      "and never touches your betslip. If the bet is there it is recorded as placed."]
        rows.append(((CHECK_LABEL, encode(sid, version, Action.CHECK)),))
        if session.history_checked and left > 0:
            lines.append(f"{RETRY_CHECKED_LABEL}: only if you have also looked in the SportyBet app. It shows a warning "
                         f"first. ({left} of {MAX_RETRIES} retries left.)")
            rows.append(((RETRY_CHECKED_LABEL, encode(sid, version, Action.RETRY_ASK)),))
        elif session.history_checked:
            lines.append(f"All {MAX_RETRIES} retries for this slate are used. Send a new request to start again.")
        else:
            lines.append(f"{RETRY_CHECKED_LABEL} appears only after a history check read the history and found nothing.")
        return Screen(_fit("\n".join(lines)), tuple(rows))
    lines += ["No bet was placed.", "", reason]
    if extra:
        lines.append(clean_text(extra, 300))
    if notice:
        lines += ["", clean_text(notice, 200)]
    if not session.retryable:
        lines += ["", "Send a new request to start again."]
    elif left <= 0:
        lines += ["", f"All {MAX_RETRIES} retries for this slate are used. Send a new request to start again."]
    else:
        if looks_like_refusal(session.message):
            lines += ["", "This looks like a real refusal by SportyBet: trying again may be refused again."]
        if session.booking:
            lines += ["", f"{TRY_AGAIN_LABEL}: reloads the same games onto your betslip, shows you the fresh odds and a NEW "
                          f"summary. Nothing is booked until you confirm it. Nothing is retried on its own. "
                          f"({left} of {MAX_RETRIES} retries left.)"]
        else:
            lines += ["", f"{TRY_AGAIN_LABEL}: reloads the same games and the same stake onto your betslip, shows you the "
                          f"fresh odds and a NEW summary. Nothing is placed until you confirm it. Nothing is retried on its "
                          f"own. ({left} of {MAX_RETRIES} retries left.)"]
        rows.append(((TRY_AGAIN_LABEL, encode(sid, version, Action.RETRY)),))
    return Screen(_fit("\n".join(lines)), tuple(rows))


def retry_warning_screen(session: Session) -> Screen:
    """The loud warning before a second attempt after an unknown result. The explicit override happens only on Yes."""
    sid, version = session.session_id, session.state_version
    lines = ["WARNING: a bet MAY already exist.", "",
             "If the first attempt actually went through, this will place a SECOND bet. Check your SportyBet bet history "
             "first. Press 'Yes, place again' only if you have checked.", "",
             "Pressing it places nothing yet: it reloads the same games with the same stake and shows a new summary "
             "that you must confirm."]
    rows = (((YES_AGAIN_LABEL, encode(sid, version, Action.RETRY_YES)), (CANCEL_LABEL, encode(sid, version, Action.BACK))),)
    return Screen("\n".join(lines), rows)


# ---------- handling a button ----------

_APPLIED_NOTICE = {Action.GO_AHEAD: "Going ahead.", Action.RETRY_PARTIAL: "Trying the failed picks again.",
                   Action.CANCEL: "Cancelled.", Action.DELETE: "Deleted.", Action.RESTORE: "Kept.",
                   Action.DONE: "Final confirm.", Action.ADJUST: "Back to adjust.", Action.CONFIRM: "Games confirmed.", Action.SKIP: "Skipped.", Action.ACCEPT: "Accepted.",
                   Action.FINAL: "Confirmed.", Action.CHANGE: "Change amount.", Action.RETRY: "Trying again.",
                   Action.CHECK: "Checking.", Action.RETRY_YES: "Placing again needs your confirmation."}


def handle_callback(data, owner: Owner, manager: SessionManager, dry_run: bool | None = None) -> Reply:
    """What to do for one button press. Buttons from an older list change nothing (AD-19). `dry_run` (the bot's
    DRY_RUN) is needed to word the final summary truthfully."""
    parsed = decode(data)
    if parsed is None:
        return Reply(NOTICE_UNKNOWN)
    session_id, version, action, arg = parsed
    result = manager.act(owner, session_id, version, action, arg)
    session = result.session
    if result.status is Status.STALE:
        return _stale_reply(result, manager, dry_run)
    if result.status is Status.REPEAT:
        return Reply(NOTICE_REPEAT)
    if result.status is Status.UNKNOWN_PICK:
        return Reply(NOTICE_GONE)
    if result.status is Status.DUPLICATE:
        return Reply("This game was already placed.", True, duplicate_screen(session, result.duplicates))
    if result.status is Status.BLOCKED:
        if action is Action.REPLACE:
            return Reply(result.notice, True)
        if session.state in (State.FAILED, State.PLACEMENT_UNKNOWN):
            return Reply(result.notice, True, failure_screen(session, result.notice, manager.now()))
        if session.state is State.AWAITING_FINAL_CONFIRM:
            return Reply(result.notice, True, summary_screen(session, dry_run=_need(dry_run), notice=result.notice))
        if session.state is State.RESOLVING_AVAILABILITY:
            return Reply(result.notice, True, issues_screen(session, result.notice))
        if session.state is State.LOAD_PARTIAL:
            return Reply(result.notice, True, partial_screen(session, result.notice))
        if session.state is State.REVIEWING_FINAL:
            return Reply(result.notice, True, final_screen(session, result.notice))
        return Reply(result.notice, True, list_screen(session, 0, result.notice))
    if result.status is Status.EXPIRED:
        return Reply(result.notice, True, summary_screen(session, dry_run=_need(dry_run), notice=result.notice))
    if result.status is Status.VIEW:
        if action is Action.RETRY_ASK:
            return Reply("", False, retry_warning_screen(session))
        if session.state is State.PLACEMENT_UNKNOWN:   # Cancel on the warning: back to the failure screen, unchanged
            return Reply("", False, failure_screen(session, now=manager.now()))
        if action is Action.KEEP:
            return Reply("Kept.", False, Screen("Kept the open slate. The new request was dropped and nothing else "
                                                "changed."))
        return Reply("", False, list_screen(session, int(arg) if arg.isdigit() else 0))
    # APPLIED
    if action is Action.REPLACE:
        return Reply("Replaced.", False, list_screen(session))
    if action is Action.OVERRIDE:
        note = "Override recorded for this exact summary. Nothing has been placed: confirm once more to place it."
        return Reply("Override recorded.", False, summary_screen(session, dry_run=_need(dry_run), notice=note))
    if action is Action.FINAL:
        return Reply(_APPLIED_NOTICE[action], False, placing_screen(_need(dry_run), manager.book_mode),
                     Job("place", session.session_id, session.state_version, result.token))
    if action is Action.CHECK:   # read-only: the job reads the bet history and nothing else
        return Reply(_APPLIED_NOTICE[action], False, slate_screen(session, dry_run),
                     Job("check", session.session_id, session.state_version))
    if action is Action.GO_AHEAD:   # AD-26: the loaded picks only; the job shows the odds, then the stake or the summary
        n = len(session.slip)
        text = (f"Going ahead with the {n} loaded pick{'' if n == 1 else 's'}."
                + "\n".join(left_out_lines(session)) + "\n\nNo bet has been placed.")
        return Reply(_APPLIED_NOTICE[action], False, Screen(_fit(text)), Job("go", session.session_id, session.state_version))
    if action is Action.CHANGE:   # back to the stake step: the prompt replaces the summary, its buttons are gone
        return Reply(_APPLIED_NOTICE[action], False, stake_prompt(session))
    if action in (Action.DELETE, Action.RESTORE):  # stay on the page the owner was editing
        return Reply(_APPLIED_NOTICE[action], False, list_screen(session, _page_of(session, arg)))
    job = Job("load", session.session_id, session.state_version) if result.load_next else None
    return Reply(_APPLIED_NOTICE[action], False, slate_screen(session, dry_run), job)


def _stale_reply(result: Result, manager: SessionManager, dry_run: bool | None) -> Reply:
    """A refused button. Nothing changed, but the owner is never left on a message with dead buttons: the pressed message
    is shown again as the CURRENT screen (its buttons carry the current version). A refusal with its own reason (a kept
    game has kicked off, the list is too old) says that reason instead of the generic 'list changed'."""
    session = result.session or manager.active
    if result.notice and session is not None:
        if session.state is State.REVIEWING_FINAL:
            return Reply(result.notice, True, final_screen(session, result.notice))
        if session.state is State.REVIEWING_LIST:
            return Reply(result.notice, True, list_screen(session, 0, result.notice))
        return Reply(result.notice, True)
    if session is None:
        return Reply(NOTICE_STALE)
    try:
        return Reply(NOTICE_STALE, False, slate_screen(session, dry_run, manager.now()))
    except ValueError:   # the summary screen needs DRY_RUN and the caller did not say: just refuse
        return Reply(NOTICE_STALE)


def _need(dry_run: bool | None) -> bool:
    if dry_run is None:
        raise ValueError("the summary needs to know whether dry run is on")
    return dry_run


def handle_stake_text(text, owner: Owner, manager: SessionManager, dry_run: bool) -> Reply | None:
    """A typed message that may be a stake (Story 5.3). None means no stake is being waited for, so the message is
    not a stake at all. Otherwise: an invalid stake gets an explanation and is asked for again; a valid one gives the
    new summary (the old confirmation, if any, stops working)."""
    result = manager.submit_stake(owner, text)
    if result.status is Status.STALE:
        return None
    if result.status is Status.BLOCKED:
        return Reply(result.notice)
    return Reply("", False, summary_screen(result.session, dry_run=dry_run))
