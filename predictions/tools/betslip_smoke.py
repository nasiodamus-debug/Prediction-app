"""OWNER-RUN smoke test for the SportyBet betslip adapter. It is DRY RUN ONLY: it can never place a bet.

Why it exists: the developer had no SportyBet login, so the login and the logged-in betslip were never seen (the bet-history
list page was inspected live and read-only on 2026-10-01 and is VERIFIED-LIVE; step 5 re-checks it). Run this on YOUR machine, with YOUR login, to find out which parts of the adapter's selectors work. Every step
prints what the adapter saw, so a wrong selector shows up here, in a harmless test, and not during a real bet.

HOW TO RUN (Windows PowerShell, from the project folder, with your bot STOPPED so nothing else uses the betslip):

    python -m predictions.tools.betslip_smoke                # headless, 2 games
    python -m predictions.tools.betslip_smoke --headed       # watch the browser while it works
    python -m predictions.tools.betslip_smoke --legs 3 --match Arsenal   # 3 games, preferring names containing Arsenal
    python -m predictions.tools.betslip_smoke --fast         # the same with FAST_BROWSER on (opt-in: measured slower), to compare seconds per step

It reads SPORTYBET_PHONE and SPORTYBET_PASSWORD from config.env (the same file the bot uses) and nothing else. It keeps
the login session in the `secrets` folder, screenshots of failures in `local_data/screenshots`, and sends nothing to
Telegram. WARNING: it CLEARS your SportyBet betslip (as the bot does) and leaves the smoke games on it: clear it by
hand afterwards, or just run the bot (loading always clears first).

What it does, in order (each prints STEP n ... OK or FAILED and what it saw):
  1. Reads the public game list and picks games (no login).
  2. Asks the adapter which of them are available (no login).
  3. Logs in, CLEARS the betslip, adds the games, reads the betslip back and verifies it holds exactly them.
  4. Enters a small stake, re-reads the betslip, and runs the placement check in DRY RUN: it proves the Place button is
     never clicked and prints what would have been placed.
  5. Reads the first page of your bet history (read only: the 5 newest bets) to see whether the history selectors work.

Send the developer the FAILED lines (they carry no secrets). Every FAILED step names the selector group involved.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMOKE_STAKE = "10"   # the smallest stake SportyBet accepts (min. 10); nothing is ever placed anyway

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictions.core.betting import Leg, UnsupportedLeg, leg_from_evaluation  # noqa: E402
from predictions.core.model import Estimate, Evaluation  # noqa: E402
from predictions.core.placement import PlacementKind  # noqa: E402
from predictions.core.types import Owner  # noqa: E402
from predictions.flow.session import Action, SessionManager  # noqa: E402

SMOKE_LINE = Decimal("2.5")


def pick_fixtures(fixtures, now: datetime, count: int, match: str = ""):
    """Up to `count` future games that list the Over/Under 2.5 market, preferring names that contain `match`."""
    usable = []
    for fixture in fixtures:
        if fixture.started or fixture.kickoff_utc <= now or not fixture.bookmaker_ref:
            continue
        for offer in fixture.markets:
            if offer.market_id == "18" and offer.specifier.replace(" ", "") == "total=2.5":
                usable.append(fixture)
                break
    usable.sort(key=lambda f: (match.lower() not in f"{f.home} {f.away}".lower() if match else False, f.kickoff_utc))
    return usable[:count]


def smoke_evaluation(fixture) -> Evaluation:
    """A stand-in pick (Over 2.5) used only to drive the session state machine to a token in this smoke test."""
    for offer in fixture.markets:
        if offer.market_id == "18" and offer.specifier.replace(" ", "") == "total=2.5":
            over = next(o for o in offer.outcomes if o.name.lower().startswith("over"))
            return Evaluation(fixture.match_id, "Over/Under 2.5", "Over 2.5", over.odds, Estimate(0.70), Decimal("0.05"),
                              True, "passed", "", "", ("smoke test stand-in",))
    raise ValueError("the game has no Over/Under 2.5")


class Reporter:
    """Prints each step and the measured seconds it took (spec 14-1: compare FAST_BROWSER on and off)."""

    def __init__(self, out=print, sanitize=lambda e: f"{type(e).__name__}: {e}", clock=time.perf_counter):
        self.out, self.sanitize, self.failed = out, sanitize, 0
        self._clock = clock
        self._mark = clock()
        self.timings: list[tuple[int, float]] = []

    def _lap(self, n) -> float:
        now = self._clock()
        seconds, self._mark = now - self._mark, now
        self.timings.append((n, seconds))
        return seconds

    def ok(self, n, name, *seen):
        self.out(f"STEP {n} OK: {name} ({self._lap(n):.1f} s)")
        for line in seen:
            self.out(f"    saw: {line}")

    def fail(self, n, name, error, hint=""):
        self.failed += 1
        self.out(f"STEP {n} FAILED: {name} ({self._lap(n):.1f} s)")
        self.out(f"    reason: {self.sanitize(error)}")
        if hint:
            self.out(f"    selector group to check: {hint}")


async def run_smoke(bookmaker, fixture_source, now: datetime, count: int = 2, match: str = "", out=print,
                    sanitize=lambda e: f"{type(e).__name__}: {e}") -> int:
    """The whole smoke test. Returns the number of failed steps. `bookmaker` MUST be in dry run."""
    if not bookmaker.dry_run:
        raise SystemExit("The smoke test only runs in dry run. Refusing.")
    rep = Reporter(out, sanitize)

    # 1. game list
    try:
        fixtures = await fixture_source.fetch_fixtures()
        chosen = pick_fixtures(fixtures, now, count, match)
        if not chosen:
            raise ValueError("no upcoming game with an Over/Under 2.5 market was found")
        evaluations = [smoke_evaluation(f) for f in chosen]
        legs: list[Leg] = [leg_from_evaluation(e) for e in evaluations]
        rep.ok(1, "read the public game list", *[f"{f.home} v {f.away} ({f.bookmaker_ref})" for f in chosen])
    except Exception as error:
        rep.fail(1, "read the public game list", error, "game list / fixtures adapter")
        return rep.failed

    # 2. availability
    try:
        availability = await bookmaker.check_availability(legs)
        bad = [a for a in availability if a.status.value != "ok"]
        if bad:
            raise ValueError("; ".join(f"{a.leg.selection}: {a.status.value}" for a in bad))
        rep.ok(2, "availability", *[f"{a.leg.selection} @ {a.outcome.odds} is offered" for a in availability])
    except Exception as error:
        rep.fail(2, "availability", error)
        return rep.failed

    # 3. login, clear, add, read back, verify
    try:
        loaded = await bookmaker.load_slip(legs)
        rep.ok(3, "log in, clear the betslip, add the games, read back and verify",
               *[f"{l.match_text} | {l.market_name} | {l.outcome_name} @ {l.odds}" for l in loaded.legs],
               f"total odds {loaded.total_odds}")
    except Exception as error:
        rep.fail(3, "log in, clear the betslip, add the games, read back and verify", error,
                 "login_*, remove_all / item_delete, outcome_cell_template, betslip_items / item_*, total_odds "
                 "(a screenshot is in local_data/screenshots)")
        return rep.failed

    # 4. a token from the real state machine, then the dry-run placement check
    try:
        manager = SessionManager()
        owner = Owner(0)   # this script is not Telegram: it plays the owner only to reach a token
        session = manager.start(evaluations, "smoke001")
        manager.act(owner, session.session_id, session.state_version, Action.DONE)
        manager.act(owner, manager.active.session_id, manager.active.state_version, Action.CONFIRM)
        manager.mark_loaded(session.session_id, manager.active.state_version, loaded)
        manager.mark_odds_shown(session.session_id, manager.active.state_version)
        manager.submit_stake(owner, SMOKE_STAKE)
        active = manager.active
        result = manager.act(owner, active.session_id, active.state_version, Action.FINAL, active.summary.summary_id)
        if result.token is None:
            raise ValueError("the state machine did not mint a token")
        placed = await bookmaker.place(result.token)
        if placed.kind is not PlacementKind.DRY_RUN:
            raise ValueError("the placement did not stay in dry run")
        rep.ok(4, "stake entry, betslip re-check and dry-run placement (the Place button is NOT clicked)", *placed.report)
    except Exception as error:
        rep.fail(4, "stake entry, betslip re-check and dry-run placement", error, "stake_input, betslip_items / total_odds")

    # 5. bet history LIST page (read only): the first page, the 5 newest bets, newest first
    try:
        history = await bookmaker.bet_history()
        rep.ok(5, "read the first page of the bet history (read only)", f"{len(history)} bet(s) read (at most 5 per page)",
               *[f"{b.placed_at:%Y-%m-%d %H:%M} UTC, {b.bet_type}, {b.status}, stake {b.stake}, "
                 f"{b.leg_count} game(s) ({len(b.shown_matches)} named), key {b.bet_id}"
                 + (", AMBIGUOUS key (identical entry)" if b.ambiguous else "") for b in history[:3]])
        out("    note: the list shows no bet id, booking code, odds or picks. The key above is built from the visible "
            "fields; a bet is recognised only by its games, stake, number of legs, status and time.")
        if not history:
            out("    note: no bets were read although the page had a recognised empty-list marker. If your history is "
                "not empty, report it.")
        if any(b.ambiguous for b in history):
            out("    note: two entries share a key (same minute, type, stake and games): they can never prove a bet.")
        if len(history) > 5:
            out("    note: more than 5 entries were read: the page layout changed, report it.")
    except Exception as error:
        rep.fail(5, "read the bet history", error, "history_url, history_items, history_time, history_type, "
                 "history_status, history_vs, history_stake, history_return, history_empty (UNVERIFIED: an empty list)")

    out("")
    out("Seconds per step: " + ", ".join(f"step {n} {seconds:.1f}" for n, seconds in rep.timings)
        + f"; total {sum(seconds for _, seconds in rep.timings):.1f}.")
    out(f"Finished: {rep.failed} step(s) failed. The Place button was never clicked (dry run).")
    out("Your betslip still holds the smoke games: clear it by hand, or start the bot (it clears first).")
    return rep.failed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run smoke test of the SportyBet betslip adapter.")
    parser.add_argument("--headed", action="store_true", help="show the browser window")
    parser.add_argument("--legs", type=int, default=2, help="how many games to load (default 2)")
    parser.add_argument("--fast", action="store_true", help="run with FAST_BROWSER on (no pictures, fonts or media) to compare the timings")
    parser.add_argument("--no-fast", action="store_true",
                        help="run as FAST_BROWSER=off (load pictures and fonts, slower looks) to compare the timings")
    parser.add_argument("--match", default="", help="prefer games whose names contain this text")
    args = parser.parse_args(argv)
    import os

    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / "config.env")
    from predictions.adapters.sportybet.betslip import Credentials, SportyBetBookmaker, parse_fast_browser
    from predictions.adapters.sportybet.fixtures import SportyBetFixtures

    try:
        credentials = Credentials(os.getenv("SPORTYBET_PHONE") or "", os.getenv("SPORTYBET_PASSWORD") or "")
    except Exception as error:
        print(f"Cannot start: {error}")
        return 2
    secrets = [credentials.phone, credentials.password]

    def sanitize(error) -> str:
        text = f"{type(error).__name__}: {error}"
        for secret in secrets:
            if secret:
                text = text.replace(secret, "[hidden]")
        return text

    try:
        fast = (parse_fast_browser(os.getenv("FAST_BROWSER")) or args.fast) and not args.no_fast
    except ValueError as error:
        print(f"Cannot start: {error}")
        return 2
    print(f"FAST_BROWSER is {'on' if fast else 'off'} for this run.")
    bookmaker = SportyBetBookmaker(credentials, dry_run=True, headless=not args.headed, fast_browser=fast)   # ALWAYS dry run
    failed = asyncio.run(run_smoke(bookmaker, SportyBetFixtures(), datetime.now(timezone.utc),
                                   max(1, min(args.legs, 5)), args.match, print, sanitize))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
