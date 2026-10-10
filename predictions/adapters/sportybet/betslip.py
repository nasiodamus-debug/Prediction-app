"""SportyBet's betslip, behind the Bookmaker port (Stories 5.1 to 5.4). ALL site knowledge for the betslip lives here.

What this adapter does, and the rules it lives by (AD-6, AD-11, AD-14, AD-16, AD-17):

* One asyncio lock wraps every operation, so a load, a check, a place or a history read never overlap.
* `load_slip` clears the owner's betslip FIRST, adds exactly the requested legs, reads the betslip back and verifies it
  holds exactly those legs. It never places anything.
* `place(token)` re-reads the real betslip and refuses to go on unless it matches the token exactly. With dry run on
  (the default) it stops there. With dry run off it takes a snapshot of the bet history, clicks Place, and reports
  success ONLY when a NEW bet appears in the bet history. Anything odd after the click is PlacementUnknown.
* Two independent guards keep the Place button unclicked in dry run: `place()` returns before the click, and the page
  driver itself (`SportyBetSite.click_place`) refuses when it was built with `allow_place=False`.
* Nothing is retried. Failures raise BookmakerError with a message that carries no URL, path or credential; a screenshot
  is saved under local_data/ (never sent anywhere).
* Login details come from the caller (app.py's already-loaded config); the Playwright storage state (session cookies)
  is kept ONLY in a directory named `secrets`, never in a synced folder.

SELECTORS: every URL, selector and piece of page text is in the `Selectors` structure below, each marked
    VERIFIED-ANON  = probed on 2026-09-30 on the public site as an anonymous visitor (desktop layout, no login);
    VERIFIED-LIVE  = inspected read-only on 2026-10-01 in the owner's logged-in session (the bet-history list page);
    UNVERIFIED     = could not be checked without logging in. The owner validates these with
                     `python -m predictions.tools.betslip_smoke` (see that file).
Every step verifies what it did, so a wrong selector makes an operation FAIL CLOSED, it never makes it guess.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Awaitable, Callable, Sequence
from urllib.parse import urlsplit

from dataclasses import replace as _replace

from predictions.adapters.sportybet.event import WANTED_MARKET_IDS, SportyBetEventMarkets
from predictions.adapters.sportybet.fixtures import SportyBetFixtures
from predictions.core.betting import (history_bet_matches, history_limits_note, Availability, BetslipLine, BetslipReading, ExpectedLine, HistoryBet, Leg,
                                      FailedLeg, LegAvailability, LoadAttempt, LoadedSlip, assess_availability, make_loaded_slip,
                                      verify_reading)
from predictions.core.engines.basketball import (AWAY_TOTAL_MARKET_ID as BB_AWAY_TOTAL_MARKET,
                                                 FH_TOTAL_MARKET_ID as BB_FH_TOTAL_MARKET,
                                                 HANDICAP_MARKET_ID as BB_HANDICAP_MARKET,
                                                 HOME_TOTAL_MARKET_ID as BB_HOME_TOTAL_MARKET,
                                                 TOTAL_MARKET_ID as BB_TOTAL_MARKET,
                                                 WINNER_MARKET_ID as BB_WINNER_MARKET)
from predictions.core.market_rules import UNDER_REASON, WHOLE_LINE_REASON, is_under, is_whole_or_quarter_line
from predictions.core.types import normalize_team
from predictions.core.placement import PlacementKind, PlacementResult, PlacementToken
from predictions.core.ports import (BetslipMismatch, BookmakerError, PickNotAdded, EventMarketsSource, FixtureFetchError, FixtureSource,
                                    OddsChanged, PlacementUnknown)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SECRETS_DIR = PROJECT_ROOT / "secrets"                      # Playwright storage state: never synced, never committed
SCREENSHOT_DIR = PROJECT_ROOT / "local_data" / "screenshots"  # failure screenshots: local only, never sent to Telegram
STATE_FILE_NAME = "sportybet_state.json"
_SYNCED_FOLDER_MARKERS = ("onedrive", "dropbox", "google drive", "googledrive", "icloud", "box sync", "megasync")

VIEWPORT = {"width": 1366, "height": 900}
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/120.0.0.0 Safari/537.36")
POLL_MS = 250
# Spec 14-1 (FAST_BROWSER, OFF unless the owner says on; measured live 2026-10-01: with it ON loading the betslip took about 90 s and\n# reading the history about 45 s, against about 54 s and 32 s with it OFF, so it is opt-in): the browser does not download pictures, fonts or media (the page
# is read as text and by selectors only), and the loops that WAIT FOR A CONDITION look at it every FAST_POLL_MS instead of
# every POLL_MS. No check is skipped: every read-back, odds, stake and history check still runs exactly as before.
BLOCKED_RESOURCE_TYPES = frozenset({"image", "font", "media"})
FAST_POLL_MS = 100


def parse_fast_browser(raw) -> bool:
    """The FAST_BROWSER setting, read once at startup. Unset or empty = OFF (the measured default). 'true'/'on' = on, 'false'/'off' = off (any case).
    Anything else raises ValueError (never echoed): the bot must not guess which behaviour the owner meant."""
    if raw is None or not str(raw).strip():
        return False
    text = str(raw).strip().lower()
    if text in ("true", "on"):
        return True
    if text in ("false", "off"):
        return False
    raise ValueError("FAST_BROWSER in config.env must be true, on, false or off. Fix it or remove the line to use the "
                     "default (off), then start the bot again.")


def parse_browser_mode(raw) -> str:
    """The BROWSER_MODE setting, read once at startup. Unset or empty = 'standard' (the bundled Chromium, as before).
    'real' = the installed Google Chrome in a visible window, with the automation flags hidden (an attempt to stop SportyBet's
    bot detection from freezing the Place button). Anything else raises ValueError (never echoed)."""
    if raw is None or not str(raw).strip():
        return "standard"
    text = str(raw).strip().lower()
    if text in ("standard", "real"):
        return text
    raise ValueError("BROWSER_MODE in config.env must be standard or real. Fix it or remove the line to use the "
                     "default (standard), then start the bot again.")


def parse_place_mode(raw) -> str:
    """The PLACE_MODE setting (AD-25), read once at startup. Unset or empty = 'book': the final confirmation only generates a
    booking code (it never clicks Place or Confirm). 'place' = the bot places the bet itself. Any case. Anything else raises
    ValueError (never echoed): the bot must not guess whether it may place a bet."""
    if raw is None or not str(raw).strip():
        return "book"
    text = str(raw).strip().lower()
    if text in ("book", "place"):
        return text
    raise ValueError("PLACE_MODE in config.env must be book or place. Fix it or remove the line to use the default (book), "
                     "then start the bot again.")


DEFAULT_PLACE_WAIT_S = 90
MIN_PLACE_WAIT_S, MAX_PLACE_WAIT_S = 30, 300


def parse_place_wait_seconds(raw) -> int:
    """PLACE_WAIT_SECONDS: how long the page is watched after the Place click for the confirm panel and the success message.
    Unset or empty = 90. A whole number from 30 to 300; anything else raises ValueError (never echoed)."""
    if raw is None or not str(raw).strip():
        return DEFAULT_PLACE_WAIT_S
    text = str(raw).strip()
    if text.isascii() and text.isdigit() and MIN_PLACE_WAIT_S <= int(text) <= MAX_PLACE_WAIT_S:
        return int(text)
    raise ValueError(f"PLACE_WAIT_SECONDS in config.env must be a whole number from {MIN_PLACE_WAIT_S} to {MAX_PLACE_WAIT_S}. "
                     f"Fix it or remove the line to use the default ({DEFAULT_PLACE_WAIT_S}), then start the bot again.")


# BROWSER_MODE=real: the owner's installed Chrome (its own User-Agent, never overridden), automation switches hidden.
REAL_BROWSER_ARGS = ["--disable-blink-features=AutomationControlled"]
REAL_BROWSER_IGNORE_DEFAULT_ARGS = ["--enable-automation"]
HIDE_WEBDRIVER_JS = "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
CHROME_MISSING_TEXT = ("BROWSER_MODE=real needs Google Chrome installed on this computer, and it could not be started. Install "
                       "Google Chrome, or set BROWSER_MODE=standard in config.env and restart the bot. Nothing was placed.")
PAGE_LOAD_MS = 60_000
LOGIN_FORM_WAIT_MS = 8_000
SESSION_LOST = ("SportyBet logged your account out while the games were being added (the site ended the session, possibly because "
                "the account was opened somewhere else). Nothing was placed. Close SportyBet on other devices and press Try again.")
LOGIN_RECHECKS = 2            # before 'not logged in' is believed, the login form must still show after this many re-checks
log = logging.getLogger("sportybet")
LOGIN_RECHECK_PAUSE_S = 2.0   # ... each one this many seconds apart (the form was seen to appear for a moment on a logged-in page)
LOGIN_DONE_WAIT_MS = 20_000
ELEMENT_WAIT_MS = 15_000
SETTLE_MS = 20_000
HISTORY_WAIT_MS = 15_000
REDIRECT_WAIT_MS = 8_000   # how long to wait for the history page to either show its rows or redirect to the login page
HISTORY_POLLS = 6            # after the click: how many times the bet history is read before giving up
HISTORY_POLL_GAP_MS = 4_000
MAX_BETSLIP_ROWS = 60           # a safety stop for clearing one row at a time
LOAD_TIMEOUT_S = 240.0
PLACE_TIMEOUT_S = 360.0   # PLUS the configured PLACE_WAIT_SECONDS (see SportyBetBookmaker.place)
BOOK_WAIT_MS = 8_000   # how long the Book Bet fallback waits for a booking code to show


# ================================================================== ALL SITE KNOWLEDGE, IN ONE PLACE

@dataclass(frozen=True)
class Selectors:
    """Every URL, selector and text the adapter relies on. Nothing about the site is written anywhere else."""

    # ---- addresses
    home_url: str = "https://www.sportybet.com/ng/"                                            # VERIFIED-ANON
    # The event page keys on the LAST path segment; the slugs before it are decoration (VERIFIED-ANON).
    event_url: str = "https://www.sportybet.com/ng/sport/football/event/event/event_vs_event/{event_ref}"
    history_url: str = "https://www.sportybet.com/ng/my_accounts/bet_history"                  # UNVERIFIED

    # ---- login (the form is in the page header for a visitor who is not logged in)
    login_phone: str = 'input[name="phone"]'                                                   # VERIFIED-ANON
    login_password: str = 'input[name="psd"]'                                                  # VERIFIED-ANON
    login_button: str = 'button[name="logIn"]'                                                 # VERIFIED-ANON
    # After a successful login the form above disappears (UNVERIFIED: no other logged-in marker is used).

    # ---- the betslip panel
    betslip_items: str = "#j_betslip .m-list .m-item"                                          # VERIFIED-ANON
    item_outcome: str = ".m-item-play span"          # e.g. "Over 2.5", "Home"               # VERIFIED-ANON
    item_teams: str = ".m-item-team"                 # e.g. "Hungary v Georgia"              # VERIFIED-ANON
    item_market: str = ".m-item-market"              # e.g. "Over/Under", "1X2"              # VERIFIED-ANON
    item_odds: str = ".m-item-odds .m-text-main"     # e.g. "2.05"                           # VERIFIED-ANON
    # Total odds of a multiple: the value next to the "Odds" label. A single selection has no such label.
    total_odds: str = "#j_betslip div:has(> .m-label .m-odds-rounded-label) > .m-value"        # VERIFIED-ANON (2 legs)
    stake_input: str = "#j_betslip .m-stake input.m-input"                                     # VERIFIED-ANON
    remove_all: str = '#j_betslip .m-text-min[data-cms-key="remove_all"]'                      # VERIFIED-ANON (2+ rows)
    item_delete: str = "#j_betslip .m-list .m-item .m-icon-delete"   # one row's delete icon    # VERIFIED-ANON
    popup_close: str = ('.es-dialog-wrap [class*="close"], [class*="modal"] [class*="close"], '
                        '[class*="popup"] [class*="close"], [role="dialog"] [aria-label*="lose"]')  # UNVERIFIED (Escape is tried first)
    dialog: str = ".es-dialog-wrap"                                                            # VERIFIED-ANON
    dialog_ok: str = '.es-dialog-wrap .es-dialog-footer a[data-ret="1"]'  # "OK" of Remove Betslip  VERIFIED-ANON

    # ---- the game page: one wrapper per market (and per line), a header title, and a row of outcome cells
    # Composed as: wrapper with this title AND this outcome text, then the cell with that outcome text.
    outcome_cell_template: str = (
        '.m-table__wrapper:has(.m-table-header-title:text-is("{title}")):has(.m-table-cell-item:text-is("{outcome}"))'
        ' .m-table-cell:has(.m-table-cell-item:text-is("{outcome}"))')                         # VERIFIED-ANON (1X2, O/U)
    # Team-total wrappers are titled "<Team> Over/Under" and the betslip row's market says the same (VERIFIED-ANON),
    # while the listing calls the market "Home O/U": the page wording is the one to look for and to verify.
    team_total_title: str = "{team} Over/Under"
    # Basketball (probed anonymously on a WNBA game page and betslip, 2026-09-30, VERIFIED-ANON): the page title AND the
    # betslip row's market say the same words, and these differ from the listing's names ("Home O/U (incl. overtime)",
    # "1st half - total"). Totals and handicaps have one wrapper per line with the line in the title; the 1st-half total
    # has one wrapper per line all titled alike, with the line only in the outcome text ("Over 74.5").
    bb_winner_title: str = "Winner (incl. overtime)"
    bb_total_title: str = "Over/Under (incl. overtime) {line}"
    bb_team_total_title: str = "{team} Over/Under (incl. overtime) {line}"
    bb_fh_total_title: str = "1st Half - Over/Under"
    bb_handicap_title: str = "Handicap (incl. overtime) {line}"

    # Soccer play types added by spec 11-1 (probed anonymously on the public game page and betslip of an international game that
    # lists them, 2026-10-01, all VERIFIED-ANON): the wrapper title AND the betslip row's market say the same words as the
    # event endpoint's market name, so `_page_market_name` simply uses the listed offer's name:
    #   Match winner  title "1X2"                    outcomes "Home" / "Away"           (betslip row: market "1X2")
    #   Double chance title "Double Chance"          outcomes "Home or Draw" / "Draw or Away" (row: market "Double Chance")
    #   Total corners title "Corners - Over/Under"   one wrapper per line, outcome "Over 9.5"    (row: market "Corners - Over/Under")
    #   Total cards   title "Bookings - Over/Under"  one wrapper per line, outcome "Over 2.5"    (row: market "Bookings - Over/Under")
    # Corners and cards lines share one title, the line is only in the outcome text, and `outcome_cell` already requires the
    # outcome text inside the wrapper, so exactly one cell matches. Loading them LOGGED IN is UNVERIFIED like every other
    # loader step (the owner's `betslip_smoke` run validates it); a mismatch fails closed.
    corners_title: str = "Corners - Over/Under"      # VERIFIED-ANON
    cards_title: str = "Bookings - Over/Under"       # VERIFIED-ANON
    double_chance_title: str = "Double Chance"       # VERIFIED-ANON

    # ---- placing (NEVER touched in dry run)
    place_button: str = 'button[data-op="desktop-betslip-place-bet-button"]'                  # VERIFIED-ANON (exists)
    # What follows the click is UNVERIFIED: a confirmation dialog with a confirm button, then a receipt.
    confirm_dialog_ok: str = '.es-dialog-wrap .es-dialog-footer a[data-ret="1"]'              # UNVERIFIED
    # Words in that dialog that mean the price moved or the bet was refused: never confirmed blindly.
    dialog_refusal_words: tuple[str, ...] = ("changed", "change", "suspend", "unavailable", "insufficient", "fail",
                                         "updated", "moved", "increase", "decrease", "reject", "limit", "expired")

    # ---- bet history. LIST PAGE inspected live and read-only on 2026-10-01 (desktop site, logged in). Newest bets first,
    # 5 bets per page; only page 1 is ever read. The page has NO bet id, NO booking code, NO odds, NO market or pick and
    # NO per-leg result: those are on the details page behind `history_detail`, which has not been inspected.
    history_url: str = "https://www.sportybet.com/ng/my_accounts/bet_history/sport_bets?isSettled=10"   # VERIFIED-LIVE (10 = all)
    history_url_unsettled: str = "https://www.sportybet.com/ng/my_accounts/bet_history/sport_bets?isSettled=0"  # VERIFIED-LIVE (running only)
    login_path: str = "/login"                                                                 # VERIFIED-LIVE 2026-10-01: the history page redirects to /ng/login?ret=...
    login_page_password: str = 'input[type="password"]'                                        # VERIFIED-ANON: the full login page (no name on the box)
    login_page_button: str = 'button.af-button--primary'                                       # VERIFIED-ANON: 'Login', disabled until both boxes are filled
    history_path: str = "/my_accounts/bet_history"                                             # VERIFIED-LIVE
    history_items: str = "div.m-order-list"                                    # one bet each          VERIFIED-LIVE
    history_time: str = ".m-order-header .m-order-left"                        # ' 30/09/2026 06:44 '  VERIFIED-LIVE (site time GMT+01:00)
    history_type: str = ".m-order-bar .m-order-type"                           # 'Multiple'/'Singles'  VERIFIED-LIVE
    history_status: str = ".m-order-bar .m-order-status"                       # 'Running'/'Won'/'Lost' (others exist, unseen) VERIFIED-LIVE
    history_detail: str = ".m-order-detail"                                    # icon to the details page; NOT used, NOT inspected  VERIFIED-LIVE (exists)
    history_vs: str = ".m-order-selection .m-order-vs-wrap div.m-order-vs"     # up to 3 games, then 'and N other matches'  VERIFIED-LIVE
    history_vs_text: str = ".vs-text"                                          # the 'v' span inside a game line  VERIFIED-LIVE
    history_stake: str = ".m-order-total-stake .stake"                         # '3,000.00'            VERIFIED-LIVE
    history_return: str = ".m-order-total-return .stake"                       # '0.00', or '--' while running  VERIFIED-LIVE
    history_pagination: str = "div.pagination"                                 # page 1 selected; NOT used (page 1 only)  VERIFIED-LIVE
    # What an empty history shows has NOT been seen (the owner's account has bets): without one of these markers an empty
    # list is an UNREADABLE page, never "no bets" (fail closed).
    history_empty: str = ".m-empty, .empty-tips, .m-no-data"                   # UNVERIFIED
    history_time_format: str = r"^(\d{2})/(\d{2})/(\d{4})\s+(\d{2}):(\d{2})$"   # dd/mm/yyyy hh:mm   VERIFIED-LIVE
    history_other_matches: str = r"^and\s+(\d+)\s+other\s+match(?:es)?$"        # 'and 2 other matches' / 'and 1 other match'  VERIFIED-LIVE
    history_money: str = r"^(?:NGN|₦)?\s*(\d+(?:\.\d+)?)$"                       # after the thousands commas are removed  VERIFIED-LIVE
    # The confirmation dialog after Place might show a booking code (UNVERIFIED; the history list never does).
    confirm_booking_code: str = r"Booking\s*Code\s*[:#]?\s*([0-9A-Za-z]{4,})"   # UNVERIFIED

    def event_page(self, event_ref: str) -> str:
        return self.event_url.format(event_ref=event_ref)

    def outcome_cell(self, title: str, outcome: str) -> str:
        for text in (title, outcome):
            if '"' in text or "\\" in text or "\n" in text:
                raise BookmakerError("a market or option name contains characters that cannot be searched for safely")
        return self.outcome_cell_template.format(title=title, outcome=outcome)


SELECTORS = Selectors()


# ================================================================== credentials, secrets folder

@dataclass(frozen=True)
class Credentials:
    """The owner's SportyBet login, handed in from app.py's already-loaded config. Never printed, never stored."""
    phone: str = field(repr=False)
    password: str = field(repr=False)

    def __post_init__(self):
        if not (self.phone or "").strip() or not (self.password or ""):
            raise BookmakerError("The SportyBet phone and password are not set in the configuration.")

    def national_phone(self) -> str:
        """The number as typed into the login box: digits only, without the country code or a leading zero
        (UNVERIFIED that this is the accepted form; a wrong form makes the login fail closed)."""
        digits = re.sub(r"\D", "", self.phone)
        if digits.startswith("234") and len(digits) >= 13:
            digits = digits[3:]
        return digits.lstrip("0") if digits.startswith("0") else digits


def check_secrets_dir(path: Path) -> Path:
    """The storage state holds session cookies: it must live in a folder named `secrets` that is not inside a synced
    folder (OneDrive, Dropbox, Google Drive, iCloud...). Returns the resolved folder, creating it with a .gitignore."""
    resolved = Path(path).resolve()
    if resolved.name.lower() != "secrets":
        raise BookmakerError("The login state must be kept in a folder named 'secrets'.")
    lowered = [part.lower() for part in resolved.parts]
    if any(marker in part for part in lowered for marker in _SYNCED_FOLDER_MARKERS):
        raise BookmakerError("The 'secrets' folder is inside a synced folder. Move the bot out of it.")
    resolved.mkdir(parents=True, exist_ok=True)
    ignore = resolved / ".gitignore"
    if not ignore.exists():
        ignore.write_text("*\n", encoding="utf-8")
    return resolved


def _parse_decimal(text: str) -> Decimal | None:
    try:
        value = Decimal(re.sub(r"[^\d.]", "", text or "") or "x")
    except InvalidOperation:
        return None
    return value if value.is_finite() and value > 0 else None


# ================================================================== the page driver (raw Playwright)

class SportyBetSite:
    """Every raw browser action, and nothing else. It knows the selectors and how to wait; it makes no decisions.
    `allow_place` is the second guard: a driver built without it can never click Place."""

    def __init__(self, page, context, credentials: Credentials | None, storage_path: Path | None,
                 allow_place: bool = False, selectors: Selectors = SELECTORS, browser=None, playwright=None,
                 poll_ms: int = POLL_MS, place_wait_ms: int = DEFAULT_PLACE_WAIT_S * 1000):
        self.page, self.context = page, context
        self._place_wait_ms = max(1, int(place_wait_ms))   # PLACE_WAIT_SECONDS: how long finish_placement watches the page
        self._poll_ms = max(1, int(poll_ms))   # the gap between looks while waiting for a condition
        self._browser, self._playwright = browser, playwright
        self._credentials, self._storage_path = credentials, storage_path
        self._allow_place = bool(allow_place)
        self.s = selectors

    # ---- small helpers
    async def _count(self, selector: str) -> int:
        return await self.page.locator(selector).count()

    async def _visible(self, selector: str) -> bool:
        loc = self.page.locator(selector)
        return await loc.count() > 0 and await loc.first.is_visible()

    async def _poll(self, check: Callable[[], Awaitable[bool]], timeout_ms: int) -> bool:
        for _ in range(max(1, timeout_ms // self._poll_ms) + 1):
            if await check():
                return True
            await self.page.wait_for_timeout(self._poll_ms)
        return False

    async def _text(self, root, selector: str) -> str:
        loc = root.locator(selector)
        if await loc.count() == 0:
            raise BookmakerError("the betslip did not look as expected")
        return (await loc.first.inner_text()).strip()

    # ---- login
    async def login(self) -> None:
        """Open the site; if the login form shows, log in with the owner's details. Uses the stored session when it
        is still valid (then the form never appears)."""
        await self.page.goto(self.s.home_url, wait_until="domcontentloaded", timeout=PAGE_LOAD_MS)
        if not await self._poll(lambda: self._visible(self.s.login_phone), LOGIN_FORM_WAIT_MS):
            return  # no login form: the stored session is still logged in
        if self._credentials is None:
            raise BookmakerError("SportyBet asked for a login and no login details are available.")
        await self.page.locator(self.s.login_phone).first.fill(self._credentials.national_phone())
        await self.page.locator(self.s.login_password).first.fill(self._credentials.password)
        await self.page.locator(self.s.login_button).first.click()

        async def form_gone() -> bool:
            return not await self._visible(self.s.login_phone)

        if not await self._poll(form_gone, LOGIN_DONE_WAIT_MS):
            raise BookmakerError("The SportyBet login did not complete (wrong details, a code or a captcha was asked for).")
        await self.save_state()

    async def save_state(self) -> None:
        if self._storage_path is not None:
            await self.context.storage_state(path=str(self._storage_path))

    async def logged_out(self) -> bool:
        """True when the page shows the login form (the home page) OR the header's Login button (the compact header on an event
        page has no boxes, only that button, which is also what a logged-out visitor sees there; probed anonymously 2026-10-09)."""
        return await self._visible(self.s.login_phone) or await self._visible(self.s.login_button)

    # ---- the betslip
    async def betslip_count(self) -> int:
        return await self._count(self.s.betslip_items)

    async def clear_betslip(self) -> None:
        """Remove every selection, then check none is left. With two or more rows the site offers Remove All (with a
        confirmation dialog); with a single row it does not, so that row's delete icon is used."""
        if await self.betslip_count() == 0:
            return
        remove = self.page.locator(self.s.remove_all)
        if await remove.count() > 0 and await remove.first.is_visible():
            await remove.first.click()
            if not await self._poll(lambda: self._visible(self.s.dialog_ok), ELEMENT_WAIT_MS):
                raise BookmakerError("The betslip could not be emptied (no confirmation appeared).")
            await self.page.locator(self.s.dialog_ok).first.click()
        else:
            for _ in range(MAX_BETSLIP_ROWS):
                remaining = await self.betslip_count()
                if remaining == 0:
                    break
                icon = self.page.locator(self.s.item_delete)
                if await icon.count() == 0:
                    raise BookmakerError("The betslip could not be emptied (no delete control was found).")
                await icon.first.click()

                async def shrank(before=remaining) -> bool:
                    return await self.betslip_count() < before

                if not await self._poll(shrank, SETTLE_MS):
                    raise BookmakerError("The betslip could not be emptied.")

        async def empty() -> bool:
            return await self.betslip_count() == 0

        if not await self._poll(empty, SETTLE_MS):
            raise BookmakerError("The betslip could not be emptied.")

    async def add_outcome(self, event_ref: str, market_titles: Sequence[str], outcome: str) -> None:
        """Open the game and click ONE outcome cell (exactly one must match), then check the betslip grew by one."""
        try:
            await self.page.goto(self.s.event_page(event_ref), wait_until="domcontentloaded", timeout=PAGE_LOAD_MS)
        except Exception as error:
            if type(error).__name__ != "TimeoutError":   # a crashed browser or driver is NOT a pick problem: fail the load
                raise
            raise PickNotAdded("The game's page did not load in time.") from None
        before = await self.betslip_count()
        chosen = ""

        async def found() -> bool:
            nonlocal chosen
            for title in market_titles:
                selector = self.s.outcome_cell(title, outcome)
                if await self._count(selector) == 1:
                    chosen = selector
                    return True
            return False

        if not await self._poll(found, ELEMENT_WAIT_MS):
            raise PickNotAdded("The selected option was not found on the game's page.")
        await self.page.locator(chosen).first.click()

        async def grew() -> bool:
            return await self.betslip_count() == before + 1

        if not await self._poll(grew, SETTLE_MS):
            # One more look: a first click that only focused the cell, or a pop-up in the way. Once only; the count is
            # checked again, so a selection can never be added twice.
            state = ""
            try:
                state = " ".join(str(await self.page.locator(chosen).first.evaluate(_CELL_STATE_JS)).split())[:200]
            except Exception:
                pass
            locked = any(word in state.lower() for word in ("lock", "disabled", "suspend"))
            if not locked and await self.betslip_count() == before:
                try:
                    await self.page.keyboard.press("Escape")
                    await self.page.locator(chosen).first.click()
                except Exception:
                    pass
                if await self._poll(grew, SETTLE_MS // 2):
                    return
            raise PickNotAdded(f"A selection did not appear on the betslip after it was clicked ({outcome} in "
                                 f"{' / '.join(market_titles)}; the option's state: {state or 'unknown'}"
                                 f"{'; it looks locked or suspended on SportyBet right now' if locked else ''}).")

    async def read_betslip(self) -> BetslipReading:
        """Every row of the real betslip, its total odds and the stake box, exactly as the page shows them."""
        items = self.page.locator(self.s.betslip_items)
        count = await items.count()
        lines: list[BetslipLine] = []
        for index in range(count):
            item = items.nth(index)
            odds = _parse_decimal(await self._text(item, self.s.item_odds))
            if odds is None:
                raise BookmakerError("The odds of a betslip selection could not be read.")
            lines.append(BetslipLine(await self._text(item, self.s.item_teams), await self._text(item, self.s.item_market),
                                     await self._text(item, self.s.item_outcome), odds))
        total: Decimal | None = None
        total_loc = self.page.locator(self.s.total_odds)
        if count > 1 and await total_loc.count() > 0:
            total = _parse_decimal(await total_loc.first.inner_text())
        elif count == 1:
            total = lines[0].odds  # a single selection shows no total-odds label (VERIFIED-ANON): its odds are the total
        stake: Decimal | None = None
        stake_loc = self.page.locator(self.s.stake_input)
        if await stake_loc.count() > 0:
            stake = _parse_decimal(await stake_loc.first.input_value())
        return BetslipReading(tuple(lines), total, stake)

    async def set_stake(self, stake: Decimal) -> None:
        box = self.page.locator(self.s.stake_input)
        if await box.count() == 0:
            raise BookmakerError("The stake box was not found on the betslip.")
        await box.first.fill(format(Decimal(stake), "f"))
        await self.page.wait_for_timeout(POLL_MS)

    # ---- placing
    async def click_place(self) -> None:
        """The ONLY place the Place button is clicked. A driver built without allow_place refuses."""
        if not self._allow_place:
            raise BookmakerError("Dry run: the Place button is never clicked.")
        self._listen_for_place_answers()
        self._place_pale = None
        await self.page.locator(self.s.place_button).first.click()

    def _listen_for_place_answers(self) -> None:
        """Diagnostic only: remember what SportyBet's SERVER answers to the requests made around the Place click (the
        reason a bet is refused is in that answer). Never raises, never changes the click."""
        self._place_answers = []
        on = getattr(self.page, "on", None)
        if on is None:
            return

        async def remember(response) -> None:
            try:
                url = str(response.url)
                low = url.lower()
                if "/api/" in low and any(word in low for word in ("order", "bet", "place", "ticket", "slip")):
                    try:
                        body = " ".join((await response.text()).split())[:220]
                    except Exception:
                        body = ""
                    path = url.split("?")[0].split("/api/", 1)[-1]
                    self._place_answers.append(f"/api/{path} -> HTTP {response.status} {body}".strip())
            except Exception:
                pass

        async def remember_any(response) -> None:
            # Wider net: ANY script-made (xhr/fetch) request after the click, whatever its address, so a bet request under an
            # unexpected name is still seen. Addresses only (no bodies, no headers, no cookies).
            try:
                if response.request.resource_type in ("xhr", "fetch"):
                    url = str(response.url).split("?")[0]
                    if len(self._place_requests) < 12:
                        self._place_requests.append(f"{response.request.method} {url[-90:]} -> {response.status}")
            except Exception:
                pass

        def navigated(frame) -> None:
            # Diagnostic only: did the page reload or navigate after the click (the owner sees the screen go white)? Path only.
            try:
                main = getattr(self.page, "main_frame", None)
                if main is not None and frame is not main:
                    return
                if len(self._place_nav) < 6:
                    self._place_nav.append("navigated " + (urlsplit(str(frame.url)).path or "/")[:60])
            except Exception:
                pass

        def loaded(kind: str):
            def handler(*_args) -> None:
                if len(self._place_nav) < 6:
                    self._place_nav.append(kind)
            return handler

        def new_document(request) -> None:
            try:
                if request.resource_type == "document" and len(self._place_nav) < 6:
                    self._place_nav.append("document request " + (urlsplit(str(request.url)).path or "/")[:60])
            except Exception:
                pass

        self._place_nav = []
        self._place_requests = []
        self._place_console = []
        try:
            on("framenavigated", navigated)
            on("load", loaded("load event"))
            on("domcontentloaded", loaded("domcontentloaded event"))
            on("request", new_document)
        except Exception:
            pass
        try:
            on("response", lambda response: asyncio.ensure_future(remember(response)))
            on("response", lambda response: asyncio.ensure_future(remember_any(response)))
            on("console", lambda message: self._place_console.append(str(message.text)[:120])
               if getattr(message, "type", "") == "error" and len(self._place_console) < 3 else None)
        except Exception:
            pass

    async def dismiss_popups(self) -> list[str]:
        """Close promotional pop-ups (VERIFIED-LIVE 2026-10-01: a 'FLASHWIN' win celebration covered the page while a bet
        was being placed and the Place click did nothing). Only a pop-up whose text has nothing to do with betting is
        touched: anything that mentions confirm, place bet, stake or odds is left exactly as it is. Escape first, then
        the pop-up's own close control. Returns the start of what was closed; never raises."""
        closed: list[str] = []
        for _ in range(3):
            try:
                text = " ".join(str(await self.page.evaluate(_OBSERVE_JS) or "").split())
                if not text or any(word in text.lower() for word in _BETTING_WORDS):
                    break
                await self.page.keyboard.press("Escape")
                await self.page.wait_for_timeout(300)
                if " ".join(str(await self.page.evaluate(_OBSERVE_JS) or "").split()):
                    close = self.page.locator(self.s.popup_close)
                    if await close.count() > 0 and await close.first.is_visible():
                        await close.first.click()
                        await self.page.wait_for_timeout(300)
                closed.append(text[:80])
            except Exception:
                break
        return closed

    async def observe_after_click(self, screenshot_dir: Path, stamp: str, label: str) -> str:
        """What the page shows right after Place (or after its confirmation): the text of any visible dialog, toast or
        message, what SportyBet's server answered to the order requests, plus a local screenshot (never sent anywhere).
        Diagnostic only: it never raises and never clicks, so it cannot change what happened. The text is clipped and
        goes to the owner through the redacting senders."""
        try:
            text = await self.page.evaluate(_OBSERVE_JS)
            text = " | ".join(str(text or "").split("\n"))[:400]
        except Exception:
            text = ""
        answers = getattr(self, "_place_answers", [])
        if answers:
            text = (text + " | " if text else "") + "server: " + " ; ".join(answers[-3:])[:420]
        else:
            seen = getattr(self, "_place_requests", [])
            text = (text + " | " if text else "") + ("requests after Place: " + " ; ".join(seen[-5:])[:420] if seen
                                                     else "no request was sent to SportyBet after the Place click")
        try:
            controls = str(await self.page.evaluate(_CONTROLS_JS) or "")
            if controls:
                text += " | visible controls: " + controls[:200]
        except Exception:
            pass
        if hasattr(self, "_place_nav"):
            nav = self._place_nav
            text += (" | the page reloaded/navigated after Place (" + "; ".join(nav) + ")" if nav
                     else " | the page did not reload after Place")
        text += await self._slip_state_note()
        errors = getattr(self, "_place_console", [])
        if errors:
            text += " | page errors: " + " ; ".join(errors)[:200]
        try:
            button = self.page.locator(self.s.place_button).first
            if await button.count():
                text += f" | Place button now: {'disabled' if await button.is_disabled() else 'enabled'}"
        except Exception:
            pass
        if getattr(self, "_place_pale", None):
            text += f" | Place button stayed pale/disabled for the whole {self._place_wait_ms // 1000} s wait"
        try:
            if not await self.logged_out():  # a login page would show the phone number
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                await self.page.screenshot(path=str(screenshot_dir / f"after-click-{label}-{stamp}.png"))
        except Exception:
            pass
        return text

    async def _slip_state_note(self) -> str:
        """One cheap read for the diagnostics: still logged in, the stake box value, the legs still on the slip. Never raises,
        never clicks, never changes a decision."""
        parts = []
        try:
            parts.append("logged out" if await self.logged_out() else "still logged in")
        except Exception:
            pass
        try:
            box = self.page.locator(self.s.stake_input)
            if await box.count() > 0:
                parts.append("stake box holds " + (str(await box.first.input_value()).strip() or "nothing"))
            else:
                parts.append("no stake box")
        except Exception:
            pass
        try:
            parts.append(f"{await self.betslip_count()} leg(s) on the slip")
        except Exception:
            pass
        return " | after the wait: " + ", ".join(parts) if parts else ""

    async def confirm_placement(self) -> str:
        """After Place, a confirmation dialog may appear (UNVERIFIED). Confirm it only if its text shows no sign that the
        price moved or the bet was refused. Returns the dialog text ('' when no dialog appeared)."""
        if not self._allow_place:
            raise BookmakerError("Dry run: nothing is confirmed.")
        if not await self._poll(lambda: self._visible(self.s.dialog), 6_000):
            return await self._confirm_by_button()
        text = (await self.page.locator(self.s.dialog).first.inner_text()).strip()
        lowered = text.lower()
        if any(word in lowered for word in self.s.dialog_refusal_words):
            raise BookmakerError("SportyBet showed a warning after Place; nothing was confirmed.")
        ok = self.page.locator(self.s.confirm_dialog_ok)
        if await ok.count() > 0:
            await ok.first.click()
        return text

    async def finish_placement(self) -> str:
        """After the Place click, watch the page (up to PLACE_WAIT_SECONDS) for what SportyBet shows, in the order it happens
        (VERIFIED-LIVE 2026-10-01 from the owner's screenshots): a 'Confirm' panel ('About to pay NGN x' with Cancel and
        Confirm) which is pressed ONCE after its amount and text are checked, then a 'Submission Successful' pop-up (total
        stake, potential win, booking code, OK). Returns the success pop-up's text, closing it with OK; '' when none came.
        Also remembers whether the Place button stayed pale/disabled at every look (diagnostics only).
        Never presses Place again."""
        pressed = False
        waited = 0
        looks = pale = 0
        self._place_pale = None
        while waited < self._place_wait_ms:
            try:
                success = str(await self.page.evaluate(_SUCCESS_JS) or "")
            except Exception:
                success = ""
            if success:
                try:
                    await self.page.evaluate(_SUCCESS_OK_JS)
                except Exception:
                    pass
                return " ".join(success.split())[:400]
            if not pressed:
                try:
                    if await self._confirm_control_visible():
                        pressed = True
                        await self._confirm_by_button()
                        continue
                    if await self._visible(self.s.dialog):
                        pressed = True
                        await self.confirm_placement()
                        continue
                except BookmakerError:
                    raise
                except Exception:
                    pass
            if waited % 1_000 < self._poll_ms:   # about once a second
                state = await self._place_button_pale()
                if state is not None:
                    looks += 1
                    pale += 1 if state else 0
            await self.page.wait_for_timeout(self._poll_ms)
            waited += self._poll_ms
        self._place_pale = looks > 0 and pale == looks
        return ""

    async def _place_button_pale(self) -> bool | None:
        """True when the Place button is disabled or marked disabled, False when usable, None when it cannot be told."""
        try:
            button = self.page.locator(self.s.place_button).first
            if not await button.count():
                return None
            if await button.is_disabled():
                return True
            return "disabled" in str(await button.get_attribute("class") or "").lower()
        except Exception:
            return None

    # ---- the Book Bet fallback (owner-approved 2026-10-01): only a booking code, never proof of a bet
    async def book_for_code(self) -> str:
        """Press the slip's 'Book Bet' link ONCE (found by its text; it never presses Place or Confirm), read the booking code
        SportyBet shows, close the pop-up. Returns the code, or '' when none could be read. Never raises."""
        try:
            before = set(_book_codes(str(await self.page.evaluate(_BOOK_TEXT_JS) or "")))
            if not await self.page.evaluate(_BOOK_CLICK_JS):
                return ""
            code = ""
            waited = 0
            while waited <= BOOK_WAIT_MS and not code:
                await self.page.wait_for_timeout(self._poll_ms)
                waited += self._poll_ms
                fresh = [c for c in _book_codes(str(await self.page.evaluate(_BOOK_TEXT_JS) or "")) if c not in before]
                code = fresh[0] if fresh else ""
            await self._close_booking_popup()
            return code
        except Exception:
            return ""

    async def _close_booking_popup(self) -> None:
        try:
            await self.page.keyboard.press("Escape")
            await self.page.wait_for_timeout(300)
            await self.page.evaluate(_BOOK_CLOSE_JS)
        except Exception:
            pass

    async def _confirm_by_button(self) -> str:
        """VERIFIED-LIVE 2026-10-01 by the owner: after Place there IS a confirming step, and it is not the .es-dialog-wrap
        dialog. Find a visible control whose whole text is 'Confirm' (button, link or role=button), read the text around it,
        refuse on any refusal word (price changed, suspended, ...), otherwise press it once. '' when no such control."""
        try:
            found = await self._poll(lambda: self._confirm_control_visible(), 4_000)
            if not found:
                return ""
            context = str(await self.page.evaluate(_CONFIRM_CONTEXT_JS) or "")
            context = " ".join(context.split())[:400]
            if any(word in context.lower() for word in self.s.dialog_refusal_words):
                raise BookmakerError("SportyBet showed a warning after Place; nothing was confirmed.")
            expected = getattr(self, "expected_stake", None)
            if expected is not None:
                shown = re.search(r"about to pay\s*(?:NGN)?\s*([\d,]+(?:\.\d+)?)", context, re.IGNORECASE)
                if shown is None or Decimal(shown.group(1).replace(",", "")) != Decimal(expected):
                    raise BookmakerError("The confirm step does not show the approved amount; nothing was confirmed.")
            await self.page.evaluate(_CONFIRM_CLICK_JS)
            await self.page.wait_for_timeout(1_000)
            return context or "Confirm pressed"
        except BookmakerError:
            raise
        except Exception:
            return ""

    async def _confirm_control_visible(self) -> bool:
        return bool(await self.page.evaluate(_CONFIRM_FIND_JS))

    # ---- bet history (a second page, so the betslip page is never navigated away)
    async def read_history(self, unsettled: bool = False) -> tuple[HistoryBet, ...]:
        """The FIRST page of the bet history, newest first (5 bets), read by the real selectors. `unsettled` reads the
        running-bets-only list instead (?isSettled=0). Fails closed: no rows and no recognised empty marker, a row
        without its expected parts, or an unreadable field is BookmakerError 'could not be read'."""
        page = await self.context.new_page()
        try:
            url = self.s.history_url_unsettled if unsettled else self.s.history_url
            await page.goto(url, wait_until="domcontentloaded", timeout=PAGE_LOAD_MS)
            await self._login_if_redirected(page, url)
            items, empty = page.locator(self.s.history_items), page.locator(self.s.history_empty)
            seen = 0
            for _ in range(HISTORY_WAIT_MS // self._poll_ms + 1):
                seen = await items.count()
                if seen or await empty.count():
                    break
                await page.wait_for_timeout(self._poll_ms)
            for _ in range(8):   # rows may still be rendering: wait until two reads in a row agree
                await page.wait_for_timeout(POLL_MS)
                again = await items.count()
                if again == seen:
                    break
                seen = again
            if seen == 0:
                if await empty.count() == 0:
                    raise BookmakerError("The bet history could not be read (no entries and no empty-list marker).")
                return ()
            return build_history([await self._read_row(items.nth(index)) for index in range(seen)], self.s)
        finally:
            await page.close()

    async def _login_if_redirected(self, page, url: str) -> None:
        """VERIFIED-LIVE 2026-10-01: a stored session can look logged in on the home page while the history page
        (a stricter, desktop page) redirects to /ng/login?ret=... In that case log in right on that page, using the
        owner's details, wait for SportyBet to leave the login page, then make sure the history page is showing.
        No login details or a login that does not complete is a plain BookmakerError (fail closed)."""
        # The redirect happens a moment AFTER the page loads (client-side), so wait until the history rows, the empty
        # marker or the login page show up before deciding.
        for _ in range(REDIRECT_WAIT_MS // self._poll_ms + 1):
            if self.s.login_path in str(getattr(page, "url", "")):
                break
            if await page.locator(self.s.history_items).count() or await page.locator(self.s.history_empty).count():
                return
            await page.wait_for_timeout(self._poll_ms)
        else:
            return  # neither: the normal wait and the fail-closed error below handle it
        if self._credentials is None:
            raise BookmakerError("SportyBet asked for a login and no login details are available.")
        # The full login page differs from the home-page pop-up: the password box has no name, the button is 'Login'.
        await page.locator(self.s.login_phone).first.fill(self._credentials.national_phone())
        await page.locator(self.s.login_page_password).first.fill(self._credentials.password)
        await page.locator(self.s.login_page_button).first.click()
        for _ in range(LOGIN_DONE_WAIT_MS // self._poll_ms + 1):
            if self.s.login_path not in str(getattr(page, "url", "")):
                break
            await page.wait_for_timeout(self._poll_ms)
        else:
            raise BookmakerError("The SportyBet login did not complete (wrong details, a code or a captcha was asked for).")
        if self.s.history_path not in str(getattr(page, "url", "")):
            await page.goto(url, wait_until="domcontentloaded", timeout=PAGE_LOAD_MS)
        await self.save_state()

    async def _read_row(self, row) -> "RawHistoryRow":
        async def one(selector: str) -> str:
            loc = row.locator(selector)
            if await loc.count() != 1:
                raise BookmakerError("The bet history could not be read (an entry did not look as expected).")
            return await loc.first.inner_text()

        vs = row.locator(self.s.history_vs)
        lines = []
        for index in range(await vs.count()):
            line = vs.nth(index)
            lines.append((await line.inner_text(), await line.locator(self.s.history_vs_text).count() > 0))
        return RawHistoryRow(await one(self.s.history_time), await one(self.s.history_type),
                             await one(self.s.history_status), tuple(lines), await one(self.s.history_stake),
                             await one(self.s.history_return))

    async def screenshot(self, path: Path) -> None:
        await self.page.screenshot(path=str(path))

    async def login_screenshot(self, path: Path) -> None:
        """A screenshot with EVERY input box masked, so a phone number or password shown in one never reaches the picture."""
        await self.page.screenshot(path=str(path), mask=[self.page.locator("input")])

    async def login_diagnostics(self) -> str:
        """What on the page matches the login-form selector: tag, name, type, class, whether it is visible, where it is and its
        parent chain. Structure only: no field value is read, so nothing typed can end up in the note."""
        info = await self.page.evaluate("""(sel) => Array.from(document.querySelectorAll(sel)).map(e => {
            const r = e.getBoundingClientRect(); const cs = getComputedStyle(e); let chain = []; let p = e.parentElement;
            for (let i = 0; i < 6 && p; i++) { chain.push((p.tagName + '.' + (p.className || '')).slice(0, 60)); p = p.parentElement; }
            return {tag: e.tagName, name: e.getAttribute('name'), type: e.getAttribute('type'), cls: (e.className || '').slice(0, 60),
                    visible: r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' && cs.display !== 'none',
                    box: [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)], chain: chain};
        })""", self.s.login_phone)
        lines = [f"url: {str(getattr(self.page, 'url', ''))[:120]}", f"selector: {self.s.login_phone}", f"matches: {len(info)}"]
        lines += [str(item) for item in info]
        return "\n".join(lines) + "\n"

    async def close(self) -> None:
        """Close the context, the browser and the Playwright driver; a failure to close is never an error."""
        for closer in (self.context.close, getattr(self._browser, "close", None), getattr(self._playwright, "stop", None)):
            if closer is None:
                continue
            try:
                await closer()
            except Exception:
                pass


@dataclass(frozen=True)
class RawHistoryRow:
    """The text of the parts of one `div.m-order-list`, as read from the page (nothing interpreted yet)."""
    time: str
    bet_type: str
    status: str
    vs_lines: tuple[tuple[str, bool], ...]   # (text of a div.m-order-vs, whether it holds the 'v' span)
    stake: str
    total_return: str


SPORTYBET_UTC_OFFSET = timedelta(hours=1)   # SportyBet Nigeria shows Lagos time (UTC+1, no daylight saving) VERIFIED-LIVE


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _parse_history_time(text: str, selectors: Selectors = SELECTORS) -> datetime | None:
    """'30/09/2026 06:44' (site time, GMT+01:00) as a UTC-aware datetime, or None when it is not exactly that."""
    found = re.match(selectors.history_time_format, _squash(text))
    if not found:
        return None
    d, mo, y, h, mi = (int(x) for x in found.groups())
    try:
        return datetime(y, mo, d, h, mi, tzinfo=timezone(SPORTYBET_UTC_OFFSET)).astimezone(timezone.utc)
    except ValueError:
        return None


def _parse_money(text: str, selectors: Selectors, allow_zero: bool = False) -> Decimal | None:
    found = re.match(selectors.history_money, _squash(text).replace(",", ""))
    if not found:
        return None
    value = Decimal(found.group(1))
    return value if value > 0 or (allow_zero and value == 0) else None


def history_key(placed_at: datetime, bet_type: str, stake: Decimal, shown: Sequence[str], others: int) -> str:
    """The SYNTHETIC key of an entry (the page shows no bet id): the minute it was placed (site time), the type, the
    stake, a short digest of the shown games in the order shown, and the 'N other' count. The status is NOT part of it, so
    the key survives the bet settling. Equal keys for two entries make both ambiguous (see build_history)."""
    local = placed_at.astimezone(timezone(SPORTYBET_UTC_OFFSET))
    plain = format(Decimal(stake).normalize(), "f")
    canonical = "|".join([local.strftime("%Y%m%d%H%M"), bet_type.casefold(), plain,
                          ";".join(normalize_team(m) for m in shown), str(others)])
    digest = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:8]
    return f"{local.strftime('%Y%m%d%H%M')}-{(bet_type[:1] or '?').upper()}{plain}-{digest}"


def history_bet_from_row(row: RawHistoryRow, selectors: Selectors = SELECTORS) -> HistoryBet:
    """One entry's parts to a HistoryBet. Raises BookmakerError ('could not be read') when any expected part is missing or
    not in the layout seen live: never a guess."""
    bad = BookmakerError("The bet history could not be read (an entry did not look as expected).")
    placed_at = _parse_history_time(row.time, selectors)
    stake = _parse_money(row.stake, selectors)
    bet_type, status = _squash(row.bet_type), _squash(row.status)
    if placed_at is None or stake is None or not bet_type or not status:
        raise bad
    shown: list[str] = []
    others = 0
    for index, (text, has_v) in enumerate(row.vs_lines):
        text = _squash(text)
        if has_v:
            if others or " v " not in f" {text} ":
                raise bad   # a game line after the 'other matches' line, or a line that is not 'Home v Away'
            shown.append(text)
            continue
        found = re.match(selectors.history_other_matches, text, re.IGNORECASE)
        if not found or others or index != len(row.vs_lines) - 1:
            raise bad
        others = int(found.group(1))
        if others < 1:
            raise bad
    if not shown:
        raise bad
    raw_return = _squash(row.total_return)
    returned = None if raw_return in ("", "--") else _parse_money(raw_return, selectors, True)
    if returned is None and raw_return not in ("", "--"):
        raise bad
    key = history_key(placed_at, bet_type, stake, shown, others)
    text = f"{_squash(row.time)} | {bet_type} | {status} | " + "; ".join(shown) + (f"; and {others} other" if others else "")
    return HistoryBet(key, "", stake, None, (), text, len(shown) + others, placed_at, tuple(shown), status, bet_type,
                      False, returned)


def build_history(rows: Sequence[RawHistoryRow], selectors: Selectors = SELECTORS) -> tuple[HistoryBet, ...]:
    """Every entry as a HistoryBet, newest first as shown. Entries whose synthetic key is not unique are ALL flagged
    `ambiguous`: two genuinely different bets can never be told apart silently, and neither can prove anything."""
    bets = [history_bet_from_row(row, selectors) for row in rows]
    counts: dict[str, int] = {}
    for bet in bets:
        counts[bet.bet_id] = counts.get(bet.bet_id, 0) + 1
    return tuple(replace(bet, ambiguous=True) if counts[bet.bet_id] > 1 else bet for bet in bets)


def history_bet_is_ours(bet: HistoryBet, token: PlacementToken, created_at: datetime | None = None) -> bool:
    """A history entry is the bet we just placed when its games, stake, number of legs, status ('Running') and time match
    the token (see core.betting.history_bet_matches). Odds and picks are NOT in the entry: the exact pre-click betslip
    check is the guard for those."""
    return history_bet_matches(bet, token.legs, token.stake, created_at)


# ================================================================== the real browser

async def _skip_heavy_resources(route) -> None:
    """Abort pictures, fonts and media; let everything else (HTML, CSS, scripts, XHR/fetch, the rest) through untouched."""
    if route.request.resource_type in BLOCKED_RESOURCE_TYPES:
        await route.abort()
    else:
        await route.continue_()


class PlaywrightSiteFactory:
    """Starts headless Chromium with the stored login state (if any). One call, one browser; the adapter closes it."""

    def __init__(self, credentials: Credentials | None, storage_dir: Path = SECRETS_DIR, allow_place: bool = False,
                 headless: bool = True, fast: bool = True, browser_mode: str = "standard",
                 place_wait_seconds: int = DEFAULT_PLACE_WAIT_S):
        self._credentials, self._storage_dir = credentials, storage_dir
        self._allow_place, self._headless = allow_place, headless
        self._fast = bool(fast)   # FAST_BROWSER: no pictures, fonts or media; quicker looks while waiting. False = as before
        self._real = browser_mode == "real"   # BROWSER_MODE=real: the installed Chrome, visible, automation flags hidden
        self._place_wait_s = int(place_wait_seconds)

    async def __call__(self) -> SportyBetSite:
        from playwright.async_api import async_playwright  # imported here: tests and the core never need a browser

        folder = check_secrets_dir(self._storage_dir)
        state = folder / STATE_FILE_NAME
        playwright = await async_playwright().start()
        browser = None
        try:
            if self._real:
                try:
                    browser = await playwright.chromium.launch(
                        channel="chrome", headless=False, args=list(REAL_BROWSER_ARGS),
                        ignore_default_args=list(REAL_BROWSER_IGNORE_DEFAULT_ARGS))
                except Exception:
                    raise BookmakerError(CHROME_MISSING_TEXT) from None   # never a silent fall back to another browser
                options = {"viewport": VIEWPORT}   # Chrome's own User-Agent: none is set here
            else:
                browser = await playwright.chromium.launch(headless=self._headless)
                options = {"viewport": VIEWPORT, "user_agent": USER_AGENT}
            if state.exists():
                options["storage_state"] = str(state)
            context = await browser.new_context(**options)
            if self._real:
                await context.add_init_script(HIDE_WEBDRIVER_JS)
            if self._fast:
                await context.route("**/*", _skip_heavy_resources)   # every page of this context, the history page too
            page = await context.new_page()
        except BaseException:
            if browser is not None:
                await browser.close()
            await playwright.stop()
            raise
        return SportyBetSite(page, context, self._credentials, state, self._allow_place, browser=browser,
                             playwright=playwright, poll_ms=FAST_POLL_MS if self._fast else POLL_MS,
                             place_wait_ms=self._place_wait_s * 1000)


# ================================================================== the Bookmaker

SiteFactory = Callable[[], Awaitable[SportyBetSite]]


class SportyBetBookmaker:
    """Implements core.ports.Bookmaker for SportyBet. Build ONE per bot: it owns the lock (AD-17)."""

    def __init__(self, credentials: Credentials | None = None, dry_run: bool = True,
                 fixture_source: FixtureSource | None = None, site_factory: SiteFactory | None = None,
                 storage_dir: Path = SECRETS_DIR, screenshot_dir: Path = SCREENSHOT_DIR,
                 clock: Callable[[], datetime] | None = None, headless: bool = True,
                 lock: asyncio.Lock | None = None,
                 sport_fixture_sources: dict[str, FixtureSource] | None = None,
                 event_markets_source: EventMarketsSource | None = None, fast_browser: bool = False,
                 browser_mode: str = "standard", place_wait_seconds: int = DEFAULT_PLACE_WAIT_S,
                 place_mode: str = "place"):
        if place_mode not in ("book", "place"):
            raise ValueError("place_mode must be 'book' or 'place'")
        self._place_mode = place_mode   # AD-25: 'book' = the final action is a booking code; Place/Confirm are never clicked
        self._dry_run = bool(dry_run)  # DRY_RUN defaults to true; only an explicit False places real bets
        self._fixtures = fixture_source or SportyBetFixtures()  # soccer
        self._sport_fixtures: dict[str, FixtureSource] = dict(sport_fixture_sources or {})  # other sports, by name
        self._event_markets: EventMarketsSource = event_markets_source or SportyBetEventMarkets()  # corners/cards (spec 11-1)
        self._site_factory: SiteFactory = site_factory or PlaywrightSiteFactory(
            credentials, storage_dir, allow_place=not self._dry_run and place_mode == "place", headless=headless, fast=fast_browser,
            browser_mode=browser_mode, place_wait_seconds=place_wait_seconds)
        self._place_wait_s = int(place_wait_seconds)
        self._screenshot_dir = Path(screenshot_dir)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = lock if lock is not None else asyncio.Lock()   # shared with the read-only readers (serialize.py)
        self._site: SportyBetSite | None = None
        self._used_tokens: set[str] = set()
        self._clicked = False  # True once the Place click was attempted in the current place() call

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @property
    def place_mode(self) -> str:
        return self._place_mode

    @property
    def lock(self) -> asyncio.Lock:
        """The one lock around every SportyBet operation. Anything else that needs the betslip takes it too."""
        return self._lock

    # ------------------------------------------------------------ helpers
    async def _confirm_logged_in(self, site, where: str) -> bool:
        """True when the page is logged in. The login form was seen to flash up on a logged-in page for a moment (a false
        'not logged in' that stopped a bet), so a showing form is checked again LOGIN_RECHECKS times, LOGIN_RECHECK_PAUSE_S apart;
        only a form that is STILL there means logged out. The first time it shows, a screenshot and a structure note (no values)
        are saved locally in the screenshots folder (never sent anywhere) so the cause can be found."""
        for attempt in range(LOGIN_RECHECKS + 1):
            if not await site.logged_out():
                if attempt:
                    log.info("the login form disappeared on re-check %d (%s): continuing", attempt, where)
                return True
            if attempt == 0:
                await self._save_login_diagnostics(site, where)
            if attempt < LOGIN_RECHECKS:
                await asyncio.sleep(LOGIN_RECHECK_PAUSE_S)
        log.warning("the login form was still showing after %d re-checks (%s)", LOGIN_RECHECKS, where)
        return False

    async def _save_login_diagnostics(self, site, where: str) -> None:
        try:
            self._screenshot_dir.mkdir(parents=True, exist_ok=True)
            stamp = self._clock().strftime("%Y%m%dT%H%M%S%f")
            base = self._screenshot_dir / f"logincheck-{where}-{stamp}"
            shoot = getattr(site, "login_screenshot", None)
            if shoot is not None:
                await shoot(base.with_suffix(".png"))
            describe = getattr(site, "login_diagnostics", None)
            if describe is not None:
                base.with_suffix(".txt").write_text(await describe(), encoding="utf-8")
            log.info("the login form showed (%s): screenshot and note saved as %s", where, base.name)
        except Exception:
            pass   # a diagnostic is a courtesy; it must never hide or change the real outcome

    @staticmethod
    async def _lost_session(site) -> bool:
        """True when the page now shows the account logged out. Never raises (a page that cannot be read is not 'logged out')."""
        try:
            return bool(await site.logged_out())
        except Exception:
            return False

    async def _fail(self, operation: str) -> None:
        """Save a local screenshot (never sent anywhere) and close the browser so the next operation starts clean."""
        site, self._site = self._site, None
        if site is None:
            return
        try:
            if await site.logged_out():
                raise RuntimeError("login form showing")  # a screenshot would show the phone number: skip it
            self._screenshot_dir.mkdir(parents=True, exist_ok=True)
            stamp = self._clock().strftime("%Y%m%dT%H%M%S%f")
            await site.screenshot(self._screenshot_dir / f"failure-{operation}-{stamp}.png")
        except Exception:
            pass  # a screenshot is a courtesy; it must never hide the real failure
        await site.close()

    async def _close(self) -> None:
        site, self._site = self._site, None
        if site is not None:
            await site.close()

    def _source_for(self, sport: str) -> FixtureSource:
        """The game list of one sport. Soccer is the injected source; any other sport is built on first use."""
        if sport == "soccer":
            return self._fixtures
        if sport not in self._sport_fixtures:
            self._sport_fixtures[sport] = SportyBetFixtures(sport=sport)
        return self._sport_fixtures[sport]

    async def _assess(self, legs: Sequence[Leg]) -> tuple[LegAvailability, ...]:
        sports = sorted({leg.match_id.sport for leg in legs}) or ["soccer"]
        fixtures = []
        for sport in sports:  # one listing per sport that has a leg; a soccer-only slip reads exactly one, as before
            try:
                fixtures.extend(await self._source_for(sport).fetch_fixtures())
            except FixtureFetchError as error:
                raise BookmakerError(f"SportyBet's game list could not be read: {error}") from None
            except ValueError:
                raise BookmakerError("SportyBet's game list is not available for one of the sports on this slip") from None
        fixtures = await self._with_event_markets(legs, fixtures)
        return assess_availability(legs, fixtures, self._clock())

    async def _with_event_markets(self, legs: Sequence[Leg], fixtures: list) -> list:
        """Corners and cards (spec 11-1) are not in the game list, only on the per-event endpoint: for the games that have
        such a leg, look the game's markets up again so the leg's option can be found. A failed lookup leaves the game as it
        is, so the option is reported missing (fail closed)."""
        wanted = {str(leg.match_id) for leg in legs if leg.market_id in WANTED_MARKET_IDS and leg.match_id.sport == "soccer"}
        targets = [f for f in fixtures if str(f.match_id) in wanted and f.bookmaker_ref]
        if not targets:
            return fixtures
        found = await self._event_markets.fetch_event_markets(targets)
        merged = []
        for fixture in fixtures:
            extra = found.get(fixture.match_id) if fixture in targets else None
            if extra:
                fixture = _replace(fixture, markets=(*fixture.markets, *extra))
            merged.append(fixture)
        return merged

    @staticmethod
    def _page_market_name(found: LegAvailability) -> str:
        """The market's name as the game page and the betslip write it. The listing says "Home O/U" for a team total;
        the page and the betslip say "<Team> Over/Under" (VERIFIED-ANON). Other markets use the listing's name."""
        leg, fixture = found.leg, found.fixture
        if leg.market_id in ("19", "20"):
            return SELECTORS.team_total_title.format(team=fixture.home if leg.market_id == "19" else fixture.away)
        if leg.match_id.sport == "soccer":   # spec 11-1, VERIFIED-ANON titles (see Selectors)
            soccer_titles = {"10": SELECTORS.double_chance_title, "166": SELECTORS.corners_title, "139": SELECTORS.cards_title}
            if leg.market_id in soccer_titles:
                return soccer_titles[leg.market_id]
        # American football uses the same SportyBet market ids and names as basketball (UNVERIFIED on the game page).
        if leg.match_id.sport in ("basketball", "american_football"):
            return SportyBetBookmaker._basketball_market_name(found)
        return found.offer.name

    @staticmethod
    def _basketball_market_name(found: LegAvailability) -> str:
        """The title basketball markets carry on the game page and the betslip (see Selectors). The line is taken from
        the listing's own specifier text so it is spelled exactly as SportyBet spells it."""
        leg, fixture = found.leg, found.fixture
        line = found.offer.specifier.partition("=")[2].strip()
        if leg.market_id == BB_WINNER_MARKET:
            return SELECTORS.bb_winner_title
        if leg.market_id == BB_TOTAL_MARKET:
            return SELECTORS.bb_total_title.format(line=line)
        if leg.market_id in (BB_HOME_TOTAL_MARKET, BB_AWAY_TOTAL_MARKET):
            team = fixture.home if leg.market_id == BB_HOME_TOTAL_MARKET else fixture.away
            return SELECTORS.bb_team_total_title.format(team=team, line=line)
        if leg.market_id == BB_FH_TOTAL_MARKET:
            return SELECTORS.bb_fh_total_title
        if leg.market_id == BB_HANDICAP_MARKET:
            return SELECTORS.bb_handicap_title.format(line=line)
        return found.offer.name

    @staticmethod
    def _expected(found: LegAvailability) -> ExpectedLine:
        fixture = found.fixture
        return ExpectedLine(found.leg.key, f"{fixture.home} v {fixture.away}", SportyBetBookmaker._page_market_name(found),
                            found.outcome.name, found.leg.selection, found.leg.odds)

    # ------------------------------------------------------------ Bookmaker port
    async def check_availability(self, legs: Sequence[Leg]) -> tuple[LegAvailability, ...]:
        async with self._lock:
            return await self._assess(legs)

    async def load_slip(self, legs: Sequence[Leg]) -> LoadedSlip:
        """Strict load: every leg must be added, or BookmakerError (a pick that cannot be added fails the load)."""
        attempt = await self._load_guarded(legs, skip_failed=False)
        if attempt.loaded is None or attempt.failed:   # cannot happen with skip_failed=False; fail closed if it does
            raise BookmakerError("A selection could not be added to the betslip. Nothing was placed.")
        return attempt.loaded

    async def load_legs(self, legs: Sequence[Leg]) -> LoadAttempt:
        """Owner request (AD-26): a pick that cannot be added is skipped and reported; the rest are loaded and verified."""
        return await self._load_guarded(legs, skip_failed=True)

    async def _load_guarded(self, legs: Sequence[Leg], skip_failed: bool) -> LoadAttempt:
        if not legs:
            raise BookmakerError("There is nothing to load.")
        if any(is_under(leg) for leg in legs):   # AD-20: fail closed, before the browser is touched
            raise BookmakerError(f"{UNDER_REASON}. Nothing was loaded.")
        if any(is_whole_or_quarter_line(leg) for leg in legs):   # AD-21: fail closed, before the browser is touched
            raise BookmakerError(f"{WHOLE_LINE_REASON}. Nothing was loaded.")
        async with self._lock:
            try:
                return await asyncio.wait_for(self._load(legs, skip_failed), LOAD_TIMEOUT_S)
            except BookmakerError:
                await self._fail("load")
                raise
            except asyncio.TimeoutError:
                await self._fail("load")
                raise BookmakerError("Loading SportyBet took too long. Nothing was placed.") from None
            except Exception as error:  # a page or driver failure: reported, never retried, nothing placed
                await self._fail("load")
                raise BookmakerError(f"Loading SportyBet failed unexpectedly ({type(error).__name__}). "
                                     "Nothing was placed.") from None

    async def _load(self, legs: Sequence[Leg], skip_failed: bool = False) -> LoadAttempt:
        # 1. Every leg must still be available right now (the game list is read again: fail closed).
        checked = await self._assess(legs)
        for result in checked:
            if result.status is not Availability.OK:
                raise BookmakerError("A confirmed game or option is no longer available. Nothing was loaded.")
        await self._close()  # always a fresh browser: nothing is inherited from an earlier run
        site = self._site = await self._site_factory()
        # 2. Log in, then clear the owner's betslip FIRST.
        await site.login()
        if not await self._confirm_logged_in(site, "load"):
            raise BookmakerError("SportyBet is not logged in.")
        await site.clear_betslip()
        if await site.betslip_count() != 0:
            raise BookmakerError("The betslip is not empty after clearing it.")
        # 3. Add exactly the confirmed legs, one at a time, each checked. A pick that cannot be added (and ONLY such a
        #    pick: PickNotAdded) is skipped and remembered when `skip_failed`; any other failure ends the whole load.
        added: list[LegAvailability] = []
        failed: list[FailedLeg] = []
        for result in checked:
            try:
                await site.add_outcome(result.fixture.bookmaker_ref, [self._page_market_name(result)], result.outcome.name)
            except PickNotAdded as error:
                if not skip_failed:
                    raise BookmakerError(str(error)) from None
                failed.append(FailedLeg(result.leg.key, str(error)))
            except Exception:
                # a page or driver failure while adding (a timeout, a page that never settled): when the account was logged out
                # meanwhile, that is the cause, and the owner is told so instead of "unexpected"
                if await self._lost_session(site):
                    raise BookmakerError(SESSION_LOST) from None
                raise
            else:
                added.append(result)
        if await self._lost_session(site):
            raise BookmakerError(SESSION_LOST)
        if not added:
            return LoadAttempt(None, tuple(failed))
        # 4. Read the betslip back and verify it holds exactly the legs that WERE added (anything else: fail closed).
        reading = await site.read_betslip()
        verified = verify_reading([self._expected(r) for r in added], reading)
        if not verified.ok:
            raise BetslipMismatch("The SportyBet betslip does not hold exactly the confirmed games: "
                                  + "; ".join(verified.problems[:3]))
        try:
            return LoadAttempt(make_loaded_slip(verified, reading), tuple(failed))
        except ValueError as error:
            raise BetslipMismatch(f"The SportyBet betslip could not be trusted: {error}") from None

    async def place(self, token: PlacementToken) -> PlacementResult:
        if not isinstance(token, PlacementToken) or not token.matches_contents():
            raise BookmakerError("That is not a valid confirmation. Nothing was placed.")
        if token.booking != (self._place_mode == "book"):
            # AD-25: a booking token can only ever read a code, and a placing token only ever places. Never crossed.
            raise BookmakerError("That confirmation does not match the bot's PLACE_MODE. No bet was placed.")
        async with self._lock:
            if token.token_hash in self._used_tokens:
                raise BookmakerError("That confirmation was already used. Nothing was placed.")
            self._used_tokens.add(token.token_hash)  # single use: consumed from here on, whatever happens
            self._clicked = False
            try:
                result = await asyncio.wait_for(self._place(token), PLACE_TIMEOUT_S + self._place_wait_s)
                await self._close()
                return result
            except _AfterClick as error:
                await self._fail("place")
                raise PlacementUnknown(str(error), error.booking_code, error.booking_tried) from None
            except OddsChanged:
                raise  # the odds moved: nothing was clicked. The page stays open: a NEW confirmation re-reads this betslip
            except BookmakerError:
                await self._fail("place")
                raise
            except asyncio.TimeoutError:
                await self._fail("place")
                if self._clicked:
                    raise PlacementUnknown("SportyBet took too long after Place was clicked. The bet may or may not "
                                           "exist: check your SportyBet bet history.") from None
                raise BookmakerError("Placing took too long before anything was clicked. Nothing was placed.") from None
            except Exception as error:
                await self._fail("place")
                if self._clicked:
                    raise PlacementUnknown(f"Something went wrong after Place was clicked ({type(error).__name__}). "
                                           "The bet may or may not exist: check your SportyBet bet history.") from None
                raise BookmakerError(f"Placing failed unexpectedly ({type(error).__name__}) before anything was "
                                     "clicked. Nothing was placed.") from None

    async def _place(self, token: PlacementToken) -> PlacementResult:
        now = self._clock()
        if token.expired(now):
            raise BookmakerError("That confirmation expired. Nothing was placed.")
        site = self._site
        if site is None:
            raise BetslipMismatch("The betslip is not loaded any more. Nothing was placed.")
        if not await self._confirm_logged_in(site, "place"):
            raise BookmakerError("SportyBet is not logged in. Nothing was placed.")
        expected = [ExpectedLine(leg.leg_key, leg.match_text, leg.market_name, leg.outcome_name, leg.selection,
                                 leg.evaluation_odds) for leg in token.legs]
        prices = [leg.odds for leg in token.legs]
        # 1. The real betslip must be exactly what was confirmed, at exactly the confirmed odds.
        reading0 = await self._check_betslip(site, expected, prices, token, with_stake=False)
        if token.booking:
            return await self._book(site, token, reading0)
        # 2. Put the confirmed stake in.
        await site.set_stake(token.stake)
        before: set[str] = set()
        if not self._dry_run:
            # 3. Snapshot the bet history BEFORE the click (a failure here means nothing was clicked). It is read
            #    BEFORE the last betslip check, so the slow part is out of the window between that check and the click.
            before = {bet.bet_id for bet in await site.read_history()}
        # 4a. The last check, made as close to the click as possible: the betslip with the stake in it.
        reading = await self._check_betslip(site, expected, prices, token, with_stake=True)
        if self._dry_run:
            return PlacementResult(PlacementKind.DRY_RUN, self._report(token, reading))
        # 4a'. A promotional pop-up (for example a win celebration) can cover the page and swallow the click: close it.
        dismiss = getattr(site, "dismiss_popups", None)
        if dismiss is not None:
            try:
                await dismiss()
            except Exception:
                pass
        # 4b. The irreversible part. From the click on, any doubt is PlacementUnknown and nothing is ever retried.
        self._clicked = True
        try:
            try:
                site.expected_stake = Decimal(token.stake)   # the confirm step must show exactly this amount
            except Exception:
                pass
            await site.click_place()
            stamp = self._clock().strftime("%Y%m%dT%H%M%S%f")
            seen = await self._observe(site, stamp, "click")
            finish = getattr(site, "finish_placement", None)
            dialog = await (finish() if finish is not None else site.confirm_placement())
            seen_after = await self._observe(site, stamp, "confirm")
            observed = " || ".join(part for part in (seen, seen_after) if part)
            return await self._prove(site, token, before, dialog, now, observed)
        except _AfterClick:
            raise
        except Exception as error:
            raise _AfterClick(f"Something went wrong after Place was clicked ({type(error).__name__}). "
                              "The bet may or may not exist: check your SportyBet bet history.") from None

    async def _book(self, site: SportyBetSite, token: PlacementToken, reading: BetslipReading) -> PlacementResult:
        """AD-25: the final action in PLACE_MODE=book. The betslip was just verified (legs and odds). No stake is typed and
        neither Place nor Confirm is ever clicked: ONLY the slip's Book Bet link is pressed, once, and the NEW code it shows is
        read. Dry run reports and presses nothing. No code is a plain failure: booking places nothing, so no bet exists."""
        if self._dry_run:
            return PlacementResult(PlacementKind.DRY_RUN, self._report(token, reading))
        book = getattr(site, "book_for_code", None)
        if book is None:
            raise BookmakerError("This browser driver cannot make a booking code.")
        try:
            code = str(await book() or "")
        except Exception:
            code = ""
        if not re.fullmatch(r"[0-9A-Za-z]{4,10}", code):
            raise BookmakerError("The booking code could not be read.")
        return PlacementResult(PlacementKind.BOOKED, (), "", code)

    async def _check_betslip(self, site: SportyBetSite, expected: list[ExpectedLine], prices: list[Decimal],
                             token: PlacementToken, with_stake: bool) -> BetslipReading:
        reading = await site.read_betslip()
        verified = verify_reading(expected, reading, prices, token.total_odds)
        if verified.odds_changed:
            try:
                # Every row matches in every word, only a price moved: those rows at the new prices are the new slip.
                loaded = make_loaded_slip(replace(verified, ok=True, problems=(), odds_changed=False), reading)
            except ValueError:
                raise BetslipMismatch("SportyBet's odds changed and the betslip could not be trusted. "
                                      "Nothing was placed.") from None
            raise OddsChanged("SportyBet's odds changed. Nothing was placed; please check the new summary.", loaded)
        if not verified.ok:
            raise BetslipMismatch("The SportyBet betslip no longer matches what you confirmed: "
                                  + "; ".join(verified.problems[:3]) + " Nothing was placed.")
        if with_stake and (reading.stake is None or Decimal(reading.stake) != Decimal(token.stake)):
            raise BetslipMismatch("The stake on the SportyBet betslip is not the stake you confirmed. Nothing was placed.")
        return reading

    @staticmethod
    def _report(token: PlacementToken, reading: BetslipReading) -> tuple[str, ...]:
        lines = [f"{leg.match_text}: {leg.outcome_name} ({leg.market_name}) @ {leg.odds}" for leg in token.legs]
        if token.booking:
            lines.append(f"Total odds {token.total_odds}. No stake: in booking mode you choose it in SportyBet.")
            lines.append("Book Bet would be pressed once for a booking code; Place and Confirm are never pressed.")
        else:
            lines.append(f"Total odds {token.total_odds}, stake NGN {format(token.stake, 'f')}")
        lines.append("The betslip was checked against your confirmation and matched exactly.")
        return tuple(lines)

    async def _observe(self, site, stamp: str, label: str) -> str:
        """Diagnostic only (see SportyBetSite.observe_after_click). A site without the method, or any failure, is ''."""
        observe = getattr(site, "observe_after_click", None)
        if observe is None:
            return ""
        try:
            return await observe(self._screenshot_dir, stamp, label)
        except Exception:
            return ""

    async def _prove(self, site: SportyBetSite, token: PlacementToken, before: set[str], dialog: str,
                     created_at: datetime | None = None, observed: str = "") -> PlacementResult:
        """Success only when a NEW entry that is ours appears in the bet history: its synthetic key is the proof. 'Ours'
        means the same games, stake and number of legs, status Running and a time not before the click (see
        history_bet_matches). The list shows no odds or picks, so the result says plainly that those were not verified
        there. An entry whose key is ambiguous, or two matching new entries, is never guessed."""
        created_at = created_at or self._clock()
        shown = _success_proof(dialog, token)
        if shown is not None:
            return shown
        for attempt in range(HISTORY_POLLS):
            if attempt:
                await site.page.wait_for_timeout(HISTORY_POLL_GAP_MS)
            try:
                bets = await site.read_history()
            except BookmakerError:
                continue
            fresh = [b for b in bets if b.bet_id not in before]
            mine = [b for b in fresh if history_bet_is_ours(b, token, created_at)]
            if len(mine) == 1:
                found = re.search(SELECTORS.confirm_booking_code, dialog or "", re.IGNORECASE)
                return PlacementResult(PlacementKind.PLACED, (), mine[0].bet_id, found.group(1) if found else "",
                                       history_limits_note(mine[0]))
            if len(mine) > 1:
                break  # ambiguous: never guess
        seen = f" After Place the page showed: \"{observed}\"." if observed else " The page showed no message after Place."
        # Owner-approved fallback: ONCE press the slip's Book Bet link for a booking code. It is never proof of a bet, it never
        # presses Place or Confirm, and a failure only means no code.
        code, tried = "", False
        book = getattr(site, "book_for_code", None)
        if book is not None:
            tried = True
            try:
                code = str(await book() or "")
            except Exception:
                code = ""
            if not re.fullmatch(r"[0-9A-Za-z]{4,10}", code):
                code = ""
        raise _AfterClick("Place was clicked but the bet was not found in your SportyBet bet history. It may or may "
                          "not exist: check your bet history. Nothing will be placed again." + seen, code, tried)

    async def bet_history(self, unsettled: bool = False) -> tuple[HistoryBet, ...]:
        """The first page of the bet history (5 newest bets); `unsettled` reads the running-bets list instead."""
        async with self._lock:
            created = self._site is None
            try:
                if created:
                    self._site = await self._site_factory()
                    await self._site.login()
                return await (self._site.read_history(unsettled=True) if unsettled else self._site.read_history())
            except BookmakerError:
                await self._fail("history")
                raise
            except Exception as error:
                await self._fail("history")
                raise BookmakerError(f"The bet history could not be read ({type(error).__name__}).") from None
            finally:
                if created:
                    await self._close()


def _success_proof(dialog: str, token: PlacementToken) -> PlacementResult | None:
    """SportyBet's own 'Submission Successful' pop-up counts as proof when it shows a booking code and a total stake equal
    to the approved stake (owner decision 2026-10-01: wait for that message before any history check). The bet id is then
    'booking:<code>'; a pop-up with a different stake, or without a code, is not proof and the history is read as before."""
    text = " ".join((dialog or "").split())
    if not re.search(r"submission\s+successful", text, re.IGNORECASE):
        return None
    code = re.search(r"Booking\s*Code\s*[:#]?\s*([0-9A-Za-z]{4,10})", text, re.IGNORECASE)
    stake = re.search(r"Total\s*Stake\s*([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if code is None or stake is None:
        return None
    try:
        if Decimal(stake.group(1).replace(",", "")) != Decimal(token.stake):
            return None
    except Exception:
        return None
    return PlacementResult(PlacementKind.PLACED, (), f"booking:{code.group(1)}", code.group(1),
                           "SportyBet showed its 'Submission Successful' message with this stake and booking code "
                           f"{code.group(1)}; the bet history was not needed to prove it.")


# Words that mark a pop-up as part of betting: dismiss_popups never touches such a pop-up.
_BETTING_WORDS = ("confirm", "place bet", "total stake", "odds", "stake", "bet slip", "betslip", "accept")

_OBSERVE_JS = """
() => {
  const sel = '.es-dialog-wrap, .es-dialog, [role=dialog], [class*=dialog], [class*=toast], [class*=popup], [class*=modal], [class*=message-box], [class*=notice]';
  const seen = [];
  document.querySelectorAll(sel).forEach(e => {
    const r = e.getBoundingClientRect();
    const t = (e.innerText || '').trim();
    if (t && r.width > 0 && r.height > 0 && !seen.some(s => s.includes(t) || t.includes(s))) seen.push(t.slice(0, 300));
  });
  return seen.slice(0, 4).join('\\n');
}
"""


_BOOK_CODE_RE = re.compile(r"Book(?:ing)?\s*Code\s*[:#]?\s*([0-9A-Z]{4,10})(?![0-9A-Za-z])", re.IGNORECASE)


_NOT_CODES = frozenset({"LOAD", "ENTER", "CODE", "BOOKING", "COPY", "SHARE", "CLOSE", "CHECK", "FACEBOOK", "SAVE", "LINK"})


def _book_codes(text: str) -> list[str]:
    """Booking codes shown in a page text: 4-10 letters/digits after 'Booking Code' (or 'Book Code'), with at least one digit (a plain
    word like 'Load' after a 'Booking Code' tab label is never a code). Upper-cased."""
    found = []
    for match in _BOOK_CODE_RE.finditer(" ".join((text or "").split())):
        raw = match.group(1)
        code = raw.upper()
        # VERIFIED-LIVE 2026-10-02: codes can be letters only (NLGYFZ). Without a digit it must be written in capitals on the page,
        # at least 5 long, and not a word of the pop-up itself.
        looks_like_code = any(ch.isdigit() for ch in raw) or (raw.isupper() and len(raw) >= 5 and code not in _NOT_CODES)
        if looks_like_code and code not in found:
            found.append(code)
    return found


_CELL_STATE_JS = """
(node) => 'class=' + String(node.className).slice(0, 80) + ' text=' + (node.innerText || '').trim().slice(0, 40)
  + (node.querySelector('[class*=lock], [class*=disable]') || /lock|disable|suspend/i.test(String(node.className)) ? ' lock-icon' : '')
"""
_BOOK_TEXT_JS = """
() => [...document.querySelectorAll('[role=dialog], [class*=dialog], [class*=modal], [class*=popup], [class*=toast], [class*=message-box], [class*=notice], [class*=betslip], [class*=slip]')]
  .filter(e => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; })
  .map(e => (e.innerText || '').trim().slice(0, 1500)).join('\\n')
"""
_BOOK_CLICK_JS = """
() => {
  const b = [...document.querySelectorAll('button, a, span, div, [role=button]')].find(e => {
    const r = e.getBoundingClientRect();
    const t = (e.innerText || '').trim().replace(/\\s+/g, ' ').toLowerCase();
    return r.width > 0 && r.height > 0 && t === 'book bet';
  });
  if (!b) return false;
  b.click();
  return true;
}
"""
_BOOK_CLOSE_JS = """
() => {
  const b = [...document.querySelectorAll('button, a, [role=button]')].find(e => {
    const r = e.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && ['ok', 'close'].includes((e.innerText || '').trim().toLowerCase());
  });
  if (b) b.click();
}
"""
_SUCCESS_JS = """
() => {
  for (const e of document.querySelectorAll('[role=dialog], [class*=dialog], [class*=modal], [class*=popup], [class*=message-box]')) {
    const r = e.getBoundingClientRect();
    const t = (e.innerText || '').trim();
    if (r.width > 0 && r.height > 0 && /submission\\s+successful/i.test(t)) return t;
  }
  return '';
}
"""
_SUCCESS_OK_JS = """
() => {
  const b = [...document.querySelectorAll('button, a, [role=button]')].find(e => {
    const r = e.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && (e.innerText || '').trim().toLowerCase() === 'ok';
  });
  if (b) b.click();
}
"""
_CONFIRM_FIND_JS = """
() => [...document.querySelectorAll('button, a, [role=button]')].some(e => {
  const r = e.getBoundingClientRect();
  return r.width > 0 && r.height > 0 && (e.innerText || '').trim().toLowerCase() === 'confirm';
})
"""
_CONFIRM_CONTEXT_JS = """
() => {
  const b = [...document.querySelectorAll('button, a, [role=button]')].find(e => {
    const r = e.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && (e.innerText || '').trim().toLowerCase() === 'confirm';
  });
  if (!b) return '';
  const box = b.closest('[role=dialog], [class*=dialog], [class*=modal], [class*=popup], [class*=betslip], [class*=slip]') || b.parentElement;
  return (box.innerText || '').trim();
}
"""
_CONFIRM_CLICK_JS = """
() => {
  const b = [...document.querySelectorAll('button, a, [role=button]')].find(e => {
    const r = e.getBoundingClientRect();
    return r.width > 0 && r.height > 0 && (e.innerText || '').trim().toLowerCase() === 'confirm';
  });
  if (b) b.click();
}
"""
_CONTROLS_JS = """
() => [...document.querySelectorAll('button, a, [role=button]')].filter(e => {
  const r = e.getBoundingClientRect();
  return r.width > 0 && r.height > 0 && /^(confirm|cancel|ok|proceed|continue|accept|yes|no)$/i.test((e.innerText || '').trim());
}).slice(0, 6).map(e => e.tagName.toLowerCase() + '.' + String(e.className).split(' ')[0] + ':' + (e.innerText || '').trim()).join(' ; ')
"""


class _AfterClick(Exception):
    """Internal: a failure after the Place click. Becomes PlacementUnknown (with the booking code the fallback read, if any)."""

    def __init__(self, message: str, booking_code: str = "", booking_tried: bool = False):
        super().__init__(message)
        self.booking_code, self.booking_tried = booking_code, booking_tried
