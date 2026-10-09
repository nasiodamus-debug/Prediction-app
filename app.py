import json
import os
import re
import sys
import asyncio
import logging
import errno
import time
import traceback
from pathlib import Path
from urllib.parse import quote
from dotenv import load_dotenv
from datetime import datetime, timezone
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    ApplicationHandlerStop,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    TypeHandler,
    filters,
)
from telegram.error import BadRequest, TelegramError
from telegram.request import HTTPXRequest
from predictions.adapters.analyst.claude import ClaudeAnalyst  # noqa: F401  (tests patch bot.ClaudeAnalyst.from_environment)
from predictions.adapters.analyst.select import build_analyst, parse_analyst_provider
from predictions.adapters.apifootball.client import ApiFootball
from predictions.flow import freeflags
from predictions.flow.reviewcache import REVIEW_CACHE, parse_review_cache_hours
from predictions.adapters.sportybet.betslip import (Credentials, SportyBetBookmaker, parse_browser_mode, parse_fast_browser,
                                                    parse_place_mode, parse_place_wait_seconds)
from predictions.adapters.sportybet.code import SportyBetCodeReader
from predictions.adapters.sportybet.event import SportyBetEventMarkets
from predictions.adapters.sportybet.serialize import SerializedCodeReader, SerializedEventMarkets, SerializedFixtures
from predictions.adapters.sportybet.fixtures import SportyBetFixtures
from predictions.adapters.footballdata.stats import FootballDataInputs
from predictions.adapters.espn.stats import EspnBasketballStats
from predictions.adapters.espn.football import EspnFootballStats
from predictions.adapters.mlb.stats import MlbStats
from predictions.adapters.nhl.stats import NhlStats
from predictions.adapters.espn.soccer import EspnSoccerStats
from predictions.adapters.understat.stats import UnderstatStats
from predictions.core.analyst import RUN_LIMIT, parse_analyst_mode, parse_call_limit, parse_daily_call_limit
from predictions.core.ports import StatsFetchError
from predictions.core.staking import parse_max_stake
from predictions.core.ttl_cache import STATS_CACHE, parse_cache_ttl_scale, parse_listing_ttl_seconds
from predictions.core.types import Owner
from predictions.flow.analyst import ANALYST_CACHE, DailyBudget
from predictions.flow.america import MultiSportFixtures, MultiSportStats, run_america
from predictions.flow.daycache import DAY_CACHE_JOB_NAME, DAY_CACHE_TIME, DayCache
from predictions.flow.gameday import run_gameday
from predictions.flow.jobs import BackgroundJobs
from predictions.flow.loading import KIND_FAILED, KIND_ISSUES, KIND_LOADED, KIND_PARTIAL, run_load
from predictions.flow.matchday import run_matchday
from predictions.flow.booking import ST_ASK, ST_NOTED, ST_RECORDED, PendingStake, press
from predictions.flow.booking import stake_reply as booking_stake_reply
from predictions.flow.placement import run_placement
from predictions.flow.recovery import attach_store, recover_on_startup
from predictions.flow.retry import run_history_check
from predictions.flow.results import (DAILY_CHECK_TIME, DAILY_JOB_NAME, ResultsGuard, ResultsReport, calibration_lines,
                                      check_results)
from predictions.flow.review import approved_evaluations, run_review
from predictions.flow.session import Action, SessionManager, State, Status, new_session_id
from predictions.flow.stats import FirstCoveringStats, add_extra_inputs, gather_stats
from predictions.flow.store import startup_database
from predictions.telegram.render import chunk_lines
from predictions.adapters.espn.scores import EspnScores
from predictions.flow.combined import parse_combined
from predictions.flow.weekly import (WEEKLY_JOB_NAME, last_completed_week, parse_weekly_time, run_weekly, weekly_catch_up_due)
from predictions.telegram import weeklyview
from predictions.telegram.slate import _compact_line as compact_pick_line
from predictions.telegram.slate import (REVIEW_ORIGIN, Job, OddsNotShown, Screen, decode, failure_screen, handle_callback,
                                        handle_stake_text, issues_screen, list_screen, odds_screen, partial_screen, placement_text,
                                        retry_notice, stake_prompt, start_screen, summary_screen)

# ---------------------------------------------------------
# ENVIRONMENT & CONFIGURATION LOAD
# ---------------------------------------------------------
env_path = Path(__file__).parent / "config.env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
SPORTY_PHONE = os.getenv("SPORTYBET_PHONE")
SPORTY_PASS = os.getenv("SPORTYBET_PASSWORD")

if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is missing. Ensure config.env exists in the application root.")

# ---------------------------------------------------------
# OWNER-ONLY GATE
# ---------------------------------------------------------
def parse_owner_id(raw):
    """Return the owner's Telegram user ID as a positive int, or raise ValueError.

    The error text never includes the raw value, so nothing sensitive is printed.
    """
    text = (raw or "").strip()
    if not text.isascii() or not text.isdigit() or int(text) <= 0:
        raise ValueError(
            "OWNER_ID is missing or is not a positive whole number. "
            "Add your Telegram user ID to config.env as OWNER_ID=<your id>."
        )
    return int(text)


OWNER_ID = parse_owner_id(os.getenv("OWNER_ID"))


def is_owner_update(update, owner_id):
    """True only when the sender and the chat are both the owner, in a private chat."""
    user = update.effective_user
    chat = update.effective_chat
    return bool(
        user is not None
        and chat is not None
        and chat.type == "private"
        and user.id == owner_id
        and chat.id == owner_id
    )


async def owner_gate(update, context):
    """Runs before every other handler. Anyone but the owner is dropped with no reply."""
    if not isinstance(update, Update) or not is_owner_update(update, OWNER_ID):
        raise ApplicationHandlerStop


# ---------------------------------------------------------
# SECRETS: REDACTION, ERROR SANITIZING, LOGGING
# ---------------------------------------------------------
REDACTED = "[hidden]"
MAX_ERROR_LENGTH = 300
# Windows paths (C:\...) and absolute POSIX paths (/home/..., /tmp/...). URLs are left alone;
# any secret inside one is removed by the exact-value redactor.
_PATH_PATTERN = re.compile(
    r"[A-Za-z]:\\(?:[^\s\"'<>|:*?]+\\?)*|(?<![\w:/.\-])/(?:[^\s\"'<>|:*?]+/?)+"
)
def _folder_pattern(folder):
    """Match a folder however it is written: any case, and / or \\ between parts."""
    parts = re.split(r"[\\/]+", folder)
    body = r"[\\/]+".join(re.escape(part) for part in parts)
    return re.compile(body + r"(?:[\\/][^\s\"'<>|:*?]*)*", re.IGNORECASE)


# Folders whose names may contain spaces, so they are removed by name first.
_KNOWN_FOLDER_PATTERNS = [
    _folder_pattern(str(Path(__file__).resolve().parent)),
    _folder_pattern(str(Path.home())),
]


def secret_values():
    """Secret values (and their common written forms) to keep out of output, read at call time."""
    phone_digits = re.sub(r"\D", "", SPORTY_PHONE or "")
    phone_forms = []
    if len(phone_digits) >= 10:
        national = phone_digits[-10:]  # 8031234567 for 08031234567
        phone_forms = [phone_digits, national, "234" + national, "+234" + national]
    candidates = [
        TELEGRAM_TOKEN,
        quote(TELEGRAM_TOKEN or "", safe=""),  # the token as it appears URL-encoded
        SPORTY_PHONE,
        *phone_forms,
        SPORTY_PASS,
        os.getenv("ANTHROPIC_API_KEY"),
        str(OWNER_ID),
    ]
    # Very short values would mangle ordinary text, so they are not redacted.
    return sorted({c for c in candidates if c and len(c) >= 4}, key=len, reverse=True)


# Shapes of the secrets, as a second net in case a value was never loaded into this process.
_SECRET_SHAPES = (
    re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{30,}"),  # Telegram bot token
    re.compile(r"sk-ant-[A-Za-z0-9_-]+"),        # Anthropic API key
)


def redact(text):
    """Replace every secret value in text. Longest values go first."""
    text = str(text)
    for secret in secret_values():
        if secret.isdigit():
            # A number is only a secret when it stands alone, not inside a longer number.
            text = re.sub(r"(?<!\d)" + re.escape(secret) + r"(?!\d)", REDACTED, text)
        else:
            text = text.replace(secret, REDACTED)
    for shape in _SECRET_SHAPES:
        text = shape.sub(REDACTED, text)
    return text


def mask_phone(phone):
    """Show only the last 4 digits of a phone number."""
    digits = str(phone or "")
    return "•" * 6 + digits[-4:] if len(digits) >= 4 else "(not set)"


def sanitize_error(error):
    """Text that is safe to send to Telegram: no secrets, no file paths, bounded length."""
    # Name the kind of failure when it is more specific than a bare Exception.
    kind = ""
    if isinstance(error, BaseException) and type(error) is not Exception:
        kind = type(error).__name__ + ": "
    text = redact(error)
    for pattern in _KNOWN_FOLDER_PATTERNS:
        text = pattern.sub("[path]", text)
    text = _PATH_PATTERN.sub("[path]", text)
    text = " ".join(text.split())
    if not text:
        text = "no details available"
    if len(text) > MAX_ERROR_LENGTH:
        text = text[:MAX_ERROR_LENGTH] + "…"
    return kind + text


class RedactingFormatter(logging.Formatter):
    """Redacts the finished log line, including any traceback."""

    def format(self, record):
        return redact(super().format(record))


NOISY_LOGGERS = ("httpx", "httpcore", "telegram", "playwright", "anthropic")


def configure_logging():
    """Send all logs through the redactor and keep chatty libraries at WARNING or above."""
    handler = logging.StreamHandler()
    handler.setFormatter(RedactingFormatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("telegram.ext").setLevel(logging.WARNING)
    _install_record_redaction()
    sys.excepthook = _log_uncaught
    # Playwright prints protocol traffic straight to stderr when DEBUG is set.
    os.environ.pop("DEBUG", None)


_record_redaction_installed = False


def _install_record_redaction():
    """Redact each log message as it is created, so every handler sees the clean text."""
    global _record_redaction_installed
    if _record_redaction_installed:
        return
    previous = logging.getLogRecordFactory()

    def factory(*args, **kwargs):
        record = previous(*args, **kwargs)
        try:
            record.msg = redact(record.getMessage())
            record.args = None
            if record.exc_info and record.exc_info[0] is not None:
                # Format the traceback now so every handler prints the redacted text.
                record.exc_text = redact("".join(traceback.format_exception(*record.exc_info)).rstrip())
        except Exception:
            pass
        return record

    logging.setLogRecordFactory(factory)
    _record_redaction_installed = True


def _log_uncaught(exc_type, exc, tb):
    """Send uncaught exceptions through the redacting logger instead of raw stderr."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc, tb)
        return
    logging.getLogger("app").critical("Uncaught exception", exc_info=(exc_type, exc, tb))


TELEGRAM_MESSAGE_LIMIT = 4000  # Telegram allows 4096 characters; keep a margin


def clip(text, limit=TELEGRAM_MESSAGE_LIMIT):
    text = str(text)
    return text if len(text) <= limit else text[: limit - 1] + "…"


async def send_to_owner(bot, text, reply_markup=None, parse_mode=None):
    """The one way the bot sends its own messages to the owner. `parse_mode` is only ever 'HTML', for the booking-code message."""
    extra = {} if parse_mode is None else {"parse_mode": parse_mode}
    if reply_markup is None:
        await bot.send_message(chat_id=OWNER_ID, text=clip(redact(text)), **extra)
    else:
        await bot.send_message(chat_id=OWNER_ID, text=clip(redact(text)), reply_markup=reply_markup, **extra)


class OwnerChat:
    """Stands in for a Telegram message when work runs outside an update (start-up recovery): `reply_text` sends to the
    owner through the one owner-message helper."""

    def __init__(self, bot):
        self._bot = bot

    async def reply_text(self, text, reply_markup=None, **kwargs):
        await send_to_owner(self._bot, text, reply_markup, kwargs.get("parse_mode"))

    def get_bot(self):
        return self._bot


def startup_banner() -> str:
    """The console line printed at start: ASCII only, so a redirected Windows stdout (cp1252) cannot crash on it."""
    return f"Matchday Betting Engine Agent Online. Stake limit: {format(MAX_STAKE, ',f')}."


async def announce_startup(application):
    state = "ON: no bet will be placed" if DRY_RUN else "OFF: bets WILL be placed"
    mode = ("Final action: booking code only (PLACE_MODE=book): I never place a bet." if PLACE_MODE == "book"
            else "Final action: the bot places the bet (PLACE_MODE=place).")
    text = f"🚀 Bot started. Dry run is {state}. {mode} Stake limit: {format(MAX_STAKE, ',f')} (MAX_STAKE)."
    if DRY_RUN_RAW is not None and DRY_RUN_RAW.strip().lower() not in ("", "true", "false"):
        text += " (DRY_RUN in config.env was not 'true' or 'false', so dry run stays on.)"
    try:
        await send_to_owner(application.bot, text)
    except Exception as error:
        logging.getLogger(__name__).warning("Could not send the startup message: %s", sanitize_error(error))
    # Story 5.5: recover before anything else is offered. A bet that was being placed is checked against the bet
    # history first; a slate at or after loading is reloaded (in the background) and needs a fresh confirmation.
    try:
        reloads = await recover_after_start(application.bot)
        for recovered in reloads:
            job = Job("load", recovered.session.session_id, recovered.session.state_version)
            application.create_task(run_slate_job(job, OwnerChat(application.bot)))
    except Exception as error:
        logging.getLogger(__name__).error("Recovery failed: %s", sanitize_error(error))
        try:
            await send_to_owner(application.bot, "Recovery after the restart failed: " + sanitize_error(error)
                                + ". Nothing was resumed and nothing was placed.")
        except Exception:
            pass
    try:   # a Monday weekly review missed while the bot was off is made up once (the week row keeps it from posting twice)
        if WEEKLY_ENABLED and STORE is not None and weekly_catch_up_due(STORE, datetime.now(timezone.utc), WEEKLY_TIME, WEEKLY_CATCHUP_DAYS):
            application.create_task(_weekly_in_background(application.bot))
    except Exception as error:
        logging.getLogger(__name__).warning("Could not check the weekly review catch-up: %s", sanitize_error(error))
    try:   # Spec 14-1: build the games cache in the background, after recovery; the bot never waits for it
        start_day_cache_build()
    except Exception as error:
        logging.getLogger(__name__).warning("Could not start the games cache build: %s", sanitize_error(error))


async def recover_after_start(bot):
    """Run the recovery (flow/recovery.py) and tell the owner what it found, through send_to_owner. Returns the recovered
    sessions that need their betslip reloaded. Never places anything."""
    if STORE is None:
        return []
    reports = await recover_on_startup(SESSIONS, STORE, get_bookmaker, sanitize_error)
    counted = sum(1 for r in reports if r.counts)
    await send_to_owner(bot, f"Recovered {counted} session(s) after the restart.")
    chat = OwnerChat(bot)
    for report in reports:
        await send_to_owner(bot, report.text)
        if report.show_list and report.session is not None:
            await send_screen(chat, list_screen(report.session))
        elif report.session is not None and report.session.state is State.PLACEMENT_UNKNOWN:
            # the result is still unknown after the restart: the owner gets the Check history again button
            await send_screen(chat, failure_screen(report.session, now=SESSIONS.now()))
        elif report.reload and report.session is not None:
            # The last confirmed games that were NOT played: shown to the owner, who decides. Nothing reloads on its own.
            await send_to_owner(bot, resume_text(report.session), reply_markup=resume_markup(report.session))
    return []   # no betslip is reloaded until the owner presses Continue (resume_callback)


def resume_text(session) -> str:
    lines = ["Your last confirmed games (nothing was placed):"]
    lines += [compact_pick_line(pick, number) for number, pick in enumerate(session.slip.picks, 1)]
    lines.append("")
    lines.append("Continue reloads these games on SportyBet and re-reads the odds. Discard drops this slate.")
    return "\n".join(lines)


def resume_markup(session):
    base = f"r:{session.session_id}:{session.state_version}"
    return InlineKeyboardMarkup([[InlineKeyboardButton("Continue", callback_data=base + ":c"),
                                  InlineKeyboardButton("Discard", callback_data=base + ":d")]])


async def resume_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Continue / Discard under the restart (or lost-network) prompt for the last confirmed, unplayed games. Only the exact open
    prompt works: a stale or repeated tap is told so and changes nothing. Continue only RELOADS the betslip and asks again; it never
    places anything (the usual stake and fresh confirmation follow)."""
    query = update.callback_query
    found = re.fullmatch(r"r:([0-9A-Za-z]{1,16}):(\d+):([cd])", query.data or "")
    if not found:
        await query.answer("That button is not valid any more.")
        return
    session_id, version, action = found.group(1), int(found.group(2)), found.group(3)
    active = SESSIONS.active
    if active is None or active.session_id != session_id or active.state_version != version or active.state is not State.LOADING_SLIP:
        await query.answer("That prompt is out of date.")
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass
        return
    try:
        await query.edit_message_reply_markup(reply_markup=None)   # the buttons are used up whichever one was pressed
    except Exception as error:
        logging.getLogger(__name__).warning("Could not remove the resume buttons: %s", sanitize_error(error))
    if action == "c":
        await query.answer("Reloading the games on SportyBet...")
        context.application.create_task(run_slate_job(Job("load", session_id, version), OwnerChat(context.bot)))
    else:
        SESSIONS.finish(session_id, State.FAILED, "Discarded at your request. Nothing was placed.", version)
        await query.answer("Discarded. Nothing was placed.")
        await send_to_owner(context.bot, "Discarded. Nothing was placed. Send a new request to start again.")


ALERT_REPEAT_SECONDS = 60
_recent_alerts = {}


async def report_error(update, context):
    """Any unhandled error: log it (redacted) and send the owner a short sanitized alert.

    The same alert is not repeated within a minute, so a failure that repeats cannot flood the chat.
    """
    logging.getLogger(__name__).error("Unhandled error", exc_info=context.error)
    text = f"⚠️ Unexpected error: {sanitize_error(context.error)}"
    now = time.monotonic()
    if now - _recent_alerts.get(text, -ALERT_REPEAT_SECONDS) < ALERT_REPEAT_SECONDS:
        return
    _recent_alerts[text] = now
    try:
        await send_to_owner(context.bot, text)
    except Exception as error:
        logging.getLogger(__name__).warning("Could not send the error alert: %s", sanitize_error(error))


# ---------------------------------------------------------
# DRY RUN AND SINGLE INSTANCE
# ---------------------------------------------------------
def parse_dry_run(raw):
    """Dry run is ON unless DRY_RUN is exactly 'false' (any case). Anything else stays safe."""
    return (raw or "").strip().lower() != "false"


DRY_RUN_RAW = os.getenv("DRY_RUN")
DRY_RUN = parse_dry_run(DRY_RUN_RAW)


LOCK_PATH = Path(__file__).resolve().parent / "local_data" / "bot.lock"


class AlreadyRunning(Exception):
    pass


def acquire_instance_lock(path=LOCK_PATH):
    """Take an exclusive lock for the life of the process. The caller must keep the returned handle.

    The operating system drops the lock if the bot crashes, so there is no stale lock to clean up.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+")
    try:
        if os.name == "nt":
            import msvcrt
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
        handle.close()
        if error.errno in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
            # The lock is held by another copy.
            raise AlreadyRunning("Another copy of the bot is already running. Close it before starting a new one.")
        raise  # a real problem (for example a read-only folder) is not "already running"
    return handle


# ---------------------------------------------------------
# FINAL LIST (Story 2.6) AND SAVED EVALUATIONS (Story 2.7)
# ---------------------------------------------------------
MAX_STAKE = parse_max_stake(os.getenv("MAX_STAKE"))  # NGN; a stake above it is refused (set MAX_STAKE in config.env)
ANALYST_MODE = parse_analyst_mode(os.getenv("ANALYST_MODE"))  # off | last_resort (default) | full; anything else = refuse to start
REVIEW_CACHE.hours = parse_review_cache_hours(os.getenv("REVIEW_CACHE_HOURS"))  # hours a finished Review is reused for the same code on the same day; default 3, 0 = off
freeflags.SOURCE = ApiFootball.from_environment()  # None without API_FOOTBALL_KEY: no free injury and lineup flags, no model check
ANALYST_PROVIDER = parse_analyst_provider(os.getenv("ANALYST_PROVIDER"))  # comma list tried in order, e.g. gemini,claude (the next only when the earlier is unavailable); default claude
RUN_LIMIT.value = parse_call_limit(os.getenv("ANALYST_CALL_LIMIT"))  # analyst calls allowed in one request; default 40
ANALYST_DAILY_CALL_LIMIT = parse_daily_call_limit(os.getenv("ANALYST_DAILY_CALL_LIMIT"))  # paid analyst calls per Lagos day; unset = 40, unreadable = refuse to start
# Spec 14-1 (speed). Each is read once here; a value that is present but unreadable stops the bot from starting.
CACHE_TTL_SCALE = parse_cache_ttl_scale(os.getenv("CACHE_TTL_SCALE"))      # multiplies the stats cache times; unset = 1, 0 = cache off
LISTING_TTL_SECONDS = parse_listing_ttl_seconds(os.getenv("LISTING_TTL_SECONDS"))  # age of the daily cache's odds before a refresh; unset = 600, 0 = off
FAST_BROWSER = parse_fast_browser(os.getenv("FAST_BROWSER"))               # no pictures/fonts/media in the betslip browser; unset = OFF (measured slower on this site)
BROWSER_MODE = parse_browser_mode(os.getenv("BROWSER_MODE"))                 # standard (default) or real: the installed Google Chrome, visible, automation flags hidden; anything else = refuse to start
PLACE_WAIT_SECONDS = parse_place_wait_seconds(os.getenv("PLACE_WAIT_SECONDS"))  # how long to watch the page after Place; unset = 90, allowed 30..300, anything else = refuse to start
PLACE_MODE = parse_place_mode(os.getenv("PLACE_MODE"))                       # AD-25: book (default) = the final confirm only generates a booking code; place = the bot places the bet; anything else = refuse to start
SHOW_BROWSER = (os.getenv("SHOW_BROWSER") or "").strip().lower() in ("true", "on", "1", "yes")  # opt-in: open a visible Chrome window for SportyBet so you can watch the Place click
STATS_CACHE.set_scale(CACHE_TTL_SCALE)
ANALYST_CACHE.enabled = CACHE_TTL_SCALE > 0
ANALYST_CACHE.directory = Path(__file__).resolve().parent / "local_data" / "analyst_cache"   # today's analyst answers survive a restart
SESSIONS = SessionManager(max_stake=MAX_STAKE, book_mode=PLACE_MODE == "book")  # the one active list session, in memory
BOOKING_DESK = PendingStake()  # AD-25: the one open 'how much did you stake?' question (in memory; a restart forgets it)
DB_PATH = Path(__file__).resolve().parent / "local_data" / "predictions.db"
BACKUP_DIR = Path(__file__).resolve().parent / "local_data" / "backups"
WEEKLY_ENABLED = (os.getenv("WEEKLY_REVIEW") or "on").strip().lower() != "off"   # WEEKLY_REVIEW=off turns the Monday review off
WEEKLY_TIME = parse_weekly_time(os.getenv("WEEKLY_REVIEW_TIME"))   # HH:MM Lagos time, default 09:30 (a bad value keeps the default)
WEEKLY_CATCHUP_DAYS = 3   # a Monday run missed because the bot was off is made up on start-up within this many days
WEEKLY_DIR = Path(__file__).resolve().parent / "local_data" / "weekly_reviews"
STORE = None  # the EvaluationStore, opened at startup in __main__ (the flow layer is its only writer)
BOOKMAKER = None  # the ONE SportyBetBookmaker
SPORTYBET_LOCK = asyncio.Lock()  # the one lock around every SportyBet operation (AD-17): the bookmaker, Matchday's game
#                                  list and Review's code reader all take it, so they run one at a time


def analyst_budget():
    """The daily analyst allowance, counted in the evaluation database so a restart does not reset it (Story 10.1)."""
    return DailyBudget(STORE, ANALYST_DAILY_CALL_LIMIT) if STORE is not None else None


def get_bookmaker():
    """The one Bookmaker. It is built on first use from the credentials already loaded above and DRY_RUN (which is
    true unless config.env says exactly 'false'). Raises BookmakerError when the login details are not set."""
    global BOOKMAKER
    if BOOKMAKER is None:
        BOOKMAKER = SportyBetBookmaker(Credentials(SPORTY_PHONE or "", SPORTY_PASS or ""), dry_run=DRY_RUN,
                                       lock=SPORTYBET_LOCK, fast_browser=FAST_BROWSER, headless=not SHOW_BROWSER,
                                       browser_mode=BROWSER_MODE, place_wait_seconds=PLACE_WAIT_SECONDS,
                                       place_mode=PLACE_MODE)
    return BOOKMAKER


def screen_markup(screen):
    """A Screen (text + rows of (label, callback data)) as inline buttons; None when it has no buttons."""
    if not screen.buttons:
        return None
    return InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=data) for label, data in row]
                                 for row in screen.buttons])


async def offer_slate(update, evaluations, origin, session_id, footer=()):
    """A trigger produced a slate (Matchday or Review). With no open slate it starts and its list is shown; with one open
    the owner is asked to confirm replacing it, and the old slate stays untouched until they do (Story 5.7)."""
    result = SESSIONS.request_start(Owner(update.effective_user.id), evaluations, origin, session_id)
    screen = start_screen(result)
    text, extra = screen.text, "\n".join(footer)
    if extra:  # Story 10.1: the safety lines end the list message (a separate message only when it would not fit)
        if len(text) + 1 + len(extra) <= TELEGRAM_MESSAGE_LIMIT:
            text = f"{text}\n{extra}"
            extra = ""
    await update.message.reply_text(redact(text), reply_markup=screen_markup(screen))
    if extra:
        await update.message.reply_text(redact(extra))


async def present_result(update, gate, origin="matchday"):
    """After a Matchday or Gameday evaluation: save every evaluation, then show the surviving picks as the final list,
    or ask whether to replace the slate that is already open. `origin` names the trigger ("matchday", "gameday", "america")."""
    session_id = new_session_id()
    if STORE is not None:
        try:
            STORE.save_evaluations(session_id, [*gate.passed, *gate.rejected])
        except Exception as error:  # reported, never retried silently, and it does not hide the picks
            await update.message.reply_text(f"Could not save this evaluation: {sanitize_error(error)}")
    if not gate.passed:
        return  # nothing to offer; a slate that is already open is left exactly as it is
    await offer_slate(update, gate.passed, origin, session_id, getattr(gate, "footer", ()))


def soccer_stats_source():
    """Soccer stats (Story 9.3): Understat (xG) first for its six domestic leagues, then ESPN (goals only) for the UEFA and
    international competitions it lists; a game covered by both uses Understat."""
    return FirstCoveringStats([UnderstatStats(), EspnSoccerStats()], sanitize_error)


def review_stats_sources():
    """The data source for each sport that has rules (soccer, basketball, and everything `America` covers)."""
    return {"soccer": soccer_stats_source(), "basketball": EspnBasketballStats(), **america_stats_sources()}


# `America` covers several sports in one request. A sport is added by adding ONE row here (its SportyBet listing and its
# stats source), registering its engine in core/engines and its odds line in telegram/render.py (SPORT_ODDS). Each value is a pair of factories so nothing is built until it is used.
AMERICA_SPORTS = {
    "baseball": (lambda: SportyBetFixtures(sport="baseball"), MlbStats),   # MLB, Story 6.3
    "hockey": (lambda: SportyBetFixtures(sport="hockey"), NhlStats),       # NHL, Stories 6.4 and 6.5
    # American football, NFL and college (Stories 7.1 to 7.4): ESPN stats, Open-Meteo wind for NFL games.
    "american_football": (lambda: SportyBetFixtures(sport="american_football"), EspnFootballStats),
}


def america_stats_sources():
    return {sport: stats() for sport, (_, stats) in AMERICA_SPORTS.items()}


def america_fixture_source():
    """Every covered sport's listing, read one after the other under the one SportyBet lock."""
    return SerializedFixtures(MultiSportFixtures({sport: fixtures() for sport, (fixtures, _) in AMERICA_SPORTS.items()}),
                              SPORTYBET_LOCK)


# ---------------------------------------------------------
# DAILY GAMES CACHE (spec 14-1, AD-23): the allowed games of today and tomorrow, in memory only, never used for placing
# ---------------------------------------------------------
async def _warm_soccer(fixtures):
    stats = await gather_stats(fixtures, soccer_stats_source())
    try:
        await add_extra_inputs(fixtures, stats, FootballDataInputs())
    except StatsFetchError:
        pass  # optional data: the real request reports it


async def _warm_basketball(fixtures):
    await gather_stats(fixtures, EspnBasketballStats())


async def _warm_america(fixtures):
    await gather_stats(fixtures, MultiSportStats(america_stats_sources(), sanitize_error))


DAY_CACHE = DayCache(
    {"Matchday": SportyBetFixtures(), "Gameday": SportyBetFixtures(sport="basketball"),
     "America": MultiSportFixtures({sport: fixtures() for sport, (fixtures, _) in AMERICA_SPORTS.items()})},
    listing_ttl_seconds=LISTING_TTL_SECONDS, lock=SPORTYBET_LOCK,
    warmers={"Matchday": _warm_soccer, "Gameday": _warm_basketball, "America": _warm_america})
DAY_CACHE_GUARD = ResultsGuard()  # one build at a time: the 06:00 job, the start-up build and `Cache refresh` never overlap
DAY_CACHE_JOB_KEY = "daycache"


async def run_day_cache_build():
    """The ONE build, used by the daily job, the start-up build and `Cache refresh`. Returns None when it is off or another
    build is running. It reads public listings and public stats only and never calls the analyst or touches the betslip."""
    if not DAY_CACHE.enabled or not DAY_CACHE_GUARD.try_begin():
        return None
    try:
        return await DAY_CACHE.build_all()
    finally:
        DAY_CACHE_GUARD.end()


def start_day_cache_build():
    """Build in the background (BackgroundJobs, AD-22): the caller returns at once. False when one is already running."""
    async def work():
        await run_day_cache_build()

    async def failed(error):
        logging.getLogger(__name__).warning("The games cache build failed: %s", sanitize_error(error))

    return JOBS.start(DAY_CACHE_JOB_KEY, work, "Games cache build", failed) is not None


async def daily_cache_job(context):
    """06:00 Lagos: build the games cache for today and tomorrow and warm their stats. Quiet unless it fails."""
    if await run_day_cache_build() is None:
        logging.getLogger(__name__).info("Daily games cache build skipped: it is off or another build is still running.")


def schedule_daily_cache(application):
    """Register the daily cache build on the job queue (06:00 Lagos, never overlapping itself, no catch-up run)."""
    queue = application.job_queue
    if queue is None:
        logging.getLogger(__name__).warning("No job queue: the daily games cache build is off.")
        return None
    return queue.run_daily(daily_cache_job, DAY_CACHE_TIME, name=DAY_CACHE_JOB_NAME,
                           job_kwargs={"misfire_grace_time": 60, "coalesce": True, "max_instances": 1})


async def save_review(update, result, session_id=None):
    """Save every evaluation a Review judged (approved or disqualified) through the same store as Matchday.
    A failed save is reported in sanitized form and never hides the review."""
    if STORE is None:
        return
    try:
        STORE.save_evaluations(session_id or new_session_id(), result.evaluations)
    except Exception as error:
        await update.message.reply_text(f"Could not save this review: {sanitize_error(error)}")


# Actions whose state change reads the placed-leg registry (SQLite) before it can answer: the answer goes out first.
_ANSWER_FIRST = (Action.FINAL, Action.OVERRIDE, Action.RETRY_ASK, Action.RETRY_YES)


async def _answer_query(query, text=None, alert=False):
    """The ONE answer Telegram accepts for a button press. A Telegram failure is logged, never raised."""
    try:
        await query.answer(text or None, show_alert=alert)
        return True
    except TelegramError as error:
        logging.getLogger(__name__).warning("Could not answer the button press: %s", sanitize_error(error))
        return False


async def slate_callback(update, context):
    """A button on the final list or on a later step. The owner gate has already run for this update.

    AD-22: the button is answered before anything slow happens. Order: pure in-memory state change (the save to SQLite is
    held back) -> answer -> save -> edit the message -> the slow job, which runs in the BACKGROUND so the next press is
    handled at once. Final confirm and Override read the placed-leg registry (SQLite), so they are answered first, with no
    text; their notice is then shown in the edited screen or a short reply.

    The state machine has already moved by the time the first Telegram call is made, so NO Telegram failure (TimedOut,
    a network error) may stop the job from running: every call below is guarded, and the job always follows."""
    started = time.monotonic()
    query = update.callback_query
    parsed = decode(query.data)
    answered_first = parsed is not None and parsed[2] in _ANSWER_FIRST
    if answered_first:
        await _answer_query(query)
    answered_at = time.monotonic()
    SESSIONS.defer_persist()
    try:
        reply = handle_callback(query.data, Owner(update.effective_user.id), SESSIONS, dry_run=DRY_RUN)
    except BaseException:
        SESSIONS.flush_deferred_persist()
        raise
    try:
        if not answered_first:
            await _answer_query(query, reply.answer, reply.alert)
            answered_at = time.monotonic()
    finally:
        SESSIONS.flush_deferred_persist()  # the save the answer did not wait for (also when the answer is cancelled)
    if reply.screen is not None:
        text, markup = redact(reply.screen.text), screen_markup(reply.screen)
        try:
            await query.edit_message_text(text, reply_markup=markup)
        except BadRequest as error:
            if "not modified" not in str(error).lower():
                await _resend_screen(query, text, markup, error)
        except TelegramError as error:  # TimedOut, network error: the change may already be applied
            await _resend_screen(query, text, markup, error)
    if answered_first and reply.answer and reply.job is None:
        await safe_reply(query.message, reply.answer)  # the early answer carried no text, so the notice is said here
    if reply.job is not None:
        _start_slate_job(reply.job, query.message)  # background: this handler returns, the queue keeps moving
    else:
        await report_persist_errors(query.message)
    logging.getLogger(__name__).info("callback took %d ms (answered after %d ms)",
                                     (time.monotonic() - started) * 1000, (answered_at - started) * 1000)


JOBS = BackgroundJobs()  # AD-22: everything slow runs here, off the Telegram update queue


def _start_slate_job(job, message):
    """Run the load or place job in the background. A bet is only placed by the PlacementToken inside the job."""
    async def work():
        try:
            await run_slate_job(job, message)
        finally:
            await report_persist_errors(message)

    async def failed(error):
        await tell_owner(message, f"⚠️ Unexpected error: {sanitize_error(error)}")

    if JOBS.start(f"slate:{job.kind}:{job.session_id}:{job.version}", work, f"Slate {job.kind} job", failed) is None:
        logging.getLogger(__name__).warning("A %s job for this slate is already running; not started twice.", job.kind)


async def report_persist_errors(message):
    """Tell the owner (sanitized, once) when the slate could not be saved: a restart would not find it as it is now."""
    errors = SESSIONS.take_persist_errors()
    if not errors:
        return
    kinds = ", ".join(sorted(set(sanitize_error(e) for e in errors)))
    await tell_owner(message, f"Warning: the slate could not be saved ({len(errors)} time(s): {kinds}). If the bot "
                              "restarts now it may not resume this slate. Check your SportyBet bet history before "
                              "doing anything again.")


async def tell_owner(message, text, reply_markup=None):
    """Send the owner a message through the one owner-message helper; a failure to send is logged, never raised."""
    try:
        await send_to_owner(message.get_bot(), text, reply_markup)
        return True
    except Exception as error:
        logging.getLogger(__name__).warning("Could not send a message to the owner: %s", sanitize_error(error))
        return False


async def _report_failure(message, session, text, warning=""):
    """Tell the owner a load or placement failed: the failure screen (whether a bet could exist, the reason and the
    owner-pressed buttons, AD-24). A session that is not in a failed state gets the plain text."""
    if session is not None and session.state in (State.FAILED, State.PLACEMENT_UNKNOWN):
        if await safe_send_screen(message, failure_screen(session, now=SESSIONS.now(), extra=warning)):
            return
    await safe_reply(message, text)


async def safe_reply(message, text):
    """Reply in the chat; when Telegram fails, send it to the owner directly. True when one of them got through."""
    try:
        await message.reply_text(redact(text))
        return True
    except TelegramError as error:
        logging.getLogger(__name__).warning("Could not reply: %s", sanitize_error(error))
    return await tell_owner(message, text)


async def safe_send_screen(message, screen):
    """Like send_screen, but a Telegram failure falls back to the owner-message helper (buttons kept). True when sent."""
    try:
        await send_screen(message, screen)
        return True
    except TelegramError as error:
        logging.getLogger(__name__).warning("Could not send a screen: %s", sanitize_error(error))
    return await tell_owner(message, screen.text, screen_markup(screen))


async def send_screen(message, screen):
    await message.reply_text(redact(screen.text), reply_markup=screen_markup(screen))


async def continue_after_load(message, session):
    """After the betslip holds the legs to bet on (a full load, or the owner's Go ahead with the loaded picks, AD-26):
    the odds first, then the stake question (place mode) or the final summary (book mode)."""
    if not await safe_send_screen(message, odds_screen(session)):  # the odds first ...
        # The owner could not be shown the odds, so no stake is asked for; the slate is ended, not left stuck.
        text = ("I could not show you the odds (Telegram did not respond), so I did not ask for a stake. "
                "Nothing was placed. Send a new request to start again.")
        SESSIONS.finish(session.session_id, State.FAILED, text, session.state_version)
        return
    shown = SESSIONS.mark_odds_shown(session.session_id, session.state_version)
    if shown.session is not None and shown.session.odds_shown and SESSIONS.book_mode:
        # AD-25: no stake question. The summary (legs, odds, total odds) is made now; the owner still confirms once.
        carried = SESSIONS.apply_booking_summary(shown.session.session_id, shown.session.state_version)
        if carried.status is Status.APPLIED and carried.session is not None:
            notice = retry_notice(carried.session) if carried.session.retries_used else ""
            await safe_send_screen(message, summary_screen(carried.session, dry_run=DRY_RUN, notice=notice))
        return
    if shown.session is not None and shown.session.odds_shown and shown.session.retry_stake is not None:
        # A retry: the stake of the failed attempt goes into a NEW summary (a new token only after the owner
        # presses confirm on it). Nothing is placed here.
        carried = SESSIONS.apply_retry_stake(shown.session.session_id, shown.session.state_version)
        if carried.status is Status.APPLIED and carried.session is not None:
            await safe_send_screen(message, summary_screen(carried.session, dry_run=DRY_RUN,
                                                           notice=retry_notice(carried.session)))
            return
        shown = SESSIONS.mark_odds_shown(session.session_id, session.state_version)   # ask for the stake as usual
    if shown.session is not None and shown.session.odds_shown:
        try:
            if not await safe_send_screen(message, stake_prompt(shown.session)):  # ... and only then the stake
                await tell_owner(message, "I could not ask for your stake (Telegram did not respond). Send your "
                                          "stake as a number, for example 500.")
        except OddsNotShown:  # cannot happen after mark_odds_shown; never ask if it does
            pass


async def run_slate_job(job, message):
    """The slow work after a button: load the betslip (Stories 5.1 to 5.3) or place (Story 5.4). Every failure is
    reported in sanitized form, nothing is retried, and only a PlacementToken from the state machine can place."""
    if job.kind == "load":
        try:
            outcome = await run_load(SESSIONS, get_bookmaker(), job.session_id, job.version, sanitize_error)
        except Exception as error:  # a bug or a missing setting: reported, the session ends, nothing was placed
            text = (f"Loading your games onto SportyBet failed: {sanitize_error(error)}. "
                    "Nothing was placed and nothing will be retried automatically.")
            # loading never clicks Place, so no bet can exist: the owner may press Try again
            failed = SESSIONS.finish(job.session_id, State.FAILED, text, job.version, retryable=True)
            await _report_failure(message, failed.session, text)
            return
        if outcome.kind == KIND_ISSUES:
            await safe_send_screen(message, issues_screen(outcome.session))
        elif outcome.kind == KIND_LOADED:
            await continue_after_load(message, outcome.session)
        elif outcome.kind == KIND_PARTIAL:   # AD-26: some picks loaded, some did not: the owner decides
            await safe_send_screen(message, partial_screen(outcome.session))
        elif outcome.kind == KIND_FAILED:
            await _report_failure(message, outcome.session, outcome.message)
        return
    if job.kind == "go":
        # AD-26: the owner pressed Go ahead with the loaded picks. The betslip already holds exactly those legs (verified at
        # load); this carries on like a normal load: the odds, then the stake question or the summary.
        active = SESSIONS.active
        if (active is not None and active.session_id == job.session_id and active.state_version == job.version
                and active.state is State.AWAITING_STAKE and active.loaded is not None):
            await continue_after_load(message, active)
        return
    if job.kind == "place":
        try:
            bookmaker = get_bookmaker()
        except Exception as error:  # cannot happen after a load, but never leave a session stuck in PLACING
            text = (f"Placing could not start: {sanitize_error(error)}. No bet was placed. Nothing will be retried "
                    "automatically.")
            failed = SESSIONS.finish(job.session_id, State.FAILED, text, retryable=True)   # nothing was clicked
            await _report_failure(message, failed.session, text)
            return
        outcome = await run_placement(SESSIONS, STORE, bookmaker, job.token, sanitize_error)
        session = outcome.session
        if (outcome.kind in ("failed", "unknown") and session is not None
                and session.state in (State.FAILED, State.PLACEMENT_UNKNOWN)):
            # the failure screen says whether a bet could exist and carries the owner-pressed buttons (AD-24)
            await _report_failure(message, session, outcome.message, outcome.warning)
        elif outcome.kind == "booked" and outcome.token is not None and outcome.result is not None:
            await send_booked(message, outcome)   # ONE message: the code, the total odds, the games, the instruction
            if outcome.warning:
                await safe_reply(message, outcome.warning)
        else:
            await safe_reply(message, placement_text(outcome))
        if outcome.kind == "odds_changed" and outcome.session is not None:
            await safe_send_screen(message, summary_screen(outcome.session, dry_run=DRY_RUN))
        return
    if job.kind == "check":
        # 'Check history again': READ-ONLY. It reads the bet history and nothing else (no place, no betslip).
        try:
            bookmaker = get_bookmaker()
        except Exception as error:
            active = SESSIONS.active
            if active is not None and active.session_id == job.session_id and active.state is State.RECONCILING:
                SESSIONS.resolve(job.session_id, State.PLACEMENT_UNKNOWN, active.message,
                                 check_text=f"I could not reach SportyBet to check ({sanitize_error(error)}).")
            await _report_failure(message, SESSIONS.active, "")
            return
        session = await run_history_check(SESSIONS, STORE, bookmaker, job.session_id, sanitize_error)
        if session is not None and session.state is State.PLACEMENT_UNKNOWN:
            await safe_send_screen(message, failure_screen(session, now=SESSIONS.now()))
        elif session is not None:
            await safe_reply(message, session.message)


async def send_booked(message, outcome):
    """The BOOKED message (AD-25): ONE message with the booking code on its first line (monospace and tap-to-copy through
    parse_mode HTML), the total odds, the games and the instruction, plus the 'I placed it' / 'I did not place it' buttons.
    When Telegram refuses the HTML the same text goes out plain. Never says that anything was placed."""
    from predictions.telegram.booked import answer_buttons, booked_message
    from predictions.telegram.slate import Screen
    plain, html_text = booked_message(outcome.token, outcome.result.booking_code)
    markup = screen_markup(Screen("", answer_buttons(outcome.token.token_hash)))
    try:
        await message.reply_text(redact(html_text), parse_mode="HTML", reply_markup=markup)
        return True
    except TelegramError as error:
        logging.getLogger(__name__).warning("Could not send the booking message as HTML: %s", sanitize_error(error))
    try:
        await message.reply_text(redact(plain), reply_markup=markup)
        return True
    except TelegramError as error:
        logging.getLogger(__name__).warning("Could not reply with the booking code: %s", sanitize_error(error))
    return await tell_owner(message, plain, markup)


async def booking_callback(update, context):
    """'I placed it' / 'I did not place it' / 'Skip' on a booking-code message or a Mybets row (AD-25). The owner gate has run.
    Each press is a compare-and-swap on the saved booking row (flow/booking.py): a repeated tap changes nothing and a stale one
    is answered as stale. The data is `b:<12 hex of the token hash>:<action>`, far under 64 bytes."""
    from predictions.telegram.booked import decode_booking, skip_buttons
    from predictions.telegram.slate import Screen
    query = update.callback_query
    parsed = decode_booking(query.data)
    if parsed is None:
        await _answer_query(query, "That button is not valid.")
        return
    key, action = parsed
    origin = BOOKING_DESK.origin
    answer = press(STORE, BOOKING_DESK, key, action)
    await _answer_query(query, answer.text if answer.status not in (ST_ASK, ST_RECORDED, ST_NOTED) else None)
    message = query.message
    if answer.status == ST_ASK:
        BOOKING_DESK.origin = (getattr(message, "chat_id", None), getattr(message, "message_id", None))
        row = STORE.find_placement_by_prefix(key) if STORE is not None else None
        buttons = skip_buttons(row["token_hash"]) if row else ()
        await safe_send_screen(message, Screen(answer.text, buttons))
    elif answer.status in (ST_RECORDED, ST_NOTED):
        await safe_reply(message, answer.text)
        if answer.status == ST_RECORDED:
            try:   # the question is answered: the buttons are no longer needed on the question or on the booking message
                await query.edit_message_reply_markup(reply_markup=None)
                if origin and origin[0] is not None and origin[1] is not None and origin[1] != getattr(message, "message_id", None):
                    await context.bot.edit_message_reply_markup(chat_id=origin[0], message_id=origin[1], reply_markup=None)
            except TelegramError:
                pass


async def mybets_request(update):
    """`Mybets`: the recent booking and placed records, the still-BOOKED ones with the same two buttons (AD-25)."""
    from predictions.telegram.booked import mybets_screens
    try:
        rows = STORE.recent_records(10) if STORE is not None else []
    except Exception as error:
        await update.message.reply_text(f"Could not read the saved records: {sanitize_error(error)}.")
        return
    for screen in mybets_screens(rows):
        await safe_send_screen(update.message, screen)


async def _resend_screen(query, text, markup, error):
    """The edit failed after the list may have changed: send the current screen as a new message so the
    owner is never left with dead buttons."""
    logging.getLogger(__name__).warning("Could not edit the list message: %s", sanitize_error(error))
    try:
        await query.message.reply_text(text, reply_markup=markup)
    except TelegramError as second:
        logging.getLogger(__name__).warning("Could not resend the list message: %s", sanitize_error(second))
        await tell_owner(query.message, text, markup)

# ---------------------------------------------------------
# RESULTS (Stories 8.1 and 8.2)
# ---------------------------------------------------------
RESULTS_GUARD = ResultsGuard()  # one results check at a time, manual or daily: a second request never stacks or runs twice


async def run_results_check():
    """The ONE results check, used by `Results` and by the daily job. Returns None when a check is already running.
    Any failure comes back as a report with a sanitized `error`; it is never retried."""
    if not RESULTS_GUARD.try_begin():
        return None
    try:
        if STORE is None:
            return ResultsReport(error="the evaluation database is not open")
        try:
            bookmaker = get_bookmaker()
        except Exception as error:
            return ResultsReport(error=sanitize_error(error))
        try:
            return await check_results(STORE, bookmaker, sanitize_error)
        except Exception as error:
            return ResultsReport(error=sanitize_error(error))
    finally:
        RESULTS_GUARD.end()


async def daily_results_job(context):
    """Story 8.2: runs once a day at DAILY_CHECK_TIME while the bot is running. If the bot was not running at that time nothing
    is invented; the next check (or `Results`) picks the results up. Quiet when there is nothing to report."""
    report = await run_results_check()
    if report is None:
        logging.getLogger(__name__).info("Daily results check skipped: another check is still running.")
        return
    if not report.has_news:
        return
    try:
        for chunk in chunk_lines(["Daily results check."] + report.lines()):
            await send_to_owner(context.bot, chunk)
    except Exception as error:
        logging.getLogger(__name__).warning("Could not send the daily results summary: %s", sanitize_error(error))


def schedule_daily_results(application):
    """Register the daily check on the job queue. No catch-up run at start-up (a short misfire grace only)."""
    queue = application.job_queue
    if queue is None:
        logging.getLogger(__name__).warning("No job queue: the daily results check is off. Send Results by hand.")
        return None
    return queue.run_daily(daily_results_job, DAILY_CHECK_TIME, name=DAILY_JOB_NAME,
                           job_kwargs={"misfire_grace_time": 60, "coalesce": True, "max_instances": 1})


# ---------------------------------------------------------
# WEEKLY POST-GAME REVIEW (docs/weekly-review-design.md)
# ---------------------------------------------------------
WEEKLY_GUARD = ResultsGuard()   # one weekly build at a time, scheduled, caught up or typed by hand


def _weekly_markup(rows):
    return InlineKeyboardMarkup([[InlineKeyboardButton(label, callback_data=data) for label, data in row] for row in rows])


async def _send_message_id(bot, text, markup=None):
    extra = {} if markup is None else {"reply_markup": markup}
    message = await bot.send_message(chat_id=OWNER_ID, text=clip(redact(text)), **extra)
    return message.message_id


async def post_weekly(bot, week_start=None, dry=False, force=False) -> str:
    """Build and post one week's review to the owner. Returns "" when it was posted, else a short note for the owner. A dry run sends
    everything without buttons and saves nothing."""
    if STORE is None:
        return "The evaluation database is not open."
    if not WEEKLY_GUARD.try_begin():
        return "A weekly review is already being built."
    try:
        outcome = await run_weekly(STORE, EspnScores(), None, week_start=week_start, dry=dry, force=force,
                                   write_dir=None if dry else WEEKLY_DIR)
        if outcome.status == "skipped":
            return outcome.note
        ids = []
        for number, chunk in enumerate(outcome.summary):
            ids.append(await _send_message_id(bot, ("DRY RUN: nothing is saved and there are no buttons.\n" if dry and number == 0 else "") + chunk))
        for rec_id, text in outcome.recommendations:
            if dry:
                await _send_message_id(bot, text)
            else:
                week, seq = weeklyview.split_id(rec_id)
                ids.append(await _send_message_id(bot, text, _weekly_markup(weeklyview.rec_keyboard(week, seq))))
        ids.append(await _send_message_id(bot, outcome.footer))
        if not dry:
            STORE.finish_weekly_review(outcome.week_start, "degraded" if outcome.status == "degraded" else "posted",
                                       json.dumps({"week": outcome.week_key}), ids, outcome.doc_path)
            weeklyview.write_approved(STORE, WEEKLY_DIR)
        return ""
    finally:
        WEEKLY_GUARD.end()


async def _weekly_in_background(bot, **kwargs):
    log = logging.getLogger(__name__)
    try:
        note = await post_weekly(bot, **kwargs)
        if note:
            await send_to_owner(bot, note)
    except Exception as error:
        log.error("The weekly review failed: %s", sanitize_error(error))
        try:
            await send_to_owner(bot, "The weekly review could not be completed: " + sanitize_error(error)
                                + ". Nothing was changed. Send 'Weekly' to try again.")
        except Exception:
            pass


async def weekly_review_job(context):
    """Every Monday at WEEKLY_REVIEW_TIME (Lagos): the review of the week that just ended."""
    await _weekly_in_background(context.bot)


def schedule_weekly_review(application):
    queue = application.job_queue
    if not WEEKLY_ENABLED:
        logging.getLogger(__name__).info("The Monday weekly review is off (WEEKLY_REVIEW=off).")
        return None
    if queue is None:
        logging.getLogger(__name__).warning("No job queue: the Monday weekly review is off. Send Weekly by hand.")
        return None
    # python-telegram-bot 20+ numbers the days 0 = Sunday ... 6 = Saturday, so Monday is 1.
    return queue.run_daily(weekly_review_job, WEEKLY_TIME, days=(1,), name=WEEKLY_JOB_NAME,
                           job_kwargs={"misfire_grace_time": 60, "coalesce": True, "max_instances": 1})


async def _weekly_request(update, argument: str):
    words = argument.split()
    dry = any(w.lower() == "dry" for w in words)
    force = any(w.lower() == "force" for w in words)
    week_start = None
    for w in words:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", w):
            try:
                week_start = datetime.strptime(w, "%Y-%m-%d").date()
            except ValueError:
                await update.message.reply_text("That date is not valid. Use YYYY-MM-DD, for example: Weekly 2026-09-30.")
                return
    if week_start is None:
        week_start = last_completed_week(datetime.now(timezone.utc).astimezone(weeklyview.LAGOS).date())[0]
    await update.message.reply_text(f"Building the weekly review for the week of {week_start.isoformat()}"
                                    + (" (dry run)" if dry else "") + ". This can take a minute.")
    asyncio.get_running_loop().create_task(_weekly_in_background(update.get_bot(), week_start=week_start, dry=dry, force=force))


async def weekly_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    decoded = weeklyview.w_decode(query.data)
    if decoded is None or STORE is None:
        await query.answer("That button is not valid any more.")
        return
    week, seq, action = decoded
    decision = weeklyview.apply_decision(STORE, week, seq, action)
    await query.answer(decision.notice[:190])
    if decision.details:
        for chunk in chunk_lines(decision.details.split("\n")):
            await query.message.reply_text(redact(chunk))
        return
    if decision.changed:
        try:
            weeklyview.write_approved(STORE, WEEKLY_DIR)
        except Exception as error:
            logging.getLogger(__name__).warning("Could not rewrite APPROVED.md: %s", sanitize_error(error))
        try:
            await query.edit_message_text(redact(weeklyview.decided_text(query.message.text or "", decision.status)),
                                          reply_markup=_weekly_markup(decision.keyboard))
        except Exception as error:   # the decision is saved either way
            logging.getLogger(__name__).warning("Could not update the weekly message: %s", sanitize_error(error))


# ---------------------------------------------------------
# TELEGRAM CONTROLLER & ROUTER
# ---------------------------------------------------------
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "⚡ **Matchday Betting Engine Agent Online**\n\n"
        "**Supported Protocol Triggers:**\n"
        "• `Matchday [Date]` — Audits Soccer (Goals, Corners, Yellow Cards). The date can be `today`, `tomorrow` or days such as `3 and 5 October`\n"
        "• `America [Date]` — Evaluates MLB, NHL and American football (NFL and college) games together (same date forms)\n"
        "• `Gameday [Date]` — Audits Basketball slates (game, team and 1H totals, spread, winner) (same date forms)\n"
        "• `Review [Booking Code]` — Reads a SportyBet code and judges each selection (soccer, basketball, MLB, NHL and American football)\n"
        "• `Results` — Reads SportyBet's own result for each of your PLACED bets and legs (also checked daily at 09:00 Lagos time)\n"
        "• `Calibration` — Win rate of your placed, graded legs by the model's probability band\n"
        "• `Mybets` — Your recent booking codes and placed bets; say whether you placed a booked one\n"
        "• `Cache` — Shows the games kept in memory for today and tomorrow (`Cache refresh` rebuilds it now)\n"
        "• Join plays into ONE list: `Gameday, Matchday and America 5 and 6 October` or `Review 23B45S and America 6 October` (a date at the end applies to every date play before it)\n"
        "• `Weekly` — The Monday post-game review of last week's lost bets with recommendations (`Weekly dry` previews it; also posted every Monday)\n"
        "• `Decisions` — What you approved from the weekly review and is waiting to be implemented\n",
        parse_mode="Markdown"
    )

ANALYST_OFF_NOTE = ("AI analyst is off (ANALYST_MODE=off or no analyst API key is set), so games with missing data stay "
                    "insufficient data.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    user_id = update.effective_user.id

    if re.match(r"^Weekly\b", text, re.I):
        await _weekly_request(update, text[len("Weekly"):])
        return
    if re.match(r"^Decisions\b", text, re.I):
        shown = "all" if text.lower().split()[1:2] == ["all"] else "implement"
        for chunk in chunk_lines((weeklyview.decisions_text(STORE, shown) if STORE is not None else "The evaluation database is not open.").split("\n")):
            await update.message.reply_text(redact(chunk))
        return
    marked = re.match(r"^(Done|Note)\s+(\d{4}-W\d{2}-\d{2})\s*(.*)$", text, re.I | re.S)
    if marked and STORE is not None:
        if marked.group(1).lower() == "done":
            answer = weeklyview.set_done(STORE, marked.group(2))
        else:
            answer = weeklyview.set_note(STORE, marked.group(2), marked.group(3).strip()[:500]) if marked.group(3).strip() else "Add the note after the id."
        try:
            weeklyview.write_approved(STORE, WEEKLY_DIR)
        except Exception as error:
            logging.getLogger(__name__).warning("Could not rewrite APPROVED.md: %s", sanitize_error(error))
        await update.message.reply_text(answer)
        return

    combined = parse_combined(text)
    if combined is not None:
        if isinstance(combined, str):
            await update.message.reply_text(combined)
            return
        busy = [p.word for p in combined if JOBS.running(p.kind)] + (["A combined request"] if JOBS.running("COMBINED") else [])
        if busy:
            await update.message.reply_text(f"{busy[0]} is still running. I will not start a second one; its result will arrive here when it "
                                            "is done. Your buttons keep working meanwhile.")
            return

        async def run_combined(update_, _argument, parts=combined):
            await _combined_request(update_, parts)
        await start_long_request(update, "COMBINED", "Combined request", " + ".join(p.word for p in combined), run_combined)
        return

    engine_type = None
    if re.match(r"^America\b", text, re.I):
        engine_type = "AMERICA"
    elif re.match(r"^Gameday\b", text, re.I):
        engine_type = "GAMEDAY"
    elif re.match(r"^Matchday\b", text, re.I):
        engine_type = "MATCHDAY"
    elif re.match(r"^Review\b", text, re.I):
        engine_type = "REVIEW"
    elif re.match(r"^Results\b", text, re.I):
        engine_type = "RESULTS"
    elif re.match(r"^Calibration\b", text, re.I):
        engine_type = "CALIBRATION"
    elif re.match(r"^Cache\b", text, re.I):
        engine_type = "CACHE"
    elif re.match(r"^Mybets\b", text, re.I):
        engine_type = "MYBETS"

    if not engine_type:
        # A typed number may be the stake the owner was just asked for (Story 5.3); the owner gate has already run.
        stake_reply = handle_stake_text(text, Owner(user_id), SESSIONS, DRY_RUN)
        if stake_reply is not None:
            try:
                if stake_reply.screen is not None:
                    await safe_send_screen(update.message, stake_reply.screen)
                else:
                    await safe_reply(update.message, stake_reply.answer)
            finally:
                await report_persist_errors(update.message)
            return
        origin = BOOKING_DESK.origin
        answered = booking_stake_reply(STORE, BOOKING_DESK, text)   # AD-25: the amount for 'I placed it' (only while that question is open)
        if answered is not None:
            await safe_reply(update.message, answered.text)
            if answered.status == ST_RECORDED and origin and origin[0] is not None and origin[1] is not None:
                try:   # the question is answered: the two buttons are no longer needed on the booking message
                    await context.bot.edit_message_reply_markup(chat_id=origin[0], message_id=origin[1], reply_markup=None)
                except Exception as error:
                    logging.getLogger(__name__).warning("Could not remove the booking buttons: %s", sanitize_error(error))
            return
        await update.message.reply_text("⚠️ Unrecognized trigger. Please start with `Matchday`, `America`, `Gameday`, `Review`, `Results`, `Calibration`, `Mybets`, `Cache`, `Weekly` or `Decisions`. Join plays into one list with `and`, e.g. `Gameday, Matchday and America 5 and 6 October` or `Review 23B45S and America 6 October`. For the three slates add a date, `today` or `tomorrow`, e.g. `Matchday today`.")
        return

    if engine_type == "CACHE":
        await _cache_request(update, text[len("Cache"):])
        return

    if engine_type == "MYBETS":
        await mybets_request(update)
        return

    if engine_type == "CALIBRATION":
        # Story 8.3: placed and graded legs only, from the local database.
        try:
            lines = calibration_lines(STORE) if STORE is not None else ["The evaluation database is not open."]
        except Exception as error:
            lines = [f"Could not read the saved results: {sanitize_error(error)}."]
        for chunk in chunk_lines(lines):
            await update.message.reply_text(redact(chunk))
        return

    # AD-22: Matchday, Gameday, America, Review and Results take a minute or more. They run in the background so this
    # handler returns at once and the owner's button presses are handled while they work.
    word = engine_type.capitalize()
    argument = text[len(word):]  # the trigger word was matched above, so it is exactly the first len(word) characters
    await start_long_request(update, engine_type, word, argument, LONG_REQUESTS[engine_type])


async def _cache_request(update, argument):
    """`Cache` shows the daily games cache; `Cache refresh` rebuilds it now, in the background. Owner only (the gate has run)."""
    word = " ".join(argument.split()).lower()
    if word == "":
        await update.message.reply_text(redact(DAY_CACHE.describe(building=DAY_CACHE_GUARD.busy)))
    elif word == "refresh":
        if not DAY_CACHE.enabled:
            await update.message.reply_text(redact(DAY_CACHE.describe()))
        elif DAY_CACHE_GUARD.busy or not start_day_cache_build():
            await update.message.reply_text("The games cache is already being built. Send Cache in a minute to see it.")
        else:
            await update.message.reply_text("Rebuilding the games cache in the background. Your buttons keep working. "
                                            "Send Cache in a minute to see it.")
    else:
        await update.message.reply_text("Send Cache to see the games cache, or Cache refresh to rebuild it now.")


async def _results_request(update, argument):
    # Story 8.1: the same check as the daily one (Story 8.2). Read-only toward SportyBet.
    report = await run_results_check()
    if report is None:
        await update.message.reply_text("A results check is already running. I will not start a second one.")
    else:
        for chunk in chunk_lines(report.lines()):
            await update.message.reply_text(redact(chunk))


async def _matchday_collect(argument):
    evaluated = []
    analyst = build_analyst(ANALYST_PROVIDER, ANALYST_MODE)  # None when ANALYST_MODE=off or without ANTHROPIC_API_KEY: the analyst is then off
    chunks = await run_matchday(argument, SerializedFixtures(SportyBetFixtures(), SPORTYBET_LOCK), sanitize_error, stats_source=soccer_stats_source(),
                                on_result=evaluated.append, analyst=analyst,
                                analyst_off_note=ANALYST_OFF_NOTE, budget=analyst_budget(),
                                extra_inputs_source=FootballDataInputs(),   # corners, cards, referee (spec 11-1)
                                event_markets_source=SerializedEventMarkets(SportyBetEventMarkets(), SPORTYBET_LOCK),
                                day_cache=DAY_CACHE)
    return chunks, evaluated


async def _matchday_request(update, argument):
    chunks, evaluated = await _matchday_collect(argument)
    for chunk in chunks:
        await update.message.reply_text(chunk)  # plain text: no Markdown parsing of SportyBet names
    if evaluated:
        await present_result(update, evaluated[0])


async def _gameday_collect(argument):
    # Stories 6.1 and 6.2: basketball through the same pipeline as Matchday (sport-parameterised slate runner).
    evaluated = []
    analyst = build_analyst(ANALYST_PROVIDER, ANALYST_MODE)  # None when ANALYST_MODE=off or without ANTHROPIC_API_KEY: the analyst is then off
    chunks = await run_gameday(argument, SerializedFixtures(SportyBetFixtures(sport="basketball"), SPORTYBET_LOCK),
                               sanitize_error, stats_source=EspnBasketballStats(), on_result=evaluated.append,
                               analyst=analyst, analyst_off_note=ANALYST_OFF_NOTE, budget=analyst_budget(),
                               day_cache=DAY_CACHE)
    return chunks, evaluated


async def _gameday_request(update, argument):
    chunks, evaluated = await _gameday_collect(argument)
    for chunk in chunks:
        await update.message.reply_text(redact(chunk))  # plain text: no Markdown parsing of SportyBet names
    if evaluated:
        await present_result(update, evaluated[0], "gameday")


async def _america_collect(argument):
    # Stories 6.3 to 6.5 and 7.1 to 7.4: MLB, NHL and American football through the same pipeline, in ONE reply and ONE list.
    evaluated = []
    analyst = build_analyst(ANALYST_PROVIDER, ANALYST_MODE)  # None when ANALYST_MODE=off or without ANTHROPIC_API_KEY: the analyst is then off
    chunks = await run_america(argument, america_fixture_source(), sanitize_error,
                               stats_source=MultiSportStats(america_stats_sources(), sanitize_error),
                               on_result=evaluated.append, analyst=analyst, analyst_off_note=ANALYST_OFF_NOTE,
                               day_cache=DAY_CACHE)
    return chunks, evaluated


async def _america_request(update, argument):
    chunks, evaluated = await _america_collect(argument)
    for chunk in chunks:
        await update.message.reply_text(redact(chunk))  # plain text: no Markdown parsing of SportyBet names
    if evaluated:
        await present_result(update, evaluated[0], "america")


async def _review_collect(argument):
    # Stories 4.1 and 4.2: a new pipeline that only READS the code anonymously and never touches the betslip.
    reviewed = []
    analyst = build_analyst(ANALYST_PROVIDER, ANALYST_MODE)  # None when ANALYST_MODE=off or without ANTHROPIC_API_KEY: the analyst is then off
    chunks = await run_review(argument, SerializedCodeReader(SportyBetCodeReader(), SPORTYBET_LOCK), sanitize_error,
                              stats_sources=review_stats_sources(), on_result=reviewed.append,
                              analyst=analyst, analyst_off_note=ANALYST_OFF_NOTE, budget=analyst_budget())
    return chunks, reviewed


async def _review_request(update, argument):
    chunks, reviewed = await _review_collect(argument)
    for chunk in chunks:
        await update.message.reply_text(redact(chunk))  # plain text: no Markdown parsing of SportyBet names
    if reviewed:
        session_id = new_session_id()
        await save_review(update, reviewed[0], session_id)
        # Story 5.6: the approved selections (never a disqualified one) continue as the same interactive list.
        approved = approved_evaluations(reviewed[0])
        if approved:
            await offer_slate(update, approved, REVIEW_ORIGIN + reviewed[0].code, session_id)


async def _combined_request(update, parts):
    """Several plays in ONE list (flow/combined.py). Each part runs exactly as on its own and reports on its own; the surviving picks
    (a slate's passing picks, a Review's approved selections) are merged and offered as a single list. A part that fails is reported
    and left out; the others still count. Nothing is placed and nothing touches the betslip until the owner confirms the one list."""
    session_id = new_session_id()
    collectors = {"MATCHDAY": _matchday_collect, "GAMEDAY": _gameday_collect, "AMERICA": _america_collect, "REVIEW": _review_collect}
    merged, footers, seen = [], [], set()
    for number, part in enumerate(parts, 1):
        label = f"{part.word} {part.argument}".strip()
        await update.message.reply_text(f"Part {number} of {len(parts)}: {label}")
        try:
            chunks, results = await collectors[part.kind](part.argument)
        except Exception as error:
            await update.message.reply_text(f"{part.word} failed: {sanitize_error(error)}. It is left out of the combined list.")
            continue
        for chunk in chunks:
            await update.message.reply_text(chunk if part.kind == "MATCHDAY" else redact(chunk))
        if not results:
            continue
        result = results[0]
        if part.kind == "REVIEW":
            await save_review(update, result, session_id)
            picks = list(approved_evaluations(result))
        else:
            if STORE is not None:
                try:
                    STORE.save_evaluations(session_id, [*result.passed, *result.rejected])
                except Exception as error:
                    await update.message.reply_text(f"Could not save this evaluation: {sanitize_error(error)}")
            picks = list(result.passed)
            footers += [line for line in getattr(result, "footer", ()) if line not in footers]
        for pick in picks:
            key = (str(pick.match_id), pick.market, pick.selection)
            if key not in seen:
                seen.add(key)
                merged.append(pick)
    if not merged:
        await update.message.reply_text("Nothing passed in the combined request, so there is no list to offer.")
        return
    await offer_slate(update, merged, "combined", session_id, tuple(footers))


# The long requests: kind -> the function that does the work. A kind runs once at a time (AD-22).
LONG_REQUESTS = {"RESULTS": _results_request, "MATCHDAY": _matchday_request, "GAMEDAY": _gameday_request,
                 "AMERICA": _america_request, "REVIEW": _review_request}   # COMBINED is started separately (handle_message)
_ACKNOWLEDGEMENTS = {
    "RESULTS": "Checking your placed bets against SportyBet. This can take a minute.",
    "REVIEW": "Reviewing that booking code. This can take a minute.",
}


async def start_long_request(update, kind, word, argument, work):
    """Start a long request in the background and return at once. One request per kind at a time; the owner gets a short
    'Analysing ...' message now and the real result (or a sanitized failure) when it is done."""
    message = update.message
    if JOBS.running(kind):
        await message.reply_text(f"{word} is still running. I will not start a second one; its result will arrive here "
                                 "when it is done. Your buttons keep working meanwhile.")
        return
    shown = " ".join(argument.split())[:60]
    ack = _ACKNOWLEDGEMENTS.get(kind) or (f"Analysing {word}" + (f" for {shown}" if shown else "")
                                          + ". This can take a minute. I will send the list here when it is ready.")
    if not shown and kind != "RESULTS":
        ack = None  # no date or code: the request is answered at once with what to send, so no 'Analysing' first

    async def run():
        if ack:
            await safe_reply(message, ack)
        await work(update, argument)

    async def failed(error):
        await safe_reply(message, f"{word} failed: {sanitize_error(error)}. Nothing was placed. Send the request again "
                                  "to retry.")

    JOBS.start(kind, run, f"{word} request", failed)


async def stop_background_jobs(application):
    """Shutdown: cancel the running requests and jobs (a bet being placed is recovered from its saved PLACING state)."""
    await JOBS.cancel_all()


# ---------------------------------------------------------
# APPLICATION ENTRYPOINT
# ---------------------------------------------------------
request_config = HTTPXRequest(
    connect_timeout=30.0,
    read_timeout=30.0
)

def register_handlers(app):
    # The owner gate goes first (group -1) and covers every update type.
    app.add_handler(TypeHandler(Update, owner_gate), group=-1)
    app.add_error_handler(report_error)

    app.add_handler(CommandHandler("start", start_cmd, filters=filters.UpdateType.MESSAGE))
    app.add_handler(CallbackQueryHandler(slate_callback, pattern=r"^s:"))  # final-list buttons (Story 2.6)
    app.add_handler(CallbackQueryHandler(resume_callback, pattern=r"^r:"))  # Continue / Discard the last confirmed, unplayed games
    app.add_handler(CallbackQueryHandler(weekly_callback, pattern=r"^w:"))  # weekly review: Implement / Later / Ignore / Details / Undo
    app.add_handler(CallbackQueryHandler(booking_callback, pattern=r"^b:"))  # AD-25: 'I placed it' / 'I did not place it' / Skip
    app.add_handler(MessageHandler(filters.UpdateType.MESSAGE & filters.TEXT & (~filters.COMMAND), handle_message))


if __name__ == "__main__":
    configure_logging()  # first, so nothing logged while starting up is unredacted
    try:
        instance_lock = acquire_instance_lock()  # keep this handle for the life of the process
    except AlreadyRunning as already_running:
        raise SystemExit(str(already_running))
    # Backup copy (keeping the last 7), then open and upgrade the evaluation database (Story 2.7).
    STORE = startup_database(DB_PATH, BACKUP_DIR, redact)
    attach_store(SESSIONS, STORE)  # the slate is saved after every step; the placed-leg registry is consulted (Story 5.5)
    app = (
        ApplicationBuilder()
        .token(TELEGRAM_TOKEN)
        .request(request_config)
        .post_init(announce_startup)
        .post_stop(stop_background_jobs)
        .build()
    )

    register_handlers(app)
    schedule_daily_results(app)
    schedule_weekly_review(app)
    schedule_daily_cache(app)

    print(startup_banner())
    app.run_polling()