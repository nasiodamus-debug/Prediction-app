"""Story 1.2: keep secrets out of messages and logs."""
import asyncio
import io
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import app as bot
from handled import drop_ack, handled
from predictions.core.ports import CodeReadError

TOKEN = "123456:TEST-TOKEN-NOT-REAL"
PHONE = "08031234567"
PASSWORD = "test-password-not-real"
API_KEY = "sk-ant-test-key-not-real"
OWNER = 111222333
SECRETS = [TOKEN, PHONE, PASSWORD, API_KEY, str(OWNER)]


@pytest.fixture(autouse=True)
def known_secrets(monkeypatch):
    monkeypatch.setattr(bot, "TELEGRAM_TOKEN", TOKEN)
    monkeypatch.setattr(bot, "SPORTY_PHONE", PHONE)
    monkeypatch.setattr(bot, "SPORTY_PASS", PASSWORD)
    monkeypatch.setattr(bot, "OWNER_ID", OWNER)
    monkeypatch.setenv("ANTHROPIC_API_KEY", API_KEY)


def assert_clean(text):
    for secret in SECRETS:
        assert secret not in text, f"secret leaked: {secret[:6]}..."


# ---------- redactor ----------

def test_redact_removes_every_secret():
    text = " | ".join(SECRETS)
    assert_clean(bot.redact(text))


def test_redact_removes_a_token_inside_a_url():
    text = f"GET https://api.telegram.org/bot{TOKEN}/getUpdates failed"
    result = bot.redact(text)
    assert TOKEN not in result
    assert "api.telegram.org" in result


def test_longer_secret_is_redacted_whole_when_it_contains_another(monkeypatch):
    """Here the password contains the phone. If the phone went first, 'pw-' and '-tail' would remain."""
    monkeypatch.setattr(bot, "SPORTY_PASS", "pw-" + PHONE + "-tail")
    result = bot.redact("login with pw-" + PHONE + "-tail failed")
    assert "pw-" not in result and "-tail" not in result
    assert PHONE not in result


def test_phone_is_caught_in_other_written_forms():
    for form in ["8031234567", "2348031234567", "+2348031234567", PHONE]:
        assert form not in bot.redact(f"user {form} not found"), form


def test_url_encoded_token_is_caught():
    encoded = "123456%3ATEST-TOKEN-NOT-REAL"
    assert encoded not in bot.redact(f"GET /bot{encoded}/getMe")


def test_secret_shapes_are_caught_even_if_the_value_was_never_loaded(monkeypatch):
    monkeypatch.setattr(bot, "TELEGRAM_TOKEN", None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    other_token = "987654321:AAH-abcdefghijklmnopqrstuvwxyz012345"
    other_key = "sk-ant-api03-AbCdEf_123-xyz"
    result = bot.redact(f"{other_token} and {other_key}")
    assert other_token not in result and other_key not in result


def test_a_number_inside_a_longer_number_is_not_mangled():
    text = "booking ref 9111222333 and 1111222333999 are fine"
    assert bot.redact(text) == text
    assert "[hidden]" in bot.redact(f"user {OWNER} did it")


def test_very_short_secrets_are_left_alone(monkeypatch):
    monkeypatch.setattr(bot, "SPORTY_PASS", "12")
    assert bot.redact("stake 12 units") == "stake 12 units"


# ---------- phone mask ----------

def test_phone_shows_only_the_last_four_digits():
    masked = bot.mask_phone(PHONE)
    assert masked.endswith("4567")
    assert PHONE not in masked
    assert "0803123" not in masked


@pytest.mark.parametrize("value", [None, "", "12"])
def test_missing_or_short_phone_is_not_shown(value):
    assert bot.mask_phone(value) == "(not set)"


# ---------- error sanitizer ----------

def test_sanitized_error_has_no_secrets():
    error = Exception(f"login failed for {PHONE} / {PASSWORD} via https://x.test/bot{TOKEN}/y")
    assert_clean(bot.sanitize_error(error))


@pytest.mark.parametrize("raw", [
    r"open failed: C:\Users\hp\Predictions app\config.env",
    r"Traceback: C:\Windows\System32\thing.dll not found",
    "cannot read /home/someone/project/secrets/state.json",
    "no such file /tmp/pw-cache/session",
])
def test_sanitized_error_has_no_file_paths(raw):
    result = bot.sanitize_error(raw)
    assert "[path]" in result
    assert "config.env" not in result and "state.json" not in result
    assert "Users" not in result and "someone" not in result and "System32" not in result


def test_project_folder_with_a_space_is_removed():
    folder = str(Path(bot.__file__).resolve().parent)
    result = bot.sanitize_error(f"failed in {folder}\\app.py line 5")
    assert folder not in result
    assert "Predictions app" not in result


@pytest.mark.parametrize("raw", [
    "failed in c:/users/hp/predictions app/app.py line 5",
    "failed in C:/Users/hp/Predictions app/app.py line 5",
    "failed in C:\\USERS\\HP\\PREDICTIONS APP\\app.py line 5",
])
def test_project_folder_is_removed_in_any_case_or_slash_style(raw):
    folder_home = str(Path.home())
    if not raw.lower().replace("/", "\\").startswith("failed in c:\\users\\hp"):
        pytest.skip("test paths assume the user folder C:\\Users\\hp")
    result = bot.sanitize_error(raw)
    assert "app.py" not in result and "predictions" not in result.lower()


def test_error_says_what_kind_of_failure_it_was():
    assert bot.sanitize_error(TimeoutError("page took too long")).startswith("TimeoutError: ")
    assert bot.sanitize_error(Exception("plain")) == "plain"


def test_empty_error_still_says_something():
    assert bot.sanitize_error(Exception("")) == "no details available"
    assert bot.sanitize_error(ValueError("")) == "ValueError: no details available"

def test_ordinary_text_and_urls_survive():
    result = bot.sanitize_error("Timeout 30000ms exceeded at https://www.sportybet.com/ng/ for 1/2 games")
    assert "Timeout 30000ms exceeded" in result
    assert "https://www.sportybet.com/ng/" in result
    assert "1/2 games" in result


def test_long_errors_are_cut_and_flattened():
    result = bot.sanitize_error("line one\nline two " + "x" * 1000)
    assert "\n" not in result
    assert len(result) <= bot.MAX_ERROR_LENGTH + 1


# ---------- logging ----------

@pytest.fixture
def clean_logging():
    root = logging.getLogger()
    saved = (list(root.handlers), root.level)
    names = list(bot.NOISY_LOGGERS) + ["telegram.ext"]
    saved_levels = {n: logging.getLogger(n).level for n in names}
    yield
    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in saved[0]:
        root.addHandler(handler)
    root.setLevel(saved[1])
    for name, level in saved_levels.items():
        logging.getLogger(name).setLevel(level)


def test_logs_are_redacted_including_tracebacks(clean_logging):
    bot.configure_logging()
    stream = io.StringIO()
    logging.getLogger().handlers[0].setStream(stream)
    log = logging.getLogger("some.module")
    log.warning("calling https://api.telegram.org/bot%s/getMe for %s", TOKEN, PHONE)
    try:
        raise RuntimeError(f"password {PASSWORD} rejected")
    except RuntimeError:
        log.exception("boom")
    output = stream.getvalue()
    assert "getMe" in output and "boom" in output
    assert_clean(output)


def test_chatty_libraries_are_quiet_below_warning(clean_logging):
    bot.configure_logging()
    stream = io.StringIO()
    logging.getLogger().handlers[0].setStream(stream)
    for name in ("httpx", "httpcore", "telegram", "telegram.ext", "playwright", "anthropic"):
        logging.getLogger(name).info("HTTP Request: POST https://api.telegram.org/bot%s/x", TOKEN)
    assert stream.getvalue() == ""
    for name in bot.NOISY_LOGGERS:
        assert logging.getLogger(name).getEffectiveLevel() >= logging.WARNING


# ---------- messages the bot sends ----------

def owner_update(text=None, data=None):
    sent = []

    async def reply_text(message, **kwargs):
        sent.append(message)

    async def edit_message_text(message, **kwargs):
        sent.append(message)

    async def answer(*args, **kwargs):
        return None

    message = SimpleNamespace(text=text, reply_text=reply_text)
    query = SimpleNamespace(data=data, answer=answer, edit_message_text=edit_message_text, message=message)
    update = SimpleNamespace(
        effective_user=SimpleNamespace(id=OWNER),
        message=message if data is None else None,
        callback_query=query if data is not None else None,
    )
    return update, sent


def test_review_failure_message_is_sanitized(monkeypatch):
    class Failing:
        async def read_code(self, code):
            raise CodeReadError(f"boom at {Path(bot.__file__).resolve()} with {TOKEN} for {PHONE}")

    monkeypatch.setattr(bot, "SportyBetCodeReader", Failing)
    update, sent = owner_update(text="Review ABC123")
    asyncio.run(handled(update))
    drop_ack(sent)
    assert sent
    assert_clean(" ".join(sent))
    assert "app.py" not in " ".join(sent)


def pending_session():
    bot.USER_SESSIONS[OWNER] = {
        "engine": "AMERICA",
        "approved": [{"id": 0, "team": "Team", "market": "m", "note": "n"}],
        "disqualified": [],
        "selected_ids": [0],
        "stake": "1000",
    }

