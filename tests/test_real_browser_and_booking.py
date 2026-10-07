"""BROWSER_MODE=real, PLACE_WAIT_SECONDS and the Book Bet fallback (owner-approved 2026-10-01 after the Place click froze the page).
Fake Playwright / fake page objects only: no browser, no network, nothing is placed."""
import asyncio
from types import SimpleNamespace

import pytest

import app as bot
from predictions.adapters.sportybet import betslip as bs
from predictions.adapters.sportybet.betslip import (CHROME_MISSING_TEXT, Credentials, PlaywrightSiteFactory, SportyBetBookmaker,
                                                    SportyBetSite, _book_codes, parse_browser_mode, parse_place_wait_seconds)
from predictions.core.ports import BookmakerError, PlacementUnknown
from predictions.flow.placement import OUTCOME_UNKNOWN
from predictions.flow.store import P_UNKNOWN
from test_epic5_flow import live_env  # noqa: F401  (fixture)
from test_fast_browser import FakePlaywright, playwright  # noqa: F401  (fixture)
from test_sportybet_betslip import live, new_bet  # noqa: F401  (fixtures)

run = asyncio.run


# ---------------------------------------------------------------- settings
@pytest.mark.parametrize("raw, expected", [(None, "standard"), ("", "standard"), (" ", "standard"), ("standard", "standard"),
                                           ("Real", "real"), (" REAL ", "real")])
def test_browser_mode_parses(raw, expected):
    assert parse_browser_mode(raw) == expected


@pytest.mark.parametrize("raw", ["chrome", "true", "1", "reall", "headless"])
def test_an_unreadable_browser_mode_refuses_to_start(raw):
    with pytest.raises(ValueError) as info:
        parse_browser_mode(raw)
    assert "BROWSER_MODE" in str(info.value) and raw not in str(info.value)


@pytest.mark.parametrize("raw, expected", [(None, 90), ("", 90), ("30", 30), ("90", 90), (" 300 ", 300), ("45", 45)])
def test_place_wait_parses(raw, expected):
    assert parse_place_wait_seconds(raw) == expected


@pytest.mark.parametrize("raw", ["29", "301", "0", "-30", "abc", "90.5", "1e2", "\u0663\u0660"])
def test_an_unreadable_place_wait_refuses_to_start(raw):
    with pytest.raises(ValueError) as info:
        parse_place_wait_seconds(raw)
    assert "PLACE_WAIT_SECONDS" in str(info.value)


def test_the_app_hands_both_settings_to_the_bookmaker(monkeypatch):
    monkeypatch.setattr(bot, "BOOKMAKER", None)
    monkeypatch.setattr(bot, "BROWSER_MODE", "real")
    monkeypatch.setattr(bot, "PLACE_WAIT_SECONDS", 120)
    factory = bot.get_bookmaker()._site_factory
    assert factory._real is True and factory._place_wait_s == 120


def test_the_default_bookmaker_is_standard_with_90_seconds():
    factory = SportyBetBookmaker(Credentials("08031234567", "x"))._site_factory
    assert factory._real is False and factory._place_wait_s == 90


# ---------------------------------------------------------------- real browser
def build(tmp_path, **kw):
    return run(PlaywrightSiteFactory(Credentials("08031234567", "pw-not-real"), tmp_path / "secrets", **kw)())


def test_real_mode_launches_installed_chrome_visible_with_automation_hidden_and_no_user_agent(playwright, tmp_path):
    site = build(tmp_path, browser_mode="real", headless=True, fast=False)
    options = playwright.launch_options
    assert options["channel"] == "chrome" and playwright.launched is False       # visible even though headless was asked
    assert "--disable-blink-features=AutomationControlled" in options["args"]
    assert options["ignore_default_args"] == ["--enable-automation"]
    context = playwright.browser.context
    assert any("navigator" in s and "webdriver" in s and "undefined" in s for s in context.init_scripts)
    assert "user_agent" not in playwright.browser.options and playwright.browser.options["viewport"] == bs.VIEWPORT
    assert isinstance(site, SportyBetSite)


def test_real_mode_keeps_the_stored_login(playwright, tmp_path):
    folder = tmp_path / "secrets"
    folder.mkdir()
    (folder / bs.STATE_FILE_NAME).write_text("{}", encoding="utf-8")
    build(tmp_path, browser_mode="real")
    assert playwright.browser.options["storage_state"].endswith(bs.STATE_FILE_NAME)


def test_standard_mode_is_unchanged(playwright, tmp_path):
    build(tmp_path, headless=True, fast=False)
    assert playwright.launch_options == {} and playwright.launched is True
    assert playwright.browser.options["user_agent"] == bs.USER_AGENT and playwright.browser.context.init_scripts == []


def test_real_mode_without_chrome_fails_plainly_and_never_falls_back(playwright, tmp_path):
    async def launch(headless=True, **more):
        playwright.attempts = getattr(playwright, "attempts", 0) + 1
        raise RuntimeError("Chromium distribution 'chrome' is not found at C:\\secret\\path")

    playwright.launch = launch
    with pytest.raises(BookmakerError) as info:
        build(tmp_path, browser_mode="real")
    assert str(info.value) == CHROME_MISSING_TEXT and "BROWSER_MODE=standard" in CHROME_MISSING_TEXT
    assert playwright.attempts == 1 and "secret" not in str(info.value)


def test_the_site_gets_the_wait_from_the_factory(playwright, tmp_path):
    assert build(tmp_path, place_wait_seconds=120)._place_wait_ms == 120_000
    assert build(tmp_path)._place_wait_ms == 90_000


# ---------------------------------------------------------------- the wait and the pale button
class _Button:
    def __init__(self, disabled=True, cls=""):
        self.disabled, self.cls = disabled, cls

    @property
    def first(self):
        return self

    async def count(self):
        return 1

    async def is_disabled(self):
        return self.disabled

    async def get_attribute(self, name):
        return self.cls


class _Missing:
    first = property(lambda self: self)

    async def count(self):
        return 0

    async def is_visible(self):
        return False


class _WaitPage:
    def __init__(self, button=None, success_after_ms=None):
        self.waited, self.button, self.success_after = 0, button, success_after_ms
        self.clicks = []

    async def wait_for_timeout(self, ms):
        self.waited += ms

    async def evaluate(self, script):
        if script is bs._SUCCESS_JS:
            if self.success_after is not None and self.waited >= self.success_after:
                return "Submission Successful Total Stake 100.00 Booking Code VWB01S"
            return ""
        if script is bs._CONFIRM_FIND_JS:
            return False
        self.clicks.append(script)
        return ""

    def locator(self, selector):
        if selector == bs.SELECTORS.place_button and self.button is not None:
            return self.button
        return _Missing()


def test_the_wait_lasts_the_configured_time_and_never_clicks_place_again():
    page = _WaitPage(_Button(True))
    site = SportyBetSite(page, None, None, None, allow_place=True, place_wait_ms=30_000)
    assert run(site.finish_placement()) == "" and 30_000 <= page.waited < 31_000
    assert site._place_pale is True and page.clicks == []


def test_a_longer_wait_catches_a_late_success():
    page = _WaitPage(_Button(True), success_after_ms=80_000)
    site = SportyBetSite(page, None, None, None, allow_place=True, place_wait_ms=90_000)
    assert "Submission Successful" in run(site.finish_placement())
    assert not site._place_pale


def test_a_button_that_became_usable_is_not_reported_pale():
    page = _WaitPage(_Button(False, "btn"))
    site = SportyBetSite(page, None, None, None, allow_place=True, place_wait_ms=2_000)
    run(site.finish_placement())
    assert site._place_pale is False


def test_the_pale_button_is_in_the_diagnostics(tmp_path):
    page = _WaitPage(_Button(True))
    site = SportyBetSite(page, None, None, None, allow_place=True, place_wait_ms=30_000)
    run(site.finish_placement())
    text = run(site.observe_after_click(tmp_path, "s", "confirm"))
    assert "Place button stayed pale/disabled for the whole 30 s wait" in text


# ---------------------------------------------------------------- the booking code
def test_booking_codes_are_found_after_the_label_only():
    assert _book_codes("Booking Code\nABC12D Copy") == ["ABC12D"]
    assert _book_codes("Booking code: vwb01s") == ["VWB01S"]
    assert _book_codes("Booking Code Load") == []                 # a tab label, not a code (not capitals)
    assert _book_codes("Booking Code NLGYFZ 02/10/2026 00:13 Facebook") == ["NLGYFZ"]   # letters-only code, as seen live
    assert _book_codes("Booking Code SHARE") == []                # a pop-up word in capitals is never a code
    assert _book_codes("Booking Code Facebook") == []
    assert _book_codes("Booking Code 123") == []                  # too short
    assert _book_codes("Booking Code 12345678901") == []          # too long
    assert _book_codes("no code here") == []


class _BookPage:
    def __init__(self, shows="Your Booking Code ABC12D", has_link=True, before=""):
        self.has_link, self.shows, self.before = has_link, shows, before
        self.pressed, self.keys, self.closed, self.scripts = 0, [], 0, []
        self.popup = False
        self.waited = 0
        self.keyboard = SimpleNamespace(press=self._press)

    async def _press(self, key):
        self.keys.append(key)
        self.popup = False

    async def wait_for_timeout(self, ms):
        self.waited += ms

    async def evaluate(self, script):
        self.scripts.append(script)
        if script is bs._BOOK_CLICK_JS:
            if self.has_link:
                self.pressed += 1
                self.popup = True
            return self.has_link
        if script is bs._BOOK_TEXT_JS:
            return self.before + (" " + self.shows if self.popup else "")
        if script is bs._BOOK_CLOSE_JS:
            self.closed += 1
            return None
        raise AssertionError("the fallback may only press Book Bet")   # never Place or Confirm


def test_book_for_code_presses_book_bet_once_reads_the_code_and_closes_the_popup():
    page = _BookPage()
    assert run(SportyBetSite(page, None, None, None).book_for_code()) == "ABC12D"
    assert page.pressed == 1 and page.keys == ["Escape"] and page.closed == 1
    assert bs._CONFIRM_CLICK_JS not in page.scripts


def test_the_book_bet_text_match_cannot_hit_place_or_confirm():
    assert "'book bet'" in bs._BOOK_CLICK_JS
    import re
    assert "confirm" not in bs._BOOK_CLICK_JS.lower() and not re.search(r"place", bs._BOOK_CLICK_JS.lower())


def test_an_old_code_already_on_the_page_is_not_taken_as_the_new_one():
    page = _BookPage(shows="Booking Code OLD123", before="Booking Code OLD123")
    assert run(SportyBetSite(page, None, None, None).book_for_code()) == ""
    assert page.waited >= bs.BOOK_WAIT_MS


def test_no_book_bet_link_or_no_code_is_an_empty_answer_never_an_error():
    assert run(SportyBetSite(_BookPage(has_link=False), None, None, None).book_for_code()) == ""
    assert run(SportyBetSite(_BookPage(shows="Something went wrong"), None, None, None).book_for_code()) == ""

    class Broken:
        async def evaluate(self, script):
            raise RuntimeError("page closed")

    assert run(SportyBetSite(Broken(), None, None, None).book_for_code()) == ""


# ---------------------------------------------------------------- the bookmaker uses it only on the unknown path
def test_the_code_rides_on_the_unknown_error_and_the_bet_is_not_called_placed(live, monkeypatch):
    monkeypatch.setattr(bs, "HISTORY_POLLS", 2)
    calls = []

    async def book():
        calls.append(1)
        return "ABC12D"

    live.site.book_for_code = book
    live.site.history_after_click = []
    with pytest.raises(PlacementUnknown) as err:
        run(live.bm.place(live.token()))
    assert err.value.booking_code == "ABC12D" and err.value.booking_tried is True
    assert calls == [1] and live.site.place_clicks == 1


def test_a_failing_booking_only_omits_the_code(live, monkeypatch):
    monkeypatch.setattr(bs, "HISTORY_POLLS", 2)

    async def book():
        raise RuntimeError("boom")

    live.site.book_for_code = book
    live.site.history_after_click = []
    with pytest.raises(PlacementUnknown) as err:
        run(live.bm.place(live.token()))
    assert err.value.booking_code == "" and err.value.booking_tried is True


def test_a_garbage_code_is_dropped(live, monkeypatch):
    monkeypatch.setattr(bs, "HISTORY_POLLS", 2)

    async def book():
        return "<script>x</script>"

    live.site.book_for_code = book
    live.site.history_after_click = []
    with pytest.raises(PlacementUnknown) as err:
        run(live.bm.place(live.token()))
    assert err.value.booking_code == ""


def test_a_proven_bet_never_books(live):
    calls = []

    async def book():
        calls.append(1)
        return "ABC12D"

    live.site.book_for_code = book
    token = live.token()
    live.site.history_after_click = [new_bet(live, bet_id="B1")]
    run(live.bm.place(token))
    assert calls == []


def test_before_click_failures_never_book(live):
    calls = []

    async def book():
        calls.append(1)
        return "X"

    live.site.book_for_code = book
    live.site.history_error_before_click = True
    with pytest.raises(BookmakerError):
        run(live.bm.place(live.token()))
    assert calls == []


# ---------------------------------------------------------------- the owner's message and the record
def test_the_unknown_message_carries_the_code_and_stays_unknown(live_env):  # noqa: F811
    env = live_env
    env.bm.place_mode = "unknown"
    env.bm.unknown_code, env.bm.unknown_tried = "ABC12D", True
    env.to_summary()
    out = env.place(env.press_final().token)
    assert out.kind == OUTCOME_UNKNOWN and env.session.state.name == "PLACEMENT_UNKNOWN"
    assert ("No success message was seen. The bet may or may not exist (check history). To place it yourself, open SportyBet "
            "and enter this booking code") in out.message
    # AD-25: the code, the total odds and the stake each on their own line (the same formatter as the BOOKED message)
    assert out.message.split(chr(10))[-3:] == ["Booking code: ABC12D", "Total odds: 2.41", "Stake: NGN 500"]
    assert "loading it onto your slip is safe" in out.message
    assert "not proof of a bet" in out.message
    row = env.store.list_placements()[0]
    assert row["state"] == P_UNKNOWN and "ABC12D" in row["detail"]
    assert env.store.list_placed_legs() == []                       # legs stay blocked, nothing registered as placed
    assert env.session.message == out.message


def test_a_code_that_could_not_be_read_is_said_plainly(live_env):  # noqa: F811
    env = live_env
    env.bm.place_mode = "unknown"
    env.bm.unknown_code, env.bm.unknown_tried = "", True
    env.to_summary()
    out = env.place(env.press_final().token)
    assert "could not read one" in out.message and "Booking code:" not in out.message


def test_without_the_fallback_the_message_is_as_before(live_env):  # noqa: F811
    env = live_env
    env.bm.place_mode = "unknown"
    env.to_summary()
    out = env.place(env.press_final().token)
    assert "booking code" not in out.message.lower() and "UNKNOWN" in out.message


# ---------------------------------------------------------------- reload / session diagnostics (diagnostic text only)
class _EventPage(_WaitPage):
    def __init__(self, logged_in=True, stake="100", legs=2):
        super().__init__(_Button(True))
        self.handlers, self.main_frame = {}, SimpleNamespace(url="https://www.sportybet.com/ng/")
        self.logged_in, self.stake, self.legs = logged_in, stake, legs

    def on(self, event, handler):
        self.handlers.setdefault(event, []).append(handler)

    def fire(self, event, *args):
        for handler in self.handlers.get(event, []):
            handler(*args)

    def locator(self, selector):
        if selector == bs.SELECTORS.stake_input:
            return SimpleNamespace(count=self._one, first=SimpleNamespace(input_value=self._stake))
        if selector == bs.SELECTORS.betslip_items:
            return SimpleNamespace(count=self._legs_count)
        return super().locator(selector)

    async def _one(self):
        return 1

    async def _stake(self):
        return self.stake

    async def _legs_count(self):
        return self.legs


def _observe(page, tmp_path, **site_kw):
    site = SportyBetSite(page, None, None, None, allow_place=True, place_wait_ms=30_000)

    async def logged_out():
        return not page.logged_in

    site.logged_out = logged_out
    site._listen_for_place_answers()
    return site


def test_a_reload_after_place_is_reported_with_the_path(tmp_path):
    page = _EventPage()
    site = _observe(page, tmp_path)
    page.fire("framenavigated", page.main_frame)
    page.fire("framenavigated", SimpleNamespace(url="https://ads.example/x"))      # a sub-frame: ignored
    page.fire("request", SimpleNamespace(resource_type="document", url="https://www.sportybet.com/ng/sport/football?x=1"))
    page.fire("request", SimpleNamespace(resource_type="xhr", url="https://www.sportybet.com/api/ng/x"))   # not a document
    page.fire("load")
    text = run(site.observe_after_click(tmp_path, "s", "confirm"))
    assert "the page reloaded/navigated after Place (navigated /ng/; document request /ng/sport/football; load event)" in text
    assert "ads.example" not in text and "x=1" not in text
    assert "still logged in, stake box holds 100, 2 leg(s) on the slip" in text


def test_no_reload_is_reported_too_and_a_logout_and_an_emptied_slip_show(tmp_path):
    page = _EventPage(logged_in=False, stake="", legs=0)
    site = _observe(page, tmp_path)
    text = run(site.observe_after_click(tmp_path, "s", "confirm"))
    assert "the page did not reload after Place" in text
    assert "logged out, stake box holds nothing, 0 leg(s) on the slip" in text


def test_the_new_diagnostics_never_raise_and_never_click(tmp_path):
    class Broken(_EventPage):
        def locator(self, selector):
            raise RuntimeError("closed")

        async def evaluate(self, script):
            raise RuntimeError("closed")

    page = Broken()
    site = _observe(page, tmp_path)
    assert isinstance(run(site.observe_after_click(tmp_path, "s", "confirm")), str)
    assert page.clicks == []
