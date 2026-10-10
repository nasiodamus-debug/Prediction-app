"""Spec 14-1, item 3: FAST_BROWSER. The betslip browser skips pictures, fonts and media and looks quicker while it waits for a
condition; nothing else changes. Fake Playwright objects only: no browser is started, nothing is logged in to."""
import asyncio
import re

import pytest

import app as bot
from predictions.adapters.sportybet import betslip
from predictions.adapters.sportybet.betslip import (BLOCKED_RESOURCE_TYPES, FAST_POLL_MS, POLL_MS, Credentials,
                                                    PlaywrightSiteFactory, SportyBetBookmaker, SportyBetSite,
                                                    parse_fast_browser)
from fakes_bookmaker import FakeBookmaker
from test_epic5_app import ListFixtures, smoke_listing, NOW
from predictions.tools import betslip_smoke

run = asyncio.run


@pytest.mark.parametrize("raw, expected", [(None, False), ("", False), ("  ", False), ("true", True), ("TRUE", True), ("on", True),
                                           (" On ", True), ("false", False), ("False", False), ("off", False), ("OFF", False)])
def test_fast_browser_is_off_unless_the_owner_says_true_or_on(raw, expected):
    assert parse_fast_browser(raw) is expected


@pytest.mark.parametrize("raw", ["0", "1", "no", "yes", "fast", "offf", "of", "tru", "disabled"])
def test_an_unreadable_fast_browser_value_refuses_to_start(raw):
    with pytest.raises(ValueError) as info:
        parse_fast_browser(raw)
    assert "FAST_BROWSER" in str(info.value)


def test_the_blocked_kinds_are_a_named_constant_of_pictures_fonts_and_media_only():
    assert BLOCKED_RESOURCE_TYPES == frozenset({"image", "font", "media"})
    assert FAST_POLL_MS < POLL_MS == 250


# ------------------------------------------------------------------ fake Playwright
class FakeRoute:
    def __init__(self, resource_type):
        self.request = type("Req", (), {"resource_type": resource_type})()
        self.action = None

    async def abort(self):
        self.action = "abort"

    async def continue_(self):
        self.action = "continue"


class FakePage:
    def __init__(self):
        self.waits = []

    async def wait_for_timeout(self, ms):
        self.waits.append(ms)


class FakeContext:
    def __init__(self):
        self.routes = []
        self.init_scripts = []
        self.page = FakePage()

    async def route(self, pattern, handler):
        self.routes.append((pattern, handler))

    async def add_init_script(self, script):
        self.init_scripts.append(script)

    async def add_init_script(self, script):
        self.init_scripts.append(script)

    async def new_page(self):
        return self.page


class FakeBrowser:
    def __init__(self):
        self.context = FakeContext()
        self.options = None

    async def new_context(self, **options):
        self.options = options
        return self.context

    async def close(self):
        pass


class FakePlaywright:
    def __init__(self):
        self.browser = FakeBrowser()
        self.chromium = self
        self.launched = None

    async def launch(self, headless=True, **more):
        self.launched = headless
        self.launch_options = more
        return self.browser

    async def stop(self):
        pass


@pytest.fixture
def playwright(monkeypatch):
    fake = FakePlaywright()

    class Starter:
        async def start(self):
            return fake

    monkeypatch.setattr("playwright.async_api.async_playwright", lambda: Starter())
    return fake


def build(tmp_path, **kw):
    factory = PlaywrightSiteFactory(Credentials("08031234567", "pw-not-real"), tmp_path / "secrets", **kw)
    return run(factory())


def test_the_route_is_installed_only_when_enabled_and_aborts_only_images_fonts_and_media(playwright, tmp_path):
    site = build(tmp_path)                                    # default: on
    routes = playwright.browser.context.routes
    assert len(routes) == 1 and routes[0][0] == "**/*" and isinstance(site, SportyBetSite)
    handler = routes[0][1]
    outcomes = {}
    for kind in ("image", "font", "media", "document", "stylesheet", "script", "xhr", "fetch", "websocket", "other", "manifest"):
        route = FakeRoute(kind)
        run(handler(route))
        outcomes[kind] = route.action
    assert {k for k, v in outcomes.items() if v == "abort"} == {"image", "font", "media"}
    assert {k for k, v in outcomes.items() if v == "continue"} == {"document", "stylesheet", "script", "xhr", "fetch",
                                                                  "websocket", "other", "manifest"}


def test_with_fast_browser_off_nothing_is_routed_and_the_waits_are_as_before(playwright, tmp_path):
    site = build(tmp_path, fast=False)
    assert playwright.browser.context.routes == [] and site._poll_ms == POLL_MS


def test_with_fast_browser_on_the_waiting_loops_look_more_often(playwright, tmp_path):
    assert build(tmp_path)._poll_ms == FAST_POLL_MS


def test_the_route_is_set_on_the_context_before_any_page_exists_so_every_page_has_it(playwright, tmp_path):
    order = []
    context = playwright.browser.context
    real_route, real_new_page = context.route, context.new_page

    async def route(pattern, handler):
        order.append("route")
        await real_route(pattern, handler)

    async def new_page():
        order.append("page")
        return await real_new_page()

    context.route, context.new_page = route, new_page
    build(tmp_path)
    assert order == ["route", "page"]


def test_a_poll_waits_for_its_condition_with_the_gap_it_was_built_with_and_for_the_same_total_time():
    for poll in (POLL_MS, FAST_POLL_MS):
        page = FakePage()
        site = SportyBetSite(page, None, None, None, poll_ms=poll)

        async def never():
            return False

        assert run(site._poll(never, 1_000)) is False
        assert set(page.waits) == {poll} and len(page.waits) * poll >= 1_000      # still waits out the whole timeout


def test_a_poll_returns_as_soon_as_the_condition_holds_without_waiting():
    page = FakePage()
    site = SportyBetSite(page, None, None, None, poll_ms=FAST_POLL_MS)

    async def yes():
        return True

    assert run(site._poll(yes, 5_000)) is True and page.waits == []


def test_a_site_built_directly_keeps_the_original_gap():
    assert SportyBetSite(FakePage(), None, None, None)._poll_ms == POLL_MS


def test_the_bookmaker_hands_the_switch_to_its_browser_factory():
    # Opt-in: measured live on 2026-10-01 it was SLOWER on this site, so the default is off.
    assert SportyBetBookmaker(Credentials("08031234567", "x"))._site_factory._fast is False
    assert SportyBetBookmaker(Credentials("08031234567", "x"), fast_browser=True)._site_factory._fast is True
    assert SportyBetBookmaker(Credentials("08031234567", "x"), fast_browser=False)._site_factory._fast is False


def test_the_app_builds_the_bookmaker_with_the_setting(monkeypatch):
    monkeypatch.setattr(bot, "BOOKMAKER", None)
    monkeypatch.setattr(bot, "FAST_BROWSER", False)
    assert bot.get_bookmaker()._site_factory._fast is False
    monkeypatch.setattr(bot, "BOOKMAKER", None)
    monkeypatch.setattr(bot, "FAST_BROWSER", True)
    assert bot.get_bookmaker()._site_factory._fast is True


def test_no_check_was_removed_from_the_site_driver():
    """The verification code is untouched: the read-back, odds, stake, history and after-click diagnostics are all still there."""
    for name in ("read_betslip", "set_stake", "read_history", "observe_after_click", "confirm_placement", "click_place",
                 "clear_betslip", "add_outcome", "save_state"):
        assert callable(getattr(SportyBetSite, name))
    source = open(betslip.__file__, encoding="utf-8").read()
    assert "HISTORY_POLL_GAP_MS" in source and "verify_reading" in source and "history_bet_matches" in source


def test_the_smoke_script_prints_the_measured_seconds_per_step(monkeypatch):
    # the script's own session manager reads the real clock, and the sample games kick off on a fixed date: pin it before them
    from predictions.flow.session import SessionManager
    monkeypatch.setattr(betslip_smoke, "SessionManager", lambda: SessionManager(clock=lambda: NOW))
    listing = smoke_listing()
    fake = FakeBookmaker(listing, dry_run=True)
    lines = []
    assert run(betslip_smoke.run_smoke(fake, ListFixtures(listing), NOW, 2, "", lines.append)) == 0
    text = "\n".join(lines)
    for n in range(1, 6):
        assert re.search(rf"STEP {n} OK: .* \(\d+\.\d s\)", text)
    assert re.search(r"Seconds per step: step 1 \d+\.\d, step 2 \d+\.\d, step 3 \d+\.\d, step 4 \d+\.\d, step 5 \d+\.\d; total \d+\.\d",
                     text)


def test_the_smoke_reporter_measures_each_step_from_the_end_of_the_last_with_its_clock():
    ticks = iter([0.0, 1.5, 4.0])
    reporter = betslip_smoke.Reporter(lambda s: None, clock=lambda: next(ticks))
    reporter.ok(1, "a")
    reporter.fail(2, "b", ValueError("x"))
    assert reporter.timings == [(1, 1.5), (2, 2.5)]
