"""The SportyBet login must be REAL: judged by the site's own account call (200 valid, 401 refused) and by the balance filling in,
not by 'no login form is showing' (a stale saved login looks logged in: My Account, no form, balance 'NGN ---', every account
call refused). Fabricated pages only; nothing touches SportyBet."""
import asyncio
from types import SimpleNamespace

import pytest

from predictions.adapters.sportybet import betslip as b
from predictions.core.ports import BookmakerError


def run(coro):
    return asyncio.run(coro)


class Loc:
    def __init__(self, page, selector):
        self.page, self.selector = page, selector

    async def count(self):
        return 1 if self._visible() else 0

    def _visible(self):
        s = self.page.site_selectors
        if self.selector in (s.login_phone, s.login_password, s.login_button):
            return self.page.form_visible
        return False

    @property
    def first(self):
        return self

    async def is_visible(self):
        return self._visible()

    async def fill(self, text):
        self.page.fills.append((self.selector, "<hidden>"))

    async def click(self):
        self.page.clicks.append(self.selector)
        if self.selector == self.page.site_selectors.login_button:
            self.page.logins += 1
            if self.page.login_works:
                self.page.session_ok = True
            self.page.form_visible = False


class Context:
    def __init__(self, page):
        self.page = page

    async def clear_cookies(self):
        self.page.cleared += 1
        self.page.session_ok = False
        self.page.looks_logged_in = False     # an anonymous visitor: the login form shows

    async def storage_state(self, path=None):
        self.page.saved += 1


class Page:
    """session_ok: the site accepts the login. looks_logged_in: the page chrome (no login form). balance: what the header shows."""

    def __init__(self, session_ok, looks_logged_in=True, login_works=True, balance_when_ok="6,187.31", balance_when_bad="---"):
        self.session_ok, self.looks_logged_in, self.login_works = session_ok, looks_logged_in, login_works
        self.balance_when_ok, self.balance_when_bad = balance_when_ok, balance_when_bad
        self.handlers, self.url, self.fills, self.clicks = [], "", [], []
        self.logins = self.cleared = self.saved = self.loads = 0
        self.form_visible = not looks_logged_in
        self.site_selectors = b.SELECTORS

    def on(self, event, callback):
        self.handlers.append(callback)

    def _load(self):
        self.loads += 1
        self.form_visible = not self.looks_logged_in and not self.session_ok
        if self.session_ok:
            self.looks_logged_in = True
        status = 200 if self.session_ok else (401 if self.looks_logged_in else None)
        if status is not None:
            for handler in self.handlers:
                handler(SimpleNamespace(url="https://www.sportybet.com/api/ng/patron/account/info", status=status))

    async def goto(self, url, **kw):
        self.url = url
        self._load()

    async def reload(self, **kw):
        self._load()

    def locator(self, selector):
        return Loc(self, selector)

    async def evaluate(self, script, *args):
        if "localStorage.clear" in script:
            return None
        if not self.looks_logged_in:
            return None                                   # an anonymous page shows no balance
        return self.balance_when_ok if self.session_ok else self.balance_when_bad

    async def wait_for_timeout(self, ms):
        return None


def site_for(page, credentials=True):
    creds = b.Credentials("08012345678", "pw") if credentials else None
    return b.SportyBetSite(page, Context(page), creds, None, poll_ms=1)


def test_a_valid_saved_login_is_kept_without_logging_in_again():
    page = Page(session_ok=True)
    run(site_for(page).login())
    assert page.logins == 0 and page.cleared == 0 and page.fills == []


def test_a_stale_saved_login_that_only_looks_logged_in_is_cleared_and_replaced_by_a_real_login():
    page = Page(session_ok=False, looks_logged_in=True)           # My Account shown, no form, balance "NGN ---", account call 401
    site = site_for(page)
    run(site.login())
    assert page.cleared == 1 and page.logins == 1
    assert page.session_ok and run(site.logged_out()) is False    # the new login holds after the reload


def test_the_old_rule_would_have_gone_on_without_a_login_here():
    page = Page(session_ok=False, looks_logged_in=True)
    site = site_for(page)
    site._listen()
    run(page.goto("https://x"))
    assert run(site._visible(b.SELECTORS.login_phone)) is False   # no form showing...
    assert run(site.logged_out()) is True                         # ...yet the site's own refusal says there is no login


def test_a_balance_stuck_at_dashes_is_waited_for_and_then_distrusted_even_when_the_account_call_says_200():
    page = Page(session_ok=True, balance_when_ok="---")
    with pytest.raises(BookmakerError, match="balance never appeared"):
        run(site_for(page).login())
    assert page.cleared == 1 and page.logins == 1                  # it tried a fresh login before giving up


def test_a_balance_the_owner_hid_with_stars_does_not_count_against_the_login():
    page = Page(session_ok=True, balance_when_ok="****")
    run(site_for(page).login())
    assert page.logins == 0


def test_a_fresh_login_the_site_refuses_after_a_reload_is_reported_not_trusted():
    page = Page(session_ok=False, looks_logged_in=True, login_works=False)
    with pytest.raises(BookmakerError):
        run(site_for(page).login())
    assert page.logins == 1


def test_no_login_details_means_a_plain_refusal():
    page = Page(session_ok=False, looks_logged_in=True)
    with pytest.raises(BookmakerError, match="no login details"):
        run(site_for(page, credentials=False).login())


def test_a_page_that_cannot_report_its_responses_keeps_the_older_rules():
    class Plain(Page):
        on = None                                                  # no response events

    page = Plain(session_ok=False, looks_logged_in=False)          # a visitor: the form shows
    del Plain.on
    site = site_for(page)
    page.session_ok = False
    run(site.login())
    assert page.logins == 1
