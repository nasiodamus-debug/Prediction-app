"""The bet-history LIST page of SportyBet, as it was seen live and read-only on 2026-10-01 (fabricated recording in
tests/data/sportybet_bet_history_sample.html). Parsing through the REAL selectors, the synthetic key, and every rule that
lets an entry prove (or refuse to prove) a bet. No browser, no network, no login."""
import asyncio
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from fake_history_page import FakeHistoryContext
from fakes_bookmaker import NOW, history_entry
from predictions.adapters.sportybet import betslip as bs
from predictions.adapters.sportybet.betslip import (SELECTORS, RawHistoryRow, SportyBetSite, build_history,
                                                    history_bet_from_row, history_key)
from predictions.core.betting import (HISTORY_LIMITS_NOTE, LoadedLeg, SETTLED_OR_RUNNING, entry_matches_legs,
                                      history_bet_matches, history_limits_note)
from predictions.core.ports import BookmakerError, PlacementUnknown
from test_sportybet_betslip import Rig as SiteRig, new_bet

REPO = Path(__file__).resolve().parent.parent
SAMPLE = (REPO / "tests" / "data" / "sportybet_bet_history_sample.html").read_text(encoding="utf-8")


def run(coro):
    return asyncio.run(coro)


def read(html=SAMPLE, unsettled=False):
    context = FakeHistoryContext(html)
    site = SportyBetSite(None, context, None, None)
    return run(site.read_history(unsettled=unsettled)), context


def raw(time=" 30/09/2026 06:44 ", kind="Multiple", status="Running", vs=(("A v B", True), ("C v D", True)),
        stake="3,000.00", back="--"):
    return RawHistoryRow(time, kind, status, tuple(vs), stake, back)


# ================================================================ parsing the recorded page

def test_the_recorded_page_is_read_by_the_real_selectors_first_page_only():
    bets, context = read()
    assert len(bets) == 5 and context.urls == [SELECTORS.history_url] and context.pages[0].closed
    assert SELECTORS.history_url.endswith("/ng/my_accounts/bet_history/sport_bets?isSettled=10")
    lost, won, running, lost2, won4 = bets
    assert (lost.bet_type, lost.status, lost.stake, lost.leg_count) == ("Multiple", "Lost", Decimal("3000.00"), 5)
    assert lost.shown_matches == ("Northgate Breakers v Southport Taipans", "BK Lindholm v Kosice Foxes",
                                  "Eastvale Kings v Westmere Bullets") and lost.total_return == Decimal("0.00")
    assert lost.placed_at == datetime(2026, 9, 30, 5, 44, tzinfo=timezone.utc)      # 06:44 site time is GMT+01:00
    assert (won.bet_type, won.status, won.leg_count, won.stake, won.total_return) == (
        "Singles", "Won", 1, Decimal("25000.00"), Decimal("42495.88")) and won.shown_matches == ("Riverton Bears v Lakeside Eagles",)
    assert (running.status, running.leg_count, running.total_return) == ("Running", 15, None)     # 3 shown + 12 other
    assert (lost2.leg_count, lost2.shown_matches) == (2, ("Alpha United v Beta City", "Gamma Rovers v Delta Athletic"))
    assert won4.leg_count == 4                                                       # 'and 1 other match'
    assert all(b.total_odds is None and b.lines == () and b.booking_code == "" and not b.ambiguous for b in bets)


def test_the_unsettled_list_has_its_own_url():
    _, context = read(unsettled=True)
    assert context.urls == [SELECTORS.history_url_unsettled] and context.urls[0].endswith("?isSettled=0")


def test_every_entry_gets_a_stable_unique_synthetic_key_that_ignores_the_status():
    bets, _ = read()
    assert len({b.bet_id for b in bets}) == 5
    again, _ = read()
    assert [b.bet_id for b in again] == [b.bet_id for b in bets]
    settled = history_bet_from_row(raw(status="Won"))
    running = history_bet_from_row(raw(status="Running"))
    assert settled.bet_id == running.bet_id                                     # settling does not change the key
    assert history_bet_from_row(raw(stake="3,001.00")).bet_id != running.bet_id
    assert history_bet_from_row(raw(vs=(("A v B", True), ("C v E", True)))).bet_id != running.bet_id
    assert history_bet_from_row(raw(vs=(("A v B", True), ("C v D", True), ("and 2 other matches", False)))).bet_id != running.bet_id
    assert history_bet_from_row(raw(time="30/09/2026 06:45")).bet_id != running.bet_id
    assert history_bet_from_row(raw(kind="Singles")).bet_id != running.bet_id


def test_two_entries_with_the_same_key_are_both_ambiguous():
    one = SAMPLE.split('<div class="m-order-list">')[1].split('<div class="m-order-list">')[0]
    doubled = SAMPLE.replace('<div class="pagination">', '<div class="m-order-list">' + one + '<div class="pagination">')
    bets, _ = read(doubled)
    keys = [b.bet_id for b in bets]
    assert len(bets) == 6 and keys.count(bets[0].bet_id) == 2
    assert [b.ambiguous for b in bets] == [True, False, False, False, False, True]


def test_the_time_is_lagos_time_and_read_to_the_minute():
    bet = history_bet_from_row(raw(time="01/01/2027 00:05"))
    assert bet.placed_at == datetime(2026, 12, 31, 23, 5, tzinfo=timezone.utc)


# ================================================================ failing closed

def test_a_page_with_no_rows_and_no_empty_marker_is_unreadable():
    with pytest.raises(BookmakerError, match="could not be read"):
        read("<div class='m-bet-history'><div class='pagination'></div></div>")
    with pytest.raises(BookmakerError, match="could not be read"):
        read("<div class='login-form'>Please log in</div>")


def test_a_recognised_empty_marker_is_an_empty_history():
    bets, _ = read("<div class='m-empty'>No bets</div>")
    assert bets == ()


@pytest.mark.parametrize("damage", [
    lambda h: h.replace('<span class="stake">3,000.00</span>', '<span class="stake">lots</span>'),
    lambda h: h.replace(" 30/09/2026 06:44 ", " yesterday "),
    lambda h: h.replace(' class="m-order-left"', ' class="gone"'),
    lambda h: h.replace('<span class="m-order-status">Lost</span>', ''),
    lambda h: h.replace('<span>and 2 other matches</span>', '<span>plus some more</span>'),
    lambda h: h.replace('<span>and 2 other matches</span>', '<span>and 0 other matches</span>'),
    lambda h: h.replace('<span class="stake"> 0.00 </span>', '<span class="stake">n/a</span>'),
    lambda h: h.replace('Northgate Breakers <span class="vs-text">v</span> Southport Taipans', 'Northgate Breakers'),
], ids=["stake", "time", "time-element", "status", "other-text", "other-zero", "return", "not-a-game"])
def test_a_row_without_its_expected_parts_makes_the_whole_read_fail(damage):
    with pytest.raises(BookmakerError, match="could not be read"):
        read(damage(SAMPLE))


def test_the_other_matches_line_must_be_last_and_single_and_games_must_exist():
    for vs in ((("and 2 other matches", False), ("A v B", True)),
               (("A v B", True), ("and 2 other matches", False), ("and 1 other match", False)),
               (("and 2 other matches", False),), ()):
        with pytest.raises(BookmakerError, match="could not be read"):
            history_bet_from_row(raw(vs=vs))


def test_an_unknown_type_or_status_word_is_kept_not_trusted():
    bet = history_bet_from_row(raw(kind="System", status="Cashed Out"))
    assert (bet.bet_type, bet.status) == ("System", "Cashed Out")
    assert not history_bet_matches(bet, LEGS2, Decimal("3000"), None, SETTLED_OR_RUNNING)


# ================================================================ the proof rules

def leg(match):
    return LoadedLeg(("m:" + match, "18", "total=2.5", "over"), match, "Over/Under", "Over 2.5", Decimal("1.85"), "Over 2.5")


LEGS2 = (leg("A v B"), leg("C v D"))
LEGS5 = tuple(leg(m) for m in ("A v B", "C v D", "E v F", "G v H", "I v J"))
STAKE = Decimal("3000")


def mine(**kw):
    kw.setdefault("matches", ("A v B", "C v D"))
    kw.setdefault("stake", "3000")
    return history_entry(**kw)


def proves(bet, legs=LEGS2, stake=STAKE, created=None):
    return history_bet_matches(bet, legs, stake, created)


def test_an_entry_with_the_games_stake_count_and_running_status_proves_the_bet():
    assert proves(mine())
    assert proves(mine(matches=("C v D", "A v B")))                     # the order of the games does not matter
    assert proves(mine(matches=("a v b", "C  V  d")))                   # team names go through the one normaliser
    assert proves(history_entry(("A v B",), "3000"), (leg("A v B"),))   # a single: 'Singles'


def test_the_stake_must_be_equal():
    assert not proves(mine(stake="2999")) and not proves(mine(stake="3001"))
    assert not proves(replace(mine(), stake=None))


def test_the_total_leg_count_must_be_equal():
    assert not proves(mine(matches=("A v B",), others=0))                              # one game for two legs
    assert not proves(mine(others=1))                                                  # 3 games for two legs
    assert not proves(replace(mine(), leg_count=None))
    assert not proves(mine(matches=("A v B", "C v D", "E v F"), others=0))
    assert proves(mine(matches=("A v B", "C v D", "E v F"), others=2), LEGS5)
    assert not proves(mine(matches=("A v B", "C v D", "E v F"), others=3), LEGS5)      # 6 games, 5 legs
    assert not proves(mine(matches=("A v B", "C v D", "E v F"), others=1), LEGS5)      # 4 games, 5 legs


def test_every_shown_game_must_be_one_of_our_legs():
    assert not proves(mine(matches=("A v B", "C v X")))
    assert not proves(mine(matches=("A v B", "A v B")))                                # one leg cannot be used twice
    assert not proves(mine(matches=("B v A", "C v D")))                                # the reverse fixture is another game
    assert not proves(mine(matches=("E v F", "G v H")))
    assert not proves(replace(mine(), shown_matches=()))


def test_only_the_first_three_games_can_be_compared_when_there_are_more_and_the_message_says_so():
    long_entry = mine(matches=("G v H", "A v B", "E v F"), others=2)
    assert proves(long_entry, LEGS5)
    assert not proves(mine(matches=("G v H", "A v B", "Z v Y"), others=2), LEGS5)      # a shown game that is not ours
    note = history_limits_note(long_entry)
    assert note.startswith(HISTORY_LIMITS_NOTE) and "first 3 of 5 games" in note and "other 2 could not be compared" in note
    assert history_limits_note(mine()) == HISTORY_LIMITS_NOTE
    assert "does not show odds or selections" in HISTORY_LIMITS_NOTE
    assert "only the games, stake and number of legs were verified" in HISTORY_LIMITS_NOTE


def test_a_brand_new_bet_must_be_running_anything_else_is_unknown():
    assert proves(mine(status="Running")) and proves(mine(status=" running "))
    for status in ("Won", "Lost", "Cashed Out", "Void", "Pending", "", "Settled"):
        assert not proves(mine(status=status)), status
    assert history_bet_matches(mine(status="Won"), LEGS2, STAKE, None, SETTLED_OR_RUNNING)      # reconciliation, later
    assert not history_bet_matches(mine(status="Cashed Out"), LEGS2, STAKE, None, SETTLED_OR_RUNNING)


def test_the_type_must_fit_the_number_of_legs():
    assert not proves(mine(bet_type="Singles"))
    assert not proves(history_entry(("A v B",), "3000", bet_type="Multiple"), (leg("A v B"),))
    assert not proves(mine(bet_type="System"))


def test_an_entry_older_than_the_placement_does_not_prove_it_two_minutes_of_slack():
    created = datetime(2026, 10, 1, 9, 0, 40, tzinfo=timezone.utc)
    assert proves(mine(placed_at=created.replace(second=0)), created=created)          # to the minute: 09:00 for 09:00:40
    assert proves(mine(placed_at=created - timedelta(minutes=2)), created=created)
    assert not proves(mine(placed_at=created - timedelta(minutes=3)), created=created)
    assert not proves(mine(placed_at=created - timedelta(days=2)), created=created)
    assert proves(mine(placed_at=created + timedelta(minutes=5)), created=created)
    assert not proves(replace(mine(), placed_at=None), created=created)               # an unreadable time is no proof


def test_an_ambiguous_entry_proves_nothing_but_can_be_seen_with_the_flag():
    twin = mine(ambiguous=True)
    assert not proves(twin)
    assert history_bet_matches(twin, LEGS2, STAKE, None, ignore_ambiguity=True)
    assert entry_matches_legs(twin, LEGS2, STAKE)


# ================================================================ the placement proof through the adapter

@pytest.fixture
def live(tmp_path):
    return SiteRig(tmp_path, dry_run=False)


def test_a_new_matching_running_entry_is_the_proof_and_its_key_is_the_bet_id(live):
    token = live.token("500")
    entry = new_bet(live, placed_at=NOW)
    live.site.history_after_click = [entry]
    result = run(live.bm.place(token))
    assert result.bet_id == entry.bet_id and result.booking_code == ""
    assert "does not show odds or selections" in result.note


@pytest.mark.parametrize("change", [
    dict(stake="600"), dict(status="Won"), dict(status="Cashed Out"), dict(placed_at=NOW - timedelta(hours=1)),
], ids=["stake", "won", "unknown-status", "older"])
def test_a_new_entry_that_breaks_a_rule_is_unknown_not_proof(live, monkeypatch, change):
    monkeypatch.setattr(bs, "HISTORY_POLLS", 1)
    token = live.token("500")
    live.site.history_after_click = [new_bet(live, **change)]
    with pytest.raises(PlacementUnknown):
        run(live.bm.place(token))
    assert live.site.place_clicks == 1


def test_an_entry_already_in_the_before_snapshot_is_not_proof(live, monkeypatch):
    monkeypatch.setattr(bs, "HISTORY_POLLS", 1)
    token = live.token("500")
    live.site.history = [new_bet(live)]
    live.site.history_after_click = [new_bet(live)]
    with pytest.raises(PlacementUnknown):
        run(live.bm.place(token))


def test_two_entries_with_an_identical_key_are_ambiguous_and_never_proof(live, monkeypatch):
    monkeypatch.setattr(bs, "HISTORY_POLLS", 1)
    token = live.token("500")
    twin = replace(new_bet(live), ambiguous=True)
    live.site.history_after_click = [twin, twin]
    with pytest.raises(PlacementUnknown):
        run(live.bm.place(token))


def test_the_entry_time_is_checked_against_the_moment_of_the_click(live):
    token = live.token("500")
    live.site.history_after_click = [new_bet(live, placed_at=NOW - timedelta(minutes=1))]     # inside the 2 minutes
    assert run(live.bm.place(token)).bet_id


def test_the_adapter_never_asks_the_list_for_odds_or_picks():
    source = (REPO / "predictions" / "adapters" / "sportybet" / "betslip.py").read_text(encoding="utf-8")
    assert source.count("VERIFIED-LIVE") >= 12
    for removed in ("history_bet_id", "history_total_odds", "history_team_separator", "parse_history_text"):
        assert removed not in source


# ---------- VERIFIED-LIVE 2026-10-01: a stored session can look logged in while the history page redirects to login

class _LoginLocator:
    def __init__(self, page, name):
        self.page, self.name = page, name

    @property
    def first(self):
        return self

    async def fill(self, text):
        self.page.filled.append((self.name, text))

    async def click(self):
        self.page.clicked.append(self.name)
        self.page.url = self.page.after_login


class _RedirectedPage:
    """A page the site first redirects to /ng/login, then (after the login button) back to the history."""

    def __init__(self, inner, login_works=True):
        self.inner, self.filled, self.clicked, self.gotos = inner, [], [], []
        self.url = "https://www.sportybet.com/ng/login?ret=/ng/my_accounts/bet_history/sport_bets"
        self.after_login = ("https://www.sportybet.com/ng/my_accounts/bet_history/sport_bets?isSettled=10"
                            if login_works else self.url)
        self.closed = False

    async def goto(self, url, **kwargs):
        self.gotos.append(url)
        await self.inner.goto(url, **kwargs)

    def locator(self, selector):
        if selector in (SELECTORS.login_phone, SELECTORS.login_page_password, SELECTORS.login_page_button):
            return _LoginLocator(self, selector)
        return self.inner.locator(selector)

    async def wait_for_timeout(self, ms):
        return None

    async def close(self):
        self.closed = True


class _RedirectContext(FakeHistoryContext):
    def __init__(self, html, login_works=True):
        super().__init__(html)
        self.login_works, self.redirected = login_works, []

    async def new_page(self):
        inner = await super().new_page()
        page = _RedirectedPage(inner, self.login_works)
        self.redirected.append(page)
        return page


class _Creds:
    password = "pw-test-not-real"

    def national_phone(self):
        return "8031234567"


def test_a_redirect_to_the_login_page_logs_in_on_that_page_and_then_reads_the_history():
    context = _RedirectContext(SAMPLE)
    site = SportyBetSite(None, context, _Creds(), None)
    bets = run(site.read_history())
    page = context.redirected[0]
    assert bets, "the history was read after logging in"
    assert [name for name, _ in page.filled] == [SELECTORS.login_phone, SELECTORS.login_page_password]
    assert page.clicked == [SELECTORS.login_page_button]


def test_a_login_that_does_not_complete_fails_closed(monkeypatch):
    monkeypatch.setattr(bs, "LOGIN_DONE_WAIT_MS", 400)
    context = _RedirectContext(SAMPLE, login_works=False)
    site = SportyBetSite(None, context, _Creds(), None)
    with pytest.raises(bs.BookmakerError) as err:
        run(site.read_history())
    assert "login did not complete" in str(err.value)


def test_a_redirect_with_no_login_details_fails_closed():
    context = _RedirectContext(SAMPLE)
    site = SportyBetSite(None, context, None, None)
    with pytest.raises(bs.BookmakerError) as err:
        run(site.read_history())
    assert "no login details" in str(err.value)


def test_no_redirect_means_no_login_attempt():
    context = FakeHistoryContext(SAMPLE)
    site = SportyBetSite(None, context, _Creds(), None)
    assert run(site.read_history())  # unchanged behaviour: the fake page's url attribute is absent
