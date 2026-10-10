"""The SportyBet betslip adapter (Stories 5.1 and 5.4): its logic against a fake page driver, and the page driver itself
against a fake Playwright page. No browser, no network, no login."""
import asyncio
import re
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from fakes_bookmaker import (KICKOFF, NOW, make_fixture, pick_evaluation, totals_offer, two_pick_setup)
from predictions.adapters.sportybet import betslip as bs
from predictions.adapters.sportybet.betslip import (SELECTORS, Credentials, SportyBetBookmaker, SportyBetSite,
                                                    check_secrets_dir, history_bet_is_ours)
from predictions.core.betting import (BetslipLine, BetslipReading, HistoryBet, LoadedLeg, LoadedSlip, Leg,
                                      leg_from_evaluation, total_odds)
from predictions.core.placement import (MINT_AUTHORITY, PlacementKind, PlacementToken, Summary, mint_token)
from predictions.core.ports import (BetslipMismatch, BookmakerError, FixtureFetchError, OddsChanged, PlacementUnknown)

REPO = Path(__file__).resolve().parent.parent


def run(coro):
    return asyncio.run(coro)


# ================================================================ a fake page driver (for the Bookmaker's logic)

class FakeSite:
    """Stands in for SportyBetSite. It keeps a betslip and a bet history and records every action in `log`."""

    def __init__(self, catalog, allow_place=False):
        self.catalog = catalog                         # event_ref -> (match_text, {outcome_name: odds}, market_name)
        self.rows: list[BetslipLine] = []
        self.stake: Decimal | None = None
        self.log: list[str] = []
        self.place_clicks = 0
        self.history: list[HistoryBet] = []
        self.history_after_click: list[HistoryBet] | None = None
        self.logged_in = True
        self.extra_stray = False        # add a stray selection while adding (a "different betslip")
        self.clear_works = True
        self.click_error: Exception | None = None
        self.history_error_before_click = False
        self.closed = False
        self.shots: list[str] = []
        self.allow_place = allow_place
        self.page = SimpleNamespace(wait_for_timeout=self._wait)
        self.odds_shift = Decimal("0")
        self.stake_ignored = False

    async def _wait(self, ms):
        return None

    async def login(self):
        self.log.append("login")

    async def logged_out(self):
        return not self.logged_in

    async def clear_betslip(self):
        self.log.append("clear")
        if self.clear_works:
            self.rows.clear()

    async def betslip_count(self):
        return len(self.rows)

    async def add_outcome(self, event_ref, market_titles, outcome):
        self.log.append(f"add:{event_ref}:{outcome}")
        match_text, offers, market = self.catalog[event_ref]
        self.rows.append(BetslipLine(match_text, market, outcome, offers[outcome]))
        if self.extra_stray:
            self.rows.append(BetslipLine("Stray v Team", "1X2", "Home", Decimal("2.00")))

    async def read_betslip(self):
        self.log.append("read")
        lines = tuple(BetslipLine(r.match_text, r.market_name, r.outcome_name, r.odds + self.odds_shift) for r in self.rows)
        total = total_odds(l.odds for l in lines) if lines else None
        return BetslipReading(lines, total, self.stake)

    async def set_stake(self, stake):
        self.log.append(f"stake:{stake}")
        if not self.stake_ignored:
            self.stake = Decimal(stake)

    async def click_place(self):
        if not self.allow_place:
            raise BookmakerError("Dry run: the Place button is never clicked.")
        self.log.append("PLACE")
        self.place_clicks += 1
        if self.click_error is not None:
            raise self.click_error
        if self.history_after_click is not None:
            self.history = list(self.history_after_click)

    async def confirm_placement(self):
        self.log.append("confirm")
        return ""

    async def read_history(self):
        self.log.append("history")
        if self.history_error_before_click and self.place_clicks == 0:
            raise BookmakerError("The bet history could not be read.")
        return tuple(self.history)

    async def screenshot(self, path):
        self.shots.append(str(path))
        Path(path).write_bytes(b"png")

    async def close(self):
        self.closed = True


class FakeFixtures:
    def __init__(self, fixtures, delay=0.0):
        self.fixtures, self.delay, self.active, self.max_active, self.calls = fixtures, delay, 0, 0, 0

    async def fetch_fixtures(self):
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(self.delay)
        self.active -= 1
        return self.fixtures


def catalog_for(evals, listing):
    out = {}
    for fixture in listing:
        prices = {}
        market = "Over/Under"
        for offer in fixture.markets:
            for outcome in offer.outcomes:
                prices[outcome.name] = outcome.odds
                market = offer.name
        out[fixture.bookmaker_ref] = (f"{fixture.home} v {fixture.away}", prices, market)
    return out


class Rig:
    """A SportyBetBookmaker over a FakeSite and a fake game list."""

    def __init__(self, tmp_path, dry_run=True, delay=0.0):
        self.evals, self.listing = two_pick_setup()
        self.fixtures = FakeFixtures(self.listing, delay)
        self.site = FakeSite(catalog_for(self.evals, self.listing), allow_place=not dry_run)
        self.factory_calls = 0

        async def factory():
            self.factory_calls += 1
            return self.site

        self.shots = tmp_path / "shots"
        self.bm = SportyBetBookmaker(None, dry_run, self.fixtures, factory, screenshot_dir=self.shots, clock=lambda: NOW)
        self.legs = [leg_from_evaluation(e) for e in self.evals]

    def token(self, stake="500", total=None, odds=None, expires=None):
        loaded = run(self.bm.load_slip(self.legs))
        legs = loaded.legs if odds is None else tuple(
            LoadedLeg(l.leg_key, l.match_text, l.market_name, l.outcome_name, o, l.selection, l.evaluation_odds)
            for l, o in zip(loaded.legs, odds))
        summary = Summary("abcd1234", legs, total or loaded.total_odds, Decimal(stake), expires or NOW + timedelta(minutes=5))
        return mint_token(MINT_AUTHORITY, summary)


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path)


@pytest.fixture
def live(tmp_path):
    return Rig(tmp_path, dry_run=False)


def new_bet(rig_, stake="500", bet_id=None, code="", placed_at=None, status="Running"):
    """The entry the real history LIST would show for the bet the rig's two evals make (games, stake, status, time)."""
    from fakes_bookmaker import history_entry
    return history_entry([f"{e.match_id.home} v {e.match_id.away}" for e in rig_.evals], stake, status, 0, placed_at,
                         bet_id=bet_id)


# ================================================================ 5.1 loading

def test_loading_clears_first_then_adds_exactly_the_legs_then_reads_back(rig):
    rig.site.rows.append(BetslipLine("Old v Junk", "1X2", "Draw", Decimal("3.00")))     # the owner's old betslip
    loaded = run(rig.bm.load_slip(rig.legs))
    assert rig.site.log[:2] == ["login", "clear"]                                       # clear FIRST
    adds = [l for l in rig.site.log if l.startswith("add:")]
    assert len(adds) == 2 and rig.site.log.index("clear") < rig.site.log.index(adds[0]) and rig.site.log[-1] == "read"
    assert [r.match_text for r in rig.site.rows] == ["Alpha v Beta", "Gamma v Delta"]    # only the confirmed games
    assert len(loaded.legs) == 2 and loaded.total_odds == Decimal("2.41")
    assert "PLACE" not in rig.site.log and rig.site.place_clicks == 0                    # nothing placed


def test_a_betslip_holding_something_else_aborts_the_load_and_saves_a_screenshot(rig):
    rig.site.extra_stray = True
    with pytest.raises(BetslipMismatch) as caught:
        run(rig.bm.load_slip(rig.legs))
    assert "does not hold exactly the confirmed games" in str(caught.value)
    assert rig.site.closed and len(list(rig.shots.glob("failure-load-*.png"))) == 1     # local screenshot only


def test_a_betslip_that_could_not_be_cleared_aborts(rig):
    rig.site.rows.append(BetslipLine("Old v Junk", "1X2", "Draw", Decimal("3.00")))
    rig.site.clear_works = False
    with pytest.raises(BookmakerError, match="not empty after clearing"):
        run(rig.bm.load_slip(rig.legs))
    assert not any(l.startswith("add:") for l in rig.site.log)                          # nothing was added


def test_a_logged_out_session_aborts_before_touching_the_betslip(rig):
    rig.site.logged_in = False
    with pytest.raises(BookmakerError, match="not logged in"):
        run(rig.bm.load_slip(rig.legs))
    assert "clear" not in rig.site.log
    # the login form is showing (phone number visible): no screenshot may be saved
    assert rig.site.shots == [] and not list(rig.shots.glob("*.png")) and rig.site.closed


def test_a_leg_that_is_gone_at_load_time_aborts_without_touching_the_site(rig):
    rig.fixtures.fixtures = rig.listing[:1]
    with pytest.raises(BookmakerError, match="no longer available"):
        run(rig.bm.load_slip(rig.legs))
    assert rig.factory_calls == 0


def test_an_unexpected_driver_error_is_reported_without_details_and_never_retried(rig):
    async def boom(*a, **k):
        raise RuntimeError("secret path C:\\x and 08031234567")

    rig.site.add_outcome = boom
    with pytest.raises(BookmakerError) as caught:
        run(rig.bm.load_slip(rig.legs))
    assert "RuntimeError" in str(caught.value) and "08031234567" not in str(caught.value) and "C:" not in str(caught.value)
    assert rig.factory_calls == 1 and rig.site.closed


def test_a_slow_load_is_stopped_and_reported(rig, monkeypatch):
    monkeypatch.setattr(bs, "LOAD_TIMEOUT_S", 0.05)

    async def slow(*a, **k):
        await asyncio.sleep(1)

    rig.site.clear_betslip = slow
    with pytest.raises(BookmakerError, match="took too long"):
        run(rig.bm.load_slip(rig.legs))


def test_an_empty_load_is_refused(rig):
    with pytest.raises(BookmakerError):
        run(rig.bm.load_slip([]))


def test_the_game_list_failing_is_reported_and_touches_no_browser(rig):
    async def broken():
        raise FixtureFetchError("SportyBet returned HTTP 500")

    rig.fixtures.fetch_fixtures = broken
    with pytest.raises(BookmakerError, match="game list could not be read"):
        run(rig.bm.check_availability(rig.legs))
    assert rig.factory_calls == 0


# ================================================================ AD-17: one lock around every operation

def test_every_sportybet_operation_runs_one_at_a_time(tmp_path):
    rig = Rig(tmp_path, delay=0.02)
    active = {"n": 0, "max": 0}
    original = rig.site.clear_betslip

    async def tracked_clear():
        active["n"] += 1
        active["max"] = max(active["max"], active["n"])
        await asyncio.sleep(0.02)
        await original()
        active["n"] -= 1

    rig.site.clear_betslip = tracked_clear

    async def go():
        return await asyncio.gather(rig.bm.check_availability(rig.legs), rig.bm.load_slip(rig.legs),
                                    rig.bm.check_availability(rig.legs), rig.bm.load_slip(rig.legs))

    run(go())
    assert rig.fixtures.max_active == 1 and active["max"] == 1     # the game list reads and the loads never overlapped
    assert rig.bm.lock.locked() is False


def test_the_lock_is_released_after_a_failure(rig):
    rig.site.extra_stray = True
    with pytest.raises(BetslipMismatch):
        run(rig.bm.load_slip(rig.legs))
    assert rig.bm.lock.locked() is False
    rig.site.extra_stray = False
    rig.site.rows.clear()
    assert len(run(rig.bm.load_slip(rig.legs)).legs) == 2


def test_a_place_waits_for_a_load_in_progress(tmp_path):
    rig = Rig(tmp_path, delay=0.03)
    token = rig.token()
    order = []
    original = rig.site.read_betslip

    async def tracked_read():
        order.append("read")
        return await original()

    rig.site.read_betslip = tracked_read

    async def go():
        loading = asyncio.create_task(rig.bm.load_slip(rig.legs))
        await asyncio.sleep(0.005)
        placing = asyncio.create_task(rig.bm.place(mint_token(MINT_AUTHORITY, Summary(
            "abcd1234", token.legs, token.total_odds, token.stake, NOW + timedelta(minutes=5)))))
        return await asyncio.gather(loading, placing)

    loaded, placed = run(go())
    assert placed.kind is PlacementKind.DRY_RUN and len(loaded.legs) == 2


def test_review_and_game_list_readers_share_the_bookmakers_lock(tmp_path):
    from predictions.adapters.sportybet.serialize import SerializedCodeReader, SerializedFixtures
    lock = asyncio.Lock()
    rig = Rig(tmp_path, delay=0.02)
    rig.bm = SportyBetBookmaker(None, True, rig.fixtures, rig.bm._site_factory, screenshot_dir=rig.shots,
                                clock=lambda: NOW, lock=lock)
    assert rig.bm.lock is lock
    events = []

    class Inner:
        async def fetch_fixtures(self):
            events.append("list:start")
            await asyncio.sleep(0.02)
            events.append("list:end")
            return []

        async def read_code(self, code):
            events.append("review:start")
            await asyncio.sleep(0.02)
            events.append("review:end")
            return []

    original = rig.site.clear_betslip

    async def clear():
        events.append("load:start")
        await asyncio.sleep(0.02)
        await original()
        events.append("load:end")

    rig.site.clear_betslip = clear

    async def go():
        await asyncio.gather(rig.bm.load_slip(rig.legs), SerializedFixtures(Inner(), lock).fetch_fixtures(),
                             SerializedCodeReader(Inner(), lock).read_code("ABC"))

    run(go())
    # no operation started while another was in progress
    depth = 0
    for event in events:
        if event.endswith(":start"):
            depth += 1
            assert depth == 1, events
        elif event.endswith(":end"):
            depth -= 1


def test_the_app_gives_matchday_review_and_the_bookmaker_the_same_lock():
    source = (REPO / "app.py").read_text(encoding="utf-8")
    assert "SerializedFixtures(SportyBetFixtures(), SPORTYBET_LOCK)" in source
    assert "SerializedCodeReader(SportyBetCodeReader(), SPORTYBET_LOCK)" in source and "lock=SPORTYBET_LOCK" in source


# ================================================================ 5.4 place: dry run

def test_the_bookmaker_is_in_dry_run_by_default_and_its_driver_cannot_place(tmp_path):
    bm = SportyBetBookmaker(Credentials("08031234567", "pw"))
    assert bm.dry_run is True and bm._site_factory._allow_place is False
    assert SportyBetBookmaker(Credentials("08031234567", "pw"), dry_run=False)._site_factory._allow_place is True
    assert SportyBetBookmaker(Credentials("08031234567", "pw"), dry_run="")._site_factory._allow_place is True   # explicit falsy
    assert SportyBetBookmaker(dry_run=True).dry_run is True


def test_dry_run_checks_everything_and_never_clicks_place(rig):
    token = rig.token("500")
    result = run(rig.bm.place(token))
    assert result.kind is PlacementKind.DRY_RUN and rig.site.place_clicks == 0 and "PLACE" not in rig.site.log
    assert any("matched exactly" in line for line in result.report) and any("500" in line for line in result.report)
    assert any(l == "stake:500" for l in rig.site.log)                    # the stake was entered and re-checked
    assert rig.site.closed                                                # the browser is closed afterwards


def test_dry_run_never_clicks_even_when_place_is_the_only_visible_button_and_the_driver_would_allow_it(tmp_path):
    rig = Rig(tmp_path, dry_run=True)
    rig.site.allow_place = True                    # a driver that WOULD click, and a Place button that is visible
    token = rig.token()
    result = run(rig.bm.place(token))
    assert result.kind is PlacementKind.DRY_RUN and rig.site.place_clicks == 0


def test_a_used_token_cannot_be_used_again(rig):
    token = rig.token()
    run(rig.bm.place(token))
    rig.site.rows.clear()
    token2 = rig.token(expires=NOW + timedelta(minutes=4))   # a fresh confirmation: its own expiry, its own hash
    with pytest.raises(BookmakerError, match="already used"):
        run(rig.bm.place(token))
    assert run(rig.bm.place(token2)).kind is PlacementKind.DRY_RUN


def test_only_a_real_untampered_token_is_accepted(rig):
    with pytest.raises(BookmakerError, match="not a valid confirmation"):
        run(rig.bm.place(SimpleNamespace(token_hash="x")))
    token = rig.token()
    forged = object.__new__(PlacementToken)
    for name in ("session_id", "legs", "total_odds", "expires_at", "token_hash", "authority"):
        object.__setattr__(forged, name, getattr(token, name))
    object.__setattr__(forged, "stake", Decimal("99999"))
    with pytest.raises(BookmakerError, match="not a valid confirmation"):
        run(rig.bm.place(forged))
    assert rig.site.place_clicks == 0


def test_an_expired_token_is_refused_without_a_click(live):
    token = live.token(expires=NOW + timedelta(minutes=5))
    live.bm._clock = lambda: NOW + timedelta(minutes=6)
    with pytest.raises(BookmakerError, match="expired"):
        run(live.bm.place(token))
    assert live.site.place_clicks == 0


# ---------------------------------------------------------------- place: the betslip must match the token exactly

def test_a_missing_leg_at_place_time_refuses_and_never_clicks(live):
    token = live.token()
    live.site.rows.pop()
    with pytest.raises(BetslipMismatch, match="no longer matches"):
        run(live.bm.place(token))
    assert live.site.place_clicks == 0 and "PLACE" not in live.site.log


def test_an_extra_leg_at_place_time_refuses(live):
    token = live.token()
    live.site.rows.append(BetslipLine("Stray v Team", "1X2", "Home", Decimal("2.00")))
    with pytest.raises(BetslipMismatch):
        run(live.bm.place(token))
    assert live.site.place_clicks == 0


def test_a_different_line_at_place_time_refuses(live):
    token = live.token()
    live.site.rows[0] = BetslipLine("Alpha v Beta", "Over/Under", "Over 3.5", Decimal("1.85"))
    with pytest.raises(BetslipMismatch):
        run(live.bm.place(token))
    assert live.site.place_clicks == 0


def test_moved_odds_at_place_time_raise_odds_changed_with_the_new_slip_and_never_click(live):
    token = live.token()
    live.site.odds_shift = Decimal("0.10")
    with pytest.raises(OddsChanged) as caught:
        run(live.bm.place(token))
    assert live.site.place_clicks == 0
    assert [l.odds for l in caught.value.loaded.legs] == [Decimal("1.95"), Decimal("1.40")]
    assert caught.value.loaded.total_odds == Decimal("2.73")
    # the page stays open, so a NEW confirmation of the NEW summary can re-read the same betslip and place
    assert live.site.closed is False
    fresh = Summary("abcd1234", caught.value.loaded.legs, caught.value.loaded.total_odds, Decimal("500"),
                    NOW + timedelta(minutes=4))
    live.site.history_after_click = [new_bet(live)]
    result = run(live.bm.place(mint_token(MINT_AUTHORITY, fresh)))
    assert result.kind is PlacementKind.PLACED and live.site.place_clicks == 1


def test_a_stake_that_did_not_go_into_the_box_refuses(live):
    token = live.token()
    live.site.stake_ignored = True
    live.site.stake = Decimal("100")              # the site's own default stake
    with pytest.raises(BetslipMismatch, match="stake"):
        run(live.bm.place(token))
    assert live.site.place_clicks == 0


def test_a_logged_out_page_at_place_time_refuses(live):
    token = live.token()
    live.site.logged_in = False
    with pytest.raises(BookmakerError, match="not logged in"):
        run(live.bm.place(token))
    assert live.site.place_clicks == 0


def test_placing_without_a_loaded_betslip_refuses(live):
    token = live.token()
    live.bm._site = None
    with pytest.raises(BetslipMismatch, match="not loaded any more"):
        run(live.bm.place(token))


# ---------------------------------------------------------------- place: real, and proof

def test_success_is_reported_only_after_the_bet_appears_in_the_history(live):
    token = live.token("500")
    live.site.history = []
    live.site.history_after_click = [new_bet(live)]
    result = run(live.bm.place(token))
    assert result.kind is PlacementKind.PLACED and result.bet_id == new_bet(live).bet_id and result.booking_code == ""
    log = live.site.log
    assert log.index("history") < log.index("PLACE") < len(log) - 1 and log.count("PLACE") == 1    # snapshot first
    assert live.site.closed


def test_a_bet_that_was_already_in_the_history_is_not_proof(live):
    token = live.token("500")
    live.site.history = [new_bet(live)]                       # already there BEFORE the click
    live.site.history_after_click = [new_bet(live)]           # nothing new after it
    with pytest.raises(PlacementUnknown, match="not found in your SportyBet bet history"):
        run(live.bm.place(token))
    assert live.site.place_clicks == 1


def test_a_click_with_no_bet_ever_showing_up_is_unknown_never_success(live, monkeypatch):
    monkeypatch.setattr(bs, "HISTORY_POLLS", 2)
    token = live.token()
    live.site.history_after_click = []
    with pytest.raises(PlacementUnknown):
        run(live.bm.place(token))
    assert live.site.place_clicks == 1 and live.site.closed


def test_an_error_on_the_click_is_unknown_because_it_cannot_be_proven_not_clicked(live):
    token = live.token()
    live.site.click_error = RuntimeError("target closed")
    with pytest.raises(PlacementUnknown, match="RuntimeError|UNKNOWN|may or may"):
        run(live.bm.place(token))
    assert live.site.place_clicks == 1


def test_a_history_that_cannot_be_read_before_the_click_means_no_click(live):
    token = live.token()
    live.site.history_error_before_click = True
    with pytest.raises(BookmakerError, match="history could not be read"):
        run(live.bm.place(token))
    assert live.site.place_clicks == 0


def test_a_history_bet_with_another_stake_or_other_games_is_not_ours(live):
    token = live.token("500")
    assert history_bet_is_ours(new_bet(live, "500"), token)
    assert not history_bet_is_ours(new_bet(live, "600"), token)
    from fakes_bookmaker import history_entry
    assert not history_bet_is_ours(history_entry(("Zulu v Yankee", "Alpha v Beta"), "500"), token)
    assert not history_bet_is_ours(HistoryBet("X2", "", None, None, (), "Alpha v Beta"), token)


def test_two_new_matching_bets_are_ambiguous_and_never_guessed(live, monkeypatch):
    monkeypatch.setattr(bs, "HISTORY_POLLS", 1)
    token = live.token("500")
    live.site.history_after_click = [new_bet(live, bet_id="BET1"), new_bet(live, bet_id="BET2")]
    with pytest.raises(PlacementUnknown):
        run(live.bm.place(token))


# ================================================================ credentials and the secrets folder

def test_credentials_never_show_in_repr_and_reject_empty_values():
    c = Credentials("08031234567", "hunter2")
    assert "0803" not in repr(c) and "hunter2" not in repr(c) and "hunter2" not in str(c)
    with pytest.raises(BookmakerError):
        Credentials("", "x")
    with pytest.raises(BookmakerError):
        Credentials("0803", "")


@pytest.mark.parametrize("phone,typed", [("08031234567", "8031234567"), ("+234 803 123 4567", "8031234567"),
                                         ("2348031234567", "8031234567"), ("8031234567", "8031234567")])
def test_the_phone_is_typed_without_country_code_or_leading_zero(phone, typed):
    assert Credentials(phone, "pw").national_phone() == typed


def test_the_storage_state_folder_must_be_named_secrets_and_not_be_synced(tmp_path):
    ok = check_secrets_dir(tmp_path / "secrets")
    assert ok.name == "secrets" and (ok / ".gitignore").read_text() == "*\n"
    with pytest.raises(BookmakerError, match="named 'secrets'"):
        check_secrets_dir(tmp_path / "state")
    for synced in ("OneDrive", "Dropbox", "Google Drive", "iCloudDrive"):
        with pytest.raises(BookmakerError, match="synced"):
            check_secrets_dir(tmp_path / synced / "bot" / "secrets")


def test_the_default_locations_are_secrets_and_local_data_inside_the_project():
    assert bs.SECRETS_DIR == REPO / "secrets" and bs.SCREENSHOT_DIR == REPO / "local_data" / "screenshots"


def test_screenshots_are_never_sent_anywhere():
    source = (REPO / "predictions" / "adapters" / "sportybet" / "betslip.py").read_text(encoding="utf-8")
    assert "telegram" not in source.lower().replace("never sent to telegram", "")
    app_source = (REPO / "app.py").read_text(encoding="utf-8")
    assert "send_photo" not in app_source and "send_document" not in app_source and "reply_photo" not in app_source


# ================================================================ the page driver against a fake Playwright page

class FakeDom:
    """A tiny model of the anonymous SportyBet pages, driven only through the selector strings in SELECTORS."""

    def __init__(self):
        self.events = {}            # event_ref -> (match_text, {(title, outcome): odds})
        self.rows = []              # [match_text, market, outcome, odds]
        self.current = None
        self.dialog_open = False
        self.stake = ""
        self.clicks = []            # every selector clicked
        self.fills = []             # (selector, value)
        self.login_form = False
        self.logged_in_after_login = True
        self.growth_broken = False
        self.odds_text = None       # override for an unreadable price
        self.gotos = []


class FakeLocator:
    def __init__(self, page, selector, index=None, row=None):
        self.page, self.selector, self.index, self.row = page, selector, index, row

    @property
    def dom(self):
        return self.page.dom

    def _elements(self):
        d, s, sel = self.dom, SELECTORS, self.selector
        if self.row is not None:                       # a sub-locator inside one betslip row
            r = d.rows[self.row]
            table = {s.item_odds: (str(r[3]) if d.odds_text is None else d.odds_text), s.item_teams: r[0],
                     s.item_market: r[1], s.item_outcome: r[2]}
            return [table[sel]] if sel in table else []
        if sel == s.betslip_items:
            return list(range(len(d.rows)))
        if sel == s.remove_all:
            return ["x"] if len(d.rows) >= 2 else []
        if sel == s.item_delete:
            return list(range(len(d.rows)))
        if sel == s.dialog_ok:
            return ["ok"] if d.dialog_open else []
        if sel == s.dialog:
            return ["dialog"] if d.dialog_open else []
        if sel == s.total_odds:
            if len(d.rows) < 2:
                return []
            total = Decimal(1)
            for r in d.rows:
                total *= Decimal(str(r[3]))
            return [str(total.quantize(Decimal("0.01")))]
        if sel == s.stake_input:
            return ["stake"]
        if sel == s.place_button:
            return ["place"]                                  # ALWAYS visible: the worst case for a dry run
        if sel == s.login_phone:
            return ["phone"] if d.login_form else []
        if sel == s.login_password:
            return ["pw"]
        if sel == s.login_button:
            return ["login"]
        if d.current is not None:
            _, offers = d.events[d.current]
            found = [k for k in offers if s.outcome_cell(*k) == sel]
            return found
        return []

    async def count(self):
        return len(self._elements())

    @property
    def first(self):
        return FakeLocator(self.page, self.selector, 0, self.row)

    def nth(self, i):
        return FakeLocator(self.page, self.selector, i, self.row)

    def locator(self, sub):
        return FakeLocator(self.page, sub, None, self.index)

    async def is_visible(self):
        return len(self._elements()) > 0

    async def click(self):
        d, s, sel = self.dom, SELECTORS, self.selector
        d.clicks.append(sel)
        if sel == s.remove_all:
            d.dialog_open = True
        elif sel == s.dialog_ok:
            d.rows.clear()
            d.dialog_open = False
        elif sel == s.item_delete:
            if d.rows:
                d.rows.pop(self.index or 0)
        elif sel == s.login_button:
            d.login_form = not d.logged_in_after_login
        elif d.current is not None and sel != s.place_button:
            key = self._elements()[0]
            if not d.growth_broken:
                match_text, offers = d.events[d.current]
                d.rows.append([match_text, key[0], key[1], offers[key]])

    async def fill(self, value):
        self.dom.fills.append((self.selector, value))
        if self.selector == SELECTORS.stake_input:
            self.dom.stake = value

    async def inner_text(self):
        els = self._elements()
        return els[self.index or 0] if self.row is not None or self.selector == SELECTORS.total_odds else "text"

    async def input_value(self):
        return self.dom.stake


class FakePage:
    def __init__(self):
        self.dom = FakeDom()
        self.shots = []

    async def goto(self, url, **kw):
        self.dom.gotos.append(url)
        for ref in self.dom.events:
            if url == SELECTORS.event_page(ref):
                self.dom.current = ref

    async def wait_for_timeout(self, ms):
        return None

    def locator(self, selector):
        return FakeLocator(self, selector)

    async def screenshot(self, path):
        self.shots.append(path)


class FakeItemLocator(FakeLocator):
    pass


# the betslip rows are read through items.nth(i).locator(...): give FakeLocator that behaviour
_orig_nth = FakeLocator.nth


def _nth(self, i):
    if self.selector == SELECTORS.betslip_items:
        return FakeLocator(self.page, self.selector, i, i)      # `row` = i marks "inside row i"
    return _orig_nth(self, i)


FakeLocator.nth = _nth


def make_site(allow_place=False, credentials=None, tmp_path=None):
    page = FakePage()
    context = SimpleNamespace(storage_state=None, close=None, new_page=None)
    saved = []

    async def storage_state(path):
        saved.append(path)

    async def close():
        return None

    context.storage_state, context.close = storage_state, close
    site = SportyBetSite(page, context, credentials, (tmp_path / "state.json") if tmp_path else None, allow_place)
    site.saved = saved
    return site, page.dom


def add_event(dom, ref, match, market, outcomes):
    dom.events[ref] = (match, {(market, name): Decimal(p) for name, p in outcomes.items()})


def test_the_driver_adds_reads_and_clears_a_betslip_including_the_single_row_case():
    site, dom = make_site()
    add_event(dom, "sr:match:1", "Alpha v Beta", "Over/Under", {"Over 2.5": "1.85", "Under 2.5": "2.00"})
    add_event(dom, "sr:match:2", "Gamma v Delta", "1X2", {"Home": "2.00"})
    run(site.add_outcome("sr:match:1", ["Over/Under"], "Over 2.5"))
    assert SELECTORS.event_page("sr:match:1") in dom.gotos
    single = run(site.read_betslip())
    assert single.total_odds == Decimal("1.85") and single.lines[0].outcome_name == "Over 2.5"   # one row: its own odds
    run(site.add_outcome("sr:match:2", ["1X2"], "Home"))
    two = run(site.read_betslip())
    assert [l.match_text for l in two.lines] == ["Alpha v Beta", "Gamma v Delta"] and two.total_odds == Decimal("3.70")
    run(site.clear_betslip())                                        # Remove All + OK for two rows
    assert dom.rows == [] and SELECTORS.remove_all in dom.clicks and SELECTORS.dialog_ok in dom.clicks
    run(site.add_outcome("sr:match:1", ["Over/Under"], "Over 2.5"))
    dom.clicks.clear()
    run(site.clear_betslip())                                        # one row: the delete icon
    assert dom.rows == [] and dom.clicks == [SELECTORS.item_delete]


def test_the_driver_needs_exactly_one_matching_option_on_the_page():
    site, dom = make_site()
    add_event(dom, "sr:match:1", "Alpha v Beta", "Over/Under", {"Over 2.5": "1.85"})
    with pytest.raises(BookmakerError, match="not found on the game's page"):
        run(site.add_outcome("sr:match:1", ["Over/Under"], "Over 9.5"))
    assert dom.rows == []
    with pytest.raises(BookmakerError, match="not found on the game's page"):
        run(site.add_outcome("sr:match:1", ["Wrong title"], "Over 2.5"))


def test_the_driver_fails_when_a_click_does_not_reach_the_betslip():
    site, dom = make_site()
    add_event(dom, "sr:match:1", "Alpha v Beta", "Over/Under", {"Over 2.5": "1.85"})
    dom.growth_broken = True
    with pytest.raises(BookmakerError, match="did not appear on the betslip"):
        run(site.add_outcome("sr:match:1", ["Over/Under"], "Over 2.5"))


def test_the_driver_fails_closed_on_unreadable_odds():
    site, dom = make_site()
    add_event(dom, "sr:match:1", "Alpha v Beta", "Over/Under", {"Over 2.5": "1.85"})
    run(site.add_outcome("sr:match:1", ["Over/Under"], "Over 2.5"))
    dom.odds_text = "SUSPENDED"
    with pytest.raises(BookmakerError, match="odds of a betslip selection could not be read"):
        run(site.read_betslip())


def test_the_driver_puts_the_stake_in_and_reads_it_back():
    site, dom = make_site()
    add_event(dom, "sr:match:1", "Alpha v Beta", "Over/Under", {"Over 2.5": "1.85"})
    run(site.add_outcome("sr:match:1", ["Over/Under"], "Over 2.5"))
    run(site.set_stake(Decimal("1250.50")))
    assert dom.fills[-1] == (SELECTORS.stake_input, "1250.50")
    assert run(site.read_betslip()).stake == Decimal("1250.50")


def test_a_driver_built_for_dry_run_cannot_click_place_even_though_the_button_is_visible():
    site, dom = make_site(allow_place=False)
    assert run(site.page.locator(SELECTORS.place_button).first.is_visible()) is True     # Place is right there
    with pytest.raises(BookmakerError, match="never clicked"):
        run(site.click_place())
    with pytest.raises(BookmakerError):
        run(site.confirm_placement())
    assert SELECTORS.place_button not in dom.clicks


def test_only_a_driver_built_to_place_clicks_the_place_button():
    site, dom = make_site(allow_place=True)
    run(site.click_place())
    assert dom.clicks == [SELECTORS.place_button]


def test_login_types_the_national_number_saves_state_and_skips_when_already_logged_in(tmp_path):
    creds = Credentials("08031234567", "s3cret-pass")
    site, dom = make_site(credentials=creds, tmp_path=tmp_path)
    dom.login_form = True
    run(site.login())
    assert (SELECTORS.login_phone, "8031234567") in dom.fills and (SELECTORS.login_password, "s3cret-pass") in dom.fills
    assert SELECTORS.login_button in dom.clicks and site.saved == [str(tmp_path / "state.json")]
    again, dom2 = make_site(credentials=creds, tmp_path=tmp_path)
    dom2.login_form = False                                  # the stored session is still valid: no form
    run(again.login())
    assert dom2.fills == [] and again.saved == []


def test_a_login_that_does_not_complete_fails_without_saving_anything(tmp_path):
    site, dom = make_site(credentials=Credentials("08031234567", "pw"), tmp_path=tmp_path)
    dom.login_form, dom.logged_in_after_login = True, False
    with pytest.raises(BookmakerError, match="login did not complete"):
        run(site.login())
    assert site.saved == []
    nobody, dom3 = make_site(credentials=None)
    dom3.login_form = True
    with pytest.raises(BookmakerError, match="no login details"):
        run(nobody.login())


def test_all_selectors_are_in_one_labelled_structure_and_the_rest_of_the_project_knows_none_of_them():
    fields = [f for f in SELECTORS.__dataclass_fields__]
    assert "place_button" in fields and "betslip_items" in fields and "login_phone" in fields
    source = (REPO / "predictions" / "adapters" / "sportybet" / "betslip.py").read_text(encoding="utf-8")
    assert source.count("VERIFIED-ANON") >= 10 and source.count("UNVERIFIED") >= 4
    for path in (REPO / "predictions").rglob("*.py"):
        if path.name == "betslip.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "j_betslip" not in text and "m-table-cell" not in text and "es-dialog" not in text, path.name


# ---------- diagnostics after the click (2026-10-01: a click that made no bet had left no trace of what the page said)

def test_what_the_page_showed_after_place_is_in_the_unknown_message(live, monkeypatch):
    monkeypatch.setattr(bs, "HISTORY_POLLS", 2)
    seen = []

    async def observe(shot_dir, stamp, label):
        seen.append(label)
        return "Confirm your bet: stake 500, potential win 825" if label == "click" else ""

    live.site.observe_after_click = observe
    live.site.history_after_click = []
    token = live.token()
    with pytest.raises(PlacementUnknown) as err:
        run(live.bm.place(token))
    assert "Confirm your bet: stake 500, potential win 825" in str(err.value)
    assert seen == ["click", "confirm"] and live.site.place_clicks == 1


def test_a_page_that_says_nothing_after_place_is_reported_as_such(live, monkeypatch):
    monkeypatch.setattr(bs, "HISTORY_POLLS", 2)
    live.site.history_after_click = []
    with pytest.raises(PlacementUnknown) as err:
        run(live.bm.place(live.token()))
    assert "showed no message after Place" in str(err.value)


def test_a_failing_observation_never_changes_the_outcome(live):
    async def broken(shot_dir, stamp, label):
        raise RuntimeError("page closed")

    live.site.observe_after_click = broken
    token = live.token("500")
    live.site.history = []
    live.site.history_after_click = [new_bet(live)]
    result = run(live.bm.place(token))
    assert result.kind is PlacementKind.PLACED   # the diagnostic cannot break a placement that is provable


# ---------- 2026-10-01: a win celebration ('FLASHWIN') covered the page and the Place click did nothing

class _PopupPage:
    """A page whose visible pop-up text is `popup` until Escape (or the close control) is used."""

    def __init__(self, popup, escape_closes=True):
        self.popup, self.escape_closes, self.keys, self.closed_by_click = popup, escape_closes, [], 0
        self.keyboard = SimpleNamespace(press=self._press)

    async def _press(self, key):
        self.keys.append(key)
        if key == "Escape" and self.escape_closes:
            self.popup = ""

    async def evaluate(self, js):
        return self.popup

    async def wait_for_timeout(self, ms):
        return None

    def locator(self, selector):
        page = self

        class _Close:
            async def count(self_inner):
                return 1

            @property
            def first(self_inner):
                return self_inner

            async def is_visible(self_inner):
                return True

            async def click(self_inner):
                page.closed_by_click += 1
                page.popup = ""

        return _Close()


def _site_with(page):
    return SportyBetSite(page, None, None, None)


def test_a_promotional_popup_is_closed_with_escape():
    page = _PopupPage("FLASHWIN\nNGN 19,317.77\nFrom Sport / Ticket ID 813751\nDetails")
    closed = run(_site_with(page).dismiss_popups())
    assert page.keys == ["Escape"] and page.popup == "" and page.closed_by_click == 0
    assert closed and closed[0].startswith("FLASHWIN")


def test_the_close_control_is_used_when_escape_does_not_close_it():
    page = _PopupPage("Congratulations, you won a gift", escape_closes=False)
    run(_site_with(page).dismiss_popups())
    assert page.closed_by_click == 1 and page.popup == ""


@pytest.mark.parametrize("text", ["Confirm your bet", "Place Bet: stake 500", "Odds have changed. Accept?",
                                  "Your betslip has 2 selections"])
def test_a_dialog_about_betting_is_never_touched(text):
    page = _PopupPage(text)
    assert run(_site_with(page).dismiss_popups()) == []
    assert page.keys == [] and page.closed_by_click == 0 and page.popup == text


def test_no_popup_means_nothing_is_pressed():
    page = _PopupPage("")
    assert run(_site_with(page).dismiss_popups()) == [] and page.keys == []


def test_dismissing_popups_never_raises():
    class Broken:
        async def evaluate(self, js):
            raise RuntimeError("page closed")

    assert run(_site_with(Broken()).dismiss_popups()) == []


def test_the_popup_is_dismissed_before_the_click_and_a_failure_changes_nothing(live):
    order = []

    async def dismiss():
        order.append("dismiss")

    original = live.site.click_place

    async def click():
        order.append("click")
        await original()

    live.site.dismiss_popups, live.site.click_place = dismiss, click
    live.site.history = []
    live.site.history_after_click = [new_bet(live)]
    run(live.bm.place(live.token("500")))
    assert order == ["dismiss", "click"]

    async def broken():
        raise RuntimeError("boom")

    live2_site = live.site
    live2_site.dismiss_popups = broken   # a failing dismissal must not stop or change the placement logic
    assert live2_site.dismiss_popups is broken


def test_the_servers_answers_to_the_order_request_reach_the_unknown_message(live, monkeypatch):
    monkeypatch.setattr(bs, "HISTORY_POLLS", 2)

    async def observe(shot_dir, stamp, label):
        return "server: /api/ng/orders/order -> HTTP 200 {\"bizCode\":19999,\"message\":\"Selection unavailable\"}" if label == "click" else ""

    live.site.observe_after_click = observe
    live.site.history_after_click = []
    with pytest.raises(PlacementUnknown) as err:
        run(live.bm.place(live.token()))
    assert "/api/ng/orders/order -> HTTP 200" in str(err.value) and "Selection unavailable" in str(err.value)



def flashing(site, answers):
    """Make the fake site's logged_out() answer from `answers` (then the last one forever), counting the calls."""
    calls = []

    async def logged_out():
        calls.append(1)
        return answers[min(len(calls) - 1, len(answers) - 1)]

    site.logged_out = logged_out
    return calls


def with_diagnostics(site):
    saved = []

    async def login_screenshot(path):
        saved.append(("png", path.name))

    async def login_diagnostics():
        return "matches: 1"

    site.login_screenshot, site.login_diagnostics = login_screenshot, login_diagnostics
    return saved


def test_a_login_form_that_only_flashes_at_place_time_is_rechecked_and_the_bet_goes_on(live):
    token = live.token()
    saved = with_diagnostics(live.site)
    calls = flashing(live.site, [True, True, False])           # the form shows twice, then is gone
    try:
        run(live.bm.place(token))
    except Exception as error:   # whatever happens after the click is another test's business: it just must not be 'not logged in'
        assert "not logged in" not in str(error)
    assert len(calls) >= 3 and live.site.place_clicks == 1       # the form was re-checked, then the bet went on to the click
    assert len(saved) == 1 and saved[0][1].startswith("logincheck-place-")   # one screenshot, taken when it first showed


def test_a_login_form_that_stays_is_refused_after_the_rechecks_with_one_saved_diagnostic(live):
    token = live.token()
    saved = with_diagnostics(live.site)
    calls = flashing(live.site, [True])
    with pytest.raises(BookmakerError, match="not logged in"):
        run(live.bm.place(token))
    assert len(saved) == 1 and live.site.place_clicks == 0
    from predictions.adapters.sportybet.betslip import LOGIN_RECHECKS
    assert len(calls) >= LOGIN_RECHECKS + 1


def test_the_login_check_at_load_time_is_rechecked_too(rig):
    saved = with_diagnostics(rig.site)
    flashing(rig.site, [True, False])
    loaded = run(rig.bm.load_slip(rig.legs))
    assert loaded is not None and len(saved) == 1 and saved[0][1].startswith("logincheck-load-")


# ---- a session that is lost WHILE the games are being added is named, not reported as an unexpected failure
def test_a_session_lost_during_the_adds_is_named_when_the_driver_times_out(rig):
    async def times_out(*a, **k):
        rig.site.logged_in = False                  # SportyBet ended the session as the event page opened
        raise TimeoutError()
    rig.site.add_outcome = times_out
    with pytest.raises(BookmakerError, match="logged your account out while the games were being added"):
        run(rig.bm.load_slip(rig.legs))
    assert rig.site.closed and "PLACE" not in rig.site.log


def test_a_session_lost_after_the_last_add_is_caught_before_the_slip_is_read(rig):
    original = rig.site.add_outcome

    async def add_then_lose(*a, **k):
        await original(*a, **k)
        rig.site.logged_in = False
    rig.site.add_outcome = add_then_lose
    with pytest.raises(BookmakerError, match="logged your account out"):
        run(rig.bm.load_slip(rig.legs))
    assert "read" not in rig.site.log


def test_a_driver_error_with_the_account_still_logged_in_stays_an_unexpected_failure(rig):
    class DriverTimeout(Exception):      # Playwright's own TimeoutError is not asyncio's
        pass

    async def boom(*a, **k):
        raise DriverTimeout()
    rig.site.add_outcome = boom
    with pytest.raises(BookmakerError, match="failed unexpectedly"):
        run(rig.bm.load_slip(rig.legs))
