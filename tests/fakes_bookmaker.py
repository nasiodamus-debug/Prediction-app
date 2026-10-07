"""Test doubles for Epic 5: a FakeBookmaker that implements the Bookmaker port, and builders for listings and picks.

The fake uses the REAL availability logic (core.betting.assess_availability), so what it says about a game or an
option is decided by the same code the real adapter uses. It records every call and every "click"."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from predictions.core.betting import (Availability, FailedLeg, LoadAttempt, LoadedLeg, LoadedSlip, assess_availability, leg_from_evaluation,
                                      total_odds)
from predictions.core.inputs import SOCCER_REGISTRY, Input
from predictions.core.model import Estimate, Evaluation
from predictions.core.placement import PlacementKind, PlacementResult, PlacementToken
from predictions.core.ports import BetslipMismatch, BookmakerError, OddsChanged, PlacementUnknown
from predictions.core.types import Fixture, MarketOffer, MatchId, Outcome

KICKOFF = datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)   # before KICKOFF
COMPETITION = "Testland - League"


def xg_inputs(attack=2.5, defence=1.5):
    values = {"matches": 10, "xg_for": attack, "goals_for": attack, "xg_against": defence, "goals_against": defence}
    return tuple(Input(f"{side}.{stat}", float(v), SOCCER_REGISTRY[stat].unit, "test")
                 for side in ("home", "away") for stat, v in values.items())


def match_id(home="Alpha", away="Beta", hour=14):
    return MatchId("soccer", COMPETITION, KICKOFF.replace(hour=hour), home, away)


def totals_offer(line, over, under):
    line = str(line)
    return MarketOffer("18", "Over/Under", f"total={line}",
                       (Outcome(f"Over {line}", Decimal(over), "12"), Outcome(f"Under {line}", Decimal(under), "13")))


def result_offer(home="2.00", draw="3.20", away="3.60"):
    return MarketOffer("1", "1X2", "", (Outcome("Home", Decimal(home), "1"), Outcome("Draw", Decimal(draw), "2"),
                                       Outcome("Away", Decimal(away), "3")))


def make_fixture(home="Alpha", away="Beta", offers=(), hour=14, started=False, ref=None):
    mid = match_id(home, away, hour)
    return Fixture(mid, COMPETITION, home, away, mid.kickoff_utc, started, tuple(offers),
                   ref if ref is not None else f"sr:match:{abs(hash((home, away))) % 10**6}")


def pick_evaluation(home="Alpha", away="Beta", market="Over/Under 2.5", selection="Over 2.5", odds="1.85", prob=0.72,
                    hour=14, inputs=None, push=0.0):
    est = Estimate(prob, xg_inputs() if inputs is None else inputs, "poisson", push)
    return Evaluation(match_id(home, away, hour), market, selection, Decimal(odds), est, Decimal("0.10"), True,
                      "passed", "", "", ())


def two_pick_setup():
    """Two confirmed picks and a listing that offers both exactly as picked."""
    a = pick_evaluation("Alpha", "Beta", "Over/Under 2.5", "Over 2.5", "1.85")
    b = pick_evaluation("Gamma", "Delta", "Over/Under 1.5", "Over 1.5", "1.30", prob=0.88, hour=16)
    listing = [make_fixture("Alpha", "Beta", [totals_offer("2.5", "1.85", "2.00")], 14),
               make_fixture("Gamma", "Delta", [totals_offer("1.5", "1.30", "3.20")], 16)]
    return [a, b], listing


class FakeBookmaker:
    """Implements core.ports.Bookmaker. `place_mode` chooses what place() does; see its branches."""

    def __init__(self, listing=(), dry_run=True, now=NOW):
        self.listing = list(listing)
        self._dry_run = dry_run
        self.now = now
        self.calls: list[tuple] = []
        self.clicks = 0                    # times the fake "clicked Place": must stay 0 in dry run
        self.odds_override: dict = {}      # leg key -> Decimal: what the betslip shows instead of the listing
        self.check_error = None
        self.load_error = None
        self.pick_failures: dict = {}      # leg key -> plain reason: that pick "cannot be added" (AD-26); the others still load
        self.unknown_code, self.unknown_tried = "", False   # the Book Bet fallback result carried by the unknown error
        self.place_mode = "ok"             # ok | odds_changed | mismatch | unknown | error | unexpected | no_proof
        self.used: set[str] = set()
        self.loaded: LoadedSlip | None = None
        self.bet_id = "B1234567"
        self.booking_code = "ABC123"
        self.history: tuple = ()
        self.book_clicks = 0               # times the fake pressed Book Bet (PLACE_MODE=book); never Place
        self.book_code = "HW63TV"          # what Book Bet shows; "" = no code could be read

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    async def check_availability(self, legs):
        self.calls.append(("check", tuple(l.key for l in legs)))
        if self.check_error is not None:
            raise self.check_error
        return assess_availability(legs, self.listing, self.now)

    def _loaded_for(self, legs) -> LoadedSlip:
        checked = assess_availability(legs, self.listing, self.now)
        loaded = []
        for found in checked:
            if found.status is not Availability.OK:
                raise BookmakerError("A confirmed game or option is no longer available. Nothing was loaded.")
            odds = self.odds_override.get(found.leg.key, found.outcome.odds)
            loaded.append(LoadedLeg(found.leg.key, f"{found.fixture.home} v {found.fixture.away}", found.offer.name,
                                    found.outcome.name, odds, found.leg.selection, found.leg.odds))
        return LoadedSlip(tuple(loaded), total_odds(l.odds for l in loaded))

    async def load_slip(self, legs):
        self.calls.append(("load", tuple(l.key for l in legs)))
        if self.load_error is not None:
            raise self.load_error
        self.loaded = self._loaded_for(legs)
        return self.loaded

    async def load_legs(self, legs):
        """AD-26: like load_slip, but a pick named in `pick_failures` is skipped and reported; the others are loaded."""
        if not any(l.key in self.pick_failures for l in legs):
            return LoadAttempt(await self.load_slip(legs), ())
        self.calls.append(("load", tuple(l.key for l in legs)))
        if self.load_error is not None:
            raise self.load_error
        failed = tuple(FailedLeg(l.key, self.pick_failures[l.key]) for l in legs if l.key in self.pick_failures)
        ok = [l for l in legs if l.key not in self.pick_failures]
        self.loaded = self._loaded_for(ok) if ok else None
        return LoadAttempt(self.loaded, failed)

    async def place(self, token: PlacementToken):
        self.calls.append(("place", token.token_hash))
        if token.token_hash in self.used:
            raise BookmakerError("That confirmation was already used. Nothing was placed.")
        self.used.add(token.token_hash)
        mode = self.place_mode
        if token.booking:   # AD-25: the same checks as above, then ONLY Book Bet; Place and Confirm are never touched
            if mode == "mismatch":
                raise BetslipMismatch("The SportyBet betslip no longer matches what you confirmed. Nothing was placed.")
            if mode == "odds_changed":
                moved = tuple(LoadedLeg(l.leg_key, l.match_text, l.market_name, l.outcome_name, l.odds + Decimal("0.10"),
                                        l.selection, l.evaluation_odds) for l in token.legs)
                raise OddsChanged("odds changed", LoadedSlip(moved, total_odds(l.odds for l in moved)))
            if self._dry_run:
                return PlacementResult(PlacementKind.DRY_RUN, ("would book: " + ", ".join(l.match_text for l in token.legs),))
            self.calls.append(("book",))
            self.book_clicks += 1
            if mode == "unexpected":
                raise RuntimeError("boom")
            if mode == "error" or not self.book_code:
                raise BookmakerError("The booking code could not be read.")
            return PlacementResult(PlacementKind.BOOKED, (), "", self.book_code)
        if mode == "mismatch":
            raise BetslipMismatch("The SportyBet betslip no longer matches what you confirmed. Nothing was placed.")
        if mode == "odds_changed":
            moved = tuple(LoadedLeg(l.leg_key, l.match_text, l.market_name, l.outcome_name, l.odds + Decimal("0.10"),
                                    l.selection, l.evaluation_odds) for l in token.legs)
            raise OddsChanged("odds changed", LoadedSlip(moved, total_odds(l.odds for l in moved)))
        if mode == "error":
            raise BookmakerError("The stake box was not found on the betslip.")
        if mode == "unexpected":
            raise RuntimeError("boom")
        if self._dry_run:
            return PlacementResult(PlacementKind.DRY_RUN, ("would place: " + ", ".join(l.match_text for l in token.legs),))
        self.clicks += 1
        if mode == "unknown":
            raise PlacementUnknown("Place was clicked but the bet was not found in your bet history.", self.unknown_code,
                                   self.unknown_tried)
        if mode == "no_proof":
            return PlacementResult(PlacementKind.PLACED, (), "", self.booking_code)
        return PlacementResult(PlacementKind.PLACED, (), self.bet_id, self.booking_code)

    async def bet_history(self):
        self.calls.append(("history",))
        return self.history


def history_entry(matches=("Alpha v Beta", "Gamma v Delta"), stake="500", status="Running", others=0, placed_at=None,
                  bet_type=None, ambiguous=False, bet_id=None):
    """A bet-history LIST entry exactly as the real page gives it: games shown, a status, a type, a time, a stake, and a
    synthetic key (no real bet id, no odds, no picks exist on that page)."""
    from predictions.adapters.sportybet.betslip import history_key
    from predictions.core.betting import HistoryBet
    placed_at = placed_at or NOW
    total = len(matches) + others
    kind = bet_type or ("Singles" if total == 1 else "Multiple")
    key = bet_id or history_key(placed_at, kind, Decimal(stake), matches, others)
    return HistoryBet(key, "", Decimal(stake), None, (), " | ".join(matches), total, placed_at, tuple(matches), status,
                      kind, ambiguous)
