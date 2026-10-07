"""Ports the core depends on. Standard library only."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from typing import Sequence

from predictions.core.analyst import AnalystQuery, AnalystReport
from predictions.core.betting import HistoryBet, Leg, LegAvailability, LoadAttempt, LoadedSlip
from predictions.core.inputs import Input
from predictions.core.stats import FixtureStats
from predictions.core.types import Fixture, MarketOffer, MatchId

if TYPE_CHECKING:  # review.py imports this module, so the type is only named here
    from predictions.core.review import CodeSelection
    from predictions.core.placement import PlacementResult, PlacementToken


class FixtureFetchError(Exception):
    """A fixture source could not deliver the complete listing. The message is safe to show the owner:
    it contains no URLs, credentials or file paths."""


class FixtureSource(Protocol):
    async def fetch_fixtures(self) -> list[Fixture]:
        """Every upcoming soccer fixture the source lists. Raises FixtureFetchError on any failure and
        never returns a partial list."""
        ...


class StatsFetchError(Exception):
    """A stats source could not be read. The message is safe to show the owner."""


class StatsSource(Protocol):
    async def fetch_stats(self, fixtures: list[Fixture]) -> dict[MatchId, FixtureStats]:
        """One FixtureStats per fixture. A league, team or game the source does not cover is INSUFFICIENT,
        an unclear team match is AMBIGUOUS; neither is guessed. Raises StatsFetchError when the source
        cannot be read, and never returns partial results."""
        ...


class InputsSource(Protocol):
    async def fetch_inputs(self, fixtures: list[Fixture]) -> dict[MatchId, list[Input]]:
        """Extra provenance inputs (corners, cards, referee) per fixture; a fixture it has nothing for is
        simply absent. Raises StatsFetchError when the source cannot be read, and never returns partial results."""
        ...


class EventMarketsSource(Protocol):
    async def fetch_event_markets(self, fixtures: list[Fixture]) -> "dict[MatchId, tuple[MarketOffer, ...] | None]":
        """The corners and cards markets (spec 11-1) one game lists, looked up once per game on the bookmaker's per-event
        endpoint (bounded concurrency, an overall time limit, no retries). A game that lists neither maps to an empty tuple;
        a game whose lookup FAILED (or was not reached in time) maps to None. Never raises for one game's failure."""
        ...


class AnalystUnavailable(Exception):
    """The analyst service could not be used (no network, outage, refusal of the key). The message is safe to
    show the owner: it contains no URLs, keys or file paths."""


class Analyst(Protocol):
    async def research(self, query: AnalystQuery) -> AnalystReport:
        """One research call about one fixture. It receives only identifiers (AnalystQuery) and returns only
        schema-validated data (AnalystReport); a malformed answer comes back as a report with `rejected` set,
        nothing repaired. Raises AnalystUnavailable when the service cannot be used."""
        ...


class CodeReadError(Exception):
    """A booking code could not be read (invalid, expired, empty, or the service failed). The message is safe to
    show the owner: it contains no URLs, paths or raw responses."""


class CodeReader(Protocol):
    async def read_code(self, code: str) -> "list[CodeSelection]":
        """Every selection on a SportyBet booking code, read WITHOUT touching anyone's betslip: an implementation
        uses a fresh, anonymous, throwaway context and never the owner's login (AD-17). Raises CodeReadError on
        any failure and never returns a partial list; an empty code is an error too."""
        ...


class BookmakerError(Exception):
    """A bookmaker operation failed BEFORE anything irreversible happened: no bet was placed. The message is safe to
    show the owner (no credentials, URLs or file paths). Never retried automatically (AD-14)."""


class PickNotAdded(BookmakerError):
    """ONE pick could not be put on the betslip (its option was not found on the game's page, it is locked or suspended, it
    did not appear after the one retry-click, or that game's page timed out). It is pick-specific: the loader skips it and
    goes on with the other picks, and the owner decides what to do with it. Anything else that fails stays a plain
    BookmakerError and fails the whole load."""


class BetslipMismatch(BookmakerError):
    """The real betslip does not hold what was confirmed (a different, missing or extra selection). Nothing was clicked."""


class OddsChanged(BookmakerError):
    """The real betslip holds exactly the confirmed selections but at different odds. Nothing was clicked. `loaded` is
    the slip as it now stands, so the owner can be shown a new summary."""

    def __init__(self, message: str, loaded: LoadedSlip):
        super().__init__(message)
        self.loaded = loaded


class PlacementUnknown(Exception):
    """Something went wrong AFTER the Place button was clicked (or it cannot be proven that it was not). The bet may or
    may not exist: it is resolved only by reading the bet history, never by placing again (AD-6).

    `booking_code` is a SportyBet booking code the adapter read from its Book Bet fallback ('' when none); `booking_tried` says
    the fallback was attempted. A booking code is NOT proof of a bet: it only lets the owner load the same picks by hand."""

    def __init__(self, message: str = "", booking_code: str = "", booking_tried: bool = False):
        super().__init__(message)
        self.booking_code, self.booking_tried = booking_code, bool(booking_tried)


class Bookmaker(Protocol):
    """Everything the bot does on the bookmaker's site, behind one door. An implementation runs one operation at a time
    (AD-17), keeps every URL and selector to itself (AD-11) and verifies what it did before it goes on."""

    @property
    def dry_run(self) -> bool:
        """True (the default) means `place` NEVER clicks Place: it checks everything and reports."""
        ...

    async def check_availability(self, legs: Sequence[Leg]) -> tuple[LegAvailability, ...]:
        """One LegAvailability per leg, in order: OK, GAME_UNAVAILABLE, or OPTION_MISSING (with the game's live listing,
        so alternatives can be judged). Does not touch the betslip. Raises BookmakerError on failure."""
        ...

    async def load_slip(self, legs: Sequence[Leg]) -> LoadedSlip:
        """Clear the owner's betslip FIRST, add exactly these legs, read the betslip back and verify it holds exactly
        them, then return what it holds with the final odds. NO bet is placed. Raises BookmakerError on any doubt."""
        ...

    async def load_legs(self, legs: Sequence[Leg]) -> LoadAttempt:
        """Like `load_slip`, but a pick that cannot be added (PickNotAdded) is SKIPPED and reported instead of failing the
        whole load: the other picks are still added, the betslip is read back and verified to hold exactly the picks that
        were added, and the result carries those (None when none was added) and one FailedLeg per skipped pick, with the
        reason. Every other failure still raises BookmakerError. NO bet is placed."""
        ...

    async def place(self, token: "PlacementToken") -> "PlacementResult":
        """Re-read the real betslip and refuse (BetslipMismatch, or OddsChanged for a price move) unless it matches the
        token exactly. With dry_run it stops there and reports; otherwise it places and returns PLACED only after the bet
        id is read back from the bet history. Raises PlacementUnknown for any failure after the click."""
        ...

    async def bet_history(self) -> tuple[HistoryBet, ...]:
        """The bets SportyBet's bet history shows, newest first. Raises BookmakerError on failure."""
        ...
