"""What a bet is made of, and how a bookmaker's betslip is checked against it (Epic 5). Standard library only.

A `Leg` is a selection the owner confirmed, in the bookmaker-neutral shape the rest of Epic 5 needs: which game,
which market, which line, which side. A `LoadedLeg` is what the bookmaker's betslip really holds for a leg after
loading (its own wording and its final odds). Everything here is pure: no browser, no network, no clock.

Fail closed: anything that cannot be read, matched or proven equal is a problem, never a guess.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import Enum
from typing import Iterable, Sequence

from predictions.core.engines.basketball import (AWAY_TOTAL_MARKET_ID as BB_AWAY_TOTAL_MARKET,
                                                 FH_TOTAL_MARKET_ID as BB_FH_TOTAL_MARKET,
                                                 HANDICAP_MARKET_ID as BB_HANDICAP_MARKET,
                                                 HOME_TOTAL_MARKET_ID as BB_HOME_TOTAL_MARKET,
                                                 TOTAL_MARKET_ID as BB_TOTAL_MARKET,
                                                 WINNER_MARKET_ID as BB_WINNER_MARKET)
from predictions.core.engines import american_football as _football
from predictions.core.engines import mlb as _mlb
from predictions.core.engines import nhl as _nhl
from predictions.core.model import Evaluation
from predictions.core.results import STATUS_RUNNING, TYPE_MULTIPLE, TYPE_SINGLES, normalise_word
from predictions.core.types import Fixture, MarketOffer, MatchId, Outcome, normalize_team

ODDS_STEP = Decimal("0.01")
TOTAL_ODDS_TOLERANCE = Decimal("0.02")  # relative: a bookmaker rounds each price, so its product can differ a little
SIDES = ("over", "under", "home", "draw", "away")

RESULT_MARKET = "1"
TOTALS_MARKET = "18"
HOME_TOTAL_MARKET = "19"
AWAY_TOTAL_MARKET = "20"
DOUBLE_CHANCE_MARKET = "10"   # spec 11-1: SportyBet Double Chance
CORNERS_MARKET = "166"        # spec 11-1: SportyBet "Corners - Over/Under" (event endpoint only)
CARDS_MARKET = "139"          # spec 11-1: SportyBet "Bookings - Over/Under" (event endpoint only)

_LINE = r"(\d+(?:\.\d+)?)"
_TOTALS_MARKET_TEXT = re.compile(rf"^Over/Under {_LINE}$")
_TOTALS_SELECTION = re.compile(rf"^(Over|Under) {_LINE}$")
_TEAM_MARKET_TEXT = re.compile(rf"^(Home|Away) team total {_LINE}$")
_TEAM_SELECTION = re.compile(rf"^(Home|Away) team (Over|Under) {_LINE}$")
_CORNERS_MARKET_TEXT = re.compile(rf"^Corners {_LINE}$")
_CORNERS_SELECTION = re.compile(rf"^Corners (Over|Under) {_LINE}$")
_CARDS_MARKET_TEXT = re.compile(rf"^Cards {_LINE}$")
_CARDS_SELECTION = re.compile(rf"^Cards (Over|Under) {_LINE}$")
# Double chance: our selection text -> the side, which is the start of SportyBet's own outcome name ("Draw or Away").
DOUBLE_CHANCE_SIDES = {"home or draw": "home or draw", "away or draw": "draw or away", "home or away": "home or away"}


class UnsupportedLeg(Exception):
    """A pick this bot cannot map onto a bookmaker market (so it can never be loaded or placed)."""


def quantize_odds(value) -> Decimal:
    return Decimal(str(value)).quantize(ODDS_STEP, rounding=ROUND_HALF_UP)


def total_odds(odds: Iterable[Decimal]) -> Decimal:
    """The multiple's total odds: the product of the leg odds, to 2 decimals (Decimal all the way, AD-15)."""
    product = Decimal(1)
    count = 0
    for price in odds:
        price = Decimal(price)
        if not price.is_finite() or price <= 0:
            raise ValueError("odds must be positive numbers")
        product *= price
        count += 1
    if count == 0:
        raise ValueError("there are no odds to multiply")
    return quantize_odds(product)


LINE_KEYS = ("total", "hcp")  # `total=2.5` (a points or goals line) and `hcp=-6.5` (a handicap, from the home side)


def _parts(specifier: str) -> tuple[str, Decimal, tuple[tuple[str, str], ...]] | None:
    """(line key, line, qualifiers) of a specifier such as `total=2.5` or `total=1.5|periodnr=1` (parts in any order).
    The qualifiers (`periodnr=1`, `inningnr=1`) are part of the market's identity, so a 1st-period total can never be
    taken for a full-game one. None when there is not exactly one readable line."""
    lines, extras = [], []
    for part in (specifier or "").split("|"):
        key, sep, value = part.partition("=")
        key, value = key.strip(), value.strip()
        if not sep or not key:
            return None
        if key in LINE_KEYS:
            try:
                line = Decimal(value)
            except InvalidOperation:
                return None
            if not line.is_finite():
                return None
            lines.append((key, line))
        else:
            extras.append((key, value))
    if len(lines) != 1:
        return None
    return lines[0][0], lines[0][1], tuple(sorted(extras))


def _key_line(specifier: str) -> tuple[str, Decimal] | None:
    found = _parts(specifier)
    return (found[0], found[1]) if found else None


def line_of(specifier: str) -> Decimal | None:
    found = _key_line(specifier)
    return found[1] if found else None


def total_specifier(line: Decimal) -> str:
    """`total=2.5`, and `total=2` for the whole line (the bookmaker's spelling has no trailing zero)."""
    return "total=" + format(Decimal(line).normalize(), "f")


def handicap_specifier(line: Decimal) -> str:
    """`hcp=-6.5`: the HOME side's handicap, spelled as the bookmaker spells it."""
    return "hcp=" + format(Decimal(line).normalize(), "f")


def specifiers_equal(a: str, b: str) -> bool:
    if not (a or "").strip() and not (b or "").strip():
        return True
    pa, pb = _parts(a), _parts(b)
    return pa is not None and pb is not None and pa == pb


@dataclass(frozen=True)
class Leg:
    """One selection to put on the betslip. `odds` are the odds the evaluation used."""
    match_id: MatchId
    market_id: str
    specifier: str
    side: str
    market: str     # the pick's market text, for display
    selection: str  # the pick's selection text, for display
    odds: Decimal

    @property
    def key(self) -> tuple[str, str, str, str]:
        """(match, market, line, side): what the placed-leg registry and the token identify a leg by."""
        return (str(self.match_id), self.market_id, normalized_specifier(self.specifier), self.side)

    @property
    def line(self) -> Decimal | None:
        return line_of(self.specifier)


def normalized_specifier(specifier: str) -> str:
    found = _parts(specifier)
    if found is None:
        return (specifier or "").strip()
    key, line, extras = found
    return "|".join([f"{key}=" + format(line.normalize(), "f"), *(f"{k}={v}" for k, v in extras)])


# ---- basketball wording (the bot's own market text, produced by core/engines/basketball.py) ----
_BB_TOTAL_MARKET_TEXT = re.compile(rf"^Total points {_LINE}$")
_BB_TOTAL_SELECTION = re.compile(rf"^(Over|Under) {_LINE}$")
_BB_TEAM_MARKET_TEXT = re.compile(rf"^(Home|Away) team total points {_LINE}$")
_BB_TEAM_SELECTION = re.compile(rf"^(Home|Away) team (Over|Under) {_LINE}$")
_BB_FH_MARKET_TEXT = re.compile(rf"^1st half total points {_LINE}$")
_BB_FH_SELECTION = re.compile(rf"^1st half (Over|Under) {_LINE}$")
_BB_HANDICAP_MARKET_TEXT = re.compile(r"^Handicap ([+-]?\d+(?:\.\d+)?)$")
_BB_HANDICAP_SELECTION = re.compile(r"^(Home|Away) \(([+-]?\d+(?:\.\d+)?)\)$")
BB_WINNER_MARKET_TEXT = "Winner (incl. OT)"


def _basketball_leg(evaluation: Evaluation) -> Leg:
    market, selection = evaluation.market.strip(), evaluation.selection.strip()
    odds, match = Decimal(evaluation.odds), evaluation.match_id
    if market == BB_WINNER_MARKET_TEXT:
        side = selection.lower()
        if side not in ("home", "away"):
            raise UnsupportedLeg(f"unknown winner selection {selection!r}")
        return Leg(match, BB_WINNER_MARKET, "", side, market, selection, odds)
    found = _BB_TOTAL_MARKET_TEXT.match(market)
    if found:
        chosen = _BB_TOTAL_SELECTION.match(selection)
        if not chosen or Decimal(chosen.group(2)) != Decimal(found.group(1)):
            raise UnsupportedLeg("the selection does not match its market line")
        return Leg(match, BB_TOTAL_MARKET, total_specifier(Decimal(found.group(1))), chosen.group(1).lower(),
                   market, selection, odds)
    found = _BB_TEAM_MARKET_TEXT.match(market)
    if found:
        chosen = _BB_TEAM_SELECTION.match(selection)
        if (not chosen or chosen.group(1) != found.group(1)
                or Decimal(chosen.group(3)) != Decimal(found.group(2))):
            raise UnsupportedLeg("the selection does not match its market line")
        market_id = BB_HOME_TOTAL_MARKET if found.group(1) == "Home" else BB_AWAY_TOTAL_MARKET
        return Leg(match, market_id, total_specifier(Decimal(found.group(2))), chosen.group(2).lower(),
                   market, selection, odds)
    found = _BB_FH_MARKET_TEXT.match(market)
    if found:
        chosen = _BB_FH_SELECTION.match(selection)
        if not chosen or Decimal(chosen.group(2)) != Decimal(found.group(1)):
            raise UnsupportedLeg("the selection does not match its market line")
        return Leg(match, BB_FH_TOTAL_MARKET, total_specifier(Decimal(found.group(1))), chosen.group(1).lower(),
                   market, selection, odds)
    found = _BB_HANDICAP_MARKET_TEXT.match(market)
    if found:
        chosen = _BB_HANDICAP_SELECTION.match(selection)
        hcp = Decimal(found.group(1))  # the HOME handicap, which is what SportyBet's specifier holds
        if not chosen:
            raise UnsupportedLeg("the selection does not match its market")
        shown = Decimal(chosen.group(2))
        if shown != (hcp if chosen.group(1) == "Home" else -hcp):
            raise UnsupportedLeg("the selection does not match its market line")
        return Leg(match, BB_HANDICAP_MARKET, handicap_specifier(hcp), chosen.group(1).lower(), market, selection, odds)
    raise UnsupportedLeg(f"no bookmaker market is known for {market!r}")


# ---- baseball (MLB) and hockey (NHL) wording: the bot's own market text, produced by core/engines/mlb.py and nhl.py ----
_SIGNED = r"([+-]?\d+(?:\.\d+)?)"


def _same_line(a: str, b: str) -> bool:
    return Decimal(a) == Decimal(b)


def _ou_rule(market_text: str, selection_text: str, market_id, prefix: str = "", per: str = ""):
    """A market "<text> L" with selection "<prefix>Over|Under L". `market_id` is a fixed id, or a function of the team
    word when the market text starts with Home/Away. Returns (market_regex, selection_regex, builder)."""
    market = re.compile(rf"^{market_text} {_LINE}$")
    selection = re.compile(rf"^{selection_text}(Over|Under) {_LINE}$")

    def build(mm, ms):
        if not _same_line(mm.group(mm.lastindex), ms.group(ms.lastindex)):
            raise UnsupportedLeg("the selection does not match its market line")
        return market_id, total_specifier(Decimal(mm.group(mm.lastindex))) + per, ms.group(ms.lastindex - 1).lower()
    return market, selection, build


def _team_rule(market_text: str, selection_text: str, ids: dict[str, str]):
    """"Home|Away <market text> L" with selection "Home|Away team Over|Under L"."""
    market = re.compile(rf"^(Home|Away) {market_text} {_LINE}$")
    selection = re.compile(rf"^(Home|Away) {selection_text} (Over|Under) {_LINE}$")

    def build(mm, ms):
        if mm.group(1) != ms.group(1) or not _same_line(mm.group(2), ms.group(3)):
            raise UnsupportedLeg("the selection does not match its market line")
        return ids[mm.group(1)], total_specifier(Decimal(mm.group(2))), ms.group(2).lower()
    return market, selection, build


def _handicap_rule(market_text: str, selection_prefix: str, market_id: str):
    """"<market text> SL" (the HOME handicap) with selection "<prefix>Home|Away (SL)"."""
    market = re.compile(rf"^{market_text} {_SIGNED}$")
    selection = re.compile(rf"^{selection_prefix}(Home|Away) \({_SIGNED}\)$")

    def build(mm, ms):
        hcp, shown = Decimal(mm.group(1)), Decimal(ms.group(2))
        if shown != (hcp if ms.group(1) == "Home" else -hcp):
            raise UnsupportedLeg("the selection does not match its market line")
        return market_id, handicap_specifier(hcp), ms.group(1).lower()
    return market, selection, build


def _fixed_rule(market_text: str, selection_regex: str, market_id: str):
    market = re.compile(rf"^{re.escape(market_text)}$")
    selection = re.compile(selection_regex)

    def build(mm, ms):
        return market_id, "", ms.group(1).lower()
    return market, selection, build


_BASEBALL_RULES = (
    _fixed_rule(_mlb.WINNER_MARKET_TEXT, r"^(Home|Away)$", _mlb.WINNER_MARKET_ID),
    _ou_rule("Total runs", "", _mlb.TOTAL_MARKET_ID),
    _team_rule("team total runs", "team", {"Home": _mlb.HOME_TOTAL_MARKET_ID, "Away": _mlb.AWAY_TOTAL_MARKET_ID}),
    _handicap_rule("Run line", "", _mlb.HANDICAP_MARKET_ID),
    _ou_rule("F5 total runs", "F5 ", _mlb.F5_TOTAL_MARKET_ID),
    _handicap_rule("F5 handicap", "F5 ", _mlb.F5_HANDICAP_MARKET_ID),
    _fixed_rule(_mlb.F5_RESULT_MARKET_TEXT, r"^F5 (Home|Draw|Away)$", _mlb.F5_RESULT_MARKET_ID),
)
_HOCKEY_RULES = (
    _fixed_rule(_nhl.WINNER_MARKET_TEXT, r"^(Home|Away)$", _nhl.WINNER_MARKET_ID),
    _ou_rule(r"Total goals \(incl\. OT\)", "", _nhl.TOTAL_OT_MARKET_ID),
    _ou_rule(r"Total goals \(regulation\)", "", _nhl.TOTAL_REG_MARKET_ID),
    _team_rule("team total goals", "team", {"Home": _nhl.HOME_TOTAL_MARKET_ID, "Away": _nhl.AWAY_TOTAL_MARKET_ID}),
    _handicap_rule(r"Puck line \(incl\. OT\)", "", _nhl.HANDICAP_OT_MARKET_ID),
    _handicap_rule(r"Handicap \(regulation\)", "", _nhl.HANDICAP_REG_MARKET_ID),
    _ou_rule("1st period total goals", "1st period ", _nhl.PERIOD_TOTAL_MARKET_ID, per="|periodnr=1"),
)
# American football (Epic 7): the same SportyBet ids as basketball, the engine's own market text (core/engines/american_football.py).
_FOOTBALL_RULES = (
    _fixed_rule(_football.WINNER_MARKET_TEXT, r"^(Home|Away)$", _football.WINNER_MARKET_ID),
    _ou_rule("Total points", "", _football.TOTAL_MARKET_ID),
    _team_rule("team total points", "team", {"Home": _football.HOME_TOTAL_MARKET_ID, "Away": _football.AWAY_TOTAL_MARKET_ID}),
    _handicap_rule("Handicap", "", _football.HANDICAP_MARKET_ID),
    _ou_rule("1st half total points", "1st half ", _football.FH_TOTAL_MARKET_ID),
)
_SPORT_RULES = {"baseball": _BASEBALL_RULES, "hockey": _HOCKEY_RULES, "american_football": _FOOTBALL_RULES}


def _rule_based_leg(evaluation: Evaluation, rules) -> Leg:
    market, selection = evaluation.market.strip(), evaluation.selection.strip()
    for market_re, selection_re, build in rules:
        found = market_re.match(market)
        if not found:
            continue
        chosen = selection_re.match(selection)
        if not chosen:
            raise UnsupportedLeg("the selection does not match its market")
        market_id, specifier, side = build(found, chosen)
        return Leg(evaluation.match_id, market_id, specifier, side, market, selection, Decimal(evaluation.odds))
    raise UnsupportedLeg(f"no bookmaker market is known for {market!r}")


def leg_from_evaluation(evaluation: Evaluation) -> Leg:
    """The Leg a passed pick stands for. Raises UnsupportedLeg for any market this bot cannot put on a betslip."""
    if evaluation.match_id.sport == "basketball":
        return _basketball_leg(evaluation)
    if evaluation.match_id.sport in _SPORT_RULES:
        return _rule_based_leg(evaluation, _SPORT_RULES[evaluation.match_id.sport])
    market, selection = evaluation.market.strip(), evaluation.selection.strip()
    if market == "1X2":
        side = selection.lower()
        if side not in ("home", "draw", "away"):
            raise UnsupportedLeg(f"unknown 1X2 selection {selection!r}")
        return Leg(evaluation.match_id, RESULT_MARKET, "", side, market, selection, Decimal(evaluation.odds))
    if market == "Double Chance":
        side = DOUBLE_CHANCE_SIDES.get(selection.lower())
        if side is None:
            raise UnsupportedLeg(f"unknown double chance selection {selection!r}")
        return Leg(evaluation.match_id, DOUBLE_CHANCE_MARKET, "", side, market, selection, Decimal(evaluation.odds))
    for market_re, selection_re, market_id in ((_CORNERS_MARKET_TEXT, _CORNERS_SELECTION, CORNERS_MARKET),
                                               (_CARDS_MARKET_TEXT, _CARDS_SELECTION, CARDS_MARKET)):
        found = market_re.match(market)
        if found:
            chosen = selection_re.match(selection)
            if not chosen or Decimal(chosen.group(2)) != Decimal(found.group(1)):
                raise UnsupportedLeg("the selection does not match its market line")
            return Leg(evaluation.match_id, market_id, total_specifier(Decimal(found.group(1))),
                       chosen.group(1).lower(), market, selection, Decimal(evaluation.odds))
    found = _TOTALS_MARKET_TEXT.match(market)
    if found:
        chosen = _TOTALS_SELECTION.match(selection)
        if not chosen or Decimal(chosen.group(2)) != Decimal(found.group(1)):
            raise UnsupportedLeg("the selection does not match its market line")
        return Leg(evaluation.match_id, TOTALS_MARKET, total_specifier(Decimal(found.group(1))),
                   chosen.group(1).lower(), market, selection, Decimal(evaluation.odds))
    found = _TEAM_MARKET_TEXT.match(market)
    if found:
        chosen = _TEAM_SELECTION.match(selection)
        if (not chosen or chosen.group(1) != found.group(1)
                or Decimal(chosen.group(3)) != Decimal(found.group(2))):
            raise UnsupportedLeg("the selection does not match its market line")
        market_id = HOME_TOTAL_MARKET if found.group(1) == "Home" else AWAY_TOTAL_MARKET
        return Leg(evaluation.match_id, market_id, total_specifier(Decimal(found.group(2))),
                   chosen.group(2).lower(), market, selection, Decimal(evaluation.odds))
    raise UnsupportedLeg(f"no bookmaker market is known for {market!r}")


def find_outcome(offers: Iterable[MarketOffer], leg: Leg) -> tuple[MarketOffer, Outcome] | None:
    """The one listed outcome that is exactly this leg (same market, same line, same side), else None."""
    for offer in offers:
        if offer.market_id != leg.market_id or not specifiers_equal(offer.specifier, leg.specifier):
            continue
        found = [o for o in offer.outcomes if o.name.strip().lower().startswith(leg.side)]
        if len(found) == 1:
            return offer, found[0]
    return None


# ---------------------------------------------------------------- availability (Story 5.2)

class Availability(Enum):
    OK = "ok"
    GAME_UNAVAILABLE = "game_unavailable"
    OPTION_MISSING = "option_missing"


@dataclass(frozen=True)
class LegAvailability:
    leg: Leg
    status: Availability
    fixture: Fixture | None = None      # the bookmaker's live listing of the game, when it has one
    offer: MarketOffer | None = None    # for OK: the market the leg sits in
    outcome: Outcome | None = None      # for OK: the outcome to click
    detail: str = ""


def assess_availability(legs: Sequence[Leg], fixtures: Iterable[Fixture], now: datetime) -> tuple[LegAvailability, ...]:
    """For each leg: OK, or the game is not (or no longer) available, or the game is there but the option is not.
    A game that has kicked off, is flagged started, or is listed more than once is unavailable."""
    by_key: dict[str, list[Fixture]] = {}
    for fixture in fixtures:
        by_key.setdefault(str(fixture.match_id), []).append(fixture)
    results = []
    for leg in legs:
        listed = by_key.get(str(leg.match_id), [])
        if len(listed) != 1:
            why = "SportyBet does not list this game" if not listed else "SportyBet lists this game more than once"
            results.append(LegAvailability(leg, Availability.GAME_UNAVAILABLE, None, None, None, why))
            continue
        fixture = listed[0]
        if fixture.started or fixture.kickoff_utc <= now or leg.match_id.kickoff_utc <= now:
            results.append(LegAvailability(leg, Availability.GAME_UNAVAILABLE, fixture, None, None,
                                           "the game has started or is about to"))
            continue
        if not fixture.bookmaker_ref:
            results.append(LegAvailability(leg, Availability.GAME_UNAVAILABLE, fixture, None, None,
                                           "SportyBet gave no way to open this game"))
            continue
        found = find_outcome(fixture.markets, leg)
        if found is None:
            results.append(LegAvailability(leg, Availability.OPTION_MISSING, fixture, None, None,
                                           "the selected option is not offered"))
            continue
        results.append(LegAvailability(leg, Availability.OK, fixture, found[0], found[1], ""))
    return tuple(results)


# ---------------------------------------------------------------- the betslip, as read from the bookmaker

@dataclass(frozen=True)
class ExpectedLine:
    """What one betslip row must say for a leg, in the bookmaker's own wording."""
    leg_key: tuple[str, str, str, str]
    match_text: str     # "Home v Away"
    market_name: str
    outcome_name: str
    selection: str = ""            # the pick's own text, for messages
    evaluation_odds: Decimal | None = None


@dataclass(frozen=True)
class BetslipLine:
    """One row read from the real betslip."""
    match_text: str
    market_name: str
    outcome_name: str
    odds: Decimal


@dataclass(frozen=True)
class BetslipReading:
    lines: tuple[BetslipLine, ...]
    total_odds: Decimal | None
    stake: Decimal | None = None   # what the stake box holds, when it was read


@dataclass(frozen=True)
class LoadedLeg:
    """A leg as the betslip really holds it, with its final odds."""
    leg_key: tuple[str, str, str, str]
    match_text: str
    market_name: str
    outcome_name: str
    odds: Decimal
    selection: str = ""
    evaluation_odds: Decimal | None = None


@dataclass(frozen=True)
class LoadedSlip:
    legs: tuple[LoadedLeg, ...]
    total_odds: Decimal


@dataclass(frozen=True)
class FailedLeg:
    """A pick the bookmaker could not add to the betslip, with the adapter's plain reason (owner-readable, no secrets)."""
    leg_key: tuple[str, str, str, str]
    reason: str


@dataclass(frozen=True)
class LoadAttempt:
    """What a load that skips un-addable picks returns: the verified betslip of the picks that WERE added (None when none
    was) and the picks that were not. `failed` empty means everything loaded."""
    loaded: LoadedSlip | None
    failed: tuple[FailedLeg, ...] = ()


def _row_key(match_text: str, market_name: str, outcome_name: str) -> tuple[str, str, str]:
    return normalize_team(match_text), normalize_team(market_name), normalize_team(outcome_name)


@dataclass(frozen=True)
class Verification:
    ok: bool
    problems: tuple[str, ...] = ()
    matched: tuple[LoadedLeg, ...] = ()   # in the order of the expected lines, with the odds actually read
    odds_changed: bool = False            # every row matches, but at least one price differs from the expected one


def verify_reading(expected: Sequence[ExpectedLine], reading: BetslipReading,
                   expected_odds: Sequence[Decimal] | None = None,
                   expected_total: Decimal | None = None) -> Verification:
    """Does the real betslip hold EXACTLY the expected rows (no more, no fewer, none different)?

    When `expected_odds` is given (one price per expected row, same order) each price must match too. A row that
    matches in every word but at a different price sets `odds_changed` and is not `ok`: the caller must
    show a new summary, never place. `expected_total` is compared the same way (to 2 decimals)."""
    problems: list[str] = []
    rows = list(reading.lines)
    if not expected:
        problems.append("there is nothing to check the betslip against")
    if len(rows) != len(expected):
        problems.append(f"the betslip holds {len(rows)} selection(s), expected {len(expected)}")
    matched: list[LoadedLeg] = []
    price_differs = False
    remaining = list(rows)
    for index, want in enumerate(expected):
        key = _row_key(want.match_text, want.market_name, want.outcome_name)
        hit = next((r for r in remaining if _row_key(r.match_text, r.market_name, r.outcome_name) == key), None)
        if hit is None:
            problems.append(f"{want.selection or want.outcome_name} ({want.match_text}) is not on the betslip as expected")
            continue
        remaining.remove(hit)
        if quantize_odds(hit.odds) <= 0:
            problems.append(f"the odds for {want.match_text} could not be read")
        matched.append(LoadedLeg(want.leg_key, hit.match_text, hit.market_name, hit.outcome_name,
                                 quantize_odds(hit.odds), want.selection, want.evaluation_odds))
        if expected_odds is not None and quantize_odds(hit.odds) != quantize_odds(expected_odds[index]):
            price_differs = True
    for extra in remaining:
        problems.append(f"the betslip holds a selection that was not confirmed: {extra.outcome_name} ({extra.match_text})")
    if reading.total_odds is None:
        problems.append("the total odds could not be read from the betslip")
    elif expected_total is not None and quantize_odds(reading.total_odds) != quantize_odds(expected_total):
        price_differs = True
    if problems:
        return Verification(False, tuple(problems), tuple(matched))
    if price_differs:
        return Verification(False, ("the odds on the betslip changed",), tuple(matched), odds_changed=True)
    return Verification(True, (), tuple(matched))


def total_is_consistent(legs: Sequence[LoadedLeg], shown_total: Decimal) -> bool:
    """The bookmaker's total against our own product of its leg odds (within a small relative tolerance)."""
    if not legs:
        return False
    mine = total_odds(leg.odds for leg in legs)
    shown = quantize_odds(shown_total)
    return abs(mine - shown) <= max(ODDS_STEP, mine * TOTAL_ODDS_TOLERANCE)


def make_loaded_slip(verified: Verification, reading: BetslipReading) -> LoadedSlip:
    if not verified.ok or reading.total_odds is None:
        raise ValueError("only a verified betslip becomes a loaded slip")
    if not total_is_consistent(verified.matched, reading.total_odds):
        raise ValueError("the total odds do not match the leg odds")
    return LoadedSlip(verified.matched, quantize_odds(reading.total_odds))


MAX_SHOWN_MATCHES = 3   # the history list shows at most this many games of a bet, then "and N other matches"


@dataclass(frozen=True)
class HistoryBet:
    """One bet as SportyBet's bet-history LIST page shows it (layout VERIFIED-LIVE 2026-10-01).

    The page has NO bet id and NO booking code, so `bet_id` is a SYNTHETIC key built from the visible fields (time, type,
    stake, the shown games and the 'N other' count; see the adapter). It is stable while the bet's status changes. Two
    entries with the same key are both flagged `ambiguous` and can never prove anything. The page shows no odds and no
    market or pick either: `total_odds` is always None and `lines` always empty on a real read."""
    bet_id: str
    booking_code: str = ""
    stake: Decimal | None = None
    total_odds: Decimal | None = None   # never shown on the list page
    lines: tuple[BetslipLine, ...] = ()  # never shown on the list page
    text: str = ""   # the entry's visible fields as one line, for messages
    leg_count: int | None = None        # TOTAL games of the bet: the shown ones plus 'N other matches'
    placed_at: datetime | None = None   # when the entry says it was placed (UTC-aware); minute precision
    shown_matches: tuple[str, ...] = ()  # the "Home v Away" lines the entry shows (at most MAX_SHOWN_MATCHES)
    status: str = ""                    # the bet's own status word: 'Running', 'Won', 'Lost', ...
    bet_type: str = ""                  # 'Multiple' or 'Singles'
    ambiguous: bool = False             # another entry has the very same key: nothing can be proven from either
    total_return: Decimal | None = None  # 'Total Return' when shown (None while running)

    @property
    def legs_in_entry(self) -> int | None:
        return self.leg_count

    @property
    def more_than_shown(self) -> int:
        """How many games of the bet the list does not name (0 when it shows them all or the count is unknown)."""
        return max(0, (self.leg_count or 0) - len(self.shown_matches))


HISTORY_LIMITS_NOTE = ("The history list does not show odds or selections, so only the games, stake and number of legs "
                       "were verified (the odds and picks were checked on the betslip just before the click).")


def history_limits_note(bet: HistoryBet) -> str:
    """The plain statement of what a history entry could and could not verify, for every message that calls a bet found."""
    note = HISTORY_LIMITS_NOTE
    hidden = bet.more_than_shown
    if hidden:
        note += (f" The list names only the first {len(bet.shown_matches)} of {bet.leg_count} games, so the other "
                 f"{hidden} could not be compared.")
    return note


# ---------------------------------------------------------------- recovery helpers (Story 5.5)

def leg_identity(leg_key: Sequence[str]) -> tuple[str, str, str]:
    """(match, market, line): what the placed-leg registry blocks on. The side is NOT part of it, so Under 2.5 is
    blocked after Over 2.5 was placed on the same game (fail closed)."""
    match_id, market_id, specifier = (str(part) for part in tuple(leg_key)[:3])
    return match_id, market_id, specifier


PLACED_AT_SLACK = timedelta(minutes=2)   # history times are to the minute, and clocks differ a little
BRAND_NEW_STATUSES = (STATUS_RUNNING,)   # what a bet placed a moment ago can show
SETTLED_OR_RUNNING = (STATUS_RUNNING, "won", "lost")   # reconciliation later: the bet may have finished since


def _match_key(text: str) -> str:
    return normalize_team(text)


def entry_matches_legs(bet: HistoryBet, legs: Sequence[LoadedLeg], stake: Decimal) -> bool:
    """The entry is the bet described by `legs` and `stake` as far as the list page can tell: the stake is equal, the
    type fits (one game = 'Singles', several = 'Multiple'), the TOTAL number of games equals the number of legs, and every
    SHOWN game is one of our legs (team names through normalize_team, order of the games ignored, each leg used once).
    When the entry shows fewer games than there are legs (more than 3), only the shown ones can be compared. Fail closed:
    doubt is a no-match. It says nothing about odds or picks (the page shows none), status or time."""
    if not legs or bet.stake is None or Decimal(bet.stake) != Decimal(stake):
        return False
    wanted_type = TYPE_SINGLES if len(legs) == 1 else TYPE_MULTIPLE
    if normalise_word(bet.bet_type) != wanted_type:
        return False
    if bet.leg_count != len(legs) or not bet.shown_matches or len(bet.shown_matches) > bet.leg_count:
        return False
    if len(bet.shown_matches) < bet.leg_count and len(bet.shown_matches) != MAX_SHOWN_MATCHES:
        return False   # the page names three games and then 'N other': anything else is not the layout we know
    ours = [_match_key(leg.match_text) for leg in legs]
    for shown in bet.shown_matches:
        key = _match_key(shown)
        if not key or key not in ours:
            return False
        ours.remove(key)
    return True


def history_bet_matches(bet: HistoryBet, legs: Sequence[LoadedLeg], stake: Decimal,
                        created_at: datetime | None = None, statuses: Sequence[str] = BRAND_NEW_STATUSES,
                        ignore_ambiguity: bool = False) -> bool:
    """A history entry proves the bet described by `legs` and `stake` when ALL hold: `entry_matches_legs`; the status is
    one of `statuses` (a brand-new bet can only show 'Running'; anything else is unknown, never proof); the entry's time
    is readable and not before `created_at` (2 minutes of slack, minute precision) when `created_at` is given; and the
    entry is not `ambiguous` (an identical key twice) unless `ignore_ambiguity` (used only to WARN about a possible
    double placement). It does NOT check odds, markets or picks: the list page shows none. Deterministic; fail closed."""
    if bet.ambiguous and not ignore_ambiguity:
        return False
    if not entry_matches_legs(bet, legs, stake):
        return False
    if normalise_word(bet.status) not in statuses:
        return False
    if created_at is not None and (bet.placed_at is None or bet.placed_at < created_at - PLACED_AT_SLACK):
        return False
    return True
