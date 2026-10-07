"""Epic 5 core logic: legs, availability, betslip verification, closest option, stake and Kelly, tokens."""
import ast
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from fakes_bookmaker import (COMPETITION, KICKOFF, NOW, make_fixture, match_id, pick_evaluation, result_offer,
                             totals_offer, xg_inputs)
from predictions.core import staking
from predictions.core.betting import (Availability, BetslipLine, BetslipReading, ExpectedLine, Leg, UnsupportedLeg,
                                      assess_availability, find_outcome, leg_from_evaluation, make_loaded_slip,
                                      quantize_odds, specifiers_equal, total_is_consistent, total_odds,
                                      total_specifier, verify_reading)
from predictions.core.model import PROBABILITY_FLOOR, Estimate, Evaluation
from predictions.core.options import as_substitute, closest_option
from predictions.core.placement import (MINT_AUTHORITY, PlacementToken, Summary, TokenNotMinted, mint_token,
                                        summary_digest)
from predictions.core.slip import Slip, SlipError, make_pick, make_pick_id
from predictions.core.betting import LoadedLeg
from predictions.core.engines import ENGINES

REPO = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- legs

def test_a_goals_pick_maps_to_the_totals_market_with_a_normalized_line():
    leg = leg_from_evaluation(pick_evaluation(market="Over/Under 2.0", selection="Over 2.0"))
    assert (leg.market_id, leg.specifier, leg.side) == ("18", "total=2", "over")
    leg = leg_from_evaluation(pick_evaluation(market="Over/Under 2.5", selection="Under 2.5"))
    assert (leg.specifier, leg.side) == ("total=2.5", "under")


def test_result_and_team_total_picks_map_too():
    assert leg_from_evaluation(pick_evaluation(market="1X2", selection="Home")).market_id == "1"
    leg = leg_from_evaluation(pick_evaluation(market="Away team total 1.5", selection="Away team Over 1.5"))
    assert (leg.market_id, leg.specifier, leg.side) == ("20", "total=1.5", "over")


@pytest.mark.parametrize("market,selection", [("Corners 9.5", "Over 9.5"), ("Over/Under 2.5", "Over 3.5"),
                                              ("1X2", "Banana"), ("Home team total 1.5", "Away team Over 1.5")])
def test_unknown_or_inconsistent_picks_are_unsupported_never_guessed(market, selection):
    with pytest.raises(UnsupportedLeg):
        leg_from_evaluation(pick_evaluation(market=market, selection=selection))


def test_specifiers_compare_by_number_not_by_spelling():
    assert specifiers_equal("total=2", "total=2.0") and specifiers_equal("", "") and not specifiers_equal("total=2", "total=3")
    assert total_specifier(Decimal("2.0")) == "total=2" and total_specifier(Decimal("2.50")) == "total=2.5"


# ---------------------------------------------------------------- availability (Story 5.2 core)

def leg_of(**kw):
    return leg_from_evaluation(pick_evaluation(**kw))


def test_availability_ok_game_missing_option_missing_and_started():
    ok_leg = leg_of()
    other = leg_of(home="Gamma", away="Delta", market="Over/Under 1.5", selection="Over 1.5", hour=16)
    started = leg_of(home="Echo", away="Foxtrot", hour=15)
    missing_game = leg_of(home="Golf", away="Hotel", hour=17)
    listing = [make_fixture("Alpha", "Beta", [totals_offer("2.5", "1.85", "2.00")]),
               make_fixture("Gamma", "Delta", [totals_offer("2.5", "1.50", "2.40")], 16),  # 1.5 line not offered
               make_fixture("Echo", "Foxtrot", [totals_offer("2.5", "1.85", "2.00")], 15, started=True)]
    got = assess_availability([ok_leg, other, started, missing_game], listing, NOW)
    assert [g.status for g in got] == [Availability.OK, Availability.OPTION_MISSING, Availability.GAME_UNAVAILABLE,
                                       Availability.GAME_UNAVAILABLE]
    assert got[0].outcome.odds == Decimal("1.85") and got[1].fixture is listing[1]


def test_a_game_that_kicked_off_after_the_request_is_unavailable_even_if_still_listed():
    leg = leg_of()
    listing = [make_fixture("Alpha", "Beta", [totals_offer("2.5", "1.85", "2.00")])]
    late = KICKOFF + timedelta(minutes=1)
    (result,) = assess_availability([leg], listing, late)
    assert result.status is Availability.GAME_UNAVAILABLE


def test_a_game_listed_twice_or_without_a_way_to_open_it_is_unavailable():
    leg = leg_of()
    twice = [make_fixture(offers=[totals_offer("2.5", "1.85", "2.00")])] * 2
    assert assess_availability([leg], twice, NOW)[0].status is Availability.GAME_UNAVAILABLE
    no_ref = [make_fixture(offers=[totals_offer("2.5", "1.85", "2.00")], ref="")]
    assert assess_availability([leg], no_ref, NOW)[0].status is Availability.GAME_UNAVAILABLE


def test_find_outcome_needs_exactly_one_matching_side():
    leg = leg_of(market="1X2", selection="Draw")
    assert find_outcome([result_offer()], leg)[1].name == "Draw"
    assert find_outcome([totals_offer("2.5", "1.8", "2.0")], leg) is None


# ---------------------------------------------------------------- odds and totals

def test_total_odds_is_a_decimal_product_rounded_half_up():
    assert total_odds([Decimal("2.05"), Decimal("2.35")]) == Decimal("4.82")   # 4.8175
    assert total_odds([Decimal("1.68"), Decimal("1.64")]) == Decimal("2.76")   # 2.7552
    assert isinstance(total_odds([Decimal("1.5")]), Decimal)
    with pytest.raises(ValueError):
        total_odds([])
    with pytest.raises(ValueError):
        total_odds([Decimal("0")])


def test_total_consistency_allows_small_rounding_but_not_a_missing_leg():
    legs = (LoadedLeg(("m", "18", "total=2.5", "over"), "A v B", "Over/Under", "Over 2.5", Decimal("2.00")),
            LoadedLeg(("n", "18", "total=2.5", "over"), "C v D", "Over/Under", "Over 2.5", Decimal("1.50")))
    assert total_is_consistent(legs, Decimal("3.00")) and total_is_consistent(legs, Decimal("3.02"))
    assert not total_is_consistent(legs, Decimal("2.00")) and not total_is_consistent((), Decimal("2"))


# ---------------------------------------------------------------- verifying a betslip against expectations

def expected(*rows):
    return [ExpectedLine((f"m{i}", "18", "total=2.5", "over"), match, market, outcome, outcome, Decimal("1.9"))
            for i, (match, market, outcome) in enumerate(rows)]


def reading(*rows, total="3.00", stake=None):
    return BetslipReading(tuple(BetslipLine(m, k, o, Decimal(p)) for m, k, o, p in rows),
                          None if total is None else Decimal(total), stake)


ROWS = [("Alpha v Beta", "Over/Under", "Over 2.5"), ("Gamma v Delta", "Over/Under", "Over 1.5")]


def test_exact_betslip_verifies_in_any_order():
    seen = reading(("Gamma v Delta", "Over/Under", "Over 1.5", "1.5"), ("Alpha v Beta", "Over/Under", "Over 2.5", "2.0"))
    verdict = verify_reading(expected(*ROWS), seen)
    assert verdict.ok and [m.odds for m in verdict.matched] == [Decimal("2.00"), Decimal("1.50")]
    assert make_loaded_slip(verdict, seen).total_odds == Decimal("3.00")


@pytest.mark.parametrize("rows,why", [
    ([("Alpha v Beta", "Over/Under", "Over 2.5", "2.0")], "holds 1 selection"),                                   # one missing
    ([("Alpha v Beta", "Over/Under", "Over 2.5", "2.0"), ("Gamma v Delta", "Over/Under", "Over 1.5", "1.5"),
      ("Echo v Fox", "1X2", "Home", "2.0")], "not confirmed"),                                                    # one extra
    ([("Alpha v Beta", "Over/Under", "Over 2.5", "2.0"), ("Gamma v Delta", "Over/Under", "Under 1.5", "1.5")], "not on the betslip"),  # wrong side
    ([("Alpha v Beta", "Over/Under", "Over 2.5", "2.0"), ("Gamma v Delta", "1X2", "Over 1.5", "1.5")], "not on the betslip"),         # wrong market
    ([("Alpha v Beta", "Over/Under", "Over 2.5", "2.0"), ("Gamma v Omega", "Over/Under", "Over 1.5", "1.5")], "not on the betslip"),   # wrong game
])
def test_any_difference_in_the_betslip_fails_verification(rows, why):
    verdict = verify_reading(expected(*ROWS), reading(*rows))
    assert not verdict.ok and not verdict.odds_changed and any(why in p for p in verdict.problems)


def test_an_unreadable_total_fails_verification():
    seen = reading(("Alpha v Beta", "Over/Under", "Over 2.5", "2.0"), ("Gamma v Delta", "Over/Under", "Over 1.5", "1.5"),
                   total=None)
    assert not verify_reading(expected(*ROWS), seen).ok


def test_same_rows_at_other_prices_is_odds_changed_not_ok():
    seen = reading(("Alpha v Beta", "Over/Under", "Over 2.5", "2.10"), ("Gamma v Delta", "Over/Under", "Over 1.5", "1.5"),
                   total="3.15")
    verdict = verify_reading(expected(*ROWS), seen, [Decimal("2.00"), Decimal("1.50")], Decimal("3.00"))
    assert not verdict.ok and verdict.odds_changed and len(verdict.matched) == 2


def test_only_a_verified_betslip_becomes_a_loaded_slip():
    bad = verify_reading(expected(*ROWS), reading(("Alpha v Beta", "Over/Under", "Over 2.5", "2.0")))
    with pytest.raises(ValueError):
        make_loaded_slip(bad, reading(("Alpha v Beta", "Over/Under", "Over 2.5", "2.0")))


# ---------------------------------------------------------------- closest option (Story 5.2 core)

def alternatives_for(pick, offers, fixture_kwargs=None):
    """What the soccer engine says about every option in the listing: the same call the flow makes."""
    fx = make_fixture(offers=offers)
    engine = ENGINES["soccer"]
    out = []
    for offer in offers:
        out.extend(engine.evaluate_exact(fx, pick.estimate.inputs, offer))
    return out


def test_closest_option_is_the_nearest_line_that_clears_the_floor():
    pick = pick_evaluation()  # Over 2.5 with xG 2.5/1.5: expected goals 4, so the lower lines are strong
    candidates = alternatives_for(pick, [totals_offer("1.5", "1.30", "3.20"), totals_offer("2", "1.60", "2.30"),
                                         totals_offer("3.5", "2.30", "1.60")])
    chosen = closest_option(leg_from_evaluation(pick), candidates)
    # the whole line 2.0 is nearer (0.5 away) but is never offered (AD-21); Over 1.5 is the nearest half line (1.0 away)
    assert chosen is not None and chosen.selection == "Over 1.5"
    assert chosen.estimate.probability >= PROBABILITY_FLOOR and chosen.passed


def test_closest_option_never_offers_below_the_floor_even_if_it_is_nearest():
    pick = pick_evaluation()
    candidates = alternatives_for(pick, [totals_offer("3.5", "2.30", "1.60"), totals_offer("4.5", "4.50", "1.20")])
    assert all(c.estimate.probability < PROBABILITY_FLOOR for c in candidates if c.selection.startswith("Over"))
    assert closest_option(leg_from_evaluation(pick), candidates) is None


def test_closest_option_keeps_the_side_the_market_and_the_match():
    pick = pick_evaluation()
    leg = leg_from_evaluation(pick)
    strong_under = alternatives_for(pick_evaluation(market="Over/Under 1.5", selection="Under 1.5", prob=0.7),
                                    [totals_offer("6.5", "9.0", "1.02")])
    other_match = pick_evaluation(home="Gamma", away="Delta", market="Over/Under 1.5", selection="Over 1.5", prob=0.9)
    assert closest_option(leg, strong_under + [other_match]) is None
    one_x_two = leg_of(market="1X2", selection="Draw")
    assert closest_option(one_x_two, [pick_evaluation(market="1X2", selection="Home")]) is None  # no line, no "nearest"


def test_closest_option_ignores_failed_or_insufficient_candidates():
    pick = pick_evaluation()
    weak = replace(pick_evaluation(market="Over/Under 2.0", selection="Over 2.0", prob=0.9), passed=False)
    insufficient = replace(pick_evaluation(market="Over/Under 1.5", selection="Over 1.5", prob=0.9), insufficient=True)
    below = pick_evaluation(market="Over/Under 3.0", selection="Over 3.0", prob=0.64)
    assert closest_option(leg_from_evaluation(pick), [weak, insufficient, below]) is None


def test_ties_go_to_the_higher_estimate_then_the_lower_line():
    leg = leg_of()
    low = pick_evaluation(market="Over/Under 1.5", selection="Over 1.5", prob=0.80)
    high = pick_evaluation(market="Over/Under 3.5", selection="Over 3.5", prob=0.70)
    assert closest_option(leg, [high, low]).selection == "Over 1.5"
    even = pick_evaluation(market="Over/Under 3.5", selection="Over 3.5", prob=0.80)
    assert closest_option(leg, [even, low]).selection == "Over 1.5"


def test_closest_option_never_offers_a_whole_or_quarter_line_nor_replaces_one():
    leg = leg_of()
    nearer = [pick_evaluation(market=f"Over/Under {x}", selection=f"Over {x}", prob=0.90) for x in ("2.0", "3.0", "2.25", "2.75")]
    assert closest_option(leg, nearer) is None
    half = pick_evaluation(market="Over/Under 1.5", selection="Over 1.5", prob=0.80)
    assert closest_option(leg, [*nearer, half]).selection == "Over 1.5"
    whole_leg = leg_of(market="Over/Under 2.0", selection="Over 2.0")
    assert closest_option(whole_leg, [half]) is None


def test_a_substitute_is_marked_and_keeps_one_pick_per_match():
    original = pick_evaluation()
    alt = pick_evaluation(market="Over/Under 3.5", selection="Over 3.5", odds="1.60", prob=0.80)
    swapped = as_substitute(original, alt)
    assert swapped.original.startswith("Over 2.5") and "Closest option" in swapped.rule
    slip = Slip.from_evaluations([original, pick_evaluation("Gamma", "Delta", "Over/Under 1.5", "Over 1.5", "1.3", 0.88, 16)])
    new = slip.substitute(make_pick_id(original), swapped)
    assert new.one_per_match and len(new) == 2 and {p.evaluation.selection for p in new.picks} == {"Over 3.5", "Over 1.5"}
    other_match = pick_evaluation("Zulu", "Yankee", "Over/Under 1.5", "Over 1.5", "1.6", 0.8)
    with pytest.raises(SlipError):
        slip.substitute(make_pick_id(original), other_match)
    whole = pick_evaluation(market="Over/Under 2.0", selection="Over 2.0", odds="1.60", prob=0.80)
    with pytest.raises(ValueError, match="Whole-line plays are not allowed"):
        slip.substitute(make_pick_id(original), whole)
    assert len(Slip.from_evaluations([original, whole])) == 1   # a whole-line evaluation is left out of a Slip, never fatal


# ---------------------------------------------------------------- stake and Kelly (Story 5.3 core)

@pytest.mark.parametrize("text,value", [("500", "500"), (" 1250.50 ", "1250.50"), ("0.01", "0.01"), ("1,500", "1500"),
                                        ("1,234,567.8", "1234567.8"), ("007", "7"), ("12.5", "12.5")])
def test_valid_stakes_parse_once_into_decimal(text, value):
    assert staking.parse_stake(text, None) == Decimal(value) and isinstance(staking.parse_stake(text, None), Decimal)


@pytest.mark.parametrize("text", ["", " ", "0", "0.00", "-5", "+5", "1e3", "5.123", "5.", ".5", "1,5", "12,34", "NaN",
                                  "Infinity", "abc", "5 000", "₦500", "500 NGN", "٥٠٠", "1..2", "0x10", None, 500])
def test_invalid_stakes_are_rejected(text):
    with pytest.raises(staking.StakeError):
        staking.parse_stake(text)


def test_kelly_suggestion_is_always_between_1_5_and_3_5_units_and_deterministic():
    low = staking.suggest_stake([0.66, 0.66], Decimal("2.30"))    # thin edge: clamps up to the minimum
    assert low.units == staking.MIN_UNITS and low.bound == "min" and low.edge
    high = staking.suggest_stake([0.95], Decimal("3.00"))         # big edge: clamps down to the maximum
    assert high.units == staking.MAX_UNITS and high.bound == "max"
    mid = staking.suggest_stake([0.82], Decimal("1.60"))
    assert staking.MIN_UNITS <= mid.units <= staking.MAX_UNITS
    assert mid == staking.suggest_stake([0.82], Decimal("1.60"))


def test_kelly_without_an_edge_shows_the_minimum_and_says_so():
    none = staking.suggest_stake([0.55, 0.55], Decimal("2.50"))
    assert none.units == staking.MIN_UNITS and not none.edge and none.kelly_fraction <= 0


def test_kelly_formula_is_quarter_kelly_of_the_product_probability():
    # p = 0.8, O = 1.8: f = (0.8 * 0.8 - 0.2) / 0.8 = 0.55; quarter Kelly at 1% per unit = 13.75 units, clamped to 3.5
    s = staking.suggest_stake([0.8], Decimal("1.8"))
    assert s.kelly_fraction == Decimal("0.55") and s.raw_units == Decimal("13.75")
    # p = 0.6 * 0.6 = 0.36, O = 3.5: f = (0.36 * 2.5 - 0.64) / 2.5 = 0.104; raw = 2.6 units
    t = staking.suggest_stake([0.6, 0.6], Decimal("3.5"))
    assert t.raw_units == Decimal("2.6") and t.units == Decimal("2.6") and t.bound == ""


@pytest.mark.parametrize("probs,odds", [([], "2"), ([0.7], "1"), ([1.2], "2"), ([0.7], "0")])
def test_kelly_refuses_nonsense_input(probs, odds):
    with pytest.raises(ValueError):
        staking.suggest_stake(probs, Decimal(odds))


# ---------------------------------------------------------------- the summary and the token

def loaded_legs():
    return (LoadedLeg(("m1", "18", "total=2.5", "over"), "Alpha v Beta", "Over/Under", "Over 2.5", Decimal("1.85"), "Over 2.5",
                      Decimal("1.85")),
            LoadedLeg(("m2", "18", "total=1.5", "over"), "Gamma v Delta", "Over/Under", "Over 1.5", Decimal("1.30"), "Over 1.5",
                      Decimal("1.30")))


def summary(**kw):
    base = dict(session_id="abcd1234", legs=loaded_legs(), total_odds=Decimal("2.41"), stake=Decimal("500"),
                expires_at=NOW + timedelta(minutes=5))
    base.update(kw)
    return Summary(**base)


def test_a_token_cannot_be_built_without_the_mint_authority():
    s = summary()
    with pytest.raises(TokenNotMinted):
        PlacementToken(s.session_id, s.legs, s.total_odds, s.stake, s.expires_at, s.digest)
    with pytest.raises(TokenNotMinted):
        mint_token(object(), s)
    with pytest.raises(TokenNotMinted):  # a forged hash is refused even with the authority
        PlacementToken(s.session_id, s.legs, s.total_odds, s.stake, s.expires_at, "0" * 64, MINT_AUTHORITY)
    assert mint_token(MINT_AUTHORITY, s).matches_contents()


def test_the_token_hash_binds_legs_odds_total_stake_and_expiry():
    base = summary()
    digests = {base.digest}
    changed_leg = (replace(base.legs[0], odds=Decimal("1.90")), base.legs[1])
    other_leg = (replace(base.legs[0], leg_key=("m1", "18", "total=3.5", "over")), base.legs[1])
    for variant in (summary(legs=changed_leg), summary(legs=other_leg), summary(total_odds=Decimal("2.50")),
                    summary(stake=Decimal("501")), summary(expires_at=NOW + timedelta(minutes=6)),
                    summary(legs=base.legs[:1])):
        digests.add(variant.digest)
    assert len(digests) == 7                                   # every change gives a different hash
    assert summary().digest == base.digest                     # deterministic
    assert summary(legs=tuple(reversed(base.legs))).digest == base.digest   # the order of legs does not matter
    assert summary(stake=Decimal("500.00")).digest == base.digest          # 500 and 500.00 are the same stake
    assert len(base.summary_id) == 12


def test_a_tampered_token_is_detected_by_recomputing():
    token = mint_token(MINT_AUTHORITY, summary())
    with pytest.raises(Exception):
        token.stake = Decimal("5000")           # frozen
    forged = object.__new__(PlacementToken)
    for name, value in (("session_id", "abcd1234"), ("legs", token.legs), ("total_odds", token.total_odds),
                        ("stake", Decimal("5000")), ("expires_at", token.expires_at), ("token_hash", token.token_hash),
                        ("authority", MINT_AUTHORITY)):
        object.__setattr__(forged, name, value)
    assert not forged.matches_contents()


def test_a_summary_needs_legs_a_positive_stake_and_a_utc_expiry():
    with pytest.raises(ValueError):
        summary(legs=())
    with pytest.raises(ValueError):
        summary(stake=Decimal("0"))
    with pytest.raises(ValueError):
        summary(expires_at=datetime(2026, 10, 1, 9, 5))


def test_only_the_session_state_machine_uses_the_mint_authority():
    offenders = []
    for path in (REPO / "predictions").rglob("*.py"):
        if path.name in ("session.py", "placement.py") and path.parent.name in ("flow", "core"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.Name):
                names = [node.id]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            if {"MINT_AUTHORITY", "mint_token"} & set(names):
                offenders.append(str(path.relative_to(REPO)))
    assert not offenders, offenders
    assert "MINT_AUTHORITY" in (REPO / "predictions" / "flow" / "session.py").read_text(encoding="utf-8")
