"""Stories 7.1 to 7.4: the American football engine (model, markets, key-number table), the key-number shield (NFL -3.5/-7.5 and
+2.5/+6.5, college -7.5 to -9.5 with the 72% Moneyline conversion), the first-half rules (top 10 first-quarter rank, the
24.5 step-down at 65% to 72%) and the wind and pace filter (15 mph, 28.5 s/play, 52% rush rate, missing wind). Boundaries
are tested at the exact values. No live call: inputs are fabricated, offers are fabricated or recorded (tests/data)."""
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from predictions.adapters.sportybet.fixtures import parse_page
from predictions.core.betting import UnsupportedLeg, leg_from_evaluation
from predictions.core.engines import ENGINES
from predictions.core.engines import american_football as af
from predictions.core.market_rules import is_whole_or_quarter_line
from predictions.core.engines.american_football import AmericanFootballEngine
from predictions.core.engines.common import Priced
from predictions.core.inputs import FOOTBALL_REGISTRY, Input
from predictions.core.model import PROBABILITY_FLOOR, Estimate, Evaluation, validate
from predictions.core.options import closest_option
from predictions.core.types import Fixture, MarketOffer, MatchId, Outcome


@pytest.fixture(autouse=True)
def _college_limits_off(monkeypatch):
    """These tests are about the model, the shield and the routing rules, with large fabricated edges: the college edge cap, the
    Over pause and the college shrinkage are put back to the NFL-style values here, and tested on their own in
    tests/test_college_limits.py."""
    monkeypatch.setitem(af.MAX_EV, af.COLLEGE, None)
    monkeypatch.setattr(af, "COLLEGE_OVERS_PAUSED", False)
    monkeypatch.setitem(af.SHRINK_GAMES, af.COLLEGE, 4.0)
    for name, old in (("MARGIN_SD", 17.0), ("TOTAL_SD", 14.0), ("TEAM_SD", 10.0), ("FH_TOTAL_SD", 9.0)):   # the thresholds below were tuned on these
        monkeypatch.setitem(getattr(af, name), af.COLLEGE, old)

D = Decimal
KICKOFF = datetime(2026, 10, 4, 17, 0, tzinfo=timezone.utc)
DATA = Path(__file__).resolve().parent / "data"
SRC = "test-source"


def make_fixture(offers=(), competition="USA - NFL"):
    mid = MatchId("american_football", competition, KICKOFF, "Home Team", "Away Team")
    return Fixture(mid, competition, "Home Team", "Away Team", KICKOFF, False, tuple(offers))


def side_inputs(side, pf=24.0, pa=24.0, fh_for=11.5, fh_against=11.5, n=6.0, q1_rank=None, pace=None, rush=None):
    out = [Input(f"{side}.matches", n, "games", SRC),
           Input(f"{side}.points_for", pf, "points per game", SRC),
           Input(f"{side}.points_against", pa, "points per game", SRC)]
    if fh_for is not None:
        out += [Input(f"{side}.fh_points_for", fh_for, "1st-half points per game", SRC),
                Input(f"{side}.fh_points_against", fh_against, "1st-half points per game", SRC)]
    if q1_rank is not None:
        out.append(Input(f"{side}.q1_rank", float(q1_rank), "rank by 1st-quarter points per game (1 = best)", SRC))
    if pace is not None:
        out.append(Input(f"{side}.pace", pace, "seconds per offensive play", SRC))
    if rush is not None:
        out.append(Input(f"{side}.rush_rate", rush, "percent of offensive plays that are rushes", SRC))
    return out


def inputs(home=None, away=None, wind=5.0, league=23.0):
    found = side_inputs("home", **(home or {})) + side_inputs("away", **(away or {}))
    found.append(Input("match.league_ppg", league, "points per team per game", SRC))
    found.append(Input("match.league_fh_ppg", league / 2, "1st-half points per team per game", SRC))
    if wind is not None:
        found.append(Input("match.wind_mph", wind, "mph", SRC))
    return found


def line_offers(market_id, name, lines, over="1.90", under="1.90", key="total", per=()):
    return [MarketOffer(market_id, name, f"{key}={line}", (Outcome(f"Over {line}", D(over)), Outcome(f"Under {line}", D(under))))
            for line in lines]


def totals(lines=("40.5", "44.5", "48.5"), **kw):
    return line_offers(af.TOTAL_MARKET_ID, "Over/Under (incl. overtime)", lines, **kw)


def team_totals(side="home", lines=("20.5", "23.5", "26.5"), **kw):
    return line_offers(af.HOME_TOTAL_MARKET_ID if side == "home" else af.AWAY_TOTAL_MARKET_ID,
                       f"{side.title()} Over/Under (incl. overtime)", lines, **kw)


def fh_totals(lines=("20.5", "21.5", "23.5", "24.5"), **kw):
    return line_offers(af.FH_TOTAL_MARKET_ID, "1st half - total", lines, **kw)


def spreads(prices):
    """prices: {shown line for the AWAY... no: home handicap: (home odds, away odds)}; the spec is the HOME handicap."""
    offers = []
    for hcp, (home, away) in prices.items():
        h = D(hcp)
        offers.append(MarketOffer(af.HANDICAP_MARKET_ID, "Handicap (incl. overtime)", f"hcp={hcp}",
                                  (Outcome(f"Home ({'+' if h > 0 else ''}{hcp})", D(home)),
                                   Outcome(f"Away ({'+' if -h > 0 else ''}{format(-h.normalize(), 'f')})", D(away)))))
    return offers


def moneyline(home="1.30", away="3.40"):
    return [MarketOffer(af.WINNER_MARKET_ID, "Winner (incl. overtime)", "", (Outcome("Home", D(home)), Outcome("Away", D(away))))]


def run(inps, offers, competition="USA - NFL"):
    return AmericanFootballEngine().evaluate(make_fixture(offers, competition), inps, offers)


def by(evals, text):
    return [e for e in evals if e.selection == text or e.market == text]


def ruled(evals, name):
    return [e for e in evals if e.rule.startswith(name)]


def home_strong(margin_boost):
    """Inputs whose home side is stronger by `margin_boost` points scored per game."""
    return inputs(home=dict(pf=24.0 + margin_boost, pa=24.0), away=dict(pf=24.0, pa=24.0 + margin_boost))


def cover_probability(inps, hcp, league="USA - NFL"):
    """The model's probability that the HOME side covers a handicap `hcp` (a half line)."""
    offers = spreads({hcp: ("1.90", "1.90")})
    e = by(run(inps, offers, league), f"Home ({'+' if D(hcp) > 0 else ''}{hcp})")[0]
    return e.estimate.probability


def bisect_boost(target, probability, low=0.0, high=30.0):
    """The smallest boost (to 4 decimals) at which probability(boost) is at least `target` and the one just below it."""
    for _ in range(60):
        mid = (low + high) / 2
        if probability(mid) >= target:
            high = mid
        else:
            low = mid
    return low, high


# ================================================================ registration, constants, the model
def test_football_is_registered_and_the_floor_is_imported_not_redefined():
    assert isinstance(ENGINES["american_football"], AmericanFootballEngine)
    source = (Path(af.__file__)).read_text(encoding="utf-8")
    assert "PROBABILITY_FLOOR =" not in source and "0.65" not in source.replace("0.65 ", "")


def test_thresholds_are_named_constants_with_the_specified_values():
    assert af.WIND_LIMIT_MPH == 15.0 and af.PACE_SLOW_SECONDS == 28.5 and af.RUSH_RATE_MIN_PERCENT == 52.0
    assert af.TOP_Q1_RANK == 10 and af.FH_STEP_LINE == D("24.5") and af.FH_STEP_TO == D("23.5") and af.FH_STEP_MAX == 0.72
    assert af.COLLEGE_MONEYLINE_MIN_WIN == 0.72 and af.COLLEGE_STEP_LINE == D("-6.5")
    assert set(af.NFL_FAVORITE_STEPS) == {D("-3.5"), D("-7.5")} and set(af.NFL_UNDERDOG_STEPS) == {D("2.5"), D("6.5")}
    assert af.NFL_FAVORITE_STEPS[D("-3.5")][0] == D("-2.5") and af.NFL_FAVORITE_STEPS[D("-7.5")][0] == D("-6.5")
    assert af.NFL_UNDERDOG_STEPS[D("2.5")] == (D("3.5"),) and af.NFL_UNDERDOG_STEPS[D("6.5")] == (D("7.5"),)


def test_the_key_number_tables_carry_the_owners_figures_and_order():
    assert af.NFL_KEY_NUMBERS[3] == 0.145 and af.NFL_KEY_NUMBERS[7] == 0.098
    nfl = sorted(af.NFL_KEY_NUMBERS, key=af.NFL_KEY_NUMBERS.get, reverse=True)
    assert nfl == [3, 7, 6, 10, 14]
    college = sorted(af.COLLEGE_KEY_NUMBERS, key=af.COLLEGE_KEY_NUMBERS.get, reverse=True)
    assert college == [7, 3, 10, 14, 4]


def test_the_margin_distribution_puts_extra_mass_on_key_numbers_and_sums_to_one():
    plain = af.normal_pmf(3.0, af.MARGIN_SD["nfl"], -af.MAX_MARGIN, af.MAX_MARGIN)
    keyed = af.margin_pmf(3.0, af.MARGIN_SD["nfl"], af.NFL_KEY_NUMBERS)
    assert abs(sum(keyed.values()) - 1.0) < 1e-9
    assert keyed[3] > plain[3] * 1.5 and keyed[7] > plain[7] * 1.3 and keyed[-3] > plain[-3] * 1.5
    assert keyed[5] < plain[5]                  # mass left the non-key margins
    # a whole-point key number is worth far more than the next half point: -2.5 covers a 3-point win, -3.5 does not
    assert keyed[3] > 0.06


def test_a_line_on_a_key_number_is_priced_differently_from_the_half_point_beside_it():
    inps = home_strong(6.0)
    gap = cover_probability(inps, "-2.5") - cover_probability(inps, "-3.5")
    plain = (af.normal_pmf(0.0, 13.5, -80, 80)[3]) * 2
    assert gap > plain   # the model knows 3 is a key number


def test_the_estimate_is_deterministic_and_missing_required_input_prices_nothing():
    offers = totals() + moneyline() + spreads({"-2.5": ("2.0", "1.8")})
    assert run(inputs(), offers) == run(inputs(), offers)
    missing = [i for i in inputs() if i.name != "home.points_for"]
    assert run(missing, offers) == []


def test_only_nfl_and_college_competitions_are_priced():
    assert af.league_of("USA - NFL") == "nfl" and af.league_of("USA - NCAA, Regular Season") == "college"
    assert af.league_of("Canada - CFL") is None and af.league_of("USA - NFL Preseason") is None
    assert run(inputs(), totals(), "Canada - CFL") == []


def test_the_registry_marks_pace_rank_and_wind_as_optional_inputs_and_points_as_required():
    assert FOOTBALL_REGISTRY["points_for"].required and FOOTBALL_REGISTRY["points_against"].required
    assert FOOTBALL_REGISTRY["matches"].required and FOOTBALL_REGISTRY["matches"].low == 3
    for stat in ("q1_rank", "pace", "rush_rate", "wind_mph", "fh_points_for"):
        assert not FOOTBALL_REGISTRY[stat].required
    assert FOOTBALL_REGISTRY["wind_mph"].scope == "match"


def test_the_moneyline_total_and_team_total_are_priced_with_sources_shown_and_gated_like_any_pick():
    offers = (moneyline("1.20", "4.5") + totals(("36.5", "44.5", "52.5"), over="1.90", under="1.90")
              + team_totals("home", ("20.5", "26.5")))
    evals = run(home_strong(8.0), offers)
    ml = by(evals, "Home")[0]
    assert ml.market == "Winner (incl. OT)" and ml.estimate.probability > 0.7
    text = " ".join(ml.notes)
    assert "home.points_for" in text and SRC in text and af.MODEL_NAME in text
    for e in evals:
        assert e.passed == validate(e.estimate, e.odds)[0] or e.rule  # the gate is core.model.validate
        assert (e.passed is False) or (e.estimate.probability >= PROBABILITY_FLOOR and e.ev > 0)
    assert {e.market.split(" 4")[0].split(" 3")[0] for e in evals if "team total" in e.market}


def test_one_line_is_shown_per_market_and_side_and_a_low_estimate_is_rejected_with_the_reason():
    evals = run(inputs(), totals(("30.5", "40.5", "44.5", "48.5", "60.5")))
    assert len([e for e in evals if e.market.startswith("Total points") and e.selection.startswith("Over")]) == 1
    assert not [e for e in evals if "Under" in e.selection]   # owner rule: no Under in any sport (AD-20)
    weak = by(run(inputs(), totals(("47.5",))), "Over 47.5")[0]
    assert not weak.passed and "below the 70% floor" in weak.reason


def test_positive_ev_is_required():
    evals = run(inputs(), totals(("20.5",), over="1.00", under="1.00"))
    over = by(evals, "Over 20.5")[0]
    assert over.estimate.probability >= PROBABILITY_FLOOR and not over.passed and "expected value" in over.reason
    assert not by(evals, "Under 20.5")


# ================================================================ Story 7.2: key-number shield, NFL
FAV = "Away ({})"


def nfl_fav(line, boost):
    """The AWAY side (the favourite) laying `line`, e.g. -3.5, with stronger away inputs."""
    return dict(inps=inputs(home=dict(pf=24.0, pa=24.0 + boost), away=dict(pf=24.0 + boost, pa=24.0)))


def away_spreads(prices):
    """Spreads where the away team is the favourite: prices maps the AWAY line (-3.5) to (home odds, away odds)."""
    return spreads({format(-D(k), "f"): v for k, v in prices.items()})


def test_an_nfl_favorite_at_minus_3_5_with_an_estimate_over_65_steps_down_to_minus_2_5_and_shows_the_original_and_rule():
    inps = nfl_fav("-3.5", 12.0)["inps"]
    offers = away_spreads({"-3.5": ("1.50", "3.00"), "-2.5": ("1.60", "1.55")})   # -3.5 has the higher EV, both clear 65%
    evals = run(inps, offers)
    away = [e for e in evals if e.market.startswith("Handicap") and e.selection.startswith("Away")]
    assert len(away) == 1
    picked = away[0]
    assert picked.selection == "Away (-2.5)" and picked.passed
    assert picked.rule.startswith(af.RULE_KEY) and "-3.5" in picked.rule and "-2.5" in picked.rule
    assert picked.original.startswith("Away (-3.5) @ 3.00") and "estimate" in picked.original


def test_an_nfl_favorite_at_minus_7_5_steps_down_to_minus_6_5():
    inps = nfl_fav("-7.5", 16.0)["inps"]
    offers = away_spreads({"-7.5": ("1.50", "2.60"), "-6.5": ("1.60", "1.75")})
    picked = [e for e in run(inps, offers) if e.selection.startswith("Away (")][0]
    assert picked.selection == "Away (-6.5)" and picked.original.startswith("Away (-7.5)") and picked.rule.startswith(af.RULE_KEY)


def test_the_half_point_is_never_bought_a_listed_whole_line_is_not_a_target_and_not_offered():
    inps = nfl_fav("-3.5", 12.0)["inps"]
    offers = away_spreads({"-3.5": ("1.50", "3.00"), "-3": ("1.55", "2.20")})   # only the whole -3 would be "buy the half"
    away = [e for e in run(inps, offers) if e.selection.startswith("Away (")]
    assert [e.selection for e in away] == ["Away (-3.5)"] and not away[0].passed       # the shield has no listed half line to land on
    assert "not offered" in away[0].reason and away[0].estimate.push == 0
    assert not any(is_whole_or_quarter_line(e) for e in run(inps, offers))
    assert af.NFL_FAVORITE_STEPS == {D("-3.5"): (D("-2.5"),), D("-7.5"): (D("-6.5"),)}   # half-line targets only


def test_only_lines_sportybet_really_lists_are_used_and_an_unlisted_shield_means_the_original_is_not_played():
    inps = nfl_fav("-3.5", 12.0)["inps"]
    offers = away_spreads({"-3.5": ("1.50", "3.00"), "-1.5": ("1.60", "1.70")})     # neither -2.5 nor -3 is listed
    picked = [e for e in run(inps, offers) if e.selection.startswith("Away (")]
    assert len(picked) == 1 and picked[0].selection == "Away (-3.5)" and not picked[0].passed
    assert "not offered" in picked[0].reason and picked[0].rule.startswith(af.RULE_KEY)


def test_the_converted_pick_stays_only_if_it_clears_the_floor_with_positive_ev():
    inps = nfl_fav("-3.5", 12.0)["inps"]
    offers = away_spreads({"-3.5": ("1.50", "3.00"), "-2.5": ("4.0", "1.12")})     # -2.5 is safer but its price kills the EV
    picked = [e for e in run(inps, offers) if e.selection.startswith("Away (")][0]
    assert picked.selection == "Away (-2.5)" and not picked.passed and "expected value" in picked.reason
    assert picked.original.startswith("Away (-3.5)")


def test_a_favorite_at_minus_3_5_below_the_floor_is_not_optimized():
    inps = nfl_fav("-3.5", 0.0)["inps"]
    offers = away_spreads({"-3.5": ("1.90", "1.90"), "-2.5": ("1.90", "1.90")})
    evals = [e for e in run(inps, offers) if e.selection.startswith("Away (")]
    assert all(not e.rule for e in evals) and all(not e.passed for e in evals)


@pytest.mark.parametrize("line,steps", [("-3.5", "-2.5"), ("-7.5", "-6.5")])
def test_the_65_percent_boundary_of_the_favorite_shield(line, steps):
    """The shield applies at an estimate of at least 65% and not below: bisect the strength around the floor."""
    offers = away_spreads({line: ("2.0", "3.0"), steps: ("2.0", "1.75")})

    def estimate(boost):
        e = [x for x in run(nfl_fav(line, boost)["inps"], away_spreads({line: ("2.0", "3.0")})) if x.selection == f"Away ({line})"][0]
        return e.estimate.probability
    below, above = bisect_boost(PROBABILITY_FLOOR, estimate, 0.0, 40.0)
    assert estimate(below) < PROBABILITY_FLOOR <= estimate(above)
    plain = [e for e in run(nfl_fav(line, below)["inps"], offers) if e.selection.startswith("Away (")]
    assert all(not e.rule for e in plain)
    shielded = [e for e in run(nfl_fav(line, above)["inps"], offers) if e.selection.startswith("Away (")]
    assert len(shielded) == 1 and shielded[0].selection == f"Away ({steps})" and shielded[0].rule.startswith(af.RULE_KEY)


@pytest.mark.parametrize("line,moves,boost", [("2.5", "3.5", 6.0), ("6.5", "7.5", 2.0)])
def test_an_nfl_underdog_at_plus_2_5_or_plus_6_5_moves_to_plus_3_5_or_plus_7_5(line, moves, boost):
    # the AWAY side is the underdog getting the points: the home handicap is the negative of the away line
    inps = inputs(home=dict(pf=24.0, pa=24.0 + boost), away=dict(pf=24.0 + boost, pa=24.0))
    offers = spreads({format(-D(line), "f"): ("2.0", "1.62"), format(-D(moves), "f"): ("2.4", "1.40")})
    picked = [e for e in run(inps, offers) if e.selection.startswith("Away (")][0]
    assert picked.selection == f"Away (+{moves})" and picked.rule.startswith(af.RULE_KEY)
    assert picked.original.startswith(f"Away (+{line})") and picked.passed and picked.estimate.probability >= PROBABILITY_FLOOR


def test_an_underdog_at_other_lines_is_not_moved():
    inps = inputs()
    offers = spreads({"-3.5": ("2.0", "1.62"), "-4.5": ("2.4", "1.40")})
    assert all(not e.rule for e in run(inps, offers) if e.selection.startswith("Away ("))


# ================================================================ Story 7.2: key-number shield, college
COLLEGE = "USA - NCAA, Regular Season"


def college_fav(boost, pa_extra=0.0):
    return inputs(home=dict(pf=24.0, pa=24.0 + boost), away=dict(pf=24.0 + boost, pa=24.0 + pa_extra))


@pytest.mark.parametrize("line", ["-7.5", "-8.5", "-9.5"])
def test_a_college_spread_from_minus_7_5_to_minus_9_5_steps_down_to_minus_6_5(line):
    offers = away_spreads({line: ("1.5", "2.2"), "-6.5": ("1.6", "1.5")})         # no Moneyline listed
    evals = run(college_fav(15.0), offers, COLLEGE)
    picked = [e for e in evals if e.selection.startswith("Away (")]
    assert len(picked) == 1
    p = picked[0]
    assert p.selection == "Away (-6.5)" and p.rule.startswith(af.RULE_KEY) and p.original.startswith(f"Away ({line})")
    assert p.passed and "-6.5" in p.rule and "no Moneyline conversion" in p.rule


@pytest.mark.parametrize("line", ["-6.5", "-10.5"])
def test_college_lines_outside_the_range_are_not_shielded(line):
    offers = away_spreads({line: ("1.5", "2.2"), "-5.5": ("1.6", "1.5")})
    assert all(not e.rule for e in run(college_fav(15.0), offers, COLLEGE) if e.selection.startswith("Away ("))


def _college_ml_win(boost=15.0):
    return [e for e in run(college_fav(boost), moneyline("4.0", "1.25"), COLLEGE) if e.selection == "Away"][0].estimate.probability


def test_the_72_percent_moneyline_threshold_is_the_specified_value_and_a_college_shield_implies_a_high_win_probability():
    assert af.COLLEGE_MONEYLINE_MIN_WIN == 0.72
    assert _college_ml_win() > 0.8       # a 65% cover at -7.5 or worse needs a much higher win probability than 72%


def test_the_college_spread_converts_to_the_moneyline_when_the_win_probability_is_at_least_the_threshold(monkeypatch):
    offers = away_spreads({"-8.5": ("1.5", "2.2"), "-6.5": ("1.6", "1.5")}) + moneyline("4.0", "1.25")
    win = _college_ml_win()
    monkeypatch.setattr(af, "COLLEGE_MONEYLINE_MIN_WIN", win)             # exactly at the threshold: converts
    at = [e for e in run(college_fav(15.0), offers, COLLEGE) if e.rule.startswith(af.RULE_KEY)]
    assert len(at) == 1 and at[0].market == af.WINNER_MARKET_TEXT and at[0].selection == "Away" and at[0].passed
    assert at[0].original.startswith("Away (-8.5)") and "Moneyline" in at[0].rule
    monkeypatch.setattr(af, "COLLEGE_MONEYLINE_MIN_WIN", win + 0.001)     # a hair below the threshold: only the step-down
    below = [e for e in run(college_fav(15.0), offers, COLLEGE) if e.rule.startswith(af.RULE_KEY)]
    assert len(below) == 1 and below[0].selection == "Away (-6.5)" and "no Moneyline conversion" in below[0].rule


def test_the_moneyline_conversion_stays_only_if_it_clears_ev_otherwise_the_step_down_is_used():
    offers = away_spreads({"-8.5": ("1.5", "2.2"), "-6.5": ("1.6", "1.5")}) + moneyline("9.0", "1.05")
    picked = [e for e in run(college_fav(15.0), offers, COLLEGE) if e.rule.startswith(af.RULE_KEY)]
    assert len(picked) == 1 and picked[0].selection == "Away (-6.5)" and picked[0].passed
    assert "converts to the Moneyline" in picked[0].rule


def test_nfl_never_converts_to_the_moneyline():
    offers = away_spreads({"-3.5": ("1.5", "3.0"), "-2.5": ("1.6", "1.55")}) + moneyline("9.0", "1.5")
    shielded = [e for e in run(nfl_fav("-3.5", 12.0)["inps"], offers) if e.rule.startswith(af.RULE_KEY)]
    assert shielded and all(e.market.startswith("Handicap") for e in shielded)


# ================================================================ Story 7.3: first-half rules
def priced_fh(line, probability, odds="1.90"):
    est = Estimate(probability)
    passed, ev, reason = validate(est, D(odds))
    mid = make_fixture().match_id
    return Priced("fh_total", "over", D(line), Evaluation(mid, f"1st half total points {line}", f"1st half Over {line}", D(odds),
                                                          est, ev, passed, reason))


def step_down(picked, pool, allowed=None):
    model = af._Model(inputs(), "nfl")
    return AmericanFootballEngine()._fh_over(model, picked, pool, allowed or pool, "", "")


@pytest.mark.parametrize("probability,steps", [(0.70, True), (0.72, True), (0.71, True), (0.699, False), (0.721, False)])
def test_a_first_half_over_at_24_5_steps_down_only_from_the_70_percent_floor_to_72_percent_inclusive(probability, steps):
    original = priced_fh("24.5", probability)
    lower = priced_fh("23.5", 0.78, "1.55")
    out = step_down(original, [original, lower])
    assert len(out) == 1
    if steps:
        assert out[0].selection == "1st half Over 23.5" and out[0].rule.startswith(af.RULE_FH_STEP) and out[0].passed
        assert out[0].original.startswith("1st half Over 24.5") and "23.5" in out[0].rule
    else:
        assert out[0].selection == "1st half Over 24.5" and not out[0].rule


def test_the_step_down_is_only_for_the_24_5_line():
    original = priced_fh("22.5", 0.70)
    assert step_down(original, [original, priced_fh("21.5", 0.8)])[0].selection == "1st half Over 22.5"


def test_the_step_down_pick_stays_only_if_it_clears_the_floor_with_positive_ev():
    original = priced_fh("24.5", 0.70)
    poor_price = priced_fh("23.5", 0.78, "1.10")     # 0.78 * 0.10 < 0.22: EV negative
    out = step_down(original, [original, poor_price])[0]
    assert out.selection == "1st half Over 23.5" and not out.passed and "expected value" in out.reason
    missing = step_down(original, [original])[0]
    assert missing.selection == "1st half Over 24.5" and not missing.passed and "not offered" in missing.reason


def test_a_single_team_first_half_total_is_also_an_accepted_step_down_when_sportybet_lists_one():
    original = priced_fh("24.5", 0.70)
    mid = make_fixture().match_id
    est = Estimate(0.80)
    single = Priced("fh_home_total", "over", D("12.5"), Evaluation(mid, "Home 1st half team total points 12.5",
                                                                 "Home 1st half team Over 12.5", D("1.55"), est,
                                                                 *validate(est, D("1.55"))))
    out = step_down(original, [original], [original, single])[0]
    assert out.selection == "Home 1st half team Over 12.5" and out.rule.startswith(af.RULE_FH_STEP)


def fh_route_inputs(home_rank, away_rank, boost=5.0):
    return inputs(home=dict(pf=24.0 + boost, pa=24.0, fh_for=13.5 + boost / 2, q1_rank=home_rank),
                  away=dict(pf=24.0 + boost, pa=24.0, fh_for=13.5 + boost / 2, q1_rank=away_rank))


@pytest.mark.parametrize("rank,routed", [(10, True), (11, False), (1, True)])
def test_a_team_in_the_top_10_for_first_quarter_efficiency_routes_the_over_to_the_first_half(rank, routed):
    offers = totals(("44.5", "48.5", "52.5")) + fh_totals(("20.5", "22.5"), over="1.85", under="1.95")
    evals = run(fh_route_inputs(rank, 25), offers)
    full_overs = [e for e in evals if e.selection.startswith("Over") and e.market.startswith("Total points")]
    fh_overs = [e for e in evals if e.selection.startswith("1st half Over")]
    assert fh_overs
    if routed:
        assert not full_overs and fh_overs[0].rule.startswith(af.RULE_Q1)
        assert f"rank {rank}" in fh_overs[0].rule and fh_overs[0].original.startswith("Over ")
    else:
        assert full_overs and not any(e.rule.startswith(af.RULE_Q1) for e in evals)


def test_the_route_can_come_from_either_team_and_stays_only_if_it_clears_the_floor():
    offers = totals(("44.5", "48.5", "52.5")) + fh_totals(("30.5",), over="1.90", under="1.90")
    evals = run(fh_route_inputs(25, 3), offers)
    routed = [e for e in evals if e.rule.startswith(af.RULE_Q1)]
    assert routed and not routed[0].passed and "below the 70% floor" in routed[0].reason


def test_a_first_half_team_total_over_is_the_route_for_the_top_team_when_sportybet_lists_one():
    """Dormant on SportyBet NG today: no 1st-half team total is listed (probed); it works on any offer named like one."""
    team = [MarketOffer("999", "Home 1st half - total", "total=8.5", (Outcome("Over 8.5", D("1.80")), Outcome("Under 8.5", D("1.90"))))]
    offers = totals(("44.5",)) + fh_totals(("20.5",)) + team
    evals = run(fh_route_inputs(2, 30), offers)
    kinds = {e.selection for e in evals if e.rule.startswith(af.RULE_Q1)}
    assert "Home 1st half team Over 8.5" in kinds and "1st half Over 20.5" in kinds
    assert af.market_kind(team[0]) == "fh_team_total" and af.fh_team_side(team[0]) == "home"


def test_without_first_half_data_nothing_is_routed_and_first_half_picks_are_insufficient():
    inps = inputs(home=dict(fh_for=None, q1_rank=2), away=dict(fh_for=None, q1_rank=4))
    offers = totals(("44.5",)) + fh_totals(("20.5",))
    evals = run(inps, offers)
    fh = [e for e in evals if e.market.startswith("1st half")]
    assert fh and all(e.insufficient for e in fh) and all("no first-half scoring data" in e.reason for e in fh)
    assert not any(e.rule.startswith(af.RULE_Q1) for e in evals)


def test_a_routed_over_with_no_first_half_market_listed_is_rejected_not_silently_dropped():
    evals = run(fh_route_inputs(1, 1), totals(("44.5", "48.5")))
    routed = [e for e in evals if e.rule.startswith(af.RULE_Q1)]
    assert routed and not routed[0].passed and "lists none" in routed[0].reason


# ================================================================ Story 7.4: wind and pace
BOOST = dict(home=dict(pf=32.0, pa=20.0), away=dict(pf=30.0, pa=21.0))     # a high-scoring matchup: Overs clear the floor


def over_evals(evals):
    return [e for e in evals if e.market.startswith("Total points") and e.selection.startswith("Over")]


def test_a_full_game_over_passes_in_calm_wind_and_the_matchup_is_a_real_over_candidate():
    evals = run(inputs(**BOOST, wind=5.0), totals(("40.5", "44.5")))
    assert any(e.passed for e in over_evals(evals))


def test_wind_of_exactly_15_mph_disqualifies_a_full_game_over_whatever_the_model_says():
    evals = run(inputs(**BOOST, wind=15.0), totals(("40.5", "44.5")))
    over = over_evals(evals)
    assert len(over) == 1 and not over[0].passed and not over[0].insufficient
    assert over[0].rule.startswith(af.RULE_WIND) and "15 mph" in over[0].reason and "whatever the model says" in over[0].reason
    assert over[0].estimate.probability >= PROBABILITY_FLOOR       # the model liked it; the wind rule overrode it


def test_wind_of_14_9_mph_does_not_disqualify():
    evals = run(inputs(**BOOST, wind=14.9), totals(("40.5", "44.5")))
    assert any(e.passed for e in over_evals(evals)) and not any(e.rule.startswith(af.RULE_WIND) for e in evals)


def test_a_wind_disqualified_over_is_not_replaced_by_a_team_total_under():   # owner rule (AD-20): no Under, the shift is gone
    inps = inputs(home=dict(pf=17.0, pa=17.0), away=dict(pf=17.0, pa=17.0), wind=20.0)
    offers = totals(("34.5",)) + team_totals("home", ("26.5",)) + team_totals("away", ("26.5",))
    evals = run(inps, offers)
    assert not [e for e in evals if "Under" in e.selection or "Under" in e.rule or "Under" in e.original]
    assert not any(e.passed for e in evals)      # nothing at all is played: wind blocks every Over and no Under replaces it
    assert any(e.rule.startswith(af.RULE_WIND) and not e.passed for e in evals)


def test_without_listed_team_totals_the_shift_has_nothing_to_move_to_and_the_over_is_simply_disqualified():
    evals = run(inputs(**BOOST, wind=20.0), totals(("40.5",)))
    assert not any(e.passed and e.selection.startswith("Over") for e in evals)
    assert not any(e.rule.startswith(af.RULE_WIND) and e.passed for e in evals)   # a rushing prop is not listed by SportyBet NG


def test_missing_wind_data_makes_every_over_insufficient_data_never_assumed_calm():
    evals = run(inputs(**BOOST, wind=None), totals(("40.5", "44.5")) + team_totals("home", ("20.5",)) + fh_totals(("20.5",)))
    overs = [e for e in evals if "Over" in e.selection]
    assert overs and all(e.insufficient and not e.passed and "no wind forecast" in e.reason for e in overs)
    assert all("never assumed calm" in e.reason for e in overs)
    assert not [e for e in evals if "Under" in e.selection]       # no Under is ever produced (AD-20)


def test_missing_wind_does_not_stop_spreads_and_the_moneyline():
    offers = spreads({"-3.5": ("1.9", "1.9")}) + moneyline()
    evals = run(inputs(**BOOST, wind=None), offers)
    assert {e.market.split(" ")[0] for e in evals} >= {"Handicap", "Winner"} and not any(e.insufficient for e in evals)


def test_wind_also_guards_first_half_and_team_total_overs():
    evals = run(inputs(**BOOST, wind=16.0), team_totals("home", ("20.5",)) + fh_totals(("20.5",)))
    overs = [e for e in evals if "Over" in e.selection]
    assert overs and all(not e.passed and e.rule.startswith(af.RULE_WIND) for e in overs)


def slow(rush=54.0, pace=29.0):
    return dict(pace=pace, rush=rush)


@pytest.mark.parametrize("pace,rush,baseline", [(28.6, 52.1, True), (28.5, 60.0, False), (29.5, 52.0, False), (28.6, 52.0, False)])
def test_a_team_slower_than_28_5_seconds_per_play_that_rushes_on_more_than_52_percent_blocks_the_over_and_plays_no_under(pace, rush, baseline):
    home = dict(pf=32.0, pa=20.0, pace=pace, rush=rush)
    evals = run(inputs(home=home, away=BOOST["away"], wind=5.0), totals(("40.5", "44.5")))
    over = over_evals(evals)
    if baseline:
        assert len(over) == 1 and not over[0].passed and over[0].rule.startswith(af.RULE_PACE)
        assert "clock-control baseline" in over[0].reason and "home" in over[0].reason
        assert not [e for e in evals if "Under" in e.selection or "Under" in e.reason]   # no Under replaces it (AD-20)
    else:
        assert any(e.passed for e in over) and not any(e.rule.startswith(af.RULE_PACE) for e in evals)


def test_the_pace_rule_needs_both_pace_and_rush_data_and_never_guesses_missing_values():
    home = dict(pf=32.0, pa=20.0, pace=29.5)          # no rush rate
    evals = run(inputs(home=home, away=BOOST["away"], wind=5.0), totals(("40.5", "44.5")))
    assert any(e.passed for e in over_evals(evals)) and not any(e.rule.startswith(af.RULE_PACE) for e in evals)


def test_a_slow_run_heavy_team_blocks_only_its_own_team_total_over():
    home = dict(pf=32.0, pa=20.0, pace=29.5, rush=55.0)
    away = dict(pf=32.0, pa=22.0, pace=27.0, rush=45.0)
    evals = run(inputs(home=home, away=away, wind=5.0), team_totals("home", ("20.5",)) + team_totals("away", ("20.5",)))
    home_over = [e for e in evals if e.selection == "Home team Over 20.5"][0]
    away_over = [e for e in evals if e.selection == "Away team Over 20.5"][0]
    assert not home_over.passed and home_over.rule.startswith(af.RULE_PACE)
    assert away_over.passed and not away_over.rule


def test_wind_is_checked_before_pace_and_a_missing_wind_reading_wins_over_the_pace_rule():
    home = dict(pf=32.0, pa=20.0, pace=29.5, rush=55.0)
    evals = run(inputs(home=home, away=BOOST["away"], wind=None), totals(("40.5",)))
    over = over_evals(evals)
    assert over[0].insufficient and over[0].rule.startswith(af.RULE_WIND)


# ================================================================ Review (exact selections), legs, listing sample
def exact(offer, inps, competition="USA - NFL"):
    return AmericanFootballEngine().evaluate_exact(make_fixture([offer], competition), inps, offer)


def test_review_judges_an_exact_selection_at_the_codes_odds_with_no_optimization():
    offer = spreads({"3.5": ("1.60", "2.40")})[0]
    inps = nfl_fav("-3.5", 12.0)["inps"]
    got = exact(offer, inps)
    assert {e.selection for e in got} == {"Home (+3.5)", "Away (-3.5)"} and not any(e.rule for e in got)
    assert {e.odds for e in got} == {D("1.60"), D("2.40")}


def test_review_still_applies_the_wind_guard_to_an_over():
    offer = totals(("40.5",))[0]
    hot = exact(offer, inputs(**BOOST, wind=15.0))
    calm = exact(offer, inputs(**BOOST, wind=14.9))
    none = exact(offer, inputs(**BOOST, wind=None))
    assert [e.passed for e in hot if e.selection.startswith("Over")] == [False]
    assert [e.passed for e in calm if e.selection.startswith("Over")] == [True]
    assert [e.insufficient for e in none if e.selection.startswith("Over")] == [True]
    assert all(e.passed is not None for e in hot)
    assert not [e for e in hot + calm + none if "Under" in e.selection]   # no Under (AD-20)


def test_review_understands_only_the_markets_the_engine_prices():
    eng = AmericanFootballEngine()
    priced = totals(("44.5",)) + spreads({"-3.5": ("1.9", "1.9")}) + moneyline() + fh_totals(("20.5",))
    assert all(eng.understands(o) for o in priced)
    other = MarketOffer("66", "1st half - handicap (2 way)", "hcp=0.5", (Outcome("Home (+0.5)", D("1.9")),))
    assert not eng.understands(other) and eng.evaluate_exact(make_fixture(), inputs(), other) == []


def sample_fixtures():
    payload = json.loads((DATA / "sportybet_football_sample.json").read_text(encoding="utf-8"))
    return parse_page(payload, "american_football")[0]


def test_the_recorded_football_listing_parses_with_the_priced_markets_and_prices_end_to_end():
    fixtures = sample_fixtures()
    assert {f.competition for f in fixtures} == {"USA - NFL", "USA - NCAA, Regular Season", "Canada - CFL"}
    assert {m.market_id for f in fixtures for m in f.markets} <= {"1", "66", "68", "219", "223", "225", "227", "228"}
    engine = AmericanFootballEngine()
    priced = 0
    for fixture in fixtures:
        league = af.league_of(fixture.competition)
        if league is None:
            assert engine.evaluate(fixture, inputs(), fixture.markets) == []
            continue
        evals = engine.evaluate(fixture, inputs(**BOOST, wind=5.0), fixture.markets)
        assert evals
        priced += len(evals)
        for e in evals:
            if not e.insufficient:
                leg = leg_from_evaluation(e)       # every priced pick maps to a SportyBet market, specifier and side
                assert leg.market_id in {"219", "223", "225", "227", "228", "68"} and leg.side in ("home", "away", "over", "under")
    assert priced > 10


def test_legs_carry_the_specifier_and_side_sportybet_uses():
    mid = make_fixture().match_id

    def ev(market, selection):
        est = Estimate(0.7)
        return Evaluation(mid, market, selection, D("1.9"), est, D("0.1"), True, "passed")
    leg = leg_from_evaluation(ev("Handicap +3.5", "Away (-3.5)"))
    assert (leg.market_id, leg.specifier, leg.side) == ("223", "hcp=3.5", "away")
    assert leg_from_evaluation(ev("Handicap -3", "Home (-3)")).specifier == "hcp=-3"
    assert leg_from_evaluation(ev("Winner (incl. OT)", "Home")).market_id == "219"
    assert leg_from_evaluation(ev("Total points 44.5", "Over 44.5")).specifier == "total=44.5"
    assert leg_from_evaluation(ev("1st half total points 23.5", "1st half Over 23.5")).market_id == "68"
    assert leg_from_evaluation(ev("Away team total points 20.5", "Away team Under 20.5")).market_id == "228"
    with pytest.raises(UnsupportedLeg):
        leg_from_evaluation(ev("Home 1st half team total points 8.5", "Home 1st half team Over 8.5"))


def test_the_closest_option_rule_knows_football_total_lines():
    mid = make_fixture().match_id

    def ev(line, p):
        est = Estimate(p)
        return Evaluation(mid, f"Total points {line}", f"Over {line}", D("1.9"), est, D("0.2"), True, "passed")
    original = leg_from_evaluation(ev("44.5", 0.7))
    assert closest_option(original, [ev("43.5", 0.72), ev("41.5", 0.8)]).selection == "Over 43.5"
