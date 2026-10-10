"""College football limits (2026-10-10 review of the first settled week: 54 legs, 72.8% predicted, 53.7% won, picks claiming more
than +50% edge won 1 of 8, Overs 2 of 10): a cap on the edge the model may claim, no college Overs, more shrinkage towards the
league level, four games before a team is priced, and no analyst fill-in for a college team the free source does not cover."""
import asyncio
from decimal import Decimal as D

import pytest

from predictions.core.engines import american_football as af
from predictions.core.inputs import Input
from predictions.core.stats import assess
from test_football_engine import (COLLEGE, fh_totals, home_strong, inputs, make_fixture, moneyline, run, side_inputs, team_totals,
                                  totals)


def probability(inps, competition):
    found = [e for e in run(inps, moneyline(home="1.01", away="9.0"), competition) if e.selection == "Home"]
    return found[0].estimate.probability


def test_a_college_pick_claiming_more_than_the_edge_cap_is_not_passed_but_the_same_pick_in_the_nfl_is():
    inps = home_strong(8.0)
    p = probability(inps, COLLEGE)
    assert p > 0.7
    big_edge = D("2.30")                                   # claims an edge of well over 20%
    college = [e for e in run(inps, moneyline(home=str(big_edge)), COLLEGE) if e.selection == "Home"][0]
    assert not college.passed and "cap for college football" in college.reason and college.ev > D("0.20")
    nfl = [e for e in run(inps, moneyline(home=str(big_edge)), "USA - NFL") if e.selection == "Home"][0]
    assert nfl.passed                                      # the NFL is not capped: its data was not separable


def test_a_college_pick_with_a_moderate_edge_still_passes():
    inps = home_strong(8.0)
    p = probability(inps, COLLEGE)
    odds = (D("1.12") / D(str(p))).quantize(D("0.01"))          # roughly 12% expected value
    college = [e for e in run(inps, moneyline(home=str(odds)), COLLEGE) if e.selection == "Home"][0]
    assert college.passed and D("0") < college.ev <= D("0.20"), (odds, college.ev, college.reason)


def test_no_college_over_is_played_but_nfl_overs_still_are():
    high = inputs(home=dict(pf=36.0, pa=36.0), away=dict(pf=36.0, pa=36.0))
    offers = totals(lines=("40.5", "44.5")) + team_totals("home", lines=("20.5", "23.5")) + fh_totals(lines=("19.5", "20.5"))
    college = [e for e in run(high, offers, COLLEGE) if e.selection.startswith("Over") or " Over " in e.selection]
    assert college and not any(e.passed for e in college)
    assert all(e.rule == af.RULE_PAUSED and "paused" in e.reason for e in college)
    nfl = run(high, offers, "USA - NFL")
    assert any(e.passed and "Over" in e.selection for e in nfl)


def test_the_college_standard_deviations_are_the_measured_ones():
    assert (af.MARGIN_SD[af.COLLEGE], af.TOTAL_SD[af.COLLEGE], af.TEAM_SD[af.COLLEGE], af.FH_TOTAL_SD[af.COLLEGE]) == (17.7, 15.5, 11.8, 11.1)
    assert af.SHRINK_GAMES == {af.NFL: 4.0, af.COLLEGE: 4.0}


def test_opponent_adjusted_rates_move_the_margin_and_winner_but_never_the_totals():
    plain = inputs(home=dict(pf=26.0, pa=24.0), away=dict(pf=26.0, pa=24.0))
    unit = "points per game, opponent-adjusted"
    extra = [Input("home.adj_points_for", 33.0, unit, "test"), Input("home.adj_points_against", 18.0, unit, "test"),
             Input("away.adj_points_for", 21.0, unit, "test"), Input("away.adj_points_against", 27.0, unit, "test")]
    without, adjusted = af._Model(plain, af.COLLEGE), af._Model(plain + extra, af.COLLEGE)
    assert not without.adjusted and adjusted.adjusted
    assert adjusted.p_home_wins() > without.p_home_wins() + 0.05            # the adjustment says the home side is much better
    assert adjusted.mu == without.mu                                         # the raw expectations, used for every total, are untouched
    assert adjusted.total_dist() == without.total_dist() and adjusted.team_dist("home") == without.team_dist("home")
    assert "opponent-adjusted" in adjusted.summary("home.points_for") and "opponent-adjusted" not in without.summary("home.points_for")
    partial = af._Model(plain + extra[:3], af.COLLEGE)                       # one of the four missing: nothing is half-applied
    assert not partial.adjusted and partial.p_home_wins() == without.p_home_wins()


def test_the_espn_source_adds_adjusted_rates_for_college_games_only():
    from predictions.adapters.espn import football as fb
    from test_football_stats import COLLEGE as COLLEGE_NAME, college_api, fetch, fixture as stats_fixture
    game = stats_fixture("Delaware Fightin\u00b4 Blue Hens", "Liberty Flames", competition=COLLEGE_NAME)
    result = fetch(college_api(), game)[game.match_id]
    names = {i.name for i in result.inputs}
    assert {"home.adj_points_for", "home.adj_points_against", "away.adj_points_for", "away.adj_points_against"} <= names
    assert all(i.unit == "points per game, opponent-adjusted" for i in result.inputs if ".adj_" in i.name)
    assert fb.adjusted_inputs([], "1", "2", "s") == []                       # no games, no ratings


def test_the_minimum_games_are_four_for_college_and_three_for_the_nfl():
    assert af.MIN_GAMES == {af.NFL: 3, af.COLLEGE: 4}


def test_the_analyst_is_never_asked_to_fill_a_college_football_team_but_is_for_other_games():
    from predictions.flow.analyst import AnalystRun, fill_missing_inputs

    class Spy:
        last_resort = False

        def __init__(self):
            self.queries = []

        async def research(self, query):
            from predictions.core.analyst import AnalystReport
            self.queries.append(query)
            return AnalystReport()

    college, nfl = make_fixture((), COLLEGE), make_fixture((), "USA - NFL")
    spy = Spy()
    stats = {college.match_id: assess(college.match_id, []), nfl.match_id: assess(nfl.match_id, [])}
    assert stats[college.match_id].status == "insufficient data"
    asyncio.run(fill_missing_inputs([college], stats, AnalystRun(spy, limit=5, cache=None)))
    assert spy.queries == []                                   # a college team below FBS/FCS: no web-found numbers
    asyncio.run(fill_missing_inputs([nfl], stats, AnalystRun(spy, limit=5, cache=None)))
    assert len(spy.queries) == 1
