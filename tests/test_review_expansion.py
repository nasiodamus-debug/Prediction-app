"""Review: Asian handicap and multigoals pricing, the goals-only fallback after the analyst, and no analyst lookups for basketball."""
import asyncio
import json
from decimal import Decimal as D

import pytest

from predictions.core.analyst import AnalystUsage, parse_report
from predictions.core.engines.soccer import SoccerEngine, poisson_cdf, result_probabilities, scaled_rates
from predictions.core.inputs import Input
from predictions.core.review import judge_selection
from predictions.core.stats import INSUFFICIENT, READY, FixtureStats, insufficient
from predictions.core.types import MarketOffer, Outcome
from predictions.flow.analyst import AnalystRun, fill_missing_inputs
from predictions.flow.review import run_review
from test_review import FakeCodeReader, FakeStats, NOW, selections
from test_soccer_optimization import HOME_FAVOURITE, TOP, fixture


def exact(market_id, name, specifier, outcome, odds="1.80", inputs=HOME_FAVOURITE):
    offer = MarketOffer(market_id, name, specifier, (Outcome(outcome, D(odds)),))
    found = SoccerEngine().evaluate_exact(fixture([], competition=TOP), inputs, offer)
    assert len(found) == 1
    return found[0]


# ---------------------------------------------------------------- Asian handicap
def test_the_two_sides_of_a_whole_handicap_and_the_refund_add_up_to_one():
    home = exact("16", "Asian Handicap", "hcp=-1", "Home (-1.0)")
    away = exact("16", "Asian Handicap", "hcp=-1", "Away (+1.0)")
    assert home.estimate.push == pytest.approx(away.estimate.push, abs=1e-3) and home.estimate.push > 0.1
    assert home.estimate.win + away.estimate.win + home.estimate.push == pytest.approx(1.0, abs=3e-3)
    assert "refunded" in home.notes[0]


def test_a_half_handicap_has_no_refund_and_its_two_sides_add_up_to_one():
    home = exact("16", "Asian Handicap", "hcp=-0.5", "Home (-0.5)")
    away = exact("16", "Asian Handicap", "hcp=-0.5", "Away (+0.5)")
    assert home.estimate.push == 0 and away.estimate.push == 0
    assert home.estimate.probability + away.estimate.probability == pytest.approx(1.0, abs=2e-3)


def test_a_quarter_handicap_splits_the_stake_over_the_two_lines_either_side():
    home, draw, away = result_probabilities(*scaled_rates(HOME_FAVOURITE))
    plus_quarter = exact("16", "Asian Handicap", "hcp=0.25", "Home (+0.25)")
    assert plus_quarter.estimate.probability == pytest.approx(home + draw, abs=3e-3)   # win, or draw refunded in half and won in half
    assert plus_quarter.estimate.push > 0


def test_the_handicap_market_is_understood_only_full_time_and_a_bad_outcome_is_skipped():
    engine = SoccerEngine()
    assert engine.understands(MarketOffer("16", "Asian Handicap", "hcp=-1", ()))
    assert not engine.understands(MarketOffer("66", "1st Half - Asian Handicap", "hcp=0.5", ()))
    offer = MarketOffer("16", "Asian Handicap", "hcp=-1", (Outcome("Home -1", D("1.8")), Outcome("Home (-1.3)", D("1.8"))))
    assert engine.evaluate_exact(fixture([], competition=TOP), HOME_FAVOURITE, offer) == []


def test_matchday_never_prices_a_handicap_or_multigoals():
    f = fixture([MarketOffer("16", "Asian Handicap", "hcp=-1", (Outcome("Home (-1.0)", D("1.8")),)),
                 MarketOffer("999", "Multigoals", "", (Outcome("2-5", D("1.5")),))], competition=TOP)
    assert SoccerEngine().evaluate(f, HOME_FAVOURITE, f.markets) == []


# ---------------------------------------------------------------- multigoals
def test_multigoals_ranges_and_open_ended_outcomes_are_priced_from_the_total_goals():
    total = sum(scaled_rates(HOME_FAVOURITE))
    two_to_five = exact("999", "Multigoals", "", "2-5", "1.40")
    assert two_to_five.estimate.probability == pytest.approx(poisson_cdf(5, total) - poisson_cdf(1, total), abs=2e-3)
    four_plus = exact("999", "Multigoals", "", "4+", "2.00")
    assert four_plus.estimate.probability == pytest.approx(1 - poisson_cdf(3, total), abs=2e-3)


def test_other_half_markets_and_unreadable_outcomes_are_not_priced():
    engine = SoccerEngine()
    assert not engine.understands(MarketOffer("999", "1st Half - Multigoals", "", ()))   # only the second-half versions are priced
    offer = MarketOffer("999", "Multigoals", "", (Outcome("5-2", D("1.5")), Outcome("some", D("1.5"))))
    assert engine.evaluate_exact(fixture([], competition=TOP), HOME_FAVOURITE, offer) == []


# ---------------------------------------------------------------- goals-only fallback after the analyst
def goals_only_analyst(with_xg=False):
    class Analyst:
        queries = []

        async def research(self, query):
            self.queries.append(query)
            values = {"matches": 10.0, "goals_for": 1.6, "goals_against": 1.0}
            if with_xg:
                values["xg_for"] = 1.5
            stats = [{"name": n, "value": values[n.partition(".")[2]], "unit": "matches" if n.endswith("matches") else
                      ("xG per match" if "xg" in n else "goals per match"), "citation": "example.com/stats"}
                     for n in query.wanted if n.partition(".")[2] in values]
            return parse_report(json.dumps({"stats": stats, "flags": []}), query, AnalystUsage(1, 1, 1))
    return Analyst()


def soccer_game():
    s = [x for x in selections() if x.sport == "soccer"][0]
    from predictions.core.types import Fixture
    return Fixture(s.match_id, s.competition, s.home, s.away, s.kickoff_utc, False)


def test_without_xg_the_analyst_goals_price_the_game_with_the_goals_only_model_when_the_fallback_is_on():
    game = soccer_game()
    stats = {game.match_id: insufficient(game.match_id, "missing required input: home.xg_for")}
    stats_off, _, _ = asyncio.run(fill_missing_inputs([game], stats, AnalystRun(goals_only_analyst())))
    assert stats_off[game.match_id].status == INSUFFICIENT
    stats_on, _, _ = asyncio.run(fill_missing_inputs([game], stats, AnalystRun(goals_only_analyst()), goals_only_fallback=True))
    entry = stats_on[game.match_id]
    assert entry.status == READY and all(not i.name.endswith(("xg_for", "xg_against")) for i in entry.inputs)
    assert all(i.source.startswith("analyst-sourced:") for i in entry.inputs)
    ev = judge_selection([x for x in selections() if x.sport == "soccer"][0], game, entry)
    assert not ev.insufficient and ev.estimate.model == "goals-only-poisson-v1"


def test_too_few_matches_stay_insufficient_even_with_the_fallback():
    game = soccer_game()
    stats = {game.match_id: insufficient(game.match_id, "missing required input: home.xg_for")}

    class Few(type(goals_only_analyst())):
        queries = []

        async def research(self, query):
            report = await super().research(query)
            return parse_report(json.dumps({"stats": [{"name": s.name, "value": 3.0 if s.name.endswith("matches") else s.value,
                                                       "unit": s.unit, "citation": "example.com/stats"} for s in report.stats],
                                            "flags": []}), query, AnalystUsage(1, 1, 1))

    result, _, _ = asyncio.run(fill_missing_inputs([game], stats, AnalystRun(Few()), goals_only_fallback=True))
    assert result[game.match_id].status == INSUFFICIENT


# ---------------------------------------------------------------- no analyst lookups for basketball in Review
class Recorder:
    def __init__(self):
        self.queries = []

    async def research(self, query):
        self.queries.append(query)
        return parse_report(json.dumps({"stats": [], "flags": []}), query, AnalystUsage(1, 1, 1))


class NoBasketballData:
    async def fetch_stats(self, fixtures):
        return {f.match_id: insufficient(f.match_id, "missing required input: home.ortg") for f in fixtures}


def test_review_never_asks_the_analyst_for_basketball_stats_but_still_does_for_soccer():
    analyst = Recorder()
    asyncio.run(run_review(" ABC123", FakeCodeReader(selections()), str,
                           stats_sources={"soccer": FakeStats(), "basketball": NoBasketballData()}, now=NOW, analyst=analyst))
    sports = {q.sport for q in analyst.queries}
    assert "basketball" not in sports and "soccer" in sports
