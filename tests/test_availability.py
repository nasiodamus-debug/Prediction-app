"""Spec 12-1: a game is analysed only when it is OPEN FOR BETTING on SportyBet now (listed, not started, an active market
the sport's engine prices). Games that fail are dropped before any stats or analyst call and counted in the WHY block and
the footer. Fabricated listings only; no live call."""
import asyncio
from datetime import timedelta
from decimal import Decimal

import pytest

from predictions.adapters.sportybet.fixtures import parse_event
from predictions.core.availability import is_open_for_betting
from predictions.core.stats import insufficient
from predictions.core.types import Fixture, MarketOffer, Outcome
from predictions.flow.america import run_america
from predictions.flow.analyst import DailyBudget
from predictions.flow.gameday import run_gameday
from predictions.flow.store import EvaluationStore
from test_competition_limits import (KICKOFF, NOW, FakeAnalyst, Listing, Table, game, good_stats, matchday, not_covered,
                                     totals)

TOURNAMENT = {"categoryName": "X", "name": "Y"}


def raw_event(markets, status=0):
    return {"eventId": "sr:match:1", "homeTeamName": "Alpha", "awayTeamName": "Beta", "status": status,
            "estimateStartTime": int(KICKOFF.timestamp() * 1000), "markets": markets}


def raw_market(market_id, status=0, active=1, odds="1.90"):
    return {"id": market_id, "name": "m", "specifier": "", "status": status,
            "outcomes": [{"id": "1", "desc": "Home", "odds": odds, "isActive": active},
                         {"id": "2", "desc": "Away", "odds": odds, "isActive": active}]}


def unpriced():
    return (MarketOffer("99999", "Something the engine ignores", "", (Outcome("Yes", Decimal("2.0"), "1"),)),)


# ---------------------------------------------------------------- the check itself
@pytest.mark.parametrize("sport", ["soccer", "basketball", "baseball", "hockey", "american_football"])
def test_a_game_with_an_open_priced_market_is_open(sport):
    assert is_open_for_betting(game("A", "B", "X - Y", sport), NOW)


@pytest.mark.parametrize("sport", ["soccer", "basketball", "baseball", "hockey", "american_football"])
def test_a_game_with_no_market_or_only_unpriced_markets_is_not_open(sport):
    assert not is_open_for_betting(game("A", "B", "X - Y", sport, markets=()), NOW)
    assert not is_open_for_betting(game("A", "B", "X - Y", sport, markets=unpriced()), NOW)


def test_a_started_or_past_game_is_not_open():
    f = game("A", "B", "X - Y")
    assert not is_open_for_betting(f, KICKOFF + timedelta(minutes=1))
    started = Fixture(f.match_id, f.competition, f.home, f.away, f.kickoff_utc, True, f.markets)
    assert not is_open_for_betting(started, NOW)


def test_the_listing_adapter_drops_suspended_markets_and_inactive_outcomes_so_the_game_is_not_open():
    suspended = parse_event(raw_event([raw_market("1", status=2)]), TOURNAMENT)
    inactive = parse_event(raw_event([raw_market("1", active=0)]), TOURNAMENT)
    live_ok = parse_event(raw_event([raw_market("1")]), TOURNAMENT)
    assert not is_open_for_betting(suspended, NOW) and not is_open_for_betting(inactive, NOW)
    assert is_open_for_betting(live_ok, NOW)
    closed = parse_event(raw_event([raw_market("1")], status=3), TOURNAMENT)
    assert not is_open_for_betting(closed, NOW)


# ---------------------------------------------------------------- the slate: nothing is analysed for a game that is not open
def test_matchday_never_looks_up_or_asks_the_analyst_about_a_game_that_is_not_open():
    fixtures = [game("Alpha", "Beta", "England - Premier League"),
                game("Shut A", "Shut B", "England - Premier League", markets=()),
                game("Odd A", "Odd B", "Spain - LaLiga", markets=unpriced())]
    analyst = FakeAnalyst()
    chunks, gate, stats = matchday(fixtures, analyst=analyst)
    assert stats.asked == ["Alpha"] and [q.home for q in analyst.queries] == ["Alpha"]
    assert chunks[0] == "NO VALID PLAY - PASS"
    assert "- 2 game(s) skipped: not open for betting on SportyBet" in chunks[1]
    assert "- 1 game(s) looked at" in chunks[1]
    assert "Shut A" not in "\n".join(chunks)


def test_the_not_open_count_is_in_the_footer_of_a_passing_matchday_list():
    ok = game("Alpha", "Beta", "Netherlands - Eredivisie", markets=[totals("1.5", "1.30", "3.20")])
    shut = game("Shut A", "Shut B", "Netherlands - Eredivisie", markets=())
    cup = game("Cup A", "Cup B", "England - FA Cup")
    chunks, gate, stats = matchday([ok, shut, cup], good_stats)
    assert gate.passed and stats.asked == ["Alpha"]
    assert gate.footer == ("1 game(s) ignored: outside your allowed competitions.",
                           "1 game(s) skipped: not open for betting on SportyBet.")


def test_only_games_in_the_allowed_competitions_are_counted_as_not_open():
    fixtures = [game("Out A", "Out B", "Germany - 2. Bundesliga", markets=()),
                game("Shut", "Down", "England - Premier League", markets=())]
    chunks, gate, stats = matchday(fixtures, analyst=FakeAnalyst())
    text = "\n".join(chunks)
    assert gate is None and stats.asked == []
    assert "No games open for betting on SportyBet" in text
    assert "1 game(s) skipped: not open for betting on SportyBet." in text
    assert "1 game(s) ignored: outside your allowed competitions." in text


def test_gameday_drops_a_game_that_is_not_open_before_stats():
    fixtures = [game("Lakers", "Celtics", "USA - NBA", "basketball"),
                game("Aces", "Liberty", "USA - WNBA", "basketball", markets=())]
    stats, gates, analyst = Table(not_covered), [], FakeAnalyst()
    chunks = asyncio.run(run_gameday("1 October", Listing(fixtures), str, NOW, stats_source=stats, on_result=gates.append,
                                     analyst=analyst))
    assert stats.asked == ["Lakers"]
    assert chunks[0] == "NO VALID PLAY - PASS" and "- 1 game(s) skipped: not open for betting on SportyBet" in chunks[1]


def test_america_drops_games_that_are_not_open_before_stats_and_analyst_and_counts_them():
    fixtures = [game("Chiefs", "Bills", "USA - NFL", "american_football"),
                game("Yankees", "Red Sox", "USA - MLB", "baseball", markets=()),
                game("Bruins", "Rangers", "USA - NHL", "hockey", markets=()),
                game("Moscow", "Minsk", "Russia - KHL", "hockey", markets=())]
    stats, gates, analyst = Table(lambda f: insufficient(f.match_id, "no stats here")), [], FakeAnalyst()
    chunks = asyncio.run(run_america("1 October", Listing(fixtures), str, NOW, stats_source=stats, on_result=gates.append,
                                     analyst=analyst))
    assert stats.asked == ["Chiefs"] and [q.home for q in analyst.queries] == ["Chiefs"]
    assert "- 2 game(s) skipped: not open for betting on SportyBet" in chunks[1]
    assert "- 1 game(s) ignored: outside your allowed competitions" in chunks[1]
    assert gates[0].footer[:2] == ("1 game(s) ignored: outside your allowed competitions.",
                                   "2 game(s) skipped: not open for betting on SportyBet.")


def test_when_nothing_is_open_no_stats_and_no_analyst_call_happen_in_any_trigger():
    for runner, sport, comp in ((None, "soccer", "England - Premier League"), (run_gameday, "basketball", "USA - NBA"),
                                (run_america, "baseball", "USA - MLB")):
        fixtures = [game("A", "B", comp, sport, markets=()), game("C", "D", comp, sport, markets=())]
        stats, analyst = Table(good_stats), FakeAnalyst()
        if runner is None:
            chunks, gate, stats = matchday(fixtures, analyst=analyst)
        else:
            chunks = asyncio.run(runner("1 October", Listing(fixtures), str, NOW, stats_source=stats, analyst=analyst))
        assert stats.asked == [] and analyst.queries == []
        assert "2 game(s) skipped: not open for betting on SportyBet." in "\n".join(chunks)


def test_the_daily_analyst_limit_counts_only_games_that_were_open(tmp_path):
    with EvaluationStore.open(tmp_path / "t.db", str) as store:
        fixtures = [game("H1", "A1", "Netherlands - Eredivisie"),
                    game("H2", "A2", "Netherlands - Eredivisie", markets=()),
                    game("H3", "A3", "Netherlands - Eredivisie", markets=())]
        analyst = FakeAnalyst()
        matchday(fixtures, analyst=analyst, budget=DailyBudget(store, 40, NOW))
        assert len(analyst.queries) == 1 and DailyBudget(store, 40, NOW).used() == 1
