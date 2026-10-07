"""Spec 11-1: the five play types Matchday considers, the best two per game, the per-game corners/cards market check,
the football-data wiring and the loader mapping of the new play types. Fabricated or recorded samples only: no network."""
import asyncio
import inspect
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from fakes_bookmaker import make_fixture, pick_evaluation, totals_offer
from listview import list_text, started
from predictions.adapters.footballdata import stats as fd
from predictions.adapters.footballdata.stats import FD_ALIASES, LEAGUES, FootballDataInputs
from predictions.adapters.sportybet import event as event_adapter
from predictions.adapters.sportybet.betslip import SELECTORS, SportyBetBookmaker
from predictions.adapters.sportybet.event import SportyBetEventMarkets, parse_event_markets
from predictions.adapters.sportybet.fixtures import MARKET_IDS, _markets
from predictions.core.betting import (Availability, Leg, UnsupportedLeg, assess_availability, find_outcome,
                                      leg_from_evaluation)
from predictions.core.engines.soccer import (MATCHDAY_MAX_PLAYS_PER_GAME, SoccerEngine, result_probabilities,
                                             scaled_rates)
from predictions.core.identity import resolve_team
from predictions.core.inputs import Input
from predictions.core.market_rules import is_under
from predictions.core.model import PROBABILITY_FLOOR, Estimate, Evaluation, validate
from predictions.core.play_labels import play_label
from predictions.core.stats import READY, FixtureStats, insufficient
from predictions.core.types import Fixture, MarketOffer, MatchId, Outcome
from predictions.flow.evaluate import _best_per_game, run_gate
from test_corners_cards import extra
from test_soccer_optimization import (EVEN, KICKOFF, NOW, TOP, fixture, make_inputs, matchday, matchday_gate, one_x_two,
                                      ready, total_offer)

DATA = Path(__file__).parent / "data"
STRONG = make_inputs(3.4, 0.8, 1.3, 2.3)    # home wins 74.4%, 2.5+ goals 74.6% (clears the 70% floor)


def offer(market_id, name, line, over, under="9.00"):
    return MarketOffer(market_id, name, f"total={line}",
                       (Outcome(f"Over {line}", Decimal(over)), Outcome(f"Under {line}", Decimal(under))))


def corners(line, over):
    return offer("166", "Corners - Over/Under", line, over)


def cards(line, over):
    return offer("139", "Bookings - Over/Under", line, over)


def double_chance(home_draw="1.25", home_away="1.20", draw_away="2.60"):
    return MarketOffer("10", "Double Chance", "", (Outcome("Home or Draw", Decimal(home_draw), "9"),
                                                   Outcome("Home or Away", Decimal(home_away), "10"),
                                                   Outcome("Draw or Away", Decimal(draw_away), "11")))


def evaluate(f, inputs):
    return SoccerEngine().evaluate(f, inputs, f.markets)


def by_selection(evals):
    return {e.selection: e for e in evals}


def all_five():
    return fixture([one_x_two("1.80", "4.00", "5.00"), double_chance(), total_offer("1.5", "1.30", "3.20"),
                    total_offer("2.5", "1.90", "1.90"), corners("7.5", "1.40"), cards("2.5", "1.39")], competition=TOP)


# ------------------------------------------------------------------ (a) to (e): each play type priced and gated
def test_each_of_the_five_play_types_is_priced_and_passes_the_usual_gate():
    found = by_selection(evaluate(all_five(), STRONG + extra()))
    assert set(found) >= {"Over 1.5", "Over 2.5", "Home", "Home or Draw", "Corners Over 7.5", "Cards Over 2.5"}
    for selection in ("Over 1.5", "Over 2.5", "Home", "Home or Draw", "Corners Over 7.5", "Cards Over 2.5"):
        e = found[selection]
        assert e.passed and e.estimate.probability >= PROBABILITY_FLOOR and e.ev > 0 and e.estimate.inputs
    assert found["Home or Draw"].market == "Double Chance" and found["Home"].market == "1X2"
    assert found["Corners Over 7.5"].estimate.model == "negbin-corners-v1"
    assert found["Cards Over 2.5"].estimate.model == "negbin-cards-v1"


def test_double_chance_is_priced_from_the_same_grid_as_the_1x2():
    home, away = scaled_rates(STRONG)
    win_h, draw, win_a = result_probabilities(home, away)
    found = by_selection(evaluate(all_five(), STRONG + extra()))
    assert found["Home or Draw"].estimate.probability == pytest.approx(win_h + draw, abs=0.002)
    assert found["Away or Draw"].estimate.probability == pytest.approx(win_a + draw, abs=0.002)
    assert found["Home"].estimate.probability == pytest.approx(win_h, abs=0.002)
    assert found["Home"].estimate.probability < found["Home or Draw"].estimate.probability


@pytest.mark.parametrize("markets,inputs,selection,why", [
    # below the 70% floor but with positive EV: still rejected
    ([total_offer("2.5", "1.90", "1.90")], EVEN, "Over 2.5", "below the 70% floor"),
    ([one_x_two("3.00", "3.40", "2.60")], EVEN, "Home", "below the 70% floor"),
    ([double_chance("1.80", "1.40", "1.90")], EVEN, "Home or Draw", "below the 70% floor"),
    ([corners("9.5", "2.00")], EVEN + extra(), "Corners Over 9.5", "below the 70% floor"),
    ([cards("4.5", "2.50")], EVEN + extra(), "Cards Over 4.5", "below the 70% floor"),
    # above the floor but with negative EV (odds too short): still rejected
    ([total_offer("1.5", "1.10", "6.00")], EVEN, "Over 1.5", "expected value"),
    ([one_x_two("1.30", "4.00", "9.00")], STRONG, "Home", "expected value"),
    ([double_chance("1.10", "1.10", "4.00")], STRONG, "Home or Draw", "expected value"),
    ([corners("7.5", "1.10")], EVEN + extra(), "Corners Over 7.5", "expected value"),
    ([cards("2.5", "1.10")], EVEN + extra(), "Cards Over 2.5", "expected value"),
])
def test_floor_and_positive_ev_apply_to_every_play_type(markets, inputs, selection, why):
    e = by_selection(evaluate(fixture(markets, competition=TOP), inputs))[selection]
    assert not e.passed and why in e.reason and not e.insufficient


def test_the_floor_is_70_percent_inclusive_and_the_estimate_is_floored_not_rounded_up():
    assert PROBABILITY_FLOOR == 0.70
    assert validate(Estimate(0.699999), Decimal("2.00"))[0] is False   # rounds DOWN to 69.9%
    assert validate(Estimate(0.700), Decimal("2.00"))[0] is True
    assert validate(Estimate(0.700), Decimal("1.40"))[0] is False      # EV 0.70 * 1.4 - 1 < 0


def goals_only(home=(3.0, 0.8), away=(0.8, 2.0)):
    out = []
    for side, (goals_for, goals_against) in (("home", home), ("away", away)):
        out += [Input(f"{side}.matches", 10.0, "matches", "t"), Input(f"{side}.goals_for", goals_for, "goals per match", "t"),
                Input(f"{side}.goals_against", goals_against, "goals per match", "t")]
    return out


def test_the_goals_only_ev_cap_applies_to_double_chance_and_match_winner_too():
    f = fixture([double_chance("2.00", "1.30", "3.00"), one_x_two("1.20", "9.00", "20.00")], competition=TOP)
    found = by_selection(evaluate(f, goals_only()))
    capped = found["Home or Draw"]
    assert capped.estimate.model == "goals-only-poisson-v1" and capped.estimate.probability >= PROBABILITY_FLOOR
    assert capped.ev > Decimal("0.20") and not capped.passed and "treated as unreliable" in capped.reason
    fine = by_selection(evaluate(fixture([double_chance("1.40", "1.30", "3.00")], competition=TOP), goals_only()))
    assert fine["Home or Draw"].passed and fine["Home or Draw"].ev <= Decimal("0.20")


# ------------------------------------------------------------------ what is NOT considered
def excluded_fixture():
    both = MarketOffer("29", "GG/NG", "", (Outcome("Yes", Decimal("1.60")), Outcome("No", Decimal("2.20"))))
    handicap = MarketOffer("16", "Handicap", "hcp=-0.5", (Outcome("Home -0.5", Decimal("1.50")),
                                                         Outcome("Away +0.5", Decimal("2.40"))))
    first_half = offer("68", "1st Half - Over/Under", "1.5", "1.60")
    markets = [both, handicap, first_half,
               offer("19", "Home O/U", "1.5", "1.30"), offer("20", "Away O/U", "0.5", "1.30"),
               total_offer("0.5", "1.05", "9.00"), total_offer("1", "1.12", "5.00"), total_offer("1.5", "1.30", "3.20"),
               one_x_two("1.80", "1.50", "5.00"), double_chance("1.25", "1.20", "1.20"),
               corners("5.5", "1.10"), corners("6.5", "1.20"), corners("7.5", "1.40"), cards("1.5", "1.12"),
               cards("2.5", "1.39"),
               MarketOffer("162", "Corners 1X2", "", (Outcome("Home", Decimal("1.3")), Outcome("Draw", Decimal("9")),
                                                       Outcome("Away", Decimal("5")))),
               offer("900300", "Home Team Total Corners", "4.5", "1.21"), offer("177", "1st Half - Total Corners", "3.5", "1.5"),
               offer("138", "Total Booking Points", "25.5", "1.39"), offer("900304", "Home Team Total Bookings", "1.5", "2.5"),
               MarketOffer("165", "Corner Handicap", "hcp=-2.5", (Outcome("Home (-2.5)", Decimal("1.4")),
                                                                  Outcome("Away (+2.5)", Decimal("2.6")))),
               MarketOffer("11", "Draw No Bet", "", (Outcome("Home", Decimal("1.2")), Outcome("Away", Decimal("4.0"))))]
    return fixture(markets, competition=TOP)


def test_only_the_five_play_types_are_ever_produced():
    evals = evaluate(excluded_fixture(), STRONG + extra())
    assert evals
    markets = {e.market for e in evals}
    assert markets == {"Over/Under 1.5", "1X2", "Double Chance", "Corners 7.5", "Cards 2.5"}
    selections = {e.selection for e in evals}
    assert selections == {"Over 1.5", "Home", "Away", "Home or Draw", "Away or Draw", "Corners Over 7.5", "Cards Over 2.5"}
    assert not any(is_under(e) for e in evals)
    for banned in ("Draw", "Yes", "No", "Home or Away", "Over 0.5", "Over 1.0"):
        assert banned not in selections
    assert not any("team" in s.lower() or "half" in s.lower() for s in selections)


def test_match_winner_and_double_chance_are_only_ever_home_or_away_sides():
    evals = evaluate(all_five(), STRONG)
    winners = {e.selection for e in evals if e.market == "1X2"}
    chances = {e.selection for e in evals if e.market == "Double Chance"}
    assert winners == {"Home", "Away"} and chances == {"Home or Draw", "Away or Draw"}


@pytest.mark.parametrize("line,present", [("1.5", True), ("1", False), ("0.5", False), ("2", False), ("3.5", True)])
def test_goal_lines_below_one_and_a_half_are_not_considered(line, present):
    f = fixture([total_offer(line, "1.60", "2.20")], competition=TOP)
    assert bool([e for e in evaluate(f, EVEN) if e.market.startswith("Over/Under")]) is present


@pytest.mark.parametrize("line,present", [("6.5", False), ("7", False), ("8", False), ("7.5", True), ("9.5", True)])
def test_corners_lines_below_seven_and_a_half_are_not_considered(line, present):
    f = fixture([corners(line, "1.60")], competition=TOP)
    assert bool(evaluate(f, EVEN + extra())) is present


@pytest.mark.parametrize("line,present", [("1.5", False), ("2", False), ("3", False), ("2.5", True), ("5.5", True)])
def test_cards_lines_below_two_and_a_half_are_not_considered(line, present):
    f = fixture([cards(line, "1.60")], competition=TOP)
    assert bool(evaluate(f, EVEN + extra())) is present


def test_a_game_without_corner_or_card_data_does_not_price_them_and_is_counted_not_listed():
    f = fixture([corners("7.5", "1.40"), cards("2.5", "1.39"), total_offer("1.5", "1.30", "3.20")], competition=TOP)
    gate = run_gate([f], {f.match_id: FixtureStats(f.match_id, READY, tuple(STRONG))})
    assert [e.selection for e in gate.passed] == ["Over 1.5"]
    assert gate.rejected == () and gate.unpriced_games == (f.match_id,)


# ------------------------------------------------------------------ the best two per game
def test_only_the_two_best_plays_by_probability_are_kept_and_the_best_is_advised():
    f = all_five()
    gate = run_gate([f], {f.match_id: FixtureStats(f.match_id, READY, tuple(STRONG + extra()))})
    assert MATCHDAY_MAX_PLAYS_PER_GAME == 2
    assert [e.selection for e in gate.passed] == ["Over 1.5", "Home or Draw"]   # 90.0% then 88.8% (of six that pass)
    assert gate.passed[0].estimate.probability > gate.passed[1].estimate.probability
    text = list_text(gate.passed)
    assert "1a" in text and "1b" in text and "1c" not in text
    assert "Over 1.5 goals @ 1.30 - 90.0% (advised)" in text and "Home or Draw @ 1.25 - 88.8%" in text
    session = started(gate.passed).active
    game = session.games()[0]
    assert [o.evaluation.selection for o in game.options] == ["Over 1.5", "Home or Draw"] and game.chosen.pick_id == game.options[0].pick_id


def test_one_play_is_shown_when_only_one_passes():
    f = fixture([total_offer("1.5", "1.30", "3.20"), total_offer("2.5", "1.90", "1.90")], competition=TOP)
    gate = run_gate([f], {f.match_id: FixtureStats(f.match_id, READY, tuple(make_inputs(2.0, 1.0, 2.0, 1.0)))})
    assert [e.selection for e in gate.passed] == ["Over 1.5"]
    assert "1a" not in list_text(gate.passed)


def _ev(selection, prob, ev, f):
    return Evaluation(f.match_id, "Double Chance", selection, Decimal("1.50"), Estimate(prob, (), "m"), Decimal(ev), True,
                      "passed")


def test_ranking_ties_go_to_the_higher_ev_then_the_selection_text():
    f = all_five()
    a, b, c, d = (_ev("Zeta", 0.80, "0.10", f), _ev("Alpha", 0.80, "0.10", f), _ev("Mid", 0.80, "0.20", f),
                  _ev("Low", 0.70, "0.90", f))
    kept = _best_per_game([a, b, c, d], [f])
    assert [e.selection for e in sorted(kept, key=lambda e: (-e.estimate.probability, -e.ev, e.selection))] == ["Mid", "Alpha"]
    assert len(kept) == 2 and d not in kept
    advised = [p.evaluation.selection for p in started(sorted(kept, key=lambda e: e.selection)).active.games()[0].options]
    assert advised == ["Mid", "Alpha"]      # the list advises in the same order


def test_the_limit_is_per_game_and_other_sports_are_untouched():
    f1, f2 = all_five(), fixture([total_offer("1.5", "1.30", "3.20")], competition=TOP, home="Gamma", away="Delta")
    plays = [_ev(s, 0.9 - i / 100, "0.1", f1) for i, s in enumerate(("A", "B", "C"))] + [_ev("X", 0.8, "0.1", f2)]
    kept = _best_per_game(plays, [f1, f2])
    assert sorted(e.selection for e in kept) == ["A", "B", "X"]
    basketball = MatchId("basketball", "USA - NBA", KICKOFF, "H", "A")
    bb = Fixture(basketball, "USA - NBA", "H", "A", KICKOFF, False)
    many = [Evaluation(basketball, "m", str(i), Decimal("1.5"), Estimate(0.7, (), "m"), Decimal("0.1"), True, "passed")
            for i in range(4)]
    assert _best_per_game(many, [bb]) == many


# ------------------------------------------------------------------ the per-game market check (flow)
class FakeEvents:
    """Stands in for the per-game lookup: `table` maps match_id -> offers (or None for a failed lookup)."""

    def __init__(self, table=None, fail_all=False):
        self.table, self.fail_all, self.asked, self.calls = table or {}, fail_all, [], 0

    async def fetch_event_markets(self, fixtures):
        self.calls += 1
        self.asked = [f.match_id for f in fixtures]
        return {f.match_id: (None if self.fail_all else self.table.get(f.match_id, ())) for f in fixtures}


def league_game(home, away, markets=(), competition=TOP):
    f = fixture([total_offer("1.5", "1.30", "3.20"), *markets], competition=competition, home=home, away=away)
    return f


def listed_corners_and_cards():
    return (corners("6.5", "1.25"), corners("7.5", "1.40"), corners("8.5", "1.60"), corners("9.5", "2.05"),
            cards("1.5", "1.12"), cards("2.5", "1.39"), cards("3.5", "1.92"), cards("4.5", "2.90"))


def test_only_games_that_list_the_markets_get_corners_and_cards_plays_priced_at_the_listed_lines_and_odds():
    with_markets, without = league_game("Gamma", "Delta"), league_game("Eps", "Zeta")
    events = FakeEvents({with_markets.match_id: listed_corners_and_cards()})
    chunks, gate = matchday_gate([with_markets, without], lambda f: ready(f, STRONG + extra()),
                                 event_markets_source=events)
    assert chunks == [] and events.calls == 1 and set(events.asked) == {with_markets.match_id, without.match_id}
    merged = gate.fixtures[with_markets.match_id]       # the game now carries the listed markets; the other one does not
    assert {m.market_id for m in merged.markets} >= {"166", "139"}
    assert not {m.market_id for m in gate.fixtures[without.match_id].markets} & {"166", "139"}
    priced = evaluate(merged, STRONG + extra())
    priced = [e for e in priced if e.market.startswith(("Corners", "Cards"))]
    assert not [e for e in evaluate(gate.fixtures[without.match_id], STRONG + extra()) if e.market.startswith(("Corners", "Cards"))]
    assert {e.match_id for e in gate.passed} == {with_markets.match_id, without.match_id}
    assert len([e for e in gate.passed if e.match_id == with_markets.match_id]) == 2     # the best two of everything that passes
    lines = {e.selection for e in priced}
    assert lines == {"Corners Over 7.5", "Corners Over 8.5", "Corners Over 9.5",
                     "Cards Over 2.5", "Cards Over 3.5", "Cards Over 4.5"}      # exactly the listed lines >= 7.5 / >= 2.5
    odds = {e.selection: e.odds for e in priced}
    assert odds["Corners Over 8.5"] == Decimal("1.60") and odds["Cards Over 3.5"] == Decimal("1.92")   # the real listed odds
    assert not any(e.match_id == without.match_id for e in priced)


def test_the_lookup_is_only_for_ready_games_and_is_not_asked_when_nothing_is_ready():
    ok, bad = league_game("Gamma", "Delta"), league_game("Eps", "Zeta")
    events = FakeEvents()
    matchday([ok, bad], lambda f: ready(f, STRONG) if f is ok else insufficient(f.match_id, "no data"),
             event_markets_source=events)
    assert events.asked == [ok.match_id]
    nothing = FakeEvents()
    matchday([bad], lambda f: insufficient(f.match_id, "no data"), event_markets_source=nothing)
    assert nothing.calls == 0


def test_a_failed_lookup_keeps_the_game_on_its_own_markets_and_adds_the_warning_line():
    g = league_game("Gamma", "Delta")
    chunks, gate = matchday_gate([g], lambda f: ready(f, STRONG + extra()),
                                 event_markets_source=FakeEvents(fail_all=True))
    assert chunks == [] and [e.selection for e in gate.passed] == ["Over 1.5"]   # still evaluated on what it had
    assert not any(e.market.startswith(("Corners", "Cards")) for e in (*gate.passed, *gate.rejected))
    assert any(line.startswith("Warning: SportyBet's per-game markets could not be read for 1 game(s)") for line in gate.footer)


def test_games_listing_corners_without_football_data_are_counted_in_the_footer_not_listed():
    g = league_game("Gamma", "Delta")
    events = FakeEvents({g.match_id: listed_corners_and_cards()})
    chunks, gate = matchday_gate([g], lambda f: ready(f, STRONG), event_markets_source=events)   # no corner/card inputs
    assert [e.selection for e in gate.passed] == ["Over 1.5"] and gate.rejected == ()
    assert "Corners/cards not available for 1 game(s)" in " ".join(gate.footer)


def test_without_a_lookup_source_nothing_changes():
    g = league_game("Gamma", "Delta", [corners("7.5", "1.40")])
    chunks, gate = matchday_gate([g], lambda f: ready(f, STRONG + extra()))
    assert any(e.selection == "Corners Over 7.5" for e in gate.passed)   # a game that already carries the market is priced
    assert all("Warning" not in line for line in gate.footer)


# ------------------------------------------------------------------ the event adapter (recorded samples)
EVENT = json.loads((DATA / "sportybet_event_sample.json").read_text(encoding="utf-8"))
EVENT_NONE = json.loads((DATA / "sportybet_event_no_corners_sample.json").read_text(encoding="utf-8"))


def test_the_recorded_event_gives_exactly_the_corners_and_cards_markets_with_real_lines_and_odds():
    offers = parse_event_markets(EVENT)
    assert {o.market_id for o in offers} == {"166", "139"}
    corner_lines = {o.specifier: o for o in offers if o.market_id == "166"}
    assert set(corner_lines) == {f"total={x}" for x in ("7.5", "8.5", "9.5", "10.5", "11.5")}
    assert corner_lines["total=9.5"].name == "Corners - Over/Under"
    assert {x.name: x.odds for x in corner_lines["total=9.5"].outcomes} == {"Over 9.5": Decimal("2.10"),
                                                                           "Under 9.5": Decimal("1.71")}
    card_lines = {o.specifier for o in offers if o.market_id == "139"}
    assert card_lines == {f"total={x}" for x in ("1.5", "2.5", "3.5", "4.5", "5.5")}
    # the markets that look similar are not taken: corners 1X2, team corners, half corners, handicaps, booking points
    assert not {o.market_id for o in offers} & {"162", "900300", "177", "165", "138"}


def test_a_league_event_without_the_markets_gives_none_and_a_bad_answer_is_unreadable():
    assert parse_event_markets(EVENT_NONE) == ()
    assert parse_event_markets({"bizCode": 19000, "data": {}}) is None
    assert parse_event_markets("junk") is None
    assert parse_event_markets({"bizCode": 10000, "data": {"markets": "x"}}) is None


def two_fixtures():
    a = make_fixture("Alpha", "Beta", [], ref="sr:match:1")
    b = make_fixture("Gamma", "Delta", [], ref="sr:match:2")
    c = make_fixture("Eps", "Zeta", [], ref="sr:match:3")
    return a, b, c


def run(coro):
    return asyncio.run(coro)


def test_the_adapter_asks_once_per_game_with_bounded_concurrency_and_marks_failures():
    a, b, c = two_fixtures()
    seen, active, peak = [], [0], [0]

    async def handler(request: httpx.Request):
        ref = request.url.params["eventId"]
        seen.append((str(request.url).split("?")[0], ref, request.url.params["productId"]))
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        await asyncio.sleep(0.01)
        active[0] -= 1
        if ref == "sr:match:3":
            return httpx.Response(500)
        return httpx.Response(200, json=EVENT if ref == "sr:match:1" else EVENT_NONE)

    source = SportyBetEventMarkets(httpx.AsyncClient(transport=httpx.MockTransport(handler)), concurrency=2)
    found = run(source.fetch_event_markets([a, b, c]))
    assert sorted(ref for _, ref, _ in seen) == ["sr:match:1", "sr:match:2", "sr:match:3"]      # once per game
    assert all(url == event_adapter.EVENT_URL and pid == "3" for url, _, pid in seen) and peak[0] <= 2
    assert {o.market_id for o in found[a.match_id]} == {"166", "139"}
    assert found[b.match_id] == () and found[c.match_id] is None                                  # none listed / failed


def test_unreadable_answers_and_missing_event_ids_are_failures_not_errors():
    a = make_fixture("Alpha", "Beta", [], ref="sr:match:1")
    no_ref = Fixture(MatchId("soccer", "T", KICKOFF, "N", "R"), "T", "N", "R", KICKOFF, False, (), "")

    async def handler(request):
        return httpx.Response(200, text="not json")

    source = SportyBetEventMarkets(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    found = run(source.fetch_event_markets([a, no_ref]))
    assert found == {a.match_id: None, no_ref.match_id: None}
    assert run(source.fetch_event_markets([])) == {}


def test_the_overall_time_limit_leaves_the_games_not_reached_as_failures():
    a, b, _ = two_fixtures()

    async def handler(request):
        await asyncio.sleep(5)
        return httpx.Response(200, json=EVENT)

    source = SportyBetEventMarkets(httpx.AsyncClient(transport=httpx.MockTransport(handler)), overall_timeout=0.05)
    assert run(source.fetch_event_markets([a, b])) == {a.match_id: None, b.match_id: None}


def test_the_listing_asks_for_the_double_chance_but_not_for_corners_and_cards():
    assert "10" in MARKET_IDS.split(",") and not {"166", "139"} & set(MARKET_IDS.split(","))


# ------------------------------------------------------------------ football-data wiring and aliases
def test_app_wires_football_data_and_the_event_lookup_into_matchday_only():
    import app as bot
    source = inspect.getsource(bot)
    matchday_call = source[source.index("run_matchday("):source.index("run_matchday(") + 900]
    assert "extra_inputs_source=FootballDataInputs()" in matchday_call and "event_markets_source=" in matchday_call
    assert "FootballDataInputs" not in source[source.index("run_gameday("):source.index("run_gameday(") + 600]
    assert "FootballDataInputs" not in source[source.index("run_america("):source.index("run_america(") + 600]


def synthetic_season(home, away):
    rows = ["Div,Date,Time,HomeTeam,AwayTeam,Referee,HC,AC,HY,AY,HR,AR"]
    for day in range(1, 7):
        rows.append(f"E0,0{day}/09/2026,15:00,{home},{away},R One,6,4,2,3,1,0")
        rows.append(f"E0,1{day}/09/2026,15:00,{away},{home},R One,5,7,3,2,0,1")
    return "\n".join(rows) + "\n"


def run_adapter(f, season):
    def handler(request):
        url = str(request.url)
        if url.endswith("fixtures.csv"):
            return httpx.Response(200, text="Div,Date,Time,HomeTeam,AwayTeam,Referee\n")
        return httpx.Response(200, text=season if "2627" in url else "Div,Date,HomeTeam,AwayTeam\n")
    return run(FootballDataInputs(httpx.AsyncClient(transport=httpx.MockTransport(handler))).fetch_inputs([f]))


# (SportyBet's spelling, football-data's spelling): the clubs that did not resolve in the 2026-10-01 alias probe
ALIAS_CASES = [
    ("England - Premier League", "Man Utd", "Hull City", "Man United", "Hull"),
    ("England - Premier League", "Coventry City", "Ipswich Town", "Coventry", "Ipswich"),
    ("Spain - LaLiga", "Espanyol", "RC Deportivo de A Coruna", "Espanol", "La Coruna"),
    ("Spain - LaLiga", "Racing Santander", "Athletic Bilbao", "Santander", "Ath Bilbao"),
    ("Germany - Bundesliga", "Cologne", "Hamburger SV", "FC Koln", "Hamburg"),
    ("Germany - Bundesliga", "Schalke", "SV 07 Elversberg", "Schalke 04", "Elversberg"),
    ("Italy - Serie A", "Como 1907", "Inter", "Como", "Inter"),
    ("France - Ligue 1", "AJ Auxerre", "Paris Saint-Germain", "Auxerre", "Paris SG"),
    ("Netherlands - Eredivisie", "Alkmaar", "Fortuna Sittard", "AZ Alkmaar", "For Sittard"),
    ("Portugal - Liga Portugal", "Sporting", "Braga", "Sp Lisbon", "Sp Braga"),
    ("Scotland - Premiership", "Heart of Midlothian FC", "Dundee FC", "Hearts", "Dundee"),
    ("Greece - Super League", "Olympiacos", "Aris Thessaloniki", "Olympiakos", "Aris"),
    ("Turkiye - Super Lig", "Istanbul BB", "Besiktas Istanbul", "Buyuksehyr", "Besiktas"),
    ("Belgium - Pro League", "Union Gilloise", "KV Mechelen", "St. Gilloise", "Mechelen"),
]


@pytest.mark.parametrize("competition,home,away,fd_home,fd_away", ALIAS_CASES)
def test_sportybet_club_names_resolve_to_football_data_names(competition, home, away, fd_home, fd_away):
    teams = {fd_home, fd_away, "Other One", "Other Two"}
    assert resolve_team(home, teams, FD_ALIASES).team == fd_home
    assert resolve_team(away, teams, FD_ALIASES).team == fd_away
    f = Fixture(MatchId("soccer", competition, KICKOFF, home, away), competition, home, away, KICKOFF, False)
    found = run_adapter(f, synthetic_season(fd_home, fd_away))
    names = {i.name for i in found[f.match_id]}
    assert {"home.corners_for", "away.corners_against", "home.cards_for"} <= names


def test_turkey_and_belgium_competitions_are_covered_under_sportybets_spelling():
    assert LEAGUES["turkiye super lig"] == "T1" and LEAGUES["belgium pro league"] == "B1"


def test_team_card_rates_count_yellow_cards_only():
    f = Fixture(MatchId("soccer", "England - Premier League", KICKOFF, "Man Utd", "Hull City"),
                "England - Premier League", "Man Utd", "Hull City", KICKOFF, False)
    found = {i.name: i.value for i in run_adapter(f, synthetic_season("Man United", "Hull"))[f.match_id]}
    assert found["home.cards_for"] == 2.0 and found["away.cards_for"] == 3.0     # yellow cards per game, reds ignored


# ------------------------------------------------------------------ labels
@pytest.mark.parametrize("market,selection,text", [
    ("Over/Under 1.5", "Over 1.5", "Over 1.5 goals"), ("Over/Under 2.0", "Over 2.0", "Over 2.0 goals"),
    ("1X2", "Home", "Home win"), ("1X2", "Away", "Away win"),
    ("Double Chance", "Home or Draw", "Home or Draw"), ("Double Chance", "Away or Draw", "Away or Draw"),
    ("Corners 9.5", "Corners Over 9.5", "Over 9.5 corners"), ("Cards 3.5", "Cards Over 3.5", "Over 3.5 yellow cards"),
])
def test_plays_are_labelled_readably(market, selection, text):
    ev = Evaluation(MatchId("soccer", "T", KICKOFF, "A", "B"), market, selection, Decimal("1.5"), Estimate(0.7), Decimal("0.1"),
                    True, "passed")
    assert play_label(ev) == text


def test_other_sports_and_unknown_markets_keep_their_own_text():
    mid = MatchId("basketball", "T", KICKOFF, "A", "B")
    ev = Evaluation(mid, "Total points 160.5", "Over 160.5", Decimal("1.5"), Estimate(0.7), Decimal("0.1"), True, "passed")
    assert play_label(ev) == "Over 160.5"
    soccer = Evaluation(MatchId("soccer", "T", KICKOFF, "A", "B"), "Home team total 1.5", "Home team Over 1.5", Decimal("1.5"),
                        Estimate(0.7), Decimal("0.1"), True, "passed")
    assert play_label(soccer) == "Home team Over 1.5"


def test_the_list_shows_the_readable_labels_with_odds_percentage_and_tags():
    f = all_five()
    gate = run_gate([f], {f.match_id: FixtureStats(f.match_id, READY, tuple(STRONG + extra()))})
    text = list_text(gate.passed)
    assert "Over 1.5 goals @ 1.30 - 90.0%" in text and "Home or Draw @ 1.25 - 88.8%" in text


# ------------------------------------------------------------------ the loader mapping
def soccer_eval(market, selection, odds="1.50"):
    return Evaluation(MatchId("soccer", "Testland - League", KICKOFF, "Alpha", "Beta"), market, selection, Decimal(odds),
                      Estimate(0.8), Decimal("0.1"), True, "passed")


@pytest.mark.parametrize("market,selection,market_id,specifier,side", [
    ("1X2", "Home", "1", "", "home"), ("1X2", "Away", "1", "", "away"),
    ("Double Chance", "Home or Draw", "10", "", "home or draw"), ("Double Chance", "Away or Draw", "10", "", "draw or away"),
    ("Corners 9.5", "Corners Over 9.5", "166", "total=9.5", "over"),
    ("Cards 3.5", "Cards Over 3.5", "139", "total=3.5", "over"),
    ("Over/Under 2.0", "Over 2.0", "18", "total=2", "over"),
])
def test_every_new_play_type_maps_to_a_bookmaker_leg(market, selection, market_id, specifier, side):
    leg = leg_from_evaluation(soccer_eval(market, selection))
    assert (leg.market_id, leg.specifier, leg.side) == (market_id, specifier, side)


@pytest.mark.parametrize("market,selection", [
    ("Corners 9.5", "Corners Over 8.5"), ("Cards 3.5", "Cards Over 2.5"), ("Double Chance", "Home or Home"),
    ("Corners 9.5", "Over 9.5"), ("Bookings 3.5", "Over 3.5"), ("Cards 3.5", "Corners Over 3.5"),
])
def test_a_selection_that_does_not_match_its_market_is_refused(market, selection):
    with pytest.raises(UnsupportedLeg):
        leg_from_evaluation(soccer_eval(market, selection))


def test_unders_of_the_new_markets_are_still_unders():
    assert is_under(leg_from_evaluation(soccer_eval("Corners 9.5", "Corners Under 9.5")))
    assert is_under(leg_from_evaluation(soccer_eval("Cards 3.5", "Cards Under 3.5")))


def listing_with(*offers):
    return make_fixture("Alpha", "Beta", offers, ref="sr:match:77")


def test_each_leg_finds_its_listed_option_and_a_missing_line_or_market_is_reported_missing():
    corner_offers = tuple(o for o in parse_event_markets(EVENT) if o.market_id == "166")
    f = listing_with(double_chance(), one_x_two("1.8", "4", "5"), *corner_offers)
    f = Fixture(f.match_id, f.competition, "Alpha", "Beta", f.kickoff_utc, False, f.markets, "sr:match:77")
    legs = [leg_from_evaluation(soccer_eval(m, s)) for m, s in (("Double Chance", "Away or Draw"), ("1X2", "Home"),
                                                                 ("Corners 9.5", "Corners Over 9.5"),
                                                                 ("Corners 6.5", "Corners Over 6.5"),
                                                                 ("Cards 2.5", "Cards Over 2.5"))]
    out = assess_availability(legs, [f], NOW)
    assert [a.status for a in out] == [Availability.OK, Availability.OK, Availability.OK, Availability.OPTION_MISSING,
                                       Availability.OPTION_MISSING]
    assert out[0].outcome.name == "Draw or Away" and out[2].outcome.name == "Over 9.5"
    assert find_outcome(f.markets, legs[0])[1].odds == Decimal("2.60")


class Listing:
    def __init__(self, fixtures):
        self.fixtures = fixtures

    async def fetch_fixtures(self):
        return self.fixtures


def test_the_bookmaker_looks_the_corners_and_cards_up_per_game_for_those_legs_only():
    f = listing_with(one_x_two("1.8", "4", "5"), totals_offer("2.5", "1.85", "2.00"))
    events = FakeEvents({f.match_id: parse_event_markets(EVENT)})
    bm = SportyBetBookmaker(None, True, Listing([f]), clock=lambda: NOW, event_markets_source=events)
    corner = leg_from_evaluation(soccer_eval("Corners 9.5", "Corners Over 9.5"))
    card = leg_from_evaluation(soccer_eval("Cards 2.5", "Cards Over 2.5"))
    winner = leg_from_evaluation(soccer_eval("1X2", "Home"))
    out = run(bm.check_availability([corner, card]))
    assert [a.status for a in out] == [Availability.OK, Availability.OK] and events.calls == 1
    assert out[0].outcome.odds == Decimal("2.10") and out[1].outcome.odds == Decimal("1.39")   # the real listed prices
    events.calls = 0
    assert run(bm.check_availability([winner]))[0].status is Availability.OK and events.calls == 0   # no lookup needed


def test_a_failed_lookup_makes_the_corner_option_missing_never_guessed():
    f = listing_with(one_x_two("1.8", "4", "5"))
    bm = SportyBetBookmaker(None, True, Listing([f]), clock=lambda: NOW, event_markets_source=FakeEvents(fail_all=True))
    out = run(bm.check_availability([leg_from_evaluation(soccer_eval("Corners 9.5", "Corners Over 9.5"))]))
    assert out[0].status is Availability.OPTION_MISSING


def test_game_page_titles_for_the_new_markets_are_the_verified_anonymous_ones():
    assert (SELECTORS.corners_title, SELECTORS.cards_title, SELECTORS.double_chance_title) == (
        "Corners - Over/Under", "Bookings - Over/Under", "Double Chance")
    f = listing_with(double_chance(), corners("9.5", "2.0"), cards("2.5", "1.4"))
    name = SportyBetBookmaker._page_market_name
    from predictions.core.betting import LegAvailability
    for market, selection, title in (("Double Chance", "Home or Draw", "Double Chance"),
                                     ("Corners 9.5", "Corners Over 9.5", "Corners - Over/Under"),
                                     ("Cards 2.5", "Cards Over 2.5", "Bookings - Over/Under")):
        leg = leg_from_evaluation(soccer_eval(market, selection))
        offer_ = next(o for o in f.markets if o.market_id == leg.market_id)
        assert name(LegAvailability(leg, Availability.OK, f, offer_, offer_.outcomes[0])) == title


# ------------------------------------------------------------------ Review keeps judging what its engine understands
def test_review_style_exact_evaluation_prices_double_chance_and_corners_but_never_unders():
    engine = SoccerEngine()
    dc = double_chance()
    f = fixture([dc], competition=TOP)
    assert engine.understands(dc)
    exact = by_selection(engine.evaluate_exact(f, STRONG, dc))
    assert set(exact) == {"Home or Draw", "Home or Away", "Away or Draw"}      # Review judges every outcome of the code's market
    assert not engine.understands(MarketOffer("14", "Handicap", "hcp=0:1", ()))
    assert engine.understands(corners("5.5", "1.5")) and engine.understands(cards("1.5", "1.2"))
    low = engine.evaluate_exact(fixture([corners("5.5", "1.5")], competition=TOP), STRONG + extra(), corners("5.5", "1.5"))
    assert [e.selection for e in low] == ["Corners Over 5.5"]                   # Review is not limited to Matchday's lines
