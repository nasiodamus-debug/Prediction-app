"""Story 2.5: corners and cards markets, referee/corner data and the football-data adapter."""
import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

from listview import detail_text, list_text
from predictions.adapters.footballdata.stats import (
    FootballDataInputs, parse_fixtures, parse_season, previous_code, season_code, team_inputs, referee_inputs)
from predictions.core.engines import soccer
from predictions.core.engines.soccer import SoccerEngine, market_kind, negbin_cdf, poisson_cdf
from predictions.core.inputs import Input, SOCCER_REGISTRY, check_inputs, missing_required
from predictions.core.model import PROBABILITY_FLOOR
from predictions.core.ports import StatsFetchError
from predictions.core.stats import READY, FixtureStats, assess, merge_inputs
from predictions.core.types import Fixture, MarketOffer, MatchId, Outcome
from predictions.flow.stats import add_extra_inputs
from test_soccer_optimization import (
    EVEN, KICKOFF, NOW, TOP, Fixtures, Stats, fixture, make_inputs, matchday, matchday_gate, ready, total_offer)

DATA = Path(__file__).parent / "data"


def prop_offer(name, line, over, under):
    return MarketOffer("999", name, f"total={line}",
                       (Outcome(f"Over {line}", Decimal(over)), Outcome(f"Under {line}", Decimal(under))))


def extra(ref=4.4, games=30.0, home_cards=2.2, away_cards=2.2, corners=(6.0, 5.0, 5.5, 5.5), source="test"):
    out = []
    if ref is not None:
        out.append(Input("match.referee_cards_avg", ref, "cards per game", source))
    if games is not None:
        out.append(Input("match.referee_games", games, "games", source))
    if home_cards is not None:
        out.append(Input("home.cards_for", home_cards, "cards per match", source))
    if away_cards is not None:
        out.append(Input("away.cards_for", away_cards, "cards per match", source))
    if corners is not None:
        for name, value in zip(("home.corners_for", "home.corners_against", "away.corners_for",
                                "away.corners_against"), corners):
            out.append(Input(name, value, "corners per match", source))
    return out


def cards_fixture(line="2.5", over="1.40", under="2.70"):
    return fixture([prop_offer("Total Cards", line, over, under)], competition=TOP)


def evals(f, inputs):
    return {e.selection: e for e in SoccerEngine().evaluate(f, inputs, f.markets)}


# ------------------------------------------------------------------ registry
def test_new_inputs_are_registered_with_units_ranges_and_scope():
    for stat in ("corners_for", "corners_against", "cards_for", "referee_cards_avg", "referee_games"):
        spec = SOCCER_REGISTRY[stat]
        assert spec.unit and spec.low >= 0 and spec.high > spec.low and not spec.required
    assert SOCCER_REGISTRY["referee_cards_avg"].scope == "match" and SOCCER_REGISTRY["cards_for"].scope == "team"


def test_match_inputs_are_named_match_and_team_inputs_home_or_away():
    kept, dropped = check_inputs(extra())
    assert len(kept) == len(extra()) and not dropped
    kept, dropped = check_inputs([Input("home.referee_cards_avg", 4.0, "cards per game", "x"),
                                  Input("match.cards_for", 2.0, "cards per match", "x")])
    assert kept == () and len(dropped) == 2


@pytest.mark.parametrize("name,value,unit", [("match.referee_cards_avg", 14.0, "cards per game"),
                                             ("match.referee_cards_avg", 4.0, "cards"),
                                             ("home.corners_for", 25.0, "corners per match"),
                                             ("away.cards_for", -1.0, "cards per match")])
def test_out_of_range_or_wrong_unit_extra_inputs_are_discarded(name, value, unit):
    kept, dropped = check_inputs([Input(name, value, unit, "x")])
    assert kept == () and dropped


def test_new_inputs_are_optional_so_goal_picks_are_not_held_up():
    assert missing_required(make_inputs(1, 1, 1, 1)) == ()
    assert assess(fixture().match_id, make_inputs(1, 1, 1, 1)).ready


def test_merge_only_touches_ready_fixtures_and_validates():
    mid = fixture().match_id
    stats = assess(mid, make_inputs(1, 1, 1, 1))
    merged = merge_inputs(stats, extra() + [Input("match.referee_cards_avg", 99.0, "cards per game", "x")])
    assert merged.ready and {i.name for i in merged.inputs} >= {"match.referee_cards_avg", "home.corners_for"}
    assert next(i for i in merged.inputs if i.name == "match.referee_cards_avg").value == 4.4  # bad duplicate dropped
    not_ready = FixtureStats(mid, "insufficient data", (), ("x",))
    assert merge_inputs(not_ready, extra()) is not_ready


# ------------------------------------------------------------------ market recognition
@pytest.mark.parametrize("name,kind", [("Total Cards", "cards"), ("Cards Over/Under", "cards"),
                                       ("Corners Over/Under", "corners"), ("Total Corners", "corners"),
                                       ("1st Half Corners", None), ("Home Team Corners", None),
                                       ("Corner Handicap", None), ("Booking Points", None),
                                       ("Total Cards Odd/Even", None), ("Over/Under", None)])
def test_corners_and_cards_markets_are_recognised_by_name_only(name, kind):
    assert market_kind(MarketOffer("999", name, "total=9.5", ())) == kind


def test_goals_markets_keep_their_ids():
    assert market_kind(MarketOffer("18", "Over/Under", "total=2.5", ())) == "goals"
    assert market_kind(MarketOffer("1", "1X2", "", ())) == "result"


# ------------------------------------------------------------------ cards: AC1
def test_cards_pick_with_a_qualifying_referee_is_evaluated_with_estimate_ev_and_inputs():
    over = evals(cards_fixture(), EVEN + extra())["Cards Over 2.5"]
    assert over.passed and over.estimate.probability >= PROBABILITY_FLOOR and over.ev > 0
    assert over.estimate.model == "negbin-cards-v1"
    note = "\n".join(over.notes)
    assert "match.referee_cards_avg 4.4" in note and "home.cards_for 2.2" in note and "expected cards 4.4" in note


def test_referee_under_3_5_cards_is_disqualified_with_the_reason():
    over = evals(cards_fixture(), EVEN + extra(ref=3.4))["Cards Over 2.5"]
    assert not over.passed and over.reason.startswith("disqualified:")
    assert "referee averages 3.4 cards per game, under the required 3.5" in over.reason
    assert over.estimate.probability > 0                       # the model number is still shown


def test_referee_at_exactly_3_5_is_allowed():
    over = evals(cards_fixture(), EVEN + extra(ref=3.5, home_cards=2.5, away_cards=2.5))["Cards Over 2.5"]
    assert "disqualified" not in over.reason and over.passed


def test_blowout_risk_of_2_goals_or_more_disqualifies_and_under_2_is_fine():
    blowout = make_inputs(3.0, 0.2, 0.4, 2.6)   # expected 3.1 vs 0.4? home (3.0+2.6)/2=2.8, away (0.4+0.2)/2=0.3
    under = evals(cards_fixture(), blowout + extra())["Cards Over 2.5"]
    assert not under.passed and "blowout risk: expected goal difference 2.5 is not under 2.0" in under.reason
    close = make_inputs(2.0, 1.0, 1.6, 1.4)     # 1.7 v 1.5
    assert "blowout" not in evals(cards_fixture(), close + extra())["Cards Over 2.5"].reason


def test_blowout_exactly_at_the_limit_is_disqualified():
    at_limit = make_inputs(2.0, 1.0, 1.0, 0.0)  # home (2.0+0.0)/2 = 1.0... away (1.0+1.0)/2 = 1.0: diff 0
    assert "blowout" not in evals(cards_fixture(), at_limit + extra())["Cards Over 2.5"].reason
    edge = make_inputs(3.0, 0.0, 0.0, 1.0)      # home (3.0+1.0)/2 = 2.0, away (0.0+0.0)/2 = 0.0: diff exactly 2.0
    assert "blowout" in evals(cards_fixture(), edge + extra())["Cards Over 2.5"].reason


def test_both_failures_are_reported_together():
    blowout = make_inputs(3.0, 0.2, 0.4, 2.6)
    over = evals(cards_fixture(), blowout + extra(ref=3.0))["Cards Over 2.5"]
    assert "referee averages 3.0" in over.reason and "blowout risk" in over.reason


def test_disqualified_cards_pick_never_passes_however_good_the_estimate():
    for e in evals(cards_fixture(over="1.30"), EVEN + extra(ref=3.4, home_cards=3.0, away_cards=3.0)).values():
        assert not e.passed


# ------------------------------------------------------------------ cards: AC3 missing data
def test_no_referee_data_is_insufficient_data_never_guessed():
    over = evals(cards_fixture(), EVEN + extra(ref=None, games=None))["Cards Over 2.5"]
    assert not over.passed and over.insufficient
    assert over.reason == "insufficient data: no referee average for this game"
    assert over.estimate.probability == 0.0 and over.ev == 0


def test_without_any_extra_inputs_every_cards_pick_is_insufficient():
    found = evals(cards_fixture(), EVEN)
    assert set(found) == {"Cards Over 2.5"}   # no Under in any market (AD-20)
    assert all(e.insufficient and not e.passed for e in found.values())


def test_thin_referee_sample_is_insufficient():
    over = evals(cards_fixture(), EVEN + extra(games=5.0))["Cards Over 2.5"]
    assert over.insufficient and "only 5 games" in over.reason


def test_missing_team_card_rate_is_insufficient():
    over = evals(cards_fixture(), EVEN + extra(away_cards=None))["Cards Over 2.5"]
    assert over.insufficient and "away.cards_for" in over.reason


# ------------------------------------------------------------------ corners: AC2 and AC3
def corners_fixture(line="8.5", over="1.60", under="2.30"):
    return fixture([prop_offer("Total Corners", line, over, under)], competition=TOP)


def test_corners_use_the_same_estimate_and_floor_rules_and_show_their_inputs():
    over = evals(corners_fixture(), EVEN + extra())["Corners Over 8.5"]
    assert over.estimate.model == "negbin-corners-v1" and over.estimate.inputs
    assert over.passed and over.estimate.probability >= PROBABILITY_FLOOR and over.ev > 0
    note = "\n".join(over.notes)
    for name in ("home.corners_for 6", "home.corners_against 5", "away.corners_for 5.5", "expected corners 11.0"):
        assert name in note
    assert "[test]" in note                                    # each input is shown with its source


def test_corners_below_the_floor_are_rejected_with_the_estimate():
    over = evals(corners_fixture("9.5", "2.00", "1.80"), EVEN + extra())["Corners Over 9.5"]
    assert not over.passed and "below the 70% floor" in over.reason and 0.5 < over.estimate.probability < 0.65


def test_corners_positive_ev_is_still_required():
    over = evals(corners_fixture("7.5", "1.10", "6.00"), EVEN + extra())["Corners Over 7.5"]
    assert over.estimate.probability >= PROBABILITY_FLOOR and not over.passed and "expected value" in over.reason


def test_corners_do_not_need_the_referee():
    assert evals(corners_fixture(), EVEN + extra(ref=None, games=None, home_cards=None, away_cards=None))[
        "Corners Over 8.5"].passed


def test_missing_corner_data_is_insufficient_data():
    found = evals(corners_fixture(), EVEN + extra(corners=None))
    assert all(e.insufficient and not e.passed and "missing corner data" in e.reason for e in found.values())
    partial = evals(corners_fixture(), EVEN + extra(corners=(6.0, 5.0, 5.5, 5.5))[:-1])
    assert all(e.insufficient for e in partial.values())


def test_corner_whole_line_is_not_a_matchday_play_but_review_prices_it_with_the_refund_maths():
    f = corners_fixture("9", "1.90", "1.90")
    assert evals(f, EVEN + extra()) == {}   # AD-21: no whole line is offered
    over = {e.selection: e for e in SoccerEngine().evaluate_exact(f, EVEN + extra(), f.markets[0])}["Corners Over 9.0"]
    assert over.estimate.push > 0 and "refund on exactly 9 corner(s)" in "\n".join(over.notes)
    assert over.estimate.probability == pytest.approx(over.estimate.win + over.estimate.push)


def test_negative_binomial_is_wider_than_poisson_and_reduces_to_it():
    assert negbin_cdf(5, 4.0, 1.0) == poisson_cdf(5, 4.0)
    assert negbin_cdf(0, 10.0, 1.3) > poisson_cdf(0, 10.0) and negbin_cdf(30, 10.0, 1.3) == pytest.approx(1.0, abs=1e-3)
    assert negbin_cdf(-1, 4.0, 1.2) == 0.0


def test_corners_and_cards_do_not_change_goal_picks():
    f = fixture([total_offer("1.5", "1.30", "3.20"), prop_offer("Total Corners", "8.5", "1.60", "2.30")],
                competition=TOP)
    with_extra = {e.selection: e for e in SoccerEngine().evaluate(f, make_inputs(2.5, 1.5, 2.5, 1.5) + extra(), f.markets)}
    without = {e.selection: e for e in SoccerEngine().evaluate(f, make_inputs(2.5, 1.5, 2.5, 1.5), f.markets)}
    a, b = with_extra["Over 1.5"], without["Over 1.5"]
    assert (a.estimate.probability, a.odds, a.ev, a.passed) == (b.estimate.probability, b.odds, b.ev, b.passed)


def test_constants_sit_at_the_top_of_the_engine():
    assert soccer.CARDS_MIN_REFEREE_AVG == 3.5 and soccer.CARDS_MAX_EXPECTED_GOAL_DIFF == 2.0


# ------------------------------------------------------------------ reply
def test_reply_shows_corner_inputs_and_insufficient_data_lines():
    f = fixture([prop_offer("Total Corners", "8.5", "1.60", "2.30"), prop_offer("Total Cards", "2.5", "1.40", "2.70")],
                competition=TOP)
    chunks, gate = matchday_gate([f], lambda fx: ready(fx, EVEN + extra(ref=None, games=None)))
    shown, detail = list_text(gate.passed), detail_text(gate.passed)
    assert chunks == [] and "Over 8.5 corners @ 1.60" in shown
    assert "inputs: home.corners_for 6 corners per match [test]" in detail
    # spec 11-1: a cards play with no data behind it is not considered (not listed, not a rejection); no Under is priced (AD-20)
    assert "cards" not in shown and gate.rejected == () and gate.unpriced_games == (f.match_id,)


def test_reply_with_only_insufficient_picks_is_no_valid_play():
    f = fixture([prop_offer("Total Cards", "2.5", "1.40", "2.70")], competition=TOP)
    chunks = matchday([f], lambda fx: ready(fx, EVEN))
    assert chunks[0] == "NO VALID PLAY - PASS"
    text = "\n".join(chunks)
    assert "insufficient data: no referee average" not in text and "estimate 0.0%" not in text   # spec 11-1: said nowhere per game


def test_disqualified_cards_show_the_reason_in_the_reply():
    f = cards_fixture()
    text = "\n".join(matchday([f], lambda fx: ready(fx, EVEN + extra(ref=3.2))))
    assert "disqualified: referee averages 3.2 cards per game, under the required 3.5" in text


# ------------------------------------------------------------------ flow: extra inputs source
class FakeExtra:
    def __init__(self, table=None, error=None):
        self.table, self.error, self.asked = table or {}, error, []

    async def fetch_inputs(self, fixtures):
        self.asked = [f.match_id for f in fixtures]
        if self.error:
            raise self.error
        return {f.match_id: self.table[f.match_id] for f in fixtures if f.match_id in self.table}


def test_extra_source_is_asked_only_about_ready_fixtures_and_inputs_are_merged():
    a, b = fixture(home="A", away="B"), fixture(home="C", away="D")
    stats = {a.match_id: ready(a, make_inputs(1, 1, 1, 1)),
             b.match_id: FixtureStats(b.match_id, "insufficient data", (), ("x",))}
    source = FakeExtra({a.match_id: extra()})
    merged = asyncio.run(add_extra_inputs([a, b], stats, source))
    assert source.asked == [a.match_id] and merged[b.match_id] is stats[b.match_id]
    assert "match.referee_cards_avg" in {i.name for i in merged[a.match_id].inputs}


def test_end_to_end_cards_pick_uses_the_extra_source():
    f = cards_fixture()
    chunks, gate = matchday_gate([f], lambda fx: ready(fx, EVEN), extra_inputs_source=FakeExtra({f.match_id: extra()}))
    assert chunks == [] and "Over 2.5 yellow cards @ 1.40" in list_text(gate.passed)
    assert "referee_cards_avg" in detail_text(gate.passed)


def test_extra_source_outage_is_reported_and_picks_become_insufficient():
    f = cards_fixture()
    chunks = matchday([f], lambda fx: ready(fx, EVEN),
                      extra_inputs_source=FakeExtra(error=StatsFetchError("football-data.co.uk could not be reached")))
    text = "\n".join(chunks)
    assert chunks[0] == "NO VALID PLAY - PASS"
    assert "Could not get corner and card data: football-data.co.uk could not be reached" in text
    assert "Corners and cards plays are not considered." in text and "insufficient data: no referee average" not in text


# ------------------------------------------------------------------ football-data adapter
SEASON = (DATA / "footballdata_E0_2627_sample.csv").read_text(encoding="utf-8")
FIXTURES_CSV = (DATA / "footballdata_fixtures_sample.csv").read_text(encoding="utf-8")


def test_season_csv_parses_played_games_and_counts_yellow_cards_only():
    games = parse_season(SEASON)
    first = games[0]
    assert (first.home, first.away, first.referee) == ("Arsenal", "Coventry", "T Bramall")
    assert (first.corners_home, first.corners_away, first.cards_home, first.cards_away) == (8, 2, 1, 1)
    red = parse_season("Div,Date,HomeTeam,AwayTeam,Referee,HC,AC,HY,AY,HR,AR\n"
                       "E0,01/08/2026,A,B,X Y,5,4,2,1,1,0\n")[0]
    assert (red.cards_home, red.cards_away) == (2, 1)   # spec 11-1: the owner plays YELLOW cards, a red is not counted


def test_unplayed_rows_are_skipped_never_read_as_zero():
    text = ("Div,Date,HomeTeam,AwayTeam,Referee,HC,AC,HY,AY,HR,AR\n"
            "E0,01/08/2026,A,B,X Y,,,,,,\nE0,02/08/2026,C,D,X Y,3,4,1,1,0,0\n")
    games = parse_season(text)
    assert [(g.home, g.away) for g in games] == [("C", "D")]


def test_fixtures_csv_carries_the_assigned_referee():
    games = parse_fixtures(FIXTURES_CSV)
    assert games and all(g.referee for g in games) and games[0].home == "Boreham Wood"


@pytest.mark.parametrize("text", ["", "a,b\n1,2\n", "Date,HomeTeam\n01/01/2026,A\n"])
def test_unexpected_shape_raises(text):
    with pytest.raises(StatsFetchError):
        parse_season(text)


def test_bad_date_raises():
    with pytest.raises(StatsFetchError):
        parse_season("Date,HomeTeam,AwayTeam,HC,AC,HY,AY,HR,AR\nnot-a-date,A,B,1,1,1,1,0,0\n")


def test_season_codes():
    assert season_code(datetime(2026, 10, 1, tzinfo=timezone.utc)) == "2627"
    assert season_code(datetime(2027, 3, 1, tzinfo=timezone.utc)) == "2627"
    assert previous_code("2627") == "2526" and previous_code("0001") == "9900"


def synthetic_season(n=6):
    rows = ["Div,Date,Time,HomeTeam,AwayTeam,Referee,HC,AC,HY,AY,HR,AR"]
    for i in range(n):
        rows.append(f"E0,{i + 1:02d}/08/2026,15:00,Man United,Leeds,R One,6,4,2,1,0,0")
        rows.append(f"E0,{i + 1:02d}/08/2026,15:00,Leeds,Man United,R Two,3,7,1,3,1,0")
    return "\n".join(rows) + "\n"


def test_team_inputs_are_averages_over_the_teams_own_side_of_each_game():
    games = parse_season(synthetic_season())
    home = team_inputs("home", "Man United", games, "src")
    values = {i.name: i.value for i in home}
    assert values == {"home.corners_for": 6.5, "home.corners_against": 3.5, "home.cards_for": 2.5}
    assert all(i.source == "src" for i in home) and [i.unit for i in home][0] == "corners per match"
    assert team_inputs("home", "Man United", games[:3], "src") == []       # under MIN_MATCHES: nothing


def test_referee_inputs_average_both_teams_cards_over_his_games():
    games = parse_season(synthetic_season())
    ref = {i.name: i for i in referee_inputs("R One", games, "src")}
    assert ref["match.referee_cards_avg"].value == 3.0 and ref["match.referee_games"].value == 6.0
    assert "R One" in ref["match.referee_cards_avg"].source
    assert referee_inputs("Nobody", games, "src") == []


def adapter(season_text, fixtures_text, calls=None, status=200):
    def handler(request):
        url = str(request.url)
        if calls is not None:
            calls.append(url)
        if status != 200:
            return httpx.Response(status)
        if url.endswith("fixtures.csv"):
            return httpx.Response(200, text=fixtures_text)
        return httpx.Response(200, text=season_text if "2627" in url else "Div,Date,HomeTeam,AwayTeam\n")
    return FootballDataInputs(httpx.AsyncClient(transport=httpx.MockTransport(handler)))


HEADER = "Div,Date,Time,HomeTeam,AwayTeam,Referee\n"


def epl_fixture(home="Manchester United", away="Leeds United", competition="England - Premier League"):
    kickoff = datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc)
    return Fixture(MatchId("soccer", competition, kickoff, home, away), competition, home, away, kickoff, False)


def test_adapter_gives_team_and_referee_inputs_for_a_listed_fixture():
    f = epl_fixture()
    listed = HEADER + "E0,10/10/2026,15:00,Man United,Leeds,R One\n"
    calls = []
    found = asyncio.run(adapter(synthetic_season(), listed, calls).fetch_inputs([f]))
    names = {i.name: i for i in found[f.match_id]}
    assert names["home.corners_for"].value == 6.5 and names["away.corners_for"].value == 3.5
    assert names["match.referee_cards_avg"].value == 3.0 and names["match.referee_games"].value == 6.0
    assert names["home.corners_for"].source == "footballdata:E0:2627"
    kept, dropped = check_inputs(found[f.match_id])
    assert len(kept) == len(found[f.match_id]) and not dropped


def test_adapter_gives_no_referee_when_the_game_is_not_in_fixtures_csv():
    f = epl_fixture()
    found = asyncio.run(adapter(synthetic_season(), HEADER).fetch_inputs([f]))
    names = {i.name for i in found[f.match_id]}
    assert "home.corners_for" in names and not {n for n in names if n.startswith("match.")}


def test_adapter_is_silent_for_uncovered_leagues_and_unknown_teams_and_makes_no_request():
    calls = []
    source = adapter(synthetic_season(), HEADER, calls)
    assert asyncio.run(source.fetch_inputs([epl_fixture(competition="Russia - Premier League")])) == {}
    assert calls == []
    assert asyncio.run(source.fetch_inputs([epl_fixture(home="Nowhere FC")])) == {}


def test_adapter_needs_enough_history_before_kickoff():
    f = epl_fixture()
    assert asyncio.run(adapter(synthetic_season(2), HEADER).fetch_inputs([f])) == {}


def test_adapter_failure_raises_a_safe_error():
    with pytest.raises(StatsFetchError) as error:
        asyncio.run(adapter("", "", status=503).fetch_inputs([epl_fixture()]))
    assert str(error.value) == "football-data.co.uk returned HTTP 503"


def test_adapter_end_to_end_with_the_real_engine():
    f = epl_fixture(home="Arsenal")      # (Manchester United v Leeds is a derby in the table)
    market = prop_offer("Total Cards", "2.5", "1.40", "2.70")
    f = Fixture(f.match_id, f.competition, f.home, f.away, f.kickoff_utc, False, (market,))
    listed = HEADER + "E0,10/10/2026,15:00,Arsenal,Leeds,R One\n"
    found = asyncio.run(adapter(synthetic_season(10).replace("Man United", "Arsenal"), listed).fetch_inputs([f]))
    stats = merge_inputs(assess(f.match_id, make_inputs(2.0, 1.0, 2.0, 1.0)), found[f.match_id])
    over = {e.selection: e for e in SoccerEngine().evaluate(f, stats.inputs, f.markets)}["Cards Over 2.5"]
    assert not over.insufficient and "footballdata:E0:2627" in "\n".join(over.notes)
