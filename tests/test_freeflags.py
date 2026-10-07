"""API-Football free flags (injuries, lineups) and the model-check warning. No network: the source is a fake or a MockTransport."""
import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import httpx
import pytest

from predictions.adapters.apifootball.client import (ApiFixture, ApiFootball, Missing, Prediction, parse_fixtures,
                                                      parse_injuries, parse_lineups_published, parse_prediction)
from predictions.core.analyst import ContextFlag
from predictions.core.model import Estimate, Evaluation
from predictions.core.ttl_cache import STATS_CACHE
from predictions.core.types import Fixture, MatchId
from predictions.flow import freeflags
from predictions.flow.freeflags import disagreement_lines, free_flags, match_fixture, merge_flags, vendor_chance

NOW = datetime(2026, 10, 3, 12, 0, tzinfo=timezone.utc)


def game(home="Atletico Mineiro", away="Flamengo", minutes=300, sport="soccer"):
    kickoff = NOW + timedelta(minutes=minutes)
    return Fixture(MatchId(sport, "Serie A", kickoff, home.lower(), away.lower()), "Serie A", home, away, kickoff, False)


def api(fid=1, home="Atletico Mineiro", away="Flamengo", minutes=300):
    return ApiFixture(fid, home, away, NOW + timedelta(minutes=minutes), "Serie A")


def out(team, n):
    return tuple(Missing(team, f"P{i}", "Missing Fixture", "Muscle Injury") for i in range(n))


class FakeSource:
    def __init__(self, fixtures=(), injuries=None, lineups=None, prediction=None):
        self._fixtures, self._injuries, self._lineups, self._prediction = list(fixtures), injuries or {}, lineups, prediction
        self.calls = []

    async def fixtures(self, day):
        self.calls.append("fixtures")
        return self._fixtures

    async def injuries(self, day):
        self.calls.append("injuries")
        return self._injuries

    async def lineups_published(self, fid):
        self.calls.append("lineups")
        return self._lineups

    async def prediction(self, fid):
        self.calls.append("prediction")
        return self._prediction


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------- parsers
def test_parsers_read_the_probed_shapes_and_skip_bad_items():
    fx = parse_fixtures([{"fixture": {"id": 7, "date": "2026-10-03T19:00:00+00:00"}, "teams": {"home": {"name": "A"}, "away": {"name": "B"}},
                          "league": {"name": "L"}}, {"broken": True}])
    assert len(fx) == 1 and fx[0].id == 7 and fx[0].kickoff_utc.hour == 19
    inj = parse_injuries([{"fixture": {"id": 7}, "team": {"name": "A"}, "player": {"name": "X", "type": "Missing Fixture", "reason": "Foot Injury"}}])
    assert inj == {7: (Missing("A", "X", "Missing Fixture", "Foot Injury"),)}
    assert parse_lineups_published([{"startXI": [1] * 11}, {"startXI": [1] * 11}]) is True
    assert parse_lineups_published([]) is False and parse_lineups_published([{"startXI": [1]}]) is False
    p = parse_prediction([{"predictions": {"percent": {"home": "45%", "draw": "25%", "away": "30%"}}}])
    assert p == Prediction(0.45, 0.25, 0.30)
    assert parse_prediction([{"predictions": {"percent": {"home": "90%", "draw": "90%", "away": "90%"}}}]) is None
    assert parse_prediction([]) is None


# ---------------------------------------------------------------- matching
def test_matching_needs_both_names_and_a_close_kickoff():
    g = game()
    assert match_fixture(g, [api()]).id == 1
    assert match_fixture(g, [api(home="Cruzeiro")]) is None
    assert match_fixture(g, [api(minutes=300 + 600)]) is None
    assert match_fixture(g, [api(minutes=300 + 60)]) is None   # an hour apart is not the same game


def test_real_spelling_differences_between_sportybet_and_api_football_are_matched():
    cases = [("Union de Santa Fe", "Union Santa Fe"), ("Independiente Rivadavia", "Independ. Rivadavia"),
             ("CA Lanus", "Lanus"), ("Atletico Mineiro MG", "Atletico-MG"), ("CA Barracas Central", "Barracas Central"),
             ("Gimnasia y Esgrima La Plata", "Gimnasia L.P.")]
    for ours, theirs in cases:
        assert match_fixture(game(ours, "Flamengo"), [api(home=theirs, away="Flamengo")]) is not None, (ours, theirs)
    assert match_fixture(game("North Macedonia", "Scotland"), [api(home="FYR Macedonia", away="Scotland")]) is not None


def test_different_clubs_reserves_and_women_are_not_matched():
    real, reserves = api(1, "Real Madrid", "Sevilla"), api(2, "Real Madrid Castilla", "Sevilla")
    assert match_fixture(game("Real Madrid", "Sevilla"), [reserves, real]).id == 1   # the real fixture outranks the reserves
    assert match_fixture(game("Real Madrid", "Sevilla"), [api(2, "Real Madrid Castilla", "Cadiz")]) is None
    assert match_fixture(game("Orlando Pride", "San Diego Wave"), [api(home="Orlando Pride W", away="San Diego Wave W")]) is None
    assert match_fixture(game("Manchester United", "Arsenal"), [api(home="Manchester City", away="Chelsea")]) is None
    assert match_fixture(game("Manchester United", "Arsenal"), [api(1, "Manchester City", "Arsenal"), api(2, "Manchester United", "Arsenal")]).id == 2
    assert match_fixture(game("Arsenal", "Chelsea"), [api(home="Chelsea", away="Arsenal")]) is None


def test_two_equally_good_candidates_match_nothing():
    g = game("Boca Juniors", "Union de Santa Fe")
    twin = [api(1, "Boca Juniors", "Union Santa Fe"), api(2, "Boca Juniors", "Union Santa Fe")]
    assert match_fixture(g, twin) is None
    assert match_fixture(g, [api(1, "Boca Juniors", "Union Santa Fe"), api(2, "Boca Jrs", "Colon Santa Fe")]).id == 1


def test_an_unmatched_spelling_gets_no_flags_and_no_error():
    src = FakeSource([api(home="Cruzeiro")], {1: out("Cruzeiro", 5)})
    assert run(free_flags([game()], NOW, src)) == {}


# ---------------------------------------------------------------- flags
def test_three_or_more_absent_players_raise_key_player_out_two_do_not():
    src = FakeSource([api()], {1: out("Atletico Mineiro", 3) + out("Flamengo", 2)})
    found = run(free_flags([game()], NOW, src))
    flags = found[game().match_id]
    assert [f.flag for f in flags] == [ContextFlag.KEY_PLAYER_OUT] and "Atletico Mineiro without 3" in flags[0].source
    assert run(free_flags([game()], NOW, FakeSource([api()], {1: out("Atletico Mineiro", 2)}))) == {}


def test_lineup_uncertain_only_near_kickoff_and_only_when_known_unpublished():
    near = game(minutes=30)
    assert [f.flag for f in run(free_flags([near], NOW, FakeSource([api(minutes=30)], lineups=False)))[near.match_id]] == [ContextFlag.LINEUP_UNCERTAIN]
    assert run(free_flags([near], NOW, FakeSource([api(minutes=30)], lineups=True))) == {}
    assert run(free_flags([near], NOW, FakeSource([api(minutes=30)], lineups=None))) == {}   # unreadable is not "uncertain"
    far = FakeSource([api(minutes=300)], lineups=False)
    assert run(free_flags([game(minutes=300)], NOW, far)) == {} and "lineups" not in far.calls


def test_only_todays_unstarted_soccer_games_cost_a_request():
    src = FakeSource([api()], {})
    assert run(free_flags([game(sport="basketball")], NOW, src)) == {} and src.calls == []
    tomorrow = game(minutes=24 * 60)
    assert run(free_flags([tomorrow], NOW, src)) == {} and src.calls == []
    assert run(free_flags([game()], NOW, None)) == {}   # no source: off


def test_merge_keeps_both_sources_flags():
    a, b = game(), game("X", "Y")
    one = {a.match_id: (freeflags.AnalystFlag(ContextFlag.KEY_PLAYER_OUT, "s1"),)}
    two = {a.match_id: (freeflags.AnalystFlag(ContextFlag.LINEUP_UNCERTAIN, "s2"),), b.match_id: (freeflags.AnalystFlag(ContextFlag.SEVERE_WEATHER, "s3"),)}
    merged = merge_flags(one, two)
    assert len(merged[a.match_id]) == 2 and len(merged[b.match_id]) == 1


# ---------------------------------------------------------------- model check
def pick(g, market, selection, p):
    return Evaluation(g.match_id, market, selection, Decimal("1.50"), Estimate(p), Decimal("0.1"), True, "passed")


def test_vendor_chance_for_1x2_and_double_chance_only():
    pr = Prediction(0.5, 0.3, 0.2)
    assert vendor_chance("Home", "1X2", pr) == 0.5 and vendor_chance("Away", "1X2", pr) == 0.2
    assert vendor_chance("Home or Draw", "Double Chance", pr) == pytest.approx(0.8)
    assert vendor_chance("Over 2.5", "Over/Under 2.5", pr) is None


def test_a_pick_the_vendor_rates_far_below_ours_gets_a_warning_and_nothing_else_changes():
    g = game()
    src = FakeSource([api()], prediction=Prediction(0.30, 0.30, 0.40))
    ev = pick(g, "1X2", "Home", 0.70)
    lines = run(disagreement_lines((ev,), {g.match_id: g}, NOW, src))
    assert len(lines) == 1 and "70.0%" in lines[0] and "30.0%" in lines[0] and "Atletico Mineiro v Flamengo" in lines[0]
    assert ev.passed and ev.estimate.probability == 0.70   # a warning only


def test_no_warning_when_close_when_vendor_is_higher_or_for_other_markets():
    g = game()
    ev = pick(g, "1X2", "Home", 0.70)
    assert run(disagreement_lines((ev,), {g.match_id: g}, NOW, FakeSource([api()], prediction=Prediction(0.55, 0.25, 0.20)))) == ()
    assert run(disagreement_lines((ev,), {g.match_id: g}, NOW, FakeSource([api()], prediction=Prediction(0.95, 0.03, 0.02)))) == ()
    goals = pick(g, "Over/Under 2.5", "Over 2.5", 0.70)
    src = FakeSource([api()], prediction=Prediction(0.1, 0.1, 0.8))
    assert run(disagreement_lines((goals,), {g.match_id: g}, NOW, src)) == () and src.calls == []
    assert run(disagreement_lines((ev,), {g.match_id: g}, NOW, FakeSource([api()], prediction=None))) == ()


# ---------------------------------------------------------------- client
def client(handler):
    return ApiFootball(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def test_client_failures_are_no_data_never_an_error():
    refused = client(lambda r: httpx.Response(200, json={"errors": {"plan": "no"}, "response": []}))
    assert run(refused.fixtures(NOW.date())) == [] and run(refused.injuries(NOW.date())) == {}
    down = client(lambda r: httpx.Response(500, json={}))
    assert run(down.lineups_published(99)) is None and run(down.prediction(99)) is None
