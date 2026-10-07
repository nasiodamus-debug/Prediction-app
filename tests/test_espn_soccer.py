"""Story 9.3: soccer without xG. The ESPN goals-only stats adapter (recorded and fabricated samples of ESPN's public API,
tests/data/espn_soccer_*.json; no test makes a live call), the goals-only path of the soccer engine, the composite stats
source and the labels shown with every goals-only pick."""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import app as bot
from handled import drop_ack, handled
from listview import detail_text, list_text
from predictions.adapters.espn import soccer as espn
from predictions.adapters.espn.soccer import EspnSoccerStats, parse_schedule, parse_scoreboard, team_inputs
from predictions.adapters.understat.stats import UnderstatStats
from predictions.core.engines.soccer import (GOALS_ONLY_CAP_REASON, GOALS_ONLY_LABEL, GOALS_ONLY_MAX_EV,
                                             GOALS_ONLY_MODEL_NAME, GOALS_ONLY_PRIOR_GOALS,
                                             GOALS_ONLY_PRIOR_MATCHES, SoccerEngine, expected_goals, is_goals_only)
from predictions.core.inputs import GOALS_ONLY_MIN_MATCHES, Input, SOCCER_REGISTRY
from predictions.core.model import PROBABILITY_FLOOR, Estimate
from predictions.core.ports import StatsFetchError
from predictions.core.stats import INSUFFICIENT, READY, FixtureStats, assess, assess_goals_only, insufficient
from predictions.core.types import Fixture, MarketOffer, MatchId, Outcome
from predictions.flow.evaluate import run_gate
from predictions.flow.matchday import run_matchday
from predictions.flow.stats import FirstCoveringStats
from predictions.telegram import render, slate

DATA = Path(__file__).parent / "data"
BOARD_SAMPLE = json.loads((DATA / "espn_soccer_scoreboard_sample.json").read_text(encoding="utf-8"))
SCHEDULE_SAMPLE = json.loads((DATA / "espn_soccer_schedule_sample.json").read_text(encoding="utf-8"))
NOW = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
KICKOFF = datetime(2026, 10, 1, 16, 0, tzinfo=timezone.utc)
NATIONS = "International - UEFA Nations League"


def fixture(home="Azerbaijan", away="Liechtenstein", competition=NATIONS, kickoff=KICKOFF, markets=()):
    return Fixture(MatchId("soccer", competition, kickoff, home, away), competition, home, away, kickoff, False,
                   tuple(markets))


# ---------------------------------------------------------------- fabricated ESPN payloads
def stamp(when):
    return when.strftime("%Y-%m-%dT%H:%MZ")


def event(event_id, when, home, away, home_goals, away_goals, completed=True):
    """home and away are (team id, name)."""
    def side(where, team, goals):
        return {"homeAway": where, "team": {"id": team[0], "displayName": team[1]},
                "score": {"value": float(goals), "displayValue": str(goals)}}
    return {"id": str(event_id), "date": stamp(when),
            "competitions": [{"id": str(event_id), "status": {"type": {"completed": completed}},
                              "competitors": [side("home", home, home_goals), side("away", away, away_goals)]}]}


def played(team, count, goals_for=2, goals_against=1, start_id=1000, days_before=7, opponent=("9999", "Opponent")):
    """`count` completed matches of `team`, one a week back from the kickoff."""
    return [event(start_id + i, KICKOFF - timedelta(days=days_before * (i + 1)), team, opponent, goals_for, goals_against)
            for i in range(count)]


class Api:
    """ESPN's soccer API from fabricated payloads: scoreboards by (slug, day), schedules by (team id, season or None)."""

    def __init__(self, boards=None, schedules=None, fail=None):
        self.boards, self.schedules, self.fail, self.requests = boards or {}, schedules or {}, fail, []

    def __call__(self, request):
        self.requests.append(request)
        if self.fail:
            return self.fail(request)
        parts = request.url.path.split("/soccer/")[1].split("/")
        if parts[-1] == "scoreboard":
            events = self.boards.get((parts[0], request.url.params["dates"]), [])
            return httpx.Response(200, json={"events": events})
        if parts[-1] == "schedule":
            team = parts[2]
            season = request.url.params.get("season")
            events = self.schedules.get((team, int(season) if season else None), [])
            return httpx.Response(200, json={"requestedSeason": {"year": int(season) if season else 2026},
                                             "events": events})
        return httpx.Response(404)

    def paths(self):
        return [r.url.path.split("/soccer/")[1] for r in self.requests]


def fetch(fixtures, api, now=NOW):
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(api)) as client:
            return await EspnSoccerStats(client).fetch_stats(fixtures)
    return asyncio.run(go())


AZE, LIE = ("581", "Azerbaijan"), ("2001", "Liechtenstein")
NATIONS_BOARD = {("uefa.nations", "20261001"): [event(1, KICKOFF, AZE, LIE, 0, 0, completed=False)]}


def nations_api(aze=10, lie=10, **kw):
    return Api(NATIONS_BOARD, {("581", None): played(AZE, aze, 1, 2, 1000), ("2001", None): played(LIE, lie, 0, 3, 2000)},
               **kw)


# ---------------------------------------------------------------- parsing (recorded samples)
def test_the_recorded_scoreboard_sample_parses():
    games = parse_scoreboard(BOARD_SAMPLE)
    assert len(games) == 3
    first = games[0]
    assert first.when == datetime(2026, 10, 1, 16, 0, tzinfo=timezone.utc)
    assert first.teams == (("581", "Azerbaijan"), ("2001", "Liechtenstein")) or {n for _, n in first.teams} == {
        "Azerbaijan", "Liechtenstein"}
    assert {n for g in games for _, n in g.teams} >= {"Denmark", "Portugal", "Germany", "Serbia"}


def test_the_recorded_schedule_sample_parses_into_completed_matches_with_goals():
    matches, year = parse_schedule(SCHEDULE_SAMPLE, "581")
    assert year == 2026 and len(matches) == 12
    assert all(m.when < KICKOFF for m in matches)
    by_id = {m.event_id: m for m in matches}
    friendly = by_id["401898013"]  # Azerbaijan 1-0 Tajikistan, 23 Sep 2026 (probed 2026-10-01)
    assert (friendly.goals_for, friendly.goals_against) == (1, 0)
    draw = by_id["401861067"]  # Lithuania 1-1 Azerbaijan, Nations League
    assert (draw.goals_for, draw.goals_against) == (1, 1)


def test_unfinished_and_unscored_events_are_left_out_and_bad_shapes_fail_loudly():
    pending = event(1, KICKOFF, AZE, LIE, 0, 0, completed=False)
    unscored = event(2, KICKOFF - timedelta(days=3), AZE, LIE, 1, 1)
    unscored["competitions"][0]["competitors"][0]["score"] = None
    ok = event(3, KICKOFF - timedelta(days=5), AZE, LIE, 3, 1)
    matches, _ = parse_schedule({"events": [pending, unscored, ok]}, "581")
    assert [(m.goals_for, m.goals_against) for m in matches] == [(3, 1)]
    for bad in (None, [], {"events": "x"}, {}):
        with pytest.raises(StatsFetchError):
            parse_schedule(bad, "581")
        with pytest.raises(StatsFetchError):
            parse_scoreboard(bad)


# ---------------------------------------------------------------- the adapter
def test_a_covered_game_becomes_ready_with_goals_only_inputs_and_provenance():
    f = fixture()
    result = fetch([f], nations_api())[f.match_id]
    assert result.status == READY and not any("xg" in i.name for i in result.inputs)
    by = {i.name: i for i in result.inputs}
    assert by["home.matches"].value == 10 and by["away.matches"].value == 10
    assert by["home.goals_for"].value == pytest.approx(1.0) and by["home.goals_against"].value == pytest.approx(2.0)
    assert by["away.goals_for"].value == pytest.approx(0.0) and by["away.goals_against"].value == pytest.approx(3.0)
    assert {i.source for i in result.inputs} == {"espn:goals-only:uefa.nations"}
    assert by["home.goals_for"].unit == "goals per match" and by["home.matches"].unit == "matches"
    assert is_goals_only(result.inputs)


def test_only_the_last_ten_matches_before_kickoff_count():
    f = fixture()
    history = played(AZE, 10, 1, 2, 1000) + played(AZE, 5, 9, 9, 5000, days_before=100)  # older, loud matches
    future = [event(7000, KICKOFF + timedelta(days=3), AZE, LIE, 8, 8)]
    unfinished = [event(7001, KICKOFF - timedelta(days=1), AZE, LIE, 8, 8, completed=False)]
    api = Api(NATIONS_BOARD, {("581", None): history + future + unfinished, ("2001", None): played(LIE, 10, 0, 3, 2000)})
    by = {i.name: i.value for i in fetch([f], api)[f.match_id].inputs}
    assert by["home.matches"] == 10 and by["home.goals_for"] == pytest.approx(1.0)
    assert by["home.goals_against"] == pytest.approx(2.0)


def test_too_little_history_is_insufficient_never_estimated_and_the_previous_season_is_read():
    f = fixture()
    api = nations_api(aze=7)
    result = fetch([f], api)[f.match_id]
    assert result.status == INSUFFICIENT and "too little history" in result.reasons[0]
    assert "Azerbaijan has 7 completed match(es)" in result.reasons[0] and str(GOALS_ONLY_MIN_MATCHES) in result.reasons[0]
    assert any(r.url.params.get("season") == "2025" for r in api.requests if "/581/" in r.url.path)
    assert result.inputs == ()


def test_the_previous_season_tops_up_a_thin_current_season():
    f = fixture()
    api = Api(NATIONS_BOARD, {("581", None): played(AZE, 4, 1, 2, 1000),
                              ("581", 2025): played(AZE, 6, 1, 2, 3000, days_before=70),
                              ("2001", None): played(LIE, 10, 0, 3, 2000)})
    result = fetch([f], api)[f.match_id]
    assert result.status == READY and {i.name: i.value for i in result.inputs}["home.matches"] == 10


def test_exactly_the_minimum_history_is_enough():
    f = fixture()
    assert fetch([f], nations_api(aze=8, lie=8))[f.match_id].status == READY


def test_a_competition_the_source_does_not_list_is_not_covered_and_costs_no_request():
    f = fixture(competition="Israel - National League")
    api = nations_api()
    result = fetch([f], api)[f.match_id]
    assert result.status == INSUFFICIENT and result.reasons == (espn.NOT_COVERED,) and api.requests == []


def test_a_game_the_scoreboard_does_not_list_is_insufficient():
    f = fixture(home="Azerbaijan", away="Malta")
    result = fetch([f], nations_api())[f.match_id]
    assert result.status == INSUFFICIENT and "'Malta' is not in the stats source" in result.reasons[0]
    other = ("3", "Malta")
    api = Api({("uefa.nations", "20261001"): [event(1, KICKOFF, AZE, LIE, 0, 0, False), event(2, KICKOFF, other, ("4", "Gibraltar"), 0, 0, False)]})
    result = fetch([fixture(home="Azerbaijan", away="Malta")], api)[fixture(home="Azerbaijan", away="Malta").match_id]
    assert result.status == INSUFFICIENT and "lists no Azerbaijan v Malta game near this kickoff" in result.reasons[0]


def test_a_listed_game_far_from_the_kickoff_is_not_the_game():
    f = fixture()
    far = {("uefa.nations", "20261001"): [event(1, KICKOFF + timedelta(days=3), AZE, LIE, 0, 0, False)]}
    result = fetch([f], Api(far))[f.match_id]
    assert result.status == INSUFFICIENT and "'Azerbaijan' is not in the stats source" in result.reasons[0]


def test_an_unclear_team_name_is_ambiguous_never_guessed():
    f = fixture(home="Dinamo", away="Rapid")
    board = {("club.friendly", "20261001"): [event(1, KICKOFF, ("1", "Dinamo Zagreb"), ("2", "Rapid Vienna"), 0, 0, False),
                                             event(2, KICKOFF, ("3", "Dinamo Tbilisi"), ("4", "Other"), 0, 0, False)]}
    result = fetch([fixture(home="Dinamo", away="Rapid Vienna", competition="International Clubs - Club Friendly Games")],
                   Api(board))
    only = next(iter(result.values()))
    assert only.status == "ambiguous" and "Dinamo Tbilisi" in only.question and "Dinamo Zagreb" in only.question


def test_women_and_youth_sides_are_matched_only_inside_their_own_competition():
    women = "International Clubs - UEFA Champions League Women"
    board = {("uefa.wchampions", "20261001"): [event(1, KICKOFF, ("10", "Arsenal"), ("11", "Chelsea"), 0, 0, False)]}
    api = Api(board, {("10", None): played(("10", "Arsenal"), 10, 2, 1, 100), ("11", None): played(("11", "Chelsea"), 10, 2, 0, 200)})
    f = fixture("Arsenal W", "Chelsea WFC", women)
    assert fetch([f], api)[f.match_id].status == READY
    # the same "W" names in a club friendly are NOT matched to the senior Arsenal and Chelsea
    club = "International Clubs - Club Friendly Games"
    senior = {("club.friendly", "20261001"): [event(1, KICKOFF, ("10", "Arsenal"), ("11", "Chelsea"), 0, 0, False)]}
    g = fixture("Arsenal W", "Chelsea W", club)
    assert fetch([g], Api(senior))[g.match_id].status == INSUFFICIENT
    youth = "International Youth - U21 UEFA European Championship, Qualification"
    board = {("uefa.euro_u21_qual", "20261001"): [event(1, KICKOFF, ("20", "Greece U21"), ("21", "Latvia U21"), 0, 0, False)]}
    api = Api(board, {("20", None): played(("20", "Greece U21"), 10, 1, 1, 100), ("21", None): played(("21", "Latvia U21"), 10, 1, 2, 200)})
    h = fixture("Greece", "Latvia", youth)
    assert fetch([h], api)[h.match_id].status == READY
    senior_board = {("uefa.nations", "20261001"): [event(1, KICKOFF, ("20", "Greece U21"), ("21", "Latvia U21"), 0, 0, False)]}
    k = fixture("Greece", "Latvia", NATIONS)
    assert fetch([k], Api(senior_board))[k.match_id].status == INSUFFICIENT


def test_a_national_team_is_not_matched_to_a_club_of_a_similar_name():
    board = {("uefa.nations", "20261001"): [event(1, KICKOFF, ("30", "Austria Vienna"), ("2001", "Liechtenstein"), 0, 0, False)]}
    f = fixture("Austria", "Liechtenstein")
    result = fetch([f], Api(board))[f.match_id]
    assert result.status in (INSUFFICIENT, "ambiguous") and result.status != READY


def test_requests_are_polite_keyless_and_never_retried():
    api = nations_api()
    fetch([fixture()], api)
    assert all(r.headers["user-agent"].startswith("python-httpx") for r in api.requests)  # ESPN answers 403 to custom UAs
    assert all("key" not in str(r.url).lower() and "authorization" not in r.headers for r in api.requests)
    assert sorted(api.paths()) == sorted(["uefa.nations/scoreboard", "all/teams/581/schedule", "all/teams/2001/schedule"])
    failing = nations_api(fail=lambda request: httpx.Response(500))
    with pytest.raises(StatsFetchError) as error:
        fetch([fixture()], failing)
    assert "HTTP 500" in str(error.value) and len(failing.requests) == 1  # no retry
    broken = nations_api(fail=lambda request: httpx.Response(200, text="<html>"))
    with pytest.raises(StatsFetchError):
        fetch([fixture()], broken)


def test_one_schedule_read_per_team_even_when_a_team_plays_twice_in_the_list():
    a, b = fixture(), fixture(kickoff=KICKOFF + timedelta(hours=1))
    api = nations_api()
    api.boards[("uefa.nations", "20261001")].append(event(2, KICKOFF + timedelta(hours=1), AZE, LIE, 0, 0, False))
    fetch([a, b], api)
    assert api.paths().count("all/teams/581/schedule") == 1


# ---------------------------------------------------------------- the goals-only engine path
def goals_inputs(matches=10, gf=2.0, ga=1.5, source="espn:goals-only:test"):
    return [Input(f"{side}.{stat}", float(value), unit, source)
            for side in ("home", "away")
            for stat, value, unit in (("matches", matches, "matches"), ("goals_for", gf, "goals per match"),
                                      ("goals_against", ga, "goals per match"))]


def xg_inputs(gf=2.0, ga=1.5):
    values = {"matches": 10, "xg_for": gf, "goals_for": gf, "xg_against": ga, "goals_against": ga}
    return [Input(f"{side}.{stat}", float(v), SOCCER_REGISTRY[stat].unit, "test")
            for side in ("home", "away") for stat, v in values.items()]


def over(line, odds, under="2.30"):
    return MarketOffer("18", "Over/Under", f"total={line}",
                       (Outcome(f"Over {line}", Decimal(odds)), Outcome(f"Under {line}", Decimal(under))))


def test_the_goals_only_registry_needs_eight_matches_and_no_xg():
    assert GOALS_ONLY_MIN_MATCHES == 8
    mid = fixture().match_id
    assert assess_goals_only(mid, goals_inputs(matches=8)).status == READY
    short = assess_goals_only(mid, goals_inputs(matches=7))
    assert short.status == INSUFFICIENT and "missing required input: home.matches" in short.reasons[0]
    assert assess(mid, goals_inputs(matches=10)).status == INSUFFICIENT  # the xG model still requires xG


def test_goals_are_shrunk_toward_the_prior_so_the_goals_only_total_is_more_conservative_than_xg():
    home, away = expected_goals(goals_inputs(10, 2.0, 2.0))
    shrunk = (10 * 2.0 + GOALS_ONLY_PRIOR_MATCHES * GOALS_ONLY_PRIOR_GOALS) / (10 + GOALS_ONLY_PRIOR_MATCHES)
    assert home == pytest.approx(shrunk) and away == pytest.approx(shrunk) and shrunk == pytest.approx(1.65)
    xg_home, xg_away = expected_goals(xg_inputs(2.0, 2.0))
    assert xg_home + xg_away == pytest.approx(4.0) and home + away < xg_home + xg_away
    # a bigger sample trusts the data more: with 40 matches the shrinkage is weaker
    big = expected_goals(goals_inputs(40, 2.0, 2.0))
    assert home < big[0] < xg_home
    # an average team gets the prior back whatever the sample size
    avg = expected_goals(goals_inputs(10, GOALS_ONLY_PRIOR_GOALS, GOALS_ONLY_PRIOR_GOALS))
    assert avg[0] == pytest.approx(GOALS_ONLY_PRIOR_GOALS)


def test_below_the_minimum_sample_the_engine_gives_no_estimate_at_all():
    f = fixture(markets=[over("1.5", "1.45")])
    assert SoccerEngine().evaluate(f, goals_inputs(matches=7), f.markets) == []


def test_goals_only_picks_are_marked_with_the_model_name_and_the_floor_and_ev_gate_still_apply():
    assert 0 < PROBABILITY_FLOOR <= 1
    f = fixture(markets=[over("1.5", "1.45"), over("2.5", "1.60"), over("1.5", "1.20")])
    evaluations = SoccerEngine().evaluate(f, goals_inputs(10, 2.0, 1.5), f.markets)
    assert evaluations and all(e.estimate.model == GOALS_ONLY_MODEL_NAME for e in evaluations)
    passed = [e for e in evaluations if e.passed]
    assert passed and all(e.estimate.probability >= PROBABILITY_FLOOR and e.ev > 0 for e in passed)
    low_odds = [e for e in evaluations if e.selection == "Over 1.5" and e.odds == Decimal("1.20")][0]
    assert low_odds.estimate.probability >= PROBABILITY_FLOOR and not low_odds.passed and "expected value" in low_odds.reason
    assert not [e for e in evaluations if "Under" in e.selection]   # owner rule: no Under in any sport (AD-20)
    # the same data through the xG model is labelled differently and is more extreme
    xg = SoccerEngine().evaluate(f, xg_inputs(2.0, 1.5), f.markets)
    assert all(e.estimate.model != GOALS_ONLY_MODEL_NAME for e in xg)
    p = lambda evs: [e for e in evs if e.selection == "Over 2.5" and e.odds == Decimal("1.60")][0].estimate.probability
    assert p(evaluations) < p(xg)


def test_the_estimate_is_deterministic():
    f = fixture(markets=[over("2.5", "1.90")])
    first = SoccerEngine().evaluate(f, goals_inputs(), f.markets)
    assert first == SoccerEngine().evaluate(f, goals_inputs(), f.markets)


def test_the_floor_is_still_defined_only_in_the_model():
    engine_source = (Path(__file__).resolve().parent.parent / "predictions" / "core" / "engines" / "soccer.py").read_text(
        encoding="utf-8")
    assert "PROBABILITY_FLOOR =" not in engine_source and "import" in engine_source


# ---------------------------------------------------------------- the labels
def gate_for(markets, inputs):
    f = fixture(markets=markets)
    stats = {f.match_id: FixtureStats(f.match_id, READY, tuple(inputs))}
    return f, run_gate([f], stats)


def test_every_goals_only_pick_line_says_so_and_xg_lines_do_not():
    f, gate = gate_for([over("1.5", "1.45"), over("3.5", "1.45", "1.90")], goals_inputs())
    assert gate.passed and gate.rejected
    for ev in gate.passed:
        assert render.evaluation_line(gate, ev).count(f"[{GOALS_ONLY_LABEL}]") == 1
    for ev in gate.rejected:
        assert f"[{GOALS_ONLY_LABEL}]" in render.evaluation_line(gate, ev, True)
    assert GOALS_ONLY_LABEL == "goals-only estimate (no xG)"
    f2, xg_gate = gate_for([over("1.5", "1.45")], xg_inputs())
    assert all("goals-only" not in render.evaluation_line(xg_gate, e) for e in [*xg_gate.passed, *xg_gate.rejected])


def test_the_matchday_reply_marks_a_goals_only_pick():
    f = fixture(markets=[over("1.5", "1.45")])

    class Listing:
        async def fetch_fixtures(self):
            return [f]

    class Stats:
        async def fetch_stats(self, fixtures):
            return {f.match_id: FixtureStats(f.match_id, READY, tuple(goals_inputs()))}
    gates = []
    chunks = asyncio.run(run_matchday("1 October", Listing(), str, NOW, stats_source=Stats(), on_result=gates.append))
    shown = list_text(gates[0].passed)
    assert chunks == [] and "Over 1.5 goals @ 1.45" in shown and "[goals-only]" in shown   # the safety label stays on the list
    assert GOALS_ONLY_LABEL in detail_text(gates[0].passed)                          # and the full wording one tap away


def test_the_review_output_and_the_final_list_show_the_label_too():
    f, gate = gate_for([over("1.5", "1.45")], goals_inputs())
    ev = gate.passed[0]
    selection = SimpleNamespace(kickoff_utc=KICKOFF, home="A", away="B", sport_label="Soccer", competition="C",
                                market_name="Over/Under", line="1.5", outcome_name="Over", odds=Decimal("1.60"))
    judged = SimpleNamespace(evaluation=ev, selection=selection, approved=True)
    lines = render._judged_lines(judged, with_inputs=True)
    assert f"    {GOALS_ONLY_LABEL}" in lines and any(GOALS_ONLY_MODEL_NAME in line for line in lines)
    pick = SimpleNamespace(evaluation=ev)
    assert slate._success(pick).endswith(f"[{GOALS_ONLY_LABEL}]")
    assert "goals-only" not in slate._success(SimpleNamespace(evaluation=SimpleNamespace(
        estimate=Estimate(0.7, (), "poisson-totals-v1"))))


# ---------------------------------------------------------------- the composite source
class Fixed:
    def __init__(self, answers=None, error=None):
        self.answers, self.error, self.asked = answers or {}, error, []

    async def fetch_stats(self, fixtures):
        self.asked.append([f.match_id for f in fixtures])
        if self.error:
            raise self.error
        return {f.match_id: self.answers[f.match_id] for f in fixtures if f.match_id in self.answers}


def covered(f, tag="x"):
    return FixtureStats(f.match_id, READY, (Input("home.matches", 10.0, "matches", tag),))


def not_covered(f):
    return insufficient(f.match_id, "this competition is not covered by the stats source")


def test_understat_answers_first_and_espn_gets_only_what_is_left():
    epl, nations, both = fixture("Arsenal", "Leeds", "England - Premier League"), fixture(), fixture("X", "Y")
    first = Fixed({epl.match_id: covered(epl, "understat"), nations.match_id: not_covered(nations),
                   both.match_id: covered(both, "understat")})
    second = Fixed({nations.match_id: covered(nations, "espn"), both.match_id: covered(both, "espn")})
    result = asyncio.run(FirstCoveringStats([first, second]).fetch_stats([epl, nations, both]))
    assert result[epl.match_id].inputs[0].source == "understat"
    assert result[both.match_id].inputs[0].source == "understat"   # covered by both: the first (xG) source wins
    assert result[nations.match_id].inputs[0].source == "espn"
    assert second.asked == [[nations.match_id]]


def test_a_specific_problem_from_the_first_source_is_not_second_guessed():
    f = fixture("Arsenal", "Leeds", "England - Premier League")
    first = Fixed({f.match_id: insufficient(f.match_id, "too little history")})
    second = Fixed({f.match_id: covered(f)})
    result = asyncio.run(FirstCoveringStats([first, second]).fetch_stats([f]))
    assert result[f.match_id].status == INSUFFICIENT and second.asked == []


def test_if_one_source_is_down_the_other_still_answers_and_the_gap_is_reported_as_unreadable():
    epl, nations = fixture("Arsenal", "Leeds", "England - Premier League"), fixture()
    down = Fixed(error=StatsFetchError("Understat returned HTTP 500"))
    second = Fixed({nations.match_id: covered(nations, "espn"), epl.match_id: not_covered(epl)})
    result = asyncio.run(FirstCoveringStats([down, second]).fetch_stats([epl, nations]))
    assert result[nations.match_id].status == READY
    assert result[epl.match_id].status == INSUFFICIENT and "could not be read" in result[epl.match_id].reasons[0]
    assert "not covered" not in result[epl.match_id].reasons[0]
    espn_down = Fixed(error=StatsFetchError("ESPN returned HTTP 500"))
    first = Fixed({epl.match_id: covered(epl, "understat"), nations.match_id: not_covered(nations)})
    result = asyncio.run(FirstCoveringStats([first, espn_down]).fetch_stats([epl, nations]))
    assert result[epl.match_id].status == READY and "could not be read" in result[nations.match_id].reasons[0]


def test_when_no_source_can_be_read_the_error_is_raised():
    f = fixture()
    both = [Fixed(error=StatsFetchError("a")), Fixed(error=StatsFetchError("b"))]
    with pytest.raises(StatsFetchError):
        asyncio.run(FirstCoveringStats(both).fetch_stats([f]))


def test_a_fixture_no_source_covers_stays_not_covered():
    f = fixture(competition="Israel - National League")
    result = asyncio.run(FirstCoveringStats([Fixed({f.match_id: not_covered(f)}), Fixed({f.match_id: not_covered(f)})]).fetch_stats([f]))
    assert result[f.match_id].reasons == ("this competition is not covered by the stats source",)


# ---------------------------------------------------------------- the wiring in app.py
def test_matchday_and_review_use_understat_first_then_espn():
    source = bot.soccer_stats_source()
    assert isinstance(source, FirstCoveringStats)
    assert [type(s) for s in source._sources] == [UnderstatStats, EspnSoccerStats]
    assert isinstance(bot.review_stats_sources()["soccer"], FirstCoveringStats)


def test_the_matchday_trigger_hands_the_composite_source_to_the_run(monkeypatch):
    captured = {}

    async def fake_run(argument, source, sanitize, **kwargs):
        captured.update(kwargs)
        return ["ok"]

    monkeypatch.setattr(bot, "run_matchday", fake_run)
    monkeypatch.setattr(bot.ClaudeAnalyst, "from_environment", classmethod(lambda cls, mode="full": None))
    sent = []

    async def reply_text(message, **kwargs):
        sent.append(message)

    update = SimpleNamespace(effective_user=SimpleNamespace(id=bot.OWNER_ID), callback_query=None,
                             message=SimpleNamespace(text="Matchday today", reply_text=reply_text))
    asyncio.run(handled(update))
    drop_ack(sent)
    assert isinstance(captured["stats_source"], FirstCoveringStats) and sent == ["ok"]


def test_the_competition_table_covers_what_the_owner_saw_on_the_slate():
    from predictions.core.types import normalize_team
    seen = ["International - UEFA Nations League", "International Youth - U21 UEFA European Championship, Qualification",
            "International Clubs - UEFA Champions League Women", "International Clubs - Club Friendly Games",
            "International - Int. Friendly Games", "International - CONCACAF Nations League"]
    assert all(normalize_team(name) in espn.COMPETITIONS for name in seen)
    assert {espn.COMPETITIONS[normalize_team(n)] for n in seen} == {
        "uefa.nations", "uefa.euro_u21_qual", "uefa.wchampions", "club.friendly", "fifa.friendly", "concacaf.nations.league"}
    assert team_inputs("home", [], "s") == []


# ---------------------------------------------------------------- the "too good to be true" EV cap
def _over25_probability(inputs):
    f = fixture(markets=[over("2.5", "2.00")])
    return SoccerEngine().evaluate(f, inputs, f.markets)[0].estimate.probability


def _odds_for_ev(p, target):
    """Odds (2 decimals) whose EV for probability p is just under/over the target."""
    return Decimal(str(round((1 + target) / p, 2)))


def _over_at_ev(inputs, target):
    p = _over25_probability(inputs)
    odds = _odds_for_ev(p, target)
    f = fixture(markets=[over("2.5", str(odds), "9.00")])
    ev = [e for e in SoccerEngine().evaluate(f, inputs, f.markets) if e.selection == "Over 2.5"][0]
    return f, ev


def test_the_cap_is_a_named_constant_and_the_floor_stays_in_the_model():
    assert GOALS_ONLY_MAX_EV == 0.20 and "20%" in GOALS_ONLY_CAP_REASON.format(cap=GOALS_ONLY_MAX_EV)


def test_a_goals_only_pick_at_plus_19_passes_and_at_plus_21_is_rejected_with_the_reason():
    inputs = goals_inputs(10, 3.5, 2.5)
    assert _over25_probability(inputs) >= PROBABILITY_FLOOR
    _, low = _over_at_ev(inputs, 0.19)
    _, high = _over_at_ev(inputs, 0.21)
    assert Decimal("0.18") < low.ev <= Decimal("0.20") and low.passed
    assert high.ev > Decimal("0.20") and not high.passed
    assert "more than 20% EV" in high.reason and "unreliable" in high.reason and "no opponent strength" in high.reason


def test_the_xg_path_is_not_capped():
    inputs = xg_inputs(3.5, 2.5)
    p = _over25_probability(inputs)
    assert p >= PROBABILITY_FLOOR
    _, ev = _over_at_ev(inputs, 0.40)
    assert ev.ev > Decimal("0.35") and ev.passed


def test_capped_picks_appear_in_the_closest_rejections_and_the_reply_with_the_reason():
    inputs = goals_inputs(10, 3.5, 2.5)
    f, ev = _over_at_ev(inputs, 0.45)
    assert not ev.passed
    f, gate = gate_for([over("2.5", str(ev.odds), "9.00")], inputs)
    assert not gate.passed
    text = chr(10).join(render.gate_lines(gate))
    assert "Closest rejections" in text and "treated as unreliable" in text


def test_team_totals_and_converted_picks_are_capped_too():
    inputs = goals_inputs(10, 3.5, 2.5)
    team = MarketOffer("19", "Home team Over/Under", "total=0.5",
                       (Outcome("Over 0.5", Decimal("1.95")), Outcome("Under 0.5", Decimal("9.00"))))
    f = fixture(markets=[team])
    # Spec 11-1: Matchday does not consider team totals at all; Review (evaluate_exact) still prices and caps them.
    assert SoccerEngine().evaluate(f, inputs, f.markets) == []
    exact = SoccerEngine().evaluate_exact(f, inputs, team)
    assert exact and any(not e.passed and "unreliable" in e.reason for e in exact if e.selection.endswith("Over 0.5"))
    # converted picks: a 1X2 home win under the floor is tried as an Over goal line (the swap rule; never a team total)
    home_strong = [Input(f"{side}.{stat}", float(v), u, "t")
                   for side, (gf, ga) in (("home", (3.5, 1.5)), ("away", (2.2, 2.5)))
                   for stat, v, u in (("matches", 10, "matches"), ("goals_for", gf, "goals per match"),
                                      ("goals_against", ga, "goals per match"))]
    result = MarketOffer("1", "1X2", "", (Outcome("Home", Decimal("2.40")), Outcome("Draw", Decimal("6.0")),
                                          Outcome("Away", Decimal("7.00"))))

    def swapped(total_odds):
        big = over("2.5", total_odds, "1.30")
        f = fixture(markets=[result, big, team])
        found = SoccerEngine().evaluate(f, home_strong, f.markets)
        assert not any("team" in e.selection.lower() for e in found)   # never a team total in Matchday
        return [e for e in found if e.rule.startswith("1X2 swap")]
    capped = swapped("3.00")
    assert capped and not any(e.passed for e in capped)
    assert any("treated as unreliable" in e.reason for e in capped)
    fine = swapped("1.50")
    assert any(e.passed and e.selection == "Over 2.5" and e.ev <= Decimal("0.20") for e in fine)


def test_review_of_a_goals_only_selection_is_disqualified_with_the_reason():
    inputs = goals_inputs(10, 3.5, 2.5)
    f, ev = _over_at_ev(inputs, 0.45)
    offer = over("2.5", str(ev.odds), "9.00")
    found = SoccerEngine().evaluate_exact(fixture(markets=[offer]), inputs, offer)
    capped = [e for e in found if e.selection == "Over 2.5"][0]
    assert not capped.passed and "unreliable" in capped.reason
    judged = SimpleNamespace(evaluation=capped, approved=False, selection=SimpleNamespace(
        kickoff_utc=KICKOFF, home="A", away="B", sport_label="Soccer", competition="C", market_name="Over/Under",
        line="2.5", outcome_name="Over", odds=ev.odds))
    assert any("disqualified" in line and "unreliable" in line for line in render._judged_lines(judged, True))
