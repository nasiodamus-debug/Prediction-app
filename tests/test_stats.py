"""Story 2.2: soccer stats for each fixture, with sources (inputs, identity, insufficient data)."""
import asyncio
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from openmarket import open_markets
from predictions.adapters.understat.stats import UnderstatStats, parse_league, season_of
from predictions.core.identity import resolve_team
from predictions.core.inputs import Input, MIN_MATCHES, SOCCER_REGISTRY, check_inputs
from predictions.core.ports import StatsFetchError
from predictions.core.stats import AMBIGUOUS, INSUFFICIENT, READY, assess
from predictions.core.types import Fixture, MatchId
from predictions.flow.matchday import run_matchday
from predictions.flow.stats import evaluable, gather_stats, summarize

SAMPLE = json.loads((Path(__file__).parent / "data" / "understat_epl_2026_sample.json").read_text(encoding="utf-8"))
NOW = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
EMPTY = {"teams": {}, "players": [], "dates": []}
KICKOFF = datetime(2026, 10, 10, 11, 30, tzinfo=timezone.utc)


def fixture(home, away, kickoff=KICKOFF, competition="England - Premier League"):
    return Fixture(MatchId("soccer", competition, kickoff, home, away), competition, home, away, kickoff, False,
                   open_markets())


def previous_season(payload):
    """The sample moved back a year: the same teams' earlier history."""
    old = copy.deepcopy(payload)
    for team in old["teams"].values():
        for row in team["history"]:
            row["date"] = "2025" + row["date"][4:]
    return old


def source_for(payloads, calls=None):
    """An adapter whose HTTP is a fake; payloads maps season -> JSON (or an int status)."""
    def handler(request):
        if calls is not None:
            calls.append(str(request.url))
        season = int(str(request.url).rstrip("/").split("/")[-1])
        value = payloads.get(season, EMPTY)
        if isinstance(value, int):
            return httpx.Response(value)
        return httpx.Response(200, json=value)
    return UnderstatStats(httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def fetch(source, fixtures):
    return asyncio.run(source.fetch_stats(fixtures))


def good_inputs(required_only=False):
    return [Input(f"{side}.{spec.stat}", max(spec.low, 1.0), spec.unit, "test")
            for side in ("home", "away") for spec in SOCCER_REGISTRY.values()
            if spec.required or not required_only]


# ---- AC1: each input has name, value, unit, source ----
def test_inputs_carry_name_value_unit_and_source():
    f = fixture("Arsenal", "Leeds United")
    stats = fetch(source_for({2026: SAMPLE, 2025: previous_season(SAMPLE)}), [f])[f.match_id]
    assert stats.status == READY
    names = {i.name for i in stats.inputs}
    assert {"home.xg_for", "home.goals_against", "away.xg_against", "away.matches", "home.ppda", "away.deep"} <= names
    for item in stats.inputs:
        assert item.name and item.unit and isinstance(item.value, float)
    assert {i.source for i in stats.inputs} == {"understat:EPL:2026+2025"}  # history came from both seasons
    assert next(i for i in stats.inputs if i.name == "home.matches").value == 10


def test_values_are_averages_of_the_teams_own_matches():
    f = fixture("Arsenal", "Leeds United")
    stats = fetch(source_for({2026: SAMPLE, 2025: EMPTY}), [f])[f.match_id]
    arsenal = next(t for t in SAMPLE["teams"].values() if t["title"] == "Arsenal")["history"]
    expected = sum(float(r["xG"]) for r in arsenal) / len(arsenal)
    got = next(i for i in stats.inputs if i.name == "home.xg_for")
    assert got.value == pytest.approx(expected, abs=1e-3) and got.unit == "xG per match"
    assert got.source == "understat:EPL:2026+2025"


def test_only_matches_before_kickoff_count():
    f = fixture("Arsenal", "Leeds United", datetime(2026, 8, 22, 12, 0, tzinfo=timezone.utc))
    payload = copy.deepcopy(SAMPLE)
    payload["dates"].append({"isResult": False, "h": {"title": "Arsenal"}, "a": {"title": "Leeds"},
                             "datetime": "2026-08-22 12:00:00"})
    stats = fetch(source_for({2026: payload}), [f])[f.match_id]
    assert stats.status == INSUFFICIENT  # only matches played before that date count: fewer than 5


# ---- AC1b: one identity resolver ties both sources to the same game ----
def test_resolver_exact_alias_and_partial():
    teams = ["Manchester United", "Manchester City", "Leeds", "Bayern Munich", "Inter", "AC Milan", "Paris FC",
             "Paris Saint Germain"]
    assert resolve_team("manchester UNITED", teams).method == "exact"
    assert resolve_team("Bayern München", teams).team == "Bayern Munich"
    assert resolve_team("Inter Milan", teams).team == "Inter"
    assert resolve_team("PSG", teams).team == "Paris Saint Germain"
    assert resolve_team("Leeds FC", teams).method == "partial"
    assert resolve_team("Leeds FC", teams).team == "Leeds"


def test_resolver_never_guesses_between_teams():
    teams = ["Manchester United", "Newcastle United", "Paris FC", "Paris Saint Germain"]
    united = resolve_team("United", teams)
    assert united.status == "ambiguous" and united.matches == ("Manchester United", "Newcastle United")
    assert resolve_team("Paris", teams).status == "ambiguous"
    assert resolve_team("Chelsea", teams).status == "none"


def test_resolver_does_not_merge_youth_or_women_teams_with_the_club():
    assert resolve_team("Arsenal U21", ["Arsenal"]).status == "none"
    assert resolve_team("Arsenal Women", ["Arsenal"]).status == "none"
    assert resolve_team("Bayern Munich II", ["Bayern Munich"]).status == "none"


def test_season_starts_in_july():
    assert season_of(datetime(2026, 10, 10, tzinfo=timezone.utc)) == 2026
    assert season_of(datetime(2027, 3, 1, tzinfo=timezone.utc)) == 2026


# ---- AC2: ambiguity is a question, never a guess ----
def test_ambiguous_team_becomes_a_question_and_is_not_evaluated():
    f = fixture("United", "Tottenham")
    stats = fetch(source_for({2026: SAMPLE}), [f])[f.match_id]
    assert stats.status == AMBIGUOUS and not stats.ready and stats.inputs == ()
    assert set(stats.options) == {"Manchester United", "Newcastle United"}
    assert "United" in stats.question


def test_two_games_that_fit_are_ambiguous():
    payload = copy.deepcopy(SAMPLE)
    payload["dates"].append({"isResult": False, "h": {"title": "Arsenal"}, "a": {"title": "Leeds"},
                             "datetime": "2026-10-10 20:00:00"})
    f = fixture("Arsenal", "Leeds")
    stats = fetch(source_for({2026: payload}), [f])[f.match_id]
    assert stats.status == AMBIGUOUS and len(stats.options) == 2


def test_both_names_resolving_to_one_team_is_ambiguous():
    f = fixture("Leeds", "Leeds United")
    assert fetch(source_for({2026: SAMPLE}), [f])[f.match_id].status == AMBIGUOUS


# ---- AC3: missing data means "insufficient data" ----
def test_uncovered_league_is_insufficient_and_makes_no_request():
    calls = []
    f = fixture("Enyimba", "Rangers", competition="Nigeria - NPFL")
    stats = fetch(source_for({}, calls), [f])[f.match_id]
    assert stats.status == INSUFFICIENT and "not covered" in stats.reasons[0] and calls == []


def test_unknown_team_and_missing_game_are_insufficient():
    a = fixture("Chelsea", "Leeds United")
    b = fixture("Arsenal", "Leeds United", datetime(2026, 10, 14, 12, 0, tzinfo=timezone.utc))
    out = fetch(source_for({2026: SAMPLE}), [a, b])
    assert out[a.match_id].status == INSUFFICIENT and "Chelsea" in out[a.match_id].reasons[0]
    assert out[b.match_id].status == INSUFFICIENT and "no Arsenal v Leeds game" in out[b.match_id].reasons[0]


def test_season_not_started_is_insufficient():
    f = fixture("Arsenal", "Leeds United")
    assert fetch(source_for({}), [f])[f.match_id].status == INSUFFICIENT


def test_too_little_history_is_insufficient():
    payload = copy.deepcopy(SAMPLE)
    for team in payload["teams"].values():
        team["history"] = team["history"][:MIN_MATCHES - 1]
    f = fixture("Arsenal", "Leeds United")
    stats = fetch(source_for({2026: payload}), [f])[f.match_id]
    assert stats.status == INSUFFICIENT and "home.matches" in stats.reasons[0]


def test_missing_required_input_marks_insufficient_and_absurd_values_are_discarded():
    good = good_inputs()
    mid = fixture("A", "B").match_id
    assert assess(mid, good).status == READY
    result = assess(mid, [i for i in good if i.name != "away.xg_for"])
    assert result.status == INSUFFICIENT and "away.xg_for" in result.reasons[0]
    absurd = [Input("away.xg_for", 55.0, "xG per match", "test") if i.name == "away.xg_for" else i for i in good]
    result = assess(mid, absurd)
    assert result.status == INSUFFICIENT and any("outside" in r for r in result.reasons)


def test_optional_input_may_be_absent():
    assert assess(fixture("A", "B").match_id, good_inputs(required_only=True)).status == READY


def test_adapter_input_beats_analyst_input_and_wrong_units_are_discarded():
    kept, discarded = check_inputs([
        Input("home.xg_for", 1.1, "xG per match", "analyst-sourced:someblog"),
        Input("home.xg_for", 1.6, "xG per match", "understat:EPL:2026"),
        Input("home.goals_for", 1.0, "goals", "understat:EPL:2026"),
        Input("home.mystery", 1.0, "x", "understat:EPL:2026"),
    ])
    assert [(i.name, i.value, i.source) for i in kept] == [("home.xg_for", 1.6, "understat:EPL:2026")]
    assert len(discarded) == 2


# ---- source failures are reported, not hidden ----
def test_http_failure_raises_a_safe_error():
    f = fixture("Arsenal", "Leeds United")
    with pytest.raises(StatsFetchError) as error:
        fetch(source_for({2026: 503}), [f])
    assert str(error.value) == "Understat returned HTTP 503"


@pytest.mark.parametrize("payload", [{"teams": [], "dates": []}, {"teams": {"1": {"title": "X"}}, "dates": []},
                                     {"teams": {}}, []])
def test_unexpected_shape_raises(payload):
    with pytest.raises(StatsFetchError):
        parse_league(payload)


# ---- flow ----
class FakeStats:
    def __init__(self, table=None, error=None):
        self.table, self.error = table or {}, error

    async def fetch_stats(self, fixtures):
        if self.error:
            raise self.error
        return {f.match_id: self.table[f.match_id] for f in fixtures if f.match_id in self.table}


class Listing:
    def __init__(self, fixtures):
        self.fixtures = fixtures

    async def fetch_fixtures(self):
        return self.fixtures


def test_flow_treats_an_unanswered_fixture_as_insufficient_and_filters_evaluable():
    ok, missing = fixture("A", "B"), fixture("C", "D")
    stats = asyncio.run(gather_stats([ok, missing], FakeStats({ok.match_id: assess(ok.match_id, good_inputs())})))
    assert stats[missing.match_id].status == INSUFFICIENT
    assert evaluable([ok, missing], stats) == [ok]
    assert summarize(stats) == (1, 1, 0)


def test_matchday_shows_which_fixtures_have_data_and_evaluates_only_ready_ones():
    later = datetime(2026, 10, 2, 14, 0, tzinfo=timezone.utc)
    a, b = fixture("Arsenal", "Leeds United", later), fixture("Enyimba", "Rangers", later, "Nigeria - NPFL")
    fake = FakeStats({a.match_id: assess(a.match_id, good_inputs()),
                      b.match_id: fetch(source_for({}), [b])[b.match_id]})
    text = "\n".join(asyncio.run(run_matchday(" 2 October", Listing([a, b]), str, NOW, stats_source=fake)))
    assert "data: ready" in text and "insufficient data, not evaluated for all 1 game(s): this competition is not covered" in text
    assert "Team stats: 1 ready, 1 insufficient data." in text
    assert "1 game(s) evaluated" in text   # spec 12-1: the game has an open priced market, so it is evaluated (and rejected here)


def test_matchday_reports_a_stats_outage_and_still_lists_games():
    a = fixture("Arsenal", "Leeds United", datetime(2026, 10, 2, 14, 0, tzinfo=timezone.utc))
    outage = FakeStats(error=StatsFetchError("Understat returned HTTP 503"))
    text = "\n".join(asyncio.run(run_matchday(" 2 October", Listing([a]), str, NOW, stats_source=outage)))
    assert "Arsenal vs Leeds United" in text and "Could not get team stats: Understat returned HTTP 503" in text


def test_matchday_question_is_shown_for_ambiguous_team():
    a = fixture("United", "Tottenham", datetime(2026, 10, 2, 14, 0, tzinfo=timezone.utc))
    stats = fetch(source_for({2026: SAMPLE}), [a])
    text = "\n".join(asyncio.run(run_matchday(" 2 October", Listing([a]), str, NOW, stats_source=FakeStats(stats))))
    assert "needs your choice" in text and "Manchester United" in text and "1 need your choice" in text


def test_matchday_without_a_stats_source_is_unchanged():
    a = fixture("Arsenal", "Leeds United", datetime(2026, 10, 2, 14, 0, tzinfo=timezone.utc))
    text = "\n".join(asyncio.run(run_matchday(" 2 October", Listing([a]), str, NOW)))
    assert "data:" not in text and "Team stats" not in text


def test_partial_match_never_pairs_a_shorter_name_with_a_longer_club():
    from predictions.core.identity import resolve_team
    paris = resolve_team("Paris FC", ["Paris Saint Germain", "Lyon"])
    assert paris.status == "ambiguous" and paris.team is None and paris.matches == ("Paris Saint Germain",)
    madrid = resolve_team("Real Madrid", ["Real Madrid Castilla"])
    assert madrid.status == "ambiguous" and madrid.team is None
    # exact names still win, and generic words are still ignored
    assert resolve_team("Real Madrid", ["Real Madrid", "Real Madrid Castilla"]).team == "Real Madrid"
    assert resolve_team("Arsenal FC", ["Arsenal"]).team == "Arsenal"
    assert resolve_team("Arsenal W", ["Arsenal"]).status == "none"
