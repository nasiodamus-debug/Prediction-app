"""Free sources first, the paid AI analyst last (research: _bmad-output/implementation-artifacts/free-sources-research.md).
ANALYST_MODE (off / last_resort / full) and the ESPN FCS scoreboard (groups=81). No network call is made."""
import asyncio
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from predictions.adapters.analyst.claude import ClaudeAnalyst
from predictions.adapters.espn import football as fb
from predictions.core.analyst import parse_analyst_mode
from predictions.core.stats import READY, assess, insufficient
from predictions.core.types import Fixture, MatchId
from test_analyst import FakeAnalyst, free_inputs, fixture, run
from test_football_stats import COLLEGE, NOW, week_board


# ---------------------------------------------------------------- ANALYST_MODE parsing
@pytest.mark.parametrize("raw, expected", [(None, "last_resort"), ("", "last_resort"), ("  ", "last_resort"),
                                           ("off", "off"), ("OFF", "off"), (" Last_Resort ", "last_resort"), ("full", "full")])
def test_analyst_mode_is_read_strictly_and_defaults_to_last_resort(raw, expected):
    assert parse_analyst_mode(raw) == expected


@pytest.mark.parametrize("raw", ["on", "lastresort", "last resort", "true", "0", "zzz-secret-9"])
def test_a_bad_analyst_mode_refuses_to_start(raw):
    with pytest.raises(ValueError, match="ANALYST_MODE"):
        parse_analyst_mode(raw)


def test_the_bad_mode_message_never_echoes_the_value():
    with pytest.raises(ValueError) as error:
        parse_analyst_mode("zzz-secret-9")
    assert "zzz-secret-9" not in str(error.value)


def test_off_builds_no_analyst_even_with_a_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    assert ClaudeAnalyst.from_environment("off") is None
    assert ClaudeAnalyst.from_environment("last_resort").last_resort is True
    assert ClaudeAnalyst.from_environment("full").last_resort is False


# ---------------------------------------------------------------- last resort in the flow
def last_resort(analyst):
    analyst.last_resort = True
    return analyst


def test_last_resort_makes_no_flags_only_call_for_a_game_the_free_sources_cover():
    fixtures = [fixture(f"T{i}", f"O{i}", hour=8 + i) for i in range(3)]
    analyst = last_resort(FakeAnalyst())
    run(fixtures, table=lambda x: assess(x.match_id, free_inputs()), analyst=analyst)
    assert analyst.queries == []                                   # full mode would make 3 flags-only calls


def test_full_mode_still_makes_the_flags_only_calls():
    fixtures = [fixture(f"T{i}", f"O{i}", hour=8 + i) for i in range(3)]
    analyst = FakeAnalyst()
    run(fixtures, table=lambda x: assess(x.match_id, free_inputs()), analyst=analyst)
    assert len(analyst.queries) == 3


def test_last_resort_still_asks_for_inputs_the_free_sources_lack():
    analyst = last_resort(FakeAnalyst())
    text, gate = run([fixture()], analyst=analyst)                 # not covered by any free source
    assert len(analyst.queries) == 1 and analyst.queries[0].wanted
    assert "analyst-sourced" in text.lower() or gate is not None


def test_last_resort_does_not_pay_for_a_team_whose_season_has_not_started():
    analyst = last_resort(FakeAnalyst())
    preseason = lambda f: insufficient(f.match_id, "Home Team has no completed games in the last 28 days")
    run([fixture()], table=preseason, analyst=analyst)
    assert analyst.queries == []
    full = FakeAnalyst()
    run([fixture()], table=preseason, analyst=full)
    assert len(full.queries) == 1                                  # full mode keeps asking


# ---------------------------------------------------------------- FCS through ESPN groups=81
FBS_TEAM, FCS_HOME, FCS_AWAY = ("Liberty Flames", "1"), ("Yale Bulldogs", "2"), ("Columbia Lions", "3")


def college_event(eid, home, away, day):
    def side(kind, team):
        return {"homeAway": kind, "team": {"id": team[1], "displayName": team[0]}, "score": "28",
                "linescores": [{"value": 7.0}, {"value": 7.0}, {"value": 7.0}, {"value": 7.0}]}
    when = datetime(2026, 9, 5, 17, 0, tzinfo=timezone.utc) + timedelta(days=day)
    return {"id": str(eid), "date": when.strftime("%Y-%m-%dT%H:%MZ"), "season": {"year": 2026, "type": 2},
            "status": {"type": {"completed": True}},
            "competitions": [{"venue": {"address": {"city": "New Haven"}},
                              "competitors": [side("home", home), side("away", away)]}]}


class FcsApi:
    """FBS scoreboard (groups=80) holds only FBS games; the FCS one (groups=81) holds the FCS games, as ESPN does."""

    def __init__(self):
        self.groups = []

    def handler(self, request: httpx.Request):
        params = dict(request.url.params)
        if not request.url.path.endswith("/scoreboard"):
            return httpx.Response(404)
        group = params.get("groups")
        self.groups.append(group)
        week = int(params.get("week", 5))
        if group == "81" and 1 <= week <= 4:
            return httpx.Response(200, json=week_board(week, [college_event(900 + week, FCS_HOME, FCS_AWAY, 7 * (week - 1))]))
        if group == "80" and 1 <= week <= 4:
            return httpx.Response(200, json=week_board(week, [college_event(800 + week, FBS_TEAM, ("Tulsa Golden Hurricane", "4"), 7 * (week - 1))]))
        return httpx.Response(200, json=week_board(week, []))

    def source(self):
        return fb.EspnFootballStats(httpx.AsyncClient(transport=httpx.MockTransport(self.handler)), clock=lambda: NOW)


def test_an_fcs_game_is_ready_from_the_fcs_scoreboard_with_no_analyst():
    api = FcsApi()
    kick = datetime(2026, 10, 3, 17, 0, tzinfo=timezone.utc)
    game = Fixture(MatchId("american_football", COLLEGE, kick, "Yale Bulldogs", "Columbia Lions"), COLLEGE,
                   "Yale Bulldogs", "Columbia Lions", kick, False)
    result = asyncio.run(api.source().fetch_stats([game]))[game.match_id]
    assert result.status == READY, result.reasons
    assert {"80", "81"} <= set(api.groups)                         # both divisions are read and merged
