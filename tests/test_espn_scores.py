"""ESPN final scores for the weekly review. No network: the scoreboard is an httpx MockTransport."""
import asyncio

import httpx

from predictions.adapters.espn.scores import EspnScores, endpoints, parse_event, parse_match_id
from predictions.core.ttl_cache import STATS_CACHE


def event(home="Las Vegas Aces", away="Indiana Fever", hs="94", as_="83", status="STATUS_FINAL", lines=((31, 31), (23, 17), (16, 17), (24, 18))):
    return {"status": {"type": {"name": status}}, "competitions": [{"competitors": [
        {"homeAway": "home", "score": hs, "team": {"displayName": home}, "linescores": [{"value": h} for h, _ in lines]},
        {"homeAway": "away", "score": as_, "team": {"displayName": away}, "linescores": [{"value": a} for _, a in lines]}]}]}


def scores(events_by_call, calls=None):
    def handler(request):
        if calls is not None:
            calls.append(str(request.url))
        return httpx.Response(200, json={"events": events_by_call})
    return EspnScores(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


MID = "basketball:usa wnba:20261002T0100Z:las vegas aces:indiana fever"


def test_the_match_id_is_parsed_and_a_bad_one_is_refused():
    parsed = parse_match_id(MID)
    assert parsed[0] == "basketball" and parsed[1] == "usa wnba" and parsed[3:] == ("las vegas aces", "indiana fever")
    assert parse_match_id("nonsense") is None and parse_match_id("a:b:notadate:c:d") is None


def test_a_final_event_gives_the_score_with_its_period_scores():
    names = parse_event(event())
    assert names[:2] == ("Las Vegas Aces", "Indiana Fever")
    score = names[2]
    assert (score.home, score.away, score.status) == (94, 83, "final") and score.periods[:2] == ((31, 31), (23, 17))


def test_postponed_suspended_unfinished_and_odd_events():
    assert parse_event(event(status="STATUS_POSTPONED"))[2].status == "postponed"
    assert parse_event(event(status="STATUS_CANCELED"))[2].status == "abandoned"
    assert parse_event(event(status="STATUS_SUSPENDED"))[2].status == "suspended"
    assert parse_event(event(status="STATUS_IN_PROGRESS")) is None
    assert parse_event(event(status="STATUS_SCHEDULED")) is None
    assert parse_event(event(hs="x")) is None
    assert parse_event({"competitions": []}) is None
    assert parse_event(event(status="STATUS_FINAL_AET"))[2].extra_time is True


def test_the_game_is_found_by_both_team_names_and_the_score_returned():
    scale = STATS_CACHE.scale
    STATS_CACHE.set_scale(0)   # no remembered scoreboard between cases
    try:
        found, why = asyncio.run(scores([event(), event("New York Liberty", "Seattle Storm", "70", "60")]).final_score(MID))
        assert found.home == 94 and why == ""
        missing, why = asyncio.run(scores([event("New York Liberty", "Seattle Storm")]).final_score(MID))
        assert missing is None and "not on the scoreboard" in why
        wrong_away, _ = asyncio.run(scores([event("Las Vegas Aces", "Seattle Storm")]).final_score(MID))
        assert wrong_away is None
    finally:
        STATS_CACHE.set_scale(scale)


def test_the_day_before_the_kickoff_is_also_read_and_unknown_leagues_give_no_score():
    scale = STATS_CACHE.scale
    STATS_CACHE.set_scale(0)   # no remembered scoreboard between cases
    try:
        calls = []
        asyncio.run(scores([], calls).final_score(MID))
        days = sorted(c.split("dates=")[1][:8] for c in calls)
        assert days == ["20261001", "20261002"]
        none, why = asyncio.run(scores([]).final_score("basketball:turkiye super lig:20261002T1700Z:a:b"))
        assert none is None and "no free score source" in why
    finally:
        STATS_CACHE.set_scale(scale)


def test_a_failing_source_is_reported_as_unreachable_not_as_a_missing_game():
    scale = STATS_CACHE.scale
    STATS_CACHE.set_scale(0)   # no remembered scoreboard between cases
    try:
        client = EspnScores(client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(500, json={}))))
        none, why = asyncio.run(client.final_score(MID))
        assert none is None and "could not be reached" in why
    finally:
        STATS_CACHE.set_scale(scale)


def test_leagues_map_to_scoreboards():
    assert endpoints("basketball", "usa nba")[0][:2] == ("basketball", "nba")
    assert [e[2] for e in endpoints("american_football", "usa ncaa regular season")] == [{"groups": "80"}, {"groups": "81"}]
    assert endpoints("american_football", "usa nfl")[0][:2] == ("football", "nfl")
    assert endpoints("soccer", "england premier league")[0][:2] == ("soccer", "eng.1")
    assert endpoints("hockey", "usa nhl")[0][:2] == ("hockey", "nhl") and endpoints("baseball", "usa mlb")[0][:2] == ("baseball", "mlb")
    assert endpoints("basketball", "turkiye super lig") == [] and endpoints("tennis", "atp") == []
