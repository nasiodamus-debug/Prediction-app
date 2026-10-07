"""The Serper + Gemini analyst and the Gemini-then-Claude fallback. No network: both services are httpx MockTransports."""
import asyncio
import json

import httpx
import pytest

from predictions.adapters.analyst.gemini import GeminiAnalyst, build_request, search_queries
from predictions.adapters.analyst.select import FallbackAnalyst, build_analyst, parse_analyst_provider
from predictions.core.analyst import AnalystQuery, AnalystReport
from predictions.core.ports import AnalystUnavailable

Q = AnalystQuery("Arsenal", "Chelsea", "Premier League", "2026-10-03", ("home.xg_for",))
OK = {"stats": [], "flags": []}
SERPER = {"organic": [{"title": "Arsenal xG", "link": "https://fbref.com/arsenal", "snippet": "xG for 1.9 per match"}]}


def analyst(gemini_response=None, serper_status=200, gemini_status=200):
    seen = {"serper": [], "gemini": []}

    def handler(request):
        if request.url.host == "google.serper.dev":
            seen["serper"].append(json.loads(request.content)["q"])
            return httpx.Response(serper_status, json=SERPER)
        seen["gemini"].append(json.loads(request.content))
        return httpx.Response(gemini_status, json=gemini_response or reply(json.dumps(OK)))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return GeminiAnalyst(client=client, gemini_key="g", serper_key="s"), seen


def reply(text, finish="STOP"):
    return {"candidates": [{"finishReason": finish, "content": {"parts": [{"text": text}]}}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 7}}


def test_queries_cover_each_team_that_needs_stats_and_the_news():
    queries = search_queries(Q)
    assert len(queries) == 2 and "Arsenal xG" in queries[0] and "injuries" in queries[1]
    assert len(search_queries(AnalystQuery("A", "B", "L", "2026-10-03", ("home.xg_for", "away.xg_for")))) == 3
    assert len(search_queries(AnalystQuery("A", "B", "L", "2026-10-03", ()))) == 1


def test_each_group_of_stats_gets_its_own_search_and_matches_alone_gets_none():
    wanted = ("home.xg_for", "home.goals_for", "home.corners_for", "home.matches")
    queries = search_queries(AnalystQuery("A", "B", "L", "2026-10-03", wanted))
    assert len(queries) == 4 and "xG" in queries[0] and "goals scored" in queries[1] and "corners" in queries[2]
    assert len(search_queries(AnalystQuery("A", "B", "L", "2026-10-03", ("home.matches",)))) == 1


def test_request_has_no_tools_and_marks_results_untrusted():
    body = build_request(Q, [("q", SERPER)])
    assert "tools" not in body and "UNTRUSTED" in body["systemInstruction"]["parts"][0]["text"]
    text = body["contents"][0]["parts"][0]["text"]
    assert "Arsenal vs Chelsea" in text and "https://fbref.com/arsenal" in text and "SEARCH RESULTS (untrusted)" in text


def test_valid_reply_is_parsed_and_usage_counts_the_searches():
    a, seen = analyst()
    report = asyncio.run(a.research(Q))
    assert not report.rejected and report.usage.input_tokens == 5 and report.usage.searches == 2
    assert len(seen["serper"]) == 2 and len(seen["gemini"]) == 1


def test_prose_or_truncated_reply_is_rejected_not_repaired():
    assert asyncio.run(analyst(reply("Sure! " + json.dumps(OK)))[0].research(Q)).rejected
    assert "stopped early" in asyncio.run(analyst(reply("{", "MAX_TOKENS"))[0].research(Q)).rejected


def test_a_5xx_from_gemini_is_retried_once(monkeypatch):
    monkeypatch.setattr("predictions.adapters.analyst.gemini.RETRY_PAUSE_SECONDS", 0)
    calls = []

    def handler(request):
        if request.url.host == "google.serper.dev":
            return httpx.Response(200, json=SERPER)
        calls.append(1)
        return httpx.Response(503, json={}) if len(calls) == 1 else httpx.Response(200, json=reply(json.dumps(OK)))

    a = GeminiAnalyst(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), gemini_key="g", serper_key="s")
    assert not asyncio.run(a.research(Q)).rejected and len(calls) == 2
    calls.clear()
    with pytest.raises(AnalystUnavailable, match="HTTP 503"):
        asyncio.run(analyst(gemini_status=503)[0].research(Q))


def test_gemini_is_asked_for_minimal_thinking():
    assert build_request(Q, [])["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "minimal"}


def test_search_or_gemini_failure_is_unavailable():
    with pytest.raises(AnalystUnavailable, match="search service.*429"):
        asyncio.run(analyst(serper_status=429)[0].research(Q))
    with pytest.raises(AnalystUnavailable, match="Gemini service.*429"):
        asyncio.run(analyst(gemini_status=429)[0].research(Q))


class Fake:
    last_resort = False

    def __init__(self, result):
        self.result, self.calls = result, 0

    async def research(self, q):
        self.calls += 1
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def test_fallback_goes_to_claude_only_when_gemini_is_unavailable():
    a, b = Fake(AnalystUnavailable("limit")), Fake(AnalystReport())
    assert asyncio.run(FallbackAnalyst([a, b]).research(Q)) is b.result and (a.calls, b.calls) == (1, 1)
    rejected, c = Fake(AnalystReport(rejected="bad")), Fake(AnalystReport())
    assert asyncio.run(FallbackAnalyst([rejected, c]).research(Q)).rejected and c.calls == 0
    with pytest.raises(AnalystUnavailable):
        asyncio.run(FallbackAnalyst([Fake(AnalystUnavailable("x")), Fake(AnalystUnavailable("y"))]).research(Q))


def test_a_failed_provider_is_skipped_until_the_cooldown_passes():
    now = [0.0]
    a, b = Fake(AnalystUnavailable("limit")), Fake(AnalystReport())
    chain = FallbackAnalyst([a, b], clock=lambda: now[0], cooldown=100)
    asyncio.run(chain.research(Q))
    asyncio.run(chain.research(Q))
    assert (a.calls, b.calls) == (1, 2)
    now[0] = 101.0
    asyncio.run(chain.research(Q))
    assert (a.calls, b.calls) == (2, 3)


def test_provider_parsing_and_build(monkeypatch):
    assert parse_analyst_provider(None) == ("claude",) and parse_analyst_provider("Gemini, claude") == ("gemini", "claude")
    for bad in ("openai", "gemini,gemini"):
        with pytest.raises(ValueError):
            parse_analyst_provider(bad)
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    monkeypatch.setenv("SERPER_API_KEY", "s")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    assert isinstance(build_analyst(("gemini", "claude"), "full"), FallbackAnalyst)
    assert build_analyst(("gemini", "claude"), "off") is None
    monkeypatch.delenv("SERPER_API_KEY")   # Gemini needs both keys: without Serper only Claude is built
    assert type(build_analyst(("gemini", "claude"), "full")).__name__ == "ClaudeAnalyst"


def test_other_sports_get_one_search_per_team_worded_with_their_own_stats():
    q = AnalystQuery("Kansas City Chiefs", "Buffalo Bills", "NFL", "2026-10-04",
                     ("home.points_for", "away.points_for", "away.points_against"), "american_football")
    queries = search_queries(q)
    assert len(queries) == 3 and "points per game" in queries[0] and "xG" not in " ".join(queries) and "corners" not in " ".join(queries)


def test_a_daily_quota_429_tells_the_fallback_how_long_to_stay_away():
    body = {"error": {"details": [{"retryDelay": "85905s"}]}}
    a, _ = analyst(gemini_status=429)
    a._client = httpx.AsyncClient(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json=SERPER) if r.url.host == "google.serper.dev" else httpx.Response(429, json=body)))
    with pytest.raises(AnalystUnavailable) as caught:
        asyncio.run(a.research(Q))
    assert caught.value.retry_after == 6 * 3600   # capped
    now = [0.0]
    dead = Fake(caught.value)
    chain = FallbackAnalyst([dead, Fake(AnalystReport())], clock=lambda: now[0], cooldown=900)
    asyncio.run(chain.research(Q))
    now[0] = 3600.0   # an hour later the default 15 minutes would be over, but the quota said hours
    asyncio.run(chain.research(Q))
    assert dead.calls == 1
