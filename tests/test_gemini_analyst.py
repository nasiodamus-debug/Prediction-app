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


def stat(name, value=1.0):
    from predictions.core.inputs import Input
    return Input(name, value, "xG per match", "analyst-sourced:test")


def test_claude_is_asked_only_for_what_gemini_left_missing_and_the_answers_are_combined():
    q = AnalystQuery("Arsenal", "Chelsea", "Premier League", "2026-10-03", ("home.xg_for", "home.xg_against", "away.xg_for"))
    asked = []

    class Spy(Fake):
        async def research(self, query):
            asked.append(query.wanted)
            return await super().research(query)

    gemini, claude = Spy(AnalystReport((stat("home.xg_for"),))), Spy(AnalystReport((stat("home.xg_against"), stat("away.xg_for"))))
    report = asyncio.run(FallbackAnalyst([gemini, claude]).research(q))
    assert asked == [q.wanted, ("home.xg_against", "away.xg_for")]          # Claude sees only the gap
    assert {i.name for i in report.stats} == set(q.wanted) and not report.incomplete
    full = Spy(AnalystReport(tuple(stat(n) for n in q.wanted)))
    never = Spy(AnalystReport())
    asyncio.run(FallbackAnalyst([full, never]).research(q))
    assert never.calls == 0                                                  # nothing missing: no paid call


def test_claude_is_asked_when_gemini_found_nothing_was_discarded_or_was_unavailable():
    for first in (AnalystReport(), AnalystReport(rejected="bad"), AnalystUnavailable("limit")):
        a, b = Fake(first), Fake(AnalystReport((stat("home.xg_for"),)))
        report = asyncio.run(FallbackAnalyst([a, b], after_empty=True).research(Q))
        assert (a.calls, b.calls) == (1, 1) and [i.name for i in report.stats] == ["home.xg_for"] and not report.rejected
    with pytest.raises(AnalystUnavailable):
        asyncio.run(FallbackAnalyst([Fake(AnalystUnavailable("x")), Fake(AnalystUnavailable("y"))]).research(Q))
    both_bad, c = Fake(AnalystReport(rejected="bad")), Fake(AnalystReport(rejected="worse"))
    assert asyncio.run(FallbackAnalyst([both_bad, c]).research(Q)).rejected


def test_a_flags_only_query_is_answered_by_the_first_analyst_alone():
    flags_only = AnalystQuery("Arsenal", "Chelsea", "Premier League", "2026-10-03", ())
    a, b = Fake(AnalystReport()), Fake(AnalystReport())
    asyncio.run(FallbackAnalyst([a, b]).research(flags_only))
    assert (a.calls, b.calls) == (1, 0)


def test_an_answer_is_marked_incomplete_when_the_paid_tier_was_unreachable_for_the_missing_stats():
    a, b = Fake(AnalystReport((stat("home.xg_for"),))), Fake(AnalystUnavailable("down"))
    q = AnalystQuery("Arsenal", "Chelsea", "Premier League", "2026-10-03", ("home.xg_for", "home.xg_against"))
    report = asyncio.run(FallbackAnalyst([a, b]).research(q))
    assert [i.name for i in report.stats] == ["home.xg_for"] and report.incomplete


def test_only_a_paid_analysts_usage_is_counted_as_spend():
    from predictions.core.analyst import AnalystUsage
    free, paid = Fake(AnalystReport(usage=AnalystUsage(100, 10, 5))), Fake(AnalystReport(usage=AnalystUsage(7, 3, 2)))
    paid.paid = True
    usage = asyncio.run(FallbackAnalyst([free, paid], after_empty=True).research(Q)).usage
    assert (usage.input_tokens, usage.output_tokens, usage.searches) == (7, 3, 2)


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


# ---- cost controls on the paid tier: essential stats only, nothing after an empty Gemini answer, a per-request limit
def paid(result):
    fake = Fake(result)
    fake.paid = True
    fake.name = "claude"
    return fake


def free(result):
    fake = Fake(result)
    fake.name = "gemini"
    return fake


WIDE = AnalystQuery("Arsenal", "Chelsea", "Premier League", "2026-10-03",
                    ("home.xg_for", "home.goals_for", "away.xg_for", "away.goals_for"), "soccer",
                    essential=("home.goals_for", "away.goals_for"))


def test_claude_is_asked_only_for_the_essential_stats_that_are_still_missing():
    asked = []

    class Spy(Fake):
        paid = True
        name = "claude"

        async def research(self, query):
            asked.append(query.wanted)
            return AnalystReport((stat("home.goals_for"),))

    first = free(AnalystReport((stat("away.xg_for"),)))          # Gemini found one stat, so the game is worth a paid look
    chain = FallbackAnalyst([first, Spy(None)])
    asyncio.run(chain.research(WIDE))
    assert asked == [("home.goals_for", "away.goals_for")]        # not the xG the lower leagues never have
    nothing_essential = FallbackAnalyst([free(AnalystReport((stat("home.goals_for"), stat("away.goals_for")))), paid(AnalystReport())])
    report = asyncio.run(nothing_essential.research(WIDE))
    assert nothing_essential._analysts[1].calls == 0 and not report.incomplete   # only xG left: not worth paying for
    assert nothing_essential.skipped_nonessential == 1


def test_claude_is_not_asked_when_gemini_answered_but_found_nothing_unless_switched_on():
    a, b = free(AnalystReport()), paid(AnalystReport((stat("home.goals_for"),)))
    chain = FallbackAnalyst([a, b])
    report = asyncio.run(chain.research(WIDE))
    assert b.calls == 0 and chain.skipped_empty == 1 and not report.stats and not report.incomplete   # final, so it is kept for the day
    a2, b2 = free(AnalystReport()), paid(AnalystReport((stat("home.goals_for"),)))
    asyncio.run(FallbackAnalyst([a2, b2], after_empty=True).research(WIDE))
    assert b2.calls == 1
    a3, b3 = free(AnalystUnavailable("limit")), paid(AnalystReport((stat("home.goals_for"),)))
    asyncio.run(FallbackAnalyst([a3, b3]).research(WIDE))
    assert b3.calls == 1                                          # Gemini being down is not evidence that nothing exists


def test_the_request_limit_on_claude_holds_and_a_held_back_answer_is_not_kept():
    b = paid(AnalystReport((stat("home.goals_for"),)))
    chain = FallbackAnalyst([free(AnalystReport((stat("away.xg_for"),))), b], paid_limit=2)
    reports = [asyncio.run(chain.research(WIDE)) for _ in range(4)]
    assert b.calls == 2 and chain.skipped_cap == 2
    assert [r.incomplete for r in reports] == [False, False, True, True]
    zero = paid(AnalystReport())
    asyncio.run(FallbackAnalyst([free(AnalystUnavailable("x")), zero], paid_limit=0).research(WIDE))   # no error: held back, not "down"
    assert zero.calls == 0


def test_the_breakdown_names_each_analyst_and_why_claude_was_skipped():
    chain = FallbackAnalyst([free(AnalystReport()), paid(AnalystReport())], paid_limit=3)
    asyncio.run(chain.research(WIDE))
    text = chain.breakdown()
    assert "Gemini 1" in text and "1 game(s) where Gemini found nothing" in text and "Claude" in text
    assert FallbackAnalyst([free(AnalystReport())]).breakdown() == ""


def test_the_paid_policy_settings_parse_and_refuse_bad_values():
    from predictions.core.analyst import parse_claude_after_empty, parse_paid_call_limit
    assert parse_paid_call_limit(None) == 10 and parse_paid_call_limit("0") == 0 and parse_paid_call_limit(" 25 ") == 25
    for bad in ("-1", "x", "1001", "2.5"):
        with pytest.raises(ValueError):
            parse_paid_call_limit(bad)
    assert parse_claude_after_empty(None) is False and parse_claude_after_empty("TRUE") is True and parse_claude_after_empty("off") is False
    with pytest.raises(ValueError):
        parse_claude_after_empty("maybe")
