"""Stories 4.1 and 4.2: Review a SportyBet booking code.

Nothing here uses the network. `tests/data/sportybet_share_code_sample.json` is FABRICATED: no real booking code was
available, so its success shape is what the adapter assumes SportyBet returns (see spec-4-1). The invalid-code sample
is the real answer to a made-up code, recorded on 2026-09-30.
"""
import asyncio
import json
from dataclasses import replace
from predictions.core.dates import LAGOS
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import app as bot
from handled import drop_ack, handled
from predictions.adapters.sportybet import code as code_adapter
from predictions.adapters.sportybet.code import SportyBetCodeReader, parse_selections
from predictions.core import engines
from predictions.core.engines.soccer import SoccerEngine
from predictions.core.inputs import SOCCER_REGISTRY, Input
from predictions.core.model import PROBABILITY_FLOOR, Evaluation
from predictions.core.ports import CodeReadError, StatsFetchError
from predictions.core.review import CodeSelection, judge_selection
from predictions.core.stats import READY, FixtureStats, insufficient
from predictions.core.types import Fixture, MarketOffer, MatchId, Outcome
from predictions.flow.review import ASK_FOR_CODE, BAD_CODE, run_review
from predictions.flow.store import EvaluationStore
from predictions.telegram.render import MESSAGE_LIMIT

DATA = Path(__file__).resolve().parent / "data"
NOW = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
SAMPLE = json.loads((DATA / "sportybet_share_code_sample.json").read_text(encoding="utf-8"))
INVALID = json.loads((DATA / "sportybet_share_code_invalid.json").read_text(encoding="utf-8"))


def selections():
    return parse_selections(SAMPLE)


def inputs_for(attack, defence, source="test-source"):
    values = {"matches": 10, "xg_for": attack, "goals_for": attack, "xg_against": defence, "goals_against": defence}
    return [Input(f"{side}.{stat}", float(v), SOCCER_REGISTRY[stat].unit, source)
            for side in ("home", "away") for stat, v in values.items()]


class FakeCodeReader:
    def __init__(self, result=None, error=None):
        self.result, self.error, self.codes = result, error, []

    async def read_code(self, code):
        self.codes.append(code)
        if self.error:
            raise self.error
        return list(self.result)


class FakeStats:
    """Ready, high-scoring stats for Arsenal, even stats for Chelsea, nothing for anybody else."""

    def __init__(self):
        self.asked = []

    async def fetch_stats(self, fixtures):
        self.asked.extend(f.home for f in fixtures)
        table = {}
        for f in fixtures:
            if f.home == "Arsenal":
                table[f.match_id] = FixtureStats(f.match_id, READY, tuple(inputs_for(2.5, 2.0)))
            elif f.home in ("Chelsea", "Sevilla"):
                table[f.match_id] = FixtureStats(f.match_id, READY, tuple(inputs_for(1.4, 1.4)))
            else:
                table[f.match_id] = insufficient(f.match_id, "missing required input: home.xg_for")
        return table


def review(reader=None, argument=" ABC123", stats=None, sources=None, **kwargs):
    reader = reader or FakeCodeReader(selections())
    stats = stats or FakeStats()
    seen = []
    chunks = asyncio.run(run_review(argument, reader, str, stats_sources=sources if sources is not None else {"soccer": stats}, now=NOW,
                                    on_result=seen.append, **kwargs))
    return chunks, seen, reader, stats


def text_of(chunks):
    return "\n".join(chunks)


# ================================================================ Story 4.1: reading the code
def test_the_sample_code_is_read_with_game_sport_market_line_and_odds():
    read = selections()
    assert len(read) == 7
    first = read[0]
    assert (first.sport, first.sport_label, first.home, first.away) == ("soccer", "Football", "Arsenal", "Leeds United")
    assert first.competition == "England - Premier League"  # the same form the fixture listing uses
    assert (first.market_id, first.market_name, first.line, first.outcome_name, first.odds) == \
        ("18", "Over/Under", "2.5", "Over 2.5", Decimal("1.85"))
    assert [s.sport for s in read] == ["soccer", "soccer", "soccer", "soccer", "basketball", "tennis", "soccer"]
    assert first.kickoff_utc == datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc)


def test_the_review_lists_every_selection_with_game_sport_market_line_and_odds():
    chunks, *_ = review()
    text = text_of(chunks)
    assert "7 selection(s) read" in text
    assert "Arsenal vs Leeds United [Football, England - Premier League]" in text
    assert "Over/Under line 2.5: Over 2.5 @ 1.85" in text
    assert "[Basketball, USA - NBA]" in text and "Total line 220.5: Over 220.5 @ 1.90" in text
    assert "[Tennis, ATP - US Open]" in text
    assert "Nothing was changed on your betslip" in text


def test_the_reader_makes_one_anonymous_get_to_the_public_share_endpoint():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json=SAMPLE)

    reader = SportyBetCodeReader(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    read = asyncio.run(reader.read_code("ABC123"))
    assert len(read) == 7 and len(requests) == 1
    request = requests[0]
    assert request.method == "GET" and str(request.url) == "https://www.sportybet.com/api/ng/orders/share/ABC123"
    assert not {"cookie", "authorization"} & {k.lower() for k in request.headers}  # no login, no session


def test_no_browser_and_no_login_anywhere_in_the_reader():
    source = Path(code_adapter.__file__).read_text(encoding="utf-8").lower()
    assert "playwright" not in source and "password" not in source and "sporty_pass" not in source
    assert not hasattr(bot, "scrape_sportybet_code")  # the legacy Playwright reader is gone


@pytest.mark.parametrize("bad", ["../x", "AB", "a b", "ABC123?x=1", "x" * 30, "AB-CD", ""])
def test_a_malformed_code_never_reaches_the_network(bad):
    def handler(request):
        raise AssertionError("a request was made")

    reader = SportyBetCodeReader(httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    with pytest.raises(CodeReadError):
        asyncio.run(reader.read_code(bad))


def reader_for(response):
    calls = []

    def handler(request):
        calls.append(request)
        if isinstance(response, Exception):
            raise response
        return response

    return SportyBetCodeReader(httpx.AsyncClient(transport=httpx.MockTransport(handler))), calls


def test_an_invalid_code_is_reported_without_internal_detail():
    reader, calls = reader_for(httpx.Response(200, json=INVALID))
    with pytest.raises(CodeReadError) as err:
        asyncio.run(reader.read_code("ZZ9QX7K"))
    message = str(err.value)
    assert "invalid or has expired" in message
    assert "19000" not in message and "http" not in message.lower() and "sportybet.com" not in message
    assert len(calls) == 1


def test_an_invalid_code_reply_through_the_whole_flow_is_a_plain_message():
    reader, _ = reader_for(httpx.Response(200, json=INVALID))
    chunks = asyncio.run(run_review(" ZZ9QX7K", reader, str, now=NOW))
    assert len(chunks) == 1 and "invalid or has expired" in chunks[0] and "19000" not in chunks[0]


@pytest.mark.parametrize("payload", [{"bizCode": 10000, "data": {"outcomes": []}}, {"bizCode": 10000, "data": {}}])
def test_an_empty_code_is_reported(payload):
    if payload["data"]:
        with pytest.raises(CodeReadError) as err:
            parse_selections(payload)
        assert "no selections" in str(err.value)
    else:
        with pytest.raises(CodeReadError):
            parse_selections(payload)


def test_an_expired_or_unknown_business_code_is_reported_safely():
    with pytest.raises(CodeReadError) as err:
        parse_selections({"bizCode": 19999, "message": "internal stack trace at Foo.java"})
    assert "Foo.java" not in str(err.value) and "19999" not in str(err.value)


def test_one_unreadable_selection_fails_the_whole_read_no_partial_results():
    broken = json.loads(json.dumps(SAMPLE))
    del broken["data"]["outcomes"][3]["markets"][0]["outcomes"][0]["odds"]
    with pytest.raises(CodeReadError) as err:
        parse_selections(broken)
    assert "could not read" in str(err.value)
    broken = json.loads(json.dumps(SAMPLE))
    broken["data"]["outcomes"][2]["homeTeamName"] = ""
    with pytest.raises(CodeReadError):
        parse_selections(broken)


@pytest.mark.parametrize("response", [
    httpx.Response(500, text="boom at C:\\secret\\path"),
    httpx.Response(200, text="<html>not json</html>"),
    httpx.Response(403, text="denied"),
    httpx.ConnectError("no route to https://internal.example"),
    httpx.ReadTimeout("slow"),
])
def test_failures_are_sanitized_and_never_retried(response):
    reader, calls = reader_for(response)
    with pytest.raises(CodeReadError) as err:
        asyncio.run(reader.read_code("ABC123"))
    message = str(err.value)
    assert "secret" not in message and "internal.example" not in message and "<html>" not in message
    assert "Nothing was reviewed" in message
    assert len(calls) == 1  # no retry


def test_review_with_no_code_asks_for_one_and_reads_nothing():
    reader = FakeCodeReader(selections())
    for argument in ("", "   "):
        assert asyncio.run(run_review(argument, reader, str, now=NOW)) == [ASK_FOR_CODE]
    assert reader.codes == []


@pytest.mark.parametrize("argument", [" two codes", " ../etc", " ab", " AB-12"])
def test_a_junk_code_is_refused_before_any_read(argument):
    reader = FakeCodeReader(selections())
    assert asyncio.run(run_review(argument, reader, str, now=NOW)) == [BAD_CODE]
    assert reader.codes == []


def test_a_game_that_has_started_is_marked_and_excluded():
    chunks, seen, _, stats = review()
    text = text_of(chunks)
    assert "ALREADY STARTED" not in text and "Napoli vs Roma" not in text   # started games are not returned, only counted
    assert "1 already started (not shown)" in text
    result = seen[0]
    assert [s.home for s in result.started] == ["Napoli"]
    assert "Napoli" not in [j.selection.home for j in result.judgements]
    assert "Napoli" not in stats.asked  # excluded before any lookup


def test_a_kickoff_in_the_past_counts_as_started_even_if_the_source_says_not_started():
    stale = [replace(s, started=False, kickoff_utc=NOW - timedelta(minutes=1)) for s in selections()[:1]]
    chunks, seen, *_ = review(FakeCodeReader(stale))
    assert len(seen) == 0  # nothing left to judge, nothing to save
    assert "already started" in text_of(chunks) and "nothing was evaluated" in text_of(chunks)


def test_a_reader_failure_gives_the_reader_message_only():
    chunks, seen, *_ = review(FakeCodeReader(error=CodeReadError("The code is invalid.")))
    assert chunks == ["The code is invalid."] and seen == []


# ================================================================ Story 4.2: judging by sport
def judgements():
    _, seen, *_ = review()
    return {j.selection.home: j for j in seen[0].judgements}


def test_a_multi_sport_code_is_judged_selection_by_selection():
    j = judgements()
    assert set(j) == {"Arsenal", "Chelsea", "Sevilla", "LA Lakers", "Alcaraz C.", "Gor Mahia"}
    assert j["Arsenal"].approved
    assert not j["Chelsea"].approved and "below the 70% floor" in j["Chelsea"].evaluation.reason
    assert not j["Chelsea"].evaluation.insufficient  # it has a real estimate


def test_the_estimate_uses_the_same_model_floor_and_odds_on_the_code():
    j = judgements()["Arsenal"].evaluation
    assert j.passed and j.estimate.probability >= PROBABILITY_FLOOR and j.ev > 0
    assert j.odds == Decimal("1.85")  # the odds on the code
    # identical to what a Matchday pick for the same market gets from the same engine
    s = selections()[0]
    fixture = Fixture(s.match_id, s.competition, s.home, s.away, s.kickoff_utc, False,
                      (MarketOffer("18", "Over/Under", "total=2.5", (Outcome("Over 2.5", Decimal("1.85")),
                                                                     Outcome("Under 2.5", Decimal("2.00")))),))
    matchday = [e for e in SoccerEngine().evaluate(fixture, inputs_for(2.5, 2.0), fixture.markets)
                if e.selection == "Over 2.5"][0]
    assert matchday.estimate.probability == j.estimate.probability and matchday.passed == j.passed
    # and the code's own odds decide the EV: worse odds on the code can turn the same estimate into a rejection
    worse = judge_selection(replace(s, odds=Decimal("1.05")), fixture,
                            FixtureStats(s.match_id, READY, tuple(inputs_for(2.5, 2.0))))
    assert worse.estimate.probability == j.estimate.probability and not worse.passed and "expected value" in worse.reason


def test_a_sport_with_no_rules_is_disqualified_by_default_and_never_approved():
    j = judgements()
    ev = j["Alcaraz C."].evaluation
    assert not j["Alcaraz C."].approved and ev.reason == "no rules for this sport" and ev.insufficient and not ev.passed
    chunks, *_ = review()
    text = text_of(chunks)
    assert text.count("disqualified: no rules for this sport") == 1
    # basketball has rules since Story 6.2: with no data source it is "insufficient data", never "no rules"
    lakers = j["LA Lakers"].evaluation
    assert not j["LA Lakers"].approved and lakers.reason.startswith("insufficient data") and lakers.insufficient


@pytest.mark.parametrize("sport", ["tennis", "rugby", "cricket"])  # baseball, hockey (6.3-6.5) and american_football (7.1) have engines
def test_every_sport_without_an_engine_is_disqualified(sport):
    s = replace(selections()[0], sport=sport, sport_label=sport.title())
    ev = judge_selection(s, Fixture(s.match_id, s.competition, s.home, s.away, s.kickoff_utc, False),
                         FixtureStats(s.match_id, READY, tuple(inputs_for(3.0, 3.0))))
    assert not ev.passed and ev.reason == "no rules for this sport"  # even with perfect data


def test_double_chance_is_judged_by_review_since_spec_11_1_and_an_unknown_market_still_has_no_rules():
    # Spec 11-1: the engine prices the double chance, so Review judges it like any other market it understands.
    ev = judgements()["Sevilla"].evaluation
    assert not ev.passed and ev.selection == "Home or Draw" and "below the 70% floor" in ev.reason
    # A market the engine has no rules for (Goal Bounds, whose meaning is not confirmed) is still "no rules for this market".
    assert not SoccerEngine().understands(MarketOffer("0", "Goal Bounds", "", (Outcome("2-5+", Decimal("1.5")),)))
    assert SoccerEngine().understands(MarketOffer("11", "Draw No Bet", "", (Outcome("Home", Decimal("1.5")),)))   # priced since 2026-10-05


def test_missing_data_is_insufficient_data_never_guessed():
    ev = judgements()["Gor Mahia"].evaluation
    assert not ev.passed and ev.insufficient and ev.reason.startswith("insufficient data: missing required input")
    assert ev.estimate.probability == 0.0 and not ev.estimate.inputs


def test_a_stats_outage_disqualifies_as_insufficient_data_with_a_safe_note():
    class Down:
        async def fetch_stats(self, fixtures):
            raise StatsFetchError("Understat could not be read")

    chunks, seen, *_ = review(sources={"soccer": Down()})
    text = text_of(chunks)
    assert "Could not get soccer stats: Understat could not be read" in text
    soccer = [j for j in seen[0].judgements if j.selection.sport == "soccer" and j.selection.market_id != "10"]
    assert soccer and all(j.evaluation.reason.startswith("insufficient data") for j in soccer)
    assert not seen[0].approved


def test_a_sport_with_rules_but_no_data_source_is_insufficient_not_approved():
    _, seen, *_ = review(sources={})
    assert not seen[0].approved
    assert all(j.evaluation.reason.startswith("insufficient data") for j in seen[0].judgements
               if j.selection.sport == "soccer" and j.selection.market_id != "10")


def test_a_derby_is_disqualified_as_no_valid_play():
    s = replace(selections()[0], home="Arsenal", away="Tottenham")
    fixture = Fixture(s.match_id, s.competition, s.home, s.away, s.kickoff_utc, False)
    ev = judge_selection(s, fixture, FixtureStats(s.match_id, READY, tuple(inputs_for(2.5, 2.0))))
    assert not ev.passed and "NO VALID PLAY - PASS" in ev.reason and "derby" in ev.reason


def test_a_suspended_market_is_not_approved():
    s = replace(selections()[0], available=False)
    fixture = Fixture(s.match_id, s.competition, s.home, s.away, s.kickoff_utc, False)
    ev = judge_selection(s, fixture, FixtureStats(s.match_id, READY, tuple(inputs_for(2.5, 2.0))))
    assert not ev.passed and "not currently offered" in ev.reason


def test_the_exact_selection_is_judged_with_no_swap_and_no_asian_line():
    # a lower-tier league would get Asian shielding on Matchday; on a code the selection stays as it is
    s = replace(selections()[0], competition="Kenya - Premier League")
    fixture = Fixture(s.match_id, s.competition, s.home, s.away, s.kickoff_utc, False)
    stats = FixtureStats(s.match_id, READY, tuple(inputs_for(2.5, 2.0)))
    assert judge_selection(s, fixture, stats).selection == "Over 2.5"
    home = replace(s, market_id="1", market_name="1X2", specifier="", outcome_name="Home", odds=Decimal("2.40"))
    ev = judge_selection(home, fixture, FixtureStats(s.match_id, READY, tuple(inputs_for(2.0, 1.0))))
    assert ev.selection == "Home" and not ev.rule and "swap" not in ev.reason.lower()


def test_a_whole_line_on_the_code_is_judged_with_its_refund():
    s = replace(selections()[0], specifier="total=2", outcome_name="Over 2")
    fixture = Fixture(s.match_id, s.competition, s.home, s.away, s.kickoff_utc, False)
    ev = judge_selection(s, fixture, FixtureStats(s.match_id, READY, tuple(inputs_for(2.5, 2.0))))
    assert ev.selection == "Over 2.0" and ev.estimate.push > 0 and any("refund" in n for n in ev.notes)


def test_team_totals_are_judged_from_the_same_rates():
    s = replace(selections()[0], market_id="19", market_name="Home Over/Under", specifier="total=0.5",
                outcome_name="Over 0.5", odds=Decimal("1.30"))
    fixture = Fixture(s.match_id, s.competition, s.home, s.away, s.kickoff_utc, False)
    ev = judge_selection(s, fixture, FixtureStats(s.match_id, READY, tuple(inputs_for(2.5, 2.0))))
    assert ev.selection == "Home team Over 0.5" and ev.estimate.probability > 0.85 and ev.passed


def test_approved_selections_show_inputs_and_sources_and_disqualified_are_listed_separately():
    chunks, *_ = review()
    text = text_of(chunks)
    approved, rest = text.split("DISQUALIFIED (")[0], text.split("DISQUALIFIED (")[1]
    assert "APPROVED (1):" in approved
    assert "inputs and sources" in approved and "home.xg_for 2.5 xG per match [test-source]" in approved
    assert "success " in approved and "%" in approved
    assert "Arsenal" not in rest  # not repeated among the disqualified
    assert "Chelsea vs Wolverhampton" in rest and "below the 70% floor" in rest
    assert "estimate " not in rest.split("Chelsea")[0]  # sanity: the disqualified block starts after the approved one
    assert text.index("APPROVED") < text.index("DISQUALIFIED")


def test_a_sport_added_later_is_judged_with_no_change_to_review(monkeypatch):
    """Epics 6 and 7: registering an engine and a data source is all it takes."""
    class FakeBasketballEngine:
        def understands(self, offer):
            return True

        def screen(self, fixture, inputs):
            return None

        def evaluate_exact(self, fixture, inputs, offer):
            o = offer.outcomes[0]
            from predictions.core.model import Estimate, validate
            estimate = Estimate(0.70, tuple(inputs), "fake-basketball")
            ok, ev, reason = validate(estimate, o.odds)
            return [Evaluation(fixture.match_id, offer.name, o.name, o.odds, estimate, ev, ok, reason)]

    class BasketballStats:
        async def fetch_stats(self, fixtures):
            return {f.match_id: FixtureStats(f.match_id, READY, (Input("home.pace", 1.0, "x", "fake"),))
                    for f in fixtures}

    monkeypatch.setitem(engines.ENGINES, "basketball", FakeBasketballEngine())
    _, seen, *_ = review(sources={"soccer": FakeStats(), "basketball": BasketballStats()})
    lakers = [j for j in seen[0].judgements if j.selection.sport == "basketball"][0]
    assert lakers.approved and lakers.evaluation.estimate.model == "fake-basketball"
    assert [j for j in seen[0].judgements if j.selection.sport == "tennis"][0].evaluation.reason == \
        "no rules for this sport"


def test_a_long_review_is_split_under_the_telegram_limit_without_losing_a_selection():
    base = selections()
    many = [replace(s, home=f"Team{n} {s.home}", away=f"Other{n} {s.away}") for n in range(40) for s in base]
    chunks, seen, *_ = review(FakeCodeReader(many))
    assert len(chunks) > 1 and all(len(c) <= MESSAGE_LIMIT <= 4000 for c in chunks)
    text = text_of(chunks)
    assert f"{len(many)} selection(s) read" in text
    assert all(f"Team{n} " in text for n in range(40))
    assert len(seen[0].judgements) + len(seen[0].started) == len(many)


# ================================================================ saving
def test_every_judged_selection_is_saved_with_the_existing_store_and_started_ones_are_not(tmp_path):
    _, seen, *_ = review()
    with EvaluationStore.open(tmp_path / "p.db", lambda t: t) as store:
        store.save_evaluations("review-1", seen[0].evaluations)
        rows = store.list_evaluations("review-1")
    assert len(rows) == 6 and {r["home"] for r in rows} == {"Arsenal", "Chelsea", "Sevilla", "LA Lakers",
                                                             "Alcaraz C.", "Gor Mahia"}
    by_home = {r["home"]: r for r in rows}
    assert by_home["Arsenal"]["passed"] == 1 and by_home["Arsenal"]["placed"] == 0
    assert by_home["Arsenal"]["odds"] == "1.85" and by_home["Arsenal"]["inputs"]
    assert by_home["LA Lakers"]["passed"] == 0 and by_home["LA Lakers"]["reason"].startswith("insufficient data")
    assert by_home["Alcaraz C."]["reason"] == "no rules for this sport"
    assert by_home["LA Lakers"]["sport"] == "basketball"


# ================================================================ the Telegram handler and the owner gate
def owner_message(text, user_id=None):
    sent = []

    async def reply_text(message, **kwargs):
        sent.append((message, kwargs))

    return SimpleNamespace(effective_user=SimpleNamespace(id=user_id or bot.OWNER_ID), callback_query=None,
                           message=SimpleNamespace(text=text, reply_text=reply_text)), sent


def test_review_runs_the_new_pipeline(monkeypatch):
    reader = FakeCodeReader(selections())
    monkeypatch.setattr(bot, "SportyBetCodeReader", lambda: reader)
    monkeypatch.setattr(bot, "review_stats_sources", lambda: {"soccer": FakeStats()})
    monkeypatch.setattr(bot.ClaudeAnalyst, "from_environment", classmethod(lambda cls, mode="full": None))
    monkeypatch.setattr(bot, "STORE", None)
    from predictions.flow.session import SessionManager
    monkeypatch.setattr(bot, "SESSIONS", SessionManager())  # a clean slate: no other test's open list
    update, sent = owner_message("Review abc123")
    asyncio.run(handled(update))
    drop_ack(sent)
    text = "\n".join(m for m, _ in sent)
    assert reader.codes == ["abc123"] and "APPROVED (1)" in text and "Processing" not in text
    assert all("parse_mode" not in kw for _, kw in sent)
    assert all("reply_markup" not in kw for _, kw in sent[:-1])  # the Review text itself carries no buttons ...
    assert "Approved selections from Review abc123" in sent[-1][0] and "reply_markup" in sent[-1][1]  # ... the list after it does (5.6)


def test_review_with_no_code_asks_for_one_through_the_handler(monkeypatch):
    async def must_not_run(*args, **kwargs):
        raise AssertionError("a read was attempted")

    monkeypatch.setattr(bot, "SportyBetCodeReader", lambda: SimpleNamespace(read_code=must_not_run))
    for text in ("Review", "review   "):
        update, sent = owner_message(text)
        asyncio.run(handled(update))
        drop_ack(sent)
        assert [m for m, _ in sent] == [ASK_FOR_CODE]


def test_review_saves_through_the_store_and_reports_a_failed_save(monkeypatch, tmp_path):
    monkeypatch.setattr(bot, "SportyBetCodeReader", lambda: FakeCodeReader(selections()))
    monkeypatch.setattr(bot, "review_stats_sources", lambda: {"soccer": FakeStats()})
    monkeypatch.setattr(bot.ClaudeAnalyst, "from_environment", classmethod(lambda cls, mode="full": None))
    store = EvaluationStore.open(tmp_path / "p.db", bot.redact)
    monkeypatch.setattr(bot, "STORE", store)
    update, sent = owner_message("Review ABC123")
    asyncio.run(handled(update))
    drop_ack(sent)
    assert len(store.list_evaluations()) == 6

    def broken(*args, **kwargs):
        raise RuntimeError("disk full at C:\\secret")

    monkeypatch.setattr(store, "save_evaluations", broken)
    update, sent = owner_message("Review ABC123")
    asyncio.run(handled(update))
    drop_ack(sent)
    assert any("Could not save this review" in m for m, _ in sent) and any("APPROVED" in m for m, _ in sent)
    assert not any("secret" in m for m, _ in sent)
    store.close()


def test_a_stranger_cannot_trigger_a_review():
    from telegram import Update
    from telegram.ext import ApplicationBuilder, ExtBot

    calls = []

    async def rec_review(*args, **kwargs):
        calls.append(args)
        return ["ok"]

    def build(monkeypatch):
        async def no_network(self):
            from telegram import User
            self._bot_user = User(id=1, first_name="TestBot", is_bot=True, username="test_bot")

        monkeypatch.setattr(ExtBot, "initialize", no_network)
        application = ApplicationBuilder().token("123456:TEST-TOKEN-NOT-REAL").build()
        bot.register_handlers(application)
        return application

    def feed(application, user_id):
        payload = {"update_id": 10, "message": {"message_id": 1, "date": 1700000000,
                                                "chat": {"id": user_id, "type": "private"},
                                                "from": {"id": user_id, "is_bot": False, "first_name": "T"},
                                                "text": "Review ABC123"}}
        update = Update.de_json(payload, application.bot)

        async def go():
            await application.initialize()
            await application.process_update(update)
            await bot.JOBS.wait()

        asyncio.run(go())

    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(bot, "run_review", rec_review)
        application = build(mp)
        feed(application, 999888777)
        assert calls == []  # dropped by the gate before any handler
        replies = []

        async def reply(self, text, **kwargs):
            replies.append(text)

        mp.setattr("telegram.Message.reply_text", reply)
        mp.setattr(bot.ClaudeAnalyst, "from_environment", classmethod(lambda cls, mode="full": None))
        feed(application, bot.OWNER_ID)
        drop_ack(replies)
        assert len(calls) == 1 and replies == ["ok"]
    finally:
        mp.undo()


# ================================================================ review order, and the finished-review cache
def test_review_lists_are_sorted_from_the_earliest_kickoff_and_every_selection_shows_date_and_time():
    reader = FakeCodeReader(list(reversed(selections())))
    chunks, seen, *_ = review(reader)
    kickoffs = [j.selection.kickoff_utc for j in seen[0].judgements]
    assert kickoffs == sorted(kickoffs)
    text = text_of(chunks)
    assert all(f"{k.astimezone(LAGOS).day} {k.astimezone(LAGOS).strftime('%b %H:%M')}" in text for k in kickoffs)


def cached_review(stats, reader, now, argument=" ABC123", seen=None):
    return asyncio.run(run_review(argument, reader, str, stats_sources={"soccer": stats}, now=now,
                                  on_result=(seen if seen is not None else []).append))


def test_the_same_code_the_same_day_is_answered_from_memory_without_reading_or_looking_up_again(monkeypatch):
    from predictions.flow.reviewcache import REVIEW_CACHE
    monkeypatch.setattr(REVIEW_CACHE, "hours", 3.0)
    stats, reader = FakeStats(), FakeCodeReader(selections())
    first = cached_review(stats, reader, NOW)
    reads, asked = len(reader.codes), len(stats.asked)
    again = cached_review(stats, reader, NOW + timedelta(minutes=30))
    assert len(reader.codes) == reads and len(stats.asked) == asked           # nothing was read or looked up again
    text = text_of(again)
    assert "Reused the review made at" in text and "3 hour(s)" in text
    assert "Reused" not in text_of(first)


def test_a_reused_review_drops_games_that_have_started_since_and_never_lists_them(monkeypatch):
    from predictions.flow.reviewcache import REVIEW_CACHE
    monkeypatch.setattr(REVIEW_CACHE, "hours", 3.0)
    stats, reader = FakeStats(), FakeCodeReader(selections())
    first_seen = []
    cached_review(stats, reader, NOW, seen=first_seen)
    first_game = min(first_seen[0].judgements, key=lambda j: j.selection.kickoff_utc).selection
    later = first_game.kickoff_utc + timedelta(minutes=1)
    seen = []
    text = text_of(cached_review(stats, reader, later, seen=seen))
    names = {j.selection.home for j in seen[0].judgements}
    assert first_game.home not in names and f"{first_game.home} vs {first_game.away}" not in text
    assert "already started (not shown)" in text and len(seen[0].started) >= 1


def test_the_cache_expires_after_the_configured_hours_and_on_a_new_day_and_can_be_turned_off(monkeypatch):
    from predictions.flow.reviewcache import REVIEW_CACHE, parse_review_cache_hours
    monkeypatch.setattr(REVIEW_CACHE, "hours", 1.0)
    stats, reader = FakeStats(), FakeCodeReader(selections())
    cached_review(stats, reader, NOW)
    reads = len(reader.codes)
    cached_review(stats, reader, NOW + timedelta(minutes=61))
    assert len(reader.codes) == reads + 1                                     # older than an hour: read again
    monkeypatch.setattr(REVIEW_CACHE, "hours", 0.0)
    cached_review(stats, reader, NOW + timedelta(minutes=62))
    assert len(reader.codes) == reads + 2                                     # off: never reused
    assert parse_review_cache_hours(None) == 3.0 and parse_review_cache_hours("1.5") == 1.5 and parse_review_cache_hours("0") == 0.0
    for bad in ("-1", "abc", "25", "1e2", "1..5"):
        with pytest.raises(ValueError):
            parse_review_cache_hours(bad)


def test_a_review_made_while_stats_failed_is_not_kept(monkeypatch):
    from predictions.flow.reviewcache import REVIEW_CACHE
    from predictions.core.ports import StatsFetchError
    monkeypatch.setattr(REVIEW_CACHE, "hours", 3.0)

    class Broken(FakeStats):
        async def fetch_stats(self, fixtures):
            raise StatsFetchError("down")

    reader = FakeCodeReader(selections())
    cached_review(Broken(), reader, NOW)
    reads = len(reader.codes)
    cached_review(FakeStats(), reader, NOW + timedelta(minutes=5))
    assert len(reader.codes) == reads + 1


def test_valid_plays_are_listed_from_the_earliest_kickoff_not_by_estimate():
    from decimal import Decimal
    from predictions.core.model import Estimate, Evaluation
    from predictions.core.types import MatchId
    from predictions.flow.evaluate import GateResult
    from predictions.telegram.render import gate_lines
    early = Fixture(MatchId("soccer", "L", NOW + timedelta(hours=2), "early home", "early away"), "L", "Early Home", "Early Away", NOW + timedelta(hours=2), False)
    late = Fixture(MatchId("soccer", "L", NOW + timedelta(hours=5), "late home", "late away"), "L", "Late Home", "Late Away", NOW + timedelta(hours=5), False)
    strong_late = Evaluation(late.match_id, "1X2", "Home", Decimal("1.50"), Estimate(0.90), Decimal("0.35"), True, "passed")
    weaker_early = Evaluation(early.match_id, "1X2", "Home", Decimal("1.50"), Estimate(0.72), Decimal("0.08"), True, "passed")
    gate = GateResult((strong_late, weaker_early), (), 2, 0, {early.match_id: early, late.match_id: late})
    lines = [l for l in gate_lines(gate) if "vs" in l]
    assert "Early Home vs Early Away" in lines[0] and "Late Home vs Late Away" in lines[1]
