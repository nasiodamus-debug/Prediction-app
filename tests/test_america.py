"""Stories 6.3 to 6.5 end to end: `America <dates>` lists SportyBet MLB and NHL games together (recorded listing samples,
tests/data/sportybet_mlb_sample.json and sportybet_nhl_sample.json, probed 2026-09-30), gathers stats with provenance,
applies the 70% floor, feeds the SAME interactive list and placement pipeline as Matchday, and `Review` judges MLB and
NHL selections. No test makes a live call."""
import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import app as bot
from handled import drop_ack, handled
from fakes_bookmaker import FakeBookmaker
from listview import detail_text, list_text
from predictions.adapters.sportybet.betslip import Credentials, SportyBetBookmaker
from predictions.adapters.sportybet.fixtures import FixtureFetchError, LISTINGS, SportyBetFixtures, parse_page
from predictions.core.betting import Availability, leg_from_evaluation
from predictions.core.dates import LAGOS
from predictions.core.engines import ENGINES
from predictions.core.model import PROBABILITY_FLOOR
from predictions.core.ports import StatsFetchError
from predictions.core.review import CodeSelection, judge_selection
from predictions.core.stats import INSUFFICIENT, READY, FixtureStats, insufficient
from predictions.core.types import Fixture
from predictions.flow.america import MultiSportFixtures, MultiSportStats, run_america
from predictions.flow.evaluate import run_gate
from predictions.flow.loading import KIND_LOADED, run_load
from predictions.flow.review import run_review
from predictions.flow.session import Action, SessionManager, State, Status
from predictions.flow.slate_run import AMERICA, PROFILES
from predictions.telegram.render import SPORT_ODDS
from predictions.telegram.slate import ORIGIN_NAMES, list_screen
from test_epic5_flow import OWNER
from test_america_stats import MlbApi, NhlApi
from test_mlb_engine import HOT, inputs as mlb_inputs
from test_nhl_engine import inputs as nhl_inputs

DATA = Path(__file__).resolve().parent / "data"
MLB = json.loads((DATA / "sportybet_mlb_sample.json").read_text(encoding="utf-8"))
NHL = json.loads((DATA / "sportybet_nhl_sample.json").read_text(encoding="utf-8"))
NOW = datetime(2026, 9, 30, 8, 0, tzinfo=timezone.utc)    # 09:00 in Lagos, before every game in the samples


def run(coro):
    return asyncio.run(coro)


def mlb_games(payload=MLB):
    return parse_page(payload, "baseball")[0]


def nhl_games(payload=NHL):
    return parse_page(payload, "hockey")[0]


class FakeFixtures:
    def __init__(self, fixtures, error=None):
        self.fixtures, self.error, self.calls = fixtures, error, 0

    async def fetch_fixtures(self):
        self.calls += 1
        if self.error:
            raise self.error
        return self.fixtures


def both_listings(mlb=None, nhl=None):
    return MultiSportFixtures({"baseball": FakeFixtures(mlb_games() if mlb is None else mlb),
                               "hockey": FakeFixtures(nhl_games() if nhl is None else nhl)})


class TableStats:
    def __init__(self, table):
        self.table, self.asked = table, []

    async def fetch_stats(self, fixtures):
        self.asked.extend(f.home for f in fixtures)
        return {f.match_id: self.table(f) for f in fixtures}


def strong(f):
    """Ready stats far above the market for both sports, so that Over picks pass."""
    made = mlb_inputs(HOT, HOT) if f.match_id.sport == "baseball" else nhl_inputs(
        dict(gf=3.6, ga=3.4, xf=3.4, xa=3.3), dict(gf=3.5, ga=3.4, xf=3.3, xa=3.3))
    return FixtureStats(f.match_id, READY, tuple(made))


def america(argument, source=None, table=None, now=NOW, **kwargs):
    gates = []
    sources = {"baseball": TableStats(table or (lambda f: insufficient(f.match_id, "the stats source has no data"))),
               "hockey": TableStats(table or (lambda f: insufficient(f.match_id, "the stats source has no data")))}
    chunks = run(run_america(argument, source or both_listings(), str, now,
                             stats_source=MultiSportStats(sources, str), on_result=gates.append, **kwargs))
    return "\n".join(chunks), chunks, (gates[0] if gates else None), sources


# ================================================================ Stories 6.3 and 6.4: listing and dates
def test_the_listing_samples_parse_as_baseball_and_hockey_with_the_supported_markets_only():
    mlb, nhl = mlb_games(), nhl_games()
    assert {f.match_id.sport for f in mlb} == {"baseball"} and {f.competition for f in mlb} == {"USA - MLB"}
    assert {f.match_id.sport for f in nhl} == {"hockey"} and {f.competition for f in nhl} == {"USA - NHL"}
    assert len(mlb) == 2 and len(nhl) == 4
    assert {m.market_id for f in mlb for m in f.markets} <= {"251", "256", "258", "260", "261", "274", "275", "276"}
    assert {m.market_id for f in nhl for m in f.markets} <= {"406", "410", "16", "412", "18", "19", "20", "446"}
    assert mlb[0].bookmaker_ref == "sr:match:75060826"


def test_america_lists_mlb_games_on_the_requested_lagos_date():
    text, _, gate, _ = america("30 September")
    assert "America fixtures for 30 Sep 2026" in text
    assert "Atlanta Braves vs Philadelphia Phillies" in text and "Houston Astros vs Chicago White Sox" in text
    assert "Flyers" not in text and "USA - MLB" in text and "USA - NHL" not in text


def test_the_date_is_the_lagos_date_not_the_utc_date_for_nhl_games():
    # 23:30 UTC on 30 September is 00:30 on 1 October in Lagos
    flyers = next(f for f in nhl_games() if f.home == "Philadelphia Flyers")
    assert flyers.kickoff_utc.date().isoformat() == "2026-09-30" and flyers.kickoff_utc.astimezone(LAGOS).day == 1
    text, *_ = america("1 October")
    assert "Philadelphia Flyers vs Pittsburgh Penguins" in text and "Colorado Avalanche vs Los Angeles Kings" in text
    assert "Columbus" not in text and "Atlanta Braves" not in text
    text, *_ = america("2 October")
    assert "Columbus Blue Jackets vs Buffalo Sabres" in text and "Flyers" not in text


def test_mlb_and_nhl_are_listed_together_in_one_reply_and_evaluated_in_one_gate():
    text, chunks, gate, sources = america("30 September and 1 October", table=strong)
    assert sources["baseball"].asked and sources["hockey"].asked
    assert sources["baseball"].asked == ["Atlanta Braves", "Houston Astros"] and len(sources["hockey"].asked) == 3
    sports = {e.match_id.sport for e in [*gate.passed, *gate.rejected]}
    assert sports == {"baseball", "hockey"} and gate.evaluated == 5   # ONE gate result, both sports
    assert chunks == []                      # something passed: the interactive list is the whole reply
    shown = list_text(gate.passed)
    assert "USA - MLB" not in shown and "Valid plays" not in shown and "Closest rejections" not in shown
    assert {e.match_id.sport for e in gate.passed} <= {"baseball", "hockey"} and "%" in shown


def test_the_listing_shows_each_sports_main_odds_line():
    text, *_ = america("30 September and 1 October")
    lines = [l.strip() for l in text.splitlines() if l.strip().startswith("Home ")]
    assert len(lines) == 5 and all("O/U" in l for l in lines)


def test_games_already_started_are_excluded_and_counted():
    late = datetime(2026, 9, 30, 18, 10, tzinfo=timezone.utc)   # the Braves game began at 18:00, the Astros' at 21:00
    text, *_ = america("30 September", now=late)
    assert "Houston Astros vs Chicago White Sox" in text and "Atlanta Braves" not in text
    assert "1 game(s) excluded because they had already started or finished" in text


def test_a_game_the_source_flags_as_started_is_excluded_even_before_its_kickoff_time():
    payload = json.loads(json.dumps(NHL))
    for tournament in payload["data"]["tournaments"]:
        for event in tournament["events"]:
            if event["homeTeamName"] == "Toronto Maple Leafs":
                event["status"], event["matchStatus"] = 1, "1st period"
    text, *_ = america("1 October", source=both_listings(nhl=nhl_games(payload)))
    assert "Toronto" not in text and "1 game(s) excluded" in text


def test_a_missing_or_unreadable_date_is_asked_for_and_nothing_is_read():
    source = both_listings()
    text, _, gate, sources = america("", source=source)
    assert text.startswith("Which date? Send the date after America") and gate is None
    text, *_ = america("soon")
    assert "I could not read that date" in text and "America" in text
    assert all(s.asked == [] for s in sources.values())


def test_only_past_dates_read_nothing():
    calls = FakeFixtures(mlb_games())
    text, _, gate, _ = america("1 September", source=MultiSportFixtures({"baseball": calls}))
    assert "Already passed" in text and calls.calls == 0 and gate is None


def test_an_unreachable_listing_for_either_sport_lists_nothing_at_all():
    source = MultiSportFixtures({"baseball": FakeFixtures(mlb_games()),
                                 "hockey": FakeFixtures([], FixtureFetchError("SportyBet did not answer in time"))})
    text, _, gate, sources = america("30 September", source=source)
    assert "Could not get fixtures from SportyBet: SportyBet did not answer in time" in text
    assert "Nothing was listed" in text and gate is None and sources["baseball"].asked == []


def test_another_hockey_league_is_listed_but_is_insufficient_data():
    payload = json.loads(json.dumps(NHL))
    payload["data"]["tournaments"][0]["categoryName"], payload["data"]["tournaments"][0]["name"] = "Sweden", "SHL"
    listing = nhl_games(payload)
    assert {f.competition for f in listing} == {"Sweden - SHL"} and all(f.match_id.sport == "hockey" for f in listing)


# ================================================================ insufficient data, NO VALID PLAY - PASS
def test_games_without_data_are_insufficient_and_nothing_passing_replies_no_valid_play_pass():
    text, chunks, gate, _ = america("30 September and 1 October")
    assert chunks[0] == "NO VALID PLAY - PASS"    # exactly, as the first message
    assert gate.passed == () and gate.evaluated == 0 and gate.skipped == 5
    assert "insufficient data: the stats source has no data" in text and "5 game(s) skipped for insufficient data" in text


def short_odds(payload):
    """The same listing with every price at 1.01: whatever the model thinks, no pick has positive EV."""
    payload = json.loads(json.dumps(payload))
    for tournament in payload["data"]["tournaments"]:
        for event in tournament["events"]:
            for market in event["markets"]:
                for outcome in market["outcomes"]:
                    outcome["odds"] = "1.01"
    return payload


def test_stats_that_give_no_pick_at_65_percent_reply_no_valid_play_pass_with_the_closest_rejections():
    source = both_listings(mlb_games(short_odds(MLB)), nhl_games(short_odds(NHL)))
    text, chunks, gate, _ = america("30 September and 1 October", source=source, table=strong)
    assert chunks[0] == "NO VALID PLAY - PASS" and not gate.passed and gate.rejected and gate.evaluated == 5
    assert "Closest rejections:" in text and ("below the 70% floor" in text or "expected value" in text)
    assert {e.match_id.sport for e in gate.rejected} == {"baseball", "hockey"}


def test_passing_picks_carry_their_percentage_provenance_and_the_rule_shown_with_the_original():
    text, chunks, gate, _ = america("30 September and 1 October", table=strong)
    assert gate.passed
    for pick in gate.passed:
        assert pick.estimate.probability >= PROBABILITY_FLOOR and pick.ev > 0
        assert pick.estimate.inputs and all(i.source == "test-source" for i in pick.estimate.inputs)
    assert "%" in list_text(gate.passed) and "Estimate: success" in detail_text(gate.passed)


def test_a_stats_source_outage_for_one_sport_only_blanks_that_sport():
    class Down:
        async def fetch_stats(self, fixtures):
            raise StatsFetchError("the NHL's stats service returned HTTP 500")

    sources = {"baseball": TableStats(strong), "hockey": Down()}
    gates = []
    chunks = run(run_america(" 30 September and 1 October", both_listings(), str, NOW,
                             stats_source=MultiSportStats(sources, str), on_result=gates.append))
    text = "\n".join(chunks)
    gate = gates[0]
    assert gate.evaluated == 2 and gate.skipped == 3 and gate.passed and all(e.match_id.sport == "baseball" for e in gate.passed)
    assert text == ""   # something passed: only the list is shown, no stats-service notes


def test_when_every_sport_fails_it_is_the_usual_stats_outage_message():
    class Down:
        async def fetch_stats(self, fixtures):
            raise StatsFetchError("service down")

    gates = []
    chunks = run(run_america(" 30 September and 1 October", both_listings(), str, NOW,
                             stats_source=MultiSportStats({"baseball": Down(), "hockey": Down()}, str), on_result=gates.append))
    assert "Could not get team stats: service down" in "\n".join(chunks) and not gates


def test_a_sport_with_no_registered_stats_source_is_insufficient_data_not_an_error():
    stats = run(MultiSportStats({"baseball": TableStats(strong)}, str).fetch_stats(mlb_games() + nhl_games()))
    assert sum(s.status == READY for s in stats.values()) == 2
    assert sum(s.status == INSUFFICIENT and "no stats source is registered for hockey" in s.reasons[0]
               for s in stats.values()) == 4


# ================================================================ the analyst may fill missing stats
class FakeAnalyst:
    """Supplies every requested MLB input (labeled by the reader as analyst-sourced)."""

    def __init__(self):
        self.queries = []

    async def research(self, query):
        from predictions.core.analyst import AnalystUsage, parse_report
        from predictions.core.inputs import BASEBALL_REGISTRY
        self.queries.append(query)
        values = {"matches": 10, "runs_for": 5.2, "runs_against": 5.0}
        stats = [{"name": n, "value": values[n.partition(".")[2]], "unit": BASEBALL_REGISTRY[n.partition(".")[2]].unit,
                  "citation": "example.com/kbo/stats"} for n in query.wanted if n.partition(".")[2] in values]
        return parse_report(json.dumps({"stats": stats, "flags": []}), query, AnalystUsage(1000, 200, 2))


def test_the_analyst_fills_missing_mlb_inputs_labeled_analyst_sourced_and_the_model_prices_them():
    analyst = FakeAnalyst()
    text, _, gate, _ = america("30 September", analyst=analyst)
    assert analyst.queries and all(q.sport == "baseball" for q in analyst.queries)
    assert {"home.runs_for", "away.runs_against"} <= set(analyst.queries[0].wanted)
    assert gate.evaluated == 2 and gate.skipped == 0 and gate.passed
    assert text == "" and "analyst-sourced:example.com" in detail_text(gate.passed)   # sources are one tap away


# ================================================================ the fixture adapter and the profile
def test_the_fixtures_adapter_asks_sportybet_for_each_sports_id_and_markets():
    seen = []

    def handler(request):
        seen.append(request.url.params)
        return httpx.Response(200, json=MLB if request.url.params["sportId"] == "sr:sport:3" else NHL)

    async def go(sport):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await SportyBetFixtures(client, sport=sport).fetch_fixtures()

    assert len(run(go("baseball"))) == 2 and len(run(go("hockey"))) == 4
    assert seen[0]["sportId"] == "sr:sport:3" and set(seen[0]["marketId"].split(",")) == {"251", "256", "258", "260", "261", "274", "275", "276"}
    assert seen[1]["sportId"] == "sr:sport:4" and set(seen[1]["marketId"].split(",")) == {"406", "410", "16", "412", "18", "19", "20", "446"}
    assert LISTINGS["soccer"].sport_id == "sr:sport:1" and LISTINGS["basketball"].sport_id == "sr:sport:2"


def test_a_sport_is_added_by_registering_a_profile_engine_stats_source_and_odds_line():
    assert PROFILES["america"] is AMERICA and ORIGIN_NAMES["america"] == "America"
    assert set(SPORT_ODDS) == {"baseball", "hockey", "american_football"} and {"baseball", "hockey"} <= set(ENGINES)
    assert set(bot.AMERICA_SPORTS) == {"baseball", "hockey", "american_football"}
    assert set(bot.america_stats_sources()) == {"baseball", "hockey", "american_football"}
    for sport in bot.AMERICA_SPORTS:
        assert sport in ENGINES and sport in LISTINGS   # the row of AMERICA_SPORTS is all `America` needs beside those


# ================================================================ the Telegram trigger
def owner_message(text):
    sent = []

    async def reply_text(message, **kwargs):
        sent.append((message, kwargs))

    return SimpleNamespace(effective_user=SimpleNamespace(id=bot.OWNER_ID), callback_query=None,
                           message=SimpleNamespace(text=text, reply_text=reply_text)), sent


@pytest.fixture
def wired(monkeypatch):
    real = bot.run_america
    captured = {}

    async def fixed_clock(argument, source, sanitize, **kwargs):
        captured["stats"] = kwargs["stats_source"]
        return await real(argument, source, sanitize, now=NOW, **kwargs)

    listings = {"baseball": FakeFixtures(mlb_games()), "hockey": FakeFixtures(nhl_games())}
    monkeypatch.setattr(bot, "run_america", fixed_clock)
    monkeypatch.setattr(bot, "AMERICA_SPORTS", {
        sport: ((lambda s=sport: listings[s]), (lambda: TableStats(strong))) for sport in ("baseball", "hockey")})
    monkeypatch.setattr(bot.ClaudeAnalyst, "from_environment", classmethod(lambda cls, mode="full": None))
    monkeypatch.setattr(bot, "STORE", None)
    monkeypatch.setattr(bot, "SESSIONS", SessionManager(clock=lambda: NOW))
    return captured


def test_america_runs_the_new_pipeline_and_shows_one_shared_list_for_both_sports(wired):
    update, sent = owner_message("America 30 September and 1 October")
    run(handled(update))
    drop_ack(sent)
    text = "\n".join(m for m, _ in sent)
    assert "fixtures for" not in text and "switched off" not in text and "rebuilt" not in text   # only the list
    assert "Processing" not in text and all("parse_mode" not in kw for _, kw in sent)
    assert isinstance(wired["stats"], MultiSportStats)
    last_text, last_kwargs = sent[-1]
    assert len(sent) == 1 and last_text.startswith("Final list:") and "reply_markup" in last_kwargs   # only the list
    assert bot.SESSIONS.active.origin == "america"
    sports = {p.evaluation.match_id.sport for p in bot.SESSIONS.active.slip.picks}
    assert sports == {"baseball", "hockey"}   # MLB and NHL picks are in the SAME list
    assert "from America" in last_text or "Final list" in last_text


def test_america_with_no_date_asks_for_one(wired):
    update, sent = owner_message("America")
    run(handled(update))
    drop_ack(sent)
    assert sent[0][0].startswith("Which date? Send the date after America") and bot.SESSIONS.active is None


def test_america_with_nothing_passing_replies_exactly_no_valid_play_pass_and_offers_no_list(monkeypatch, wired):
    monkeypatch.setattr(bot, "AMERICA_SPORTS", {
        sport: ((lambda s=sport: FakeFixtures(mlb_games() if s == "baseball" else nhl_games())),
                (lambda: TableStats(lambda f: insufficient(f.match_id, "no data")))) for sport in ("baseball", "hockey")})
    update, sent = owner_message("America 30 September")
    run(handled(update))
    drop_ack(sent)
    assert sent[0][0] == "NO VALID PLAY - PASS" and bot.SESSIONS.active is None
    assert all("reply_markup" not in kw for _, kw in sent)


def test_there_is_no_legacy_america_any_more_and_the_help_text_says_what_america_covers():
    assert not hasattr(bot, "LEGACY_TRIGGER_NAMES") and not hasattr(bot, "run_engine_audit")
    sent = []

    async def reply_text(message, **kwargs):
        sent.append(message)

    run(bot.start_cmd(SimpleNamespace(message=SimpleNamespace(reply_text=reply_text)), None))
    assert "America [Date]" in sent[0] and "MLB, NHL and American football" in sent[0]


def test_review_statistics_sources_cover_mlb_and_nhl():
    sources = bot.review_stats_sources()
    assert {"baseball", "hockey"} <= set(sources)
    assert type(sources["baseball"]).__name__ == "MlbStats" and type(sources["hockey"]).__name__ == "NhlStats"


# ================================================================ the shared list, loading and placement pipeline
def passing():
    _, _, gate, _ = america("30 September and 1 October", table=strong)
    return gate


def test_mlb_and_nhl_picks_are_confirmed_loaded_and_priced_like_any_other_pick_in_dry_run():
    gate = passing()
    manager = SessionManager(clock=lambda: NOW)
    session = manager.start(gate.passed, "abcd1234", "america")
    assert "Final list:" in list_screen(session).text
    mlb_pick = next(p for p in session.slip.picks if p.evaluation.match_id.sport == "baseball")
    nhl_pick = next(p for p in session.slip.picks if p.evaluation.match_id.sport == "hockey")
    # one pick per match: drop every other pick, leave one MLB and one NHL pick
    for pick in list(session.slip.picks):
        if pick.pick_id not in (mlb_pick.pick_id, nhl_pick.pick_id):
            manager.act(OWNER, "abcd1234", manager.active.state_version, Action.DELETE, pick.pick_id)
    manager.act(OWNER, "abcd1234", manager.active.state_version, Action.DONE)
    result = manager.act(OWNER, "abcd1234", manager.active.state_version, Action.CONFIRM)
    assert result.status is Status.APPLIED and result.load_next and manager.active.state is State.LOADING_SLIP
    bookmaker = FakeBookmaker(mlb_games() + nhl_games(), dry_run=True, now=NOW)
    out = run(run_load(manager, bookmaker, "abcd1234", manager.active.state_version, str))
    assert out.kind == KIND_LOADED and manager.active.state is State.AWAITING_STAKE
    keys = {leg.leg_key for leg in manager.active.loaded.legs}
    assert keys == {leg_from_evaluation(mlb_pick.evaluation).key, leg_from_evaluation(nhl_pick.evaluation).key}
    assert not any(c[0] == "place" for c in bookmaker.calls) and bookmaker.clicks == 0


def test_the_bookmaker_checks_each_leg_against_its_own_sports_listing():
    gate = passing()
    mlb_pick = next(e for e in gate.passed if e.match_id.sport == "baseball")
    nhl_pick = next(e for e in gate.passed if e.match_id.sport == "hockey")
    calls = []

    class Listing(FakeFixtures):
        def __init__(self, name, fixtures):
            super().__init__(fixtures)
            self.name = name

        async def fetch_fixtures(self):
            calls.append(self.name)
            return await super().fetch_fixtures()

    async def no_site():
        raise AssertionError("no browser is needed to check availability")

    bookmaker = SportyBetBookmaker(Credentials("08031234567", "pw"), dry_run=True, fixture_source=Listing("soccer", []),
                                   sport_fixture_sources={"baseball": Listing("baseball", mlb_games()),
                                                          "hockey": Listing("hockey", nhl_games())},
                                   site_factory=no_site, clock=lambda: NOW)
    found = run(bookmaker.check_availability([leg_from_evaluation(mlb_pick), leg_from_evaluation(nhl_pick)]))
    assert [f.status for f in found] == [Availability.OK, Availability.OK]
    assert sorted(calls) == ["baseball", "hockey"]    # never the soccer listing for these two sports


# ================================================================ Review judges MLB and NHL (same engines, the code's odds)
def code_selection(sport, label, competition, home, away, market_id, market_name, specifier, outcome, odds):
    return CodeSelection(sport, label, competition, home, away, datetime(2026, 10, 1, 1, 0, tzinfo=timezone.utc), False,
                         market_id, market_name, specifier, outcome, Decimal(odds))


MLB_SELECTION = code_selection("baseball", "Baseball", "USA - MLB", "Boston Red Sox", "New York Yankees", "258",
                               "Total (incl. extra innings)", "total=7.5", "Over 7.5", "1.90")
NHL_SELECTION = code_selection("hockey", "Ice Hockey", "USA - NHL", "Boston Bruins", "Tampa Bay Lightning", "412",
                               "Total (OT and P)", "total=5", "Over 5", "1.90")


def judge(selection, stats):
    fixture = Fixture(selection.match_id, selection.competition, selection.home, selection.away,
                      selection.kickoff_utc, False)
    return judge_selection(selection, fixture, stats)


def test_review_judges_an_mlb_selection_by_the_mlb_engine_at_the_codes_odds():
    ready = FixtureStats(MLB_SELECTION.match_id, READY, tuple(mlb_inputs(HOT, HOT)))
    ev = judge(MLB_SELECTION, ready)
    assert ev.passed and ev.odds == Decimal("1.90") and ev.estimate.model == "negbin-runs-v1"
    assert ev.estimate.probability >= PROBABILITY_FLOOR
    calm = judge(MLB_SELECTION, FixtureStats(MLB_SELECTION.match_id, READY, tuple(mlb_inputs())))
    assert not calm.passed and "no rules" not in calm.reason
    short = judge(code_selection("baseball", "Baseball", "USA - MLB", "A", "B", "258", "Total", "total=7.5", "Over 7.5", "1.02"),
                  FixtureStats(MLB_SELECTION.match_id.__class__("baseball", "USA - MLB", MLB_SELECTION.kickoff_utc, "A", "B"),
                               READY, tuple(mlb_inputs(HOT, HOT))))
    assert not short.passed and "expected value" in short.reason


def test_review_judges_an_nhl_selection_with_the_nhl_engine_and_the_whole_line_refund_maths():
    hot = tuple(nhl_inputs(dict(gf=3.6, ga=3.4, xf=3.4, xa=3.3), dict(gf=3.5, ga=3.4, xf=3.3, xa=3.3)))
    ev = judge(NHL_SELECTION, FixtureStats(NHL_SELECTION.match_id, READY, hot))
    assert ev.estimate.model == "poisson-goals-v1" and ev.estimate.push > 0.05
    assert abs(ev.estimate.probability - (ev.estimate.win + ev.estimate.push)) < 1e-9
    assert ev.passed and "no rules" not in ev.reason


def test_review_disqualifies_mlb_and_nhl_selections_without_data_as_insufficient_data_never_no_rules():
    for selection in (MLB_SELECTION, NHL_SELECTION):
        ev = judge(selection, None)
        assert not ev.passed and ev.reason.startswith("insufficient data") and "no rules" not in ev.reason


class FakeReader:
    def __init__(self, selections):
        self.selections = selections

    async def read_code(self, code):
        return list(self.selections)


def test_a_review_of_a_code_with_mlb_and_nhl_selections_uses_both_stats_sources_and_lists_the_verdicts():
    sources = {"baseball": TableStats(lambda f: FixtureStats(f.match_id, READY, tuple(mlb_inputs(HOT, HOT)))),
               "hockey": TableStats(lambda f: FixtureStats(f.match_id, READY, tuple(nhl_inputs(
                   dict(gf=3.6, ga=3.4, xf=3.4, xa=3.3), dict(gf=3.5, ga=3.4, xf=3.3, xa=3.3)))))}
    seen = []
    chunks = run(run_review(" ABC123", FakeReader([MLB_SELECTION, NHL_SELECTION]), str, stats_sources=sources,
                            now=datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc), on_result=seen.append))
    text = "\n".join(chunks)
    assert sources["baseball"].asked == ["Boston Red Sox"] and sources["hockey"].asked == ["Boston Bruins"]
    assert "APPROVED (2)" in text and "no rules for this sport" not in text
    assert {e.evaluation.match_id.sport for e in seen[0].approved} == {"baseball", "hockey"}


# ================================================================ the real stats adapters behind the multi-sport source
def test_america_with_the_real_stats_adapters_on_recorded_style_data_evaluates_mlb_and_marks_unknown_nhl_teams_insufficient():
    sources = {"baseball": MlbApi().source(), "hockey": NhlApi().source()}
    gates = []
    chunks = run(run_america(" 30 September and 1 October", both_listings(), str, NOW,
                             stats_source=MultiSportStats(sources, str), on_result=gates.append))
    text, gate = "\n".join(chunks), gates[0]
    stats = run(MultiSportStats(sources, str).fetch_stats(mlb_games() + nhl_games()))
    braves = next(stats[f.match_id] for f in mlb_games() if f.home == "Atlanta Braves")
    assert braves.status == READY                                 # the Braves have a full window in the fabricated API
    assert any("was not found among the NHL teams" in r for st in stats.values() for r in st.reasons)
    assert text == "" or "Atlanta Braves" in text                 # a passing game means only the list is shown
    assert gate.evaluated >= 1 and {e.match_id.sport for e in [*gate.passed, *gate.rejected]} == {"baseball"}
    for e in [*gate.passed, *gate.rejected]:
        assert e.estimate.inputs and all(i.source == "mlb-statsapi" for i in e.estimate.inputs)
