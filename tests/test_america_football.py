"""Story 7.1 end to end: `America <dates>` lists NFL and college football together with MLB and NHL in ONE reply and ONE list
(recorded SportyBet listing samples, probed 2026-09-30), excludes started games, applies the 70% floor and positive EV,
replies exactly NO VALID PLAY - PASS when nothing passes, lets the analyst fill missing stats (labeled analyst-sourced),
feeds the shared interactive list / load / dry-run pipeline, and `Review` judges football selections. No live call."""
import asyncio
import json
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest

import app as bot
from handled import drop_ack, handled
from fakes_bookmaker import FakeBookmaker
from listview import detail_text
from predictions.adapters.sportybet.fixtures import LISTINGS, SportyBetFixtures, parse_page
from predictions.core.betting import Availability, leg_from_evaluation
from predictions.core.engines import ENGINES
from predictions.core.model import PROBABILITY_FLOOR
from predictions.core.review import CodeSelection, judge_selection
from predictions.core.stats import INSUFFICIENT, READY, FixtureStats, insufficient
from predictions.core.types import Fixture
from predictions.flow.america import MultiSportFixtures, MultiSportStats, run_america
from predictions.flow.loading import KIND_LOADED, run_load
from predictions.flow.review import run_review
from predictions.flow.session import Action, SessionManager, State, Status
from predictions.telegram.slate import list_screen
from test_america import FakeFixtures, TableStats, mlb_games, nhl_games, strong, NOW as NOW_MLB
from test_epic5_flow import OWNER
from test_football_engine import BOOST, af, inputs as fb_inputs
from test_football_stats import Espn, fixture as fb_fixture, values

DATA = Path(__file__).resolve().parent / "data"
FOOTBALL = json.loads((DATA / "sportybet_football_sample.json").read_text(encoding="utf-8"))
NOW = datetime(2026, 9, 30, 8, 0, tzinfo=timezone.utc)      # before every game in the samples


def run(coro):
    return asyncio.run(coro)


def football_games(payload=FOOTBALL):
    return parse_page(payload, "american_football")[0]


def football_stats(f):
    """Ready football stats for NFL and college (a high-scoring matchup with 5 mph wind); the CFL has no source."""
    if af.league_of(f.competition) is None:
        return insufficient(f.match_id, f"no free stats source covers {f.competition}")
    return FixtureStats(f.match_id, READY, tuple(fb_inputs(**BOOST, wind=5.0)))


def all_listings(football=None):
    return MultiSportFixtures({"baseball": FakeFixtures(mlb_games()), "hockey": FakeFixtures(nhl_games()),
                               "american_football": FakeFixtures(football_games() if football is None else football)})


def america(argument, source=None, table=None, now=NOW, quiet=False, **kwargs):
    """`quiet=True`: no sport has data, so nothing passes and the reply is the full fixture listing."""
    gates = []
    none = lambda f: insufficient(f.match_id, "the stats source has no data")   # noqa: E731
    sources = {"baseball": TableStats(none if quiet else strong), "hockey": TableStats(none if quiet else strong),
               "american_football": TableStats(none if quiet else (table or football_stats))}
    chunks = run(run_america(argument, source or all_listings(), str, now, stats_source=MultiSportStats(sources, str),
                             on_result=gates.append, **kwargs))
    return "\n".join(chunks), chunks, (gates[0] if gates else None), sources


# ================================================================ 7.1: listing
def test_the_football_listing_is_parsed_as_american_football_with_the_priced_markets():
    games = football_games()
    assert {f.match_id.sport for f in games} == {"american_football"} and len(games) == 7
    assert LISTINGS["american_football"].sport_id == "sr:sport:16"
    assert set(LISTINGS["american_football"].market_ids.split(",")) == {"219", "223", "225", "227", "228", "68"}
    assert games[0].bookmaker_ref == "sr:match:71515842" and games[0].home == "Cleveland Browns"


def test_the_fixtures_adapter_asks_sportybet_for_sport_16_and_its_market_ids():
    seen = []

    def handler(request):
        seen.append(request.url.params)
        return httpx.Response(200, json=FOOTBALL)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await SportyBetFixtures(client, sport="american_football").fetch_fixtures()
    assert len(run(go())) == 7
    assert seen[0]["sportId"] == "sr:sport:16" and set(seen[0]["marketId"].split(",")) == {"219", "223", "225", "227", "228", "68"}


def test_football_is_listed_together_with_mlb_and_nhl_in_one_reply_and_one_gate():
    listing, *_ = america("30 September and 2 October", quiet=True)      # nothing passes: the full listing
    assert "USA - MLB" in listing and "USA - NFL" in listing and "USA - NCAA, Regular Season" in listing
    assert "Cleveland Browns vs Pittsburgh Steelers" in listing and "Atlanta Braves vs" in listing
    text, chunks, gate, sources = america("30 September and 2 October")
    assert chunks == []                                              # something passed: only the list is shown
    assert sources["american_football"].asked and set(sources["american_football"].asked) >= {"Cleveland Browns"}
    sports = {e.match_id.sport for e in [*gate.passed, *gate.rejected]}
    assert {"baseball", "american_football"} <= sports              # ONE gate result, several sports
    assert "game(s) listed" in listing and "game(s) listed" not in text


def test_the_listing_shows_the_football_main_odds_line():
    text, *_ = america("2 October", quiet=True)
    line = text.split("America fixtures", 1)[1].split("Cleveland Browns vs Pittsburgh Steelers")[1].splitlines()[1].strip()
    assert line.startswith("Home 2.15 / Away 1.64") and "O/U" in line


def test_the_date_is_the_lagos_date_for_football():
    # 00:15 UTC on 2 October is 01:15 in Lagos (UTC+1); 23:00 UTC on 2 October is 00:00 on 3 October
    text, *_ = america("2 October", quiet=True)
    assert "Cleveland Browns vs Pittsburgh Steelers" in text and "Delaware" not in text and "Baltimore" not in text
    text, *_ = america("3 October", quiet=True)
    assert "Delaware" in text and "Cleveland Browns" not in text
    text, *_ = america("4 October", quiet=True)
    assert "Baltimore Ravens vs Tennessee Titans" in text and "Washington Commanders vs Indianapolis Colts" in text


def test_football_games_already_started_are_excluded_and_counted():
    late = datetime(2026, 10, 4, 13, 45, tzinfo=timezone.utc)      # the Commanders game began at 13:30 UTC, the Ravens' at 17:00
    text, *_ = america("4 October", now=late, quiet=True)
    assert "Baltimore Ravens vs Tennessee Titans" in text and "Washington Commanders" not in text
    assert "1 game(s) excluded because they had already started or finished" in text


def test_a_game_the_source_flags_as_started_is_excluded_even_before_its_kickoff_time():
    payload = json.loads(json.dumps(FOOTBALL))
    payload["data"]["tournaments"][0]["events"][0]["status"] = 1
    text, *_ = america("2 October", source=all_listings(football_games(payload)), quiet=True)
    assert "Cleveland Browns" not in text


def test_the_cfl_is_ignored_as_outside_the_allowed_competitions():
    # spec 12-1: America only looks at the NFL and college football; the CFL is counted as ignored and never looked up.
    text, _, gate, _ = america("3 October", quiet=True)
    assert "Ottawa Redblacks" not in text and "ignored: outside your allowed competitions" in text


# ================================================================ 7.1: evaluation
def test_football_picks_carry_percentage_provenance_and_the_floor_is_applied():
    _, _, gate, _ = america("2 October")
    football = [e for e in [*gate.passed, *gate.rejected] if e.match_id.sport == "american_football"]
    assert football and gate.passed
    for e in gate.passed:
        assert e.estimate.probability >= PROBABILITY_FLOOR and e.ev > 0
        assert e.estimate.inputs and all(i.source == "test-source" for i in e.estimate.inputs)
    assert any(e.estimate.probability < PROBABILITY_FLOOR and not e.passed for e in football)


def test_nothing_passing_replies_exactly_no_valid_play_pass():
    table = lambda f: insufficient(f.match_id, "the stats source has no data")   # noqa: E731
    sources = {"baseball": TableStats(table), "hockey": TableStats(table), "american_football": TableStats(table)}
    chunks = run(run_america(" 2 October", all_listings(), str, NOW, stats_source=MultiSportStats(sources, str)))
    assert chunks[0] == "NO VALID PLAY - PASS"
    gates = []
    run(run_america(" 2 October", all_listings(), str, NOW, stats_source=MultiSportStats(sources, str), on_result=gates.append))
    assert not gates[0].passed                                     # nothing to offer: no list


def test_a_football_outage_blanks_only_football():
    class Down:
        async def fetch_stats(self, fixtures):
            from predictions.core.ports import StatsFetchError
            raise StatsFetchError("ESPN returned HTTP 500")

    sources = {"baseball": TableStats(strong), "hockey": TableStats(strong), "american_football": Down()}
    gates = []
    chunks = run(run_america(" 30 September and 2 October", all_listings(), str, NOW,
                             stats_source=MultiSportStats(sources, str), on_result=gates.append))
    text = "\n".join(chunks)
    assert text == ""                                             # something passed: only the list is shown
    assert {e.match_id.sport for e in [*gates[0].passed, *gates[0].rejected]} == {"baseball", "hockey"}


class FakeAnalyst:
    """Supplies every requested football input, labeled by the reader as analyst-sourced."""

    def __init__(self):
        self.queries = []

    async def research(self, query):
        from predictions.core.analyst import AnalystUsage, parse_report
        from predictions.core.inputs import FOOTBALL_REGISTRY
        self.queries.append(query)
        values_ = {"matches": 5, "points_for": 28.0, "points_against": 19.0}
        stats = [{"name": n, "value": values_[n.partition(".")[2]], "unit": FOOTBALL_REGISTRY[n.partition(".")[2]].unit,
                  "citation": "example.com/cfb/stats"} for n in query.wanted if n.partition(".")[2] in values_]
        return parse_report(json.dumps({"stats": stats, "flags": []}), query, AnalystUsage(1000, 200, 2))


def test_the_analyst_fills_a_college_game_no_free_source_covers_labeled_analyst_sourced():
    def only_missing(f):
        return insufficient(f.match_id, "no free stats source covers this game")
    analyst = FakeAnalyst()
    text, _, gate, _ = america("2 October", table=only_missing, analyst=analyst)
    football_queries = [q for q in analyst.queries if q.sport == "american_football"]
    assert football_queries and {"home.points_for", "away.points_against", "home.matches"} <= set(football_queries[0].wanted)
    assert text == ""   # the hockey pick still passes; at the 70% floor no college pick does, but each is priced from the analyst's inputs
    inputs_ = [i for e in [*gate.passed, *gate.rejected] if e.match_id.sport == "american_football" for i in e.estimate.inputs]
    assert inputs_ and any(i.source == "analyst-sourced:example.com/cfb/stats" for i in inputs_)


def test_an_over_on_an_analyst_filled_game_is_insufficient_because_wind_is_not_something_the_analyst_supplies():
    analyst = FakeAnalyst()
    _, _, gate, _ = america("2 October", table=lambda f: insufficient(f.match_id, "no data"), analyst=analyst)
    football = [e for e in [*gate.passed, *gate.rejected] if e.match_id.sport == "american_football"]
    overs = [e for e in football if "Over" in e.selection]
    assert overs and all(e.insufficient and "no wind forecast" in e.reason for e in overs)


# ================================================================ the real stats adapter behind the multi-sport source
def test_america_with_the_real_football_adapter_on_fabricated_espn_data():
    api = Espn(current=_upcoming_for(FOOTBALL))
    sample = football_games()[0]                                  # Cleveland Browns v Pittsburgh Steelers on 2 October
    browns = Fixture(replace(sample.match_id, away="Buffalo Bills"), sample.competition, sample.home, "Buffalo Bills",
                     sample.kickoff_utc, False, sample.markets, sample.bookmaker_ref)      # the fabricated ESPN league has the Bills
    listed = [browns] + [f for f in football_games() if f.match_id != sample.match_id]
    sources = {"american_football": api.source()}
    gates = []
    chunks = run(run_america(" 2 October", MultiSportFixtures({"american_football": FakeFixtures(listed)}), str,
                             datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc), stats_source=MultiSportStats(sources, str),
                             on_result=gates.append))
    stats = run(MultiSportStats(sources, str).fetch_stats(listed))
    assert stats[browns.match_id].status == READY
    assert any("was not found among the COLLEGE teams" in r for st in stats.values() for r in st.reasons)
    football = [e for e in [*gates[0].passed, *gates[0].rejected] if e.match_id.sport == "american_football"]
    assert football and all(i.source.startswith(("espn-football", "open-meteo", "stadium-table")) for e in football
                            for i in e.estimate.inputs)


def _upcoming_for(payload):
    from test_football_stats import current_board, event
    when = datetime(2026, 10, 2, 0, 15, tzinfo=timezone.utc)
    return current_board([event(700, when, "Cleveland Browns", "Buffalo Bills", "Cleveland", completed=False)])


# ================================================================ Telegram trigger, shared list, loading, dry run
def owner_message(text):
    from types import SimpleNamespace
    sent = []

    async def reply_text(message, **kwargs):
        sent.append((message, kwargs))

    return SimpleNamespace(effective_user=SimpleNamespace(id=bot.OWNER_ID), callback_query=None,
                           message=SimpleNamespace(text=text, reply_text=reply_text)), sent


@pytest.fixture
def wired(monkeypatch):
    real = bot.run_america

    async def fixed_clock(argument, source, sanitize, **kwargs):
        return await real(argument, source, sanitize, now=NOW, **kwargs)

    listings = {"baseball": FakeFixtures(mlb_games()), "hockey": FakeFixtures(nhl_games()),
                "american_football": FakeFixtures(football_games())}
    tables = {"baseball": strong, "hockey": strong, "american_football": football_stats}
    monkeypatch.setattr(bot, "run_america", fixed_clock)
    monkeypatch.setattr(bot, "AMERICA_SPORTS", {s: ((lambda s=s: listings[s]), (lambda s=s: TableStats(tables[s]))) for s in listings})
    monkeypatch.setattr(bot.ClaudeAnalyst, "from_environment", classmethod(lambda cls, mode="full": None))
    monkeypatch.setattr(bot, "STORE", None)
    monkeypatch.setattr(bot, "SESSIONS", SessionManager(clock=lambda: NOW))
    return listings


def test_america_shows_one_shared_list_with_mlb_nhl_and_football_picks(wired):
    update, sent = owner_message("America 30 September and 1 October and 2 October")
    run(handled(update))
    drop_ack(sent)
    assert len(sent) == 1                                             # only the list: no fixture listing
    last_text, last_kwargs = sent[-1]
    assert "USA - NFL" not in last_text and last_text.startswith("Final list:") and "reply_markup" in last_kwargs
    assert bot.SESSIONS.active.origin == "america"
    sports = {p.evaluation.match_id.sport for p in bot.SESSIONS.active.slip.picks}
    assert "american_football" in sports and sports >= {"baseball"}


def test_america_football_only_with_nothing_passing_replies_exactly_no_valid_play_pass(monkeypatch, wired):
    monkeypatch.setattr(bot, "AMERICA_SPORTS", {"american_football": (lambda: FakeFixtures(football_games()),
                                                                      lambda: TableStats(lambda f: insufficient(f.match_id, "no data")))})
    update, sent = owner_message("America 2 October")
    run(handled(update))
    drop_ack(sent)
    assert sent[0][0] == "NO VALID PLAY - PASS" and bot.SESSIONS.active is None


def test_a_football_pick_is_confirmed_loaded_and_priced_in_dry_run_without_placing():
    _, _, gate, _ = america("2 October")
    pick = next(e for e in gate.passed if e.match_id.sport == "american_football")
    manager = SessionManager(clock=lambda: NOW)
    session = manager.start(gate.passed, "abcd1234", "america")
    assert "Final list:" in list_screen(session).text
    mine = next(p for p in session.slip.picks if p.evaluation.match_id.sport == "american_football")
    for other in list(session.slip.picks):
        if other.pick_id != mine.pick_id:
            manager.act(OWNER, "abcd1234", manager.active.state_version, Action.DELETE, other.pick_id)
    manager.act(OWNER, "abcd1234", manager.active.state_version, Action.DONE)
    result = manager.act(OWNER, "abcd1234", manager.active.state_version, Action.CONFIRM)
    assert result.status is Status.APPLIED and manager.active.state is State.LOADING_SLIP
    bookmaker = FakeBookmaker(football_games() + mlb_games() + nhl_games(), dry_run=True, now=NOW)
    out = run(run_load(manager, bookmaker, "abcd1234", manager.active.state_version, str))
    assert out.kind == KIND_LOADED and manager.active.state is State.AWAITING_STAKE
    assert {leg.leg_key for leg in manager.active.loaded.legs} == {leg_from_evaluation(mine.evaluation).key}
    assert not any(c[0] == "place" for c in bookmaker.calls) and bookmaker.clicks == 0


def test_every_football_pick_the_engine_can_pass_is_available_on_its_listing():
    _, _, gate, _ = america("2 October")
    picks = [e for e in gate.passed if e.match_id.sport == "american_football"]
    assert picks
    fixtures = football_games()
    from predictions.core.betting import assess_availability
    found = assess_availability([leg_from_evaluation(e) for e in picks], fixtures, NOW)
    assert all(f.status is Availability.OK for f in found)


# ================================================================ Review judges football (same engine, the code's odds)
def code_selection(market_id, market_name, specifier, outcome, odds, competition="USA - NFL"):
    return CodeSelection("american_football", "American Football", competition, "Cleveland Browns", "Pittsburgh Steelers",
                         datetime(2026, 10, 2, 0, 15, tzinfo=timezone.utc), False, market_id, market_name, specifier, outcome,
                         Decimal(odds))


def judge(selection, stats):
    fixture = Fixture(selection.match_id, selection.competition, selection.home, selection.away, selection.kickoff_utc, False)
    return judge_selection(selection, fixture, stats)


def test_review_judges_a_football_selection_by_the_football_engine_at_the_codes_odds():
    selection = code_selection("225", "Over/Under (incl. overtime)", "total=40.5", "Over 40.5", "1.90")
    ready = FixtureStats(selection.match_id, READY, tuple(fb_inputs(**BOOST, wind=5.0)))
    ev = judge(selection, ready)
    assert ev.passed and ev.odds == Decimal("1.90") and ev.estimate.model == af.MODEL_NAME and "no rules" not in ev.reason
    windy = judge(selection, FixtureStats(selection.match_id, READY, tuple(fb_inputs(**BOOST, wind=15.0))))
    assert not windy.passed and "15 mph" in windy.reason and windy.rule.startswith(af.RULE_WIND)
    unknown = judge(selection, FixtureStats(selection.match_id, READY, tuple(fb_inputs(**BOOST, wind=None))))
    assert not unknown.passed and unknown.insufficient and "no wind forecast" in unknown.reason


def test_review_disqualifies_a_football_selection_without_data_as_insufficient_data_not_no_rules():
    ev = judge(code_selection("219", "Winner (incl. overtime)", "", "Home", "1.80"), None)
    assert not ev.passed and ev.reason.startswith("insufficient data") and "no rules" not in ev.reason


def test_review_says_no_rules_for_a_football_market_the_engine_does_not_price():
    ev = judge(code_selection("66", "1st half - handicap (2 way)", "hcp=0.5", "Home (+0.5)", "1.90"),
               FixtureStats(code_selection("66", "x", "", "y", "1.9").match_id, READY, tuple(fb_inputs(**BOOST))))
    assert not ev.passed and ev.reason == "no rules for this market"


class FakeReader:
    def __init__(self, selections):
        self.selections = selections

    async def read_code(self, code):
        return list(self.selections)


def test_a_review_of_a_code_with_a_football_selection_uses_the_football_stats_source():
    sources = {"american_football": TableStats(football_stats)}
    seen = []
    chunks = run(run_review(" ABC123", FakeReader([code_selection("225", "Over/Under (incl. overtime)", "total=40.5",
                                                               "Over 40.5", "1.90")]),
                            str, stats_sources=sources, now=datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc),
                            on_result=seen.append))
    text = "\n".join(chunks)
    assert sources["american_football"].asked == ["Cleveland Browns"]
    assert "APPROVED (1)" in text and "no rules for this sport" not in text


def test_review_statistics_sources_include_american_football():
    sources = bot.review_stats_sources()
    assert type(sources["american_football"]).__name__ == "EspnFootballStats" and "american_football" in ENGINES
