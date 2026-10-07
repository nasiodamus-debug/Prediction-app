"""Stories 6.1 and 6.2 end to end: `Gameday <dates>` lists SportyBet basketball games (recorded listing sample,
tests/data/sportybet_basketball_sample.json, probed 2026-09-30), gathers stats with provenance, applies the 70% floor,
feeds the SAME interactive list and placement pipeline as Matchday, and `Review` judges basketball selections.
No test makes a live call."""
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
from openmarket import open_markets
from predictions.adapters.sportybet import fixtures as sb_fixtures
from predictions.adapters.sportybet.betslip import Credentials, SportyBetBookmaker
from predictions.adapters.sportybet.fixtures import FixtureFetchError, SportyBetFixtures, parse_page
from predictions.core.betting import Availability, leg_from_evaluation
from predictions.core.dates import LAGOS
from predictions.core.engines import ENGINES
from predictions.core.inputs import BASKETBALL_REGISTRY, Input
from predictions.core.model import PROBABILITY_FLOOR
from predictions.core.review import CodeSelection, judge_selection
from predictions.core.stats import READY, FixtureStats, insufficient
from predictions.core.types import Fixture, MarketOffer, MatchId, Outcome, Owner
from predictions.flow.gameday import run_gameday
from predictions.flow.loading import KIND_LOADED, run_load
from predictions.flow.matchday import run_matchday
from predictions.flow.review import run_review
from predictions.flow.session import Action, SessionManager, State, Status
from predictions.telegram.slate import MARK_KEPT, MARK_REMOVED, encode, handle_callback, list_screen
from test_basketball_engine import inputs as bb_inputs
from test_epic5_flow import OWNER

DATA = Path(__file__).resolve().parent / "data"
SAMPLE = json.loads((DATA / "sportybet_basketball_sample.json").read_text(encoding="utf-8"))
NOW = datetime(2026, 9, 30, 8, 0, tzinfo=timezone.utc)          # 09:00 in Lagos, before every game in the sample
NOW_NBA = datetime(2026, 10, 20, 10, 0, tzinfo=timezone.utc)    # the day of Detroit v Boston (kickoff 19:00 UTC)


def listing(payload=SAMPLE):
    return parse_page(payload, "basketball")[0]


def by_home(home, payload=SAMPLE):
    return next(f for f in listing(payload) if f.home == home)


def run(coro):
    return asyncio.run(coro)


class FakeFixtures:
    def __init__(self, fixtures=None, error=None):
        self.fixtures, self.error, self.calls = fixtures if fixtures is not None else listing(), error, 0

    async def fetch_fixtures(self):
        self.calls += 1
        if self.error:
            raise self.error
        return self.fixtures


class TableStats:
    """A stats source: `table(fixture)` gives the FixtureStats of each game."""

    def __init__(self, table):
        self.table, self.asked = table, []

    async def fetch_stats(self, fixtures):
        self.asked.extend(f.home for f in fixtures)
        return {f.match_id: self.table(f) for f in fixtures}


def ready(f, **kw):
    return FixtureStats(f.match_id, READY, tuple(_inputs_for(f, **kw)))


def _inputs_for(f, high=False, home=None, away=None, league_pace=None):
    """Team stats far above the NBA market (`high`) so that Over picks pass, else stats in line with it (a total of 219)."""
    strong = dict(ortg=107.0, drtg=124.0, pace=105.0) if high else dict(ortg=107.0, drtg=111.5, pace=100.0)
    inputs = bb_inputs(home or strong, away or strong, league_pace)
    return [Input(i.name, i.value, i.unit, i.source) for i in inputs]


def gameday(argument, source=None, table=None, now=NOW, **kwargs):
    gates = []
    stats = TableStats(table or (lambda f: insufficient(f.match_id, "the stats source has no data")))
    chunks = run(run_gameday(argument, source or FakeFixtures(), str, now, stats_source=stats, on_result=gates.append,
                             **kwargs))
    return "\n".join(chunks), chunks, (gates[0] if gates else None), stats


# ================================================================ Story 6.1: listing and dates
def test_the_listing_sample_parses_as_basketball_with_the_supported_markets_only():
    fixtures = listing()
    assert {f.match_id.sport for f in fixtures} == {"basketball"} and len(fixtures) == 6
    ids = {m.market_id for f in fixtures for m in f.markets}
    assert ids <= {"219", "223", "225", "227", "228", "68"} and {"219", "225", "68"} <= ids
    nba = by_home("Detroit Pistons")
    assert nba.competition == "USA - NBA" and nba.bookmaker_ref == "sr:match:73682354"
    assert not any(m.market_id == "225" and m.specifier in ("total=216.5", "total=217.5") for m in nba.markets)  # suspended lines


def test_gameday_lists_only_games_on_the_requested_lagos_dates():
    text, _, gate, _ = gameday("1 October")
    assert "Gameday fixtures for 1 Oct 2026" in text and "Washington Mystics vs Atlanta Dream" in text
    assert "Panathinaikos" not in text  # 30 September in Lagos
    # 23:00 UTC on 30 September is 00:00 on 1 October in Lagos: the date is the Lagos date, not the UTC date
    mystics = by_home("Washington Mystics")
    assert mystics.kickoff_utc.date().isoformat() == "2026-09-30" and mystics.kickoff_utc.astimezone(LAGOS).day == 1
    text, *_ = gameday("30 September")
    assert "Panathinaikos BC vs ASVEL Lyon-Villeurbanne" in text and "Maccabi Tel-Aviv vs Besiktas JK" in text
    assert "Washington Mystics" not in text


def test_gameday_accepts_several_dates():
    text, *_ = gameday("30 September and 20 October")
    assert "Detroit Pistons vs Boston Celtics" in text and "Maccabi" in text and "Mystics" not in text


def test_the_listing_shows_the_main_basketball_odds_with_the_line():
    text, *_ = gameday("20 October")
    assert "Home 1.81 / Away 2.05 | O/U 221.5: 1.90/1.87" in text  # the most balanced listed total


def test_games_already_started_are_excluded_and_counted():
    late = datetime(2026, 9, 30, 18, 10, tzinfo=timezone.utc)  # Maccabi kicked off at 18:05, Panathinaikos at 18:15
    text, _, gate, _ = gameday("30 September", now=late)
    assert "Panathinaikos BC vs ASVEL" in text and "Maccabi" not in text
    assert "1 game(s) excluded because they had already started or finished" in text


def test_a_game_the_source_flags_as_started_is_excluded_even_before_its_kickoff_time():
    payload = json.loads(json.dumps(SAMPLE))
    for t in payload["data"]["tournaments"]:
        for e in t["events"]:
            if e["homeTeamName"] == "Detroit Pistons":
                e["status"], e["matchStatus"] = 1, "1st quarter"
    text, *_ = gameday("20 October", source=FakeFixtures(listing(payload)))
    assert "Detroit" not in text.split("excluded")[0].splitlines()[0] and "1 game(s) excluded" in text


def test_a_missing_or_unreadable_date_is_asked_for_not_defaulted():
    text, _, gate, stats = gameday("")
    assert text.startswith("Which date? Send the date after Gameday") and gate is None and stats.asked == []
    text, *_ = gameday("soon")
    assert "I could not read that date" in text and "Gameday" in text


def test_only_past_dates_read_nothing():
    source = FakeFixtures()
    text, _, gate, _ = gameday("1 September", source=source)
    assert "Already passed" in text and source.calls == 0 and gate is None


def test_month_and_year_default_to_the_current_ones():
    text, *_ = gameday("30", now=NOW)
    assert "Gameday fixtures for 30 Sep 2026" in text


def test_an_unreachable_listing_is_reported_and_nothing_is_evaluated():
    text, _, gate, stats = gameday("30 September", source=FakeFixtures(error=FixtureFetchError("SportyBet did not answer in time")))
    assert "Could not get fixtures from SportyBet: SportyBet did not answer in time" in text
    assert "Nothing was listed" in text and gate is None and stats.asked == []


def test_the_fixtures_adapter_asks_sportybet_for_basketball_and_its_markets():
    seen = []

    def handler(request):
        seen.append(request.url.params)
        return httpx.Response(200, json=SAMPLE)

    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await SportyBetFixtures(client, sport="basketball").fetch_fixtures()

    fixtures = run(go())
    assert len(fixtures) == 6 and all(f.match_id.sport == "basketball" for f in fixtures)
    assert seen[0]["sportId"] == "sr:sport:2" and set(seen[0]["marketId"].split(",")) == {"219", "223", "225", "227", "228", "68"}
    assert sb_fixtures.SPORT_ID == "sr:sport:1"  # the soccer listing is untouched
    with pytest.raises(ValueError):
        SportyBetFixtures(sport="curling")


# ================================================================ Story 6.1: insufficient data, NO VALID PLAY - PASS
def test_games_without_data_are_insufficient_and_nothing_passing_replies_no_valid_play_pass():
    text, chunks, gate, _ = gameday("30 September")
    assert chunks[0] == "NO VALID PLAY - PASS"  # exactly, as the first message
    assert gate.passed == () and gate.evaluated == 0 and gate.skipped == 2
    assert "insufficient data: the stats source has no data" in text and "2 game(s) skipped for insufficient data" in text


def test_stats_that_give_no_pick_at_65_percent_reply_no_valid_play_pass_with_the_closest_rejections():
    text, chunks, gate, _ = gameday("20 October", table=lambda f: ready(f))
    assert chunks[0] == "NO VALID PLAY - PASS" and not gate.passed and gate.rejected
    assert "Closest rejections:" in text and "below the 70% floor" in text or "expected value" in text


def test_a_stats_outage_is_reported_and_no_game_is_evaluated():
    from predictions.core.ports import StatsFetchError

    def boom(f):
        raise StatsFetchError("ESPN returned HTTP 500")

    text, _, gate, _ = gameday("20 October", table=boom)
    assert "Could not get team stats: ESPN returned HTTP 500" in text and gate is None


def test_passing_picks_are_listed_with_their_percentage_and_the_original_rule_is_shown():
    text, chunks, gate, stats = gameday("20 October", table=lambda f: ready(f, high=True))
    assert gate.passed and chunks == []                     # the list is the whole reply
    over = [e for e in gate.passed if e.selection.startswith("Over")]
    assert over and over[0].estimate.probability >= PROBABILITY_FLOOR and over[0].ev > 0
    assert any(i.source == "test-source" for i in over[0].estimate.inputs)  # provenance travels with the pick
    shown = list_text(gate.passed)
    assert "%" in shown and "Over" in shown and "Valid plays" not in shown and "Estimate:" in detail_text(gate.passed)


# ================================================================ Story 6.1: the analyst may fill missing stats
class FakeAnalyst:
    """Supplies every requested basketball input, labeled by the reader as analyst-sourced."""

    VALUES = {"matches": 10, "ortg": 107.0, "drtg": 124.0, "pace": 105.0}

    def __init__(self, values=None):
        self.queries, self.values = [], values or self.VALUES

    async def research(self, query):
        from predictions.core.analyst import AnalystUsage, parse_report
        self.queries.append(query)
        stats = []
        for name in query.wanted:
            stat = name.partition(".")[2]
            if stat in self.values:
                stats.append({"name": name, "value": self.values[stat], "unit": BASKETBALL_REGISTRY[stat].unit,
                              "citation": "example.com/euroleague/stats"})
        return parse_report(json.dumps({"stats": stats, "flags": []}), query, AnalystUsage(1000, 200, 2))


def test_gameday_never_asks_the_analyst_for_missing_basketball_stats():
    # ortg, drtg and pace are not in search snippets: every lookup came back empty (live, 2026-10-03), so none is made
    analyst = FakeAnalyst()
    text, _, gate, _ = gameday("30 September", analyst=analyst)
    assert analyst.queries == [] and gate.evaluated == 0 and gate.skipped == 2 and "insufficient data" in text


def test_the_analyst_fills_missing_basketball_stats_labeled_analyst_sourced_and_the_model_prices_them(monkeypatch):
    # the path Review still uses: with the Gameday switch on, the analyst fills basketball stats and the model prices them
    from dataclasses import replace
    from predictions.flow import gameday as gameday_flow
    monkeypatch.setattr(gameday_flow, "GAMEDAY", replace(gameday_flow.GAMEDAY, analyst_fills_stats=True))
    analyst = FakeAnalyst()
    text, _, gate, _ = gameday("30 September", analyst=analyst)
    assert analyst.queries and all(q.sport == "basketball" for q in analyst.queries)
    assert {"home.ortg", "away.pace"} <= set(analyst.queries[0].wanted)
    assert gate.evaluated == 2 and gate.skipped == 0  # both Euroleague games were evaluated after the analyst filled them
    assert gate.passed and text == ""                      # only the list; sources are one tap away
    assert "analyst-sourced:example.com" in detail_text(gate.passed)
    # the estimate comes only from the model: every evaluation's inputs are the labeled ones
    for e in [*gate.passed, *gate.rejected]:
        assert all(i.source.startswith("analyst-sourced:") for i in e.estimate.inputs if i.name.endswith(".ortg"))


def test_an_analyst_value_outside_the_plausible_range_is_discarded_and_the_game_stays_insufficient():
    text, _, gate, _ = gameday("30 September", analyst=FakeAnalyst({"matches": 10, "ortg": 400.0, "drtg": 124.0, "pace": 105.0}))
    assert gate.evaluated == 0 and gate.skipped == 2 and "insufficient data" in text


def test_the_analyst_request_for_basketball_names_basketball_and_the_ratings():
    from predictions.adapters.analyst import claude
    from predictions.core.analyst import AnalystQuery
    request = claude.build_request(AnalystQuery("Panathinaikos BC", "ASVEL", "Euroleague", "2026-09-30",
                                                ("home.ortg", "away.pace"), "basketball"))
    assert "basketball" in request["system"] and "football" not in request["system"]
    body = request["messages"][0]["content"]
    assert '"home.ortg"' in body and "points per 100 possessions" in body and "possessions per game" in body


# ================================================================ Story 6.1: the shared list and placement pipeline
def picks_for_nba():
    _, _, gate, _ = gameday("20 October", table=lambda f: ready(f, high=True), now=NOW_NBA)
    return gate


def test_basketball_picks_appear_in_the_same_list_with_percentages_and_can_be_deleted():
    gate = picks_for_nba()
    manager = SessionManager(clock=lambda: NOW_NBA)
    session = manager.start(gate.passed, "abcd1234", "gameday")
    screen = list_screen(session)
    assert "Final list:" in screen.text and " - " in screen.text and "%" in screen.text and "Detroit Pistons" in screen.text
    assert any("Delete" in label for row in screen.buttons for label, _ in row)
    pick = session.slip.picks[0]
    reply = handle_callback(encode("abcd1234", session.state_version, Action.DELETE, pick.pick_id), OWNER, manager)
    assert reply.screen is not None and pick.pick_id not in manager.active.slip
    assert any(label.endswith("Keep") for row in reply.screen.buttons for label, _ in row)  # removed: Keep brings it back


def test_several_basketball_picks_for_one_match_are_options_of_one_game_with_the_advised_one_chosen():
    wnba = by_home("Washington Mystics")
    stats = ready(wnba, high=True, league_pace=80.0)
    from predictions.flow.evaluate import run_gate
    gate = run_gate([wnba], {wnba.match_id: stats})
    assert len(gate.passed) > 1  # a game total, team totals and a first-half total all clear the floor here
    manager = SessionManager(clock=lambda: NOW)
    session = manager.start(gate.passed, "abcd1234", "gameday")
    text = list_screen(session).text
    assert f"1a {MARK_KEPT}" in text and "(advised)" in text and f"1b {MARK_REMOVED}" in text   # lettered plays, advised first and kept
    assert len(session.slip) == 1 and session.slip.one_per_match            # one game contributes exactly one play
    other = session.games()[0].options[1]
    manager.act(OWNER, "abcd1234", session.state_version, Action.RESTORE, other.pick_id)
    assert manager.active.slip.one_per_match and [p.pick_id for p in manager.active.slip.picks] == [other.pick_id]


def test_a_basketball_pick_is_confirmed_loaded_priced_and_placed_like_any_other_in_dry_run():
    gate = picks_for_nba()
    nba = listing()
    manager = SessionManager(clock=lambda: datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc))
    manager.start(gate.passed, "abcd1234", "gameday")
    pick = manager.active.slip.picks[0]
    manager.act(OWNER, "abcd1234", manager.active.state_version, Action.RESTORE, pick.pick_id)
    manager.act(OWNER, "abcd1234", manager.active.state_version, Action.DONE)
    result = manager.act(OWNER, "abcd1234", manager.active.state_version, Action.CONFIRM)
    assert result.status is Status.APPLIED and result.load_next and manager.active.state is State.LOADING_SLIP
    bookmaker = FakeBookmaker(nba, dry_run=True, now=datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc))
    out = run(run_load(manager, bookmaker, "abcd1234", manager.active.state_version, str))
    assert out.kind == KIND_LOADED and manager.active.state is State.AWAITING_STAKE
    leg = leg_from_evaluation(pick.evaluation)
    loaded = manager.active.loaded.legs[0]
    assert loaded.leg_key == leg.key and loaded.odds == pick.evaluation.odds  # the pick's exact option is on the slip
    assert not any(c[0] == "place" for c in bookmaker.calls) and bookmaker.clicks == 0


def test_the_bookmaker_checks_a_basketball_leg_against_the_basketball_listing():
    gate = picks_for_nba()
    leg = leg_from_evaluation(gate.passed[0])
    calls = []

    class SoccerListing:
        async def fetch_fixtures(self):
            calls.append("soccer")
            return []

    class BasketballListing(FakeFixtures):
        async def fetch_fixtures(self):
            calls.append("basketball")
            return await super().fetch_fixtures()

    async def no_site():
        raise AssertionError("no browser is needed to check availability")

    bookmaker = SportyBetBookmaker(Credentials("08031234567", "pw"), dry_run=True, fixture_source=SoccerListing(),
                                   sport_fixture_sources={"basketball": BasketballListing()}, site_factory=no_site,
                                   clock=lambda: datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc))
    (found,) = run(bookmaker.check_availability([leg]))
    assert found.status is Availability.OK and found.outcome.odds == gate.passed[0].odds
    assert calls == ["basketball"]  # a basketball-only slip never reads the soccer listing


# ================================================================ Review judges basketball (Story 6.2)
def lakers_selection(outcome="Over 220.5", odds="1.90", market_id="225", specifier="total=220.5", market_name="Total"):
    return CodeSelection("basketball", "Basketball", "USA - NBA", "LA Lakers", "Boston Celtics",
                         datetime(2026, 10, 9, 23, 40, tzinfo=timezone.utc), False, market_id, market_name,
                         specifier, outcome, Decimal(odds))


def judge(selection, stats):
    fixture = Fixture(selection.match_id, selection.competition, selection.home, selection.away,
                      selection.kickoff_utc, False)
    return judge_selection(selection, fixture, stats)


def test_review_judges_a_basketball_selection_with_the_basketball_rules_and_the_same_floor():
    s = lakers_selection()
    strong = FixtureStats(s.match_id, READY, tuple(_inputs_for(SimpleNamespace(), high=True)))
    ev = judge(s, strong)
    assert ev.passed and ev.reason == "passed" and ev.estimate.probability >= PROBABILITY_FLOOR and ev.odds == Decimal("1.90")
    assert ev.estimate.model == "normal-points-v1"
    weak = judge(s, FixtureStats(s.match_id, READY, tuple(_inputs_for(SimpleNamespace()))))
    assert not weak.passed and "below the 70% floor" in weak.reason
    short = judge(lakers_selection(odds="1.02"), strong)
    assert not short.passed and "expected value" in short.reason


def test_review_no_longer_says_no_rules_for_this_sport_for_basketball():
    s = lakers_selection()
    ev = judge(s, None)
    assert ev.reason.startswith("insufficient data") and "no rules" not in ev.reason
    unknown = judge(lakers_selection(market_id="74", market_name="1st half - odd/even", specifier="", outcome="Odd"), None)
    assert unknown.reason == "no rules for this market"  # a basketball market the engine does not know


def test_review_judges_a_first_half_total_and_disqualifies_it_without_first_half_data():
    s = lakers_selection("Over 108.5", market_id="68", specifier="total=108.5", market_name="1st half - total")
    no_fh = judge(s, FixtureStats(s.match_id, READY, tuple(_inputs_for(SimpleNamespace(), high=True))))
    assert not no_fh.passed and no_fh.insufficient and "first-half" in no_fh.reason


class FakeReader:
    def __init__(self, selections):
        self.selections = selections

    async def read_code(self, code):
        return list(self.selections)


def test_a_review_of_a_code_uses_the_basketball_stats_source_and_lists_the_verdict():
    s = lakers_selection()
    source = TableStats(lambda f: FixtureStats(f.match_id, READY, tuple(_inputs_for(f, high=True))))
    seen = []
    chunks = run(run_review(" ABC123", FakeReader([s]), str, stats_sources={"basketball": source},
                            now=datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc), on_result=seen.append))
    text = "\n".join(chunks)
    assert source.asked == ["LA Lakers"] and "APPROVED (1)" in text and "Over 220.5 @ 1.90" in text
    assert seen[0].approved and "no rules for this sport" not in text


def test_a_review_of_basketball_with_no_data_source_is_insufficient_data_not_no_rules():
    seen = []
    chunks = run(run_review(" ABC123", FakeReader([lakers_selection()]), str, stats_sources={},
                            now=datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc), on_result=seen.append))
    text = "\n".join(chunks)
    assert "DISQUALIFIED (1)" in text and "insufficient data" in text and "no rules for this sport" not in text


def test_the_stats_source_for_basketball_is_registered_for_review():
    sources = bot.review_stats_sources()
    assert set(sources) == {"soccer", "basketball", "baseball", "hockey", "american_football"}
    assert type(sources["basketball"]).__name__ == "EspnBasketballStats"
    assert "basketball" in ENGINES


# ================================================================ the Telegram trigger
def owner_message(text):
    sent = []

    async def reply_text(message, **kwargs):
        sent.append((message, kwargs))

    return SimpleNamespace(effective_user=SimpleNamespace(id=bot.OWNER_ID), callback_query=None,
                           message=SimpleNamespace(text=text, reply_text=reply_text)), sent


def test_there_are_no_legacy_triggers_left():
    assert not hasattr(bot, "LEGACY_TRIGGER_NAMES") and not hasattr(bot, "LEGACY_TRIGGERS_ENABLED")
    assert not hasattr(bot, "run_engine_audit") and not hasattr(bot, "USER_SESSIONS")


def test_gameday_runs_the_new_pipeline_and_shows_the_shared_list(monkeypatch):
    real = bot.run_gameday
    captured = {}

    async def fixed_clock(argument, source, sanitize, **kwargs):
        captured["stats"] = kwargs["stats_source"]
        return await real(argument, source, sanitize, now=NOW_NBA, **kwargs)

    class Espn(TableStats):
        def __init__(self):
            super().__init__(lambda f: ready(f, high=True))

    listing_kwargs = {}

    def fake_listing(**kwargs):
        listing_kwargs.update(kwargs)
        return FakeFixtures()

    monkeypatch.setattr(bot, "run_gameday", fixed_clock)
    monkeypatch.setattr(bot, "SportyBetFixtures", fake_listing)
    monkeypatch.setattr(bot, "EspnBasketballStats", Espn)
    monkeypatch.setattr(bot.ClaudeAnalyst, "from_environment", classmethod(lambda cls, mode="full": None))
    monkeypatch.setattr(bot, "STORE", None)
    monkeypatch.setattr(bot, "SESSIONS", SessionManager(clock=lambda: NOW_NBA))
    update, sent = owner_message("Gameday 20 October")
    asyncio.run(handled(update))
    drop_ack(sent)
    assert listing_kwargs == {"sport": "basketball"} and isinstance(captured["stats"], Espn)
    text = "\n".join(m for m, _ in sent)
    assert "fixtures for" not in text and "switched off" not in text and "Processing" not in text and len(sent) == 1
    assert all("parse_mode" not in kw for _, kw in sent)
    last_text, last_kwargs = sent[-1]
    assert last_text.startswith("Final list:") and "reply_markup" in last_kwargs  # the shared interactive list
    assert bot.SESSIONS.active.origin == "gameday"


def test_matchday_still_works_through_the_same_runner_and_keeps_its_wording():
    from datetime import date
    fixtures = [Fixture(MatchId("soccer", "England - Premier League", datetime(2026, 10, 3, 14, 0, tzinfo=timezone.utc),
                                "Alpha", "Beta"), "England - Premier League", "Alpha", "Beta",
                        datetime(2026, 10, 3, 14, 0, tzinfo=timezone.utc), False, open_markets())]
    text = "\n".join(run(run_matchday("3 October", FakeFixtures(fixtures), str, datetime(2026, 10, 1, 9, tzinfo=timezone.utc))))
    assert "Matchday fixtures for 3 Oct 2026" in text and "Gameday" not in text
    assert run(run_matchday("", FakeFixtures(fixtures), str)) == [
        "Which date? Send the date after Matchday, for example: Matchday 20 and 22 September, Matchday today or Matchday tomorrow. I never pick a day for you."]


# ================================================================ the betslip wording (probed anonymously, 2026-09-30)
def _found(market_id, side, offer):
    from predictions.core.betting import Leg, assess_availability
    fixture = by_home("Washington Mystics")
    leg = Leg(fixture.match_id, market_id, offer.specifier, side, "m", "s", Decimal("1.5"))
    (found,) = assess_availability([leg], [fixture], NOW)
    assert found.status is Availability.OK, found.detail
    return found


def test_basketball_markets_are_looked_for_under_the_titles_the_game_page_and_betslip_really_use():
    fixture = by_home("Washington Mystics")
    first = lambda market_id: next(o for o in fixture.markets if o.market_id == market_id)  # noqa: E731
    name = SportyBetBookmaker._page_market_name
    total, team_home, first_half = first("225"), first("227"), first("68")
    line = total.specifier.partition("=")[2]
    assert name(_found("225", "over", total)) == f"Over/Under (incl. overtime) {line}"
    assert name(_found("227", "over", team_home)) == \
        f"Washington Mystics Over/Under (incl. overtime) {team_home.specifier.partition('=')[2]}"
    assert name(_found("228", "under", first("228"))) == \
        f"Atlanta Dream Over/Under (incl. overtime) {first('228').specifier.partition('=')[2]}"
    assert name(_found("68", "over", first_half)) == "1st Half - Over/Under"  # the line is in the outcome text only
    assert name(_found("219", "home", first("219"))) == "Winner (incl. overtime)"
    hcp = first("223")
    assert name(_found("223", "home", hcp)) == f"Handicap (incl. overtime) {hcp.specifier.partition('=')[2]}"


# ================================================================ every basketball evaluation is saved (FR-12)
def test_basketball_evaluations_are_saved_with_inputs_sources_and_the_rule_applied(tmp_path):
    from predictions.flow.store import EvaluationStore
    from test_basketball_engine import make_fixture, run as run_engine, team_totals, totals
    evals = run_engine(bb_inputs(dict(ortg=112.0), dict(ortg=100.0)),
                       totals() + team_totals("home", lines=("104.5",)))
    locked = next(e for e in evals if e.rule.startswith("ORTG lock"))
    with EvaluationStore.open(tmp_path / "p.db", lambda t: t) as store:
        store.save_evaluations("gameday-1", evals)
        rows = store.list_evaluations("gameday-1")
    row = next(r for r in rows if r["selection"] == locked.selection)
    assert row["sport"] == "basketball" and row["rule"].startswith("ORTG lock") and row["original"]
    assert row["inputs"] and row["placed"] == 0 and len(rows) == len(evals)
