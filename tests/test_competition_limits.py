"""Story 10.1: Matchday and Gameday look only at the competitions in predictions/core/competitions.py; the filter runs before
any stats or analyst work; Review disqualifies a selection outside the list; America is untouched; ESPN covers the allowed
top divisions; the analyst has a daily call limit kept in the database; passing replies end with the safety lines.
Fabricated listings only: no test makes a live call. Competition names are the real SportyBet spellings (probed 2026-10-01)."""
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

import app as bot
from predictions.adapters.espn import soccer as espn
from predictions.core.analyst import (DEFAULT_DAILY_CALL_LIMIT, AnalystUsage, parse_daily_call_limit, parse_report)
from predictions.core.competitions import REASON_OUTSIDE, is_allowed
from predictions.core.inputs import SOCCER_REGISTRY, Input
from predictions.core.stats import INSUFFICIENT, READY, FixtureStats, assess, insufficient
from predictions.core.types import Fixture, MarketOffer, MatchId, Outcome
from predictions.flow.america import run_america
from predictions.flow.analyst import ANALYST_CACHE, AnalystRun, DailyBudget
from predictions.flow.gameday import run_gameday
from predictions.flow.matchday import run_matchday
from predictions.flow.review import run_review
from predictions.flow.slate_run import AMERICA, GAMEDAY, MATCHDAY
from predictions.flow.stats import FirstCoveringStats
from predictions.flow.store import EvaluationStore
from openmarket import open_markets
from test_espn_soccer import KICKOFF as ESPN_KICKOFF, Api, event, fetch, played
from test_review import FakeCodeReader, selections

NOW = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
KICKOFF = datetime(2026, 10, 1, 16, 0, tzinfo=timezone.utc)
STATS = ("matches", "xg_for", "xg_against", "goals_for", "goals_against")
GOOD = {"matches": 10, "xg_for": 2.5, "goals_for": 2.5, "xg_against": 1.5, "goals_against": 1.5}


# ---------------------------------------------------------------- the table
@pytest.mark.parametrize("competition", [
    "England - Premier League", "Spain - LaLiga", "Italy - Serie A", "Germany - Bundesliga", "France - Ligue 1",
    "Netherlands - Eredivisie", "Portugal - Liga Portugal", "Belgium - Pro League", "Turkiye - Super Lig",
    "Scotland - Premiership", "Brazil - Brasileiro Serie A", "Argentina - Primera LPF", "USA - MLS", "Mexico - Liga MX",
    "Canada - Canadian Premier League", "Chile - Primera Division", "Colombia - Liga DIMAYOR",
    "International Clubs - UEFA Champions League", "International Clubs - UEFA Europa League",
    "International Clubs - UEFA Conference League", "International Clubs - CONMEBOL Libertadores",
    "International Clubs - CONMEBOL Sudamericana", "International Clubs - UEFA Champions League Women",
    "International - UEFA Nations League", "International - Int. Friendly Games",
    "International - CONCACAF Nations League", "International - Africa Cup of Nations Qualification",
    "International - FIFA World Cup Qualification UEFA", "International - World Cup, Qualification CAF",
    "International - European Championship Qualification", "International - Copa America",
    "International - World Cup Women", "International - Women World Cup Qualification Europe",
    "International - UEFA Nations League Women", "International - Int. Friendly Games, Women",
])
def test_allowed_soccer_competitions(competition):
    allowed, reason = is_allowed("Matchday", "soccer", competition)
    assert allowed and reason


@pytest.mark.parametrize("competition", [
    "England - EFL Cup", "Scotland - League Cup", "Japan - J.League Cup", "Portugal - Taca da Liga",
    "France - Coupe de la Ligue", "Argentina - Copa de la Liga",
])
def test_domestic_league_cups_are_allowed(competition):
    assert is_allowed("Matchday", "soccer", competition)[0]


@pytest.mark.parametrize("competition", [
    "England - Premier League 2", "England - Premier League, Reserves", "Argentina - Primera LPF, Reserves",
    "England Amateur - U21 Professional Development League", "International Youth - U21 UEFA European Championship, Qualification",
    "International Youth - U19 Friendly Games", "International Youth - U23 Friendly Games", "International - World Cup U20",
    "England Amateur - Super League Women", "Mexico - Liga MX, Women", "Switzerland - Super League, Women",
    "Argentina - Primera Division, Women", "Brazil - Brasileiro A1, Women", "USA - National Womens Soccer League",
    "England - FA Cup", "England - EFL Trophy", "England - National League Cup", "Israel - League Cup, National", "Germany - DFB Pokal", "Brazil - Copa do Brasil",
    "International Clubs - Club Friendly Games", "Simulated Reality League - SRL Club Friendlies",
    "Simulated Reality League - UEFA Nations League SRL", "England - Championship", "Germany - 2. Bundesliga",
    "Spain - LALIGA HYPERMOTION", "Portugal - Liga Portugal 2", "Greece - Super League 2", "Costa Rica - Liga de Ascenso",
    "Australia - A-League", "Japan - J2 League", "Ghana - Premier League", "Egypt - 2. Division A",
    "International - Asian Games, Women", "International - FIFA ASEAN Cup", "", None, "Premier League",
    "Colombia - Torneo DIMAYOR", "Sweden - Damallsvenskan",
])
def test_excluded_soccer_competitions(competition):
    allowed, reason = is_allowed("Matchday", "soccer", competition)
    assert not allowed and REASON_OUTSIDE in reason


@pytest.mark.parametrize("competition", [
    "USA - NBA", "USA - WNBA", "International - Euroleague", "Spain - Liga ACB",
    "Italy - Serie A", "Germany - BBL", "France - LNB Elite", "Greece - Basketball League", "Turkiye - Super Lig",
    "International - Liga ABA", "International - BNXT League", "Poland - PLK", "England - SLB", "Lithuania - LKL",
    "Austria - Superliga", "Belgium - Top Division 1", "Croatia - Premijer liga", "Denmark - Basketligaen",
    "Finland - Korisliiga", "Georgia - Superleague", "Hungary - NB I. A", "Sweden - SBL",
])
def test_allowed_basketball_competitions(competition):
    assert is_allowed("Gameday", "basketball", competition)[0]


@pytest.mark.parametrize("competition", [
    "USA - NCAA", "USA - NBA G League", "USA - NCAA, Regular Season", "France - LNB Elite 2", "Germany - Pro A",
    "Turkiye - Super Lig, Women", "Finland - Korisliiga, Women", "Greece - Elite League", "Australia - NBL",
    "International - Eurocup, Women", "International - EuroCup", "International - WABA Liga, Women", "USA - NBA, Summer League",
    "Mexico - LNBP", "Argentina - LNB", "Australia - WNBL", "Poland - 1. Liga", "Lithuania - NKL", "Sweden - Superettan",
    "Czechia - 1. Liga", "Estonia - I Liiga", "Russia - Super League", "Japan - B League One", "Hungary - NB I. A, Women",
    "Spain - Primera FEB", "France - Nationale 1", "Iceland - Iceland First Division", "International - 3x3 Europe Cup",
    "Germany - BBL U19", "",
])
def test_excluded_basketball_competitions(competition):
    allowed, reason = is_allowed("Gameday", "basketball", competition)
    assert not allowed and REASON_OUTSIDE in reason


def test_a_basketball_name_is_not_allowed_for_soccer_and_vice_versa():
    assert not is_allowed("Matchday", "soccer", "USA - NBA")[0]
    assert not is_allowed("Gameday", "basketball", "England - Premier League")[0]


@pytest.mark.parametrize("sport, competition", [
    ("hockey", "USA - NHL"), ("baseball", "USA - MLB"), ("american_football", "USA - NFL"),
    ("american_football", "USA - NCAA, Regular Season"),
])
def test_allowed_america_competitions(sport, competition):
    allowed, reason = is_allowed("America", sport, competition)
    assert allowed and reason != "not restricted"


@pytest.mark.parametrize("sport, competition", [
    ("hockey", "Russia - KHL"), ("hockey", "Sweden - SHL"), ("hockey", "Finland - Liiga"), ("hockey", "USA - AHL"),
    ("hockey", "USA - NCAA, Regular Season"), ("hockey", "Canada - OHL"), ("hockey", "International - Champions Hockey League"),
    ("baseball", "Japan - NPB"), ("baseball", "South Korea - KBO"), ("baseball", "Taiwan - CPBL"), ("baseball", "USA - MiLB"),
    ("baseball", "USA - NCAA"), ("american_football", "Canada - CFL"), ("american_football", "USA - XFL"),
    ("american_football", "USA - NCAA, Regular Season, Women"), ("american_football", "USA - Arena Football League"),
    ("hockey", ""), ("baseball", None),
])
def test_excluded_america_competitions(sport, competition):
    allowed, reason = is_allowed("America", sport, competition)
    assert not allowed and REASON_OUTSIDE in reason


def test_the_america_tables_are_per_sport_and_basketball_names_do_not_leak():
    assert not is_allowed("America", "hockey", "USA - MLB")[0] and not is_allowed("America", "baseball", "USA - NHL")[0]
    assert not is_allowed("America", "american_football", "USA - NHL")[0]
    assert not is_allowed("Gameday", "basketball", "USA - NCAA, Regular Season")[0]


def test_review_is_never_restricted_for_any_sport():
    for sport, competition in (("american_football", "Canada - CFL"), ("baseball", "Japan - NPB"), ("hockey", "Russia - KHL"),
                               ("soccer", "Germany - 2. Bundesliga"), ("basketball", "USA - NCAA"), ("soccer", "")):
        assert is_allowed("Review", sport, competition) == (True, "not restricted")
    assert MATCHDAY.restrict and GAMEDAY.restrict and AMERICA.restrict


# ---------------------------------------------------------------- helpers
def totals(line, over, under):
    return MarketOffer("18", "Over/Under", f"total={line}",
                       (Outcome(f"Over {line}", Decimal(over)), Outcome(f"Under {line}", Decimal(under))))


def game(home, away, competition, sport="soccer", hour=16, markets=None, day=1):
    """markets=None gives one open priced market (the game is open for betting); markets=() gives none (it is not)."""
    kickoff = KICKOFF.replace(hour=hour, day=day)
    offers = open_markets(sport) if markets is None else tuple(markets)
    return Fixture(MatchId(sport, competition, kickoff, home, away), competition, home, away, kickoff, False, offers)


class Listing:
    def __init__(self, fixtures):
        self.fixtures = fixtures

    async def fetch_fixtures(self):
        return self.fixtures


class Table:
    """A stats source that remembers which games it was asked about."""

    def __init__(self, answer):
        self.answer, self.asked = answer, []

    async def fetch_stats(self, fixtures):
        self.asked.extend(f.home for f in fixtures)
        return {f.match_id: self.answer(f) for f in fixtures}


class FakeAnalyst:
    def __init__(self, error=None):
        self.queries, self.error = [], error

    async def research(self, query):
        self.queries.append(query)
        import json
        if getattr(query, "sport", "soccer") != "soccer":   # other sports: nothing found, the call is only recorded
            return parse_report(json.dumps({"stats": [], "flags": []}), query, AnalystUsage(1000, 200, 2))
        stats = [{"name": n, "value": GOOD[n.split(".")[1]], "unit": SOCCER_REGISTRY[n.split(".")[1]].unit,
                  "citation": "example.com/table"} for n in query.wanted]
        return parse_report(json.dumps({"stats": stats, "flags": []}), query, AnalystUsage(1000, 200, 2))


def not_covered(f):
    return insufficient(f.match_id, "this competition is not covered by the stats source")


def good_stats(f):
    return assess(f.match_id, [Input(f"{side}.{stat}", float(GOOD[stat]), SOCCER_REGISTRY[stat].unit, "test")
                               for side in ("home", "away") for stat in STATS])


def matchday(fixtures, answer=not_covered, argument="1 October", now=NOW, **kw):
    stats, gates = Table(answer), []
    chunks = asyncio.run(run_matchday(argument, Listing(fixtures), str, now, stats_source=stats,
                                      on_result=gates.append, **kw))
    return chunks, (gates[0] if gates else None), stats


# ---------------------------------------------------------------- the filter runs before stats and the analyst
def test_games_outside_the_list_never_reach_the_stats_source_or_the_analyst():
    fixtures = [game("Alpha", "Beta", "England - Premier League"),
                game("Reserve A", "Reserve B", "England - Premier League, Reserves"),
                game("Cup A", "Cup B", "England - FA Cup"), game("Girls A", "Girls B", "Mexico - Liga MX, Women"),
                game("Sim A", "Sim B", "Simulated Reality League - UEFA Nations League SRL"),
                game("Youth A", "Youth B", "International Youth - U19 Friendly Games")]
    analyst = FakeAnalyst()
    chunks, gate, stats = matchday(fixtures, analyst=analyst)
    assert stats.asked == ["Alpha"]
    assert [q.home for q in analyst.queries] == ["Alpha"]
    assert set(gate.fixtures) == {fixtures[0].match_id}
    assert "Reserve" not in "\n".join(chunks) and "FA Cup" not in "\n".join(chunks)


def test_the_no_valid_play_why_block_counts_the_ignored_games():
    fixtures = [game("Alpha", "Beta", "England - Premier League")] + [
        game(f"H{i}", f"A{i}", "Germany - 2. Bundesliga") for i in range(3)]
    chunks, gate, stats = matchday(fixtures)
    assert chunks[0] == "NO VALID PLAY - PASS"
    assert "- 3 game(s) ignored: outside your allowed competitions" in chunks[1]
    assert "- 1 game(s) looked at" in chunks[1]  # only the allowed game was looked at
    assert "2. Bundesliga" not in "\n".join(chunks)


def test_when_every_game_is_outside_the_list_nothing_is_looked_up_and_the_reply_says_so():
    fixtures = [game(f"H{i}", f"A{i}", "Germany - 2. Bundesliga") for i in range(4)]
    chunks, gate, stats = matchday(fixtures, analyst=FakeAnalyst())
    assert gate is None and stats.asked == []
    text = "\n".join(chunks)
    assert "No games in your allowed competitions" in text and "4 game(s) ignored: outside your allowed competitions." in text


def test_gameday_filters_basketball_before_stats():
    fixtures = [game("Lakers", "Celtics", "USA - NBA", "basketball"), game("Aces", "Liberty", "USA - WNBA", "basketball"),
                game("Duke", "Kansas", "USA - NCAA", "basketball"), game("Reno", "Iowa", "USA - NBA G League", "basketball"),
                game("Real", "Barca", "Spain - Liga ACB", "basketball"), game("Vilnius", "Kaunas", "Lithuania - NKL", "basketball")]
    stats, gates = Table(not_covered), []
    chunks = asyncio.run(run_gameday("1 October", Listing(fixtures), str, NOW, stats_source=stats, on_result=gates.append))
    assert sorted(stats.asked) == ["Aces", "Lakers", "Real"]
    assert chunks[0] == "NO VALID PLAY - PASS" and "3 game(s) ignored: outside your allowed competitions" in chunks[1]


def test_america_filters_each_sport_by_its_own_table_before_stats():
    fixtures = [game("Ohio State", "Michigan", "USA - NCAA, Regular Season", "american_football"),
                game("Chiefs", "Bills", "USA - NFL", "american_football"),
                game("Yankees", "Red Sox", "USA - MLB", "baseball"), game("Bruins", "Rangers", "USA - NHL", "hockey"),
                game("Toronto", "Calgary", "Canada - CFL", "american_football"),
                game("Moscow", "Minsk", "Russia - KHL", "hockey"), game("Tokyo", "Osaka", "Japan - NPB", "baseball"),
                game("Seoul", "Busan", "South Korea - KBO", "baseball"), game("Duke", "UNC", "USA - NCAA", "basketball"),
                game("Maine", "Boston U", "USA - NCAA, Regular Season", "hockey")]
    stats, gates, analyst = Table(lambda f: insufficient(f.match_id, "no stats here")), [], FakeAnalyst()
    chunks = asyncio.run(run_america("1 October", Listing(fixtures), str, NOW, stats_source=stats, on_result=gates.append,
                                     analyst=analyst))
    assert sorted(stats.asked) == ["Bruins", "Chiefs", "Ohio State", "Yankees"]
    assert chunks[0] == "NO VALID PLAY - PASS" and "- 6 game(s) ignored: outside your allowed competitions" in chunks[1]
    text = "\n".join(chunks)
    for out in ("KHL", "NPB", "KBO", "CFL"):
        assert out not in text
    assert set(gates[0].fixtures) == {fixtures[i].match_id for i in (0, 1, 2, 3)}


def test_america_when_every_game_is_outside_nothing_is_looked_up():
    fixtures = [game("Moscow", "Minsk", "Russia - KHL", "hockey"), game("Toronto", "Calgary", "Canada - CFL", "american_football")]
    stats, gates = Table(not_covered), []
    chunks = asyncio.run(run_america("1 October", Listing(fixtures), str, NOW, stats_source=stats, on_result=gates.append))
    assert stats.asked == [] and not gates
    assert "No games in your allowed competitions" in "\n".join(chunks)


def test_america_passing_footer_has_the_ignored_count():
    ok = game("Chiefs", "Bills", "USA - NFL", "american_football")
    out = game("Toronto", "Calgary", "Canada - CFL", "american_football")
    stats, gates = Table(lambda f: insufficient(f.match_id, "no stats here")), []
    asyncio.run(run_america("1 October", Listing([ok, out]), str, NOW, stats_source=stats, on_result=gates.append))
    assert gates[0].footer == ("1 game(s) ignored: outside your allowed competitions.",)


# ---------------------------------------------------------------- Review is NOT restricted
def test_review_judges_a_lower_division_soccer_selection_with_stats_and_analyst():
    s = replace(selections()[0], competition="England - Premier League, Reserves", home="Zed", away="Yan")
    reader, stats, analyst, seen = FakeCodeReader([s]), Table(good_stats), FakeAnalyst(), []
    chunks = asyncio.run(run_review(" ABC123", reader, str, stats_sources={"soccer": stats}, now=NOW,
                                    on_result=seen.append, analyst=analyst))
    assert stats.asked == ["Zed"]
    assert seen[0].judgements[0].evaluation.reason != REASON_OUTSIDE
    assert "allowed competitions" not in "\n".join(chunks)


def test_review_judges_a_selection_in_the_list_as_before():
    s = replace(selections()[0], competition="England - Premier League")
    stats, seen = Table(good_stats), []
    asyncio.run(run_review(" ABC123", FakeCodeReader([s]), str, stats_sources={"soccer": stats}, now=NOW,
                           on_result=seen.append))
    assert stats.asked == [s.home] and seen[0].judgements[0].evaluation.reason != REASON_OUTSIDE


def test_review_judges_a_basketball_selection_from_a_league_gameday_ignores():
    base = [x for x in selections() if x.sport == "basketball"]
    s = replace(base[0], competition="USA - NCAA") if base else None
    if s is None:   # build one from a soccer selection
        s = replace(selections()[0], sport="basketball", sport_label="Basketball", competition="USA - NCAA")
    stats, seen = Table(lambda f: insufficient(f.match_id, "no data")), []
    asyncio.run(run_review(" ABC123", FakeCodeReader([s]), str, stats_sources={"basketball": stats}, now=NOW,
                           on_result=seen.append))
    assert stats.asked == [s.home] and seen[0].judgements[0].evaluation.reason != REASON_OUTSIDE


@pytest.mark.parametrize("sport, label, competition", [("baseball", "Baseball", "Japan - NPB"), ("hockey", "Ice Hockey", "Russia - KHL"),
                                                       ("american_football", "American Football", "Canada - CFL"),
                                                       ("american_football", "American Football", "USA - NCAA, Regular Season")])
def test_review_judges_american_sports_whatever_the_competition(sport, label, competition):
    s = replace(selections()[0], sport=sport, sport_label=label, competition=competition)
    stats, seen = Table(lambda f: insufficient(f.match_id, "no data")), []
    asyncio.run(run_review(" ABC123", FakeCodeReader([s]), str, stats_sources={sport: stats}, now=NOW, on_result=seen.append))
    assert stats.asked == [s.home] and seen[0].judgements[0].evaluation.reason != REASON_OUTSIDE


def test_review_does_not_apply_the_open_for_betting_check_or_the_matchday_play_types():
    # A booking-code selection has no listing markets at all: it is judged on its own odds.
    s = replace(selections()[0], competition="Germany - 2. Bundesliga")
    stats, seen = Table(good_stats), []
    asyncio.run(run_review(" ABC123", FakeCodeReader([s]), str, stats_sources={"soccer": stats}, now=NOW, on_result=seen.append))
    assert stats.asked == [s.home] and not any("not open for betting" in j.evaluation.reason for j in seen[0].judgements)


# ---------------------------------------------------------------- ESPN: more leagues, same composite order
def test_espn_has_slugs_for_the_top_divisions_understat_lacks_and_the_continental_cups():
    for competition, slug in (("Netherlands - Eredivisie", "ned.1"), ("Portugal - Liga Portugal", "por.1"),
                              ("Turkiye - Super Lig", "tur.1"), ("Scotland - Premiership", "sco.1"),
                              ("Belgium - Pro League", "bel.1"), ("Greece - Super League", "gre.1"),
                              ("Brazil - Brasileiro Serie A", "bra.1"), ("Argentina - Primera LPF", "arg.1"),
                              ("USA - MLS", "usa.1"), ("Mexico - Liga MX", "mex.1"),
                              ("International Clubs - CONMEBOL Libertadores", "conmebol.libertadores"),
                              ("International Clubs - CONMEBOL Sudamericana", "conmebol.sudamericana"),
                              ("International Clubs - UEFA Champions League", "uefa.champions"),
                              ("International Clubs - UEFA Europa League", "uefa.europa"),
                              ("International Clubs - UEFA Conference League", "uefa.europa.conf"),
                              ("International Clubs - CAF Champions League", "caf.champions"),
                              ("International Clubs - CONCACAF Champions Cup", "concacaf.champions"),
                              ("International - FIFA World Cup Qualification UEFA", "fifa.worldq.uefa"),
                              ("International - World Cup Qualification CAF", "fifa.worldq.caf"),
                              ("International - World Cup", "fifa.world"),
                              ("International - European Championship Qualification", "uefa.euroq"),
                              ("International - Africa Cup of Nations", "caf.nations"),
                              ("International - Copa America", "conmebol.america"),
                              ("International - Asian Cup", "afc.asian.cup"),
                              ("International - UEFA Nations League Women", "uefa.w.nations"),
                              ("International - Women World Cup Qualification Europe", "fifa.wworldq.uefa")):
        assert espn.slug_for(competition) == slug, competition
    for youth in ("International Youth - U19 Friendly Games", "International - World Cup U20", "England - Premier League, Reserves"):
        assert espn.slug_for(youth) is None


def test_the_six_leagues_understat_covers_are_left_to_understat_and_a_league_with_no_slug_costs_no_request():
    from predictions.adapters.understat.stats import LEAGUES
    assert not set(LEAGUES) & set(espn.DOMESTIC)
    f = Fixture(MatchId("soccer", "Poland - Ekstraklasa", ESPN_KICKOFF, "Legia", "Lech"), "Poland - Ekstraklasa", "Legia",
                "Lech", ESPN_KICKOFF, False)
    api = Api()
    result = fetch([f], api)[f.match_id]
    assert result.reasons == (espn.NOT_COVERED,) and api.requests == []


def test_an_eredivisie_game_is_answered_by_espn_goals_only():
    ajax, psv = ("139", "Ajax"), ("148", "PSV Eindhoven")
    f = Fixture(MatchId("soccer", "Netherlands - Eredivisie", ESPN_KICKOFF, "Ajax", "PSV Eindhoven"),
                "Netherlands - Eredivisie", "Ajax", "PSV Eindhoven", ESPN_KICKOFF, False)
    api = Api({("ned.1", "20261001"): [event(1, ESPN_KICKOFF, ajax, psv, 0, 0, completed=False)]},
              {("139", None): played(ajax, 10, 2, 1, 1000), ("148", None): played(psv, 10, 2, 1, 2000)})
    result = fetch([f], api)[f.match_id]
    assert result.status == READY and result.inputs[0].source == "espn:goals-only:ned.1"


def test_the_composite_order_is_still_understat_then_espn():
    from predictions.adapters.footballdata.goals import FootballDataGoalsStats
    from predictions.adapters.understat.stats import UnderstatStats
    source = bot.soccer_stats_source()
    assert isinstance(source, FirstCoveringStats)
    assert [type(s) for s in source._sources] == [UnderstatStats, espn.EspnSoccerStats, FootballDataGoalsStats]


# ---------------------------------------------------------------- the daily analyst limit
@pytest.fixture
def store(tmp_path):
    with EvaluationStore.open(tmp_path / "t.db", str) as opened:
        yield opened


def test_the_daily_count_is_persisted_across_a_store_reopen_and_resets_on_the_next_lagos_day(tmp_path):
    path = tmp_path / "usage.db"
    with EvaluationStore.open(path, str) as first:
        budget = DailyBudget(first, 3, NOW)
        budget.record(2)
        assert budget.used() == 2 and budget.remaining() == 1
    with EvaluationStore.open(path, str) as second:
        assert DailyBudget(second, 3, NOW).used() == 2          # a restart does not reset it
        assert DailyBudget(second, 3, NOW + timedelta(days=1)).used() == 0
        late = datetime(2026, 10, 1, 23, 30, tzinfo=timezone.utc)  # 00:30 on 2 October in Lagos
        assert DailyBudget(second, 3, late).day == "2026-10-02"
        assert second.schema_version == 7


def test_the_daily_limit_is_enforced_across_requests_and_the_analyst_is_skipped_with_a_note(store):
    def several(count):
        return [game(f"H{i}", f"A{i}", "Netherlands - Eredivisie", hour=12 + i) for i in range(count)]
    first = FakeAnalyst()
    chunks, gate, _ = matchday(several(2), analyst=first, budget=DailyBudget(store, 3, NOW))
    assert len(first.queries) == 2 and DailyBudget(store, 3, NOW).used() == 2
    ANALYST_CACHE.clear()  # spec 14-1: the same games would be answered from the cache; this test is about the limit
    second = FakeAnalyst()  # a second request the same day: only one call is left
    chunks, gate, _ = matchday(several(3), analyst=second, budget=DailyBudget(store, 3, NOW))
    text = "\n".join(chunks)
    assert len(second.queries) == 1 and DailyBudget(store, 3, NOW).used() == 3
    assert "Daily analyst limit of 3 calls reached" in text
    third = FakeAnalyst()  # the limit is used up: nothing is asked
    chunks, gate, _ = matchday(several(2), analyst=third, budget=DailyBudget(store, 3, NOW))
    assert third.queries == [] and "daily limit (3 calls) was reached" in "\n".join(chunks)
    fresh = FakeAnalyst()  # the next day starts again
    tomorrow = [game("H", "A", "Netherlands - Eredivisie", day=2)]
    matchday(tomorrow, analyst=fresh, budget=DailyBudget(store, 3, NOW + timedelta(days=1)), argument="2 October",
             now=NOW + timedelta(days=1))
    assert len(fresh.queries) == 1


def test_the_per_request_cap_still_applies_beside_the_daily_limit(store):
    run = AnalystRun(FakeAnalyst(), limit=1, budget=DailyBudget(store, 40, NOW))
    f = game("H", "A", "England - Premier League")
    asyncio.run(run.ask(f, ("home.matches",)))
    other = game("H2", "A2", "England - Premier League")   # a different game: the same one would come from the cache (spec 14-1)
    assert asyncio.run(run.ask(other, ("home.matches",))) is None and run.limit_hit and not run.daily_hit


def test_a_store_that_cannot_be_read_does_not_block_the_request():
    class Broken:
        def analyst_calls_on(self, day):
            raise RuntimeError("disk")

        def add_analyst_calls(self, day, count=1):
            raise RuntimeError("disk")
    budget = DailyBudget(Broken(), 3, NOW)
    assert budget.remaining() == 3 and budget.broken


@pytest.mark.parametrize("raw, expected", [(None, DEFAULT_DAILY_CALL_LIMIT), ("", DEFAULT_DAILY_CALL_LIMIT),
                                           ("  ", DEFAULT_DAILY_CALL_LIMIT), ("25", 25), (" 40 ", 40)])
def test_the_daily_limit_setting_is_read_like_max_stake(raw, expected):
    assert parse_daily_call_limit(raw) == expected


@pytest.mark.parametrize("raw", ["abc", "0", "-5", "2.5", "1e3", "forty", "99999999", "٣"])
def test_an_unreadable_daily_limit_refuses_to_start_and_never_echoes_the_value(raw):
    with pytest.raises(ValueError) as error:
        parse_daily_call_limit(raw)
    assert (len(raw) < 3 or raw not in str(error.value)) and "ANALYST_DAILY_CALL_LIMIT" in str(error.value)


def test_the_default_is_forty_and_app_reads_it_at_startup():
    # The owner's real config.env may set ANALYST_DAILY_CALL_LIMIT (it is loaded when app is imported), so this test
    # checks the default through the parser and only that the startup value is a sane positive whole number.
    assert DEFAULT_DAILY_CALL_LIMIT == 40
    assert bot.parse_daily_call_limit(None) == 40 and bot.parse_daily_call_limit("") == 40
    assert isinstance(bot.ANALYST_DAILY_CALL_LIMIT, int) and bot.ANALYST_DAILY_CALL_LIMIT > 0
    assert isinstance(bot.analyst_budget(), (DailyBudget, type(None)))


# ---------------------------------------------------------------- the safety lines on a passing reply
def test_a_passing_reply_ends_with_the_analyst_cost_line_and_the_ignored_count(store):
    fixtures = [game("Alpha", "Beta", "Netherlands - Eredivisie", markets=[totals("1.5", "1.30", "3.20")]),
                game("Cup A", "Cup B", "England - FA Cup")]
    analyst = FakeAnalyst()
    chunks, gate, _ = matchday(fixtures, analyst=analyst, budget=DailyBudget(store, 40, NOW))
    assert gate.passed and chunks == []  # the interactive list is the whole reply; the footer rides on it
    assert 1 <= len(gate.footer) <= 3
    assert gate.footer[0] == "1 game(s) ignored: outside your allowed competitions."
    assert gate.footer[1].startswith("AI analyst: 1 call(s) used, about $") and "estimate" in gate.footer[1]
    assert all(len(line) < 120 for line in gate.footer)


def test_a_passing_reply_warns_when_a_stats_service_could_not_be_read():
    ok = game("Alpha", "Beta", "Netherlands - Eredivisie", markets=[totals("1.5", "1.30", "3.20")])
    down = game("Gamma", "Delta", "Spain - LaLiga")

    def answer(f):
        if f is down:
            return insufficient(f.match_id, "a stats service could not be read: Understat returned HTTP 500")
        return good_stats(f)
    chunks, gate, _ = matchday([ok, down], answer)
    assert gate.passed and gate.footer == ("Warning: the soccer stats service could not be read: games from it were not evaluated.",)
    assert not any("AI analyst" in line for line in gate.footer)  # no analyst call, no cost line


def test_a_clean_passing_reply_has_no_footer():
    ok = game("Alpha", "Beta", "Netherlands - Eredivisie", markets=[totals("1.5", "1.30", "3.20")])
    chunks, gate, _ = matchday([ok], good_stats)
    assert gate.passed and gate.footer == ()


def test_offer_slate_puts_the_footer_at_the_end_of_the_list_message():
    from types import SimpleNamespace
    ok = game("Alpha", "Beta", "Netherlands - Eredivisie", markets=[totals("1.5", "1.30", "3.20")])
    chunks, gate, _ = matchday([ok], good_stats)
    sent = []

    async def reply_text(text, reply_markup=None, **kw):
        sent.append(text)
    update = SimpleNamespace(effective_user=SimpleNamespace(id=bot.OWNER_ID), message=SimpleNamespace(reply_text=reply_text))
    bot.SESSIONS.end()
    asyncio.run(bot.offer_slate(update, gate.passed, "matchday", "sess1", ("line one", "line two")))
    assert len(sent) == 1 and sent[0].endswith("line one\nline two")
    bot.SESSIONS.end()


def test_no_valid_play_stays_exactly_the_same_first_message():
    chunks, gate, _ = matchday([game("Alpha", "Beta", "Netherlands - Eredivisie")])
    assert chunks[0] == "NO VALID PLAY - PASS"
