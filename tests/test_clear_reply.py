"""Story 9.2: a clearer reply when nothing is evaluated or passes. The first message stays exactly NO VALID PLAY - PASS
(FR-3); the next one starts with a plain-language WHY block; skipped games are listed compactly."""
import asyncio
from datetime import datetime, timezone
from decimal import Decimal

from openmarket import open_markets
from predictions.core.inputs import Input, SOCCER_REGISTRY
from predictions.core.stats import (CAT_AMBIGUOUS, CAT_ANALYST, CAT_GAME_UNLISTED, CAT_HISTORY, CAT_NOT_COVERED,
                                    CAT_OTHER, CAT_TEAM_UNKNOWN, CAT_UNREADABLE, READY, FixtureStats, ambiguous,
                                    insufficient, skip_category)
from predictions.core.types import Fixture, MarketOffer, MatchId, Outcome
from predictions.flow.america import run_america
from predictions.flow.gameday import run_gameday
from predictions.flow.matchday import run_matchday

NOW = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
KICKOFF = datetime(2026, 10, 1, 16, 0, tzinfo=timezone.utc)
NOT_COVERED = "this competition is not covered by the stats source"


def game(home, away, competition, sport="soccer", markets=()):
    return Fixture(MatchId(sport, competition, KICKOFF, home, away), competition, home, away, KICKOFF, False,
                   tuple(markets) or open_markets(sport))


class Listing:
    def __init__(self, fixtures):
        self.fixtures = fixtures

    async def fetch_fixtures(self):
        return self.fixtures


class Table:
    def __init__(self, answer):
        self.answer = answer

    async def fetch_stats(self, fixtures):
        return {f.match_id: self.answer(f) for f in fixtures}


def many(competition, count, start=0, sport="soccer"):
    return [game(f"Home{start + i}", f"Away{start + i}", competition, sport) for i in range(count)]


def chunks_for(fixtures, answer, runner=run_matchday, argument="1 October", **kw):
    return asyncio.run(runner(argument, Listing(fixtures), str, NOW, stats_source=Table(answer), **kw))


def not_covered(f):
    return insufficient(f.match_id, NOT_COVERED)


# ---- the exact first message and the WHY block ----
def test_first_message_is_exactly_no_valid_play_and_the_next_starts_with_the_why_block():
    fixtures = (many("International - UEFA Nations League", 8) + many("Region - Cup", 3, 100))
    chunks = chunks_for(fixtures, not_covered)
    assert chunks[0] == "NO VALID PLAY - PASS"
    assert chunks[1].startswith("Why there is no valid play:\n")
    assert chunks[1].splitlines()[1] == "- 11 game(s) looked at: 0 evaluated, 11 skipped."
    assert ("- Skipped, not covered by the stats source: 11 game(s) (International - UEFA Nations League 8, "
            "Region - Cup 3).") in chunks[1]
    assert chunks[1].index("Why there is no valid play") < chunks[1].index("Matchday fixtures for")


def test_the_why_block_shows_the_top_competitions_by_count_and_how_many_more():
    fixtures = []
    for n, count in enumerate((9, 8, 7, 6, 5, 2, 1)):
        fixtures += many(f"Region - Cup {n}", count, start=n * 100)
    block = chunks_for(fixtures, not_covered)[1]
    assert ("(Region - Cup 0 9, Region - Cup 1 8, Region - Cup 2 7, Region - Cup 3 6, Region - Cup 4 5 "
            "and 2 more competition(s))") in block
    assert "38 game(s) looked at: 0 evaluated, 38 skipped." in block


def test_reasons_are_grouped_with_counts():
    def answer(f):
        n = int(f.home.removeprefix("Home"))
        if n < 4:
            return not_covered(f)
        if n < 6:
            return insufficient(f.match_id, "too little history: X has 3 completed match(es)")
        if n == 6:
            return ambiguous(f.match_id, "Which team is Y?", ("Y1", "Y2"))
        if n == 7:
            return insufficient(f.match_id, "the home team Z is not in the stats source's list")
        if n == 8:
            return insufficient(f.match_id, "the stats source lists no A v B game near this kickoff")
        return insufficient(f.match_id, "something odd")
    block = chunks_for(many("Region - League", 10), answer)[1]
    assert "- Skipped, not covered by the stats source: 4 game(s) (Region - League 4)." in block
    assert "- Skipped, too little history: 2 game(s)." in block
    assert "- Skipped, ambiguous team match (needs your choice): 1 game(s)." in block
    assert "- Skipped, team not found in the stats source: 1 game(s)." in block
    assert "- Skipped, game not listed by the stats source: 1 game(s)." in block
    assert "- Skipped, other data problem: 1 game(s)." in block
    assert block.index("not covered") < block.index("too little history")  # biggest group first


def test_hint_says_what_could_change_it_and_mentions_the_analyst_only_when_it_is_off():
    fixtures = many("Region - League", 3)
    off = chunks_for(fixtures, not_covered, analyst_off_note="AI analyst is off (no ANTHROPIC_API_KEY is set).")[1]
    hint = [line for line in off.splitlines() if line.startswith("What could change it:")]
    assert len(hint) == 1 and "another date" in hint[0] and "ANTHROPIC_API_KEY" in hint[0]
    on = [line for line in chunks_for(fixtures, not_covered)[1].splitlines() if line.startswith("What could change it:")]
    assert len(on) == 1 and "another date" in on[0] and "ANTHROPIC_API_KEY" not in on[0]


def totals(line, over, under):
    return MarketOffer("18", "Over/Under", f"total={line}",
                       (Outcome(f"Over {line}", Decimal(over)), Outcome(f"Under {line}", Decimal(under))))


def ready_inputs(attack, defence):
    values = {"matches": 10, "xg_for": attack, "goals_for": attack, "xg_against": defence, "goals_against": defence}
    return [Input(f"{side}.{stat}", float(v), SOCCER_REGISTRY[stat].unit, "test")
            for side in ("home", "away") for stat, v in values.items()]


def test_evaluated_but_rejected_games_are_counted_in_the_why_block():
    ok = game("Alpha", "Beta", "Testland - League", markets=[totals("2.5", "1.90", "1.90")])
    off = game("Gamma", "Delta", "Region - Cup")

    def answer(f):
        return FixtureStats(f.match_id, READY, tuple(ready_inputs(2.0, 1.0))) if f is ok else not_covered(f)
    chunks = chunks_for([ok, off], answer)
    assert chunks[0] == "NO VALID PLAY - PASS"
    block = chunks[1]
    assert "- 2 game(s) looked at: 1 evaluated, 1 skipped." in block
    assert "pick(s) were priced from the 1 evaluated game(s) but none reached 70% with positive EV." in block
    assert "another date may have picks that clear the floor" in block
    assert block.index("Why there is no valid play") < block.index("Closest rejections:")


def test_no_why_block_when_a_pick_passes():
    ok = game("Alpha", "Beta", "Testland - League", markets=[totals("1.5", "1.60", "2.20")])
    chunks = chunks_for([ok], lambda f: FixtureStats(f.match_id, READY, tuple(ready_inputs(2.5, 1.5))))
    assert chunks == []   # something passes: no WHY block, no NO VALID PLAY, no listing; the interactive list is the reply


# ---- the compact listing ----
def test_skipped_games_of_one_competition_share_one_reason_line():
    fixtures = many("International - UEFA Nations League", 8) + many("Other - League", 3, 100)
    text = "\n".join(chunks_for(fixtures, not_covered)[1:])
    assert "insufficient data, not evaluated for all 8 game(s): " + NOT_COVERED in text
    assert "insufficient data, not evaluated for all 3 game(s): " + NOT_COVERED in text
    assert "    insufficient data: " + NOT_COVERED not in text  # no per-game long line any more
    assert text.count("insufficient data, not evaluated for") == 2  # one line per competition, not per game
    header = text.index("International - UEFA Nations League\n")
    assert text[header:].splitlines()[1].startswith("    insufficient data, not evaluated for all 8 game(s)")


def test_evaluated_games_and_specific_problems_keep_their_detail():
    ok = game("Alpha", "Beta", "Mixed - League", markets=[totals("2.5", "1.90", "1.90")])
    short = game("Gamma", "Delta", "Mixed - League")
    dark = game("Eta", "Theta", "Mixed - League")
    amb = game("Iota", "Kappa", "Mixed - League")

    def answer(f):
        if f is ok:
            return FixtureStats(f.match_id, READY, tuple(ready_inputs(2.0, 1.0)))
        if f is short:
            return insufficient(f.match_id, "too little history: Gamma has 3 completed match(es)")
        if f is amb:
            return ambiguous(f.match_id, "Which team is Iota?", ("I1", "I2"))
        return not_covered(f)
    text = "\n".join(chunks_for([ok, short, dark, amb], answer)[1:])
    assert "data: ready (10 inputs)" in text
    assert "insufficient data: too little history: Gamma has 3 completed match(es)" in text
    assert "needs your choice: Which team is Iota?" in text
    assert "insufficient data, not evaluated for 1 game(s): " + NOT_COVERED in text
    assert "    insufficient data: " + NOT_COVERED not in text


def test_a_game_the_analyst_could_not_fill_keeps_its_own_line():
    f = game("Alpha", "Beta", "Region - League")
    text = "\n".join(chunks_for([f], lambda x: insufficient(x.match_id, "the AI analyst call limit (10) was reached",
                                                           NOT_COVERED))[1:])
    assert "insufficient data: the AI analyst call limit (10) was reached" in text
    assert "- Skipped, the AI analyst could not fill the gap: 1 game(s)." in text


# ---- Gameday and America share the code ----
def test_gameday_and_america_get_the_same_first_message_and_why_block():
    basket = many("Region - Basketball League", 4, sport="basketball")
    chunks = chunks_for(basket, not_covered, runner=run_gameday)
    assert chunks[0] == "NO VALID PLAY - PASS" and chunks[1].startswith("Why there is no valid play:")
    assert "- Skipped, not covered by the stats source: 4 game(s) (Region - Basketball League 4)." in chunks[1]
    assert "Gameday fixtures for 1 Oct 2026" in chunks[1]
    mixed = many("USA - MLB", 2, sport="baseball") + many("USA - NHL", 3, 50, sport="hockey")
    chunks = chunks_for(mixed, not_covered, runner=run_america)
    assert chunks[0] == "NO VALID PLAY - PASS" and "5 game(s) looked at: 0 evaluated, 5 skipped." in chunks[1]
    assert "(USA - NHL 3, USA - MLB 2)" in chunks[1]


# ---- the categories ----
def test_skip_categories():
    def cat(*reasons):
        return skip_category(insufficient(game("a", "b", "c").match_id, *reasons))
    assert cat(NOT_COVERED) == CAT_NOT_COVERED
    assert cat("no free stats source covers X (ESPN: NBA and WNBA only)") == CAT_NOT_COVERED
    assert cat("no stats source is registered for hockey") == CAT_NOT_COVERED
    assert cat("missing required input: home.xg_for") == CAT_HISTORY
    assert cat("too little history: A has 2 completed match(es)") == CAT_HISTORY
    assert cat("the home team A is not in the stats source's EPL list") == CAT_TEAM_UNKNOWN
    assert cat("the stats source lists no A v B game near this kickoff") == CAT_GAME_UNLISTED
    assert cat("the hockey stats service could not be read: HTTP 500") == CAT_UNREADABLE
    assert cat("the stats source gave no answer for this game") == CAT_UNREADABLE
    assert cat("the AI analyst call limit (10) was reached", NOT_COVERED) == CAT_ANALYST
    assert cat("after the analyst lookup: missing required input: home.ortg", NOT_COVERED) == CAT_ANALYST
    assert cat("weird") == CAT_OTHER
    assert skip_category(ambiguous(game("a", "b", "c").match_id, "q", ())) == CAT_AMBIGUOUS
