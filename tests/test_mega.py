"""Mega: football only, every competition, ONE pick per game on a 60% floor with odds of at least 1.25 and no EV test,
one list across all the requested dates, capped by MEGA_MAX_PICKS. Fabricated samples only: no network."""
import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from predictions.core.model import REASON_PASSED
from predictions.core.types import Fixture, MatchId
from predictions.flow.evaluate import run_gate
from predictions.flow.mega import (DEFAULT_MEGA_MAX_PICKS, MEGA_FLOOR, MEGA_MIN_ODDS, mega_gate, parse_mega_max_picks,
                                   run_mega)
from test_matchday_plays import STRONG
from test_soccer_optimization import (EVEN, KICKOFF, NOW, Fixtures, Stats, fixture, make_inputs, ready, total_offer)

WEAK = make_inputs(0.8, 1.6, 0.9, 1.7)   # low-scoring: Over 1.5 is far from 60%
OUTSIDE = "Testland - League"              # in no allow-list: Matchday ignores it, Mega looks at it


def strong_game(home="Alpha", away="Beta", competition=OUTSIDE, over_15="1.10", over_25="1.40"):
    return fixture([total_offer("1.5", over_15, "7.00"), total_offer("2.5", over_25, "3.00")],
                   competition=competition, home=home, away=away)


def gate_of(fixtures, table=lambda f: ready(f, STRONG), **kwargs):
    stats = {f.match_id: table(f) for f in fixtures}
    return mega_gate(fixtures, stats, None, **kwargs)


def mega(fixtures, table=lambda f: ready(f, STRONG), argument="10 October", **kwargs):
    gates = []
    chunks = asyncio.run(run_mega(argument, Fixtures(fixtures), str, NOW, stats_source=Stats(table),
                                  on_result=gates.append, **kwargs))
    return chunks, (gates[0] if gates else None)


def test_the_constants_are_the_owners_rules():
    assert MEGA_FLOOR == 0.60 and MEGA_MIN_ODDS == Decimal("1.25") and DEFAULT_MEGA_MAX_PICKS == 20


def test_one_pick_per_game_the_highest_probability_with_odds_of_1_25_or_more():
    f = strong_game()
    gate = gate_of([f])
    assert len(gate.passed) == 1
    pick = gate.passed[0]
    # Over 1.5 @ 1.10 has the higher probability but its odds are under 1.25: discarded, the next highest is used
    assert pick.selection == "Over 2.5" and pick.odds == Decimal("1.40")
    assert pick.passed and pick.reason == REASON_PASSED and pick.estimate.probability >= MEGA_FLOOR
    discarded = [e for e in gate.rejected if e.selection == "Over 1.5"]
    assert discarded and "below the minimum of 1.25" in discarded[0].reason and not discarded[0].passed


def test_no_positive_ev_is_needed_but_matchday_would_refuse_the_same_pick():
    f = strong_game(over_25="1.25")
    pick = gate_of([f]).passed[0]
    assert pick.selection == "Over 2.5" and pick.ev < 0
    matchday = run_gate([f], {f.match_id: ready(f, STRONG)})
    assert "Over 2.5" not in {e.selection for e in matchday.passed}


def test_the_floor_is_sixty_percent_not_seventy():
    f = fixture([total_offer("2.5", "1.45", "2.70")], competition=OUTSIDE)
    between = make_inputs(2.2, 1.1, 2.2, 1.1)   # about 3.3 expected goals: Over 2.5 near 64%
    gate = gate_of([f], lambda g: ready(g, between))
    assert len(gate.passed) == 1 and 0.60 <= gate.passed[0].estimate.probability < 0.70
    assert not run_gate([f], {f.match_id: ready(f, between)}).passed


def test_a_pick_under_sixty_percent_is_never_listed():
    f = fixture([total_offer("2.5", "2.50", "1.55")], competition=OUTSIDE)
    gate = gate_of([f], lambda g: ready(g, WEAK))
    assert not gate.passed
    assert all("below the 60% floor" in e.reason for e in gate.rejected if not e.insufficient)


def test_every_competition_is_looked_at():
    chunks, gate = mega([strong_game()])
    assert chunks == [] and gate is not None and len(gate.passed) == 1   # "Testland - League" is outside Matchday's list


def test_a_game_that_has_kicked_off_is_never_listed():
    kicked_off = strong_game("Past", "Game")                              # 14:00 UTC
    upcoming = replace(strong_game("Next", "Game"), kickoff_utc=KICKOFF + timedelta(hours=4))
    upcoming = replace(upcoming, match_id=MatchId("soccer", OUTSIDE, upcoming.kickoff_utc, "Next", "Game"))
    gates = []
    now = KICKOFF + timedelta(hours=1)
    asyncio.run(run_mega("10 October", Fixtures([kicked_off, upcoming]), str, now,
                         stats_source=Stats(lambda f: ready(f, STRONG)), on_result=gates.append))
    assert [e.match_id.home for e in gates[0].passed] == ["Next"]


def test_all_the_dates_make_one_list_and_the_highest_probabilities_come_first():
    later = replace(strong_game("Gamma", "Delta"), kickoff_utc=KICKOFF + timedelta(days=1))
    later = replace(later, match_id=MatchId("soccer", OUTSIDE, later.kickoff_utc, "Gamma", "Delta"))
    chunks, gate = mega([strong_game(), later], argument="10 11 October")
    assert chunks == [] and len(gate.passed) == 2
    assert [e.estimate.probability for e in gate.passed] == sorted((e.estimate.probability for e in gate.passed), reverse=True)


def test_the_limit_keeps_the_highest_probabilities_and_says_how_many_were_left_out():
    games = [strong_game(f"Home{i}", f"Away{i}") for i in range(4)]
    inputs = {f"Home{i}": make_inputs(3.4 - 0.3 * i, 0.8, 1.3, 2.3) for i in range(4)}
    gate = gate_of(games, lambda f: ready(f, inputs[f.home]), max_picks=2)
    assert len(gate.passed) == 2
    kept = {e.estimate.probability for e in gate.passed}
    dropped = [e for e in gate.rejected if "Mega limit of 2" in e.reason]
    assert len(dropped) == 2 and all(d.estimate.probability <= min(kept) for d in dropped)
    assert any("2 more qualifying game(s) are not listed" in line for line in gate.footer)


def test_a_derby_is_still_passed():
    derby = fixture([total_offer("2.5", "1.40", "3.00")], competition="England - Premier League", home="Arsenal",
                    away="Tottenham Hotspur")
    gate = gate_of([derby])
    assert not gate.passed and [p.rule for p in gate.passes] == ["Derby pass"]


def test_nothing_qualifying_is_no_valid_play_and_the_wording_is_mega_s_own():
    chunks, gate = mega([fixture([total_offer("2.5", "2.50", "1.55")], competition=OUTSIDE)], lambda f: ready(f, WEAK))
    text = "\n".join(chunks)
    assert chunks[0] == "NO VALID PLAY" or chunks[0].startswith("NO VALID PLAY")
    assert "60%" in text and "70%" not in text and "positive EV" not in text
    assert gate is not None and not gate.passed and gate.floor == MEGA_FLOOR and not gate.ev_required


def test_a_missing_date_asks_for_one():
    chunks = asyncio.run(run_mega("", Fixtures([]), str, NOW))
    assert len(chunks) == 1 and "Mega" in chunks[0]


@pytest.mark.parametrize("raw,expected", [(None, 20), ("", 20), (" 12 ", 12), ("1", 1), ("200", 200)])
def test_max_picks_setting_is_read(raw, expected):
    assert parse_mega_max_picks(raw) == expected


@pytest.mark.parametrize("raw", ["0", "201", "-3", "abc", "2.5", "１２"])
def test_an_unreadable_max_picks_stops_the_bot_from_starting(raw):
    with pytest.raises(ValueError):
        parse_mega_max_picks(raw)


def test_the_bot_knows_the_trigger_and_names_the_list():
    import app as bot
    from predictions.telegram.slate import ORIGIN_NAMES
    assert "MEGA" in bot.LONG_REQUESTS and ORIGIN_NAMES["mega"] == "Mega"
    assert bot.MEGA_MAX_PICKS == 20 or bot.MEGA_MAX_PICKS >= 1
