"""Story 2.3: Estimate and the 65% gate."""
import asyncio
import re
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

from listview import detail_text, list_text
from predictions.core.engines.soccer import SoccerEngine, expected_goals, poisson_cdf, probability_over
from predictions.core.inputs import Input, SOCCER_REGISTRY
from predictions.core.model import PROBABILITY_FLOOR, Estimate, expected_value, validate
from predictions.core.stats import FixtureStats, READY, assess, insufficient
from predictions.core.types import Fixture, MarketOffer, MatchId, Outcome
from predictions.flow.evaluate import run_gate
from predictions.flow.matchday import run_matchday

KICKOFF = datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc)
NOW = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parent.parent


def inputs_for(attack, defence):
    """Both sides identical: xG and goals equal, so each side's expected goals = (attack + defence) / 2."""
    values = {"matches": 10, "xg_for": attack, "goals_for": attack, "xg_against": defence, "goals_against": defence}
    return [Input(f"{side}.{stat}", float(v), SOCCER_REGISTRY[stat].unit, "test")
            for side in ("home", "away") for stat, v in values.items()]


def totals(line, over, under):
    return MarketOffer("18", "Over/Under", f"total={line}",
                       (Outcome(f"Over {line}", Decimal(over)), Outcome(f"Under {line}", Decimal(under))))


def fixture(home="Alpha", away="Beta", markets=(), competition="Testland - League"):
    mid = MatchId("soccer", competition, KICKOFF, home, away)
    return Fixture(mid, competition, home, away, KICKOFF, False, tuple(markets))


class FakeStats:
    def __init__(self, table):
        self.table = table

    async def fetch_stats(self, fixtures):
        return {f.match_id: self.table(f) for f in fixtures}


class FakeFixtures:
    def __init__(self, fixtures):
        self.fixtures = fixtures

    async def fetch_fixtures(self):
        return self.fixtures


def run(fixtures, table):
    return asyncio.run(run_matchday("10 October", FakeFixtures(fixtures), str, NOW, stats_source=FakeStats(table)))


def ready(f, inputs):
    return FixtureStats(f.match_id, READY, tuple(inputs))


# ---- floor and EV ----
def test_649_rejected_with_estimate_and_reason():
    ok, ev, reason = validate(Estimate(0.649), Decimal("2.00"))
    assert not ok and "64.9%" in reason and "floor" in reason


def test_650_passes_with_positive_ev():
    ok, ev, reason = validate(Estimate(0.70), Decimal("1.60"))
    assert ok and ev > 0


def test_positive_ev_also_required():
    ok, ev, reason = validate(Estimate(0.70), Decimal("1.40"))  # 0.7 * 1.4 - 1 = -0.02
    assert not ok and ev < 0 and "expected value" in reason


def test_zero_ev_is_rejected():
    ok, ev, _ = validate(Estimate(0.8), Decimal("1.25"))
    assert ev == 0 and not ok


def test_ev_is_probability_times_odds_minus_one():
    assert expected_value(0.7, Decimal("1.6")) == Decimal("0.12")


def test_floor_value_and_rounding():
    assert PROBABILITY_FLOOR == 0.70
    # Kept to 0.1%, rounded DOWN: nothing under the floor can be lifted over it by rounding.`n    assert Estimate(0.6499).probability == 0.649`n    assert Estimate(0.6496).probability == 0.649`n    assert Estimate(0.65).probability == 0.65`n    assert Estimate(0.6509).probability == 0.65
    with pytest.raises(ValueError):
        Estimate(1.2)


def test_floor_defined_only_in_model():
    pattern = re.compile(r"(?<![\w.])0?\.65(?!\d)")
    offenders = []
    for path in (ROOT / "predictions").rglob("*.py"):
        if path.name == "model.py" and path.parent.name == "core":
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{path.name}:{n}")
    assert not offenders, offenders
    assert "PROBABILITY_FLOOR = " in (ROOT / "predictions" / "core" / "model.py").read_text(encoding="utf-8")


# ---- model ----
def test_poisson_cdf_known_values():
    assert poisson_cdf(2, 3.0) == pytest.approx(0.42319, abs=1e-4)
    assert poisson_cdf(0, 1.0) == pytest.approx(0.36788, abs=1e-4)


def test_expected_goals_average_attack_and_defence():
    home, away = expected_goals(inputs_for(2.0, 1.0))
    assert home == pytest.approx(1.5) and away == pytest.approx(1.5)


def test_over_probability_from_inputs():
    assert probability_over(inputs_for(2.0, 1.0), Decimal("2.5")) == pytest.approx(0.5768, abs=1e-3)


def test_deterministic_same_inputs_same_estimate():
    f = fixture(markets=[totals("2.5", "1.90", "1.90"), totals("1.5", "1.30", "3.20")])
    engine = SoccerEngine()
    a = engine.evaluate(f, inputs_for(2.5, 1.5), f.markets)
    b = engine.evaluate(f, list(reversed(inputs_for(2.5, 1.5))), f.markets)
    assert a == b and len(a) == 2   # the two Overs; no Under exists (AD-20)


def test_engine_picks_only_the_over_for_half_lines_in_a_top_league():
    # Top tier, not low tempo: half lines stand and the whole line is not used (no shield anywhere, AD-21).
    f = fixture(markets=[totals("2.5", "1.90", "1.90"), totals("2", "1.5", "2.5")],
                competition="Germany - Bundesliga")
    evals = SoccerEngine().evaluate(f, inputs_for(2.0, 1.0), f.markets)
    # owner rules: no Under in any sport (AD-20) and no whole line (AD-21): the listed whole line 2.0 is not a play
    assert {e.selection for e in evals} == {"Over 2.5"}
    assert next(e for e in evals if e.selection == "Over 2.5").rule == ""   # the half line stands: top tier, no shielding
    over = next(e for e in evals if e.selection == "Over 2.5")
    assert over.estimate.inputs and over.estimate.model


def test_engine_gives_nothing_without_required_inputs():
    f = fixture(markets=[totals("2.5", "1.90", "1.90")])
    assert SoccerEngine().evaluate(f, [], f.markets) == []


def test_engine_gate_marks_pass_and_reject():
    f = fixture(markets=[totals("1.5", "1.30", "3.20")])
    evals = {e.selection: e for e in SoccerEngine().evaluate(f, inputs_for(2.5, 1.5), f.markets)}
    assert evals["Over 1.5"].passed and evals["Over 1.5"].estimate.probability == pytest.approx(0.908, abs=1e-3)
    assert "Under 1.5" not in evals     # the 3.20 Under is never even priced (AD-20)


# ---- gate over fixtures ----
def test_not_ready_fixture_is_skipped():
    f = fixture(markets=[totals("1.5", "1.30", "3.20")])
    gate = run_gate([f], {f.match_id: insufficient(f.match_id, "no data")})
    assert gate.skipped == 1 and gate.evaluated == 0 and not gate.passed and not gate.rejected


def test_missing_stats_entry_is_skipped():
    f = fixture(markets=[totals("1.5", "1.30", "3.20")])
    assert run_gate([f], {}).skipped == 1


def test_assess_ready_stats_feed_the_engine():
    f = fixture(markets=[totals("1.5", "1.30", "3.20")])
    stats = assess(f.match_id, inputs_for(2.5, 1.5))
    assert stats.ready and run_gate([f], {f.match_id: stats}).passed


# ---- rendering ----
def test_reply_shows_passing_picks_with_percentages():
    f = fixture(markets=[totals("1.5", "1.30", "3.20")])
    gates = []
    chunks = asyncio.run(run_matchday("10 October", FakeFixtures([f]), str, NOW,
                                      stats_source=FakeStats(lambda fx: ready(fx, inputs_for(2.5, 1.5))),
                                      on_result=gates.append))
    shown, detail = list_text(gates[0].passed), detail_text(gates[0].passed)
    assert chunks == []                                    # the list is the whole reply: no listing, no WHY block
    assert "Alpha vs Beta" in shown and "Over 1.5 goals @ 1.30 - 90.8%" in shown
    assert "success 90.8%" in detail and "Expected value: +18.0%" in detail


def test_no_valid_play_reply_is_exact_with_closest_rejections():
    f = fixture(markets=[totals("2.5", "1.90", "1.90")])
    chunks = run([f], lambda fx: ready(fx, inputs_for(2.0, 1.0)))
    assert chunks[0] == "NO VALID PLAY - PASS"
    text = "\n".join(chunks[1:])
    assert "Closest rejections:" in text and "Over 2.5" in text and "57.6%" in text and "floor" in text


def test_all_insufficient_is_no_valid_play():
    f = fixture(markets=[totals("2.5", "1.90", "1.90")])
    chunks = run([f], lambda fx: insufficient(fx.match_id, "no data"))
    assert chunks[0] == "NO VALID PLAY - PASS"
    assert "1 game(s) skipped for insufficient data" in "\n".join(chunks)


def test_rejections_capped_and_chunks_within_limit():
    fixtures = [fixture(home=f"Team{i}", away=f"Other{i}", markets=[totals("2.5", "1.90", "1.90")])
                for i in range(300)]
    chunks = run(fixtures, lambda fx: ready(fx, inputs_for(2.0, 1.0)))
    assert chunks[0] == "NO VALID PLAY - PASS"
    assert "\n".join(chunks).count("Closest rejections:") == 1
    assert "\n".join(chunks).count("is below the") == 5
    assert all(len(c) <= 4000 for c in chunks)


def test_no_stats_source_keeps_plain_listing():
    f = fixture(markets=[totals("2.5", "1.90", "1.90")])
    chunks = asyncio.run(run_matchday("10 October", FakeFixtures([f]), str, NOW))
    assert "NO VALID PLAY" not in "\n".join(chunks) and "nothing evaluated" in chunks[0]
