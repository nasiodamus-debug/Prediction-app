"""Spec 14-1, item 4: the analyst's per-fixture calls run a few at a time (flow/analyst.py) while the per-request cap, the daily
limit, the earliest-kickoff-first rule, the outage stance and the schema validation hold exactly. Fake analysts that sleep."""
import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest

from predictions.core.analyst import AnalystFlag, AnalystUsage, ContextFlag, parse_report
from predictions.core.inputs import SOCCER_REGISTRY
from predictions.core.ports import AnalystUnavailable
from predictions.core.stats import INSUFFICIENT, insufficient
from predictions.flow.analyst import (ANALYST_CONCURRENCY, AnalystRun, DailyBudget, check_flags, fill_missing_inputs,
                                      gather_in_order)
from test_competition_limits import GOOD, game

run = asyncio.run
NOW = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)
COMPETITION = "Netherlands - Eredivisie"


def games(count):
    return [game(f"H{i}", f"A{i}", COMPETITION, hour=10 + i) for i in range(count)]


def lacking(fixtures):
    return {f.match_id: insufficient(f.match_id, "not covered") for f in fixtures}


class SlowAnalyst:
    """Sleeps per call (the earliest games are the slowest, so completion order is the REVERSE of launch order), records the
    peak number of calls in flight and the order the calls started in."""

    def __init__(self, delay=0.02, fail_on=None, flags=(), reverse=True):
        self.delay, self.fail_on, self.flags, self.reverse = delay, fail_on, flags, reverse
        self.started, self.now, self.peak = [], 0, 0
        self.finished = []

    async def research(self, query):
        index = len(self.started)
        self.started.append(query.home)
        self.now += 1
        self.peak = max(self.peak, self.now)
        try:
            await asyncio.sleep(self.delay * (3 - index % 3) if self.reverse else self.delay)
            if self.fail_on is not None and index == self.fail_on:
                raise AnalystUnavailable("the service is down")
            stats = [{"name": n, "value": GOOD[n.split(".")[1]], "unit": SOCCER_REGISTRY[n.split(".")[1]].unit,
                      "citation": "example.com/table"} for n in query.wanted]
            flags = [{"flag": flag, "source": "example.com/news"} for flag in self.flags]
            self.finished.append(query.home)
            return parse_report(json.dumps({"stats": stats, "flags": flags}), query, AnalystUsage(1000, 200, 2))
        finally:
            self.now -= 1


def fill(fixtures, analyst, limit=10, budget=None, concurrency=ANALYST_CONCURRENCY):
    stats, ar = lacking(fixtures), AnalystRun(analyst, limit=limit, budget=budget, cache=None)
    result = run(fill_missing_inputs(fixtures, stats, ar, concurrency))
    return result, ar


def test_the_default_concurrency_is_three_and_a_named_constant():
    assert ANALYST_CONCURRENCY == 3


def test_calls_run_a_few_at_a_time_never_more_than_the_bound():
    analyst = SlowAnalyst()
    (stats, flags, skipped), ar = fill(games(9), analyst)
    assert analyst.peak == 3 and ar.calls == 9 and skipped == 0
    assert all(s.status != INSUFFICIENT for s in stats.values())
    analyst = SlowAnalyst()
    fill(games(5), analyst, concurrency=1)
    assert analyst.peak == 1


def test_concurrent_is_faster_than_one_at_a_time():
    import time
    one, three = SlowAnalyst(0.03, reverse=False), SlowAnalyst(0.03, reverse=False)
    started = time.perf_counter()
    fill(games(6), one, concurrency=1)
    sequential = time.perf_counter() - started
    started = time.perf_counter()
    fill(games(6), three)
    assert time.perf_counter() - started < sequential * 0.7 and sequential >= 0.18


def test_calls_start_in_earliest_kickoff_order_even_when_the_list_is_not_sorted():
    fixtures = list(reversed(games(7)))                       # the latest kickoff comes first in the list
    analyst = SlowAnalyst()
    fill(fixtures, analyst)
    assert analyst.started == [f"H{i}" for i in range(7)]


def test_the_per_request_cap_bites_the_latest_games_and_is_never_exceeded():
    analyst = SlowAnalyst()
    (stats, flags, skipped), ar = fill(list(reversed(games(8))), analyst, limit=4)
    assert ar.calls == 4 and len(analyst.started) == 4 and skipped == 4 and ar.limit_hit
    assert sorted(analyst.started) == ["H0", "H1", "H2", "H3"]                      # the four earliest kickoffs
    for i in range(4):
        assert stats[games(8)[i].match_id].status != INSUFFICIENT
    for i in range(4, 8):
        assert any("call limit (4) was reached" in r for r in stats[games(8)[i].match_id].reasons)


def test_the_daily_limit_is_counted_atomically_under_concurrency(tmp_path):
    from predictions.flow.store import EvaluationStore
    with EvaluationStore.open(tmp_path / "d.db", str) as store:
        analyst = SlowAnalyst()
        budget = DailyBudget(store, 5, NOW)
        (stats, flags, skipped), ar = fill(games(12), analyst, limit=10, budget=budget)
        assert len(analyst.started) == 5 and ar.calls == 5 and DailyBudget(store, 5, NOW).used() == 5
        assert ar.daily_hit and skipped == 7
        assert sorted(analyst.started) == [f"H{i}" for i in range(5)]
        assert all(any("daily limit (5 calls)" in r for r in stats[g.match_id].reasons) for g in games(12)[5:])


def test_results_do_not_depend_on_completion_order():
    fixtures = games(6)
    reverse, forward = SlowAnalyst(reverse=True, flags=("key_player_out",)), SlowAnalyst(reverse=False, flags=("key_player_out",))
    (stats_a, flags_a, skipped_a), _ = fill(fixtures, reverse)
    (stats_b, flags_b, skipped_b), _ = fill(fixtures, forward)
    assert reverse.finished != reverse.started   # the calls really did finish in a different order than they started
    assert stats_a == stats_b and flags_a == flags_b and skipped_a == skipped_b
    assert set(flags_a) == {f.match_id for f in fixtures} and all(
        v[0].flag is ContextFlag.KEY_PLAYER_OUT for v in flags_a.values())


def test_after_an_outage_no_new_call_is_launched_and_the_rest_are_marked_unavailable():
    analyst = SlowAnalyst(fail_on=0, reverse=False)
    (stats, flags, skipped), ar = fill(games(9), analyst)
    assert ar.unavailable and len(analyst.started) == 3          # only the first batch was in flight; nothing launched after
    assert all(any("unavailable" in r for r in s.reasons) for s in stats.values() if s.status == INSUFFICIENT)
    assert len([s for s in stats.values() if s.status == INSUFFICIENT]) >= 6 and skipped == 0


def test_an_unexpected_error_cancels_the_other_calls_and_is_raised():
    class Boom(SlowAnalyst):
        async def research(self, query):
            if query.home == "H1":
                raise RuntimeError("bug")
            return await super().research(query)

    with pytest.raises(RuntimeError):
        fill(games(5), Boom())


def test_check_flags_is_concurrent_ordered_capped_and_skips_games_already_asked():
    fixtures = list(reversed(games(8)))
    analyst = SlowAnalyst(flags=("lineup_uncertain",))
    ar = AnalystRun(analyst, limit=5, cache=None)
    ar.asked.add(games(8)[0].match_id)                           # H0 was asked in this run already
    ids = {f.match_id for f in fixtures}
    found = run(check_flags(fixtures, ids, ar))
    assert analyst.peak == 3 and ar.calls == 5 and sorted(analyst.started) == ["H1", "H2", "H3", "H4", "H5"]
    assert set(found) == {games(8)[i].match_id for i in range(1, 6)}


def test_check_flags_stops_launching_after_an_outage():
    analyst = SlowAnalyst(fail_on=0, reverse=False, flags=("lineup_uncertain",))
    ar = AnalystRun(analyst, cache=None)
    found = run(check_flags(games(9), {f.match_id for f in games(9)}, ar))
    assert ar.unavailable and len(analyst.started) == 3
    assert len(found) == 2                                       # the two calls already in flight still answered


def test_gather_in_order_returns_results_in_job_order_with_a_bounded_number_in_flight():
    peak = {"now": 0, "max": 0}

    def job(i):
        async def work():
            peak["now"] += 1
            peak["max"] = max(peak["max"], peak["now"])
            await asyncio.sleep(0.01 * (5 - i))
            peak["now"] -= 1
            return i
        return work

    assert run(gather_in_order([job(i) for i in range(5)], 2)) == [0, 1, 2, 3, 4] and peak["max"] == 2


def test_no_secret_reaches_the_analyst_only_identifiers_do():
    seen = []

    class Spy(SlowAnalyst):
        async def research(self, query):
            seen.append(query)
            return await super().research(query)

    fill(games(3), Spy(reverse=False))
    assert all(set(vars(q)) == {"home", "away", "league", "kickoff_date", "wanted", "sport"} for q in seen)
