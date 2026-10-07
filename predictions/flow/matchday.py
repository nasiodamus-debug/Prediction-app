"""Run a Matchday request (soccer). The work is done by the sport-parameterised slate runner (flow/slate_run.py);
this module keeps the Matchday entry point and its wording."""
from __future__ import annotations

from datetime import datetime
from typing import Callable

from predictions.core.ports import Analyst, EventMarketsSource, FixtureSource, InputsSource, StatsSource
from predictions.flow.analyst import DailyBudget
from predictions.flow.evaluate import GateResult
from predictions.flow.slate_run import MATCHDAY, MAX_PROBLEMS_SHOWN, SportProfile, ask_for_date, run_slate  # noqa: F401

ASK_FOR_DATE = ask_for_date(MATCHDAY)


async def run_matchday(argument: str, source: FixtureSource, sanitize: Callable[[object], str],
                       now: datetime | None = None, stats_source: StatsSource | None = None,
                       extra_inputs_source: InputsSource | None = None,
                       on_result: Callable[[GateResult], None] | None = None,
                       analyst: Analyst | None = None, analyst_off_note: str = "",
                       budget: DailyBudget | None = None,
                       event_markets_source: EventMarketsSource | None = None, day_cache=None) -> list[str]:
    """Return the message chunks to send. `argument` is the request text after the word Matchday.
    When the games were evaluated, `on_result` is called with the GateResult so the caller can save the
    evaluations and start the final-list review (Stories 2.6 and 2.7). `analyst` (Epic 3) fills inputs the free
    sources lack and may flag a fixture; `analyst_off_note` is the one line shown when it is switched off.
    `extra_inputs_source` gives corner, card and referee inputs (football-data.co.uk) and `event_markets_source` looks up, per
    game, whether SportyBet lists corners and cards markets (spec 11-1)."""
    return await run_slate(MATCHDAY, argument, source, sanitize, now, stats_source, extra_inputs_source, on_result,
                           analyst, analyst_off_note, budget, event_markets_source, day_cache)
