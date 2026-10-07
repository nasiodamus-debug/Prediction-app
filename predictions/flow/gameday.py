"""Run a Gameday request (basketball, Story 6.1): the same slate runner as Matchday with the basketball profile."""
from __future__ import annotations

from datetime import datetime
from typing import Callable

from predictions.core.ports import Analyst, FixtureSource, StatsSource
from predictions.flow.analyst import DailyBudget
from predictions.flow.evaluate import GateResult
from predictions.flow.slate_run import GAMEDAY, run_slate


async def run_gameday(argument: str, source: FixtureSource, sanitize: Callable[[object], str],
                      now: datetime | None = None, stats_source: StatsSource | None = None,
                      on_result: Callable[[GateResult], None] | None = None,
                      analyst: Analyst | None = None, analyst_off_note: str = "",
                      budget: DailyBudget | None = None, day_cache=None) -> list[str]:
    """The message chunks for `Gameday <dates>`. `source` lists basketball games (SportyBet, sr:sport:2) and
    `stats_source` gives their team stats with provenance. Games that started, and games with no data, are handled
    exactly as for Matchday; when no pick passes the reply is NO VALID PLAY - PASS."""
    return await run_slate(GAMEDAY, argument, source, sanitize, now, stats_source, None, on_result, analyst,
                           analyst_off_note, budget, day_cache=day_cache)
