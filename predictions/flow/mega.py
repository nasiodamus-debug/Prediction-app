"""Mega (soccer only): one combined list across the requested dates, ONE pick per game, judged on a lower floor.

The owner's rules (2026-10-17):
  * every football competition is looked at (no allow-list), and every game that has some or all of the data the model needs;
  * a pick qualifies at a probability of at least MEGA_FLOOR (60%), with odds of at least MEGA_MIN_ODDS (1.25). There is NO
    positive-EV test: the bookmaker margin makes almost every pick slightly negative, so EV is shown on each pick for
    information only (the estimate is 75% bookmaker price, so EV is about -margin whatever the pick);
  * per game the qualifying pick with the highest probability is kept (tie: higher odds); a pick under 1.25 is discarded and the
    next highest probability still at 60% or more is used instead;
  * all kept picks go in the list, best first, up to MEGA_MAX_PICKS (config.env MEGA_MAX_PICKS, default 20), then the owner reviews
    the list and confirms before any booking code is made (the same interactive list as Matchday).
The owner's standing rules still apply in the gate: no Under (AD-20), no whole or quarter line (AD-21), derbies and cup ties
passed, the goals-only "too good to be true" cap, and an analyst flag. Games that have kicked off are never listed.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from decimal import Decimal
from functools import partial
from typing import Callable

from predictions.core.model import REASON_PASSED, Evaluation, is_floor_or_ev_reason, percent
from predictions.core.ports import Analyst, EventMarketsSource, FixtureSource, InputsSource, StatsSource
from predictions.flow.analyst import DailyBudget
from predictions.flow.evaluate import GateResult, run_gate
from predictions.flow.slate_run import MEGA, run_slate

MEGA_FLOOR = 0.60
MEGA_MIN_ODDS = Decimal("1.25")
DEFAULT_MEGA_MAX_PICKS = 20
MAX_MEGA_MAX_PICKS = 200
NOT_CHOSEN_REASON = "Mega keeps one pick per game and a better one was chosen"
OVER_LIMIT_REASON = "over the Mega limit of {n} picks (the highest probabilities are kept)"


def parse_mega_max_picks(raw) -> int:
    """The MEGA_MAX_PICKS setting, read once at startup. Unset or empty = 20. Otherwise a whole number from 1 to 200; anything
    else raises ValueError (never echoed)."""
    if raw is None or not str(raw).strip():
        return DEFAULT_MEGA_MAX_PICKS
    text = str(raw).strip()
    if not text.isascii() or not text.isdigit() or len(text) > 3 or not 1 <= int(text) <= MAX_MEGA_MAX_PICKS:
        raise ValueError(f"MEGA_MAX_PICKS in config.env is not a whole number from 1 to {MAX_MEGA_MAX_PICKS} (for example 20). "
                         f"Fix it or remove the line to use {DEFAULT_MEGA_MAX_PICKS}, then start the bot again.")
    return int(text)


def _qualifies(ev: Evaluation) -> bool:
    """A priced pick Mega may consider: the engine's own gate passed it, or the ONLY things against it were the 70% floor and
    the EV test (not a rule's refusal such as the goals-only cap), and it clears Mega's floor and minimum odds."""
    if ev.insufficient or not (ev.passed or is_floor_or_ev_reason(ev.reason)):
        return False
    return ev.estimate.probability >= MEGA_FLOOR and ev.odds >= MEGA_MIN_ODDS


def _mega_reason(ev: Evaluation) -> str:
    """Why a priced pick does not qualify for Mega, in Mega's own terms (the engine's wording quotes the 70% floor and EV)."""
    if ev.estimate.probability < MEGA_FLOOR:
        return f"estimate {percent(ev.estimate.probability)} is below the {MEGA_FLOOR:.0%} floor"
    return f"odds {ev.odds:.2f} are below the minimum of {MEGA_MIN_ODDS:.2f}"


def mega_gate(fixtures, stats, flags=None, max_picks: int = DEFAULT_MEGA_MAX_PICKS) -> GateResult:
    """Run the soccer gate with every play kept, then choose ONE pick per game by Mega's rules and cap the list."""
    base = run_gate(fixtures, stats, flags, per_game_limit=False)
    pool = [*base.passed, *base.rejected]
    best: dict = {}
    for ev in pool:
        if _qualifies(ev):
            key = (-ev.estimate.probability, -ev.odds, ev.selection)
            if ev.match_id not in best or key < best[ev.match_id][0]:
                best[ev.match_id] = (key, ev)
    ranked = sorted(best.values(), key=lambda item: (item[0][0], str(item[1].match_id), item[0][1], item[0][2]))
    chosen_pairs = ranked[:max_picks]
    chosen = {id(ev) for _, ev in chosen_pairs}
    over_limit = {id(ev) for _, ev in ranked[max_picks:]}
    passed = [replace(ev, passed=True, reason=REASON_PASSED) for _, ev in chosen_pairs]
    rejected = []
    for ev in pool:
        if id(ev) in chosen:
            continue
        if id(ev) in over_limit:
            rejected.append(replace(ev, passed=False, reason=OVER_LIMIT_REASON.format(n=max_picks)))
        elif _qualifies(ev):
            rejected.append(replace(ev, passed=False, reason=NOT_CHOSEN_REASON))
        elif not ev.insufficient and is_floor_or_ev_reason(ev.reason):
            rejected.append(replace(ev, passed=False, reason=_mega_reason(ev)))
        else:
            rejected.append(replace(ev, passed=False))
    rejected.sort(key=lambda e: (-e.estimate.probability, str(e.match_id), e.selection))
    footer = ()
    if over_limit:
        footer = (f"{len(over_limit)} more qualifying game(s) are not listed: the limit is {max_picks} picks "
                  "(MEGA_MAX_PICKS in config.env).",)
    return replace(base, passed=tuple(passed), rejected=tuple(rejected), floor=MEGA_FLOOR, ev_required=False, footer=footer)


async def run_mega(argument: str, source: FixtureSource, sanitize: Callable[[object], str],
                   now: datetime | None = None, stats_source: StatsSource | None = None,
                   extra_inputs_source: InputsSource | None = None,
                   on_result: Callable[[GateResult], None] | None = None,
                   event_markets_source: EventMarketsSource | None = None,
                   max_picks: int = DEFAULT_MEGA_MAX_PICKS, analyst: Analyst | None = None,
                   budget: DailyBudget | None = None) -> list[str]:
    """The message chunks for `Mega <dates>`. When at least one game qualifies the reply is empty and `on_result` receives the
    GateResult (its `passed` is the one-pick-per-game list); otherwise the reply is NO VALID PLAY and the reasons. The AI analyst
    is off unless one is passed (Mega may look at hundreds of games, so the paid lookups are not made by default)."""
    profile = replace(MEGA, gate=partial(mega_gate, max_picks=max_picks))
    return await run_slate(profile, argument, source, sanitize, now, stats_source, extra_inputs_source, on_result, analyst, "",
                           budget, event_markets_source)
