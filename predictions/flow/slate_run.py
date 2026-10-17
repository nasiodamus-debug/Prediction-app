"""The sport-parameterised slate runner: read dates, fetch fixtures, select, gather stats, evaluate, render.

`Matchday` (soccer), `Gameday` (basketball) and `America` are the same run with a different `SportProfile`. A sport is added by
adding a profile here, an engine in `core/engines/`, a fixture listing in `adapters/sportybet/fixtures.py` and a
stats source (see the progress log). Nothing else in the run is sport-specific.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Callable

from predictions.core.availability import is_open_for_betting, not_open_line
from predictions.core.competitions import REASON_OUTSIDE, is_allowed
from predictions.core.dates import lagos_today, parse_dates
from predictions.core.ports import (Analyst, EventMarketsSource, FixtureFetchError, FixtureSource, InputsSource,
                                    StatsFetchError, StatsSource)
from predictions.core.schedule import select_fixtures
from predictions.core.stats import INSUFFICIENT
from predictions.core.ttl_cache import STATS_CACHE
from predictions.core.types import Fixture
from predictions.flow.daycache import odds_as_of_line
from predictions.flow.analyst import AnalystRun, DailyBudget, check_flags, fill_missing_inputs
from predictions.flow.freeflags import disagreement_lines, free_flags, merge_flags
from predictions.flow.evaluate import GateResult, run_gate
from predictions.flow.stats import add_extra_inputs, gather_stats
from predictions.telegram.render import basketball_main_odds, main_odds, multi_sport_main_odds, render_matchday

MAX_PROBLEMS_SHOWN = 5
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SportProfile:
    trigger: str                            # the word the owner sends: "Matchday"
    sport: str                              # the canonical sport name engines are registered under
    example_dates: str                      # shown when a date is missing or unreadable
    odds_summary: Callable[[Fixture], str]  # the one line of main odds shown under each listed game
    restrict: bool = False                  # Story 10.1: only the competitions in core/competitions.py are looked at
    analyst_fills_stats: bool = True        # False: the AI analyst is never asked for missing stats in this list (Gameday: the
                                            # required ortg, drtg and pace are not in search snippets, so every lookup came back empty)
    gate: Callable | None = None            # (fixtures, stats, flags) -> GateResult; None = the standard run_gate (Mega supplies its own)


MATCHDAY = SportProfile("Matchday", "soccer", "20 and 22 September", main_odds, restrict=True)
GAMEDAY = SportProfile("Gameday", "basketball", "20 and 22 October", basketball_main_odds, restrict=True,
                       analyst_fills_stats=False)
# `America` is one request over several sports (MLB, NHL and American football): its fixtures come from a
# MultiSportFixtures, its stats from a MultiSportStats (flow/america.py), and each game's odds line from its own sport.
AMERICA = SportProfile("America", "baseball, hockey, american_football", "20 and 22 October", multi_sport_main_odds,
                       restrict=True)
# `Mega` (flow/mega.py): football only, every competition, one pick per game on a 60% floor; its gate is supplied by run_mega.
MEGA = SportProfile("Mega", "soccer", "17 and 18 October", main_odds, restrict=False, analyst_fills_stats=False)
PROFILES = {"matchday": MATCHDAY, "gameday": GAMEDAY, "america": AMERICA, "mega": MEGA}  # by slate origin (see telegram/slate.py)


def ask_for_date(profile: SportProfile) -> str:
    return (f"Which date? Send the date after {profile.trigger}, for example: {profile.trigger} "
            f"{profile.example_dates}, {profile.trigger} today or {profile.trigger} tomorrow. "
            "I never pick a day for you.")


def ignored_line(count: int) -> str:
    return f"{count} game(s) ignored: {REASON_OUTSIDE}."


def _footer(profile: SportProfile, ignored: int, stats, run: AnalystRun | None, lookup_failed: int = 0,
            unpriced: int = 0, not_open: int = 0, odds_read_at=None) -> tuple[str, ...]:
    """The short lines that end a passing list: how many games were ignored, what the analyst cost, and a warning when a
    stats service could not be read (Story 10.1). The last two are the safety lines."""
    lines = [ignored_line(ignored)] if ignored else []
    if not_open:   # spec 12-1: listed by SportyBet but not open for betting
        lines.append(not_open_line(not_open))
    if run is not None and (run.calls or run.cache_hits):
        lines.append(run.short_summary())
    if odds_read_at is not None:   # spec 14-1: the list was built from the daily games cache, so say when its odds were read
        lines.append(odds_as_of_line(odds_read_at))
    if stats and any(e.status == INSUFFICIENT and any("could not be read" in r for r in e.reasons) for e in stats.values()):
        lines.append(f"Warning: the {profile.sport} stats service could not be read: games from it were not evaluated.")
    if lookup_failed:   # spec 11-1: the per-game corners/cards lookup failed for some games
        lines.append(f"Warning: SportyBet's per-game markets could not be read for {lookup_failed} game(s): corners and "
                     "cards were not considered for them.")
    if unpriced:
        lines.append(f"Corners/cards not available for {unpriced} game(s): listed by SportyBet but no corner or card data.")
    return tuple(lines)


async def _with_event_markets(fixtures: list[Fixture], stats, source: EventMarketsSource) -> tuple[list[Fixture], int]:
    """Spec 11-1, the per-game market check: for every READY game look up its corners and cards markets once and add them to
    the game's own markets. Returns the fixtures (same MatchIds) and how many lookups failed (those games keep the markets
    they had). Cached nowhere: a new request looks again."""
    ready = [f for f in fixtures if f.match_id in stats and stats[f.match_id].ready]
    if not ready:
        return fixtures, 0
    found = await source.fetch_event_markets(ready)
    failed = sum(1 for f in ready if found.get(f.match_id) is None)
    merged = []
    for fixture in fixtures:
        extra = found.get(fixture.match_id)
        if extra:
            have = {(m.market_id, m.specifier) for m in fixture.markets}
            fixture = replace(fixture, markets=(*fixture.markets, *(m for m in extra if (m.market_id, m.specifier) not in have)))
        merged.append(fixture)
    return merged, failed


def _join(note: str, more: str) -> str:
    return f"{note}\n{more}" if note else more


async def run_slate(profile: SportProfile, argument: str, source: FixtureSource, sanitize: Callable[[object], str],
                    now: datetime | None = None, stats_source: StatsSource | None = None,
                    extra_inputs_source: InputsSource | None = None,
                    on_result: Callable[[GateResult], None] | None = None,
                    analyst: Analyst | None = None, analyst_off_note: str = "",
                    budget: DailyBudget | None = None,
                    event_markets_source: EventMarketsSource | None = None, day_cache=None) -> list[str]:
    """Return the message chunks to send. `argument` is the request text after the trigger word.
    When the games were evaluated, `on_result` is called with the GateResult so the caller can save the
    evaluations and start the final-list review (Stories 2.6 and 2.7). `analyst` (Epic 3) fills inputs the free
    sources lack and may flag a fixture; `analyst_off_note` is the one line shown when it is switched off. `budget` is the
    daily analyst allowance (flow/analyst.py). A restricted profile (Matchday, Gameday, America) drops games outside its
    allowed competitions, and every profile drops games that are not open for betting on SportyBet, BEFORE any stats or
    analyst work, so they cost nothing. `day_cache` (flow/daycache.py) lets a restricted profile read the allowed games of
    today and tomorrow from memory instead of the live listing; the same checks (not started, open for betting) are
    applied to them here, and the list then says when their odds were read."""
    now = now or datetime.now(timezone.utc)
    trigger = profile.trigger
    gate_of = profile.gate or run_gate

    ignored = not_open = 0

    def render(*args, **kwargs):
        return render_matchday(*args, trigger=trigger, odds=profile.odds_summary,
                               analyst_off=analyst is None and bool(analyst_off_note), ignored=ignored,
                               not_open=not_open, **kwargs)

    parsed = parse_dates(argument, now)
    if parsed.problems:
        shown = parsed.problems[:MAX_PROBLEMS_SHOWN]
        more = len(parsed.problems) - len(shown)
        extra = f" (and {more} more)" if more > 0 else ""
        return ["I could not read that date: " + "; ".join(shown) + extra + ". "
                f"Please send a valid date, for example: {trigger} {profile.example_dates} (or today / tomorrow)."]
    if not parsed.dates:
        return [ask_for_date(profile)]

    today = lagos_today(now)
    if all(d < today for d in parsed.dates):
        return render(select_fixtures([], parsed.dates, now), parsed.dates)

    cached = None
    if day_cache is not None and profile.restrict:
        cached = await day_cache.read(trigger, parsed.dates, now)
    odds_read_at = cached.snapshot_at if cached is not None else None
    if cached is not None:
        fixtures = cached.fixtures
    else:
        try:
            fixtures = await source.fetch_fixtures()
        except FixtureFetchError as error:  # no retry, no partial list
            return [f"Could not get fixtures from SportyBet: {sanitize(error)}. Nothing was listed. "
                    "Please try again later."]
    # Any other exception is a bug, not an outage: it goes to the bot's error handler, which logs it
    # and alerts the owner, instead of being reported as a SportyBet problem.
    selection = select_fixtures(fixtures, parsed.dates, now)
    if profile.restrict:
        kept = [f for f in selection.fixtures if is_allowed(trigger, f.match_id.sport, f.competition)[0]]
        ignored = len(selection.fixtures) - len(kept) + (cached.ignored if cached is not None else 0)
        selection.fixtures = kept
    # Spec 12-1: only games open for betting on SportyBet right now are analysed (no stats or analyst call otherwise).
    open_games = [f for f in selection.fixtures if is_open_for_betting(f, now)]
    not_open = len(selection.fixtures) - len(open_games) + (cached.not_open if cached is not None else 0)
    selection.fixtures = open_games
    stats, stats_note, gate, run, lookup_failed = None, "", None, None, 0
    skipped = 0
    cache_before = STATS_CACHE.counts()
    if stats_source is not None and selection.fixtures:
        try:
            stats = await gather_stats(selection.fixtures, stats_source)
            if extra_inputs_source is not None:
                try:
                    stats = await add_extra_inputs(selection.fixtures, stats, extra_inputs_source)
                except StatsFetchError as error:  # optional data: corners and cards picks become "insufficient data"
                    stats_note = (f"Could not get corner and card data: {sanitize(error)}. "
                                  "Corners and cards plays are not considered.")
            flags, run = {}, None
            if analyst is not None:
                run = AnalystRun(analyst, budget=budget)
                if profile.analyst_fills_stats:
                    stats, flags, skipped = await fill_missing_inputs(selection.fixtures, stats, run)
            elif analyst_off_note and any(s.status == INSUFFICIENT for s in stats.values()):
                stats_note = _join(stats_note, analyst_off_note)
            if event_markets_source is not None:   # Matchday only: which games list corners and cards (spec 11-1)
                selection.fixtures, lookup_failed = await _with_event_markets(selection.fixtures, stats,
                                                                              event_markets_source)
            flags = merge_flags(flags, await free_flags(selection.fixtures))   # API-Football injuries and lineups (no AI call)
            gate = gate_of(selection.fixtures, stats, flags)
            if run is not None:
                # Flags matter most for picks that could be bet: check those with what is left of the call cap.
                more = await check_flags(selection.fixtures, {e.match_id for e in gate.passed}, run)
                if more:
                    flags = merge_flags(flags, more)
                    gate = gate_of(selection.fixtures, stats, flags)
                if run.calls or run.daily_hit or run.cache_hits:
                    stats_note = _join(stats_note, run.summary(skipped))
        except StatsFetchError as error:  # no retry; the fixtures are still listed, none is evaluable
            stats_note = f"Could not get team stats: {sanitize(error)}. No game has data, so none can be evaluated."
    if stats_source is not None and selection.fixtures:
        hits, misses = (after - before for after, before in zip(STATS_CACHE.counts(), cache_before))
        log.info("%s request: stats cache %d hit(s), %d miss(es); games from %s", trigger, hits, misses,
                 "the daily games cache" if cached is not None else "the live listing")
    if gate is not None and profile.restrict:
        gate = replace(gate, footer=_footer(profile, ignored, stats, run, lookup_failed, len(gate.unpriced_games), not_open,
                                            odds_read_at))
    if gate is not None and gate.passed:   # the vendor's own model disagreeing sharply with a passing pick: a warning only
        warnings = await disagreement_lines(gate.passed, gate.fixtures)
        if warnings:
            gate = replace(gate, footer=gate.footer + warnings)
    if gate is not None and on_result is not None:
        on_result(gate)
    return render(selection, parsed.dates, stats, stats_note, gate)
