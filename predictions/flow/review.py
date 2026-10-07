"""Run a Review request: read the code, drop games already started, judge each selection by its own sport's
rules, render (Epic 4). The flow owns the order of work; the rules live in the engines (core) and everything about
SportyBet's endpoint lives in the CodeReader adapter (AD-11). Review only READS a code: it never touches a
betslip and never places anything (AD-17). Review is NOT restricted by the competition limits (core/competitions.py) and
does not run the "open for betting" pre-check: every selection on the code is judged on its own odds (spec-12-1)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Mapping

from predictions.core.analyst import RULE_ANALYST_FLAG, describe_flags
from predictions.core.engines import ENGINES
from predictions.core.market_rules import is_under, is_whole_or_quarter_line
from predictions.core.model import PROBABILITY_FLOOR, Evaluation, FixturePass
from predictions.core.ports import (Analyst, CodeReader, CodeReadError, InputsSource, StatsFetchError, StatsSource)
from predictions.core.review import (CodeSelection, Judgement, ReviewResult, judge_selection,
                                     valid_code_format, INSUFFICIENT_DATA)
from predictions.core.stats import FixtureStats
from predictions.core.types import Fixture, MatchId
from predictions.flow.analyst import AnalystRun, DailyBudget, check_flags, fill_missing_inputs
from predictions.core.dates import LAGOS
from predictions.flow.freeflags import free_flags, merge_flags
from predictions.flow.reviewcache import REVIEW_CACHE
from predictions.flow.stats import add_extra_inputs, gather_stats
from predictions.telegram.render import render_review

ASK_FOR_CODE = "Which booking code? Send it after Review, for example: Review 23B45S."
BAD_CODE = ("That does not look like a booking code. A code is letters and numbers only, "
            "for example: Review 23B45S.")


def _fixtures(selections: list[CodeSelection]) -> dict[MatchId, Fixture]:
    """One Fixture per game on the code, built the way the SportyBet listing builds them (same competition
    text, same MatchId), so the stats sources match teams with the existing identity resolver."""
    found: dict[MatchId, Fixture] = {}
    for s in selections:
        found.setdefault(s.match_id, Fixture(s.match_id, s.competition, s.home, s.away, s.kickoff_utc, False))
    return found


def _join(note: str, more: str) -> str:
    return f"{note}\n{more}" if note else more


def _judge_all(open_selections: list[CodeSelection], fixtures: dict[MatchId, Fixture],
               stats: dict[MatchId, FixtureStats], flags: dict) -> tuple[Judgement, ...]:
    judged = []
    for s in open_selections:
        if is_under(s):   # AD-20: no Under is ever judged, so it costs no stats and no analyst call
            judged.append(Judgement(s, judge_selection(s, None, None)))
            continue
        flag = None
        if flags.get(s.match_id):
            flag = FixturePass(s.match_id, RULE_ANALYST_FLAG, describe_flags(flags[s.match_id]))
        judged.append(Judgement(s, judge_selection(s, fixtures[s.match_id], stats.get(s.match_id), flag)))
    return tuple(judged)


async def run_review(argument: str, reader: CodeReader, sanitize: Callable[[object], str],
                     stats_sources: Mapping[str, StatsSource] | None = None,
                     extra_inputs_sources: Mapping[str, InputsSource] | None = None,
                     now: datetime | None = None,
                     on_result: Callable[[ReviewResult], None] | None = None,
                     analyst: Analyst | None = None, analyst_off_note: str = "",
                     budget: DailyBudget | None = None) -> list[str]:
    """The message chunks to send. `argument` is the text after the word Review. `stats_sources` maps a
    canonical sport to its StatsSource: a sport with no engine in `ENGINES` is disqualified by default without
    any lookup, so Epics 6 and 7 add a sport by registering an engine and a source, and nothing here changes.
    `on_result` receives the ReviewResult so the caller can save the evaluations (Story 2.7's store)."""
    now = now or datetime.now(timezone.utc)
    parts = argument.split()
    if not parts:
        return [ASK_FOR_CODE]
    if len(parts) > 1 or not valid_code_format(parts[0]):
        return [BAD_CODE]
    code = parts[0]

    hit = REVIEW_CACHE.get(code, now)
    if hit is not None:   # the same code asked again the same day: no new reads, stats or analyst calls
        made, cached, cached_notes = hit
        live = tuple(j for j in cached.judgements if not j.selection.is_started(now))
        if live:
            gone = cached.started + tuple(j.selection for j in cached.judgements if j.selection.is_started(now))
            result = ReviewResult(code, live, gone)
            if on_result is not None:
                on_result(result)
            note = (f"Reused the review made at {made.astimezone(LAGOS).strftime('%H:%M')} Lagos time (kept for up to "
                    f"{REVIEW_CACHE.hours:g} hour(s) on the same day); odds and stats are as they were then and games that "
                    "have started were removed.")
            return render_review(result, _join(cached_notes, note))

    try:
        selections = await reader.read_code(code)  # no retry, no partial list
    except CodeReadError as error:
        return [sanitize(error)]
    # Any other exception is a bug, not a bad code: it goes to the bot's error handler.

    started = tuple(s for s in selections if s.is_started(now))
    open_selections = [s for s in selections if not s.is_started(now)]
    # An Under selection is disqualified outright (AD-20): it gets no fixture, no stats and no analyst work.
    fixtures = _fixtures([s for s in open_selections if not is_under(s)])
    stats: dict[MatchId, FixtureStats] = {}
    flags: dict = {}
    notes = ""
    run = None

    # Only sports that have an engine are looked up; the others are disqualified by default in judge_selection.
    for sport in sorted({s.sport for s in open_selections if s.sport in ENGINES and not is_under(s)}):
        source = (stats_sources or {}).get(sport)
        games = [f for mid, f in fixtures.items() if mid.sport == sport]
        if source is None or not games:
            continue  # no data source for a sport that has rules: its games become "insufficient data"
        try:
            found = await gather_stats(games, source)
            extra = (extra_inputs_sources or {}).get(sport)
            if extra is not None:
                try:
                    found = await add_extra_inputs(games, found, extra)
                except StatsFetchError as error:  # optional data: corners and cards become "insufficient data"
                    notes = _join(notes, f"Could not get extra {sport} data: {sanitize(error)}.")
            if analyst is not None:  # the analyst knows every sport that has an input registry (core/inputs.py)
                run = run or AnalystRun(analyst, budget=budget)
                if sport != "basketball":   # ortg, drtg and pace are not in search snippets: every basketball lookup came back empty
                    found, more_flags, skipped = await fill_missing_inputs(games, found, run, goals_only_fallback=True)
                    flags.update(more_flags)
            elif analyst_off_note and any(not s.ready for s in found.values()):
                notes = _join(notes, analyst_off_note)
            stats.update(found)
        except StatsFetchError as error:  # no retry: those games are "insufficient data", never guessed
            notes = _join(notes, f"Could not get {sport} stats: {sanitize(error)}. Those selections are "
                                 f"disqualified as {INSUFFICIENT_DATA}.")
    flags = merge_flags(flags, await free_flags(list(fixtures.values())))   # API-Football injuries and lineups (no AI call)
    judgements = _judge_all(open_selections, fixtures, stats, flags)

    if run is not None:
        # Analyst flags matter most for selections that could be approved: check those with the calls left.
        passing = {j.selection.match_id for j in judgements if j.approved}
        more = await check_flags([fixtures[m] for m in passing], passing, run)
        if more:
            flags = merge_flags(flags, more)
            judgements = _judge_all(open_selections, fixtures, stats, flags)
        if run.calls or run.daily_hit or run.cache_hits:
            notes = _join(notes, run.summary())

    judgements = tuple(sorted(judgements, key=lambda j: j.selection.kickoff_utc))   # earliest kickoff first
    result = ReviewResult(code, judgements, started)
    if on_result is not None and result.judgements:
        on_result(result)
    if "Could not get" not in notes and "became unavailable" not in notes:   # a review made during an outage is not kept
        REVIEW_CACHE.put(code, now, result, notes)
    return render_review(result, notes)


def approved_evaluations(result: ReviewResult) -> tuple[Evaluation, ...]:
    """The picks a Review may hand on to a bet (Story 5.6): ONLY approved selections, with everything the later steps
    need already on them (market, line, the odds on the code, the estimate and its inputs). A disqualified selection
    can never be returned: it must be approved, passed, priced (not "insufficient") and clear the floor. Review itself
    never touches the betslip; the normal flow loads these fresh (AD-17). A whole-line or quarter-line selection is judged on the review but is never handed on (AD-21). The one-pick-per-match rule is applied by the
    Slip, so two approved selections on one game make the owner choose one."""
    kept: list[Evaluation] = []
    for judgement in result.judgements:
        ev = judgement.evaluation
        if (judgement.approved and ev.passed and not ev.insufficient
                and ev.estimate.probability >= PROBABILITY_FLOOR and ev not in kept
                and not is_whole_or_quarter_line(ev)):   # AD-21: Review still judges a whole line, but never hands it on to a bet
            kept.append(ev)
    return tuple(kept)
