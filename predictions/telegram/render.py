"""Turn fixtures into plain-text Telegram messages. Nothing here uses Markdown or HTML parsing."""
from __future__ import annotations

import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation
from itertools import groupby

from predictions.core.availability import not_open_line
from predictions.core.dates import LAGOS
from predictions.core.engines.soccer import GOALS_ONLY_LABEL, GOALS_ONLY_MODEL_NAME
from predictions.core.inputs import ANALYST_PREFIX
from predictions.core.market_rules import UNDER_REASON, WHOLE_LINE_REASON, is_under, is_whole_or_quarter_line
from predictions.core.model import PROBABILITY_FLOOR, Evaluation, percent, signed_percent
from predictions.core.play_labels import play_label
from predictions.core.schedule import Selection
from predictions.core.stats import (AMBIGUOUS, COMPETITION_LEVEL, INSUFFICIENT, READY, FixtureStats,
                                     skip_category)
from predictions.core.types import Fixture, MatchId
from predictions.flow.evaluate import GateResult

MESSAGE_LIMIT = 4000  # Telegram allows 4096 characters; keep a margin
MAX_TEXT_LENGTH = 60


def clean_text(value, limit: int = MAX_TEXT_LENGTH) -> str:
    """Externally sourced text made safe: control and format characters removed, spaces collapsed, length capped."""
    text = "".join(" " if unicodedata.category(c) in ("Cc", "Zl", "Zp") else c
                   for c in str(value) if unicodedata.category(c) not in ("Cf", "Cs", "Co", "Cn"))
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def format_date(value: date) -> str:
    return f"{value.day} {value.strftime('%b %Y')}"


def _odds(value: Decimal) -> str:
    return f"{value:.2f}"


def _pick(outcomes, *prefixes):
    """One outcome per prefix, found by name (Home, Draw, Away / Over 2.5, Under 2.5), never by position."""
    found = []
    for prefix in prefixes:
        match = [o for o in outcomes if o.name.strip().lower().startswith(prefix)]
        if len(match) != 1:
            return None
        found.append(match[0])
    return found


def _is_total(specifier: str, line: str) -> bool:
    key, _, value = specifier.partition("=")
    try:
        return key.strip() == "total" and Decimal(value.strip()) == Decimal(line)
    except InvalidOperation:
        return False


def main_odds(fixture: Fixture) -> str:
    parts = []
    for market in fixture.markets:
        if market.market_id == "1":
            picked = _pick(market.outcomes, "home", "draw", "away")
            if picked:
                home, draw, away = picked
                parts.append(f"1 {_odds(home.odds)} X {_odds(draw.odds)} 2 {_odds(away.odds)}")
                break
    for market in fixture.markets:
        if market.market_id == "18" and _is_total(market.specifier, "2.5"):
            picked = _pick(market.outcomes, "over", "under")
            if picked:
                over, under = picked
                parts.append(f"O/U 2.5: {_odds(over.odds)}/{_odds(under.odds)}")
                break
    return " | ".join(parts) if parts else "odds not listed"


def winner_and_total_odds(fixture: Fixture, winner_id: str, total_id: str) -> str:
    """Winner odds and the most balanced listed total (the line whose over and under prices are closest)."""
    parts = []
    for market in fixture.markets:
        if market.market_id == winner_id:
            picked = _pick(market.outcomes, "home", "away")
            if picked:
                parts.append(f"Home {_odds(picked[0].odds)} / Away {_odds(picked[1].odds)}")
                break
    best = None
    for market in fixture.markets:
        if market.market_id != total_id:
            continue
        picked = _pick(market.outcomes, "over", "under")
        key, _, value = market.specifier.partition("=")
        if not picked or key.strip() != "total":
            continue
        try:
            line = Decimal(value.strip())
        except InvalidOperation:
            continue
        rank = (abs(picked[0].odds - picked[1].odds), line)
        if best is None or rank < best[0]:
            best = (rank, line, picked)
    if best is not None:
        parts.append(f"O/U {best[1].normalize():f}: {_odds(best[2][0].odds)}/{_odds(best[2][1].odds)}")
    return " | ".join(parts) if parts else "odds not listed"


def basketball_main_odds(fixture: Fixture) -> str:
    return winner_and_total_odds(fixture, "219", "225")


def baseball_main_odds(fixture: Fixture) -> str:
    return winner_and_total_odds(fixture, "251", "258")


def hockey_main_odds(fixture: Fixture) -> str:
    return winner_and_total_odds(fixture, "406", "412")


def football_main_odds(fixture: Fixture) -> str:
    return winner_and_total_odds(fixture, "219", "225")


# The line of main odds by sport, for a request that lists several sports (`America`). A sport added later adds its line here.
SPORT_ODDS = {"baseball": baseball_main_odds, "hockey": hockey_main_odds, "american_football": football_main_odds}


def multi_sport_main_odds(fixture: Fixture) -> str:
    return SPORT_ODDS.get(fixture.match_id.sport, main_odds)(fixture)


def fixture_line(fixture: Fixture, odds=None) -> str:
    local = fixture.kickoff_utc.astimezone(LAGOS)
    stamp = f"{local.day} {local.strftime('%b %H:%M')}"
    summary = (odds or main_odds)(fixture)
    return f"{stamp} {clean_text(fixture.home)} vs {clean_text(fixture.away)}\n    {summary}"


def stats_line(stats: FixtureStats) -> str:
    if stats.status == READY:
        return f"    data: ready ({len(stats.inputs)} inputs)"
    if stats.status == AMBIGUOUS:
        return f"    needs your choice: {clean_text(stats.question, 200)} Left out until you choose."
    return f"    insufficient data: {clean_text('; '.join(stats.reasons[:1]), 160)}"


def chunk_lines(lines: list[str], limit: int = MESSAGE_LIMIT) -> list[str]:
    chunks, current = [], ""
    for line in lines:
        line = line if len(line) <= limit else line[: limit - 1] + "…"
        if current and len(current) + 1 + len(line) > limit:
            chunks.append(current)
            current = line
        else:
            current = f"{current}\n{line}" if current else line
    if current:
        chunks.append(current)
    return chunks


NO_VALID_PLAY = "NO VALID PLAY - PASS"
MAX_REJECTIONS_SHOWN = 5


def _who(gate: GateResult, match_id: MatchId) -> str:
    fixture = gate.fixtures.get(match_id)
    if fixture is None:
        return str(match_id)
    local = fixture.kickoff_utc.astimezone(LAGOS)
    return f"{local.day} {local.strftime('%b %H:%M')} {clean_text(fixture.home)} vs {clean_text(fixture.away)}"


def evaluation_line(gate: GateResult, ev: Evaluation, with_reason: bool = False) -> str:
    who = _who(gate, ev.match_id)
    if ev.insufficient:
        return f"{who}: {clean_text(play_label(ev))} @ {_odds(ev.odds)} - {clean_text(ev.reason, 200)}"
    text = f"{who}: {clean_text(play_label(ev))} @ {_odds(ev.odds)}, estimate {percent(ev.estimate.probability)}"
    if ev.estimate.push:
        text += f" (win {percent(ev.estimate.win)} + refund {percent(ev.estimate.push)})"
    text += f" - {clean_text(ev.reason, 300)}" if with_reason else f", EV {signed_percent(ev.ev)}"
    if ev.estimate.model == GOALS_ONLY_MODEL_NAME:
        text += f" [{GOALS_ONLY_LABEL}]"  # Story 9.3: no xG behind this estimate, said on every such pick
    if ev.original:
        text += f"\n    original pick: {clean_text(ev.original, 120)}"
    if ev.rule:
        text += f"\n    rule applied: {clean_text(ev.rule, 240)}"
    for note in ev.notes:
        text += f"\n    {clean_text(note, 400)}"
    sourced = [i for i in ev.estimate.inputs if i.source.startswith(ANALYST_PREFIX)]
    if sourced:  # Story 3.1: web-sourced values are always shown with the pick, with their citation
        text += "\n    analyst-sourced inputs: " + "; ".join(
            f"{clean_text(i.name, 40)} {i.value:g} [{clean_text(i.source, 120)}]" for i in sourced)
    return text


def pass_line(gate: GateResult, refusal) -> str:
    return f"{_who(gate, refusal.match_id)}: {NO_VALID_PLAY} ({clean_text(refusal.rule)}: {clean_text(refusal.reason, 200)})"


def under_line(count: int) -> str:
    """AD-20: the one short line that counts Under plays refused for the owner's rule (never listed one by one)."""
    return f"{count} under play(s) ignored: {UNDER_REASON.lower()}."


def whole_line_line(count: int) -> str:
    """AD-21: the one short line that counts whole-line plays refused for the owner's rule (never listed one by one)."""
    return f"{count} whole-line play(s) ignored: {WHOLE_LINE_REASON.lower()}."


def _rejected_not_under(gate: GateResult) -> tuple[list, int]:
    """The rejected picks that may be shown, and how many Under plays are being ignored (central gate and any that slipped)."""
    shown = [e for e in gate.rejected if not is_under(e) and not is_whole_or_quarter_line(e)]
    return shown, gate.under_ignored + sum(1 for e in gate.rejected if is_under(e))


def _whole_ignored(gate: GateResult) -> int:
    """How many whole-line plays are being ignored (central gate and any that slipped into `rejected`; an Under is counted once, as an Under)."""
    return gate.whole_ignored + sum(1 for e in gate.rejected if is_whole_or_quarter_line(e) and not is_under(e))


def gate_lines(gate: GateResult) -> list[str]:
    """Passing picks, or the closest rejections when nothing passes. Never Markdown."""
    skipped = f"{gate.skipped} game(s) skipped for insufficient data." if gate.skipped else ""
    refused = [pass_line(gate, r) for r in gate.passes]
    rejected, unders = _rejected_not_under(gate)
    if gate.passed:
        lines = [f"Valid plays (estimate at least {PROBABILITY_FLOOR:.0%} and positive EV):"]
        def kickoff(ev):   # earliest game first, the better estimate first within one kickoff
            fixture = gate.fixtures.get(ev.match_id)
            return (fixture.kickoff_utc if fixture is not None else ev.match_id.kickoff_utc, -ev.estimate.probability)
        lines += [evaluation_line(gate, e) for e in sorted(gate.passed, key=kickoff)]
        lines.append(f"{len(gate.passed)} valid play(s) from {gate.evaluated} evaluated game(s). "
                     f"{len(rejected)} other pick(s) rejected." + (f" {skipped}" if skipped else ""))
        return lines + refused
    lines = []
    if rejected:
        lines.append("Closest rejections:")
        lines += [evaluation_line(gate, e, True) for e in rejected[:MAX_REJECTIONS_SHOWN]]
    lines += refused
    if unders:
        lines.append(under_line(unders))
    if _whole_ignored(gate):
        lines.append(whole_line_line(_whole_ignored(gate)))
    lines.append(f"{gate.evaluated} game(s) evaluated." + (f" {skipped}" if skipped else ""))
    return lines


MAX_COMPETITIONS_SHOWN = 5
NOT_EVALUATED_NOTE = "insufficient data, not evaluated"


def _competition_counts(competitions) -> str:
    counts: dict[str, int] = {}
    for name in competitions:
        counts[name] = counts.get(name, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    shown = ", ".join(f"{clean_text(name, 50)} {n}" for name, n in ordered[:MAX_COMPETITIONS_SHOWN])
    more = len(ordered) - MAX_COMPETITIONS_SHOWN
    return shown + (f" and {more} more competition(s)" if more > 0 else "")


def why_lines(gate: GateResult, stats: dict[MatchId, FixtureStats] | None, analyst_off: bool = False,
              ignored: int = 0, not_open: int = 0) -> list[str]:
    """Story 9.2: the plain-language WHY block that starts the message after NO VALID PLAY - PASS. It says how many games
    were evaluated and skipped, groups the skip reasons with counts (the top competitions for competition-level reasons),
    and gives one line on what could change the answer. Plain text."""
    stats = stats or {}
    total = len(gate.fixtures)
    lines = ["Why there is no valid play:",
             f"- {total} game(s) looked at: {gate.evaluated} evaluated, {gate.skipped} skipped."]
    if ignored:
        lines.append(f"- {ignored} game(s) ignored: outside your allowed competitions (not looked at, no cost).")
    if not_open:
        lines.append(f"- {not_open} game(s) skipped: not open for betting on SportyBet (not looked at, no cost).")
    groups: dict[str, list[Fixture]] = {}
    for match_id, fixture in gate.fixtures.items():
        entry = stats.get(match_id)
        if entry is not None and entry.status != READY:
            groups.setdefault(skip_category(entry), []).append(fixture)
    for category, games in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        text = f"- Skipped, {category}: {len(games)} game(s)"
        if category in COMPETITION_LEVEL:
            text += f" ({_competition_counts(g.competition for g in games)})"
        lines.append(text + ".")
    if gate.passes:
        lines.append(f"- Passed outright (derby, cup tie or analyst flag): {len(gate.passes)} game(s).")
    rejected, unders = _rejected_not_under(gate)
    if unders:
        lines.append("- " + under_line(unders))
    wholes = _whole_ignored(gate)
    if wholes:
        lines.append("- " + whole_line_line(wholes))
    if gate.evaluated and rejected:
        lines.append(f"- {len(rejected)} pick(s) were priced from the {gate.evaluated} evaluated game(s) but none "
                     f"reached {PROBABILITY_FLOOR:.0%} with positive EV.")
    elif gate.evaluated and not gate.passes and not unders and not wholes:
        lines.append(f"- The {gate.evaluated} evaluated game(s) offered no pick the model can price.")
    hints = []
    if groups:
        hints.append("try another date with games in covered competitions")
    if analyst_off and groups:
        hints.append("turn on the AI analyst by setting ANTHROPIC_API_KEY, so it can fill missing stats")
    if gate.evaluated and rejected:
        hints.append("another date may have picks that clear the floor")
    if hints:
        lines.append("What could change it: " + ", or ".join(hints) + ".")
    return lines


def _compact_reason(entry: FixtureStats) -> str:
    return clean_text("; ".join(entry.reasons[:1]), 160)


def render_matchday(selection: Selection, dates: list[date], stats: dict[MatchId, FixtureStats] | None = None,
                    stats_note: str = "", gate: GateResult | None = None, trigger: str = "Matchday",
                    odds=None, analyst_off: bool = False, ignored: int = 0, not_open: int = 0) -> list[str]:
    """The reply for a request whose dates were read successfully. `trigger` names the request ("Matchday",
    "Gameday") and `odds` gives the one line of main odds for a game (default: the soccer line)."""
    shown = ", ".join(format_date(d) for d in dates)
    lines = []
    if selection.passed_dates:
        passed = ", ".join(format_date(d) for d in selection.passed_dates)
        lines.append(f"Already passed (nothing listed): {passed}.")
    listed = [d for d in dates if d not in selection.passed_dates]
    if not selection.fixtures:
        if listed and not_open:
            lines.append(f"No games open for betting on SportyBet for {', '.join(format_date(d) for d in listed)} "
                         f"(Lagos time). {not_open_line(not_open)}")
            if ignored:
                lines.append(f"{ignored} game(s) ignored: outside your allowed competitions.")
        elif listed and ignored:
            lines.append(f"No games in your allowed competitions for {', '.join(format_date(d) for d in listed)} "
                         f"(Lagos time). {ignored} game(s) ignored: outside your allowed competitions.")
        elif listed:
            lines.append(f"No games found for {', '.join(format_date(d) for d in listed)} (Lagos time).")
    else:
        note = "kickoff in Lagos time" if gate is not None else "kickoff in Lagos time, listing only, nothing evaluated"
        lines.append(f"{trigger} fixtures for {shown} ({note}):")
        first_kickoff = {}
        for f in selection.fixtures:
            first_kickoff.setdefault(f.competition, f.kickoff_utc)
        ordered = sorted(selection.fixtures, key=lambda f: (first_kickoff[f.competition], f.competition, f.kickoff_utc))
        for competition, games in groupby(ordered, key=lambda f: f.competition):
            lines.append("")
            lines.append(clean_text(competition, 80))
            games = list(games)
            # One reason line for the games that share a competition-level reason (not covered by the stats source);
            # games that were evaluated, or have a specific problem, keep their own data line.
            compact = {g.match_id for g in games if stats is not None and g.match_id in stats
                       and stats[g.match_id].status == INSUFFICIENT
                       and skip_category(stats[g.match_id]) in COMPETITION_LEVEL}
            if compact:
                first = next(stats[g.match_id] for g in games if g.match_id in compact)
                everyone = "all " if len(compact) == len(games) else ""
                lines.append(f"    {NOT_EVALUATED_NOTE} for {everyone}{len(compact)} game(s): {_compact_reason(first)}")
            for g in games:
                lines.append(fixture_line(g, odds))
                if stats is not None and g.match_id in stats and g.match_id not in compact:
                    lines.append(stats_line(stats[g.match_id]))
        lines.append("")
        lines.append(f"{len(selection.fixtures)} game(s) listed.")
        if ignored and gate is None:
            lines.append(f"{ignored} game(s) ignored: outside your allowed competitions.")
        if not_open and gate is None:
            lines.append(not_open_line(not_open))
        if stats:
            ready = sum(s.status == READY for s in stats.values())
            asked = sum(s.status == AMBIGUOUS for s in stats.values())
            lines.append(f"Team stats: {ready} ready, {len(stats) - ready - asked} insufficient data"
                         + (f", {asked} need your choice." if asked else "."))
        if stats_note:
            lines.append(stats_note)
    if selection.excluded_started:
        lines.append(f"{selection.excluded_started} game(s) excluded because they had already started or finished.")
    if gate is None:
        return chunk_lines(lines)
    if gate.passed:
        return []  # something passed: the interactive list is the whole reply (only games with an advised play, no noise)
    return [NO_VALID_PLAY] + chunk_lines(why_lines(gate, stats, analyst_off, ignored, not_open) + [""] + gate_lines(gate) + [""] + lines)


# ---------------------------------------------------------------- Review (Epic 4)
def _kickoff(when) -> str:
    local = when.astimezone(LAGOS)
    return f"{local.day} {local.strftime('%b %H:%M')}"


def selection_lines(s, first: str = "") -> list[str]:
    """One selection as the owner reads it: game, sport, market, line and odds (Story 4.1)."""
    line = f" line {clean_text(s.line, 12)}" if s.line else ""
    game = (f"{_kickoff(s.kickoff_utc)} {clean_text(s.home)} vs {clean_text(s.away)} "
            f"[{clean_text(s.sport_label, 30)}, {clean_text(s.competition, 60)}]")
    return [game, f"    {clean_text(s.market_name, 60)}{line}: {clean_text(s.outcome_name)} @ {_odds(s.odds)}{first}"]


def _judged_lines(judgement, with_inputs: bool) -> list[str]:
    ev = judgement.evaluation
    if ev.insufficient:
        lines = selection_lines(judgement.selection)
        lines.append(f"    disqualified: {clean_text(ev.reason, 300)}")
        return lines
    detail = f", success {percent(ev.estimate.probability)}, EV {signed_percent(ev.ev)}"
    lines = selection_lines(judgement.selection, detail)
    if ev.estimate.model == GOALS_ONLY_MODEL_NAME:
        lines.append(f"    {GOALS_ONLY_LABEL}")
    if ev.estimate.push:
        lines.append(f"    win {percent(ev.estimate.win)} + refund {percent(ev.estimate.push)}")
    if judgement.approved:
        lines.append(f"    approved (estimate at least {PROBABILITY_FLOOR:.0%} and positive EV)")
    else:
        lines.append(f"    disqualified: {clean_text(ev.reason, 300)}")
    if ev.rule:
        lines.append(f"    rule applied: {clean_text(ev.rule, 240)}")
    for note in ev.notes:
        lines.append(f"    {clean_text(note, 400)}")
    if with_inputs:
        lines.append(f"    inputs and sources ({ev.estimate.model}):")
        for i in ev.estimate.inputs:
            lines.append(f"      {clean_text(i.name, 40)} {i.value:g} {clean_text(i.unit, 40)} "
                         f"[{clean_text(i.source, 120)}]")
    return lines


def render_review(result, notes: str = "") -> list[str]:
    """The Review reply: approved selections with their inputs and sources, disqualified ones listed separately
    with the reason, and games already started listed as excluded. Plain text, split under Telegram's limit."""
    total = len(result.judgements) + len(result.started)
    lines = [f"Review of booking code {clean_text(result.code, 24)}: {total} selection(s) read "
             f"(kickoff in Lagos time). Nothing was changed on your betslip."]
    if result.approved:
        lines += ["", f"APPROVED ({len(result.approved)}):"]
        for n, j in enumerate(result.approved, 1):
            lines += [f"{n}."] + _judged_lines(j, with_inputs=True)
    if result.disqualified:
        lines += ["", f"DISQUALIFIED ({len(result.disqualified)}):"]
        for n, j in enumerate(result.disqualified, 1):
            lines += [f"{n}."] + _judged_lines(j, with_inputs=False)
    lines += ["", f"{len(result.approved)} approved, {len(result.disqualified)} disqualified, "
                  f"{len(result.started)} already started (not shown)."]
    if notes:
        lines.append(notes)
    if not result.judgements:
        lines.append("Every game on this code has already started or finished, so nothing was evaluated.")
    return chunk_lines(lines)
