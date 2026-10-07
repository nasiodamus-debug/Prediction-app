"""The Monday weekly post-game review: gather the previous Monday-to-Sunday week's lost bets, settle each leg from final scores (our own
diagnostic settlement; the bookmaker's result stays the official one), say why each failed, and make detailed recommendations the owner
accepts or ignores. Design: docs/weekly-review-design.md.

This module only BUILDS the report and saves its rows (flow layer = the only writer, AD-12). Sending to Telegram, the buttons and the
scheduler live in app.py / telegram/weeklyview.py. `ScoreSource` is anything with `async final_score(match_id) -> (FinalScore | None,
why)` (adapters/espn/scores.py), so tests use a fake.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path

from predictions.adapters.espn.scores import parse_match_id
from predictions.core.betting import UnsupportedLeg, leg_from_evaluation
from predictions.core.dates import LAGOS
from predictions.core.inputs import Input
from predictions.core.model import Estimate, Evaluation
from predictions.core.postgame import (ANECDOTE, MODERATE, STRONG, WEAK, BetReview, Coverage, LegRecord, Recommendation, R_UNEXPLAINED,
                                       SettledLeg, cut_stats, evidence_level, market_kind, recommend, review_bet, settle,
                                       settled_from)
from predictions.core.settle import LOSS, UNVERIFIABLE, VOID, WIN, FinalScore
from predictions.core.types import MatchId
from predictions.flow.store import PLACED_LIKE, EvaluationStore

log = logging.getLogger("weekly")

WEEKLY_JOB_NAME = "weekly-review"
DEFAULT_WEEKLY_TIME = time(9, 30, tzinfo=LAGOS)
MAX_EXTRA_LEGS = 150                 # approved-but-unplaced picks kept as extra evidence per week (the latest kickoffs first)
MAX_RECOMMENDATIONS = 5              # the post is capped; the rest wait for the next week
IGNORE_GROWTH = 1.5                  # an ignored finding returns only when its evidence grew by this factor (and 3+ legs)
LATER_DAYS = 14
SCORE_CONCURRENCY = 6
MAX_LEG_LINES = 12
_EXPECTED = re.compile(r"expected (?:points|goals|runs) ([\d.]+) - ([\d.]+)")
_UNCLEAR_NAME = re.compile(r"Which team is '([^']+)'")


def parse_weekly_time(raw) -> time:
    """WEEKLY_REVIEW_TIME as HH:MM Lagos time; unset or unreadable keeps the default (09:30)."""
    text = str(raw or "").strip()
    if not text:
        return DEFAULT_WEEKLY_TIME
    match = re.fullmatch(r"([01]?\d|2[0-3]):([0-5]\d)", text)
    return time(int(match.group(1)), int(match.group(2)), tzinfo=LAGOS) if match else DEFAULT_WEEKLY_TIME


def week_containing(day: date) -> tuple[date, date]:
    monday = day - timedelta(days=day.weekday())
    return monday, monday + timedelta(days=6)


def last_completed_week(today: date) -> tuple[date, date]:
    return week_containing(today - timedelta(days=today.weekday() + 1))   # the Monday-Sunday week that ended before this Monday


def week_key(monday: date) -> str:
    iso = monday.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def _range_utc(monday: date, sunday: date) -> tuple[str, str]:
    start = datetime.combine(monday, time(0, 0), tzinfo=LAGOS).astimezone(timezone.utc)
    end = datetime.combine(sunday + timedelta(days=1), time(0, 0), tzinfo=LAGOS).astimezone(timezone.utc)
    return start.isoformat(), end.isoformat()


def weekly_catch_up_due(store: EvaluationStore, now: datetime, weekly_time: time = DEFAULT_WEEKLY_TIME, catchup_days: int = 3) -> bool:
    """True when the bot starts after the Monday run was missed: it is Monday after the run time, or up to `catchup_days` days later,
    and the last completed week has no built review (or only a stale, failed build). The unique week row keeps this idempotent."""
    local = now.astimezone(LAGOS)
    weekday = local.weekday()
    if weekday == 0 and local.time() < weekly_time.replace(tzinfo=None):
        return False   # the scheduled run has not happened yet today
    if weekday > catchup_days:
        return False
    monday, _ = last_completed_week(local.date())
    row = store.weekly_review(monday.isoformat())
    if row is None:
        return True
    if row["status"] != "building":
        return False
    from predictions.flow.store import parse_stamp
    started = parse_stamp(row["built_at_utc"])
    return started is None or now - started >= timedelta(minutes=15)


@dataclass
class WeeklyOutcome:
    week_start: str
    week_key: str
    status: str                                               # posted | dry | degraded | skipped
    summary: list[str] = field(default_factory=list)          # message 1 (split under Telegram's limit)
    recommendations: list[tuple[str, str]] = field(default_factory=list)   # (recommendation id, message text), one message each
    footer: str = ""
    doc_text: str = ""
    doc_path: str = ""
    note: str = ""                                            # why nothing was built ("already posted", ...)


# ---------------------------------------------------------------- gathering
def _kickoff(match_id: str) -> datetime | None:
    parsed = parse_match_id(match_id)
    return parsed[2] if parsed else None


def _expected(notes: str) -> tuple[float, float] | None:
    found = _EXPECTED.search(notes or "")
    return (float(found.group(1)), float(found.group(2))) if found else None


def _soccer_expected(inputs: tuple[tuple[str, float, str], ...]) -> tuple[float, float] | None:
    try:
        from predictions.core.engines.soccer import scaled_rates
        return scaled_rates([Input(n, v, "", s) for n, v, s in inputs])
    except Exception:   # the stored inputs may not be enough for the model: then no expectation, never a guess
        return None


def _leg_record(leg_id: str, week: str, placed: bool, bet_id: str, match_id: str, market_id: str, specifier: str, side: str,
                market_text: str, selection_text: str, odds: float, evaluation: dict | None, inputs: list[dict]) -> LegRecord:
    parsed = parse_match_id(match_id)
    sport, league, home, away = (parsed[0], parsed[1], parsed[3], parsed[4]) if parsed else ("", "", "", "")
    if evaluation is not None:   # the readable names of the game (the match id only carries normalized ones)
        league, home, away = evaluation["league"], evaluation["home"], evaluation["away"]
    stored = tuple((i["name"], float(i["value"]), i["source"]) for i in inputs)
    expected = None
    probability = model = rule = original = None
    ev = None
    if evaluation is not None:
        probability, model = evaluation["probability"], evaluation["model"]
        rule, original = evaluation["rule"], evaluation["original"]
        try:
            ev = float(evaluation["ev"])
        except (TypeError, ValueError):
            ev = None
        expected = _expected(evaluation.get("notes", ""))
    if expected is None and sport == "soccer" and stored:
        expected = _soccer_expected(stored)
    return LegRecord(leg_id, week, placed, bet_id, match_id, sport, league, home, away, str(market_id), specifier, side, market_text,
                     selection_text, odds, probability, model or "", rule or "", original or "", ev, stored, None, "", expected)


def _with_score(leg: LegRecord, score: FinalScore | None, note: str) -> LegRecord:
    from dataclasses import replace
    return replace(leg, score=score, score_note=note)


async def _fetch_scores(scores, match_ids: list[str]) -> dict[str, tuple[FinalScore | None, str]]:
    gate = asyncio.Semaphore(SCORE_CONCURRENCY)
    unique = sorted(set(match_ids))

    async def one(match_id):
        async with gate:
            try:
                return match_id, await scores.final_score(match_id)
            except Exception as error:   # a score source must never fail the report
                log.warning("score lookup failed: %s", type(error).__name__)
                return match_id, (None, "the score source failed")

    return dict(await asyncio.gather(*(one(m) for m in unique)))


def _gather(store: EvaluationStore, monday: date, sunday: date, key: str):
    """Placed bets of the week (by the kickoff of their last leg) and the approved-but-unplaced picks, as LegRecords without scores."""
    start_utc, end_utc = _range_utc(monday, sunday)
    outcomes = {o["placement_id"]: o["result"] for o in store.list_bet_outcomes()}
    placed_keys: set[tuple[str, str]] = set()
    bets: list[dict] = []
    for row in store.list_placements():
        if row["dry_run"] or row["state"] not in PLACED_LIKE:
            continue
        try:
            legs = json.loads(row["legs"])
        except (TypeError, ValueError):
            continue
        kicks = [k for k in (_kickoff(leg["leg_key"][0]) for leg in legs) if k]
        if not kicks or not start_utc <= max(kicks).isoformat() < end_utc:
            continue
        records = []
        for index, leg in enumerate(legs):
            match_id, market_id, specifier, side = leg["leg_key"]
            candidates = [c for c in store.evaluation_candidates(match_id, leg.get("selection", "")) if not c["insufficient"]]
            evaluation = candidates[-1] if candidates else None
            inputs = store.evaluation_inputs(evaluation["id"]) if evaluation else []
            placed_keys.add((match_id, leg.get("selection", "")))
            records.append(_leg_record(f"placed:{row['id']}:{index}", key, True, f"#{row['id']}", match_id, market_id, specifier, side,
                                       leg.get("market", ""), leg.get("selection", ""), float(leg.get("odds", 0) or 0), evaluation, inputs))
        bets.append({"row": row, "legs": records, "bookmaker": outcomes.get(row["id"], "")})
    extra: list[LegRecord] = []
    latest: dict[tuple[str, str, str], dict] = {}
    for evaluation in store.evaluations_between(start_utc, end_utc, passed_only=True):
        latest[(evaluation["match_id"], evaluation["market"], evaluation["selection"])] = evaluation
    for evaluation in sorted(latest.values(), key=lambda e: e["kickoff_utc"], reverse=True):
        if (evaluation["match_id"], evaluation["selection"]) in placed_keys or len(extra) >= MAX_EXTRA_LEGS:
            continue
        parsed = parse_match_id(evaluation["match_id"])
        if parsed is None:
            continue
        try:
            stub = Evaluation(MatchId(parsed[0], parsed[1], parsed[2], parsed[3], parsed[4]), evaluation["market"], evaluation["selection"],
                              Decimal(evaluation["odds"]), Estimate(float(evaluation["probability"])), Decimal(evaluation["ev"]), True, "passed")
            leg = leg_from_evaluation(stub)
        except (UnsupportedLeg, ValueError, ArithmeticError):
            continue
        inputs = store.evaluation_inputs(evaluation["id"])
        extra.append(_leg_record(f"eval:{evaluation['id']}", key, False, "", evaluation["match_id"], leg.market_id, leg.specifier, leg.side,
                                 evaluation["market"], evaluation["selection"], float(evaluation["odds"]), evaluation, inputs))
    return bets, extra


def _coverage(store: EvaluationStore, monday: date, sunday: date) -> Coverage:
    start_utc, end_utc = _range_utc(monday, sunday)
    cov = Coverage()
    seen = set()
    for row in store.evaluations_between(start_utc, end_utc):
        key = (row["match_id"], row["market"], row["selection"])
        if key in seen:
            continue
        seen.add(key)
        cov.total += 1
        reason = str(row.get("reason", ""))
        game = f"{row['home']} v {row['away']} ({row['league']})"
        if not row["insufficient"] and row["probability"] is not None and reason != "no rules for this market" and "no rules" not in reason:
            cov.judged += 1
            continue
        if "no rules for this market" in reason:
            cov.add("no rules for market", f"{row['market']} / {row['selection']} ({row['sport']})", f"{row['market']} ({row['sport']})")
        elif "no rules for this sport" in reason:
            cov.add("no rules for sport", row["sport"])
        elif "analyst call limit" in reason:
            cov.add("analyst call limit", game, f"{row['league']} ({row['sport']})")
        elif "team names are unclear" in reason:
            found = _UNCLEAR_NAME.search(reason)
            if found:
                cov.unclear_names.append(found.group(1))
            cov.add("insufficient data", game, f"{row['league']} ({row['sport']})")
        elif "insufficient data" in reason:
            cov.add("insufficient data", game, f"{row['league']} ({row['sport']})")
        elif "not currently offered" in reason:
            continue
    return cov


# ---------------------------------------------------------------- report text
def _money(x) -> str:
    try:
        return f"{float(x):,.0f}"
    except (TypeError, ValueError):
        return "?"


def _leg_line(d) -> str:
    leg = d.leg
    p = "" if leg.probability is None else f", estimate {leg.probability:.0%}"
    return f"{leg.home} v {leg.away}: {leg.selection_text or leg.market_text} @ {leg.odds:.2f}{p} - {d.settlement.verdict.upper()} ({d.settlement.reason})"


def _bet_block(number: int, review: BetReview) -> list[str]:
    odds = f"@{review.total_odds:.2f}" if review.total_odds else ""
    stake = f", stake {_money(review.stake)}" if review.stake else ""
    joint = review.joint_probability
    lines = [f"{number}) Bet {review.bet_id} {odds}{stake}: {len(review.legs)} leg(s)" + (f", joint estimate {joint:.0%}" if joint is not None else "")]
    for d in review.legs[:MAX_LEG_LINES]:
        lines.append("   - " + _leg_line(d))
    lines.append(f"   Cause: {review.cause}")
    for d in review.failed_legs[:3]:
        for text in d.detail[:4]:
            lines.append(f"     * {text}")
    return lines


def _rec_text(rec_id: str, rec: Recommendation, note: str = "") -> str:
    lines = [f"[{rec_id}] {rec.title}" + (f"  ({note})" if note else ""),
             f"Evidence: {rec.confidence} ({rec.n} case(s), {rec.weeks} week(s)); category: {rec.category}",
             f"Finding: {rec.finding}",
             f"Proposed change: {rec.proposed_change}",
             f"Expected effect: {rec.expected_effect}",
             f"Risk: {rec.risk}",
             f"Effort: {rec.effort}",
             f"How to check it worked: {rec.validation}",
             f"Rollback: {rec.rollback}"]
    return "\n".join(lines)


def _confidence_rank(level: str) -> int:
    return {STRONG: 0, MODERATE: 1, WEAK: 2, ANECDOTE: 3}.get(level, 4)


def _watch_lines(history: list[SettledLeg]) -> list[str]:
    groups: dict[str, list[SettledLeg]] = {}
    for leg in history:
        groups.setdefault(leg.cut, []).append(leg)
    out = []
    for cut, legs in sorted(groups.items()):
        stats = cut_stats(legs)
        if evidence_level(stats.n, stats.weeks) == WEAK and stats.gap >= 0.08:
            out.append(f"{cut.replace('|', ' / ')}: {stats.wins} of {stats.n} won ({stats.hit_rate:.0%}) against an estimate of "
                       f"{stats.mean_estimate:.0%} (not enough legs yet to recommend anything)")
    return out


# ---------------------------------------------------------------- the build
async def run_weekly(store: EvaluationStore, scores, now: datetime | None = None, week_start: date | None = None, dry: bool = False,
                     force: bool = False, write_dir: Path | None = None) -> WeeklyOutcome:
    """Build (and, unless `dry`, save) the review of one week; returns what to post. A week already posted is not built again unless
    `force`; a `dry` run reads everything but writes no row, no history and no file."""
    now = now or datetime.now(timezone.utc)
    monday, sunday = week_containing(week_start) if week_start else last_completed_week(now.astimezone(LAGOS).date())
    key = week_key(monday)
    outcome = WeeklyOutcome(monday.isoformat(), key, "dry" if dry else "posted")
    if not dry and not store.begin_weekly_review(monday.isoformat(), key, now, force=force):
        outcome.status, outcome.note = "skipped", f"The review of {key} was already built (send 'Weekly force' to build it again)."
        return outcome
    try:
        bets, extra = _gather(store, monday, sunday, key)
        scored = await _fetch_scores(scores, [leg.match_id for bet in bets for leg in bet["legs"]] + [leg.match_id for leg in extra])
        reachable = any(score is not None for score, _ in scored.values())
        for bet in bets:
            bet["legs"] = [_with_score(leg, *scored.get(leg.match_id, (None, "no score lookup"))) for leg in bet["legs"]]
        extra = [_with_score(leg, *scored.get(leg.match_id, (None, "no score lookup"))) for leg in extra]
        reviews = [review_bet(f"#{b['row']['id']}", b["legs"], _float(b["row"]["total_odds"]), _float(b["row"]["owner_stake"] or b["row"]["stake"]),
                              b["bookmaker"]) for b in bets]
        history_rows = []
        for leg in [l for b in bets for l in b["legs"]] + extra:
            record = settled_from(leg, settle(leg))
            if record is not None:
                history_rows.append(record)
        coverage = _coverage(store, monday, sunday)
        coverage.placed_legs = sum(len(b["legs"]) for b in bets)
        # only a missing SOURCE or a market a score cannot settle is a coverage gap; a game not finished yet is just timing
        coverage.unverifiable_legs = sum(1 for r in reviews for d in r.legs if d.settlement.verdict == UNVERIFIABLE
                                         and not d.leg.score_note.startswith(("the game was not on the scoreboard", "the score source")))
        if not dry:
            store.save_leg_settlements([{"leg_id": r.leg_id, "week_key": r.week, "placed": r.placed, "cut": r.cut, "sport": r.sport,
                                         "kind": r.kind, "model": r.model, "rule": r.rule, "analyst": r.analyst,
                                         "small_sample": r.small_sample, "probability": r.probability, "odds": r.odds,
                                         "verdict": r.verdict, "ev": r.ev} for r in history_rows], now)
        history = _history(store, history_rows, dry)
        picked, suppressed = _select(store, recommend(history, coverage, reviews), key, now)
        lost = [r for r in reviews if r.failed_legs or r.bookmaker_result == "loss"]
        if not reachable and (bets or extra):
            outcome.status = "dry" if dry else "degraded"
        outcome.summary = _summary(monday, sunday, key, reviews, lost, history_rows, coverage, picked, suppressed, history, reachable)
        saved = []
        for seq, (rec, note) in enumerate(picked, 1):
            rec_id = f"{key}-{seq:02d}"
            saved.append((rec_id, rec, note))
            if not dry:
                store.save_recommendation(rec_id, monday.isoformat(), seq, rec.topic_key, rec.title, rec.category,
                                          json.dumps(rec.__dict__, default=list), rec.n, rec.weeks, rec.confidence, now)
        outcome.recommendations = [(rec_id, _rec_text(rec_id, rec, note)) for rec_id, rec, note in saved]
        outcome.footer = _footer(suppressed, store)
        outcome.doc_text = _document(monday, sunday, key, reviews, extra, saved, coverage, history, suppressed)
        if not dry and write_dir is not None:
            write_dir.mkdir(parents=True, exist_ok=True)
            path = write_dir / f"{key}.md"
            path.write_text(outcome.doc_text, encoding="utf-8")
            outcome.doc_path = str(path)
        return outcome
    except BaseException:
        if not dry:
            store.abort_weekly_review(monday.isoformat())   # a failed build can be run again straight away
        raise


def _float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _history(store: EvaluationStore, this_week: list[SettledLeg], dry: bool) -> list[SettledLeg]:
    """All settled legs: the saved history plus this week's (a dry run has not saved them, so they are added here)."""
    saved = {row["leg_id"]: SettledLeg(row["leg_id"], row["week_key"], bool(row["placed"]), row["cut"], row["sport"], row["kind"], row["model"],
                                       row["rule"], bool(row["analyst"]), bool(row["small_sample"]), row["probability"], row["odds"],
                                       row["verdict"], row["ev"]) for row in store.list_leg_settlements()}
    for leg in this_week:
        saved[leg.leg_id] = leg
    return list(saved.values())


def _select(store: EvaluationStore, recs: list[Recommendation], key: str, now: datetime) -> tuple[list[tuple[Recommendation, str]], Counter]:
    """Apply the decisions of earlier weeks: ignored findings stay out until their evidence grew, 'implement' ones wait for the developer,
    'later' ones return after their date. At most MAX_RECOMMENDATIONS are kept, the best evidence first."""
    chosen: list[tuple[Recommendation, str]] = []
    suppressed: Counter = Counter()
    today = now.astimezone(LAGOS).date().isoformat()
    for rec in sorted(recs, key=lambda r: (_confidence_rank(r.confidence), -r.n)):
        previous = store.latest_recommendation_for_topic(rec.topic_key)
        note = ""
        if previous is not None and not previous["id"].startswith(key):
            status = previous["status"]
            if status == "ignore":
                base = previous["ignored_n"] or previous["n"]
                if rec.n < base * IGNORE_GROWTH or rec.n - base < 3:
                    suppressed["ignored"] += 1
                    continue
                note = f"re-raised: the evidence grew from {base} to {rec.n}"
            elif status == "implement":
                suppressed["awaiting"] += 1
                continue
            elif status == "later":
                if (previous["resurface_after"] or "") > today:
                    suppressed["later"] += 1
                    continue
                note = f"revisited from {previous['id']}"
            elif status == "done":
                suppressed["done"] += 1
                continue
            else:
                note = f"still undecided from {previous['id']}"
        chosen.append((rec, note))
    if len(chosen) > MAX_RECOMMENDATIONS:
        suppressed["over the cap"] += len(chosen) - MAX_RECOMMENDATIONS
    return chosen[:MAX_RECOMMENDATIONS], suppressed


def _footer(suppressed: Counter, store: EvaluationStore) -> str:
    parts = []
    if suppressed.get("ignored"):
        parts.append(f"{suppressed['ignored']} ignored finding(s) unchanged and not repeated")
    if suppressed.get("later"):
        parts.append(f"{suppressed['later']} 'later' item(s) not due yet")
    if suppressed.get("awaiting"):
        parts.append(f"{suppressed['awaiting']} item(s) you approved are waiting for the developer")
    if suppressed.get("over the cap"):
        parts.append(f"{suppressed['over the cap']} more finding(s) held back (the post shows at most {MAX_RECOMMENDATIONS})")
    waiting = len(store.list_recommendations(status="implement"))
    text = "; ".join(parts)
    return (text + ". " if text else "") + f"Press Details under a recommendation for its full text. Send Decisions to list what you approved ({waiting} waiting). The whole report is saved on the PC in local_data/weekly_reviews."


def _summary(monday: date, sunday: date, key: str, reviews: list[BetReview], lost: list[BetReview], history_rows: list[SettledLeg],
             coverage: Coverage, picked, suppressed: Counter, history: list[SettledLeg], reachable: bool) -> list[str]:
    from predictions.telegram.render import chunk_lines
    lines = [f"WEEKLY REVIEW {key}: {monday.strftime('%d %b')} to {sunday.strftime('%d %b %Y')} (Lagos). "
             "Results below are OUR settlement from final scores, a diagnosis: SportyBet's own result is the official one."]
    unverified = [r for r in reviews if not r.verified]
    won = [r for r in reviews if r.verified and not r.failed_legs]
    lines.append(f"Bets you placed this week: {len(reviews)}. Lost (shown below): {len(lost)}. Not lost: {len(won)}. "
                 f"With a leg we could not settle: {len(unverified)}.")
    if not reachable and reviews:
        lines.append("WARNING: no score source could be reached, so no reasons are given this week. Send 'Weekly force' to try again.")
    mismatches = [r for r in reviews if r.bookmaker_result == "loss" and not r.failed_legs and r.verified]
    if mismatches:
        lines.append(f"CHECK: SportyBet says {len(mismatches)} bet(s) lost but our settlement found no losing leg ("
                     + ", ".join(r.bet_id for r in mismatches) + "): our settlement or a score source may be wrong for them.")
    week_legs = [r for r in history_rows if r.week == key]
    if week_legs:
        stats = cut_stats(week_legs)
        lines.append(f"Calibration this week ({len(week_legs)} decided legs, {sum(1 for r in week_legs if r.placed)} placed, the rest approved but "
                     f"unplaced): {stats.wins} won ({stats.hit_rate:.0%}) against a mean estimate of {stats.mean_estimate:.0%}.")
    if not reviews:
        lines.append("No bet you placed in this week was recorded, so there is nothing to explain. Review coverage below still applies.")
    for number, review in enumerate(lost, 1):
        lines.append("")
        lines += _bet_block(number, review)
    open_bets = [r for r in unverified if r not in lost]
    if open_bets:
        lines += ["", "Bets with a leg we could not settle (so we cannot say whether or why they lost; check them on SportyBet):"]
        for review in open_bets:
            why = "; ".join(sorted({d.settlement.reason for d in review.legs if d.settlement.verdict == UNVERIFIABLE}))[:200]
            lines.append(f"- Bet {review.bet_id}: {review.cause}. {why}")
    tally = Counter(d.primary for r in lost for d in r.failed_legs)
    if tally:
        lines += ["", "Why the failed legs failed: " + "; ".join(f"{name or R_UNEXPLAINED} {count}" for name, count in tally.most_common())]
    if coverage.total:
        lines += ["", f"Review coverage: {coverage.judged} of {coverage.total} reviewed selections could be priced; the rest lacked data or rules."]
    watch = _watch_lines(history)
    if watch:
        lines += ["", "Watching (too few legs to recommend yet):"] + [f"- {w}" for w in watch[:6]]
    lines += ["", (f"{len(picked)} recommendation(s) follow, one message each, with buttons: Implement, Later or Ignore. Nothing changes until you decide."
                   if picked else "No recommendation reached the evidence bar this week.")]
    return chunk_lines(lines)


def _document(monday: date, sunday: date, key: str, reviews: list[BetReview], extra: list[LegRecord], saved, coverage: Coverage,
              history: list[SettledLeg], suppressed: Counter) -> str:
    out = [f"# Weekly review {key} ({monday} to {sunday}, Lagos)", "",
           "Results are our own settlement from final scores (diagnosis only); SportyBet's result is the official one.", ""]
    out.append("## Bets you placed")
    if not reviews:
        out.append("None recorded for this week.")
    for review in reviews:
        out.append(f"\n### Bet {review.bet_id}  odds {review.total_odds}  stake {review.stake}  bookmaker result: {review.bookmaker_result or 'not graded'}")
        out.append(f"Cause: {review.cause}")
        for d in review.legs:
            leg = d.leg
            out.append(f"- {leg.home} v {leg.away} [{leg.league}] {leg.market_text} / {leg.selection_text} @ {leg.odds:.2f}: "
                       f"{d.settlement.verdict} ({d.settlement.reason}); primary reason: {d.primary or 'n/a'}")
            out.append(f"  - model {leg.model or 'n/a'}, estimate {leg.probability if leg.probability is not None else 'n/a'}, rule '{leg.rule}'")
            out += [f"  - {text}" for text in d.detail]
            out += [f"  - input {n} = {v:g} [{s}]" for n, v, s in leg.inputs]
    out.append(f"\n## Approved but unplaced picks used as extra evidence: {len(extra)}")
    out.append("\n## Review coverage")
    out.append(f"{coverage.judged} of {coverage.total} selections priced. Reasons: {dict(coverage.reasons)}")
    for category, examples in coverage.examples.items():
        out.append(f"- {category}: " + "; ".join(examples))
    out.append("\n## Recommendations")
    if not saved:
        out.append("None this week.")
    for rec_id, rec, note in saved:
        out.append("\n" + _rec_text(rec_id, rec, note))
        if rec.evidence:
            out.append("Evidence ids: " + ", ".join(rec.evidence))
    out.append(f"\n## Pooled history\n{len(history)} decided legs saved across weeks.")
    return "\n".join(out) + "\n"
