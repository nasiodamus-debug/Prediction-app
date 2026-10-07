"""Epic 8: results (8.1), the daily check (8.2) and calibration (8.3). Fabricated bet-history LIST entries (bet-level
status only: the real list page shows no per-leg result) and the fake bookmaker; nothing touches SportyBet."""
import asyncio
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest
from telegram import User
from telegram.ext import ApplicationBuilder, ExtBot

import app as bot
from fakes_bookmaker import FakeBookmaker, history_entry, pick_evaluation, two_pick_setup
from predictions.core.betting import LoadedLeg, leg_from_evaluation
from predictions.core.calibration import MIN_BAND_SAMPLE, GradedLeg, band_label, summarise
from predictions.core.placement import MINT_AUTHORITY, PlacementToken, summary_digest
from predictions.core.results import (LOSS, PENDING, PUSH, UNKNOWN, VOID, WIN, MULTIPLE_LOST_REASON, grade_entry)
from predictions.flow.results import (DAILY_CHECK_TIME, ResultsGuard, ResultsReport, calibration_lines, check_results)
from predictions.flow.session import SessionManager
from predictions.flow.store import EvaluationStore
from test_final_list import Chain

OWNER_ID = 111222333
STRANGER = 999888777
EXPIRES = datetime(2026, 12, 1, tzinfo=timezone.utc)


def run(coro):
    return asyncio.run(coro)


@pytest.fixture
def store(tmp_path):
    s = EvaluationStore.open(tmp_path / "e8.db", str)
    yield s
    s.close()


def loaded(ev):
    leg = leg_from_evaluation(ev)
    m = ev.match_id
    return LoadedLeg(leg.key, f"{m.home} v {m.away}", "Over/Under", ev.selection, ev.odds, ev.selection, ev.odds)


def place(store, evals, bet_id="B1234567", session="s1", dry_run=False, state="PLACED", stake="500"):
    """Save the evaluations (once per session), then record a placement of them (and, when PLACED, its legs)."""
    legs = tuple(loaded(e) for e in evals)
    total = Decimal("1")
    for l in legs:
        total *= l.odds
    digest = summary_digest(session, legs, total, Decimal(stake), EXPIRES)
    token = PlacementToken(session, legs, total, Decimal(stake), EXPIRES, digest, MINT_AUTHORITY)
    store.record_placing(token, dry_run)
    if state == "PLACED":
        store.update_placement(token.token_hash, "PLACED", bet_id=bet_id)
        store.record_placed_legs(token.token_hash, legs)
    elif state == "DRY_RUN":
        store.update_placement(token.token_hash, "DRY_RUN")
    return token


BET_ID = "B1234567"   # the placement's stored key (a real one is the synthetic history key)


def entry(bet_id=BET_ID, status="Running", matches=("Alpha v Beta", "Gamma v Delta"), stake="500", others=0, **kw):
    """One history LIST entry: games, stake, a BET-level status; nothing per leg, no odds, no picks."""
    return history_entry(matches, stake, status, others, bet_id=bet_id, **kw)


def two_evals():
    return two_pick_setup()[0]


def setup_bet(store, **kw):
    evals = two_evals()
    store.save_evaluations("s1", evals)
    place(store, evals, **kw)
    return evals


def bm_with(*bets):
    bm = FakeBookmaker()
    bm.history = tuple(bets)
    return bm


def multi(status, **kw):
    return entry(status=status, **kw)


def single_setup(store, bet_id=BET_ID, stake="500"):
    evals = [pick_evaluation("Alpha", "Beta", "Over/Under 2.5", "Over 2.5", "1.85")]
    store.save_evaluations("s1", evals)
    place(store, evals, bet_id=bet_id, stake=stake)
    return evals


def single(status, **kw):
    return entry(status=status, matches=("Alpha v Beta",), **kw)


# ---------------------------------------------------------------- 8.1 core grading (fail closed)

def test_grade_entry_maps_the_bet_status_for_singles():
    assert grade_entry("Singles", "Won", 1) == grade_entry("singles", " won ", 1)
    won, lost = grade_entry("Singles", "Won", 1), grade_entry("Singles", "Lost", 1)
    assert (won.bet, won.legs, won.inferred) == (WIN, WIN, False)
    assert (lost.bet, lost.legs, lost.inferred) == (LOSS, LOSS, False)


def test_grade_entry_a_won_multiple_infers_every_leg_a_lost_multiple_leaves_the_legs_ungraded():
    won = grade_entry("Multiple", "Won", 5)
    assert (won.bet, won.legs, won.inferred) == (WIN, WIN, True) and "void" in won.note
    lost = grade_entry("Multiple", "Lost", 5)
    assert (lost.bet, lost.legs) == (LOSS, None) and lost.note == MULTIPLE_LOST_REASON
    assert "which leg lost is only on the bet details page" in MULTIPLE_LOST_REASON


def test_grade_entry_running_is_pending_and_anything_else_is_unknown():
    assert grade_entry("Multiple", "Running", 4) == grade_entry("Singles", "Running", 1) and grade_entry("Singles", "Running", 1).bet == PENDING
    for word in ("Cashed Out", "Void", "Cancelled", "Half Won", "Settled", ""):
        assert grade_entry("Multiple", word, 3).bet == UNKNOWN and grade_entry("Multiple", word, 3).legs is None
    for kind, count in (("Singles", 3), ("Multiple", 1), ("System", 3), ("", 2), ("Singles", None)):
        assert grade_entry(kind, "Won", count).bet == UNKNOWN, (kind, count)    # the type and the count must fit


# ---------------------------------------------------------------- 8.1 the check

def test_a_won_multiple_saves_every_leg_as_a_win_inferred_from_the_bet(store):
    setup_bet(store)
    report = run(check_results(store, bm_with(multi("Won")), str))
    assert report.checked == 1 and len(report.changes) == 1 and "WON" in report.changes[0] and not report.problems
    assert "inferred from the bet result" in report.changes[0]
    legs, bets = store.list_leg_outcomes(), store.list_bet_outcomes()
    assert [l["result"] for l in legs] == ["win", "win"] and {l["source"] for l in legs} == {"sportybet"}
    assert {l["raw_status"] for l in legs} == {"won (inferred from the bet result)"}
    assert [b["result"] for b in bets] == ["win"] and bets[0]["source"] == "sportybet" and bets[0]["raw_status"] == "Won"
    ids = {e["id"] for e in store.list_evaluations()}
    assert {l["evaluation_id"] for l in legs} == ids                  # each leg is saved against its evaluation
    assert store.open_placements_for_results() == []                   # settled: not read again


def test_a_lost_multiple_is_a_bet_level_loss_and_its_legs_stay_ungraded(store):
    setup_bet(store)
    report = run(check_results(store, bm_with(multi("Lost")), str))
    assert store.list_leg_outcomes() == []                             # NO leg is a win and NO leg is a loss
    assert [(b["result"], b["raw_status"]) for b in store.list_bet_outcomes()] == [("loss", "Lost")]
    assert "LOST" in report.changes[0] and MULTIPLE_LOST_REASON in report.changes[0] and report.ungraded == 2
    assert store.graded_legs() == [] and store.ungraded_legs_of_settled_bets() == 2
    assert store.open_placements_for_results() == []


def test_a_single_takes_the_bet_status_as_its_leg_result(store):
    single_setup(store)
    report = run(check_results(store, bm_with(single("Won")), str))
    assert [l["result"] for l in store.list_leg_outcomes()] == ["win"] and "WON" in report.changes[0]
    assert "inferred" not in report.changes[0] and store.list_leg_outcomes()[0]["raw_status"] == "Won"
    assert [b["result"] for b in store.list_bet_outcomes()] == ["win"]


def test_a_lost_single_is_a_lost_leg(store):
    single_setup(store)
    run(check_results(store, bm_with(single("Lost")), str))
    assert [l["result"] for l in store.list_leg_outcomes()] == ["loss"]
    assert [b["result"] for b in store.list_bet_outcomes()] == ["loss"] and store.ungraded_legs_of_settled_bets() == 0


def test_running_stays_pending_and_is_checked_again_next_time(store):
    setup_bet(store)
    bm = bm_with(multi("Running"))
    report = run(check_results(store, bm, str))
    assert report.pending == 1 and not report.changes and not report.problems
    assert store.list_leg_outcomes() == [] and store.list_bet_outcomes() == []
    bm.history = (multi("Won"),)
    run(check_results(store, bm, str))
    assert [l["result"] for l in store.list_leg_outcomes()] == ["win", "win"]
    assert [b["result"] for b in store.list_bet_outcomes()] == ["win"]
    assert [c[0] for c in bm.calls].count("history") == 2
    again = run(check_results(store, bm, str))
    assert again.checked == 0 and [c[0] for c in bm.calls].count("history") == 2   # nothing open: history not even read


@pytest.mark.parametrize("word", ["Cashed Out", "Void", "Cancelled", "Half Won", "Settled"])
def test_unknown_wording_stays_pending_and_is_reported_never_guessed(store, word):
    setup_bet(store)
    report = run(check_results(store, bm_with(multi(word)), str))
    assert store.list_leg_outcomes() == [] and store.list_bet_outcomes() == []
    assert report.pending == 1 and any("not understood" in p and "do not guess" in p for p in report.problems)


def test_an_entry_that_no_longer_matches_the_placement_is_left_pending(store):
    setup_bet(store)
    for odd in (multi("Won", stake="900"), multi("Won", matches=("Alpha v Beta", "Zulu v Yankee")),
                multi("Won", others=1), entry(status="Won", matches=("Alpha v Beta", "Gamma v Delta"), bet_type="Singles")):
        report = run(check_results(store, bm_with(odd), str))
        assert store.list_bet_outcomes() == [] and any("no longer matches" in p for p in report.problems)


def test_an_entry_with_an_ambiguous_key_is_never_graded(store):
    setup_bet(store)
    report = run(check_results(store, bm_with(multi("Won", ambiguous=True)), str))
    assert store.list_bet_outcomes() == [] and any("cannot be told apart" in p for p in report.problems)


def test_dry_run_and_unplaced_bets_are_never_graded(store):
    evals = two_evals()
    store.save_evaluations("s1", evals)                                # evaluated, never placed
    place(store, evals, session="s2", dry_run=True, state="DRY_RUN")
    bm = bm_with(multi("Won"))
    report = run(check_results(store, bm, str))
    assert report.checked == 0 and store.list_leg_outcomes() == [] and store.list_bet_outcomes() == []
    assert bm.calls == [] and "No placed bet is waiting" in report.text   # no history read for nothing
    assert store.graded_legs() == []


def test_a_bet_that_is_not_in_the_history_stays_pending(store):
    setup_bet(store)
    report = run(check_results(store, bm_with(entry(bet_id="B9999999", status="Won")), str))
    assert report.pending == 1 and store.list_bet_outcomes() == [] and any("not found" in p for p in report.problems)
    assert any("only the 5 newest bets" in p for p in report.problems)


def test_a_history_read_failure_is_reported_sanitized_and_saves_nothing(store):
    from predictions.core.ports import BookmakerError
    setup_bet(store)
    bm = bm_with()

    async def broken():
        raise BookmakerError("could not read 08031234567")
    bm.bet_history = broken
    report = run(check_results(store, bm, lambda e: str(e).replace("08031234567", "[hidden]")))
    assert "[hidden]" in report.error and "08031234567" not in report.text
    assert "will not be retried automatically" in report.text and store.list_leg_outcomes() == []


def test_the_check_only_reads_history_and_never_places_or_loads(store):
    setup_bet(store)
    bm = bm_with(multi("Won"))
    run(check_results(store, bm, str))
    assert [c[0] for c in bm.calls] == ["history"] and bm.clicks == 0


def test_the_outcome_tables_only_accept_sportybet_results(store):
    setup_bet(store)
    import sqlite3
    conn = store._conn
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO bet_outcomes (placement_id, result, source, decided_at_utc) VALUES (1,'win','model','x')")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO bet_outcomes (placement_id, result, source, decided_at_utc) VALUES (1,'maybe','sportybet','x')")


def test_migration_v4_is_forward_only_and_keeps_earlier_rows(tmp_path):
    from predictions.flow.store import MIGRATIONS
    db = tmp_path / "m.db"
    with EvaluationStore.open(db, str, migrations=MIGRATIONS[:3]) as old:
        old.save_evaluations("s", two_evals())
    backups = tmp_path / "b"
    with EvaluationStore.open(db, str, backup_dir=backups) as new:
        assert new.schema_version == 7 and len(new.list_evaluations()) == 2 and new.list_leg_outcomes() == []
    assert list(backups.glob("*-before-v4.db"))


# ---------------------------------------------------------------- 8.2 the daily check

def test_the_daily_check_is_scheduled_once_at_a_fixed_lagos_time():
    application = ApplicationBuilder().token("123456:TEST-TOKEN-NOT-REAL").build()
    job = bot.schedule_daily_results(application)
    assert job is not None and job.name == "daily-results-check"
    assert application.job_queue.get_jobs_by_name("daily-results-check") == (job,)
    assert DAILY_CHECK_TIME.hour == 9 and DAILY_CHECK_TIME.minute == 0 and str(DAILY_CHECK_TIME.tzinfo) == "Africa/Lagos"
    trigger = job.job.trigger
    fields = {f.name: str(f) for f in trigger.fields}
    assert fields["hour"] == "9" and fields["minute"] == "0" and str(trigger.timezone) == "Africa/Lagos"
    assert job.job.max_instances == 1 and job.job.coalesce and job.job.misfire_grace_time == 60


class Ctx:
    def __init__(self):
        self.sent = []
        outer = self

        class B:
            async def send_message(self, chat_id, text, **kw):
                outer.sent.append(text)
        self.bot = B()


@pytest.fixture
def daily(monkeypatch, store):
    monkeypatch.setattr(bot, "STORE", store)
    monkeypatch.setattr(bot, "RESULTS_GUARD", ResultsGuard())
    bm = bm_with()
    monkeypatch.setattr(bot, "BOOKMAKER", bm)
    return bm


def test_the_daily_job_runs_the_same_check_and_messages_the_owner_a_summary(daily, store):
    setup_bet(store)
    daily.history = (multi("Won"),)
    ctx = Ctx()
    run(bot.daily_results_job(ctx))
    assert len(ctx.sent) == 1 and "Daily results check" in ctx.sent[0] and "WON" in ctx.sent[0]
    assert [b["result"] for b in store.list_bet_outcomes()] == ["win"]


def test_the_daily_job_is_quiet_when_nothing_changed(daily, store):
    ctx = Ctx()
    run(bot.daily_results_job(ctx))
    assert ctx.sent == []
    setup_bet(store)
    daily.history = (multi("Running"),)
    run(bot.daily_results_job(ctx))
    assert ctx.sent == []


def test_a_failed_daily_check_is_reported_sanitized_and_not_retried(daily, store):
    setup_bet(store)

    async def broken():
        daily.calls.append(("history",))
        raise RuntimeError("boom test-password-not-real")
    daily.bet_history = broken
    ctx = Ctx()
    run(bot.daily_results_job(ctx))
    assert len(ctx.sent) == 1 and "failed" in ctx.sent[0] and "test-password-not-real" not in ctx.sent[0]
    assert [c[0] for c in daily.calls] == ["history"]                 # one attempt only


def test_overlapping_checks_never_stack_or_double_run(daily, store):
    setup_bet(store)
    gate = asyncio.Event()
    started = []

    async def slow():
        started.append(1)
        await gate.wait()
        return (multi("Won"),)
    daily.bet_history = slow

    async def scenario():
        first = asyncio.create_task(bot.run_results_check())
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        second = await bot.run_results_check()                         # while the first is still reading
        ctx = Ctx()
        await bot.daily_results_job(ctx)                               # the daily job is skipped too
        gate.set()
        return await first, second, ctx.sent

    first, second, sent = run(scenario())
    assert second is None and sent == [] and started == [1] and first.changes
    assert not bot.RESULTS_GUARD.busy                                  # released, so the next check can run


def test_the_guard_is_released_after_an_error(monkeypatch, daily, store):
    setup_bet(store)

    async def broken():
        raise RuntimeError("x")
    daily.bet_history = broken
    report = run(bot.run_results_check())
    assert report.error and not bot.RESULTS_GUARD.busy


def test_no_missed_run_is_invented_at_startup():
    application = ApplicationBuilder().token("123456:TEST-TOKEN-NOT-REAL").build()
    job = bot.schedule_daily_results(application)
    assert job.job.misfire_grace_time <= 60 and job.job.coalesce   # a run missed while stopped is dropped, not replayed


# ---------------------------------------------------------------- 8.3 calibration

def gl(p, r):
    return GradedLeg(p, r)


def test_bands_and_their_edges():
    assert [band_label(p) for p in (0.65, 0.699, 0.70, 0.749, 0.75, 0.799, 0.80, 0.95, 0.60)] == \
           ["65-70%", "65-70%", "70-75%", "70-75%", "75-80%", "75-80%", "80%+", "80%+", "under 65%"]


def test_summary_counts_voids_and_pushes_apart_and_the_win_rate_excludes_them():
    legs = [gl(0.72, WIN)] * 3 + [gl(0.72, LOSS)] + [gl(0.72, VOID), gl(0.73, PUSH), gl(0.86, WIN)]
    bands = {b.label: b for b in summarise(legs)}
    b = bands["70-75%"]
    assert (b.legs, b.wins, b.losses, b.voids, b.pushes) == (6, 3, 1, 1, 1) and b.win_rate == 0.75 and b.too_small
    assert bands["80%+"].wins == 1 and "65-70%" not in bands


def test_a_band_of_only_voids_has_no_win_rate():
    (b,) = summarise([gl(0.7, VOID)])
    assert b.win_rate is None and b.too_small


def test_a_band_needs_20_decided_legs_to_be_judged():
    legs = [gl(0.72, WIN)] * 14 + [gl(0.72, LOSS)] * 5
    assert summarise(legs)[0].too_small
    assert not summarise(legs + [gl(0.72, WIN)])[0].too_small and MIN_BAND_SAMPLE == 20


def test_calibration_says_so_when_nothing_is_graded(store):
    assert "nothing to calibrate" in calibration_lines(store)[0]
    setup_bet(store)                                                   # placed but not graded
    assert "nothing to calibrate" in calibration_lines(store)[0]


def test_calibration_uses_placed_graded_legs_with_their_stored_estimate(store):
    evals = setup_bet(store)                                            # estimates 0.72 and 0.88
    run(check_results(store, bm_with(multi("Won")), str))
    text = "\n".join(calibration_lines(store))
    assert "2 graded leg(s)" in text
    assert "70-75%: 1 leg(s): 1 won, 0 lost, 0 void, 0 push. Win rate 100%" in text
    assert "80%+: 1 leg(s): 1 won, 0 lost" in text and "too small a sample to judge" in text
    assert "push-protected" in text and "neither a win nor a loss" in text
    assert "inferred from the bet" in text and "ungraded" in text and "lost multiple" in text   # the one-line limitation


def test_calibration_reports_the_legs_of_lost_multiples_apart_and_never_counts_them(store):
    setup_bet(store)                                                    # a two-leg multiple that LOST: legs ungraded
    run(check_results(store, bm_with(multi("Lost")), str))
    lines = calibration_lines(store)
    assert "nothing to calibrate" in lines[0]                           # no leg has a known result
    assert any("2 leg(s) of lost multiples are ungraded" in l and "never counted as wins or losses" in l for l in lines)
    # a won single next to it: the band holds ONLY the known leg; the lost multiple's two legs are the separate count
    evals = [pick_evaluation("Zeta", "Eta", "Over/Under 2.5", "Over 2.5", "1.85", prob=0.78, hour=18)]
    store.save_evaluations("s9", evals)
    place(store, evals, bet_id="B2", session="s9", stake="700")
    run(check_results(store, bm_with(multi("Lost"), entry(bet_id="B2", status="Won", matches=("Zeta v Eta",), stake="700")), str))
    text = "\n".join(calibration_lines(store))
    assert "1 graded leg(s)" in text and "75-80%: 1 leg(s): 1 won, 0 lost" in text and "Win rate 100%" in text
    assert "2 leg(s) of lost multiples are ungraded" in text and "70-75%" not in text and "80%+" not in text


def test_calibration_never_counts_evaluated_only_or_dry_run_picks(store):
    evals = two_evals()
    store.save_evaluations("s1", evals)
    place(store, evals, session="s2", dry_run=True, state="DRY_RUN")
    run(check_results(store, bm_with(multi("Won")), str))
    assert "nothing to calibrate" in calibration_lines(store)[0]


def test_a_graded_leg_without_a_stored_estimate_is_reported_not_banded(store):
    evals = two_evals()                                                 # placed WITHOUT saving the evaluations
    place(store, evals)
    run(check_results(store, bm_with(multi("Won")), str))
    text = "\n".join(calibration_lines(store))
    assert "could not be linked" in text and "no bands to show" in text


def test_the_latest_evaluation_of_the_same_session_is_the_one_linked(store):
    older = pick_evaluation(prob=0.66)
    store.save_evaluations("other", [older])
    evals = two_evals()
    store.save_evaluations("s1", evals)
    place(store, evals)
    run(check_results(store, bm_with(multi("Won")), str))
    probs = sorted(r["probability"] for r in store.graded_legs())
    assert probs == [0.72, 0.88]


# ---------------------------------------------------------------- the triggers through the real handler chain

@pytest.fixture
def chain(monkeypatch, tmp_path, store):
    sent = []

    async def no_network(self):
        self._bot_user = User(id=1, first_name="TestBot", is_bot=True, username="test_bot")

    async def send_message(self, chat_id, text, **kwargs):
        sent.append((text, kwargs))

    monkeypatch.setattr(ExtBot, "initialize", no_network)
    monkeypatch.setattr(ExtBot, "send_message", send_message)
    monkeypatch.setattr(bot, "SESSIONS", SessionManager())
    monkeypatch.setattr(bot, "STORE", store)
    monkeypatch.setattr(bot, "RESULTS_GUARD", ResultsGuard())
    bm = bm_with()
    monkeypatch.setattr(bot, "BOOKMAKER", bm)
    application = ApplicationBuilder().token("123456:TEST-TOKEN-NOT-REAL").build()
    bot.register_handlers(application)
    c = Chain(application, sent, [], [])
    c.store, c.bm = store, bm
    return c


def test_results_trigger_reads_and_replies_for_the_owner(chain):
    setup_bet(chain.store)
    chain.bm.history = (multi("Won"),)
    chain.text(OWNER_ID, "Results")
    assert "Results check: 1 placed bet(s)" in chain.sent[-1][0] and "WON" in chain.sent[-1][0]
    assert len(chain.store.list_bet_outcomes()) == 1


def test_calibration_trigger_replies_for_the_owner(chain):
    chain.text(OWNER_ID, "calibration")
    assert "nothing to calibrate" in chain.sent[-1][0]


def test_a_stranger_gets_nothing_from_results_or_calibration(chain):
    setup_bet(chain.store)
    chain.bm.history = (multi("Won"),)
    chain.text(STRANGER, "Results")
    chain.text(STRANGER, "Calibration")
    assert chain.sent == [] and chain.bm.calls == [] and chain.store.list_leg_outcomes() == []


def test_results_says_so_when_a_check_is_already_running(chain):
    assert bot.RESULTS_GUARD.try_begin()
    chain.text(OWNER_ID, "Results")
    assert "already running" in chain.sent[-1][0] and chain.bm.calls == []


def test_help_and_the_unrecognised_reply_name_the_new_triggers(chain):
    chain.text(OWNER_ID, "hello there")
    assert "`Results`" in chain.sent[-1][0] and "`Calibration`" in chain.sent[-1][0]
    chain.feed({"message": {"message_id": 9, "date": 1700000000, "chat": {"id": OWNER_ID, "type": "private"},
                            "from": {"id": OWNER_ID, "is_bot": False, "first_name": "T"}, "text": "/start",
                            "entities": [{"type": "bot_command", "offset": 0, "length": 6}]}})
    start = chain.sent[-1][0]
    assert "Results" in start and "Calibration" in start


def test_results_report_text_when_nothing_is_open_and_when_a_check_failed():
    assert "No placed bet is waiting" in ResultsReport().text
    assert "failed: x" in ResultsReport(error="x").text
