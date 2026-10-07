"""Story 2.7: every evaluation is saved (SQLite, flow layer only), with versioning, backups and no secrets."""
import ast
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

import app as bot
from predictions.core.inputs import Input
from predictions.core.model import Estimate, Evaluation
from predictions.core.types import MatchId
from predictions.flow.store import (KEEP_BACKUPS, MIGRATIONS, DatabaseTooNew, EvaluationStore, backup_database,
                                    startup_database)

ROOT = Path(__file__).resolve().parent.parent
KICKOFF = datetime(2026, 10, 10, 14, 0, tzinfo=timezone.utc)
T0 = datetime(2026, 10, 1, 8, 0, tzinfo=timezone.utc)

# Test-only values, set by conftest.py; they are not real credentials.
TOKEN = "123456:TEST-TOKEN-NOT-REAL"
PASSWORD = "test-password-not-real"
API_KEY = "sk-ant-test-key-not-real"
PHONE = "08031234567"


def keep(text):
    return text


def evaluation(selection="Over 2.5", passed=True, source="understat", rule="", original="", notes=("inputs: a",),
               home="Alpha", away="Beta", push=0.0):
    inputs = (Input("home.xg_for", 1.8, "xG per match", source),
              Input("away.xg_against", 1.4, "xG per match", "understat"))
    return Evaluation(MatchId("soccer", "Testland - League", KICKOFF, home, away), "Over/Under", selection,
                      Decimal("1.85"), Estimate(0.72 if passed else 0.55, inputs, "poisson", push), Decimal("0.332"),
                      passed, "passed" if passed else "below the 70% floor", rule, original, tuple(notes))


@pytest.fixture
def db(tmp_path):
    return tmp_path / "local_data" / "predictions.db"


# ---------- what is saved ----------

def test_an_evaluation_is_saved_with_estimate_inputs_sources_rule_and_placed_flag(db):
    ev = evaluation(rule="Asian line", original="Over 2.5", push=0.10)
    with EvaluationStore.open(db, keep) as store:
        store.save_evaluations("sess0001", [ev], now=T0)
        (row,) = store.list_evaluations()
    assert row["session_id"] == "sess0001" and row["evaluated_at_utc"].startswith("2026-10-01T08:00:00")
    assert row["selection"] == "Over 2.5" and row["market"] == "Over/Under" and row["odds"] == "1.85"
    assert row["probability"] == 0.72 and row["push"] == 0.1 and row["model"] == "poisson"
    assert row["rule"] == "Asian line" and row["original"] == "Over 2.5" and row["ev"] == "0.332"
    assert row["passed"] == 1 and row["placed"] == 0  # nothing is ever placed yet
    assert (row["home"], row["away"], row["sport"]) == ("Alpha", "Beta", "soccer")
    assert row["kickoff_utc"].startswith("2026-10-10T14:00:00") and row["match_id"].startswith("soccer:")
    assert [(i["name"], i["value"], i["unit"], i["source"]) for i in row["inputs"]] == [
        ("home.xg_for", 1.8, "xG per match", "understat"), ("away.xg_against", 1.4, "xG per match", "understat")]


def test_rejected_evaluations_are_saved_too(db):
    with EvaluationStore.open(db, keep) as store:
        store.save_evaluations("s", [evaluation(), evaluation("Under 2.5", passed=False)])
        rows = store.list_evaluations("s")
        assert [r["passed"] for r in rows] == [1, 0] and store.list_evaluations("other") == []


def test_analyst_sourced_provenance_is_kept(db):
    ev = evaluation(source="analyst-sourced:https://example.com/stats")
    with EvaluationStore.open(db, keep) as store:
        store.save_evaluations("s", [ev])
        assert store.list_evaluations()[0]["inputs"][0]["source"] == "analyst-sourced:https://example.com/stats"


def test_a_failed_save_leaves_nothing_half_written(db):
    good = evaluation()
    bad = Evaluation(good.match_id, good.market, good.selection, good.odds,
                     Estimate(0.7, (Input("home.xg_for", "not a number", "u", "s"),)), good.ev, True, "x")
    with EvaluationStore.open(db, keep) as store:
        with pytest.raises(ValueError):
            store.save_evaluations("s", [good, bad])
        assert store.list_evaluations() == []


def test_only_the_tables_the_stories_so_far_need_exist(db):
    with EvaluationStore.open(db, keep):
        pass
    conn = sqlite3.connect(db)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert tables - {"sqlite_sequence"} == {"evaluations", "inputs", "placements", "placed_legs", "sessions",
                                                "leg_outcomes", "bet_outcomes", "analyst_usage", "weekly_reviews", "recommendations",
                                                "leg_settlements"}  # v2: Epic 5, v3: sessions, v4: Epic 8 outcomes, v5: analyst usage, v7: weekly review


# ---------- restart ----------

def test_saved_evaluations_survive_a_restart(db):
    first = EvaluationStore.open(db, keep)
    first.save_evaluations("s1", [evaluation(), evaluation("Over 1.5", home="Gamma", away="Delta")], now=T0)
    first.close()
    second = EvaluationStore.open(db, keep)  # the bot starts again
    try:
        rows = second.list_evaluations()
        assert [r["selection"] for r in rows] == ["Over 2.5", "Over 1.5"] and len(rows[0]["inputs"]) == 2
        assert second.schema_version == len(MIGRATIONS)
        second.save_evaluations("s2", [evaluation()])
        assert len(second.list_evaluations()) == 3
    finally:
        second.close()


# ---------- no secrets ----------

def test_secrets_are_redacted_before_they_reach_the_file(db):
    ev = evaluation(selection=f"Over 2.5 {TOKEN}", source=PASSWORD, rule=f"rule {API_KEY}",
                    original=f"call {PHONE}", notes=(f"note +234{PHONE[1:]} and {PASSWORD}",),
                    home=f"Team {TOKEN}")
    store = EvaluationStore.open(db, bot.redact)
    store.save_evaluations(f"sess-{PASSWORD}", [ev])
    row = store.list_evaluations()[0]
    store.close()
    blob = " ".join(str(v) for v in row.values()) + " ".join(str(i) for i in row["inputs"])
    raw = b"".join(p.read_bytes() for p in db.parent.glob("predictions.db*"))
    for secret in (TOKEN, PASSWORD, API_KEY, PHONE, PHONE[1:]):
        assert secret not in blob, secret
        assert secret.encode() not in raw, secret
    assert bot.REDACTED in row["selection"] and row["inputs"][0]["source"] == bot.REDACTED


def test_a_store_cannot_be_opened_without_a_redactor(db):
    with pytest.raises(TypeError):
        EvaluationStore.open(db)


# ---------- schema version and forward-only migrations ----------

def test_schema_version_is_in_pragma_user_version(db):
    with EvaluationStore.open(db, keep) as store:
        assert store.schema_version == len(MIGRATIONS) == 7
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
    conn.close()


def test_a_later_structure_change_upgrades_in_order_after_a_backup(db, tmp_path):
    backups = tmp_path / "backups"
    with EvaluationStore.open(db, keep) as store:
        store.save_evaluations("s", [evaluation()])
    order = []

    base = len(MIGRATIONS)

    def step_two(conn):
        order.append(base + 1)
        conn.execute("ALTER TABLE evaluations ADD COLUMN reviewed INTEGER NOT NULL DEFAULT 0")

    def step_three(conn):
        order.append(base + 2)
        conn.execute("CREATE TABLE later (id INTEGER PRIMARY KEY)")

    migrations = (*MIGRATIONS, step_two, step_three)
    with EvaluationStore.open(db, keep, backup_dir=backups, migrations=migrations, now=T0) as store:
        assert store.schema_version == base + 2 and order == [base + 1, base + 2]
        rows = store.list_evaluations()
        assert len(rows) == 1 and rows[0]["reviewed"] == 0  # the data came through
    (copy,) = backups.glob("predictions-*.db")
    old = sqlite3.connect(copy)
    assert old.execute("PRAGMA user_version").fetchone()[0] == base  # the backup is the file BEFORE the upgrade
    assert old.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == 1
    assert "reviewed" not in [c[1] for c in old.execute("PRAGMA table_info(evaluations)")]
    old.close()


def test_a_failing_migration_rolls_back_and_keeps_the_old_version(db, tmp_path):
    with EvaluationStore.open(db, keep):
        pass

    def broken(conn):
        conn.execute("ALTER TABLE evaluations ADD COLUMN half_done INTEGER")
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        EvaluationStore.open(db, keep, backup_dir=tmp_path / "b", migrations=(*MIGRATIONS, broken))
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == len(MIGRATIONS)
    assert "half_done" not in [c[1] for c in conn.execute("PRAGMA table_info(evaluations)")]
    conn.close()


def test_a_newer_database_is_refused_never_downgraded(db):
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    conn.execute("PRAGMA user_version = 99")
    conn.commit()
    conn.close()
    with pytest.raises(DatabaseTooNew):
        EvaluationStore.open(db, keep)
    conn = sqlite3.connect(db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 99
    conn.close()


# ---------- startup backups ----------

def test_no_backup_when_there_is_no_database_yet(db, tmp_path):
    assert backup_database(db, tmp_path / "backups") is None
    assert not (tmp_path / "backups").exists()


def test_only_the_last_seven_backups_are_kept(db, tmp_path):
    with EvaluationStore.open(db, keep) as store:
        store.save_evaluations("s", [evaluation()])
    backups = tmp_path / "backups"
    made = [backup_database(db, backups, now=T0 + timedelta(days=d)) for d in range(10)]
    kept = sorted(backups.glob("predictions-*.db"))
    assert KEEP_BACKUPS == 7 and kept == sorted(made[3:])  # the 7 newest
    copy = sqlite3.connect(kept[0])
    assert copy.execute("SELECT COUNT(*) FROM evaluations").fetchone()[0] == 1
    copy.close()


def test_startup_backs_up_the_existing_database_then_opens_it(db, tmp_path):
    backups = tmp_path / "backups"
    first = startup_database(db, backups, keep, now=T0)
    assert not backups.exists()  # nothing to copy on the very first start
    first.save_evaluations("s", [evaluation()])
    first.close()
    second = startup_database(db, backups, keep, now=T0 + timedelta(hours=1))
    try:
        assert len(list(backups.glob("predictions-*.db"))) == 1 and len(second.list_evaluations()) == 1
    finally:
        second.close()
    for day in range(2, 12):  # more starts: the folder never holds more than seven
        startup_database(db, backups, keep, now=T0 + timedelta(days=day)).close()
    assert len(list(backups.glob("predictions-*.db"))) == 7


# ---------- who may touch the database ----------

def test_only_the_flow_layer_touches_sqlite():
    offenders = []
    for path in (ROOT / "predictions").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            if any(n.split(".")[0] == "sqlite3" for n in names):
                offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == ["predictions/flow/store.py"]


def test_app_takes_the_backup_at_startup_in_main_and_never_at_import():
    source = (ROOT / "app.py").read_text(encoding="utf-8")
    head, _, main = source.partition('if __name__ == "__main__":')
    assert "startup_database(DB_PATH, BACKUP_DIR, redact)" in main
    assert "startup_database(" not in head.replace("import startup_database", "")
    assert bot.STORE is None  # importing the app opens no database
    assert bot.DB_PATH.parent.name == "local_data" and bot.BACKUP_DIR.parent.name == "local_data"
