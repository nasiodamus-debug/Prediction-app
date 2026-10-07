"""The evaluation database (AD-12, AD-16). The flow layer is its only writer; the core never touches it.

Every text value passes through the injected `redact` function before it is written, so a secret can never
reach the file (AD-9). The schema version is `PRAGMA user_version`; migrations only move forward, each in its
own transaction, and a copy of the file is made before an upgrade. `backup_database` keeps the last 7 copies.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Iterable, Sequence

from predictions.core.betting import LoadedLeg, leg_identity
from predictions.core.model import Evaluation
from predictions.core.placement import DuplicateLeg, PlacementToken

KEEP_BACKUPS = 7
BACKUP_PREFIX = "predictions-"
BACKUP_SUFFIX = ".db"


class DatabaseTooNew(Exception):
    """The file was written by a newer version of the bot. It is never opened or downgraded."""


def _v1_evaluations_and_inputs(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            evaluated_at_utc TEXT NOT NULL,
            match_id TEXT NOT NULL,
            sport TEXT NOT NULL,
            league TEXT NOT NULL,
            home TEXT NOT NULL,
            away TEXT NOT NULL,
            kickoff_utc TEXT NOT NULL,
            market TEXT NOT NULL,
            selection TEXT NOT NULL,
            odds TEXT NOT NULL,
            probability REAL NOT NULL,
            push REAL NOT NULL,
            model TEXT NOT NULL,
            ev TEXT NOT NULL,
            passed INTEGER NOT NULL,
            insufficient INTEGER NOT NULL,
            reason TEXT NOT NULL,
            rule TEXT NOT NULL,
            original TEXT NOT NULL,
            notes TEXT NOT NULL,
            placed INTEGER NOT NULL DEFAULT 0
        )""")
    conn.execute("""
        CREATE TABLE inputs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluation_id INTEGER NOT NULL REFERENCES evaluations(id),
            name TEXT NOT NULL,
            value REAL NOT NULL,
            unit TEXT NOT NULL,
            source TEXT NOT NULL
        )""")
    conn.execute("CREATE INDEX inputs_by_evaluation ON inputs(evaluation_id)")
    conn.execute("CREATE INDEX evaluations_by_session ON evaluations(session_id)")


def _v2_placements(conn: sqlite3.Connection) -> None:
    """Epic 5: one row per PlacementToken (written BEFORE the click, so a crash leaves a trace) and the registry of
    legs that were really placed. Recovery after a crash (Story 5.5) reads these; nothing here holds a secret."""
    conn.execute("""
        CREATE TABLE placements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            token_hash TEXT NOT NULL UNIQUE,
            session_id TEXT NOT NULL,
            state TEXT NOT NULL,
            dry_run INTEGER NOT NULL,
            stake TEXT NOT NULL,
            total_odds TEXT NOT NULL,
            legs TEXT NOT NULL,
            expires_at_utc TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL,
            bet_id TEXT NOT NULL DEFAULT '',
            booking_code TEXT NOT NULL DEFAULT '',
            detail TEXT NOT NULL DEFAULT ''
        )""")
    conn.execute("CREATE INDEX placements_by_state ON placements(state)")
    conn.execute("""
        CREATE TABLE placed_legs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            placement_id INTEGER NOT NULL REFERENCES placements(id),
            match_id TEXT NOT NULL,
            market_id TEXT NOT NULL,
            specifier TEXT NOT NULL,
            side TEXT NOT NULL,
            selection TEXT NOT NULL,
            odds TEXT NOT NULL,
            placed_at_utc TEXT NOT NULL
        )""")
    conn.execute("CREATE INDEX placed_legs_by_leg ON placed_legs(match_id, market_id, specifier, side)")


def _v3_sessions(conn: sqlite3.Connection) -> None:
    """Story 5.5: the state of the slate in progress, saved after each step so a restart can resume or reconcile it.
    `payload` is the deterministic JSON of the flow session (flow/snapshot.py); it holds no secrets."""
    conn.execute("""
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            state TEXT NOT NULL,
            state_version INTEGER NOT NULL,
            origin TEXT NOT NULL DEFAULT '',
            payload TEXT NOT NULL,
            created_at_utc TEXT NOT NULL,
            updated_at_utc TEXT NOT NULL
        )""")
    conn.execute("CREATE INDEX sessions_by_update ON sessions(updated_at_utc)")


def _v4_outcomes(conn: sqlite3.Connection) -> None:
    """Epic 8: the results of PLACED bets, exactly as SportyBet's bet history shows them (AD-13). One row per leg and one
    per bet, written only when SportyBet gave a final result; a game that is still pending has NO row. `source` can only
    be 'sportybet': nothing else may grade a bet. `evaluation_id` links a leg to the saved estimate it was placed on."""
    conn.execute("""
        CREATE TABLE leg_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            placed_leg_id INTEGER NOT NULL UNIQUE REFERENCES placed_legs(id),
            placement_id INTEGER NOT NULL REFERENCES placements(id),
            evaluation_id INTEGER REFERENCES evaluations(id),
            result TEXT NOT NULL CHECK (result IN ('win', 'loss', 'void', 'push')),
            source TEXT NOT NULL CHECK (source = 'sportybet'),
            raw_status TEXT NOT NULL DEFAULT '',
            decided_at_utc TEXT NOT NULL
        )""")
    conn.execute("CREATE INDEX leg_outcomes_by_placement ON leg_outcomes(placement_id)")
    conn.execute("""
        CREATE TABLE bet_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            placement_id INTEGER NOT NULL UNIQUE REFERENCES placements(id),
            result TEXT NOT NULL CHECK (result IN ('win', 'loss', 'void', 'push')),
            source TEXT NOT NULL CHECK (source = 'sportybet'),
            raw_status TEXT NOT NULL DEFAULT '',
            decided_at_utc TEXT NOT NULL
        )""")


def _v5_analyst_usage(conn: sqlite3.Connection) -> None:
    """Story 10.1: how many paid analyst calls were made on each Lagos day, so the daily limit survives a restart."""
    conn.execute("""
        CREATE TABLE analyst_usage (
            day TEXT PRIMARY KEY,
            calls INTEGER NOT NULL CHECK (calls >= 0)
        )""")


def _v6_booking_records(conn: sqlite3.Connection) -> None:
    """AD-25: booking-code records and the owner's own confirmation that he placed one. `source` is '' for every older row
    and 'owner-confirmed' for a booking the owner said he placed; `owner_stake` is the stake he stated (NULL when he skipped
    it). Both columns are additive, so an older database upgrades in place and every older row keeps its meaning."""
    conn.execute("ALTER TABLE placements ADD COLUMN source TEXT NOT NULL DEFAULT ''")
    conn.execute("ALTER TABLE placements ADD COLUMN owner_stake TEXT")


def _v7_weekly_reviews(conn: sqlite3.Connection) -> None:
    """The Monday post-game review: one row per posted week, the recommendations it made (with the owner's decision on each), and the
    settled legs kept as pooled evidence across weeks. Additive: nothing older changes. The flow layer is the only writer."""
    conn.execute("""
        CREATE TABLE weekly_reviews (
            week_start TEXT PRIMARY KEY,
            week_key TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('building', 'posted', 'dry', 'degraded')),
            built_at_utc TEXT NOT NULL,
            posted_at_utc TEXT,
            summary_json TEXT NOT NULL DEFAULT '{}',
            message_ids TEXT NOT NULL DEFAULT '[]',
            doc_path TEXT NOT NULL DEFAULT '',
            revision INTEGER NOT NULL DEFAULT 1
        )""")
    conn.execute("""
        CREATE TABLE recommendations (
            id TEXT PRIMARY KEY,
            week_start TEXT NOT NULL REFERENCES weekly_reviews(week_start),
            seq INTEGER NOT NULL,
            topic_key TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT NOT NULL DEFAULT '',
            body_json TEXT NOT NULL,
            n INTEGER NOT NULL DEFAULT 0,
            weeks INTEGER NOT NULL DEFAULT 0,
            confidence TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'new' CHECK (status IN ('new', 'implement', 'ignore', 'later', 'done')),
            decided_at_utc TEXT,
            note TEXT NOT NULL DEFAULT '',
            ignored_n INTEGER,
            resurface_after TEXT,
            created_at_utc TEXT NOT NULL,
            UNIQUE (week_start, seq)
        )""")
    conn.execute("CREATE INDEX recommendations_by_status ON recommendations(status)")
    conn.execute("CREATE INDEX recommendations_by_topic ON recommendations(topic_key)")
    conn.execute("""
        CREATE TABLE leg_settlements (
            leg_id TEXT PRIMARY KEY,
            week_key TEXT NOT NULL,
            placed INTEGER NOT NULL,
            cut TEXT NOT NULL,
            sport TEXT NOT NULL,
            kind TEXT NOT NULL,
            model TEXT NOT NULL DEFAULT '',
            rule TEXT NOT NULL DEFAULT '',
            analyst INTEGER NOT NULL DEFAULT 0,
            small_sample INTEGER NOT NULL DEFAULT 0,
            probability REAL NOT NULL,
            odds REAL NOT NULL,
            verdict TEXT NOT NULL CHECK (verdict IN ('win', 'loss')),
            ev REAL,
            settled_at_utc TEXT NOT NULL
        )""")
    conn.execute("CREATE INDEX leg_settlements_by_week ON leg_settlements(week_key)")


# Forward-only: index i upgrades version i to version i + 1. Never edit or remove one; append a new one.
MIGRATIONS: tuple[Callable[[sqlite3.Connection], None], ...] = (_v1_evaluations_and_inputs, _v2_placements,
                                                                _v3_sessions, _v4_outcomes, _v5_analyst_usage,
                                                                _v6_booking_records, _v7_weekly_reviews)

# Placement states (the `state` column). PLACING is written before the click; only PLACED is proof of a bet.
P_PLACING = "PLACING"
P_DRY_RUN = "DRY_RUN"              # dry run: checked, reported, never clicked
P_PLACED = "PLACED"                # a bet id was read back from SportyBet's bet history
P_FAILED = "FAILED"                # an error before the click: no bet
P_UNKNOWN = "PLACEMENT_UNKNOWN"    # an error after the click (or it cannot be proven there was none)
P_VOIDED = "VOIDED_ODDS_CHANGED"   # the odds moved before the click: nothing clicked, a new summary was issued
# AD-25 (PLACE_MODE=book): a booking code was generated and NO bet was placed. It is not a placement: it never blocks a
# future list, never enters the placed-leg registry, never counts for Results, Calibration or MAX_STAKE accounting. The row
# keeps the legs and the code so the code can always be found again. `state` is free text, so no migration is needed.
P_BOOKED = "BOOKED"
# AD-25: the owner pressed 'I placed it' on a booking. His word, not SportyBet's: source = 'owner-confirmed'. Its legs ARE
# registered in the placed-leg registry (so they block future lists exactly like PLACED), and Results grades it only when the
# bet history shows a matching bet.
P_PLACED_BY_OWNER = "PLACED_BY_OWNER"
OWNER_SOURCE = "owner-confirmed"
PLACEMENT_STATES = (P_PLACING, P_DRY_RUN, P_PLACED, P_FAILED, P_UNKNOWN, P_VOIDED, P_BOOKED, P_PLACED_BY_OWNER)
# States that hold real, registered legs: the registry blocks on them and Results/Calibration may grade them.
PLACED_LIKE = (P_PLACED, P_PLACED_BY_OWNER)


def parse_stamp(text) -> datetime | None:
    """The datetime of a stored UTC stamp ('...Z' text written by the store), or None."""
    try:
        return datetime.fromisoformat(str(text).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        pass
    try:
        return datetime.strptime(str(text), "%Y%m%dT%H%M%S%fZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _stamp(now: datetime | None) -> str:
    return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "Z"


def backup_database(db_path: Path, backup_dir: Path, keep: int = KEEP_BACKUPS, now: datetime | None = None,
                    label: str = "") -> Path | None:
    """Copy the database into `backup_dir` (a consistent copy, even while it is open) and delete all but the
    newest `keep` copies. Returns the new copy, or None when there is no database yet to copy."""
    db_path, backup_dir = Path(db_path), Path(backup_dir)
    if not db_path.exists():
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / f"{BACKUP_PREFIX}{_stamp(now)}{label}{BACKUP_SUFFIX}"
    source = sqlite3.connect(db_path)
    try:
        copy = sqlite3.connect(target)
        try:
            source.backup(copy)
        finally:
            copy.close()
    finally:
        source.close()
    copies = sorted(p for p in backup_dir.glob(f"{BACKUP_PREFIX}*{BACKUP_SUFFIX}") if p.is_file())
    for old in (copies[:-keep] if keep > 0 else copies):
        old.unlink()
    return target


class EvaluationStore:
    def __init__(self, conn: sqlite3.Connection, redact: Callable[[str], str]):
        self._conn = conn
        self._redact = redact

    @classmethod
    def open(cls, db_path: Path, redact: Callable[[str], str], backup_dir: Path | None = None,
             migrations: Sequence[Callable[[sqlite3.Connection], None]] = MIGRATIONS,
             now: datetime | None = None) -> "EvaluationStore":
        """Open (creating if needed) and upgrade the database. `redact` is required: there is no way to open
        a store that does not filter secrets. When an existing database is about to be upgraded and
        `backup_dir` is given, a copy is made first."""
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, isolation_level=None)  # transactions are explicit below
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > len(migrations):
                raise DatabaseTooNew(f"the database is version {version}; this bot only knows up to {len(migrations)}")
            if 0 < version < len(migrations) and backup_dir is not None:
                backup_database(db_path, backup_dir, now=now, label=f"-before-v{version + 1}")
            for target in range(version + 1, len(migrations) + 1):
                conn.execute("BEGIN")
                try:
                    migrations[target - 1](conn)
                    conn.execute(f"PRAGMA user_version = {int(target)}")
                    conn.execute("COMMIT")
                except BaseException:
                    conn.execute("ROLLBACK")
                    raise
        except BaseException:
            conn.close()
            raise
        return cls(conn, redact)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EvaluationStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        return self._conn.execute("PRAGMA user_version").fetchone()[0]

    def analyst_calls_on(self, day: str) -> int:
        """Analyst calls recorded for one Lagos day (ISO date text); 0 when none."""
        row = self._conn.execute("SELECT calls FROM analyst_usage WHERE day = ?", (str(day),)).fetchone()
        return int(row[0]) if row else 0

    def add_analyst_calls(self, day: str, count: int = 1) -> int:
        """Add `count` calls to that day's total (written at once, so a crash cannot lose a call that was made); the new total."""
        if count < 0:
            raise ValueError("a call count cannot be negative")
        self._conn.execute("INSERT INTO analyst_usage(day, calls) VALUES (?, ?) "
                           "ON CONFLICT(day) DO UPDATE SET calls = calls + excluded.calls", (str(day), int(count)))
        return self.analyst_calls_on(day)

    def _text(self, value) -> str:
        return self._redact(str(value))

    def save_evaluations(self, session_id: str, evaluations: Iterable[Evaluation],
                         now: datetime | None = None) -> list[int]:
        """Save every evaluation shown, passed or not, with its estimate, inputs (with provenance), the rule
        applied and whether it was placed (never yet: placing is Epic 5). All or nothing."""
        when = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        ids: list[int] = []
        conn, text = self._conn, self._text
        conn.execute("BEGIN")
        try:
            for ev in evaluations:
                mid = ev.match_id
                cursor = conn.execute(
                    "INSERT INTO evaluations (session_id, evaluated_at_utc, match_id, sport, league, home, away,"
                    " kickoff_utc, market, selection, odds, probability, push, model, ev, passed, insufficient,"
                    " reason, rule, original, notes, placed) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)",
                    (text(session_id), when, text(mid), text(mid.sport), text(mid.league), text(mid.home),
                     text(mid.away), mid.kickoff_utc.astimezone(timezone.utc).isoformat(), text(ev.market),
                     text(ev.selection), str(ev.odds), ev.estimate.probability, ev.estimate.push,
                     text(ev.estimate.model), str(ev.ev), int(ev.passed), int(ev.insufficient), text(ev.reason),
                     text(ev.rule), text(ev.original), text("\n".join(ev.notes))))
                evaluation_id = cursor.lastrowid
                ids.append(evaluation_id)
                for item in ev.estimate.inputs:
                    conn.execute("INSERT INTO inputs (evaluation_id, name, value, unit, source) VALUES (?,?,?,?,?)",
                                 (evaluation_id, text(item.name), float(item.value), text(item.unit), text(item.source)))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        return ids

    # ---------- placements (Epic 5) ----------

    @staticmethod
    def _now_text(now: datetime | None) -> str:
        return (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()

    @staticmethod
    def _legs_json(legs: Sequence[LoadedLeg]) -> str:
        return json.dumps([{"leg_key": list(leg.leg_key), "match": leg.match_text, "market": leg.market_name,
                            "outcome": leg.outcome_name, "odds": str(leg.odds), "selection": leg.selection,
                            "evaluation_odds": str(leg.evaluation_odds) if leg.evaluation_odds is not None else ""}
                           for leg in legs], sort_keys=True)

    def record_placing(self, token: PlacementToken, dry_run: bool, now: datetime | None = None) -> int:
        """Write the PLACING row for this token. Called BEFORE the adapter is asked to place, and it must succeed or
        nothing is clicked. The token hash is unique, so the same token can never be recorded twice."""
        when = self._now_text(now)
        text = self._text
        cursor = self._conn.execute(
            "INSERT INTO placements (token_hash, session_id, state, dry_run, stake, total_odds, legs, expires_at_utc,"
            " created_at_utc, updated_at_utc) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (token.token_hash, text(token.session_id), P_PLACING, int(bool(dry_run)), str(token.stake),
             str(token.total_odds), text(self._legs_json(token.legs)),
             token.expires_at.astimezone(timezone.utc).isoformat(), when, when))
        return cursor.lastrowid

    def update_placement(self, token_hash: str, state: str, bet_id: str = "", booking_code: str = "",
                         detail: str = "", now: datetime | None = None) -> None:
        """Move a placement to its next state. Raises if the token was never recorded."""
        if state not in PLACEMENT_STATES:
            raise ValueError("unknown placement state")
        cursor = self._conn.execute(
            "UPDATE placements SET state = ?, bet_id = ?, booking_code = ?, detail = ?, updated_at_utc = ?"
            " WHERE token_hash = ?",
            (state, self._text(bet_id), self._text(booking_code), self._text(detail), self._now_text(now), token_hash))
        if cursor.rowcount != 1:
            raise LookupError("no placement is recorded for this token")

    def annotate_placement(self, token_hash: str, note: str, now: datetime | None = None) -> None:
        """The audit record of an owner choice about a placement (for example 'Yes, place again' after an unknown
        result): the note, the time and the placement id are appended to its detail. The STATE never changes, so the
        row stays UNKNOWN and its legs stay blocked. Raises if the token was never recorded."""
        row = self._conn.execute("SELECT id, detail FROM placements WHERE token_hash = ?", (token_hash,)).fetchone()
        if row is None:
            raise LookupError("no placement is recorded for this token")
        line = f"[{self._now_text(now)}] placement #{row[0]}: {note}"
        detail = f"{row[1]} | {line}" if row[1] else line
        self._conn.execute("UPDATE placements SET detail = ? WHERE token_hash = ?", (self._text(detail), token_hash))

    def record_placed_legs(self, token_hash: str, legs: Sequence[LoadedLeg], now: datetime | None = None) -> None:
        """The registry of legs that were really placed (a bet id was read back). All or nothing."""
        when = self._now_text(now)
        conn, text = self._conn, self._text
        row = conn.execute("SELECT id FROM placements WHERE token_hash = ?", (token_hash,)).fetchone()
        if row is None:
            raise LookupError("no placement is recorded for this token")
        conn.execute("BEGIN")
        try:
            for leg in legs:
                match_id, market_id, specifier, side = leg.leg_key
                conn.execute(
                    "INSERT INTO placed_legs (placement_id, match_id, market_id, specifier, side, selection, odds,"
                    " placed_at_utc) VALUES (?,?,?,?,?,?,?,?)",
                    (row[0], text(match_id), text(market_id), text(specifier), text(side), text(leg.selection),
                     str(leg.odds), when))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    # ---------- the slate in progress (Story 5.5) ----------

    def save_session(self, session_id: str, state: str, state_version: int, payload: str, origin: str = "",
                     now: datetime | None = None) -> None:
        """Insert or update the one row for this session. The payload is redacted like every other text, and must
        still be valid JSON afterwards (else nothing is written)."""
        when = self._now_text(now)
        clean = self._text(payload)
        json.loads(clean)  # raises ValueError: a payload the redactor broke is never stored
        self._conn.execute(
            "INSERT INTO sessions (session_id, state, state_version, origin, payload, created_at_utc, updated_at_utc)"
            " VALUES (?,?,?,?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET state = excluded.state,"
            " state_version = excluded.state_version, origin = excluded.origin, payload = excluded.payload,"
            " updated_at_utc = excluded.updated_at_utc",
            (self._text(session_id), self._text(state), int(state_version), self._text(origin), clean, when, when))

    def load_latest_session(self) -> dict | None:
        """The most recently saved session: with one active session at a time, this is the one that matters."""
        rows = self._rows("SELECT * FROM sessions ORDER BY updated_at_utc DESC, rowid DESC LIMIT 1")
        return rows[0] if rows else None

    def list_sessions(self) -> list[dict]:
        return self._rows("SELECT * FROM sessions ORDER BY rowid")

    # ---------- the placed-leg registry (Story 5.5) ----------

    def find_blocking_legs(self, identities: Iterable[Sequence[str]]) -> list[DuplicateLeg]:
        """Which of these legs (match, market, line) were already placed, or are in a placement whose result was
        never proven (PLACING or PLACEMENT_UNKNOWN, real money only)? The side is ignored on purpose. A dry-run
        placement never registers anything and never blocks. Neither does a BOOKED one (a booking code is not a bet)."""
        wanted = {tuple(str(p) for p in identity[:3]) for identity in identities}
        found: dict[tuple, DuplicateLeg] = {}
        for row in self._rows("SELECT l.match_id, l.market_id, l.specifier, l.selection, p.bet_id FROM placed_legs l"
                              " JOIN placements p ON p.id = l.placement_id WHERE p.state IN (?, ?) ORDER BY l.id",
                              PLACED_LIKE):
            identity = (row["match_id"], row["market_id"], row["specifier"])
            if identity in wanted:
                found[identity] = DuplicateLeg(identity, "placed", row["bet_id"], "", row["selection"])
        # Rows whose result is not proven (PLACING, PLACEMENT_UNKNOWN), and PLACED rows that have NO placed_legs rows
        # (a crash between the two writes must never let a proven bet go unblocked): their saved legs block.
        for row in self._rows("SELECT p.legs, p.state, p.bet_id FROM placements p WHERE p.dry_run = 0 AND (p.state IN (?, ?)"
                              " OR (p.state IN (?, ?) AND NOT EXISTS (SELECT 1 FROM placed_legs l WHERE l.placement_id = p.id)))"
                              " ORDER BY p.id", (P_PLACING, P_UNKNOWN, P_PLACED, P_PLACED_BY_OWNER)):
            try:
                legs = json.loads(row["legs"])
            except ValueError:
                raise LookupError("a saved placement could not be read, so the placed-leg registry cannot be trusted")
            source = "placed" if row["state"] in PLACED_LIKE else "unknown"
            for leg in legs:
                identity = leg_identity(leg["leg_key"])
                if identity in wanted and identity not in found:
                    found[identity] = DuplicateLeg(identity, source, row["bet_id"], leg.get("match", ""),
                                                   leg.get("selection", ""))
        return list(found.values())

    def resolve_placement_found(self, token_hash: str, bet_id: str, booking_code: str, detail: str,
                                legs: Sequence[LoadedLeg], now: datetime | None = None) -> None:
        """Reconciliation found the bet in the bet history: mark the placement PLACED and register its legs (once),
        all or nothing."""
        when = self._now_text(now)
        conn, text = self._conn, self._text
        row = conn.execute("SELECT id FROM placements WHERE token_hash = ?", (token_hash,)).fetchone()
        if row is None:
            raise LookupError("no placement is recorded for this token")
        conn.execute("BEGIN")
        try:
            conn.execute("UPDATE placements SET state = ?, bet_id = ?, booking_code = ?, detail = ?, updated_at_utc = ?"
                         " WHERE token_hash = ?", (P_PLACED, text(bet_id), text(booking_code), text(detail), when,
                                                   token_hash))
            registered = conn.execute("SELECT COUNT(*) FROM placed_legs WHERE placement_id = ?", (row[0],)).fetchone()[0]
            if not registered:
                for leg in legs:
                    match_id, market_id, specifier, side = leg.leg_key
                    conn.execute(
                        "INSERT INTO placed_legs (placement_id, match_id, market_id, specifier, side, selection, odds,"
                        " placed_at_utc) VALUES (?,?,?,?,?,?,?,?)",
                        (row[0], text(match_id), text(market_id), text(specifier), text(side), text(leg.selection),
                         str(leg.odds), when))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    # ---------- results and calibration (Epic 8) ----------

    def open_placements_for_results(self) -> list[dict]:
        """Real-money PLACED bets (with a bet id) whose bet-level result is not saved yet, each with its `placed_legs`
        (each marked `decided` when a result is already saved for it). Dry-run and unproven placements never appear."""
        rows = self._rows("SELECT p.* FROM placements p WHERE p.dry_run = 0 AND ((p.state = ? AND p.bet_id <> '') OR p.state = ?)"
                          " AND NOT EXISTS (SELECT 1 FROM bet_outcomes b WHERE b.placement_id = p.id) ORDER BY p.id",
                          (P_PLACED, P_PLACED_BY_OWNER))
        for row in rows:
            row["placed_legs"] = self._rows(
                "SELECT l.*, EXISTS (SELECT 1 FROM leg_outcomes o WHERE o.placed_leg_id = l.id) AS decided"
                " FROM placed_legs l WHERE l.placement_id = ? ORDER BY l.id", (row["id"],))
        return rows

    def evaluation_candidates(self, match_id: str, selection: str) -> list[dict]:
        """Saved evaluations of this game and selection text, oldest first (no inputs)."""
        return self._rows("SELECT * FROM evaluations WHERE match_id = ? AND selection = ? ORDER BY id",
                          (match_id, selection))

    def record_results(self, placement_id: int, bet_result: str | None,
                       legs: Sequence[tuple[int, int | None, str, str]], raw_bet: str = "",
                       now: datetime | None = None) -> None:
        """Save SportyBet's final results: `legs` is (placed_leg_id, evaluation_id or None, result, raw wording); a leg
        already saved is left as it is. `bet_result` (None while the bet is not decided) closes the bet. All or nothing."""
        when = self._now_text(now)
        conn, text = self._conn, self._text
        conn.execute("BEGIN")
        try:
            for placed_leg_id, evaluation_id, result, raw in legs:
                conn.execute("INSERT OR IGNORE INTO leg_outcomes (placed_leg_id, placement_id, evaluation_id, result,"
                             " source, raw_status, decided_at_utc) VALUES (?,?,?,?,'sportybet',?,?)",
                             (placed_leg_id, placement_id, evaluation_id, result, text(raw), when))
            if bet_result is not None:
                conn.execute("INSERT OR IGNORE INTO bet_outcomes (placement_id, result, source, raw_status, decided_at_utc)"
                             " VALUES (?,?,'sportybet',?,?)", (placement_id, bet_result, text(raw_bet), when))
            conn.execute("COMMIT")
        except BaseException:
            conn.execute("ROLLBACK")
            raise

    def graded_legs(self) -> list[dict]:
        """Every graded leg of a real-money PLACED bet: result plus the stored estimate (probability, push) when the leg
        could be linked to a saved evaluation (else None). Nothing that was never placed can appear here."""
        return self._rows(
            "SELECT o.id, o.result, o.evaluation_id, e.probability AS probability, e.push AS push FROM leg_outcomes o"
            " JOIN placements p ON p.id = o.placement_id LEFT JOIN evaluations e ON e.id = o.evaluation_id"
            " WHERE p.state IN (?, ?) AND p.dry_run = 0 ORDER BY o.id", PLACED_LIKE)

    def ungraded_legs_of_settled_bets(self) -> int:
        """Legs of real-money PLACED bets whose BET has a saved result but which have no result of their own: the legs of a
        lost multiple (the list page does not say which leg lost). They are never wins or losses anywhere."""
        rows = self._rows(
            "SELECT COUNT(*) AS n FROM placed_legs l JOIN placements p ON p.id = l.placement_id"
            " JOIN bet_outcomes b ON b.placement_id = p.id"
            " WHERE p.state IN (?, ?) AND p.dry_run = 0 AND NOT EXISTS (SELECT 1 FROM leg_outcomes o WHERE o.placed_leg_id = l.id)",
            PLACED_LIKE)
        return int(rows[0]["n"])

    def list_leg_outcomes(self) -> list[dict]:
        return self._rows("SELECT * FROM leg_outcomes ORDER BY id")

    def list_bet_outcomes(self) -> list[dict]:
        return self._rows("SELECT * FROM bet_outcomes ORDER BY id")

    def _rows(self, query: str, args: tuple = ()) -> list[dict]:
        conn = self._conn
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute(query, args)]
        finally:
            conn.row_factory = None

    def get_placement(self, token_hash: str) -> dict | None:
        rows = self._rows("SELECT * FROM placements WHERE token_hash = ?", (token_hash,))
        return rows[0] if rows else None

    def list_placements(self, state: str | None = None) -> list[dict]:
        if state is None:
            return self._rows("SELECT * FROM placements ORDER BY id")
        return self._rows("SELECT * FROM placements WHERE state = ? ORDER BY id", (state,))

    def find_placement_by_prefix(self, prefix: str) -> dict | None:
        """The placement whose token hash starts with `prefix` (a booking button carries 12 characters of it). None unless
        exactly one row matches."""
        prefix = str(prefix)
        if not prefix or not all(c in "0123456789abcdef" for c in prefix):
            return None
        rows = self._rows("SELECT * FROM placements WHERE token_hash LIKE ? ORDER BY id", (prefix + "%",))
        return rows[0] if len(rows) == 1 else None

    def resolve_owner_placed(self, token_hash: str, stake, now: datetime | None = None) -> bool:
        """AD-25: the owner says he placed this booked bet. Only a BOOKED, real-money row may move (compare-and-swap in the
        UPDATE itself, so a second tap or a race changes nothing); then the row becomes PLACED_BY_OWNER (source
        'owner-confirmed', his stake or NULL) and its legs go into the placed-leg registry, in ONE transaction. Returns False
        when the row was not BOOKED (nothing changed)."""
        when = self._now_text(now)
        conn, text = self._conn, self._text
        row = conn.execute("SELECT id, legs FROM placements WHERE token_hash = ?", (token_hash,)).fetchone()
        if row is None:
            raise LookupError("no placement is recorded for this token")
        stake_text = None if stake is None else format(Decimal(stake).normalize(), "f")
        conn.execute("BEGIN")
        try:
            cursor = conn.execute(
                "UPDATE placements SET state = ?, source = ?, owner_stake = ?, stake = COALESCE(?, stake),"
                " detail = ?, updated_at_utc = ? WHERE token_hash = ? AND state = ? AND dry_run = 0",
                (P_PLACED_BY_OWNER, OWNER_SOURCE, stake_text, stake_text,
                 text("the owner confirmed that he placed this booked bet" + ("" if stake_text else "; stake not given")),
                 when, token_hash, P_BOOKED))
            if cursor.rowcount != 1:
                conn.execute("ROLLBACK")
                return False
            registered = conn.execute("SELECT COUNT(*) FROM placed_legs WHERE placement_id = ?", (row[0],)).fetchone()[0]
            if not registered:
                for leg in json.loads(row[1]):
                    match_id, market_id, specifier, side = leg["leg_key"]
                    conn.execute(
                        "INSERT INTO placed_legs (placement_id, match_id, market_id, specifier, side, selection, odds,"
                        " placed_at_utc) VALUES (?,?,?,?,?,?,?,?)",
                        (row[0], text(match_id), text(market_id), text(specifier), text(side), text(leg.get("selection", "")),
                         str(leg["odds"]), when))
            conn.execute("COMMIT")
        except BaseException:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        return True

    def link_owner_bet(self, placement_id: int, bet_id: str, now: datetime | None = None) -> None:
        """Results found exactly one matching entry in the bet history for an owner-confirmed bet: remember its key."""
        self._conn.execute("UPDATE placements SET bet_id = ?, updated_at_utc = ? WHERE id = ? AND state = ?",
                           (self._text(bet_id), self._now_text(now), int(placement_id), P_PLACED_BY_OWNER))

    def recent_records(self, limit: int = 10) -> list[dict]:
        """The newest real-money records that matter to the owner (booked, placed by him, placed, unknown), newest first."""
        return self._rows("SELECT * FROM placements WHERE dry_run = 0 AND state IN (?, ?, ?, ?) ORDER BY id DESC LIMIT ?",
                          (P_BOOKED, P_PLACED_BY_OWNER, P_PLACED, P_UNKNOWN, int(limit)))

    def list_booked(self) -> list[dict]:
        """Every BOOKED row (booking code only, nothing placed), oldest first, each with its saved legs and its code."""
        return self.list_placements(P_BOOKED)

    def list_placed_legs(self) -> list[dict]:
        return self._rows("SELECT * FROM placed_legs ORDER BY id")

    def list_evaluations(self, session_id: str | None = None) -> list[dict]:
        """Saved evaluations, oldest first, each with its `inputs`."""
        conn = self._conn
        conn.row_factory = sqlite3.Row
        try:
            query, args = "SELECT * FROM evaluations", ()
            if session_id is not None:
                query, args = query + " WHERE session_id = ?", (session_id,)
            rows = [dict(r) for r in conn.execute(query + " ORDER BY id", args)]
            for row in rows:
                row["inputs"] = [dict(r) for r in conn.execute(
                    "SELECT name, value, unit, source FROM inputs WHERE evaluation_id = ? ORDER BY id", (row["id"],))]
        finally:
            conn.row_factory = None
        return rows

    # ---- the Monday weekly review (migration v7): rows, recommendations and the settled-leg history
    def weekly_review(self, week_start: str) -> dict | None:
        rows = self._rows("SELECT * FROM weekly_reviews WHERE week_start = ?", (str(week_start),))
        return rows[0] if rows else None

    def begin_weekly_review(self, week_start: str, week_key: str, now: datetime | None = None, force: bool = False,
                            stale_minutes: int = 15) -> bool:
        """Claim a week to build. False when it is already posted (unless `force`, which makes a new revision) or another build of
        it started less than `stale_minutes` ago. The unique week_start makes the Monday job and a restart catch-up idempotent."""
        stamp, conn = self._now_text(now), self._conn
        current = self.weekly_review(week_start)
        if current is None:
            conn.execute("INSERT INTO weekly_reviews (week_start, week_key, status, built_at_utc) VALUES (?, ?, 'building', ?)",
                         (str(week_start), self._text(week_key), stamp))
            return True
        if current["status"] in ("posted", "dry", "degraded") and not force:
            return False
        if current["status"] == "building" and not force:
            started = parse_stamp(current["built_at_utc"])
            if started is not None and (now or datetime.now(timezone.utc)) - started < timedelta(minutes=stale_minutes):
                return False
        conn.execute("UPDATE weekly_reviews SET status = 'building', built_at_utc = ?, revision = revision + ? WHERE week_start = ?",
                     (stamp, 1 if current["status"] != "building" else 0, str(week_start)))
        return True

    def finish_weekly_review(self, week_start: str, status: str, summary_json: str, message_ids: Sequence[int] = (),
                             doc_path: str = "", now: datetime | None = None) -> None:
        self._conn.execute(
            "UPDATE weekly_reviews SET status = ?, posted_at_utc = ?, summary_json = ?, message_ids = ?, doc_path = ? WHERE week_start = ?",
            (status, self._now_text(now), self._text(summary_json), json.dumps(list(message_ids)), self._text(doc_path), str(week_start)))

    def abort_weekly_review(self, week_start: str) -> None:
        """A build that failed: make its 'building' row stale at once so the next run may claim the week again."""
        self._conn.execute("UPDATE weekly_reviews SET built_at_utc = ? WHERE week_start = ? AND status = 'building'",
                           ("1970-01-01T00:00:00+00:00", str(week_start)))

    def save_leg_settlements(self, rows: Iterable[dict], now: datetime | None = None) -> int:
        count = 0
        for r in rows:
            self._conn.execute(
                "INSERT OR REPLACE INTO leg_settlements (leg_id, week_key, placed, cut, sport, kind, model, rule, analyst, small_sample, "
                "probability, odds, verdict, ev, settled_at_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (r["leg_id"], r["week_key"], int(bool(r["placed"])), r["cut"], r["sport"], r["kind"], self._text(r.get("model", "")),
                 self._text(r.get("rule", "")), int(bool(r.get("analyst"))), int(bool(r.get("small_sample"))), float(r["probability"]),
                 float(r["odds"]), r["verdict"], r.get("ev"), self._now_text(now)))
            count += 1
        return count

    def list_leg_settlements(self) -> list[dict]:
        return self._rows("SELECT * FROM leg_settlements ORDER BY week_key, leg_id")

    def evaluation_inputs(self, evaluation_id: int) -> list[dict]:
        return self._rows("SELECT name, value, unit, source FROM inputs WHERE evaluation_id = ? ORDER BY id", (int(evaluation_id),))

    def evaluations_between(self, start_utc: str, end_utc: str, passed_only: bool = False) -> list[dict]:
        """Evaluations of games kicking off in [start, end) (ISO UTC text), oldest first; `passed_only` keeps the approved picks."""
        query = "SELECT * FROM evaluations WHERE kickoff_utc >= ? AND kickoff_utc < ?"
        if passed_only:
            query += " AND passed = 1 AND insufficient = 0"
        return self._rows(query + " ORDER BY id", (start_utc, end_utc))

    def latest_recommendation_for_topic(self, topic_key: str) -> dict | None:
        rows = self._rows("SELECT * FROM recommendations WHERE topic_key = ? ORDER BY week_start DESC, seq DESC LIMIT 1", (str(topic_key),))
        return rows[0] if rows else None

    def save_recommendation(self, rec_id: str, week_start: str, seq: int, topic_key: str, title: str, category: str, body_json: str,
                            n: int, weeks: int, confidence: str, now: datetime | None = None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO recommendations (id, week_start, seq, topic_key, title, category, body_json, n, weeks, confidence, "
            "created_at_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (rec_id, str(week_start), int(seq), self._text(topic_key), self._text(title), self._text(category), self._text(body_json),
             int(n), int(weeks), self._text(confidence), self._now_text(now)))

    def get_recommendation(self, rec_id: str) -> dict | None:
        rows = self._rows("SELECT * FROM recommendations WHERE id = ?", (str(rec_id),))
        return rows[0] if rows else None

    def list_recommendations(self, status: str | None = None, week_start: str | None = None) -> list[dict]:
        query, args = "SELECT * FROM recommendations", []
        clauses = []
        if status is not None:
            clauses.append("status = ?")
            args.append(status)
        if week_start is not None:
            clauses.append("week_start = ?")
            args.append(str(week_start))
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        return self._rows(query + " ORDER BY week_start, seq", tuple(args))

    def set_recommendation_note(self, rec_id: str, note: str) -> int:
        return self._conn.execute("UPDATE recommendations SET note = ? WHERE id = ?", (self._text(note), str(rec_id))).rowcount

    def set_recommendation_status(self, rec_id: str, from_statuses: Sequence[str], to_status: str, note: str | None = None,
                                  ignored_n: int | None = None, resurface_after: str | None = None,
                                  now: datetime | None = None) -> int:
        """Move one recommendation. The WHERE holds the old status, so a repeated tap or a race changes nothing (returns 0)."""
        marks = ", ".join("?" for _ in from_statuses)
        cursor = self._conn.execute(
            f"UPDATE recommendations SET status = ?, decided_at_utc = ?, note = COALESCE(?, note), ignored_n = ?, resurface_after = ? "
            f"WHERE id = ? AND status IN ({marks})",
            (to_status, self._now_text(now), None if note is None else self._text(note), ignored_n, resurface_after, str(rec_id),
             *from_statuses))
        return cursor.rowcount


def startup_database(db_path: Path, backup_dir: Path, redact: Callable[[str], str],
                     now: datetime | None = None) -> EvaluationStore:
    """What the bot does at startup: a backup copy of the existing database (keeping the last 7), then open
    it, upgrading it in order if its structure is older (with its own copy first)."""
    backup_database(db_path, backup_dir, now=now)
    return EvaluationStore.open(db_path, redact, backup_dir=backup_dir, now=now)
