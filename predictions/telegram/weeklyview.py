"""The owner's side of the Monday weekly review: the buttons, what each tap records, and the files the developer reads.

Callback data is `w:<week>:<seq>:<action>` (under Telegram's 64-byte limit), for example `w:2026W40:01:i`. Actions: i implement, l later,
g ignore, d details, u undo. Every handler is idempotent: a repeated tap says "Already recorded" and changes nothing, because the store's
UPDATE holds the old status. Decisions are the OWNER's (the callbacks only run for the owner, the existing gate), and nothing here changes
the engine: an approved item waits in `Decisions` and `local_data/weekly_reviews/APPROVED.md` until a developer implements it and sends
`Done <id>`.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from predictions.core.dates import LAGOS
from predictions.flow.store import EvaluationStore

CALLBACK_LIMIT = 64
LATER_DAYS = 14
_PATTERN = re.compile(r"^w:(\d{4}W\d{2}):(\d{2}):([ilgdu])$")
_ID = re.compile(r"^(\d{4})-W(\d{2})-(\d{2})$")
ACTION_STATUS = {"i": "implement", "l": "later", "g": "ignore"}
LABELS = {"implement": "IMPLEMENT", "later": "LATER", "ignore": "IGNORE", "done": "DONE", "new": "undecided"}


def w_encode(week_key: str, seq: int, action: str) -> str:
    data = f"w:{week_key.replace('-', '')}:{int(seq):02d}:{action}"
    if len(data.encode("utf-8")) > CALLBACK_LIMIT or not _PATTERN.match(data):
        raise ValueError("a weekly callback must be short and well formed")
    return data


def w_decode(data) -> tuple[str, int, str] | None:
    found = _PATTERN.match(str(data or ""))
    if not found:
        return None
    raw, seq, action = found.groups()
    return f"{raw[:4]}-{raw[4:]}", int(seq), action


def rec_id(week_key: str, seq: int) -> str:
    return f"{week_key}-{int(seq):02d}"


def split_id(text: str) -> tuple[str, int] | None:
    found = _ID.match(str(text or "").strip())
    return (f"{found.group(1)}-W{found.group(2)}", int(found.group(3))) if found else None


def rec_keyboard(week_key: str, seq: int) -> tuple[tuple[tuple[str, str], ...], ...]:
    """Rows of (label, callback data) under one recommendation: the three decisions, then Details."""
    return (((("Implement"), w_encode(week_key, seq, "i")), ("Later", w_encode(week_key, seq, "l")), ("Ignore", w_encode(week_key, seq, "g"))),
            (("Details", w_encode(week_key, seq, "d")),))


def decided_keyboard(week_key: str, seq: int) -> tuple[tuple[tuple[str, str], ...], ...]:
    return ((("Change: Implement", w_encode(week_key, seq, "i")), ("Later", w_encode(week_key, seq, "l")), ("Ignore", w_encode(week_key, seq, "g"))),
            (("Undo", w_encode(week_key, seq, "u")), ("Details", w_encode(week_key, seq, "d"))))


@dataclass(frozen=True)
class Decision:
    notice: str                                   # the short answer shown to the owner
    status: str = ""                              # the status now, "" when nothing changed
    keyboard: tuple = ()                          # the buttons to leave under the message (empty: leave them)
    details: str = ""                             # the full text for a Details tap
    changed: bool = False


def _today(now: datetime):
    return now.astimezone(LAGOS).date()


def details_text(row: dict) -> str:
    body = json.loads(row["body_json"] or "{}")
    if not body:
        return f"[{row['id']}] {row['title']}"
    lines = [f"[{row['id']}] {row['title']}", f"Status: {LABELS.get(row['status'], row['status'])}",
             f"Evidence: {body.get('confidence', '')} ({body.get('n', 0)} case(s), {body.get('weeks', 0)} week(s)); category: {body.get('category', '')}",
             f"Finding: {body.get('finding', '')}", f"Proposed change: {body.get('proposed_change', '')}",
             f"Expected effect: {body.get('expected_effect', '')}", f"Risk: {body.get('risk', '')}", f"Effort: {body.get('effort', '')}",
             f"How to check it worked: {body.get('validation', '')}", f"Rollback: {body.get('rollback', '')}"]
    evidence = body.get("evidence") or []
    if evidence:
        lines.append("Evidence ids: " + ", ".join(str(e) for e in evidence))
    if row.get("note"):
        lines.append(f"Your note: {row['note']}")
    return "\n".join(lines)


def apply_decision(store: EvaluationStore, week_key: str, seq: int, action: str, now: datetime | None = None) -> Decision:
    """What one button tap records. Idempotent: the same tap twice changes nothing and says so."""
    now = now or datetime.now(timezone.utc)
    row = store.get_recommendation(rec_id(week_key, seq))
    if row is None:
        return Decision("That recommendation is not on file any more.")
    if action == "d":
        return Decision("Details below.", row["status"], details=details_text(row))
    current = row["status"]
    if current == "done":
        return Decision("That item is already marked done.", current)
    if action == "u":
        if current == "new":
            return Decision("Already recorded: it is undecided.", current)
        store.set_recommendation_status(row["id"], (current,), "new", now=now)
        return Decision("Decision removed: it is undecided again.", "new", rec_keyboard(week_key, seq), changed=True)
    target = ACTION_STATUS.get(action)
    if target is None:
        return Decision("Unknown button.")
    if current == target:
        return Decision("Already recorded.", current)
    resurface = (_today(now) + timedelta(days=LATER_DAYS)).isoformat() if target == "later" else None
    ignored_n = row["n"] if target == "ignore" else None
    changed = store.set_recommendation_status(row["id"], (current,), target, ignored_n=ignored_n, resurface_after=resurface, now=now)
    if not changed:
        return Decision("Already recorded.", target)
    extra = f" It will come back on {resurface} if the problem persists." if resurface else ""
    return Decision(f"Recorded: {LABELS[target]}." + extra, target, decided_keyboard(week_key, seq), changed=True)


def decided_text(original: str, status: str, now: datetime | None = None) -> str:
    """The message text after a decision: its first line plus what was decided (the long body stays in Details and the file)."""
    now = now or datetime.now(timezone.utc)
    first = original.split("\n", 1)[0] if original else ""
    return f"{first}\n=> {LABELS.get(status, status)} (recorded {now.astimezone(LAGOS).strftime('%d %b %H:%M')} Lagos). Press Details for the full text."


def set_done(store: EvaluationStore, rec: str, now: datetime | None = None) -> str:
    row = store.get_recommendation(rec)
    if row is None:
        return f"No recommendation {rec}."
    if row["status"] != "implement":
        return f"{rec} is {LABELS.get(row['status'], row['status'])}, not waiting to be implemented."
    store.set_recommendation_status(rec, ("implement",), "done", now=now)
    return f"{rec} marked done."


def set_note(store: EvaluationStore, rec: str, text: str) -> str:
    row = store.get_recommendation(rec)
    if row is None:
        return f"No recommendation {rec}."
    store.set_recommendation_note(rec, text)
    return f"Note saved on {rec}."


def decisions_text(store: EvaluationStore, which: str = "implement") -> str:
    """The read-only `Decisions` answer: what the owner approved and is waiting to be implemented (or `all` decisions)."""
    rows = store.list_recommendations()
    wanted = rows if which == "all" else [r for r in rows if r["status"] == "implement"]
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in ("new", "implement", "later", "ignore", "done")}
    head = (f"Waiting to be implemented: {counts['implement']}. Undecided: {counts['new']}. Later: {counts['later']}. "
            f"Ignored: {counts['ignore']}. Done: {counts['done']}.")
    if not wanted:
        return head + (" Nothing is waiting." if which != "all" else "")
    lines = [head, ""]
    for r in wanted:
        body = json.loads(r["body_json"] or "{}")
        lines.append(f"[{r['id']}] {r['title']} ({LABELS.get(r['status'], r['status'])}" +
                     (f", decided {r['decided_at_utc'][:10]}" if r["decided_at_utc"] else "") + ")")
        if body.get("proposed_change"):
            lines.append(f"   Change: {body['proposed_change']}")
        if r.get("note"):
            lines.append(f"   Your note: {r['note']}")
    return "\n".join(lines)


def approved_markdown(store: EvaluationStore) -> str:
    rows = [r for r in store.list_recommendations() if r["status"] == "implement"]
    out = ["# Approved weekly-review recommendations (waiting for a developer)", "",
           "The owner chose Implement for each item below in the Telegram weekly review. Implement one at a time, add or update tests, then the "
           "owner sends `Done <id>`. Each item's full evidence is in the weekly file for its week.", ""]
    if not rows:
        out.append("Nothing is waiting.")
    for r in rows:
        out.append(f"## {r['id']}: {r['title']}")
        out.append("")
        out += ["- " + line for line in details_text(r).split("\n")[1:]]
        out.append("")
    return "\n".join(out) + "\n"


def write_approved(store: EvaluationStore, directory: Path) -> Path:
    """Rewrite APPROVED.md atomically (temp file then rename) after every change."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "APPROVED.md"
    temp = directory / "APPROVED.md.tmp"
    temp.write_text(approved_markdown(store), encoding="utf-8")
    temp.replace(path)
    return path
