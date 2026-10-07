"""Combined requests: several plays merged into ONE final list, for example

    Review 3ercer and America 6 October
    Gameday, Matchday and America 5 and 6 October

Each part runs exactly as it does on its own (its own rules, floor, no-Under and no-whole-line rules, its own report); only the surviving
picks are merged into a single list, which the Slip then groups by game (two picks on one game become options of that game). Pure parsing
here, standard library only; app.py runs the parts and builds the one list.

Grammar: the text starts with a trigger (Matchday, Gameday, America or Review); each trigger owns the words up to the next trigger, with
the joining words (and, comma, +, &, plus, then, with) trimmed off its ends. `Review` takes exactly one booking code. A date trigger written
WITHOUT dates borrows the dates of the nearest date trigger after it that has some (else the nearest before it), so a date at the end
applies to all of them: "Gameday, Matchday and America 5 and 6 October".
"""
from __future__ import annotations

import re
from dataclasses import dataclass

TRIGGERS = {"matchday": ("MATCHDAY", "Matchday"), "gameday": ("GAMEDAY", "Gameday"), "america": ("AMERICA", "America"),
            "review": ("REVIEW", "Review")}
DATE_KINDS = ("MATCHDAY", "GAMEDAY", "AMERICA")
CONNECTORS = frozenset({"and", "&", "+", "plus", "then", "with", ","})
MAX_PARTS = 4
USAGE = ("A combined request joins plays into one list, for example: Review 23B45S and America 6 October, or Gameday, Matchday and "
         "America 5 and 6 October. Each of Matchday, Gameday and America may be used once and takes dates; Review takes one booking code.")


@dataclass(frozen=True)
class SubRequest:
    kind: str       # MATCHDAY, GAMEDAY, AMERICA or REVIEW (the keys of app.py's requests)
    word: str       # the word as shown to the owner
    argument: str   # the dates (date triggers) or the booking code (Review) it will run with


def _tokens(text: str) -> list[str]:
    return text.replace(",", " , ").split()


def _trim(tokens: list[str]) -> list[str]:
    start, end = 0, len(tokens)
    while start < end and tokens[start].lower() in CONNECTORS:
        start += 1
    while end > start and tokens[end - 1].lower() in CONNECTORS:
        end -= 1
    return tokens[start:end]


def _join(tokens: list[str]) -> str:
    return " ".join(tokens).replace(" ,", ",")


def parse_combined(text: str) -> list[SubRequest] | str | None:
    """None when the text is not a combined request (fewer than two triggers, or it does not start with one: the normal routing
    applies); a list of parts when it is one; or a short message for the owner when it is meant as one but cannot be read."""
    tokens = _tokens(str(text or ""))
    marks = [i for i, t in enumerate(tokens) if t.lower() in TRIGGERS]
    if len(marks) < 2 or marks[0] != 0:
        return None
    parts: list[tuple[str, str, list[str]]] = []
    for n, at in enumerate(marks):
        stop = marks[n + 1] if n + 1 < len(marks) else len(tokens)
        kind, word = TRIGGERS[tokens[at].lower()]
        parts.append((kind, word, _trim(tokens[at + 1:stop])))
    kinds = [kind for kind, _, _ in parts]
    if len(set(kinds)) != len(kinds):
        return "Each of Matchday, Gameday, America and Review can appear only once in a combined request. " + USAGE
    if len(parts) > MAX_PARTS:
        return f"A combined request joins at most {MAX_PARTS} plays. " + USAGE
    out: list[SubRequest] = []
    for kind, word, words in parts:
        if kind == "REVIEW":
            if len(words) != 1:
                return "Review takes exactly one booking code in a combined request, for example: Review 23B45S and Matchday today. " + USAGE
            out.append(SubRequest(kind, word, words[0]))
        else:
            out.append(SubRequest(kind, word, _join(words)))
    # a date trigger without dates borrows them: first from the nearest later date trigger that has some, else the nearest earlier one
    for i, part in enumerate(out):
        if part.kind in DATE_KINDS and not part.argument:
            donors = [p for p in out[i + 1:] if p.kind in DATE_KINDS and p.argument] or \
                     [p for p in reversed(out[:i]) if p.kind in DATE_KINDS and p.argument]
            if donors:
                out[i] = SubRequest(part.kind, part.word, donors[0].argument)
    missing = [part.word for part in out if part.kind in DATE_KINDS and not part.argument]
    if missing:
        return (f"{', '.join(missing)} needs a date: put it at the end, for example: Gameday, Matchday and America 5 and 6 October, "
                "or write it after each play. " + USAGE)
    return out
