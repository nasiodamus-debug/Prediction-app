"""The AI analyst's vocabulary and the strict reader of what it returns (Epic 3, AD-2, AD-10). Standard library only.

The analyst may do two things and nothing else: supply MISSING stat inputs (each labeled
"analyst-sourced:<citation>", range-checked by the input registry, never beating a free source) and raise
context flags from a CLOSED set. A flag can only make an evaluation stricter (the fixture becomes a PASS); there
is no code path where a flag or an analyst value raises an estimate or approves a pick. Anything outside the
schema is discarded whole, never repaired.
"""
from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Any

from predictions.core.inputs import ANALYST_PREFIX, SIDES, Input, registry_for
from predictions.core.types import Fixture

ANALYST_CALL_LIMIT = 40        # research calls per run (one request: a Matchday, a Review of one code); further lookups are not made
MAX_CALL_LIMIT = 1000


class RunLimit:
    """The per-run cap in force: app.py sets `value` once at startup from ANALYST_CALL_LIMIT (a run reads it when it starts)."""
    value = ANALYST_CALL_LIMIT


RUN_LIMIT = RunLimit()


def parse_call_limit(raw) -> int:
    """The ANALYST_CALL_LIMIT setting (analyst calls allowed in ONE request), read once at startup. Unset or empty keeps the default.
    A value that is present but is not a whole number from 1 to 1000 raises ValueError. The message never echoes the value."""
    if raw is None or not str(raw).strip():
        return ANALYST_CALL_LIMIT
    text = str(raw).strip()
    if not text.isascii() or not text.isdigit() or not 1 <= int(text) <= MAX_CALL_LIMIT:
        raise ValueError("ANALYST_CALL_LIMIT in config.env is not a whole number from 1 to 1000 (for example 40). Fix it or remove the "
                         f"line to use the default of {ANALYST_CALL_LIMIT}, then start the bot again.")
    return int(text)


# (the daily limit follows)

DEFAULT_DAILY_CALL_LIMIT = 40  # analyst calls per Lagos day across all requests (ANALYST_DAILY_CALL_LIMIT overrides it)
MAX_CITATION_LENGTH = 200
MAX_IDENTIFIER_LENGTH = 80
MAX_STATS_PER_REPORT = 30
MAX_FLAGS_PER_REPORT = 5


class ContextFlag(Enum):
    """The fixed set of context flags. Adding a member here is the only way to widen what the analyst may say."""
    KEY_PLAYER_OUT = "key_player_out"
    LINEUP_UNCERTAIN = "lineup_uncertain"
    SEVERE_WEATHER = "severe_weather"
    DERBY_OR_RIVALRY = "derby_or_rivalry"
    UNUSUAL_MATCH_CIRCUMSTANCES = "unusual_match_circumstances"


FLAG_LABELS = {
    ContextFlag.KEY_PLAYER_OUT: "key player out",
    ContextFlag.LINEUP_UNCERTAIN: "lineup uncertain",
    ContextFlag.SEVERE_WEATHER: "severe weather",
    ContextFlag.DERBY_OR_RIVALRY: "derby or rivalry",
    ContextFlag.UNUSUAL_MATCH_CIRCUMSTANCES: "unusual match circumstances",
}
RULE_ANALYST_FLAG = "Analyst flag"


@dataclass(frozen=True)
class AnalystFlag:
    flag: ContextFlag
    source: str  # the citation the analyst gave


@dataclass(frozen=True)
class AnalystQuery:
    """Everything the analyst is ever told about a fixture: identifiers, and the input names still missing.
    No credentials, no settings, no database content, no odds."""
    home: str
    away: str
    league: str
    kickoff_date: str  # ISO date, UTC
    wanted: tuple[str, ...] = ()  # "<side>.<stat>" names still missing; empty means flags only
    sport: str = "soccer"         # which sport's registry the names belong to


@dataclass(frozen=True)
class AnalystUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    searches: int = 0


@dataclass(frozen=True)
class AnalystReport:
    """What one research call produced, already validated. `rejected` says why the whole response was
    discarded (then stats and flags are empty)."""
    stats: tuple[Input, ...] = ()
    flags: tuple[AnalystFlag, ...] = ()
    rejected: str = ""
    usage: AnalystUsage = AnalystUsage()


def clean_identifier(value: object) -> str:
    """A team or league name made safe to put in a prompt: no control or format characters, one line, capped."""
    text = "".join(" " if unicodedata.category(c) in ("Cc", "Zl", "Zp") else c
                   for c in str(value) if unicodedata.category(c) not in ("Cf", "Cs", "Co", "Cn"))
    return " ".join(text.split())[:MAX_IDENTIFIER_LENGTH]


def query_for(fixture: Fixture, wanted: tuple[str, ...] = ()) -> AnalystQuery:
    return AnalystQuery(clean_identifier(fixture.home), clean_identifier(fixture.away),
                        clean_identifier(fixture.competition), fixture.kickoff_utc.date().isoformat(), wanted,
                        fixture.match_id.sport)


def _text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or len(value) > MAX_CITATION_LENGTH or not value.isprintable():
        return None
    return value


def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def parse_report(text: str, query: AnalystQuery, usage: AnalystUsage = AnalystUsage()) -> AnalystReport:
    """Read the analyst's answer. The answer must be exactly a JSON object {"stats": [...], "flags": [...]}:
    stats are {"name", "value", "unit", "citation"} for names in `query.wanted`, flags are {"flag", "source"}
    with `flag` in ContextFlag. ANY deviation (extra keys, text around the JSON, unknown flag, unrequested or
    repeated name, wrong type) discards the WHOLE response; nothing is repaired or partially kept.
    Range and unit checks against the input registry happen later, per input (core.inputs.check_inputs)."""
    def reject(why: str) -> AnalystReport:
        return AnalystReport(rejected=why, usage=usage)

    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        return reject("the answer was not valid JSON")
    if not isinstance(data, dict) or set(data) != {"stats", "flags"}:
        return reject("the answer did not have exactly the keys stats and flags")
    raw_stats, raw_flags = data["stats"], data["flags"]
    if not isinstance(raw_stats, list) or not isinstance(raw_flags, list):
        return reject("stats and flags must be lists")
    if len(raw_stats) > MAX_STATS_PER_REPORT or len(raw_flags) > MAX_FLAGS_PER_REPORT:
        return reject("the answer had too many items")
    stats: list[Input] = []
    seen: set[str] = set()
    for item in raw_stats:
        if not isinstance(item, dict) or set(item) != {"name", "value", "unit", "citation"}:
            return reject("a stat did not match the schema")
        name, unit, citation = item["name"], item["unit"], _text(item["citation"])
        if not isinstance(name, str) or name not in query.wanted or name in seen:
            return reject("a stat named an input that was not requested, or repeated one")
        if not _number(item["value"]) or not isinstance(unit, str) or citation is None:
            return reject("a stat had a bad value, unit or citation")
        seen.add(name)
        stats.append(Input(name, float(item["value"]), unit, ANALYST_PREFIX + citation))
    flags: list[AnalystFlag] = []
    for item in raw_flags:
        if not isinstance(item, dict) or set(item) != {"flag", "source"}:
            return reject("a flag did not match the schema")
        source = _text(item["source"])
        try:
            flag = ContextFlag(item["flag"])
        except (ValueError, TypeError):
            return reject("a flag was outside the fixed set")
        if source is None:
            return reject("a flag had no usable source")
        flags.append(AnalystFlag(flag, source))
    return AnalystReport(tuple(stats), tuple(flags), "", usage)


def describe_flags(flags: tuple[AnalystFlag, ...]) -> str:
    return "; ".join(f"{FLAG_LABELS[f.flag]} [{ANALYST_PREFIX}{f.source}]" for f in flags)


def wanted_names_valid(names: tuple[str, ...], sport: str = "soccer") -> bool:
    """True when every name is a team input of the sport's registry (used by adapters to refuse odd queries)."""
    registry = registry_for(sport)
    for name in names:
        side, _, stat = name.partition(".")
        spec = registry.get(stat)
        if side not in SIDES or spec is None or spec.scope != "team":
            return False
    return True


def parse_daily_call_limit(raw) -> int:
    """The ANALYST_DAILY_CALL_LIMIT setting, read once at startup. Unset or empty keeps the default. A value that is present
    but is not a positive whole number raises ValueError (the bot must not start with a cost limit the owner did not
    intend). The message never echoes the value."""
    if raw is None or not str(raw).strip():
        return DEFAULT_DAILY_CALL_LIMIT
    text = str(raw).strip()
    if not text.isascii() or not text.isdigit() or int(text) <= 0 or len(text) > 6:
        raise ValueError("ANALYST_DAILY_CALL_LIMIT in config.env is not a positive whole number (for example 40). "
                         f"Fix it or remove the line to use the default of {DEFAULT_DAILY_CALL_LIMIT}, then start the bot again.")
    return int(text)


ANALYST_MODES = ("off", "last_resort", "full")
DEFAULT_ANALYST_MODE = "last_resort"


def parse_analyst_mode(raw) -> str:
    """The ANALYST_MODE setting, read once at startup. `off`: no analyst is ever built or called. `last_resort` (the default
    when unset): the analyst is asked only for required inputs that every free source left missing, never for a flags-only
    check and never for a team whose season has not started. `full`: the earlier behaviour (also flags-only checks). Any case;
    anything else raises ValueError (the bot must not start with a cost setting the owner did not intend). The message never
    echoes the value."""
    if raw is None or not str(raw).strip():
        return DEFAULT_ANALYST_MODE
    text = str(raw).strip().lower()
    if text not in ANALYST_MODES:
        raise ValueError("ANALYST_MODE in config.env is not one of off, last_resort or full. Fix it or remove the line to use "
                         f"{DEFAULT_ANALYST_MODE}, then start the bot again.")
    return text
