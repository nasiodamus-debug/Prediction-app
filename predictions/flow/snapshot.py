"""Saving and restoring a flow Session as deterministic JSON (Story 5.5, AD-12).

`to_json` is a pure function of the session: sorted keys, fixed separators, Decimals as text, datetimes as UTC ISO text,
so the same session always gives the same bytes and `from_json(to_json(s)) == s`. Nothing secret is ever part of a
Session (names, markets, odds, estimates and inputs only); the store redacts the text again before it is written.

`from_json` fails closed: anything missing, of the wrong type, or inconsistent (for example a pick id that is not the
one its pick derives) raises SnapshotError, and the caller restores nothing rather than guessing.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Any

from predictions.core.betting import LoadedLeg, LoadedSlip
from predictions.core.inputs import Input
from predictions.core.market_rules import is_under, is_whole_or_quarter_line
from predictions.core.model import Estimate, Evaluation
from predictions.core.placement import Summary
from predictions.core.slip import Pick, Slip, make_pick_id
from predictions.core.types import MatchId
from predictions.flow.session import LIST_STATES, Action, DroppedPick, Issue, PickFailure, Session, State

SNAPSHOT_VERSION = 1


class SnapshotError(Exception):
    """A saved session could not be read back exactly. Nothing is restored from it."""


def _dt(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(timezone.utc).isoformat()


def _parse_dt(value: Any) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise SnapshotError("a saved time has no time zone")
    return parsed.astimezone(timezone.utc)


def _dec(value: Decimal | None) -> str | None:
    return None if value is None else str(value)


def _parse_dec(value: Any) -> Decimal | None:
    if value is None:
        return None
    try:
        parsed = Decimal(value)
    except (InvalidOperation, TypeError, ValueError):
        raise SnapshotError("a saved number could not be read") from None
    if not parsed.is_finite():
        raise SnapshotError("a saved number is not finite")
    return parsed


# ---------------------------------------------------------------- to data

def _match_to_data(match_id: MatchId) -> dict:
    return {"sport": match_id.sport, "league": match_id.league, "kickoff": _dt(match_id.kickoff_utc),
            "home": match_id.home, "away": match_id.away}


def evaluation_to_data(ev: Evaluation) -> dict:
    return {"match": _match_to_data(ev.match_id), "market": ev.market, "selection": ev.selection, "odds": str(ev.odds),
            "estimate": {"probability": ev.estimate.probability, "push": ev.estimate.push, "model": ev.estimate.model,
                         "inputs": [{"name": i.name, "value": i.value, "unit": i.unit, "source": i.source}
                                    for i in ev.estimate.inputs]},
            "ev": str(ev.ev), "passed": ev.passed, "reason": ev.reason, "rule": ev.rule, "original": ev.original,
            "notes": list(ev.notes), "insufficient": ev.insufficient}


def _leg_to_data(leg: LoadedLeg) -> dict:
    return {"key": list(leg.leg_key), "match": leg.match_text, "market": leg.market_name, "outcome": leg.outcome_name,
            "odds": str(leg.odds), "selection": leg.selection, "evaluation_odds": _dec(leg.evaluation_odds)}


def session_to_data(session: Session) -> dict:
    last = None if session.last is None else [session.last[0], session.last[1].value, session.last[2]]
    summary = None
    if session.summary is not None:
        s = session.summary
        summary = {"session_id": s.session_id, "legs": [_leg_to_data(l) for l in s.legs],
                   "total_odds": str(s.total_odds), "stake": str(s.stake), "expires_at": _dt(s.expires_at),
                   "booking": bool(s.booking)}
    loaded = None
    if session.loaded is not None:
        loaded = {"legs": [_leg_to_data(l) for l in session.loaded.legs], "total_odds": str(session.loaded.total_odds)}
    return {
        "snapshot": SNAPSHOT_VERSION,
        "session_id": session.session_id, "state": session.state.value, "state_version": session.state_version,
        "origin": session.origin,
        "picks": [{"pick_id": p.pick_id, "evaluation": evaluation_to_data(p.evaluation)} for p in session.slip.picks],
        "menu": [{"pick_id": p.pick_id, "evaluation": evaluation_to_data(p.evaluation)} for p in session.menu],
        "chosen": list(session.chosen), "removed_games": list(session.removed_games),
        "last": last, "created_at": _dt(session.created_at), "earliest_kickoff": _dt(session.earliest_kickoff),
        "issues": [{"pick_id": i.pick_id, "kind": i.kind, "detail": i.detail,
                    "alternative": None if i.alternative is None else evaluation_to_data(i.alternative)}
                   for i in session.issues],
        "loaded": loaded, "odds_shown": session.odds_shown, "stake": _dec(session.stake), "summary": summary,
        "token_hash": session.token_hash, "message": session.message,
        "overrides": [list(o) for o in session.overrides], "override_summary": session.override_summary,
        "retries_used": session.retries_used, "retryable": session.retryable,
        "history_checked": session.history_checked, "retry_stake": _dec(session.retry_stake),
        "retry_override": session.retry_override, "unknown_at": _dt(session.unknown_at),
        "check_text": session.check_text, "booking": session.booking,
        "failed_picks": [{"pick_id": f.pick_id, "reason": f.reason, "times": f.times} for f in session.failed_picks],
        "dropped": [{"pick_id": d.pick_id, "label": d.label, "reason": d.reason} for d in session.dropped],
    }


def to_json(session: Session) -> str:
    """Deterministic: the same session always gives the same text."""
    return json.dumps(session_to_data(session), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


# ---------------------------------------------------------------- from data

def _evaluation_from_data(data: dict) -> Evaluation:
    m, e = data["match"], data["estimate"]
    match_id = MatchId(m["sport"], m["league"], _parse_dt(m["kickoff"]), m["home"], m["away"])
    inputs = tuple(Input(i["name"], float(i["value"]), i["unit"], i["source"]) for i in e["inputs"])
    estimate = Estimate(float(e["probability"]), inputs, e["model"], float(e["push"]))
    return Evaluation(match_id, data["market"], data["selection"], _parse_dec(data["odds"]), estimate,
                      _parse_dec(data["ev"]), bool(data["passed"]), data["reason"], data["rule"], data["original"],
                      tuple(data["notes"]), bool(data["insufficient"]))


def _leg_from_data(data: dict) -> LoadedLeg:
    key = tuple(data["key"])
    if len(key) != 4 or not all(isinstance(k, str) for k in key):
        raise SnapshotError("a saved leg key is malformed")
    return LoadedLeg(key, data["match"], data["market"], data["outcome"], _parse_dec(data["odds"]), data["selection"],
                     _parse_dec(data["evaluation_odds"]))


def session_from_data(data: dict) -> Session:
    try:
        if data["snapshot"] != SNAPSHOT_VERSION:
            raise SnapshotError("a saved session has an unknown format")
        def _read_picks(rows) -> list[Pick]:
            out = []
            for row in rows:
                evaluation = _evaluation_from_data(row["evaluation"])
                if row["pick_id"] != make_pick_id(evaluation):
                    raise SnapshotError("a saved pick does not match its id")
                if is_under(evaluation):   # AD-20: a saved Under (from before the rule) is never restored
                    continue
                if is_whole_or_quarter_line(evaluation):   # AD-21: a saved whole-line play is never restored either
                    continue
                out.append(Pick(row["pick_id"], evaluation))
            return out

        picks = _read_picks(data["picks"])
        # Older snapshots have no menu, chosen or removed_games: every game is kept with its advised option chosen.
        menu = _read_picks(data["menu"]) if "menu" in data else []
        chosen = tuple(str(c) for c in data.get("chosen", []))
        removed_games = tuple(str(k) for k in data.get("removed_games", []))
        if menu and not {p.pick_id for p in picks} <= {p.pick_id for p in menu}:
            raise SnapshotError("a saved pick is not one of the saved options")
        last = None
        if data["last"] is not None:
            try:
                last = (int(data["last"][0]), Action(data["last"][1]), str(data["last"][2]))
            except ValueError:
                last = None  # an action that no longer exists (the old Open and option-select buttons)
        def _alternative(row):
            if row is None:
                return None
            found = _evaluation_from_data(row)
            return None if is_under(found) or is_whole_or_quarter_line(found) else found   # AD-20, AD-21

        issues = tuple(Issue(i["pick_id"], i["kind"], i["detail"], _alternative(i["alternative"]))
                       for i in data["issues"])
        loaded = None
        if data["loaded"] is not None:
            loaded = LoadedSlip(tuple(_leg_from_data(l) for l in data["loaded"]["legs"]),
                                _parse_dec(data["loaded"]["total_odds"]))
        summary = None
        if data["summary"] is not None:
            s = data["summary"]
            summary = Summary(s["session_id"], tuple(_leg_from_data(l) for l in s["legs"]),
                              _parse_dec(s["total_odds"]), _parse_dec(s["stake"]), _parse_dt(s["expires_at"]),
                              bool(s.get("booking", False)))   # older snapshots have no flag: a placing summary
        session = Session(
            session_id=str(data["session_id"]), state=State(data["state"]), state_version=int(data["state_version"]),
            slip=Slip(tuple(picks)), menu=tuple(menu), chosen=chosen, removed_games=removed_games, last=last, created_at=_parse_dt(data["created_at"]),
            earliest_kickoff=_parse_dt(data["earliest_kickoff"]), issues=issues, loaded=loaded,
            odds_shown=bool(data["odds_shown"]), stake=_parse_dec(data["stake"]), summary=summary,
            token_hash=str(data["token_hash"]), message=str(data["message"]), origin=str(data["origin"]),
            overrides=tuple(tuple(str(p) for p in o) for o in data["overrides"]),
            override_summary=str(data["override_summary"]),
            # Older snapshots have none of the retry fields: no retries used, nothing retryable, nothing checked.
            retries_used=max(0, int(data.get("retries_used", 0))), retryable=bool(data.get("retryable", False)),
            history_checked=bool(data.get("history_checked", False)),
            retry_stake=_parse_dec(data.get("retry_stake")), retry_override=bool(data.get("retry_override", False)),
            unknown_at=_parse_dt(data.get("unknown_at")), check_text=str(data.get("check_text", "")),
            booking=bool(data.get("booking", False)),
            # Older snapshots have no partial-load fields: nothing failed, nothing was left out (AD-26).
            failed_picks=tuple(PickFailure(str(f["pick_id"]), str(f["reason"]), max(1, int(f.get("times", 1))))
                               for f in data.get("failed_picks", [])),
            dropped=tuple(DroppedPick(str(d["pick_id"]), str(d["label"]), str(d["reason"]))
                          for d in data.get("dropped", [])))
        if session.state in LIST_STATES:
            session = session.with_decisions()  # the Slip is always exactly the chosen option of each kept game
        return session
    except SnapshotError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as error:
        raise SnapshotError(f"a saved session could not be read ({type(error).__name__})") from None


def from_json(text: str) -> Session:
    try:
        data = json.loads(text)
    except (TypeError, ValueError):
        raise SnapshotError("a saved session is not valid JSON") from None
    if not isinstance(data, dict):
        raise SnapshotError("a saved session is not an object")
    return session_from_data(data)
