"""Test helpers: what the owner sees on the interactive list for a gate's passing evaluations.

Since the list became the whole reply when something passes (the fixture listing, the evaluated-games lines and the
rejections are only shown with NO VALID PLAY), tests that used to read the plays out of the reply read them here.
"""
from datetime import datetime, timezone

from predictions.core.model import signed_percent
from predictions.core.types import Owner
from predictions.flow.session import SessionManager
from predictions.telegram import slate

NOW = datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc)


def started(evaluations, origin="matchday", clock=None):
    manager = SessionManager(clock=clock or (lambda: NOW))
    manager.start(list(evaluations), "abcd1234", origin)
    return manager


def list_text(evaluations, origin="matchday") -> str:
    """The main list text (every page), compact: games with their advised plays."""
    manager = started(evaluations, origin)
    session = manager.active
    pages = slate.page_count(session)
    return "\n".join(slate.list_screen(session, page).text for page in range(pages))


def detail_of(pick) -> str:
    """Test-only: the full record of one play (market, selection, estimate, rule, notes, every input and its source).
    The owner no longer has an Open screen for this (the list is Keep/Delete only); the tests still check that the
    evaluations carry it."""
    ev, est = pick.evaluation, pick.evaluation.estimate
    lines = [f"Market: {ev.market}", f"Selection: {ev.selection} @ {ev.odds:.2f}", f"Estimate: {slate._success(pick)}",
             f"Expected value: {signed_percent(ev.ev)}"]
    if ev.original:
        lines.append(f"Original pick: {ev.original}")
    if ev.rule:
        lines.append(f"Rule applied: {ev.rule}")
    if est.model:
        lines.append(f"Model: {est.model}")
    lines += list(ev.notes)
    lines += [f"{i.name}: {i.value:g} ({i.unit}) from {i.source}" for i in est.inputs]
    return "\n".join(lines)


def detail_text(evaluations, origin="matchday") -> str:
    """The records of every offered play of a gate's evaluations."""
    manager = started(evaluations, origin)
    return "\n".join(detail_of(pick) for pick in manager.active.menu)
