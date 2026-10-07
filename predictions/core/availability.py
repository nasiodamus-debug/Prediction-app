"""Is a listed game actually OPEN FOR BETTING on SportyBet right now (spec-12-1)? Standard library only.

A game is analysed (stats, analyst) only when it passes this check, so a game SportyBet shows but will not take a bet on
costs nothing. The fixture listing already did the hard part: the SportyBet adapter keeps only markets whose status is
open (not suspended, not banned) and outcomes that are active, with a real price. So here a game is open when it has not
started, its kickoff is in the future, and at least one of its remaining markets is one its sport's engine prices
(`engine.understands`). No new network call is made. Corners and cards (soccer) are not part of the listing; Matchday
looks them up per game afterwards, and that lookup is untouched.
"""
from __future__ import annotations

from datetime import datetime

from predictions.core.engines import ENGINES
from predictions.core.types import Fixture

NOT_OPEN_REASON = "not open for betting on SportyBet"


def is_open_for_betting(fixture: Fixture, now: datetime) -> bool:
    if fixture.started or fixture.kickoff_utc <= now:
        return False
    engine = ENGINES.get(fixture.match_id.sport)
    if engine is None:
        return False
    return any(engine.understands(offer) for offer in fixture.markets)


def not_open_line(count: int) -> str:
    return f"{count} game(s) skipped: {NOT_OPEN_REASON}."
