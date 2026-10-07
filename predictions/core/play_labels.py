"""Readable wording for a soccer play (spec 11-1). Standard library only, display only: the evaluation's own market and
selection text (what grading, the loader and the saved rows use) is never changed.

    "Over/Under 2.5" + "Over 2.5"        -> "Over 2.5 goals"        "Double Chance" + "Home or Draw" -> "Home or Draw"
    "1X2" + "Home" / "Away"              -> "Home win" / "Away win"  "Corners 9.5" + "Corners Over 9.5" -> "Over 9.5 corners"
    "Cards 3.5" + "Cards Over 3.5"       -> "Over 3.5 yellow cards"

Any other market, any other sport and anything unrecognised keeps its own selection text.
"""
from __future__ import annotations

import re

_LINE = r"(\d+(?:\.\d+)?)"
_GOALS = re.compile(rf"^Over/Under {_LINE}$")
_OVER = re.compile(rf"^Over {_LINE}$")
_CORNERS = re.compile(rf"^Corners {_LINE}$")
_CORNERS_SEL = re.compile(rf"^Corners Over {_LINE}$")
_CARDS = re.compile(rf"^Cards {_LINE}$")
_CARDS_SEL = re.compile(rf"^Cards Over {_LINE}$")
_WINNER_WORDS = {"Home": "Home win", "Away": "Away win"}


def play_label(evaluation) -> str:
    """The text the owner reads for one play. `evaluation` is an Evaluation (or anything with match_id, market, selection)."""
    market, selection = str(evaluation.market).strip(), str(evaluation.selection).strip()
    if getattr(evaluation.match_id, "sport", "") != "soccer":
        return selection
    if market == "1X2":
        return _WINNER_WORDS.get(selection, selection)
    found, chosen = _GOALS.match(market), _OVER.match(selection)
    if found and chosen and found.group(1) == chosen.group(1):
        return f"Over {chosen.group(1)} goals"
    found, chosen = _CORNERS.match(market), _CORNERS_SEL.match(selection)
    if found and chosen and found.group(1) == chosen.group(1):
        return f"Over {chosen.group(1)} corners"
    found, chosen = _CARDS.match(market), _CARDS_SEL.match(selection)
    if found and chosen and found.group(1) == chosen.group(1):
        return f"Over {chosen.group(1)} yellow cards"
    return selection
