"""Market rules the owner fixed for every sport. Standard library only, a leaf module (it imports nothing of ours).

OWNER RULE: "In all sports no under predictions are allowed." An Under play is never produced, listed, offered, converted
to, kept, loaded onto the betslip or placed, in any sport, trigger or market (full-game, team, half, period, F5, whole
or Asian line, corners, cards). `is_under` is the one definition; every chokepoint calls it (AD-20).

OWNER RULE 2 ("leave out whole-line then"): a WHOLE-LINE or QUARTER-LINE play (any line that is not a half line x.5: a play that
can end in a refund or a half refund, such as Over 2.0, Over 2.25, -1.0, spread -3, total 44) is never produced, ranked, offered,
converted to, kept, loaded or placed, in any sport or trigger (Matchday, Gameday, America). `is_whole_or_quarter_line` is the one
definition and the chokepoints are the same as for the Under rule (AD-21). REVIEW is exempt: it judges whatever is on the booking
code, with the existing push maths, so Review's `evaluate_exact` is NOT filtered.
"""
from __future__ import annotations

import re
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from typing import Iterable

SIDES_PLAYED = ("over",)   # the over/under sides an engine may price: Under is never one of them
UNDER_REASON = "Under plays are not allowed"
RULE_NO_UNDER = "No Under plays"

# "Under" as a word ("Under 2.5", "Home team Under 1.5", "1st half Under 24.5", "Corners Under 9.5"), or the short form
# "U 2.5" / "U2.5" / "u 2.5" at the start of the text or after a team/half/period prefix. "Underdog" is not an Under.
_UNDER_WORD = re.compile(r"(?<![a-z0-9/])under(?![a-z])", re.IGNORECASE)
_UNDER_SHORT = re.compile(r"(?:^|(?<=\s))u\s*\d+(?:[.,]\d+)?\s*$", re.IGNORECASE)
_SHORT_PREFIX = re.compile(r"^\s*(?:(?:home|away)\s+(?:team\s+)?|1st\s+half\s+|2nd\s+half\s+|(?:1st|2nd|3rd)\s+period\s+|"
                           r"first\s+half\s+|corners\s+|cards\s+|f5\s+)*u\s*\d+(?:[.,]\d+)?\s*$", re.IGNORECASE)


def _text_is_under(text: str) -> bool:
    text = " ".join(str(text).replace("_", " ").split())
    return bool(_UNDER_WORD.search(text) or _SHORT_PREFIX.match(text))


def is_under(item) -> bool:
    """True when the selection is an Under. `item` is the selection text, or any object that carries one: an Evaluation
    (`selection`), a Pick (`evaluation`), a CodeSelection (`outcome_name`), a Leg or Priced (`side`), an Outcome
    (`name`). Fails safe: for an object, any of its readable fields saying Under counts."""
    if item is None:
        return False
    if isinstance(item, str):
        return _text_is_under(item)
    found = False
    for attribute in ("evaluation", "selection", "outcome_name", "name", "side"):
        value = getattr(item, attribute, None)
        if value is None:
            continue
        if isinstance(value, str):
            found = found or _text_is_under(value)
        elif attribute == "evaluation":
            found = found or is_under(value)
    return found


def reject_under(evaluation):
    """The same evaluation, refused for the owner's rule: never passed, never a pick (no estimate is touched)."""
    return replace(evaluation, passed=False, reason=UNDER_REASON, rule=RULE_NO_UNDER)


def without_unders(evaluations: Iterable) -> list:
    """The evaluations with every Under removed (the engines' last step before they hand picks on)."""
    return [e for e in evaluations if not is_under(e)]


# ---------------------------------------------------------------- AD-21: no whole-line or quarter-line plays
WHOLE_LINE_REASON = "Whole-line plays are not allowed"
RULE_NO_WHOLE_LINE = "No whole-line plays"
_LINE_KEYS = ("total", "hcp", "handicap", "line")
_NUMBER = re.compile(r"(?<![\w.])[+-]?\d+(?:[.,]\d+)?(?![\w.])")   # "2.0", "-1.5"; not "1st", "F5", "1X2", "04.5x"
_TEXT_ATTRIBUTES = ("selection", "outcome_name")
_PLAIN_TEXT_ATTRIBUTES = ("selection", "market", "outcome_name", "market_name", "name")


def _bad_number(text) -> bool:
    """True when this one number is not a half line x.5 (0 counts as not a half line)."""
    try:
        value = Decimal(str(text).strip().replace(",", "."))
    except (InvalidOperation, ValueError):
        return False
    if not value.is_finite():
        return True
    return abs(value) % 1 != Decimal("0.5")


def _specifier_is_whole(text: str) -> bool | None:
    """None when the text has no line key (`total=`, `hcp=`); else whether any of its lines is whole or quarter."""
    found = None
    for part in text.split("|"):
        key, sep, value = part.partition("=")
        if sep and key.strip().lower() in _LINE_KEYS:
            found = bool(found) or _bad_number(value)
    return found


def _free_text_is_whole(text: str) -> bool:
    text = " ".join(str(text).replace("_", " ").split())
    return any(_bad_number(token) for token in _NUMBER.findall(text))


def is_whole_or_quarter_line(item) -> bool:
    """True when the selection's line is NOT a half line x.5 (2, 2.0, 3.0, 2.25, 2.75, -1.0, 0 are all True; 2.5, -1.5 are not).
    `item` is a number, a specifier ('total=2', 'hcp=-1.0'), a selection text ('Over 2.0', 'Home (-1)'), or an object that
    carries one: an Evaluation (`selection`, `market`), a Pick (`evaluation`), a Leg / MarketOffer / CodeSelection (`specifier`,
    and `selection` or `outcome_name`) or an Outcome (`name`). Text with no number (a winner, a double chance) is not a line.
    Fails safe: for an object, any readable field saying whole or quarter counts."""
    if item is None or isinstance(item, bool):
        return False
    if isinstance(item, (int, float, Decimal)):
        return _bad_number(item)
    if isinstance(item, str):
        spec = _specifier_is_whole(item)
        if spec is not None:
            return spec
        stripped = item.strip()
        if re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", stripped):
            return _bad_number(stripped)
        return _free_text_is_whole(item)
    found = False
    inner = getattr(item, "evaluation", None)
    if inner is not None and not isinstance(inner, str):
        found = is_whole_or_quarter_line(inner)
    specifier = getattr(item, "specifier", None)
    if isinstance(specifier, str):
        spec = _specifier_is_whole(specifier)
        found = found or bool(spec)
        names = _TEXT_ATTRIBUTES   # the market name of a bookmaker may say "Innings 1 to 5": only the pick's own text is read
    else:
        names = _PLAIN_TEXT_ATTRIBUTES
    for attribute in names:
        value = getattr(item, attribute, None)
        if isinstance(value, str) and (attribute != "name" or specifier is None):
            found = found or _free_text_is_whole(value)
    return found


def reject_whole_line(evaluation):
    """The same evaluation, refused for the owner's rule: never passed, never a pick (no estimate is touched)."""
    return replace(evaluation, passed=False, reason=WHOLE_LINE_REASON, rule=RULE_NO_WHOLE_LINE)


def without_whole_lines(evaluations: Iterable) -> list:
    """The evaluations with every whole-line or quarter-line play removed (the engines' last step before they hand picks on)."""
    return [e for e in evaluations if not is_whole_or_quarter_line(e)]


def offers_without_whole_lines(offers: Iterable) -> list:
    """The market offers an engine may price for an offer list: any offer whose own line (specifier) is whole or quarter is not
    priced at all, so line shopping and every rule only ever see half lines (a winner or double chance has no line and stays)."""
    return [o for o in offers if not is_whole_or_quarter_line(getattr(o, "specifier", "") or "")]
