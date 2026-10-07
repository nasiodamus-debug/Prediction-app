"""AD-25: what the owner sees after a booking code was generated, the two answer buttons, and the Mybets list.

The BOOKED message is ONE message. Its first line is `Booking code: <code>` (the code in monospace when the sender uses HTML), then
`Total odds: n.nn`, then the games with picks and odds one per line, then the instruction. It never says 'placed' of anything the
bot did: nothing was placed. Nothing here imports the telegram library.
"""
from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from decimal import Decimal

from predictions.core.placement import PlacementToken, booking_lines
from predictions.core.staking import format_money
from predictions.flow.booking import ACT_NOT_PLACED, ACT_PLACED, ACT_SKIP
from predictions.flow.store import P_BOOKED, P_PLACED, P_PLACED_BY_OWNER, P_UNKNOWN
from predictions.telegram.render import MESSAGE_LIMIT, clean_text
from predictions.telegram.slate import Screen

BOOKING_PREFIX = "b"
_DATA = re.compile(r"^b:([0-9a-f]{12}):([a-z])$")
KEY_LENGTH = 12
PLACED_LABEL = "I placed it"
NOT_PLACED_LABEL = "I did not place it"
SKIP_LABEL = "Skip"
INSTRUCTION = ("Open SportyBet, enter the code, choose your stake and press Place Bet yourself. "
               "Nothing was placed by the bot.")


def encode_booking(token_hash: str, action: str) -> str:
    data = f"{BOOKING_PREFIX}:{token_hash[:KEY_LENGTH]}:{action}"
    if len(data.encode("utf-8")) > 64:
        raise ValueError("callback data is longer than Telegram allows")
    return data


def decode_booking(data) -> tuple[str, str] | None:
    match = _DATA.match(data) if isinstance(data, str) else None
    if match is None or match.group(2) not in (ACT_PLACED, ACT_NOT_PLACED, ACT_SKIP):
        return None
    return match.group(1), match.group(2)


def answer_buttons(token_hash: str) -> tuple[tuple[tuple[str, str], ...], ...]:
    return (((PLACED_LABEL, encode_booking(token_hash, ACT_PLACED)),
             (NOT_PLACED_LABEL, encode_booking(token_hash, ACT_NOT_PLACED))),)


def skip_buttons(token_hash: str) -> tuple[tuple[tuple[str, str], ...], ...]:
    return (((SKIP_LABEL, encode_booking(token_hash, ACT_SKIP)),),)


def _game_lines(legs) -> list[str]:
    return [f"{number}. {clean_text(leg.match_text)}: {clean_text(leg.selection or leg.outcome_name)} "
            f"({clean_text(leg.market_name, 40)}) @ {leg.odds:.2f}" for number, leg in enumerate(legs, start=1)]


def booked_message(token: PlacementToken, code: str) -> tuple[str, str]:
    """(plain text, HTML text) of the one BOOKED message. The HTML differs only by wrapping the code in <code> and escaping."""
    head = booking_lines(code, token.total_odds)           # ['Booking code: X', 'Total odds: n.nn']
    games = _game_lines(token.legs)
    body = [head[1], "", *games, "", INSTRUCTION]
    plain = "\n".join([head[0], *body])
    html_text = "\n".join([f"Booking code: <code>{html.escape(code)}</code>", *(html.escape(line) for line in body)])
    if len(plain) > MESSAGE_LIMIT:                         # a very long slip: the games yield, the code and odds never do
        keep = MESSAGE_LIMIT - len("\n".join([head[0], *head[1:], "", INSTRUCTION])) - 40
        shown, used = [], 0
        for line in games:
            if used + len(line) + 1 > keep:
                break
            shown.append(line)
            used += len(line) + 1
        games = shown + [f"and {len(token.legs) - len(shown)} more"]
        body = [head[1], "", *games, "", INSTRUCTION]
        plain = "\n".join([head[0], *body])
        html_text = "\n".join([f"Booking code: <code>{html.escape(code)}</code>", *(html.escape(line) for line in body)])
    return plain, html_text


# ---------- Mybets ----------

STATUS_WORDS = {P_BOOKED: "BOOKED (nothing placed by the bot)", P_PLACED_BY_OWNER: "PLACED BY YOU", P_PLACED: "PLACED",
                P_UNKNOWN: "UNKNOWN (a bet may exist)"}


def _stake_words(row: dict) -> str:
    if row["state"] == P_BOOKED:
        return "none (you choose it in SportyBet)"
    if row["state"] == P_PLACED_BY_OWNER:
        return f"NGN {format_money(row['owner_stake'])}" if row.get("owner_stake") else "not recorded"
    try:
        return f"NGN {format_money(Decimal(row['stake']))}"
    except Exception:
        return "unknown"


def record_text(row: dict) -> str:
    try:
        games = ", ".join(clean_text(l["match"], 40) for l in json.loads(row["legs"]))
    except Exception:
        games = "(games not readable)"
    try:
        odds = f"{Decimal(row['total_odds']):.2f}"
    except Exception:
        odds = "?"
    code = clean_text(row["booking_code"] or "-", 12)
    when = clean_text(str(row["created_at_utc"])[:16].replace("T", " "), 20)
    return "\n".join([f"{STATUS_WORDS.get(row['state'], row['state'])}",
                      f"Booking code: {code}", f"Total odds: {odds}", f"Stake: {_stake_words(row)}",
                      f"{when} UTC | {clean_text(games, 300)}"])


def mybets_screens(rows: list[dict]) -> list[Screen]:
    """One header, one message (with the two buttons) per still-BOOKED row, and one message for all the other records."""
    if not rows:
        return [Screen("No booking codes or placed bets are recorded yet.")]
    screens = [Screen(f"Your {len(rows)} most recent records (newest first). A BOOKED record means a booking code was made and "
                      "nothing was placed by the bot: tell me if you placed it.")]
    others = []
    for row in rows:
        if row["state"] == P_BOOKED:
            screens.append(Screen(record_text(row), answer_buttons(row["token_hash"])))
        else:
            others.append(record_text(row))
    if others:
        screens.append(Screen("\n\n".join(others)[:MESSAGE_LIMIT - 1]))
    return screens
