"""Settle ONE leg from final scores, for the Monday post-game review's DIAGNOSIS only. The bookmaker decides every official result
(AD-13); nothing here grades a bet or touches the betslip. Standard library only.

A leg is identified by what the placed-leg registry stores (sport, market_id, specifier, side), never by the market text. The
`hcp` in a specifier is the HOME handicap, so the away line is its negative. A market this module cannot settle from the data
given (corners, cards, tennis sets, event-time markets, missing period scores) is UNVERIFIABLE: it is shown as such and never
guessed, never counted as a loss.

Score conventions (spec: docs/weekly-review-design.md section 2): soccer markets use the 90-minute score; basketball, American
football, baseball and hockey winner / handicap / total markets include overtime, extra innings and the shootout; first-half markets
use the first two periods only; the MLB first-five-innings markets use innings 1 to 5.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

WIN, LOSS, PUSH, VOID = "win", "loss", "push", "void"
HALF_WIN, HALF_LOSS = "half win", "half loss"
UNVERIFIABLE = "unverifiable"

LOST = frozenset({LOSS, HALF_LOSS})
KEPT = frozenset({WIN, HALF_WIN})


@dataclass(frozen=True)
class FinalScore:
    """What a score source returned for one game."""
    home: int                       # the score a full-game market uses (soccer: the 90-minute score; others: including overtime)
    away: int
    status: str = "final"           # final | postponed | abandoned | suspended
    periods: tuple[tuple[int, int], ...] = ()   # (home, away) per quarter / half / period / inning, in order
    regulation: tuple[int, int] | None = None   # score after regulation when home/away include overtime or a shootout (NHL)
    extra_time: bool = False        # soccer: the game went to extra time or penalties (home/away may then not be the 90-minute score)


@dataclass(frozen=True)
class Settlement:
    verdict: str
    reason: str

    @property
    def lost(self) -> bool:
        return self.verdict in LOST


def _unverifiable(why: str) -> Settlement:
    return Settlement(UNVERIFIABLE, why)


def parse_specifier(specifier: str) -> dict[str, str]:
    """'total=8.5|periodnr=1' -> {'total': '8.5', 'periodnr': '1'}."""
    out = {}
    for part in str(specifier or "").split("|"):
        key, _, value = part.partition("=")
        if key.strip():
            out[key.strip()] = value.strip()
    return out


def _line(spec: dict[str, str], key: str) -> Decimal | None:
    try:
        value = Decimal(spec[key])
    except (KeyError, InvalidOperation):
        return None
    return value if value.is_finite() else None


def _single(value: Decimal) -> str:
    return WIN if value > 0 else PUSH if value == 0 else LOSS


def _combine(results: list[str]) -> str:
    if len(results) == 1:
        return results[0]
    first, second = sorted(results)
    pair = {first, second}
    if pair == {WIN} or pair == {LOSS} or pair == {PUSH}:
        return first
    if pair == {WIN, PUSH}:
        return HALF_WIN
    if pair == {LOSS, PUSH}:
        return HALF_LOSS
    return PUSH   # a win and a loss cannot meet on adjacent quarter lines; never reached


def _cover(margin: float, line: Decimal) -> str:
    """The bet wins when margin + line is above 0 (a handicap, from the bet side's point of view) and pushes at exactly 0."""
    if (line * 2) % 1 == 0:
        return _single(Decimal(str(margin)) + line)
    quarter = Decimal("0.25")
    return _combine([_single(Decimal(str(margin)) + line - quarter), _single(Decimal(str(margin)) + line + quarter)])


def _over_under(total: float, line: Decimal, side: str) -> str | None:
    if side == "over":
        return _cover(total, -line)
    if side == "under":
        return _cover(-total, line)
    return None


def _describe(verdict: str, text: str) -> Settlement:
    return Settlement(verdict, text)


# ---------------------------------------------------------------- the market tables (market ids as SportyBet lists them)
SOCCER_1X2, SOCCER_DC, SOCCER_TOTAL, SOCCER_AH, SOCCER_HOME_TOTAL, SOCCER_AWAY_TOTAL = "1", "10", "18", "16", "19", "20"
US_WINNER, US_HANDICAP, US_TOTAL, US_HOME_TOTAL, US_AWAY_TOTAL, US_FIRST_HALF_TOTAL = "219", "223", "225", "227", "228", "68"
BASEBALL_WINNER, BASEBALL_HANDICAP, BASEBALL_TOTAL = "251", "256", "258"
BASEBALL_HOME_TOTAL, BASEBALL_AWAY_TOTAL = "260", "261"
BASEBALL_F5_RESULT, BASEBALL_F5_HANDICAP, BASEBALL_F5_TOTAL = "274", "275", "276"
HOCKEY_WINNER, HOCKEY_HANDICAP, HOCKEY_TOTAL = "406", "410", "412"


def settle_leg(sport: str, market_id: str, specifier: str, side: str, score: FinalScore | None) -> Settlement:
    """The verdict for one placed leg given the game's final score (None when no score could be found)."""
    side = str(side).strip().lower()
    if score is None:
        return _unverifiable("no final score was found for this game")
    if score.status in ("postponed", "abandoned"):
        return Settlement(VOID, f"the game was {score.status}: the stake is refunded")
    if score.status != "final":
        return _unverifiable(f"the game status is {score.status!r}")
    spec = parse_specifier(specifier)
    market_id = str(market_id)
    if sport == "soccer":
        return _soccer(market_id, spec, side, score)
    if sport in ("basketball", "american_football"):
        return _us_style(sport, market_id, spec, side, score)
    if sport == "baseball":
        return _baseball(market_id, spec, side, score)
    if sport == "hockey":
        return _hockey(market_id, spec, side, score)
    return _unverifiable(f"no settlement rule for {sport}")


def _soccer(market_id: str, spec: dict, side: str, score: FinalScore) -> Settlement:
    if score.regulation is not None:
        home, away = score.regulation
    elif score.extra_time:
        return _unverifiable("the game went to extra time or penalties and the 90-minute score is not known")
    else:
        home, away = score.home, score.away
    shown = f"{home}-{away}"
    if market_id == SOCCER_1X2:
        actual = "home" if home > away else "away" if home < away else "draw"
        return _describe(WIN if side == actual else LOSS, f"{shown} (90 min): {actual}") if side in ("home", "draw", "away") \
            else _unverifiable(f"unknown 1X2 side {side!r}")
    if market_id == SOCCER_DC:
        wins = {"home or draw": home >= away, "draw or away": away >= home, "home or away": home != away}
        if side not in wins:
            return _unverifiable(f"unknown double chance side {side!r}")
        return _describe(WIN if wins[side] else LOSS, f"{shown} (90 min)")
    if market_id in (SOCCER_TOTAL, SOCCER_HOME_TOTAL, SOCCER_AWAY_TOTAL):
        line = _line(spec, "total")
        if line is None:
            return _unverifiable("the line could not be read")
        count = {SOCCER_TOTAL: home + away, SOCCER_HOME_TOTAL: home, SOCCER_AWAY_TOTAL: away}[market_id]
        verdict = _over_under(count, line, side)
        return _unverifiable(f"unknown side {side!r}") if verdict is None else _describe(verdict, f"{shown} (90 min): {count} vs {line}")
    if market_id == SOCCER_AH:
        hcp = _line(spec, "hcp")
        if hcp is None or side not in ("home", "away"):
            return _unverifiable("the handicap could not be read")
        margin = (home - away) if side == "home" else (away - home)
        return _describe(_cover(margin, hcp if side == "home" else -hcp), f"{shown} (90 min) with handicap {hcp:+} for the home side")
    return _unverifiable("this soccer market needs data a final score does not give (corners, cards, halves or goal times)")


def _first_half(score: FinalScore) -> tuple[int, int] | None:
    if len(score.periods) < 2:
        return None
    return (score.periods[0][0] + score.periods[1][0], score.periods[0][1] + score.periods[1][1])


def _us_style(sport: str, market_id: str, spec: dict, side: str, score: FinalScore) -> Settlement:
    home, away = score.home, score.away
    shown = f"{home}-{away}"
    if market_id == US_WINNER:
        if home == away:
            return _unverifiable(f"{shown} is a tie: how SportyBet settles a tied winner market is not known")
        actual = "home" if home > away else "away"
        return _describe(WIN if side == actual else LOSS, f"{shown} (incl. overtime): {actual}") if side in ("home", "away") \
            else _unverifiable(f"unknown winner side {side!r}")
    if market_id == US_HANDICAP:
        hcp = _line(spec, "hcp")
        if hcp is None or side not in ("home", "away"):
            return _unverifiable("the handicap could not be read")
        margin = (home - away) if side == "home" else (away - home)
        return _describe(_cover(margin, hcp if side == "home" else -hcp), f"{shown} (incl. overtime) with handicap {hcp:+} for the home side")
    if market_id in (US_TOTAL, US_HOME_TOTAL, US_AWAY_TOTAL):
        line = _line(spec, "total")
        if line is None:
            return _unverifiable("the line could not be read")
        count = {US_TOTAL: home + away, US_HOME_TOTAL: home, US_AWAY_TOTAL: away}[market_id]
        verdict = _over_under(count, line, side)
        return _unverifiable(f"unknown side {side!r}") if verdict is None else _describe(verdict, f"{shown} (incl. overtime): {count} vs {line}")
    if market_id == US_FIRST_HALF_TOTAL:
        line = _line(spec, "total")
        half = _first_half(score)
        if line is None:
            return _unverifiable("the line could not be read")
        if half is None:
            return _unverifiable("no quarter scores were found, so the first half cannot be settled")
        verdict = _over_under(half[0] + half[1], line, side)
        return _unverifiable(f"unknown side {side!r}") if verdict is None else _describe(
            verdict, f"first half {half[0]}-{half[1]}: {half[0] + half[1]} vs {line}")
    return _unverifiable(f"this {sport.replace('_', ' ')} market is not settled from final scores")


def _baseball(market_id: str, spec: dict, side: str, score: FinalScore) -> Settlement:
    home, away = score.home, score.away
    shown = f"{home}-{away}"
    if market_id == BASEBALL_WINNER:
        if home == away:
            return _unverifiable(f"{shown} is a tie")
        actual = "home" if home > away else "away"
        return _describe(WIN if side == actual else LOSS, f"{shown} (incl. extra innings): {actual}") if side in ("home", "away") \
            else _unverifiable(f"unknown winner side {side!r}")
    if market_id == BASEBALL_HANDICAP:
        hcp = _line(spec, "hcp")
        if hcp is None or side not in ("home", "away"):
            return _unverifiable("the run line could not be read")
        margin = (home - away) if side == "home" else (away - home)
        return _describe(_cover(margin, hcp if side == "home" else -hcp), f"{shown} with run line {hcp:+} for the home side")
    if market_id in (BASEBALL_TOTAL, BASEBALL_HOME_TOTAL, BASEBALL_AWAY_TOTAL):
        line = _line(spec, "total")
        if line is None:
            return _unverifiable("the line could not be read")
        count = {BASEBALL_TOTAL: home + away, BASEBALL_HOME_TOTAL: home, BASEBALL_AWAY_TOTAL: away}[market_id]
        verdict = _over_under(count, line, side)
        return _unverifiable(f"unknown side {side!r}") if verdict is None else _describe(verdict, f"{shown}: {count} vs {line}")
    if market_id in (BASEBALL_F5_RESULT, BASEBALL_F5_HANDICAP, BASEBALL_F5_TOTAL):
        if not score.periods:
            return _unverifiable("no inning scores were found, so the first five innings cannot be settled")
        if len(score.periods) < 5:
            return Settlement(VOID, "the game ended before five innings: the first-five-innings bet is void")
        h5 = sum(p[0] for p in score.periods[:5])
        a5 = sum(p[1] for p in score.periods[:5])
        shown5 = f"first five innings {h5}-{a5}"
        if market_id == BASEBALL_F5_RESULT:
            actual = "home" if h5 > a5 else "away" if h5 < a5 else "draw"
            return _describe(WIN if side == actual else LOSS, f"{shown5}: {actual}") if side in ("home", "draw", "away") \
                else _unverifiable(f"unknown side {side!r}")
        if market_id == BASEBALL_F5_HANDICAP:
            hcp = _line(spec, "hcp")
            if hcp is None or side not in ("home", "away"):
                return _unverifiable("the handicap could not be read")
            margin = (h5 - a5) if side == "home" else (a5 - h5)
            return _describe(_cover(margin, hcp if side == "home" else -hcp), f"{shown5} with handicap {hcp:+} for the home side")
        line = _line(spec, "total")
        if line is None:
            return _unverifiable("the line could not be read")
        verdict = _over_under(h5 + a5, line, side)
        return _unverifiable(f"unknown side {side!r}") if verdict is None else _describe(verdict, f"{shown5}: {h5 + a5} vs {line}")
    return _unverifiable("this baseball market is not settled from final scores")


def _hockey(market_id: str, spec: dict, side: str, score: FinalScore) -> Settlement:
    home, away = score.home, score.away
    shown = f"{home}-{away}"
    if market_id == HOCKEY_WINNER:
        if home == away:
            return _unverifiable(f"{shown} is a tie")
        actual = "home" if home > away else "away"
        return _describe(WIN if side == actual else LOSS, f"{shown} (incl. overtime and shootout): {actual}") if side in ("home", "away") \
            else _unverifiable(f"unknown winner side {side!r}")
    if market_id == HOCKEY_HANDICAP:
        hcp = _line(spec, "hcp")
        if hcp is None or side not in ("home", "away"):
            return _unverifiable("the puck line could not be read")
        margin = (home - away) if side == "home" else (away - home)
        return _describe(_cover(margin, hcp if side == "home" else -hcp), f"{shown} with puck line {hcp:+} for the home side")
    if market_id == HOCKEY_TOTAL:
        line = _line(spec, "total")
        if line is None:
            return _unverifiable("the line could not be read")
        verdict = _over_under(home + away, line, side)
        return _unverifiable(f"unknown side {side!r}") if verdict is None else _describe(verdict, f"{shown}: {home + away} vs {line}")
    return _unverifiable("this hockey market needs the regulation score or period scores and is not settled")
