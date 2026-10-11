"""OWNER-RUN walk-forward backtest of the soccer goals model AGAINST THE MARKET. READ ONLY: it reads football-data.co.uk's public
CSV files (results plus past bookmaker odds) and never touches SportyBet, Telegram, the database or any account.

Why it exists: those CSVs hold the closing odds of Pinnacle and the market average next to every result, so for the first time the model can be
judged against a price. For every game it rebuilds what the bot's goals model would have said from ONLY the team's earlier games, takes the
market's probabilities (the odds with the bookmaker's margin removed), and scores three things on the real results: the model alone, the
market alone, and blends of the two (`--weight` is the share given to the model). It then shows what happens when the model claims an edge.

HOW TO RUN (from the project folder):

    python -m predictions.tools.soccer_backtest
    python -m predictions.tools.soccer_backtest --league E0 --league D1 --season 2425 --season 2526

Lower log loss / Brier score is better. A season is read once and kept in local_data/backtest/soccer/ (the current season is always read again).

What it cannot tell you: the bot's xG model (Understat) is not rebuilt here (past xG per game is not in these files), so the MODEL side is the
bot's goals-only model on rolling goals, the weaker of the two; the market side is as good as it gets. SportyBet's own prices are not in the files.
"""
from __future__ import annotations

import argparse
import csv
import io
import math
import sys
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402

from predictions.core.engines import soccer as sc  # noqa: E402
from predictions.core.inputs import Input  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "local_data" / "backtest" / "soccer"
URL = "https://www.football-data.co.uk/mmz4281/{season}/{code}.csv"
DEFAULT_LEAGUES = ("E0", "E1", "D1", "SP1", "I1", "F1", "N1", "B1", "P1", "T1", "SC0")
DEFAULT_SEASONS = ("2223", "2324", "2425", "2526", "2627")
LAST_N = 10
MIN_PRIOR = sc.GOALS_ONLY_MIN_MATCHES
WEIGHTS = (1.0, 0.75, 0.5, 0.25, 0.0)   # share of the model in a blend: 1 = model alone, 0 = market alone
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
# Odds columns in the order tried: closing market average, closing Pinnacle, then the earlier average.
ODDS_1X2 = (("AvgCH", "AvgCD", "AvgCA"), ("PSCH", "PSCD", "PSCA"), ("AvgH", "AvgD", "AvgA"))
ODDS_OU = (("AvgC>2.5", "AvgC<2.5"), ("PC>2.5", "PC<2.5"), ("Avg>2.5", "Avg<2.5"))
EDGE_STEPS = (0.0, 0.03, 0.05, 0.08, 0.12)   # "the model claims at least this much more than the market"


@dataclass(frozen=True)
class Game:
    day: date
    home: str
    away: str
    hg: int
    ag: int
    market_1x2: tuple[float, float, float] | None     # fair probabilities (home, draw, away)
    market_over: float | None                          # fair P(over 2.5)


def _float(value) -> float | None:
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return number if number > 1.0 else None


def _fair(odds: Sequence[float | None]) -> tuple[float, ...] | None:
    """Probabilities with the bookmaker's margin taken out proportionally."""
    if any(o is None for o in odds):
        return None
    raw = [1.0 / o for o in odds]
    total = sum(raw)
    return tuple(r / total for r in raw)


def _first(row: dict, groups: Iterable[Sequence[str]]) -> tuple[float, ...] | None:
    for cols in groups:
        fair = _fair([_float(row.get(c)) for c in cols])
        if fair is not None:
            return fair
    return None


def parse_games(text: str) -> list[Game]:
    games = []
    for row in csv.DictReader(io.StringIO(text.lstrip("﻿"))):
        try:
            day = datetime.strptime(row["Date"].strip(), "%d/%m/%Y").date() if len(row["Date"].strip()) == 10 else \
                datetime.strptime(row["Date"].strip(), "%d/%m/%y").date()
            hg, ag = int(row["FTHG"]), int(row["FTAG"])
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        ou = _first(row, ODDS_OU)
        games.append(Game(day, row["HomeTeam"].strip(), row["AwayTeam"].strip(), hg, ag,
                          _first(row, ODDS_1X2), ou[0] if ou else None))
    return sorted(games, key=lambda g: g.day)


def season_text(code: str, season: str, client: httpx.Client, current: bool) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{season}_{code}.csv"
    if path.exists() and not current:
        return path.read_text(encoding="utf-8")
    response = client.get(URL.format(season=season, code=code))
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    return response.text


def model_probabilities(history: list[Game], home: str, away: str) -> tuple[tuple[float, float, float], float] | None:
    """What the bot's goals-only model says (P home/draw/away, P over 2.5) from the teams' earlier games only."""
    def side(team: str, label: str) -> list[Input] | None:
        mine = [g for g in history if team in (g.home, g.away)][-LAST_N:]
        if len(mine) < MIN_PRIOR:
            return None
        gf = sum(g.hg if g.home == team else g.ag for g in mine) / len(mine)
        ga = sum(g.ag if g.home == team else g.hg for g in mine) / len(mine)
        return [Input(f"{label}.matches", float(len(mine)), "matches", "backtest"),
                Input(f"{label}.goals_for", gf, "goals", "backtest"),
                Input(f"{label}.goals_against", ga, "goals", "backtest")]

    h, a = side(home, "home"), side(away, "away")
    if h is None or a is None:
        return None
    inputs = h + a
    lam_h, lam_a = sc.scaled_rates(inputs)
    return sc.result_probabilities(lam_h, lam_a), sc.probability_over(inputs, Decimal("2.5"))


@dataclass(frozen=True)
class Row:
    model_1x2: tuple[float, float, float]
    model_over: float
    market_1x2: tuple[float, float, float] | None
    market_over: float | None
    result: int            # 0 home, 1 draw, 2 away
    over: bool


def build_rows(leagues: Sequence[str], seasons: Sequence[str], client: httpx.Client) -> list[Row]:
    rows = []
    start = date.today().year if date.today().month >= 7 else date.today().year - 1
    current = f"{start % 100:02d}{(start + 1) % 100:02d}"
    for code in leagues:
        for season in seasons:
            try:
                games = parse_games(season_text(code, season, client, season == current))
            except httpx.HTTPError:
                continue
            for i, g in enumerate(games):
                prior = games[:i]
                priced = model_probabilities(prior, g.home, g.away)
                if priced is None:
                    continue
                result = 0 if g.hg > g.ag else 1 if g.hg == g.ag else 2
                rows.append(Row(priced[0], priced[1], g.market_1x2, g.market_over, result, g.hg + g.ag > 2.5))
    return rows


# ---------------------------------------------------------------- scores
def _blend(model: Sequence[float], market: Sequence[float], weight: float) -> list[float]:
    mixed = [weight * m + (1.0 - weight) * k for m, k in zip(model, market)]
    total = sum(mixed)
    return [x / total for x in mixed]


def log_loss(prob: Sequence[float], actual: int) -> float:
    return -math.log(max(prob[actual], 1e-9))


def brier(prob: Sequence[float], actual: int) -> float:
    return sum((p - (1.0 if i == actual else 0.0)) ** 2 for i, p in enumerate(prob))


def score_table(rows: Sequence[Row]) -> list[tuple[str, float, float, float, float, int]]:
    """(label, 1X2 log loss, 1X2 Brier, Over 2.5 log loss, Over 2.5 Brier, games) per blend weight, on games with BOTH prices."""
    result = []
    full = [r for r in rows if r.market_1x2 is not None and r.market_over is not None]
    for w in WEIGHTS:
        ll = br = lo = bo = 0.0
        for r in full:
            p = _blend(r.model_1x2, r.market_1x2, w)
            q = _blend((r.model_over, 1 - r.model_over), (r.market_over, 1 - r.market_over), w)
            ll += log_loss(p, r.result)
            br += brier(p, r.result)
            lo += log_loss(q, 0 if r.over else 1)
            bo += brier(q, 0 if r.over else 1)
        n = max(len(full), 1)
        label = "model alone" if w == 1.0 else "market alone" if w == 0.0 else f"{w:.0%} model + {1 - w:.0%} market"
        result.append((label, ll / n, br / n, lo / n, bo / n, len(full)))
    return result


def edge_table(rows: Sequence[Row]) -> list[tuple[float, int, float, float, float]]:
    """When the model rates a 1X2 outcome at least `edge` above the market: (edge, picks, hit rate, model's claim, market's claim)."""
    out = []
    for edge in EDGE_STEPS:
        picks = hits = 0
        claim = price = 0.0
        for r in rows:
            if r.market_1x2 is None:
                continue
            for i in range(3):
                if r.model_1x2[i] - r.market_1x2[i] >= max(edge, 1e-9) and r.model_1x2[i] >= 0.5:
                    picks += 1
                    hits += r.result == i
                    claim += r.model_1x2[i]
                    price += r.market_1x2[i]
        n = max(picks, 1)
        out.append((edge, picks, hits / n, claim / n, price / n))
    return out


def over_edge_table(rows: Sequence[Row]) -> list[tuple[float, int, float, float, float]]:
    out = []
    for edge in EDGE_STEPS:
        picks = hits = 0
        claim = price = 0.0
        for r in rows:
            if r.market_over is None:
                continue
            if r.model_over - r.market_over >= max(edge, 1e-9) and r.model_over >= 0.5:
                picks += 1
                hits += r.over
                claim += r.model_over
                price += r.market_over
        n = max(picks, 1)
        out.append((edge, picks, hits / n, claim / n, price / n))
    return out


def format_report(rows: Sequence[Row], title: str) -> str:
    lines = [title, f"games with a model price: {len(rows)}", ""]
    lines.append("Scores (lower is better), games with both prices:")
    lines.append(f"{'':<26}{'1X2 loss':>10}{'1X2 brier':>11}{'O2.5 loss':>11}{'O2.5 brier':>12}")
    table = score_table(rows)
    for label, ll, br, lo, bo, n in table:
        lines.append(f"{label:<26}{ll:>10.4f}{br:>11.4f}{lo:>11.4f}{bo:>12.4f}")
    if table:
        lines.append(f"({table[0][5]} games)")
    for name, data in (("1X2 outcome", edge_table(rows)), ("Over 2.5", over_edge_table(rows))):
        lines += ["", f"When the model rates a {name} at least X above the market (and at least 50%):",
                  f"{'edge':>8}{'picks':>8}{'hit rate':>10}{'model said':>12}{'market said':>13}"]
        for edge, picks, hit, claim, price in data:
            lines.append(f"{edge:>8.0%}{picks:>8}{hit:>10.1%}{claim:>12.1%}{price:>13.1%}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Soccer model against the market, walk-forward, read only.")
    parser.add_argument("--league", action="append", help="football-data code, e.g. E0 (default: the bot's leagues)")
    parser.add_argument("--season", action="append", help="e.g. 2425 (default: the last four seasons and this one)")
    args = parser.parse_args(argv)
    leagues = args.league or DEFAULT_LEAGUES
    seasons = args.season or DEFAULT_SEASONS
    with httpx.Client(timeout=30.0, headers=HEADERS, follow_redirects=True) as client:
        rows = build_rows(leagues, seasons, client)
    print(format_report(rows, f"Soccer goals model vs market: {', '.join(leagues)} / {', '.join(seasons)}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
