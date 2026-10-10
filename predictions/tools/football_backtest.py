"""OWNER-RUN walk-forward backtest of the American football model. READ ONLY: it reads public ESPN scoreboards and never touches
SportyBet, Telegram, the database or any account, and it places nothing.

Why it exists: the bot's own settled picks are a few dozen legs, far too few to judge the model (the first week: 54 legs, 72.8%
predicted, 53.7% won). ESPN already holds every completed game of a season, so this tool replays the season week by week. For each
week W it rebuilds, from ONLY the games of the weeks before W (the same window, inputs and model code as the bot uses), what the bot
would have expected, and compares that with what happened. It answers: how far off are the expected margins and totals, are the
standard deviations the model assumes about right, and when the model says 70% or 80% how often is it right?

HOW TO RUN (from the project folder):

    python -m predictions.tools.football_backtest --league college --year 2025
    python -m predictions.tools.football_backtest --league college --year 2025 --year 2026 --min-week 5
    python -m predictions.tools.football_backtest --league nfl --year 2025

A season is read once and kept in local_data/backtest/ (the current year is always read again). `--variant` picks the way the model
inputs are built: `production` is exactly what the bot does now; other variants are added here to be compared on the same games.

What it cannot tell you: ESPN gives scores, not past bookmaker odds. A model can be well calibrated and still lose to the bookmaker.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Callable, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx  # noqa: E402

from predictions.adapters.espn import football as fb  # noqa: E402
from predictions.core.engines import american_football as af  # noqa: E402
from predictions.core.engines.common import handicap_probabilities, line_probabilities  # noqa: E402
from predictions.core.inputs import Input  # noqa: E402
from predictions.core.ports import StatsFetchError  # noqa: E402
from predictions.core.ratings import adjusted_ratings, results_of  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "local_data" / "backtest"
SOURCE = "backtest"
MAX_WEEK = 18
# Lines are placed around the model's own expectation (so every probability from about 5% to 95% is tested), in whole-point steps
# from a half line, so no line can push.
LINE_OFFSETS = (-14, -10, -7, -3, 0, 3, 7, 10, 14)
BUCKETS = tuple((0.5 + 0.05 * i, 0.5 + 0.05 * (i + 1) if i < 9 else 1.0001) for i in range(10))   # 50-55% ... 95-100%


@dataclass(frozen=True)
class Game:
    week: int
    game: fb.PlayedGame


@dataclass(frozen=True)
class Event:
    kind: str        # winner, handicap, total, team_total
    p: float         # the model's probability that the event happens
    hit: bool        # whether it happened


@dataclass
class Prediction:
    week: int
    mu_margin: float
    mu_total: float
    mu_home: float
    mu_away: float
    margin: float
    total: float
    home_points: float
    away_points: float
    events: list[Event] = field(default_factory=list)
    mu_fh_total: float | None = None
    fh_total: float | None = None


# ---------------------------------------------------------------- reading a season
def fetch_games(league: str, year: int, client: httpx.Client, weeks: Iterable[int] = range(1, MAX_WEEK + 1)) -> list[Game]:
    """Every completed regular-season game of the season, with its week, from the same scoreboards the bot reads."""
    found: dict[str, Game] = {}
    for group in fb.GROUPS[league]:
        for week in weeks:
            params = {**fb.EXTRA_PARAMS[league], "dates": str(year), "seasontype": str(fb.REGULAR_SEASON), "week": str(week)}
            if group:
                params["groups"] = group
            response = client.get(fb.BASE_URL.format(slug=fb.SLUGS[league]) + "/scoreboard", params=params)
            if response.status_code == 404:
                continue
            response.raise_for_status()
            for played in _parse_page(response.json()):
                found.setdefault(played.game_id, Game(week, played))
    return sorted(found.values(), key=lambda g: (g.week, g.game.when, g.game.game_id))


def _parse_page(payload) -> list[fb.PlayedGame]:
    """The completed games of one scoreboard page. The bot's parser refuses a whole page for one malformed entry (fail closed);
    for a backtest over hundreds of pages one empty entry (seen in ESPN's 2025 FCS week 8) should only lose that entry."""
    try:
        return list(fb.parse_scoreboard(payload).played)
    except StatsFetchError:
        events = payload.get("events") if isinstance(payload, dict) else None
        if not isinstance(events, list):
            return []
        games: list[fb.PlayedGame] = []
        for event in events:
            try:
                games.extend(fb.parse_scoreboard({"events": [event]}).played)
            except StatsFetchError:
                continue
        return games


def _side_to_json(side: fb.Side) -> dict:
    return {"team_id": side.team_id, "name": side.name, "score": side.score, "quarters": list(side.quarters)}


def _side_from_json(data: dict) -> fb.Side:
    return fb.Side(str(data["team_id"]), str(data["name"]), int(data["score"]), tuple(float(q) for q in data["quarters"]))


def save_games(path: Path, games: Sequence[Game]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"week": g.week, "game_id": g.game.game_id, "when": g.game.when.isoformat(),
             "home": _side_to_json(g.game.home), "away": _side_to_json(g.game.away)} for g in games]
    path.write_text(json.dumps(rows), encoding="utf-8")


def load_games(path: Path) -> list[Game] | None:
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return [Game(int(r["week"]), fb.PlayedGame(str(r["game_id"]), datetime.fromisoformat(r["when"]),
                                                   _side_from_json(r["home"]), _side_from_json(r["away"]))) for r in rows]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def season_games(league: str, year: int, client: httpx.Client | None = None, refresh: bool = False,
                 cache_dir: Path = CACHE_DIR, today: datetime | None = None) -> list[Game]:
    """A season from the local copy when there is one (the current year is always read again), else from ESPN."""
    path = cache_dir / f"football-{league}-{year}.json"
    current = year >= (today or datetime.now(timezone.utc)).year
    if not refresh and not current:
        cached = load_games(path)
        if cached:
            return cached
    own = client is None
    client = client or httpx.Client(timeout=fb.TIMEOUT_SECONDS)
    try:
        games = fetch_games(league, year, client)
    finally:
        if own:
            client.close()
    if games:
        save_games(path, games)
    return games


# ---------------------------------------------------------------- how model inputs are built (the thing being compared)
Variant = Callable[[Sequence[fb.PlayedGame], str, str, str], "list[Input] | None"]


def raw_inputs(history: Sequence[fb.PlayedGame], home_id: str, away_id: str, league: str) -> list[Input] | None:
    """The bot's inputs before the opponent adjustment: each team's last games in the window and the league level."""
    home, away = fb.team_window(list(history), home_id), fb.team_window(list(history), away_id)
    needed = af.MIN_GAMES[league]
    if len(home) < needed or len(away) < needed:
        return None
    return [*fb.team_inputs("home", home, None, SOURCE), *fb.team_inputs("away", away, None, SOURCE),
            *fb.league_inputs(list(history), SOURCE)]


def production_inputs(history: Sequence[fb.PlayedGame], home_id: str, away_id: str, league: str) -> list[Input] | None:
    """Exactly what the bot builds now: the raw inputs plus, for college, the opponent-adjusted rates the engine uses for margins."""
    base = raw_inputs(history, home_id, away_id, league)
    if base is None:
        return None
    if league == af.COLLEGE:
        base = base + fb.adjusted_inputs(list(history), home_id, away_id, SOURCE)
    return base


def sos_variant(ridge: float) -> Variant:
    """Production inputs with each team's full-game points for and against replaced by opponent-adjusted ratings (core/ratings.py),
    found over every game of the window, so a team that beat weak opponents is not flattered. Everything else is unchanged."""
    def build(history: Sequence[fb.PlayedGame], home_id: str, away_id: str, league: str) -> list[Input] | None:
        base = raw_inputs(history, home_id, away_id, league)
        if base is None:
            return None
        ratings = adjusted_ratings(results_of(history), ridge=ridge)
        if ratings is None:
            return base
        replaced = {f"{side}.points_for": ratings.rate_for(team) for side, team in (("home", home_id), ("away", away_id))}
        replaced |= {f"{side}.points_against": ratings.rate_against(team) for side, team in (("home", home_id), ("away", away_id))}
        return [Input(i.name, replaced[i.name], i.unit, i.source) if replaced.get(i.name) is not None else i for i in base]
    return build


VARIANTS: dict[str, Variant] = {"production": production_inputs, "raw": raw_inputs, "sos": sos_variant(1.0),
                                "sos3": sos_variant(3.0), "sos0": sos_variant(0.0)}


# ---------------------------------------------------------------- the walk-forward replay
def backtest(games: Sequence[Game], league: str, variant: Variant = production_inputs, min_week: int = 5) -> tuple[list[Prediction], int]:
    """(predictions, games skipped for too little history). Week W is predicted from the games of weeks W-LOOKBACK_WEEKS .. W-1
    only, so nothing that happened on or after the game is ever used."""
    weeks = sorted({g.week for g in games})
    predictions: list[Prediction] = []
    skipped = 0
    for week in weeks:
        if week < min_week:
            continue
        history = [g.game for g in games if week - fb.LOOKBACK_WEEKS <= g.week < week]
        for entry in (g for g in games if g.week == week):
            game = entry.game
            inputs = variant(history, game.home.team_id, game.away.team_id, league)
            if inputs is None:
                skipped += 1
                continue
            try:
                model = af._Model(inputs, league)
            except af.MissingInput:
                skipped += 1
                continue
            predictions.append(_predict(model, game, week))
    return predictions, skipped


def _half_line(center: float, offset: int) -> Decimal:
    return Decimal(str(math.floor(center) + 0.5 + offset))


def _predict(model: "af._Model", game: fb.PlayedGame, week: int) -> Prediction:
    margin = game.home.score - game.away.score
    total = game.home.score + game.away.score
    mu_margin = model.mu["home"] - model.mu["away"]
    mu_total = model.mu["home"] + model.mu["away"]
    result = Prediction(week, mu_margin, mu_total, model.mu["home"], model.mu["away"], float(margin), float(total),
                        float(game.home.score), float(game.away.score))
    if model.has_fh and len(game.home.quarters) >= 2 and len(game.away.quarters) >= 2:
        result.mu_fh_total = sum(model.mu1.values())
        result.fh_total = float(sum(game.home.quarters[:2]) + sum(game.away.quarters[:2]))
    if margin != 0:
        result.events.append(Event("winner", model.p_home_wins(), margin > 0))
    dist = model.margin()
    for offset in LINE_OFFSETS:
        hcp = _half_line(-mu_margin, offset)                       # the home side is given hcp points
        probs = handicap_probabilities(dist, hcp)
        if probs is not None:
            result.events.append(Event("handicap", probs[0], margin + float(hcp) > 0))
        line = _half_line(mu_total, offset)
        probs = line_probabilities(model.total_dist(), line)
        if probs is not None:
            result.events.append(Event("total", probs[0], total > float(line)))
        for side, points in (("home", game.home.score), ("away", game.away.score)):
            line = _half_line(model.mu[side], offset // 2)
            probs = line_probabilities(model.team_dist(side), line)
            if probs is not None:
                result.events.append(Event("team_total", probs[0], points > float(line)))
    return result


# ---------------------------------------------------------------- the report
def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / (len(values) - 1))


def accuracy(predictions: Sequence[Prediction]) -> dict[str, float]:
    margin_err = [p.margin - p.mu_margin for p in predictions]
    total_err = [p.total - p.mu_total for p in predictions]
    team_err = [e for p in predictions for e in (p.home_points - p.mu_home, p.away_points - p.mu_away)]
    fh_err = [p.fh_total - p.mu_fh_total for p in predictions if p.fh_total is not None]
    winners = [e for p in predictions for e in p.events if e.kind == "winner"]
    return {"games": float(len(predictions)),
            "margin_mae": _mean([abs(e) for e in margin_err]), "margin_bias": _mean(margin_err), "margin_sd": _sd(margin_err),
            "total_mae": _mean([abs(e) for e in total_err]), "total_bias": _mean(total_err), "total_sd": _sd(total_err),
            "team_sd": _sd(team_err), "team_bias": _mean(team_err),
            "fh_games": float(len(fh_err)), "fh_bias": _mean(fh_err), "fh_sd": _sd(fh_err),
            "winner_accuracy": _mean([1.0 if (e.p >= 0.5) == e.hit else 0.0 for e in winners]),
            "winner_brier": _mean([(e.p - (1.0 if e.hit else 0.0)) ** 2 for e in winners])}


def calibration(events: Iterable[Event]) -> list[tuple[str, int, float, float]]:
    """For the side the model favours in each event: (bucket label, n, mean predicted, share that happened)."""
    rows = []
    sided = [(max(e.p, 1.0 - e.p), e.hit if e.p >= 0.5 else not e.hit) for e in events]
    for low, high in BUCKETS:
        chosen = [(q, hit) for q, hit in sided if low <= q < high]
        if chosen:
            label = f"{low:.0%}-{min(high, 1.0):.0%}"
            rows.append((label, len(chosen), _mean([q for q, _ in chosen]), _mean([1.0 if hit else 0.0 for _, hit in chosen])))
    return rows


def format_report(predictions: Sequence[Prediction], league: str, label: str, skipped: int = 0) -> str:
    if not predictions:
        return f"{label}: no games could be predicted (too little history or no data)."
    acc = accuracy(predictions)
    assumed = {"margin": af.MARGIN_SD[league], "total": af.TOTAL_SD[league], "team": af.TEAM_SD[league]}
    weeks = sorted({p.week for p in predictions})
    lines = [f"=== {label}: {league} football, {int(acc['games'])} games predicted (weeks {weeks[0]}-{weeks[-1]}), {skipped} skipped for "
             f"too little history ===",
             "",
             "How far off the expectations are (actual minus expected; the standard deviation is what the model ASSUMES):",
             f"  margin : mean error {acc['margin_bias']:+5.2f}  average miss {acc['margin_mae']:5.2f}  actual sd {acc['margin_sd']:5.2f}"
             f"  (model assumes {assumed['margin']:g})",
             f"  total  : mean error {acc['total_bias']:+5.2f}  average miss {acc['total_mae']:5.2f}  actual sd {acc['total_sd']:5.2f}"
             f"  (model assumes {assumed['total']:g})",
             f"  one team's points: mean error {acc['team_bias']:+5.2f}  actual sd {acc['team_sd']:5.2f}  (model assumes {assumed['team']:g})",
             f"  1st-half total ({int(acc['fh_games'])} games): mean error {acc['fh_bias']:+5.2f}  actual sd {acc['fh_sd']:5.2f}"
             f"  (model assumes {af.FH_TOTAL_SD[league]:g})",
             f"  winner : right {acc['winner_accuracy']:.1%} of games, Brier {acc['winner_brier']:.3f} (0.250 = coin flip)",
             "",
             "When the model favours a side, how often is it right? (all lines around its own expectation)"]
    groups = (("winner", [e for p in predictions for e in p.events if e.kind == "winner"]),
              ("handicap", [e for p in predictions for e in p.events if e.kind == "handicap"]),
              ("total", [e for p in predictions for e in p.events if e.kind == "total"]),
              ("team_total", [e for p in predictions for e in p.events if e.kind == "team_total"]),
              ("ALL", [e for p in predictions for e in p.events]))
    for name, events in groups:
        rows = calibration(events)
        if not rows:
            continue
        lines.append(f"  {name}:")
        lines.append("    model says    n   predicted   happened   gap")
        for bucket, n, predicted, actual in rows:
            lines.append(f"    {bucket:9s} {n:5d}     {predicted:6.1%}    {actual:6.1%}  {actual - predicted:+6.1%}")
    high = [(max(e.p, 1 - e.p), e.hit if e.p >= 0.5 else not e.hit) for p in predictions for e in p.events]
    high = [(q, hit) for q, hit in high if q >= 0.70]
    if high:
        lines += ["", f"At 70% or more (all kinds): {len(high)} events, predicted {_mean([q for q, _ in high]):.1%}, "
                      f"happened {_mean([1.0 if hit else 0.0 for _, hit in high]):.1%}"]
    return "\n".join(lines)


def high_confidence(predictions: Sequence[Prediction], floor: float = 0.70) -> tuple[int, float, float]:
    """(events, mean predicted, share that happened) for the favoured side of every event the model puts at `floor` or more."""
    sided = [(max(e.p, 1.0 - e.p), e.hit if e.p >= 0.5 else not e.hit) for p in predictions for e in p.events]
    sided = [(q, hit) for q, hit in sided if q >= floor]
    return len(sided), _mean([q for q, _ in sided]), _mean([1.0 if hit else 0.0 for _, hit in sided])


def format_comparison(results: dict[str, Sequence[Prediction]]) -> str:
    lines = ["Variants on the SAME games (lower misses and a smaller gap are better):",
             "  variant        games  margin miss  total miss  margin sd  total sd  winner Brier  70%+: said / happened (gap)"]
    for name, predictions in results.items():
        if not predictions:
            lines.append(f"  {name:12s}  no games")
            continue
        acc = accuracy(predictions)
        n, said, happened = high_confidence(predictions)
        lines.append(f"  {name:12s} {int(acc['games']):6d}  {acc['margin_mae']:11.2f}  {acc['total_mae']:10.2f}  {acc['margin_sd']:9.2f}"
                     f"  {acc['total_sd']:8.2f}  {acc['winner_brier']:12.3f}  {said:7.1%} / {happened:6.1%} ({happened - said:+.1%})")
    return "\n".join(lines)


# ---------------------------------------------------------------- command line
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Walk-forward backtest of the football model on past ESPN scoreboards (read only).")
    parser.add_argument("--league", choices=("college", "nfl", "both"), default="college")
    parser.add_argument("--year", type=int, action="append", help="a season (repeat for several); default: this year and last")
    parser.add_argument("--min-week", type=int, default=5, help="the first week to predict (earlier weeks are only history)")
    parser.add_argument("--variant", choices=sorted(VARIANTS), action="append",
                        help="how model inputs are built (repeat to compare several on the same games); default: production")
    parser.add_argument("--refresh", action="store_true", help="read the seasons from ESPN again even when a local copy exists")
    args = parser.parse_args(argv)
    this_year = datetime.now(timezone.utc).year
    years = args.year or [this_year - 1, this_year]
    leagues = (af.COLLEGE, af.NFL) if args.league == "both" else (af.COLLEGE if args.league == "college" else af.NFL,)
    names = args.variant or ["production"]
    for league in leagues:
        seasons: dict[int, list[Game]] = {}
        for year in years:
            try:
                seasons[year] = season_games(league, year, refresh=args.refresh)
            except (httpx.HTTPError, ValueError, StatsFetchError) as error:
                print(f"{league} {year}: could not be read ({type(error).__name__}); skipped")
                continue
            print(f"{league} {year}: {len(seasons[year])} completed games")
        results: dict[str, list[Prediction]] = {}
        for name in names:
            predictions: list[Prediction] = []
            skipped = 0
            for games in seasons.values():
                more, thin = backtest(games, league, VARIANTS[name], args.min_week)
                predictions += more
                skipped += thin
            results[name] = predictions
            print()
            print(format_report(predictions, league, f"variant {name}, seasons {', '.join(map(str, seasons))}", skipped))
        if len(results) > 1:
            print()
            print(format_comparison(results))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
