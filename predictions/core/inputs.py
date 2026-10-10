"""Model inputs with provenance and the input registry (AD-3). Standard library only."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

ANALYST_PREFIX = "analyst-sourced:"
SIDES = ("home", "away")
MATCH = "match"  # scope of inputs about the game itself (the referee), named "match.<stat>"
MIN_MATCHES = 5  # fewer past matches than this is not enough history
GOALS_ONLY_MIN_MATCHES = 8  # soccer without xG (ESPN, Story 9.3): a goals-only estimate needs a bigger sample
FOOTBALL_MIN_MATCHES = 3  # a football team has few games early in the season (17 in an NFL year)


@dataclass(frozen=True)
class Input:
    """One model input: where it came from is part of it. `name` is "<side>.<stat>", e.g. "home.xg_for"."""
    name: str
    value: float
    unit: str
    source: str  # an adapter name, or "analyst-sourced:<citation>"


@dataclass(frozen=True)
class InputSpec:
    stat: str
    unit: str
    low: float
    high: float
    required: bool
    scope: str = "team"  # "team": one per side ("home.xg_for"); "match": one per game ("match.referee_cards_avg")


# Soccer team inputs, one set per side. Values outside [low, high] are discarded, never clamped.
SOCCER_REGISTRY: dict[str, InputSpec] = {s.stat: s for s in (
    InputSpec("matches", "matches", MIN_MATCHES, 40, True),
    InputSpec("xg_for", "xG per match", 0.0, 6.0, True),
    InputSpec("xg_against", "xG per match", 0.0, 6.0, True),
    InputSpec("goals_for", "goals per match", 0.0, 8.0, True),
    InputSpec("goals_against", "goals per match", 0.0, 8.0, True),
    InputSpec("ppda", "opponent passes per defensive action", 1.0, 60.0, False),
    InputSpec("deep", "deep completions per match", 0.0, 40.0, False),
    InputSpec("deep_allowed", "deep completions allowed per match", 0.0, 40.0, False),
    # Story 2.5, all optional: a corners or cards pick without them is "insufficient data", never guessed.
    InputSpec("corners_for", "corners per match", 0.0, 20.0, False),
    InputSpec("corners_against", "corners per match", 0.0, 20.0, False),
    InputSpec("cards_for", "cards per match", 0.0, 10.0, False),  # yellow cards the team receives (spec 11-1: reds are not counted)
    InputSpec("referee_cards_avg", "cards per game", 0.0, 12.0, False, MATCH),  # yellow cards, per game refereed
    InputSpec("referee_games", "games", 0.0, 2000.0, False, MATCH),
)}


# Soccer WITHOUT xG (Story 9.3): ESPN gives goals only. A separate registry so the xG model keeps its required xG inputs;
# a fixture is READY here with goals and a sample of at least GOALS_ONLY_MIN_MATCHES matches. The soccer engine
# recognises such a fixture by the absence of xG inputs and prices it with its goals-only model.
GOALS_ONLY_REGISTRY: dict[str, InputSpec] = {s.stat: s for s in (
    InputSpec("matches", "matches", GOALS_ONLY_MIN_MATCHES, 40, True),
    InputSpec("goals_for", "goals per match", 0.0, 8.0, True),
    InputSpec("goals_against", "goals per match", 0.0, 8.0, True),
)}

# Basketball (Story 6.1/6.2). One set per side ("home.ortg"), plus one per game ("match.league_pace").
# ORTG/DRTG are points scored/allowed per 100 possessions over the team's last games; possessions are ESTIMATED
# (see adapters/espn/stats.py and spec-6-1). Values outside [low, high] are discarded, never clamped.
BASKETBALL_REGISTRY: dict[str, InputSpec] = {s.stat: s for s in (
    InputSpec("matches", "games", MIN_MATCHES, 40, True),
    InputSpec("ortg", "points per 100 possessions", 70.0, 150.0, True),
    InputSpec("drtg", "points per 100 possessions", 70.0, 150.0, True),
    InputSpec("pace", "possessions per game", 55.0, 125.0, True),
    InputSpec("points_for", "points per game", 40.0, 160.0, False),
    InputSpec("points_against", "points per game", 40.0, 160.0, False),
    # First-half scoring: only the 1st-half total markets and the "explosive start" route need it. Missing means
    # those picks are "insufficient data", never guessed.
    InputSpec("fh_points_for", "1st-half points per game", 15.0, 90.0, False),
    InputSpec("fh_points_against", "1st-half points per game", 15.0, 90.0, False),
    InputSpec("league_pace", "possessions per game", 55.0, 125.0, False, MATCH),
)}

# Baseball, MLB (Story 6.3). One set per side ("home.runs_for"), plus per-game inputs ("match.league_runs"). Runs are per
# game over the team's last games (see adapters/mlb/stats.py); F5 = the first five innings; sp_* is the probable
# starting pitcher's season figure. Optional inputs missing means the picks that need them are "insufficient data".
BASEBALL_REGISTRY: dict[str, InputSpec] = {s.stat: s for s in (
    InputSpec("matches", "games", MIN_MATCHES, 40, True),
    InputSpec("runs_for", "runs per game", 0.0, 15.0, True),
    InputSpec("runs_against", "runs per game", 0.0, 15.0, True),
    InputSpec("f5_runs_for", "runs per 5 innings", 0.0, 10.0, False),
    InputSpec("f5_runs_against", "runs per 5 innings", 0.0, 10.0, False),
    InputSpec("sp_runs_per9", "runs allowed per 9 innings", 0.0, 15.0, False),
    InputSpec("sp_starts", "starts", 0.0, 40.0, False),
    InputSpec("league_runs", "runs per team per game", 2.0, 8.0, False, MATCH),
    InputSpec("league_f5_runs", "runs per team per 5 innings", 1.0, 6.0, False, MATCH),
)}

# Ice hockey, NHL (Stories 6.4 and 6.5). xg_* and p1_xg_* are PROXIES built by adapters/nhl/stats.py from shot locations
# (the NHL API publishes no expected goals); they are labeled as such in their unit. Period-1 and shot inputs are
# optional: without them the first-period picks are "insufficient data".
HOCKEY_REGISTRY: dict[str, InputSpec] = {s.stat: s for s in (
    InputSpec("matches", "games", MIN_MATCHES, 40, True),
    InputSpec("goals_for", "goals per game", 0.0, 10.0, True),
    InputSpec("goals_against", "goals per game", 0.0, 10.0, True),
    InputSpec("xg_for", "expected goals per game (proxy)", 0.0, 8.0, True),
    InputSpec("xg_against", "expected goals per game (proxy)", 0.0, 8.0, True),
    InputSpec("shots_for", "shots on goal per game", 5.0, 70.0, False),
    InputSpec("shots_against", "shots on goal per game", 5.0, 70.0, False),
    InputSpec("p1_goals_for", "1st-period goals per game", 0.0, 5.0, False),
    InputSpec("p1_goals_against", "1st-period goals per game", 0.0, 5.0, False),
    InputSpec("p1_xg_for", "1st-period expected goals per game (proxy)", 0.0, 4.0, False),
    InputSpec("p1_xg_against", "1st-period expected goals per game (proxy)", 0.0, 4.0, False),
    InputSpec("league_goals", "goals per team per game", 1.5, 5.0, False, MATCH),
    InputSpec("league_p1_goals", "1st-period goals per team per game", 0.3, 2.5, False, MATCH),
)}

# American football, NFL and college (Stories 7.1 to 7.4). One set per side ("home.points_for") plus per-game inputs. Points
# are per game over the team's last games from ESPN's scoreboard (adapters/espn/football.py); q1_* is the first quarter,
# fh_* the first half; q1_rank is the team's rank (1 = best) by first-quarter points per game over the league sample (a PROXY
# for "first-quarter offensive efficiency"); pace = possession seconds per offensive play and rush_rate = rushing attempts
# as a percent of offensive plays, both season to date; match.wind_mph = projected sustained wind at kickoff (Open-Meteo, or 0
# for a fixed-roof stadium). Optional inputs missing means the rules that need them do not apply, except wind: no wind means
# an Over is "insufficient data".
FOOTBALL_REGISTRY: dict[str, InputSpec] = {s.stat: s for s in (
    InputSpec("matches", "games", FOOTBALL_MIN_MATCHES, 40, True),
    InputSpec("points_for", "points per game", 0.0, 70.0, True),
    InputSpec("points_against", "points per game", 0.0, 70.0, True),
    # opponent-adjusted rates (core/ratings.py), college only: what the team would score / allow against an average opponent
    InputSpec("adj_points_for", "points per game, opponent-adjusted", -10.0, 90.0, False),
    InputSpec("adj_points_against", "points per game, opponent-adjusted", -10.0, 90.0, False),
    InputSpec("q1_points_for", "1st-quarter points per game", 0.0, 21.0, False),
    InputSpec("fh_points_for", "1st-half points per game", 0.0, 45.0, False),
    InputSpec("fh_points_against", "1st-half points per game", 0.0, 45.0, False),
    InputSpec("q1_rank", "rank by 1st-quarter points per game (1 = best)", 1.0, 200.0, False),
    InputSpec("pace", "seconds per offensive play", 15.0, 45.0, False),
    InputSpec("rush_rate", "percent of offensive plays that are rushes", 0.0, 100.0, False),
    InputSpec("league_ppg", "points per team per game", 10.0, 40.0, False, MATCH),
    InputSpec("league_fh_ppg", "1st-half points per team per game", 4.0, 25.0, False, MATCH),
    InputSpec("wind_mph", "mph", 0.0, 80.0, False, MATCH),
)}

# The registry of each sport. A sport added later adds its registry here (see the "add a sport" note in the progress log).
REGISTRIES: dict[str, dict[str, InputSpec]] = {"soccer": SOCCER_REGISTRY, "basketball": BASKETBALL_REGISTRY,
                                               "baseball": BASEBALL_REGISTRY, "hockey": HOCKEY_REGISTRY,
                                               "american_football": FOOTBALL_REGISTRY}


def registry_for(sport: str) -> dict[str, InputSpec]:
    """The input registry of a sport; an unknown sport has no registry, so nothing about it is ever accepted."""
    return REGISTRIES.get(sport, {})


def _spec_for(name: str, registry: dict[str, InputSpec]) -> InputSpec | None:
    side, _, stat = name.partition(".")
    spec = registry.get(stat)
    if spec is None:
        return None
    if side in SIDES and spec.scope == "team":
        return spec
    return spec if side == MATCH and spec.scope == MATCH else None


def check_inputs(raw: Iterable[Input], registry: dict[str, InputSpec] | None = None
                 ) -> tuple[tuple[Input, ...], tuple[str, ...]]:
    """Keep inputs that are known, in units, finite and in range. Adapter data beats analyst data for the
    same name. Returns (kept inputs sorted by name, human-readable reasons for each discarded input)."""
    registry = SOCCER_REGISTRY if registry is None else registry
    kept: dict[str, Input] = {}
    discarded: list[str] = []
    for item in raw:
        spec = _spec_for(item.name, registry)
        if spec is None:
            discarded.append(f"{item.name}: not a known input")
        elif item.unit != spec.unit:
            discarded.append(f"{item.name}: unit {item.unit!r} is not {spec.unit!r}")
        elif isinstance(item.value, bool) or not isinstance(item.value, (int, float)) or not math.isfinite(item.value):
            discarded.append(f"{item.name}: value is not a number")
        elif not spec.low <= item.value <= spec.high:
            discarded.append(f"{item.name}: {item.value:g} is outside {spec.low:g} to {spec.high:g}")
        else:
            current = kept.get(item.name)
            if current is None or (current.source.startswith(ANALYST_PREFIX)
                                   and not item.source.startswith(ANALYST_PREFIX)):
                kept[item.name] = item
    return tuple(sorted(kept.values(), key=lambda i: i.name)), tuple(discarded)


def missing_required(inputs: Iterable[Input], registry: dict[str, InputSpec] | None = None) -> tuple[str, ...]:
    registry = SOCCER_REGISTRY if registry is None else registry
    have = {i.name for i in inputs}
    return tuple(f"{side}.{spec.stat}" for side in SIDES for spec in registry.values()
                 if spec.required and spec.scope == "team" and f"{side}.{spec.stat}" not in have)
