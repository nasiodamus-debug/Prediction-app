"""Opponent-adjusted scoring ratings (a simple rating system), standard library only.

A team's raw points-for and points-against average flatter a team that played weak opponents and hurt one that played strong ones.
Here every game says: what a team scored is about the league level + its own attack + the opponent's defensive weakness, and what
it allowed is about the league level + its own defensive weakness + the opponent's attack. The attack and defence of every team are
then found together by repeating the same averaging until they settle (each pass uses the other teams' latest values). `ridge` adds
that many phantom average games to every team, so a team with few games stays close to the league level.

    rate_for(team)     = league + attack(team)    (points it would score against an average opponent)
    rate_against(team) = league + defence(team)   (points it would allow to an average opponent)

Pure arithmetic over played games: no I/O, deterministic.
"""
from __future__ import annotations

from typing import Iterable, NamedTuple, Sequence


class Result(NamedTuple):
    team_id: str
    opponent_id: str
    scored: float
    allowed: float


class Ratings(NamedTuple):
    league: float                 # mean points per team per game over the games given
    attack: dict[str, float]      # points above (+) or below (-) the league level against an average opponent
    defence: dict[str, float]     # points allowed above (+, a weaker defence) or below the league level

    def rate_for(self, team_id: str) -> float | None:
        return None if team_id not in self.attack else self.league + self.attack[team_id]

    def rate_against(self, team_id: str) -> float | None:
        return None if team_id not in self.defence else self.league + self.defence[team_id]


def results_of(games: Iterable) -> list[Result]:
    """Two results per game (one from each side). `games` are objects with .home and .away sides carrying team_id and score."""
    rows: list[Result] = []
    for game in games:
        rows.append(Result(game.home.team_id, game.away.team_id, float(game.home.score), float(game.away.score)))
        rows.append(Result(game.away.team_id, game.home.team_id, float(game.away.score), float(game.home.score)))
    return rows


def adjusted_ratings(results: Sequence[Result], ridge: float = 1.0, iterations: int = 40) -> Ratings | None:
    """None when there are no results. With ridge 0 and a connected schedule the passes settle on the plain least-squares answer;
    a positive ridge pulls every team towards the league level in proportion to how few games it has."""
    if not results:
        return None
    league = sum(r.scored for r in results) / len(results)
    by_team: dict[str, list[Result]] = {}
    for r in results:
        by_team.setdefault(r.team_id, []).append(r)
    attack = {t: 0.0 for t in by_team}
    defence = {t: 0.0 for t in by_team}
    for _ in range(max(1, iterations)):
        new_attack, new_defence = {}, {}
        for team, games in by_team.items():
            n = len(games) + ridge
            new_attack[team] = sum(g.scored - league - defence.get(g.opponent_id, 0.0) for g in games) / n
            new_defence[team] = sum(g.allowed - league - attack.get(g.opponent_id, 0.0) for g in games) / n
        attack, defence = new_attack, new_defence
    # An attack and a defence are only fixed relative to each other (adding to every attack and taking the same from every defence
    # predicts the same scores), so centre both on the league: an "average opponent" then has no attack and no defence to speak of.
    mean_attack, mean_defence = sum(attack.values()) / len(attack), sum(defence.values()) / len(defence)
    attack = {t: v - mean_attack for t, v in attack.items()}
    defence = {t: v - mean_defence for t, v in defence.items()}
    return Ratings(league, attack, defence)
