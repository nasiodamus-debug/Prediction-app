"""The Slip: the picks the owner is keeping, at most one per match (AD-7). Standard library only.

A Slip is immutable: `delete` and `choose` return a new Slip. A Pick has a stable `pick_id` derived from what
the pick IS (match, market, selection), never from its position in a list, so a button that names a pick
still names the same pick after other picks are deleted (AD-19).
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from decimal import Decimal

from predictions.core.market_rules import UNDER_REASON, WHOLE_LINE_REASON, is_under, is_whole_or_quarter_line
from predictions.core.model import Evaluation
from predictions.core.types import MatchId

log = logging.getLogger(__name__)
PICK_ID_LENGTH = 12  # hex characters; keeps callback data far below Telegram's 64-byte limit


class SlipError(Exception):
    pass


class UnknownPick(SlipError):
    """The pick id is not in the slip (already deleted, or never there)."""


def match_key(match_id: MatchId) -> str:
    """The one-per-match key: the canonical MatchId string (names normalized by the identity resolver)."""
    return str(match_id)


def make_pick_id(evaluation: Evaluation) -> str:
    text = "|".join((match_key(evaluation.match_id), evaluation.market, evaluation.selection,
                     str(Decimal(evaluation.odds))))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:PICK_ID_LENGTH]


@dataclass(frozen=True)
class Pick:
    pick_id: str
    evaluation: Evaluation

    @property
    def match_id(self) -> MatchId:
        return self.evaluation.match_id

    @property
    def key(self) -> str:
        return match_key(self.evaluation.match_id)


def make_pick(evaluation: Evaluation) -> Pick:
    """Only a pick that passed the floor and EV gate may enter a Slip."""
    if is_under(evaluation):   # AD-20: last line of defence, whatever the caller thinks of the evaluation
        raise ValueError(f"{UNDER_REASON}: {evaluation.selection}")
    if is_whole_or_quarter_line(evaluation):   # AD-21: no whole or quarter line is ever a pick
        raise ValueError(f"{WHOLE_LINE_REASON}: {evaluation.selection}")
    if not evaluation.passed or evaluation.insufficient:
        raise ValueError("only a passed evaluation can become a pick")
    return Pick(make_pick_id(evaluation), evaluation)


@dataclass(frozen=True)
class Slip:
    picks: tuple[Pick, ...] = ()

    @classmethod
    def from_evaluations(cls, evaluations) -> "Slip":
        """A Slip of every passed evaluation given. The same pick twice counts once."""
        seen: dict[str, Pick] = {}
        for evaluation in evaluations:
            if is_under(evaluation):   # excluded and noted, never fatal (AD-20)
                log.warning("%s: left out of the slip: %s", UNDER_REASON, evaluation.selection)
                continue
            if is_whole_or_quarter_line(evaluation):   # excluded and noted, never fatal (AD-21)
                log.warning("%s: left out of the slip: %s", WHOLE_LINE_REASON, evaluation.selection)
                continue
            pick = make_pick(evaluation)
            if pick.pick_id in seen and seen[pick.pick_id].evaluation != evaluation:
                raise SlipError("two different picks share one pick id")
            seen.setdefault(pick.pick_id, pick)
        return cls(tuple(seen.values()))

    def __len__(self) -> int:
        return len(self.picks)

    def get(self, pick_id: str) -> Pick:
        for pick in self.picks:
            if pick.pick_id == pick_id:
                return pick
        raise UnknownPick(pick_id)

    def __contains__(self, pick_id: str) -> bool:
        return any(p.pick_id == pick_id for p in self.picks)

    def picks_for(self, key: str) -> tuple[Pick, ...]:
        return tuple(p for p in self.picks if p.key == key)

    def conflicts(self) -> dict[str, tuple[Pick, ...]]:
        """Matches that have more than one pick, keyed by match key."""
        grouped: dict[str, list[Pick]] = {}
        for pick in self.picks:
            grouped.setdefault(pick.key, []).append(pick)
        return {k: tuple(v) for k, v in grouped.items() if len(v) > 1}

    def in_conflict(self, pick_id: str) -> bool:
        return len(self.picks_for(self.get(pick_id).key)) > 1

    @property
    def one_per_match(self) -> bool:
        return not self.conflicts()

    @property
    def matches(self) -> int:
        return len({p.key for p in self.picks})

    def delete(self, pick_id: str) -> "Slip":
        self.get(pick_id)
        return Slip(tuple(p for p in self.picks if p.pick_id != pick_id))

    def choose(self, pick_id: str) -> "Slip":
        """Keep this pick and drop every other pick for the same match."""
        chosen = self.get(pick_id)
        return Slip(tuple(p for p in self.picks if p.key != chosen.key or p.pick_id == pick_id))

    def substitute(self, pick_id: str, evaluation: Evaluation) -> "Slip":
        """Swap one pick for another pick on the SAME match (Story 5.2: the owner accepted the closest option). The
        one-pick-per-match rule holds by construction, and a replacement for a different match is refused."""
        old = self.get(pick_id)
        new = make_pick(evaluation)
        if new.key != old.key:
            raise SlipError("a substitute must be for the same match")
        if new.pick_id != old.pick_id and new.pick_id in self:
            raise SlipError("the substitute is already in the slip")
        return Slip(tuple(new if p.pick_id == pick_id else p for p in self.picks))


# ---------- games and their options (the owner's list: one game, its valid options, one chosen play) ----------

def advised_order(picks) -> tuple[Pick, ...]:
    """The options of one game, the ADVISED one first. Every pick here already passed the floor with positive EV; the
    advised play is the one with the highest success estimate, then the highest EV, then the selection text."""
    return tuple(sorted(picks, key=lambda p: (-p.evaluation.estimate.probability, -p.evaluation.ev,
                                              p.evaluation.selection, p.pick_id)))


@dataclass(frozen=True)
class Game:
    """One match on the list: every valid option, which one is chosen, and whether the game is kept. A kept game
    contributes exactly its chosen option to the Slip; a removed game contributes nothing (AD-7 by construction)."""
    key: str
    options: tuple[Pick, ...]   # advised first
    chosen_id: str
    kept: bool

    @property
    def advised(self) -> Pick:
        return self.options[0]

    @property
    def chosen(self) -> Pick:
        return next(p for p in self.options if p.pick_id == self.chosen_id)

    @property
    def match_id(self) -> MatchId:
        return self.options[0].match_id


def build_games(menu, chosen_ids=(), removed_keys=()) -> tuple[Game, ...]:
    """Group `menu` (every offered pick) by match, in list order (kickoff, then match key). A chosen id that is not an
    option of its game is ignored (the advised option is chosen). Pure and deterministic."""
    grouped: dict[str, list[Pick]] = {}
    for pick in menu:
        if is_under(pick):   # AD-20: an Under pick never becomes an option of a game
            log.warning("%s: left out of the list: %s", UNDER_REASON, pick.evaluation.selection)
            continue
        if is_whole_or_quarter_line(pick):   # AD-21: a whole or quarter line never becomes an option of a game
            log.warning("%s: left out of the list: %s", WHOLE_LINE_REASON, pick.evaluation.selection)
            continue
        grouped.setdefault(pick.key, []).append(pick)
    chosen, removed = set(chosen_ids), set(removed_keys)
    games = []
    for key, picks in grouped.items():
        options = advised_order(picks)
        pick_id = next((p.pick_id for p in options if p.pick_id in chosen), options[0].pick_id)
        games.append(Game(key, options, pick_id, key not in removed))
    return tuple(sorted(games, key=lambda g: (g.match_id.kickoff_utc, g.key)))


def slip_of(games) -> Slip:
    """The Slip: the chosen option of every kept game."""
    return Slip(tuple(g.chosen for g in games if g.kept))
