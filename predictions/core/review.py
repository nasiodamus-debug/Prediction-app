"""Review of a SportyBet booking code (Epic 4). Standard library only.

A booking code is read into CodeSelections (Story 4.1) and each one is judged by the engine of ITS OWN sport
(Story 4.2). Engines are looked up in a registry by sport, so a sport added later needs no change here: a sport
with no engine is disqualified by default and is never approved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Mapping

from predictions.core.engines import ENGINES
from predictions.core.ports import CodeReadError  # noqa: F401  (re-exported for the flow)
from predictions.core.market_rules import RULE_NO_UNDER, UNDER_REASON, is_under
from predictions.core.model import Estimate, Evaluation, FixturePass
from predictions.core.stats import AMBIGUOUS, FixtureStats
from predictions.core.types import Fixture, MarketOffer, MatchId, Outcome

CODE_PATTERN = re.compile(r"[A-Za-z0-9]{3,20}")  # what a booking code looks like; anything else is never sent anywhere

NO_RULES_SPORT = "no rules for this sport"
NO_RULES_MARKET = "no rules for this market"
INSUFFICIENT_DATA = "insufficient data"
RULE_NO_RULES = "No rules built"
RULE_UNAVAILABLE = "Not offered"
NO_VALID_PLAY = "NO VALID PLAY - PASS"


@dataclass(frozen=True)
class CodeSelection:
    """One selection on a booking code, in the bot's own vocabulary. `sport` is the canonical sport name the
    engine registry uses ("soccer", "basketball", ...); `sport_label` is what SportyBet calls it."""
    sport: str
    sport_label: str
    competition: str
    home: str
    away: str
    kickoff_utc: datetime
    started: bool           # the source says the game has started or finished
    market_id: str
    market_name: str
    specifier: str          # e.g. "total=2.5", "" when the market has no line
    outcome_name: str       # e.g. "Over 2.5", "Home"
    odds: Decimal           # the odds on the code
    available: bool = True  # False when the source shows this market or outcome as suspended

    @property
    def match_id(self) -> MatchId:
        return MatchId(self.sport, self.competition, self.kickoff_utc, self.home, self.away)

    @property
    def line(self) -> str:
        """The line or handicap in the specifier ("2.5"), or "" when there is none."""
        key, _, value = self.specifier.partition("=")
        return value.strip() if key.strip() else ""

    def offer(self) -> MarketOffer:
        """The market as a one-outcome offer, so an engine judges exactly this outcome at the code's odds."""
        return MarketOffer(self.market_id, self.market_name, self.specifier, (Outcome(self.outcome_name, self.odds),))

    def is_started(self, now: datetime) -> bool:
        return self.started or self.kickoff_utc <= now


@dataclass(frozen=True)
class Judgement:
    """One selection and its verdict. The Evaluation is the same object a Matchday pick has, so it is saved by the
    same store. An evaluation with `insufficient` set has no estimate (nothing was guessed)."""
    selection: CodeSelection
    evaluation: Evaluation

    @property
    def approved(self) -> bool:
        return self.evaluation.passed


@dataclass(frozen=True)
class ReviewResult:
    code: str
    judgements: tuple[Judgement, ...]
    started: tuple[CodeSelection, ...]  # excluded from the review: the game had already started or finished

    @property
    def approved(self) -> tuple[Judgement, ...]:
        return tuple(j for j in self.judgements if j.approved)

    @property
    def disqualified(self) -> tuple[Judgement, ...]:
        return tuple(j for j in self.judgements if not j.approved)

    @property
    def evaluations(self) -> tuple[Evaluation, ...]:
        return tuple(j.evaluation for j in self.judgements)


def valid_code_format(code: str) -> bool:
    return CODE_PATTERN.fullmatch(code) is not None


def _display_market(selection: CodeSelection) -> str:
    line = selection.line
    return f"{selection.market_name} {line}".strip() if line else selection.market_name


def disqualified(selection: CodeSelection, reason: str, rule: str = "") -> Evaluation:
    """A disqualification with no estimate (the engine could not or would not price it)."""
    return Evaluation(selection.match_id, _display_market(selection), selection.outcome_name, selection.odds,
                      Estimate(0.0), Decimal(0), False, reason, rule, "", (), True)


def judge_selection(selection: CodeSelection, fixture: Fixture, stats: FixtureStats | None,
                    flag: FixturePass | None = None,
                    engines: Mapping[str, object] | None = None) -> Evaluation:
    """Judge one open selection by the rules of its own sport. Never approves anything the engine did not
    approve: no engine, an unknown market, missing data or a suspended market is a disqualification. An Under is
    disqualified first of all (AD-20), before any engine, data or analyst is consulted."""
    if is_under(selection):
        return disqualified(selection, UNDER_REASON, RULE_NO_UNDER)
    engine = (ENGINES if engines is None else engines).get(selection.sport)
    if engine is None:
        return disqualified(selection, NO_RULES_SPORT, RULE_NO_RULES)
    offer = selection.offer()
    if not engine.understands(offer):
        return disqualified(selection, NO_RULES_MARKET, RULE_NO_RULES)
    if not selection.available:
        return disqualified(selection, "this selection is not currently offered on SportyBet (suspended)",
                            RULE_UNAVAILABLE)
    ready = stats is not None and stats.ready
    refusal = engine.screen(fixture, stats.inputs if ready else ())
    if refusal is None:
        refusal = flag
    if refusal is not None:
        return disqualified(selection, f"{NO_VALID_PLAY} ({refusal.rule}: {refusal.reason})", refusal.rule)
    if not ready:
        if stats is None:
            why = "no team data was found for this game"
        elif stats.status == AMBIGUOUS:
            why = f"the team names are unclear ({stats.question})"
        else:
            why = stats.reasons[0] if stats.reasons else "the data for this game is incomplete"
        return disqualified(selection, f"{INSUFFICIENT_DATA}: {why}")
    found = engine.evaluate_exact(fixture, stats.inputs, offer)
    if not found:
        return disqualified(selection, NO_RULES_MARKET, RULE_NO_RULES)
    if len(found) != 1:
        return disqualified(selection, f"{INSUFFICIENT_DATA}: this selection could not be priced on its own")
    return found[0]
