"""Chooses the analyst provider(s) (ANALYST_PROVIDER: a comma list from claude | gemini, tried in order; default claude).
`gemini,claude` uses the free Serper-search + Gemini-reading analyst first and falls back to Claude when it is unavailable
(a free limit reached, HTTP 429, a failed search, or any service error). A provider that fails is skipped for
COOLDOWN_SECONDS so a dead one is not retried on every call. Anything else in the list refuses to start."""
from __future__ import annotations

import time

from predictions.adapters.analyst.claude import ClaudeAnalyst
from predictions.adapters.analyst.gemini import GeminiAnalyst
from predictions.core.ports import AnalystUnavailable

PROVIDERS = {"claude": ClaudeAnalyst, "gemini": GeminiAnalyst}
COOLDOWN_SECONDS = 900


def parse_analyst_provider(raw) -> tuple[str, ...]:
    names = tuple(n.strip().lower() for n in (raw or "").split(",") if n.strip()) or ("claude",)
    if any(n not in PROVIDERS for n in names) or len(set(names)) != len(names):
        raise ValueError("ANALYST_PROVIDER in config.env must be distinct names from claude and gemini, for example "
                         "gemini,claude. Fix it or remove the line to use claude, then start the bot again.")
    return names


class FallbackAnalyst:
    """Tries each analyst in order; moves to the next only when one is unavailable (never on a rejected answer).
    An analyst that was unavailable is not tried again until the cooldown has passed, unless every analyst is cooling."""

    def __init__(self, analysts, clock=time.monotonic, cooldown: float = COOLDOWN_SECONDS):
        self._analysts = list(analysts)
        self._clock, self._cooldown = clock, cooldown
        self._failed_at: dict[int, float] = {}
        self._wait: dict[int, float] = {}
        self.last_resort = self._analysts[0].last_resort

    def _cooling(self, index: int) -> bool:
        at = self._failed_at.get(index)
        return at is not None and self._clock() - at < self._wait.get(index, self._cooldown)

    async def research(self, query):
        order = [i for i in range(len(self._analysts)) if not self._cooling(i)] or list(range(len(self._analysts)))
        error = None
        for index in order:
            try:
                return await self._analysts[index].research(query)
            except AnalystUnavailable as e:
                self._failed_at[index] = self._clock()
                self._wait[index] = getattr(e, "retry_after", None) or self._cooldown   # a daily-quota 429 says how long
                error = e
        raise error


def build_analyst(providers, mode: str):
    """The analyst for the chosen providers (those without their API keys are skipped), or None when the analyst is off
    or none of them can be built."""
    analysts = [a for a in (PROVIDERS[p].from_environment(mode) for p in providers) if a is not None]
    if not analysts:
        return None
    return analysts[0] if len(analysts) == 1 else FallbackAnalyst(analysts)
