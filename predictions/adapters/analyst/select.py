"""Chooses the analyst provider(s) (ANALYST_PROVIDER: a comma list from claude | gemini, tried in order; default claude).
`gemini,claude` is a hierarchy: the free Serper-search + Gemini-reading analyst first, then Claude (paid, last resort) for
whatever stats Gemini left missing, whether Gemini found nothing, found only some, gave a discarded answer or was unavailable
(a free limit reached, HTTP 429, a failed search, or any service error). A provider that fails is skipped for
COOLDOWN_SECONDS so a dead one is not retried on every call. Anything else in the list refuses to start."""
from __future__ import annotations

import time
from dataclasses import replace

from predictions.adapters.analyst.claude import ClaudeAnalyst
from predictions.adapters.analyst.gemini import GeminiAnalyst
from predictions.core.analyst import AnalystReport, AnalystUsage
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
        """The hierarchy: the first analyst (free Gemini) is asked for everything wanted; the next (paid Claude) is asked ONLY for
        the stats still missing, and only because they are still missing. A failed, unavailable or discarded answer counts as
        nothing found. A flags-only query (nothing wanted) is answered by the first analyst alone. The stats and flags of
        every analyst that answered are combined; only a paid analyst's usage is counted as spend."""
        order = [i for i in range(len(self._analysts)) if not self._cooling(i)] or list(range(len(self._analysts)))
        stats: list = []
        flags: list = []
        usage = AnalystUsage()
        error = rejected = None
        answered = last_failed = False
        remaining = tuple(query.wanted)
        current = query
        for index in order:
            analyst = self._analysts[index]
            try:
                report = await analyst.research(current)
            except AnalystUnavailable as e:
                self._failed_at[index] = self._clock()
                self._wait[index] = getattr(e, "retry_after", None) or self._cooldown   # a daily-quota 429 says how long
                error, last_failed = e, True
                continue
            answered, last_failed = True, False
            if getattr(analyst, "paid", False):
                usage = AnalystUsage(usage.input_tokens + report.usage.input_tokens,
                                     usage.output_tokens + report.usage.output_tokens, usage.searches + report.usage.searches)
            if report.rejected:
                rejected = report.rejected
                continue
            stats.extend(report.stats)
            flags.extend(f for f in report.flags if f not in flags)
            have = {i.name for i in stats}
            remaining = tuple(n for n in query.wanted if n not in have)
            if not remaining:
                break
            current = replace(query, wanted=remaining)
        if not answered:
            raise error
        if not stats and not flags and rejected is not None and (remaining or not query.wanted):
            return AnalystReport(rejected=rejected, usage=usage)
        # the last analyst tried could not be reached while stats were still missing: this answer is real but not final, so it is not kept for the day
        incomplete = bool(remaining) and last_failed
        return AnalystReport(tuple(stats), tuple(flags), "", usage, incomplete)


def build_analyst(providers, mode: str):
    """The analyst for the chosen providers (those without their API keys are skipped), or None when the analyst is off
    or none of them can be built."""
    analysts = [a for a in (PROVIDERS[p].from_environment(mode) for p in providers) if a is not None]
    if not analysts:
        return None
    return analysts[0] if len(analysts) == 1 else FallbackAnalyst(analysts)
