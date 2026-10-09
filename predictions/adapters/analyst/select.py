"""Chooses the analyst provider(s) (ANALYST_PROVIDER: a comma list from claude | gemini, tried in order; default claude).
`gemini,claude` is a hierarchy: the free Serper-search + Gemini-reading analyst first, then Claude (paid, last resort) for
whatever stats Gemini left missing, whether Gemini found nothing, found only some, gave a discarded answer or was unavailable
(a free limit reached, HTTP 429, a failed search, or any service error), within the Claude limits below (CLAUDE_CALL_LIMIT per
request, nothing for a game Gemini found nothing about, and only the "essential" stats). A provider that fails is skipped for
COOLDOWN_SECONDS so a dead one is not retried on every call. Anything else in the list refuses to start."""
from __future__ import annotations

import time
from dataclasses import replace

from predictions.adapters.analyst.claude import ClaudeAnalyst
from predictions.adapters.analyst.gemini import GeminiAnalyst
from predictions.core.analyst import PAID_POLICY, AnalystReport, AnalystUsage
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
    """The hierarchy (free Gemini first, paid Claude last), one lookup per game. Gemini is asked for everything wanted. Claude is
    asked ONLY for the stats still missing, and only when it is worth paying for:
      * only the stats in the query's `essential` part (for a Review of a soccer game: the goals-only set, not xG);
      * not when Gemini answered but found nothing at all (unless CLAUDE_AFTER_EMPTY is on);
      * not beyond CLAUDE_CALL_LIMIT lookups for this request (the answer is then not kept for the day, so a later request retries).
    A failed, unavailable or discarded Gemini answer counts as "could not answer", not as "nothing exists", so Claude is asked.
    A flags-only query is answered by the first analyst alone. Answers are combined; only a paid analyst's usage is spend.
    One of these is built per request, so its counters are that request's (`breakdown()` shows them to the owner)."""

    def __init__(self, analysts, clock=time.monotonic, cooldown: float = COOLDOWN_SECONDS,
                 paid_limit: int | None = None, after_empty: bool | None = None):
        self._analysts = list(analysts)
        self._clock, self._cooldown = clock, cooldown
        self._failed_at: dict[int, float] = {}
        self._wait: dict[int, float] = {}
        self.last_resort = self._analysts[0].last_resort
        self.paid_limit = PAID_POLICY.call_limit if paid_limit is None else paid_limit
        self.after_empty = PAID_POLICY.after_empty if after_empty is None else after_empty
        self.calls: dict[str, int] = {}          # lookups made, by analyst name
        self.skipped_empty = 0                   # Claude not asked: Gemini found nothing for the game
        self.skipped_cap = 0                     # Claude not asked: this request's Claude limit was used up
        self.skipped_nonessential = 0            # Claude not asked: only stats not worth paying for were missing

    def _cooling(self, index: int) -> bool:
        at = self._failed_at.get(index)
        return at is not None and self._clock() - at < self._wait.get(index, self._cooldown)

    @staticmethod
    def _name(analyst) -> str:
        return getattr(analyst, "name", type(analyst).__name__.removesuffix("Analyst").lower())

    def breakdown(self) -> str:
        """e.g. "Gemini 41, Claude 10; Claude not used for 12 game(s) where Gemini found nothing and 5 over its limit of 10"."""
        if not self.calls and not (self.skipped_empty or self.skipped_cap or self.skipped_nonessential):
            return ""
        text = ", ".join(f"{name.title()} {count}" for name, count in self.calls.items())
        why = []
        if self.skipped_empty:
            why.append(f"{self.skipped_empty} game(s) where Gemini found nothing")
        if self.skipped_nonessential:
            why.append(f"{self.skipped_nonessential} that only lacked stats not worth paying for")
        if self.skipped_cap:
            why.append(f"{self.skipped_cap} over its limit of {self.paid_limit}")
        return text + (f"; Claude not used for {' and '.join(why)}" if why else "")

    async def research(self, query):
        order = [i for i in range(len(self._analysts)) if not self._cooling(i)] or list(range(len(self._analysts)))
        stats: list = []
        flags: list = []
        usage = AnalystUsage()
        error = rejected = None
        answered = last_failed = cap_hit = policy_stop = free_empty = False
        remaining = tuple(query.wanted)
        current = query
        for index in order:
            analyst = self._analysts[index]
            name = self._name(analyst)
            paid = bool(getattr(analyst, "paid", False))
            if paid:
                names = tuple(n for n in remaining if query.essential is None or n in query.essential)
                if query.wanted and not names:
                    if remaining:
                        self.skipped_nonessential += 1
                    policy_stop = True
                    break
                if free_empty and not self.after_empty:
                    self.skipped_empty += 1
                    policy_stop = True
                    break
                if self.calls.get(name, 0) >= self.paid_limit:
                    self.skipped_cap += 1
                    cap_hit = True
                    break
                current = replace(query, wanted=names)
            self.calls[name] = self.calls.get(name, 0) + 1
            try:
                report = await analyst.research(current)
            except AnalystUnavailable as e:
                self._failed_at[index] = self._clock()
                self._wait[index] = getattr(e, "retry_after", None) or self._cooldown   # a daily-quota 429 says how long
                error, last_failed = e, True
                continue
            answered, last_failed = True, False
            if paid:
                usage = AnalystUsage(usage.input_tokens + report.usage.input_tokens,
                                     usage.output_tokens + report.usage.output_tokens, usage.searches + report.usage.searches)
            if report.rejected:
                rejected = report.rejected
                continue
            stats.extend(report.stats)
            flags.extend(f for f in report.flags if f not in flags)
            if not paid and not report.stats and not report.flags:
                free_empty = True
            have = {i.name for i in stats}
            remaining = tuple(n for n in query.wanted if n not in have)
            if not remaining:
                break
            current = replace(query, wanted=remaining)
        if not answered and not (cap_hit or policy_stop):
            raise error
        if not stats and not flags and rejected is not None and (remaining or not query.wanted):
            return AnalystReport(rejected=rejected, usage=usage)
        # not final: the last analyst tried was unreachable, the paid step was held back by its limit, or nothing at all could
        # answer. Real but not kept for the day, so a later request can retry.
        incomplete = bool(remaining) and (last_failed or cap_hit or (policy_stop and not answered))
        return AnalystReport(tuple(stats), tuple(flags), "", usage, incomplete)


def build_analyst(providers, mode: str):
    """The analyst for the chosen providers (those without their API keys are skipped), or None when the analyst is off
    or none of them can be built."""
    analysts = [a for a in (PROVIDERS[p].from_environment(mode) for p in providers) if a is not None]
    if not analysts:
        return None
    return analysts[0] if len(analysts) == 1 else FallbackAnalyst(analysts)
