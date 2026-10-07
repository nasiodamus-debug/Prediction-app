"""Remembers finished Reviews for a while so the same booking code, asked again the same day, is answered from memory
without reading the code or paying for stats and analyst calls again.

Kept in memory only (a restart empties it). An entry is used only while it is younger than `ttl_seconds` and only on the
same Lagos calendar day it was made. A reused review never lists a game that has started since: those are removed at
once (flow/review.py). A review made while a stats or analyst service was failing is not kept (it would repeat the
failure for hours). REVIEW_CACHE_HOURS sets the time (default 3, 0 = off); a bad value stops the bot from starting.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from predictions.core.dates import LAGOS
from predictions.core.review import ReviewResult

DEFAULT_REVIEW_CACHE_HOURS = 3.0
MAX_REVIEW_CACHE_HOURS = 24.0


def parse_review_cache_hours(raw) -> float:
    """The REVIEW_CACHE_HOURS setting, read once at startup. Unset or empty = 3 hours. '0' = the review cache is off. Otherwise a
    number of hours from 0 to 24 written plainly (digits and one dot, for example 1.5). Anything else raises ValueError (the
    bot must not start with a setting the owner did not intend). The message never echoes the value."""
    if raw is None or not str(raw).strip():
        return DEFAULT_REVIEW_CACHE_HOURS
    text = str(raw).strip()
    plain = text.replace(".", "", 1)
    if not text.isascii() or not plain.isdigit() or float(text) > MAX_REVIEW_CACHE_HOURS:
        raise ValueError("REVIEW_CACHE_HOURS in config.env is not a number of hours from 0 to 24 (for example 3 or 1.5; 0 turns "
                         f"the review cache off). Fix it or remove the line to use {DEFAULT_REVIEW_CACHE_HOURS:g}, then start the "
                         "bot again.")
    return float(text)


class ReviewCache:
    def __init__(self, hours: float = DEFAULT_REVIEW_CACHE_HOURS):
        self.hours = hours
        self._items: dict[str, tuple[datetime, ReviewResult, str]] = {}

    @property
    def enabled(self) -> bool:
        return self.hours > 0

    def clear(self) -> None:
        self._items.clear()

    @staticmethod
    def _key(code: str) -> str:
        return code.strip().upper()

    def get(self, code: str, now: datetime) -> tuple[datetime, ReviewResult, str] | None:
        """(when it was made, the review, the notes it carried) or None when there is no usable entry."""
        if not self.enabled:
            return None
        entry = self._items.get(self._key(code))
        if entry is None:
            return None
        made, result, notes = entry
        if now - made >= timedelta(hours=self.hours) or made.astimezone(LAGOS).date() != now.astimezone(LAGOS).date():
            del self._items[self._key(code)]
            return None
        return entry

    def put(self, code: str, now: datetime, result: ReviewResult, notes: str) -> None:
        if self.enabled and result.judgements:
            self._items[self._key(code)] = (now, result, notes)


REVIEW_CACHE = ReviewCache()   # app.py sets REVIEW_CACHE.hours from REVIEW_CACHE_HOURS at startup
