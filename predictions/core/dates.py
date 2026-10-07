"""Parse the dates in a Matchday request as Africa/Lagos calendar dates. Standard library only."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

LAGOS = ZoneInfo("Africa/Lagos")

_MONTHS = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3, "april": 4, "apr": 4,
    "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7, "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9, "october": 10, "oct": 10, "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}
_FILLER = {"and", "on", "for", "the", "of", "&"}
# Words that name a day relative to now (Lagos). Only these two; nothing else is ever inferred.
RELATIVE_DAYS = {"today": 0, "tomorrow": 1}
_SEPARATORS = {",", ";", "."}
_TOKEN = re.compile(r"\d+(?:st|nd|rd|th)?|[^\W\d_]+|[^\s\w]", re.UNICODE | re.IGNORECASE)
_ORDINAL = re.compile(r"^(\d+)(?:st|nd|rd|th)$", re.IGNORECASE)
# 20/09, 20-09-2026 and 20-22 could each mean more than one thing, so they are never guessed at.
_NUMERIC_DATE = re.compile(r"\d\s*[-–—/]\s*\d")
_NUMERIC_HELP = ('numeric dates and ranges such as 20/09 or 20-22 are not supported; '
                 'list each day with the month as a word, for example "20 and 22 September"')


@dataclass
class DateParse:
    dates: list[date] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)  # what could not be read; empty on success

    @property
    def ok(self) -> bool:
        return bool(self.dates) and not self.problems


def lagos_today(now: datetime) -> date:
    return now.astimezone(LAGOS).date()


def parse_dates(text: str, now: datetime) -> DateParse:
    """Read dates like "20 and 22 September", "20, 22", "5 Oct 2026", "today", "tomorrow" or "today and 3 October".
    Month and year default to now's (Lagos); today and tomorrow are Lagos calendar days of `now`.

    A day written with no month of its own uses the month (and year) most recently written before it,
    or the current ones if none was. Never invents a day: with no date, or anything unreadable, `ok` is
    False and `problems` says why (an empty `problems` with no dates means nothing was given).
    """
    today = lagos_today(now)
    result = DateParse()
    if _NUMERIC_DATE.search(text or ""):
        result.problems.append(_NUMERIC_HELP)
        return result
    tokens = _TOKEN.findall(text or "")

    groups: list[list[list]] = []  # each item: [day, month, year]; a group shares a month and year
    pending: list[list] = []  # days waiting for a month written after them ("20 and 22 September")
    month_first = None  # a month written before its days ("October 5 and 6")
    month_first_raw = None
    month_first_used = True
    last_group = None
    carry_month = None  # the month most recently written, for days that name none
    carry_year = None
    relative: list[date] = []  # days named with today / tomorrow

    def close_pending(month):
        nonlocal pending, last_group
        if pending:
            for item in pending:
                item[1] = month if month is not None else carry_month
            groups.append(pending)
            last_group = pending
            pending = []

    def check_month_used():
        if not month_first_used:
            result.problems.append(f'the month "{month_first_raw}" has no day with it')

    for raw in tokens:
        low = raw.lower()
        if low in _FILLER or raw in _SEPARATORS:
            continue
        if low in RELATIVE_DAYS:
            close_pending(None)  # a bare day before it has no month of its own: the current one
            check_month_used()
            month_first, month_first_raw, month_first_used = None, None, True
            last_group = None
            relative.append(today + timedelta(days=RELATIVE_DAYS[low]))
            continue
        if low in _MONTHS:
            carry_month = _MONTHS[low]
            if pending:
                close_pending(carry_month)
                month_first = None
            else:
                check_month_used()
                month_first, month_first_raw, month_first_used = carry_month, raw, False
                last_group = None
            continue
        ordinal = _ORDINAL.match(raw)
        if raw.isdigit() or ordinal:
            number = int(ordinal.group(1) if ordinal else raw)
            if raw.isdigit() and len(raw) == 4:
                if last_group is None:
                    result.problems.append(f"the year {raw} has no day and month before it")
                else:
                    for item in last_group:
                        item[2] = number
                    carry_year = number
                continue
            item = [number, month_first, carry_year]
            if month_first is not None:
                month_first_used = True
                last_group = [item]
                groups.append(last_group)
            else:
                pending.append(item)
            continue
        result.problems.append(f'could not read "{raw}"')

    close_pending(None)  # days with no month use the last month written, else the current month
    check_month_used()

    seen = set()
    for value in relative:
        if value not in seen:
            seen.add(value)
            result.dates.append(value)
    for group in groups:
        for day, month, year in group:
            month = month or today.month
            year = today.year if year is None else year
            try:
                value = date(year, month, day)
            except (ValueError, OverflowError):
                result.problems.append(f"{day}/{month}/{year} is not a real date")
                continue
            if value not in seen:
                seen.add(value)
                result.dates.append(value)
    result.dates.sort()
    return result
