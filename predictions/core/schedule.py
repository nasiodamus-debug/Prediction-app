"""Pick the fixtures a request asks for. Pure, standard library only."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from predictions.core.dates import LAGOS, lagos_today
from predictions.core.types import Fixture


@dataclass
class Selection:
    fixtures: list[Fixture] = field(default_factory=list)  # not started, on requested dates, by kickoff
    excluded_started: int = 0
    passed_dates: list[date] = field(default_factory=list)  # requested dates before today in Lagos


def select_fixtures(fixtures: list[Fixture], dates: list[date], now: datetime) -> Selection:
    today = lagos_today(now)
    passed = sorted({d for d in dates if d < today})
    wanted = {d for d in dates if d >= today}
    result = Selection(passed_dates=passed)
    seen = set()
    for fixture in fixtures:
        if fixture.kickoff_utc.astimezone(LAGOS).date() not in wanted:
            continue
        if fixture.match_id in seen:
            continue
        seen.add(fixture.match_id)
        if fixture.started or fixture.kickoff_utc <= now:
            result.excluded_started += 1
        else:
            result.fixtures.append(fixture)
    result.fixtures.sort(key=lambda f: (f.kickoff_utc, f.competition, f.home))
    return result
