"""Story 9.1: `today` and `tomorrow` as explicit dates (Lagos time) for Matchday, Gameday and America."""
import asyncio
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import app as bot
from handled import drop_ack, handled
from openmarket import open_markets
from predictions.core.dates import parse_dates
from predictions.core.types import Fixture, MatchId
from predictions.flow.america import run_america
from predictions.flow.gameday import run_gameday
from predictions.flow.matchday import ASK_FOR_DATE, run_matchday

NOW = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)  # 10:00 on 1 Oct in Lagos
ROOT = Path(__file__).resolve().parent.parent


def test_today_and_tomorrow_are_lagos_calendar_days():
    assert parse_dates("today", NOW).dates == [date(2026, 10, 1)]
    assert parse_dates("tomorrow", NOW).dates == [date(2026, 10, 2)]
    assert parse_dates("Today", NOW).ok and parse_dates("TOMORROW", NOW).ok


def test_lagos_day_not_the_utc_day_decides():
    late = datetime(2026, 10, 1, 23, 30, tzinfo=timezone.utc)  # 00:30 on 2 Oct in Lagos
    assert parse_dates("today", late).dates == [date(2026, 10, 2)]
    assert parse_dates("tomorrow", late).dates == [date(2026, 10, 3)]


def test_month_and_year_roll_over():
    end = datetime(2026, 12, 31, 12, 0, tzinfo=timezone.utc)
    assert parse_dates("tomorrow", end).dates == [date(2027, 1, 1)]


@pytest.mark.parametrize("text, expected", [
    ("today and 3 October", [date(2026, 10, 1), date(2026, 10, 3)]),
    ("3 October and today", [date(2026, 10, 1), date(2026, 10, 3)]),
    ("tomorrow and 5 and 6 November", [date(2026, 10, 2), date(2026, 11, 5), date(2026, 11, 6)]),
    ("today, tomorrow", [date(2026, 10, 1), date(2026, 10, 2)]),
    ("tomorrow today", [date(2026, 10, 1), date(2026, 10, 2)]),
    ("today and today", [date(2026, 10, 1)]),
    ("for today", [date(2026, 10, 1)]),
    ("for the 20 and tomorrow", [date(2026, 10, 2), date(2026, 10, 20)]),
    ("on 3 Oct, for tomorrow", [date(2026, 10, 2), date(2026, 10, 3)]),
    ("today 22 September", [date(2026, 9, 22), date(2026, 10, 1)]),
])
def test_relative_days_mix_with_other_dates_and_filler(text, expected):
    result = parse_dates(text, NOW)
    assert result.ok and result.dates == expected


@pytest.mark.parametrize("text", ["todays", "tomorow", "yesterday", "next week", "today 20/09", "October today", "tonight"])
def test_anything_else_is_a_problem_and_no_day_is_picked(text):
    result = parse_dates(text, NOW)
    assert not result.ok and result.problems


def test_no_date_still_asks_and_picks_nothing():
    result = parse_dates("for the", NOW)
    assert not result.dates and not result.problems


def game(kickoff, home="Home FC", away="Away FC", competition="Testland - League", sport="soccer"):
    return Fixture(MatchId(sport, competition, kickoff, home, away), competition, home, away, kickoff, False,
                   open_markets(sport))


class Listing:
    def __init__(self, fixtures):
        self.fixtures = fixtures

    async def fetch_fixtures(self):
        return self.fixtures


TODAY_GAME = game(datetime(2026, 10, 1, 15, 0, tzinfo=timezone.utc), "Alpha", "Beta")
TOMORROW_GAME = game(datetime(2026, 10, 2, 15, 0, tzinfo=timezone.utc), "Gamma", "Delta")
LATER_GAME = game(datetime(2026, 10, 3, 15, 0, tzinfo=timezone.utc), "Eta", "Theta")
ALL = [TODAY_GAME, TOMORROW_GAME, LATER_GAME]


def text_of(run_fn, argument, **kw):
    return "\n".join(asyncio.run(run_fn(argument, Listing(ALL), str, NOW, **kw)))


def test_matchday_for_today_lists_only_todays_games():
    text = text_of(run_matchday, " for today")
    assert "Matchday fixtures for 1 Oct 2026" in text and "Alpha vs Beta" in text
    assert "Gamma" not in text and "Eta" not in text


def test_matchday_tomorrow_and_a_date_lists_both_and_nothing_else():
    text = text_of(run_matchday, " tomorrow and 3 October")
    assert "Gamma vs Delta" in text and "Eta vs Theta" in text and "Alpha" not in text


def test_gameday_and_america_share_the_parser():
    basket = [game(g.kickoff_utc, g.home, g.away, "USA - NBA", "basketball") for g in ALL]
    gd = "\n".join(asyncio.run(run_gameday("today", Listing(basket), str, NOW)))
    assert "Gameday fixtures for 1 Oct 2026" in gd and "Alpha vs Beta" in gd and "Gamma" not in gd
    am = "\n".join(asyncio.run(run_america("tomorrow", Listing([game(g.kickoff_utc, g.home, g.away, "USA - MLB", "baseball")
                                                                for g in ALL]), str, NOW)))
    assert "America fixtures for 2 Oct 2026" in am and "Gamma vs Delta" in am and "Alpha" not in am


def test_the_ask_for_a_date_reply_mentions_today_and_tomorrow_and_picks_nothing():
    for runner, trigger in ((run_matchday, "Matchday"), (run_gameday, "Gameday"), (run_america, "America")):
        reply = asyncio.run(runner("", Listing(ALL), str, NOW))
        assert len(reply) == 1 and "today" in reply[0] and "tomorrow" in reply[0] and "I never pick a day" in reply[0]
        assert reply[0].startswith(f"Which date? Send the date after {trigger}")
    assert "tomorrow" in ASK_FOR_DATE


def test_a_misspelt_day_is_refused_not_guessed():
    reply = asyncio.run(run_matchday("tomorow", Listing(ALL), str, NOW))
    assert len(reply) == 1 and "could not read" in reply[0] and "today / tomorrow" in reply[0]
    assert "Alpha" not in reply[0]


def test_help_texts_mention_today_and_tomorrow(monkeypatch):
    sent = []

    async def reply_text(message, **kwargs):
        sent.append(message)

    asyncio.run(bot.start_cmd(SimpleNamespace(message=SimpleNamespace(reply_text=reply_text)), None))
    assert "`today`" in sent[0] and "`tomorrow`" in sent[0]
    update = SimpleNamespace(effective_user=SimpleNamespace(id=bot.OWNER_ID), callback_query=None,
                             message=SimpleNamespace(text="hello", reply_text=reply_text))
    asyncio.run(handled(update))
    drop_ack(sent)
    assert "Unrecognized trigger" in sent[-1] and "today" in sent[-1] and "tomorrow" in sent[-1]


def test_readme_documents_the_new_dates():
    readme = (ROOT / "Readme.md").read_text(encoding="utf-8")
    assert "`today`" in readme and "`tomorrow`" in readme and "Matchday for today" in readme
