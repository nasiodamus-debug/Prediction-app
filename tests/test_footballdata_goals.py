"""Goals-only stats from football-data.co.uk's season files: the last source in the soccer chain (lower divisions)."""
import asyncio
from datetime import datetime, timedelta, timezone

import httpx

from predictions.adapters.footballdata.goals import FootballDataGoalsStats, NOT_COVERED, parse_goals, team_inputs
from predictions.core.stats import skip_category, CAT_NOT_COVERED
from predictions.core.types import Fixture, MatchId

KICK = datetime(2026, 10, 17, 14, 0, tzinfo=timezone.utc)


def season_csv(games):
    lines = ["Div,Date,HomeTeam,AwayTeam,FTHG,FTAG"]
    start = datetime(2026, 8, 1)
    for i, (home, away, hg, ag) in enumerate(games):
        lines.append(f"E2,{(start + timedelta(days=i)).strftime('%d/%m/%Y')},{home},{away},{hg},{ag}")
    return "\n".join(lines)


def fixture(competition="England - League One", home="Oxford United", away="Mansfield Town"):
    mid = MatchId("soccer", competition, KICK, home, away)
    return Fixture(mid, competition, home, away, KICK, False)


def run(games, fx):
    def handler(request):
        text = season_csv(games) if "2627" in str(request.url) else "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG"
        return httpx.Response(200, text=text)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return asyncio.run(FootballDataGoalsStats(client).fetch_stats([fx]))[fx.match_id]


def test_parse_skips_rows_without_a_score():
    text = "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG\nE2,01/08/2026,A,B,2,1\nE2,08/08/2026,B,A,,\n"
    assert [(g[1], g[3], g[4]) for g in parse_goals(text)] == [("A", 2, 1)]


def test_team_inputs_are_goals_for_and_against_from_the_teams_side():
    games = [(datetime(2026, 8, 1).date(), "A", "B", 3, 1), (datetime(2026, 8, 8).date(), "B", "A", 0, 2)]
    got = {i.name: i.value for i in team_inputs("home", "A", games, "t")}
    assert got == {"home.matches": 2.0, "home.goals_for": 2.5, "home.goals_against": 0.5}


def test_a_lower_division_game_with_enough_history_is_ready():
    games = [("Oxford", "Mansfield", 1, 0)] + [("Oxford", "X", 2, 1), ("Mansfield", "Y", 1, 1)] * 8
    stats = run(games, fixture())
    assert stats.ready
    assert {i.name for i in stats.inputs} >= {"home.goals_for", "away.goals_against", "home.matches"}


def test_too_little_history_is_insufficient_not_guessed():
    stats = run([("Oxford", "Mansfield", 1, 0)] * 3, fixture())
    assert not stats.ready and "too little history" in stats.reasons[0]


def test_other_competitions_are_not_covered_so_the_chain_can_move_on():
    stats = run([], fixture("Kenya - Premier League", "A", "B"))
    assert NOT_COVERED in stats.reasons[0] and skip_category(stats) == CAT_NOT_COVERED
