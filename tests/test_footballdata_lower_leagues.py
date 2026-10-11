"""Lower divisions on football-data.co.uk (2026-10-11): their SportyBet names map to the right files and club spellings resolve."""
import pytest

from predictions.adapters.footballdata.stats import FD_ALIASES, LEAGUES
from predictions.core.identity import resolve_team
from predictions.core.types import normalize_team

LOWER = {"England - League One": "E2", "England - League Two": "E3", "England - National League": "EC",
         "Scotland - Championship": "SC1", "Scotland - League 1": "SC2", "Scotland - League 2": "SC3",
         "Spain - LaLiga 2": "SP2", "Italy - Serie B": "I2", "France - Ligue 2": "F2"}


@pytest.mark.parametrize("name,code", LOWER.items())
def test_lower_division_names_map_to_their_file(name, code):
    assert LEAGUES[normalize_team(name)] == code


@pytest.mark.parametrize("sportybet,file_spelling,league", [
    ("Queens Park Rangers", "QPR", ["QPR", "Hull", "Leeds"]),
    ("Sheffield Wednesday", "Sheffield Weds", ["Sheffield Weds", "Sheffield United"]),
    ("Stade Lavallois MFC", "Laval", ["Laval", "Lens"]),
    ("Ayr United FC", "Ayr", ["Ayr", "Airdrie Utd"]),
    ("Cardiff City", "Cardiff", ["Cardiff", "Bristol City", "Charlton"]),
])
def test_club_spellings_resolve_to_the_file_spelling(sportybet, file_spelling, league):
    match = resolve_team(sportybet, {normalize_team(t) for t in league} | set(league), FD_ALIASES)
    assert match.status == "matched" and normalize_team(match.team) == normalize_team(file_spelling)
