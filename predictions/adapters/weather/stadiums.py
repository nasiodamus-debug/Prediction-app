"""Stadium coordinates for the wind lookup (Story 7.4): a documented table, keyed by the venue's CITY as ESPN spells it.

Why by city and not by team: a game can be played away from the home team's stadium (NFL international games, neutral
sites), and ESPN's schedule names the real venue and its city for every game. A game whose venue city is not in this
table has NO wind figure, and an Over on it is "insufficient data" (never assumed calm). The table covers the NFL
stadium cities (Inglewood and East Rutherford each host two teams) and the cities that host NFL international games.
COLLEGE stadiums are NOT in this table (a college team's city can also hold an NFL stadium with a different roof);
college stadiums live in college_stadiums.py (keyed by home team).

Coordinates are approximate (two to four decimals; a wind forecast does not need more) and were entered by hand from
public knowledge of the stadiums, not fetched: a wrong row would only move the forecast a few kilometres.
Roofs: "open" = outdoors; "dome" = fixed roof, indoors (wind is 0 mph by construction, no forecast is read);
"retractable" = the roof may be open or shut, and the schedule does not say which, so the OUTDOOR forecast is used
(the cautious reading for an Over).
"""
from __future__ import annotations

from dataclasses import dataclass

from predictions.core.types import normalize_team


@dataclass(frozen=True)
class Stadium:
    name: str
    latitude: float
    longitude: float
    roof: str  # "open", "dome" or "retractable"


ROOFS = ("open", "dome", "retractable")

_ROWS = (
    # city as ESPN spells it, stadium, latitude, longitude, roof
    ("Glendale", "State Farm Stadium", 33.5276, -112.2626, "retractable"),
    ("Atlanta", "Mercedes-Benz Stadium", 33.7554, -84.4008, "retractable"),
    ("Baltimore", "M&T Bank Stadium", 39.2780, -76.6227, "open"),
    ("Orchard Park", "Highmark Stadium", 42.7738, -78.7870, "open"),
    ("Charlotte", "Bank of America Stadium", 35.2258, -80.8528, "open"),
    ("Chicago", "Soldier Field", 41.8623, -87.6167, "open"),
    ("Cincinnati", "Paycor Stadium", 39.0955, -84.5161, "open"),
    ("Cleveland", "Huntington Bank Field", 41.5061, -81.6995, "open"),
    ("Arlington", "AT&T Stadium", 32.7473, -97.0945, "retractable"),
    ("Denver", "Empower Field at Mile High", 39.7439, -105.0201, "open"),
    ("Detroit", "Ford Field", 42.3400, -83.0456, "dome"),
    ("Green Bay", "Lambeau Field", 44.5013, -88.0622, "open"),
    ("Houston", "NRG Stadium", 29.6847, -95.4107, "retractable"),
    ("Indianapolis", "Lucas Oil Stadium", 39.7601, -86.1639, "retractable"),
    ("Jacksonville", "EverBank Stadium", 30.3239, -81.6373, "open"),
    ("Kansas City", "GEHA Field at Arrowhead Stadium", 39.0489, -94.4839, "open"),
    ("Las Vegas", "Allegiant Stadium", 36.0909, -115.1833, "dome"),
    ("Inglewood", "SoFi Stadium", 33.9535, -118.3392, "open"),
    ("Miami Gardens", "Hard Rock Stadium", 25.9580, -80.2389, "open"),
    ("Minneapolis", "U.S. Bank Stadium", 44.9735, -93.2575, "dome"),
    ("Foxborough", "Gillette Stadium", 42.0909, -71.2643, "open"),
    ("New Orleans", "Caesars Superdome", 29.9511, -90.0812, "dome"),
    ("East Rutherford", "MetLife Stadium", 40.8135, -74.0745, "open"),
    ("Philadelphia", "Lincoln Financial Field", 39.9008, -75.1675, "open"),
    ("Pittsburgh", "Acrisure Stadium", 40.4468, -80.0158, "open"),
    ("Santa Clara", "Levi's Stadium", 37.4030, -121.9700, "open"),
    ("Seattle", "Lumen Field", 47.5952, -122.3316, "open"),
    ("Tampa", "Raymond James Stadium", 27.9759, -82.5033, "open"),
    ("Nashville", "Nissan Stadium", 36.1665, -86.7713, "open"),
    ("Landover", "Northwest Stadium", 38.9076, -76.8645, "open"),
    # cities that host NFL international games (one coordinate per city: the stadiums are a few km apart)
    ("London", "London (Tottenham Hotspur Stadium / Wembley)", 51.5800, -0.1700, "open"),
    ("Munich", "Allianz Arena", 48.2188, 11.6247, "open"),
    ("Frankfurt", "Deutsche Bank Park", 50.0686, 8.6455, "open"),
    ("Mexico City", "Estadio Azteca", 19.3029, -99.1505, "open"),
    ("Madrid", "Santiago Bernabeu", 40.4530, -3.6883, "retractable"),
    ("Dublin", "Croke Park", 53.3607, -6.2510, "open"),
    ("Rio de Janeiro", "Maracana", -22.9121, -43.2302, "open"),
    ("Sao Paulo", "Neo Quimica Arena", -23.5453, -46.4742, "open"),
    ("Melbourne", "Melbourne Cricket Ground", -37.8200, 144.9834, "open"),
)

NFL_VENUES: dict[str, Stadium] = {normalize_team(city): Stadium(name, lat, lon, roof)
                                  for city, name, lat, lon, roof in _ROWS}


def stadium_for_city(city: str | None) -> Stadium | None:
    """The stadium for a venue city as ESPN spells it, or None (unknown city: no wind figure)."""
    return NFL_VENUES.get(normalize_team(city)) if city else None
