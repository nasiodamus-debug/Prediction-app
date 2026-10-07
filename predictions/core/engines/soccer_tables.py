"""Static soccer reference tables for the Optimization stage (Story 2.4). Standard library only.

These are hand-written judgement data, NOT measured here. Each table says what it is and what it is not.
The numbers that decide behaviour (the tempo cut-off, the tier cut-off) are named constants at the top of
`core/engines/soccer.py`; this file only holds the data they are applied to.
"""
from __future__ import annotations

from dataclasses import dataclass

from predictions.core.identity import resolve_team
from predictions.core.types import normalize_team


@dataclass(frozen=True)
class LeagueProfile:
    tier: int          # 1 = the five biggest European leagues, 2 = other first divisions, 3 = lower/other
    avg_goals: float   # rough multi-season goals per match: the league's TEMPO. Approximate, rounded.


# Keyed by the normalized SportyBet competition name, "<country> - <league>" (see normalize_team).
# PROXY / APPROXIMATE: tiers are a football-standing judgement and avg_goals are rounded long-run averages
# from memory of published league tables, not fetched. The owner should refresh them from time to time.
# A league that is not listed is treated as tier 3 (see soccer.league_profile).
LEAGUE_PROFILES: dict[str, LeagueProfile] = {
    "england premier league": LeagueProfile(1, 2.85),
    "germany bundesliga": LeagueProfile(1, 3.15),
    "spain laliga": LeagueProfile(1, 2.55),
    "spain la liga": LeagueProfile(1, 2.55),
    "italy serie a": LeagueProfile(1, 2.65),
    "france ligue 1": LeagueProfile(1, 2.75),
    "netherlands eredivisie": LeagueProfile(2, 3.15),
    "portugal liga portugal": LeagueProfile(2, 2.55),
    "portugal primeira liga": LeagueProfile(2, 2.55),
    "belgium jupiler pro league": LeagueProfile(2, 2.9),
    "turkey super lig": LeagueProfile(2, 2.85),
    "scotland premiership": LeagueProfile(2, 2.8),
    "russia premier league": LeagueProfile(2, 2.55),
    "england championship": LeagueProfile(2, 2.55),
    "germany 2 bundesliga": LeagueProfile(2, 3.05),
    "usa mls": LeagueProfile(2, 3.0),
    "brazil serie a": LeagueProfile(2, 2.4),
    "brazil brasileirao serie a": LeagueProfile(2, 2.4),
    "argentina liga profesional": LeagueProfile(2, 2.2),
    "mexico liga mx": LeagueProfile(2, 2.8),
    "norway eliteserien": LeagueProfile(2, 3.0),
    "sweden allsvenskan": LeagueProfile(2, 2.9),
    "denmark superliga": LeagueProfile(2, 2.9),
    "switzerland super league": LeagueProfile(2, 3.0),
    "austria bundesliga": LeagueProfile(2, 3.0),
    "greece super league": LeagueProfile(2, 2.2),
    "japan j1 league": LeagueProfile(2, 2.6),
    "france ligue 2": LeagueProfile(3, 2.45),
    "spain laliga 2": LeagueProfile(3, 2.35),
    "italy serie b": LeagueProfile(3, 2.4),
    "england league one": LeagueProfile(3, 2.6),
}


@dataclass(frozen=True)
class Derby:
    name: str
    country: str         # normalized country as SportyBet prefixes its competitions ("england")
    teams: tuple[str, str]  # names as core/identity's ALIASES targets spell them (normalized)


# A short, hand-written list of the well-known rivalries in the leagues the bot covers. It is a STATIC TABLE:
# a local rivalry that is missing here is not detected. A pair counts in any competition of that country
# (league or domestic cup); European ties between clubs of one country are not covered.
DERBIES: tuple[Derby, ...] = (
    Derby("North London derby", "england", ("arsenal", "tottenham")),
    Derby("Merseyside derby", "england", ("liverpool", "everton")),
    Derby("Manchester derby", "england", ("manchester united", "manchester city")),
    Derby("North West derby", "england", ("manchester united", "liverpool")),
    Derby("Tyne-Wear derby", "england", ("newcastle united", "sunderland")),
    Derby("London derby (Arsenal v Chelsea)", "england", ("arsenal", "chelsea")),
    Derby("London derby (Chelsea v Tottenham)", "england", ("chelsea", "tottenham")),
    Derby("London derby (West Ham v Tottenham)", "england", ("west ham", "tottenham")),
    Derby("London derby (West Ham v Chelsea)", "england", ("west ham", "chelsea")),
    Derby("London derby (Fulham v Chelsea)", "england", ("fulham", "chelsea")),
    Derby("M23 derby", "england", ("crystal palace", "brighton")),
    Derby("Roses derby", "england", ("leeds", "manchester united")),
    Derby("West Midlands derby", "england", ("aston villa", "wolverhampton")),
    Derby("El Clasico", "spain", ("real madrid", "barcelona")),
    Derby("Madrid derby", "spain", ("real madrid", "atletico madrid")),
    Derby("Catalan derby", "spain", ("barcelona", "espanyol")),
    Derby("Seville derby", "spain", ("sevilla", "real betis")),
    Derby("Basque derby", "spain", ("athletic club", "real sociedad")),
    Derby("Der Klassiker", "germany", ("bayern munich", "borussia dortmund")),
    Derby("Revierderby", "germany", ("borussia dortmund", "schalke 04")),
    Derby("Nordderby", "germany", ("hamburger sv", "werder bremen")),
    Derby("Rhine derby", "germany", ("fc cologne", "borussia m gladbach")),
    Derby("Berlin derby", "germany", ("union berlin", "hertha berlin")),
    Derby("Derby della Madonnina", "italy", ("ac milan", "inter")),
    Derby("Derby della Capitale", "italy", ("as roma", "lazio")),
    Derby("Derby della Mole", "italy", ("juventus", "torino")),
    Derby("Derby d'Italia", "italy", ("juventus", "inter")),
    Derby("Derby della Lanterna", "italy", ("genoa", "sampdoria")),
    Derby("Le Classique", "france", ("paris saint germain", "marseille")),
    Derby("Derby du Rhone", "france", ("lyon", "saint etienne")),
    Derby("Derby du Nord", "france", ("lille", "lens")),
    Derby("Derby de l'Ouest", "france", ("nantes", "rennes")),
    Derby("Moscow derby (Spartak v CSKA)", "russia", ("spartak moscow", "cska moscow")),
    Derby("Moscow derby (Spartak v Dynamo)", "russia", ("spartak moscow", "dynamo moscow")),
    Derby("Moscow derby (Spartak v Lokomotiv)", "russia", ("spartak moscow", "lokomotiv moscow")),
    Derby("Moscow derby (CSKA v Dynamo)", "russia", ("cska moscow", "dynamo moscow")),
    Derby("O Classico", "portugal", ("benfica", "porto")),
    Derby("Lisbon derby", "portugal", ("benfica", "sporting cp")),
    Derby("Porto v Sporting", "portugal", ("porto", "sporting cp")),
    Derby("De Klassieker", "netherlands", ("ajax", "feyenoord")),
    Derby("Old Firm", "scotland", ("celtic", "rangers")),
    Derby("Intercontinental derby (Galatasaray v Fenerbahce)", "turkey", ("galatasaray", "fenerbahce")),
    Derby("Istanbul derby (Besiktas v Fenerbahce)", "turkey", ("besiktas", "fenerbahce")),
    Derby("Istanbul derby (Galatasaray v Besiktas)", "turkey", ("galatasaray", "besiktas")),
)

# Words in a competition's name that make it a cup. A PROXY for "cup tie": it also catches cup finals and
# tournaments between national teams, and misses a two-legged European tie in a competition not named "cup".
CUP_WORDS = frozenset({"cup", "copa", "coppa", "pokal", "coupe", "taca", "beker", "cupa", "kupa", "kubok",
                       "trophy"})


def is_cup(competition: str) -> bool:
    return bool(CUP_WORDS & set(normalize_team(competition).split()))


def find_derby(competition: str, home: str, away: str) -> Derby | None:
    """The derby this game is, or None. Needs the country of the competition and both clubs to match."""
    comp = normalize_team(competition)
    for derby in DERBIES:
        if not (comp == derby.country or comp.startswith(derby.country + " ")):
            continue
        a, b = derby.teams
        h, w = resolve_team(home, derby.teams), resolve_team(away, derby.teams)
        if h.team and w.team and {h.team, w.team} == {a, b}:
            return derby
    return None
