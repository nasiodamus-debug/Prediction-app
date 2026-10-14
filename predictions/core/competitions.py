"""Which competitions Matchday (soccer), Gameday (basketball) and America (MLB, NHL, American football) may look at
(spec-10-1, spec-12-1). Standard library only.

WHY: every game outside this list would cost stats look-ups and, worse, paid analyst calls. A game outside the list is not
listed, not looked up and not sent to the analyst. Review is NOT restricted: every booking-code selection is judged.

HOW TO EDIT: change the tables below, nothing else. SportyBet names a competition "<Country> - <League>" (for example
"England - Premier League", "International Clubs - UEFA Champions League Women"). Each entry is the NORMALISED form of that
text (lower case, no accents, every punctuation mark a single space; see `normalize`), so "Turkiye - Super Lig" is
"turkiye super lig". Matching is EXACT, so "england premier league" never admits "England - Premier League, Reserves" or
"Premier League 2". To allow another league, add its normalised "<country> <league>" to the right table with a label. To
find SportyBet's exact spelling, run Matchday once and read the competition headings in the NO VALID PLAY reply or the
SportyBet website itself. A separate safety net (`BLOCKED_WORDS`) refuses any string that mentions reserves, youth ages,
amateur, women (outside the women's entries), simulated or virtual football, even if a table entry would otherwise match.
"""
from __future__ import annotations

import re
import unicodedata

# ----------------------------------------------------------------------------------------------------------------------
# Normalisation (the same rule as predictions.core.types.normalize_team, repeated here so this file stands alone)
# ----------------------------------------------------------------------------------------------------------------------


def normalize(text) -> str:
    decomposed = unicodedata.normalize("NFKD", str(text))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    kept = "".join(c if c.isalnum() else " " for c in stripped.casefold())
    return " ".join(kept.split())


# Words that can never be in an allowed competition (checked on the whole normalised string, as whole words).
BLOCKED_WORDS = frozenset({"reserve", "reserves", "youth", "junior", "juniors", "amateur", "srl", "simulated", "virtual",
                           "esoccer", "ebasketball"})
_AGE_GROUP = re.compile(r"^u\d{2}$")
_WOMEN_WORDS = frozenset({"women", "womens", "woman", "ladies", "feminine", "femenino", "femenina", "frauen", "w"})

# ----------------------------------------------------------------------------------------------------------------------
# SOCCER (Matchday): men's top divisions, men's continental club cups, senior national-team competitions, UEFA Women's CL
# ----------------------------------------------------------------------------------------------------------------------
# (a) The top division of a men's national league (Europe, South America, North America). Key = normalised
# "<country> <league>" as SportyBet writes it. Value = label shown in the reasons. Spellings marked [seen] were on SportyBet's
# listing on 2026-10-01; the others are likely spellings that were not listed that day (they do nothing until SportyBet
# lists them with exactly that name; a different spelling needs one more line here).
SOCCER_DOMESTIC: dict[str, str] = {
    # --- Europe
    "england premier league": "England Premier League",                      # [seen]
    "spain laliga": "Spain LaLiga",                                           # [seen]
    "italy serie a": "Italy Serie A",                                         # [seen]
    "germany bundesliga": "Germany Bundesliga",                               # [seen]
    "france ligue 1": "France Ligue 1",                                       # [seen]
    "netherlands eredivisie": "Netherlands Eredivisie",                       # [seen]
    "portugal liga portugal": "Portugal Primeira Liga",                       # [seen]
    "belgium pro league": "Belgium Pro League",                               # [seen]
    "turkiye super lig": "Turkey Super Lig",                                  # [seen]
    "scotland premiership": "Scotland Premiership",                           # [seen]
    "austria bundesliga": "Austria Bundesliga",                               # [seen]
    "switzerland super league": "Switzerland Super League",                   # [seen]
    "denmark superliga": "Denmark Superliga",                                 # [seen]
    "norway eliteserien": "Norway Eliteserien",                               # [seen]
    "sweden allsvenskan": "Sweden Allsvenskan",                               # [seen]
    "finland veikkausliiga": "Finland Veikkausliiga",                         # [seen]
    "greece super league": "Greece Super League",                             # [seen]
    "russia premier league": "Russia Premier League",                         # [seen]
    "ukraine premier league": "Ukraine Premier League",                       # [seen]
    "poland ekstraklasa": "Poland Ekstraklasa",                               # [seen]
    "czechia 1 liga": "Czechia 1. Liga",                                      # [seen]
    "serbia superliga": "Serbia Superliga",                                   # [seen]
    "romania superliga": "Romania Superliga",                                 # [seen]
    "bulgaria parva liga": "Bulgaria First League",                           # [seen]
    "slovakia superliga": "Slovakia Superliga",                               # [seen]
    "slovenia prvaliga": "Slovenia PrvaLiga",                                 # [seen]
    "cyprus 1st division": "Cyprus First Division",                           # [seen]
    "ireland premier division": "Ireland Premier Division",                   # [seen]
    "northern ireland premiership": "Northern Ireland Premiership",           # [seen]
    "wales cymru premier": "Wales Cymru Premier",                             # [seen]
    "estonia premium liiga": "Estonia Premium Liiga",                         # [seen]
    "croatia hnl": "Croatia HNL",                                             # not seen
    "croatia 1 hnl": "Croatia HNL",                                           # not seen
    "croatia supersport hnl": "Croatia HNL",                                  # not seen
    "hungary nb i": "Hungary NB I",                                           # not seen
    "iceland besta deild": "Iceland Besta deild",                             # not seen (only the women's one was listed)
    "lithuania a lyga": "Lithuania A Lyga",                                   # not seen (1 Lyga is the 2nd division)
    "latvia virsliga": "Latvia Virsliga",                                     # not seen
    "albania superliga": "Albania Superliga",                                 # not seen
    "bosnia and herzegovina premier liga": "Bosnia Premier Liga",             # not seen (Prva Liga is the 2nd division)
    "north macedonia first league": "North Macedonia First League",           # not seen
    "montenegro 1 cfl": "Montenegro First League",                            # not seen
    "belarus premier league": "Belarus Premier League",                       # not seen
    "moldova super liga": "Moldova Super Liga",                               # not seen
    "georgia erovnuli liga": "Georgia Erovnuli Liga",                         # not seen
    "armenia premier league": "Armenia Premier League",                       # not seen
    "azerbaijan premier league": "Azerbaijan Premier League",                 # not seen
    "israel ligat haal": "Israel Ligat ha'Al",                                # not seen
    "luxembourg bgl ligue": "Luxembourg National Division",                   # not seen
    "malta premier league": "Malta Premier League",                           # not seen
    "faroe islands premier league": "Faroe Islands Premier League",           # not seen
    # --- South America
    "brazil brasileiro serie a": "Brazil Serie A",                            # [seen]
    "argentina primera lpf": "Argentina Primera Division",                    # [seen]
    "argentina primera division": "Argentina Primera Division",               # not seen (older spelling)
    "chile primera division": "Chile Primera Division",                       # [seen]
    "colombia liga dimayor": "Colombia Primera A",                            # [seen] (Torneo DIMAYOR is the 2nd division)
    "ecuador ligapro primera a": "Ecuador LigaPro Serie A",                   # [seen]
    "paraguay division de honor": "Paraguay Division de Honor",               # [seen]
    "peru liga 1": "Peru Liga 1",                                             # [seen]
    "uruguay primera division": "Uruguay Primera Division",                   # [seen]
    "bolivia division profesional": "Bolivia Division Profesional",           # not seen
    "venezuela liga futve": "Venezuela Liga FUTVE",                           # not seen
    # --- North America (incl. Central America and the Caribbean where SportyBet files them by country)
    "usa mls": "USA MLS",                                                     # [seen]
    "mexico liga mx": "Mexico Liga MX",                                       # [seen]
    "canada canadian premier league": "Canada Premier League",                # [seen]
    "costa rica primera division": "Costa Rica Primera Division",             # not seen (Liga de Ascenso is the 2nd division)
    "costa rica liga promerica": "Costa Rica Primera Division",               # not seen
    "el salvador primera division": "El Salvador Primera Division",           # [seen]
    "panama liga prom": "Panama Liga Prom",                                   # [seen]
    "honduras liga nacional": "Honduras Liga Nacional",                       # not seen
    "guatemala liga nacional": "Guatemala Liga Nacional",                     # not seen
    "jamaica premier league": "Jamaica Premier League",                       # not seen
}

# (b) Men's continental club cups (category "International Clubs").
SOCCER_CONTINENTAL: dict[str, str] = {
    # --- Domestic league cups (top-flight clubs; the Toto Cup "League Cup, National", EFL Trophy and National League Cup are not)
    "england efl cup": "England EFL Cup (Carabao Cup)",                      # [seen]
    "england league cup": "England EFL Cup (Carabao Cup)",                   # not seen
    "scotland league cup": "Scotland League Cup",                            # not seen
    "scotland premier sports cup": "Scotland League Cup",                    # not seen
    "japan j league cup": "Japan J.League Cup",                              # [seen]
    "portugal taca da liga": "Portugal Taca da Liga",                        # not seen
    "portugal league cup": "Portugal Taca da Liga",                          # not seen
    "france coupe de la ligue": "France Coupe de la Ligue",                  # not seen
    "argentina copa de la liga": "Argentina Copa de la Liga",                # not seen
    "argentina copa de la liga profesional": "Argentina Copa de la Liga",    # not seen
    "international clubs uefa champions league": "UEFA Champions League",           # [seen]
    "international clubs uefa europa league": "UEFA Europa League",                 # [seen]
    "international clubs uefa conference league": "UEFA Conference League",         # [seen]
    "international clubs uefa europa conference league": "UEFA Conference League",  # not seen (older spelling)
    "international clubs conmebol libertadores": "Copa Libertadores",               # [seen]
    "international clubs copa libertadores": "Copa Libertadores",                   # not seen
    "international clubs conmebol sudamericana": "Copa Sudamericana",              # [seen]
    "international clubs copa sudamericana": "Copa Sudamericana",                   # not seen
    "international clubs concacaf champions cup": "CONCACAF Champions Cup",         # not seen
    "international clubs concacaf champions league": "CONCACAF Champions Cup",      # not seen (old name)
    "international clubs caf champions league": "CAF Champions League",             # not seen
}

# (d) The UEFA Women's Champions League is the only women's club competition allowed.
SOCCER_WOMEN_CLUB: dict[str, str] = {
    "international clubs uefa champions league women": "UEFA Women's Champions League",  # [seen]
    "international clubs uefa womens champions league": "UEFA Women's Champions League",  # not seen
    "international clubs uefa women s champions league": "UEFA Women's Champions League",  # not seen
}

# (c) Senior national-team competitions, men AND women. SportyBet files them under the category "International" (youth
# sides are "International Youth", which is never allowed). A league name is allowed when it starts with one of these
# normalised competition names and whatever FOLLOWS is only words from NATIONAL_SUFFIX_WORDS (a region, "qualification",
# "women", a group letter...). So "FIFA World Cup Qualification CAF" and "UEFA Nations League" pass, while "World Cup U20",
# "Asian Games, Women" and "Premier League 2" do not.
NATIONAL_CATEGORY = "international"
NATIONAL_COMPETITIONS: dict[str, str] = {
    "fifa world cup": "FIFA World Cup (and qualifiers)",
    "world cup": "FIFA World Cup (and qualifiers)",
    "european championship": "UEFA European Championship (and qualifiers)",
    "uefa european championship": "UEFA European Championship (and qualifiers)",
    "uefa euro": "UEFA European Championship (and qualifiers)",
    "euro": "UEFA European Championship (and qualifiers)",
    "uefa nations league": "UEFA Nations League",
    "concacaf nations league": "CONCACAF Nations League",
    "concacaf gold cup": "CONCACAF Gold Cup",
    "concacaf championship": "CONCACAF competitions",
    "concacaf": "CONCACAF competitions",
    "africa cup of nations": "Africa Cup of Nations (and qualifiers)",
    "african cup of nations": "Africa Cup of Nations (and qualifiers)",
    "afcon": "Africa Cup of Nations (and qualifiers)",
    "copa america": "Copa America",
    "conmebol copa america": "Copa America",
    "asian cup": "AFC Asian Cup (and qualifiers)",
    "afc asian cup": "AFC Asian Cup (and qualifiers)",
    "arab cup": "FIFA Arab Cup",
    "int friendly games": "International friendlies (senior)",
    "international friendly games": "International friendlies (senior)",
    "international friendlies": "International friendlies (senior)",
    "friendly international": "International friendlies (senior)",
}
NATIONAL_SUFFIX_WORDS = frozenset({
    "qualification", "qualifications", "qualifier", "qualifiers", "qualifying", "women", "womens", "uefa", "caf", "afc",
    "concacaf", "conmebol", "ofc", "fifa", "europe", "european", "africa", "african", "asia", "asian", "america",
    "north", "south", "central", "oceania", "group", "groups", "stage", "round", "play", "offs", "playoff", "playoffs",
    "final", "finals", "preliminary", "intercontinental", "league", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j",
    "k", "l", "1", "2", "3", "4", "1st", "2nd", "3rd", "s",
})

# ----------------------------------------------------------------------------------------------------------------------
# BASKETBALL (Gameday): the NBA, the WNBA, the EuroLeague and the FIRST division of each European country's men's league.
# Key = normalised "<country> <league>" as SportyBet writes it. [seen] = on SportyBet's listing on 2026-10-01 (spec-12-1).
# Second divisions (LNB Elite 2, Pro A, Primera FEB, Superettan...), the EuroCup, NCAA, G League, youth, 3x3 and every
# women's league except the WNBA are NOT here, so they are ignored.
# ----------------------------------------------------------------------------------------------------------------------
BASKETBALL_ALLOWED: dict[str, str] = {
    "usa nba": "NBA",                                                         # [seen]
    "usa wnba": "WNBA",                                                       # [seen]
    # "australia wnbl": "Australian WNBL",   # enable if the owner wants the Australian WNBL (also add its key to _WOMEN_OK)
    "international euroleague": "EuroLeague",                                 # [seen]
    "international bnxt league": "BNXT League (Belgium and Netherlands top league)",  # [seen]
    "international liga aba": "Adriatic ABA League",                          # [seen]
    "spain liga acb": "Spain ACB",                                            # [seen]
    "italy serie a": "Italy LBA (Lega Basket Serie A)",                       # [seen]
    "germany bbl": "Germany BBL",                                             # [seen]
    "france lnb elite": "France LNB Elite (Betclic Elite)",                   # [seen] (LNB Elite 2 is the 2nd division)
    "greece basketball league": "Greece Basket League",                       # [seen] (Elite League is not the top league)
    "turkiye super lig": "Turkey BSL",                                        # [seen] (the women's one reads "Super Lig, Women")
    "lithuania lkl": "Lithuania LKL",                                         # [seen] (NKL is the 2nd division)
    "poland plk": "Poland PLK",                                               # [seen] (1. Liga is the 2nd division)
    "austria superliga": "Austria Superliga",                                 # [seen]
    "belgium top division 1": "Belgium Top Division 1",                       # [seen]
    "croatia premijer liga": "Croatia Premijer liga",                         # [seen]
    "denmark basketligaen": "Denmark Basketligaen",                           # [seen]
    "england slb": "England Super League Basketball",                         # [seen]
    "finland korisliiga": "Finland Korisliiga",                               # [seen] (the women's one reads "Korisliiga, Women")
    "georgia superleague": "Georgia Superleague",                             # [seen]
    "hungary nb i a": "Hungary NB I. A",                                      # [seen] (the women's one reads "NB I. A, Women")
    "sweden sbl": "Sweden SBL",                                               # [seen] (Superettan is the 2nd division)
    # Not on the listing on 2026-10-01 (off season, or filed under another name): they do nothing until SportyBet lists them
    # with exactly this spelling.
    "serbia kls": "Serbia KLS",                                               # not seen (only the women's league was listed)
    "slovenia liga nova kbm": "Slovenia Liga Nova KBM",                       # not seen
    "israel super league": "Israel Super League",                             # not seen
    "portugal lpb": "Portugal LPB",                                           # [seen] 2026-10-01 (live probe)
    "netherlands dbl": "Netherlands DBL",                                     # not seen
    "ukraine superleague": "Ukraine Superleague",                             # not seen
    "latvia lbl": "Latvia LBL",                                               # not seen
    "bulgaria nbl": "Bulgaria NBL",                                           # not seen
    "bosnia and herzegovina premier league": "Bosnia Premier League",         # not seen
    "montenegro prva liga": "Montenegro Prva Liga",                           # not seen
    "romania lnb": "Romania LNB",                                             # not seen (only the women's league was listed)
    "russia vtb united league": "Russia VTB United League",                   # not seen (Russia - Super League is the 2nd tier)
    "international vtb united league": "Russia VTB United League",            # not seen
}

# ----------------------------------------------------------------------------------------------------------------------
# AMERICA: one table per sport (the key is the same normalised "<country> <league>"). Everything else, such as the CFL, the
# KHL, SHL, Liiga, AHL, junior leagues, NCAA hockey, KBO, NPB, minor leagues, is ignored.
# ----------------------------------------------------------------------------------------------------------------------
BASEBALL_ALLOWED: dict[str, str] = {
    "usa mlb": "MLB",                                                         # [seen] (NPB, KBO, CPBL and minor leagues are out)
}
HOCKEY_ALLOWED: dict[str, str] = {
    "usa nhl": "NHL",                                                         # [seen] (AHL, KHL, SHL, Liiga, NCAA, juniors are out)
}
FOOTBALL_ALLOWED: dict[str, str] = {
    "usa nfl": "NFL",                                                         # [seen]
    "usa ncaa regular season": "NCAA football (FBS and FCS)",                 # [seen] (SportyBet does not split FBS and FCS; CFL is out)
    "usa ncaa postseason": "NCAA football (bowls and playoff)",               # not seen (off season)
    "usa ncaa bowl games": "NCAA football (bowls and playoff)",               # not seen
    "usa ncaa playoffs": "NCAA football (bowls and playoff)",                 # not seen
}

TABLES = {"soccer": SOCCER_DOMESTIC, "basketball": BASKETBALL_ALLOWED, "baseball": BASEBALL_ALLOWED,
          "hockey": HOCKEY_ALLOWED, "american_football": FOOTBALL_ALLOWED}
# Review is NOT restricted: the owner wants every booking-code selection judged whatever its competition or country.
RESTRICTED_TRIGGERS = frozenset({"matchday", "gameday", "america"})
_WOMEN_OK = frozenset({"usa wnba"})   # the only women's competition allowed outside soccer

REASON_OUTSIDE = "outside your allowed competitions"


def _blocked(words: list[str]) -> str:
    for w in words:
        if w in BLOCKED_WORDS or _AGE_GROUP.match(w):
            return w
    return ""


def _national(league_words: list[str]) -> str:
    """The label of the national-team competition this league name starts with, '' when none (longest name first)."""
    while league_words and league_words[0] in ("women", "womens"):  # "Women World Cup Qualification Europe"
        league_words = league_words[1:]
    text = " ".join(league_words)
    for name in sorted(NATIONAL_COMPETITIONS, key=len, reverse=True):
        if text == name or text.startswith(name + " "):
            rest = text[len(name):].split()
            if all(w in NATIONAL_SUFFIX_WORDS for w in rest):
                return NATIONAL_COMPETITIONS[name]
    return ""


def _soccer(competition: str, words: list[str], key: str) -> tuple[bool, str]:
    blocked = _blocked(words)
    if blocked:
        return False, f"{REASON_OUTSIDE} (reserve, youth, amateur or simulated: '{blocked}')"
    if key in SOCCER_WOMEN_CLUB:
        return True, SOCCER_WOMEN_CLUB[key]
    if key in SOCCER_CONTINENTAL:
        return True, SOCCER_CONTINENTAL[key]
    # Everything below is a men's competition, except the national-team ones where women are allowed.
    if key in SOCCER_DOMESTIC:
        if any(w in _WOMEN_WORDS for w in words):
            return False, f"{REASON_OUTSIDE} (women's league)"
        return True, SOCCER_DOMESTIC[key]
    return False, REASON_OUTSIDE  # national-team competitions are read in is_allowed, from the raw "Country - League"


def is_allowed(trigger: str, sport: str, competition) -> tuple[bool, str]:
    """(allowed, reason) for a game of `sport` in SportyBet's `competition` ("Country - League") asked for by `trigger`
    ("Matchday", "Gameday" or "America"). Review is never restricted, and a sport with no table is allowed unchanged.
    An empty or unknown competition is NOT allowed."""
    if str(trigger).strip().lower() not in RESTRICTED_TRIGGERS or sport not in TABLES:
        return True, "not restricted"
    text = str(competition or "")
    country, sep, league = text.partition(" - ")
    key = normalize(text)
    words = key.split()
    if not words:
        return False, REASON_OUTSIDE
    if sport != "soccer":
        blocked = _blocked(words)
        if blocked:
            return False, f"{REASON_OUTSIDE} ('{blocked}')"
        table = TABLES[sport]
        if key in table:
            if any(w in _WOMEN_WORDS for w in words) and key not in _WOMEN_OK:
                return False, f"{REASON_OUTSIDE} (women's league)"
            return True, table[key]
        return False, REASON_OUTSIDE
    # soccer
    allowed, reason = _soccer(text, words, key)
    if allowed or not sep:
        return allowed, reason
    if normalize(country) == NATIONAL_CATEGORY:
        league_words = normalize(league).split()
        blocked = _blocked(league_words)
        if blocked:
            return False, f"{REASON_OUTSIDE} (youth, reserve or simulated: '{blocked}')"
        label = _national(league_words)
        if label:
            return True, label
    return False, REASON_OUTSIDE
