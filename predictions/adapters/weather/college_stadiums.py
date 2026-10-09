"""College-football home stadiums for the wind lookup, keyed by the HOME TEAM as ESPN and SportyBet spell it.

Why by team and not by city (as the NFL table is): a college team's city can also hold an NFL stadium with a different roof,
so a city key could pick the wrong roof. The game is still checked against the schedule: when ESPN names the venue city and
it is not this team's home city (a neutral-site or relocated game), NO wind figure is given. A team not in this table has
no wind figure and its Overs stay "insufficient data" (never assumed calm).

Coverage: the FBS teams plus the FCS teams that have appeared on the book. Coordinates are approximate (a wind forecast does
not need more than a few km) and were entered by hand from public knowledge of the stadiums, not fetched.
Roofs: "open" = outdoors; "dome" = fixed roof, indoors (wind is 0 mph by construction, no forecast is read).
"""
from __future__ import annotations

from predictions.adapters.weather.stadiums import Stadium
from predictions.core.types import normalize_team

_ROWS = (
    # team as ESPN spells it, stadium, home city, latitude, longitude, roof
    ("Air Force Falcons", "Falcon Stadium", "Colorado Springs", 38.9969, -104.8439, "open"),
    ("Akron Zips", "InfoCision Stadium", "Akron", 41.0725, -81.5085, "open"),
    ("Alabama Crimson Tide", "Bryant-Denny Stadium", "Tuscaloosa", 33.2084, -87.5503, "open"),
    ("Alabama State Hornets", "ASU Stadium", "Montgomery", 32.3643, -86.2946, "open"),
    ("App State Mountaineers", "Kidd Brewer Stadium", "Boone", 36.2128, -81.6853, "open"),
    ("Arizona State Sun Devils", "Mountain America Stadium", "Tempe", 33.4264, -111.9325, "open"),
    ("Arizona Wildcats", "Arizona Stadium", "Tucson", 32.2288, -110.9488, "open"),
    ("Arkansas Razorbacks", "Razorback Stadium", "Fayetteville", 36.0680, -94.1788, "open"),
    ("Arkansas State Red Wolves", "Centennial Bank Stadium", "Jonesboro", 35.8478, -90.6773, "open"),
    ("Army Black Knights", "Michie Stadium", "West Point", 41.3874, -73.9633, "open"),
    ("Auburn Tigers", "Jordan-Hare Stadium", "Auburn", 32.6023, -85.4897, "open"),
    ("Ball State Cardinals", "Scheumann Stadium", "Muncie", 40.2120, -85.4090, "open"),
    ("Baylor Bears", "McLane Stadium", "Waco", 31.5584, -97.1155, "open"),
    ("Boise State Broncos", "Albertsons Stadium", "Boise", 43.6020, -116.1963, "open"),
    ("Boston College Eagles", "Alumni Stadium", "Chestnut Hill", 42.3350, -71.1664, "open"),
    ("Bowling Green Falcons", "Doyt Perry Stadium", "Bowling Green", 41.3794, -83.6280, "open"),
    ("Buffalo Bulls", "UB Stadium", "Amherst", 43.0000, -78.7770, "open"),
    ("BYU Cougars", "LaVell Edwards Stadium", "Provo", 40.2576, -111.6546, "open"),
    ("California Golden Bears", "California Memorial Stadium", "Berkeley", 37.8714, -122.2509, "open"),
    ("Central Michigan Chippewas", "Kelly/Shorts Stadium", "Mount Pleasant", 43.5880, -84.7748, "open"),
    ("Charlotte 49ers", "Jerry Richardson Stadium", "Charlotte", 35.3098, -80.7250, "open"),
    ("Cincinnati Bearcats", "Nippert Stadium", "Cincinnati", 39.1316, -84.5160, "open"),
    ("Clemson Tigers", "Memorial Stadium", "Clemson", 34.6786, -82.8433, "open"),
    ("Coastal Carolina Chanticleers", "Brooks Stadium", "Conway", 33.7937, -79.0169, "open"),
    ("Colorado Buffaloes", "Folsom Field", "Boulder", 40.0094, -105.2669, "open"),
    ("Colorado State Rams", "Canvas Stadium", "Fort Collins", 40.5710, -105.1370, "open"),
    ("Delaware Blue Hens", "Delaware Stadium", "Newark", 39.6781, -75.7497, "open"),
    ("Duke Blue Devils", "Wallace Wade Stadium", "Durham", 36.0014, -78.9423, "open"),
    ("East Carolina Pirates", "Dowdy-Ficklen Stadium", "Greenville", 35.5967, -77.3669, "open"),
    ("Eastern Michigan Eagles", "Rynearson Stadium", "Ypsilanti", 42.2515, -83.6240, "open"),
    ("Florida A&M Rattlers", "Bragg Memorial Stadium", "Tallahassee", 30.4270, -84.2860, "open"),
    ("Florida Atlantic Owls", "FAU Stadium", "Boca Raton", 26.3711, -80.1021, "open"),
    ("Florida Gators", "Ben Hill Griffin Stadium", "Gainesville", 29.6500, -82.3486, "open"),
    ("Florida International Panthers", "Pitbull Stadium", "Miami", 25.7510, -80.3770, "open"),
    ("Florida State Seminoles", "Doak Campbell Stadium", "Tallahassee", 30.4380, -84.3040, "open"),
    ("Fresno State Bulldogs", "Valley Children's Stadium", "Fresno", 36.8123, -119.7580, "open"),
    ("Georgia Bulldogs", "Sanford Stadium", "Athens", 33.9496, -83.3735, "open"),
    ("Georgia Southern Eagles", "Paulson Stadium", "Statesboro", 32.4210, -81.7840, "open"),
    ("Georgia State Panthers", "Center Parc Stadium", "Atlanta", 33.7350, -84.3890, "open"),
    ("Georgia Tech Yellow Jackets", "Bobby Dodd Stadium", "Atlanta", 33.7724, -84.3928, "open"),
    ("Hawai'i Rainbow Warriors", "Clarence T.C. Ching Complex", "Honolulu", 21.2970, -157.8190, "open"),
    ("Houston Cougars", "TDECU Stadium", "Houston", 29.7218, -95.3422, "open"),
    ("Illinois Fighting Illini", "Memorial Stadium", "Champaign", 40.0992, -88.2360, "open"),
    ("Indiana Hoosiers", "Memorial Stadium", "Bloomington", 39.1808, -86.5234, "open"),
    ("Iowa Hawkeyes", "Kinnick Stadium", "Iowa City", 41.6586, -91.5511, "open"),
    ("Iowa State Cyclones", "Jack Trice Stadium", "Ames", 42.0140, -93.6358, "open"),
    ("Jacksonville State Gamecocks", "AmFirst Stadium", "Jacksonville", 33.8226, -85.7640, "open"),
    ("James Madison Dukes", "Bridgeforth Stadium", "Harrisonburg", 38.4350, -78.8690, "open"),
    ("Kansas Jayhawks", "David Booth Kansas Memorial Stadium", "Lawrence", 38.9633, -95.2468, "open"),
    ("Kansas State Wildcats", "Bill Snyder Family Stadium", "Manhattan", 39.2018, -96.5937, "open"),
    ("Kennesaw State Owls", "Fifth Third Bank Stadium", "Kennesaw", 34.0300, -84.5810, "open"),
    ("Kent State Golden Flashes", "Dix Stadium", "Kent", 41.1480, -81.3450, "open"),
    ("Kentucky Wildcats", "Kroger Field", "Lexington", 38.0222, -84.5050, "open"),
    ("Liberty Flames", "Williams Stadium", "Lynchburg", 37.3510, -79.1790, "open"),
    ("Louisiana Ragin' Cajuns", "Cajun Field", "Lafayette", 30.2140, -92.0220, "open"),
    ("Louisiana Tech Bulldogs", "Joe Aillet Stadium", "Ruston", 32.5320, -92.6460, "open"),
    ("Louisville Cardinals", "L&N Federal Credit Union Stadium", "Louisville", 38.2058, -85.7592, "open"),
    ("LSU Tigers", "Tiger Stadium", "Baton Rouge", 30.4120, -91.1836, "open"),
    ("Marshall Thundering Herd", "Joan C. Edwards Stadium", "Huntington", 38.4230, -82.4340, "open"),
    ("Maryland Terrapins", "SECU Stadium", "College Park", 38.9900, -76.9476, "open"),
    ("Memphis Tigers", "Simmons Bank Liberty Stadium", "Memphis", 35.1210, -89.9770, "open"),
    ("Miami (OH) RedHawks", "Yager Stadium", "Oxford", 39.5072, -84.7303, "open"),
    ("Miami Hurricanes", "Hard Rock Stadium", "Miami Gardens", 25.9580, -80.2389, "open"),
    ("Michigan State Spartans", "Spartan Stadium", "East Lansing", 42.7284, -84.4846, "open"),
    ("Michigan Wolverines", "Michigan Stadium", "Ann Arbor", 42.2658, -83.7487, "open"),
    ("Middle Tennessee Blue Raiders", "Johnny 'Red' Floyd Stadium", "Murfreesboro", 35.8490, -86.3650, "open"),
    ("Minnesota Golden Gophers", "Huntington Bank Stadium", "Minneapolis", 44.9765, -93.2246, "open"),
    ("Missouri State Bears", "Plaster Stadium", "Springfield", 37.1960, -93.2820, "open"),
    ("Missouri Tigers", "Faurot Field", "Columbia", 38.9360, -92.3331, "open"),
    ("Navy Midshipmen", "Navy-Marine Corps Memorial Stadium", "Annapolis", 38.9850, -76.5080, "open"),
    ("NC State Wolfpack", "Carter-Finley Stadium", "Raleigh", 35.8003, -78.7195, "open"),
    ("Nebraska Cornhuskers", "Memorial Stadium", "Lincoln", 40.8206, -96.7056, "open"),
    ("Nevada Wolf Pack", "Mackay Stadium", "Reno", 39.5470, -119.8160, "open"),
    ("New Mexico Lobos", "University Stadium", "Albuquerque", 35.0770, -106.6180, "open"),
    ("New Mexico State Aggies", "Aggie Memorial Stadium", "Las Cruces", 32.2790, -106.7440, "open"),
    ("North Carolina Tar Heels", "Kenan Memorial Stadium", "Chapel Hill", 35.9060, -79.0477, "open"),
    ("North Texas Mean Green", "DATCU Stadium", "Denton", 33.2030, -97.1560, "open"),
    ("Northern Illinois Huskies", "Huskie Stadium", "DeKalb", 41.9330, -88.7720, "open"),
    ("Northwestern Wildcats", "Ryan Field", "Evanston", 42.0658, -87.6924, "open"),
    ("Notre Dame Fighting Irish", "Notre Dame Stadium", "Notre Dame", 41.6984, -86.2339, "open"),
    ("Ohio Bobcats", "Peden Stadium", "Athens", 39.3260, -82.0990, "open"),
    ("Ohio State Buckeyes", "Ohio Stadium", "Columbus", 40.0017, -83.0197, "open"),
    ("Oklahoma Sooners", "Gaylord Family Oklahoma Memorial Stadium", "Norman", 35.2058, -97.4423, "open"),
    ("Oklahoma State Cowboys", "Boone Pickens Stadium", "Stillwater", 36.1260, -97.0660, "open"),
    ("Old Dominion Monarchs", "S.B. Ballard Stadium", "Norfolk", 36.8860, -76.3070, "open"),
    ("Ole Miss Rebels", "Vaught-Hemingway Stadium", "Oxford", 34.3622, -89.5343, "open"),
    ("Oregon Ducks", "Autzen Stadium", "Eugene", 44.0584, -123.0685, "open"),
    ("Oregon State Beavers", "Reser Stadium", "Corvallis", 44.5593, -123.2812, "open"),
    ("Penn State Nittany Lions", "Beaver Stadium", "University Park", 40.8122, -77.8561, "open"),
    ("Pittsburgh Panthers", "Acrisure Stadium", "Pittsburgh", 40.4468, -80.0158, "open"),
    ("Purdue Boilermakers", "Ross-Ade Stadium", "West Lafayette", 40.4346, -86.9186, "open"),
    ("Rice Owls", "Rice Stadium", "Houston", 29.7160, -95.4090, "open"),
    ("Rutgers Scarlet Knights", "SHI Stadium", "Piscataway", 40.5139, -74.4655, "open"),
    ("Sam Houston Bearkats", "Elliott T. Bowers Stadium", "Huntsville", 30.7150, -95.5490, "open"),
    ("San Diego State Aztecs", "Snapdragon Stadium", "San Diego", 32.7840, -117.1220, "open"),
    ("San Jose State Spartans", "CEFCU Stadium", "San Jose", 37.3200, -121.8650, "open"),
    ("SMU Mustangs", "Gerald J. Ford Stadium", "Dallas", 32.8400, -96.7830, "open"),
    ("South Alabama Jaguars", "Hancock Whitney Stadium", "Mobile", 30.6970, -88.1750, "open"),
    ("South Carolina Gamecocks", "Williams-Brice Stadium", "Columbia", 33.9728, -81.0192, "open"),
    ("South Florida Bulls", "Raymond James Stadium", "Tampa", 27.9759, -82.5033, "open"),
    ("Southern Miss Golden Eagles", "M.M. Roberts Stadium", "Hattiesburg", 31.3280, -89.3390, "open"),
    ("Stanford Cardinal", "Stanford Stadium", "Stanford", 37.4346, -122.1609, "open"),
    ("Syracuse Orange", "JMA Wireless Dome", "Syracuse", 43.0361, -76.1364, "dome"),
    ("TCU Horned Frogs", "Amon G. Carter Stadium", "Fort Worth", 32.7098, -97.3677, "open"),
    ("Temple Owls", "Lincoln Financial Field", "Philadelphia", 39.9008, -75.1675, "open"),
    ("Tennessee Volunteers", "Neyland Stadium", "Knoxville", 35.9550, -83.9250, "open"),
    ("Texas A&M Aggies", "Kyle Field", "College Station", 30.6100, -96.3406, "open"),
    ("Texas Longhorns", "Darrell K Royal-Texas Memorial Stadium", "Austin", 30.2840, -97.7325, "open"),
    ("Texas State Bobcats", "UFCU Stadium", "San Marcos", 29.8980, -97.9210, "open"),
    ("Texas Tech Red Raiders", "Jones AT&T Stadium", "Lubbock", 33.5906, -101.8728, "open"),
    ("Toledo Rockets", "Glass Bowl", "Toledo", 41.6180, -83.6150, "open"),
    ("Troy Trojans", "Veterans Memorial Stadium", "Troy", 31.8010, -85.9570, "open"),
    ("Tulane Green Wave", "Yulman Stadium", "New Orleans", 29.9400, -90.1190, "open"),
    ("Tulsa Golden Hurricane", "H.A. Chapman Stadium", "Tulsa", 36.1510, -95.9460, "open"),
    ("UAB Blazers", "Protective Stadium", "Birmingham", 33.5100, -86.8130, "open"),
    ("UCF Knights", "FBC Mortgage Stadium", "Orlando", 28.6080, -81.1930, "open"),
    ("UCLA Bruins", "Rose Bowl", "Pasadena", 34.1613, -118.1676, "open"),
    ("UConn Huskies", "Pratt & Whitney Stadium", "East Hartford", 41.7710, -72.6150, "open"),
    ("UL Monroe Warhawks", "JPS Field at Malone Stadium", "Monroe", 32.5290, -92.0690, "open"),
    ("UMass Minutemen", "McGuirk Alumni Stadium", "Amherst", 42.3700, -72.5340, "open"),
    ("UNLV Rebels", "Allegiant Stadium", "Las Vegas", 36.0909, -115.1833, "dome"),
    ("USC Trojans", "Los Angeles Memorial Coliseum", "Los Angeles", 34.0141, -118.2879, "open"),
    ("Utah State Aggies", "Maverik Stadium", "Logan", 41.7560, -111.8100, "open"),
    ("Utah Utes", "Rice-Eccles Stadium", "Salt Lake City", 40.7600, -111.8488, "open"),
    ("UTEP Miners", "Sun Bowl", "El Paso", 31.7710, -106.5070, "open"),
    ("UTSA Roadrunners", "Alamodome", "San Antonio", 29.4169, -98.4781, "dome"),
    ("Vanderbilt Commodores", "FirstBank Stadium", "Nashville", 36.1436, -86.8115, "open"),
    ("Virginia Cavaliers", "Scott Stadium", "Charlottesville", 38.0310, -78.5140, "open"),
    ("Virginia Tech Hokies", "Lane Stadium", "Blacksburg", 37.2200, -80.4180, "open"),
    ("Wake Forest Demon Deacons", "Allegacy Federal Credit Union Stadium", "Winston-Salem", 36.1317, -80.2542, "open"),
    ("Washington Huskies", "Husky Stadium", "Seattle", 47.6504, -122.3016, "open"),
    ("Washington State Cougars", "Gesa Field", "Pullman", 46.7316, -117.1600, "open"),
    ("West Virginia Mountaineers", "Milan Puskar Stadium", "Morgantown", 39.6500, -79.9550, "open"),
    ("Western Kentucky Hilltoppers", "Houchens Industries-L.T. Smith Stadium", "Bowling Green", 36.9840, -86.4570, "open"),
    ("Western Michigan Broncos", "Waldo Stadium", "Kalamazoo", 42.2830, -85.6130, "open"),
    ("Wisconsin Badgers", "Camp Randall Stadium", "Madison", 43.0700, -89.4126, "open"),
    ("Wyoming Cowboys", "War Memorial Stadium", "Laramie", 41.3120, -105.5670, "open"),
)

COLLEGE_VENUES: dict[str, tuple[str, Stadium]] = {
    normalize_team(team): (normalize_team(city), Stadium(stadium, lat, lon, roof))
    for team, stadium, city, lat, lon, roof in _ROWS}


def college_stadium_for(home_team: str, venue_city: str | None) -> Stadium | None:
    """The home team's stadium, or None: the team is not in the table, or the schedule says the game is in another city
    (neutral site, relocated game), where the home stadium's weather would be the wrong place."""
    row = COLLEGE_VENUES.get(normalize_team(home_team))
    if row is None:
        return None
    home_city, stadium = row
    if venue_city and normalize_team(venue_city) != home_city:
        return None
    return stadium
