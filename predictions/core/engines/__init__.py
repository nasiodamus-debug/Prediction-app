"""One engine per sport (AD-4). A sport is added by registering its engine here (see the progress log)."""
from predictions.core.engines.american_football import AmericanFootballEngine
from predictions.core.engines.basketball import BasketballEngine
from predictions.core.engines.mlb import MlbEngine
from predictions.core.engines.nhl import NhlEngine
from predictions.core.engines.soccer import SoccerEngine

# Keys are the canonical sport names of SportyBet's sport ids (adapters/sportybet/code.py SPORTS): MLB is "baseball",
# the NHL is "hockey". American football (NFL and college, Epic 7) is "american_football".
ENGINES = {"soccer": SoccerEngine(), "basketball": BasketballEngine(), "baseball": MlbEngine(), "hockey": NhlEngine(),
           "american_football": AmericanFootballEngine()}
