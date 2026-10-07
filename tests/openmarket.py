"""One open, priced market per sport, so a fabricated fixture counts as OPEN FOR BETTING (spec-12-1)."""
from decimal import Decimal

from predictions.core.types import MarketOffer, Outcome

# SportyBet market ids each engine prices: soccer 1X2, basketball/football 219 winner, MLB 251, NHL 406.
_WINNER_IDS = {"soccer": "1", "basketball": "219", "american_football": "219", "baseball": "251", "hockey": "406"}


def open_markets(sport="soccer"):
    return (MarketOffer(_WINNER_IDS[sport], "Winner", "", (Outcome("Home", Decimal("1.90"), "1"),
                                                           Outcome("Away", Decimal("1.90"), "2"))),)
