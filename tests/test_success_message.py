"""Owner decision 2026-10-01: SportyBet's own 'Submission Successful' pop-up (with the approved total stake and a booking
code) proves a placement; anything else is not proof and the bet history is read as before."""
from decimal import Decimal
from types import SimpleNamespace

import pytest

from predictions.adapters.sportybet.betslip import _success_proof
from predictions.core.placement import PlacementKind

POPUP = "Submission Successful Total Stake 100.00 Potential Win 132.00 Booking Code VWB01S OK Rebet Check Bet History"


def token(stake="100"):
    return SimpleNamespace(stake=Decimal(stake))


def test_the_success_popup_with_the_approved_stake_and_a_code_is_proof():
    result = _success_proof(POPUP, token())
    assert result.kind is PlacementKind.PLACED and result.booking_code == "VWB01S" and result.bet_id == "booking:VWB01S"


def test_a_different_stake_is_not_proof():
    assert _success_proof(POPUP, token("200")) is None


@pytest.mark.parametrize("text", ["", "About to pay NGN 100.00 Cancel Confirm", "Submission Successful Total Stake 100.00",
                                  "Submission Successful Booking Code VWB01S", "Submission Failed Total Stake 100.00 Booking Code VWB01S"])
def test_anything_else_is_not_proof(text):
    assert _success_proof(text, token()) is None
