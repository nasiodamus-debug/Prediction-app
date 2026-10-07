"""Suspended markets and inactive outcomes are never offered (snippets recorded from the live listing)."""
from predictions.adapters.sportybet.fixtures import parse_event

TOURNAMENT = {"categoryName": "England", "name": "Premier League"}


def _event(markets):
    return {"homeTeamName": "Arsenal", "awayTeamName": "Leeds", "estimateStartTime": 1789237359000,
            "status": 0, "markets": markets}


def _ou(status, active=1, line="2.5"):
    return {"id": "18", "specifier": f"total={line}", "desc": "Over/Under", "status": status, "name": "Over/Under",
            "banned": False, "outcomes": [{"id": "12", "odds": "1.90", "isActive": active, "desc": f"Over {line}"},
                                          {"id": "13", "odds": "1.90", "isActive": 1, "desc": f"Under {line}"}]}


def test_suspended_market_status_2_is_skipped():
    fixture = parse_event(_event([_ou(2), _ou(0, line="1.5")]), TOURNAMENT)
    assert [m.specifier for m in fixture.markets] == ["total=1.5"]


def test_absent_status_is_open_and_string_zero_is_open():
    open_absent = _ou(0)
    del open_absent["status"]
    fixture = parse_event(_event([open_absent, _ou("0", line="3.5")]), TOURNAMENT)
    assert len(fixture.markets) == 2


def test_unreadable_status_and_banned_are_skipped():
    banned = _ou(0)
    banned["banned"] = True
    fixture = parse_event(_event([_ou("x"), banned]), TOURNAMENT)
    assert fixture.markets == ()


def test_inactive_outcome_is_dropped_but_others_stay():
    for flag in (0, False, "0"):
        fixture = parse_event(_event([_ou(0, active=flag)]), TOURNAMENT)
        assert [o.name for o in fixture.markets[0].outcomes] == ["Under 2.5"]


def test_market_with_no_active_outcome_is_dropped():
    market = _ou(0, active=0)
    market["outcomes"][1]["isActive"] = 0
    assert parse_event(_event([market]), TOURNAMENT).markets == ()
