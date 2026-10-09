"""Shared test setup: test-only values, so real config.env secrets are never needed."""
import os
import sys
from pathlib import Path

os.environ["OWNER_ID"] = "111222333"
os.environ["TELEGRAM_BOT_TOKEN"] = "123456:TEST-TOKEN-NOT-REAL"
os.environ["SPORTYBET_PHONE"] = "08031234567"
os.environ["SPORTYBET_PASSWORD"] = "test-password-not-real"
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test-key-not-real"
# The owner's real keys for the other services must never be used by a test (no network, no spent quota).
for _name in ("API_FOOTBALL_KEY", "GEMINI_API_KEY", "SERPER_API_KEY"):
    os.environ[_name] = ""
os.environ["ANALYST_PROVIDER"] = "claude"
os.environ["DRY_RUN"] = "true"  # the tests never depend on (or are changed by) the owner's config.env
os.environ["PLACE_WAIT_SECONDS"] = "90"  # likewise: never the value in the owner's config.env
os.environ["PLACE_MODE"] = "place"  # AD-25: the older tests exercise place mode; the booking tests switch to book mode explicitly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Story 10.1: Matchday and Gameday only look at the competitions in predictions/core/competitions.py. Many older tests use
# made-up competitions ("Testland - League"); these test-only names are admitted for every test so they keep exercising
# the stats, gate and rendering code. None of them is a real SportyBet competition, so tests/test_competition_limits.py
# (which uses real names) is unaffected.
import pytest

TEST_SOCCER_COMPETITIONS = (
    "Testland - League", "Nowhere - League", "Region - League", "Other - League", "Mixed - League", "A League", "B League",
    "Kenya - Premier League", "Nigeria - NPFL", "International Youth - U19 Friendly Games",
    *(f"Country - League {i}" for i in range(7)), "Region - Cup",
    *(f"Region - Cup {i}" for i in range(7)),
)
TEST_BASKETBALL_COMPETITIONS = ("Region - Basketball League", "Testland - League", "Nowhere - League")


@pytest.fixture(autouse=True)
def _test_only_competitions(monkeypatch):
    from predictions.core import competitions
    for name in TEST_SOCCER_COMPETITIONS:
        monkeypatch.setitem(competitions.SOCCER_DOMESTIC, competitions.normalize(name), "test competition")
    for name in TEST_BASKETBALL_COMPETITIONS:
        monkeypatch.setitem(competitions.BASKETBALL_ALLOWED, competitions.normalize(name), "test competition")


@pytest.fixture(autouse=True)
def _fresh_caches():
    """Spec 14-1: the stats cache is OFF and the analyst cache is empty for every test, so one test never sees another's data.
    Tests of the caches themselves switch them on (see tests/test_speed_caches.py)."""
    from predictions.core.ttl_cache import STATS_CACHE
    from predictions.flow.analyst import ANALYST_CACHE
    STATS_CACHE.clear()
    STATS_CACHE.set_scale(0)
    STATS_CACHE.hits = STATS_CACHE.misses = 0
    ANALYST_CACHE.clear()
    ANALYST_CACHE.enabled = True
    ANALYST_CACHE.directory = None   # importing app.py points it at local_data; a test must never read or write the real files
    app = sys.modules.get("app")   # the bot module: its daily games cache must never reach the network from a test
    saved = None
    if app is not None and hasattr(app, "DAY_CACHE"):
        saved = app.DAY_CACHE.ttl
        app.DAY_CACHE.ttl = 0
        app.DAY_CACHE.clear()
    yield
    if saved is not None:
        app.DAY_CACHE.ttl = saved
    STATS_CACHE.clear()
    STATS_CACHE.set_scale(1.0)
    ANALYST_CACHE.clear()


@pytest.fixture(autouse=True)
def _no_live_api_football(monkeypatch):
    from predictions.flow import freeflags
    monkeypatch.setattr(freeflags, "SOURCE", None)


@pytest.fixture(autouse=True)
def _fresh_review_cache(monkeypatch):
    from predictions.flow.reviewcache import REVIEW_CACHE
    monkeypatch.setattr(REVIEW_CACHE, "hours", 0.0)   # tests of the cache itself turn it on; no other test shares a remembered review
    REVIEW_CACHE.clear()


@pytest.fixture(autouse=True)
def _no_login_recheck_pause(monkeypatch):
    from predictions.adapters.sportybet import betslip
    monkeypatch.setattr(betslip, "LOGIN_RECHECK_PAUSE_S", 0.0)   # the re-check is tested without waiting seconds


@pytest.fixture(autouse=True)
def _no_weekly_catch_up_by_the_calendar(monkeypatch):
    """The Monday catch-up depends on today's date: tests that want it turn it on themselves."""
    import sys
    if "app" in sys.modules:
        monkeypatch.setattr(sys.modules["app"], "WEEKLY_ENABLED", False)
