"""Owner bug report: 'the list changed, use the newest list' when pressing Final confirm after deciding on every game.
The owner's exact sequence is driven through the REAL handler chain with a realistic, paged slate."""
from datetime import datetime, timedelta, timezone

import app as bot
from predictions.flow.session import SessionManager
from test_final_list import KICKOFF, OWNER_ID, ev, chain, find, open_list  # noqa: F401  (chain is a fixture)


def realistic_slate():
    """14 games (two pages), several with 2 or 3 plays, kicking off from 12:00 to 19:00 UTC."""
    evals = []
    for i in range(14):
        home, away, hour = f"Home{i:02d}", f"Away{i:02d}", 12 + i // 2
        evals.append(ev(home, away, "Over 1.5", 0.90 - i * 0.01, "1.30", hour=hour))
        if i % 3 == 0:
            evals.append(ev(home, away, "Over 2.5", 0.75, "1.85", hour=hour))
        if i % 5 == 0:
            evals.append(ev(home, away, "BTTS Yes", 0.70, "1.90", hour=hour))
    return evals


def current(chain):
    return [(b.text, b.callback_data) for row in chain.events[-1][1]["reply_markup"].inline_keyboard for b in row]


def decide_every_game(chain):
    """Press Delete on every odd game's first play and Keep on the second play of multi-play games; page with Next/Prev."""
    seen = 0
    for page in (0, 1):
        if page == 1:
            chain.press(OWNER_ID, find(current(chain), "Next"))
        labels = [t for t, _ in current(chain)]
        for label in [l for l in labels if l.endswith("Delete") or l.endswith("Keep")]:
            if label.endswith("Delete") and (int(label.split()[0].rstrip("abc")) % 2 == 1):
                chain.press(OWNER_ID, find(current(chain), label))
                seen += 1
            elif label.endswith("Keep") and label.split()[0].endswith("b"):
                chain.press(OWNER_ID, find(current(chain), label))
                seen += 1
        if page == 0:
            chain.press(OWNER_ID, find(current(chain), "Next"))
            chain.press(OWNER_ID, find(current(chain), "Previous"))
    return seen


def test_owner_sequence_keep_delete_on_every_game_then_final_confirm_is_not_stale(chain):
    open_list(chain, realistic_slate())
    decide_every_game(chain)
    version = bot.SESSIONS.active.state_version
    assert all(f":{version}:" in d for _, d in current(chain)), "every live button carries the current version"
    chain.press(OWNER_ID, find(current(chain), "Final confirm"))
    assert chain.answers[-1][0] != "This list changed. Please use the newest list."
    assert "Final edit" in chain.events[-1][0]


FIRST_KICKOFF = KICKOFF.replace(hour=12)


def set_clock(when):
    bot.SESSIONS._clock = lambda: when


def test_deleting_the_games_that_already_kicked_off_lets_final_confirm_work(chain):
    """THE REPORTED BUG. The slate was made before 12:00; the owner reviews it after the 12:00 games kicked off, deletes
    them (the only sensible thing to do) and decides on the rest. Final confirm used to answer 'This list changed.
    Please use the newest list.' because the expiry check looked at the DELETED games' kickoff, not the kept ones."""
    set_clock(FIRST_KICKOFF - timedelta(minutes=30))     # the list is made 30 minutes before the first games
    open_list(chain, realistic_slate())
    set_clock(FIRST_KICKOFF + timedelta(minutes=20))     # games 1 and 2 (12:00) are under way
    for label in ("1a Delete", "2 Delete"):
        chain.press(OWNER_ID, find(current(chain), label))
    chain.press(OWNER_ID, find(current(chain), "Next"))
    version = bot.SESSIONS.active.state_version
    assert all(f":{version}:" in d for _, d in current(chain))
    chain.press(OWNER_ID, find(current(chain), "Final confirm"))
    assert chain.answers[-1][0] != "This list changed. Please use the newest list."
    assert "Final edit" in chain.events[-1][0]
    chain.press(OWNER_ID, find(current(chain), "Confirm games"))
    assert chain.jobs and chain.jobs[-1].kind == "load"


def test_final_confirm_with_a_kept_game_that_kicked_off_explains_and_keeps_the_buttons_working(chain):
    set_clock(FIRST_KICKOFF - timedelta(minutes=30))     # the list is made 30 minutes before the first games
    open_list(chain, realistic_slate())
    set_clock(FIRST_KICKOFF + timedelta(minutes=20))
    chain.press(OWNER_ID, find(current(chain), "Final confirm"))
    answer, alert = chain.answers[-1]
    assert answer != "This list changed. Please use the newest list." and "kicked off" in answer and alert
    text, kwargs = chain.events[-1]
    assert "kicked off" in text and "Home00 vs Away00" in text           # the list is shown again with the reason
    version = bot.SESSIONS.active.state_version
    assert all(f":{version}:" in d for _, d in current(chain))
    assert bot.SESSIONS.active.state.value == "REVIEWING_LIST"
    chain.press(OWNER_ID, find(current(chain), "1a Delete"))               # and the owner can fix it
    assert chain.answers[-1][0] == "Deleted."


def test_a_genuinely_stale_button_is_refused_but_the_current_list_is_shown_again(chain):
    old = open_list(chain, realistic_slate())
    chain.press(OWNER_ID, find(old, "1a Delete"))
    before, shown = bot.SESSIONS.active, len(chain.events)
    chain.press(OWNER_ID, find(old, "3 Delete"))
    assert len(chain.events) == shown + 1                                   # the pressed message was shown again                           # a button from the older version
    assert chain.answers[-1][0] == "This list changed. Please use the newest list."
    assert bot.SESSIONS.active == before                                    # nothing was changed by the stale tap
    version = before.state_version
    assert all(f":{version}:" in d for _, d in current(chain))              # the pressed message now has working buttons
    assert "Final list" in chain.events[-1][0]
