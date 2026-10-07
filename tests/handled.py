"""Matchday, Gameday, America, Review and Results run in the background (AD-22): a test that calls the handler directly
must also wait for the background request to finish."""


async def handled(update, context=None):
    import app as bot
    await bot.handle_message(update, context)
    await bot.JOBS.wait()


ACK_PREFIXES = ("Analysing ", "Reviewing that booking code.", "Checking your placed bets")


def drop_ack(sent):
    """Remove the short 'Analysing ...' acknowledgement the bot sends first, from a list of replies (strings or
    (text, kwargs) pairs), so a test can look at the real result."""
    sent[:] = [item for item in sent
               if not str(item[0] if isinstance(item, tuple) else item).startswith(ACK_PREFIXES)]
