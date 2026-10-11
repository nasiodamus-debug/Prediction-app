# Predictions app

A private Telegram bot (owner only) that evaluates sports bets, lets you confirm a slate, loads it onto SportyBet, and
later reads back how your placed bets did. Everything below that touches SportyBet or the model's live output is
UNVERIFIED against the real sites unless stated: treat every first run as a rehearsal.

## Setup

```
cd "C:\Users\hp\Predictions app"
pip install "python-telegram-bot[job-queue]" playwright python-dotenv httpx anthropic==1.9.0 tzdata==2026.4
playwright install chromium
```

## How to run it

1. Put your settings in `config.env` (never share it):
   - `TELEGRAM_BOT_TOKEN`, `SPORTYBET_PHONE`, `SPORTYBET_PASSWORD` (login for the placement browser).
   - `OWNER_ID`: your Telegram user ID (message @userinfobot to find it). Required; the bot answers only this account, in
     a private chat, and will not start without it.
   - `DRY_RUN`: **keep it `true` (or unset).** Dry run is the default: everything is checked and reported and the Place
     button is NEVER clicked. Real bets are placed only if it says exactly `DRY_RUN=false`. Do a full dry-run journey
     first and compare it with the real SportyBet betslip.
   - `MAX_STAKE`: the largest stake accepted, in NGN (default 100000).
   - `ANTHROPIC_API_KEY`: optional; turns on the AI analyst that may fill missing stats (labelled as analyst-sourced).
   - `ANALYST_MODE`: optional; makes the paid analyst a last resort. `last_resort` (the default when unset): free sources are always tried first, and the analyst is asked only for required stats that every free source left missing; it is never asked for a flags-only check on a game the free sources already cover, and never for a team whose season has not started (for example NHL preseason: nothing to find). `off`: no analyst is built and no Claude call is ever made, even with `ANTHROPIC_API_KEY` set; games with missing data stay "insufficient data". `full`: the earlier behaviour (also flags-only checks). Any case; anything else stops the bot from starting. Free data that already exists always wins over an analyst value, and an analyst value is always labelled `analyst-sourced`.
   - `ANALYST_DAILY_CALL_LIMIT`: optional; the most paid analyst calls per Lagos day across Matchday, Gameday and Review (default 40). The count is kept in the local database, so a restart does not reset it. When it is reached the analyst is skipped with a one-line note. A value that is not a positive whole number stops the bot from starting. (One request is still capped at 10 calls.)
     Without it, games with missing data stay "insufficient data".
   - `CACHE_TTL_SCALE`: optional; multiplies how long team form and wind figures are remembered in memory (form 3 hours, wind and announced starting pitchers 1 hour). Unset = 1 (as stated), `0.5` = half as long, `0` = **cache off** (every request reads everything again, and an analyst answer is no longer reused). A number from 0 to 24; anything else stops the bot from starting. Odds, markets, betting status and your account are never cached.
   - `LISTING_TTL_SECONDS`: optional; how old the odds in the daily games cache (below) may be before a request refreshes them. Unset = 600 (10 minutes); a whole number from 1 to 3600; `0` = **daily games cache off** (every request reads SportyBet's list itself, as before). Anything else stops the bot from starting.
   - `FAST_BROWSER`: optional; `true`/`on` turns on a mode where the placement browser (OFF when unset: measured live on 2026-10-01 it was SLOWER on this site, about 90 s against 54 s to load the betslip, so leave it off unless you test it yourself) skip pictures, fonts and media and look at the page more often while it waits; `false`/`off` restores the exact earlier behaviour. Every check (betslip read-back, odds, stake, history) runs either way. Anything else stops the bot from starting. If the SportyBet login ever asks for a picture code, set `FAST_BROWSER=off`.
   - `BROWSER_MODE`: optional; `standard` (default when unset) is the bundled hidden Chromium as before. `real` starts your installed **Google Chrome in a visible window** with the automation flags hidden (no custom user-agent), an attempt to stop SportyBet's bot detection from freezing the page after Place. If Chrome is not installed the bot says so and does nothing (it never falls back silently): install Chrome or set `BROWSER_MODE=standard`. Anything else stops the bot from starting. Your saved login is still used. Not verified live.
   - `PLACE_WAIT_SECONDS`: optional; how long the bot watches the page after the Place click for the Confirm panel and the "Submission Successful" pop-up. Unset = 90; a whole number from 30 to 300; anything else stops the bot from starting. If the Place button stayed pale/disabled the whole time, the diagnostics say so. Place is never clicked a second time.
   - `PLACE_MODE`: optional; `book` (the default when unset) or `place`, any case; anything else stops the bot from starting. **`book`: the bot never places a bet.** After your one final confirmation it checks the real SportyBet betslip (games and odds), presses ONLY the slip's `Book Bet` link once and sends you the booking code; it never clicks Place Bet or Confirm and never types a stake, so you choose the stake and place the bet yourself in SportyBet (`Booking Code` tab, enter the code, stake, Place Bet). In this mode you are not asked for a stake at all (no stake question, no `Change amount`, no potential winnings, `MAX_STAKE` is not used). `place`: the earlier behaviour, unchanged (stake question, the bot clicks Place, then waits for the success message; an unclear result is UNKNOWN and carries a booking code from the same Book Bet fallback). `DRY_RUN=true` still wins over both: nothing is clicked, no code is made. Set `PLACE_MODE=place` in config.env to go back to bot-placed bets.
   - `ANALYST_PROVIDER`: optional; a comma list tried in order, `claude` (the default when unset) or, for example, `gemini,claude`. With `gemini,claude` the analyst is a hierarchy: the free sources run first (always), then the free Gemini analyst (Serper search, Gemini reads), then Claude (paid, at most 2 web searches per lookup) and ONLY for the required stats Gemini left missing, whether it found nothing, found only some, gave a discarded answer or was unavailable. A flags-only check never reaches Claude. Needs `GEMINI_API_KEY` and `SERPER_API_KEY` for the Gemini step. Only Claude's usage is counted as spend in the cost line.
   - `CLAUDE_CALL_LIMIT`: optional; the most Claude (paid) lookups ONE request may make when `ANALYST_PROVIDER` has Gemini first (unset = 10; `0` = never use Claude; a whole number from 0 to 1000). A game held back by the limit is left as it was and not kept for the day, so a later request can retry it. Anything else stops the bot from starting.
   - `CLAUDE_AFTER_EMPTY`: optional; `true` asks Claude even for a game where Gemini answered but found nothing at all (unset/`false` = it is not asked: such a game rarely pays off). A Gemini that failed, was unavailable or gave a discarded answer still goes on to Claude. Anything else stops the bot from starting.
   - `KEY_PLAYERS_OUT_MIN`: optional; how many DISTINCT players API-Football must list as out for one team before a soccer game is flagged `key_player_out` (which can only turn a pick into NO VALID PLAY - PASS). Unset = 5; a whole number from 1 to 30; anything else stops the bot from starting. A player listed for several issues counts once, a player listed only as a doubt ("Questionable") does not count, and once a team reaches the minimum its squad list is read once (cached for the day) and players the squad gives as under 21 are not counted (a squad that cannot be read filters nothing).
   - In a Review, Claude is only asked for a soccer game's goals-only stats (matches, goals scored and conceded), not for xG, because a game with no xG is priced from goals alone. The cost line shows how many lookups each analyst made and why Claude was skipped.
   - Analyst answers are kept for the rest of the Lagos day, whichever analyst gave them, and written to `local_data/analyst_cache/`, so a restart does not lose them. A later request for the same or fewer stats on the same game is served from there at no cost and uses no call limit, including stats that were looked for and not found. An answer is not kept if the paid step could not be reached for stats still missing, or if it was discarded.
   - `ANALYST_CALL_LIMIT`: optional; how many AI analyst calls ONE request may make (a Matchday, a Review of one code). Unset = 40; a whole number from 1 to 1000. This is separate from `ANALYST_DAILY_CALL_LIMIT`, the total per Lagos day. Anything else stops the bot from starting.
   - `WEEKLY_REVIEW` / `WEEKLY_REVIEW_TIME`: optional; the Monday post-game review (see below). `WEEKLY_REVIEW=off` turns the Monday post off (the `Weekly` command still works); `WEEKLY_REVIEW_TIME` is `HH:MM` Lagos time (unset = 09:30; an unreadable value keeps 09:30).
   - `REVIEW_CACHE_HOURS`: optional; how many hours a finished `Review <code>` is kept in memory and reused when the same code is sent again on the same Lagos day (unset = 3; a number from 0 to 24 such as `1.5`; `0` = off). A reused review makes no new reads, stats lookups or analyst calls, shows when it was made, and drops any game that has started since. A review made while a stats or analyst service was failing is not kept. Anything else stops the bot from starting.
   - `SHOW_BROWSER`: optional; `true` opens a visible Chrome window for SportyBet (default: hidden) so you can watch the Place click. Use it with a small `MAX_STAKE`; do not touch the window while the bot works.
2. Before trusting the bot with the login, run the smoke script (dry run, it never places anything):
   `python -m predictions.tools.betslip_smoke`
3. Start the bot: `python app.py` (one copy at a time). It sends you a start-up message that states whether dry run is on.

## Triggers (send them to the bot in Telegram)

| Trigger | What it does |
| --- | --- |
| `Matchday <dates>` | Soccer (domestic leagues with xG from Understat; UEFA and international competitions on a goals-only estimate from ESPN, see below). It considers only five play types per game: Over goals from 1.5 (half lines only: no whole-line or Asian plays are ever offered in Matchday, Gameday or America; Review still judges them), match winner (Home or Away), double chance (Home or Draw, Away or Draw), Over yellow cards from 2.5 and Over corners from 7.5 (cards and corners only for games where SportyBet lists them, checked per game; data from football-data.co.uk for the major leagues). Plays must clear the 70% floor with positive EV; the best two per game are listed |
| `Gameday <dates>` | Basketball slates (game, team and 1H totals, spread, winner) |
| `America <dates>` | MLB, NHL and American football (NFL and college) together, in one list |
| `Review <booking code>` | Reads a SportyBet booking code and judges each selection |
| `Results` | Reads SportyBet's own bet-level result (Won, Lost, Running) for your PLACED bets from the first page of the bet history (5 newest bets). Running stays pending. See the Results limitation below |
| `Mybets` | Your 10 most recent records, newest first: `BOOKED` (a booking code was made, nothing placed by the bot), `PLACED BY YOU` (you said you placed a booked one), `PLACED` (the bot placed it and SportyBet's history proved it) and `UNKNOWN`, each with its code, total odds and stake. Every still-`BOOKED` record has the buttons `I placed it` / `I did not place it`, so you can answer later |
| `Calibration` | Win rate of your placed, graded legs by the model's probability band (65-70%, 70-75%, 75-80%, 80%+), voids and pushes counted apart, small samples flagged |
| `Cache` | Shows the daily games cache (below): when it was built, the two days it covers, how many games it holds per trigger, and how old the oldest and newest odds snapshot is. `Cache refresh` rebuilds it now, in the background |

**The bot stays responsive.** `Matchday`, `Gameday`, `America`, `Review` and `Results` take a minute or more, so the bot first replies "Analysing ... This can take a minute" and sends the list (or NO VALID PLAY) when it is ready. While it works, your buttons still answer at once, and loading and placing run in the background the same way. One request of each kind runs at a time: a second `Matchday` while one is running gets "still running". The console logs how long each button and each request took (for example `Matchday request took 41.3 s`).

`<dates>` for Matchday, Gameday and America can be `today`, `tomorrow` (Lagos time), days such as `3 and 5 October`, or a mix
("today and 3 October"). Filler words are fine ("Matchday for today"). The bot never picks a day you did not name.

When something passes, the reply is just the list of games that have an advised play (nothing else). Every play has its own `Keep` / `Delete` pair on one row: a game with one play is `1` (`1 Keep` / `1 Delete`), a game with several valid plays lists them as `1a`, `1b`, `1c` (the advised one, the highest estimate, first and kept by default) with a row each. A game keeps at most ONE play: `Keep` on another play replaces the kept one, `Delete` on the kept play takes the game out (it stays listed and any of its plays can be kept again). Press `Final confirm` at the bottom for the review screen (only the kept plays), then `Confirm games` to load them or `Back to adjust` to keep editing. Nothing is loaded or bet before Confirm, and DRY RUN stays on unless you turn it off.

The final page (before the bet) starts with a banner (`*** LIVE: this will place a REAL bet on SportyBet ***` or `DRY RUN: nothing will be placed`), lists the games, shows the total odds and your stake, and shows `Potential winnings` = total odds x stake (rounded down, your stake included, with a `Profit if it wins` line; an estimate, as SportyBet may adjust for bonuses or limits). The `Change amount` button next to the confirm button takes you back to the stake step (the loaded betslip and odds are kept, nothing is reloaded): type the new amount and you get a new final page with updated winnings. Changing the amount voids the earlier confirmation, so only the new page's button can place the bet. Once placing has started the button no longer works.

**Booking-code mode (`PLACE_MODE=book`, the default).** The flow is the same up to the odds: after `Confirm games` the bot loads the slip, verifies the games and odds on the real betslip, shows the odds and goes straight to the final page, which says `Confirm to get your booking code` and has the single button `Get booking code` (no stake question, no `Change amount`, no winnings: you set the stake in SportyBet). Pressing it uses the same single-use confirmation and the same exact check of the real betslip as before; then the bot presses only `Book Bet`, once, reads the new booking code and closes the pop-up. You get ONE message:

```
Booking code: HW63TV            (the code in monospace: tap to copy)
Total odds: 2.41

1. Alpha v Beta: Over 2.5 (Over/Under) @ 1.85
2. Gamma v Delta: Over 1.5 (Over/Under) @ 1.30

Open SportyBet, enter the code, choose your stake and press Place Bet yourself. Nothing was placed by the bot.
```

with the buttons `I placed it` and `I did not place it`. Nothing here is ever called "placed" by the bot. If the code cannot be read it says `No bet was placed. The booking code could not be read.` with the usual `Try again` (3 retries per slate; it reloads the same games, shows fresh odds and the same final page). `I placed it` asks `How much did you stake? Reply with the amount, or press Skip`; then the booking becomes **PLACED BY YOU** (your word, source `owner-confirmed`, stake or none), its games go into the placed-games list (they block future lists exactly like a bet the bot placed) and the bot says `Recorded as placed (confirmed by you).` `I did not place it` says `Noted: not placed.` and changes nothing (a BOOKED record never blocks anything and never counts as a bet; you can still answer later with `Mybets`). Taps are safe to repeat; if the bot restarts while it waits for the amount, the record simply stays BOOKED and the buttons work again. `Results` also grades a PLACED BY YOU bet, but only when the bet history shows exactly one matching bet (same games, your stake, not older than the booking); otherwise it is listed as `not graded` and nothing is invented. A booking code on its own is never counted in `Results`, `Calibration` or any stake accounting.

**If one pick cannot be added to the betslip (for example a handicap SportyBet does not add), the others still load and you decide.** The bot clicks each pick once more if it did not appear, then skips it and carries on. When everything loaded nothing changes. When some did not, you get `Loaded 4 of 5 picks on SportyBet. 1 could not be added.` with the loaded picks and their real odds, the failed pick and the reason (including `locked or suspended` when SportyBet shows it that way), and the total odds of the LOADED picks only, with three buttons: `Go ahead with the 4 loaded` (the failed pick is dropped from this bet for good and named again on the odds screen and the final summary; in booking mode you then get `Get booking code` with the new total odds, in place mode the stake question as usual), `Retry the failed picks` (clears your betslip and adds all the picks again, then shows this screen again; if a pick fails twice it says so; it uses one of the 3 retries for the slate, and at the cap only Go ahead and Cancel remain) and `Cancel`. Nothing is skipped or dropped without you pressing a button, nothing is placed or booked at this step, and the confirmation only covers the picks that are really on the slip. If NO pick could be added you get the usual `No bet was placed` with `Try again`. A problem that is not about one pick (login, the betslip cannot be emptied, a browser crash, a timeout) still fails the whole load. If the bot restarts on this screen it reloads all the picks and shows the result again.

**If something fails, you decide whether to try again: nothing is ever retried on its own.** Every failure message says plainly whether a bet could exist. If it says `No bet was placed.` (the failure was before the Place click), a `Try again` button reloads the same games with the same stake onto your betslip, shows the fresh odds and a new final page; you must press `Confirm and place bet` on that new page (a reload never places anything). If it says `A bet MAY exist.` (Place was clicked but the result is unknown), you get `Check history again`, which only reads your SportyBet bet history (it never places or touches the betslip): if the bet is there it is recorded as placed; if not, it says so (history can lag by a minute). Only after a check found nothing does `Try again, I checked` appear; it shows a warning that a SECOND bet may be placed if the first went through, and `Yes, place again` still needs your confirmation on a new page. Each slate allows 3 retries in total, then start again with a new request.

When nothing is evaluated or nothing passes, the first message is exactly `NO VALID PLAY - PASS`; the next message starts with a
short "Why" block (games evaluated and skipped, the skip reasons with counts, and what could change it).

Soccer coverage: Understat (xG) for England, Spain, Germany, Italy, France and Russia; ESPN's free public data for the UEFA
Nations League, Champions/Europa/Conference League (men and women), U21 European qualification, FIFA friendlies and club
friendlies. ESPN has no xG, so those games use a stricter GOALS-ONLY model (at least 8 past matches, strong shrinkage to the
average) and every such pick is labelled "goals-only estimate (no xG)". The 70% floor and positive EV still apply.

Results are also checked automatically once a day at 09:00 Lagos time while the bot is running (you get a short message
only when something changed or failed). If the bot was not running then, nothing is invented: send `Results`. Only
SportyBet decides an outcome; a pick that was never placed (or was only a dry run) is never graded.

**No Under plays, in any sport.** The bot never recommends, lists, loads or places an Under (totals, team totals, halves, periods, corners, cards). A booking-code Under is disqualified in Review.

## Speed: parallel reads, caches and the daily games cache

A real Matchday request took about 41 s, 29 to 32 s of it SportyBet's game list (17 pages of 100 games read one after the other). Now:

- **The game list is read in parallel**: page 1 first (it announces the total), then the other pages six at a time. All pages must succeed (never a partial list), the announced total may drift by 2 games (or 0.5%), games are de-duplicated, and there is still a 120 s overall limit and no retries. `America` reads its three listings at the same time. Measured on the real soccer list: 28.7 s one at a time, 12.9 s three at a time, 10.4 s five at a time, 7.0 s eight at a time.
- **Team form is remembered in memory** (ESPN, MLB, NHL, Understat, football-data: 3 hours; Open-Meteo wind and announced MLB starting pitchers: 1 hour). Odds, markets, "open for betting", bet history, the betslip and anything about your account are never cached; an error is never cached. The log says per request how many reads were hits and misses. Switch off or shorten with `CACHE_TTL_SCALE`.
- **AI analyst answers are remembered for the rest of the Lagos day** per game and requested inputs, so a repeated request never pays twice for the same game. A reused answer makes no call, uses neither the per-request cap (10) nor the daily limit, costs nothing, and the cost line says "N answer(s) reused from today's cache at no cost". Up to 3 analyst calls run at the same time; the per-request and daily limits, the earliest-kickoff-first rule and the outage handling are unchanged.
- **A faster betslip browser** (`FAST_BROWSER`, OFF by default because it measured slower): no pictures, fonts or media, quicker looks while waiting; games are still added one at a time on one page and every check still runs. The smoke script prints the measured seconds per step (`--no-fast` runs it with the switch off to compare).
- **Daily games cache** (memory only, nothing is written to the database): at 06:00 Lagos time every day, and in the background when the bot starts, the bot reads the three SportyBet listings together and keeps, per trigger, the games in your allowed competitions that are open for betting and not started and kick off **today or tomorrow (Lagos days)**, with their competition, teams, kickoff, odds and the time the odds were read. It then fetches the team stats of those games ahead of time (no analyst call is ever made for this). A `Matchday`, `Gameday` or `America` request whose dates are all inside those two days reads its games from the cache when the odds are younger than `LISTING_TTL_SECONDS` (10 minutes by default), so no list is read at all; an older snapshot is refreshed first (one parallel read of the live list, which also picks up newly listed games). Each request still checks that every game has not started and is still open for betting, and a request for any other date works as before. When a list comes from the cache it ends with one line, `Odds as of HH:MM (Lagos)`. **The cache never feeds placement**: loading still reads the real betslip, the summary shows the odds read from the betslip and the bet is placed only if they match what you confirmed. `Review` does not use it. Turn it off with `LISTING_TTL_SECONDS=0`.

## Which competitions are covered

To keep the cost down, Matchday, Gameday and America only look at the competitions below. Everything else is not listed, not looked up and never sent to the paid analyst; the reply says how many games were ignored. **Review is not limited by this list**: it judges every selection of a booking code, whatever its competition or country.

- **Matchday (soccer):** the top division of the men's national leagues in Europe, South America and North America (England Premier League, Spain LaLiga, Italy Serie A, Germany Bundesliga, France Ligue 1, Netherlands Eredivisie, Portugal, Belgium, Turkey, Scotland, Brazil Serie A, Argentina, USA MLS, Mexico Liga MX and so on); the men's continental cups (Champions League, Europa League, Conference League, Copa Libertadores, Copa Sudamericana, CONCACAF Champions Cup, CAF Champions League); senior national-team games, men and women (World Cup and qualifiers, European Championship and qualifiers, Nations League, Africa Cup of Nations, Copa America, Asian Cup, senior friendlies); and the UEFA Women's Champions League. Not covered: second and lower divisions, reserve, youth and under-age sides, women's domestic leagues, domestic cups, club friendlies, simulated (SRL) games, and Asia, Africa and Oceania domestic leagues.
- **Gameday (basketball):** NBA, WNBA, EuroLeague, the Adriatic ABA League, the BNXT League, and the first division of each European country's men's league as SportyBet files them: Spain ACB, Italy LBA, Germany BBL, France LNB Elite, Greece Basket League, Turkey BSL, Lithuania LKL, Poland PLK, Austria, Belgium, Croatia, Denmark, England SLB, Finland, Georgia, Hungary, Portugal, Sweden (a few more are listed in the table for when SportyBet shows them). Not covered: the EuroCup, every second division, NCAA, G League, women's leagues other than the WNBA, youth, 3x3, and the Australian WNBL and NBL (the WNBL is a one-line entry you can enable in the table, marked "enable if the owner wants the Australian WNBL").
- **America (one table per sport):** MLB (baseball), NHL (hockey), NFL and college football (SportyBet's "USA - NCAA, Regular Season", which holds FBS and FCS together; the free ESPN data now reads both the FBS and FCS scoreboards, so FCS games no longer need the analyst). **College football limits (set 2026-10-10 after the first settled week: 54 legs, 72.8% predicted, 53.7% won):** a college pick is not passed when it claims more than +20% expected value (NFL picks are not capped); no college Over (game, team or 1st-half total) is played; a college team needs 4 completed games (the NFL needs 3); the analyst is never asked to fill in a college team ESPN does not cover; the college standard deviations are the ones measured by the backtest (margin 17.7, total 15.5, one side 11.8, 1st-half total 11.1 points); and college handicaps and winners use OPPONENT-ADJUSTED scoring rates (a team that beat weak opponents is not flattered), while every total keeps the raw averages (the backtest found the adjustment helps margins and winners and slightly hurts totals). The constants are `MAX_EV`, `COLLEGE_OVERS_PAUSED`, `MIN_GAMES`, `SHRINK_GAMES` and the `*_SD` tables in `predictions/core/engines/american_football.py`. **Backtest:** `python -m predictions.tools.football_backtest --league college --year 2025 --year 2026` replays past ESPN seasons week by week (each week predicted from the earlier weeks only, the bot's own code) and reports how far off the expected margins and totals are, whether the assumed standard deviations are right, and how often "70%" and "80%" come true; `--variant production --variant raw` compares input recipes on the same games. It reads public ESPN scoreboards only, keeps seasons in `local_data/backtest/`, and needs no account. It cannot judge the odds: ESPN gives scores, not past bookmaker prices. Not covered: the CFL, every other hockey league (KHL, SHL, Liiga, AHL, juniors, NCAA hockey), every other baseball league (KBO, NPB, CPBL, minor leagues), college sports other than football, Arena and other football.
- **Review:** not restricted. A booking-code selection is judged by its sport's rules whatever the competition, and it is judged on the odds on the code even if SportyBet has suspended the game now (only games that have already started are excluded).

**Open for betting.** Before any analysis, Matchday, Gameday and America check that each game is actually open for betting on SportyBet right now: it is in SportyBet's current listing, not started, and has at least one active market the sport's engine prices. A listed game with every market suspended (or only markets the engine ignores) is skipped before any stats or analyst call, and the reply says "N game(s) skipped: not open for betting on SportyBet". Soccer corners and cards are still looked up per game afterwards.

**To change the list**, edit the tables at the top of `predictions/core/competitions.py` (one commented line per competition, written the way SportyBet names it, "Country - League", lower case without punctuation). Matching is exact, so adding `england premier league` never lets in "Premier League 2" or "Premier League, Reserves". Nothing else needs changing. Run `python -m pytest tests -q -p no:cacheprovider` afterwards.

## Unverified-live caveats

- The SportyBet bet-history LIST page was inspected live (read-only) on 2026-10-01 and its selectors are VERIFIED-LIVE; the login,
  the logged-in betslip, what follows the Place click and the empty-history marker are still unverified. Everything fails closed.
- The history list shows NO bet id, booking code, odds, market or pick. A placed bet is recognised only by its games, stake,
  number of legs, status Running and time (with a synthetic key), and the reply says so: odds and picks are guaranteed only by the
  exact betslip check made just before the click. With more than 3 games only the first 3 can be compared.
- Results limitation for multiples: the list gives one result per BET. A lost multiple is recorded as a lost bet but its legs stay
  UNGRADED (which leg lost is only on the bet details page, not read yet), so it never enters the Calibration win rates (they are
  reported as a separate count). A won multiple's legs are recorded as wins inferred from the bet (a void leg would count as a
  win). Singles are exact. Wording other than Won / Lost / Running (cashed out, void, ...) is reported and left pending.
- Only the first history page (the 5 newest bets) is read: send `Results` before 5 newer bets push a bet off the page.
- MLB, NHL and football game-page wording, and the model's estimates, are uncalibrated. Use `Calibration` over time, and
  treat a band with under 20 decided legs as too small to judge.
- **Soccer market blend (2026-10-11).** For the match result, double chance and goal-total (half line) picks the probability is 75% the bookmaker's own price (its margin removed, read from the same game's listing) and 25% the model (`MARKET_WEIGHT` in `predictions/core/engines/soccer.py`; `0` switches it off). Reason: `python -m predictions.tools.soccer_backtest` replays 11,978 games from football-data.co.uk with closing odds, and the market scored better than the model on the result and on Over 2.5, with the market right whenever the two disagreed. Fewer soccer picks pass as a result (the margin now counts against them); each blended pick says so, and a game with no complete price is priced by the model alone and says that. Corners, cards, team totals and the Review-only markets are not blended. The backtest tests the goals-only model (past xG is not in those files).
- Goals-only soccer picks (ESPN-sourced UEFA and international games, no xG) are experimental and capped: the model has no opponent strength, no home advantage and no calibration, so a goals-only pick whose expected value is above +20% is rejected as unreliable (`GOALS_ONLY_MAX_EV` in `predictions/core/engines/soccer.py`). Review applies the same cap.
- **Book Bet fallback (after the Place click only).** If the wait ends with no success message and the bet is not in your history, the bot presses the slip's `Book Bet` link once (never Place or Confirm), reads the booking code and adds to the Telegram message and the placement record: "No success message was seen. The bet may or may not exist (check history). To place it yourself, open SportyBet and enter booking code XXXXX (stake NGN n, total odds x) - loading it onto your slip is safe." If no code can be read it says so. **A booking code is not proof of a bet**: the result stays UNKNOWN, the games stay blocked, and Check history again / Try again work as before. Not verified live (the Book Bet link and pop-up wording are assumed from your screenshots).
- Failures are reported to you and never retried silently.

## Tests

Install pytest first (`pip install pytest pytest-asyncio`), then:

`python -m pytest tests -q`

## The Monday weekly post-game review (docs/weekly-review-design.md)
Every Monday (09:30 Lagos by default) the bot reviews the Monday to Sunday week that just ended and posts to you: your **lost bets**, which leg failed
and why, and **detailed recommendations** on how the review engine can improve. You decide each one; nothing changes by itself.
- **How legs are judged:** each leg of a placed bet is settled by OUR OWN rules from final scores (ESPN scoreboards: NBA, WNBA, NFL, NCAA, MLB, NHL and many
  soccer leagues). This is a diagnosis; SportyBet's result stays the official one, and a mismatch is flagged. A leg we cannot settle (a league with no free
  score, corners, cards, tennis sets, a game not finished) is listed as unverifiable, never guessed and never counted as a loss. Only bets you confirmed placing
  (`I placed it`) are seen; approved picks you did not place are extra evidence only.
- **Why a leg failed:** one primary reason by priority: void/unsettleable, rule artefact (a swap or step-down changed the pick and the original would have won),
  data quality (AI-analyst inputs, goals-only model, small sample), model miss (the result was 1.5+ standard deviations against the model), or variance (the
  default for an estimate of 70%+: "no change warranted from this bet"). "Unexplained" is allowed.
- **Recommendations:** at most five, each with the finding and its numbers, the exact change (file, constant, old and new value), a replay of that change on the
  saved history, the expected effect, risk, effort, how to check it worked and how to roll it back. Evidence is graded anecdote, weak, moderate or strong by the
  number of settled legs and weeks (Wilson interval); anecdotes and weak findings are only listed under "Watching", never recommended.
- **Your decision:** buttons under each recommendation: **Implement**, **Later** (returns after 14 days) or **Ignore** (stays quiet until its evidence grows 1.5x),
  plus **Details** and **Undo**. `Decisions` lists what you approved; `Decisions all` lists everything; `Done <id>` marks one implemented; `Note <id> <text>`
  adds a note. The same list is kept in `local_data/weekly_reviews/APPROVED.md` for the developer, and each week's full report in `local_data/weekly_reviews/<week>.md`.
- **Commands:** `Weekly` (last completed week), `Weekly 2026-09-30` (the week containing that date), `Weekly dry` (preview, nothing saved, no buttons),
  `Weekly force` (build a posted week again). A Monday missed because the bot was off is made up once on the next start (up to three days later).

## After a restart or a lost network
The bot saves your list after every step. If it restarts (or the network drops and the load or booking failed before anything was placed), it
shows you the **last confirmed games that were not played**, one line each, and asks: **Continue** reloads them on SportyBet and re-reads the odds,
then asks for your stake and a fresh confirmation as usual (it never places anything by itself); **Discard** drops the slate. A slate too old, or
with a game already started, is ended instead. The buttons work once; a stale one says so.

## "SportyBet is not logged in" and the login re-check
SportyBet's login form was seen to flash up for a moment on a page that is really logged in, which stopped a bet with a false "not logged in". The bot
now looks again (twice, 2 seconds apart) before it believes it, and the first time the form shows it saves a screenshot (every input box masked) and a
structure note (no values) as `local_data/screenshots/logincheck-<load|place>-<time>.png` and `.txt`, stored on the PC only and never sent anywhere.
Send those files to the developer if the error still appears.

## Soccer markets Review can price (Review only; Matchday and the other slates still use only their five plays)
Besides 1X2, double chance, goal totals, team totals, corners and cards, `Review <code>` now prices, from the same scoreline model: **Asian handicap**
(whole, half and quarter lines, with the refund maths), **Multigoals** (full time), **GG/NG** (both teams to score), **Draw No Bet** (a draw refunds),
**Home / Draw / Away Team or Over L** (the "Yes" side), **1X2 - 2UP** (priced as the plain 1X2 chance: the early-payout promotion is not counted, so the
estimate is conservative), the **3-way handicap** (Handicap 0:1 and the like) and the **second-half** Multigoals and Double Chance and **Both Halves Under**
(these use an ASSUMED split of 45% of a game's goals in the first half and 55% in the second, because no half data is stored; each one says so).
A side that is an Under in disguise (GG/NG "No", "Home Team or Over" "No", Both Halves Under "Yes") is shown as disqualified by your no-Under rule.
Still without rules: Goal Bounds and Excluded Number of Goals (meaning not confirmed), Early Goals (needs goal times), team shots, and tennis.

## Combining plays into one list
You can join plays in one message and get ONE final list, for example `Review 3ercer and America 6 October` or `Gameday, Matchday and America 5 and 6 October`.
- **Each part runs exactly as it does alone** (its own rules, floor, no-Under and no-whole-line rules) and posts its own report, headed "Part 1 of 3: ...".
  Only the surviving picks are merged: a slate's passing picks and a Review's approved selections. The same pick found twice is listed once, and two picks on
  one game become options of that game, as usual. The merged list is saved under one session and is offered like any other (Keep / Delete / Final confirm).
- **Words:** `Matchday`, `Gameday`, `America` and `Review`, joined by `and`, a comma, `+` or `&`, in any order and any case; each at most once, up to four parts.
  `Review` takes one booking code. A date play written without dates uses the dates of the next date play (else the one before), so a date at the end
  applies to all of them. A part that fails is reported and left out; the others still count. If nothing passes anywhere, no list is offered.
- **One at a time:** a combined request will not start while one of its parts (or another combined request) is still running.
