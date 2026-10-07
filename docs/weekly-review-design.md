# Monday weekly post-game review: design (merged from the team brainstorm, 2026-10-03)

Owner's decisions: lost bets the owner placed are the main subject; approved-but-unplaced picks that missed are extra
evidence only; each leg of a lost bet is settled by OUR OWN settlement from final scores (a diagnosis, never the official
result: the bookmaker decides). The owner then chooses per recommendation: Implement, Ignore or Later.

## 1. What is real today (data audit)
- `placements`: 14 rows (a "placed" bet is state PLACED_BY_OWNER, dry_run = 0). `leg_outcomes` / `bet_outcomes` are empty, and a lost
  multiple does not say which leg lost, so our own settlement is the only leg-level source.
- `evaluations.placed` is 0 on every row: join `placed_legs` (market_id, specifier, side are authoritative) to the LATEST evaluation of
  the same match and market at or before `placed_at_utc`.
- Final scores are not stored anywhere: the review fetches and caches them.
- Free score coverage (peer probes, 2026-10-01): ESPN scoreboard by date for NBA, WNBA, NFL, NCAA, MLB, NHL and about 219 soccer
  leagues (quarter, period and inning line scores); football-data.co.uk CSVs for Argentina, Brazil, Mexico, Japan, USA, Poland, Ireland,
  Finland (results lag days; half-time goals, corners and cards only for major leagues); EuroLeague keyless API. Nothing free for Turkish,
  Greek, Polish or Finnish basketball, nor for Colombia, Guatemala, Bolivia soccer. Such legs are reported as UNVERIFIABLE, never guessed.

## 2. Settlement rules (core/settle.py)
Leg identity = (sport, market_id, specifier, side) from `placements.legs[].leg_key`, never the market text.
- `hcp` in the specifier is the HOME handicap; the away line is its negative. Margin + hcp > 0 wins, = 0 pushes, < 0 loses.
  A quarter line splits the stake over the two adjacent lines (half win / half loss).
- Totals: total above the line = Over; equal = push on a whole line.
- Soccer markets use the 90-minute score only (no extra time, no penalties). Basketball, American football and baseball markets include
  overtime / extra innings. Hockey winner / handicap / total include OT and shootout.
- 1st-half markets (basketball and NFL id 68): first two periods only, overtime excluded; needs period scores.
- MLB first five innings (274-276): innings 1-5; a game that did not reach five full innings voids the leg.
- Corners, cards, tennis sets, early goals: UNVERIFIABLE (needs counts or event times).
- Postponed / abandoned: VOID, never blamed on the model. An NFL tie in a winner market: UNVERIFIABLE (rule unknown).
- A pushed or voided leg cannot be the leg that lost a bet.

## 3. Failure-reason taxonomy (core/postgame.py)
One loss can fire several tests; ONE primary reason is chosen by priority, and "unexplained" is allowed (never forced):
P1 void/unsettleable > P2 game-state (needs an event feed; otherwise cannot fire) > P3 rule artefact (swap / step-down / shield whose
ORIGINAL pick would have won) > P4 data quality (analyst-sourced inputs, goals-only model, small sample, flagged but placed) > P5 model
miss (|z| >= 1.5 against the pick AND the line missed by >= 1 sd) > P6 market-specific (pooled evidence only) > P7 variance (default for
p >= 70% with nothing else firing: "no change warranted from this bet"). Per bet: joint probability = product of leg probabilities and
the weakest failed leg; a multiple with joint p < 0.5 is variance by construction.

## 4. Statistical guardrails
Unit = the LEG, denominators include wins, pooled across weeks since the last constant change. Wilson 95% interval vs mean estimate per cut
(sport x market kind x model, rule, source type). n < 10 anecdote (listed once, never recommended), 10-19 weak (watch), >= 20 moderate,
strong needs n >= 30, >= 3 distinct weeks, Wilson upper bound below the estimate and a stated mechanism in code. A mechanism fix (bug,
wrong settlement, missing input) may be a single instance. Unplaced approved picks count as labelled extra evidence only.

## 5. Recommendation record
id (`2026-W41-03`), topic_key (stable slug, e.g. `goals-only-ev-cap:soccer`), title, category, finding with numbers (n, wins, hit rate,
Wilson interval, mean estimate, gap, weeks), evidence leg ids (placed vs extra), proposed change (file, constant, old -> new), expected
effect, risk, effort, confidence (anecdote / weak / moderate / strong), validation, rollback, status (new / implement / ignore / later /
done), decided_at, note. Ten templates (core/postgame.py): goals-only EV cap, goals-only haircut, analyst-sourced inputs, market-band
calibration, step-down rules, 1X2 swap, derby/cup table, lineup freshness, small-sample gate, accumulator size / weather and pace guards.
Anti-overfitting: nothing changes from fewer than 3 weeks and 30 legs unless a mechanism is shown; one change at a time per cut; at most
5 findings posted; ignored items stay muted until their evidence grows 1.5x (and 3 more legs); "later" items resurface after 14 days.

## 6. Delivery (flow/weekly.py, telegram/weeklyview.py)
- Every Monday 09:30 Africa/Lagos (WEEKLY_REVIEW_TIME), for the previous Monday-Sunday Lagos week; week key = ISO week of that Monday.
- Restart catch-up: if the last completed week has no posted row and today is within WEEKLY_CATCHUP_DAYS (3) after the Monday, run once. A
  UNIQUE week_start row makes the job and the catch-up idempotent. WEEKLY_REVIEW=off disables the job.
- Manual: `Weekly`, `Weekly <date>`, `Weekly dry` (renders, saves nothing), `Weekly force` (new revision, never overwrites decisions).
- Message 1 = summary (counts, one block per lost bet, reasons tally); then ONE message per recommendation with buttons
  [Implement] [Later] [Ignore] [Details]; a footer; the full detail is written to `local_data/weekly_reviews/<week>.md` and sent as a document.
- Decisions: callback `w:<week>:<seq>:<action>` (<= 64 bytes), idempotent, Undo; `Decisions` lists approved items; every change rewrites
  `local_data/weekly_reviews/APPROVED.md` for the developer. Tables `weekly_reviews` and `recommendations` (store migration v7).
- Failure modes: zero lost bets (short message, row still recorded); unverified legs ("cause unverified", never a recommendation on its own);
  score sources down (degraded summary without reasons, re-run with `Weekly force`).

## 7. Build order
A. core/settle.py + tests. B. score sources (ESPN scoreboards, cached) + name matching. C. core/postgame.py (diagnostics, guardrails,
recommendations) + tests. D. store migration, flow/weekly.py, Telegram view, scheduler, commands + tests. E. README and a dry run on real data.
