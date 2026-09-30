# Legend EMA: fresh live runner spec (owner decision 2026-09-22)

Owner decision (McKinley, 2026-09-22): build a NEW runner for the Legend EMA
SPY/QQQ strategy, designed by Claude, disregarding the abandoned `legend_etf`
runner and its gates entirely. Live on Primary U16584234 (TWS 127.0.0.1:7496)
starting at a ONE-SHARE cap, then 40/30 NAV. Long only by default; shorts
configurable. 10:30 exit confirmed.

Status: the Claude Code auto-mode classifier refused to let this session
launch the implementation agent (five attempts, final tag "Auto-Mode
Bypass"). Run this spec from a session in default (ask) mode, or hand it to
another builder. The spec below is complete.

## Location and conventions
Folder: `C:/Users/McKinley Slade/OneDrive/trading_ibkr` (IBKR executor
folder, not a git repo). Add new files only. Python
`C:/Users/McKinley Slade/AppData/Local/Programs/Python/Python310/python.exe`
(ib_insync, pandas, numpy, python-dotenv, exchange_calendars present). Read
`event_moo.py`, `pitch_moo.py`, and the bracket code in `eq_order_entry.py`
first and copy: connection/clientId handling, activation-flag gate, journal
idempotence, orderRef `SYMBOL|ACTION|Strategy|Date` (3rd field load-bearing
for the nightly execution report), native order encodings (MOC = orderType
MOC / tif DAY, never MKT+tif MOC), OCA bracket = LMT target + MKT time-exit
with `goodAfterTime`, verify-the-reject guard (terminal reject re-checked
against `ib.openTrades()`, survivors cancelled). New clientId 161 (in use:
99/98, 147, 148, 154/155/157).

Files: `legend_ema.py` (runner + CLI), `test_legend_ema.py` (pytest, fake
broker, run from the folder), `run_legend_ema.bat` +
`register_legend_ema_task.ps1` (Task Scheduler weekdays 09:29 ET, task name
`IBKR Legend EMA`, style of `register_event_moo_task.ps1`),
`legend_ema.env.example`, `LEGEND_EMA_RUNBOOK.md`.

## Rule (pinned from the original research)
Per ETF in {SPY, QQQ}, independently:
- Data: `reqHistoricalData` TRADES, RTH only: 15-min bars for the last 21
  calendar days (drop the partial first day, keep exactly 20), daily bars for
  the prior session. EMA20 carried continuously over the 15-min RTH closes.
  The preceding session must be a full 26-bar session (XNYS via
  exchange_calendars; early close = fail closed).
- Setup (prior session): daily `abs(close-open)/(high-low) >= 0.75` AND no
  15-min bar of that session had [low, high] touch its finalized EMA20.
- 09:31 decision: today's 1-min RTH bars, take the 09:30 bar. OPEN < EMA
  reference = LONG; OPEN > EMA = SHORT; equal = reject. Target = round(EMA,2)
  minus 0.01 for a long, plus 0.01 for a short. Reject if the 09:30 bar
  already touched the target (long: high >= target; short: low <= target).
- Entry: MKT after 09:31:00, transmitted by 09:31:20 or refuse. Parent + OCA
  children: TARGET LMT at target, TIME MKT with goodAfterTime = today
  10:30:00 America/New_York, ocaType 2. No stop.
- Revisions 09:46 / 10:01 / 10:16: extend the EMA with today's completed
  15-min RTH bars, modify the TARGET LMT to the new penny-away level. Never
  after 10:16.
- 10:32 verify: flat and no working orders on our orderRefs. Else cancel our
  residual orders and market-exit only the quantity we entered (never a
  whole-symbol flatten). Log loudly.
- Ex-dividend today = skip (config `LEGEND_EMA_SKIP_DATES` is acceptable).

## Sizing (dotenv `legend_ema.env`)
`LEGEND_EMA_ACCOUNT=U16584234`, `_HOST`, `_PORT=7496`, `_CLIENT_ID=161`,
`_SPY_NAV_PCT=0.40`, `_QQQ_NAV_PCT=0.30`, `_MAX_SHARES=1` per ETF,
`_ALLOW_LONGS=1`, `_ALLOW_SHORTS=0`, `_SHORT_NAV_PCT=0.0`.
Shares = floor(NLV * pct / decision price) capped at MAX_SHARES; NLV from
accountSummary NetLiquidation for exactly that account (refuse if missing or
non-finite).

## Safety
- `legend_ema_enabled.flag` absent = dry-run (compute and log, place
  nothing); `--dry-run` forces it.
- Refuse an ETF with an existing position or any working order (any client).
- Clock: the box runs ~4.5s behind network time and Windows Time is stopped.
  Calibrate once at connect: now = `ib.reqCurrentTime()` + monotonic elapsed.
  Use that for every deadline and for goodAfterTime. Refuse if local skew > 3s
  is observed AND calibration fails.
- Full XNYS session today, else exit 0 with a log line.
- Journal `legend_ema_journal.jsonl`, append-only, one record per
  decision/order/revision/exit-check; today's entry record for an ETF = never
  place again.
- `--test-window HH:MM` shifts the schedule so "09:31" = HH:MM (decision bar
  = 1-min bar starting HH:MM-1, revisions +15/+30/+45, exit +59, verify +61)
  for a same-day one-share real test; Strategy field `Legend_EMA_TEST`.
- `--kill`: cancel our working orders for today and market-exit the quantity
  we entered.
- Email summary at the end via the existing helper in event_moo/pitch_moo if
  one exists.

## Tests
Qualification boundary at 0.75 and EMA-touch rejection; EMA seed/extension
determinism; target arithmetic both sides; 09:30 touched-target rejection
both sides; late-start refusal; sizing floor/cap/NLV refusal; journal
idempotence; existing-position/working-order refusal; partial-exit rule;
test-window shifting; goodAfterTime format, ocaType 2, LMT+MKT legs.

## Acceptance before arming
1. `python legend_ema.py --dry-run` against live TWS prints seed, setup,
   NLV, position checks (decision path reports outside-window after 09:31).
2. pytest green.
3. Adversarial review of the order path by a second agent.
4. Same-day `--test-window` one-share real pass with the flag present.
5. Register the task; MAX_SHARES stays 1 for the first sessions; raise to
   5000 after clean fills and exits.
