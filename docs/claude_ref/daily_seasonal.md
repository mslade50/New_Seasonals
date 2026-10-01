# Daily Seasonal

Built 2026-09-30 from `docs/seasonal_agent_design_2026-09-30.md` (owner
decisions the same day: MORNING run after the pitch, moved 2026-10-01 to 04:30, before the pitch; 30 bps default with the
agent free to pick 15-50 bps by conviction; the site board's ticket channel
stays live; own negative registry, the pitch's read-only).

A second agent product that owns the seasonal surface end to end: every rank
outlier, cycle cell and calendar cell, horizons 3 to 63 sessions, exits
developed from the pattern's shape. It is the Daily Pitch's machinery with a
product switch, not a second pipeline. Nothing places orders in v1.

## Live rule

Flow (weekdays 04:30 local, moved from 07:00 on 2026-10-01, Task Scheduler job "Daily Seasonal (agent)"):
`scripts/run_daily_seasonal.bat` ->
`pull_scan_caches.py --set pitch` (includes the ROOT `atr_seasonal_ranks.parquet`) ->
`grade_pitch_journal.py --product seasonal` ->
`scripts/build_seasonal_state.py` (`data/seasonal_state.json` + `data/seasonal_tape.json`) ->
`scripts/invoke_daily_seasonal_agent.ps1` (`claude -p /daily-seasonal`, opus / xhigh,
6300 s timeout, bypassPermissions: the pitch's pins exactly) ->
`check_pitch_delivered.py --product seasonal --require-r2`.
The skill (`.claude/skills/daily-seasonal/SKILL.md`) writes
`scratch/seasonal_checks/<date>/00_surface_map.md` plus `.py` checks, composes
`data/seasonal_agent_ideas.json`, and publishes with
`python daily_pitch.py --product seasonal --ideas data/seasonal_agent_ideas.json`.

The register script mirrors the pitch's exactly: it points Task Scheduler at
the `.bat` beside itself, in whatever checkout it is run from, with `python`
from PATH (no venv, no pinned worktree). Run it from the checkout that should
own the job. Not registered as of 2026-09-30.

### Product switch (`pitch_products.py`)

| | pitch (default) | seasonal |
|---|---|---|
| checks root | `scratch/pitch_checks` | `scratch/seasonal_checks` |
| journal (R2 key) | `data/pitch_journal.jsonl` | `data/seasonal_agent_journal.jsonl` (`seasonal_agent_journal.jsonl`) |
| scoreboard | `data/pitch_scoreboard.json` | `data/seasonal_agent_scoreboard.json` |
| watchlist | `data/pitch_watchlist.json` | `data/seasonal_agent_watchlist.json` |
| negative registry | `data/pitch_negative_registry.md` | `data/seasonal_agent_negative_registry.md` (own registry, standalone; the pitch registry is not read) |
| state | `data/pitch_state.json` | `data/seasonal_state.json` |
| receipts (local / R2 prefix) | `data/pitch_delivery_receipts/` / `pitch_delivery_receipts/` | `data/seasonal_agent_delivery_receipts/` / `seasonal_agent_delivery_receipts/` |
| subject | `Daily Pitch - <date> - N ideas` | `Daily Seasonal - <date> - N ideas` (stand-down `... - NO TRADES (k killed)`) |
| Sheets tab | `Pitch` | `Seasonal Agent` (pitch columns + `Trail_Arm_ATR`, `Trail_ATR`) |
| idea id | `<date>-N` | `<date>-SN` (never collides with a pitch id or a `Pitch-{id}` fill tag) |
| Scan_Source | `Pitch` | `Seasonal_Agent` |
| site Pitch tab payload | yes | no (skipped; the site tab is the pitch's) |
| fills approvals in the grader | yes | no |

The pitch reads its module globals (`pitch_journal.JOURNAL_PATH`,
`pitch_delivery.RECEIPT_DIR`, `pitch_grammar.CHECKS_ROOT`, `daily_pitch.TAB_NAME`
...) at call time, so every pitch path and behaviour is unchanged; guarded by
`tests/test_seasonal_agent_publish_paths.py`.

The seasonal publisher also blocks any idea whose fingerprint the PITCH journal
shows inside the 10-td repetition window (read-only), unless `changed_since`
says what changed. Since the 2026-10-01 move to 04:30 the run precedes the
pitch, so this dedup covers prior days only and a same-day duplicate is possible.

### Grammar extensions (`pitch_grammar.py`, `product="seasonal"`)

- `horizon_td` 1..63 (`MAX_HORIZON_TD_BY_PRODUCT`); `exit.time_td` 1..horizon.
  The pitch's cap was already 63 and is unchanged.
- Optional `exit.trail {arm_atr > 0, trail_atr > 0}`: once MFE from entry
  reaches `arm_atr` ATR, a stop trails `trail_atr` ATR behind the best close
  (above it for shorts). Legal with or without `stop_atr`. A trail makes the
  idea `manual` in `auto_placement`. A trail on a PITCH idea is an error.
- Sizing: always `risk_bps`, 15..50 (default 30 when omitted);
  `stop_atr_for_sizing` REQUIRED and >= 1.0; with no `stop_atr` (time-only
  exit) it is the catastrophe distance and must be >= 3.0.
- `novelty_axis` is product-scoped (`NOVELTY_AXES_BY_PRODUCT`): the seasonal
  takes `SEASONAL_NOVELTY_AXES` (rank_outlier, cycle_cell, calendar_cell,
  path_turn, relative_value, instrument_translation, inversion,
  historical_analogue), the skill's stage B2 list. The pitch-only axes
  (interaction_cell, flow_mechanics, event_fingerprint) are refused on a
  seasonal idea and the seasonal-only ones on a pitch idea. Added 2026-10-01:
  the first live run's valid seasonal axes failed against the pitch set.
- Everything else is the pitch's: survey-map gate, `dev_script`, short slate
  and stand-down floors, kill-lint, one grade C, and the ATR-risk caps
  (60 bps per idea, 150 bps per slate, 4 legs).

### Grader trail replay (`scripts/grade_pitch_journal.py`)

Rows with `Trail_Arm_ATR` / `Trail_ATR` arm once a bar after the fill reaches
the arm MFE (bar high for longs, low for shorts); from the NEXT bar a stop sits
`Trail_ATR` behind the best close since the fill, ratcheting one way, and the
tighter of it and any fixed stop rules. It fills like a stop (3 bps, +10 on a
gap) and books `trail` / `trail_gap`. Rows without the columns replay exactly
as before.

### State builder (`scripts/build_seasonal_state.py`)

Reuses `build_pitch_state` for calendar, tape, risk, book, earnings (widened to
63 td and to the outlier names), history, watchlist and pipeline. Adds:

- `ranks`: outliers of the root rank file. Rank <= 10 / >= 90 at 5/10/21/63
  with two ADJACENT horizons in the same tail, or one horizon <= 5 / >= 95;
  $5M 21d ADV floor on shares/ETFs (indices, FX, futures, crypto exempt);
  sorted by max |rank-50|, capped at 80. Each carries class (pitch B1 table,
  extended), sector (`data/sector_map.parquet`), agreeing horizons, path turn
  (expected-path nadir/peak within 5 sessions; 0 = never adverse), cycle and
  all-years n/k/mean from a T+1 entry over the smallest agreeing horizon
  (`seasonal_window_returns(entry_lag=1)`; mean_atr uses each year's ATR at
  the entry bar), trailing-return percentiles, ATR, close, ADV.
- `calendar_cells`: month-end, turn-of-month, holiday pre/post, weekday x month
  and opex/FOMC/CPI/NFP/VIX-expiry/quad-witching inside [today, +10 td], each
  with its historical anchor count (Market Context anchor definitions).
- `board`: `daily_seasonal_ideas.build(grades=None)` in-process, stdout and
  warnings swallowed, nothing written, no ledger append; rows with
  `evidence.TICKET` only.
- `history.pitch_recent_fingerprints`, `negative_registry` (text + entries).
  Own registry, standalone; the pitch registry is not read.
  History: 2026-09-30, owner decision (McKinley) removed the read-only
  `pitch_negative_registry` block from the state and the skill.

## Aligned sites, change together

- `pitch_products.py` (the product table)
- `pitch_grammar.py` (extensions), `daily_pitch.py` (`--product`, tab, subject,
  receipts, site skip), `pitch_delivery.py` (receipt namespace, journal key),
  `pitch_journal.py` (`r2_key_for`)
- `scripts/grade_pitch_journal.py`, `scripts/check_pitch_delivered.py`,
  `scripts/build_seasonal_state.py`, `scripts/build_pitch_state.py` (shared blocks)
- `scripts/run_daily_seasonal.bat`, `scripts/invoke_daily_seasonal_agent.ps1`,
  `scripts/register_daily_seasonal_task.ps1`
- `.claude/skills/daily-seasonal/SKILL.md` (names in the contract above)
- `.gitignore` (seasonal state/tape/ideas/receipts ignored; journal, scoreboard,
  watchlist and registry committed)

## Guard tests

`tests/test_seasonal_agent_grammar.py`, `tests/test_build_seasonal_state.py`,
`tests/test_seasonal_agent_publish_paths.py`, `tests/test_seasonal_agent_grader.py`,
plus every Daily Pitch guard (the switch runs through the pitch's code).

## Not built yet

- The pitch state builder does not yet read the seasonal journal's
  fingerprints (the design's reverse direction); the pitch publisher does not
  block seasonal fingerprints. Since the 04:30 move the seasonal run also
  precedes the pitch, so a same-day duplicate across the two products is possible.
- No site tab, no auto-staging, no overflow-tier ranks (the 1,025 rank names only).
