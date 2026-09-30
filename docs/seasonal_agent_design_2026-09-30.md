# Daily Seasonal Agent: design brief (2026-09-30)

Status: proposal, not built. Owner decision needed on the five open questions at the end.

## Why

The site's seasonal board is a deterministic scanner: fixed horizons (5/10/21), fixed gates, a fixed 0.6-0.8 ATR ticket. Its forward record since the 2026-07-24 gates is 1 win in 11 completed tickets (-9.3R), and the 2026-09-30 audit found its cycle cells were anchored before the tradeable entry (fixed the same day). The four tickets it printed today all died under the Daily Pitch battery: two were sector beta, one flipped sign at the entry, one was a real 26-year pattern that has decayed since 2016 and sits in its losing regime branch. Nothing in the scanner could have said any of that. A scanner ranks cells; an agent falsifies trades.

The Daily Pitch already does this once a day, but seasonality is one of seven novelty axes there and gets one or two checks on a busy morning. The proposal is a second agent that owns the seasonal surface end to end: every rank outlier, every cycle cell, every calendar cell, with horizons from 3 to 63 sessions and an exit chosen to fit the shape of the pattern rather than a fixed 2:1 ticket.

## What it reuses (nothing new below this line is machinery)

| need | existing asset |
|---|---|
| seasonal ranks, 1,025 names, 6 horizons (5..252d), 25/75 cycle blend | `atr_seasonal_ranks.parquet` at repo root (refreshed on every deploy) |
| day-of-year window stats, cycle cohorts, expected path | `scripts/seasonal_edge.py` (`seasonal_window_returns`, `expected_seasonal_path`, `cycle_phase`) |
| turn-of-month, holiday, weekday-month, cycle-cell sweeps | `scripts/build_context_state.py` (`month_window_anchors`, `holiday_adjacent_anchors`, `sweep_seasonal_cells`) |
| event calendar 2000-2027 | `macro_calendar.py`, `data/macro_events.csv` |
| falsification toolkit | `pitch_lab.py` (`battery`, `horizon_scan`, `episode_paths`, `filter_vs_reanchor`, `sign_test`, `cluster_note`) |
| grammar, sizing caps, disk-evidence gates, delivery receipts, journal, grader | `pitch_grammar.py`, `daily_pitch.py`, `pitch_delivery.py`, `pitch_journal.py`, `scripts/grade_pitch_journal.py` |
| launcher chain | `scripts/run_daily_pitch.bat` + `invoke_daily_pitch_agent.ps1` + `register_daily_pitch_task.ps1` |

## Shape of the run

**Stage A, state (code).** `scripts/build_seasonal_state.py` writes `data/seasonal_state.json`:

- the rank cross-section for today across all 1,025 names and all six horizons, with the class map from the pitch's B1 table (US large, small, rates, credit, gold, metals, energy, dollar, international, vol) plus GICS sector from `data/sector_map.parquet`
- outliers: rank <= 10 or >= 90 at any of 5/10/21/63d, flagged when two adjacent horizons agree, plus the rank-path turn (the expected path's nadir/peak within the next 5 sessions, from `expected_seasonal_path`)
- the cycle-cohort raw counts for each outlier, re-anchored at T+1 and at the path turn (never lag 0)
- calendar cells live today: turn-of-month, holiday adjacency, opex/FOMC/CPI offsets, month-of-year, cycle year, all from the context-state sweep helpers
- the same context the pitch carries: dial, P/C state, tape extremes, staged signals, earnings inside 63 sessions, and the seasonal journal's recent fingerprints and watchlist

**Stage B, survey.** A `00_surface_map.md` on disk, enforced by the publisher exactly as the pitch does it. Three enumerations: every rank outlier by asset class and sector, every calendar cell inside the next 10 sessions, every active watchlist entry. Coverage floors: four asset classes, at least one macro (rates/FX/commodity) candidate, at least one non-equity vehicle, at least one short. Select 8 to 12.

**Stage C, falsification.** The pitch battery plus five seasonal-specific rules written from today's kills:

1. Entry-anchored windows only. The historical window starts at the close the order would fill at, never the day-of-year close.
2. Index and sector residual is mandatory. Report the return net of SPY and net of the sector ETF, with its own hit rate and sign test. A seasonal whose residual is zero is an index seasonal and must be pitched as one or killed.
3. Cycle cells report drop-best and drop-two-best, and the non-cycle cohort alongside, so a conditioner that adds nothing is visible.
4. Regime branch: split the history by SPY within 2% of its 52-week high at entry and by the fragility dial where the PIT history allows. Report which branch today sits in.
5. Recency: the last 10 years individually with max adverse excursion in ATR. A 25-year record with a coin-flip last decade is grade C at best.

Round 3 develops the exit from the shape: `horizon_scan` over 1..63 sessions picks the hold, `episode_paths` shows where the dip sits, and the stop is chosen from the adverse-excursion table (time-only, catastrophe at 3 ATR, or a trail armed after a stated MFE), never assumed.

**Stage D, compose and publish.** Same grammar as the pitch with two extensions: `horizon_td` up to 63, and an `exit.trail` block (arm after k ATR of MFE, trail at m ATR). Sizing is `risk_bps` off the chosen stop or, for time-only exits, off a 3 ATR catastrophe stop; the pitch caps stay (100 bps risk per idea, 60 bps ATR risk per idea, 150 bps across the slate). Up to three ideas, short-slate and stand-down blocks as in the pitch. Delivery is a separate email ("Daily Seasonal") and its own Sheets tab. No auto-staging in v1.

**After.** Own journal (`data/seasonal_journal.jsonl`), own scoreboard, own watchlist and negative registry. The pitch state builder reads the seasonal journal's fingerprints so the pitch never re-pitches a seasonal idea inside 10 sessions; the seasonal agent reads the pitch registry read-only.

## Timing

Recommend an evening run, about 17:45 ET after the post-close deploy refreshes the rank file. Seasonal cells are known in advance, so the ideas can carry close-anchored limits for the next session and McKinley reads them with the context brief rather than at 5 AM. The morning pitch then sees the seasonal slate in its state and skips those cells.

## Build plan

| phase | work | new files |
|---|---|---|
| 0 (1 session) | state builder; skill file cloned from the pitch with the seasonal stages above; validator gains `horizon_td <= 63` and `exit.trail`; publisher gains a `--product seasonal` switch for paths, email subject and tab name | `scripts/build_seasonal_state.py`, `.claude/skills/daily-seasonal/SKILL.md`, tests for the grammar extensions |
| 1 (1 session) | journal, scoreboard, grader replay for trails; launcher and Task Scheduler registration; `docs/claude_ref/daily_seasonal.md` | `scripts/run_daily_seasonal.bat`, `scripts/register_daily_seasonal_task.ps1`, `scripts/grade_seasonal_journal.py` (or a `--product` flag on the pitch grader) |
| 2 (after 20 graded ideas) | decide whether the board's ticket channel is retired, kept as the agent's candidate feed only, or left as is | none |

Cost: one opus run at the pitch's effort, roughly the pitch's token bill, per trading day.

## Open decisions for McKinley

1. Evening run (recommended) or a second morning run after the pitch?
2. Universe: the 1,025 rank names only, or also the overflow tier (needs `append_atr_seasonal_ranks`)?
3. Default risk per idea: 30 bps as the pitch, or a lower 20 bps while the scoreboard is under 20 ideas?
4. Keep the board's ticket channel live on the site while the agent runs, or demote it to "candidates" until phase 2?
5. Share the pitch negative registry or keep a separate seasonal one (recommended separate, cross-read)?
