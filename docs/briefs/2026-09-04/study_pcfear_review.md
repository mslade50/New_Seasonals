# Brief: study_pcfear_review (post-ship review, part 1, of the P/C-fear family band table)

Date 2026-09-04. Plan: `docs/plan_2026-09-04.md` (section 0 hard rules, D8, section 6 report format). Type: RESEARCH. Read-only on the repo; you write only under `scratch/ultracode_research/pcfear_review_2026-09-04/` (scripts, results JSON, and one `REVIEW.md`), which the mind will commit as evidence.

## Decision and why
On 2026-08-05 the six frag-band carriers (Weak Close Decent Sznls, SPY QQQ MonFri Reversion, Monday Dip, Indices Oversold Bounce, 3x Bear ETF Overbot Fade, Monthly Weak Close) went live with a fear-selected band table: fear ON `[[0,50,1.25],[50,999,1.0]]`, fear OFF `[[0,50,1.0],[50,999,0.0]]`, stale `[[0,50,1.0],[50,999,0.25]]`. It shipped ahead of its gates as an appetite decision; the prereg (`scratch/ultracode_research/family_pc_fear_band_prereg_2026-08-05.md`, read it in full first) converted gates 1-3 into a post-ship review whose part 1 has never been run. Since 2026-07-30 the dial's 10d-MA 63d has been above 50 with fear OFF, so the family has been staged at 0 shares for five weeks; that zeroed set includes three of the top four trailing-24-month Sharpe contributors. The mind needs the review numbers to decide, per leg, STAND or ROLL BACK to the incumbent 0.25x. You compute; you do not decide and you change no config.

## Files you own
`scratch/ultracode_research/pcfear_review_2026-09-04/` only. Nothing else.

## Hard rules
Section 0 of the plan. Do not modify `pc_fear.py`, `strategy_config.py`, any test, or any parquet. Do not rebuild the production ledger. If `data/backtest_trades_pcfear_shadow.parquet` is absent locally, you may download it read-only from R2 into your folder via `cache_io.download_to_local` if creds exist; otherwise reconstruct the zeroed set from the main ledger plus the band logic and say so.

## Intent
Reproduce the prereg's protocol exactly, on two dial vintages, with the lag-1 fear state:
- Dial vintage A (primary): the extended point-in-time series in `scratch/ultracode_sizing_2026-09-02/dd_pit/` (expanding-window weights, through 2026-09-01; find the parquet and its README). Vintage B: the live `data/rd2_fragility.parquet` 63d column, 10d MA. State for each trade which vintage scored it.
- Fear state: `pc_fear.py`'s definition, lag-1 by data date (newest row dated <= D-1 bday), from `data/cboe_putcall.parquet`. Do not re-implement the percentile; import the module's function.
- Population: every family trade in `data/backtest_trades_full.parquet` (tranche rows collapsed to positions; R = sum PnL / sum risk) 2016-06 onward, plus the zeroed signals from the shadow parquet for the 2026-08-05 onward window.

Compute and report:
1. The 2x2 (dial < 50 / >= 50 by fear OFF / ON) with N, avgR, win rate, date-clustered t, and Mann-Whitney, on vintage A and vintage B.
2. Gate 1a: the no-fear hi-frag deficit, clustered sigma vs the no-fear lo-frag cell (prereg requires <= -1.5).
3. Gate 1b: the fear-ON hi-frag cell avgR, date-clustered (prereg requires >= +0.3R).
4. Gate 1c: the sensitivity grid, fear threshold {80, 85, 90} x dial threshold {45, 50, 55}, showing 1a and 1b at every cell.
5. Leg B non-inferiority: fear-ON dial<50 vs no-fear dial<50, clustered difference (must not be worse by more than 0.1R).
6. Gate 3 LOYO on the fear-ON hi-frag cell by episode-year (2021, 2022, 2026 and any new).
7. Aug-2026 out-of-sample scoring for leg C: every family signal since 2026-08-05 that the fear-OFF table zeroed. For each: strategy, ticker, signal date, dial, fear pctile, and what it would have earned at 0.25x and at 1.0x from the shadow pass (or from bars if the shadow is unavailable, stated). Totals in R and flat dollars. This is a report line, not a gate; say plainly that it is one episode.
8. Live-regime status line: dial 10d-MA 63d and fear pctile for each session since 2026-07-30, so the mind can see how long the family has been off and whether release is near.

## Recon first
`scratch/ultracode_research/pcfear_review_2026-09-04/00_plan.md`: the files you will read, the vintage you found, how you will import `pc_fear` headless (look for the `_NoOp` Streamlit stub pattern under `scripts/`), and the clustering method.

## Verification
`scratch/ultracode_research/pcfear_review_2026-09-04/checks.json` written by your script:
`{"vintage_a_path": "...", "vintage_a_last_date": "...", "trades_scored": int, "gate_1a_sigma_A": float, "gate_1a_sigma_B": float, "gate_1b_avgR_A": float, "gate_1b_t_A": float, "gate_1b_n": int, "grid_cells_passing_both": int, "grid_cells_total": 9, "legB_diff_R": float, "loyo_min_avgR": float, "loyo_all_positive": bool, "aug2026_zeroed_n": int, "aug2026_at_025_R": float, "aug2026_at_100_R": float, "aug2026_at_100_usd": float, "family_off_sessions_since_0730": int}`.
No screenshots.

## Report
Section 6 format. Findings: the gate table (gate, threshold, value A, value B, PASS/FAIL). Handoff: for each leg (A, B, C) state which gates it passes and fails on each vintage. Do not recommend a multiplier; the decision set is closed and belongs to the mind.
