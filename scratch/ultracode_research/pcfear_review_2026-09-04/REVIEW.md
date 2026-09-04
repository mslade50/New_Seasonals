# P/C-fear family band: post-ship review, part 1 (2026-09-04)

Brief: `docs/briefs/2026-09-04/study_pcfear_review.md`. Prereg:
`scratch/ultracode_research/family_pc_fear_band_prereg_2026-08-05.md` rev 3.
This folder computes; it decides nothing. Every number below is in
`02_review.log` / `results.json` / `checks.json` / `trades_scored.csv`
(written by `02_review.py`) or `01_shadow_2026.log` (written by
`01_shadow_2026.py`). Recon: `00_plan.md`.

## Inputs

- Population: `data/backtest_trades_pcfear_shadow.parquet` (pc_fear
  DISABLED pass = incumbent 0.25x table everywhere; local build
  2026-08-07T19:12Z, sha 34afae8, 1163 family rows, last signal
  2026-07-29) plus 6 positions after that date from this folder's
  family-only engine re-run with `pc_fear_enabled=False`
  (`shadow_2026_pcfear_off.parquet`). No R2 copy of the shadow exists
  (`build_trade_ledger.py` uploads only the main ledger key), so the
  post-07-29 window is reconstructed from bars BY THE PRODUCTION ENGINE
  (`process_signals_fast`, 6 family strategies, 42 tickers, candidates from
  2026-01-01), not by hand. Parity of that re-run (`01_shadow_2026.log`):
  the production-rule pass reproduces the 2026 family rows of
  `data/backtest_trades_full.parquet` (gha:33852895307, built 2026-09-04)
  26/26 positions, max |R diff| 0.0000, max |PnL diff| $0; the pc_fear-OFF
  pass reproduces the 2026-08-07 shadow 26/26, max |R diff| 0.0006.
- 557 positions 2016-06-01 .. 2026-09-02 (557 rows; the family carries no
  tranche rows, so the collapse changed 0 positions). Fear state: 451 off,
  106 on, 0 stale.
- Dial vintage A: `scratch/ultracode_sizing_2026-09-02/dd_pit/pit_dial_extended.parquet`
  column `pit`, non-null 2018-01-02 .. 2026-09-01 (vintage Y-1 expanding
  weights score year Y; 2026 on weights through 2025-12-31; basis raw 63d ->
  rolling(5) -> rolling(10)). Scores 504 positions; the 53 signals before
  2018-01-29 have no A value and are scored on B only.
- Dial vintage B: `data/rd2_fragility.parquet` 63d -> rolling(10), the live
  sizing series (frozen_through 2026-09-02, last date 2026-09-03), looked up
  the engine's way (`_frag_score_series`: daily grid, ffill limit 5). Scores
  539 positions; 18 unscored (13 in June 2016 before the series starts
  2016-07-05, 5 on 2017 dates where the 63d column is NaN) — the engine
  also sizes those at 1.0x (no score).
- Fear: `pc_fear.fear_state_asof` (imported), lag-1 by data date, from
  `data/cboe_putcall.parquet` (2006-11-01 .. 2026-09-03). The 80/90 grid
  cells re-threshold the module's own `pct`.
- Statistics: cluster = signal date (a date sits in exactly one cell, since
  dial and fear are date-level). "Clustered sigma" for a two-cell
  difference = Welch t on the two cells' per-date mean-R series, the
  prereg's method (`scratch/family_dial_pc_ttest.py`); Mann-Whitney on the
  same series. Gate values are trade-level avgR (the prereg's 2x2 quoted
  trade avgR). A cluster-robust (Liang-Zeger) t on trade-level R is printed
  beside each as a secondary.

## 1. The 2x2 (dial 50 / fear 85), both vintages (`02_review.log`)

| cell | A: n / dates / avgR / win / t_date | B: n / dates / avgR / win / t_date |
|---|---|---|
| dial<50, fear OFF | 281 / 141 / +0.633 / 72% / 5.96 | 347 / 179 / +0.608 / 73% / 5.95 |
| dial>=50, fear OFF | 118 / 64 / +0.006 / 52% / 0.67 | 86 / 51 / -0.166 / 40% / 0.41 |
| dial<50, fear ON | 84 / 42 / +0.695 / 77% / 3.07 | 89 / 47 / +0.727 / 76% / 3.29 |
| dial>=50, fear ON | 21 / 14 / +0.791 / 81% / 1.12 | 17 / 10 / +0.706 / 88% / 0.80 |

Prereg contrast within dial>=50 (fear ON minus OFF): A +0.785, Welch t
+0.86 (p=.404), MW p=.097; B +0.873, Welch t +0.62 (p=.545), MW p=.202.
The prereg reported t=1.21 / MW p=.057 on the recompute dial with 19 vs 70
trades; the fear-ON hi-frag cell is still not separable from the no-fear
cell by a two-sample test on either vintage.

## 2. Gate table

| gate | prereg threshold | vintage A | vintage B | A | B |
|---|---|---|---|---|---|
| 1a no-fear hi-frag deficit, hi minus lo, Welch t on date means | <= -1.5 | -2.82 (diff -0.627R; cluster-robust t -3.22; MW p=.005) | -2.22 (diff -0.774R; cluster-robust t -3.73; MW p=.014) | PASS | PASS |
| 1b fear-ON hi-frag avgR | >= +0.3R | +0.791 (n=21, 14 dates, t_date 1.12, t_cluster 2.04) | +0.706 (n=17, 10 dates, t_date 0.80, t_cluster 1.70) | PASS | PASS |
| 1c grid, cells passing both 1a and 1b | not one knife-edge cell | 8 of 9 (fails fear>90 x dial>=55: 1a t -1.36) | 7 of 9 (fails fear>90 x dial>=50 and >=55: 1b avgR +0.289 on n=13) | PASS | PASS (2 fails, both at the sparsest fear threshold) |
| leg B non-inferiority, fear-ON lo minus no-fear lo | not worse by > 0.1R | +0.062 trade-level; date-mean -0.065; Welch t -0.33 | +0.119 trade-level; date-mean +0.053; Welch t +0.27 | PASS | PASS |
| 3 LOYO on fear-ON hi-frag by year | all remainders > 0 | by year 2018 n=1 -0.04, 2021 n=6 +1.71, 2022 n=3 -0.99, 2026 n=11 +0.85; drop-one min +0.421 (drop 2021) | 2021 n=9 +1.13, 2022 n=3 -0.99, 2026 n=5 +0.96; drop-one min +0.230 (drop 2021) | PASS | PASS |
| 2 new OOS fear-ON hi-frag episodes since 2026-08-05 | >= 2 | 0 (fear has been OFF every session since 2026-08-04; `results.json` live_status) | 0 | not runnable | not runnable |

Grid detail (`results.json` -> vintage_A/B -> grid), 1a Welch t | 1b avgR (n):

| fear thr | dial>=45 (A / B) | dial>=50 (A / B) | dial>=55 (A / B) |
|---|---|---|---|
| >80 | -3.45, +0.831 (27) / -2.27, +0.827 (30) | -2.91, +0.858 (25) / -2.41, +0.803 (21) | -1.94, +1.078 (17) / -1.97, +0.530 (16) |
| >85 | -3.43, +0.765 (23) / -2.11, +0.772 (24) | -2.82, +0.791 (21) / -2.22, +0.706 (17) | -1.78, +1.037 (13) / -1.76, +0.381 (14) |
| >90 | -3.25, +0.648 (16) / -2.15, +0.738 (18) | -2.83, +0.758 (15) / -1.81, +0.289 (13) | -1.36, +0.581 (9) / -1.60, +0.289 (13) |

## 3. What the gate table does not say by itself

- The fear-ON hi-frag cell remains small and episode-carried: 2022's three
  trades are -0.99R on both vintages (one -3.50R SPXS fade on 2022-01-18);
  the cell's sign is carried by 2021 and 2026. LOYO passes because dropping
  2021 still leaves +0.42 (A) / +0.23 (B), but the B remainder without
  2021 is 8 trades.
- The 2026 part of that cell is entirely PRE-ship: March 2026 (5 trades on
  both vintages, dial_B 73-81) and, on vintage A only, 2026-07-23..29 (six
  3x Bear fade legs at dial_A 50.4-56.0 while dial_B was 46.6-49.9, i.e.
  lo_on on the live series, sized 1.0x live). The prereg's motivating 2x2
  ran to 2026-05, so the July-2026 legs are new to the cell but not
  post-ship. Zero fear-ON hi-frag signal dates have occurred since
  2026-08-05.
- The 2018 vintage-A hi_on trade (2018-10-24, -0.04R) exists only because
  vintage-2017 weights put the dial at 51.1 that day; B has it at lo.
- Vintage A scores 118 no-fear hi-frag trades vs B's 86 and its cell is
  +0.006R rather than -0.166R: the extended PIT dial arms more often and
  earlier than the frozen-weight live series (dd_pit README: PIT-vs-live
  >=50 agreement 88.3%). Gate 1a passes on both because the lo cell is
  +0.61..0.63R either way.

## 4. Aug-2026 out-of-sample, leg C (ONE episode; a report line, not a gate)

Every family signal since 2026-08-05 that the fear-OFF table zeroed
(present in the pc_fear-OFF re-run, absent from the production-rule
re-run; `01_shadow_2026.log`, `results.json` -> aug2026). No family signal
fired 2026-07-30 .. 08-04.

| strategy | ticker | signal | exit (type) | dial B / A | fear pct | R | $ at 0.25x | $ at 1.0x |
|---|---|---|---|---|---|---|---|---|
| SPY QQQ MonFri Reversion | QQQ | 2026-08-10 | 08-13 (Time) | 63.7 / 72.2 | 50.8 | +0.866 | +852 | +3,406 |
| Monday Dip | SMH | 2026-08-10 | 08-13 (Time) | 63.7 / 72.2 | 50.8 | +0.671 | +283 | +1,148 |
| SPY QQQ MonFri Reversion | QQQ | 2026-08-17 | 08-20 (Time) | 84.8 / 92.9 | 38.9 | -0.496 | -488 | -1,960 |
| SPY QQQ MonFri Reversion | SPY | 2026-08-28 | 09-01 (Stop) | 87.6 / 96.4 | 39.7 | -1.044 | -1,028 | -4,112 |
| Weak Close Decent Sznls | XBI | 2026-08-28 | 09-02 (Time) | 87.6 / 96.4 | 39.7 | +1.239 | +1,829 | +7,315 |
| 3x Bear ETF Overbot Fade | SOXS | 2026-09-02 | 09-03 (open, marked at the last close) | 87.9 / n/a (A ends 09-01 at 95.6) | 51.6 | +0.974 | +685 | +2,742 |

Totals: n=6, sum +2.209R, avg +0.368R, 4 of 6 winners; +$2,133 at 0.25x,
+$8,539 at 1.0x (flat $750k, `shadow_2026_bands_off.parquet` for the 1.0x
dollars, which keep the gap derate on SMH). The SOXS leg is still open at
the 2026-09-03 bar and its number is provisional. The R is identical at any
multiplier; only the dollars scale. This is one episode of one hi-frag
regime with the sign OPPOSITE to the historical no-fear hi-frag cell
(-0.17R on B over 86 trades); it neither confirms nor refutes leg C and
is recorded for the "+20 hi-frag family trades" re-exam, which now stands
at 86 + 6 = 92 no-fear hi-frag trades on B.

## 5. Live-regime status since 2026-07-30 (`results.json` -> live_status)

26 sessions 2026-07-30 .. 2026-09-03. Live dial (B) arms at 50.3 on
07-30 and peaks 89.5 on 08-21/24, last 87.8 on 09-03. Fear (lag-1 pct252
of the 10d-MA equity P/C) was ON on 07-30 (95.6), 07-31 (91.7), 08-03
(88.9), then OFF from 08-04 (84.9) and never back above 85: 50.8 on 08-10,
low 38.9 on 08-17, 54.4 on 09-03. The family has therefore been zeroed on
23 of the 26 sessions (every session since 2026-08-04). Vintage A reads
95.6 on 09-01. Release needs the live 10d-MA 63d below 50 or fear above
85; neither is near on the last row (87.8 / 54.4).

## 6. Leg summary (which gates each leg passes, per vintage)

- Leg A (dial>=50 & fear ON -> 1.0x): gate 1a PASS/PASS, 1b PASS/PASS
  (+0.79 / +0.71 on 21 / 17 trades; the one-sample date t is 1.12 / 0.80,
  the fear-vs-no-fear contrast is not significant on either vintage), 1c
  8/9 and 7/9, gate 3 LOYO PASS/PASS (min remainder +0.42 / +0.23), gate 2
  NOT RUNNABLE (0 new episodes).
- Leg B (dial<50 & fear ON -> 1.25x): non-inferiority PASS/PASS (+0.06 /
  +0.12 trade-level; date-mean -0.07 / +0.05; all inside 0.1R). Gate 2
  applies to it as well and is not runnable.
- Leg C (dial>=50 & fear OFF -> 0x): gate 1a PASS/PASS (the deficit
  survives PIT on both vintages at -2.8 / -2.2). Aug-2026 shadow: 6 zeroed
  signals, +2.2R, one episode.

Nothing here is a recommendation. The decision set is the prereg's closed
set (STAND or ROLL BACK to the incumbent 0.25x table, per leg).
