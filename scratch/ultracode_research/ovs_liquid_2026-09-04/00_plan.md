# study_ovs_liquid — recon and method plan (2026-09-04)

Brief: `docs/briefs/2026-09-04/study_ovs_liquid.md`. Pre-registration is frozen; this file records
what the ledger actually carries and how each pre-registered quantity is computed. Written before
the study script ran.

## Ledger recon (`data/backtest_trades_full.parquet`, vintage `ledger_build_utc=2026-09-04T08:24:57Z`, `ledger_source=gha:33852895307`, 4701 rows)

- OVS rows: `Strategy == 'Overbot Vol Spike'`, 2487 rows, signal dates 2003-04-22 .. 2026-07-28.
- **Tier**: ledger column `Tier` in {'Liquid', 'Overflow'}. Verified: every Liquid-tier OVS ticker is
  in `strategy_config.LIQUID_PLUS_COMMODITIES` and no Overflow-tier ticker is (both set differences
  empty), so the ledger column and the brief's definition agree. I use the ledger column.
- **Tranche**: column `Tranche` in {'near', 'far', ''}. 1215 near/far pairs + 57 single rows
  (Tranche '' = EOD-DD days, one row per position). Grouping by (Ticker, Signal Date, Tier) yields
  exactly 2 rows (near+far) or 1 row (''), never more, so that key identifies a position.
  Position R = sum(PnL_flat_750k) / sum(Risk_flat_750k) over the group (brief definition).
- **Path (P1/P2)**: NOT a ledger column. Inferred exactly the way the engine and order_staging
  decide it: gap = (`T+1 Open` - `Signal Close`) / `ATR`; P1 if gap > 0.25, else P2 (gap <= 0 never
  appears in the ledger: min 2e-6, i.e. skips are already excluded). Cross-check: `Size_Mult` is
  1.0 or 0.75 (cycle mult) for P1 and 0.2 / 0.15 / smaller (P2 cap pro-rata) for P2; the script
  reports the agreement rate between the gap rule and the Size_Mult rule.
- **Exit type**: column `Exit Type` in {'Target', 'Time', 'EOD-DD'} per tranche row. Position-level
  label: 'EOD-DD' if any row is EOD-DD; 'Target' if all rows Target; 'Time' if all rows Time;
  'Mixed' (near Target, far Time) otherwise. Cut (viii) reports the tranche-row mix AND the
  position-level mix; "time-exit avgR" is reported for positions labelled Time (every tranche timed
  out) and, separately, for far-tranche rows that exited on Time.
- **Ranks**: NOT in the ledger (only `Range %`). Computed from `data/master_prices.parquet`
  (adjusted OHLCV, mtime 2026-09-04 05:10) with the `indicators.py` definition, verbatim:
  `ret_{w}d = Close.pct_change(w, fill_method=None)`;
  `rank_ret_{w}d = ret_{w}d.expanding(min_periods=252).rank(pct=True) * 100`, w in {2,5,10,21}.
  Joined on (ticker, Signal Date). Sanity check reported: share of liquid OVS signals whose four
  computed ranks are all > 85 (the strategy's own perf filter), which should be ~100% if the
  recomputation matches the ledger build's basis. Extremity = mean(rank_2d, rank_5d, rank_10d,
  rank_21d); bottom cell < 94, top cell >= 94 (brief (iv)).
- **Sector**: `data/sector_map.parquet` (ticker -> sector). BNY and DOV are missing and are mapped
  by hand (BNY Financial Services, DOV Industrials); ETFs/indices absent from the map are labelled
  'ETF/Index'. Semis theme set (declared here, before running):
  {MU, INTC, AMD, NVDA, AMAT, ADI, TXN, QCOM, AVGO, SMH}. Mega-cap tech set:
  {AAPL, MSFT, GOOG, AMZN, META, NVDA, AVGO, XLK, QQQ, ^NDX}. Union = "semis / mega-cap tech".
- **Control strategies**: overflow OVS = same OVS rows with Tier == 'Overflow'; short-side control =
  `Strategy == '3x ETF Overbot Fade'` (87 rows, all Liquid, single-row positions, from 2011).

## Eras
- Base: signal date 2010-01-01 .. 2023-12-31. Recent: signal date >= 2024-01-01.
- 2003-2009 OVS rows are excluded from every statistic (outside the registered window).

## Statistics
- Primary: diff = avgR(recent) - avgR(base). t clustered by signal date: OLS of R on an era dummy,
  cluster-robust (CR1, small-sample factor G/(G-1) * (N-1)/(N-2)) covariance with clusters =
  signal dates, implemented in the script (no statsmodels dependency; statsmodels 0.13.5 is present
  and is used only as an independent check that the manual CR1 t agrees).
- Monthly block bootstrap: positions are grouped by signal month (YYYY-MM). Each draw resamples,
  within each era separately, the era's signal-months with replacement (same number of months as
  observed), concatenates the positions of the drawn months and recomputes avgR; diff = recent -
  base. 10,000 draws, seed 20260904, percentile 2.5 / 97.5 interval.
- Every cut recomputes the same primary (diff, clustered t) on the stated subset; per-year table
  uses unclustered means.
- `explained_by_extremity` (for checks.json) := top-cell clustered t > -1.5 AND bottom-cell
  clustered t <= -1.5 (the deficit lives only in the < 94 cell). `decision_inputs_all_hold` :=
  t_clustered <= -2.0 AND cut_i_t <= -1.5 AND cut_iv_top_cell_t <= -1.5 AND {2024, 2025, 2026}
  all individually below the base mean.

## Outputs
- `study_ovs_liquid.py` (the script), `checks.json`, `results.md` (every table), `positions.csv`
  (the collapsed position table with path, ranks, extremity, sector, era) so every number is
  re-derivable.
