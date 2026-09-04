# Recon plan: P/C-fear post-ship review, part 1 (2026-09-04)

Brief: `docs/briefs/2026-09-04/study_pcfear_review.md`. Prereg read in full:
`scratch/ultracode_research/family_pc_fear_band_prereg_2026-08-05.md` (rev 3).
Everything written here lives under this folder. No config, test, parquet or
ledger under `data/` is touched.

## Inputs found (all read-only)

| input | path | what I found (`00_inspect.py`) |
|---|---|---|
| Vintage A dial | `scratch/ultracode_sizing_2026-09-02/dd_pit/pit_dial_extended.parquet`, column `pit` | 2016-07-05 .. 2026-09-01; `pit` non-null from 2018-01-02 (vintage-2017 weights score 2018; expanding-window edges, vintage Y-1 scores year Y; 2026 on weights fit through 2025-12-31). Basis = raw 63d -> rolling(5) -> rolling(10), i.e. already the 10d-MA live basis. README: `pit_hedge_dd.md`, generator `02_pit_dial_extended.py` |
| Vintage B dial | `data/rd2_fragility.parquet` column `63d` | 2016-07-05 .. 2026-09-03; append-only PIT since 2026-07-02, recompute vintage before. Scored exactly as the engine does: `63d.dropna().rolling(10, min_periods=1).mean()`, daily grid, `ffill(limit=5)` (`pages/strat_backtester._frag_score_series`) |
| Fear state | `pc_fear.fear_state_asof(signal_date)` from `data/cboe_putcall.parquet` (2006-11-01 .. 2026-09-03) | imported, not re-implemented; lag-1 by data date, stale > 3 bd. The 80/90 grid cells re-threshold the module's own `pct` |
| Population | `data/backtest_trades_pcfear_shadow.parquet` (local build 2026-08-07, sha 34afae8, 1163 family rows, last signal 2026-07-29) | the pc_fear-DISABLED pass = incumbent 0.25x table everywhere, so it holds every family trade INCLUDING the ones the live table zeroes. The main ledger (`data/backtest_trades_full.parquet`, gha:33852895307, built 2026-09-04) has only 1083 family rows because the fear-OFF/dial>=50 rows are dropped at the shares floor; it cannot populate the no-fear hi-frag cell. Both are checked against each other on (Strategy, Ticker, Signal Date) |
| Post-07-29 window | my own family-only engine re-run (`01_shadow_2026.py`) | the local shadow ends 2026-07-29 and no R2 copy exists (`build_trade_ledger.py` uploads only the main ledger key; grep finds no other writer). The brief allows a bar-based reconstruction when the shadow is unavailable; the reconstruction IS the engine (`process_signals_fast`, family strategies only, candidates from 2026-01-01) run three ways: pc_fear ON (parity with the main ledger), pc_fear OFF (= shadow, 0.25x), bands stripped (1.0x). Written to this folder only |

Family = the six `pc_fear_bands` carriers: Weak Close Decent Sznls, SPY QQQ
MonFri Reversion, Monday Dip, Indices Oversold Bounce, 3x Bear ETF Overbot
Fade, Monthly Weak Close. None is overflow-eligible; combined universe 48
tickers, so the re-run is cheap. Family rows carry `Tranche == ''`, but
the collapse to positions (sum PnL / sum risk per Strategy-Ticker-Signal
Date-Entry Date) is applied anyway as the brief asks.

## Headless import

`pages/strat_backtester.py` is imported the way `scripts/build_trade_ledger.py`
imports it (streamlit is installed; its decorators are inert outside a
Streamlit run). `fragility_core` is not needed here, so the `_NoOp`
streamlit stub in `dd_pit/02_pit_dial_extended.py` is not required; it is
kept as the fallback pattern if the import fails.

## Statistics (fixed before running)

- Position R = sum PnL_flat / sum Risk_flat. Trade avgR = mean of position R;
  win = share of R > 0.
- Cluster = signal date. "Date-clustered" statistics use the per-date mean R
  series (the prereg's own method in `scratch/family_dial_pc_ttest.py`):
  - one-sample t of a cell (gate 1b): t = mean(date means) / SE(date means).
    The gate VALUE is the trade-level avgR (the prereg's 2x2 quoted trade
    avgR, +0.75R); the date-mean avgR and its t are printed beside it.
  - two-cell difference (gate 1a, leg B): Welch t between the two cells'
    date-mean series, sign = first cell minus second. Mann-Whitney on the
    same two date-mean series.
  - secondary, printed only: cluster-robust (Liang-Zeger, clusters = signal
    date) SE on trade-level R, so the trade-weighted version is visible.
- Gate 1a: (no-fear, dial>=50) minus (no-fear, dial<50), Welch t <= -1.5.
- Gate 1b: (fear ON, dial>=50) trade avgR >= +0.3.
- Gate 1c: fear pct threshold {80,85,90} x dial threshold {45,50,55}; a cell
  passes when both 1a and 1b pass in it. `grid_cells_passing_both` is the
  vintage-A count.
- Leg B: (fear ON, dial<50) minus (no-fear, dial<50), Welch on date means;
  must be >= -0.1R.
- Gate 3 LOYO: fear-ON dial>=50 cell, drop each signal-year in turn; all
  remaining averages must stay > 0.
- Vintage A scores only trades with a `pit` value at the signal date
  (>= 2018-01-02, ffill limit 5); vintage B scores the full 2016-06+ set.
  Each trade row in `trades_scored.csv` states both scores and which vintage
  bucketed it.
- Aug-2026 (item 7, report line only): signals >= 2026-08-05 present in the
  pc_fear-OFF re-run and absent from the pc_fear-ON re-run, each with live
  dial, fear pct, R, PnL at 0.25x (from the OFF pass) and at 1.0x (from the
  bands-stripped pass, matched on Strategy/Ticker/Signal Date). Signals
  2026-07-30 .. 08-04 are listed separately (the live book was still on the
  0.25x table then; the ledger replays the new rule over them).
- Item 8: every `rd2_fragility` session from 2026-07-30: live dial (B), PIT
  dial (A), fear pct + state via `fear_state_asof(session)`. `family_off`
  = sessions where B >= 50 and state == off.

## Outputs

`01_shadow_2026.py` -> `shadow_2026_{pcfear_on,pcfear_off,bands_off}.parquet`,
`01_shadow_2026.log`. `02_review.py` -> `trades_scored.csv`,
`results.json`, `checks.json`, `02_review.log`. `REVIEW.md` hand-written
from those files (every number cites the file it came from).
