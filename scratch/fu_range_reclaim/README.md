# Rising-200-day range undercut/reclaim, revision 2

User correction, 2026-09-22: stocks must be above a rising 200-day average, with a defined range and a clear support line before the undercut. This replaces the first version's single post-breakout pivot anchor with repeatedly tested horizontal support. A preceding breakout is not required in this range-based revision.

Research only. Existing checkout/branch; unique source paths. No production changes, new branches, worktrees, uploads or orders. The original FU worktree and results remain intact.

## Specification frozen before reading v2 returns

- Daily bars, 200-day simple moving average. Previous and signal closes must be above it. The prior session's SMA200 must exceed its value 20 sessions earlier, and the signal day's SMA200 must exceed the prior day's. The support floor must also exceed the prior SMA200.
- Fixed 40-session range immediately before signal. Determine floor (minimum low) and ceiling (maximum high) from the first 35 sessions; both are therefore known at least five completed sessions before signal. Reference ATR is the 14-session simple true-range average at that formation cutoff.
- At least three strict confirmed 2-left/2-right pivot lows within 0.35 reference ATR of the floor, separated by at least five sessions and spanning at least 15 sessions. Price must rebound at least one reference ATR between successive support tests.
- At least two confirmed pivot highs within 0.5 ATR of the ceiling, separated by at least five sessions. Confirmation must occur by the formation cutoff.
- Range width is 2–8 ATR and at most 15% of support. Prior 40-session close-regression slope implies no more than 1 ATR of absolute drift from beginning to end. Prior closes remain within floor/ceiling; the final five sessions cannot breach either boundary (wicks included). The formation's extrema themselves define its boundaries.
- Signal: low is strictly below support by at most 0.5 ATR, close is strictly above support and below/equal ceiling, and close is in the upper half of its candle. No future confirmation bars. Entry next open.
- Default 20-session signal cooldown per ticker, so one support event is not repeatedly counted. It is causal and independent of future exits.

## Review first

`detect` writes signals, coverage, quality diagnostics, a manifest, and a self-contained chart gallery. Latest 15 signals are selected by date, not outcome. Each chart initially hides bars after the signal. Support tests and the SMA200 are marked. `evaluate` is a separate invocation, allowing visual inspection before reading returns. No parameter search or outcome-driven chart selection.

## Execution/evaluation

Reuse the first research version's next-open mechanics in a small local module: fixed 5/10/20-session exits (entry is day 1), plus a separate 10-session stop-only policy at signal low minus 0.1 ATR. Skip stop-policy entries opening at/below the stop; exit later gaps at the open. 10 bps round-trip cost, 0/25 bps sensitivity. No R-based sizing. No portfolio CAGR or Sharpe.

Both traded name and same-date SPY pay the same assumed cost. Monthly block-bootstrap confidence intervals, 2,000 resamples, seed 9202026. Adjacent-month dependence remains. Periods: 2000–2017, 2018–2022, 2023+. These are retrospective slices, not untouched holdout data. Current/retained constituents create survivorship and selection bias; source prices may contain corporate-action defects. Strong returns alone would not validate this idea.

Research input snapshot uses only requested OHLC columns, is hashed, and is saved under artifacts. Source stat is checked before/after loading to reject a changing cache. Duplicate sessions fail; full tickers with invalid OHLC or weekend records are excluded. Large overnight gaps (absolute 40%+) are flagged for review, not silently removed. No shared cache changes.

## Commands

```powershell
python scratch/fu_range_reclaim/research.py detect --prices data/master_prices.parquet --out artifacts/fu_range_reclaim/run_v2b
python -m pytest tests/test_fu_range_reclaim.py -q -p no:cacheprovider
python scratch/fu_range_reclaim/research.py evaluate --out artifacts/fu_range_reclaim/run_v2b
```

Existing output files are preserved: detection refuses an existing run directory and evaluation refuses an existing trades.csv. Inputs, rules, code hashes and chart selection are recorded in manifest.json.

Visual-QA revision: the initial `run_v2` used only the 20-session SMA comparison. The POST example showed that this can admit a currently falling SMA. `run_v2b` adds the current-day rise check; its regression test failed before the fix and passed afterward. The original 31-signal run is preserved. No range or execution settings were selected from returns.
