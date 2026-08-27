# Trend V2 research harness

This directory is an isolated, research-only laboratory. It does not import or
modify `trend_sleeve.py`, does not read or write its state, and has no Sheets,
broker, R2, deployment, or network path. The current production sleeve remains
the frozen comparator.

## What it runs

1. **Frozen production core benchmark (not a candidate trial).** The 12 ETFs, 12-1
   momentum above zero, price above the 10-month moving average, inverse-63-day
   volatility slots, 20% asset cap, production 1% rebalance band, long/flat.
   This is a copied research
   specification frozen on 2026-08-27; production code is never changed. The
   book-level month-entry fragility gate is deliberately held outside this
   price-only signal comparison and must be replayed in the later portfolio
   integration gate.
2. **Multi-speed time-series ETF family.** Four preregistered global trials use
   3/6/12-month return-sign or channel-breakout votes. Entry/exit hysteresis,
   inverse-volatility allocation, a 10% ex-ante portfolio-volatility ceiling,
   20% asset and 100% gross caps, a 1% no-trade band, and a 50% soft monthly
   turnover cap are applied identically to every ETF. There is no ticker-level
   optimization.
3. **Stock residual-trend family.** A separate, optional research family—not an
   extension or replacement of the ETF sleeve. It removes market beta using
   rolling betas known before each return, ranks 126-day residual momentum only
   against contemporaneous sector peers, and gives active sectors equal risk
   budgets. Stock mode requires dated sector history; undated current-sector
   maps are rejected. Sector budgets use cap-aware water filling: if one active
   sector cannot deploy its nominal budget under the 5% name cap, every active
   sector is reduced to that same feasible budget rather than allowing broader
   sectors to dominate.

## Point-in-time and execution semantics

- Signals at month-end `t` use closes and histories available through `t`.
- Month-end features require the exact final NYSE-session close. Missing daily
  closes in the fixed ETF panel fail the run; an earlier bar is never silently
  substituted or forward-filled.
- Rolling market betas used for day `t` residuals are shifted and therefore use
  data only through `t-1`.
- Dated sector snapshots are forward-filled but never backfilled. Effective
  intervals are applied only inside their stated dates.
- Builders emit **desired** targets only. At each next-period boundary, the
  simulator first drifts prior executed weights through realized returns, then
  applies the no-trade band and soft turnover cap against those drifted weights.
  Hard name/gross-cap repairs always override the soft turnover budget. Costs
  are charged on the actual executed delta, not `target.diff()`.
- Targets formed at month-end `t` execute at the exact first NYSE session of
  `t+1`. Open-to-open returns use the exact first sessions; a later available
  bar is never substituted. Close-only exploration uses exact NYSE month-end
  closes and the same prior-target shift.
- A missing exact boundary price for a held security, an internal missing
  holding return, or a gap in target months raises an error. It is never dropped
  or time-compressed. Only a trailing period whose next boundary is genuinely
  beyond source coverage is omitted as incomplete.
- CAGR uses elapsed calendar time between represented monthly periods rather
  than assuming `N/12` years.
- ETF evaluation starts on one common next-period clock. Months before a
  strategy's first active signal remain in cash rather than being discarded,
  so delayed activation retains its opportunity cost. The support note does
  not select an in-sample winner and never ranks the separate stock family
  against the ETF family.
- Cash earns the prior exact-month-end `^IRX` annual yield divided by 12 when
  that series is present; absent months are explicitly modeled as zero and the
  assumption is recorded in the manifest.
- Adjusted-price provenance is a caller responsibility; the loader cannot infer
  adjustment status from numeric values.

These calculations remove signal and execution lookahead. They do not cure
constituent survivorship. When supplied, dated membership—not the sector map—
defines the exact stock universe, and missing price history fails loudly. The
artifact writer re-materializes sector and membership panels from the hashed
source files and requires them to match the audited panels before the PIT gate
can pass. Without membership, stock mode may run for engineering diagnosis,
but the manifest marks the gate failed and the results must not be described as
PIT-ready.

## Local input formats

The price parquet can be:

- long: case-insensitive `date`, `ticker`, `Close`, and optional `Open`; or
- wide close-only: DatetimeIndex and one ticker per column.

Sector history can be:

- dated long snapshots: `date`, `ticker`, `sector`;
- effective intervals: `ticker`, `sector`, `effective_from`, optional
  `effective_to`; or
- a wide dated panel with tickers as columns.

Historical membership can be:

- dated long booleans: `date`, `ticker`, and one of `in_universe`, `member`, or
  `eligible`;
- effective intervals: `ticker`, `effective_from`, optional `effective_to`; or
- a wide dated boolean panel.

No downloader exists in this package.

## Run

```powershell
python scripts/run_trend_v2_research.py `
  --prices data/master_prices.parquet `
  --output-dir artifacts/trend_v2/2026-08-27_etf_only
```

Optional stock mode:

```powershell
python scripts/run_trend_v2_research.py `
  --prices artifacts/research_inputs/adjusted_prices.parquet `
  --sector-history artifacts/research_inputs/sector_history.parquet `
  --membership-history artifacts/research_inputs/membership_history.parquet `
  --market-ticker SPY `
  --output-dir artifacts/trend_v2/2026-08-27_with_stocks
```

The output directory is required, must live under `artifacts/`, and must be
empty. The runner refuses source, production-data, site, and deployment paths.
The command-line entry point further confines output to this worktree's own
ignored `artifacts/` root; it cannot write beside production modules or data.
It writes:

- `summary.csv`: gross/net performance, turnover, drawdown, costs, and frozen
  benchmark correlation;
- `trial_details.json`: complete specifications and metrics;
- `manifest.json`: input provenance, execution semantics, exact sanitized
  artifact slugs, explicit trial/parameter counts, and the stock PIT gate;
- `monthly_returns/`: auditable return/cost series;
- `desired_targets/`, `pretrade_weights/`, and `executed_weights/`: separate
  intent, drift, and actual execution panels;
- `stock_pit_audit/` when stock mode runs: source hashes, exact universe,
  materialized sector/membership panels, scores, ranks, and missing-sector
  coverage; and
- `support_note.md`: a brief research-only first read.

Read [PREREGISTRATION.md](PREREGISTRATION.md) before interpreting or extending
the grid. Any extra threshold, universe, cost, start-date, or unpublished rerun
is an additional trial.
