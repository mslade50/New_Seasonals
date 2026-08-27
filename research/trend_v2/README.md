# Trend V2 research harness

This directory is an isolated, research-only laboratory. It does not import or
modify `trend_sleeve.py`, does not read or write its state, and has no Sheets,
broker, R2, deployment, or network path. The current production sleeve remains
the frozen comparator.

## What it runs

1. **Frozen production core benchmark (not a candidate trial).** The 12 ETFs, 12-1
   momentum above zero, price above the 10-month moving average, inverse-63-day
   volatility slots, 20% asset cap, long/flat. This is a copied research
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
   maps are rejected.

## Point-in-time and execution semantics

- Signals at month-end `t` use closes and histories available through `t`.
- Rolling market betas used for day `t` residuals are shifted and therefore use
  data only through `t-1`.
- Dated sector snapshots are forward-filled but never backfilled. Effective
  intervals are applied only inside their stated dates.
- Targets formed at month-end `t` are shifted into the next holding period.
  When adjusted Open prices exist, results are next-open to next-open. A
  close-only file uses next-close to next-close, still with the prior target.
- Missing returns for a held security invalidate that month rather than being
  silently treated as zero.
- Adjusted-price provenance is a caller responsibility; the loader cannot infer
  adjustment status from numeric values.

These calculations remove signal and execution lookahead. They do not cure
constituent survivorship. A stock study must encode historical membership with
missing prices or an upstream point-in-time universe before its results can be
considered reliable.

## Local input formats

The price parquet can be:

- long: case-insensitive `date`, `ticker`, `Close`, and optional `Open`; or
- wide close-only: DatetimeIndex and one ticker per column.

Sector history can be:

- dated long snapshots: `date`, `ticker`, `sector`;
- effective intervals: `ticker`, `sector`, `effective_from`, optional
  `effective_to`; or
- a wide dated panel with tickers as columns.

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
- `manifest.json`: input provenance, execution semantics, and explicit trial
  and parameter counts;
- `monthly_returns/`: auditable return/cost series;
- `targets/`: auditable target-weight panels; and
- `support_note.md`: a brief research-only first read.

Read [PREREGISTRATION.md](PREREGISTRATION.md) before interpreting or extending
the grid. Any extra threshold, universe, cost, start-date, or unpublished rerun
is an additional trial.
