# Wide Opportunity Book v0

This is a deterministic, research-only funnel over the union of
`LIQUID_PLUS_COMMODITIES` and `CSV_UNIVERSE` (currently about 1,025 unique
tickers). It allocates research attention; it does not produce trading
recommendations, sizing, staging rows, portfolio changes, emails, uploads or
broker actions.

## Design

1. Read one local `master_prices.parquet`; never fetch from the network.
2. Discard every bar after the requested `asof` before calculating anything.
3. Give every requested ticker a coverage row and a deterministic first
   rejection when it is ineligible.
4. Calculate returns, gaps, ATR/volatility, participation shocks, 52-week and
   moving-average distances, plus market and optional sector residuals.
5. Rank eligible names independently in five research archetypes. There is no
   universal alpha score.
6. Round-robin across archetypes into a bounded review queue, derive a bounded
   deep-test queue, and draw a seeded random coverage-audit sample.
7. Emit JSON, CSVs and a self-contained local HTML report.

The command-line entry point refuses to write outside this worktree's ignored
`artifacts/` directory. The library stays pure until its explicit writer is
called, which keeps unit tests and downstream research composition possible.

Each promoted research card carries research actionability, a potential
variant wedge, why-now evidence, a first rejection test, what makes the setup
researchable, a kill condition, and the next research workflow. These are
hypothesis-design fields, not recommendations.

[`CODEX_HANDOFF.md`](CODEX_HANDOFF.md) defines the bounded second stage: Codex
may perform source diligence on no more than three of the ten deep-test names,
write local artifacts, or reject the slate. It deliberately never invokes the
production Daily Pitch publisher or an order-capable surface.

Every rank input, archetype score, rank, percentile, exclusion and freshness
field is retained in the output bundle.

Features degrade independently when inputs are unavailable: close history can
remain eligible when OHLC or volume is absent, while the dependent fields are
reported in `Missing_Features`. A 52-week high/low is populated only after 252
bars; it is never relabeled from a shorter history.

## Local use

```powershell
python scripts/build_wide_opportunity_book.py `
  --output-dir artifacts/opportunity_book/2026-08-27
```

Without `--asof`, the runner uses the latest local SPY bar—not the wall-clock
date—as the inclusive cutoff. This prevents a pre-close run from mixing a few
new-calendar-date instruments into an otherwise prior-session US equity book.
Pass `--asof` only when deliberately replaying a frozen historical date.

Useful development overrides:

```powershell
python scripts/build_wide_opportunity_book.py `
  --prices artifacts/fixtures/prices.parquet `
  --tickers "SPY,AAPL,MSFT,NVDA" `
  --sector-map artifacts/fixtures/sectors.json `
  --asof 2026-08-27 `
  --output-dir artifacts/opportunity_book/dev
```

The optional sector map may be JSON (`{"AAPL": "XLK"}`) or CSV with
`ticker` and `sector`, `sector_ticker`, or `benchmark` columns. When the value
names an available price series it is used as a rolling benchmark; otherwise
it remains a category label and the output includes same-date cross-sectional
sector residuals.
