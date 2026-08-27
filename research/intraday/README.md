# Intraday Research Lab v0

This package is an isolated research surface. It reads explicitly supplied
local 15-minute parquets or DataFrames and writes research artifacts only. It
does not import the strategy book, scan/staging code, broker code, Sheets,
workflows, cache credentials, or R2 helpers. A result cannot promote itself.

## Input contract

Each ticker has regular-session bars with columns:

```text
ts, open, high, low, close, volume
```

`ts` is the 15-minute bar's **start** timestamp in New York time. Existing
naive ET parquets are accepted. A bar labelled `13:00` is not fully known
until `13:15`. Inputs outside 09:30--15:45, off the 15-minute grid, duplicated,
or with impossible OHLC values fail loudly. Missing bars are retained as a
data-quality fact rather than silently manufactured.

## Point-in-time eligibility

The execution-grade gate uses only quote-free proxies available before the
session being classified:

- prior-session close >= $5;
- trailing 20-session median of `sum(close * volume)` >= $25 million;
- trailing bar completeness >= 95%, relative to 26 expected bars;
- at least 10 completed historical sessions.

All rolling inputs are shifted one full session. Current-day eventual volume
or completeness therefore cannot make the current day eligible. These gates
are coarse research filters, not proof of spread, depth, borrow, or capacity.

## Pre-registered v0 templates

1. `gap_first_hour_residual_continuation_v0`
   - residual = asset return minus a 50/50 SPY and sector-proxy return;
   - if sector proxy is SPY, SPY is used once;
   - residual overnight gap must be at least 1.0% in magnitude;
   - the four completed bars from 09:30 through 10:30 ET must produce an
     aligned residual response of at least 0.5% (the last consumed bar is
     labelled 10:15 and closes at 10:30);
   - decide at 10:30 and enter the 10:45 bar open, in the gap direction;
   - exit the scheduled 15:45 bar close (16:00).

2. `intraday_residual_shock_reversal_v0`
   - measure the cumulative session-open move through the bar labelled 13:00,
     which closes and becomes available at 13:15;
   - residual shock must be at least 1.5% in magnitude;
   - decide at 13:15 and enter the 13:30 bar open, opposite the residual shock;
   - exit under the same close policy.

The thresholds and directions are hypotheses chosen before looking at results,
not claims of edge. Every output includes day, ticker, sector, sector proxy,
decision/feature/entry/exit clocks, signal features, eligibility proxies, and
day/ticker/sector cluster keys.

## Optional capital-reuse feasibility

No account treatment is inferred from account size. The optional feasibility
layer runs only when the caller supplies a per-trade notional and at least one
of: starting settled cash, intraday buying power, or maximum concurrent
notional. Same-day reuse defaults to **off** and can be enabled only through an
explicit parameter. The audit preserves both accepted and rejected trades,
with rejection reasons such as `settled_cash`, `intraday_buying_power`, and
`max_concurrent_notional`.

Trades sharing an entry timestamp need a deterministic capacity priority. The
CLI defaults to descending `signal_strength`; `--capital-priority input_order`
is available only when a simultaneous batch is not oversubscribed. Otherwise
input-order mode fails loudly rather than letting ticker order allocate scarce
capital. The audit exposes the final `capital_sequence`, so this choice remains
testable instead of hidden.

This is deliberately just per-session arithmetic. It does not determine PDT
status, interpret a broker's margin rules, model a T+1 settlement calendar, or
assert that the account is permitted to reuse proceeds. Confirmed account
terms must be supplied from outside the research engine. Reuse releases the
requested notional, not realized proceeds or P&L.

## Local CLI

```powershell
python scripts/run_intraday_research.py `
  --data-dir data/intraday `
  --sector-map data/sector_map.parquet `
  --tickers AAPL MSFT NVDA `
  --round-trip-cost-bps 8 `
  --per-trade-notional 5000 `
  --starting-settled-cash 20000 `
  --max-concurrent-notional 15000
```

When `--tickers` is present, the runner also loads SPY and the required sector
proxies. With no explicit output path, timestamped files go beneath
`artifacts/intraday_research/`. The runner never downloads missing data;
missing parquets abort the run. Outputs are `eligibility.parquet`,
`signals.parquet`, `trades.parquet`, `execution_rejections.parquet`,
`summary.csv`, and `run_manifest.json`.
When capital constraints are supplied, it additionally writes the full capital
audit, feasible trades, rejections, and a feasible-trade summary.

## Deliberate limitations

- A 15-minute bar cannot reveal within-bar sequence, spread, queue position,
  partial fills, opening/closing auction behavior, or whether the displayed
  open was realistically obtainable.
- The cost input is a single round-trip bps stress, not a quote-aware model.
- There is no news/earnings/halts/corporate-action filter yet.
- There is no short-locate or borrow model. Short candidates are research
  observations only.
- Sector residuals use fixed ETF proxies, not fitted point-in-time betas.
- The sector map itself is today's classification, not point-in-time history.
- Simultaneous trades can be clustered after the run, but v0 does not size a
  portfolio, recycle capital, or cap factor/sector exposure.
- Missing scheduled entry or close bars are rejected from return-producing
  trades and preserved in `execution_rejections.parquet`. There is no backward
  fallback. Known exchange early closes need an ex-ante calendar schedule,
  which v0 does not yet implement.
- Canonical sessions are the union of dates observed across the supplied
  frames. This exposes a wholly missing proxy session when another supplied
  ticker has that date, but it cannot detect a date missing from every input;
  an exchange calendar is still required for that case.
- Results are not recommendations and have no path to production execution.

The next validity step is to run unchanged v0 definitions over a frozen local
sample, inspect coverage and data failures first, then evaluate day-clustered
and ticker/sector-held-out performance under materially stressed costs.
