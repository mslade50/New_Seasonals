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

## Streaming real-data runner

`scripts/run_intraday_streaming_research.py` is the full-history evaluation
path. It requires an explicit local data directory, sector map, and frozen
candidate universe. It loads SPY and each required sector proxy once, reduces
those proxies to daily event features, then loads, evaluates, and releases one
candidate parquet at a time. It never imports a downloader, R2 client, broker,
strategy scanner, or production writer.

```powershell
python scripts/run_intraday_streaming_research.py `
  --data-dir artifacts/intraday_input/frozen_2026-08-26 `
  --sector-map artifacts/intraday_input/sector_map.parquet `
  --universe-file artifacts/intraday_input/liquid_single_stocks.csv `
  --output-dir artifacts/intraday_streaming_research/pilot_2026-08-26
```

The output path must be a fresh directory under this worktree's ignored
`artifacts/` root. The manifest is written last and declares the run
research-only, order-free, production-write-free, and ineligible for automatic
promotion. There is intentionally no investor-facing HTML output.

The event clocks, thresholds, directions, SPY/sector residual arithmetic, and
prior-session eligibility rules remain the locked v0 definitions. The
real-data path layers on stricter validity gates:

- expected sessions come from the repository's NYSE `TRADING_DAY` calendar,
  exposing dates absent from every loaded input;
- an exact observed 09:30--12:45 14-bar SPY tape is labeled and excluded as an
  observed early close; other incomplete SPY sessions remain separately
  labeled rather than becoming opaque missing-close trades;
- the gap feature requires all four 09:30, 09:45, 10:00, and 10:15 bars, and
  the shock feature requires every 15-minute bar from 09:30 through 13:00;
- all candidate and proxy feature-window bars must have positive volume;
- scheduled candidate entry and exit bars must exist and have positive volume;
- sector metadata and the mapped sector-proxy parquet are mandatory per
  candidate. Affected candidates are explicitly excluded; there is no silent
  SPY fallback;
- raw overnight ratios near common split factors (1:2, 1:3, 1:4, 1:5, 1:10
  and their reverse-split counterparts, within 12%) are flagged and filtered
  from the gap template. This is a conservative discontinuity heuristic, not
  proof of a corporate action.

The locked primary result uses 10 bps round-trip cost. The default stress grid
is 5/10/15/20/30 bps. Outputs include full coverage and calendar audits,
signal-generation and execution rejection audits, event-level trades, raw
day-cluster returns, two-sided day-cluster t/p values, deterministic
day-cluster bootstrap confidence intervals, and Holm adjustment across the two
10-bps primary template tests. Annual and rolling five-calendar-year
train/one-year test tables diagnose stability without refitting or selecting a
rule. Ticker and sector summaries expose concentration.

Top-strength K=1/3/5/10 overlays are explicitly **slot-based**: each day takes
the K strongest pre-existing signals, splits notional equally across K slots,
and leaves unused slots in cash. They do not model integer shares, per-share
commissions, account settlement, buying power, short permissions, spreads, or
broker capacity.

## Deliberate limitations

- A 15-minute bar cannot reveal within-bar sequence, spread, queue position,
  partial fills, opening/closing auction behavior, or whether the displayed
  open was realistically obtainable.
- The v0 CLI cost input is a single round-trip bps stress. The streaming path
  provides a fixed grid, but neither path is a quote-aware cost model.
- There is no news, earnings, halt, or authoritative corporate-action source.
  The streaming split-factor heuristic is deliberately only a raw-price
  discontinuity filter.
- There is no short-locate or borrow model. Short candidates are research
  observations only.
- Sector residuals use fixed ETF proxies, not fitted point-in-time betas.
- The sector map itself is today's classification, not point-in-time history.
- Simultaneous trades can be clustered after the run. The optional arithmetic
  gate tests explicitly supplied notional ceilings and reuse assumptions, but
  v0 does not optimize portfolio sizing or cap factor/sector exposure.
- The streaming K overlays model scarce signal slots, not a brokerage account.
  Neither path provides integer-share sizing, per-share commissions, or a
  spread/participation capacity model.
- Missing or zero-volume scheduled entry/close bars are rejected from
  return-producing trades and preserved in `execution_rejections.parquet`.
  There is no backward fallback. The streaming path labels exact observed SPY
  early-close tapes, but it still lacks an authoritative ex-ante early-close
  schedule.
- Canonical sessions are the union of dates observed across the supplied
  frames. This exposes a wholly missing proxy session when another supplied
  ticker has that date, but it cannot detect a date missing from every input;
  an exchange calendar is still required for that case.
- Current-universe and current-sector-map tests have survivorship and
  classification lookahead. Results cannot be described as point-in-time
  universe evidence without historical membership and classification inputs.
- Results are not recommendations and have no path to production execution.

The next validity step is to run unchanged v0 definitions over a frozen local
sample, inspect coverage and data failures first, then evaluate day-clustered
and ticker/sector-held-out performance under materially stressed costs.
