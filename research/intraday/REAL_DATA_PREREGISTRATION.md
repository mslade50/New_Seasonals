# Intraday v0 Real-Data Preregistration

## Status and purpose

This document freezes the first real-data evaluation of the two intraday v0
templates before any full-sample returns are computed. The run is a
research-priority screen, not a trading recommendation or production gate.
The definitions remain unchanged unless a separately named v1 experiment is
preregistered against a new holdout.

## Frozen input

- Read-only R2 snapshot: `intraday-r2-snapshot.v1`
- Snapshot date: 2026-08-27
- R2 index SHA-256:
  `7b64e8f3a024a20c344a1aca08893a97e506deff191638c5032e1f216b399b27`
- Files: 197 ticker parquets plus the index
- Bars: 28,141,725
- Nominal range: 2003-09-10 through 2026-08-26; individual histories vary
- Price basis: raw/unadjusted regular-session 15-minute bars, originally
  backfilled from FMP and incrementally maintained with yfinance
- Sorted current liquid-universe SHA-256:
  `fc587b0949515549ed7c11aa2722baf8458a9100d04025727ab94853e892ab51`
- Explicit 162-name single-stock request file:
  `research/intraday/liquid_single_stock_pilot_2026-08-27.csv`
- Request-file SHA-256:
  `87772864149408afaa8d3bfd4b7f87171e4044f31a9babb653d24424319cf22b`
- Frozen sector-map SHA-256:
  `a1c53ca5be39a2fe68d0a181b6699163b8f725760dd7cfc17bc6b112ae7aab99`

The snapshot covers the liquid cache, not the 1,025-name base-plus-overflow
universe. Requested, available, missing, stale, and rejected names must be
reported explicitly. No result from this run may be described as a 1,025-name
or full-overflow result.

## Primary candidate universe

The primary candidate set is determined without reference to returns:

1. Start with the current `LIQUID_PLUS_COMMODITIES` configuration.
2. Intersect with the frozen R2 snapshot.
3. Remove `OLV_CAP_EXEMPT_ETFS` and `INDICES_SPOT`; the primary test is the
   liquid single-stock sleeve, not indices, sector ETFs, commodity products,
   or volatility products.
4. Require a non-unknown sector in the frozen `data/sector_map.parquet` and an
   available sector proxy from the pre-existing fixed SPDR mapping.
5. Record unsupported sectors/proxies, missing parquets, stale symbols, and
   non-tradeable instruments as coverage rejections. Never replace a missing
   sector proxy with SPY silently.

ETF and market-only residual results may be produced as secondary exploratory
tables, but they cannot advance or rescue either primary template.

## Locked templates and execution clocks

The signal definitions remain those in `research/intraday/templates.py`:

- `gap_first_hour_residual_continuation_v0`: residual overnight gap of at
  least 1.0%, aligned residual first-hour response of at least 0.5%, decision
  at 10:30, entry at the 10:45 bar open, exit at the 15:45 bar close.
- `intraday_residual_shock_reversal_v0`: absolute residual move through the
  13:00 bar of at least 1.5%, decision after that bar closes at 13:15, entry
  at the 13:30 bar open, exit at the 15:45 bar close.

Residuals use the pre-existing 50/50 SPY and sector-proxy reference, with SPY
used once when the sector proxy and market proxy coincide. Thresholds,
directions, clocks, and reference weights are not optimized in this run.

## Data and execution gates

- Eligibility uses completed sessions through T-1: prior close of at least
  $5, trailing 20-session median dollar volume of at least $25 million,
  trailing completeness of at least 95%, and at least 10 historical sessions.
- Expected full sessions come from the repository NYSE calendar. Dates absent
  from every input are surfaced rather than silently disappearing.
- Known/observed NYSE half-days are excluded because v0 specifies a 15:45
  exit and has no preregistered half-day exit.
- Every scheduled bar in the feature window, plus the exact entry and exit
  bars, is required. No endpoint-only shortcut or backward bar fallback is
  permitted.
- Feature, entry, and exit bars with zero volume are rejections.
- Missing market or sector-proxy observations are rejection reasons, not
  silent signal-generation skips.
- The gap template excludes conservative raw-price discontinuity suspects.
  A day is flagged when the open/prior-close ratio is at least 20% from one
  and lies within 3% of a common split factor
  (`0.10, 0.20, 0.25, 1/3, 0.50, 2/3, 0.75, 0.80, 1.25, 4/3, 1.50, 2,
  3, 4, 5, 10`), or when the ratio is below 0.20 or above 5. The unfiltered
  count and a sensitivity table remain visible.

## Evaluation design

These rules contain no trained parameter. The honest first test is therefore
a locked chronological event study with walk-forward stability diagnostics,
not a claim that thresholds were fitted walk-forward.

### Primary endpoint

For each template, the primary endpoint is the mean of daily equal-notional
portfolio returns after 10 bps round-trip costs. Trades on the same day are
equal-weighted before inference so a broad market event does not count as many
independent observations.

- Null: the mean daily net return is zero.
- Report a two-sided day-cluster t-test and a deterministic day-block
  bootstrap confidence interval.
- Apply Holm correction across the two primary template tests.
- Long and short sides are reported separately but are not alternative
  winner-selection tests.

### Prespecified robustness views

- Round-trip cost grid: 5, 10, 15, 20, and 30 bps.
- Calendar-year results and leave-one-year-out results.
- Rolling five-calendar-year history followed by one-calendar-year test
  diagnostics, beginning only when five complete prior years exist. The
  training window is descriptive and does not change the fixed v0 rule.
- Ticker and sector contribution/concentration tables.
- Signal-strength capacity overlays retaining the top 1, 3, 5, and 10 signals
  per template/day. The top-three overlay is the primary small-footprint
  diagnostic; no account or broker rule is inferred from this slot count.
- Gross-return sensitivity with the discontinuity filter disabled is an
  audit view only and cannot be used to advance the gap template.

## Research-priority interpretation

A template can be labeled `Advance to deeper work` only if all of the
following hold:

1. Primary 10-bps mean daily net return is positive with Holm-adjusted
   p-value below 0.05.
2. Mean daily net return remains positive at 20 bps.
3. At least 60% of eligible chronological test years are positive at 10 bps.
4. The top-three capacity overlay remains positive at 10 bps.
5. The edge is not wholly explained by one year, ticker, or sector.

Positive economics that fail one or more robustness gates are `Watch / needs
new holdout`. A non-positive primary result is `Reject v0`. These statuses
allocate research attention only; none authorizes sizing, staging, orders,
deployment, scheduling, or automatic strategy promotion.

## Known limits that the run cannot cure

- The backfill uses today's liquid universe and therefore has survivorship
  bias; it is not point-in-time membership history.
- The sector map is static and therefore has classification lookahead.
- FMP and yfinance volumes differ by tape/source; volume-gated results need a
  source-seam sensitivity before reliance.
- Fifteen-minute bars do not identify spreads, queues, partial fills,
  within-bar sequencing, borrow, halts, or news state.
- Short observations do not imply borrow availability or permission to sell
  short.
- A future all-1,025-name study requires separate artifact-only backfills and
  point-in-time universe work; it is outside this frozen pilot.
