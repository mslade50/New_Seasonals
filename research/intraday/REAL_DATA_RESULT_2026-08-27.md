# Intraday v0 Real-Data Result — 2026-08-27

## Decision

Both locked v0 templates are **Reject v0**. Neither is suitable for production,
paper promotion, sizing, staging, or scheduling. This is a research-priority
decision under `REAL_DATA_PREREGISTRATION.md`, not an investment recommendation.

| Locked template | 10 bps mean active-day return | 95% bootstrap CI | Holm p-value | 20 bps mean | Positive complete test years | K=3, 10 bps mean/session | Verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| Gap + first-hour residual continuation | -10.07 bps | [-13.59, -6.56] bps | 6.37e-9 | -20.07 bps | 1 / 17 | -5.08 bps | Reject v0 |
| Intraday residual-shock reversal | -8.34 bps | [-9.25, -7.42] bps | 1.07e-73 | -18.34 bps | 0 / 17 | -10.60 bps | Reject v0 |

The small p-values establish statistically clear **negative** net results; they
are not evidence for an edge. Leave-one-year-out means remain negative for all
24 omissions for both templates.

## Economics

- At 5 bps round-trip cost, the active-day means are still negative: -5.07 bps
  for gap continuation and -3.34 bps for shock reversal.
- Before the flat cost proxy, gap continuation is effectively zero at the
  daily portfolio level (-0.07 bps). Shock reversal has a small +1.66 bps gross
  daily mean, far below even the lowest prespecified 5 bps cost case.
- The top-three signal-strength overlays remain negative at the primary 10 bps
  cost. The gap template's top-one overlay is positive at 5 bps but negative at
  10 bps, with extreme volatility and drawdown; that cannot rescue v0.
- The raw-price discontinuity filter removed 17 gap events. Turning the filter
  off makes gross daily results slightly worse, so the conclusion is not an
  artifact of that filter.

## Scope and accounting

- Frozen R2 cache: 197 ticker files, 28,141,725 bars, 2003-09-10 through
  2026-08-26, with differing per-name histories.
- Requested liquid single stocks: 162; evaluated: 155.
- Exclusions: BNY, DOV, ED, and EIX lacked frozen sector metadata; MRSH lacked a
  parquet; PSA and SPG required the absent XLRE proxy.
- Executed observations: 18,534 gap-continuation trades and 97,790 shock-reversal
  trades. Seven additional signals were rejected for missing or zero-volume
  scheduled execution bars. Signal accounting reconciles exactly.
- The source manifest, universe file, and sector map were hash-validated before
  output. All 166 loaded candidate/proxy file hashes matched the immutable
  snapshot manifest.

This is not an all-1,025-name intraday test. The frozen cache contains 197 names
and overlaps only 196 of the configured base-plus-overflow universe. An honest
all-universe study requires artifact-only backfills and point-in-time membership
work first.

## Research implication

Do not refine these thresholds on the same sample and call the result validated.
If intraday work continues, the cleanest new v1 wedge is to test whether the
small gross shock-reversal effect increases monotonically with prespecified
signal strength, liquidity, or market-state gates on a new holdout. It must
clear realistic spread, fee, slippage, share-rounding, and borrow assumptions.
Gap continuation should be deprioritized unless a materially different economic
mechanism is proposed and preregistered.

## Limits

The study uses today's liquid universe and static sector classifications, so it
has survivorship and classification lookahead. The cache combines an FMP
backfill with yfinance incremental maintenance. Fifteen-minute bars cannot
identify spreads, queues, partial fills, within-bar sequencing, halts, news, or
borrow availability. Capacity overlays are signal slots, not an account or
broker simulation.

