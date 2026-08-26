# UVXY VIX-Compression × Fragility Research

This package tests one frozen, research-only long-volatility hypothesis. It does
not add a production strategy or touch order/portfolio state.

## Verdict as of 2026-08-25

Reject as alpha. The primary rule produced 11 non-overlapping trades from
2018-03-01 through 2026-08-25:

| Metric | Result |
|---|---:|
| Mean net UVXY return | -0.39% |
| Median net return | -1.90% |
| Win rate | 36% |
| Event-level t-stat | -0.13 |
| 90% year-block bootstrap interval | -4.70% to +4.84% |
| Bootstrap share of resampled means <= 0 | 58% |
| Genuinely PIT qualifying trades | 0 |

The rule improved on UVXY's unconditional five-session decay but did not create
positive absolute expectancy. It was also slightly worse than VIX Range
Compression alone, negative after removing the two best trades, and negative at
50 bps round-trip cost. Threshold and holding-period neighbors change sign and
remain small-sample.

### Short-horizon overlay result

The follow-up freezes 5d and 21d ex-VRC fragility overlays instead of choosing a
horizon after observing returns. Each version recomputes the six registered
non-VRC component histories under one current code vintage, applies linear decay
over the named horizon, equal-weights the components, smooths once for five
sessions, causally ranks the score over the preceding 504 sessions, and uses the
prior day's rank. Execution and the VRC activation rule are unchanged.

| Overlay at 67th-percentile gate | Trades | Mean net | Median | Win rate | t-stat |
|---|---:|---:|---:|---:|---:|
| VRC only | 22 | -0.35% | -3.13% | 36% | -0.14 |
| Original 63d sizing-basis control | 11 | -0.39% | -1.90% | 36% | -0.13 |
| Native 5d ex-VRC | 12 | -1.57% | -3.08% | 33% | -0.55 |
| Native 21d ex-VRC | 12 | -1.54% | -3.08% | 33% | -0.54 |
| 5d and 21d both high | 11 | -1.21% | -1.90% | 36% | -0.41 |
| Native 63d ex-VRC | 14 | +0.65% | -3.08% | 36% | +0.18 |

The 5d and 21d ranks correlate 0.82, so they provide little independent
confirmation. The native 63d row is not evidence of alpha: removing its best two
trades changes its mean to -3.80%, its median is negative, and its year-block 90%
interval spans roughly -4.80% to +9.14%. None of the recomputed overlay history
is genuinely point in time. Shortening the fragility memory therefore did not
solve this VRC-activation, next-open-to-fifth-close setup in the current sample.

## Frozen primary rule

1. Compute the production VIX Range Compression signal from VIX closes:
   21-session closing range below its trailing 504-session 15th percentile,
   VIX above 13, and VIX above its 20-session average.
2. Require five consecutive VRC-off sessions before an activation.
3. Remove VRC's equal-weight contribution from the registered seven-signal
   simple fragility shadow. This avoids testing VRC against a dial that already
   contains VRC.
4. Smooth the resulting ex-VRC score using the production convention and rank
   it against only its preceding 504 sessions.
5. On the VRC activation date, require the **prior session's** ex-VRC fragility
   rank to be at least 67 (upper tercile).
6. Buy UVXY at the next session's open and sell at the fifth post-signal
   session's close. Use 12 bps round-trip cost; do not overlap positions.

The primary sample begins after UVXY changed its daily objective from 2× to 1.5×
at the February 27, 2018 close. Adjusted OHLC handles reverse splits. UVXY tracks
a rolling short-term VIX-futures index, not spot VIX, so term structure is
reported as a diagnostic rather than retrofitted as an optimized filter.

## Run

From the repository root:

```powershell
python research/uvxy_vol_alpha/run_backtest.py `
  --data-dir data `
  --output-dir artifacts/uvxy-vol-alpha
```

The isolated development worktree has no copied price cache, so its verified run
used the parent workspace's read-only data directory:

```powershell
python research/uvxy_vol_alpha/run_backtest.py `
  --data-dir "C:\Users\McKinley Slade\dev\New_Seasonals\data" `
  --output-dir artifacts/uvxy-vol-alpha
```

Outputs:

- `uvxy_vol_alpha_report.html` — first-read PM report
- `uvxy_vol_alpha_summary.json` — machine-readable verdict and tests
- `uvxy_vol_alpha_trades.csv` — primary trade ledger
- `uvxy_vol_alpha_sensitivity.csv` — threshold/hold/cost grid
- `uvxy_vol_alpha_short_horizon_overlays.csv` — 5d/21d/63d overlay grid
- `uvxy_vol_alpha_short_horizon_trades.csv` — per-trade short-overlay audit ledger
- `uvxy_vol_alpha_leave_one_year_out.csv` — year concentration
- `uvxy_vol_alpha_matched_controls.csv` — calendar/VIX/VIX3M-slope controls

## Evidence limits

The original fragility histories before July 2026 and all short-overlay
component histories are recomputed using current signal definitions; they are
not clean out-of-sample vintages. The native short overlays avoid full-sample
outcome weights and exclude VRC, but current signal definitions and
constituent/classification inputs still create research lookahead. No primary
event occurs in the genuinely append-only simple-dial period, and the native
short overlays have zero genuinely PIT rows. These facts alone block live
deployment even if a neighboring grid cell looks attractive. Stored incumbent
5d/21d results are retained only as circular diagnostics because those dials
contain VRC and outcome-fitted weights.

The matched-control comparison is descriptive: controls can overlap and be
reused, so it does not claim an independent paired p-value. VIX/VIX3M is a
volatility-slope proxy, not direct VIX-futures roll carry. The JSON summary
fingerprints the explicitly loaded caches, the classification and seasonal-rank
files used by the current-vintage recomputation, and the relevant local source
files. This identifies the local research vintage but is not a full Python
environment lockfile.

See the sponsor's [UVXY product page](https://www.proshares.com/our-etfs/strategic/uvxy)
for the current daily objective and the distinction between UVXY's VIX-futures
benchmark and spot VIX.
