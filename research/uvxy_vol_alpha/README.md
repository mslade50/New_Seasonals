# UVXY VIX-Compression × Fragility Research

This package tests one frozen, research-only long-volatility hypothesis. It does
not add a production strategy or touch order/portfolio state.

## Verdict as of 2026-08-24

Reject as alpha. The primary rule produced 11 non-overlapping trades from
2018-03-01 through 2026-08-24:

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
- `uvxy_vol_alpha_leave_one_year_out.csv` — year concentration
- `uvxy_vol_alpha_matched_controls.csv` — calendar/VIX/VIX3M-slope controls

## Evidence limits

The fragility histories before July 2026 are recomputed using current signal
definitions; they are not clean out-of-sample vintages. The equal-weight simple
dial avoids full-sample outcome weights, but current signal definitions and
constituent/classification inputs still create research lookahead. No primary
event occurs in the genuinely append-only simple-dial period. These facts alone
block live deployment even if a neighboring grid cell looks attractive.

The matched-control comparison is descriptive: controls can overlap and be
reused, so it does not claim an independent paired p-value. VIX/VIX3M is a
volatility-slope proxy, not direct VIX-futures roll carry. The JSON summary
fingerprints every local input so this exact cache vintage can be identified.

See the sponsor's [UVXY product page](https://www.proshares.com/our-etfs/strategic/uvxy)
for the current daily objective and the distinction between UVXY's VIX-futures
benchmark and spot VIX.
