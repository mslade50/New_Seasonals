# Primary-Stock 100-Day Breakout — Frozen Exploratory Specification

Status: research only. This study may produce local artifacts under `artifacts/`.
It may not write scanner, staging, portfolio, broker, cache, website, R2, Sheets,
or any other production state, and it cannot promote itself.

## Research question

Does a simple long-only fresh-breakout portfolio retain a positive, diversified,
net-of-cost return profile when applied to the stock-like portion of the current
primary (non-overflow) universe?

The test is intentionally exploratory because the current curated universe is
applied backward and the local master cache is adjusted, rolling-vintage OHLCV.
A favorable result can only advance the idea to point-in-time membership and
immutable raw/as-traded price validation.

## Frozen universe

- Start with `strategy_config.LIQUID_PLUS_COMMODITIES`.
- Exclude every ticker in `strategy_config.OLV_CAP_EXEMPT_ETFS`.
- The expected run-time result is 162 stock-like tickers and 35 excluded ETFs,
  spot indices, and commodity vehicles.
- Freeze the exact sorted ticker list and SHA-256 in the run bundle.
- Missing requested tickers are disclosed. No fallback to another universe.

## Primary signal and execution

For ticker *i* on session *t*:

1. `breakout[t] = Close[t] > max(High[t-100:t-1])` using 100 complete prior
   benchmark-calendar sessions. Today is excluded from the threshold.
2. `fresh[t] = breakout[t] AND NOT breakout[t-1]`. Missing prior state cannot
   create a fresh signal.
3. `ROC100[t] = Close[t] / Close[t-100] - 1`, on the same adjusted basis as
   the OHLC used by the exploratory test. "Raw" means unscaled and
   non-volatility-normalized, not unadjusted dollars.
4. A signal while already held is discarded, never queued.
5. At session *t+1* open, all valid fresh candidates are sorted by ROC100
   descending, then ticker ascending for deterministic ties.
6. Entry execution is opening price plus 5 bps. No pyramiding.

## Chande–Kroll long stop

- True range uses prior close.
- Wilder ATR period `p=10`, seeded with the arithmetic mean of the first 10
  consecutive true-range observations and recursively updated thereafter.
- `preliminary[t] = max(High[t-9:t]) - 3 * ATR10[t]`.
- `CK[t] = max(preliminary[t-8:t])` (`q=9`).
- Initial stop for a next-open entry is CK from the signal close.
- Skip an entry whose modeled fill is at or below the initial stop.
- An active stop never loosens: after a surviving close,
  `active_stop[t+1] = max(active_stop[t], CK[t])`.
- The stop active at the start of a session remains fixed for that session.
  A close-computed update cannot reach backward into the same day's low.
- If `Open <= active_stop`, exit at the open less exit cost. Otherwise, if
  `Low <= active_stop`, exit at the active stop less exit cost.
- There is no time exit. Terminal close liquidation is only an end-of-sample
  reporting convention and is identified separately.

## Portfolio construction

- Starting equity comes from `strategy_config.ACCOUNT_VALUE` (expected
  $750,000 at freeze time). Results are primarily percentage based.
- Target risk is at most 0.5% of prior-close equity for each admitted entry.
- Per-share modeled risk includes entry and regular stop-exit cost:
  `entry_fill - initial_stop * (1 - cost_bps/10000)`.
- Whole shares only; size is the minimum allowed by:
  - the 0.5% target risk;
  - 10% of prior-close equity in one name;
  - available opening cash;
  - 100% gross admission cap;
  - 5% aggregate current stop-risk admission cap.
- Caps constrain new entries only. Appreciation never forces rebalancing or
  trimming. Gap-stop exits at the open free cash for open entries; intraday
  exits do not.
- Continue down the ROC ranking after an infeasible candidate.
- No current book overlays, `GLOBAL_RISK_MULTIPLIER`, fragility scaling,
  earnings filters, pooled strategy caps, or other production rules apply.

## Data and causal controls

- Runner requires an explicit local parquet path and never downloads data.
- Benchmark calendar is the observed SPY session calendar; no OHLC forward fill.
- A held position missing an OHLC session fails the run.
- Indicators reset/withhold values after missing observations until their full
  windows are valid again.
- Input file, universe, config, preregistration, and code hashes are recorded.

## Prespecified outputs

- Full period; 2000s, 2010s, 2020+; and 2015+ holdout views.
- Strategy, SPY, and a no-cost daily-rebalanced equal-weight comparator of the
  same static primary stocks.
- CAGR, volatility, Sharpe, Sortino, Calmar, maximum drawdown, turnover,
  exposure, positions, trade expectancy in R, realized versus target risk,
  exit path, and contribution concentration.
- Moving-block bootstrap interval for monthly Sharpe.
- Full trade, candidate, equity, yearly-return, period, and robustness support
  files plus one standalone HTML report.

## Robustness grid (not optimization)

The primary row is 100-day breakout, CK 10/3/9, 5 bps per side. Run these
neighbors without selecting a winner:

- Costs: 0, 10, and 20 bps per side.
- Breakout window: 80 and 120 sessions, with ROC window matched.
- CK ATR multiple: 2.5 and 3.5.

## Exploratory advance gates

All must pass for the result to say only "advance to raw / point-in-time
validation":

1. Positive CAGR and Sharpe in the 2000s, 2010s, and 2020+ segments.
2. Positive full-period Sharpe at 20 bps per side.
3. Positive full-period Sharpe for all four breakout/stop neighbors.
4. Maximum drawdown smaller in magnitude than the equal-weight comparator.
5. Top-five positive contributors account for less than 50% of total positive
   ticker contribution.

Failure means do not advance the current formulation. Passing never means ship,
stage, size a live position, or change production.
