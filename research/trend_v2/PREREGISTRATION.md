# Trend V2 preregistration and promotion gates

Version: `trend_v2.v0`, frozen 2026-08-27. This document governs research only.
Nothing in this harness can promote a strategy, change the production sleeve,
stage an order, or allocate capital.

## Locked comparison

The comparator is the production 12-ETF core signal/weight family as of the
freeze date:

- universe: SPY, QQQ, IWM, EFA, EEM, FXI, VNQ, GLD, SLV, DBC, TLT, LQD;
- month-end signal: 12-1 momentum > 0 **and** close > 10-month SMA;
- inverse-63-day-volatility slots over eligible assets, 20% cap, long/flat;
- next-period execution and 5 bps per side.

The live month-entry fragility gate is a separate book-state overlay, not a
candidate trend definition. It is held outside this price-only comparison and
must be replayed unchanged during the portfolio-increment gate. This keeps an
overlay with shorter history from changing the trend-family trial sample.

It is a benchmark, not one of the new trials. A fingerprint of the complete
specification is written to every run manifest.

## Candidate families and trial count

### A. Multi-speed time-series ETF family: four trials

All trials use 3/6/12-month horizons, 63-day volatility, 126-day covariance,
10% portfolio-volatility ceiling, 20% asset cap, 100% gross cap, 1% no-trade
band, 50% soft turnover cap, and 5 bps per side.

| Trial | Votes | Enter | Exit |
| --- | --- | ---: | ---: |
| `ms_sign_2in_1out` | positive horizon return | 2 | <=1 |
| `ms_sign_unanimous_in` | positive horizon return | 3 | <=1 |
| `ms_breakout_2in_1out` | channel state | 2 | <=1 |
| `ms_breakout_unanimous_in` | channel state | 3 | <=1 |

Channel state enters above the prior horizon high and exits below the prior
half-horizon low. Entry and exit use the same rule for every asset.

### B. Stock residual-trend family: one trial

This is a separate family and should not be pooled with the four ETF variants
when discussing what “improved” the production trend sleeve.

- 126-day beta estimated through the prior close;
- 126-day sum of market-residual daily returns;
- monthly rebalance after ranking within the sector known on that date;
- long-only top sector quintile for v0;
- equal budget across sectors, inverse vol within sector;
- 5% asset cap, 100% gross cap, 25 bp no-trade band;
- 75% soft monthly turnover cap and 10 bps per side;
- minimum 252 daily observations.

The locked candidate count is five when stock data is supplied, or four for an
ETF-only run. The benchmark row does not increase this count. All exploratory
variations—including discarded ones—must be added to the family trial count.

## Primary questions

1. Does a multi-speed ensemble improve the frozen sleeve's net return/drawdown
   or turnover trade-off without relying on one asset or episode?
2. Does hysteresis reduce turnover enough to matter after costs while retaining
   the benchmark's extended-bear behavior?
3. Does the stock residual family deliver independent trend exposure after
   controlling market and sector effects, rather than repackaged equity beta?

The maximum full-sample Sharpe across trials is descriptive, not a selection
criterion.

## Required robustness before shadow consideration

Every candidate must pass all of these on a separately recorded run:

1. **Data integrity:** adjusted OHLC provenance, split checks, duplicate audit,
   coverage report, and historical sector/universe membership. A current
   survivor list or undated sector map fails the stock-family gate.
2. **Timing:** identical conclusions under true next-open execution. A
   close-only run can explore but cannot promote.
3. **Trial ledger:** every tested family, threshold, horizon, universe, cost,
   and start/end-date variation is counted, including unpublished failures.
4. **Holdout:** the last 20% of months remains untouched until the rule is
   frozen. Net holdout return must be positive, and its Sharpe cannot be less
   than half the development-period Sharpe.
5. **Leave-one-era-out:** no omitted five-year block may flip the full candidate
   result below cash. No single calendar year may contribute more than 35% of
   cumulative net profit.
6. **Cost stress:** the conclusion survives at 2x preregistered costs and a
   one-session execution delay. Gross-to-net degradation is reported, never
   hidden behind a zero-cost headline.
7. **Concentration:** no ETF exceeds its 20% cap; the stock family respects 5%
   names and equal sector budgets. P&L attribution must not be dominated by one
   ticker, sector, or crisis.
8. **Portfolio increment:** compared on common months, adding a fixed-risk
   candidate to the frozen sleeve must improve at least one of net Sharpe,
   maximum drawdown, or left-tail return without materially degrading the other
   two. High correlation is acceptable only with a measurable turnover or
   drawdown improvement.
9. **Operational fit:** stale/missing inputs fail closed, target turnover is
   executable, and any short version separately proves borrow and financing.

Passing these gates authorizes only a human review for shadow tracking. Shadow
promotion, live promotion, sizing, and retirement require separate explicit
decisions and out-of-sample evidence.

## Overfitting cautions

- Three horizons are one economic ensemble, not three independent samples.
- Hundreds of tickers do not create hundreds of independent observations;
  market days and trend episodes are the relevant clusters.
- Expanding from diversified ETFs to correlated stocks changes the exposure
  into cross-sectional equity momentum. Do not claim it is merely “more of” the
  production time-series sleeve.
- Signal “cleanliness,” hit rate, and a smooth full-sample curve are not goals.
  Trend whipsaws are expected; removing all of them usually removes convexity.
- Universe, sector-history, and ETF-availability hindsight can dominate small
  rule improvements. Report them before performance.
- The frozen benchmark already reflects prior research choices. Comparisons are
  conditional on that selection history, not pristine first tests.
