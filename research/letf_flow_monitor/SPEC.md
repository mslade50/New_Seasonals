# Leveraged ETF Flow Monitor — Frozen Research Specification

## Objective

Measure whether leveraged-ETF creations/redemptions and mechanical daily
rebalancing can anticipate (1) volatility expansion, (2) late-day price
continuation, and (3) subsequent market turns. This is a research monitor,
not a production trading rule.

## Version-one universe

The first version uses 16 ProShares funds across Nasdaq-100, S&P 500, Dow 30,
and Russell 2000 complexes. Each complex includes +3x, -3x, +2x, and -2x
funds. ProShares is used because it publishes a free official historical CSV
with NAV, prior NAV, shares outstanding, and AUM for every included fund.

## Measurements

For fund `i` on day `t`:

1. Primary-market flow:

   `flow(i,t) = split-adjusted change in shares outstanding × NAV(t)`

2. Flow-associated benchmark exposure:

   `flow exposure(i,t) = leverage(i) × flow(i,t)`

3. Mechanical rebalance demand:

   `mechanical(i,t) = AUM(i,t-1) × L × (L-1) × benchmark return(t)`

4. Estimated total benchmark demand:

   `estimated demand = flow exposure + mechanical demand`

Positive demand means buy the benchmark; negative demand means sell it.
Mechanical and total demand are estimates, not observed market orders.

## Information timing

- Official shares outstanding and AUM for day `t` are treated as post-close
  information and may first predict day `t+1`.
- The 15:30 ET mechanical estimate uses only day `t-1` AUM and the benchmark
  return through 15:30. It may predict the final 30 minutes of day `t`.
- Rolling thresholds and z-scores exclude the current observation.
- No same-day primary flow is used in the intraday forecast.

## Predefined hypotheses

1. Gross primary flow above its prior 252-session 90th percentile predicts
   higher next-five-day realized volatility relative to trailing volatility.
2. Absolute estimated demand relative to proxy dollar volume above its prior
   252-session 90th percentile predicts the same volatility expansion.
3. Net bearish flow exposure below its prior 10th percentile is a contrarian
   positive next-five-day turn signal.
4. Net bullish flow exposure above its prior 90th percentile is a contrarian
   negative next-five-day turn signal.
5. Extreme modeled pressure at 15:30 continues in the same direction through
   the close.
6. Extreme modeled pressure at 15:30 reverses in the next session.

Consecutive signals are reduced to the first event following five quiet
benchmark sessions. All market-complex observations on the same date are
then clustered into one event-date observation, and inference uses Newey-West
standard errors. `p < 0.10` is only an exploratory support threshold;
production promotion requires stability checks, costs, and a frozen
out-of-sample gate.

## Known limitations

- Authorized participants and swap counterparties can pre-hedge or execute at
  times different from the model's assumed window.
- The free first version excludes leveraged funds from issuers without a
  similarly accessible official historical shares series, notably Direxion.
- End-of-day shares outstanding cannot identify intraday creations in real
  time. A paid feed is the clean upgrade path if the research survives.
