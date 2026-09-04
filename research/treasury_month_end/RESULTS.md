# Treasury Month-End Benchmark-Demand Strategy: Results

## Verdict

**GRADUATES under the frozen decision rule.**

- 2020+ holdout mean after 10 bp is 0.255% (pass).
- Full-sample t-stat is 3.54 (pass versus >2.0).
- Positive years: 20/25 (pass).
- Mechanism/placebo checks passed: 3/3 (maturity=True, timing=True, baseline=True).

## Primary evidence

| Sample | N | Mean/event | t-stat | Win rate | Profit factor | Worst event |
|---|---:|---:|---:|---:|---:|---:|
| Full sample | 288 | 0.330% | 3.54 | 59.0% | 1.69 | -4.13% |
| Through 2019 | 209 | 0.359% | 3.44 | 62.7% | 1.81 | -4.13% |
| 2020+ holdout | 79 | 0.255% | 1.28 | 49.4% | 1.44 | -4.07% |

The full sample compounds to 149.4% while invested
only five sessions per month, with a -12.10% event-curve
maximum drawdown and 0.72 annualized event Sharpe.
The month-bootstrap mean is 0.330% with a 95% interval of
[0.149%, 0.512%] and P(mean <= 0) =
0.0001. Leaving out any one year gives a mean
between 0.303% (omit 2014) and
0.351% (omit 2017).

## Mechanism and placebo checks

- **Maturity:** TLT averages 0.430% gross versus
  0.269% for IEF over the identical T-5 to T
  window (difference 0.161%).
- **Timing:** the primary window averages 0.430%
  gross versus 0.179% for T-10 to T-5,
  and -0.267% for T to T+5.
- **Ordinary five-day blocks:** non-event windows average
  -0.049% versus
  0.430% in the month-end window
  (difference 0.480%, Welch
  t=4.10).

## Execution and portfolio fit

- The gross mean is 43.0 bp per event, which is
  also the friction break-even before alpha reaches zero. The frozen 10 bp cost
  consumes less than one-quarter of that gross mean, and the rule remains
  positive at the 20 bp robustness assumption.
- 33.7 bp accrues by the month-end
  open and another 9.3 bp accrues
  from that open to the month-end close (intraday t-stat
  2.48). The MOC exit is therefore part
  of the edge, not an interchangeable convenience.
- The current full trade ledger contains 0 TLT,
  IEF, LQD, or HYG trades, so this is not a duplicate strategy-book position.
  TLT event returns correlate -0.233 with
  SPY over the same dates and 0.087
  with monthly ledger exit P&L (the latter is a rough, non-mark-to-market
  diversification check).
- The maximum losing streak is 4 events. The two worst
  events are 2008-10 at
  -4.13% and
  2022-12 at
  -4.07%.

## Robustness map

The frozen primary row is T-5 at 10 bp. Other rows are sensitivity checks, not
alternative strategies selected after seeing results.

| Entry | Cost (bp) | N | Mean net | t-stat | Win rate |
|---:|---:|---:|---:|---:|---:|
| T-3 | 0 | 288 | 0.368% | 4.72 | 62.8% |
| T-3 | 4 | 288 | 0.328% | 4.21 | 62.2% |
| T-3 | 10 | 288 | 0.268% | 3.44 | 60.1% |
| T-3 | 20 | 288 | 0.168% | 2.16 | 56.6% |
| T-4 | 0 | 288 | 0.352% | 4.00 | 60.8% |
| T-4 | 4 | 288 | 0.312% | 3.55 | 60.1% |
| T-4 | 10 | 288 | 0.252% | 2.86 | 59.4% |
| T-4 | 20 | 288 | 0.152% | 1.73 | 56.2% |
| T-5 | 0 | 288 | 0.430% | 4.62 | 60.8% |
| T-5 | 4 | 288 | 0.390% | 4.19 | 60.1% |
| T-5 | 10 | 288 | 0.330% | 3.54 | 59.0% |
| T-5 | 20 | 288 | 0.230% | 2.47 | 56.6% |
| T-6 | 0 | 288 | 0.483% | 4.51 | 60.4% |
| T-6 | 4 | 288 | 0.443% | 4.13 | 59.4% |
| T-6 | 10 | 288 | 0.383% | 3.57 | 58.7% |
| T-6 | 20 | 288 | 0.283% | 2.64 | 56.6% |
| T-7 | 0 | 288 | 0.479% | 4.07 | 60.1% |
| T-7 | 4 | 288 | 0.439% | 3.73 | 59.4% |
| T-7 | 10 | 288 | 0.379% | 3.22 | 58.0% |
| T-7 | 20 | 288 | 0.279% | 2.37 | 56.2% |

## Calendar years

| Year | N | Mean/event | Compounded |
|---:|---:|---:|---:|
| 2002 | 5 | 0.869% | 4.39% |
| 2003 | 12 | 0.031% | 0.17% |
| 2004 | 12 | -0.084% | -1.11% |
| 2005 | 12 | 0.241% | 2.88% |
| 2006 | 12 | -0.106% | -1.39% |
| 2007 | 12 | 0.584% | 7.18% |
| 2008 | 12 | 0.230% | 2.61% |
| 2009 | 12 | 0.083% | 0.72% |
| 2010 | 12 | 0.736% | 9.09% |
| 2011 | 12 | 0.338% | 3.91% |
| 2012 | 12 | 0.835% | 10.31% |
| 2013 | 12 | 0.110% | 1.23% |
| 2014 | 12 | 0.954% | 12.01% |
| 2015 | 12 | 0.190% | 2.09% |
| 2016 | 12 | 0.732% | 9.00% |
| 2017 | 12 | -0.149% | -1.87% |
| 2018 | 12 | 0.400% | 4.83% |
| 2019 | 12 | 0.761% | 9.46% |
| 2020 | 12 | 0.313% | 3.58% |
| 2021 | 12 | 0.030% | 0.21% |
| 2022 | 12 | -0.065% | -1.02% |
| 2023 | 12 | -0.056% | -0.82% |
| 2024 | 12 | 0.533% | 6.30% |
| 2025 | 12 | 0.876% | 10.93% |
| 2026 | 7 | 0.081% | 0.52% |

## Risk boundary

This is long-duration exposure, not arbitrage. A hawkish policy surprise,
inflation shock, or disorderly selloff can overwhelm benchmark demand. There is
no stop because daily gaps dominate stop execution; sizing must treat the worst
historical five-session loss as a floor, not a bound. The post-publication
sample is the key decay check and should be reviewed annually without changing
the rule between reviews.
