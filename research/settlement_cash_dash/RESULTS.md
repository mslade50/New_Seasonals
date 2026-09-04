# Settlement-Adjusted Month-End Cash-Dash Reversal: Results

## Verdict

**GRADUATES under the frozen decision rule.**

- 2014+ holdout mean after costs is 0.171% (pass).
- Positive calendar years: 18/27; each settlement-regime mean positive = True.
- Mechanism checks passed: 2/3 (Friday=True, pressure=True, timing=False; predicted offset was best in 1/3 regimes).

The primary test contains 319 monthly events from
2000-01-25 through 2026-08-03.
All reported strategy returns below include the frozen 4 bp round-trip cost.

## Primary evidence

| Sample | N | Mean/event | t-stat | Win rate | Profit factor | Worst event |
|---|---:|---:|---:|---:|---:|---:|
| Full sample | 319 | 0.254% | 2.23 | 55.8% | 1.46 | -6.39% |
| Paper era through 2013 | 168 | 0.329% | 1.81 | 56.0% | 1.55 | -6.39% |
| Untouched holdout, 2014+ | 151 | 0.171% | 1.30 | 55.6% | 1.34 | -5.28% |

The full-sample bootstrap mean is 0.254%, with a 95%
month-resampled interval of [0.038%,
0.480%] and P(mean <= 0) =
0.0097. The event-only compounded
return is 110.6%, the annualized return is
2.85%, and maximum event-curve drawdown is
-14.80% while invested roughly three sessions per
month.

Leaving out any single calendar year produces mean returns from
0.181% (omit 2008) to
0.295% (omit 2021).

## Mechanism checks

1. **Friday payment overlap.** Friday month-ends average
   0.420% versus 0.200% for
   other weekdays (difference 0.220%, Welch
   t=0.93).
2. **Prior selling pressure.** The Pearson correlation between the five-session
   return ending at entry and the subsequent gross reversal is
   -0.342 (p=0.0000); the
   Spearman correlation is -0.168. Events following
   negative pressure average 0.712%
   gross versus -0.082% after
   nonnegative pressure.
3. **Settlement timing.** The matrix below shows each settlement regime against
   each candidate entry offset. The frozen rule uses the bold economic mapping
   T+3/T-4, T+2/T-3, and T+1/T-2; controls are diagnostic, not optimized
   replacements.

| Regime | Entry offset | N | Mean net | t-stat |
|---|---:|---:|---:|---:|
| T+3 | -4 | 212 | 0.312% | 2.08 |
| T+2 | -4 | 80 | 0.412% | 1.91 |
| T+1 | -4 | 28 | 0.098% | 0.32 |
| T+3 | -3 | 212 | 0.114% | 0.93 |
| T+2 | -3 | 80 | 0.161% | 0.81 |
| T+1 | -3 | 28 | 0.102% | 0.47 |
| T+3 | -2 | 212 | 0.212% | 1.61 |
| T+2 | -2 | 80 | 0.387% | 2.34 |
| T+1 | -2 | 27 | 0.071% | 0.24 |

As a broad baseline, non-event, non-overlapping three-session SPY blocks average
0.082% gross versus 0.294%
for the scheduled event windows (difference 0.212%,
Welch t=1.72).

## Robustness map

Only the 3-session, 4-bp row is the frozen primary specification.

| Hold sessions | Cost (bp) | N | Mean net | t-stat | Win rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 320 | 0.113% | 1.58 | 52.2% |
| 1 | 4 | 320 | 0.073% | 1.02 | 50.0% |
| 1 | 10 | 320 | 0.013% | 0.19 | 45.3% |
| 1 | 20 | 320 | -0.087% | -1.21 | 40.0% |
| 2 | 0 | 320 | 0.297% | 3.24 | 59.4% |
| 2 | 4 | 320 | 0.257% | 2.80 | 57.5% |
| 2 | 10 | 320 | 0.197% | 2.15 | 55.3% |
| 2 | 20 | 320 | 0.097% | 1.05 | 51.6% |
| 3 | 0 | 319 | 0.294% | 2.58 | 56.7% |
| 3 | 4 | 319 | 0.254% | 2.23 | 55.8% |
| 3 | 10 | 319 | 0.194% | 1.70 | 53.3% |
| 3 | 20 | 319 | 0.094% | 0.82 | 51.4% |
| 4 | 0 | 319 | 0.353% | 2.86 | 57.1% |
| 4 | 4 | 319 | 0.313% | 2.53 | 56.4% |
| 4 | 10 | 319 | 0.253% | 2.05 | 54.9% |
| 4 | 20 | 319 | 0.153% | 1.24 | 52.4% |
| 5 | 0 | 319 | 0.524% | 4.04 | 60.5% |
| 5 | 4 | 319 | 0.484% | 3.73 | 59.9% |
| 5 | 10 | 319 | 0.424% | 3.27 | 58.3% |
| 5 | 20 | 319 | 0.324% | 2.50 | 55.8% |

## Calendar-year returns

| Year | N | Mean/event | Compounded |
|---:|---:|---:|---:|
| 2000 | 12 | -0.318% | -4.05% |
| 2001 | 12 | -0.299% | -3.85% |
| 2002 | 12 | -0.049% | -1.05% |
| 2003 | 12 | 0.404% | 4.87% |
| 2004 | 12 | 0.348% | 4.11% |
| 2005 | 12 | 0.202% | 2.38% |
| 2006 | 12 | 0.509% | 6.25% |
| 2007 | 12 | -0.074% | -1.16% |
| 2008 | 12 | 2.119% | 26.35% |
| 2009 | 12 | 0.382% | 4.53% |
| 2010 | 12 | -0.297% | -3.61% |
| 2011 | 12 | 1.257% | 15.86% |
| 2012 | 12 | 0.190% | 2.18% |
| 2013 | 12 | 0.229% | 2.74% |
| 2014 | 12 | 0.205% | 2.46% |
| 2015 | 12 | 0.257% | 2.81% |
| 2016 | 12 | 0.381% | 4.62% |
| 2017 | 12 | 0.323% | 3.91% |
| 2018 | 12 | 0.120% | 1.35% |
| 2019 | 12 | 0.155% | 1.80% |
| 2020 | 12 | -0.385% | -4.78% |
| 2021 | 12 | -0.809% | -9.42% |
| 2022 | 12 | 0.815% | 9.77% |
| 2023 | 12 | 0.956% | 12.01% |
| 2024 | 12 | -0.462% | -5.44% |
| 2025 | 12 | -0.289% | -3.49% |
| 2026 | 7 | 1.512% | 10.97% |

## Interpretation boundary

This is evidence for a recurring liquidity premium, not proof that every event
is caused by month-end payments. It is a low-frequency equity-beta trade with
gap risk and no stop. Its main live risk is structural decay: settlement has
already shortened to T+1, and the T+1 regime has a much smaller sample. The
strategy should therefore remain a research sleeve until the modern regime has
enough observations for a prospective recheck.
