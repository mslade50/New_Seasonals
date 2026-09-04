# Treasury Month-End Benchmark-Demand Strategy

Status: **frozen before the repo data were tested**

Frozen: 2026-08-09

## Economic premise

This is a structural-flow strategy. Life insurers, pension funds, index funds,
and other benchmarked fixed-income investors are predictable buyers of coupon
Treasuries at month-end. Their demand is linked to portfolio rebalancing,
benchmark-extension trades, and reporting/window-dressing constraints rather
than a discretionary view on rates. The strategy takes the other side before
that price-insensitive demand arrives and exits into it at month-end.

The specification follows Hartley and Schwarz, *Predictable End-of-Month
Treasury Returns* (2019), which reports that coupon-Treasury excess returns are
positive and significant in the final five sessions of the month, near zero on
other days, and stronger at longer maturities. Because the paper was published
in 2019, January 2020 onward is the untouched external holdout.

## Primary rule

- Instrument: TLT (liquid, long-duration U.S. Treasury ETF).
- Direction: long.
- Month anchor `T`: the final TLT trading session of the calendar month.
- Entry: market-on-close on `T-5` (five close-to-close return sessions before
  the month-end close).
- Exit: market-on-close on `T`.
- Frequency: every complete calendar month.
- No rate, trend, volatility, equity, weekday, or macro filters.
- No stop or target; the month-end close is the fixed time exit.
- Trading friction: 10 basis points round trip, deducted once per event.
- Returns use adjusted TLT bars because entry and exit are recomputed relative
  prices, not frozen absolute dollar levels.

## Frozen evaluation

Primary outcomes:

1. Net event return and one-sample t-statistic for the full ETF sample.
2. Same statistics for the pre-publication sample through 2019 and the
   untouched January 2020+ holdout.
3. Win rate, payoff ratio, profit factor, worst event, event-only maximum
   drawdown, calendar-time annualized return, and annualized event Sharpe.

Mechanism and placebo tests:

1. The same T-5 to T window should be directionally stronger in TLT than IEF
   because the paper finds the effect increasing with Treasury maturity.
2. T-5 to T should outperform equal-length windows immediately before it
   (T-10 to T-5) and after it (T to T+5).
3. T-5 to T should outperform non-overlapping five-session TLT windows outside
   the month-end exposure window.

Robustness tests (not selection candidates):

- T-3, T-4, T-5, T-6, and T-7 entry closes, always exiting at T.
- Gross and net results at 0, 4, 10, and 20 bps round-trip friction.
- Leave-one-year-out means and a month-level bootstrap confidence interval.
- Calendar-year and pre/post-publication breakdowns.

## Decision rule

The strategy graduates only if the 2020+ holdout mean is positive after 10 bps,
the full-sample mean has a two-sided t-statistic above 2.0, at least half of the
calendar years are profitable, and at least two of the three mechanism/placebo
tests point in the predicted direction. Otherwise it is documented as rejected
without adding a newly discovered filter.

