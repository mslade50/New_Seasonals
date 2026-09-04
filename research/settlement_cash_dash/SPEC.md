# Settlement-Adjusted Month-End Cash-Dash Reversal

Status: **frozen before the repo data were tested**

Frozen: 2026-08-09

## Economic premise

This is a liquidity-provision strategy. Institutions that must fund pensions,
payroll, redemptions, and other month-end payments are price-insensitive sellers
of liquid assets. The sale deadline is determined by the prevailing settlement
cycle. Once that deadline passes, the temporary price pressure should reverse.

The design follows *Dash for Cash: Monthly Market Impact of Institutional
Liquidity Needs* (Review of Financial Studies, 2020). Its U.S. sample ends in
2013, so 2014 onward is treated as the untouched external holdout.

## Primary rule

- Instrument: SPY.
- Direction: long.
- Month anchor `T`: the final SPY trading session of the calendar month.
- Entry: market-on-close on the settlement-dependent liquidity deadline:
  - T+3 era, through 2017-09-01: `T-4`.
  - T+2 era, 2017-09-05 through 2024-05-24: `T-3`.
  - T+1 era, from 2024-05-28: `T-2`.
- Exit: market-on-close three SPY sessions after entry.
- No price, trend, volatility, weekday, or macro filters.
- No stop or target; the three-session time exit is the risk boundary.
- Trading friction: 4 basis points round trip, deducted once per event.
- Returns use adjusted SPY bars because all levels are recomputed from the same
  series; there is no frozen absolute dollar level.

## Frozen evaluation

Primary outcomes:

1. Net event return and one-sample t-statistic for the full sample.
2. Same statistics for the paper-era sample through 2013 and the untouched
   2014+ holdout.
3. Win rate, payoff ratio, profit factor, worst event, maximum drawdown of the
   event-only equity curve, and exposure-adjusted annualized return.

Mechanism tests (validation, not entry filters):

1. Friday month-end events should have larger average returns than other
   month-ends because monthly pension payments and weekly payroll coincide.
2. The five-session return ending on the entry date should be negatively
   related to the next three-session reversal return.
3. The entry offset should move one session closer to month-end after each U.S.
   settlement-cycle change. Static T-4 and T-3 alternatives are controls.

Robustness tests (not selection candidates):

- One- through five-session holding periods around the frozen three-session
  exit.
- Gross and net results at 0, 4, 10, and 20 bps round-trip friction.
- Leave-one-year-out means and a month-level bootstrap confidence interval.
- Calendar-year and settlement-regime breakdowns.
- Unconditional SPY returns over all other non-overlapping three-session
  windows as a baseline.

## Decision rule

The strategy graduates only if the 2014+ holdout is positive after 4 bps,
the effect is not confined to one year or one settlement regime, and at least
two of the three mechanism tests point in the predicted direction. Otherwise it
is documented as rejected, not repaired with a newly discovered filter.

