# TLT Month-End Benchmark-Demand Strategy

This standalone research sleeve buys TLT at the close five trading sessions
before calendar month-end and exits at the month-end close. It is grounded in
predictable, price-insensitive Treasury demand from insurers and benchmarked
fixed-income portfolios.

## Trade ticket

- Entry: buy TLT MOC on the sixth-last exchange session of the month (`T-5`).
- Exit: sell TLT MOC on the final exchange session (`T`).
- Hold: exactly five close-to-close sessions.
- Stop/target: none; month-end MOC is mandatory.
- Pilot size: 10% of NAV, capped so all TLT exposure across this and any trend
  sleeve remains at or below 20% of NAV.
- Expected frequency: 12 trades per year.

The 10% pilot is a risk choice, not an optimized backtest parameter. The worst
historical event was -4.13% at full notional, or about -41 bp of NAV at a 10%
allocation. That historical loss is not a hard bound.

## Files

- `SPEC.md` — frozen thesis, rule, tests, and graduation criteria.
- `backtest.py` — reproducible research using `data/master_prices.parquet`.
- `strategy.py` — isolated calendar, instruction, and pilot-sizing helpers.
- `test_strategy.py` — unit tests for dates, state transitions, and exposure cap.
- `RESULTS.md` — generated evidence and risk discussion.
- `events.csv`, `yearly.csv`, `robustness.csv`, `leave_one_year_out.csv`, and
  `summary.json` — generated audit trail.

## Reproduce

```powershell
python research/treasury_month_end/backtest.py
python -m pytest research/treasury_month_end/test_strategy.py -q
```

## Live-integration boundary

This package intentionally does not edit the production scanner, strategy book,
or order-staging code. A future integration must supply an authoritative NYSE
session calendar; the helper refuses to approximate exchange holidays. MOC is
part of the strategy specification, so converting it to next-open execution is
not equivalent. Existing TLT exposure must be passed to `pilot_order` so the
seasonal trade is an incremental delta rather than a duplicate full position.

