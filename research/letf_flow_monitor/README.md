# Leveraged ETF Flow Monitor

This standalone research package tracks official ProShares creations and
redemptions, estimates leveraged-fund mechanical demand, and tests whether
those pressures precede volatility expansion, late-day continuation, or
market turns.

Run from the repository root:

```powershell
python -m research.letf_flow_monitor.monitor --refresh
```

Use `--refresh` for the first run or whenever fresh official and R2 data are
wanted. Omit it to reuse the local raw cache. Outputs:

- `outputs/REPORT.md` — current snapshot and hypothesis results
- `outputs/latest_snapshot.csv` — machine-readable latest complex readings
- `outputs/latest_fund_flows.csv` — latest creations/redemptions by fund
- `outputs/event_studies.csv` — event-study statistics
- `outputs/pressure_robustness.csv` — benchmark and subperiod stability checks
- `outputs/monitor.json` — combined payload for a future dashboard
- `data/*.parquet` — normalized fund, daily, and intraday research tables

The package does not alter `daily_scan.py`, the strategy book, or live order
staging. See `SPEC.md` for the frozen definitions and timing rules.
