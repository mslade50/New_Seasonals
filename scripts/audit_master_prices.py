"""Audit data/master_prices.parquet — surfaces stale tickers, short histories,
and tickers with mid-range gaps. Useful before running a backtest to know the
data state both pages will see.

Usage:
    python scripts/audit_master_prices.py
"""
import os
import sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "data", "master_prices.parquet")


def main():
    if not os.path.exists(PATH):
        print(f"ERROR: {PATH} missing — run scripts/build_master_prices.py first")
        return 1

    df = pd.read_parquet(PATH)
    today = pd.Timestamp.today().normalize()

    summary = df.groupby("ticker").agg(
        first=("date", "min"),
        last=("date", "max"),
        bars=("date", "size"),
    )
    summary["gap_days"] = (today - summary["last"]).dt.days
    summary["span_days"] = (summary["last"] - summary["first"]).dt.days
    summary["expected_bars"] = (summary["span_days"] / 365.25 * 252).round().astype(int).clip(lower=1)
    summary["coverage"] = (summary["bars"] / summary["expected_bars"]).clip(upper=1.0)

    print(f"=== master_prices.parquet ===")
    print(f"  rows:    {len(df):,}")
    print(f"  tickers: {df['ticker'].nunique()}")
    print(f"  range:   {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"  file:    {os.path.getsize(PATH)/1e6:.1f} MB\n")

    stale = summary[summary["gap_days"] > 7].sort_values("gap_days", ascending=False)
    print(f"=== Stale ({len(stale)} tickers, gap > 7 days from today) ===")
    if len(stale):
        print(stale[["last", "gap_days", "bars"]].head(40))
    print()

    short = summary[summary["bars"] < 1000].sort_values("bars")
    print(f"=== Short history ({len(short)} tickers, < 1000 bars / ~4 yrs) ===")
    if len(short):
        print(short[["first", "last", "bars"]].head(40))
    print()

    bad_cov = summary[(summary["coverage"] < 0.90) & (summary["span_days"] > 365)].sort_values("coverage")
    print(f"=== Mid-range gaps ({len(bad_cov)} tickers, coverage < 90% over 1+ yrs) ===")
    if len(bad_cov):
        print(bad_cov[["first", "last", "bars", "expected_bars", "coverage"]].head(40))
    print()

    print(f"=== Healthy: {(summary['gap_days'] <= 7).sum()} tickers fresh, "
          f"{(summary['coverage'] >= 0.95).sum()} with >=95% coverage ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
