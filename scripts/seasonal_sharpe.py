"""Sharpe / Sortino of the segmented seasonal-ideas portfolio.

Takes a backtest parquet, dedups (one open per ticker+direction), EXCLUDES single-
stock shorts entirely (keeps macro long+short + stock long), and reports the
risk-adjusted profile.

Ratios are computed on the monthly R-stream (R booked at exit). Because every
trade risks the same 1R, the series is the strategy's PnL at constant per-trade
risk, so Sharpe/Sortino are invariant to the dollar risk fraction chosen. Sortino
uses a 0 target (downside deviation of negative months).
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = r"C:\Users\McKinley Slade\dev\New_Seasonals"
DEFAULT = os.path.join(ROOT, "data", "seasonal_ideas_backtest.parquet")


def dedup(d):
    d = d.sort_values(["ticker", "direction", "entry_date"])
    keep, last = [], {}
    for r in d.itertuples():
        k = (r.ticker, r.direction)
        if last.get(k) is None or r.entry_date > last[k]:
            keep.append(r.Index); last[k] = r.exit_date
    return d.loc[keep]


def ratios(R_series, periods):
    r = R_series.astype(float)
    mu, sd = r.mean(), r.std(ddof=1)
    downside = np.sqrt((np.minimum(r, 0.0) ** 2).mean())
    sharpe = (mu / sd) * np.sqrt(periods) if sd else np.nan
    sortino = (mu / downside) * np.sqrt(periods) if downside else np.nan
    return sharpe, sortino


def main(path=DEFAULT):
    df = pd.read_parquet(path)
    df["entry_date"] = pd.to_datetime(df["entry_date"]); df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["asset"] = np.where(df["channel"] == "detect_seasonal", "stock", "macro")
    df = dedup(df)
    book = df[~((df.asset == "stock") & (df.direction == "short"))].copy()  # exclude stock shorts
    print(f"trades after dedup: {len(df)} | segmented book (no stock shorts): {len(book)}")

    R = book["R"].astype(float)
    w, l = R[R > 0], R[R < 0]
    pf = w.sum() / abs(l.sum())
    eq = book.sort_values("exit_date")["R"].cumsum()
    maxdd = float((eq - eq.cummax()).min())
    print(f"\nPer-trade: N {len(book)} | Win% {100*(R>0).mean():.1f} | AvgR {R.mean():.3f} | "
          f"TotR {R.sum():.1f} | PF {pf:.2f} | maxDD {maxdd:.1f} R")

    # R booked at exit, on the trading calendar
    daily = book.groupby(book["exit_date"].dt.normalize())["R"].sum()
    full = pd.date_range(book["exit_date"].min().normalize(), book["exit_date"].max().normalize(), freq="B")
    daily = daily.reindex(full, fill_value=0.0)
    monthly = daily.resample("ME").sum()

    sh_m, so_m = ratios(monthly, 12)
    sh_d, so_d = ratios(daily, 252)
    print(f"\n=== RISK-ADJUSTED (R-stream, constant per-trade risk) ===")
    print(f"Monthly  Sharpe {sh_m:.2f} | Sortino {so_m:.2f}  "
          f"(n={len(monthly)} months, {100*(monthly>0).mean():.0f}% positive, "
          f"best {monthly.max():.1f}R / worst {monthly.min():.1f}R)")
    print(f"Daily    Sharpe {sh_d:.2f} | Sortino {so_d:.2f}  (lumpier — R books on exit day)")

    # by cycle for context
    print("\nMonthly Sharpe by cycle year:")
    for c in sorted(book["cycle"].unique()):
        bc = book[book.cycle == c]
        dc = bc.groupby(bc["exit_date"].dt.normalize())["R"].sum().reindex(full, fill_value=0.0).resample("ME").sum()
        s, so = ratios(dc, 12)
        print(f"  cycle {int(c)}: Sharpe {s:.2f} | Sortino {so:.2f} | TotR {bc['R'].sum():.0f} (N {len(bc)})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT)
