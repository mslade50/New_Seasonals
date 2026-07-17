"""Heston-Sadka same-calendar-month momentum on the house universe (2026-07-17).

The claim (Heston & Sadka 2008): a stock's return in a given calendar month
positively predicts its return in the SAME month of later years, out to 20
annual lags — seasonality is cross-sectionally persistent per stock, which
is different structure from the (mostly arbitraged) factor-level January
effects.

Test here, on data/master_prices.parquet (adjusted, ~2000 tickers, 2000+):
- signal for stock i, month t = mean of i's returns in the same calendar
  month at lags 1..10 years (>= 5 observations required)
- eligibility: prior-month close >= $5, prior-month median dollar volume
  >= $2M
- rank cross-sectionally each month (>= 100 eligible names), quintiles
- report Q5-Q1 (long-short) and Q5 minus universe (long-only tilt),
  full period + era halves + ex-January

Known bias, stated up front: the cache is today's universe, so this is
survivorship-flattered like every study on it (~21 of 22 major 2020s
delistings absent). Treat magnitudes as upper bounds; structure/monotonic
quintile ordering is the more robust read.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tstat(x: pd.Series) -> float:
    x = x.dropna()
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 else np.nan


def report(label: str, s: pd.Series):
    x = s.dropna()
    print(f"  {label:<28} {x.mean():+.2f}%/mo  t={tstat(x):+.2f}  "
          f"hit {100*(x>0).mean():.0f}%  n={len(x)}")


def main():
    mp = pd.read_parquet(os.path.join(ROOT, "data", "master_prices.parquet"),
                         columns=["ticker", "date", "Close", "Volume"])
    mp["date"] = pd.to_datetime(mp["date"])
    px = mp.pivot_table(index="date", columns="ticker", values="Close")
    dv = mp.pivot_table(index="date", columns="ticker", values="Volume")
    dollar = px * dv

    close_m = px.resample("ME").last()
    ret_m = close_m.pct_change() * 100.0
    dollar_m = dollar.resample("ME").median()

    months = ret_m.index
    q5q1, q5uni, dates = [], [], []
    for i, t in enumerate(months):
        if i < 130:               # need ~10y of lags + warmup
            continue
        # same-calendar-month lags 1..10y: rows at t-12, t-24, ... t-120
        lag_rows = [ret_m.iloc[i - 12 * k] for k in range(1, 11) if i - 12 * k >= 0]
        lag_df = pd.DataFrame(lag_rows)
        obs = lag_df.notna().sum()
        sig = lag_df.mean()
        sig[obs < 5] = np.nan

        elig = (close_m.iloc[i - 1] >= 5.0) & (dollar_m.iloc[i - 1] >= 2e6) & sig.notna()
        fwd = ret_m.iloc[i]
        elig &= fwd.notna()
        if elig.sum() < 100:
            continue
        s = sig[elig]
        f = fwd[elig]
        q = pd.qcut(s.rank(method="first"), 5, labels=False)
        q5 = f[q == 4].mean()
        q1 = f[q == 0].mean()
        q5q1.append(q5 - q1)
        q5uni.append(q5 - f.mean())
        dates.append(t)

    ls = pd.Series(q5q1, index=pd.DatetimeIndex(dates))
    lo = pd.Series(q5uni, index=pd.DatetimeIndex(dates))
    print(f"Heston-Sadka same-month momentum — {ls.index.min().date()} -> "
          f"{ls.index.max().date()} ({len(ls)} months)")
    print("\nQ5 - Q1 (long-short quintile spread):")
    report("full period", ls)
    report("first half", ls[ls.index < ls.index[len(ls) // 2]])
    report("second half", ls[ls.index >= ls.index[len(ls) // 2]])
    report("ex-January", ls[ls.index.month != 1])
    report("post-2018", ls[ls.index.year >= 2018])
    print("\nQ5 - universe (long-only tilt):")
    report("full period", lo)
    report("second half", lo[lo.index >= lo.index[len(lo) // 2]])
    report("post-2018", lo[lo.index.year >= 2018])

    # quintile monotonicity — the structural read that survives survivorship
    print("\nPer-quintile mean fwd return (full period):")
    qmeans = {k: [] for k in range(5)}
    for i, t in enumerate(months):
        if i < 130:
            continue
        lag_rows = [ret_m.iloc[i - 12 * k] for k in range(1, 11) if i - 12 * k >= 0]
        lag_df = pd.DataFrame(lag_rows)
        obs = lag_df.notna().sum()
        sig = lag_df.mean()
        sig[obs < 5] = np.nan
        elig = (close_m.iloc[i - 1] >= 5.0) & (dollar_m.iloc[i - 1] >= 2e6) & sig.notna()
        fwd = ret_m.iloc[i]
        elig &= fwd.notna()
        if elig.sum() < 100:
            continue
        s, f = sig[elig], fwd[elig]
        q = pd.qcut(s.rank(method="first"), 5, labels=False)
        for k in range(5):
            qmeans[k].append(f[q == k].mean())
    for k in range(5):
        print(f"  Q{k+1}: {np.mean(qmeans[k]):+.2f}%/mo")
    print("\nCAVEAT: survivorship-flattered universe; magnitudes are upper bounds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
