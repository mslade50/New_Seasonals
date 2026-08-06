"""Drill-down on the three surviving cells from event_seasonality_sweep:

1. September post-quad-witching weakness (post-opex week, extended to
   month-end) — per-year, cross-ticker, vs rest-of-September baseline.
2. December post-opex strength (Santa anchored at Dec opex -> year end).
3. Pre-FOMC drift ex-midterm-years — per-year totals, drop-best-year floor.

Run: python scratch/event_sweep_drilldown.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from macro_calendar import event_dates  # noqa: E402

TICKERS = ["SPY", "QQQ", "IWM", "TLT"]


def load(tkr: str) -> pd.DataFrame:
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         columns=["ticker", "date", "Open", "Close"])
    df = (mp[mp["ticker"] == tkr].set_index("date").sort_index()
          [["Open", "Close"]])
    df.index = pd.to_datetime(df.index).normalize()
    df = df[~df.index.duplicated(keep="last")]
    return df[df.index >= "2000-01-01"]


def win(df: pd.DataFrame, anchor: pd.Timestamp, a: int, b: int) -> float:
    idx = df.index
    p = idx.searchsorted(anchor)
    lo, hi = p + a, p + b
    if lo - 1 < 0 or hi >= len(idx) or p >= len(idx):
        return np.nan
    return float(df["Close"].iloc[hi] / df["Close"].iloc[lo - 1] - 1)


def to_month_end(df: pd.DataFrame, anchor: pd.Timestamp) -> float:
    idx = df.index
    p = idx.searchsorted(anchor)
    if p >= len(idx):
        return np.nan
    me = idx.searchsorted(pd.Timestamp(anchor.year, anchor.month, 28)
                          + pd.Timedelta(days=4), side="left") - 1
    if me <= p or me >= len(idx):
        return np.nan
    return float(df["Close"].iloc[me] / df["Close"].iloc[p] - 1)


def stats(x: pd.Series, label: str) -> str:
    x = x.dropna()
    if len(x) < 3:
        return f"{label:34s} N={len(x)}"
    t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    return (f"{label:34s} mean {x.mean()*1e4:+7.1f} bps  t {t:+5.2f}  "
            f"N {len(x):3d}  hit {(x>0).mean():.2f}  "
            f"worst {x.min()*1e4:+7.1f}  best {x.max()*1e4:+7.1f}")


px = {t: load(t) for t in TICKERS}
spy = px["SPY"]
opex = event_dates("opex")
opex = opex[(opex >= spy.index.min()) & (opex <= spy.index.max())]

print("=" * 90)
print("1. SEPTEMBER POST-QUAD-WITCHING")
print("=" * 90)
sep = [d for d in opex if d.month == 9]
for tkr in TICKERS:
    w = pd.Series([win(px[tkr], d, 1, 5) for d in sep], index=sep)
    print(stats(w, f"{tkr} post-opex wk (+1..+5)"))
for tkr in TICKERS:
    w = pd.Series([to_month_end(px[tkr], d) for d in sep], index=sep)
    print(stats(w, f"{tkr} opex close -> Sep month-end"))

print("\nSPY per-year (opex close -> month-end):")
w = pd.Series([to_month_end(spy, d) for d in sep], index=sep).dropna()
for d, v in w.items():
    mid = " MIDTERM" if d.year % 4 == 2 else ""
    print(f"  {d.year}: {v*1e4:+8.1f} bps{mid}")
print(stats(w, "all"))
print(stats(w[[d.year % 4 == 2 for d in w.index]], "midterm only"))
print(stats(w[[d.year % 4 != 2 for d in w.index]], "ex-midterm"))
# drop best + drop worst
s = w.sort_values()
print(stats(s.iloc[1:], "drop worst year"))
print(stats(s.iloc[:-1], "drop best year"))

# baseline: same length window starting at Sep 1
sep1 = [pd.Timestamp(y, 9, 1) for y in range(2000, 2026)]
base = pd.Series([win(spy, pd.Timestamp(d), 1, 5) for d in sep1])
print(stats(base, "baseline: first wk of Sep (+1..+5)"))

print()
print("=" * 90)
print("2. DECEMBER POST-OPEX -> YEAR-END (Santa anchored at opex)")
print("=" * 90)
dec = [d for d in opex if d.month == 12]
for tkr in TICKERS:
    w = pd.Series([win(px[tkr], d, 1, 5) for d in dec], index=dec)
    print(stats(w, f"{tkr} post-opex wk (+1..+5)"))
for tkr in TICKERS:
    w = pd.Series([to_month_end(px[tkr], d) for d in dec], index=dec)
    print(stats(w, f"{tkr} opex close -> year-end"))
print("\nSPY per-year (opex close -> year-end):")
w = pd.Series([to_month_end(spy, d) for d in dec], index=dec).dropna()
for d, v in w.items():
    print(f"  {d.year}: {v*1e4:+8.1f} bps")
s = w.sort_values()
print(stats(s.iloc[:-1], "drop best year"))
print(stats(s.iloc[1:], "drop worst year"))

print()
print("=" * 90)
print("3. PRE-FOMC DRIFT EX-MIDTERM (-3..0 close-to-close)")
print("=" * 90)
fomc = event_dates("fomc_decision")
fomc = fomc[(fomc >= spy.index.min()) & (fomc <= spy.index.max())]
recs = pd.Series([win(spy, d, -3, 0) for d in fomc], index=fomc).dropna()
ex = recs[[d.year % 4 != 2 for d in recs.index]]
mid = recs[[d.year % 4 == 2 for d in recs.index]]
print(stats(recs, "all meetings"))
print(stats(ex, "ex-midterm"))
print(stats(mid, "midterm only"))
print("\nper-year sum (bps), ex-midterm flagged:")
yr = recs.groupby(recs.index.year).sum() * 1e4
pos = 0
for y, v in yr.items():
    tag = " MIDTERM" if y % 4 == 2 else ""
    print(f"  {y}: {v:+8.1f}{tag}")
ex_yr = yr[[y % 4 != 2 for y in yr.index]]
print(f"\nex-midterm yearly: mean {ex_yr.mean():+.1f} bps, "
      f"{(ex_yr>0).mean():.0%} positive years, "
      f"worst {ex_yr.min():+.1f} ({ex_yr.idxmin()}), "
      f"drop-best-year mean {ex_yr.drop(ex_yr.idxmax()).mean():+.1f}")
print("\nQQQ / IWM confirm (ex-midterm):")
for tkr in ("QQQ", "IWM"):
    r = pd.Series([win(px[tkr], d, -3, 0) for d in fomc], index=fomc).dropna()
    print(stats(r[[d.year % 4 != 2 for d in r.index]], f"{tkr} ex-midterm"))
