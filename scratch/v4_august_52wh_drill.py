"""V4 POSTOPEX_VOL drill (2026-08-21, McKinley's two questions on the first
live window): (1) is the AUGUST opex window itself positive EV despite the
Aug 2015 -21.5% tail, and (2) how does the trade look when entered with SPY
at/near a 52-week high?

Same basis as the prereg grid (scratch/svxy_postevent_grid.py): synthetic
-0.5x short-vol legs from UVXY OHLC, compounded, 2011-10+. Window = opex
close -> close +3 sessions, ex-September (the live spec). Conditioning uses
LAG-1 SPY (the session before opex), matching the sleeve's staging
convention. Small cells get an exact binomial sign test (house doctrine),
not just a t-stat.
"""
from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from macro_calendar import event_dates  # noqa: E402


def load(tkr: str, cols=("Open", "Close")) -> pd.DataFrame:
    mp = pd.read_parquet(ROOT / "data" / "master_prices.parquet",
                         columns=["ticker", "date", *cols])
    df = mp[mp["ticker"] == tkr].set_index("date").sort_index()
    df.index = pd.to_datetime(df.index).normalize()
    return df[~df.index.duplicated(keep="last")][list(cols)]


u = load("UVXY")
ovn = -(u["Open"] / u["Close"].shift(1) - 1) / 3.0
intra = -(u["Close"] / u["Open"] - 1) / 3.0
idx = u.index

spy = load("SPY", cols=("Close",))["Close"]
spy_hi252 = spy.rolling(252).max()
dist_52wh = (spy / spy_hi252 - 1) * 100.0   # 0 = at the 52w high (adj close)


def window_ret(p: int, exit_k: int = 3) -> float:
    hi = p + exit_k
    if hi >= len(idx) or p < 1:
        return np.nan
    total = 1.0
    for i in range(p + 1, hi + 1):
        total *= (1 + ovn.iloc[i]) * (1 + intra.iloc[i])
    return total - 1


def sign_p(wins: int, n: int) -> float:
    """Exact one-sided binomial P(>= wins | p=0.5)."""
    return sum(comb(n, k) for k in range(wins, n + 1)) / 2 ** n


def cell(name: str, sub: pd.DataFrame) -> None:
    x = sub["ret"].dropna()
    if not len(x):
        print(f"{name:38s}  n=0")
        return
    n, mean, hit = len(x), x.mean() * 1e4, (x > 0).mean()
    t = (x.mean() / (x.std(ddof=1) / np.sqrt(n))) if n > 2 else np.nan
    p = sign_p(int((x > 0).sum()), n)
    print(f"{name:38s}  n={n:3d}  avg {mean:+7.1f} bps  t {t:5.2f}  "
          f"hit {hit:.0%}  sign-p {p:.3f}  worst {x.min()*100:+.1f}%")


rows = []
for d in event_dates("opex"):
    d = pd.Timestamp(d).normalize()
    if d.month == 9 or d not in idx.to_series().index:
        continue
    p = idx.get_loc(d)
    lag1 = spy.index[spy.index < d]
    dist = dist_52wh.reindex(lag1).iloc[-1] if len(lag1) else np.nan
    rows.append({"date": d, "month": d.month, "ret": window_ret(p),
                 "dist": dist})
df = pd.DataFrame(rows).dropna(subset=["ret"])

print(f"V4 windows (opex close -> +3 close, ex-Sep, synthetic -0.5x): "
      f"{len(df)}  span {df['date'].min().date()} -> {df['date'].max().date()}")
print()
cell("ALL windows", df)
cell("August only", df[df["month"] == 8])
print("\nAugust year by year:")
for _, r in df[df["month"] == 8].iterrows():
    print(f"  {r['date'].date()}  {r['ret']*100:+7.1f}%   "
          f"(SPY {r['dist']:+.1f}% vs 52w high at entry)")

print("\nBy SPY distance from 252d high (lag-1) at entry:")
cell("  at high: dist > -1%", df[df["dist"] > -1])
cell("  near: -3% < dist <= -1%", df[(df["dist"] > -3) & (df["dist"] <= -1)])
cell("  mid: -10% < dist <= -3%", df[(df["dist"] > -10) & (df["dist"] <= -3)])
cell("  deep: dist <= -10%", df[df["dist"] <= -10])
print()
cell("within 2% of high (today's state)", df[df["dist"] > -2])
cell("more than 2% off the high", df[df["dist"] <= -2])
cell("Aug AND within 2% of high", df[(df["month"] == 8) & (df["dist"] > -2)])
cell("2018+ (real -0.5x era) within 2%", df[(df["dist"] > -2) &
                                            (df["date"] >= "2018-03-01")])

last = dist_52wh.dropna()
print(f"\nToday's condition: SPY {last.iloc[-1]:+.2f}% vs its 252d high "
      f"as of {last.index[-1].date()} (lag-1 for the 2026-08-21 entry)")
