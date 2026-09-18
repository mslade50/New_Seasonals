"""The three-way's five episodes sit at trading-day-of-month 2,2,5,3,4.
The registry's reusable finding is that TLT has a strong unconditional tdom
profile with NO event anywhere, so a short-duration cell whose whole support
sits early in the month owes that control. Re-derived from scratch here rather
than borrowed (registry rule on inherited numbers).

Also: where does TODAY's entry sit? Compute the live tdom exactly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["DBC", "TLT", "IEF"])
raw = load_prices(["DBC", "TLT"])
IDX = px.index
d3 = pd.DatetimeIndex(IDX)
TDOM = pd.Series(range(len(d3)), index=d3).groupby([d3.year, d3.month]).rank(method="first").astype(int)

# live tdom: today 2026-09-08 is the ENTRY session (bars stop 2026-09-04)
sep26 = [d for d in d3 if d.year == 2026 and d.month == 9]
print("September 2026 sessions present in the cache:", [str(x.date()) for x in sep26])
print("  (cache stops 2026-09-04; 09-05..09-07 were the Labor Day closure, "
      "so the ENTRY session 2026-09-08 is the next one)")
n_sep_bars = len(sep26)
print(f"  sessions in Sept 2026 through the last bar: {n_sep_bars} "
      f"-> the 2026-09-08 entry is trading day {n_sep_bars + 1} of the month")

for name, legs in (("SHORT TLT", [("TLT", -1.0)]), ("SHORT IEF", [("IEF", -1.0)])):
    for H in (3, 8):
        ret = vehicle_ret(px, legs, H, 1)
        valid = ret.notna()
        rows = []
        for lo, hi in ((1, 2), (2, 6), (6, 9), (9, 13), (13, 17), (17, 24)):
            m = (TDOM >= lo) & (TDOM < hi)
            v = ret[valid & m.values].values
            rows.append(summarize(v, f"tdom [{lo},{hi})"))
        rows.append(summarize(ret[valid].values, "all tdom"))
        show(rows, f"{name} h={H}  UNCONDITIONAL drift by trading-day-of-month "
                   f"(no event anywhere)")
        a = ret[valid & ((TDOM >= 2) & (TDOM <= 5)).values].values
        b = ret[valid & (TDOM == n_sep_bars + 1).values].values
        print(f"  the cell's support tdom 2-5: {100*a.mean():+.3f}% (n={len(a)})   "
              f"vs the LIVE entry tdom {n_sep_bars+1}: {100*b.mean():+.3f}% (n={len(b)})"
              f"   => free tailwind at the support worth "
              f"{100*(a.mean()-b.mean()):+.3f}pp")

# tdom-matched control for the three-way itself
print("\n" + "=" * 78)
print("THREE-WAY vs its own tdom-matched control, h=8 short TLT")
print("=" * 78)
state = (1.0 - raw["DBC"]["Close"] /
         rolling_on_valid(raw["DBC"]["Close"], lambda x: x.rolling(252).max())).reindex(IDX) <= 0.0025
mom = (raw["TLT"]["Close"] /
       rolling_on_valid(raw["TLT"]["Close"], lambda x: x.rolling(252).min()) - 1.0).reindex(IDX) <= 0.02


def flag(kinds, h, lag=1, k=0):
    ev = load_events(list(kinds))["date"]
    pos, _ = anchor_positions(IDX, ev, offset=k)
    e = np.asarray(pd.DatetimeIndex([IDX[p] for p in pos]).values, dtype="datetime64[ns]")
    out = np.zeros(len(IDX), dtype=bool)
    for i in range(len(IDX)):
        if i + lag + h >= len(IDX):
            continue
        out[i] = bool(((e > np.datetime64(IDX[i + lag])) &
                       (e <= np.datetime64(IDX[i + lag + h]))).any())
    return pd.Series(out, index=IDX)


H = 8
ret = vehicle_ret(px, [("TLT", -1.0)], H, 1)
valid = ret.notna()
dd = IDX[(state & mom & flag(("ppi", "cpi"), H, 1)).values & valid.values]
e = declusters(dd, H, IDX)
tds = TDOM.loc[e]
parts, w = [], []
for t_, c in tds.value_counts().items():
    pool = ret[valid & (TDOM == t_).values]
    if len(pool):
        parts.append(pool.mean()); w.append(c)
ctrl = float(np.average(parts, weights=w))
v = ret.loc[e].values
print(f"three-way episodes n={len(v)}, tdom {list(tds.values)}, mean {100*v.mean():+.3f}%")
print(f"tdom-MATCHED control = {100*ctrl:+.3f}%  =>  excess {100*(v.mean()-ctrl):+.3f}pp")
print(f"all-days control     = {100*ret[valid].mean():+.3f}%  =>  excess "
      f"{100*(v.mean()-ret[valid].mean()):+.3f}pp")
print(f"\nlive entry tdom would be {n_sep_bars+1}; observed support "
      f"{int(tds.min())}-{int(tds.max())}")
