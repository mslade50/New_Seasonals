"""The Aug-11 seasonal for SPY: 18-8 up in the sweep. Does it survive anything?

The engine's seasonal_doy cell takes one anchor per prior year at the matching
trading day of year (+/- 2) and reports h1. n=26, mean +0.525%, median +0.205%,
18-8 up, sign p 0.0378. Controls owed: August drift, the neighbouring days, era
stability, concentration, and what the midterm subset does.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, fwd_ret, summarize, era_split, sign_test, cluster_note,
)

px = close_panel(["SPY", "^GSPC", "QQQ", "IWM"])
px = px[px.index >= "1999-01-01"]
dates = px.index


def one_per_year(sel):
    out = []
    for _, g in pd.Series(0, index=sel).groupby(sel.year):
        out.append(g.index[len(g) // 2])
    return pd.DatetimeIndex(out)


def show(sub, idx, label, h=1):
    f = fwd_ret(px[sub].dropna(), h)
    v = f.reindex(pd.DatetimeIndex(sorted(set(idx)))).dropna()
    if len(v) < 3:
        print(f"   {label:56s} n={len(v)} too few")
        return None
    r = summarize(v.values, "")
    up = int((v.values > 0).sum())
    print(f"   {label:56s} n={r['n']:4d} mean {r['mean_pct']:+.3f}% "
          f"med {r['median_pct']:+.3f}% hit {r['hit']:.1f}% t={r['t']:+.2f} | "
          f"{up}-{len(v)-up} up-p {sign_test(up, len(v)):.4f}")
    return v


aug11 = dates[(dates.month == 8) & (dates.day >= 9) & (dates.day <= 13)]
anch = one_per_year(aug11)
print(f"Aug-11 anchors, one per year: {len(anch)}  {sorted(anch.year)}")
print(f"anchor dates: {[str(d.date()) for d in anch]}")

print("\n--- the cell, h1 ---")
v = show("SPY", anch, "SPY, Aug-11 anchor (one per year)")
print(f"      {cluster_note(v.index, v.values)}")
for e in era_split(v.index, v.values):
    if e.get("n", 0):
        print(f"      era n={e['n']:3d} mean {e['mean_pct']:+.3f}% hit {e['hit']:.1f}% "
              f"t={e['t']:+.2f}")
print(f"      per-year: {[(str(d.year), round(100*x, 2)) for d, x in v.items()]}")
for sub in ("^GSPC", "QQQ", "IWM"):
    show(sub, anch, f"{sub}, same anchors")

print("\n--- controls ---")
show("SPY", dates, "SPY, ALL sessions")
show("SPY", dates[dates.month == 8], "SPY, all August sessions")
show("SPY", aug11, "SPY, every Aug 9-13 session (not one per year)")
for lo in (2, 5, 16, 19, 23):
    sel = dates[(dates.month == 8) & (dates.day >= lo) & (dates.day <= lo + 4)]
    show("SPY", one_per_year(sel), f"SPY, Aug {lo}-{lo+4} anchor (one per year)")
for m in (7, 9):
    sel = dates[(dates.month == m) & (dates.day >= 9) & (dates.day <= 13)]
    show("SPY", one_per_year(sel), f"SPY, {m:02d}/09-13 anchor (one per year)")

print("\n--- cycle split ---")
for lab, mod in (("midterm", 2), ("pre-election", 3), ("election", 0), ("post-election", 1)):
    show("SPY", anch[anch.year % 4 == mod], f"SPY, Aug-11 anchor, {lab} years")

print("\n--- and the same cell at longer horizons ---")
for h in (2, 3, 5, 10):
    show("SPY", anch, f"SPY, Aug-11 anchor, h{h}", h=h)
