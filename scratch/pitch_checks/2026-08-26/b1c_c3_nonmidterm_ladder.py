"""b1c / C3: KILL or NEAR-MISS?

The midterm cell is wrong-signed on both vehicles and today is midterm, so
C3 cannot trade this morning either way. The remaining question is whether
the NON-midterm cell is a parked candidate (registry precedent: "Re-check
this cell in a non-midterm year; it is parked, not dead") or whether the
whole thing is a flat surface with no dose response in any regime.

Decision rule, fixed before the run: park it only if the non-midterm cell
shows a MONOTONE dose response in the rank (tighter rung pays more) AND the
tightest rung clears 5x the DX round trip. Otherwise the midterm split is
just where a flat surface happened to land and the cell is dead.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

pd.set_option("display.width", 220)

px = close_panel(["DX-Y.NYB", "UUP"]).dropna(subset=["DX-Y.NYB"])
rk = pct_rank(px["DX-Y.NYB"], 21, 252)

for h in (3, 5, 10):
    ret = vehicle_ret(px, [("DX-Y.NYB", 1.0)], h, 1)
    valid = ret.dropna().index
    print(f"\n=== NON-MIDTERM dose response, LONG DX h={h} (episodes) ===")
    rows = []
    for k in (1, 2, 5, 10, 20, 50):
        tt = px.index[(rk <= k).values].intersection(valid)
        epi = declusters(tt, h, valid)
        yrs = pd.DatetimeIndex(epi).year
        nm = epi[(yrs % 4) != 2]
        d = summarize(ret.loc[nm].values, f"rank21<={k}")
        d["bps"] = round(d["mean_pct"] * 100, 1)
        d["x_cost_1.5bp"] = round(d["mean_pct"] * 100 / 1.5, 1)
        rows.append(d)
    base = summarize(ret.loc[valid[(pd.DatetimeIndex(valid).year % 4) != 2]].values,
                     "ALL non-midterm days")
    base["bps"] = round(base["mean_pct"] * 100, 1)
    rows.append(base)
    show(rows)

# stability of the non-midterm cell
print("\n=== non-midterm rank<=2 cell, h=5: is it one year? ===")
ret = vehicle_ret(px, [("DX-Y.NYB", 1.0)], 5, 1)
valid = ret.dropna().index
tt = px.index[(rk <= 2).values].intersection(valid)
epi = declusters(tt, 5, valid)
yrs = pd.DatetimeIndex(epi).year
nm = epi[(yrs % 4) != 2]
v = ret.loc[nm].values
print(" ", cluster_note(nm, v, k=3))
order = np.argsort(-v)
for k in (1, 2, 3):
    keep = np.ones(len(v), bool); keep[order[:k]] = False
    print(f"  drop-best-{k}: {100*v[keep].mean():+.3f}% ({100*v[keep].mean()*100/1.5:.1f}x cost), N={keep.sum()}")
byy = pd.Series(v).groupby(pd.DatetimeIndex(nm).year.values).agg(["size", "mean"])
byy["mean"] = (byy["mean"] * 100).round(3)
print(byy.to_string())
print(f"  years positive: {int((byy['mean']>0).sum())} of {len(byy)}")
