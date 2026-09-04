"""C1 round 2d -- the arbitration.

a2b established the month-of-year profile on this cell PERSISTS split-half
(rank corr +0.71, permutation P(sim>=obs)=0.025) and that August is its NULL
month (+0.020% parent, 50.0% hit, N=24), coherent across IEF/LQD and inverted
in ^TNX. a2c showed the live cell's 4 August obs are 0-for-4.

So: is the CPI-on-eve gate ORTHOGONAL to the month shape (in which case August
only applies the parent-level drag and the live cell keeps most of its edge), or
does the gate's edge itself disappear once each observation is charged for the
month it landed in? Leave-one-out month adjustment, so no observation is graded
against itself. LOYO on the live cell as the companion.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["TLT"])
tl = px["TLT"].dropna()
idx = tl.index
c = tl.values
N = len(c)
d1 = np.full(N, np.nan)
d1[1:] = c[1:] / c[:-1] - 1.0
base_hit = float((d1[~np.isnan(d1)] > 0).mean())

ev = load_events()
sp = lambda k: sorted({int(idx.searchsorted(x, "left"))
                       for x in ev[ev.event == k]["date"]
                       if 0 <= int(idx.searchsorted(x, "left")) < N})
ppi_l = [p for p in sp("ppi") if 1 <= p < N and not np.isnan(d1[p])]
cpi_all = set(sp("cpi"))
v = np.array([d1[p] for p in ppi_l])
dt = pd.DatetimeIndex([idx[p] for p in ppi_l])
mo, yr = dt.month.values, dt.year.values
L = np.array([(p - 1) in cpi_all for p in ppi_l])

print("=" * 100)
print("1. LEAVE-ONE-OUT MONTH-ADJUSTED EXCESS")
print("   each observation minus the mean of its own month computed WITHOUT it")
print("=" * 100)
adj = np.full(len(v), np.nan)
for i in range(len(v)):
    peers = (mo == mo[i]) & (np.arange(len(v)) != i)
    adj[i] = v[i] - v[peers].mean()


def rep(x, lbl):
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"cell": lbl, "N": 0}
    w = int((x > 0).sum())
    sd = x.std(ddof=1) if len(x) > 1 else np.nan
    return {"cell": lbl, "N": len(x), "mean_bps": round(100 * 100 * x.mean(), 2),
            "hit": round(100 * w / len(x), 1),
            "t": round(x.mean() / (sd / np.sqrt(len(x))), 2),
            "signp": round(sign_test(w, len(x), base_hit), 4)}


print(pd.DataFrame([
    rep(v, "raw: parent"),
    rep(adj, "month-adj: parent (0 by construction)"),
    rep(v[L], "raw: LIVE CELL"),
    rep(adj[L], "*** month-adj: LIVE CELL ***"),
    rep(v[~L], "raw: not live"),
    rep(adj[~L], "month-adj: not live"),
]).to_string(index=False))
a, b = adj[L], adj[~L]
se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
print(f"\n  month-adjusted gate lift = {100*(a.mean()-b.mean()):+.4f}pp  "
      f"welch t {(a.mean()-b.mean())/se:+.2f}")
print(f"  bootstrap P(mean<=0) on the month-adjusted live cell = "
      f"{bootstrap_p_le0(a):.3f}")
print("\n  -> if the gate SURVIVES month adjustment, the August drag is the")
print("     parent's (~-10 bps) and not the gate's; expected value today is")
print("     roughly  parent-August (+2 bps)  +  the gate's month-adj lift.")

print("\n" + "=" * 100)
print("2. THE RESULTING HONEST POINT ESTIMATE FOR TONIGHT")
print("=" * 100)
aug_parent = v[mo == 8].mean()
gate_adj = adj[L].mean()
print(f"  parent August mean               {100*100*aug_parent:+7.2f} bps  (N=24)")
print(f"  month-adjusted gate lift         {100*100*gate_adj:+7.2f} bps  "
      f"(N={int(L.sum())})")
print(f"  additive estimate for tonight    "
      f"{100*100*(aug_parent+gate_adj):+7.2f} bps  vs 2.5 bps round trip = "
      f"{(aug_parent+gate_adj)*10000/2.5:+.2f}x")
print(f"  headline (unconditional) estimate {100*100*v[L].mean():+7.2f} bps  = "
      f"{v[L].mean()*10000/2.5:+.2f}x")
print(f"  realised August live-cell        {100*100*v[L & (mo==8)].mean():+7.2f} bps"
      f"  (N=4, 0-for-4)")

print("\n" + "=" * 100)
print("3. LOYO on the live cell (drop each year, recompute the mean)")
print("=" * 100)
ys = sorted(set(yr[L]))
mins, out = None, []
for y in ys:
    s = v[L & (yr != y)]
    out.append((y, 100 * s.mean(), 100 * (s > 0).mean()))
df = pd.DataFrame(out, columns=["dropped_year", "mean_pct", "hit"]).round(4)
print(df.to_string(index=False))
print(f"\n  LOYO floor {df['mean_pct'].min():+.4f}% (dropping "
      f"{int(df.loc[df['mean_pct'].idxmin(), 'dropped_year'])}), "
      f"ceiling {df['mean_pct'].max():+.4f}%")
print(f"  every LOYO fold positive: {bool((df['mean_pct'] > 0).all())}")

print("\n" + "=" * 100)
print("4. SAME LOYO ON THE MONTH-ADJUSTED LIVE CELL")
print("=" * 100)
out = []
for y in ys:
    s = adj[L & (yr != y)]
    out.append((y, 100 * s.mean(), 100 * (s > 0).mean()))
df2 = pd.DataFrame(out, columns=["dropped_year", "madj_pct", "hit"]).round(4)
print(df2.to_string(index=False))
print(f"  LOYO floor {df2['madj_pct'].min():+.4f}%  all folds positive: "
      f"{bool((df2['madj_pct'] > 0).all())}")
