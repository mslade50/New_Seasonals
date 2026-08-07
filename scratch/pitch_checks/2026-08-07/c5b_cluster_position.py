"""C5 follow-up: first-of-cluster (+0.18%, t=0.45) and cluster-MEAN (+1.08%, t=3.59)
disagree, and day-level is NEGATIVE (-0.25%, t=-1.67). Resolve it:
  (a) where in its cluster does 2026-08-06 sit? that fixes the reference class.
  (b) is the cluster-mean edge era-stable, or an equal-weighting artifact?
  (c) return by day-position inside the cluster.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _study import *  # noqa

import numpy as np
import pandas as pd

H, LAG = 5, 1
P = close_panel(["GLD", "GDX"]).dropna()
sp = (P["GDX"].pct_change(21) - P["GLD"].pct_change(21)) * 100
fw = pair_fwd(P, "GLD", "GDX", H, LAG)
valid = fw.dropna().index
trig = sp.index[(sp >= 8.0).fillna(False)]
ASOF = P.index[-1]

# (a) current cluster position -- count consecutive prior trigger days
run = 0
i = P.index.get_loc(ASOF)
while i - run >= 0 and sp.iloc[i - run] >= 8.0:
    run += 1
print(f"(a) TODAY {ASOF.date()} spread={sp.loc[ASOF]:+.2f}pp")
print(f"    consecutive trigger days ending today = {run}")
print(f"    last 12 spread values: {[round(x,2) for x in sp.iloc[i-11:i+1].values]}")
print(f"    -> today is day {run} of its cluster; reference class = "
      f"{'FIRST-of-cluster' if run == 1 else f'day-{run} of cluster'}")

pos = pd.Series(range(len(P.index)), index=P.index)
cl, cur, last = [], [], None
for d in sorted(pd.DatetimeIndex(trig)):
    p = pos[d]
    if last is not None and p - last >= H:
        cl.append(cur); cur = []
    cur.append(d); last = p
if cur:
    cl.append(cur)
sizes = np.array([len(c) for c in cl])
print(f"\n(b) {len(cl)} clusters, sizes: median={np.median(sizes):.0f} mean={sizes.mean():.1f} "
      f"max={sizes.max()}; {100*(sizes==1).mean():.0f}% are single-day")

cm = np.array([np.nanmean(fw.reindex(c).values) for c in cl])
cd = pd.DatetimeIndex([c[0] for c in cl])
ok = ~np.isnan(cm)
cm, cd = cm[ok], cd[ok]
show(era_split(cd, cm), "(b) cluster-MEAN era split")
show([summarize(cm[sizes[ok] == 1], "single-day clusters"),
      summarize(cm[sizes[ok] > 1], "multi-day clusters")], "(b2) by cluster size")
order = np.argsort(cm)
print(f"    cluster-mean all={100*cm.mean():+.2f}% t={summarize(cm)['t']:.2f}; "
      f"drop-best={100*np.delete(cm, order[-1]).mean():+.2f}%; "
      f"drop-2-best={100*np.delete(cm, order[-2:]).mean():+.2f}%")
yrs = cd.year.values
print("    LOYO cluster-mean:", [(y, round(100*cm[yrs != y].mean(), 2)) for y in sorted(set(yrs))])

# (c) return by position inside cluster
rows = []
for k in (1, 2, 3, 4, 5):
    v = [fw.get(c[k-1], np.nan) for c in cl if len(c) >= k]
    rows.append(summarize(np.array(v, float), f"cluster day {k}"))
v = [fw.get(c[-1], np.nan) for c in cl if len(c) >= 6]
rows.append(summarize(np.array(v, float), "last day of 6+ clusters"))
show(rows, "(c) C5 pair return by position inside the trigger cluster")

# honest overlap-free alternative: weight each day by 1/cluster_size (day-level, decluttered)
w = np.concatenate([np.full(len(c), 1.0/len(c)) for c in cl])
dd = pd.DatetimeIndex([d for c in cl for d in c])
vv = fw.reindex(dd).values
m = ~np.isnan(vv)
print(f"\n(d) size-weighted day-level mean = {100*np.average(vv[m], weights=w[m]):+.3f}% "
      f"(equals cluster-mean by construction); plain day-level = {100*np.nanmean(vv):+.3f}%")
