"""C8 robustness: the day-level (+4.4%) and first-of-cluster episode (-2.4%) means
disagree in SIGN. That can only happen if the winning days sit late inside clusters.
Test three episode conventions + per-episode listing. Also re-run C5/C6 the same way.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _study import *  # noqa

import numpy as np
import pandas as pd

LAG = 1
P = close_panel(["GDX", "GLD", "SLV"]).dropna(subset=["GDX", "GLD"])


def clusters(trig: pd.DatetimeIndex, valid: pd.DatetimeIndex, gap: int):
    pos = pd.Series(range(len(valid)), index=valid)
    t = [d for d in sorted(pd.DatetimeIndex(trig).intersection(valid))]
    out, cur, last = [], [], None
    for d in t:
        p = pos[d]
        if last is not None and p - last >= gap:
            out.append(cur)
            cur = []
        cur.append(d)
        last = p
    if cur:
        out.append(cur)
    return out


def conventions(fw: pd.Series, trig, gap: int, label: str):
    valid = fw.dropna().index
    cl = clusters(trig, valid, gap)
    first = np.array([fw.loc[c[0]] for c in cl])
    mean_ = np.array([np.nanmean(fw.loc[c].values) for c in cl])
    last_ = np.array([fw.loc[c[-1]] for c in cl])
    rows = []
    for nm, v in (("first-of-cluster", first), ("cluster-MEAN", mean_), ("last-of-cluster", last_)):
        r = summarize(v, f"{label} {nm}")
        r["boot_p"] = bootstrap_p_le0(v)
        rows.append(r)
    return cl, rows, mean_


g = P["GDX"]
r21 = g.pct_change(21) * 100
rk63 = pct_rank(g, 63)
trig8 = r21.index[((r21 >= 12.0) & (rk63 < 30.0)).fillna(False)]

for H in (5, 10, 21):
    fw = fwd_lag(g, H, LAG)
    cl, rows, cm = conventions(fw, trig8, H, f"C8 h{H}")
    show(rows, f"C8 episode-convention robustness h{H} (n_clusters={len(cl)})")

fw = fwd_lag(g, 10, LAG)
cl, _, _ = conventions(fw, trig8, 10, "C8 h10")
print("\n=== C8 h10 per-cluster detail (cluster-mean fwd ret) ===")
for c in cl:
    v = fw.loc[c].values
    print(f"  {c[0].date()} .. {c[-1].date()}  ndays={len(c):2d}  first={100*v[0]:+7.2f}%  "
          f"mean={100*np.nanmean(v):+7.2f}%  min={100*np.nanmin(v):+7.2f}%  max={100*np.nanmax(v):+7.2f}%")

# --- same treatment for C5 and C6 so the comparison is apples to apples ---
sp5 = (P["GDX"].pct_change(21) - P["GLD"].pct_change(21)) * 100
trig5 = sp5.index[(sp5 >= 8.0).fillna(False)]
fw5 = pair_fwd(P, "GLD", "GDX", 5, LAG)
_, rows5, _ = conventions(fw5, trig5, 5, "C5 h5 (GLD-GDX)")
show(rows5, "C5 episode-convention robustness")

Q = close_panel(["SLV", "GLD"]).dropna()
sp6 = (Q["SLV"].pct_change(63) - Q["GLD"].pct_change(63)) * 100
r5s = Q["SLV"].pct_change(5) * 100
trig6 = sp6.index[((sp6 <= -8.0) & (r5s > 0)).fillna(False)]
fw6 = pair_fwd(Q, "SLV", "GLD", 10, LAG)
_, rows6, _ = conventions(fw6, trig6, 10, "C6 h10 (SLV-GLD)")
show(rows6, "C6 episode-convention robustness")

# C8 stability: drop the single worst cluster, and LOYO
cl, _, cm = conventions(fw, trig8, 10, "x")
order = np.argsort(cm)
print(f"\nC8 h10 cluster-mean: all={100*cm.mean():+.2f}% (n={len(cm)}); "
      f"drop-worst={100*np.delete(cm, order[0]).mean():+.2f}%; "
      f"drop-best={100*np.delete(cm, order[-1]).mean():+.2f}%")
yrs = np.array([c[0].year for c in cl])
loyo = [(y, round(100 * cm[yrs != y].mean(), 2)) for y in sorted(set(yrs))]
print("C8 h10 leave-one-year-out cluster-mean:", loyo)
