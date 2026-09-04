"""Evening check, 2026-08-11: gold-miner thrust, follow or fade?

Tonight's tape: GDX +15.7% over 5 sessions (99th pctile of its trailing
year), +22.8% over 21. Question for a possible IDEA post: after a 5d
thrust of 12%+ in GDX, what happens next, entering at the NEXT close
(lag=1)? Cluster-first only, declustered 10td, controlled.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd

import pitch_lab as pl

px = pl.close_panel(["GDX", "GLD"])
px = px.dropna(subset=["GDX"])

r5 = px["GDX"].pct_change(5)
mask = (r5 >= 0.12).fillna(False)
trig_all = px.index[mask]
trig = pl.declusters(trig_all, 10, px.index)
print(f"trigger days: {len(trig_all)} raw, {len(trig)} declustered (10td)")
print("last 5 declustered:", [str(d.date()) for d in trig[-5:]])
print("fires tonight (2026-08-11)?", pd.Timestamp("2026-08-11") in trig_all)
recent = [d for d in trig_all if d >= pd.Timestamp("2026-08-01")]
print("august trigger days:", [str(d.date()) for d in recent])

H, LAG = 5, 1
fwd = pl.fwd_lag(px["GDX"], H, LAG)
rows = []
v = fwd.reindex(trig).dropna()
rows.append(pl.summarize(v.values, f"GDX fwd{H} lag1, cluster-first"))
up = int((v > 0).sum())
print(f"sign test (up): {up}/{len(v)} positive, "
      f"p={pl.sign_test(up, len(v)):.4f}")

ctrl = pl.local_control(px.index, trig, win=126)
rows.append(pl.summarize(fwd.reindex(ctrl).dropna().values,
                         "local control (+/-126td)"))
rows.append(pl.summarize(fwd.dropna().values, "all days"))

# late-in-cluster (tonight may be one): the honesty check
late = trig_all.difference(trig)
rows.append(pl.summarize(fwd.reindex(late).dropna().values,
                         "late-in-cluster lag-1"))

pl.show(rows, "GDX 5d thrust >= 12% -> next 5, lag-1")
print(pl.cluster_note(v.index, v.values))
for row in pl.era_split(v.index, v.values):
    print(row)

# horizon scan for a time stop, both anchors
for label, anchor in (("cluster-first", trig), ("late", late)):
    for h in (2, 3, 5, 10, 21):
        vv = pl.fwd_lag(px["GDX"], h, LAG).reindex(anchor).dropna()
        if not len(vv):
            continue
        s = pl.summarize(vv.values, "")
        u = int((vv > 0).sum())
        print(f"{label:14s} h={h:2d}: n={s['n']:3d} mean={s['mean_pct']:+.2f}% "
              f"hit={s['hit']:.0f}% t={s['t']:.2f} worst={s['worst_pct']:+.1f}% "
              f"sign_p(up)={pl.sign_test(u, s['n']):.4f}")
