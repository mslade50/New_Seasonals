"""C9 round 2 - the h=7 spike, concentration, regime, and the near-miss number.

Round 1 killed the LITERAL state on population (3 days ever) and showed the
intact-trend gate is a NEGATIVE-value filter (bare r5<=5 +0.368% h=3, joint
+0.234%, and the BROKEN-trend complement r63<=30 pays MORE at +0.418%).  The
one thing that looked alive was h=7 (+1.613%, t 2.65, 9-of-13).  This round
decides whether that is a trade or a horizon-scan maximum on 13 episodes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 240)

SECTORS = ["XLK", "XLV", "XLP", "XLU", "XLI", "XLF", "XLY", "XLE", "XLB"]
px = close_panel(SECTORS + ["SPY"])
r5 = pct_rank(px["XLI"], 5)
r63 = pct_rank(px["XLI"], 63)
hi = rolling_on_valid(px["XLI"], lambda x: x.rolling(252).max())
d52 = px["XLI"] / hi - 1.0
MAIN = ((r5 <= 5) & (r63 >= 30) & (r63 <= 60) & (d52 >= -0.05)).fillna(False)

print("########## A. h=7 CONCENTRATION AND STRUCTURE ##########")
ret7 = vehicle_ret(px, [("XLI", 1.0)], 7)
v7 = ret7.dropna().index
e7 = declusters(px.index[MAIN.values].intersection(v7), 7, v7)
vals = ret7.loc[e7].values
print(f"  N={len(vals)}  mean {100*vals.mean():+.3f}%  record "
      f"{int((vals>0).sum())}-{int((vals<=0).sum())}  sign p="
      f"{sign_test(int((vals>0).sum()), len(vals)):.4f}")
print(f"  XLI unconditional h=7 drift {100*ret7.loc[v7].mean():+.3f}%")
print(f"  {cluster_note(e7, vals)}")
o = np.argsort(-vals)
print(f"  drop best 1: {100*np.delete(vals, o[0]).mean():+.3f}%   "
      f"drop best 2: {100*np.delete(vals, o[:2]).mean():+.3f}%   "
      f"drop best 3: {100*np.delete(vals, o[:3]).mean():+.3f}%")
for d, x in zip(e7, vals):
    print(f"    {d.date()}  {100*x:+7.3f}%")
show(era_split(e7, vals), "era split h=7")
mid = np.array([d.year % 4 == 2 for d in e7])
show([summarize(vals[mid], f"midterm (N={int(mid.sum())})"),
      summarize(vals[~mid], f"non-midterm (N={int((~mid).sum())})")], "midterm h=7")

print("\n########## B. HORIZON PLATEAU TEST - is h=7 a spike or a shelf? ##########")
rows = []
for h in range(1, 11):
    r = vehicle_ret(px, [("XLI", 1.0)], h)
    vv = r.dropna().index
    e = declusters(px.index[MAIN.values].intersection(vv), h, vv)
    s = summarize(r.loc[e].values, f"h={h}")
    s["drift_pct"] = round(100 * r.loc[vv].mean(), 3)
    s["edge_pct"] = round(s["mean_pct"] - 100 * r.loc[vv].mean(), 3)
    rows.append(s)
show(rows, "long XLI, main gate, h=1..10")

print("\n########## C. GATE ATTRIBUTION AT h=7 (does the gate survive its best horizon?) ##########")
rows = []
for lbl, m in [
    ("r5<=5 bare", (r5 <= 5).fillna(False)),
    ("r5<=5 & r63[30,60]", ((r5 <= 5) & (r63 >= 30) & (r63 <= 60)).fillna(False)),
    ("r5<=5 & r63[30,60] & nr high (MAIN)", MAIN),
    ("r5<=5 & r63<=30 (BROKEN trend)", ((r5 <= 5) & (r63 <= 30)).fillna(False)),
    ("r5<=5 & r63>=70", ((r5 <= 5) & (r63 >= 70)).fillna(False)),
]:
    e = declusters(px.index[m.values].intersection(v7), 7, v7)
    s = summarize(ret7.loc[e].values, lbl)
    s["edge_pct"] = round(s.get("mean_pct", np.nan) - 100 * ret7.loc[v7].mean(), 3)
    rows.append(s)
show(rows, f"h=7   (XLI drift {100*ret7.loc[v7].mean():+.3f}%)")

print("\n########## D. FRAGILITY DIAL ##########")
frag = pd.read_parquet("data/rd2_fragility.parquet")
frag.index = pd.to_datetime(frag.index)
dial = frag["63d"].rolling(10).mean()
dv = dial.reindex(e7)
print(f"  dial readings on the 13 episodes: "
      f"{[(str(d.date()), None if pd.isna(x) else round(x,1)) for d, x in zip(e7, dv.values)]}")
print(f"  today 89.5;  max episode dial "
      f"{np.nanmax(dv.values) if dv.notna().any() else float('nan'):.1f}")

print("\n########## E. NEAR-MISS ARITHMETIC (what would turn it on) ##########")
print("  The gate must ADD value, not subtract it.  Today the intact-trend gate")
print("  moves the h=3 parent from +0.368% (bare r5<=5, N=192 episodes) to")
print("  +0.234% (N=14), and the broken-trend complement pays MORE (+0.418%).")
print("  Turn-on: the joint cell has to beat the BARE washout at the same")
print("  horizon on a populated rung.  At h=7 the comparison is:")
e_bare = declusters(px.index[(r5 <= 5).fillna(False).values].intersection(v7), 7, v7)
print(f"    bare r5<=5  h=7: {100*ret7.loc[e_bare].mean():+.3f}%  N={len(e_bare)}")
print(f"    MAIN        h=7: {100*vals.mean():+.3f}%  N={len(vals)}")
print(f"    difference: {100*(vals.mean()-ret7.loc[e_bare].mean()):+.3f}pp")
