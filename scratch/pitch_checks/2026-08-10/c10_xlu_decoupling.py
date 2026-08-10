"""C10 round 1 -- utilities falling while yields ALSO fell: decoupling, not washout.

Friday 2026-08-07 killed the outright XLU washout long (episodes -0.123% vs a
+0.207% own drift; the SPY-near-high gate HURTS: +0.605% ungated -> -0.123%
gated) and the XLU-vs-SPY pair (episodes -0.311%, welch t=-0.65, bootstrap
P(mean<=0)=0.774). The ONLY new thing today is the yield conditioner.

So this script is a GATE-ATTRIBUTION test and nothing else. If adding
"^TNX fell over the last 5 sessions" does not move the number, the idea dies
for the third time and the honest statement is what the trade actually keys on.

It also prices the gate: a rank is not a magnitude (registry). ^TNX rank5d
15.9 today corresponds to a specific bp move -- printed below.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["XLU", "SPY", "^TNX", "TLT", "XLP"]).dropna(subset=["XLU", "^TNX"])
idx = px.index

xlu_z = zscore(px["XLU"], 10)
tnx_r5 = pct_rank(px["^TNX"], 5)
tnx_d5 = px["^TNX"].diff(5) * 100.0        # bps
spy_hi = px["SPY"] / px["SPY"].rolling(252).max() - 1.0

print("=" * 88)
print("0. WHAT THE GATE ACTUALLY BUYS TODAY (a rank is not a magnitude)")
print("=" * 88)
print(f"  XLU z10            {xlu_z.iloc[-1]:+.2f}")
print(f"  ^TNX rank5d        {tnx_r5.iloc[-1]:.1f}")
print(f"  ^TNX 5d change     {tnx_d5.iloc[-1]:+.1f} bps")
sel = (tnx_r5 < 20) & tnx_d5.notna()
q = 100 * float((tnx_d5[sel] <= tnx_d5.iloc[-1]).mean())
print(f"  within the rank5d<20 cell (n={int(sel.sum())}): median "
      f"{tnx_d5[sel].median():+.1f} bps, 10th pctile "
      f"{np.percentile(tnx_d5[sel], 10):+.1f} bps")
print(f"  TODAY'S {tnx_d5.iloc[-1]:+.1f} bps sits at the {q:.1f}th percentile "
      f"of that cell -> the gate fires on the SHALLOWEST end of what it claims")
print(f"  SPY dist from 52w high {100*spy_hi.iloc[-1]:.2f}% (the gate Friday "
      f"showed HURTS)")

wash = xlu_z <= -2.0
yfall_rank = tnx_r5 < 20
yfall_sign = tnx_d5 < 0
yfall_hard = tnx_d5 <= -15.0        # a real 5d rally in duration
print(f"\n  state counts: XLU z10<=-2 {int(wash.sum())} days | "
      f"^TNX rank5d<20 {int(yfall_rank.sum())} | d5<0 {int(yfall_sign.sum())} "
      f"| d5<=-15bps {int(yfall_hard.sum())}")
print(f"  fires today: wash {bool(wash.iloc[-1])}  rank {bool(yfall_rank.iloc[-1])} "
      f"  sign {bool(yfall_sign.iloc[-1])}  hard {bool(yfall_hard.iloc[-1])}")

cases = {
    "W  washout only (z10<=-2)  [FRIDAY'S DEAD CELL]": wash,
    "W + yields fell (rank5d<20)  [TODAY'S GATE]": wash & yfall_rank,
    "W + yields fell (d5 < 0)": wash & yfall_sign,
    "W + yields fell HARD (d5 <= -15bps)": wash & yfall_hard,
    "W + yields ROSE (d5 > 0)  [the inverse gate]": wash & (tnx_d5 > 0),
    "yields fell only (rank5d<20), no washout": yfall_rank & ~wash,
    "W + yields fell + SPY within 1% of 52w high (today)": wash & yfall_rank & (spy_hi >= -0.01),
}

for legs, tag, cost in [([("XLU", 1.0)], "LONG XLU", 3.0),
                        ([("XLU", 1.0), ("SPY", -1.0)], "LONG XLU / SHORT SPY", 3.0)]:
    for H in (3, 5, 10):
        r = vehicle_ret(px, legs, H, 1)
        allr = r.dropna()
        rows = []
        for lbl, m in cases.items():
            d = idx[m.reindex(idx, fill_value=False).values & r.notna().values]
            d = declusters(d, max(H, 10), allr.index)
            if len(d) == 0:
                rows.append({"label": lbl, "n": 0})
                continue
            v = r.loc[d].values
            s = summarize(v, lbl)
            s["edge_pp"] = round(s["mean_pct"] - 100 * allr.mean(), 3)
            w = int((v > 0).sum())
            s["sign_p"] = round(sign_test(w, len(v)), 4)
            s["boot"] = round(bootstrap_p_le0(v), 3) if len(v) >= 3 else np.nan
            rows.append(s)
        show(rows, f"{tag}  h={H}  (own drift all-days {100*allr.mean():+.3f}%)")

print("\n" + "=" * 88)
print("GATE ATTRIBUTION SUMMARY: ungated vs gated, long XLU, every horizon")
print("=" * 88)
for H in (1, 2, 3, 5, 10, 21):
    r = vehicle_ret(px, [("XLU", 1.0)], H, 1)
    allr = r.dropna()
    out = []
    for lbl, m in [("ungated", wash), ("+rank gate", wash & yfall_rank),
                   ("+sign gate", wash & yfall_sign),
                   ("+hard gate", wash & yfall_hard),
                   ("+INVERSE (yields rose)", wash & (tnx_d5 > 0))]:
        d = declusters(idx[m.reindex(idx, fill_value=False).values
                           & r.notna().values], max(H, 10), allr.index)
        v = r.loc[d].values if len(d) else np.array([])
        out.append(f"{lbl} N={len(v):3d} {100*np.mean(v) if len(v) else np.nan:+.3f}%")
    print(f"h={H:2d}  ctrl {100*allr.mean():+.3f}%  |  " + "  |  ".join(out))

print("\n" + "=" * 88)
print("DETAIL: the exact cell the pitch would ship (gated, h=5)")
print("=" * 88)
battery(px, wash & yfall_rank, [("XLU", 1.0)], h=5,
        title="XLU long, z10<=-2 AND ^TNX rank5d<20", cost_bps=3.0, lag=1,
        min_gap=10, event_kinds=("cpi",),
        variants={"z10<=-1.5 + gate": (xlu_z <= -1.5) & yfall_rank,
                  "z10<=-2.5 + gate": (xlu_z <= -2.5) & yfall_rank,
                  "z10<=-2 + rank<10": wash & (tnx_r5 < 10),
                  "z10<=-2 + rank<30": wash & (tnx_r5 < 30)})

print("\n" + "=" * 88)
print("MIDTERM SPLIT on the gated cell, h=5")
print("=" * 88)
r5 = vehicle_ret(px, [("XLU", 1.0)], 5, 1)
d = declusters(idx[(wash & yfall_rank).reindex(idx, fill_value=False).values
                   & r5.notna().values], 10, r5.dropna().index)
v = r5.loc[d].values
mid = d.year % 4 == 2
show([summarize(v[mid], f"midterm N={int(mid.sum())}"),
      summarize(v[~mid], f"non-midterm N={int((~mid).sum())}")], "")
print(cluster_note(d, v, k=3))
print("episodes:", ", ".join(f"{x.date()} {100*y:+.1f}%" for x, y in zip(d, v)))
