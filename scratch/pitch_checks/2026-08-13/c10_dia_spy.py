"""C10 - DIA 5d break inside its own 63d leadership, against SPY.

Cell as specified: DIA rank5 <= 20 & DIA rank63 >= 80 & SPY 5d return > 0.
Recon says the joint state has ONE day in 26 years: today. Registry rule
"Count occurrences of a JOINT state before designing the trade -- unmeasurable
is a kill" (2026-08-07). This script prices the occurrence count, then walks
out through the definition neighbours to see whether any measurable family
exists, then prices the DIA/SPY vs QQQ/SPY spread correlation the brief asks
for.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TK = ["DIA", "SPY", "QQQ"]
px = close_panel(TK)
# registry trap 2026-08-12: distance-to-extreme / ranks off the OWN series
raw = load_prices(TK)
dia, spy, qqq = raw["DIA"]["Close"], raw["SPY"]["Close"], raw["QQQ"]["Close"]

d_r5, d_r63 = pct_rank(dia, 5), pct_rank(dia, 63)
s_r5 = pct_rank(spy, 5)


def mask(r5_max=20, r63_min=80, spy_pos=True):
    m = (d_r5 <= r5_max) & (d_r63 >= r63_min)
    if spy_pos:
        m &= spy.pct_change(5) > 0
    return m.reindex(px.index).fillna(False)


print("=== occurrence counts, definition neighbourhood ===")
rows = []
for r5 in (10, 15, 20, 25, 30, 40):
    for r63 in (70, 80, 90):
        for sp in (True, False):
            m = mask(r5, r63, sp)
            d = px.index[m.values]
            rows.append({"r5<=": r5, "r63>=": r63, "spy_pos": sp,
                         "n_days": len(d),
                         "n_epi": len(declusters(d, 5, px.index)),
                         "first": str(d[0].date()) if len(d) else "-",
                         "last": str(d[-1].date()) if len(d) else "-"})
print(pd.DataFrame(rows).to_string(index=False))

# --- beta of DIA on SPY (daily, full history) ---
rd, rs, rq = dia.pct_change(), spy.pct_change(), qqq.pct_change()
al = pd.concat([rd, rs, rq], axis=1).dropna()
al.columns = ["dia", "spy", "qqq"]
b_dia = np.polyfit(al["spy"], al["dia"], 1)[0]
b_qqq = np.polyfit(al["spy"], al["qqq"], 1)[0]
print(f"\nbeta DIA~SPY = {b_dia:.3f}   beta QQQ~SPY = {b_qqq:.3f}   "
      f"corr(DIA,SPY) = {al['dia'].corr(al['spy']):.3f}")

print("\n=== registry: DIA/SPY spread vs the 08-11 pitched QQQ/SPY spread ===")
for h in (1, 3, 5):
    sd = vehicle_ret(px, [("DIA", 1.0), ("SPY", -b_dia)], h)
    sq = vehicle_ret(px, [("QQQ", 1.0), ("SPY", -b_qqq)], h)
    j = pd.concat([sd, sq], axis=1).dropna()
    print(f"  h={h}: corr(DIA/SPY resid, QQQ/SPY resid) = {j.iloc[:, 0].corr(j.iloc[:, 1]):+.3f}"
          f"   (equal-dollar corr = "
          f"{pd.concat([vehicle_ret(px, [('DIA', 1.0), ('SPY', -1.0)], h), vehicle_ret(px, [('QQQ', 1.0), ('SPY', -1.0)], h)], axis=1).dropna().corr().iloc[0, 1]:+.3f})")

# --- the loosest measurable relative of the cell, run properly ---
print("\n\n########## loosest measurable relative: r5<=30, r63>=70, spy_pos ##########")
m = mask(30, 70, True)
trig = px.index[m.values]
print(f"n_days={len(trig)} epi={len(declusters(trig, 5, px.index))}")
if len(trig) >= 5:
    for h in (1, 3, 5):
        battery(px, m, [("DIA", 1.0), ("SPY", -b_dia)], h,
                f"loose C10 beta-neutral DIA/SPY (beta {b_dia:.2f})", 10.0)
        battery(px, m, [("DIA", 1.0)], h, "loose C10 long DIA outright", 5.0)

print("\n\n########## as-specified cell, no spy_pos gate (the only measurable form) ##########")
m2 = mask(20, 80, False)
trig2 = px.index[m2.values]
print(f"n_days={len(trig2)} epi={len(declusters(trig2, 5, px.index))}")
if len(trig2) >= 5:
    variants = {"r5<=15": mask(15, 80, False), "r5<=25": mask(25, 80, False),
                "r63>=70": mask(20, 70, False), "r63>=90": mask(20, 90, False)}
    for h in (3, 5):
        battery(px, m2, [("DIA", 1.0), ("SPY", -b_dia)], h,
                f"C10 no-SPY-gate beta-neutral (beta {b_dia:.2f})", 10.0, variants)
        battery(px, m2, [("DIA", 1.0)], h, "C10 no-SPY-gate long DIA outright", 5.0)
    # how much does the SPY-positive gate cost / where does it live
    print("\n--- SPY-positive gate attribution on the no-gate population ---")
    spos = (spy.pct_change(5) > 0).reindex(px.index).fillna(False)
    for h in (3, 5):
        r = vehicle_ret(px, [("DIA", 1.0), ("SPY", -b_dia)], h)
        e = declusters(trig2, 5, px.index)
        g = spos.loc[e].values
        show([summarize(r.loc[e[g]].values, f"h={h} SPY 5d>0 (N={int(g.sum())})"),
              summarize(r.loc[e[~g]].values, f"h={h} SPY 5d<=0 (N={int((~g).sum())})")])
    print("\n--- midterm split (year%4==2), episodes, h=3 ---")
    r = vehicle_ret(px, [("DIA", 1.0), ("SPY", -b_dia)], 3)
    e = declusters(trig2, 5, px.index)
    mt = np.array([d.year % 4 == 2 for d in e])
    show([summarize(r.loc[e[mt]].values, f"midterm (N={int(mt.sum())})"),
          summarize(r.loc[e[~mt]].values, f"non-midterm (N={int((~mt).sum())})")])
