"""C4 -- "fear without damage": VIX 1d >= +5% on a day SPY barely moved.

THE WHOLE TEST IS GATE ATTRIBUTION, three ways:
  (i)  plain "SPY 1d > -0.75%" days
  (ii) plain "VIX 1d >= +5%" days regardless of spot
  (iii) local +/-126td control
If the joint state does not beat all three, the divergence carries nothing.

Vehicle is CASH EQUITY only (SPY / QQQ). Vol ETPs are registry-dead
(roll drag in every direction-consistent vehicle; SVXY is two instruments
across the 2018-02-28 leverage change).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["SPY", "QQQ", "^VIX"])
px = px.dropna(subset=["SPY", "^VIX"])
spy, vix, qqq = px["SPY"], px["^VIX"], px["QQQ"]
r_spy, r_vix = spy.pct_change(), vix.pct_change()
print(f"panel {px.index[0].date()} .. {px.index[-1].date()}  n={len(px)}")
print(f"live 2026-08-17: VIX 1d {100*r_vix.iloc[-1]:+.2f}% level {vix.iloc[-1]:.2f} | "
      f"SPY 1d {100*r_spy.iloc[-1]:+.2f}%")

VIX_T, SPY_T = 0.05, -0.0075
joint = (r_vix >= VIX_T) & (r_spy > SPY_T)
gate_spy = (r_spy > SPY_T) & r_vix.notna()
gate_vix = (r_vix >= VIX_T) & r_spy.notna()
print(f"\njoint N={int(joint.sum())}  SPY-only N={int(gate_spy.sum())}  "
      f"VIX-only N={int(gate_vix.sum())}")

# ---------------------------------------------------------------------------
# 1. GATE ATTRIBUTION at every horizon, long SPY, lag=1
# ---------------------------------------------------------------------------
def epi(mask, gap):
    d = px.index[mask.reindex(px.index, fill_value=False).values]
    return declusters(d, gap, px.index)


for h in (1, 2, 3, 5, 10):
    ret = vehicle_ret(px, [("SPY", 1.0)], h, lag=1)
    valid = ret.notna()
    gap = max(h, 5)
    dj, ds, dv = epi(joint & valid, gap), epi(gate_spy & valid, gap), epi(gate_vix & valid, gap)
    loc = local_control(px.index[valid.values],
                        px.index[(joint & valid).reindex(px.index, fill_value=False).values])
    rows = [
        summarize(ret.loc[dj].values, f"JOINT episodes (N={len(dj)})"),
        summarize(ret[joint & valid].values, f"JOINT day-level (N={int((joint&valid).sum())})"),
        summarize(ret.loc[ds].values, f"(i) SPY>-0.75% only, epi (N={len(ds)})"),
        summarize(ret.loc[dv].values, f"(ii) VIX>=+5% only, epi (N={len(dv)})"),
        summarize(ret.loc[loc].values, "(iii) local +/-126td ex-trigger"),
        summarize(ret[valid].values, "CTRL-b all days"),
    ]
    show(rows, f"1. GATE ATTRIBUTION, long SPY, h={h}")
    m = {r["label"].split(" ")[0]: r["mean_pct"] for r in rows}
    beat = (rows[0]["mean_pct"] > rows[2]["mean_pct"],
            rows[0]["mean_pct"] > rows[3]["mean_pct"],
            rows[0]["mean_pct"] > rows[4]["mean_pct"])
    print(f"  joint beats (i)/(ii)/(iii): {beat}  -> all three: {all(beat)}")

# ---------------------------------------------------------------------------
# 2. horizon scan 1..10 on episodes, joint cell
# ---------------------------------------------------------------------------
dj_all = px.index[joint.values]
show(horizon_scan(px, dj_all, [("SPY", 1.0)], hs=tuple(range(1, 11)), min_gap=5),
     "2. horizon scan 1..10, JOINT episodes, long SPY (edge_pct = vs all days)")
show(horizon_scan(px, dj_all, [("QQQ", 1.0)], hs=tuple(range(1, 11)), min_gap=5),
     "2b. same, long QQQ")
# the OTHER gate for reference: does the plain VIX pop carry it?
dv_all = px.index[gate_vix.values]
show(horizon_scan(px, dv_all, [("SPY", 1.0)], hs=tuple(range(1, 11)), min_gap=5),
     "2c. horizon scan, PLAIN VIX>=+5% episodes (the gate being tested)")

# ---------------------------------------------------------------------------
# 3. era split, joint cell
# ---------------------------------------------------------------------------
for h in (3, 5, 10):
    ret = vehicle_ret(px, [("SPY", 1.0)], h, lag=1)
    d = epi(joint & ret.notna(), max(h, 5))
    v = ret.loc[d].values
    rows = []
    for cut in ("2018-01-01", "2021-01-01"):
        m = d < pd.Timestamp(cut)
        rows.append(summarize(v[m], f"pre-{cut[:4]} (N={int(m.sum())})"))
        rows.append(summarize(v[~m], f"{cut[:4]}+ (N={int((~m).sum())})"))
    show(rows, f"3. era split, JOINT episodes, h={h}")
    print("  concentration:", cluster_note(d, v))

# ---------------------------------------------------------------------------
# 4. THRESHOLD NEIGHBOURS on both legs (12-cell grid -- charged as a search)
# ---------------------------------------------------------------------------
for h in (3, 5):
    ret = vehicle_ret(px, [("SPY", 1.0)], h, lag=1)
    rows = []
    for vt in (0.04, 0.05, 0.06, 0.08):
        for st in (-0.005, -0.0075, -0.010):
            m = (r_vix >= vt) & (r_spy > st) & ret.notna()
            d = epi(m, max(h, 5))
            r = summarize(ret.loc[d].values, f"VIX>={100*vt:.0f}% SPY>{100*st:+.2f}%")
            r["n_days"] = int(m.sum())
            rows.append(r)
    show(rows, f"4. threshold neighbours, h={h} (episodes)")
    signs = [np.sign(r.get("mean_pct", 0)) for r in rows if r.get("n", 0) > 3]
    print(f"  sign stability: {int(sum(1 for s in signs if s>0))}+ / "
          f"{int(sum(1 for s in signs if s<0))}- of {len(signs)} cells")

# ---------------------------------------------------------------------------
# 5. is the LIVE VIX LEVEL inside the support of the trigger set?
# ---------------------------------------------------------------------------
lvl = vix.loc[dj_all]
r21 = pct_rank(vix, 21).loc[dj_all]
print(f"\n5. LIVE-STATE SUPPORT.  live VIX level 15.19, VIX 21d rank 12.7")
print(f"   trigger-day VIX level: min {lvl.min():.2f} p10 {lvl.quantile(.10):.2f} "
      f"median {lvl.median():.2f} p90 {lvl.quantile(.90):.2f} max {lvl.max():.2f}")
print(f"   share of trigger days with VIX level <= 16: "
      f"{100*(lvl <= 16).mean():.1f}%  ( <=17: {100*(lvl<=17).mean():.1f}% )")
print(f"   trigger-day VIX 21d rank: median {r21.median():.1f}, "
      f"share <= 25: {100*(r21 <= 25).mean():.1f}%")
for h in (3, 5, 10):
    ret = vehicle_ret(px, [("SPY", 1.0)], h, lag=1)
    sub_lvl = joint & (vix <= 17) & ret.notna()
    sub_rk = joint & (pct_rank(vix, 21) <= 25) & ret.notna()
    show([summarize(ret.loc[epi(sub_lvl, max(h, 5))].values,
                    f"joint & VIX level<=17 (N_days={int(sub_lvl.sum())})"),
          summarize(ret.loc[epi(sub_rk, max(h, 5))].values,
                    f"joint & VIX 21d rank<=25 (N_days={int(sub_rk.sum())})"),
          summarize(ret.loc[epi(joint & (vix > 17) & ret.notna(), max(h, 5))].values,
                    "joint & VIX level>17")],
         f"5b. live-like subset, h={h}")
