"""C11 round 2 — reproduce the registry number exactly, then the charged grid,
the reference class, decluster-gap sign stability, NVDA-print variance and the
book overlap.

Round 1 (e1_c11_qqq_laggard.py) established: the SPY gate moves the parent by
-0.036pp while dropping 67% of the sample, the excluded half pays MORE, and the
cell's whole dial support tops out at 80.4 against today's 89.5. This file
discharges the remaining mandatory tests so the kill is complete, and prices the
grid separately from the pre-specified cell.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

pd.set_option("display.width", 250)
rng = np.random.default_rng(20260825)
ROOT = Path(__file__).resolve().parents[3]

REF = ["XLK", "XLV", "XLF", "XLI", "XLE", "XLP", "XLU", "XLY", "XLB",
       "SMH", "IWM", "QQQ", "DIA", "IBB", "XRT", "ITB", "IYT", "KRE", "GDX", "XBI"]
NAMES = sorted(set(REF + ["SPY"]))
pxa = load_prices(NAMES)
CAL = pxa["SPY"]["Close"].dropna().index
px = pd.DataFrame({t: pxa[t]["Close"] for t in NAMES}).reindex(CAL)


def rv(t, n=63):
    return pct_rank(pxa[t]["Close"].dropna(), n).reindex(CAL)


sr = rv("SPY")
qr = rv("QQQ")
CELL = ((qr <= 20) & (sr > 20)).fillna(False)
CELL_SCRIPT = ((qr <= 20) & (sr >= 25)).fillna(False)

print("=" * 104)
print("A. REPRODUCE THE REGISTRY NUMBER EXACTLY (c3_c8 used min_gap=10, CAL cut at 2026-08-21)")
print("=" * 104)
BAR = pd.Timestamp("2026-08-21")
cal_b = CAL[CAL <= BAR]
px_b = px.reindex(cal_b)
for lbl, m in (("script form QQQ<=20 & SPY>=25", CELL_SCRIPT),
               ("prose form QQQ<=20 & SPY>20", CELL)):
    ret = vehicle_ret(px_b, [("QQQ", 1.0)], 5)
    v = ret.dropna().index
    t = cal_b[m.reindex(cal_b, fill_value=False).values].intersection(v)
    for gap in (5, 10):
        e = declusters(t, gap, v)
        val = ret.loc[e].values
        print(f"  {lbl:34s} gap={gap:>2d}  N={len(e):>3d}  mean {100*val.mean():+.3f}%  "
              f"hit {100*(val>0).mean():.1f}%  t {val.mean()/(val.std(ddof=1)/np.sqrt(len(val))):+.2f}")
print("  registry sentence: 'QQQ LONG pays +0.508% at h=5'")

print("\n" + "=" * 104)
print("B. DECLUSTER-GAP SIGN STABILITY (registry: a sign that moves on the gap is a cluster artifact)")
print("=" * 104)
ret5 = vehicle_ret(px, [("QQQ", 1.0)], 5)
v5 = ret5.dropna().index
trig = CAL[CELL.values].intersection(v5)
drift = float(ret5.loc[v5].mean())
for gap in (1, 5, 10, 21, 63):
    e = declusters(trig, gap, v5)
    val = ret5.loc[e].values
    print(f"  gap={gap:>2d} td  N={len(e):>3d}  mean {100*val.mean():+.3f}%  "
          f"EDGE over all-days drift {100*(val.mean()-drift):+.3f}pp  "
          f"t {val.mean()/(val.std(ddof=1)/np.sqrt(len(val))):+.2f}")
print(f"  QQQ all-days h=5 drift = {100*drift:+.3f}%")

print("\n" + "=" * 104)
print("C. REFERENCE CLASS — does EVERY laggard revert? same rule, 20 vehicles")
print("=" * 104)
rows = []
for t in REF:
    if t not in px.columns or px[t].dropna().empty:
        continue
    r = rv(t)
    m = ((r <= 20) & (sr > 20)).fillna(False)
    ret = vehicle_ret(px, [(t, 1.0)], 5)
    v = ret.dropna().index
    tt = CAL[m.values].intersection(v)
    if len(tt) < 20:
        continue
    e = declusters(tt, 5, v)
    val = ret.loc[e].values
    own = float(ret.loc[v].mean())
    rows.append({"vehicle": t, "n_epi": len(e), "mean_pct": round(100 * val.mean(), 3),
                 "own_drift_pct": round(100 * own, 3),
                 "edge_pp": round(100 * (val.mean() - own), 3),
                 "hit": round(100 * (val > 0).mean(), 1),
                 "t_edge": round(float((val.mean() - own) / (val.std(ddof=1) / np.sqrt(len(val)))), 2),
                 "live": bool(m.iloc[-1])})
rc = pd.DataFrame(rows).sort_values("edge_pp", ascending=False)
print(rc.to_string(index=False))
k = len(rc)
qedge = float(rc.loc[rc["vehicle"] == "QQQ", "edge_pp"].iloc[0])
rank = int((rc["edge_pp"] >= qedge).sum())
print(f"\n  QQQ edge over its own drift = {qedge:+.3f}pp -> rank {rank} of {k} vehicles; "
      f"P(random member >= QQQ) = {rank/k:.3f}")
print(f"  members with POSITIVE edge: {int((rc['edge_pp']>0).sum())} of {k}  "
      f"(median edge {rc['edge_pp'].median():+.3f}pp)")
tmax = rc["t_edge"].abs().max()
print(f"  max |t_edge| across the class = {tmax:.2f} on "
      f"{rc.loc[rc['t_edge'].abs().idxmax(),'vehicle']}")
# permutation max-of-k under rotation: rotate the mask, keep vehicle returns fixed
print("\n  ROTATION PERMUTATION, max-of-k over the 20-vehicle class (500 rotations):")
n = len(CAL)
maxts = []
base = {t: vehicle_ret(px, [(t, 1.0)], 5) for t in rc["vehicle"]}
masks = {t: ((rv(t) <= 20) & (sr > 20)).fillna(False).values for t in rc["vehicle"]}
for _ in range(500):
    sh = int(rng.integers(63, n - 63))
    best = 0.0
    for t in rc["vehicle"]:
        mm = np.roll(masks[t], sh)
        ret = base[t]
        v = ret.dropna().index
        tt = CAL[mm].intersection(v)
        if len(tt) < 10:
            continue
        e = declusters(tt, 5, v)
        val = ret.loc[e].values
        if len(val) < 5 or val.std(ddof=1) == 0:
            continue
        own = float(ret.loc[v].mean())
        tv = abs((val.mean() - own) / (val.std(ddof=1) / np.sqrt(len(val))))
        best = max(best, tv)
    maxts.append(best)
maxts = np.array(maxts)
print(f"    P(rotated grid max |t_edge| >= observed {tmax:.2f}) = {float((maxts>=tmax).mean()):.3f}")
qt = abs(float(rc.loc[rc['vehicle'] == 'QQQ', 't_edge'].iloc[0]))
print(f"    P(rotated grid max |t_edge| >= QQQ's {qt:.2f}) = {float((maxts>=qt).mean()):.3f}")

print("\n" + "=" * 104)
print("D. THE GRID CHARGE (36 definition cells from e1 section 8, Sidak + rotation)")
print("=" * 104)
qr_by = {lb: rv("QQQ", lb) for lb in (42, 63, 126)}
sr_by = {lb: rv("SPY", lb) for lb in (42, 63, 126)}
gm = {}
ts = []
for lb in (42, 63, 126):
    for q in (15, 20, 25, 30):
        for s in (20, 25, 30):
            m = ((qr_by[lb] <= q) & (sr_by[lb] > s)).fillna(False)
            tt = CAL[m.values].intersection(v5)
            e = declusters(tt, 5, v5)
            val = ret5.loc[e].values
            if len(val) < 5:
                continue
            tv = (val.mean() - drift) / (val.std(ddof=1) / np.sqrt(len(val)))
            gm[(lb, q, s)] = m.values
            ts.append(abs(tv))
ts = np.array(ts)
K = len(ts)
obs_max = ts.max()
print(f"  cells with an EDGE t computed: {K}; observed max |t_edge| = {obs_max:.2f}")
from math import erf, sqrt
p_one = 2 * (1 - 0.5 * (1 + erf(obs_max / sqrt(2))))
print(f"  naive two-sided p of the max = {p_one:.4f};  Sidak over {K}: "
      f"{1 - (1 - p_one) ** K:.4f}")
maxr = []
keys = list(gm)
for _ in range(500):
    sh = int(rng.integers(63, n - 63))
    best = 0.0
    for kk in keys:
        mm = np.roll(gm[kk], sh)
        tt = CAL[mm].intersection(v5)
        e = declusters(tt, 5, v5)
        val = ret5.loc[e].values
        if len(val) < 5 or val.std(ddof=1) == 0:
            continue
        best = max(best, abs((val.mean() - drift) / (val.std(ddof=1) / np.sqrt(len(val)))))
    maxr.append(best)
maxr = np.array(maxr)
print(f"  ROTATION NULL: P(grid max |t_edge| >= {obs_max:.2f}) = {float((maxr>=obs_max).mean()):.3f}")
pre = abs(float((ret5.loc[declusters(trig, 5, v5)].mean() - drift) /
                (ret5.loc[declusters(trig, 5, v5)].std(ddof=1) /
                 np.sqrt(len(declusters(trig, 5, v5))))))
print(f"  the PRE-SPEC cell's own edge |t| = {pre:.2f}  -> "
      f"P(rotation grid max >= that) = {float((maxr>=pre).mean()):.3f}")

print("\n" + "=" * 104)
print("E. NVDA PRINT INSIDE EVERY HOLD — realised variance of the cell's hold window")
print("=" * 104)
ep = ret5.loc[declusters(trig, 5, v5)].values
print(f"  cell h=5 episode sd = {100*ep.std(ddof=1):.2f}%, worst {100*ep.min():.2f}%, "
      f"mean {100*ep.mean():+.3f}%  -> mean/sd = {ep.mean()/ep.std(ddof=1):.3f}")
nv = pxa.get("QQQ")
print("  brief's measured NVDA-reaction sd: SMH 3.41% since 2020, XLK 1.54%.")
qd = px["QQQ"].pct_change()
print(f"  QQQ daily sd (2020+) = {100*qd[qd.index>='2020-01-01'].std():.2f}%  "
      f"-> one NVDA reaction session is ~{1.54/ (100*qd[qd.index>='2020-01-01'].std()):.2f}x "
      "a normal QQQ session on the XLK read-across")

print("\n" + "=" * 104)
print("F. BOOK OVERLAP — staged OLV semis (AMKR, ON, POWI) vs QQQ")
print("=" * 104)
for t in ("AMKR", "ON", "POWI"):
    try:
        s = load_prices([t])[t]["Close"].dropna()
    except Exception:
        continue
    d = s.pct_change().reindex(CAL).dropna()
    q = qd.reindex(d.index)
    ok = d.notna() & q.notna()
    d, q = d[ok], q[ok]
    d5, q5 = d[d.index >= "2024-01-01"], q[q.index >= "2024-01-01"]
    beta = float(np.polyfit(q5, d5, 1)[0])
    corr = float(np.corrcoef(q5, d5)[0, 1])
    print(f"  {t:5s} vs QQQ (2024+): corr {corr:+.3f}  beta {beta:.3f}  N={len(d5)}")
print("  (brief: those same names measure +0.727 corr / 1.617 beta vs XLK, +0.817 / 1.213 vs SMH)")

print("\n" + "=" * 104)
print("G. COST")
print("=" * 104)
print(f"  QQQ round trip assumed 2.5 bps (1 leg, penny-wide, ~$25bn ADV).")
print(f"  pre-spec episode mean {100*ep.mean():.3f}% = {10000*ep.mean():.1f} bps -> "
      f"{10000*ep.mean()/2.5:.1f}x cost")
print(f"  EDGE over QQQ's own drift {100*(ep.mean()-drift):.3f}% = "
      f"{10000*(ep.mean()-drift):.1f} bps -> {10000*(ep.mean()-drift)/2.5:.1f}x cost")
print("  the honest denominator is the EDGE, not the mean: buying and holding QQQ costs nothing.")
