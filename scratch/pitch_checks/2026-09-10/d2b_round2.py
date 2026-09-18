"""D2 round 2 -- the live state, not the cell average.

d2 established: the armed cell reproduces (N=29 filter-then-decluster at
+34.7 bps; N=25 decluster-then-filter at +39.4 bps, which is the entry's own
number, so the parked 39.4 came from the DTF order), it is genuinely
duration-neutral (corr with the yield change over the hold +0.028), and the
thrust leg alone is worth nothing (+4.3 bps on n=120 against a +4.3 bps
unconditional) -- it works only interacted with the max touch.

This round asks the questions that decide the LIVE trade:

 A. THE ARM IS AN IN-SAMPLE MEDIAN SPLIT. +78 bp is the episode median of the
    cell's own support, chosen from four candidate conditioners. Ladder the
    threshold and see whether 78 is a plateau or an edge, and charge the
    conditioner choice.
 B. THE CLEARANCE DOSE. Today clears by +1.1 bp. Continuous dose, quartiles,
    and the live bucket priced against the entry's own 5x cost bar.
 C. FIRST CROSSING vs deep inside a run.
 D. MULTIPLICITY charged over the walk the entry itself declared (180 cells,
    1080 with the lookbacks), long-only, with the null-max distribution shown
    so the charge is auditable.
 E. Regime split: the level of yields, and the direction of the curve trend.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403,E402
from pitch_lab import (close_panel, vehicle_ret, summarize, sign_test,
                       rolling_on_valid, show, bootstrap_p_le0)

warnings.filterwarnings("ignore")
pd.set_option("display.width", 260)

px = close_panel(["^TNX", "TLT", "IEF"]).dropna(how="any")
idx = px.index
tnx = px["^TNX"]
hi252 = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
off_hi = tnx / hi252 - 1.0
LEVEL = off_hi >= -0.0025
chg252 = (tnx - tnx.shift(252)) * 100.0
d = px[["TLT", "IEF"]].pct_change().dropna()
BETA = float(np.polyfit(d["IEF"].values, d["TLT"].values, 1)[0])
FLAT = [("IEF", 1.0), ("TLT", -1.0 / BETA)]
COST = 4.423
POS = {dd: i for i, dd in enumerate(idx)}
H = 8
RET = vehicle_ret(px, FLAT, H, 1)


def fdc(sig, gap):
    keep, last = [], -10 ** 9
    for dd in sig:
        p = POS.get(dd)
        if p is None:
            continue
        if p - last >= gap:
            keep.append(dd)
            last = p
    return pd.DatetimeIndex(keep)


def cellv(mask, h=H, gap=10):
    r = vehicle_ret(px, FLAT, h, 1)
    sig = idx[mask.reindex(idx, fill_value=False).values & r.notna().values]
    ep = fdc(sig, max(h, gap))
    return ep, r.loc[ep].values


def bl(v, label):
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"label": label, "n": 0}
    s = summarize(v, label)
    return {"label": label, "n": len(v), "bps": round(1e4 * v.mean(), 1),
            "hit": round(s["hit"], 1), "t": round(s["t"], 2),
            "rec": f"{int((v>0).sum())}-{int((v<0).sum())}",
            "signp": round(sign_test(int((v > 0).sum()), len(v)), 4),
            "x_cost": round(1e4 * v.mean() / COST, 2)}


print("=" * 124)
print("A. THE ARM IS AN IN-SAMPLE MEDIAN SPLIT OF THE CELL'S OWN SUPPORT")
print("=" * 124)
ep_l, v_l = cellv(LEVEL)
med = float(np.nanmedian(chg252.reindex(ep_l).values))
print(f"  LEVEL cell episode median of the 252d yield change = {med:+.1f} bp")
print(f"  -> the published arm '+78 bp' IS that median, rounded. Live {chg252.iloc[-1]:+.1f} bp.")
print("  threshold ladder (filter-then-decluster, h=8):")
rows = []
for thr in (40, 50, 60, 70, 74, 78, 82, 90, 100, 120):
    rows.append(bl(cellv(LEVEL & (chg252 >= thr))[1], f"chg252 >= {thr:3d} bp"))
show(rows, "is 78 a plateau or an edge?")
print("  the three conditioners the entry ALSO walked (its own text names four):")
chg63 = (tnx - tnx.shift(63)) * 100.0
rng252 = (rolling_on_valid(tnx, lambda x: x.rolling(252).max())
          - rolling_on_valid(tnx, lambda x: x.rolling(252).min())) * 100.0
chg21 = (tnx - tnx.shift(21)) * 100.0
for lab, ser in (("252d chg", chg252), ("63d chg", chg63),
                 ("252d range width", rng252), ("21d chg", chg21)):
    e = ser.reindex(ep_l).values
    m = float(np.nanmedian(e))
    hi_ = v_l[e > m]
    lo_ = v_l[e <= m]
    print(f"    {lab:18s} median {m:+7.1f}  HIGH half {1e4*hi_.mean():+6.1f} bps "
          f"(n={len(hi_)})  LOW half {1e4*lo_.mean():+6.1f} bps (n={len(lo_)})  "
          f"spread {1e4*(hi_.mean()-lo_.mean()):+6.1f} bps   live={float(ser.iloc[-1]):+7.1f} "
          f"-> {'HIGH' if float(ser.iloc[-1]) > m else 'LOW'}")
print("  the 252d-change conditioner was SELECTED as the one with a dose; that is")
print("  a 1-of-4 choice on top of the walk charged in section D.")

print("\n" + "=" * 124)
print("B. THE CLEARANCE DOSE -- today clears the bar by +1.1 bp")
print("=" * 124)
ARMED = LEVEL & (chg252 >= 78.0)
ep_a, v_a = cellv(ARMED)
clr = (chg252.reindex(ep_a) - 78.0).values
b, a = np.polyfit(clr, v_a, 1)
print(f"  regression ret = {1e4*a:+.1f} bps + {1e4*b:+.4f} bps per bp of clearance "
      f" corr {np.corrcoef(clr, v_a)[0,1]:+.3f}  n={len(v_a)}")
print(f"  fitted value AT today's clearance (+1.1 bp) = {1e4*(a + b*1.1):+.1f} bps "
      f"-> {(a + b*1.1)*1e4/COST:.2f}x cost")
q = np.percentile(clr, [25, 50, 75])
rows = []
edges = [0] + list(q) + [np.inf]
for i in range(4):
    m = (clr >= edges[i]) & (clr < edges[i + 1])
    rows.append(bl(v_a[m], f"clearance [{edges[i]:.0f},{edges[i+1]:.0f}) bp"))
show(rows, f"quartiles of clearance; today sits in the FIRST bucket at +1.1 bp")
for cut in (5, 10, 15, 20):
    m = clr <= cut
    print(f"  clearance <= {cut:2d} bp: n={int(m.sum())}  {1e4*v_a[m].mean():+6.1f} bps  "
          f"{1e4*v_a[m].mean()/COST:.2f}x  record "
          f"{int((v_a[m]>0).sum())}-{int((v_a[m]<0).sum())}   |   > {cut}: "
          f"n={int((~m).sum())}  {1e4*v_a[~m].mean():+6.1f} bps  "
          f"{1e4*v_a[~m].mean()/COST:.2f}x")

print("\n" + "=" * 124)
print("C. FIRST CROSSING vs deep inside a run")
print("=" * 124)
armb = ARMED.reindex(idx, fill_value=False)
run = armb & ~armb.shift(1, fill_value=False)
print("  ALL armed days:", int(armb.sum()), " first-crossing days:", int(run.sum()))
print("  today is a FIRST crossing (the arm was not satisfied on 2026-09-08).")
show([bl(cellv(ARMED & run)[1], "FIRST crossing of the armed state"),
      bl(cellv(ARMED & ~run)[1], "already inside an armed run"),
      bl(v_a, "all armed episodes (declustered gap 10)")],
     "freshness split")
fr = idx[run.values & RET.notna().values]
print(f"  first-crossing dates: {[str(x.date()) for x in fr]}")
frclr = (chg252.reindex(fr) - 78.0)
print(f"  their clearances: {[round(float(x),1) for x in frclr]}")
thin_fresh = fr[(frclr <= 5).values]
print(f"  FIRST crossings that were also THIN (<=5 bp) -- today's exact state: "
      f"n={len(thin_fresh)} {[str(x.date()) for x in thin_fresh]}")
if len(thin_fresh):
    vv = RET.reindex(thin_fresh).dropna().values
    print(f"    -> {1e4*vv.mean():+.1f} bps, record {int((vv>0).sum())}-{int((vv<0).sum())}, "
          f"{1e4*vv.mean()/COST:.2f}x cost")

print("\n" + "=" * 124)
print("D. MULTIPLICITY over the walk THE ENTRY DECLARED (long-only)")
print("=" * 124)
VEH = {"curve": FLAT, "IEF": [("IEF", 1.0)], "TLT": [("TLT", 1.0)]}
HS = list(range(1, 11))
PROX = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.03]
LBS = [63, 126, 189, 252, 378, 504]
rets = {(vn, h): vehicle_ret(px, legs, h, 1) for vn, legs in VEH.items() for h in HS}
arr = {k: np.nan_to_num(v.values, nan=0.0) for k, v in rets.items()}
val = {k: ~np.isnan(v.values) for k, v in rets.items()}
n_all = len(idx)
pos_arr = np.arange(n_all)
GAPS = sorted({max(h, 10) for h in HS})


def keeps_for(mask):
    sp = pos_arr[mask.reindex(idx, fill_value=False).values]
    out = {}
    for g in GAPS:
        kp, last = [], -10 ** 9
        for pp in sp:
            if pp - last >= g:
                kp.append(pp)
                last = pp
        out[g] = np.array(kp, dtype=int)
    return out


MASKS = {}
for lb in LBS:
    hi = rolling_on_valid(tnx, lambda x, L=lb: x.rolling(L).max())
    oh = tnx / hi - 1.0
    for p in PROX:
        MASKS[(lb, p)] = keeps_for(((oh >= -p) if p > 0 else (tnx >= hi - 1e-12))
                                   & (chg252 >= 78.0))
obs = float(v_a.mean())
print(f"  DEFENDED statistic: mean h=8 curve return over the 29 declustered armed")
print(f"  episodes = {1e4*obs:+.1f} bps.")
rng = np.random.default_rng(11)
NB = 3000
for gname, use_lb in (("GRID A: 180 cells (3 veh x 10 h x 6 prox, lookback 252)", False),
                      ("GRID B: 1080 cells (+ 6 lookbacks)", True)):
    keys = [k for k in MASKS if (use_lb or k[0] == 252)]
    nulls = np.empty(NB)
    unch = 0
    for b in range(NB):
        sh = int(rng.integers(21, n_all - 21))
        best = -9.9
        for k in keys:
            kk = MASKS[k]
            for h in HS:
                kp = kk[max(h, 10)]
                if len(kp) < 8:
                    continue
                src = (kp + sh) % n_all
                for vn in VEH:
                    vd = val[(vn, h)][src]
                    if vd.sum() < 8:
                        continue
                    mu = float(arr[(vn, h)][src][vd].mean())
                    if mu > best:
                        best = mu
                    if vn == "curve" and h == 8 and k == (252, 0.0025) and mu >= obs:
                        unch += 1
        nulls[b] = best
    print(f"  {gname}")
    print(f"    UNCHARGED p = {unch/NB:.4f}")
    print(f"    CHARGED   p = {(nulls >= obs).mean():.4f}")
    print(f"    null-max distribution (bps): p50 {1e4*np.percentile(nulls,50):.1f}  "
          f"p90 {1e4*np.percentile(nulls,90):.1f}  p95 {1e4*np.percentile(nulls,95):.1f}  "
          f"max {1e4*nulls.max():.1f}   vs observed {1e4*obs:.1f}")

print("\n" + "=" * 124)
print("E. REGIME SPLIT")
print("=" * 124)
lvl_at = tnx.reindex(ep_a).values
print(f"  live ^TNX level {tnx.iloc[-1]:.3f}")
show([bl(v_a[lvl_at < 3.0], "^TNX < 3.0% at signal"),
      bl(v_a[(lvl_at >= 3.0) & (lvl_at < 4.0)], "^TNX 3.0-4.0%"),
      bl(v_a[lvl_at >= 4.0], "^TNX >= 4.0% -- today's bucket")],
     "yield level regime")
print(f"  episode dates at ^TNX >= 4.0: "
      f"{[str(d.date()) for d, x in zip(ep_a, lvl_at) if x >= 4.0]}")
