"""C1 round 2b: (1) dose on the ABSOLUTE 252d change at A88 episode starts,
(2) parent era split, (3) the charge re-run with the dose dimension added
(the walk that produced the re-arm: 180 cells x the thresholds {78, 88}),
observed = A88 FTD curve h=8 mean. Draw-by-draw the union grid's null max is
>= the 78-only grid's, so this can only raise the 09-10 P of 0.7097."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
from pitch_lab import close_panel, vehicle_ret, rolling_on_valid

warnings.filterwarnings("ignore")
px = close_panel(["^TNX", "TLT", "IEF"]).dropna(how="any")
idx = px.index
tnx = px["^TNX"]
chg252 = (tnx - tnx.shift(252)) * 100.0
d = px[["TLT", "IEF"]].pct_change().dropna()
BETA = float(np.polyfit(d["IEF"].values, d["TLT"].values, 1)[0])
FLAT = [("IEF", 1.0), ("TLT", -1.0 / BETA)]
n_all = len(idx)
pos_arr = np.arange(n_all)
R8 = vehicle_ret(px, FLAT, 8, 1)


def fdc_pos(sp, g):
    kp, last = [], -10 ** 9
    for pp in sp:
        if pp - last >= g:
            kp.append(pp)
            last = pp
    return np.array(kp, dtype=int)


hi252 = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
lev = (tnx / hi252 - 1.0) >= -0.0025
m88 = (lev & (chg252 >= 88)).values & R8.notna().values
k88 = fdc_pos(pos_arr[m88], 10)
v88 = R8.values[k88]
c88 = chg252.values[k88]
print("A88 episodes by ABSOLUTE 252d change at start")
for lo, hi in ((88, 100), (100, 113), (113, 150), (150, 999)):
    m = (c88 >= lo) & (c88 < hi)
    vv = v88[m]
    print(f"  [{lo},{hi}) n={m.sum():2d} mean {1e4*vv.mean():+6.1f} bps  rec {int((vv>0).sum())}-{int((vv<0).sum())}")
print(f"  spearman(chg, ret) {pd.Series(c88).corr(pd.Series(v88), method='spearman'):+.3f}")
dd = idx[k88]
for lab, m in (("pre-2018", dd < "2018-01-01"), ("2018+", dd >= "2018-01-01")):
    vv = v88[m]
    print(f"  A88 {lab}: n={m.sum()} {1e4*vv.mean():+.1f} bps ({1e4*vv.mean()/4.423:.2f}x) rec {int((vv>0).sum())}-{int((vv<0).sum())}")
ex22 = dd.year != 2022
print(f"  A88 ex-2022: n={ex22.sum()} {1e4*v88[ex22].mean():+.1f} bps ({1e4*v88[ex22].mean()/4.423:.2f}x)")

VEH = {"curve": FLAT, "IEF": [("IEF", 1.0)], "TLT": [("TLT", 1.0)]}
HS = list(range(1, 11))
PROX = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.03]
arr, val = {}, {}
for vn, legs in VEH.items():
    for h in HS:
        r = vehicle_ret(px, legs, h, 1)
        arr[(vn, h)] = np.nan_to_num(r.values, nan=0.0)
        val[(vn, h)] = ~np.isnan(r.values)
MASKS = {}
for p in PROX:
    lvl = (tnx >= hi252 - 1e-12) if p == 0 else ((tnx / hi252 - 1.0) >= -p)
    for thr in (78, 88):
        sp = pos_arr[(lvl & (chg252 >= thr)).values]
        MASKS[(p, thr)] = {g: fdc_pos(sp, g) for g in (10,)}
obs = float(v88.mean())
rng = np.random.default_rng(14)
NB = 800
nulls = np.empty(NB)
for b in range(NB):
    sh = int(rng.integers(21, n_all - 21))
    best = -9.9
    for k, kk in MASKS.items():
        kp = kk[10]
        src = (kp + sh) % n_all
        for h in HS:
            for vn in VEH:
                vd = val[(vn, h)][src]
                if vd.sum() < 8:
                    continue
                mu = float(arr[(vn, h)][src][vd].mean())
                if mu > best:
                    best = mu
    nulls[b] = best
print(f"CHARGE 360 cells (3 veh x 10 h x 6 prox x thr{{78,88}}), obs A88 {1e4*obs:+.1f} bps: "
      f"P = {(nulls >= obs).mean():.4f}; null-max p50 {1e4*np.percentile(nulls,50):.1f} "
      f"p90 {1e4*np.percentile(nulls,90):.1f} p95 {1e4*np.percentile(nulls,95):.1f}")
