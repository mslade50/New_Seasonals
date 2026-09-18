"""C1 round 2c: the honest PARENT. If the dose does not filter, what does the
trade key on? Ladder the thrust gate down to zero (LEVEL only), with all-days
and local +/-126td controls and the era split, same FTD gap-10 h=8 curve."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
from pitch_lab import close_panel, vehicle_ret, rolling_on_valid, local_control, sign_test

warnings.filterwarnings("ignore")
px = close_panel(["^TNX", "TLT", "IEF"]).dropna(how="any")
idx = px.index
tnx = px["^TNX"]
chg252 = (tnx - tnx.shift(252)) * 100.0
hi252 = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
LEVEL = (tnx / hi252 - 1.0) >= -0.0025
d = px[["TLT", "IEF"]].pct_change().dropna()
BETA = float(np.polyfit(d["IEF"].values, d["TLT"].values, 1)[0])
FLAT = [("IEF", 1.0), ("TLT", -1.0 / BETA)]
R = vehicle_ret(px, FLAT, 8, 1)
valid = R.notna().values
allmean = float(R.dropna().mean())
print(f"all-days curve h8 {1e4*allmean:+.1f} bps; pre-2018 {1e4*R[:'2017'].mean():+.1f}; 2018+ {1e4*R['2018':].mean():+.1f}")


def ftd(mask):
    return declusters(idx[mask.values & valid], 10, idx)


def line(lab, ep):
    v = R.reindex(ep).values
    loc = local_control(idx[valid], ep, 126)
    lm = float(R.reindex(loc).mean())
    pre = v[ep < "2018-01-01"]
    post = v[ep >= "2018-01-01"]
    w = int((v > 0).sum())
    print(f"  {lab:28s} n={len(v):3d} {1e4*v.mean():+6.1f} bps rec {w}-{len(v)-w} p {sign_test(w, len(v)):.4f} "
          f"| local {1e4*lm:+5.1f} | pre-2018 n={len(pre)} {1e4*np.nanmean(pre) if len(pre) else float('nan'):+6.1f} "
          f"| 2018+ n={len(post)} {1e4*np.nanmean(post) if len(post) else float('nan'):+6.1f}")


for thr in (None, 0, 40, 60, 78, 88):
    m = LEVEL if thr is None else (LEVEL & (chg252 >= thr))
    line("LEVEL only" if thr is None else f"LEVEL & chg>={thr}", ftd(m))
line("LEVEL & chg<78 (complement)", ftd(LEVEL & (chg252 < 78)))
line("LEVEL & chg<0 (falling yr)", ftd(LEVEL & (chg252 < 0)))
