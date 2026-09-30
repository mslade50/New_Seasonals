"""C1 follow-up (C1 is killed in round 1 on era sign and live bucket).

Only to size the one sub-cell that carries the stated credit mechanism, the
PIT beta-IEF residual rank <= 10 inside the child, so the verdict can say
whether it is a parkable object: era split, top-2 share, h=3/5/10, and whether
09-23 sits in it (it does not: resid rank 23.8, HYG/IEF r21 91.7).  This cell
was FOUND by a split, so it carries a search charge.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
BAR = pd.Timestamp("2026-09-23")
px = close_panel(["SPY", "HYG", "IEF"]).dropna().loc[:BAR]
spy, hyg, ief = px["SPY"], px["HYG"], px["IEF"]
LEGS = [("SPY", -1.0)]
hr21 = pct_rank(hyg, 21)
dh, di = hyg.pct_change(), ief.pct_change()
beta = (dh.rolling(252).cov(di) / di.rolling(252).var()).shift(1)
res_rank = (dh - beta * di).rolling(21).sum().rolling(252).rank(pct=True) * 100
near2 = spy / spy.rolling(252).max() - 1.0 >= -0.02
cells = {"CHILD all": (hr21 <= 10) & near2,
         "CHILD & resid<=10 (credit-driven)": (hr21 <= 10) & near2 & (res_rank <= 10),
         "CHILD & resid>10 (TODAY's bucket)": (hr21 <= 10) & near2 & (res_rank > 10)}
for h in (3, 5, 10):
    r = vehicle_ret(px, LEGS, h)
    for k, m in cells.items():
        d = declusters(px.index[m.values & r.notna().values], h, px.index)
        v = r.loc[d].values
        w = int((v > 0).sum())
        e = era_split(d, v)
        print(f"h={h:2d} {k:36s} n={len(v):3d} mean {100*v.mean():+.3f}% rec {w}-{len(v)-w} "
              f"p {sign_test(w, len(v)):.4f} | pre-2018 {e[0].get('mean_pct', np.nan):+.3f} "
              f"(n={e[0]['n']}) 2018+ {e[1].get('mean_pct', np.nan):+.3f} (n={e[1]['n']})")
        if "credit" in k:
            print("      ", cluster_note(d, v))
