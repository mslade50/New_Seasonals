import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = close_panel(["HYG", "IEF", "SPY"])
px = px[px["HYG"].notna() & px["IEF"].notna() & px["SPY"].notna()]
hyg, ief = px["HYG"], px["IEF"]
ratio = hyg / ief
hyg_r5 = pct_rank(hyg, 5); rat_r5 = pct_rank(ratio, 5); ief_r5 = pct_rank(ief, 5)

cells = {"child r5<=10 rat>=80": (hyg_r5 <= 10) & (rat_r5 >= 80),
         "direct HYG r5<=10 & IEF r5<=10": (hyg_r5 <= 10) & (ief_r5 <= 10),
         "parent HYG r5<=10": hyg_r5 <= 10}
for lbl, m in cells.items():
    d = px.index[m.reindex(px.index, fill_value=False).values]
    show(horizon_scan(px, d, [("HYG", 1.0)], hs=tuple(range(1, 11)), min_gap=5), f"horizon scan long HYG | {lbl} (min_gap 5)")

# trimmed view of the wider ratio neighbours (the -7.8% outlier)
H = 5
ret = vehicle_ret(px, [("HYG", 1.0)], H)
v = ret.notna()
for lbl, m in (("r5<=10 rat>=70", (hyg_r5 <= 10) & (rat_r5 >= 70)),
               ("r5<=15 rat>=70", (hyg_r5 <= 15) & (rat_r5 >= 70)),
               ("r5<=15 rat>=80", (hyg_r5 <= 15) & (rat_r5 >= 80))):
    s = px.index[m.reindex(px.index, fill_value=False).values & v.values]
    e = declusters(s, H, px.index)
    vv = ret.loc[e]
    worst = vv.idxmin()
    ex = vv.drop(worst)
    wins = int((ex > 0).sum())
    print(f"{lbl}: n={len(vv)} mean {100*vv.mean():+.3f}% worst {100*vv.min():+.2f}% on {worst.date()} | "
          f"ex-worst mean {100*ex.mean():+.3f}% record {wins}-{len(ex)-wins} sign p {sign_test(wins, len(ex)):.3f} | "
          f"ex-2022 mean {100*vv[vv.index.year != 2022].mean():+.3f}% (n={int((vv.index.year != 2022).sum())})")

# child 2022 vs rest record + sign test vs HYG own h=5 up-rate
s = px.index[cells["child r5<=10 rat>=80"].reindex(px.index, fill_value=False).values & v.values]
e = declusters(s, H, px.index)
vv = ret.loc[e]
base = float((ret[v] > 0).mean())
w = int((vv > 0).sum())
print(f"\nchild record {w}-{len(vv)-w}, sign p(0.5) {sign_test(w, len(vv)):.3f}, vs HYG h5 up-rate {base:.3f}: p {sign_test(w, len(vv), base):.3f}")
print("child 2022 episodes:", [(str(d.date()), round(100 * x, 3)) for d, x in vv[vv.index.year == 2022].items()])
print(f"live 2026-09-11 HYG close {hyg.iloc[-1]:.2f}")
