"""C4 round 1: credit failing to confirm an index high. HYG 5d rank <= 5 AND LQD
within 1% of its 252 low AND SPY within 1% of its 252 high. Trades (one decision):
SHORT SPY h=5/10, or LONG HYG. Mandatory split: IEF flushed alongside (IEF 5d rank
<= 20, duration-driven, as today 7.9) vs not (spread-driven, the W53 arm form).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

T = ["SPY", "HYG", "LQD", "IEF"]
raw = load_prices(T)
IDX = raw["SPY"]["Close"].index
px = pd.DataFrame({t: raw[t]["Close"].reindex(IDX).ffill(limit=2) for t in T})


def own(t, f):
    s = raw[t]["Close"].dropna()
    return f(s).reindex(IDX).ffill(limit=2)


HR5 = own("HYG", lambda s: pct_rank(s, 5, 252))
IR5 = own("IEF", lambda s: pct_rank(s, 5, 252))
LQL = own("LQD", lambda s: s / s.rolling(252).min() - 1)
SPH = own("SPY", lambda s: s / s.rolling(252).max() - 1)
print(f"LIVE {IDX[-1].date()}: HYG r5 {HR5.iloc[-1]:.1f}, LQD over 252lo {100*LQL.iloc[-1]:.2f}%, "
      f"SPY vs 252hi {100*SPH.iloc[-1]:.2f}%, IEF r5 {IR5.iloc[-1]:.1f}")

spy_hi = SPH >= -0.01
hyg_f = HR5 <= 5
lqd_lo = LQL <= 0.01
cell = hyg_f & lqd_lo & spy_hi
dur = IR5 <= 20
print(f"days: cell {int(cell.sum())} (dur {int((cell & dur).sum())}, spread {int((cell & ~dur).sum())}); "
      f"HYG&SPYhi {int((hyg_f & spy_hi).sum())}; SPYhi {int(spy_hi.sum())}")
print("cell dates:", ", ".join(str(d.date()) for d in IDX[cell.fillna(False).to_numpy()]))

VEH = {"shortSPY": [("SPY", -1.0)], "longHYG": [("HYG", 1.0)]}


def ep(mask, legs, h, gap=10):
    ret = vehicle_ret(px, legs, h, 1)
    m = mask.reindex(IDX, fill_value=False).fillna(False).to_numpy() & ret.notna().to_numpy()
    s = IDX[m]
    if len(s) == 0:
        return "n/a", np.array([]), s
    e = declusters(s, gap, IDX)
    v = ret.loc[e].to_numpy()
    w = int((v > 0).sum())
    return f"{100*v.mean():+.3f} {w}-{len(v)-w} p{sign_test(w, len(v)):.2f}", v, e


rows = []
for name, legs in VEH.items():
    for h in (1, 3, 5, 10):
        ret = vehicle_ret(px, legs, h, 1).dropna()
        ret = ret[ret.index >= "2008-04-01"]
        r = {"veh": name, "h": h, "alldays": f"{100*ret.mean():+.3f}"}
        for lab, m in (("CELL", cell), ("cell dur", cell & dur), ("cell spread", cell & ~dur),
                       ("HYG&SPYhi", hyg_f & spy_hi), ("SPYhi only", spy_hi & ~hyg_f), ("HYGr5<=5", hyg_f),
                       ("HYG&LQD", hyg_f & lqd_lo)):
            r[lab] = ep(m, legs, h)[0]
        rows.append(r)
print("\n=== EPISODES (declustered 10td), % mean + record + sign p, pitched sign; controls 2008-04+ ===")
print(pd.DataFrame(rows).to_string(index=False))

for h in (5, 10):
    battery(px, cell, VEH["shortSPY"], h, f"C4 short SPY h={h}", 3.0, min_gap=10,
            variants={"HYG r5<=10": (HR5 <= 10) & lqd_lo & spy_hi, "HYG r5<=2.5": (HR5 <= 2.5) & lqd_lo & spy_hi,
                      "LQD within 2%": hyg_f & (LQL <= 0.02) & spy_hi, "LQD within 0.5%": hyg_f & (LQL <= 0.005) & spy_hi,
                      "SPY within 2%": hyg_f & lqd_lo & (SPH >= -0.02), "SPY within 0.5%": hyg_f & lqd_lo & (SPH >= -0.005),
                      "no LQD leg": hyg_f & spy_hi})
battery(px, cell, VEH["longHYG"], 5, "C4 long HYG h=5", 3.0, min_gap=10,
        variants={"dur (IEF r5<=20)": cell & dur, "spread (IEF r5>20)": cell & ~dur, "no LQD leg": hyg_f & spy_hi})

_, v5, e5 = ep(cell, VEH["shortSPY"], 5)
r10 = vehicle_ret(px, VEH["shortSPY"], 10, 1)
rh = vehicle_ret(px, VEH["longHYG"], 5, 1)
print("\nC4 episodes:")
for d, v in zip(e5, v5):
    print(f"  {d.date()}  HYGr5 {HR5[d]:.1f} LQD+{100*LQL[d]:.2f}% SPY {100*SPH[d]:+.2f}% IEFr5 {IR5[d]:.1f}  "
          f"shortSPY h5 {100*v:+.2f} h10 {100*r10[d]:+.2f}  longHYG h5 {100*rh[d]:+.2f}")
