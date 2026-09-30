"""C4 round 1b: the pre-specified cell has ZERO precedents (fires only 2026-09-25),
so measure its definition neighbours (charged: this is a grid I walked) with the
mandatory duration/spread split (IEF 5d rank <= 20 = duration-driven)."""
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
dur = IR5 <= 20
R = {("shortSPY", h): vehicle_ret(px, [("SPY", -1.0)], h, 1) for h in (5, 10)}
R.update({("longHYG", h): vehicle_ret(px, [("HYG", 1.0)], h, 1) for h in (5, 10)})
base = {k: v[v.index >= "2008-04-01"].mean() for k, v in R.items()}


def cellstr(mask, k):
    ret = R[k]
    m = mask.reindex(IDX, fill_value=False).fillna(False).to_numpy() & ret.notna().to_numpy()
    s = IDX[m]
    if len(s) == 0:
        return "n/a"
    e = declusters(s, 10, IDX)
    v = ret.loc[e].to_numpy()
    w = int((v > 0).sum())
    return f"{100*v.mean():+.2f} {w}-{len(v)-w}"


rows = []
for hr in (5, 10):
    for lq in (0.01, 0.02, 0.03, None):
        for sp in (0.01, 0.02):
            m = (HR5 <= hr) & (SPH >= -sp)
            if lq is not None:
                m = m & (LQL <= lq)
            live = bool(m.iloc[-1])
            r = {"HYGr5<=": hr, "LQD<=": lq if lq else "-", "SPY>=-": sp, "live": live}
            for k in R:
                r[f"{k[0]}{k[1]}"] = cellstr(m, k)
                r[f"{k[0]}{k[1]} dur"] = cellstr(m & dur, k)
                r[f"{k[0]}{k[1]} spr"] = cellstr(m & ~dur, k)
            rows.append(r)
df = pd.DataFrame(rows)
print("controls 2008-04+:", {f"{k[0]}{k[1]}": round(100 * v, 3) for k, v in base.items()})
pd.set_option("display.width", 250)
print(df[[c for c in df.columns if "SPY" in c or c in ("HYGr5<=", "LQD<=", "live")]].to_string(index=False))
print()
print(df[[c for c in df.columns if "HYG" in c or c in ("LQD<=", "SPY>=-", "live")]].to_string(index=False))

# the tightest live neighbour with the most history, episodes listed
m = (HR5 <= 5) & (SPH >= -0.01) & (LQL <= 0.03)
e = declusters(IDX[m.fillna(False).to_numpy() & R[("shortSPY", 10)].notna().to_numpy()], 10, IDX)
print("\nneighbour HYGr5<=5, LQD<=3%, SPY>=-1% episodes:")
for d in e:
    print(f"  {d.date()} HYGr5 {HR5[d]:.1f} LQD+{100*LQL[d]:.2f}% SPY {100*SPH[d]:+.2f}% IEFr5 {IR5[d]:.1f} "
          f"shortSPY h5 {100*R[('shortSPY', 5)][d]:+.2f} h10 {100*R[('shortSPY', 10)][d]:+.2f} "
          f"longHYG h5 {100*R[('longHYG', 5)][d]:+.2f} h10 {100*R[('longHYG', 10)][d]:+.2f}")
