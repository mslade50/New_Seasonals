"""K3 c8 neighbour table: is the wrong sign a knife-edge or the whole family?
Belly-led selloff = IEF n-day rank <= a while TLT n-day rank >= b, lookbacks
3/5/10, a in {2,5,10}, b in {10,20,30,50}. Long IEF / -0.523 TLT, lag=1,
declustered at gap max(h,5). Also the discarded complement for each rung."""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
BAR = pd.Timestamp("2026-09-16")
px = close_panel(["IEF", "TLT"])
px = px[px.index <= BAR].dropna()
D = px.index
LEGS = [("IEF", 1.0), ("TLT", -0.523)]
R = {h: vehicle_ret(px, LEGS, h, 1) for h in (3, 5, 10)}


def ep(mask, h):
    ret = R[h]
    sig = D[(mask & ret.notna()).values]
    e = declusters(sig, max(h, 5), D)
    return ret.loc[e].values


rows = []
for n in (3, 5, 10):
    ri = pct_rank(px["IEF"], n)
    rt = pct_rank(px["TLT"], n)
    for a in (2, 5, 10):
        par = ri <= a
        for b in (10, 20, 30, 50):
            cell = par & (rt >= b)
            live = bool(cell.iloc[-1])
            rec = {"lb": n, "ief<=": a, "tlt>=": b, "live": live}
            for h in (3, 5, 10):
                v = ep(cell, h)
                c = ep(par & ~cell, h)
                rec[f"n{h}"] = len(v)
                rec[f"bp{h}"] = round(1e4 * v.mean(), 1) if len(v) else np.nan
                rec[f"w{h}"] = int((v > 0).sum())
                rec[f"disc_bp{h}"] = round(1e4 * c.mean(), 1) if len(c) else np.nan
            rows.append(rec)
df = pd.DataFrame(rows)
print(df.to_string(index=False))
for h in (3, 5, 10):
    sub = df[df[f"n{h}"] >= 3]
    print(f"h={h}: cells with n>=3: {len(sub)}; cell < discard in "
          f"{int((sub[f'bp{h}'] < sub[f'disc_bp{h}']).sum())}; cell < 0 in "
          f"{int((sub[f'bp{h}'] < 0).sum())}")
