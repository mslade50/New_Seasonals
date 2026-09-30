"""kB_b2_b -- kill confirmation for B2: definition neighbours + volume-gate
attribution, so the round-1 kill is not a one-cell accident.

Grid: resid <= -1.0/-1.5/-2.0% x volx >= none/1.5/2/3, h = 1/3/5, family
date-clustered + XLF alone (episodes). Also the lag=0 contrast at h=1 (the
reversal the lag=1 entry cannot reach), reported for the registry, not traded.
Reuses kB_b2's definitions by import.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch_lab import *  # noqa
import io, contextlib  # noqa: E402
with contextlib.redirect_stdout(io.StringIO()):
    import kB_b2_xlf_shock_short as b2  # noqa: E402

rows = []
for thr in (-0.010, -0.015, -0.020):
    for vx in (0, 1.5, 2.0, 3.0):
        r = {"resid<=": f"{100*thr:.1f}%", "volx>=": vx or "none"}
        for h in (1, 3, 5):
            df, V, D = b2.member_table(h, thr=thr, vx=vx)
            s, tt = b2.dcl(V, D)
            r[f"h{h}_dates"] = len(s)
            r[f"h{h}_dcl"] = round(100 * s.mean(), 3)
            r[f"h{h}_t"] = round(tt, 2)
            ret = b2.trade("XLF", h)
            m = b2.shock("XLF", thr=thr, vx=vx)
            days = b2.CAL[m.values & ret.notna().values]
            e = declusters(days, h, b2.CAL)
            v = ret.loc[e].values
            r[f"XLF_h{h}"] = f"{100*v.mean():+.2f}/{int((v>0).sum())}-{int((v<=0).sum())}"
        rows.append(r)
print("=== NEIGHBOUR GRID (family date-clustered mean %, t; XLF mean%/record)")
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== lag=0 contrast, family h=1 (entering AT the shock close), headline")
vs, ds = [], []
for t in b2.FAM:
    ret0 = b2.trade(t, 1, lag=0)
    m = b2.shock(t)
    days = b2.CAL[m.values & ret0.notna().values]
    vs.append(ret0.loc[days].values); ds.extend(list(days))
V0 = np.concatenate(vs)
s0, t0 = b2.dcl(V0, ds)
w = int((s0 > 0).sum())
print(f"  lag=0 h=1 short: dcl {100*s0.mean():+.3f}% t {t0:+.2f} rec {w}-{len(s0)-w} "
      f"(negative = the sector BOUNCES vs beta the next day)")
show(era_split(pd.DatetimeIndex(s0.index), s0.values), "lag=0 h=1 era split")
