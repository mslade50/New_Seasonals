"""C3 round 1b: gate attribution of the MOVE leg on top of the calm-VIX parent,
drop-best-2, and the only non-dead corner (h=1). Charged: 4 vehicles x 5 horizons
were walked in b3, so any single h pick carries a 20-cell search."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

exec(open(Path(__file__).with_name("b3_move_vix.py")).read().split("B = 1.48")[0].split("import pandas as pd", 1)[1])
B = 1.48
VEH = {"shortSVXYres": [("SVXY", -1.0), ("SPY", B)], "shortSPY": [("SPY", -1.0)], "longVIX": [("^VIX", 1.0)]}


def ep(mask, legs, h, lo=None):
    ret = vehicle_ret(px, legs, h, 1)
    m = mask.reindex(px.index, fill_value=False).to_numpy() & ret.notna().to_numpy()
    if lo:
        m &= (px.index >= lo)
    e = declusters(px.index[m], 10, px.index)
    return ret.loc[e].to_numpy(), e


out = []
for name, legs in VEH.items():
    lo = "2018-03-01" if name == "shortSVXYres" else None
    for h in (1, 5):
        v, e = ep(cell, legs, h, lo)
        p, _ = ep(calm & ~mvg, legs, h, lo)
        mo, _ = ep(mvg & ~calm, legs, h, lo)
        s = np.sort(v)[::-1]
        w = int((v > 0).sum())
        out.append({"veh": name, "h": h, "cell": f"{100*v.mean():+.3f} {w}-{len(v)-w} p{sign_test(w, len(v)):.3f}",
                    "cell_median": round(100 * np.median(v), 3), "drop_best2": round(100 * s[2:].mean(), 3),
                    "calm_parent": f"{100*p.mean():+.3f} med {100*np.median(p):+.3f}",
                    "gate_pp": round(100 * (v.mean() - p.mean()), 3),
                    "gate_pp_ex2": round(100 * (s[2:].mean() - p.mean()), 3),
                    "MOVE_only": f"{100*mo.mean():+.3f}",
                    "pre18": f"{100*v[e < '2018-01-01'].mean():+.3f}" if (e < '2018-01-01').any() else "n/a",
                    "post18": f"{100*v[e >= '2018-01-01'].mean():+.3f}"})
show(out, "C3 gate attribution (episodes, pitched sign)")

# live-form: both calm legs true as today (VIX < 16 AND rangepct <= 15)
both = mvg & (VX < 16) & (RP <= 15)
for name, legs in VEH.items():
    v, e = ep(both, legs, 5)
    print(f"both-calm-legs live form {name} h5: {[round(100*x, 2) for x in v]} on {[str(d.date()) for d in e]}")
