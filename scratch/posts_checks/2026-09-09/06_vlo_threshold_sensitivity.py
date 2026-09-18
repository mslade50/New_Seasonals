"""VLO z10-threshold sensitivity (2026-09-09) — robustness, NOT selection.

The cell reported in 02/04 uses the pre-specified z10 >= 1.5. Tonight VLO's
actual z10 is 2.93, so the honest question is whether the edge is a property of
"stretched at a new high" or an artifact of one threshold. The sweep is printed
so the report can say which; the SHIPPED form stays at 1.5, because tightening
to tonight's own reading after seeing the table is a post-hoc choice.

Same conventions: _metrics_for z10 and dist_52w_high, lag-1 MOC (entry close
D+1, exit close D+1+h), 5-session declustering.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403

import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-09-09")
vlo = load_prices(["VLO"])["VLO"]
vlo = vlo[vlo.index <= ASOF]
c = vlo["Close"].astype(float)
z = c.pct_change(10) / (c.pct_change().rolling(21).std() * np.sqrt(10))
dh = c / c.rolling(252).max() - 1.0
print(f"VLO tonight: close {float(c.iloc[-1]):.4f}  z10 {float(z.iloc[-1]):+.2f}  "
      f"dist_52w_high {100*float(dh.iloc[-1]):+.2f}%")

rows = []
for thr in (0.0, 1.0, 1.5, 2.0, 2.5, 2.93):
    cond = (dh >= 0.0) & (z >= thr)
    t = vlo.index[cond.fillna(False).values]
    t = t[t < ASOF]
    e = declusters(t, 5, vlo.index)
    for h in (3, 5, 10):
        s = (c.shift(-(1 + h)) / c.shift(-1) - 1.0)
        v = s.reindex(e).dropna()
        up, dn = int((v > 0).sum()), int((v < 0).sum())
        r = summarize(v.values, f"z10>={thr} h={h}")
        r.pop("sd_pct", None)
        r["record"] = f"{up}-{dn}"
        r["sign_p_up"] = round(sign_test(up, len(v)), 4)
        base = s.dropna()
        r["ctrl_all_days_pct"] = round(100 * base.mean(), 3)
        r["edge_pp"] = round(r["mean_pct"] - 100 * base.mean(), 3)
        rows.append(r)
show(rows, "VLO at a 252d high, z10 threshold sweep (episodes, lag-1 MOC)")
print("\nDONE.")
