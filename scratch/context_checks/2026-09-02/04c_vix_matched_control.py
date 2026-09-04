"""Lock the numbers for the lead. 04b's matched control used a two-condition
calm state while the published cell uses one, so the control did not match the
cell. Rebuild it on exactly the published condition, and add the September and
midterm subsamples that the brief has to disclose.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

vx = load_prices(["^VIX"])["^VIX"]["Close"].dropna()
ev = load_events(["nfp"])["date"]
pos, _ = anchor_positions(vx.index, ev, offset=-2)
anch = vx.index[pos]
anch = anch[anch < vx.index[-1]]
r2 = fwd_ret(vx, 2)
rk = pct_rank(vx, 63)
print(f"live ^VIX {vx.iloc[-1]:.2f}, 63d return rank {rk.iloc[-1]:.1f}, "
      f"63d return {100*(vx.iloc[-1]/vx.dropna().iloc[-64]-1):+.1f}%")

calm_anch = anch[(rk.reindex(anch) <= 25).fillna(False).values]
calm_ctl = vx.index[(rk <= 25).fillna(False).values].difference(anch)


def line(v, lab):
    dn = int((v < 0).sum())
    return {"label": lab, "n": len(v), "down": dn, "up": len(v) - dn,
            "pct_down": round(100 * dn / len(v), 1),
            "median_pct": round(100 * float(np.median(v)), 3),
            "mean_pct": round(100 * float(v.mean()), 3),
            "sign_p_down": round(sign_test(dn, len(v)), 5)}


rows = [line(r2.reindex(calm_anch).dropna(), "pre-payrolls anchor, VIX 63d rank <= 25"),
        line(r2.reindex(calm_ctl).dropna(), "SAME state, no payrolls (matched ctl)"),
        line(r2.reindex(anch).dropna(), "pre-payrolls anchor, any state"),
        line(r2.dropna(), "every 2-session span")]
v = r2.reindex(calm_anch).dropna()
for lab, m in [("  cell, pre-2018", v.index < pd.Timestamp("2018-01-01")),
               ("  cell, 2018+", v.index >= pd.Timestamp("2018-01-01")),
               ("  cell, midterm years", (v.index.year % 4) == 2)]:
    sub = v[np.asarray(m)]
    if len(sub) >= 5:
        rows.append(line(sub, lab))
sep = anch[anch.month == 9]
rows.append(line(r2.reindex(sep).dropna(), "  September anchors, any state"))
show(rows, "^VIX, anchor close -> payrolls close")
print(cluster_note(v.index, v.values, k=2))
print("\nmost recent 10 episodes of the cell:")
for d, x in v.tail(10).items():
    print(f"  {d.date()} -> {100*x:+6.2f}%")
