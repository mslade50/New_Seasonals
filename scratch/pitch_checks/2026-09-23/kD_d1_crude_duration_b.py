"""kD D1 round 2 (kill confirmation). Round 1: 7 episodes, IEF -0.341%, 1-6.
Here: (A) crude lookback x threshold ladder on USO and CL=F, TNX ladder,
(B) gate attribution both legs at the widest honest neighbour, (C) mechanism
in its own window: does ^TNX actually fall over the hold after a crude
collapse with yields elevated, vs the rates-only state? (D) horizon scan on
the pre-specified sign, for the record only (no flip)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["IEF", "TLT", "USO", "CL=F", "^TNX"])
cl = px["CL=F"].copy()
cl[cl <= 0] = np.nan
px["CL=F"] = cl
IDX = px.index
H = 5
ret = vehicle_ret(px, [("IEF", 1.0)], H, 1)
valid = ret.notna()
tnx_chg = (px["^TNX"].shift(-(1 + H)) - px["^TNX"].shift(-1))  # yield pts over hold
r63 = pct_rank(px["^TNX"], 63)


def epi_stats(m, lbl, r=ret):
    dd = IDX[m.reindex(IDX, fill_value=False).fillna(False).values & r.notna().values]
    if len(dd) == 0:
        return {"label": lbl, "n": 0}
    e = declusters(dd, H, IDX)
    s = summarize(r.loc[e].values, lbl)
    s["n_days"] = len(dd)
    return s


print("A. CRUDE LOOKBACK x THRESHOLD LADDER (TNX r63>=80), long IEF h=5, episodes")
rows = []
for src in ("USO", "CL=F"):
    for lb in (3, 5, 10):
        rk = pct_rank(px[src], lb)
        for th in (2, 3, 5, 10):
            rows.append(epi_stats((rk <= th) & (r63 >= 80), f"{src} r{lb}<={th}"))
show(rows)
pos = sum(1 for r in rows if r.get("n", 0) and r["mean_pct"] > 0.069)
print(f"  cells beating IEF all-days drift (+0.069%): {pos} of {len(rows)}")

print("\nTNX ladder with USO r5<=3")
rk5 = pct_rank(px["USO"], 5)
show([epi_stats((rk5 <= 3) & (r63 >= t), f"TNX r63>={t}") for t in (50, 60, 70, 80, 90, 95)])

print("\nB. GATE ATTRIBUTION at the widest neighbour USO r5<=10 (and pre-spec <=3)")
for th in (3, 10):
    c = rk5 <= th
    show([epi_stats(c & (r63 >= 80), f"CELL crude<= {th} & rates"),
          epi_stats(c & (r63 < 80), "crude, rates gate DISCARDS (complement)"),
          epi_stats((r63 >= 80) & ~c, "rates, crude leg DISCARDS"),
          epi_stats(pd.Series(True, index=IDX), "all days")],
          f"crude threshold {th}")

print("\nC. MECHANISM: ^TNX change (yield pts) over the hold, episodes")
show([epi_stats((rk5 <= 3) & (r63 >= 80), "CELL", tnx_chg),
      epi_stats((rk5 <= 10) & (r63 >= 80), "CELL wide (USO r5<=10)", tnx_chg),
      epi_stats((r63 >= 80) & (rk5 > 10), "rates-only state", tnx_chg),
      epi_stats(pd.Series(True, index=IDX), "all days", tnx_chg)],
     "mean_pct column here = 100 x yield points (e.g. -5 = -5 bp)")

print("\nD. HORIZON SCAN pre-spec cell, LONG IEF (record only; no sign flip)")
dd = IDX[((rk5 <= 3) & (r63 >= 80)).fillna(False).values]
show(horizon_scan(px, dd, [("IEF", 1.0)], hs=tuple(range(1, 11))))
dd10 = IDX[((rk5 <= 10) & (r63 >= 80)).fillna(False).values]
show(horizon_scan(px, dd10, [("IEF", 1.0)], hs=tuple(range(1, 11))), "wide USO r5<=10")
