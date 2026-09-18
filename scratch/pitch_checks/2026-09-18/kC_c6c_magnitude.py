"""c6 round 2 addendum: MAGNITUDE-MATCHED gate attribution.

The threshold ladder rises with the threshold (h5 lag1 +0.391% at 1.5%, +0.997%
at 2.0%, +1.501% at 2.5%), and the single-name parent / anti-cell beat the
conjunction at 1.5%. Is the ladder just SLV's OWN move size? Bucket SLV's
signal-day move and compare complex-confirmed (GLD & GDX >= 1.5%, DX <= 0)
against SLV-only days of the SAME size. Live SLV move +3.37% -> the >= 3% bucket.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kC_common import *  # noqa

GAP = 5
px = panel(["GLD", "SLV", "GDX", "DX-Y.NYB"], "GLD", ffill=("DX-Y.NYB",)).loc["2006-05-22":]
r1 = {t: dret(px[t]) for t in px.columns}
cx = ((r1["GLD"] >= 0.015) & (r1["GDX"] >= 0.015) & (r1["DX-Y.NYB"] <= 0)).fillna(False)
L = [("SLV", 1.0)]
rows = []
for lo, hi in [(0.015, 0.02), (0.02, 0.03), (0.03, 9.0), (0.025, 9.0)]:
    sb = ((r1["SLV"] >= lo) & (r1["SLV"] < hi)).fillna(False)
    for lbl, m in [("complex-confirmed", sb & cx), ("SLV-only (not confirmed)", sb & ~cx)]:
        for h in (1, 3, 5):
            for lag in (0, 1):
                s, _, _ = cellstats(px, m, L, h, f"SLV [{100*lo:.1f},{100*hi if hi < 9 else 99:.0f}) {lbl} h={h} L{lag}", GAP, lag)
                rows.append({k: s.get(k) for k in ("label", "n", "mean_pct", "edge_pp", "rec", "p_coin")})
show(rows)
