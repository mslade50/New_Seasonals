"""C6e: the same fine lookback ladder on the FULL DX/TNX panel (2000+),
not the GLD-truncated one c67e used.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

px = close_panel(["DX-Y.NYB", "^TNX"]).dropna()
dx, tnx = px["DX-Y.NYB"], px["^TNX"]
print("panel", px.index[0].date(), "->", px.index[-1].date())

for h in (3, 5, 10):
    ret = vehicle_ret(px, [("DX-Y.NYB", -1.0)], h)
    valid = px.index[ret.notna().values]
    rows = []
    for ln in (10, 13, 15, 18, 21, 25, 30, 42, 63):
        rt, rd, rr = pct_rank(tnx, ln), pct_rank(dx, ln), tnx.pct_change(ln)
        m = ((rr > 0) & (rt >= 65) & (rd <= 20)).fillna(False)
        epi = declusters(px.index[m.values & ret.notna().values], 21, valid)
        v = ret.loc[epi].values
        s = summarize(v, f"lookback {ln}d" + ("  <-- PITCHED" if ln == 21 else ""))
        if s["n"]:
            s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
            s["bps"] = round(100*s["mean_pct"], 1)
            s["x_cost"] = round(100*s["mean_pct"]/1.5, 1)
        rows.append(s)
    show(rows, f"C6 short DXY spot, h={h}, gap=21, FULL panel")
    print("  positive at %d of %d lookbacks"
          % (sum(1 for r in rows if r.get("mean_pct", 0) > 0), len(rows)))
