"""C6/C7e: the FAIR lookback test. 10/42/63 are arguably different states;
15, 18, 25 and 30 sessions are true near neighbours of 21. If the cell only
exists at exactly 21, that is definition fragility rather than a slow dial.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

px = close_panel(["DX-Y.NYB", "^TNX", "GLD"]).dropna(subset=["DX-Y.NYB", "^TNX", "GLD"])
dx, tnx = px["DX-Y.NYB"], px["^TNX"]

def run(legs, h, cost_bps, title):
    print(f"\n=== {title} (h={h}, gap=21) ===")
    ret = vehicle_ret(px, legs, h)
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
            s["x_cost"] = round(100*s["mean_pct"]/cost_bps, 1)
        rows.append(s)
    show(rows)
    pos = sum(1 for r in rows if r.get("mean_pct", 0) > 0)
    print(f"  positive at {pos} of {len(rows)} lookbacks")

run([("GLD", 1.0)], 3, 2.0, "C7 long GLD")
run([("GLD", 1.0)], 5, 2.0, "C7 long GLD")
run([("DX-Y.NYB", -1.0)], 5, 1.5, "C6 short DXY spot")
run([("DX-Y.NYB", -1.0)], 3, 1.5, "C6 short DXY spot")
