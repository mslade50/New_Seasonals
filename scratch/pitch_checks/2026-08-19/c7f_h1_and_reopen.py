"""C7f: is h=1 (the strongest single cell) robust to the lookback ladder,
and what is the exact 'both dials at force' cell that would reopen this?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

px = close_panel(["DX-Y.NYB", "^TNX", "GLD"]).dropna(subset=["DX-Y.NYB", "^TNX", "GLD"])
dx, tnx = px["DX-Y.NYB"], px["^TNX"]

print("=== h=1 lookback ladder (the strongest single cell) ===")
ret = vehicle_ret(px, [("GLD", 1.0)], 1)
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
    rows.append(s)
show(rows)
print("  positive at %d of %d lookbacks"
      % (sum(1 for r in rows if r.get("mean_pct", 0) > 0), len(rows)))

print("\n=== the 'both dials at force' cell (the reopen condition) ===")
rk_tnx, rk_dx = pct_rank(tnx, 21), pct_rank(dx, 21)
r21 = tnx.pct_change(21)
lvl = tnx - tnx.shift(21)
for dxr, lv, lbl in [(10, 0.20, "DX rank<=10 AND TNX 21d rise>=+0.20pt"),
                     (10, 0.25, "DX rank<=10 AND TNX 21d rise>=+0.25pt"),
                     (15, 0.20, "DX rank<=15 AND TNX 21d rise>=+0.20pt"),
                     (20, 0.25, "DX rank<=20 AND TNX 21d rise>=+0.25pt")]:
    m = ((r21 > 0) & (rk_tnx >= 65) & (rk_dx <= dxr) & (lvl >= lv)).fillna(False)
    out = []
    for h in (1, 3, 5):
        r = vehicle_ret(px, [("GLD", 1.0)], h)
        vd = px.index[r.notna().values]
        epi = declusters(px.index[m.values & r.notna().values], 21, vd)
        v = r.loc[epi].values
        s = summarize(v, f"h={h} {lbl}")
        if s["n"]:
            s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        out.append(s)
    show(out)

print("\n=== today's readings against those gates ===")
print("  DX 21d rank %.1f (gate 10 / 15 / 20)" % rk_dx.iloc[-1])
print("  TNX 21d level rise %+0.3f pt (gate +0.20 / +0.25)" % lvl.iloc[-1])
