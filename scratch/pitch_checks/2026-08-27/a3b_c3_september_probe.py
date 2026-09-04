"""C3 round-2 probe -- the ONLY cell in C3 that looked alive: the 9 September
instances of the exact PPI@+9 / CPI@+10 configuration (+1.378%, 8-1,
sign p 0.020, excess +0.909pp against SPY's ALL-DAY 10 td drift).

Three questions decide it:
  (a) does it beat SEPTEMBER's own drift, not the all-day drift? (a
      conditional cell must beat the instrument's own drift over the same
      horizon AND WINDOW -- September is SPY's weakest month, so an all-day
      control is the wrong denominator in the flattering direction)
  (b) the offset placebo ladder restricted to the September pairs
  (c) the month-of-year selection is a 12-cell grid entered AFTER the parent
      came in below drift -> Sidak charge
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pxd = load_prices(["SPY"])
px = pd.DataFrame({"SPY": pxd["SPY"]["Close"]}).dropna()
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
H, LAG = 10, 1
ret = vehicle_ret(px, [("SPY", 1.0)], H, LAG)
v = ret.notna()

ppi = load_events(["ppi"])["date"]
cpi = set(load_events(["cpi"])["date"])
pairs = [int(p) for d in ppi
         if (p := pos.get(d)) is not None and p + 1 < len(idx) and idx[p + 1] in cpi]
sig = idx[[p - 10 for p in pairs if p - 10 >= 0]]
sig = sig[ret.reindex(sig).notna().values]
sep = sig[sig.month == 9]
print(f"exact-config signals N={len(sig)}, September N={len(sep)}")
print("  September signal dates:", ", ".join(str(d.date()) for d in sep))
vals = ret.loc[sep].values
print("  per-instance:", ", ".join(f"{d.year}:{100*x:+.2f}%"
                                   for d, x in zip(sep, vals)))

print("\n(a) CONTROLS -- the denominator matters")
rows = [summarize(vals, f"COND September pairs (N={len(sep)})"),
        summarize(ret[v].values, "CTRL all days, all months"),
        summarize(ret[v & (idx.month == 9)].values, "CTRL all SEPTEMBER days"),
        summarize(ret[v & (idx.month == 9) & (idx.day <= 15)].values,
                  "CTRL September days 1-15 (the anchor's own half-month)")]
show(rows)
sep_drift = ret[v & (idx.month == 9)].mean()
half = ret[v & (idx.month == 9) & (idx.day <= 15)].mean()
print(f"  excess vs ALL-day drift      = {100*(vals.mean()-ret[v].mean()):+.3f}pp")
print(f"  excess vs SEPTEMBER drift    = {100*(vals.mean()-sep_drift):+.3f}pp")
print(f"  excess vs Sep 1-15 drift     = {100*(vals.mean()-half):+.3f}pp")
w = int((vals > 0).sum())
print(f"  record {w}-{len(vals)-w}, sign p (vs coin) = {sign_test(w, len(vals)):.4f}")
sep_hit = float((ret[v & (idx.month == 9)] > 0).mean())
print(f"  September base hit rate = {100*sep_hit:.1f}% -> sign p vs that base "
      f"= {sign_test(w, len(vals), sep_hit):.4f}")

print("\n(b) OFFSET PLACEBO LADDER restricted to the September pairs")
sep_pairs = [p for p in pairs if idx[p - 10].month == 9 and p - 22 >= 0]
rows = []
for k in range(-12, 13):
    dts = idx[[p - 10 + k for p in sep_pairs if 0 <= p - 10 + k < len(idx)]]
    vv = ret.loc[dts].dropna().values
    if len(vv) < 5:
        continue
    rows.append({"k": k, "n": len(vv), "mean_pct": round(100 * vv.mean(), 3),
                 "hit": round(100 * (vv > 0).mean(), 1),
                 "vs_sep_drift_pp": round(100 * (vv.mean() - sep_drift), 3)})
d = pd.DataFrame(rows).sort_values("mean_pct", ascending=False).reset_index(drop=True)
rank = int(d.index[d["k"] == 0][0]) + 1
print(f"  TRUE ANCHOR k=0 RANKS {rank} of {len(d)}")
print(d.to_string(index=False))

print("\n(c) SEARCH CHARGE -- month-of-year is a 12-cell selection made after")
print("    the parent came in BELOW drift.")
p_raw = sign_test(w, len(vals))
print(f"    raw sign p {p_raw:.4f} -> Sidak over 12 months = "
      f"{1-(1-p_raw)**12:.4f}")
print("    per-month table of the exact-config cell (the grid it came from):")
rows = []
for m in range(1, 13):
    s = sig[sig.month == m]
    if len(s) == 0:
        continue
    x = ret.loc[s].values
    md = ret[v & (idx.month == m)].mean()
    rows.append({"month": m, "n": len(x), "mean_pct": round(100 * x.mean(), 3),
                 "month_drift_pct": round(100 * md, 3),
                 "excess_pp": round(100 * (x.mean() - md), 3),
                 "hit": round(100 * (x > 0).mean(), 1)})
print(pd.DataFrame(rows).to_string(index=False))
