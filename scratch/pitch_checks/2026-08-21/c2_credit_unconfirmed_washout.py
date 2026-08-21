"""C2 round 1: long SPY on a 5d washout while HYG sits at its 52w high.

The whole test is GATE ATTRIBUTION: measure the equity washout alone, then
add the credit gate, and report what the gate is worth in pp.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change

TK = ["SPY", "HYG", "IWM", "LQD"]
px = close_panel(TK)
px = px.loc[px["HYG"].notna()]          # HYG inception 2007-04
print("panel", px.index[0].date(), "..", px.index[-1].date(), len(px))

spy5 = _valid_pct_change(px["SPY"], 5)
spy5r = pct_rank(px["SPY"], 5)
spy_hi = rolling_on_valid(px["SPY"], lambda x: x.rolling(252).max())
spy_off = px["SPY"] / spy_hi - 1.0
hyg_hi = rolling_on_valid(px["HYG"], lambda x: x.rolling(252).max())
hyg_off = px["HYG"] / hyg_hi - 1.0
hyg5 = _valid_pct_change(px["HYG"], 5)
sma200 = rolling_on_valid(px["SPY"], lambda x: x.rolling(200).mean())

print("\nLIVE 2026-08-20: spy5 %.2f%% rank %.1f off-high %.2f%% | "
      "hyg off-high %.2f%% hyg5 %.2f%%"
      % (100 * spy5.iloc[-1], spy5r.iloc[-1], 100 * spy_off.iloc[-1],
         100 * hyg_off.iloc[-1], 100 * hyg5.iloc[-1]))

EQ = (spy5r <= 10)                       # the live equity leg, rank form
CR = (hyg_off >= -0.005)                 # HYG within 0.5% of its 52w high

H = 5
ret = fwd_lag(px["SPY"], H, 1)
valid = ret.notna()
alldays = px.index[valid.values]
base = ret.loc[alldays]


def cell(mask, label, h=H, min_gap=None):
    r = fwd_lag(px["SPY"], h, 1)
    v = r.notna()
    d = px.index[mask.reindex(px.index, fill_value=False).values & v.values]
    if len(d) == 0:
        return {"label": label, "n": 0}, d, np.array([])
    e = declusters(d, min_gap or h, px.index[v.values])
    s = summarize(r.loc[e].values, label)
    s["n_days"] = len(d)
    s["edge_pp"] = round(s["mean_pct"] - 100 * r.loc[v].mean(), 3)
    return s, e, r.loc[e].values


# ---------------------------------------------------------------- 1. gate attribution
rows = [summarize(base.values, f"CTRL-b all days h={H}")]
for lbl, m in [("EQ alone (spy5 rank<=10)", EQ),
               ("EQ + CREDIT (hyg within 0.5% of 52wh)", EQ & CR),
               ("EQ + NOT-credit (hyg >0.5% off high)", EQ & ~CR),
               ("CREDIT alone", CR)]:
    s, _, _ = cell(m, lbl)
    rows.append(s)
show(rows, "1. GATE ATTRIBUTION (episodes, h=5, lag=1)")

s_eq, _, _ = cell(EQ, "eq")
s_both, epi_both, v_both = cell(EQ & CR, "both")
print(f"\n  CREDIT GATE IS WORTH: {s_both['mean_pct'] - s_eq['mean_pct']:+.3f}pp "
      f"(episodes {s_both['n']} vs {s_eq['n']})")

# ---------------------------------------------------------------- 2. definition neighbours
vr = []
for tol in [0.0025, 0.005, 0.01, 0.02]:
    s, _, _ = cell(EQ & (hyg_off >= -tol), f"hyg within {100*tol:.2f}% of 52wh")
    vr.append(s)
for k in [5, 10, 20]:
    s, _, _ = cell((spy5r <= k) & CR, f"spy5 rank<={k} + credit")
    vr.append(s)
for mg in [-0.01, -0.015, -0.02, -0.03]:
    s, _, _ = cell((spy5 <= mg) & CR, f"spy5 <= {100*mg:.1f}% + credit")
    vr.append(s)
for hr in [0.0, -0.005, -0.01]:
    s, _, _ = cell(EQ & (hyg5 >= hr), f"spy5 rank<=10 + HYG 5d ret >= {100*hr:.1f}%")
    vr.append(s)
s, _, _ = cell(EQ & (spy_off >= -0.03), "spy5 rank<=10 + SPY within 3% of 52wh (no credit)")
vr.append(s)
s, _, _ = cell(EQ & CR & (spy_off >= -0.03), "FULL live state (all three)")
vr.append(s)
show(vr, "2. definition neighbours (episodes h=5)")

# ---------------------------------------------------------------- 3. is the gate a bull-tape selector?
above = (px["SPY"] > sma200)
bt = above.reindex(px.index).fillna(False)
d_eq = px.index[EQ.reindex(px.index, fill_value=False).values & valid.values]
d_both = px.index[(EQ & CR).reindex(px.index, fill_value=False).values & valid.values]
print("\n3. bull-tape over-selection (SPY > 200d)")
print(f"   base rate all days          {100*bt.loc[alldays].mean():.1f}%")
print(f"   EQ-alone trigger days       {100*bt.loc[d_eq].mean():.1f}%  (N={len(d_eq)})")
print(f"   EQ+CREDIT trigger days      {100*bt.loc[d_both].mean():.1f}%  (N={len(d_both)})")

# ---------------------------------------------------------------- 4. full battery on the pitched cell
battery(px, EQ & CR, [("SPY", 1.0)], H,
        "C2 long SPY | 5d rank<=10 + HYG within 0.5% of 52w high",
        cost_bps=2.0,
        variants={"hyg 1%": EQ & (hyg_off >= -0.01),
                  "hyg 2%": EQ & (hyg_off >= -0.02),
                  "eq rank<=20": (spy5r <= 20) & CR},
        event_kinds=("cpi", "fomc_decision"))

# ---------------------------------------------------------------- 5. era + horizon
show(horizon_scan(px, epi_both, [("SPY", 1.0)], hs=(1, 2, 3, 5, 10)),
     "5. horizon scan on the gated cell (episodes)")
_, epi_eq, v_eq = cell(EQ, "eq")
show(horizon_scan(px, epi_eq, [("SPY", 1.0)], hs=(1, 2, 3, 5, 10)),
     "5b. horizon scan, EQ ALONE (the thing the gate must beat)")

print("\n6. concentration, gated cell:", cluster_note(epi_both, v_both))
print("   concentration, eq alone :", cluster_note(epi_eq, v_eq))
show(era_split(epi_both, v_both), "7. era split, gated cell")
show(era_split(epi_eq, v_eq), "7b. era split, EQ alone")
