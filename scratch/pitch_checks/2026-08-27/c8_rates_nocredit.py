"""C8 -- IEF and LQD near 52w lows while HYG is AT a 52w high.

COUNT FIRST (registry: a "dollar into CPI" cell had occurred ZERO times in 318
events). Then, only if the rung exists, measure. Watchlist 1 is the SPREAD form
of the same state and is episode-count blocked at 4 of 8 since 2007; C8 must
produce a DIFFERENT object or die as a duplicate.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

TK = ["IEF", "LQD", "HYG", "TLT", "SPY", "AGG"]
px = close_panel(TK)
px = px[px.index >= "2007-04-11"]   # HYG inception

def near_low(s, tol):
    lo = rolling_on_valid(s, lambda x: x.rolling(252).min())
    return (s / lo - 1.0) <= tol

def near_high(s, tol):
    hi = rolling_on_valid(s, lambda x: x.rolling(252).max())
    return (s / hi - 1.0) >= -tol

for t in ("IEF", "LQD", "HYG", "TLT"):
    s = px[t]
    lo = rolling_on_valid(s, lambda x: x.rolling(252).min())
    hi = rolling_on_valid(s, lambda x: x.rolling(252).max())
    print(f"today {t}: {s.iloc[-1]:.2f}  +{100*(s.iloc[-1]/lo.iloc[-1]-1):.2f}% "
          f"above 52w low, {100*(s.iloc[-1]/hi.iloc[-1]-1):+.2f}% vs 52w high")

# ---------- STEP 1: COUNT, before anything is measured ----------
legs_def = {
    "IEF<=1.5% above low": near_low(px["IEF"], 0.015),
    "LQD<=1.5% above low": near_low(px["LQD"], 0.015),
    "HYG within 0.25% of high": near_high(px["HYG"], 0.0025),
}
print("\n=== STEP 1: counts, each leg and the joint ===")
for k, v in legs_def.items():
    print(f"  {k:32s} {int(v.fillna(False).sum()):5d} days")
joint = (legs_def["IEF<=1.5% above low"] & legs_def["LQD<=1.5% above low"]
         & legs_def["HYG within 0.25% of high"]).fillna(False)
print(f"  {'JOINT (exact rung)':32s} {int(joint.sum()):5d} days")
epi_j = declusters(px.index[joint], 21, px.index)
print("  joint declustered (21td) episodes:", len(epi_j),
      [str(d.date()) for d in epi_j])
print("  joint by year:", {int(k): int(v) for k, v in
                           joint.groupby(px.index.year).sum().items() if v})
print("  today fires?", bool(joint.iloc[-1]))

# ---------- STEP 2: ONE stated widening ----------
wide = (near_low(px["IEF"], 0.03) & near_low(px["LQD"], 0.03)
        & near_high(px["HYG"], 0.01)).fillna(False)
epi_w = declusters(px.index[wide], 21, px.index)
print("\n=== STEP 2: ONE widening -- IEF/LQD <=3% above low, HYG within 1% of high ===")
print("  days:", int(wide.sum()), "| declustered episodes:", len(epi_w))
print("  episodes:", [str(d.date()) for d in epi_w])
print("  by year:", {int(k): int(v) for k, v in
                     wide.groupby(px.index.year).sum().items() if v})

use = wide if len(epi_w) >= 8 else joint
label = "WIDENED" if len(epi_w) >= 8 else "EXACT"
print(f"\n>>> measuring the {label} rung (episodes={len(declusters(px.index[use], 21, px.index))})")

variants = {
    "exact rung": joint,
    "widened rung": wide,
    "IEF+LQD low only (NO HYG gate)":
        (near_low(px["IEF"], 0.03) & near_low(px["LQD"], 0.03)).fillna(False),
    "HYG high only (NO rates gate)": near_high(px["HYG"], 0.01).fillna(False),
    "all days": pd.Series(True, index=px.index),
}

for legs, cost, nm in (([("TLT", 1.0)], 4.0, "LONG TLT"),
                       ([("IEF", 1.0)], 3.0, "LONG IEF"),
                       ([("HYG", 1.0)], 4.0, "LONG HYG"),
                       ([("SPY", 1.0)], 3.0, "LONG SPY"),
                       ([("LQD", 1.0), ("HYG", -1.0)], 4.0, "LQD - HYG spread")):
    for h in (5, 10):
        battery(px, use, legs, h, f"C8 {nm} | {label} rung", cost,
                variants=variants if nm == "LONG TLT" else None,
                min_gap=21, event_kinds=("jackson_hole",))

# ---------- the watchlist-1 duplication test ----------
print("\n\n### duplication test vs watchlist 1 (LQD residual against IEF) ###")
for h in (5, 10):
    rl = fwd_lag(px["LQD"], h, 1)
    ri = fwd_lag(px["IEF"], h, 1)
    e = declusters(px.index[use.values & rl.notna().values & ri.notna().values],
                   21, px.index)
    if len(e) < 2:
        print(f"h={h}: too few"); continue
    x = ri.loc[e].values; y = rl.loc[e].values
    b = np.polyfit(x, y, 1)
    print(f"h={h}: LQD = {b[0]:.3f}*IEF + {100*b[1]:+.3f}pp  "
          f"(LQD mean {100*y.mean():+.3f}%, IEF mean {100*x.mean():+.3f}%, N={len(e)})")

led = pd.read_parquet(ROOT / "data" / "backtest_trades_full.parquet")
led["Signal Date"] = pd.to_datetime(led["Signal Date"])
ov = led[led["Signal Date"].isin(set(px.index[use]))]
print("\n### book overlap ###  signals:", len(ov), "avgR:",
      round(ov["R_Multiple"].mean(), 3) if len(ov) else "n/a",
      "| book-wide", round(led["R_Multiple"].mean(), 3))
