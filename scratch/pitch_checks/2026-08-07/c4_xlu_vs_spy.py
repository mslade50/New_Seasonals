"""C4: Utilities vs the index. Long XLU / short SPY equal dollar, 5 td.

Trigger identical to C1 (close of D): XLU z10 <= -2.0 AND SPY within 1.5% of
its 52w high. Entry MOC D+1, exit MOC D+6.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa
from _engine import battery, vehicle_ret

import numpy as np
import pandas as pd

H = 5
LEGS = [("XLU", 1.0), ("SPY", -1.0)]
px = close_panel(["SPY", "XLU", "XLP"]).dropna()
z = zscore(px["XLU"], 10)
spy_off = px["SPY"] / px["SPY"].rolling(252).max() - 1.0


def trig(zt: float = -2.0, spyt: float = -0.015) -> pd.Series:
    return (z <= zt) & (spy_off >= spyt)


base = trig()
print(f"C4 trigger fires on 2026-08-06: {bool(base.iloc[-1])}  "
      f"(XLU z10={z.iloc[-1]:.3f}, SPY {100*spy_off.iloc[-1]:.2f}% off 52wh)")

battery(px, base, LEGS, H, "C4 XLU-SPY spread, 5td", cost_bps=2,
        variants={
            "z<=-1.75": trig(-1.75), "z<=-2.00": trig(-2.00), "z<=-2.25": trig(-2.25),
            "z<=-2.50": trig(-2.50),
            "z<=-2.0 SPY within 3%": trig(-2.0, -0.03),
            "z<=-2.0 SPY within 0.75%": trig(-2.0, -0.0075),
            "z<=-2.0 NO SPY gate": (z <= -2.0),
        })

# --- leg attribution: where does any pair PnL come from? ---------------------
r_pair = vehicle_ret(px, LEGS, H, 1)
sig = px.index[base.fillna(False).values & r_pair.notna().values]
epi = declusters(sig, H, px.index)
show([summarize(vehicle_ret(px, [("XLU", 1.0)], H, 1).loc[epi].values, "long XLU leg"),
      summarize(vehicle_ret(px, [("SPY", 1.0)], H, 1).loc[epi].values, "SPY leg (as long)"),
      summarize(r_pair.loc[epi].values, "PAIR XLU-SPY"),
      summarize(r_pair[px.index >= sig[0]].values, "PAIR unconditional, same span")],
     "leg attribution (episodes)")

# --- beta reality check: equal-dollar is not beta-neutral --------------------
rx = px["XLU"].pct_change()
rs = px["SPY"].pct_change()
beta = (rx.rolling(252).cov(rs) / rs.rolling(252).var())
print(f"\nXLU 252d beta to SPY at 2026-08-06 = {beta.iloc[-1]:.2f}; "
      f"median over trigger episodes = {beta.loc[epi].median():.2f}")
print("  -> equal-dollar leaves residual short-beta; in a tape pinned to 52w highs "
      "that is a structural drag, not an edge.")
rb = vehicle_ret(px, [("XLU", 1.0)], H, 1) - beta.loc[:].reindex(px.index) * vehicle_ret(px, [("SPY", 1.0)], H, 1)
show([summarize(rb.loc[epi].values, "beta-hedged pair (episodes)"),
      summarize(rb[px.index >= sig[0]].values, "beta-hedged uncond, same span")],
     "beta-hedged variant")

# --- horizon sweep -----------------------------------------------------------
rows = []
for h in (1, 2, 3, 5, 10, 21):
    rr = vehicle_ret(px, LEGS, h, 1)
    s = px.index[base.fillna(False).values & rr.notna().values]
    e = declusters(s, h, px.index)
    a = summarize(rr.loc[e].values, f"h={h}")
    a["ctrl_pct"] = summarize(rr[(px.index >= s[0]) & (px.index <= s[-1])].values, "")["mean_pct"]
    a["excess_pct"] = a["mean_pct"] - a["ctrl_pct"]
    a["boot_p_le0"] = bootstrap_p_le0(rr.loc[e].values)
    rows.append(a)
show(rows, "horizon sweep (episodes)")
