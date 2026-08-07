"""C1: Utilities washout snapback, OUTRIGHT long XLU.

Trigger (measured on close of D): XLU z10 <= -2.0 AND SPY within 1.5% of its
52w high. Entry MOC D+1, exit MOC D+6 (5 td hold).

Adversarial brief: kill it. Everything below is measured, nothing assumed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa
from _engine import battery, vehicle_ret

import numpy as np
import pandas as pd

H = 5
px = close_panel(["SPY", "XLU", "XLP", "TLT", "IEF", "^TNX"]).dropna()
z = zscore(px["XLU"], 10)
spy_off = px["SPY"] / px["SPY"].rolling(252).max() - 1.0


def trig(zt: float, spyt: float = -0.015) -> pd.Series:
    return (z <= zt) & (spy_off >= spyt)


base = trig(-2.0)
print(f"C1 trigger fires on 2026-08-06: {bool(base.iloc[-1])}  "
      f"(z10={z.iloc[-1]:.3f}, SPY {100*spy_off.iloc[-1]:.2f}% off 52wh)")

battery(px, base, [("XLU", 1.0)], H, "C1 XLU outright, 5td", cost_bps=2,
        variants={
            "z<=-1.75": trig(-1.75), "z<=-2.00": trig(-2.00), "z<=-2.25": trig(-2.25),
            "z<=-2.50": trig(-2.50),
            "z<=-2.0 SPY within 3%": trig(-2.0, -0.03),
            "z<=-2.0 SPY within 0.75%": trig(-2.0, -0.0075),
            "z<=-2.0 NO SPY gate": (z <= -2.0),
        })

# --- does the SPY-near-high gate add anything, or is it decoration? ----------
print("\n--- gate decomposition (episodes, h=5) ---")
r = vehicle_ret(px, [("XLU", 1.0)], H, 1)
rows = []
for lbl, m in [("z<=-2 & SPY nr high", trig(-2.0)),
               ("z<=-2 & SPY NOT nr high", (z <= -2.0) & (spy_off < -0.015)),
               ("z<=-2 any tape", (z <= -2.0)),
               ("SPY nr high only", (spy_off >= -0.015) & z.notna())]:
    s = px.index[m.fillna(False).values & r.notna().values]
    if len(s) == 0:
        continue
    e = declusters(s, H, px.index)
    rows.append(summarize(r.loc[e].values, f"{lbl} (N_days={len(s)})"))
show(rows, "gate decomposition")

# --- horizon sweep: is 5td the only place it works? -------------------------
print("\n--- horizon sweep (episodes) ---")
rows = []
for h in (1, 2, 3, 5, 10, 21):
    rr = vehicle_ret(px, [("XLU", 1.0)], h, 1)
    s = px.index[base.fillna(False).values & rr.notna().values]
    e = declusters(s, h, px.index)
    a = summarize(rr.loc[e].values, f"h={h}")
    c = summarize(rr[(px.index >= s[0]) & (px.index <= s[-1])].values, "")
    a["ctrl_mean_pct"] = c["mean_pct"]
    a["excess_pct"] = a["mean_pct"] - c["mean_pct"]
    rows.append(a)
show(rows, "horizon sweep")
