"""C2: Intra-defensive spread. Long XLU / short XLP equal dollar, 5 td.

Trigger on close of D: (XLU 21d ret - XLP 21d ret) <= -4pp AND XLU z10 <= -1.5.
Entry MOC D+1, exit MOC D+6.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa
from _engine import battery, vehicle_ret

import numpy as np
import pandas as pd

H = 5
LEGS = [("XLU", 1.0), ("XLP", -1.0)]
px = close_panel(["SPY", "XLU", "XLP"]).dropna()
z = zscore(px["XLU"], 10)
sp21 = px["XLU"].pct_change(21) - px["XLP"].pct_change(21)


def trig(spt: float = -0.04, zt: float = -1.5) -> pd.Series:
    return (sp21 <= spt) & (z <= zt)


base = trig()
print(f"C2 trigger fires on 2026-08-06: {bool(base.iloc[-1])}  "
      f"(spread21={100*sp21.iloc[-1]:.2f}pp, XLU z10={z.iloc[-1]:.3f})")

battery(px, base, LEGS, H, "C2 XLU-XLP spread, 5td", cost_bps=2,
        variants={
            "sp<=-3pp z<=-1.5": trig(-0.03, -1.5),
            "sp<=-4pp z<=-1.5": trig(-0.04, -1.5),
            "sp<=-5pp z<=-1.5": trig(-0.05, -1.5),
            "sp<=-6pp z<=-1.5": trig(-0.06, -1.5),
            "sp<=-4pp z<=-1.25": trig(-0.04, -1.25),
            "sp<=-4pp z<=-1.75": trig(-0.04, -1.75),
            "sp<=-4pp z<=-2.0": trig(-0.04, -2.0),
            "sp<=-4pp NO z gate": (sp21 <= -0.04) & z.notna(),
            "z<=-1.5 NO spread gate": (z <= -1.5),
        })

# --- is the spread just XLU mean reversion with an XLP hedge that costs? -----
print("\n--- leg attribution on the SAME episodes (h=5, lag=1) ---")
sig = px.index[base.fillna(False).values]
r_pair = vehicle_ret(px, LEGS, H, 1)
sig = px.index[base.fillna(False).values & r_pair.notna().values]
epi = declusters(sig, H, px.index)
rows = [summarize(vehicle_ret(px, [("XLU", 1.0)], H, 1).loc[epi].values, "long XLU leg only"),
        summarize(vehicle_ret(px, [("XLP", 1.0)], H, 1).loc[epi].values, "XLP leg (as long)"),
        summarize(r_pair.loc[epi].values, "PAIR XLU-XLP"),
        summarize((vehicle_ret(px, LEGS, H, 1))[px.index >= sig[0]].values, "PAIR unconditional, same span")]
show(rows, "leg attribution")

# --- horizon sweep + excess vs same-span control -----------------------------
print("\n--- horizon sweep (episodes) ---")
rows = []
for h in (1, 2, 3, 5, 10, 21):
    rr = vehicle_ret(px, LEGS, h, 1)
    s = px.index[base.fillna(False).values & rr.notna().values]
    e = declusters(s, h, px.index)
    a = summarize(rr.loc[e].values, f"h={h}")
    a["ctrl_mean_pct"] = summarize(rr[(px.index >= s[0]) & (px.index <= s[-1])].values, "")["mean_pct"]
    a["excess_pct"] = a["mean_pct"] - a["ctrl_mean_pct"]
    a["boot_p_le0"] = bootstrap_p_le0(rr.loc[e].values)
    rows.append(a)
show(rows, "horizon sweep")
