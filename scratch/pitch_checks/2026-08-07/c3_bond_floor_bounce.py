"""C3: Bond floor bounce. Long TLT (and IEF as an alt vehicle) when TLT closes
within 1.5% of its 52w low AND ^TNX 63d return rank >= 85. Horizons 5 and 10 td.

^TNX IS present in master_prices (2000-present), so the rate proxy is the real
10y yield, not a bond-derived stand-in.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa
from _engine import battery, vehicle_ret

import numpy as np
import pandas as pd

px = close_panel(["TLT", "IEF", "^TNX", "SPY"]).dropna()
lo52 = px["TLT"].rolling(252).min()
off_lo = px["TLT"] / lo52 - 1.0
tnx63 = pct_rank(px["^TNX"], 63)


def trig(d: float = 0.015, rk: float = 85.0) -> pd.Series:
    return (off_lo <= d) & (tnx63 >= rk)


base = trig()
print(f"C3 trigger fires on 2026-08-06: {bool(base.iloc[-1])}  "
      f"(TLT {100*off_lo.iloc[-1]:.2f}% above 52w low, ^TNX rank63={tnx63.iloc[-1]:.1f})")
print(f"panel span {px.index[0].date()} .. {px.index[-1].date()}  (TLT/IEF start 2002-07-30)")

VAR = {
    "d<=1.0% rk>=85": trig(0.010, 85), "d<=1.5% rk>=85": trig(0.015, 85),
    "d<=2.0% rk>=85": trig(0.020, 85), "d<=3.0% rk>=85": trig(0.030, 85),
    "d<=1.5% rk>=80": trig(0.015, 80), "d<=1.5% rk>=90": trig(0.015, 90),
    "d<=1.5% NO rate gate": (off_lo <= 0.015) & tnx63.notna(),
    "rk>=85 NO floor gate": (tnx63 >= 85) & off_lo.notna(),
}

for h in (5, 10):
    battery(px, base, [("TLT", 1.0)], h, f"C3 TLT outright, {h}td", cost_bps=2, variants=VAR)

# --- IEF as the vehicle ------------------------------------------------------
print("\n\n########## C3-alt: IEF vehicle, same TLT-based trigger ##########")
for h in (5, 10):
    battery(px, base, [("IEF", 1.0)], h, f"C3-alt IEF outright, {h}td", cost_bps=2, variants=None)

# --- horizon sweep + excess, both vehicles -----------------------------------
print("\n--- horizon sweep, excess over same-span unconditional (episodes) ---")
rows = []
for tkr in ("TLT", "IEF"):
    for h in (1, 2, 3, 5, 10, 21, 42):
        rr = vehicle_ret(px, [(tkr, 1.0)], h, 1)
        s = px.index[base.fillna(False).values & rr.notna().values]
        if len(s) == 0:
            continue
        e = declusters(s, h, px.index)
        a = summarize(rr.loc[e].values, f"{tkr} h={h}")
        a["ctrl_pct"] = summarize(rr[(px.index >= s[0]) & (px.index <= s[-1])].values, "")["mean_pct"]
        a["excess_pct"] = a["mean_pct"] - a["ctrl_pct"]
        a["boot_p_le0"] = bootstrap_p_le0(rr.loc[e].values)
        rows.append(a)
show(rows, "horizon sweep")

# --- how clustered are these episodes really? --------------------------------
print("\n--- episode concentration by year (day-level trigger count) ---")
sig = px.index[base.fillna(False).values]
print(pd.Series(1, index=sig).groupby(sig.year).sum().to_string())
