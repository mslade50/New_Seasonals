"""Does each candidate's trigger ACTUALLY fire on the 2026-08-06 close?

Recomputes z10 / ranks / 52w distances from master_prices directly.
No trust in the pitch-state summary numbers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import pandas as pd

TK = ["SPY", "XLU", "XLP", "TLT", "IEF", "^TNX"]
px = close_panel(TK)
px = px.dropna()  # common trading calendar
D = pd.Timestamp("2026-08-06")
assert px.index[-1] == D, f"last bar is {px.index[-1].date()}, expected 2026-08-06"

z10 = {t: zscore(px[t], 10) for t in TK}
r5 = {t: pct_rank(px[t], 5) for t in TK}
r21 = {t: pct_rank(px[t], 21) for t in TK}
r63 = {t: pct_rank(px[t], 63) for t in TK}
hi52 = {t: px[t].rolling(252).max() for t in TK}
lo52 = {t: px[t].rolling(252).min() for t in TK}

print("=== recomputed levels at 2026-08-06 (my numbers, not the brief's) ===")
rows = []
for t in TK:
    rows.append(dict(
        tkr=t, close=px[t].loc[D],
        z10=z10[t].loc[D], rank5=r5[t].loc[D], rank21=r21[t].loc[D], rank63=r63[t].loc[D],
        ret5_pct=100 * (px[t].loc[D] / px[t].shift(5).loc[D] - 1),
        ret21_pct=100 * (px[t].loc[D] / px[t].shift(21).loc[D] - 1),
        pct_below_52wh=100 * (px[t].loc[D] / hi52[t].loc[D] - 1),
        pct_above_52wl=100 * (px[t].loc[D] / lo52[t].loc[D] - 1),
    ))
show(rows, "levels")

print("\n=== trigger evaluation ===")
xlu_z = z10["XLU"].loc[D]
spy_off_hi = 100 * (px["SPY"].loc[D] / hi52["SPY"].loc[D] - 1)
c1 = (xlu_z <= -2.0) and (spy_off_hi >= -1.5)
print(f"C1/C4: XLU z10={xlu_z:.3f} <= -2.0 ? {xlu_z <= -2.0}   "
      f"SPY {spy_off_hi:.2f}% off 52wh, within 1.5% ? {spy_off_hi >= -1.5}")
print(f"  -> C1/C4 FIRES: {c1}")

sp21 = (px["XLU"].loc[D] / px["XLU"].shift(21).loc[D] - 1) - (px["XLP"].loc[D] / px["XLP"].shift(21).loc[D] - 1)
c2 = (100 * sp21 <= -4.0) and (xlu_z <= -1.5)
print(f"C2: XLU21-XLP21 = {100*sp21:.2f}pp <= -4pp ? {100*sp21 <= -4.0}   "
      f"XLU z10 <= -1.5 ? {xlu_z <= -1.5}")
print(f"  -> C2 FIRES: {c2}")

tlt_off_lo = 100 * (px["TLT"].loc[D] / lo52["TLT"].loc[D] - 1)
ief_off_lo = 100 * (px["IEF"].loc[D] / lo52["IEF"].loc[D] - 1)
tnx63 = r63["^TNX"].loc[D]
c3 = (tlt_off_lo <= 1.5) and (tnx63 >= 85)
print(f"C3: TLT {tlt_off_lo:.2f}% above 52w low, within 1.5% ? {tlt_off_lo <= 1.5}   "
      f"^TNX rank63={tnx63:.1f} >= 85 ? {tnx63 >= 85}")
print(f"  (IEF is {ief_off_lo:.2f}% above its own 52w low)")
print(f"  -> C3 FIRES: {c3}")

print("\n=== brief-vs-recompute deltas (sanity) ===")
brief = {"SPY_rank5": 96, "SPY_rank21": 75.4, "SPY_z10": 1.46, "XLU_z10": -2.31,
         "XLP_z10": 0.59, "TNX_rank63": 87.7, "IEF_rank63": 13.1, "TLT_rank63": 11.1}
mine = {"SPY_rank5": r5["SPY"].loc[D], "SPY_rank21": r21["SPY"].loc[D], "SPY_z10": z10["SPY"].loc[D],
        "XLU_z10": z10["XLU"].loc[D], "XLP_z10": z10["XLP"].loc[D], "TNX_rank63": r63["^TNX"].loc[D],
        "IEF_rank63": r63["IEF"].loc[D], "TLT_rank63": r63["TLT"].loc[D]}
for k in brief:
    print(f"  {k:12s} brief={brief[k]:7.2f}  mine={mine[k]:7.2f}  d={mine[k]-brief[k]:+.2f}")
