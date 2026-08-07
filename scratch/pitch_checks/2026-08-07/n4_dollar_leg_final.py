"""N4: the last open cell. Short dollar on the NFP/rates-floor trigger.

N2 gave UUP short +0.361% at h=5, 61.9% hit, t=1.331, 9x cost. Weak but not
absurd, and it is the mirror of the bond leg rather than an independent bet.
The bond leg died on the midterm conditioner. If the dollar leg dies the same
way, the whole NFP x rates complex is closed for today and the morning is a
stand-down.

Also sweeps DX-Y.NYB directly (the futures proxy McKinley can trade) since
UUP carries an expense drag the futures leg does not.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import numpy as np
import pandas as pd

PX = close_panel(["TLT", "UUP", "DX-Y.NYB"]).dropna(subset=["TLT"])
CAL = PX.index
POS = pd.Series(range(len(CAL)), index=CAL)
EV = load_events()
NFP = [d for d in EV.loc[EV.event == "nfp", "date"] if d in POS.index]
FLOOR = 100.0 * (PX["TLT"] / PX["TLT"].rolling(252).min() - 1.0)
GATE = 3.0
sub = [d for d in NFP if FLOOR.get(d, np.nan) <= GATE]


def fwd(sym, d, h):
    p = POS[d]
    if p + h >= len(CAL) or pd.isna(PX[sym].iloc[p]):
        return np.nan
    return PX[sym].iloc[p + h] / PX[sym].iloc[p] - 1.0


for sym, cost in (("UUP", 0.04), ("DX-Y.NYB", 0.015)):
    print("\n" + "=" * 92)
    print(f"{sym}  SHORT on the NFP + TLT-at-floor trigger   (cost {cost:.3f}% round trip)")
    print("=" * 92)
    for h in (3, 5):
        v = np.array([fwd(sym, d, h) for d in sub])
        keep = ~np.isnan(v)
        dts = pd.DatetimeIndex([d for d, k in zip(sub, keep) if k])
        v = -v[keep]                       # SHORT side
        if len(v) < 5:
            print(f"  h={h}: N={len(v)} too few")
            continue
        s = summarize(v, f"{sym} short +{h}td  (N={len(v)})")
        s["p_le0_boot"] = bootstrap_p_le0(v)
        s["x_cost"] = round(100 * v.mean() / cost, 1)
        show([s], "")
        mid = [i for i, d in enumerate(dts) if d.year % 4 == 2]
        non = [i for i, d in enumerate(dts) if d.year % 4 != 2]
        show([summarize(v[mid], f"  midterm (N={len(mid)})"),
              summarize(v[non], f"  non-midterm (N={len(non)})")], "")
        for e in era_split(dts, v):
            print(f"      era {e['label']:<10} n={e['n']:<4} "
                  f"mean={e['mean_pct']:+.3f} t={e['t']:+.2f}")

print("\n" + "=" * 92)
print("CLOSING THE COMPLEX")
print("=" * 92)
print("  Bond leg  (TLT h=3): midterm +0.071%, t=0.17, N=12   -> dead today")
print("  Equity leg (XLU h=3): midterm -0.538%, t=-0.83, N=12 -> wrong sign today")
print("  Today's exact cell (midterm AND CPI-inside): N=1, 2022-05-06")
print("  If the dollar leg is also midterm-dead, every expression of the cell")
print("  the AM run missed is closed, and the correct verdict is a stand-down.")
