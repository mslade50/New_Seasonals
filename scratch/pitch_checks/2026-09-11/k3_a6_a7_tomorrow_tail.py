"""Round-1 item 6 for A6 and A7: the LIVE hold entered 2026-09-11 contains the
FOMC decision (2026-09-16, +3 td) and quad witching (2026-09-18, +5 td). Split
both cells on whether those events landed inside the hold historically.
"""
import sys
from pathlib import Path

ROOTP = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOTP))
import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
from pitch_lab import *  # noqa: E402,F403

H = 5
px = close_panel(["SPY", "IWM", "TLT", "IEF", "LQD", "DBC"])
cal = px["SPY"].dropna().index
px = px.reindex(cal)
raw = load_prices(["^TNX"])["^TNX"]["Close"]
tnx = raw.reindex(cal).ffill()
at_max = (tnx >= rolling_on_valid(tnx, lambda x: x.rolling(252).max()) - 1e-9)


def near_high(t, p):
    s = px[t]
    mx = rolling_on_valid(s, lambda x: x.rolling(252).max())
    return ((mx - s) / mx * 100 <= p) & mx.notna()


def near_low(t, p):
    s = px[t]
    mn = rolling_on_valid(s, lambda x: x.rolling(252).min())
    return ((s - mn) / mn * 100 <= p) & mn.notna()


CELLS = [
    ("A6 SPY+1/IWM-1, ^TNX at 252d max", at_max,
     [("SPY", 1.0), ("IWM", -1.0)]),
    ("A7 SHORT SPY, joint inflation state",
     near_high("DBC", 0.5) & near_low("IEF", 1.0) & near_low("LQD", 1.0),
     [("SPY", -1.0)]),
    ("A7 LONG IWM, joint inflation state",
     near_high("DBC", 0.5) & near_low("IEF", 1.0) & near_low("LQD", 1.0),
     [("IWM", 1.0)]),
]

for title, m, legs in CELLS:
    r = vehicle_ret(px, legs, H, 1)
    sig = cal[m.reindex(cal, fill_value=False).values].intersection(
        r.dropna().index)
    epi = declusters(sig, 21, cal)          # 21td independence, not h
    v = r.loc[epi].values
    print("\n" + "=" * 78)
    print(f"{title}   h={H}, episodes declustered at 21 td (N={len(v)})")
    if len(v) == 0:
        print("  no episodes")
        continue
    for kinds in [("fomc_decision",), ("quad_witching",), ("opex",),
                  ("fomc_decision", "quad_witching")]:
        fl = event_in_window(epi, cal, H, 1, kinds)
        lab = "+".join(kinds)
        if fl.sum() == 0:
            print(f"  {lab:<30} 0 of {len(fl)} episodes -- NO historical "
                  f"instance of the live configuration")
            continue
        wi = int((v[fl] > 0).sum())
        wo = int((v[~fl] > 0).sum())
        print(f"  {lab:<30} IN  N={int(fl.sum()):>3} "
              f"{100*v[fl].mean():+.3f}%  record {wi}-{int(fl.sum())-wi}  "
              f"sign p {sign_test(wi, int(fl.sum())):.4f}")
        print(f"  {'':<30} OUT N={int((~fl).sum()):>3} "
              f"{100*v[~fl].mean():+.3f}%  record {wo}-{int((~fl).sum())-wo}")
    print(f"  worst episode {100*v.min():+.2f}% on "
          f"{epi[int(np.argmin(v))].date()}")
print("\nDONE")
