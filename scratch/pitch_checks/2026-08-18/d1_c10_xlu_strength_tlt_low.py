"""C10 -- short XLU when utilities are bid (5d rank >= 85) while TLT sits at a
52-week low.

Written to KILL. Order of operations:
  0. trigger sanity against the recon (XLU rank5 88.1, TLT 0.00% off 52w low)
  1. OVERLAP with the six dead utilities expressions, computed not asserted
  2. GATE ATTRIBUTION -- XLU-strength alone, TLT-at-low alone, joint. If the
     joint does not beat BOTH singles the "divergence" carries nothing.
  3. horizon 1..10, declustered episodes only
  4. full battery at the best horizon (controls, era split, concentration)
  5. threshold neighbours on both legs
  6. DIRECTION -- is the resolution XLU falling or TLT rising? measure the TLT
     leg on the same days.
  7. cost on a sector-ETF short.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TKRS = ["XLU", "TLT", "IEF", "SPY", "XLP"]
px = close_panel(TKRS)
idx = px.index

rank5_xlu = pct_rank(px["XLU"], 5)
rank21_xlu = pct_rank(px["XLU"], 21)
z10_xlu = zscore(px["XLU"], 10)
tlt_low = px["TLT"] / px["TLT"].rolling(252).min() - 1.0
ief_low = px["IEF"] / px["IEF"].rolling(252).min() - 1.0
spy_high = px["SPY"] / px["SPY"].rolling(252).max() - 1.0

print("=== 0. TRIGGER SANITY (must match 01_live_state_recon.txt) ===")
print(f"  last bar          : {idx[-1].date()}")
print(f"  XLU pct_rank(5)   : {rank5_xlu.iloc[-1]:.1f}   (recon r5 88.1)")
print(f"  XLU pct_rank(21)  : {rank21_xlu.iloc[-1]:.1f}   (recon r21 29.4)")
print(f"  TLT above 52w low : {100*tlt_low.iloc[-1]:.2f}%  (recon 0.00%)")
print(f"  IEF above 52w low : {100*ief_low.iloc[-1]:.2f}%  (recon 0.73%)")
print(f"  SPY off 52w high  : {100*spy_high.iloc[-1]:.2f}% (recon -0.67%)")


def mk(xlu_thr: float = 85.0, tlt_thr: float = 0.01) -> pd.Series:
    return ((rank5_xlu >= xlu_thr) & (tlt_low <= tlt_thr)).reindex(
        idx, fill_value=False).fillna(False)


A = (rank5_xlu >= 85).reindex(idx, fill_value=False).fillna(False)   # XLU bid
B = (tlt_low <= 0.01).reindex(idx, fill_value=False).fillna(False)   # TLT at low
J = (A & B)

# common valid window: both legs need 252d history + TLT exists from 2002-07
valid = rank5_xlu.notna() & tlt_low.notna()
A, B, J = A & valid, B & valid, J & valid
print(f"\ntrigger counts (days): XLU-strength {int(A.sum())}  "
      f"TLT-at-low {int(B.sum())}  JOINT {int(J.sum())}")
if int(J.sum()):
    jd = idx[J.values]
    print(f"  joint span {jd[0].date()} .. {jd[-1].date()}")
    print("  joint years:", dict(pd.Series(jd.year).value_counts().sort_index()))
print(f"  fires today: {bool(J.iloc[-1])}")

# ---------------------------------------------------------------------------
# 1. OVERLAP with the six dead utilities expressions
# ---------------------------------------------------------------------------
print("\n=== 1. OVERLAP with the dead utilities corpses (day level) ===")
corpses = {
    "washout z10<=-2 (2026-08-07 C1)": (z10_xlu <= -2.0),
    "washout z10<=-2 & SPY nr high": (z10_xlu <= -2.0) & (spy_high >= -0.015),
    "rank21<=5 washout (2026-08-12)": (rank21_xlu <= 5),
    "any XLU weakness rank5<=15": (rank5_xlu <= 15),
}
jset = set(idx[J.values])
for lbl, m in corpses.items():
    ms = set(idx[m.reindex(idx, fill_value=False).fillna(False).values & valid.values])
    inter = len(jset & ms)
    print(f"  {lbl:38s} N={len(ms):4d}  overlap with joint = {inter} "
          f"({100*inter/max(len(jset),1):.1f}% of joint days)")

# ---------------------------------------------------------------------------
# 2. GATE ATTRIBUTION -- the whole test
# ---------------------------------------------------------------------------
print("\n=== 2. GATE ATTRIBUTION: short XLU, episodes only, h=1..10 ===")
LEGS = [("XLU", -1.0)]
for lbl, m in [("XLU-strength ALONE", A), ("TLT-at-low ALONE", B),
               ("JOINT (divergence)", J)]:
    d = idx[m.values]
    print(f"\n-- {lbl}  (N days {len(d)})")
    show(horizon_scan(px, d, LEGS, hs=(1, 2, 3, 5, 10)), "")

# unconditional short-XLU drift over the joint span, for the honest control
if int(J.sum()):
    jd = idx[J.values]
    span = (idx >= jd[0]) & (idx <= jd[-1])
    rows = []
    for h in (1, 2, 3, 5, 10):
        r = vehicle_ret(px, LEGS, h, 1)
        rows.append(summarize(r[span].values, f"short-XLU drift h={h}, joint span"))
    show(rows, "CTRL: unconditional short XLU over the joint span")

# ---------------------------------------------------------------------------
# 3+4. full battery at h=5 and h=3
# ---------------------------------------------------------------------------
for H in (3, 5):
    battery(px, J, LEGS, H, f"C10 short XLU joint, h={H}", cost_bps=10,
            variants={
                "xlu>=80 & tlt<=1%": mk(80, 0.01),
                "xlu>=85 & tlt<=1%": mk(85, 0.01),
                "xlu>=90 & tlt<=1%": mk(90, 0.01),
                "xlu>=95 & tlt<=1%": mk(95, 0.01),
                "xlu>=85 & tlt<=0.5%": mk(85, 0.005),
                "xlu>=85 & tlt<=2%": mk(85, 0.02),
                "xlu>=85 & tlt<=3%": mk(85, 0.03),
                "xlu>=85 & tlt<=5%": mk(85, 0.05),
            })

# ---------------------------------------------------------------------------
# 6. DIRECTION -- what actually resolves?
# ---------------------------------------------------------------------------
print("\n=== 6. DIRECTION CHECK on the same joint episodes ===")
if int(J.sum()):
    jd = idx[J.values]
    for H in (3, 5, 10):
        epi = declusters(jd, H, idx)
        rows = []
        for lbl, legs in [("short XLU", [("XLU", -1.0)]),
                          ("long TLT", [("TLT", 1.0)]),
                          ("long IEF", [("IEF", 1.0)]),
                          ("short XLU vs short SPY (resid)",
                           [("XLU", -1.0), ("SPY", 1.0)]),
                          ("short XLU vs short XLP (resid)",
                           [("XLU", -1.0), ("XLP", 1.0)]),
                          ("long TLT vs long XLU (the divergence closing)",
                           [("TLT", 1.0), ("XLU", -1.0)])]:
            r = vehicle_ret(px, legs, H, 1)
            rows.append(summarize(r.loc[epi.intersection(r.dropna().index)].values,
                                  f"h={H} {lbl}"))
        # unconditional controls for the two single legs
        for lbl, legs in [("h=%d short XLU ALL DAYS" % H, [("XLU", -1.0)]),
                          ("h=%d long TLT ALL DAYS" % H, [("TLT", 1.0)])]:
            r = vehicle_ret(px, legs, H, 1)
            rows.append(summarize(r.dropna().values, lbl))
        show(rows, f"direction at h={H} (episodes, min_gap={H})")
