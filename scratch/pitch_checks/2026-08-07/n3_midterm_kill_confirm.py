"""N3: is the midterm zero real, or is it the CPI-inside cell in disguise?

N2 found the NFP x rates-floor cell survives everything EXCEPT the conditioner
that describes today. Midterm N=12 gives +0.071% (t=0.17); non-midterm N=13
gives +0.978% (t=2.72). Separately, CPI-inside-the-hold gave +1.614% on N=5
with a 100% hit rate, and CPI IS inside today's hold.

If those 5 CPI-inside observations are all non-midterm, the CPI split is the
midterm split wearing a different label and today's cell is still dead. If
they straddle, there is a live sub-cell. This decides the morning.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa

import numpy as np
import pandas as pd

PX = close_panel(["TLT", "IEF", "XLU"]).dropna(subset=["TLT"])
CAL = PX.index
POS = pd.Series(range(len(CAL)), index=CAL)
EV = load_events()
NFP = [d for d in EV.loc[EV.event == "nfp", "date"] if d in POS.index]
CPI = set(EV.loc[EV.event == "cpi", "date"])
FLOOR = 100.0 * (PX["TLT"] / PX["TLT"].rolling(252).min() - 1.0)

H, GATE = 3, 3.0
sub = [d for d in NFP if FLOOR.get(d, np.nan) <= GATE and POS[d] + H < len(CAL)]

rows = []
for d in sub:
    p = POS[d]
    rows.append({
        "date": d.date(), "year": d.year,
        "midterm": d.year % 4 == 2,
        "cpi_in": any(x in CPI for x in CAL[p + 1: p + H + 1]),
        "tlt_ret": 100 * (PX["TLT"].iloc[p + H] / PX["TLT"].iloc[p] - 1.0),
        "floor_pct": round(FLOOR[d], 2),
    })
df = pd.DataFrame(rows)

print("=" * 92)
print("THE 2x2: midterm x CPI-inside-the-hold")
print("=" * 92)
print(df.to_string(index=False))

print("\n--- cell means ---")
cells = []
for mt in (True, False):
    for ci in (True, False):
        v = df[(df.midterm == mt) & (df.cpi_in == ci)]["tlt_ret"].values
        cells.append({"midterm": mt, "cpi_in": ci, "n": len(v),
                      "mean_pct": round(v.mean(), 3) if len(v) else None,
                      "hit": round(100 * (v > 0).mean(), 1) if len(v) else None})
print(pd.DataFrame(cells).to_string(index=False))

print("\n" + "=" * 92)
print("VERDICT LOGIC")
print("=" * 92)
mt_cpi = df[(df.midterm) & (df.cpi_in)]
print(f"  Today's exact cell is midterm=True, cpi_in=True.")
print(f"  Historical occurrences of that exact cell: N = {len(mt_cpi)}")
if len(mt_cpi):
    print(f"    dates: {[str(x) for x in mt_cpi.date.tolist()]}")
    print(f"    returns: {[round(x, 3) for x in mt_cpi.tlt_ret.tolist()]}")
    print(f"    mean: {mt_cpi.tlt_ret.mean():+.3f}%")

cpi_rows = df[df.cpi_in]
print(f"\n  CPI-inside observations: N={len(cpi_rows)}, "
      f"midterm share {cpi_rows.midterm.sum()}/{len(cpi_rows)}")
print(f"  If that share is 0, the CPI split IS the midterm split relabelled.")

print("\n--- midterm cell, every horizon (is ANY horizon alive in midterms?) ---")
out = []
for h in (1, 2, 3, 4, 5, 6, 10):
    v = np.array([100 * (PX["TLT"].iloc[POS[d] + h] / PX["TLT"].iloc[POS[d]] - 1.0)
                  for d in sub if d.year % 4 == 2 and POS[d] + h < len(CAL)])
    out.append(summarize(v / 100, f"midterm +{h}td"))
show(out, "")

print("\n--- same, XLU (the stronger leg in N1) in midterms only ---")
out = []
for h in (3, 5):
    v = np.array([100 * (PX["XLU"].iloc[POS[d] + h] / PX["XLU"].iloc[POS[d]] - 1.0)
                  for d in sub if d.year % 4 == 2 and POS[d] + h < len(CAL)])
    out.append(summarize(v / 100, f"XLU midterm +{h}td"))
show(out, "")
