"""Per-episode value tables for the two cells worth reporting (2026-09-09).

No new methodology: the masks and forms are copied from 03 (the drill's ^IRX x
^GSPC cell, ZIRP-removed, lag-0 context convention) and from 02/04 (VLO at a
252d high with z10 >= 1.5, lag-1 MOC). This script exists so the report can
quote individual episodes rather than only aggregates, and so the recency
concentration of the ^IRX cell is visible.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa: F401,F403

import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-09-09")

# --- cell 1: ^IRX 63d rank >= 95 x ^GSPC within 3% of 252d high, ^IRX >= 0.50
px = load_prices(["^IRX", "^GSPC", "^VIX"])
irx = px["^IRX"]["Close"].dropna()
gspc = px["^GSPC"]["Close"].dropna()
vix = px["^VIX"]["Close"].dropna()
r63 = pct_rank(px["^IRX"]["Close"], 63, 252)
near = gspc >= 0.97 * gspc.rolling(252).max()
trig = pd.DatetimeIndex(r63.index[(r63 >= 95.0).fillna(False).values])
trig = trig.intersection(pd.DatetimeIndex(near.index[near.fillna(False).values]))
trig = trig.intersection(pd.DatetimeIndex(irx.index[(irx >= 0.5).values]))
epi = declusters(trig.intersection(gspc.index), 10, gspc.index)

rows = []
for d in epi:
    rows.append({
        "episode": str(d.date()),
        "irx": round(float(irx.loc[d]), 3),
        "gspc_vs_252hi_%": round(100 * float(gspc.loc[d] / gspc.rolling(252).max().loc[d] - 1), 2),
        "vix_close": round(float(vix.loc[d]), 2) if d in vix.index else None,
        "vix_h1_%": round(100 * float(fwd_ret(vix, 1).loc[d]), 2),
        "vix_h5_%": round(100 * float(fwd_ret(vix, 5).loc[d]), 2) if pd.notna(fwd_ret(vix, 5).get(d)) else None,
        "vix_h10_%": round(100 * float(fwd_ret(vix, 10).loc[d]), 2) if pd.notna(fwd_ret(vix, 10).get(d)) else None,
        "gspc_h5_%": round(100 * float(fwd_ret(gspc, 5).loc[d]), 2) if pd.notna(fwd_ret(gspc, 5).get(d)) else None,
    })
print("=" * 96)
print("CELL 1 — ^IRX 63d rank >=95 x ^GSPC within 3% of its 252d high, ^IRX >= 0.50")
print("  lag-0 close-to-close (context convention). 18 episodes, 10td declustered.")
print("=" * 96)
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n  tonight {ASOF.date()} is itself a trigger session; the previous "
      f"independent episode was {epi[-1].date()} "
      f"({int(pd.Series(range(len(gspc.index)), index=gspc.index).loc[ASOF] - pd.Series(range(len(gspc.index)), index=gspc.index).loc[epi[-1]])} sessions ago), "
      "so tonight declusters into that episode rather than being a new one.")

# --- cell 2: VLO at a 252d high with z10 >= 1.5, lag-1 MOC ------------------
vlo = load_prices(["VLO"])["VLO"]
vlo = vlo[vlo.index <= ASOF]
c = vlo["Close"].astype(float)
z = c.pct_change(10) / (c.pct_change().rolling(21).std() * np.sqrt(10))
dh = c / c.rolling(252).max() - 1.0
cond = (dh >= 0.0) & (z >= 1.5)
t2 = vlo.index[cond.fillna(False).values]
t2 = t2[t2 < ASOF]
e2 = declusters(t2, 5, vlo.index)
m5 = (c.shift(-6) / c.shift(-1) - 1.0)
print("\n" + "=" * 96)
print("CELL 2 — VLO at a 252d high AND z10 >= 1.5, lag-1 MOC h=5 (entry close "
      "D+1, exit close D+6)")
print("=" * 96)
v = m5.reindex(e2).dropna()
print(f"  {len(v)} episodes.  by year:")
byy = pd.Series(100 * v.values, index=pd.DatetimeIndex(v.index).year)
tab = byy.groupby(level=0).agg(["count", "mean", "median", "min", "max"]).round(2)
print(tab.to_string())
worst = v.nsmallest(5)
best = v.nlargest(5)
print("\n  5 worst episodes: " + ", ".join(f"{d.date()} {100*x:+.2f}%" for d, x in worst.items()))
print("  5 best  episodes: " + ", ".join(f"{d.date()} {100*x:+.2f}%" for d, x in best.items()))
print(f"\n  most recent 8 episodes: "
      + ", ".join(f"{d.date()} {100*x:+.2f}%" for d, x in v.tail(8).items()))
print("\nDONE.")
