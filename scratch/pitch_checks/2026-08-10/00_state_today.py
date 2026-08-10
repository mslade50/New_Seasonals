"""Pin every gate threshold to TODAY's actual value before any cell is measured.
A rank gate is not a magnitude gate (registry). This prints both."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

T = ["SPY", "TLT", "IEF", "^TNX", "^MOVE", "HYG", "LQD", "XLU", "^VIX"]
px = close_panel(T)
print("last bar:", px.index[-1].date())

for t in T:
    s = px[t].dropna()
    lo52 = s.rolling(252).min()
    hi52 = s.rolling(252).max()
    row = {
        "last": s.iloc[-1],
        "dist_52w_low_pct": 100 * (s.iloc[-1] / lo52.iloc[-1] - 1),
        "dist_52w_high_pct": 100 * (s.iloc[-1] / hi52.iloc[-1] - 1),
        "z10": zscore(s, 10).iloc[-1],
        "rank5d": pct_rank(s, 5).iloc[-1],
        "rank21d": pct_rank(s, 21).iloc[-1],
        "rank63d": pct_rank(s, 63).iloc[-1],
        "lvl_pctile_252": 100 * float((s.iloc[-252:] <= s.iloc[-1]).mean()),
        "ret5d_pct": 100 * (s.iloc[-1] / s.iloc[-6] - 1),
    }
    print(f"{t:7s} " + "  ".join(f"{k}={v:.2f}" for k, v in row.items()))

# how many bp of yield does ^TNX rank5d 15.9 correspond to today?
tnx = px["^TNX"].dropna()
d5 = tnx.diff(5)
print(f"\n^TNX 5d change today = {d5.iloc[-1]:+.3f} (index pts = 10x pct)"
      f"  -> {10*d5.iloc[-1]:+.1f} bps")
print("  distribution of 5d change on days with rank5d<20 (trailing 252 rank):")
r5 = pct_rank(tnx, 5)
sel = (r5 < 20) & d5.notna()
print(f"    n={int(sel.sum())} median {10*d5[sel].median():+.1f}bps  "
      f"today's {10*d5.iloc[-1]:+.1f}bps sits at the "
      f"{100*float((d5[sel] <= d5.iloc[-1]).mean()):.1f}th pctile of that cell")
