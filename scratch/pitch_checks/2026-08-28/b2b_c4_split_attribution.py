"""C4 round 2 follow-up: the one sub-cell that looked alive.

Section 5 of b2 found the C4 cell splits by 5d rank the way watchlist 30
predicted -- 5d rank < 15 pays +1.437% (N=53, t 2.15) while 5d rank >= 25 pays
+0.619%. Two questions that decide whether that is anything:
  (a) does the 63d-rank gate do work INSIDE the 5d<15 sub-cell, or is
      "21d thrust + 5d pullback" the whole story (gate attribution again)?
  (b) EEM is at 5d rank 63.1 today, so which side is the LIVE instance on?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from b0_pool import pooled, series  # noqa
from pitch_lab import load_prices, pct_rank, show  # noqa

H, MIN_GAP = 10, 10
FAM = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "EWJ", "FXI", "EWZ",
       "XLK", "XLV", "XLF", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE", "XLC",
       "SMH", "XBI", "IBB", "KRE", "IHI", "ITB", "XME", "XLE", "XOP", "OIH"]
px = load_prices(FAM)

rows = []
for lbl, fn in [
    ("21d>=90 & 63d<=10 & 5d<15  (C4 sub-cell)",
     lambda s: (pct_rank(s, 21) >= 90) & (pct_rank(s, 63) <= 10) & (pct_rank(s, 5) < 15)),
    ("21d>=90 & 5d<15   (drop the 63d gate)",
     lambda s: (pct_rank(s, 21) >= 90) & (pct_rank(s, 5) < 15)),
    ("21d>=90 & 63d>10 & 5d<15  (complement)",
     lambda s: (pct_rank(s, 21) >= 90) & (pct_rank(s, 63) > 10) & (pct_rank(s, 5) < 15)),
    ("5d<15 alone",
     lambda s: pct_rank(s, 5) < 15),
    ("21d>=90 & 63d<=10 & 5d>=25 (WHERE EEM IS)",
     lambda s: (pct_rank(s, 21) >= 90) & (pct_rank(s, 63) <= 10) & (pct_rank(s, 5) >= 25)),
    ("21d>=90 & 5d>=25  (drop the 63d gate)",
     lambda s: (pct_rank(s, 21) >= 90) & (pct_rank(s, 5) >= 25)),
]:
    p = pooled(px, FAM, fn, H, MIN_GAP, lbl)
    rows.append({k: v for k, v in p.items() if not k.startswith("_")})
show(rows, "gate attribution inside the 5d-rank split (pooled episodes, h=10)")

a = rows[0]["mean_pct"]
b = rows[1]["mean_pct"]
print("\n  63d-gate delta inside 5d<15 : %+.3f pp (%.3f vs %.3f)" % (a - b, a, b))
c, d = rows[4]["mean_pct"], rows[5]["mean_pct"]
print("  63d-gate delta inside 5d>=25: %+.3f pp (%.3f vs %.3f)" % (c - d, c, d))
eem = series(px, "EEM")
print("\n  LIVE EEM 2026-08-27: 5d rank %.1f -> lands in the 5d>=25 cell"
      % float(pct_rank(eem, 5).iloc[-1]))
print("  the 5d rank would have to fall below 15 for the live instance to sit "
      "in the sub-cell that pays; today it is %.1f"
      % float(pct_rank(eem, 5).iloc[-1]))
