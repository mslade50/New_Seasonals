"""Data probe for C8 / C9 / C11 -- history spans, current state, sanity."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["^SKEW", "^VIX", "SPY", "^GSPC", "EFA", "HYG", "IWM", "EWJ", "LQD",
      "DX-Y.NYB", "UUP", "SVXY", "UVXY", "TLT"]
px = load_prices(TK)
for t in TK:
    if t not in px:
        continue
    s = px[t]["Close"].dropna()
    print(f"{t:12s} n={len(s):6d}  {s.index[0].date()} .. {s.index[-1].date()}"
          f"  last={s.iloc[-1]:.4f}")

# gaps in ^SKEW
if "^SKEW" in px:
    s = px["^SKEW"]["Close"].dropna()
    sp = px["SPY"]["Close"].dropna()
    common = s.index.intersection(sp.index)
    print(f"\n^SKEW vs SPY common sessions: {len(common)}  "
          f"({common[0].date()} .. {common[-1].date()})")
    # coverage by year
    print(s.groupby(s.index.year).size().to_string())

# put/call cache
import pc_fear  # noqa
pcs = pc_fear.pct_series()
print(f"\nP/C pctile series: n={len(pcs)}  {pcs.index[0].date()} .. "
      f"{pcs.index[-1].date()}  last={pcs.iloc[-1]:.1f}")

# today's state values
p = close_panel(["^SKEW", "SPY", "^VIX", "EFA", "HYG", "IWM", "EWJ",
                 "DX-Y.NYB"])
p = p.dropna(how="all")
print("\nlast 3 rows:")
print(p.tail(3).to_string())

sk = p["^SKEW"].dropna()
sk_r21 = pct_rank(sk, 21)
sk_lvl_r = sk.rolling(252).rank(pct=True) * 100
print(f"\n^SKEW last={sk.iloc[-1]:.2f}  ret21-rank={sk_r21.iloc[-1]:.1f}  "
      f"LEVEL 252d pctile={sk_lvl_r.iloc[-1]:.1f}")

spy = p["SPY"].dropna()
hi = spy.rolling(252).max()
print(f"SPY dist from 252d high = {100*(spy.iloc[-1]/hi.iloc[-1]-1):.3f}%")
for t in ["EFA", "HYG", "IWM", "EWJ"]:
    s = p[t].dropna()
    h = s.rolling(252).max()
    print(f"{t} dist 252d high = {100*(s.iloc[-1]/h.iloc[-1]-1):.3f}%")

dx = p["DX-Y.NYB"].dropna()
print(f"\nDX z10={zscore(dx,10).iloc[-1]:.2f}  rank21={pct_rank(dx,21).iloc[-1]:.1f}"
      f"  rank63={pct_rank(dx,63).iloc[-1]:.1f}  "
      f"dist52wh={100*(dx.iloc[-1]/dx.rolling(252).max().iloc[-1]-1):.3f}%")
