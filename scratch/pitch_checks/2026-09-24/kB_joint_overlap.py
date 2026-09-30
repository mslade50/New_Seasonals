"""kB joint: D1, G1 and E1 all fire on 09-23. On the historical sessions where
more than one fires, how correlated are their outcomes? One macro bet or three?

D1 = long DX-Y.NYB; G1 = short GLD; E1 = short EEM vs ex-ante beta-SPY.
All use the -0.00505 DX near-high threshold (today misses the literal 0.5% by 0.2 bp).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

TK = ["DX-Y.NYB", "GLD", "EEM", "SPY", "^TNX"]
raw = load_prices(TK)
cl = {t: raw[t]["Close"].dropna() for t in TK}

dx = cl["DX-Y.NYB"]
dx_off = dx / dx.rolling(252).max() - 1.0
dx1 = dx.pct_change()
r1 = dx.pct_change()
z10 = (dx / dx.shift(10) - 1.0) / (r1.rolling(21).std() * np.sqrt(10))
tnx = cl["^TNX"]
tnx_hi = tnx >= tnx.rolling(252).max() - 1e-12
near = dx_off >= -0.00505

# common calendar: sessions where all four trade
px = pd.DataFrame({t: cl[t] for t in ["DX-Y.NYB", "GLD", "EEM", "SPY"]}).dropna()
idx = px.index
rr = px.pct_change()
beta = rr["EEM"].rolling(252).cov(rr["SPY"]) / rr["SPY"].rolling(252).var()

D1 = ((z10 >= 2.0) & near).reindex(idx).fillna(False).astype(bool)
G1 = ((px["GLD"].pct_change() <= -0.015) & tnx_hi.reindex(idx).fillna(False).astype(bool)
      & near.reindex(idx).fillna(False).astype(bool))
E1 = ((dx1 >= 0.004) & near).reindex(idx).fillna(False).astype(bool)
print("fires on last session:", idx[-1].date(), "D1", bool(D1.iloc[-1]), "G1", bool(G1.iloc[-1]),
      "E1", bool(E1.iloc[-1]))
print(f"trigger days (common calendar): D1 {D1.sum()}  G1 {G1.sum()}  E1 {E1.sum()}  "
      f"D1&E1 {(D1&E1).sum()}  D1&G1 {(D1&G1).sum()}  G1&E1 {(G1&E1).sum()}  all3 {(D1&G1&E1).sum()}")

for h in (1, 5):
    out = pd.DataFrame({
        "D1_longDX": fwd_lag(px["DX-Y.NYB"], h, 1),
        "G1_shortGLD": -fwd_lag(px["GLD"], h, 1),
        "E1_shortEEMres": -fwd_lag(px["EEM"], h, 1) + beta * fwd_lag(px["SPY"], h, 1),
    })
    multi = (D1.astype(int) + G1.astype(int) + E1.astype(int)) >= 2
    co = out.loc[multi.values].dropna()
    print(f"\n=== h={h}: sessions with >=2 firing: {len(co)} ===")
    print("  pairwise corr of outcomes on co-fire sessions:")
    print(co.corr().round(3).to_string())
    print("  same corr on ALL days (baseline):")
    print(out.dropna().corr().round(3).to_string())
    show([summarize(co[c].values, c) for c in co.columns], f"outcomes on co-fire sessions h={h}")
    both_d1e1 = out.loc[(D1 & E1).values].dropna()
    e = declusters(both_d1e1.index, h, out.dropna().index)
    b = both_d1e1.loc[e]
    agree = ((b["D1_longDX"] > 0) == (b["E1_shortEEMres"] > 0)).mean()
    print(f"  D1&E1 declustered episodes N={len(b)}: sign agreement {100*agree:.1f}%; "
          f"equal-weight D1+E1 mean {100*b.sum(axis=1).mean()/2:+.3f}%")
    print("  co-fire list (G1 involved):")
    g = out.loc[(G1 & (D1 | E1)).values]
    print((100 * g).round(3).to_string())
