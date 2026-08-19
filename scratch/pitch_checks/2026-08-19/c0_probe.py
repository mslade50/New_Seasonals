"""C6/C7/C8 probe: data availability, live state values, trigger counts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

T = ["DX-Y.NYB", "UUP", "^TNX", "TLT", "IEF", "SPY", "GLD", "GDX", "SLV"]
px = close_panel(T)
print("panel span", px.index[0].date(), "->", px.index[-1].date(), "rows", len(px))
for t in T:
    s = px[t].dropna()
    print(f"  {t:10s} {s.index[0].date()} .. {s.index[-1].date()}  n={len(s)}  last={s.iloc[-1]:.4f}")

# live state
print("\n--- live state (bar 2026-08-18) ---")
for t in ["DX-Y.NYB", "UUP", "^TNX", "TLT", "GLD", "GDX", "SPY"]:
    s = px[t].dropna()
    r21 = s.pct_change(21)
    rk21 = pct_rank(s, 21)
    print(f"  {t:10s} 21d ret {100*r21.iloc[-1]:+7.2f}%  rank21 {rk21.iloc[-1]:5.1f}")

# TNX 21d CHANGE in yield points (level diff) vs pct
tnx = px["^TNX"].dropna()
print(f"  ^TNX level {tnx.iloc[-1]:.3f}  21d level chg {tnx.iloc[-1]-tnx.iloc[-22]:+.3f} pts")
