"""kC probe: data availability + live readings for c6/c7/c8 (bar 2026-09-17)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root
from pitch_lab import *  # noqa

T = ["GLD", "SLV", "GDX", "DX-Y.NYB", "UUP", "USO", "XLE", "XOP", "TLT", "IEF",
     "LQD", "ITB", "XHB", "XLRE", "KRE", "XLU", "XLF", "XLK", "XLB", "XLP",
     "SPY", "^TNX", "SIL", "SLVP"]
raw = load_prices(T)
for t in T:
    df = raw.get(t)
    if df is None or len(df) == 0:
        print(f"{t:10s} MISSING")
        continue
    c = df["Close"].dropna()
    r = c.pct_change(fill_method=None)
    print(f"{t:10s} {c.index[0].date()} .. {c.index[-1].date()}  n={len(c)}  "
          f"last {c.iloc[-1]:.3f}  d1 {100*r.iloc[-1]:+.2f}%  "
          f"vs252min {100*(c.iloc[-1]/c.rolling(252).min().iloc[-1]-1):+.2f}%  "
          f"vs252max {100*(c.iloc[-1]/c.rolling(252).max().iloc[-1]-1):+.2f}%")
