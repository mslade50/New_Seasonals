"""Data availability probe for the 2026-08-19 sector-rotation candidates."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TICKERS = ["XLV", "XLK", "XLE", "XLF", "XLI", "XLP", "XLU", "XLB", "XLY",
           "SPY", "QQQ", "SMH", "IWM", "XLC", "XLRE"]

px = close_panel(TICKERS)
print("panel shape", px.shape)
for t in TICKERS:
    if t not in px.columns:
        print(f"{t:6s} MISSING")
        continue
    s = px[t].dropna()
    print(f"{t:6s} {s.index[0].date()} .. {s.index[-1].date()}  n={len(s)}")

print()
print("last 3 rows XLV/XLK/SPY/SMH/QQQ:")
print(px[["XLV", "XLK", "SPY", "SMH", "QQQ"]].tail(3))
r = px.pct_change()
print()
print("2026-08-18 daily returns (%):")
print((100 * r.iloc[-1]).round(2).sort_values())
