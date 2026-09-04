"""Recon: print the thing each candidate is NAMED after, with a PIT percentile.

The 2026-08-24 registry's top lesson - three of ten candidates died on a false
premise checkable in one line. This runs that line for every candidate before
any battery.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TK = ["SPY", "QQQ", "IWM", "XLK", "XLV", "XLP", "XLU", "XLI", "XLF", "XLY", "XLE",
      "XLB", "XLC", "XLRE", "SMH", "IBB", "IHI", "OIH", "XOP", "USO", "TLT", "IEF",
      "^TNX", "HYG", "LQD", "GLD", "GDX", "SLV", "UUP", "DX-Y.NYB", "EFA", "EEM",
      "FXI", "^VIX", "^VIX3M", "^SKEW", "SVXY", "NVDA", "^GSPC"]
px = close_panel(TK)
print("panel", px.index[0].date(), "->", px.index[-1].date(), px.shape)

def pit(s, label, n=252):
    """PIT trailing-252 percentile of the CURRENT value of series s."""
    v = s.dropna()
    cur = v.iloc[-1]
    win = v.iloc[-n:]
    p = (win < cur).mean() * 100
    print(f"  {label:<46} now={cur:>9.3f}   PIT252 pctile={p:>5.1f}")
    return cur, p

print("\n=== P1  tech-vs-defensive 5d rotation spread ===")
r5 = px.pct_change(5)
for a, b in [("XLV", "XLK"), ("XLP", "XLK"), ("XLV", "QQQ"), ("XLP", "SMH"), ("XLV", "SMH")]:
    sp = (r5[a] - r5[b]) * 100
    pit(sp, f"{a} - {b}  5d spread (pp)")
    v = sp.dropna()
    print(f"      full-sample pctile {(v < v.iloc[-1]).mean()*100:5.1f}   N={len(v)}")

print("\n=== P2  semis relative state into the NVDA print ===")
r63 = px.pct_change(63)
pit((r63["SMH"] - r63["SPY"]) * 100, "SMH - SPY 63d (pp)")
pit((r5["SMH"] - r5["SPY"]) * 100, "SMH - SPY  5d (pp)")
pit(px["SMH"] / px["SMH"].rolling(252).max() * 100, "SMH pct of its own 252d high")

print("\n=== P3  utilities dumped without a rate cause ===")
r21 = px.pct_change(21)
pit(r21["XLU"] * 100, "XLU 21d return (%)")
pit(r21["TLT"] * 100, "TLT 21d return (%)")
pit((r21["XLU"] - r21["TLT"]) * 100, "XLU - TLT 21d spread (pp)")
pit((r21["XLU"] - r21["SPY"]) * 100, "XLU - SPY 21d spread (pp)")

print("\n=== P4  oil services vs E&P ===")
pit((r63["OIH"] - r63["XOP"]) * 100, "OIH - XOP 63d (pp)")
pit((r5["OIH"] - r5["XLE"]) * 100, "OIH - XLE  5d (pp)")
print(f"  XLE off 252d high: {(px['XLE'].iloc[-1]/px['XLE'].rolling(252).max().iloc[-1]-1)*100:.2f}%")
print(f"  OIH off 252d high: {(px['OIH'].iloc[-1]/px['OIH'].rolling(252).max().iloc[-1]-1)*100:.2f}%")

print("\n=== P5  credit quality divergence (watchlist W2) ===")
for t in ["HYG", "LQD"]:
    hi = px[t].rolling(252).max().iloc[-1]
    lo = px[t].rolling(252).min().iloc[-1]
    print(f"  {t}: off 52w high {(px[t].iloc[-1]/hi-1)*100:+.2f}%   above 52w low {(px[t].iloc[-1]/lo-1)*100:+.2f}%")

print("\n=== P6  dollar ===")
pit(r21["DX-Y.NYB"] * 100, "DXY 21d return (%)")
pit(r21["UUP"] * 100, "UUP 21d return (%)")
pit(px["^TNX"] / px["^TNX"].rolling(252).max() * 100, "^TNX pct of 252d high")

print("\n=== P7  vol term structure ===")
ts = px["^VIX"] / px["^VIX3M"]
pit(ts, "VIX / VIX3M ratio")
pit(px["^VIX"], "VIX level")
pit(r5["^VIX"] * 100, "VIX 5d return (%)")
pit(px["^SKEW"], "SKEW level")

print("\n=== P8  gold complex ===")
pit(r21["GDX"] * 100, "GDX 21d (%)")
pit((r21["GDX"] - r21["GLD"]) * 100, "GDX - GLD 21d (pp)")
print(f"  GLD off 52w high: {(px['GLD'].iloc[-1]/px['GLD'].rolling(252).max().iloc[-1]-1)*100:.2f}%")

print("\n=== P9  calendar position ===")
import pandas as pd
d = px.index[-1]
print("  last bar", d.date(), "weekday", d.day_name())
ev = load_events()
up = ev[ev["date"] > d].sort_values("date").head(12)
print(up.to_string(index=False))
