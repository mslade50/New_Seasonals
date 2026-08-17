"""Probe: which of today's candidate instruments are actually in the cache,
and what is TODAY's value of every trigger statistic I am about to define.

Registry rule (2026-08-11, the broken-rank incident): print TODAY's value of
any new trigger and confirm it matches the tape file, or the population being
measured may not even contain today's state.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

WANT = ["SMH", "QQQ", "SPY", "XLE", "XOP", "OIH", "USO", "EWZ", "EEM",
        "NVDA", "AVGO", "AMAT", "MU", "AMD", "INTC", "TXN", "QCOM", "ADI",
        "GLW", "XOM", "CVX", "COP", "SLB", "HAL", "VLO", "^GSPC", "^NDX"]
px = load_prices(WANT)
print("\npresent:", sorted(px))
for t in sorted(px):
    s = px[t]["Close"].dropna()
    print(f"  {t:6s} {s.index[0].date()} .. {s.index[-1].date()}  N={len(s)}  last={s.iloc[-1]:.2f}")

print("\n--- today's trigger statistics (bar 2026-08-14) ---")
pan = close_panel([t for t in WANT if t in px])
for t in ("SMH", "QQQ", "XLE", "EWZ", "EEM", "USO", "SPY"):
    if t not in pan:
        continue
    s = pan[t].dropna()
    r63 = pct_rank(s, 63).iloc[-1]
    r21 = pct_rank(s, 21).iloc[-1]
    r5 = pct_rank(s, 5).iloc[-1]
    z10 = zscore(s, 10).iloc[-1]
    hi52 = s.rolling(252).max().iloc[-1]
    sma200 = s.rolling(200).mean().iloc[-1]
    print(f"{t:5s} rank63={r63:6.1f} rank21={r21:6.1f} rank5={r5:6.1f} "
          f"z10={z10:+6.2f} off52wh={100*(s.iloc[-1]/hi52-1):+6.2f}% "
          f"vs200d={100*(s.iloc[-1]/sma200-1):+6.2f}% ret5={100*(s.iloc[-1]/s.iloc[-6]-1):+6.2f}%")
