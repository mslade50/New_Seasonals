"""Data probe: history depth + today's exact state for the C1/C1b/C7 inputs.

Not a check. Establishes what is measurable before any mask is built.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pitch_lab import load_prices, pct_rank  # noqa: E402

ASOF = pd.Timestamp("2026-08-13")
TK = ["^SKEW", "^VIX", "^VIX3M", "SPY", "SVXY", "^GSPC", "QQQ", "IWM",
      "VIXY", "UVXY", "^MOVE"]
px = load_prices(TK)

print("=" * 78)
print("history depth")
print("=" * 78)
for t in TK:
    if t not in px:
        print(f"  {t:9s} MISSING")
        continue
    c = px[t]["Close"].dropna()
    print(f"  {t:9s} {c.index[0].date()} .. {c.index[-1].date()}  n={len(c)}")


def lvl_pctile(s: pd.Series, lookback: int = 252) -> pd.Series:
    """Percentile of the LEVEL within its own trailing `lookback` closes."""
    return s.rolling(lookback).rank(pct=True) * 100.0


print("\n" + "=" * 78)
print("today's state (anchor 2026-08-13)")
print("=" * 78)
for t in ["^SKEW", "^VIX", "^VIX3M", "^MOVE"]:
    c = px[t]["Close"].dropna()
    c = c[c.index <= ASOF]
    print(f"  {t:8s} last {c.iloc[-1]:8.2f}  LEVEL pctile252 "
          f"{lvl_pctile(c).iloc[-1]:5.1f}  ret21rank {pct_rank(c,21).iloc[-1]:5.1f}"
          f"  ret5rank {pct_rank(c,5).iloc[-1]:5.1f}")

spy = px["SPY"]["Close"].dropna(); spy = spy[spy.index <= ASOF]
svxy = px["SVXY"]["Close"].dropna(); svxy = svxy[svxy.index <= ASOF]
print(f"  SPY   dist52wh {100*(spy.iloc[-1]/spy.iloc[-252:].max()-1):+.2f}%  "
      f"above200d {100*(spy.iloc[-1]/spy.rolling(200).mean().iloc[-1]-1):+.2f}%")
print(f"  SVXY  dist52wh {100*(svxy.iloc[-1]/svxy.iloc[-252:].max()-1):+.2f}%  "
      f"trailing21d {100*(svxy.iloc[-1]/svxy.iloc[-22]-1):+.2f}%  "
      f"rank63 {pct_rank(svxy,63).iloc[-1]:.1f}")

print("\n" + "=" * 78)
print("how rare is the state? (counts on the FULL ^SKEW history)")
print("=" * 78)
sk = px["^SKEW"]["Close"].dropna()
vx = px["^VIX"]["Close"].dropna()
sp = px["SPY"]["Close"].dropna()
idx = sk.index.intersection(vx.index).intersection(sp.index)
sk, vx, sp = sk.loc[idx], vx.loc[idx], sp.loc[idx]

sk_p = lvl_pctile(sk)
vx_p = lvl_pctile(vx)
sp_hi = sp / sp.rolling(252).max() - 1.0

for skt in (2, 5, 10, 20):
    m = sk_p <= skt
    print(f"  SKEW lvl pctile <= {skt:2d}: {int(m.sum()):5d} days, "
          f"first {sk_p[m].index[0].date() if m.any() else '-'}")
for vxt in (5, 10, 20):
    m = vx_p <= vxt
    print(f"  VIX  lvl pctile <= {vxt:2d}: {int(m.sum()):5d} days")
for hit in (0.0, -0.005, -0.01):
    m = sp_hi >= hit
    print(f"  SPY within {abs(hit)*100:.1f}% of 52wh: {int(m.sum()):5d} days")

print("\n  joint cells:")
for skt in (2, 5, 10, 20):
    for vxt in (5, 10, 20):
        for hit in (0.0, -0.01):
            m = (sk_p <= skt) & (vx_p <= vxt) & (sp_hi >= hit)
            yrs = sorted(set(m[m].index.year))
            print(f"    SKEW<={skt:2d} VIX<={vxt:2d} SPYhi>={hit*100:+.0f}%: "
                  f"N={int(m.sum()):4d} yrs={yrs}")
