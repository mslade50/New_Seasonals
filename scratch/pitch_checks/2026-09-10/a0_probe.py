"""a0 — data availability probe for the three adversarial checks.

Confirms which tickers the cache carries, their first bar, and the live
readings the candidates claim, before any statistic is computed.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

WANT = [
    "^TNX", "XLRE", "IYR", "VNQ", "XLF", "SPY", "QQQ", "IWM", "DIA",
    "XLB", "XLC", "XLE", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY",
    "SMH", "XBI", "IBB", "ITA", "IHI", "ITB", "XHB", "XRT", "XME",
    "XOP", "OIH", "KRE", "GDX", "EFA", "EEM", "USO", "DBC",
]

px = load_prices(WANT)
rows = []
for t in WANT:
    if t not in px:
        rows.append({"ticker": t, "first": "MISSING", "last": "", "n": 0})
        continue
    s = px[t]["Close"].dropna()
    rows.append({"ticker": t, "first": str(s.index[0].date()),
                 "last": str(s.index[-1].date()), "n": len(s),
                 "close": round(float(s.iloc[-1]), 3)})
print(pd.DataFrame(rows).to_string(index=False))

# live readings the candidates assert
p = close_panel([t for t in WANT if t in px])
print("\nlive readings as of last bar", p.index[-1].date())
for t, lbl in [("^TNX", "10y yield"), ("XLRE", "XLRE"), ("XLF", "XLF"),
               ("XLE", "XLE"), ("XLY", "XLY"), ("SMH", "SMH")]:
    if t not in p:
        continue
    r5 = pct_rank(p[t], 5).iloc[-1]
    r21 = pct_rank(p[t], 21).iloc[-1]
    r63 = pct_rank(p[t], 63).iloc[-1]
    hi252 = rolling_on_valid(p[t], lambda x: x.rolling(252).max()).iloc[-1]
    print(f"  {lbl:10s} close={p[t].iloc[-1]:9.3f} r5={r5:5.1f} r21={r21:5.1f} "
          f"r63={r63:5.1f}  252d max={hi252:9.3f} at_high={p[t].iloc[-1] >= hi252 - 1e-9}")

# 21d spread XLE - XLY today and its percentile
sp = _valid_pct_change(p["XLE"], 21) - _valid_pct_change(p["XLY"], 21)
spr = rolling_on_valid(sp, lambda x: x.rolling(252).rank(pct=True) * 100)
print(f"\nXLE-XLY 21d spread today = {100*sp.iloc[-1]:+.2f}pp, "
      f"trailing-252 pctile = {spr.iloc[-1]:.1f}")
