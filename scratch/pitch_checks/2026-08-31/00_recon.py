"""Live-state recon for the 2026-08-31 surface map.

Verifies each near-armed watchlist trigger against today's tape with the
SAME statistic the entry names, and prints today's value. Per the
2026-08-11 registry entry, every new trigger is sanity-checked by printing
its live value and confirming it matches data/pitch_tape.json.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

ASOF = pd.Timestamp("2026-08-28")  # freshest bar

TK = ["SPY","QQQ","IWM","TLT","IEF","LQD","HYG","^TNX","^MOVE","^VIX","^VIX3M","^SKEW",
      "GLD","GDX","SLV","XLU","XLE","XOP","OIH","USO","UUP","DX-Y.NYB","EEM","EFA","FXI",
      "SMH","XLK","XLV","XLF","XLI","XLP","XLY","XLB","XLRE","XLC","IYR","VNQ","XRT","TJX","XME"]
px = load_prices(TK)
print("loaded:", {k: (str(v.index[-1].date()), len(v)) for k, v in list(px.items())[:3]})

def last(t, col="Close"):
    return px[t][col].dropna()

def dist_to_extreme(t, win=252, kind="low"):
    s = last(t)
    w = s.iloc[-win:]
    if kind == "low":
        return 100.0 * (s.iloc[-1] / w.min() - 1.0)
    return 100.0 * (s.iloc[-1] / w.max() - 1.0)

print("\n=== W5 / W31: IG complex at 52w lows, HYG at high ===")
for t in ["TLT","IEF","LQD"]:
    print(f"  {t:5s} above trailing-252 low: {dist_to_extreme(t,252,'low'):+.3f}%")
print(f"  HYG   below trailing-252 high: {dist_to_extreme('HYG',252,'high'):+.3f}%")
print("  W5 rung  (TLT<=0.5, IEF<=1.0, LQD<=1.0):",
      dist_to_extreme('TLT')<=0.5 and dist_to_extreme('IEF')<=1.0 and dist_to_extreme('LQD')<=1.0)
print("  W31 rung (IEF<=1.5, LQD<=1.5, HYG within 0.25 of high):",
      dist_to_extreme('IEF')<=1.5 and dist_to_extreme('LQD')<=1.5 and abs(dist_to_extreme('HYG',252,'high'))<=0.25)

print("\n=== W11: SPY at 52w high AND TLT at 52w low ===")
print(f"  SPY below trailing-252 high: {dist_to_extreme('SPY',252,'high'):+.3f}%  (needs >= -0.5)")
print(f"  TLT above trailing-252 low : {dist_to_extreme('TLT',252,'low'):+.3f}%  (needs <= 1.0)")

print("\n=== W21: TNX within 0.25% of trailing-252 high ===")
print(f"  ^TNX below trailing-252 high: {dist_to_extreme('^TNX',252,'high'):+.3f}%  (needs >= -0.25)")
print(f"  ^TNX level {last('^TNX').iloc[-1]:.3f}, 252d max {last('^TNX').iloc[-252:].max():.3f}")

print("\n=== W6: SKEW 5d return rank >= 95 ===")
sk = pct_rank(last("^SKEW"), 5, 252)
print(f"  pct_rank(^SKEW,5) today = {sk.iloc[-1]:.1f}   (needs >= 95); midterm year blocks anyway")
print(f"  SKEW level {last('^SKEW').iloc[-1]:.2f}, trailing-252 level pctile "
      f"{100.0*(last('^SKEW').iloc[-252:] <= last('^SKEW').iloc[-1]).mean():.1f}")

print("\n=== W26: XLU 21d rank <= 5 with TLT also hit ===")
xr = pct_rank(last("XLU"), 21, 252)
print(f"  pct_rank(XLU,21) today = {xr.iloc[-1]:.1f}  (needs <= 5)")
tr = pct_rank(last("TLT"), 21, 252)
print(f"  pct_rank(TLT,21) today = {tr.iloc[-1]:.1f}")

print("\n=== W24: OIH minus XOP 63d return spread, PIT trailing-252 pctile ===")
o = last("OIH"); x = last("XOP")
common = o.index.intersection(x.index)
o, x = o.reindex(common), x.reindex(common)
spread = o.pct_change(63) - x.pct_change(63)
pit = spread.rolling(252).apply(lambda w: 100.0*(w[:-1] <= w[-1]).mean(), raw=True)
print(f"  spread today = {100*spread.iloc[-1]:+.2f}pp   PIT pctile = {pit.iloc[-1]:.1f}  (needs <= 2.5)")

print("\n=== W22: energy complex count at z10 >= 2.0 ===")
ENER = ["XLE","XOP","USO","OIH"]
zs = {t: zscore(last(t), 10).iloc[-1] for t in ENER}
print("  ", {k: round(float(v),2) for k,v in zs.items()}, " count>=2.0:", sum(v>=2.0 for v in zs.values()))

print("\n=== W30/W33: 21d rank>=90 AND 63d rank<=10 AND 5d rank<15 ===")
hits = []
for t in TK:
    s = last(t)
    if len(s) < 400: continue
    r21, r63, r5 = pct_rank(s,21,252).iloc[-1], pct_rank(s,63,252).iloc[-1], pct_rank(s,5,252).iloc[-1]
    if r21 >= 90 and r63 <= 10:
        hits.append((t, round(float(r21),1), round(float(r63),1), round(float(r5),1)))
print("  holders (t, r21, r63, r5):", hits or "NONE")
print(f"  SMH: r21={pct_rank(last('SMH'),21,252).iloc[-1]:.1f} r63={pct_rank(last('SMH'),63,252).iloc[-1]:.1f} r5={pct_rank(last('SMH'),5,252).iloc[-1]:.1f}")

print("\n=== Vol term structure + level percentiles ===")
v, v3 = last("^VIX").iloc[-1], last("^VIX3M").iloc[-1]
print(f"  VIX {v:.2f} (trailing-252 level pctile {100.0*(last('^VIX').iloc[-252:]<=v).mean():.1f})")
print(f"  VIX3M {v3:.2f} (trailing-252 level pctile {100.0*(last('^VIX3M').iloc[-252:]<=v3).mean():.1f})")
print(f"  VIX3M/VIX = {v3/v:.4f}, trailing-252 pctile "
      f"{100.0*((last('^VIX3M')/last('^VIX')).iloc[-252:] <= v3/v).mean():.1f}")
mv = last("^MOVE").iloc[-1]
print(f"  MOVE {mv:.2f} (trailing-252 level pctile {100.0*(last('^MOVE').iloc[-252:]<=mv).mean():.1f})")

print("\n=== calendar position ===")
d = last("SPY").index
print("  last 5 sessions:", [str(x.date()) for x in d[-5:]])
print("  2026-08-31 is the last trading day of August (ME-0); next session opens September.")
