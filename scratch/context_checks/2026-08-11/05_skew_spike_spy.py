"""^SKEW thrust into the top 5% of its year: what does the S&P do next?

The engine measured SKEW's own forward return (n=359, -1.352%, t=-8.02) which
is near-tautological for a bounded fast-reverting index. The useful question is
the S&P's forward return after the thrust, and whether a CPI landing the next
session changes it.

Tonight: ^SKEW 135.59, 5d return +7.26% at the 98.4th percentile of its year,
while its 21d return sits at the 8.3rd. A sharp re-steepening of the tail bid
from a low base.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = load_prices(["^SKEW", "SPY", "^VIX"])
skew = px["^SKEW"]["Close"].dropna()
spy = px["SPY"]["Close"].dropna()
vix = px["^VIX"]["Close"].dropna()

common = skew.index.intersection(spy.index)
skew = skew.reindex(common)
spy = spy.reindex(common)

r5 = skew.pct_change(5)
rank5 = r5.rolling(252, min_periods=252).rank(pct=True) * 100
trig = pd.DatetimeIndex([d for d in common if pd.notna(rank5.get(d, np.nan)) and rank5[d] >= 95])
print(f"raw trigger days: {len(trig)}  {trig[0].date()} .. {trig[-1].date()}")

dec = declusters(trig, 5, common)
print(f"declustered (5td gap): {len(dec)} episodes")

sr = spy.pct_change()
for h in (1, 2, 3, 5, 10, 21):
    fw = (spy.shift(-h) / spy - 1.0)
    v = fw.reindex(dec).dropna()
    s = summarize(v.values, f"h{h}")
    ctrl_idx = local_control(common, dec, 126)
    vc = fw.reindex(ctrl_idx).dropna()
    sc = summarize(vc.values, "ctrl")
    print(f"  SPY h{h:<2} n={s['n']:<4} mean {s['mean_pct']:+7.3f}%  hit {s['hit']:5.1f}%  "
          f"t={s['t']:+5.2f}  up {(v>0).sum()}-{(v<0).sum()}  "
          f"sign p {sign_test(int((v>0).sum()), int(len(v))):.4f}   "
          f"| local ctrl {sc['mean_pct']:+.3f}% hit {sc['hit']:.1f}%  edge {s['mean_pct']-sc['mean_pct']:+.3f}%")

fw1 = (spy.shift(-1) / spy - 1.0)
v = fw1.reindex(dec).dropna()
print("\n  h1 era:", [(e['label'], e['n'], round(e['mean_pct'], 3), round(e['hit'], 1), round(e['t'], 2)) for e in era_split(v.index, v.values)])
print("  h1 concentration:", cluster_note(v.index, v.values))

fw5 = (spy.shift(-5) / spy - 1.0)
v5 = fw5.reindex(dec).dropna()
print("  h5 era:", [(e['label'], e['n'], round(e['mean_pct'], 3), round(e['hit'], 1), round(e['t'], 2)) for e in era_split(v5.index, v5.values)])
print("  h5 concentration:", cluster_note(v5.index, v5.values))

# does SKEW spiking while VIX is LOW differ? that is tonight's shape
pctv = vix.rolling(252, min_periods=252).rank(pct=True) * 100
lowvix = pd.DatetimeIndex([d for d in dec if pd.notna(pctv.get(d, np.nan)) and pctv[d] < 40])
print(f"\n  subset: SKEW thrust with VIX in the bottom 40% of its year -> n={len(lowvix)}")
for h in (1, 5, 10, 21):
    fw = (spy.shift(-h) / spy - 1.0)
    v = fw.reindex(lowvix).dropna()
    if len(v) < 5:
        continue
    s = summarize(v.values, f"h{h}")
    print(f"    h{h:<2} n={s['n']:<3} mean {s['mean_pct']:+7.3f}% hit {s['hit']:5.1f}% t={s['t']:+5.2f} "
          f"up {(v>0).sum()}-{(v<0).sum()} sign p {sign_test(int((v>0).sum()), int(len(v))):.4f}")

# and the SKEW-up / VIX-down shape specifically (tonight: SKEW 5d +7.26, VIX 5d -7.39)
vr5 = vix.pct_change(5)
diverge = pd.DatetimeIndex([d for d in dec if pd.notna(vr5.get(d, np.nan)) and vr5[d] < 0])
print(f"\n  subset: SKEW thrust WHILE VIX fell over the same 5 sessions -> n={len(diverge)}")
for h in (1, 5, 10, 21):
    fw = (spy.shift(-h) / spy - 1.0)
    v = fw.reindex(diverge).dropna()
    if len(v) < 5:
        continue
    s = summarize(v.values, f"h{h}")
    print(f"    h{h:<2} n={s['n']:<3} mean {s['mean_pct']:+7.3f}% hit {s['hit']:5.1f}% t={s['t']:+5.2f} "
          f"up {(v>0).sum()}-{(v<0).sum()} sign p {sign_test(int((v>0).sum()), int(len(v))):.4f}")
v = (spy.shift(-21) / spy - 1.0).reindex(diverge).dropna()
if len(v):
    print("    h21 era:", [(e['label'], e['n'], round(e['mean_pct'], 2), round(e['hit'], 1)) for e in era_split(v.index, v.values)])
    print("    h21 concentration:", cluster_note(v.index, v.values))
    print("    years:", sorted(set(diverge.year)))
