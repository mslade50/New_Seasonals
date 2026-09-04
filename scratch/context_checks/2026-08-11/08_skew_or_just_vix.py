"""Does the SKEW thrust add anything, or is a falling VIX the whole signal?

Drill 07 found next-10d realized vol of 13.57% after a SKEW thrust with VIX
falling, against a 16.08% local control. A falling VIX is itself a calm-regime
marker, so the control has to be VIX-matched or the claim belongs to VIX.
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
common = skew.index.intersection(spy.index).intersection(vix.index)
skew, spy, vix = skew.reindex(common), spy.reindex(common), vix.reindex(common)

sk5 = skew.pct_change(5)
rank5 = sk5.rolling(252, min_periods=252).rank(pct=True) * 100
vx5 = vix.pct_change(5)
pctv = vix.rolling(252, min_periods=252).rank(pct=True) * 100
fwvol = spy.pct_change().rolling(10).std().shift(-10) * np.sqrt(252) * 100

def stat(idx, lab):
    v = fwvol.reindex(pd.DatetimeIndex(idx)).dropna()
    print(f"  {lab:<52} n={len(v):<5} next-10d rvol mean {v.mean():5.2f}%  median {v.median():5.2f}%")
    return v

print("next-10d SPY realized vol, annualized")
skew_hot = (rank5 >= 95)
vix_dn = (vx5 < 0)
both = pd.DatetimeIndex([d for d in common if bool(skew_hot.get(d, False)) and bool(vix_dn.get(d, False))])
vixonly = pd.DatetimeIndex([d for d in common if not bool(skew_hot.get(d, False)) and bool(vix_dn.get(d, False))])
skewonly = pd.DatetimeIndex([d for d in common if bool(skew_hot.get(d, False)) and not bool(vix_dn.get(d, False))])
neither = pd.DatetimeIndex([d for d in common if not bool(skew_hot.get(d, False)) and not bool(vix_dn.get(d, False))])

a = stat(declusters(both, 5, common), "SKEW top-5% thrust AND VIX down on the week")
b = stat(vixonly, "VIX down on the week, no SKEW thrust (all days)")
c = stat(declusters(skewonly, 5, common), "SKEW thrust, VIX up on the week")
d_ = stat(neither, "neither")

# VIX-percentile matched: restrict the VIX-down control to the same VIX decile mix
print("\nVIX-percentile matched control (VIX down weeks, matched on VIX pctile decile)")
dec_both = declusters(both, 5, common)
pb = pctv.reindex(dec_both).dropna()
print(f"  episode VIX pctile: mean {pb.mean():.1f}  median {pb.median():.1f}")
bins = np.arange(0, 101, 10)
wts = pd.cut(pb, bins, include_lowest=True).value_counts(normalize=True).sort_index()
pc_ctrl = pctv.reindex(vixonly).dropna()
cb = pd.cut(pc_ctrl, bins, include_lowest=True)
parts = []
for b_, w in wts.items():
    sub = fwvol.reindex(pc_ctrl.index[cb == b_]).dropna()
    if len(sub):
        parts.append((w, sub.mean(), len(sub)))
matched = sum(w * m for w, m, _ in parts) / sum(w for w, _, _ in parts)
print(f"  matched control mean: {matched:.2f}%   vs episodes {a.mean():.2f}%   diff {a.mean()-matched:+.2f}pp")
for w, m, n in parts:
    print(f"    weight {w:.3f}  ctrl mean {m:5.2f}%  n={n}")
print(f"\n  today's VIX pctile: {pctv.iloc[-1]:.1f}   VIX {vix.iloc[-1]:.2f}   SKEW {skew.iloc[-1]:.2f}")
