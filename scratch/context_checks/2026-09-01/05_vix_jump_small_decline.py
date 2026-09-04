"""^VIX rose 9.52% while SPY fell only 0.69%. The trigger inventory has no
line for a vol response that large against an equity move that small (P10c
wants +10% on VIX outright and does not look at SPY), so build the cell.

Definition: VIX up 8% or more on a session SPY closed DOWN but by less than
1%. Forward SPY and VIX, lag=0, episodes declustered at 5td.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["SPY", "^VIX", "^VIX3M"]).dropna(subset=["SPY", "^VIX"])
idx = px.index
spy_r = px["SPY"].pct_change()
vix_r = px["^VIX"].pct_change()
print(f"panel {idx[0].date()} .. {idx[-1].date()}, {len(px)} sessions")
print(f"today: SPY {100*spy_r.iloc[-1]:+.2f}%, ^VIX {100*vix_r.iloc[-1]:+.2f}% "
      f"to {px['^VIX'].iloc[-1]:.2f}")

mask = (vix_r >= 0.08) & (spy_r < 0) & (spy_r > -0.01)
dts = idx[mask.fillna(False).values]
print(f"\nVIX +8% or more with SPY down less than 1%: {len(dts)} sessions "
      f"({100*len(dts)/len(px):.2f}% of the panel)")
epi = declusters(dts, 5, idx)
print(f"episodes at a 5td gap: {len(epi)}")
print("  years: " + str(dict(pd.Series(epi.year).value_counts().sort_index())))


def block(subj, dates, label, hs=(1, 3, 5, 10, 21)):
    rows = []
    for h in hs:
        r = fwd_ret(px[subj], h)
        v = r.loc[r.index.intersection(dates)].dropna()
        s = summarize(v.values, f"h={h}")
        if s["n"]:
            w = int((v.values > 0).sum())
            s["record"] = f"{w}-{s['n']-w}"
            s["sign_p"] = round(sign_test(w, s["n"]), 4)
            s["edge_pct"] = round(s["mean_pct"] - 100 * r.dropna().mean(), 3)
        rows.append(s)
    show(rows, f"{subj}: {label}")


block("SPY", epi, f"after the cell ({len(epi)} episodes)")
block("^VIX", epi, "VIX itself", hs=(1, 5, 10))

print("\n=== controls ===")
r5 = fwd_ret(px["SPY"], 5)
base = r5.dropna()
ctl = local_control(idx, dts, 126)
print(f"  SPY h=5: episodes {100*r5.loc[r5.index.intersection(epi)].mean():+.3f}%  "
      f"local +/-126td {100*r5.loc[r5.index.intersection(ctl)].mean():+.3f}%  "
      f"all days {100*base.mean():+.3f}%")
# the obvious confound: any SPY down day under 1% without the VIX jump
plain = idx[((spy_r < 0) & (spy_r > -0.01) & (vix_r < 0.08)).fillna(False).values]
block("SPY", plain, "control: SPY down <1% WITHOUT a VIX jump", hs=(1, 5, 10))

print("\n=== era split and concentration, SPY h=5 and h=10 ===")
for h in (5, 10):
    r = fwd_ret(px["SPY"], h)
    v = r.loc[r.index.intersection(epi)].dropna()
    show(era_split(v.index, v.values), f"SPY h={h}")
    print("  " + cluster_note(v.index, v.values, k=2))

print("\n=== does the VIX level matter? split at its own 252d median ===")
med = rolling_on_valid(px["^VIX"], lambda x: x.rolling(252, min_periods=200).median())
lo = epi[(px["^VIX"].reindex(epi) <= med.reindex(epi)).fillna(False).values]
hi = epi.difference(lo)
print(f"  VIX at or below its 252d median: {len(lo)} episodes; above: {len(hi)}")
print(f"  tonight VIX {px['^VIX'].iloc[-1]:.2f} vs 252d median {med.iloc[-1]:.2f}")
for name, d in (("VIX <= median", lo), ("VIX > median", hi)):
    block("SPY", d, name, hs=(1, 5, 10))
