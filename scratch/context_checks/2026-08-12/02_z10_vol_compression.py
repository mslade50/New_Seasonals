"""^GSPC printed z10 = +2.16 tonight on a 5-day return of +0.32% (46th percentile of
its year). The z is high because the denominator collapsed: 21d realized vol 13.7%,
0.62x its 63-day norm. Split the historical z10>=2 cell on WHICH side of the ratio
produced the reading and see whether they behave differently.

z10 definition matches build_pitch_state._metrics_for exactly:
    10d return / (21d stdev of daily returns * sqrt(10))
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, era_split, fwd_ret, sign_test, summarize, cluster_note  # noqa: E402

px = close_panel(["^GSPC", "SPY", "^VIX"])
s = px["^GSPC"].dropna()

ret10 = s.pct_change(10)
vol21 = s.pct_change().rolling(21).std()
z10 = ret10 / (vol21 * np.sqrt(10))
vol63 = s.pct_change().rolling(63).std()
vol_ratio = vol21 / vol63                      # < 1 = compressing
rank5 = s.pct_change(5).rolling(252).rank(pct=True) * 100

print("=== tonight's reading, reproduced ===")
print(f"  ^GSPC z10 {z10.iloc[-1]:+.2f}   10d ret {100 * ret10.iloc[-1]:+.2f}%   "
      f"5d rank {rank5.iloc[-1]:.1f}   vol21/vol63 {vol_ratio.iloc[-1]:.2f}   "
      f"rvol21 {100 * vol21.iloc[-1] * np.sqrt(252):.1f}%")

trig = z10 >= 2.0
base = trig[trig].index
print(f"\n=== base cell: ^GSPC z10 >= 2, n={len(base)} ===")


def show(label, idx, h=1, tkr="^GSPC"):
    f = fwd_ret(px[tkr].dropna(), h)
    a = pd.DatetimeIndex(idx).intersection(f.dropna().index)
    v = f.loc[a].values
    if len(v) == 0:
        print(f"  {label:<52} n=0")
        return None
    d = summarize(v)
    up = int((v > 0).sum())
    print(f"  {label:<52} n={len(v):<4} mean={d['mean_pct']:+.3f}%  "
          f"med={d['median_pct']:+.3f}%  hit={d['hit']:.1f}%  t={d['t']:+.2f}  "
          f"{up}-{len(v) - up} up  sign p={sign_test(up, len(v)):.4f}")
    return a, v


for h in (1, 2, 5, 10):
    show(f"z10>=2, h{h}", base, h)

allf = fwd_ret(s, 1).dropna()
d = summarize(allf.values)
print(f"  {'all sessions control, h1':<52} n={len(allf):<4} "
      f"mean={d['mean_pct']:+.3f}%  med={d['median_pct']:+.3f}%  hit={d['hit']:.1f}%")

# --- the split ------------------------------------------------------------------
print("\n=== split: was the numerator big or the denominator small? ===")
comp = trig & (vol_ratio < 0.85)      # vol compressing into the reading
noncomp = trig & (vol_ratio >= 0.85)
print(f"  compressed-vol z10>=2: {int(comp.sum())}   normal-vol z10>=2: {int(noncomp.sum())}")
for h in (1, 5, 10):
    show(f"z10>=2 AND vol21<0.85x vol63, h{h}", comp[comp].index, h)
    show(f"z10>=2 AND vol21>=0.85x vol63, h{h}", noncomp[noncomp].index, h)
    print()

# tonight is more extreme than 0.85; use the actual reading as the cut
cut = float(vol_ratio.iloc[-1])
tight = trig & (vol_ratio <= cut + 0.03)
print(f"=== tighter: vol21/vol63 <= {cut + 0.03:.2f} (tonight is {cut:.2f}) ===")
for h in (1, 5, 10, 21):
    show(f"z10>=2 AND vol ratio <= {cut + 0.03:.2f}, h{h}", tight[tight].index, h)

r = show("h5 for the era check", tight[tight].index, 5)
if r:
    print("\n  era split, h5:")
    for part in era_split(r[0], r[1]):
        print(f"    {part['label']:<10} n={part['n']:<4} mean={part['mean_pct']:+.3f}%  "
              f"hit={part['hit']:.1f}%  t={part['t']:+.2f}")
    print(f"  {cluster_note(r[0], r[1])}")

# --- add the 5d-quiet condition: the index barely moved over the last week --------
print("\n=== z10>=2 while the 5d return sits below its own median rank ===")
quiet = trig & (rank5 <= 50)
loud = trig & (rank5 > 50)
print(f"  quiet-5d n={int(quiet.sum())}   loud-5d n={int(loud.sum())}")
for h in (1, 5, 10):
    show(f"z10>=2 AND 5d rank <= 50, h{h}", quiet[quiet].index, h)
    show(f"z10>=2 AND 5d rank  > 50, h{h}", loud[loud].index, h)
    print()

r = show("quiet-5d h10 for era", quiet[quiet].index, 10)
if r:
    for part in era_split(r[0], r[1]):
        print(f"    {part['label']:<10} n={part['n']:<4} mean={part['mean_pct']:+.3f}%  "
              f"hit={part['hit']:.1f}%  t={part['t']:+.2f}")
    print(f"  {cluster_note(r[0], r[1])}")
    print("  most recent 8 occurrences:")
    for dt in list(r[0])[-8:]:
        print(f"    {dt.date()}")

# --- realized vol AFTER, which is the honest question for a compression reading ---
print("\n=== what happens to realized vol after a compressed-vol z10>=2 ===")
fwd_vol10 = s.pct_change().rolling(10).std().shift(-10) * np.sqrt(252) * 100
for label, idx in (("compressed", tight[tight].index), ("all sessions", s.index)):
    v = fwd_vol10.reindex(idx).dropna()
    print(f"  {label:<14} n={len(v):<5} next-10d realized vol mean {v.mean():.2f}%  "
          f"median {v.median():.2f}%")
