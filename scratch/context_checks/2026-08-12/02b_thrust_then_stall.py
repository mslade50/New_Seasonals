"""02 killed its own premise. ^GSPC's z10 of +2.16 is NOT vol compression: vol21/vol63
is 1.00 and the 10-day return is +5.91%. What is unusual is the SHAPE. The index gained
5.91% over ten sessions of which the last five contributed +0.32% (46th percentile), and
it is sitting 0.12% under a 52-week high. A thrust that stopped dead at the top.

Cells:
  A  z10 >= 2 base, h1 hit rate against control      (engine gave 49-71 down)
  B  10d ret >= 5% AND 5d ret <= 0.5%                (thrust then stall)
  C  B, and within 0.5% of a 52-week high            (tonight exactly)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, cluster_note, declusters, era_split, fwd_ret, local_control,
    sign_test, summarize,
)

px = close_panel(["^GSPC", "SPY", "^VIX"])
s = px["^GSPC"].dropna()
dates = s.index

ret5, ret10 = s.pct_change(5), s.pct_change(10)
vol21 = s.pct_change().rolling(21).std()
z10 = ret10 / (vol21 * np.sqrt(10))
hi252 = s.rolling(252).max()
near_high = (s / hi252 - 1.0) >= -0.005

print("tonight: z10 %+.2f  10d %+.2f%%  5d %+.2f%%  under 52wh %.2f%%"
      % (z10.iloc[-1], 100 * ret10.iloc[-1], 100 * ret5.iloc[-1],
         100 * (s.iloc[-1] / hi252.iloc[-1] - 1)))


def show(label, idx, h=1, tkr="^GSPC", quiet=False):
    f = fwd_ret(px[tkr].dropna(), h)
    a = pd.DatetimeIndex(idx).intersection(f.dropna().index)
    v = f.loc[a].values
    if len(v) == 0:
        print(f"  {label:<50} n=0")
        return None
    d = summarize(v)
    up = int((v > 0).sum())
    if not quiet:
        print(f"  {label:<50} n={len(v):<4} mean={d['mean_pct']:+.3f}%  "
              f"med={d['median_pct']:+.3f}%  hit={d['hit']:.1f}%  t={d['t']:+.2f}  "
              f"{up}-{len(v) - up} up  sign p={sign_test(up, len(v)):.4f}")
    return a, v


print("\n=== A. the base z10>=2 cell, and the control it should be read against ===")
A = (z10 >= 2)[z10 >= 2].index
for h in (1, 2, 3, 5, 10, 21):
    show(f"z10>=2  h{h}", A, h)
f1 = fwd_ret(s, 1).dropna()
d = summarize(f1.values)
print(f"  {'ALL sessions h1':<50} n={len(f1):<4} mean={d['mean_pct']:+.3f}%  "
      f"med={d['median_pct']:+.3f}%  hit={d['hit']:.1f}%")
ctrl = local_control(f1.index, A.intersection(f1.index), 126)
v = f1.loc[ctrl.intersection(f1.index)].values
d = summarize(v)
print(f"  {'local control +/-126td h1':<50} n={len(v):<4} mean={d['mean_pct']:+.3f}%  "
      f"med={d['median_pct']:+.3f}%  hit={d['hit']:.1f}%")

r = show("z10>=2 h1 (era)", A, 1, quiet=True)
print("  era split h1:")
for part in era_split(r[0], r[1]):
    print(f"    {part['label']:<10} n={part['n']:<4} mean={part['mean_pct']:+.3f}%  "
          f"hit={part['hit']:.1f}%  t={part['t']:+.2f}")
dec = declusters(r[0], 5, dates)
v = f1.loc[dec.intersection(f1.index)].values
up = int((v > 0).sum())
d = summarize(v)
print(f"  declustered 5td: n={len(v)} mean={d['mean_pct']:+.3f}% hit={d['hit']:.1f}% "
      f"{up}-{len(v) - up} up sign p={sign_test(up, len(v)):.4f}")

print("\n=== B. thrust then stall: 10d >= 5% while the last 5d <= 0.5% ===")
B = ((ret10 >= 0.05) & (ret5 <= 0.005))
Bi = B[B].index
print(f"  raw occurrences {len(Bi)}, declustered(10td) {len(declusters(Bi, 10, dates))}")
for h in (1, 2, 5, 10, 21):
    show(f"thrust-stall  h{h}", Bi, h)
Bd = declusters(Bi, 10, dates)
print("  declustered:")
for h in (1, 5, 10, 21):
    show(f"thrust-stall declustered  h{h}", Bd, h)

r = show("B h5 era", Bd, 5, quiet=True)
print("  era split h5 (declustered):")
for part in era_split(r[0], r[1]):
    print(f"    {part['label']:<10} n={part['n']:<4} mean={part['mean_pct']:+.3f}%  "
          f"hit={part['hit']:.1f}%  t={part['t']:+.2f}")
print(f"  {cluster_note(r[0], r[1])}")

print("\n=== C. the same, sitting within 0.5% of a 52-week high ===")
C = B & near_high
Ci = C[C].index
Cd = declusters(Ci, 10, dates)
print(f"  raw {len(Ci)}, declustered {len(Cd)}")
for h in (1, 2, 5, 10, 21):
    show(f"thrust-stall at a 52wh  h{h}", Cd, h)
print("  occurrences:")
for dt in Cd:
    print(f"    {dt.date()}")

notC = B & ~near_high
for h in (1, 5, 21):
    show(f"thrust-stall NOT near a high  h{h}", declusters(notC[notC].index, 10, dates), h)

print("\n=== how often does a 10d thrust >= 5% happen with the last week flat? ===")
thrust = ret10 >= 0.05
print(f"  10d >= 5%: {int(thrust.sum())} sessions of {len(s)}")
print(f"  of those, 5d <= 0.5%: {int((thrust & (ret5 <= 0.005)).sum())} "
      f"({100 * (thrust & (ret5 <= 0.005)).sum() / thrust.sum():.1f}%)")
print(f"  median 5d return inside a 10d thrust: "
      f"{100 * ret5[thrust].median():+.2f}%, tonight {100 * ret5.iloc[-1]:+.2f}%")
