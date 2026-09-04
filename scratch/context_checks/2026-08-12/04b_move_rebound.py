"""04's real finding is not in the bonds, it is in ^MOVE itself: a >=5% bond-vol crush
that leaves the 10-year yield unchanged rebuilds. Pin it against the right control,
split the era, and check the concentration before it publishes.
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

px = close_panel(["^MOVE", "^TNX", "TLT", "IEF"])
move = px["^MOVE"].dropna()
tnx = px["^TNX"].dropna()
m1 = move.pct_change()
t1 = tnx.pct_change().reindex(move.index)

B = (m1 <= -0.05) & (t1.abs() <= 0.005)
Bi = declusters(B[B].index, 5, move.index)
print(f"cell B: MOVE 1d <= -5% with |TNX 1d| <= 0.5%   raw {int(B.sum())}  "
      f"declustered(5td) {len(Bi)}")
print(f"tonight: MOVE {100 * m1.iloc[-1]:+.2f}%  TNX {100 * t1.iloc[-1]:+.2f}%  "
      f"MOVE level {move.iloc[-1]:.1f}")


def stat(v):
    d = summarize(v)
    up = int((v > 0).sum())
    return (f"n={len(v):<4} mean={d['mean_pct']:+.3f}%  med={d['median_pct']:+.3f}%  "
            f"hit={d['hit']:.1f}%  t={d['t']:+.2f}  {up}-{len(v) - up} up  "
            f"sign p={sign_test(up, len(v)):.4f}")


print("\n=== ^MOVE forward, cell vs three controls ===")
for h in (1, 3, 5, 10, 21):
    f = fwd_ret(move, h)
    a = Bi.intersection(f.dropna().index)
    print(f"  h{h:<3} cell            {stat(f.loc[a].values)}")
    fa = f.dropna()
    print(f"       all sessions    {stat(fa.values)}")
    ctrl = local_control(fa.index, a, 126)
    print(f"       local +/-126td  {stat(fa.loc[ctrl.intersection(fa.index)].values)}")
    # vol-level matched control: same MOVE percentile decile, no crush
    pct = move.rolling(252).rank(pct=True)
    lo, hi = pct.loc[Bi].quantile([0.25, 0.75])
    matched = pct[(pct >= lo) & (pct <= hi)].index.difference(Bi)
    print(f"       MOVE-pctile matched {stat(fa.loc[matched.intersection(fa.index)].values)}")
    print()

print("=== era split, h10 ===")
f10 = fwd_ret(move, 10)
a = Bi.intersection(f10.dropna().index)
v = f10.loc[a].values
for part in era_split(a, v):
    print(f"  {part['label']:<10} n={part['n']:<4} mean={part['mean_pct']:+.3f}%  "
          f"med={part['median_pct']:+.3f}%  hit={part['hit']:.1f}%  t={part['t']:+.2f}")
print(f"  {cluster_note(a, v)}")

print("\n=== drop the biggest 3 episodes ===")
order = np.argsort(-np.abs(v))
keep = np.setdiff1d(np.arange(len(v)), order[:3])
print(f"  {stat(v[keep])}")

print("\n=== where is MOVE right now relative to the cell's history ===")
pct = move.rolling(252).rank(pct=True) * 100
print(f"  tonight MOVE 252d pctile {pct.iloc[-1]:.1f}")
print(f"  cell median entry pctile {pct.loc[Bi].median():.1f}")
hi_start = Bi[pct.loc[Bi] <= 50]
lo_start = Bi[pct.loc[Bi] > 50]
for lbl, idx in (("entered from a LOW MOVE pctile", hi_start),
                 ("entered from a HIGH MOVE pctile", lo_start)):
    a2 = pd.DatetimeIndex(idx).intersection(f10.dropna().index)
    print(f"  {lbl:<34} {stat(f10.loc[a2].values)}")
