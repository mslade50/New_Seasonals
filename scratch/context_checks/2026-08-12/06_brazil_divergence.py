"""^BVSP closed down a 6th straight session with its 5-day return in the 0.4th
percentile of its own year, on a day US indices printed 52-week highs. The engine gave
the bare down-streak cell (n=144, +0.401%, 59.0%, sign p 0.0184). The cross is the
question: does a Brazilian washout INTO a US high behave like one that happens with
the US falling too?
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

px = close_panel(["^BVSP", "EWZ", "SPY", "^GSPC"])
bv, ewz, spy = px["^BVSP"].dropna(), px["EWZ"].dropna(), px["SPY"].dropna()

r1 = bv.pct_change()
streak = (r1 < 0).astype(int)
run = streak * (streak.groupby((streak != streak.shift()).cumsum()).cumcount() + 1)
rank5 = bv.pct_change(5).rolling(252).rank(pct=True) * 100
spy_hi = spy.rolling(252).max().reindex(bv.index).ffill()
spy_near_hi = (spy.reindex(bv.index).ffill() / spy_hi - 1.0) >= -0.005

print(f"tonight: ^BVSP run of {int(run.iloc[-1])} down closes, 5d "
      f"{100 * bv.pct_change(5).iloc[-1]:+.2f}% (rank {rank5.iloc[-1]:.1f}), "
      f"SPY {100 * (spy.iloc[-1] / spy_hi.iloc[-1] - 1):+.2f}% off its high")


def show(label, idx, h=1, tkr="^BVSP"):
    f = fwd_ret(px[tkr].dropna(), h)
    a = pd.DatetimeIndex(idx).intersection(f.dropna().index)
    v = f.loc[a].values
    if len(v) == 0:
        print(f"  {label:<48} {tkr:<6} n=0")
        return None
    d = summarize(v)
    up = int((v > 0).sum())
    print(f"  {label:<48} {tkr:<6} n={len(v):<4} mean={d['mean_pct']:+.3f}%  "
          f"med={d['median_pct']:+.3f}%  hit={d['hit']:.1f}%  t={d['t']:+.2f}  "
          f"{up}-{len(v) - up} up  sign p={sign_test(up, len(v)):.4f}")
    return a, v


base = (run >= 5)
print(f"\n=== base: >=5 consecutive down closes, n={int(base.sum())} ===")
bi = declusters(base[base].index, 5, bv.index)
for tkr in ("^BVSP", "EWZ"):
    for h in (1, 5, 10):
        show(f"5+ down closes, h{h}", bi, h, tkr)
    print()

print("=== the cross: same streak, but SPY within 0.5% of a 52-week high ===")
cross = base & spy_near_hi
ci = declusters(cross[cross].index, 5, bv.index)
anti = base & ~spy_near_hi
ai = declusters(anti[anti].index, 5, bv.index)
print(f"  streak into a US high: {len(ci)}   streak with the US off its high: {len(ai)}")
for tkr in ("^BVSP", "EWZ"):
    for h in (1, 5, 10, 21):
        show(f"streak INTO a US high, h{h}", ci, h, tkr)
    print()
for tkr in ("^BVSP", "EWZ"):
    for h in (1, 5, 10):
        show(f"streak, US NOT at a high, h{h}", ai, h, tkr)
    print()

print("=== tighten with the 5d washout rank ===")
tight = base & spy_near_hi & (rank5 <= 5)
ti = declusters(tight[tight].index, 5, bv.index)
print(f"  streak + 5d in the bottom 5% + US at a high: {len(ti)}")
for tkr in ("^BVSP", "EWZ"):
    for h in (1, 5, 10, 21):
        show(f"full cross, h{h}", ti, h, tkr)
    print()
print("  occurrences:")
for d0 in ti:
    print(f"    {d0.date()}  5d {100 * bv.pct_change(5).loc[d0]:+.2f}%  run {int(run.loc[d0])}")

print("\n=== era + concentration on the base streak, h5 ===")
r = show("base streak h5", bi, 5, "^BVSP")
if r:
    for part in era_split(r[0], r[1]):
        print(f"    {part['label']:<10} n={part['n']:<4} mean={part['mean_pct']:+.3f}%  "
              f"hit={part['hit']:.1f}%  t={part['t']:+.2f}")
    print(f"  {cluster_note(r[0], r[1])}")

print("\n=== controls ===")
for tkr in ("^BVSP", "EWZ"):
    f = fwd_ret(px[tkr].dropna(), 5).dropna()
    d = summarize(f.values)
    print(f"  {tkr:<6} h5 all sessions n={len(f)} mean={d['mean_pct']:+.3f}% "
          f"med={d['median_pct']:+.3f}% hit={d['hit']:.1f}%")
    ctrl = local_control(f.index, ci.intersection(f.index), 126)
    v = f.loc[ctrl.intersection(f.index)].values
    d = summarize(v)
    print(f"  {tkr:<6} h5 local control n={len(v)} mean={d['mean_pct']:+.3f}% "
          f"hit={d['hit']:.1f}%")
