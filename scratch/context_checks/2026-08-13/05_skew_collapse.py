"""Two sessions ago this brief published SKEW at the 98.4th percentile as a
record tail bid. Tonight SKEW's 21-day return is in the 2nd percentile of its
year, -9.52%, close 134.37.

The sweep tagged this `solid` on SKEW's own forward return, n=350, t=6.26. That
is SKEW mean-reverting in SKEW, which is mechanical and not worth Scott's time.
The publishable questions:

  1. what does SPY do after the tail bid drains
  2. what does VIX do, given VIX is already at a 63d rank of 20.6
  3. how fast was the round trip, and does the whipsaw itself mean anything
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, declusters, era_split,  # noqa: E402
                       fwd_ret, local_control, sign_test, summarize,
                       cluster_note)

px = close_panel(["^SKEW", "SPY", "^VIX"])
px = px[px.index >= "1999-01-01"]
sk = px["^SKEW"].dropna()

r21 = sk / sk.shift(21) - 1.0
rank21 = r21.rolling(252).rank(pct=True) * 100
fire = (rank21 <= 5.0)

print(f"tonight: SKEW {sk.iloc[-1]:.2f}, 21d return {100 * r21.iloc[-1]:+.2f}%, "
      f"rank {rank21.iloc[-1]:.1f}")
print(f"SKEW history starts {sk.index[0].date()}, {len(sk)} sessions\n")

trig = sk.index[fire.fillna(False).to_numpy()]
dc = declusters(pd.DatetimeIndex(trig), 21, sk.index)
print(f"raw firing sessions {len(trig)}, declustered episodes at 21td gap {len(dc)}")
print("last 10 episodes:", [str(d.date()) for d in dc[-10:]], "\n")


def fwd(subject, dates, label, hs=(1, 5, 10, 21), era=False):
    s = px[subject].dropna()
    keep = []
    for h in hs:
        r = fwd_ret(s, h).reindex(pd.DatetimeIndex(dates)).dropna()
        if len(r) < 4:
            continue
        d = summarize(r.to_numpy(), label)
        up = int((r > 0).sum())
        print(f"   {label:<44} {subject:<6} h{h:<3} n={len(r):>4} "
              f"mean={d['mean_pct']:+6.2f}% med={d['median_pct']:+6.2f}% "
              f"up={up}-{len(r) - up} ({100 * up / len(r):4.1f}%) "
              f"t={d['t']:+5.2f} signp={sign_test(up, len(r)):.4f}")
        keep.append((h, r))
    if era and keep:
        h, r = keep[min(2, len(keep) - 1)]
        for e in era_split(r.index, r.to_numpy()):
            if e["n"]:
                print(f"        era {e['label']}: n={e['n']} mean={e['mean_pct']:+.2f}% "
                      f"hit={e['hit']:.1f}%")
        print("        ", cluster_note(r.index, r.to_numpy()))
    return keep


print("=== A. SPY after a SKEW 21d collapse (declustered episodes) ===")
fwd("SPY", dc, "SKEW 21d in bottom 5%", era=True)
print()
lc = local_control(sk.index, pd.DatetimeIndex(dc), win=126)
fwd("SPY", lc, "local +/-126td control")
print()
fwd("SPY", sk.index, "all sessions since 1999")

print("\n=== B. VIX after the same episodes ===")
fwd("^VIX", dc, "SKEW 21d in bottom 5%", era=True)
print()
fwd("^VIX", lc, "local +/-126td control")
print()
fwd("^VIX", sk.index, "all sessions since 1999")

print("\n=== C. realized vol after, not direction (the 08-11 brief's measure) ===")
spy = px["SPY"].dropna()
rets = np.log(spy / spy.shift(1))
rv10 = rets.rolling(10).std().shift(-10) * np.sqrt(252) * 100
for nm, dates in [("SKEW 21d bottom 5%", dc), ("local control", lc),
                  ("all sessions", sk.index)]:
    v = rv10.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(v):
        print(f"   {nm:<28} n={len(v):>4}  next-10d realized vol "
              f"mean={v.mean():5.2f}%  median={np.median(v):5.2f}%")

print("\n=== D. the round trip: how fast did the tail bid drain ===")
r5 = sk / sk.shift(5) - 1.0
rank5 = r5.rolling(252).rank(pct=True) * 100
hot = (rank5 >= 95).fillna(False)
# a 21d-collapse episode that had a 5d top-5% surge inside the prior 10 sessions
whip = []
for d in dc:
    i = sk.index.get_loc(d)
    lo = max(0, i - 10)
    if bool(hot.iloc[lo:i + 1].any()):
        whip.append(d)
whip = pd.DatetimeIndex(whip)
print(f"   episodes preceded by a 5d top-5% SKEW surge inside 10 sessions: "
      f"{len(whip)} of {len(dc)}")
print("   dates:", [str(d.date()) for d in whip[-12:]])
if len(whip) >= 5:
    fwd("SPY", whip, "whipsaw episodes only")
    print()
    fwd("^VIX", whip, "whipsaw episodes only")

print("\n=== E. where SKEW sits in level terms, not change terms ===")
lvl_rank = sk.rolling(252).rank(pct=True) * 100
print(f"   SKEW level rank in its trailing year: {lvl_rank.iloc[-1]:.1f}")
print(f"   the collapse is in the CHANGE, the LEVEL is mid-range: "
      f"{sk.iloc[-1]:.2f} vs 252d min {sk.rolling(252).min().iloc[-1]:.2f} "
      f"max {sk.rolling(252).max().iloc[-1]:.2f}")
both = sk.index[(fire.fillna(False) & (lvl_rank <= 25)).to_numpy()]
print(f"   episodes where the change AND the level were both bottom-quartile: "
      f"{len(declusters(pd.DatetimeIndex(both), 21, sk.index))}")
