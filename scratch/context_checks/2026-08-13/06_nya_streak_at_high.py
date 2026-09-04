"""The sweep's ^NYA up-streak cell: 5+ consecutive up closes, n=239, next
session 109-130 down, sign p 0.0978, era-stable. Weak on its own.

Tonight it comes with two extra conditions: ^NYA closed AT a 52-week high and
its 21-day return is in the 90.5th percentile of its year. Does the streak cell
sharpen when the run ends at a high, or is it the same tired mean-reversion?

Drill 04 already covered the joint-high cluster with SPY forward. This is a
different object: ^NYA's own streak, ^NYA forward. If it overlaps 04 it does
not ship.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, declusters, era_split,  # noqa: E402
                       fwd_ret, sign_test, summarize)

px = close_panel(["^NYA", "SPY"])
px = px[px.index >= "1999-01-01"]
s = px["^NYA"].dropna()

up = (s > s.shift(1))
run = up * (up.groupby((~up).cumsum()).cumcount() + 1)
hi = s.rolling(252, min_periods=200).max()
at_high = (s / hi - 1.0) >= -0.0005

print(f"tonight: ^NYA run length {int(run.iloc[-1])}, "
      f"at a 52w high {bool(at_high.iloc[-1])}, "
      f"{100 * (s.iloc[-1] / hi.iloc[-1] - 1):+.2f}% from it")


def rep(name, mask, hs=(1, 5, 10), gap=5, era=False):
    dates = s.index[np.asarray(mask, bool)]
    dc = declusters(pd.DatetimeIndex(dates), gap, s.index)
    out = None
    for h in hs:
        r = fwd_ret(s, h).reindex(dc).dropna()
        if len(r) < 4:
            continue
        d = summarize(r.to_numpy(), name)
        u = int((r > 0).sum())
        print(f"   {name:<40} h{h:<3} n={len(r):>4} mean={d['mean_pct']:+6.3f}% "
              f"med={d['median_pct']:+6.3f}% up={u}-{len(r) - u} "
              f"({100 * u / len(r):4.1f}%) t={d['t']:+5.2f} "
              f"signp={sign_test(u, len(r)):.4f}")
        if h == 1:
            out = r
    if era and out is not None:
        for e in era_split(out.index, out.to_numpy()):
            if e["n"]:
                print(f"       era {e['label']}: n={e['n']} mean={e['mean_pct']:+.3f}% "
                      f"hit={e['hit']:.1f}%")
    return out


streak = (run >= 5).to_numpy()
ah = at_high.fillna(False).to_numpy()

print("\n=== the bare cell and the two conditioned versions ===")
rep("5+ up closes (bare)", streak, era=True)
print()
rep("5+ up closes ENDING at a 52w high", streak & ah, era=True)
print()
rep("5+ up closes NOT at a 52w high", streak & ~ah, era=True)
print()
rep("at a 52w high, no streak", ~streak & ah)
print()
rep("all sessions", np.ones(len(s), dtype=bool))

print("\n=== overlap with drill 04's cell (SPY+IWM+HYG joint highs) ===")
px2 = close_panel(["SPY", "IWM", "HYG"])
px2 = px2[px2.index >= "1999-01-01"]
common = px2.dropna().index
jm = None
for t in ("SPY", "IWM", "HYG"):
    v = px2[t]
    m = (v / v.rolling(252, min_periods=200).max() - 1.0) >= -0.0005
    jm = m if jm is None else (jm & m)
joint = set(common[jm.reindex(common).fillna(False).to_numpy()])
cell = set(s.index[streak & ah])
print(f"   ^NYA streak-at-high sessions: {len(cell)}")
print(f"   drill 04 joint-high sessions: {len(joint)}")
print(f"   overlap: {len(cell & joint)} "
      f"({100 * len(cell & joint) / max(1, len(cell)):.0f}% of this cell)")
