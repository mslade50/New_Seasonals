"""C8 round 2 -- confirm the interaction cell is EMPTY, and price the two
fallbacks the round-1 output exposed.

Round 1's decisive line: at every yield gate from 0.5% to 2.0%, the number of
Aug ME-1 anchors with ^TNX near its trailing-252 high is **N=0 over 24 years**.
The only anchor inside 5% is 2003, and it LOSES at h=10 (-1.527%) and h=21
(-5.934%). So the pitched conjunction has no precedent; today would be the
first instance. This script (a) proves that from the raw distance series rather
than from an empty groupby, (b) charges the bare August anchor its own
max-of-12, and (c) prices the "any September session with the yield at a 52w
high" fallback the gate-attribution line surfaced (+1.506% / +2.386% short,
N=26 days).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np
import pandas as pd

px = close_panel(["^TNX", "TLT", "IEF"])
idx = px.index
tnx = px["^TNX"]
tnx_dist = tnx / rolling_on_valid(tnx, lambda x: x.rolling(252).max()) - 1.0


def month_end_anchors(index, month):
    s = pd.Series(index, index=index)
    out = []
    for p, g in s.groupby(index.to_period("M")):
        if p.month == month and len(g) >= 2:
            out.append(g.iloc[-2])
    return pd.DatetimeIndex(out)


print("=== 1. THE INTERACTION CELL IS EMPTY -- raw distances, every August ===")
a = month_end_anchors(idx, 8)
d = tnx_dist.reindex(a)
print("  ^TNX distance below its trailing-252 high on each Aug ME-1 anchor:")
for dt, val in d.items():
    flag = "  <-- would qualify" if val >= -0.02 else ""
    print(f"    {dt.date()}  {100*val:+7.2f}%{flag}")
print(f"\n  anchors within 2.0% of the 252d yield high: "
      f"{int((d >= -0.02).sum())} of {len(d)}   |  within 5.0%: "
      f"{int((d >= -0.05).sum())}   |  TODAY: -0.527%")
print("  => the pitched conjunction has ZERO precedent in 24 years. Today is "
      "instance #1, so there is nothing to measure it on.")

print("\n=== 2. charge the BARE August anchor its own max-of-12 ===")
rng = np.random.default_rng(3)
for veh in ("TLT", "IEF"):
    for h in (5, 10, 21):
        r = vehicle_ret(px, [(veh, -1.0)], h, 1)
        vals, sizes, means = [], [], {}
        for m in range(1, 13):
            am = month_end_anchors(idx, m)
            am = am[r.reindex(am).notna().values]
            v = r.loc[am].values
            vals.append(v); sizes.append(len(v)); means[m] = float(v.mean())
        allv = np.concatenate(vals)
        obs = means[8]
        rank = sorted(means, key=lambda m: -means[m]).index(8) + 1
        mx = []
        for _ in range(20000):
            p = rng.permutation(allv)
            i, best = 0, -np.inf
            for s in sizes:
                best = max(best, p[i:i + s].mean()); i += s
            mx.append(best)
        print(f"  SHORT {veh} h={h:2d}: AUGUST anchor {100*obs:+.3f}% ranks "
              f"{rank} of 12; P(max-of-12 >= Aug) = {(np.asarray(mx) >= obs).mean():.4f}")

print("\n=== 3. FALLBACK: ANY September session with ^TNX near its 52w high ===")
yh = tnx_dist >= -0.01
for h in (5, 10, 21):
    r = vehicle_ret(px, [("TLT", -1.0)], h, 1)
    ok = r.notna()
    days = idx[yh.values & ok.values & (idx.month == 9)]
    if len(days) == 0:
        print(f"  h={h}: no days"); continue
    e = declusters(days, max(h, 10), idx)
    v = r.loc[e].values
    ctrl_sep = r[ok & (idx.month == 9)].values
    ctrl_yh = r.loc[idx[yh.values & ok.values]].values
    print(f"  h={h:2d}: days {len(days)} in years {sorted(set(days.year))} -> "
          f"{len(e)} episodes")
    show([summarize(v, f"Sep x yield-high episodes"),
          summarize(ctrl_sep, "CTRL all Sep days"),
          summarize(ctrl_yh, "CTRL all yield-high days"),
          summarize(r[ok].values, "CTRL all days")], f"short TLT h={h}")
    carry = 1.79 * h
    bps = 100 * float(np.nanmean(v)) * 100
    print(f"    net of {3.0+carry:.1f} bp cost+carry: {bps-3.0-carry:+.1f} bps "
          f"-> {(bps-3.0-carry)/(3.0+carry):.2f}x  |  episodes: " +
          ", ".join(f"{str(x.date())} {100*y:+.2f}%" for x, y in zip(e, v)))
