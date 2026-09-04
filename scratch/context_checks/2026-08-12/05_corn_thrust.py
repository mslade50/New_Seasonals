"""Corn closed +10.13% today, at a 52-week high, with its 21-day return in the 96th
percentile of its year. Three price triggers on one instrument. The engine's forward
cells are all empty, so the question is not what corn does next, it is how far outside
its own distribution the session was.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, declusters, fwd_ret, sign_test, summarize  # noqa: E402

px = close_panel(["ZC=F", "ZW=F", "ZS=F", "DBC"])
zc = px["ZC=F"].dropna()
r1 = zc.pct_change()

print(f"corn close {zc.iloc[-1]:.2f}  1d {100 * r1.iloc[-1]:+.2f}%  "
      f"n sessions {len(zc)} from {zc.index[0].date()}")

print("\n=== how big was today by corn's own standards ===")
bigger = int((r1 > r1.iloc[-1]).sum())
print(f"  sessions with a larger 1-day gain: {bigger} of {r1.notna().sum()} "
      f"({100 * bigger / r1.notna().sum():.2f}%)")
print(f"  1-day gain percentile: {100 * (r1 < r1.iloc[-1]).mean():.2f}")
sd = r1.rolling(252).std().iloc[-2]
print(f"  today in trailing-252 sigmas: {r1.iloc[-1] / sd:+.2f}")
prev = r1[r1 >= 0.08].index
print(f"\n  prior sessions >= +8%: {len(prev[prev < zc.index[-1]])}")
for d in prev[prev < zc.index[-1]][-12:]:
    print(f"    {d.date()}  {100 * r1.loc[d]:+.2f}%")
last = prev[prev < zc.index[-1]]
if len(last):
    gap = np.busday_count(last[-1].date(), zc.index[-1].date())
    print(f"  business days since the last one: {gap} ({last[-1].date()})")

print("\n=== was it at a 52-week high too? ===")
hi = zc.rolling(252).max()
at_hi = zc >= hi * 0.9999
both = (r1 >= 0.08) & at_hi
print(f"  >= +8% AND closing at a 252d high: {int(both.sum())} in the whole history")
for d in both[both].index:
    print(f"    {d.date()}  {100 * r1.loc[d]:+.2f}%")

print("\n=== forward, the >= +8% cell (declustered 10td) ===")
cell = declusters(r1[r1 >= 0.08].index, 10, zc.index)


def show(label, idx, h, tkr="ZC=F"):
    f = fwd_ret(px[tkr].dropna(), h)
    a = pd.DatetimeIndex(idx).intersection(f.dropna().index)
    v = f.loc[a].values
    if len(v) == 0:
        print(f"  {label:<40} n=0")
        return
    d = summarize(v)
    up = int((v > 0).sum())
    print(f"  {label:<40} n={len(v):<4} mean={d['mean_pct']:+.3f}%  "
          f"med={d['median_pct']:+.3f}%  hit={d['hit']:.1f}%  t={d['t']:+.2f}  "
          f"{up}-{len(v) - up} up  sign p={sign_test(up, len(v)):.4f}")


for h in (1, 5, 21):
    show(f"corn >= +8% day, h{h}", cell, h)
fa = fwd_ret(zc, 5).dropna()
d = summarize(fa.values)
print(f"  {'all sessions h5':<40} n={len(fa):<4} mean={d['mean_pct']:+.3f}%  "
      f"med={d['median_pct']:+.3f}%  hit={d['hit']:.1f}%")

print("\n=== did the rest of the grain complex follow ===")
for t in ("ZW=F", "ZS=F"):
    s = px[t].dropna()
    print(f"  {t} today {100 * s.pct_change().iloc[-1]:+.2f}%")
for h in (1, 5):
    for t in ("ZW=F", "ZS=F"):
        show(f"after a corn >= +8% day, {t} h{h}", cell, h, t)
