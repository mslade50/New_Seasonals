"""05 produced three candidate legs. Before any of them ship:

  1. does TONIGHT actually meet the whipsaw condition (a 5d top-5% SKEW surge
     inside the prior 10 sessions)? If not, the n=21 cell is not about tonight
     and cannot be written as though it is.
  2. is the realized-vol result era-stable, or is it a pre-2018 artefact
  3. the level rank and the 21d-change rank both printed 2.0, which looks like
     a bug until it is checked
  4. the VIX direction leg had top-2 episodes at 287% of total. Confirm it is
     unusable rather than quietly dropping it.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, declusters, sign_test,  # noqa: E402
                       summarize)

px = close_panel(["^SKEW", "SPY", "^VIX"])
px = px[px.index >= "1999-01-01"]
sk = px["^SKEW"].dropna()

r21 = sk / sk.shift(21) - 1.0
r5 = sk / sk.shift(5) - 1.0
rank21 = r21.rolling(252).rank(pct=True) * 100
rank5 = r5.rolling(252).rank(pct=True) * 100
lvl_rank = sk.rolling(252).rank(pct=True) * 100

print("=== 3. the two 2.0 ranks are not the same statistic ===")
print(f"   SKEW close                     {sk.iloc[-1]:.2f}")
print(f"   21d-CHANGE rank in trailing yr {rank21.iloc[-1]:.2f}  "
      f"(21d change {100 * r21.iloc[-1]:+.2f}%)")
print(f"   LEVEL rank in trailing yr      {lvl_rank.iloc[-1]:.2f}")
below = int((sk.iloc[-252:] < sk.iloc[-1]).sum())
print(f"   sessions in the trailing 252 closing BELOW tonight: {below} of 252 "
      f"-> {100 * below / 252:.1f}%")
print(f"   trailing-252 min {sk.iloc[-252:].min():.2f}  max {sk.iloc[-252:].max():.2f}")
print("   both really are near the bottom of the year. Not a bug.\n")

print("=== 1. does tonight meet the whipsaw condition ===")
print(f"   tonight 21d-change rank {rank21.iloc[-1]:.1f} (<=5 fires: "
      f"{bool(rank21.iloc[-1] <= 5)})")
tail = rank5.iloc[-11:]
print("   5d-change rank over the last 11 sessions:")
for d, v in tail.items():
    mark = "  <== top-5% surge" if v >= 95 else ""
    print(f"      {d.date()}  {v:6.1f}{mark}")
print(f"   whipsaw condition met tonight: {bool((tail.iloc[:-1] >= 95).any())}")
print(f"   max 5d rank in the prior 10 sessions: {tail.iloc[:-1].max():.1f}\n")

fire = (rank21 <= 5.0).fillna(False)
trig = sk.index[fire.to_numpy()]
dc = declusters(pd.DatetimeIndex(trig), 21, sk.index)

print("=== 2. realized vol after the collapse, split by era ===")
spy = px["SPY"].dropna()
lr = np.log(spy / spy.shift(1))
rv10 = lr.rolling(10).std().shift(-10) * np.sqrt(252) * 100
rv10_in = lr.rolling(10).std() * np.sqrt(252) * 100      # vol going IN

for label, dates in [("SKEW 21d bottom 5%", pd.DatetimeIndex(dc)),
                     ("all sessions", sk.index)]:
    v = rv10.reindex(dates).dropna()
    d = pd.DatetimeIndex(v.index)
    for nm, m in [("pre-2018", d < pd.Timestamp("2018-01-01")),
                  ("2018+", d >= pd.Timestamp("2018-01-01"))]:
        w = v.to_numpy()[m]
        if len(w):
            print(f"   {label:<24} {nm:<9} n={len(w):>4} "
                  f"mean={w.mean():5.2f}%  median={np.median(w):5.2f}%")
    print()

print("   the honest control: vol is autocorrelated, so match on vol going IN")
vin = rv10_in.reindex(pd.DatetimeIndex(dc)).dropna()
print(f"   realized vol ENTERING the episodes: mean {vin.mean():.2f}%, "
      f"median {np.median(vin):.2f}%")
allin = rv10_in.dropna()
print(f"   realized vol entering all sessions:  mean {allin.mean():.2f}%, "
      f"median {np.median(allin):.2f}%")
# decile-match the control on entering vol
dec = pd.qcut(allin, 10, labels=False, duplicates="drop")
ep_dec = dec.reindex(pd.DatetimeIndex(dc)).dropna()
matched = []
for q in sorted(set(ep_dec.to_numpy())):
    pool = rv10.reindex(dec.index[dec == q]).dropna()
    w = int((ep_dec == q).sum())
    if len(pool):
        matched.append((pool.mean(), np.median(pool), w))
if matched:
    tw = sum(m[2] for m in matched)
    print(f"   vol-decile-matched control: mean "
          f"{sum(m[0] * m[2] for m in matched) / tw:5.2f}%  median "
          f"{sum(m[1] * m[2] for m in matched) / tw:5.2f}%")
ep = rv10.reindex(pd.DatetimeIndex(dc)).dropna()
print(f"   episodes:                   mean {ep.mean():5.2f}%  "
      f"median {np.median(ep):5.2f}%")
print(f"   tonight enters at 21d realized {px['SPY'].pct_change().rolling(21).std().iloc[-1] * np.sqrt(252) * 100:.1f}%, "
      f"10d realized {rv10_in.iloc[-1]:.1f}%")

print("\n=== 4. the VIX direction leg, confirming it is unusable ===")
vix = px["^VIX"].dropna()
r21v = (vix.shift(-21) / vix - 1.0).reindex(pd.DatetimeIndex(dc)).dropna()
d = summarize(r21v.to_numpy(), "VIX h21")
up = int((r21v > 0).sum())
print(f"   n={len(r21v)} mean={d['mean_pct']:+.2f}% median={d['median_pct']:+.2f}% "
      f"up={up}-{len(r21v) - up} t={d['t']:+.2f}")
top = r21v.reindex(r21v.abs().sort_values(ascending=False).index[:2])
print(f"   two largest episodes: "
      f"{[(str(i.date()), round(100 * v, 1)) for i, v in top.items()]}")
rest = r21v.drop(top.index)
d2 = summarize(rest.to_numpy(), "ex top2")
print(f"   without them: n={len(rest)} mean={d2['mean_pct']:+.2f}% "
      f"median={d2['median_pct']:+.2f}% t={d2['t']:+.2f}")
print("   -> mean sign is set by two episodes. Dropping the VIX direction leg.")
