"""^VIX +6.60% on a ^GSPC session that fell only 0.52%, with ^VVIX +7.36%,
^MOVE +8.70% and ^SKEW +3.29%.

No trigger fired: P10c wants VIX +10%, P9d wants VIX up on an UP day. Neither
describes a vol bid this size against an index move this small. What follows a day
where the vol complex moves far more than the tape justifies?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, fwd_ret, declusters, local_control, summarize, era_split,
    sign_test, cluster_note,
)

px = close_panel(["SPY", "^GSPC", "^VIX", "^VIX3M", "^VVIX", "^MOVE"]).dropna(how="all")
spx, vix = px["^GSPC"].dropna(), px["^VIX"].dropna()
spx_r = spx / spx.shift(1) - 1.0
vix_r = vix / vix.shift(1) - 1.0

print(f"today: SPX {100*spx_r.iloc[-1]:+.2f}%  VIX {100*vix_r.iloc[-1]:+.2f}%  "
      f"VIX level {vix.iloc[-1]:.2f}")
ts = px["^VIX"] / px["^VIX3M"]
print(f"VIX/VIX3M {ts.iloc[-1]:.3f} (was {ts.iloc[-2]:.3f})")


def line(d):
    if d.get("n", 0) == 0:
        return "n=0"
    up = int(round(d["hit"] / 100 * d["n"]))
    return (f"n={d['n']:5d} mean={d['mean_pct']:+.3f}% med={d['median_pct']:+.3f}% "
            f"hit={d['hit']:5.1f}% t={d['t']:+.2f} rec={up}-{d['n']-up} "
            f"signp={sign_test(up, d['n']):.4f}")


common = spx_r.dropna().index.intersection(vix_r.dropna().index)
mask = pd.Series(False, index=common)
mask[:] = (vix_r.reindex(common) >= 0.05) & (spx_r.reindex(common) > -0.01) & \
          (spx_r.reindex(common) < 0)
trig = common[mask.values]
print(f"\n=== VIX +5% or more, SPX down but less than 1% : n = {len(trig)} ===")
for t in ["SPY", "^VIX"]:
    for h in [1, 5, 10]:
        print(f"  {t:6s} h{h:<3d} {line(summarize(fwd_ret(px[t], h).reindex(trig).dropna().values))}")

print("\n=== controls ===")
for t in ["SPY", "^VIX"]:
    for h in [1, 5, 10]:
        print(f"  {t:6s} h{h:<3d} all days   "
              f"{line(summarize(fwd_ret(px[t], h).dropna().values))}")
print()
for t in ["SPY", "^VIX"]:
    ctrl = local_control(px[t].dropna().index, trig, win=126).difference(trig)
    for h in [1, 5]:
        print(f"  {t:6s} h{h:<3d} local      "
              f"{line(summarize(fwd_ret(px[t], h).reindex(ctrl).dropna().values))}")

print("\n=== contrast: VIX +5%+ on a session SPX fell MORE than 1% ===")
big = common[((vix_r.reindex(common) >= 0.05) & (spx_r.reindex(common) <= -0.01)).values]
for t in ["SPY", "^VIX"]:
    for h in [1, 5]:
        print(f"  {t:6s} h{h:<3d} {line(summarize(fwd_ret(px[t], h).reindex(big).dropna().values))}")

print("\n=== contrast: SPX down 0-1% with VIX NOT up 5% ===")
mild = common[((vix_r.reindex(common) < 0.05) & (spx_r.reindex(common) > -0.01)
               & (spx_r.reindex(common) < 0)).values]
for t in ["SPY", "^VIX"]:
    for h in [1, 5]:
        print(f"  {t:6s} h{h:<3d} {line(summarize(fwd_ret(px[t], h).reindex(mild).dropna().values))}")

print("\n=== declustered 5td, era, concentration (SPY h1) ===")
dec = declusters(trig, 5, px.index)
print(f"  n declustered = {len(dec)}")
for t in ["SPY", "^VIX"]:
    for h in [1, 5]:
        print(f"  {t:6s} h{h:<3d} {line(summarize(fwd_ret(px[t], h).reindex(dec).dropna().values))}")
v = fwd_ret(px["SPY"], 1).reindex(trig).dropna()
for part in era_split(v.index, v.values):
    u = int(round(part["hit"] / 100 * part["n"]))
    print(f"  SPY h1 {part.get('label',''):12s} n={part['n']:4d} "
          f"mean={part['mean_pct']:+.3f}% rec={u}-{part['n']-u}")
print(f"  {cluster_note(v.index, v.values, k=2)}")

# tighten toward tonight: shallow tape, vol bid, and VIX still historically low
print("\n=== tighter: same, and VIX 21d rank in the bottom third of its year ===")
rank21 = vix.rolling(252).apply(lambda w: 100.0 * (w <= w[-1]).mean(), raw=True)
print(f"  VIX 252d percentile today = {rank21.iloc[-1]:.0f}")
tight = trig[(rank21.reindex(trig) <= 40).values]
print(f"  n = {len(tight)}")
for t in ["SPY", "^VIX"]:
    for h in [1, 5]:
        print(f"  {t:6s} h{h:<3d} {line(summarize(fwd_ret(px[t], h).reindex(tight).dropna().values))}")

print("\n=== how unusual is the VIX move given the tape? ===")
# rank today's |VIX move| among days with a similar-size index move
similar = common[(np.abs(spx_r.reindex(common) - spx_r.iloc[-1]) <= 0.002).values]
vv = vix_r.reindex(similar).dropna()
print(f"  sessions with SPX within 0.2pp of {100*spx_r.iloc[-1]:+.2f}%: n={len(vv)}")
print(f"  their VIX moves: mean {100*vv.mean():+.2f}%  median {100*vv.median():+.2f}%  "
      f"today {100*vix_r.iloc[-1]:+.2f}% ranks at the "
      f"{100*(vv <= vix_r.iloc[-1]).mean():.0f}th percentile")
