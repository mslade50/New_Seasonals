"""TLT closed exactly at its trailing-252 low while ^MOVE jumped 8.70%.

No trigger fired: P2 wants the first 52w low in 30+ calendar days and TLT also closed
at one on 2026-07-31. The state is live anyway and bond vol spiking into it has no
trigger at all. What has followed?
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

px = close_panel(["TLT", "IEF", "^TNX", "^MOVE", "SPY", "LQD"]).dropna(how="all")
tlt, move = px["TLT"].dropna(), px["^MOVE"].dropna()
print(f"TLT   {tlt.index[0].date()} .. {tlt.index[-1].date()}  n={len(tlt)}")
print(f"MOVE  {move.index[0].date()} .. {move.index[-1].date()}  n={len(move)}")

at_low = tlt <= tlt.rolling(252).min() + 1e-9
move_r = move / move.shift(1) - 1.0
common = tlt.index.intersection(move.index)

print(f"\ntoday: TLT at 252d low = {bool(at_low.iloc[-1])}, "
      f"MOVE 1d = {100*move_r.iloc[-1]:+.2f}%, "
      f"TLT 1d = {100*(tlt.iloc[-1]/tlt.iloc[-2]-1):+.2f}%")


def line(d):
    if d.get("n", 0) == 0:
        return "n=0"
    up = int(round(d["hit"] / 100 * d["n"]))
    return (f"n={d['n']:5d} mean={d['mean_pct']:+.3f}% med={d['median_pct']:+.3f}% "
            f"hit={d['hit']:5.1f}% t={d['t']:+.2f} rec={up}-{d['n']-up} "
            f"signp={sign_test(up, d['n']):.4f}")


def report(name, idx, tickers=("TLT", "^TNX", "SPY"), hs=(1, 5, 21)):
    idx = pd.DatetimeIndex(sorted(set(idx)))
    print(f"\n--- {name}  (n anchors = {len(idx)}) ---")
    for t in tickers:
        for h in hs:
            v = fwd_ret(px[t], h).reindex(idx).dropna()
            print(f"  {t:6s} h{h:<3d} {line(summarize(v.values))}")
    return idx


# 1. the bare state
base = report("TLT at a 252d low (every such close)", tlt.index[at_low])

# 2. with bond vol spiking
cond = [d for d in common if at_low.get(d, False) and move_r.get(d, np.nan) >= 0.05]
spike = report("TLT at a 252d low AND ^MOVE +5% or more", cond)

cond8 = [d for d in common if at_low.get(d, False) and move_r.get(d, np.nan) >= 0.08]
report("TLT at a 252d low AND ^MOVE +8% or more", cond8)

# 3. contrast: MOVE spike WITHOUT the low
cond_nolow = [d for d in common if (not at_low.get(d, False))
              and move_r.get(d, np.nan) >= 0.05]
report("^MOVE +5% or more, TLT NOT at a low", cond_nolow)

# 4. controls
print("\n--- controls ---")
for t in ["TLT", "^TNX", "SPY"]:
    for h in [1, 5, 21]:
        v = fwd_ret(px[t], h).dropna()
        print(f"  {t:6s} h{h:<3d} all days      {line(summarize(v.values))}")
print()
for t in ["TLT", "^TNX"]:
    ctrl = local_control(px[t].dropna().index, spike, win=126).difference(spike)
    for h in [1, 5, 21]:
        v = fwd_ret(px[t], h).reindex(ctrl).dropna()
        print(f"  {t:6s} h{h:<3d} local +/-126  {line(summarize(v.values))}")

# 5. declustered, era, concentration on the spike cell
print("\n--- spike cell, declustered 21td ---")
dec = declusters(pd.DatetimeIndex(spike), 21, px.index)
print(f"  n declustered = {len(dec)}: {[str(x.date()) for x in dec]}")
for t in ["TLT", "^TNX", "SPY"]:
    for h in [5, 21]:
        v = fwd_ret(px[t], h).reindex(dec).dropna()
        print(f"  {t:6s} h{h:<3d} {line(summarize(v.values))}")

print("\n--- spike cell, era split (h21 TLT) ---")
v = fwd_ret(px["TLT"], 21).reindex(spike).dropna()
for part in era_split(v.index, v.values):
    print(f"  {part.get('label',''):12s} {line(part)}")
print(f"  concentration: {cluster_note(v.index, v.values, k=2)}")

print("\n--- declustered episodes, TLT path ---")
for d in dec:
    row = [f"{100*fwd_ret(px['TLT'], h).get(d, np.nan):+6.2f}%" for h in (1, 5, 21, 63)]
    print(f"  {str(d.date()):12s} MOVE {100*move_r.get(d,np.nan):+5.1f}%  "
          f"TLT h1/h5/h21/h63 " + " ".join(row))
