"""Gold's 5-session move as a MAGNITUDE cell, not a percentile cell.

The 2026-08-09 brief already published the P5 rank cell (GC 5d return in the
top 5% of its year). GC is up 10.29% over five sessions, which is a much rarer
object than the 95th percentile of a quiet year. Silver +14.24%, platinum
+9.03%, palladium +11.0% over the same window, so the complex version is worth
a look too. Anchor = the session the state printed, lag=0.
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

TIX = ["GC=F", "SI=F", "PL=F", "PA=F", "DX-Y.NYB", "SPY", "^TNX"]
px = close_panel(TIX)
px = px[px.index >= "1999-01-01"]
dates = px.index
for t in TIX:
    s = px[t].dropna()
    print(f"  {t:10s} {len(s):5d} bars {s.index.min().date()} -> {s.index.max().date()}")

r5 = {t: px[t] / px[t].shift(5) - 1.0 for t in ("GC=F", "SI=F", "PL=F", "PA=F")}
print("\nlive 5d: " + "  ".join(f"{t} {100*r5[t].iloc[-1]:+.2f}%" for t in r5))
dxy21 = px["DX-Y.NYB"] / px["DX-Y.NYB"].shift(21) - 1.0
print(f"live DXY 21d {100*dxy21.iloc[-1]:+.2f}%")


def run(mask, label, subjects=("GC=F", "SI=F"), gap=10, hs=(1, 5, 21)):
    trig = dates[mask.reindex(dates).fillna(False).values]
    trig = trig[trig <= dates[-2]]
    if len(trig) == 0:
        print(f"\n=== {label}: NONE")
        return
    dc = declusters(trig, gap, dates)
    ctrl = local_control(dates, dc, win=126)
    print(f"\n=== {label}")
    print(f"   raw {len(trig)} sessions -> {len(dc)} episodes; years "
          f"{sorted(set(pd.DatetimeIndex(dc).year))}")
    for sub in subjects:
        s = px[sub].dropna()
        for h in hs:
            f = fwd_ret(s, h)
            v = f.reindex(dc).dropna()
            if len(v) < 3:
                continue
            r = summarize(v.values, "")
            base = summarize(f.dropna().values, "")
            loc = summarize(f.reindex(ctrl).dropna().values, "")
            up = int((v.values > 0).sum())
            print(f"   {sub:10s} h{h:<3d} n={r['n']:3d} mean {r['mean_pct']:+.2f}% "
                  f"med {r['median_pct']:+.2f}% hit {r['hit']:.0f}% t={r['t']:+.2f} | "
                  f"{up}-{len(v)-up} up-p {sign_test(up, len(v)):.4f} "
                  f"dn-p {sign_test(len(v)-up, len(v)):.4f} | all {base['mean_pct']:+.2f}% "
                  f"local {loc['mean_pct']:+.2f}%")
            if h == 5:
                print(f"          {cluster_note(v.index, v.values)}")
                for e in era_split(v.index, v.values):
                    if e.get("n", 0) == 0:
                        print("          era n=  0")
                        continue
                    print(f"          era n={e['n']:3d} mean {e['mean_pct']:+.2f}% "
                          f"hit {e['hit']:.0f}% t={e['t']:+.2f}")


for thr in (0.06, 0.08, 0.10):
    run(r5["GC=F"] >= thr, f"GC=F 5d return >= +{100*thr:.0f}%")

# the complex, all four precious metals stretched at once
run((r5["GC=F"] >= 0.05) & (r5["SI=F"] >= 0.08) & (r5["PL=F"] >= 0.05) & (r5["PA=F"] >= 0.05),
    "gold +5%, silver +8%, platinum +5% and palladium +5% over the same 5 sessions",
    subjects=("GC=F", "SI=F", "SPY"))

# is it a dollar move or a gold move
run((r5["GC=F"] >= 0.08) & (dxy21 <= -0.01),
    "GC=F 5d >= +8% with the dollar down over 21 sessions", subjects=("GC=F", "DX-Y.NYB"))
run((r5["GC=F"] >= 0.08) & (dxy21 > -0.01),
    "GC=F 5d >= +8% with the dollar NOT down over 21 sessions", subjects=("GC=F", "DX-Y.NYB"))

# gold still well below its own high while thrusting (today: -16.35%)
dh = px["GC=F"] / px["GC=F"].rolling(252).max() - 1.0
print(f"\nlive GC dist-52w-high {100*dh.iloc[-1]:+.2f}%")
run((r5["GC=F"] >= 0.08) & (dh <= -0.10), "GC=F 5d >= +8% while still 10%+ below its 52w high")
run((r5["GC=F"] >= 0.08) & (dh > -0.10), "GC=F 5d >= +8% within 10% of its 52w high")
