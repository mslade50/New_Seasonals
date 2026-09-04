"""Europe closed at 52w highs on a 5-session run while the US closed lower.

^FCHI: 5+ consecutive up closes, n=205, mean -0.158%, hit 43.9%, t=-2.03,
90-115, sign p 0.0467, era stable per the sweep. Today it printed that streak
AT a 52w high, with ^GDAXI also at its high (63d ranks 99.2 and 95.2) while
SPY, QQQ and IWM all closed lower. Anchor = the session the state printed.
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

TIX = ["^FCHI", "^GDAXI", "SPY", "^GSPC"]
px = close_panel(TIX)
px = px[px.index >= "1999-01-01"]
for t in TIX:
    s = px[t].dropna()
    print(f"  {t:12s} {len(s):5d} bars {s.index.min().date()} -> {s.index.max().date()}"
          if len(s) else f"  {t:12s} EMPTY")
dates = px.index


def up_streak(s, k=5):
    up = s.pct_change() > 0
    run = up.groupby((~up).cumsum()).cumcount() + 1
    return up & (run >= k)


def dist_high(s, win=252):
    return s / s.rolling(win).max() - 1.0


fc, dx = px["^FCHI"], px["^GDAXI"]
print(f"\nlive: ^FCHI dist-high {100*dist_high(fc).iloc[-1]:+.3f}%  "
      f"^GDAXI dist-high {100*dist_high(dx).iloc[-1]:+.3f}%")
st = up_streak(fc)
print(f"^FCHI on a 5+ up streak today: {bool(st.iloc[-1])}; "
      f"streak sessions in history: {int(st.sum())}")


def run(mask, label, subjects=("^FCHI", "^GDAXI", "SPY"), gap=10, hs=(1, 5)):
    trig = dates[mask.reindex(dates).fillna(False).values]
    trig = trig[trig <= dates[-2]]
    if len(trig) == 0:
        print(f"\n=== {label}: NONE")
        return
    dc = declusters(trig, gap, dates)
    ctrl = local_control(dates, dc, win=126)
    print(f"\n=== {label}")
    print(f"   raw {len(trig)} -> {len(dc)} episodes; years "
          f"{sorted(set(pd.DatetimeIndex(dc).year))}")
    for sub in subjects:
        s = px[sub].dropna()
        if s.empty:
            continue
        for h in hs:
            f = fwd_ret(s, h)
            v = f.reindex(dc).dropna()
            if len(v) < 3:
                continue
            r = summarize(v.values, "")
            base = summarize(f.dropna().values, "")
            loc = summarize(f.reindex(ctrl).dropna().values, "")
            up = int((v.values > 0).sum())
            print(f"   {sub:10s} h{h:<3d} n={r['n']:3d} mean {r['mean_pct']:+.3f}% "
                  f"med {r['median_pct']:+.3f}% hit {r['hit']:.0f}% t={r['t']:+.2f} | "
                  f"{up}-{len(v)-up} up-p {sign_test(up, len(v)):.4f} "
                  f"dn-p {sign_test(len(v)-up, len(v)):.4f} | all {base['mean_pct']:+.3f}% "
                  f"local {loc['mean_pct']:+.3f}%")
            if h == 1 and sub == "^FCHI":
                print(f"          {cluster_note(v.index, v.values)}")
                for e in era_split(v.index, v.values):
                    if e.get("n", 0):
                        print(f"          era n={e['n']:3d} mean {e['mean_pct']:+.3f}% "
                              f"hit {e['hit']:.0f}% t={e['t']:+.2f}")


run(st, "^FCHI 5+ consecutive up closes (the sweep's cell, declustered)", gap=1)
run(st, "^FCHI 5+ up closes, 10td declustered")
run(st & (dist_high(fc) > -0.002), "^FCHI 5+ up closes AND at a 52w high")
run(st & (dist_high(fc) > -0.002) & (dist_high(dx) > -0.002),
    "^FCHI 5+ up closes, BOTH ^FCHI and ^GDAXI at 52w highs")
run(dist_high(fc) > -0.002, "^FCHI at a 52w high (streak ignored)", gap=10)

# the divergence: Europe at a high, US down on the day
us_dn = px["SPY"].pct_change() < 0
run(st & (dist_high(fc) > -0.002) & us_dn,
    "^FCHI 5+ up closes at a 52w high while SPY closed lower")
