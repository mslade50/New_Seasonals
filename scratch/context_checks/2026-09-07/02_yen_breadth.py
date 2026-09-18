"""Drill 02 — the yen complex went in together. The engine scored each cross alone.

Friday: JPY=X -2.05%, GBPJPY -1.72, EURJPY -1.69, CADJPY -1.67, AUDJPY -1.53,
CHFJPY -1.39, NZDJPY -1.47. Six crosses landed in the bottom 5% of their own
trailing-year 5d range on ONE session, and EURJPY/JPY=X also printed >= 2 ATR
down days.

`P5:rank5_extreme` gave JPY=X n=326 +0.121% h1 sign p 0.0005 and EURJPY n=286
+0.109% sign p 0.0004, both BH-pass. But those are per-cross cells. The real
state is BREADTH: how often does the whole complex go at once, and does the
bounce look different when it does? A single cross washing out is common; six
at once is a different animal.

Convention: anchor is the session the state printed, h1 is the next session.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, fwd_ret, pct_rank, summarize, era_split, cluster_note,
    sign_test, declusters, local_control, show,
)

CROSSES = ["JPY=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "CADJPY=X",
           "CHFJPY=X", "NZDJPY=X"]
px = close_panel(CROSSES + ["^GSPC", "SPY"])
ASOF = pd.Timestamp("2026-09-04")   # FX prints on US holidays; cut at the asof session
px = px[px.index <= ASOF]
fx = px[CROSSES].dropna(how="all")
print(f"fx panel {fx.index[0].date()} .. {fx.index[-1].date()}  n={len(fx)}")

# 5d return percentile inside the trailing 252 sessions, per cross.
# pct_rank takes the PRICE series and differences internally -- passing a
# pre-computed return double-differences it and silently rescores the cell.
rank = pd.DataFrame({c: pct_rank(fx[c], 5, 252) for c in CROSSES})
low = rank < 5.0                       # bottom 5% of its own year
n_low = low.sum(axis=1)
valid = rank.notna().sum(axis=1)
n_low = n_low.where(valid >= 6)

today = fx.index[-1]
print(f"\nlatest session {today.date()}: {int(n_low.loc[today])} of "
      f"{int(valid.loc[today])} crosses in the bottom 5%")
print("  per-cross 5d rank: " +
      ", ".join(f"{c} {rank[c].loc[today]:.1f}" for c in CROSSES))

BREADTH = 6
trig_all = n_low.index[n_low >= BREADTH]
trig = declusters(trig_all, 5, fx.index)      # 5td min gap, one per episode
print(f"\nbreadth >= {BREADTH}: {len(trig_all)} raw sessions -> "
      f"{len(trig)} declustered episodes")
print(f"  episodes: {[str(d.date()) for d in trig]}")

# a single cross washing out, for contrast
trig1_all = n_low.index[n_low == 1]
trig1 = declusters(trig1_all, 5, fx.index)

for sub in ["JPY=X", "EURJPY=X", "AUDJPY=X"]:
    s = fx[sub].dropna()
    f1, f5 = fwd_ret(s, 1), fwd_ret(s, 5)
    base = f1.dropna()
    ctrl = local_control(base.index, trig, 126)
    rows = [
        summarize(f1.reindex(trig).dropna().values, f"complex washout (>= {BREADTH})"),
        summarize(f1.reindex(trig1).dropna().values, "one cross only"),
        summarize(f1.reindex(ctrl).dropna().values, "local +/-126td control"),
        summarize(base.values, "all days"),
    ]
    show(rows, f"=== {sub} h1 ===")
    v = f1.reindex(trig).dropna()
    if len(v) >= 5:
        w, n = int((v.values > 0).sum()), len(v)
        print(f"  record {w}-{n - w} up, sign p(up) {sign_test(w, n):.4f}")
        for e in era_split(v.index, v.values):
            print(f"    era {e['label']}: n={e['n']} mean {e['mean_pct']:+.3f}% "
                  f"hit {e['hit']:.1f}")
        print(f"    conc: {cluster_note(v.index, v.values)}")
        v5 = f5.reindex(trig).dropna()
        print(f"    h5: n={len(v5)} mean {100 * v5.mean():+.3f}% "
              f"hit {100 * (v5.values > 0).mean():.1f}")
    print()

# does a yen-complex washout say anything about US equities the next session?
print("=== ^GSPC h1 after a yen-complex washout (spillover check) ===")
sp = px["^GSPC"].dropna()
f1s = fwd_ret(sp, 1)
trig_sp = pd.DatetimeIndex([d for d in trig if d in set(sp.index)])
print(f"  {len(trig_sp)} of {len(trig)} episodes fall on an NYSE session")
show([summarize(f1s.reindex(trig_sp).dropna().values, "after yen washout"),
      summarize(f1s.dropna().values, "all days")], "^GSPC")
vs = f1s.reindex(trig_sp).dropna()
if len(vs):
    w = int((vs.values > 0).sum())
    print(f"  record {w}-{len(vs) - w} up, sign p {sign_test(w, len(vs)):.4f}")
