"""QQQ underperforming SPY by a full point on a session the index barely moved.

No engine trigger covers relative performance, so the sweep is silent on the
most distinctive thing today's tape did: SPY -0.68% while QQQ fell 1.69% and
IWM 1.26%. This cell is built from the tape rather than inherited, which is
recorded in the cell map as a tape-derived cross.

The cell: QQQ - SPY <= -1.0pp on a session where SPY itself fell less than 1%.
That isolates a rotation out of large-cap tech from a general index selloff.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, summarize, show, sign_test, era_split, cluster_note,
    declusters, local_control,
)

TK = ["SPY", "QQQ", "IWM"]
px = close_panel(TK)
idx = px.index
r = px.pct_change(fill_method=None)
spread = r["QQQ"] - r["SPY"]

print("today: SPY %+.2f%%  QQQ %+.2f%%  IWM %+.2f%%  QQQ-SPY %+.2fpp" % (
    100 * r["SPY"].iloc[-1], 100 * r["QQQ"].iloc[-1], 100 * r["IWM"].iloc[-1],
    100 * spread.iloc[-1]))
print("spread percentile in trailing 252d: %.1f" % (
    100 * (spread.tail(252) < spread.iloc[-1]).mean()))

mask = (spread <= -0.01) & (r["SPY"] < 0) & (r["SPY"] > -0.01)
trig = idx[mask.fillna(False).values]
trig = trig[trig < idx[-1]]  # exclude today, it has no forward return
print(f"\nraw trigger days: {len(trig)}")
epi = declusters(pd.DatetimeIndex(trig), 5, idx)
print(f"declustered episodes (5td): {len(epi)}")
print("  last twelve:", [str(d.date()) for d in epi][-12:])

print("\n" + "=" * 74)
print("A. what follows, per index, episode level")
print("=" * 74)
for t in TK:
    out = []
    for h in (1, 3, 5, 10, 21):
        f = px[t].shift(-h) / px[t] - 1.0
        v = f.loc[f.index.intersection(epi)].dropna().values
        row = summarize(v, f"{t} h={h}")
        row["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
        row["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        base = f.dropna()
        row["ctl_all_pct"] = round(100 * base.mean(), 3)
        row["edge_pp"] = round(row["mean_pct"] - 100 * base.mean(), 3)
        out.append(row)
    show(out, f"{t} after a 1pp tech underperformance on a shallow down day")

print("\n" + "=" * 74)
print("B. does the SPREAD keep going, or snap back?")
print("=" * 74)
cq = px["QQQ"] / px["SPY"]
out = []
for h in (1, 3, 5, 10, 21):
    f = cq.shift(-h) / cq - 1.0
    v = f.loc[f.index.intersection(epi)].dropna().values
    row = summarize(v, f"QQQ/SPY ratio h={h}")
    row["rec"] = f"{int((v > 0).sum())}-{int((v <= 0).sum())}"
    row["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    base = f.dropna()
    row["ctl_all_pct"] = round(100 * base.mean(), 3)
    row["edge_pp"] = round(row["mean_pct"] - 100 * base.mean(), 3)
    out.append(row)
show(out, "the relative leg: QQQ over SPY")

print("\n" + "=" * 74)
print("C. era, concentration and local control on the best horizon")
print("=" * 74)
for t, h in [("QQQ", 5), ("SPY", 5), ("QQQ", 1)]:
    f = px[t].shift(-h) / px[t] - 1.0
    s = f.loc[f.index.intersection(epi)].dropna()
    show(era_split(s.index, s.values), f"{t} h={h}: era split")
    print(" ", cluster_note(s.index, s.values, k=2))
    valid = f.dropna().index
    loc = local_control(valid, pd.DatetimeIndex(epi).intersection(valid), win=126)
    print(f"  CTRL local +/-126td ex-trigger: {100*f.loc[loc].mean():+.3f}% "
          f"(n={len(loc)})")

print("\n" + "=" * 74)
print("D. how unusual is a 1pp gap on a sub-1% down day at all?")
print("=" * 74)
shallow = (r["SPY"] < 0) & (r["SPY"] > -0.01)
print(f"  shallow down days: {int(shallow.sum())}")
print(f"  of those, QQQ lagged by >=1pp: {int((shallow & (spread <= -0.01)).sum())} "
      f"({100*(spread[shallow] <= -0.01).mean():.1f}%)")
print(f"  QQQ-SPY spread on all days: mean {100*spread.mean():+.3f}pp, "
      f"sd {100*spread.std():.3f}pp")
by_yr = spread[shallow & (spread <= -0.01)].groupby(
    idx[shallow & (spread <= -0.01)].year).size()
print("  trigger count by year:", dict(by_yr.tail(12)))
