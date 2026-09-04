"""The dollar at a 21-day-return floor, recovered from the sweep's per-trigger cap.

DX-Y.NYB and UUP were both dropped from P5b:rank21_extreme by the cap (kept 8,
ranked by session move) because the dollar barely moved today. But the STATE is
the point: DXY's 21d return sits in the 2nd percentile of its own year while gold
is up 10.3% over the same window. This recomputes the dropped cell rather than
inheriting it, and cross-checks against JPY=X, which did survive the cap
(n=385, +0.087%, 223-162, sign p 0.0011, BH pass).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import close_panel, cluster_note, declusters, era_split, local_control, sign_test, summarize  # noqa


def report(label, v):
    v = np.asarray(v)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        print(f"  {label:<48} n=0")
        return
    st = summarize(v, label)
    up = int((v > 0).sum())
    print(
        f"  {label:<48} n={st['n']:<5} mean={st['mean_pct']:+.3f}%  med={st['median_pct']:+.3f}%"
        f"  {up}-{st['n'] - up}  hit={st['hit']:.1f}%  t={st['t']:+.2f}  sign_p={sign_test(up, st['n']):.4f}"
    )


cp = close_panel(["DX-Y.NYB", "JPY=X", "GC=F"])

for tkr in ("DX-Y.NYB", "JPY=X"):
    px = cp[tkr].dropna()
    idx = px.index
    r21 = px.pct_change(21)
    rank21 = r21.rolling(252).apply(lambda w: (w[-1] > w[:-1]).mean() * 100, raw=True).values
    fwd1 = px.pct_change().shift(-1).values
    fwd5 = (px.shift(-5) / px - 1.0).values
    fwd21 = (px.shift(-21) / px - 1.0).values

    print(f"\n=== {tkr}  {idx[0].date()} to {idx[-1].date()} ===")
    print(f"  live: close {px.iloc[-1]:.4f}, 21d {r21.iloc[-1] * 100:+.2f}%, 21d rank {rank21[-1]:.1f}")

    cell = (rank21 < 5) & ~np.isnan(fwd1)
    report("21d return in the bottom 5% of its year, h1", fwd1[cell])
    report("  same cell, h5", fwd5[cell & ~np.isnan(fwd5)])
    report("  same cell, h21", fwd21[cell & ~np.isnan(fwd21)])
    print("  controls:")
    report("  all sessions, h1 (own drift)", fwd1)
    lc = local_control(idx, idx[cell], win=126)
    report("  local +/-126td, h1", px.pct_change().shift(-1).reindex(lc).values)

    # declustered so a single multi-month slump does not vote 60 times
    dc = declusters(idx[cell], min_gap_td=21, all_dates=idx)
    dv = px.pct_change().shift(-1).reindex(dc).values
    report("  declustered at 21td, h1", dv)
    dv5 = (px.shift(-5) / px - 1.0).reindex(dc).values
    report("  declustered at 21td, h5", dv5)

    print("  era split (h1, full cell):")
    for e in era_split(idx[cell], fwd1[cell]):
        print(f"    {e['label']:<9} n={e['n']:<4} mean={e['mean_pct']:+.3f}%  hit={e['hit']:.1f}%  t={e['t']:+.2f}")
    print("  concentration:", cluster_note(idx[cell], fwd1[cell]))
    print(f"  declustered episodes: {len(dc)}, most recent: {[str(d.date()) for d in dc[-6:]]}")

# the thing that makes tonight's version distinctive: gold ripping at the same time
print("\n=== the joint state: DXY 21d floor WHILE gold 21d is in the top decile ===")
dxy = cp["DX-Y.NYB"].dropna()
gold = cp["GC=F"].reindex(dxy.index)
d_rank = dxy.pct_change(21).rolling(252).apply(lambda w: (w[-1] > w[:-1]).mean() * 100, raw=True)
g_rank = gold.pct_change(21).rolling(252).apply(lambda w: (w[-1] > w[:-1]).mean() * 100, raw=True)
print(f"  live: DXY 21d rank {d_rank.iloc[-1]:.1f}, gold 21d rank {g_rank.iloc[-1]:.1f}")
joint = (d_rank < 5) & (g_rank > 90)
g_f5 = (gold.shift(-5) / gold - 1.0)
g_f21 = (gold.shift(-21) / gold - 1.0)
d_f21 = (dxy.shift(-21) / dxy - 1.0)
m = joint & g_f21.notna()
print(f"  joint days: {int(m.sum())}")
report("  gold h5 from the joint state", g_f5[m].values)
report("  gold h21 from the joint state", g_f21[m].values)
report("  dollar h21 from the joint state", d_f21[m].values)
report("  gold h21, all days (control)", g_f21.dropna().values)
dcj = declusters(dxy.index[m], min_gap_td=21, all_dates=dxy.index)
report("  gold h21, joint state declustered 21td", g_f21.reindex(dcj).values)
print(f"  declustered joint episodes: {len(dcj)} -> {[str(d.date()) for d in dcj]}")
