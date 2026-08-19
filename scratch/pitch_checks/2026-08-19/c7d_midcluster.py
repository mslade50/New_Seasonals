"""C7d: the mid-cluster trap (today is deep inside the trigger run), the
decluster-order disagreement, and gold's REAL live state (below its 200d,
63d rank 30.6, -19.6% off the 252d high) rather than the map's 21d read.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import pandas as pd, numpy as np

px = close_panel(["DX-Y.NYB", "^TNX", "GLD"]).dropna(subset=["DX-Y.NYB", "^TNX", "GLD"])
dx, tnx, gld = px["DX-Y.NYB"], px["^TNX"], px["GLD"]
rk_tnx, rk_dx, rk_gld = pct_rank(tnx, 21), pct_rank(dx, 21), pct_rank(gld, 21)
r21_tnx = tnx.pct_change(21)
base = ((r21_tnx > 0) & (rk_tnx >= 65) & (rk_dx <= 20)).fillna(False)

# ---- how deep into the run is today? ----
run = 0
for v in base.values[::-1]:
    if v:
        run += 1
    else:
        break
print("=== (a) MID-CLUSTER: the joint trigger has been ON for %d consecutive "
      "sessions (today included) ===" % run)

# day-in-run for every trigger day
dayin = np.zeros(len(base), int)
c = 0
for i, v in enumerate(base.values):
    c = c + 1 if v else 0
    dayin[i] = c
dayin = pd.Series(dayin, index=px.index)

for h in (1, 3, 5):
    ret = vehicle_ret(px, [("GLD", 1.0)], h)
    valid = ret.notna().values
    rows = []
    for lo, hi, lbl in [(1, 1, "day 1 of the run (the episode entry)"),
                        (2, 5, "days 2-5"), (6, 15, "days 6-15 (TODAY = %d)" % run),
                        (16, 999, "day 16+")]:
        m = base.values & valid & (dayin.values >= lo) & (dayin.values <= hi)
        v = ret.values[m]
        s = summarize(v, f"h={h} {lbl}")
        if s["n"]:
            s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        rows.append(s)
    show(rows, f"gold forward by position in the trigger run, h={h} "
               f"(day-level, overlapping)")

# ---- (b) decluster-order disagreement on the live sub-cell ----
print("\n=== (b) the gold-strength sub-cell answers differently depending on "
      "WHEN you decluster ===")
for h in (3, 5):
    ret = vehicle_ret(px, [("GLD", 1.0)], h)
    valid = px.index[ret.notna().values]
    # order A: decluster the PARENT, then split
    epiA = declusters(px.index[base.values & ret.notna().values], 21, valid)
    hiA = (rk_gld.reindex(epiA) >= 75).values
    vA = ret.loc[epiA].values[hiA]
    # order B: gate first, then decluster
    mB = (base & (rk_gld >= 75)).fillna(False)
    epiB = declusters(px.index[mB.values & ret.notna().values], 21, valid)
    vB = ret.loc[epiB].values
    print("  h=%d  A: decluster-then-split  N=%d mean %+0.3f%% hit %.1f%%   |   "
          "B: gate-then-decluster N=%d mean %+0.3f%% hit %.1f%%   (gap %+0.3fpp)"
          % (h, len(vA), 100*vA.mean(), 100*(vA > 0).mean(),
             len(vB), 100*vB.mean(), 100*(vB > 0).mean(),
             100*(vB.mean()-vA.mean())))

# ---- (c) gold's REAL live state ----
print("\n=== (c) gold's real live state, and the cell restricted to it ===")
sma200 = gld.rolling(200).mean()
hi252 = gld.rolling(252).max()
dd = gld/hi252 - 1.0
rk63g = pct_rank(gld, 63)
print("  TODAY: GLD %.2f, 200d %.2f (BELOW), 63d rank %.1f, %.1f%% off the "
      "252d high, 21d rank %.1f"
      % (gld.iloc[-1], sma200.iloc[-1], rk63g.iloc[-1], 100*dd.iloc[-1],
         rk_gld.iloc[-1]))
live = (base & (gld < sma200) & (rk_gld >= 75)).fillna(False)
for h in (1, 3, 5, 10):
    ret = vehicle_ret(px, [("GLD", 1.0)], h)
    valid = px.index[ret.notna().values]
    epi = declusters(px.index[live.values & ret.notna().values], 21, valid)
    v = ret.loc[epi].values
    s = summarize(v, "h=%d joint & GLD<200d & rank21>=75 (TODAY'S STATE)" % h)
    if s["n"]:
        s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        s["edge_pp"] = round(s["mean_pct"] - 100*ret.loc[valid].mean(), 3)
    show([s])
    if h == 3:
        print("   episodes:", ", ".join(str(d.date()) for d in epi))

# ---- (d) 21d-rank-hot but 63d-rank-cold, i.e. a bounce inside a drawdown ----
print("\n=== (d) the map's 'gold is hot' is a 21d read; the 63d rank is 30.6 ===")
for h in (3, 5):
    ret = vehicle_ret(px, [("GLD", 1.0)], h)
    valid = px.index[ret.notna().values]
    rows = []
    for lbl, m in [("63d rank >= 60", base & (rk63g >= 60)),
                   ("63d rank 30-60 (TODAY 30.6)", base & (rk63g >= 30) & (rk63g < 60)),
                   ("63d rank < 30", base & (rk63g < 30))]:
        epi = declusters(px.index[m.fillna(False).values & ret.notna().values],
                         21, valid)
        v = ret.loc[epi].values
        s = summarize(v, f"h={h} {lbl}")
        if s["n"]:
            s["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
        rows.append(s)
    show(rows)
