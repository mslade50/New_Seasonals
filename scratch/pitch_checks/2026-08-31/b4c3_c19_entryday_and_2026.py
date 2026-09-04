"""C19 addendum - the two objections the teardown did not close.

E1. THE ENTRY DAY.  The order is MOC TODAY, one session after the trigger.
    Whatever SLV does between Friday's close and today's close is known at
    15:55 and is not in any of the statistics so far.  Does the h=1 edge
    depend on the entry-day move?  If it only exists when silver keeps
    falling on the entry day, then a bounce today is a different setup.

E2. THE 2026 LOAD.  cluster_note reported best year 2026 = +19.7pp of a
    +63.19pp h=1 total.  Eleven of the 119 episodes are 2026.  Price the
    cell ex-2026 and per-year.

E3. lag contrast and day-of-week/month-position sanity.
"""
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

pd.set_option("display.width", 250)
BAR = pd.Timestamp("2026-08-28")
GAP = 5
px = close_panel(["GLD", "SLV", "GDX"]).dropna().loc[:BAR]
r1 = {t: px[t] / px[t].shift(1) - 1.0 for t in px.columns}
trig = (r1["GLD"] <= -0.02) & (r1["SLV"] <= -0.02) & (r1["GDX"] <= -0.02)
pos = pd.Series(range(len(px.index)), index=px.index)

r = vehicle_ret(px, [("SLV", -1.0)], 1, 1)
base = r.dropna()
P0 = float((base > 0).mean())
days = px.index[trig.values & r.notna().values]
epi = declusters(days, GAP, px.index)
vals = r.loc[epi].values
print(f"episodes {len(epi)}   h=1 mean {100*vals.mean():+.3f}%   "
      f"SLV down-rate {100*P0:.2f}%")

# ---------------------------------------------------------------- E1 entry day
print("\n" + "=" * 100)
print("E1. THE ENTRY-DAY MOVE (knowable at 15:55 today; not in any prior stat)")
print("=" * 100)
entry_move = np.array([px["SLV"].iloc[pos[d] + 1] / px["SLV"].iloc[pos[d]] - 1.0
                       for d in epi])
print(f"  entry-day SLV move across episodes: mean {100*entry_move.mean():+.3f}%  "
      f"median {100*np.median(entry_move):+.3f}%  "
      f"down on the entry day {100*(entry_move < 0).mean():.1f}% of the time")
for lbl, m in [("entry day DOWN (continuation)", entry_move < 0),
               ("entry day DOWN > 1%", entry_move < -0.01),
               ("entry day FLAT/UP (bounce)", entry_move >= 0),
               ("entry day UP > 1%", entry_move > 0.01),
               ("entry day UP > 2%", entry_move > 0.02)]:
    if m.sum() < 3:
        continue
    v = vals[m]
    w = int((v > 0).sum())
    print(f"    {lbl:30s} N={m.sum():3d}  mean {100*v.mean():+7.3f}%  "
          f"median {100*np.median(v):+7.3f}%  record {w}-{len(v)-w}  "
          f"sign p vs {100*P0:.1f}% = {sign_test(w, len(v), P0):.4f}")
a = vals[entry_move < 0]
b = vals[entry_move >= 0]
se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
print(f"  DOWN-vs-UP diff {100*(a.mean()-b.mean()):+.3f}pp   welch t "
      f"{(a.mean()-b.mean())/se:+.2f}")
print("  (correlation entry-day move vs h=1 short return: "
      f"{np.corrcoef(entry_move, vals)[0,1]:+.3f})")

# ------------------------------------------------------------------ E2 2026
print("\n" + "=" * 100)
print("E2. THE 2026 LOAD")
print("=" * 100)
yr = pd.DatetimeIndex(epi).year
by = pd.Series(100 * vals, index=yr).groupby(level=0).agg(["count", "sum", "mean"])
print(by.round(2).to_string())
tot = vals.sum()
for cut in (2026, 2025, 2020, 2008):
    m = yr != cut
    v = vals[m]
    w = int((v > 0).sum())
    print(f"  ex-{cut}: N={len(v)}  mean {100*v.mean():+.3f}%  record "
          f"{w}-{len(v)-w}  sign p {sign_test(w, len(v), P0):.4f}  "
          f"({100*vals[yr==cut].sum():+.2f}pp of {100*tot:+.2f}pp total = "
          f"{100*vals[yr==cut].sum()/tot:.0f}%)")
m = ~np.isin(yr, [2026, 2020, 2008])
v = vals[m]
w = int((v > 0).sum())
print(f"  ex-2008/2020/2026 (the three crisis-ish years): N={len(v)}  mean "
      f"{100*v.mean():+.3f}%  record {w}-{len(v)-w}  sign p "
      f"{sign_test(w, len(v), P0):.4f}")
print("\n  per-year win: "
      f"{int((by['sum'] > 0).sum())} of {len(by)} years positive, sign p "
      f"{sign_test(int((by['sum'] > 0).sum()), len(by)):.4f}")
print(f"  LAST 5 YEARS ONLY (2022+): ", end="")
m5 = yr >= 2022
v = vals[m5]
w = int((v > 0).sum())
print(f"N={len(v)} mean {100*v.mean():+.3f}% record {w}-{len(v)-w} "
      f"sign p {sign_test(w, len(v), P0):.4f}")

# ------------------------------------------------------------------- E3 sanity
print("\n" + "=" * 100)
print("E3. LAG CONTRAST + calendar sanity")
print("=" * 100)
r0 = vehicle_ret(px, [("SLV", -1.0)], 1, 0)
print(f"  lag=0 (untradeable, signal-close entry): "
      f"{100*r0.loc[epi].mean():+.3f}%   lag=1 (the order): "
      f"{100*vals.mean():+.3f}%")
r2 = vehicle_ret(px, [("SLV", -1.0)], 1, 2)
print(f"  lag=2 (one day late): {100*r2.reindex(epi).dropna().mean():+.3f}%  "
      f"-> the edge is {'gone' if abs(r2.reindex(epi).dropna().mean()) < 0.001 else 'partly present'} a session later")
dow = pd.DatetimeIndex(epi).dayofweek
for d in range(5):
    m = dow == d
    if m.sum() < 5:
        continue
    v = vals[m]
    w = int((v > 0).sum())
    print(f"    trigger on {['Mon','Tue','Wed','Thu','Fri'][d]}: N={m.sum():3d} "
          f"mean {100*v.mean():+7.3f}%  record {w}-{len(v)-w}")
print(f"  today's trigger was a FRIDAY (2026-08-28) -> entry Monday 08-31, "
      f"which is also the LAST TRADING DAY of August.")
me = np.array([pd.Timestamp(d).month != px.index[pos[d] + 1].month
               or pos[d] + 2 >= len(px.index)
               or px.index[pos[d] + 2].month != px.index[pos[d] + 1].month
               for d in epi])
if me.sum() >= 3:
    v = vals[me]
    w = int((v > 0).sum())
    print(f"  entry day is a MONTH-END session: N={int(me.sum())} mean "
          f"{100*v.mean():+.3f}% record {w}-{len(v)-w}  (rest "
          f"{100*vals[~me].mean():+.3f}%)")
