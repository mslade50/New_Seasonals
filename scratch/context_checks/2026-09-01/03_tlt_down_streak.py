"""TLT 5+ consecutive down closes. The engine's strongest cell tonight:
n=93, +0.303% h1, 62-31, sign p 0.0009, era-stable, BH pass.

Three things the base cell cannot answer:
  1. how long is the current streak, and does the edge depend on length
  2. does it survive when the streak ends AT a 52-week low, which is tonight
  3. is it duration or is it TLT (check IEF and LQD on the same construction),
     and do the top two episodes carry the mean

Plus: the E:seasonal_doy TLT midterm h5 arm (5 of 5 down) sits beside this and
points the OTHER way. Check whether the two overlap at all.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

TKRS = ["TLT", "IEF", "LQD", "SPY"]
px = close_panel(TKRS).dropna(subset=["TLT"])
print(f"panel {px.index[0].date()} .. {px.index[-1].date()}, {len(px)} sessions")


def down_streak(s: pd.Series) -> pd.Series:
    dn = (s.diff() < 0).astype(int)
    out, run = [], 0
    for v in dn.values:
        run = run + 1 if v else 0
        out.append(run)
    return pd.Series(out, index=s.index)


for t in ["TLT", "IEF", "LQD"]:
    st = down_streak(px[t])
    print(f"{t}: current down streak {int(st.iloc[-1])} sessions "
          f"(last 8 closes {[round(x,2) for x in px[t].iloc[-8:].tolist()]})")

st = down_streak(px["TLT"])
low252 = rolling_on_valid(px["TLT"], lambda x: x.rolling(252, min_periods=200).min())
near_low = px["TLT"] <= low252 * 1.01

fired = (st >= 5) & (st.shift(1) < 5)   # first session the streak reaches 5
print(f"\nTLT streak reaches 5: {int(fired.sum())} first-hits since {px.index[0].date()}")

# the engine's construction is every session with streak>=5; keep that for parity
mask_all = (st >= 5)
print(f"TLT sessions with an ACTIVE 5+ streak: {int(mask_all.sum())} (engine cell n=93)")

all_dates = px.index


def cell(mask, label, subj="TLT", hs=(1, 5, 10, 21)):
    dts = all_dates[mask.fillna(False).values]
    rows = []
    for h in hs:
        r = fwd_ret(px[subj], h)
        v = r.loc[r.index.intersection(dts)].dropna()
        s = summarize(v.values, f"h={h}")
        if s["n"]:
            wins = int((v.values > 0).sum())
            s["record"] = f"{wins}-{s['n']-wins}"
            s["sign_p"] = round(sign_test(wins, s["n"]), 4)
            base = r.dropna()
            s["edge_pct"] = round(s["mean_pct"] - 100 * base.mean(), 3)
        rows.append(s)
    show(rows, f"{subj}: {label}")
    return dts


print("\n=== 1. the base cell, and by streak length ===")
d_all = cell(mask_all, "any active 5+ down streak")
for lo, hi in ((5, 5), (6, 6), (7, 99)):
    cell((st >= lo) & (st <= hi), f"streak exactly {lo}" if lo == hi else f"streak {lo}+",
         hs=(1, 5))

print("\n=== 2. the streak AT a 52-week low (within 1%), which is tonight ===")
d_low = cell(mask_all & near_low, "5+ streak with TLT within 1% of its 252d low")
print(f"  ({int((mask_all & near_low).sum())} sessions, "
      f"{int((mask_all & ~near_low).sum())} sessions NOT near the low)")
cell(mask_all & ~near_low, "5+ streak, NOT near the 252d low", hs=(1, 5))

print("\n=== 3. same construction on IEF and LQD ===")
for t in ["IEF", "LQD"]:
    s_t = down_streak(px[t])
    cell(s_t >= 5, f"{t} 5+ down streak", subj=t, hs=(1, 5))

print("\n=== era split and concentration, TLT h=1 on the base cell ===")
r1 = fwd_ret(px["TLT"], 1)
v1 = r1.loc[r1.index.intersection(d_all)].dropna()
show(era_split(v1.index, v1.values), "TLT h=1, 5+ streak")
print("  " + cluster_note(v1.index, v1.values, k=2))

print("\n=== episode view: decluster to one per streak, 5td gap ===")
epi = declusters(pd.DatetimeIndex(d_all), 5, all_dates)
for h in (1, 5, 10):
    r = fwd_ret(px["TLT"], h)
    v = r.loc[r.index.intersection(epi)].dropna()
    wins = int((v.values > 0).sum())
    s = summarize(v.values, f"h={h}")
    print(f"  h={h}: n {s['n']}, {wins}-{s['n']-wins}, mean {s['mean_pct']:+.3f}%, "
          f"median {s['median_pct']:+.3f}%, t {s['t']:.2f}, "
          f"sign p {sign_test(wins, s['n']):.4f}, worst {s['worst_pct']:+.2f}%")

print("\n=== 4. does the seasonal_doy midterm arm overlap this cell? ===")
# early-September anchors in midterm years, TLT h5 (the 5-of-5-down arm)
mt = [y for y in range(2002, 2027) if y % 4 == 2]
r5 = fwd_ret(px["TLT"], 5)
print("  year   anchor      TLT h5      active 5+ down streak at the anchor?")
for y in mt:
    cand = all_dates[(all_dates.year == y) & (all_dates.month == 9) &
                     (all_dates.day <= 4)]
    if len(cand) == 0:
        continue
    a = cand[0]
    val = r5.get(a, np.nan)
    print(f"  {y}   {a.date()}   {100*val:+7.2f}%    {int(st.get(a, 0))}")
