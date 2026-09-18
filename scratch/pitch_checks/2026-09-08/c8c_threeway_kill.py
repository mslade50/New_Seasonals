"""Last probe: the ONE cell in today's three candidates with a clean record.

C8's fully-live three-way (DBC within 0.25% of its 252d high AND TLT within 2%
of its own 252d LOW AND a print inside the hold) is 5-for-5 short TLT at h=8,
mean +1.670%, worst +0.357%, sign p 0.03125. Small N is not a legal kill, so
this has to die on a substantive ground or it survives.

Tests:
  1. WHEN are the 5 episodes? An all-2021/22 set is the regime kill.
  2. Placebo ladder ON THE THREE-WAY at h=8. If nonsense offsets also produce
     clean records the calendar leg is decoration.
  3. Is the TLT@low leg a filter that filters, or an anchor swap? Parent =
     cmdty-high + print WITHOUT the TLT leg (54 epi), and the TLT@low leg
     WITHOUT the print (its own parent).
  4. The leg that was found by looking: cross it with the killing conditioner.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["DBC", "TLT", "IEF", "SPY"])
raw = load_prices(["DBC", "TLT"])
IDX = px.index


def dist_to_high(t, n=252):
    s = raw[t]["Close"]
    return (1.0 - s / rolling_on_valid(s, lambda x: x.rolling(n).max())).reindex(IDX)


def dist_above_low(t, n=252):
    s = raw[t]["Close"]
    return (s / rolling_on_valid(s, lambda x: x.rolling(n).min()) - 1.0).reindex(IDX)


def flag(kinds, h, lag=1, k=0):
    ev = load_events(list(kinds))["date"]
    pos, _ = anchor_positions(IDX, ev, offset=k)
    e = np.asarray(pd.DatetimeIndex([IDX[p] for p in pos]).values, dtype="datetime64[ns]")
    out = np.zeros(len(IDX), dtype=bool)
    for i in range(len(IDX)):
        if i + lag + h >= len(IDX):
            continue
        out[i] = bool(((e > np.datetime64(IDX[i + lag])) &
                       (e <= np.datetime64(IDX[i + lag + h]))).any())
    return pd.Series(out, index=IDX)


LAG, H = 1, 8
state = dist_to_high("DBC") <= 0.0025
mom = dist_above_low("TLT") <= 0.02
ret = vehicle_ret(px, [("TLT", -1.0)], H, LAG)
valid = ret.notna()
pr = flag(("ppi", "cpi"), H, LAG)

three = state & mom & pr
dd = IDX[three.values & valid.values]
e = declusters(dd, H, IDX)
print("=" * 78)
print("1. WHEN are the three-way episodes?  (short TLT, h=8)")
print("=" * 78)
print(f"day-level trigger days: {len(dd)}  -> episodes: {len(e)}")
for x in e:
    print(f"   {x.date()}   TLT h=8 short = {100*ret.loc[x]:+.3f}%")
print(f"   all trigger DAYS: {', '.join(str(x.date()) for x in dd)}")
yrs = sorted(set(pd.DatetimeIndex(e).year))
print(f"   distinct years: {yrs}")
print(f"   distinct calendar months: "
      f"{sorted(set((x.year, x.month) for x in e))}")

print("\n" + "=" * 78)
print("2. PLACEBO LADDER ON THE THREE-WAY (h=8, k=-5..+5)")
print("=" * 78)
rows, means = [], []
for k in range(-5, 6):
    m = state & mom & flag(("ppi", "cpi"), H, LAG, k)
    d2 = IDX[m.values & valid.values]
    e2 = declusters(d2, H, IDX)
    v = ret.loc[e2].values
    r = summarize(v, f"k={k:+d}" + ("  <-- TRUE" if k == 0 else ""))
    if r["n"]:
        r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(r); means.append(r.get("mean_pct", np.nan))
show(rows, "three-way, short TLT h=8")
print(f"  true anchor ranks {int(np.sum(np.asarray(means) >= means[5]))} of 11 by mean")
clean = [i - 5 for i, r in enumerate(rows)
         if r.get("n", 0) >= 4 and r.get("hit", 0) == 100.0]
print(f"  offsets with a PERFECT record at n>=4: k = {clean}")

print("\n" + "=" * 78)
print("3. GATE ATTRIBUTION on the TLT@low leg  (does it filter, or re-anchor?)")
print("=" * 78)
rows = []
for m, nm in ((state & pr, "cmdty-high + print (parent, no TLT leg)"),
              (state & pr & mom, "  + TLT@low  (THREE-WAY)"),
              (state & pr & ~mom, "  + TLT NOT at low"),
              (state & mom, "cmdty-high + TLT@low, NO print"),
              (mom & pr, "TLT@low + print, no cmdty leg"),
              (mom, "TLT@low alone"),
              (pd.Series(True, index=IDX), "all days")):
    d2 = IDX[m.values & valid.values]
    if len(d2) == 0:
        rows.append({"label": nm, "n": 0}); continue
    e2 = declusters(d2, H, IDX)
    v = ret.loc[e2].values
    r = summarize(v, nm)
    r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    r["n_days"] = len(d2)
    rows.append(r)
show(rows, "short TLT h=8, episode level")

print("\n" + "=" * 78)
print("4. CROSS THE RESCUING LEG WITH THE KILLING ONE")
print("=" * 78)
d_par = IDX[(state & pr).values & valid.values]
e_par = declusters(d_par, H, IDX)
v_par = ret.loc[e_par].values
infl = pd.DatetimeIndex(e_par).year.isin([2007, 2008, 2021, 2022])
print(f"parent (cmdty-high+print) h=8: n={len(v_par)} mean {100*v_par.mean():+.3f}%")
print(f"  ex 2007/08/21/22: n={int((~infl).sum())} mean "
      f"{100*v_par[~infl].mean():+.3f}% hit {100*(v_par[~infl]>0).mean():.1f}%")
ei = pd.DatetimeIndex(e)
print(f"\nthree-way episodes inside 2007/08/21/22: "
      f"{int(ei.year.isin([2007,2008,2021,2022]).sum())} of {len(ei)}")
v3 = ret.loc[e].values
m3 = ei.year.isin([2007, 2008, 2021, 2022])
if (~m3).sum():
    print(f"  three-way EX those years: n={int((~m3).sum())}, mean "
          f"{100*v3[~m3].mean():+.3f}%, hit {100*(v3[~m3]>0).mean():.1f}%, "
          f"sign p {sign_test(int((v3[~m3]>0).sum()), int((~m3).sum())):.4f}")
else:
    print("  three-way EX those years: ZERO observations -- the cell exists "
          "only inside the inflation-shock regime.")

# how many DISTINCT non-overlapping regimes does the 5-episode set span?
print("\n  season/cycle location of the three-way episodes:")
for x in e:
    print(f"   {x.date()}  month={x.month:02d}  midterm={'Y' if x.year%4==2 else 'N'}")
print(f"\n  live analogue: 2026-09-04 signal, month=09, midterm=Y  -> "
      f"September episodes in the set: "
      f"{int((pd.DatetimeIndex(e).month==9).sum())}; "
      f"midterm episodes: {int(((pd.DatetimeIndex(e).year%4)==2).sum())}")

print("\n" + "=" * 78)
print("5. COST + tdom on the three-way")
print("=" * 78)
print(f"  episode mean {100*v3.mean():.3f}% = {10000*v3.mean():.1f} bps vs "
      f"3 bps TLT round trip = {v3.mean()*10000/3:.1f}x")
d3 = pd.DatetimeIndex(IDX)
tdom = pd.Series(range(len(d3)), index=d3).groupby([d3.year, d3.month]).rank(method="first").astype(int)
print(f"  tdom of the 5 episodes: {[int(tdom.loc[x]) for x in e]}")
