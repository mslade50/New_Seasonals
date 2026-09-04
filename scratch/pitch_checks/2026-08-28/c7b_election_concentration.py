"""C7b -- make the C7 headline kill a number rather than an eyeball.

The 16 trigger days decluster to 9 episodes at h=3, and the dates read as two
US post-presidential-election risk-on rotations plus stragglers.  If most of the
cell's mass sits inside a few weeks of a presidential election, "the defensive
complex washed out at an index high" is a post-election rotation wearing a
sector-breadth label, and 2026 is a MIDTERM year in which that reference class
is not available.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pitch_lab import (load_prices, load_events, pct_rank, fwd_lag,  # noqa: E402
                       summarize, show, declusters)

DEF = ["XLP", "XLU", "XLRE"]
ASOF = pd.Timestamp("2026-08-27")
px = load_prices(DEF + ["SPY"])
S = {t: px[t]["Close"].dropna().loc[:ASOF] for t in px}
spy = S["SPY"]
spy_dist = spy / spy.rolling(252).max() - 1.0

cal = S["XLRE"].index
R = pd.DataFrame({t: pct_rank(S[t], 21).reindex(cal) for t in DEF}).dropna()
sd = spy_dist.reindex(R.index)
MASK = (R <= 20).all(axis=1) & (sd >= -0.01)

bk = None
for t in DEF:
    s = S[t].reindex(cal).ffill()
    bk = (s / s.iloc[0]) / 3 if bk is None else bk + (s / s.iloc[0]) / 3
BK = bk.dropna()

ev = load_events(["election"])
elections = pd.DatetimeIndex(ev["date"])
# presidential = every 4th, year % 4 == 0
pres = pd.DatetimeIndex([d for d in elections if d.year % 4 == 0])
print(f"elections in file: {len(elections)}  presidential: "
      f"{[str(d.date()) for d in pres]}")

trig = BK.index[MASK.reindex(BK.index, fill_value=False).values]
print(f"\ntrigger days: {len(trig)}")


def days_to_nearest(d, anchors):
    return int(min(abs((d - a).days) for a in anchors))


rows = []
for d in trig:
    rows.append({"date": str(d.date()),
                 "days_from_pres_election": days_to_nearest(d, pres),
                 "days_from_any_election": days_to_nearest(d, elections)})
T = pd.DataFrame(rows)
print(T.to_string(index=False))

for win in (30, 45, 60, 90):
    n = int((T["days_from_pres_election"] <= win).sum())
    print(f"  within {win} calendar days of a PRESIDENTIAL election: "
          f"{n} of {len(T)} trigger days ({100*n/len(T):.0f}%)")

# what fraction of the sample is within that window, as a base rate?
alld = BK.index
base = np.mean([days_to_nearest(d, pres) <= 60 for d in alld])
hit = float((T["days_from_pres_election"] <= 60).mean())
print(f"\n  BASE RATE: {100*base:.1f}% of all sessions in the 2016-11..2026-08 "
      f"sample sit within 60 days of a presidential election.")
print(f"  TRIGGER RATE: {100*hit:.1f}%.  Enrichment = {hit/base:.1f}x")

print("\n=== does the cell survive OUTSIDE the election window? ===")
for h in (3, 5, 10):
    r = fwd_lag(BK, h, 1)
    m = MASK.reindex(BK.index, fill_value=False).values & r.notna().values
    dts = BK.index[m]
    near = pd.DatetimeIndex([d for d in dts if days_to_nearest(d, pres) <= 60])
    far = pd.DatetimeIndex([d for d in dts if days_to_nearest(d, pres) > 60])
    en, ef = declusters(near, h, BK.index), declusters(far, h, BK.index)
    show([summarize(r.loc[en].values, f"WITHIN 60d of pres election (epi {len(en)})"),
          summarize(r.loc[ef].values, f"OUTSIDE (epi {len(ef)})"),
          summarize(r.dropna().values, "all days")], f"h={h}")
    print(f"    outside-window episode dates: {[str(d.date()) for d in ef]}")

print("\n=== midterm availability of the reference class ===")
yrs = sorted(set(pd.DatetimeIndex(trig).year))
print(f"  trigger years: {yrs}")
print(f"  midterm trigger days (year%4==2): "
      f"{int(sum(1 for d in trig if d.year % 4 == 2))} of {len(trig)}")
print("  2026 is a midterm year: the post-presidential-election rotation that")
print("  supplies most of this cell's mass is structurally unavailable.")
