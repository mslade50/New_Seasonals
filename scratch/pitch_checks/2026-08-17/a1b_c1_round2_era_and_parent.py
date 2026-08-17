"""C1 round 2 - the parent window, the era, and the rate regime.

Round 1 produced two things that decide the candidate:
  * the August tdom ladder is a PLATEAU (tdom 4-12 all pay +0.76 to +1.15pp
    excess at h=10), which the registry names as the signature of month position
    rather than an event;
  * the cell's own episode era split is +1.696% pre-2018 on 15-1 against
    -0.066% on 3-5 from 2018.

So the honest question is not about the cell at all, it is about its PARENT: is
TLT's mid-August drift still alive? If the parent died with the bond bull, the
cell is a fossil and nothing conditioned on it can trade.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["TLT"])
tlt = px["TLT"].dropna()
IDX = tlt.index
tdom = pd.Series(IDX.to_series().groupby([IDX.year, IDX.month]).cumcount() + 1,
                 index=IDX)
f10 = fwd_lag(tlt, 10, 1)
f5 = fwd_lag(tlt, 5, 1)

cell = declusters(IDX[((IDX.month == 8) & (IDX.day >= 15) & (IDX.day <= 19))], 5, IDX)
parent = pd.DatetimeIndex([d for d in IDX if d.month == 8 and 6 <= d.day <= 16])

# --------------------------- 1. cell against the plateau, EX the cell's own days
print("=" * 92)
print("1. the cell against its own neighbourhood, cell days REMOVED from control")
print("=" * 92)
nb = pd.DatetimeIndex([d for d in IDX if d.month == 8 and 4 <= tdom.loc[d] <= 14])
nb_ex = nb.difference(IDX[((IDX.month == 8) & (IDX.day >= 15) & (IDX.day <= 19))])
for h, f in ((5, f5), (10, f10)):
    c = f.reindex(cell).dropna()
    n = f.reindex(nb_ex).dropna()
    show([summarize(c.values, f"CELL Aug15-19 episodes h={h}"),
          summarize(n.values, f"CTRL August tdom 4-14, cell days removed h={h}"),
          summarize(f.reindex(parent).dropna().values,
                    f"CTRL August day 6-16 (registry parent) h={h}")], f"h={h}")
    se = np.sqrt(c.var(ddof=1) / len(c) + n.var(ddof=1) / len(n))
    print(f"  h={h} cell-minus-neighbourhood {100*(c.mean()-n.mean()):+.3f}pp  "
          f"welch t {(c.mean()-n.mean())/se:+.2f}")

# ------------------------------------------- 2. era of the PARENT, not the cell
print("\n" + "=" * 92)
print("2. IS THE PARENT STILL ALIVE? TLT mid-August drift by era")
print("=" * 92)
for h, f in ((5, f5), (10, f10)):
    v = f.reindex(parent).dropna()
    rows = []
    for lo, hi, lbl in ((2002, 2009, "2002-2009"), (2010, 2017, "2010-2017"),
                        (2018, 2020, "2018-2020"), (2021, 2025, "2021-2025")):
        s = v[(v.index.year >= lo) & (v.index.year <= hi)]
        r = summarize(s.values, f"August 6-16 {lbl}")
        r["yrs"] = len(set(s.index.year))
        rows.append(r)
    rows.append(summarize(v[v.index.year >= 2018].values, "August 6-16 2018+"))
    show(rows, f"parent window h={h}")
    # per-year means, one number per year = the honest record
    yr = pd.Series(v.values).groupby(v.index.year.values).mean() * 100
    print(f"  h={h} per-YEAR mean (one obs per year): "
          + ", ".join(f"{y}:{m:+.2f}" for y, m in yr.items()))
    w = int((yr > 0).sum())
    print(f"  per-year record {w}-{len(yr)-w}, sign p {sign_test(w, len(yr)):.4f} | "
          f"2018+ record {int((yr[yr.index>=2018]>0).sum())}-"
          f"{int((yr[yr.index>=2018]<=0).sum())}")

# --------------------------------------------- 3. rate regime, the mechanism
print("\n" + "=" * 92)
print("3. RATE REGIME - was the August duration drift a bond-bull artifact?")
print("=" * 92)
tnx = close_panel(["^TNX"])["^TNX"].reindex(IDX).ffill()
# regime = sign of the trailing 252d change in the 10y yield
dy = tnx.diff(252)
for h, f in ((5, f5), (10, f10)):
    v = f.reindex(parent).dropna()
    dd = dy.reindex(v.index)
    show([summarize(v[(dd < 0).values].values, f"August 6-16, yields FALLING 1y h={h}"),
          summarize(v[(dd >= 0).values].values, f"August 6-16, yields RISING 1y h={h}")],
         f"h={h} by 1y yield direction")
    # all-days version, so we can tell regime from calendar
    a = f.dropna()
    da = dy.reindex(a.index)
    show([summarize(a[(da < 0).values].values, f"ALL DAYS, yields FALLING h={h}"),
          summarize(a[(da >= 0).values].values, f"ALL DAYS, yields RISING h={h}")],
         f"h={h} all-days by regime (is August adding anything?)")

print("\ntoday's regime: 10y yield 1y change = "
      f"{dy.iloc[-1]:+.2f}pp ({'RISING' if dy.iloc[-1] >= 0 else 'FALLING'})")

# ------------------------------------------- 4. definition neighbours (window)
print("\n" + "=" * 92)
print("4. DEFINITION NEIGHBOURS - move the calendar window")
print("=" * 92)
rows = []
for lo, hi in ((13, 17), (14, 18), (15, 19), (16, 20), (17, 21), (18, 22)):
    s = declusters(pd.DatetimeIndex([d for d in IDX if d.month == 8
                                     and lo <= d.day <= hi]), 5, IDX)
    v = f10.reindex(s).dropna()
    v18 = v[v.index.year >= 2018]
    rows.append({"aug window": f"{lo}-{hi}", "n": len(v),
                 "mean_pct": round(100 * v.mean(), 3),
                 "hit": round(100 * (v > 0).mean(), 1),
                 "n_2018+": len(v18),
                 "mean_2018+": round(100 * v18.mean(), 3) if len(v18) else np.nan})
show(rows, "calendar-window neighbours, TLT h=10")

# same window in the neighbouring months (is August the month?)
rows = []
for mo in (6, 7, 8, 9, 10, 11):
    s = declusters(pd.DatetimeIndex([d for d in IDX if d.month == mo
                                     and 15 <= d.day <= 19]), 5, IDX)
    v = f10.reindex(s).dropna()
    v18 = v[v.index.year >= 2018]
    rows.append({"month": mo, "n": len(v), "mean_pct": round(100 * v.mean(), 3),
                 "hit": round(100 * (v > 0).mean(), 1),
                 "mean_2018+": round(100 * v18.mean(), 3) if len(v18) else np.nan})
show(rows, "the SAME day-window in other months, TLT h=10")
