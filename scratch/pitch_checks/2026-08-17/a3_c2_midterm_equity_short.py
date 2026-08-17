"""C2 round 1 - Short SPY / QQQ into late August in a midterm year.

Registry collision to clear first (2026-08-07): "midterm mid-August seasonality
- N=6, carried entirely by 2002 (+8.68%); drop-two-best is negative. The midterm
restriction ANTI-WORKS at 21 td." Also 2026-08-07 "the run into August opex" and
2026-08-14 "the pre-opex WEEK entered on the Friday before", both dead.

Decisive questions, in order:
  1. does the ALL-YEARS cell (N=26) stand on its own against the tdom+month
     control, i.e. is there anything to condition?
  2. is the midterm subset DIFFERENT from the non-midterm subset, or six draws
     from one distribution? (sign test on the record, per-year table,
     drop-worst-year, two-sample test between subsets)
  3. today's entry is at a 52w high with z10 +1.44. Does conditioning on that
     help or hurt? A seasonal short at a high is a different trade.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

BAR = pd.Timestamp("2026-08-14")
px = close_panel(["SPY", "QQQ"])
IDX = px.index
tdom = pd.Series(IDX.to_series().groupby([IDX.year, IDX.month]).cumcount() + 1,
                 index=IDX)

mask_aug = pd.Series((IDX.month == 8) & (IDX.day >= 15) & (IDX.day <= 19), index=IDX)
sig = IDX[mask_aug.values]
epi = declusters(sig, 5, IDX)
mid = pd.DatetimeIndex([d for d in epi if d.year % 4 == 2])
non = epi.difference(mid)
print(f"episodes N={len(epi)} ({epi[0].date()}..{epi[-1].date()}), "
      f"midterm N={len(mid)} {[d.year for d in mid]}")

for tkr in ("SPY", "QQQ"):
    battery(px, mask_aug, [(tkr, -1.0)], 10,
            f"C2 SHORT {tkr}, Aug 15-19 entry analogue", cost_bps=3.0,
            min_gap=5, event_kinds=("jackson_hole",))

# ---------------------------------------------- 1. controls (tdom + month)
print("\n" + "=" * 92)
print("1. CONTROL LADDER for the SHORT (all values are SHORT returns)")
print("=" * 92)
for tkr in ("SPY", "QQQ"):
    s = px[tkr].dropna()
    for h in (5, 10):
        f = -fwd_lag(s, h, 1)
        valid = f.notna()
        cell = f.reindex(epi).dropna()
        rows = [summarize(cell.values, f"CELL short {tkr} h={h} (N={len(cell)})"),
                summarize(f[valid].values, "CTRL all days"),
                summarize(f[((tdom >= 10) & (tdom <= 12) & valid).values].values,
                          "CTRL tdom 10-12"),
                summarize(f[(IDX.month == 8) & valid.values].values, "CTRL all August"),
                summarize(f[(IDX.month == 8) & (tdom.values >= 9)
                            & (tdom.values <= 13) & valid.values].values,
                          "CTRL August tdom 9-13")]
        show(rows, f"{tkr} h={h}")

# ------------------------------------------- 2. is midterm actually different?
print("\n" + "=" * 92)
print("2. MIDTERM vs NON-MIDTERM: one distribution or two?")
print("=" * 92)
for tkr in ("SPY", "QQQ"):
    s = px[tkr].dropna()
    for h in (5, 10):
        f = -fwd_lag(s, h, 1)
        m, n = f.reindex(mid).dropna(), f.reindex(non).dropna()
        show([summarize(m.values, f"{tkr} h={h} MIDTERM"),
              summarize(n.values, f"{tkr} h={h} non-midterm")], f"{tkr} h={h}")
        se = np.sqrt(m.var(ddof=1) / len(m) + n.var(ddof=1) / len(n))
        w = int((m > 0).sum())
        print(f"  midterm-minus-nonmidterm {100*(m.mean()-n.mean()):+.3f}pp  "
              f"welch t {(m.mean()-n.mean())/se:+.2f}   midterm record {w}-{len(m)-w}"
              f"  sign p {sign_test(w, len(m)):.4f}")
        print("  midterm years:", {d.year: round(100 * v, 2) for d, v in m.items()})
        srt = np.sort(m.values)
        print(f"  drop-best-year mean {100*srt[:-1].mean():+.3f}%  "
              f"drop-two-best {100*srt[:-2].mean():+.3f}%  "
              f"median {100*np.median(m.values):+.3f}%")

# ---------------------------------------- 3. midterm placebo across cycle years
print("\n" + "=" * 92)
print("3. CYCLE-YEAR PLACEBO: all four residues, short SPY h=10")
print("=" * 92)
f10 = -fwd_lag(px["SPY"].dropna(), 10, 1)
rows = []
for r in range(4):
    sel = pd.DatetimeIndex([d for d in epi if d.year % 4 == r])
    v = f10.reindex(sel).dropna()
    rows.append({"year%4": r, "n": len(v), "mean_pct": round(100 * v.mean(), 3),
                 "hit": round(100 * (v > 0).mean(), 1),
                 "median_pct": round(100 * np.median(v.values), 3)})
show(rows, "short SPY h=10 by cycle residue (midterm = 2)")

# --------------------------- 4. does 'entering at a 52w high' help or hurt?
print("\n" + "=" * 92)
print("4. CONDITIONER: entering the seasonal short AT a 52w high (today: -0.20%)")
print("=" * 92)
for tkr in ("SPY", "QQQ"):
    s = px[tkr].dropna()
    offhi = (s / s.rolling(252).max() - 1.0) * 100
    print(f"  {tkr} today {offhi.loc[BAR]:.2f}% off its 52w high")
    for h in (5, 10):
        f = -fwd_lag(s, h, 1)
        near = pd.DatetimeIndex([d for d in epi if offhi.get(d, -99) >= -1.0])
        far = pd.DatetimeIndex([d for d in epi if offhi.get(d, -99) < -1.0])
        show([summarize(f.reindex(near).dropna().values,
                        f"{tkr} h={h} cell x within 1% of 52w high"),
              summarize(f.reindex(far).dropna().values,
                        f"{tkr} h={h} cell x more than 1% below"),
              summarize(f[(offhi >= -1.0).reindex(f.index, fill_value=False).values
                          & f.notna().values].values,
                        f"{tkr} h={h} ALL DAYS within 1% of high (no seasonal)")],
             f"{tkr} h={h}")
        near_mid = pd.DatetimeIndex([d for d in near if d.year % 4 == 2])
        v = f.reindex(near_mid).dropna()
        print(f"  live intersection (midterm x within 1% of high): N={len(v)}"
              + (f"  mean {100*v.mean():+.3f}%  years "
                 f"{[d.year for d in v.index]}" if len(v) else ""))

# ------------------------------------------------- 5. era stability all-years
print("\n" + "=" * 92)
print("5. ERA: does the all-years short cell survive 2010+/2018+?")
print("=" * 92)
for tkr in ("SPY", "QQQ"):
    s = px[tkr].dropna()
    f = -fwd_lag(s, 10, 1)
    v = f.reindex(epi).dropna()
    show([summarize(v[v.index < "2010-01-01"].values, f"{tkr} pre-2010"),
          summarize(v[(v.index >= "2010-01-01") & (v.index < "2018-01-01")].values,
                    f"{tkr} 2010-2017"),
          summarize(v[v.index >= "2018-01-01"].values, f"{tkr} 2018+")],
         f"{tkr} short h=10 by era")
    print("  ", cluster_note(v.index, v.values))
