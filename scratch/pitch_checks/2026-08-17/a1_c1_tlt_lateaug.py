"""C1 round 1 - Long TLT into the late-August window (h ~8-10).

Registry debts this script must pay BEFORE anything else (both are rules, not
suggestions):
  * 2026-08-10 "a control that does not control": TLT's own drift moves with
    the TRADING DAY OF MONTH. Today is tdom 11. Excess must be quoted against a
    tdom-matched control.
  * 2026-08-14 "an event cell inside one month owes a MONTH-OF-YEAR control":
    TLT's h=10 lag-1 drift is +0.494% in August against -0.432% in October. The
    Jackson Hole cell already died to exactly this.

Then the attribution question the surface map named: "late August" and "opex
week" are the same days here. Only one can be the mechanism, so the opex anchor
is measured OUTSIDE August and the August window is measured OUTSIDE opex week.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

BAR = pd.Timestamp("2026-08-14")
H_MAIN = 10

px = close_panel(["TLT"])
tlt = px["TLT"].dropna()
IDX = tlt.index
px = pd.DataFrame({"TLT": tlt})

# trading day of month, 1-based
tdom = pd.Series(IDX.to_series().groupby([IDX.year, IDX.month]).cumcount() + 1,
                 index=IDX)
print(f"TLT span {IDX[0].date()} .. {IDX[-1].date()}  N={len(IDX)}")
print(f"today's analogue: 2026-08-17 would be tdom 11 (Aug 14 is tdom {int(tdom.loc[BAR])})")

# ---------------------------------------------------------------- the cell
mask_aug = pd.Series((IDX.month == 8) & (IDX.day >= 15) & (IDX.day <= 19),
                     index=IDX)
sig = IDX[mask_aug.values]
epi = declusters(sig, 5, IDX)
print(f"\nAug 15-19 trigger days N={len(sig)}, declustered(5td) N={len(epi)}")

battery(px, mask_aug, [("TLT", 1.0)], H_MAIN,
        "C1 TLT long, Aug 15-19 entry analogue", cost_bps=3.0,
        min_gap=5, event_kinds=("jackson_hole", "opex"))

# --------------------------------------------------- 1. THE MANDATORY CONTROLS
print("\n" + "=" * 92)
print("1. CONTROL LADDER - all-days vs tdom-matched vs month-matched vs both")
print("=" * 92)
for h in (5, 8, 10):
    f = fwd_lag(tlt, h, 1)
    valid = f.notna()
    cell = f.reindex(epi).dropna()
    rows = [summarize(cell.values, f"CELL Aug15-19 episodes h={h}")]
    # CTRL all days
    rows.append(summarize(f[valid].values, "CTRL all days"))
    # CTRL tdom-matched (tdom 11 exactly, and the 10-12 band)
    for lo, hi, lbl in ((11, 11, "CTRL tdom==11 (all months)"),
                        (10, 12, "CTRL tdom 10-12 (all months)")):
        m = (tdom >= lo) & (tdom <= hi) & valid
        rows.append(summarize(f[m.values].values, lbl))
    # CTRL month-matched
    m = (IDX.month == 8) & valid.values
    rows.append(summarize(f[m].values, "CTRL all August days"))
    # CTRL month x tdom
    m = (IDX.month == 8) & (tdom.values >= 9) & (tdom.values <= 13) & valid.values
    rows.append(summarize(f[m].values, "CTRL August tdom 9-13"))
    show(rows, f"h={h}")
    c_all = f[valid].mean()
    c_td = f[((tdom >= 10) & (tdom <= 12) & valid).values].mean()
    c_aug = f[(IDX.month == 8) & valid.values].mean()
    print(f"  h={h}: excess vs all-days {100*(cell.mean()-c_all):+.3f}pp | "
          f"vs tdom 10-12 {100*(cell.mean()-c_td):+.3f}pp | "
          f"vs all-August {100*(cell.mean()-c_aug):+.3f}pp")

# --------------------------------------------- 2. PLACEBO LADDER OVER AUG tdom
print("\n" + "=" * 92)
print("2. PLACEBO LADDER: TLT h=10 by AUGUST trading-day-of-month")
print("   spike -> the cell is real; plateau -> it is the month, not the window")
print("=" * 92)
f10 = fwd_lag(tlt, 10, 1)
c_all10 = f10.dropna().mean()
rows = []
for td in range(1, 23):
    m = (IDX.month == 8) & (tdom.values == td) & f10.notna().values
    v = f10[m]
    if len(v) < 5:
        continue
    rows.append({"aug_tdom": td, "n": len(v), "mean_pct": round(100 * v.mean(), 3),
                 "excess_all_pct": round(100 * (v.mean() - c_all10), 3),
                 "hit": round(100 * (v > 0).mean(), 1)})
show(rows, "August tdom ladder (h=10, lag=1)")

print("\n   ...and the same ladder for ALL MONTHS (is tdom 11 special anywhere?)")
rows = []
for td in range(1, 23):
    m = (tdom.values == td) & f10.notna().values
    v = f10[m]
    if len(v) < 20:
        continue
    rows.append({"tdom": td, "n": len(v), "mean_pct": round(100 * v.mean(), 3),
                 "excess_all_pct": round(100 * (v.mean() - c_all10), 3)})
show(rows, "all-month tdom ladder (h=10)")

# ------------------------------------------------- 3. MONTH-OF-YEAR CONTROL
print("\n" + "=" * 92)
print("3. MONTH-OF-YEAR: TLT h=10 lag-1 drift by calendar month (registry 08-14)")
print("=" * 92)
rows = []
for mo in range(1, 13):
    v = f10[(IDX.month == mo) & f10.notna().values]
    rows.append({"month": mo, "n": len(v), "mean_pct": round(100 * v.mean(), 3),
                 "hit": round(100 * (v > 0).mean(), 1)})
show(rows, "TLT h=10 by month")

# --------------------------------------- 4. OPEX vs LATE-AUGUST ATTRIBUTION
print("\n" + "=" * 92)
print("4. ATTRIBUTION: is it opex-minus-4, or is it mid-August?")
print("=" * 92)
opex = load_events(["opex"])["date"]
posn = pd.Series(range(len(IDX)), index=IDX)
anchor = []
for d in opex:
    p = posn.get(pd.Timestamp(d))
    if p is None or p - 4 < 0:
        continue
    anchor.append(IDX[p - 4])
anchor = pd.DatetimeIndex(sorted(set(anchor)))
print(f"opex-4 anchors N={len(anchor)}")
a_aug = pd.DatetimeIndex([d for d in anchor if d.month == 8])
a_non = anchor.difference(a_aug)
rows = [summarize(f10.reindex(anchor).dropna().values, "opex-4 ALL months"),
        summarize(f10.reindex(a_non).dropna().values, "opex-4 EX-August"),
        summarize(f10.reindex(a_aug).dropna().values, "opex-4 August only"),
        summarize(f10.dropna().values, "CTRL all days")]
show(rows, "opex anchor, in and out of August (h=10)")
c_non = f10.reindex(a_non).dropna()
print(f"  opex-4 ex-August excess vs all-days: "
      f"{100*(c_non.mean()-c_all10):+.3f}pp  (sign {int((c_non>0).sum())}-"
      f"{int((c_non<=0).sum())}, p {sign_test(int((c_non>0).sum()), len(c_non)):.4f})")

# August days NOT in opex week (opex-4 .. opex)
opex_week = set()
for d in opex:
    p = posn.get(pd.Timestamp(d))
    if p is None:
        continue
    for k in range(0, 6):
        if p - k >= 0:
            opex_week.add(IDX[p - k])
aug_days = pd.DatetimeIndex([d for d in IDX if d.month == 8])
aug_in = pd.DatetimeIndex([d for d in aug_days if d in opex_week])
aug_out = aug_days.difference(aug_in)
show([summarize(f10.reindex(aug_in).dropna().values, "August IN opex week"),
      summarize(f10.reindex(aug_out).dropna().values, "August OUT of opex week"),
      summarize(f10.reindex(pd.DatetimeIndex([d for d in IDX if d.month == 8
                                              and 6 <= d.day <= 16]))
                .dropna().values, "August day 6-16 (registry cell)")],
     "August split by opex week (h=10)")

# ------------------------------------------------------- 5. MIDTERM SUBSET
print("\n" + "=" * 92)
print("5. midterm subset of the cell (N=6) and the cycle control")
print("=" * 92)
for h in (5, 8, 10):
    f = fwd_lag(tlt, h, 1)
    cell = f.reindex(epi).dropna()
    mid = pd.DatetimeIndex([d for d in cell.index if d.year % 4 == 2])
    non = cell.index.difference(mid)
    show([summarize(cell.reindex(mid).values, f"midterm h={h}"),
          summarize(cell.reindex(non).values, f"non-midterm h={h}")], f"h={h}")

# ---------------------------------------------------------- 6. SEARCH CHARGE
print("\n" + "=" * 92)
print("6. multiplicity: this cell came out of a grid (3 anchors x 10 classes x")
print("   6 horizons ~ 180 cells, plus a 12-month x 22-tdom seasonal grid).")
print("   A nominal p of 0.05 buys nothing at that width; the excess must stand")
print("   against the tdom+month control on its own size.")
