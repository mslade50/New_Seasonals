"""D6 - DX-Y.NYB (spot dollar index) drift in the sessions BEFORE a CPI release.

Same convention as D5: a k-session pre-CPI window ENTERS at the close k+1
trading days before the CPI date and EXITS at the last close before the print.
The real 2026 order (entry MOC 08-07, exit MOC 08-11) is k = 2, not 3.

DX-Y.NYB is a SPOT index, so it is a proxy for the ICE DX futures contract and
carries no roll/carry. Rate differentials make the futures basis non-zero, so
the spot series slightly misstates a futures P&L; direction and magnitude of a
3-session move are close enough to grade an edge this small.

Conditional cell: dollar 21d return rank depressed inside a strong 63d uptrend
(today: rank21 19.8, rank63 90.5), both measured at the ENTRY close.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import bootstrap_p_le0, close_panel, era_split, load_events, pct_rank, show, summarize  # noqa: E402

px = close_panel(["DX-Y.NYB"])
d = px["DX-Y.NYB"].dropna()
cal = d.index
cpi = load_events(["cpi"])["date"]
cpi = cpi[(cpi >= cal[0]) & (cpi <= cal[-1])]
r21, r63 = pct_rank(d, 21), pct_rank(d, 63)


def window(k: int):
    ed, rv = [], []
    for x in cpi:
        j = int(np.searchsorted(cal.values, np.datetime64(x)))
        ex, en = j - 1, j - 1 - k
        if en < 0 or ex >= len(cal):
            continue
        ed.append(cal[en])
        rv.append(d.iloc[ex] / d.iloc[en] - 1.0)
    return pd.DatetimeIndex(ed), np.array(rv)


print(f"DX-Y.NYB {cal[0].date()}..{cal[-1].date()}  CPI events in range: {len(cpi)}")
print("Decluster note: CPI is monthly, k<=5 -> min-gap declustering is a NO-OP. "
      "Day-level N == episode-level N throughout.")

rows, base = [], []
for k in (1, 2, 3, 4, 5):
    ed, rv = window(k)
    rows.append(summarize(rv, f"pre-CPI k={k}"))
    base.append(summarize((d.shift(-k) / d - 1.0).loc[ed[0]:ed[-1]].dropna().values, f"all-days h={k}"))
show(rows + base, "1. pre-CPI window vs DX's own unconditional drift (same span)")

ed2, rv2 = window(2)
a21, a63 = r21.reindex(ed2).values, r63.reindex(ed2).values
cond = (a21 < 20) & (a63 > 90)
show([summarize(rv2, "k=2 ALL CPI (control)"),
      summarize(rv2[cond], "k=2 & rank21<20 & rank63>90"),
      summarize(rv2[~cond], "k=2 & NOT cond"),
      summarize(rv2[a21 < 20], "k=2 & rank21<20 only"),
      summarize(rv2[a63 > 90], "k=2 & rank63>90 only")],
     "1b. conditional: depressed 21d rank inside a strong 63d uptrend")

for lab, m in [("unconditional", np.ones(len(rv2), bool)), ("cond", cond)]:
    v, dt = rv2[m], ed2[m]
    db = np.sort(v)[:-1] if len(v) > 2 else v
    show(era_split(dt, v) + [summarize(db, "drop-best-episode")], f"2. era + drop-best [k=2, {lab}]")

print("\n=== 3. bootstrap P(mean<=0) (episodes == days) ===")
for k in (1, 2, 3):
    ed, rv = window(k)
    c = (r21.reindex(ed).values < 20) & (r63.reindex(ed).values > 90)
    print(f"  k={k}: all N={len(rv)} P={bootstrap_p_le0(rv):.3f} | "
          f"cond N={int(c.sum())} P={bootstrap_p_le0(rv[c]):.3f} "
          f"mean={100 * np.nanmean(rv[c]) if c.sum() else float('nan'):.3f}%")

grid = []
for k in (1, 2, 3):
    ed, rv = window(k)
    q21, q63 = r21.reindex(ed).values, r63.reindex(ed).values
    for t21 in (10, 20, 30, 40):
        row = {"k": k, "rank21<": t21}
        for t63 in (80, 90, 95):
            m = (q21 < t21) & (q63 > t63)
            s = summarize(rv[m])
            row[f">{t63}"] = (f"{s.get('mean_pct', float('nan')):.3f}/t{s.get('t', float('nan')):.1f}"
                              f"/n{s.get('n', 0)}")
        grid.append(row)
show(grid, "4. sensitivity: mean% / t / N across k x rank21 x rank63")

sc = summarize(rv2[cond])
bps = 100 * sc.get("mean_pct", float("nan"))
print("\n=== 5. cost sanity ===")
print(f"  target cell k=2 cond: mean {bps:.1f} bps (|edge| {abs(bps):.1f})")
print(f"  DX futures 1.5 bps round trip -> {abs(bps) / 1.5:.1f}x  (need ~5x)")
print(f"  UUP        6.0 bps round trip -> {abs(bps) / 6.0:.1f}x  (need ~5x)")

print("\n=== 6. cost of a late exit: the CPI-day session itself ===")
pr = []
for x in cpi:
    j = int(np.searchsorted(cal.values, np.datetime64(x)))
    if j <= 0 or j >= len(cal) or cal[j] != x:
        continue
    pr.append((cal[j - 1], d.iloc[j] / d.iloc[j - 1] - 1.0))
pdd, pv = pd.DatetimeIndex([a for a, _ in pr]), np.array([b for _, b in pr])
cp = (r21.reindex(pdd).values < 20) & (r63.reindex(pdd).values > 90)
show([summarize(pv, "CPI-day move, all"), summarize(pv[cp], "CPI-day, cond into it")],
     "6. DX on the print")

# ---- 7. how rare is today's DX state at all? -----------------------------
st = (r21 < 20) & (r63 > 90)
st = st & r63.notna()
print(f"\n=== 7. state rarity: rank21<20 & rank63>90 fires on {int(st.sum())} of "
      f"{int(r63.notna().sum())} days ({100 * st.mean():.2f}%), "
      f"{len(pd.Series(st[st].index).dt.year.unique())} distinct years ===")
print("  years:", sorted(pd.Series(st[st].index).dt.year.unique().tolist()))
rows = []
for h in (2, 3, 5, 10):
    f = (d.shift(-h) / d - 1.0)
    rows.append(summarize(f[st].values, f"state h={h}"))
    rows.append(summarize(f.dropna().values, f"all-days h={h}"))
show(rows, "7. the state OFF the CPI calendar (all days, day-level, overlapping)")
