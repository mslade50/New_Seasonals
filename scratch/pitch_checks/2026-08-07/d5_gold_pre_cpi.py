"""D5 - GLD drift in the sessions BEFORE a CPI release.

Convention: the pre-CPI window of k sessions ENTERS at the close k+1 trading
days before the CPI date and EXITS at the close 1 trading day before it
(the last close before the 08:30 print). The real order for 2026 is entry
MOC 2026-08-07, exit MOC 2026-08-11 => k = 2 (verified in d0_fires_today.py;
the candidate's "3 sessions" is off by one).

Conditional variant: GLD trailing 5d return > thr measured at the ENTRY close
(knowable at an MOC entry).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import bootstrap_p_le0, close_panel, era_split, load_events, show, summarize  # noqa: E402

px = close_panel(["GLD"])
g = px["GLD"].dropna()
cal = g.index
pos = pd.Series(range(len(cal)), index=cal)
cpi = load_events(["cpi"])["date"]
cpi = cpi[(cpi >= cal[0]) & (cpi <= cal[-1])]
r5 = g.pct_change(5)


def window(k: int):
    """(entry_dates, rets) for the k-session pre-CPI window."""
    ed, rv = [], []
    for d in cpi:
        j = int(np.searchsorted(cal.values, np.datetime64(d)))  # first cal day >= d
        ex = j - 1 if (j < len(cal) and cal[j] == d) else j - 1
        en = ex - k
        if en < 0 or ex >= len(cal) or en >= len(cal):
            continue
        ed.append(cal[en])
        rv.append(g.iloc[ex] / g.iloc[en] - 1.0)
    return pd.DatetimeIndex(ed), np.array(rv)


print(f"GLD {cal[0].date()}..{cal[-1].date()}  CPI events in range: {len(cpi)}")
print("Decluster note: CPI is monthly (~21 td apart) and k<=5, so min-gap "
      "declustering is a strict NO-OP. Day-level N == episode-level N below.")

# ---- 1. does the pattern exist -------------------------------------------
rows, base = [], []
for k in (1, 2, 3, 4, 5):
    ed, rv = window(k)
    rows.append(summarize(rv, f"pre-CPI k={k}"))
    allday = (g.shift(-k) / g - 1.0).loc[ed[0]:ed[-1]].dropna().values
    base.append(summarize(allday, f"all-days h={k}"))
show(rows + base, "1. pre-CPI window vs GLD's own unconditional drift (same span)")

# ---- conditional on already rallying -------------------------------------
ed2, rv2 = window(2)
c5 = r5.reindex(ed2).values
rows = [summarize(rv2, "k=2 ALL CPI (control)"),
        summarize(rv2[c5 > 0], "k=2 & 5d ret > 0"),
        summarize(rv2[c5 <= 0], "k=2 & 5d ret <= 0")]
show(rows, "1b. conditional on gold already rallying (today: 5d +3.32%)")

# ---- 2. era, worst, drop-best --------------------------------------------
for lab, mask in [("unconditional", np.ones(len(rv2), bool)), ("5d>0", c5 > 0)]:
    v, d = rv2[mask], ed2[mask]
    db = np.sort(v)[:-1] if len(v) > 2 else v
    show(era_split(d, v) + [summarize(db, "drop-best-episode")],
         f"2. era split + drop-best  [k=2, {lab}]")

# ---- 3. bootstrap ---------------------------------------------------------
print("\n=== 3. bootstrap P(mean<=0), episodes == days (no-op decluster) ===")
for k in (1, 2, 3):
    ed, rv = window(k)
    c = r5.reindex(ed).values
    print(f"  k={k}: all N={len(rv)} P={bootstrap_p_le0(rv):.3f} | "
          f"5d>0 N={int((c > 0).sum())} P={bootstrap_p_le0(rv[c > 0]):.3f}")

# ---- 4. sensitivity grid --------------------------------------------------
grid = []
for k in (1, 2, 3, 4):
    ed, rv = window(k)
    c = r5.reindex(ed).values
    row = {"k": k}
    for thr, lab in [(-99, "all"), (-0.01, ">-1%"), (0.0, ">0"), (0.01, ">+1%"), (0.02, ">+2%")]:
        m = c > thr
        s = summarize(rv[m])
        row[lab] = f"{s.get('mean_pct', np.nan):.2f}/t{s.get('t', np.nan):.1f}/n{s.get('n', 0)}"
    grid.append(row)
show(grid, "4. sensitivity: mean% / t / N across k x 5d-return threshold")

# ---- 5. cost --------------------------------------------------------------
s2 = summarize(rv2[c5 > 0])
print(f"\n=== 5. cost sanity (GLD ~1 bp round trip; need ~5 bps) ===")
print(f"  best conditional cell k=2 & 5d>0: mean {100 * s2['mean_pct'] / 100:.3f}% "
      f"= {100 * s2['mean_pct']:.1f} bps -> {100 * s2['mean_pct'] / 1.0:.0f}x round trip")

# ---- 6. holding THROUGH the print ----------------------------------------
print("\n=== 6. cost of a late exit: the CPI-day session itself (1 td) ===")
pr = []
for d in cpi:
    j = int(np.searchsorted(cal.values, np.datetime64(d)))
    if j <= 0 or j >= len(cal) or cal[j] != d:
        continue
    pr.append((cal[j - 1], g.iloc[j] / g.iloc[j - 1] - 1.0))
pd_, pv = pd.DatetimeIndex([x[0] for x in pr]), np.array([x[1] for x in pr])
c5p = r5.reindex(pd_).values
show([summarize(pv, "CPI-day move, all"),
      summarize(pv[c5p > 0], "CPI-day, 5d>0 into it"),
      summarize(pv[c5p <= 0], "CPI-day, 5d<=0 into it")],
     "6. GLD on the print (what a late exit eats)")
