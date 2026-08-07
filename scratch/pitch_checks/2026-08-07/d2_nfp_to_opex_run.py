"""D2 - "The run into August opex": SPY MOC on the NFP-day close -> MOC on that month's
opex close. Real nfp/opex dates from macro_events.csv.

Candidate = August only, entry MOC 2026-08-07 (the NFP close), exit MOC 2026-08-21 (opex).
Tests: August cell, all-months cell (is August special or is this just a 10 td long?),
midterm-year cell, vs matched-horizon unconditional drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa: F401,F403

px = load_prices(["SPY", "IWM"])
spy = px["SPY"]["Close"]
spy = spy[spy.index <= "2026-08-06"]
idx = spy.index
pos = pd.Series(range(len(idx)), index=idx)

ev = load_events()
nfp = ev[ev.event == "nfp"][["date"]].rename(columns={"date": "nfp"})
opex = ev[ev.event == "opex"][["date"]].rename(columns={"date": "opex"})


def snap(d):
    """Nearest trading day <= d. None past the last bar so forward-dated events
    don't all collapse onto the final session and fake extra cells."""
    if d > idx[-1]:
        return None
    s = idx[idx <= d]
    if not len(s) or (d - s[-1]).days > 4:
        return None
    return s[-1]


def build_cells() -> pd.DataFrame:
    """One row per (year, month): NFP close -> opex close."""
    rows = []
    for _, n in nfp.iterrows():
        y, m = n.nfp.year, n.nfp.month
        o = opex[(opex.opex.dt.year == y) & (opex.opex.dt.month == m)]
        if o.empty:
            continue
        a, b = snap(n.nfp), snap(o.opex.iloc[0])
        if a is None or b is None or a >= b:
            continue
        pa, pb = pos.get(a), pos.get(b)
        if pa is None or pb is None:
            continue
        rows.append({"year": y, "month": m, "entry": a, "exit": b, "td": pb - pa,
                     "ret": spy.iloc[pb] / spy.iloc[pa] - 1.0})
    return pd.DataFrame(rows)


cells = build_cells()
cells = cells[cells.entry >= idx[0]]
print("### D2 NFP close -> opex close | entry MOC on NFP day, exit MOC on opex day")
print(f"cells: {len(cells)}  {cells.entry.min().date()} .. {cells.entry.max().date()}")
print(f"holding td distribution: {cells.td.value_counts().sort_index().to_dict()}")

aug = cells[cells.month == 8]
print(f"\nAugust cells: {len(aug)}, td: {sorted(aug.td.unique())}")
print(aug.assign(ret_pct=(100 * aug.ret).round(2))[["year", "entry", "exit", "td", "ret_pct"]]
      .to_string(index=False))

# ---------- 1. pattern vs TWO controls ----------
rows = [summarize(aug.ret.values, "AUGUST nfp->opex"),
        summarize(cells.ret.values, "ALL MONTHS nfp->opex"),
        summarize(cells[cells.month != 8].ret.values, "non-August nfp->opex")]
# control A: matched-horizon unconditional drift on the same sample window
for h in sorted(cells.td.unique()):
    pass
med_td = int(cells.td.median())
uncond = (spy.shift(-med_td) / spy - 1.0).dropna()
rows.append(summarize(uncond.values, f"ctrl A: SPY any-day h{med_td} 2000+"))
aug_days = uncond[uncond.index.month == 8]
rows.append(summarize(aug_days.values, f"ctrl B: SPY any AUGUST day h{med_td}"))
show(rows, "1. conditional vs controls")

# midterm split
mid = aug[aug.year % 4 == 2]
rows = [summarize(mid.ret.values, "AUGUST midterm only"),
        summarize(aug[aug.year % 4 != 2].ret.values, "AUGUST non-midterm"),
        summarize(cells[cells.year % 4 == 2].ret.values, "ALL MONTHS midterm")]
show(rows, "1b. midterm split")
print("  midterm August rows: " + ", ".join(f"{r.year}:{100*r.ret:+.2f}%" for r in mid.itertuples()))

# ---------- 2/3. episodes, era, drop-best, bootstrap ----------
# August cells are one per year -> already non-overlapping (no declustering needed).
print("\n(August cells are 1/yr and non-overlapping -> day-level == episode-level)")
rows = []
for label, d in [("AUGUST", aug), ("ALL MONTHS", cells)]:
    dd = declusters(pd.DatetimeIndex(d.entry), med_td, idx)
    sub = d[d.entry.isin(dd)]
    s = summarize(sub.ret.values, f"{label} declustered(gap={med_td})")
    s["boot_P<=0"] = bootstrap_p_le0(sub.ret.values)
    r = sub.ret.values
    s["drop_best_mean"] = 100 * np.delete(r, np.argmax(r)).mean()
    s["drop_worst_mean"] = 100 * np.delete(r, np.argmin(r)).mean()
    rows.append(s)
show(rows, "2/3. declustered + bootstrap + drop-best")

rows = []
for label, d in [("AUGUST", aug), ("ALL MONTHS", cells)]:
    for s in era_split(pd.DatetimeIndex(d.entry), d.ret.values):
        s["label"] = f"{label} {s['label']}"
        rows.append(s)
show(rows, "2b. era stability")

# ---------- 4. sensitivity: shift entry/exit one session each way ----------
rows = []
for de in (-1, 0, 1):
    for dx in (-1, 0, 1):
        if abs(de) + abs(dx) > 1 and (de, dx) != (0, 0):
            continue
        v = []
        for r in aug.itertuples():
            pa, pb = pos[r.entry] + de, pos[r.exit] + dx
            if 0 <= pa < len(spy) and 0 <= pb < len(spy) and pb > pa:
                v.append(spy.iloc[pb] / spy.iloc[pa] - 1.0)
        s = summarize(np.array(v), f"entry{de:+d} exit{dx:+d}")
        rows.append(s)
show(rows, "4. sensitivity: entry/exit shifted one session (August cell)")

# start-year sensitivity
rows = []
for y0 in (2000, 2005, 2010, 2015):
    rows.append(summarize(aug[aug.year >= y0].ret.values, f"August, {y0}+"))
show(rows, "4b. sensitivity: sample start year")

# ---------- 6. CPI inside the window ----------
cpi = set(load_events(["cpi"])["date"])
flag = np.array([any(a < c <= b for c in cpi) for a, b in zip(aug.entry, aug.exit)])
show([summarize(aug.ret.values[flag], "August, CPI inside"),
      summarize(aug.ret.values[~flag], "August, no CPI inside")], "6. CPI-in-window split")
print(f"  2026 window 08-07 -> 08-21 contains CPI 08-12 and PPI 08-13: "
      f"{any(pd.Timestamp('2026-08-07') < c <= pd.Timestamp('2026-08-21') for c in cpi)}")

# ---------- 5. cost ----------
m = 100 * aug.ret.mean()
print(f"\n5. cost sanity: August mean {m:+.3f}% vs ~0.01% round trip -> {m/0.01:.0f}x cost")
print(f"   but edge OVER the matched control is {m - 100*uncond.mean():+.3f} pp "
      f"-> {(m - 100*uncond.mean())/0.01:.0f}x cost")
