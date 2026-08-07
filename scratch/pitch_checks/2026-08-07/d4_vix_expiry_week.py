"""D4 - "VIX-expiry week drift": SPY MOC 3 sessions before VIX expiry -> MOC on expiry day.
Thesis under test: hedge unwind into the VIX settle lifts the index.

Uses real vix_expiry dates from macro_events.csv. Also asks the killer question: is this
just "the 3 sessions before monthly opex" (opex is normally 2 sessions AFTER vix expiry),
i.e. a relabelled pre-opex drift?
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _common import *  # noqa: F401,F403

px = load_prices(["SPY"])["SPY"]
spy = px["Close"][px.index <= "2026-08-06"]
idx = spy.index
pos = pd.Series(range(len(idx)), index=idx)

ev = load_events()
vix = ev[ev.event == "vix_expiry"].date
opex = ev[ev.event == "opex"].date
qw = set(ev[ev.event == "quad_witching"].date)


def snap(d):
    """Nearest session <= d. None for forward-dated events (past the last bar) so the
    2026-08-19+ expiries do NOT all collapse onto 2026-08-06 and fake extra cells."""
    if d > idx[-1]:
        return None
    s = idx[idx <= d]
    if not len(s) or (d - s[-1]).days > 4:
        return None
    return s[-1]


def window(anchor_dates, lead: int) -> pd.DataFrame:
    """entry MOC `lead` sessions before the anchor, exit MOC on the anchor."""
    rows = []
    for d in anchor_dates:
        b = snap(d)
        if b is None or b < idx[0]:
            continue
        pb = pos[b]
        pa = pb - lead
        if pa < 0:
            continue
        rows.append({"anchor": b, "entry": idx[pa], "ret": spy.iloc[pb] / spy.iloc[pa] - 1.0,
                     "year": b.year, "month": b.month})
    return pd.DataFrame(rows)


vx = window(vix, 3)
op = window(opex, 3)
print("### D4 VIX-expiry week | entry MOC expiry-3, exit MOC on expiry day (3 td hold)")
print(f"vix cells {len(vx)}  {vx.anchor.min().date()} .. {vx.anchor.max().date()}")

# gap between vix expiry and same-month opex, in sessions
gaps = []
for d in vix:
    b = snap(d)
    o = opex[(opex.dt.year == d.year) & (opex.dt.month == d.month)]
    if b is None or o.empty:
        continue
    ob = snap(o.iloc[0])
    if ob is not None and b in pos.index and ob in pos.index:
        gaps.append(pos[ob] - pos[b])
print(f"sessions from vix_expiry to same-month opex: {pd.Series(gaps).value_counts().to_dict()}")

# ---------- 1. pattern vs TWO controls ----------
u3 = (spy.shift(-3) / spy - 1.0).dropna()
rows = [summarize(vx.ret.values, "VIX-EXPIRY wk (exp-3 -> exp)"),
        summarize(op.ret.values, "OPEX wk (opex-3 -> opex)"),
        summarize(u3.values, "ctrl A: SPY any day h3 2000+"),
        summarize(u3[u3.index >= vx.entry.min()].values, "ctrl B: SPY any day h3, same window")]
show(rows, "1. conditional vs controls")

# is it just pre-opex? overlap + residual
vset, oset = set(vx.entry), set(op.entry)
print(f"\nentry-day overlap vix-week vs opex-week: {len(vset & oset)} of {len(vset)}")
rows = [summarize(vx[~vx.entry.isin(oset)].ret.values, "vix-week NOT overlapping opex-week"),
        summarize(op[~op.entry.isin(vset)].ret.values, "opex-week NOT overlapping vix-week")]
# strictly disjoint construction: vix-week = exp-3..exp ; pre-opex = opex-3..opex which
# for a 2-session gap covers exp-1..opex. Test the clean pre-VIX-only leg exp-3 -> exp-1.
rows.append(summarize(
    np.array([spy.iloc[pos[a] ] / spy.iloc[pos[a] - 0] - 1.0 for a in []]), "(placeholder)"))
rows = [r for r in rows if r.get("n", 0) > 0]
show(rows, "1b. vix-week vs opex-week disentangle")

# leg decomposition: exp-3 -> exp-1 (pure pre-vix) and exp-1 -> exp (settle day)
legs = []
for r in vx.itertuples():
    p = pos[r.anchor]
    legs.append({"anchor": r.anchor, "year": r.year,
                 "leg_pre": spy.iloc[p - 1] / spy.iloc[p - 3] - 1.0,
                 "leg_settle": spy.iloc[p] / spy.iloc[p - 1] - 1.0})
legs = pd.DataFrame(legs)
show([summarize(legs.leg_pre.values, "leg exp-3 -> exp-1 (2 td)"),
      summarize(legs.leg_settle.values, "leg exp-1 -> exp (1 td, the settle)"),
      summarize((spy.shift(-1) / spy - 1.0).dropna().values, "ctrl: SPY any day h1")],
     "1c. where inside the week does the return live?")

# ---------- 2/3. decluster, era, drop-best, bootstrap ----------
d = declusters(pd.DatetimeIndex(vx.entry), 3, idx)
sub = vx[vx.entry.isin(d)]
s = summarize(sub.ret.values, "VIX-week declustered(gap=3)")
s["boot_P<=0"] = bootstrap_p_le0(sub.ret.values)
r = sub.ret.values
s["drop_best_mean"] = 100 * np.delete(r, np.argmax(r)).mean()
s["drop_worst_mean"] = 100 * np.delete(r, np.argmin(r)).mean()
show([s], "2/3. declustered + bootstrap + drop-best")
print("(monthly anchors 3 td apart are already non-overlapping -> day == episode level)")

rows = era_split(pd.DatetimeIndex(vx.entry), vx.ret.values)
rows += era_split(pd.DatetimeIndex(vx.entry), vx.ret.values, cut="2013-01-01")
show(rows, "2b. era stability (2018 cut and 2013 cut)")

# worst years / concentration
by_yr = vx.groupby("year").ret.mean() * 100
print(f"\nbest 3 years: {by_yr.nlargest(3).round(2).to_dict()}")
print(f"worst 3 years: {by_yr.nsmallest(3).round(2).to_dict()}")
print(f"worst single window: {100*vx.ret.min():.2f}% on {vx.loc[vx.ret.idxmin(),'anchor'].date()}")

# ---------- 4. sensitivity: lead 1..6, and month exclusions ----------
rows = []
for lead in range(1, 7):
    w = window(vix, lead)
    s = summarize(w.ret.values, f"lead={lead}")
    s["boot_P<=0"] = bootstrap_p_le0(w.ret.values)
    u = (spy.shift(-lead) / spy - 1.0).dropna()
    s["vs_ctrl_pp"] = s["mean_pct"] - 100 * u.mean()
    rows.append(s)
show(rows, "4. sensitivity: entry lead (sessions before expiry)")

rows = []
for label, m in [("all months", vx),
                 ("ex quad-witching months", vx[~vx.month.isin([3, 6, 9, 12])]),
                 ("quad-witching months only", vx[vx.month.isin([3, 6, 9, 12])]),
                 ("August only", vx[vx.month == 8]),
                 ("midterm years", vx[vx.year % 4 == 2])]:
    rows.append(summarize(m.ret.values, label))
show(rows, "4b. sensitivity: month / year subsets")

# ---------- 5/6. cost + CPI ----------
cpi = set(load_events(["cpi"])["date"])
fl = np.array([any(a < c <= b for c in cpi) for a, b in zip(vx.entry, vx.anchor)])
show([summarize(vx.ret.values[fl], "CPI inside the 3 td"),
      summarize(vx.ret.values[~fl], "no CPI inside")], "6. CPI-in-window split")
print("  2026 cell: entry 2026-08-14 -> exit 2026-08-19; CPI 08-12 and PPI 08-13 land "
      "BEFORE entry, so this specific window is print-free.")

m = 100 * vx.ret.mean()
print(f"\n5. cost sanity: mean {m:+.3f}% -> {m/0.01:.0f}x the ~1 bp round trip; "
      f"edge over h3 control {m - 100*u3.mean():+.3f} pp -> {(m - 100*u3.mean())/0.01:.0f}x cost")
