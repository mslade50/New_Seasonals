"""C2 watchlist arming check -- the surviving cell is 'long TLT, MOC the close
before a PPI print, exit MOC the print close'. This week that entry close is
Wed 2026-08-12, which is ITSELF a CPI session (CPI 08-12, PPI 08-13, gap +1).

So before parking it: does the cell still work when the entry close is a CPI
day? Historically PPI usually PRECEDES CPI (227 of 316), so most entries are
ordinary sessions. Split it, and split by refunding month, and state the arm
date.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["TLT", "IEF"]).dropna(subset=["TLT"])
idx = px.index
ev = load_events(["ppi", "cpi"])
c = px["TLT"].values
ci = px["IEF"].values

ppi_pos = [int(idx.searchsorted(x, "left")) for x in ev[ev.event == "ppi"]["date"]]
ppi_pos = [p for p in ppi_pos if 1 < p < len(idx)]
cpi_sess = {idx[int(idx.searchsorted(x, "left"))]
            for x in ev[ev.event == "cpi"]["date"]
            if 0 <= int(idx.searchsorted(x, "left")) < len(idx)}

v = np.array([c[p] / c[p - 1] - 1.0 for p in ppi_pos])
vi = np.array([ci[p] / ci[p - 1] - 1.0 for p in ppi_pos])
d = pd.DatetimeIndex([idx[p] for p in ppi_pos])
entry_is_cpi = np.array([idx[p - 1] in cpi_sess for p in ppi_pos])
ref = np.isin(d.month, [2, 5, 8, 11])
base = px["TLT"].pct_change().dropna()

print(f"PPI prints: {len(v)}   entry close is a CPI session: "
      f"{int(entry_is_cpi.sum())} (this week's shape)")
print(f"TLT all-days daily drift control: {100*base.mean():+.4f}%\n")

rows = []
for nm, m in [("ALL PPI prints", np.ones(len(v), bool)),
              ("entry close IS a CPI session (THIS WEEK)", entry_is_cpi),
              ("entry close is NOT a CPI session", ~entry_is_cpi),
              ("  ...and refunding month (AUG=yes)", entry_is_cpi & ref),
              ("  ...CPI-entry, 2018+", entry_is_cpi & (d.year >= 2018)),
              ("  ...CPI-entry, pre-2018", entry_is_cpi & (d.year < 2018)),
              ("  ...CPI-entry, midterm (2026=yes)", entry_is_cpi & (d.year % 4 == 2)),
              ("refunding month", ref), ("other months", ~ref),
              ("2018+", d.year >= 2018), ("pre-2018", d.year < 2018)]:
    if m.sum() == 0:
        continue
    s = summarize(v[m], nm)
    s["edge_pp"] = round(s["mean_pct"] - 100 * base.mean(), 4)
    s["sign_p"] = round(sign_test(int((v[m] > 0).sum()), int(m.sum())), 4)
    s["boot"] = round(bootstrap_p_le0(v[m]), 3) if m.sum() >= 3 else np.nan
    rows.append(s)
show(rows, "long TLT, MOC close before PPI -> MOC PPI close")

print("\nIEF same split (coherence):")
for nm, m in [("ALL", np.ones(len(v), bool)),
              ("entry close IS a CPI session", entry_is_cpi),
              ("entry close is NOT", ~entry_is_cpi)]:
    s = summarize(vi[m], nm)
    print(f"  {nm:34s} N={s['n']:3d} {s['mean_pct']:+.3f}% hit {s['hit']:.1f}% "
          f"t {s['t']:+.2f} sign p "
          f"{sign_test(int((vi[m] > 0).sum()), int(m.sum())):.4f}")

print("\n" + "=" * 88)
print("ARM DATE")
print("=" * 88)
fut = ev[(ev.event == "ppi") & (ev.date > "2026-08-07")]["date"].head(5)
for x in fut:
    print(f"  PPI {x.date()} ({x.day_name()})  -> entry MOC the prior session, "
          f"refunding month: {x.month in (2,5,8,11)}")
print("\n  NEXT ARM: the morning of the session immediately before PPI "
      "2026-08-13 = Wednesday 2026-08-12.")
print("  It is NOT armed from a Monday run: entering Mon 08-10 or Tue 08-11 "
      "buys the pre-print sessions, which average")
print("  -0.009%/-0.142% per session in 2018+ vs the print session's +0.133%.")
