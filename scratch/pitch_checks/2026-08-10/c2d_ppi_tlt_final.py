"""C2 final -- the exact cells a MONDAY pitch can execute, fully split.

The headline cell from round 2 (enter the close BEFORE the print, exit the
print close) needs an entry MOC Wed 2026-08-12 and is therefore NOT pitchable
today. Only these are:
  (a) enter MOC Tue 08-11 -> exit MOC Thu 08-13 (the PPI close)      p-2 -> p+0
  (b) enter MOC Tue 08-11 -> exit MOC Fri 08-14                      p-2 -> p+1
  (c) enter MOC Mon 08-10 -> exit MOC Fri 08-14                      p-3 -> p+1

Each gets: full sample, CPI-inside-the-window subset (this week's shape),
midterm, era, refunding month, concentration, cost.
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

ppi_pos = [int(idx.searchsorted(x, "left")) for x in ev[ev.event == "ppi"]["date"]]
ppi_pos = [p for p in ppi_pos if 0 < p < len(idx)]
cpi_sess = {idx[int(idx.searchsorted(x, "left"))]
            for x in ev[ev.event == "cpi"]["date"]
            if 0 <= int(idx.searchsorted(x, "left")) < len(idx)}

FORMS = [("(a) enter p-2, exit p+0  [Tue -> Thu, 2td]", 2, 0),
         ("(b) enter p-2, exit p+1  [Tue -> Fri, 3td]", 2, 1),
         ("(c) enter p-3, exit p+1  [Mon -> Fri, 4td]", 3, 1)]

for lbl, k, j in FORMS:
    v, dts, has_cpi = [], [], []
    for p in ppi_pos:
        if p - k < 0 or p + j >= len(idx):
            continue
        v.append(c[p + j] / c[p - k] - 1.0)
        dts.append(idx[p])
        has_cpi.append(bool(set(idx[p - k + 1: p + j + 1]) & cpi_sess))
    v = np.array(v)
    d = pd.DatetimeIndex(dts)
    has_cpi = np.array(has_cpi)
    h = k + j
    base = (px["TLT"].shift(-h) / px["TLT"] - 1.0).dropna()
    ctrl = 100 * base.mean()

    print("\n" + "=" * 92)
    print(f"{lbl}   ctrl (TLT own {h}td drift, all days) {ctrl:+.3f}%")
    print("=" * 92)
    rows = []
    ref = np.isin(d.month, [2, 5, 8, 11])
    mid = d.year % 4 == 2
    pre = d.year < 2018
    for nm, m in [("FULL SAMPLE", np.ones(len(v), bool)),
                  ("CPI inside window (THIS WEEK'S SHAPE)", has_cpi),
                  ("no CPI inside", ~has_cpi),
                  ("refunding month Feb/May/Aug/Nov (AUG=YES)", ref),
                  ("other months", ~ref),
                  ("CPI inside AND refunding month (EXACT)", has_cpi & ref),
                  ("midterm year (2026 = YES)", mid),
                  ("non-midterm", ~mid),
                  ("pre-2018", pre), ("2018+", ~pre),
                  ("2002-2012", d.year <= 2012), ("2013+", d.year >= 2013)]:
        if m.sum() == 0:
            continue
        s = summarize(v[m], nm)
        s["edge_pp"] = round(s["mean_pct"] - ctrl, 3)
        w = int((v[m] > 0).sum())
        s["sign_p"] = round(sign_test(w, int(m.sum())), 4)
        s["boot"] = round(bootstrap_p_le0(v[m]), 3) if m.sum() >= 3 else np.nan
        rows.append(s)
    show(rows, "")
    print(" ", cluster_note(d, v, k=3))
    yr = pd.Series(100 * v, index=d).groupby(d.year).sum()
    print(f"  years positive {int((yr > 0).sum())}/{len(yr)}; "
          f"drop best year ({yr.idxmax()}): mean "
          f"{100*v[d.year != yr.idxmax()].mean():+.4f}% (full {100*v.mean():+.4f}%)")
    print(f"  cost: 2 bps round trip vs {100*100*v.mean():.1f} bps edge -> "
          f"{100*100*v.mean()/2:.1f}x")

# ---------------------------------------------------------------- IEF cross
print("\n" + "=" * 92)
print("CROSS-INSTRUMENT COHERENCE (same forms on IEF, 7-10y belly)")
print("=" * 92)
ci = px["IEF"].values
for lbl, k, j in FORMS:
    v = [ci[p + j] / ci[p - k] - 1.0 for p in ppi_pos
         if p - k >= 0 and p + j < len(idx) and not np.isnan(ci[p - k])]
    v = np.array([x for x in v if not np.isnan(x)])
    h = k + j
    base = (px["IEF"].shift(-h) / px["IEF"] - 1.0).dropna()
    s = summarize(v, lbl)
    print(f"  {lbl:44s} N={s['n']:3d} mean {s['mean_pct']:+.3f}% "
          f"hit {s['hit']:5.1f}% t {s['t']:+5.2f} edge "
          f"{s['mean_pct']-100*base.mean():+.3f}pp sign p "
          f"{sign_test(int((v>0).sum()), len(v)):.4f}")

# ------------------------------------------------------ MULTIPLICITY LEDGER
print("\n" + "=" * 92)
print("MULTIPLICITY LEDGER (what was searched, so the p-values can be read)")
print("=" * 92)
print("  pre-specified, no direction prior: PPI session, SPY and TLT. 2 tests.")
print("  scans run afterwards: 12-cell entry/exit grid, 7 horizons, 4 anchor")
print("  offsets, 3 lags, 4 instruments, 12 calendar months, 4 era cuts.")
print("  -> the SINGLE number that owes no correction is the pre-specified")
print("     PPI-session cell (p-1 -> p+0), and it is not the one pitchable")
print("     today. Everything below is a selected cell; read it as such.")
