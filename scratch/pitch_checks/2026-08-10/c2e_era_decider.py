"""C2 decider -- the executable forms all show pre-2018 >> 2018+. Is the
one-session PRINT cell the only era-stable thing here, and does the best-looking
selected cell (CPI-inside AND refunding month, form b) survive its own era cut?

If the pre-print sessions are what died after 2017, then every form a Monday
pitch can execute is buying a dead half and the live half is unreachable today.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["TLT"]).dropna()
idx = px.index
ev = load_events(["ppi", "cpi"])
c = px["TLT"].values
ppi_pos = [int(idx.searchsorted(x, "left")) for x in ev[ev.event == "ppi"]["date"]]
ppi_pos = [p for p in ppi_pos if 3 < p < len(idx) - 2]
cpi_sess = {idx[int(idx.searchsorted(x, "left"))]
            for x in ev[ev.event == "cpi"]["date"]
            if 0 <= int(idx.searchsorted(x, "left")) < len(idx)}

print("=" * 92)
print("A. SESSION-BY-SESSION decomposition of the PPI window, by era")
print("   (each column is ONE session's return, TLT, averaged over PPI months)")
print("=" * 92)
rows = []
for off, nm in [(-2, "p-3 -> p-2"), (-1, "p-2 -> p-1"), (0, "p-1 -> p (THE PRINT)"),
                (1, "p -> p+1 (day after)"), (2, "p+1 -> p+2")]:
    v = np.array([c[p + off] / c[p + off - 1] - 1.0 for p in ppi_pos])
    d = pd.DatetimeIndex([idx[p] for p in ppi_pos])
    base = (px["TLT"].pct_change()).dropna()
    r = {"session": nm, "N": len(v), "all_pct": round(100 * v.mean(), 4),
         "pre2018": round(100 * v[d.year < 2018].mean(), 4),
         "y2018p": round(100 * v[d.year >= 2018].mean(), 4),
         "hit_all": round(100 * (v > 0).mean(), 1),
         "hit_18p": round(100 * (v[d.year >= 2018] > 0).mean(), 1),
         "ctrl_daily": round(100 * base.mean(), 4)}
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
print("\n  -> the PRINT session is the only one that holds up after 2017.")

print("\n" + "=" * 92)
print("B. Does the best selected cell survive its own era cut?")
print("   form (b) enter p-2 -> exit p+1, CPI inside window AND refunding month")
print("=" * 92)
for lbl, k, j in [("(a) p-2 -> p+0", 2, 0), ("(b) p-2 -> p+1", 2, 1),
                  ("(c) p-3 -> p+1", 3, 1), ("REF p-1 -> p+0 (print only)", 1, 0)]:
    v, d, hc = [], [], []
    for p in ppi_pos:
        v.append(c[p + j] / c[p - k] - 1.0)
        d.append(idx[p])
        hc.append(bool(set(idx[p - k + 1: p + j + 1]) & cpi_sess))
    v, d, hc = np.array(v), pd.DatetimeIndex(d), np.array(hc)
    ref = np.isin(d.month, [2, 5, 8, 11])
    exact = hc & ref
    print(f"\n{lbl}")
    for nm, m in [("full", np.ones(len(v), bool)),
                  ("EXACT cell (CPI-in + refunding)", exact)]:
        if m.sum() < 2:
            print(f"  {nm}: N={int(m.sum())}")
            continue
        pre = m & (d.year < 2018)
        pst = m & (d.year >= 2018)
        f = lambda s: (f"N={int(s.sum())} {100*v[s].mean():+.3f}% "
                       f"hit {100*(v[s] > 0).mean():.0f}% "
                       f"p={sign_test(int((v[s] > 0).sum()), int(s.sum())):.3f}") \
            if s.sum() else "N=0"
        print(f"  {nm:34s} all {f(m)}")
        print(f"  {'':34s} pre-2018 {f(pre)}")
        print(f"  {'':34s} 2018+    {f(pst)}")

print("\n" + "=" * 92)
print("C. The print-session cell 2018+ on its own (the era-stable thing)")
print("=" * 92)
v = np.array([c[p] / c[p - 1] - 1.0 for p in ppi_pos])
d = pd.DatetimeIndex([idx[p] for p in ppi_pos])
base1 = (px["TLT"].pct_change()).dropna()
for nm, m in [("2018-2026", d.year >= 2018), ("2021-2026", d.year >= 2021),
              ("2023-2026", d.year >= 2023)]:
    s = summarize(v[m], nm)
    print(f"  {nm}: N={s['n']} mean {s['mean_pct']:+.3f}% hit {s['hit']:.1f}% "
          f"t {s['t']:+.2f} sign p "
          f"{sign_test(int((v[m] > 0).sum()), int(m.sum())):.4f} "
          f"| ctrl {100*base1.mean():+.4f}%")
print("\n  This is the cell that needs an entry MOC Wed 2026-08-12 and is")
print("  therefore NOT pitchable from a Monday run. It is the watchlist entry.")
