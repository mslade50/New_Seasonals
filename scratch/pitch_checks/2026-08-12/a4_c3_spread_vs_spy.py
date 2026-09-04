"""C3 round 1 -- long duration against short SPY across the PPI print session.

The question is not whether the spread is positive (two positive-sign legs
usually are) but whether it beats the OUTRIGHT after doubling the cost, and
whether the SPY leg carries information or is noise being paid for twice.

Registry rules applied: price the legs BEFORE the spread; report the regression
beta of TLT on SPY over these windows and test the spread beta-weighted, not
only equal-dollar.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

px = close_panel(["TLT", "SPY", "IEF"]).dropna()
idx = px.index
N = len(idx)
rt = px["TLT"].pct_change().values
rs = px["SPY"].pct_change().values
ri = px["IEF"].pct_change().values

ev = load_events()
sp = lambda k: sorted({int(idx.searchsorted(x, "left"))
                       for x in ev[ev.event == k]["date"]
                       if 0 <= int(idx.searchsorted(x, "left")) < N})
ppi_l = [p for p in sp("ppi") if 1 <= p < N and not np.isnan(rt[p])]
cpi_all = set(sp("cpi"))
dt = pd.DatetimeIndex([idx[p] for p in ppi_l])
ceve = np.array([(p - 1) in cpi_all for p in ppi_l])
mo, yr = dt.month.values, dt.year.values

T = np.array([rt[p] for p in ppi_l])
S = np.array([rs[p] for p in ppi_l])
bh_t = float((rt[1:] > 0).mean())

print("=" * 104)
print("1. THE LEGS, PRICED SEPARATELY (registry rule)")
print("=" * 104)
rows = []
for lbl, v, drift, cost in [("TLT long", T, np.nanmean(rt), 2.5),
                            ("SPY short", -S, -np.nanmean(rs), 1.0)]:
    w = int((v > 0).sum())
    bps = 100 * 100 * (v.mean() - drift)
    rows.append({"leg": lbl, "N": len(v), "raw_pct": round(100 * v.mean(), 4),
                 "own_drift_pct": round(100 * drift, 4),
                 "excess_bps": round(bps, 2), "hit": round(100 * w / len(v), 1),
                 "cost_bps": cost, "x_cost": round(bps / cost, 2)})
print(pd.DataFrame(rows).to_string(index=False))
print("\n  The SPY leg on its own: is short-SPY-into-a-PPI-print a real cell?")
w = int(((-S) > 0).sum())
print(f"    short SPY N={len(S)}  {100*(-S).mean():+.4f}%  hit {100*w/len(S):.1f}%"
      f"  sign p vs SPY's own down-rate "
      f"{sign_test(w, len(S), 1-float((rs[1:] > 0).mean())):.4f}")

print("\n" + "=" * 104)
print("2. BETA OF TLT ON SPY")
print("=" * 104)
ok = ~np.isnan(rt) & ~np.isnan(rs)
b_full = np.polyfit(rs[ok], rt[ok], 1)[0]
b_cell = np.polyfit(S, T, 1)[0]
lv = ceve
b_live = np.polyfit(S[lv], T[lv], 1)[0]
print(f"  full sample  beta(TLT on SPY) = {b_full:+.3f}   "
      f"corr {np.corrcoef(rs[ok], rt[ok])[0,1]:+.3f}")
print(f"  PPI-print days beta           = {b_cell:+.3f}   "
      f"corr {np.corrcoef(S, T)[0,1]:+.3f}")
print(f"  live cell (CPI on eve) beta   = {b_live:+.3f}   "
      f"corr {np.corrcoef(S[lv], T[lv])[0,1]:+.3f}")
print("  A NEGATIVE beta means short SPY is a LONG-duration proxy, so the")
print("  spread doubles the same bet instead of hedging it.")

print("\n" + "=" * 104)
print("3. SPREAD FORMS vs THE OUTRIGHT (parent and live cell)")
print("=" * 104)


def line(v, lbl, cost, n_leg):
    w = int((v > 0).sum())
    bps = 100 * 100 * v.mean()
    sd = v.std(ddof=1)
    return {"form": lbl, "N": len(v), "mean_bps": round(bps, 2),
            "hit": round(100 * w / len(v), 1),
            "sd_pct": round(100 * sd, 3),
            "mean_per_sd": round(v.mean() / sd, 4),
            "cost_bps": cost, "net_bps": round(bps - cost, 2),
            "x_cost": round(bps / cost, 2)}


for nm, m in [("PARENT (N=286)", np.ones(len(T), bool)),
              ("LIVE CELL (CPI on eve)", lv)]:
    print(f"\n--- {nm} ---")
    rows = [line(T[m], "outright long TLT", 2.5, 1),
            line(T[m] - S[m], "TLT - SPY, equal dollar", 3.5, 2),
            line(T[m] - b_cell * S[m], f"TLT - {b_cell:.2f}*SPY (beta-wtd)", 3.5, 2),
            line(-S[m], "outright short SPY", 1.0, 1)]
    print(pd.DataFrame(rows).to_string(index=False))
    o = T[m]
    s2 = T[m] - S[m]
    se = np.sqrt(o.var(ddof=1) / len(o) + s2.var(ddof=1) / len(s2))
    print(f"  spread minus outright = {100*100*(s2.mean()-o.mean()):+.2f} bps "
          f"gross, {100*100*(s2.mean()-o.mean())-1.0:+.2f} bps net of the extra "
          f"leg;  paired t = "
          f"{(s2-o).mean()/((s2-o).std(ddof=1)/np.sqrt(len(o))):+.2f}")

print("\n" + "=" * 104)
print("4. RESIDUAL TEST: is the SPY leg adding anything the TLT leg lacks?")
print("   regress the cell's TLT return on SPY, look at the intercept")
print("=" * 104)
for nm, m in [("parent", np.ones(len(T), bool)), ("live cell", lv)]:
    b, a0 = np.polyfit(S[m], T[m], 1)
    res = T[m] - (a0 + b * S[m])
    print(f"  {nm:10s} alpha = {100*100*a0:+.2f} bps   beta {b:+.3f}   "
          f"resid sd {100*res.std(ddof=1):.3f}%   raw mean "
          f"{100*100*T[m].mean():+.2f} bps")
print("  If alpha ~= the raw mean, SPY explains none of it and shorting SPY")
print("  buys variance, not signal.")

print("\n" + "=" * 104)
print("5. TODAY'S CONDITIONERS ON THE SPREAD (August, midterm)")
print("=" * 104)
aug, mid = mo == 8, (yr % 4) == 2
for nm, m in [("live cell, all", lv), ("live cell August", lv & aug),
              ("live cell midterm", lv & mid),
              ("live cell Aug-or-midterm", lv & (aug | mid))]:
    if m.sum() < 2:
        print(f"  {nm:28s} N={int(m.sum())}")
        continue
    s2 = T[m] - S[m]
    print(f"  {nm:28s} outright {100*100*T[m].mean():+7.2f} bps | "
          f"spread {100*100*s2.mean():+7.2f} bps  "
          f"(N={int(m.sum())}, spread hit {100*(s2>0).mean():.1f}%)")

print("\n" + "=" * 104)
print("6. BOOK-OVERLAP CHECK: the systematic book is long-biased equity")
print("=" * 104)
print("  A short-SPY leg is a partial hedge of existing book exposure rather")
print("  than a standalone idea; that is a reason to prefer it ONLY if it pays")
print("  for itself. Numbers above decide that, not the framing.")
print(f"  short-SPY leg excess on the live cell: "
      f"{100*100*((-S[lv]).mean() - (-np.nanmean(rs))):+.2f} bps vs 1.0 bps cost")
