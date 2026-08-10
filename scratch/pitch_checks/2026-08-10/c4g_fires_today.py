"""C4 final -- does it fire today, and what are the exact entry/exit sessions?

The anchor arithmetic is load-bearing: the freshest bar is 2026-08-07 and the
CPI print is 2026-08-12, so the signal bar sits k=3 trading days before the
print and the MOC entry is TODAY (2026-08-10).  Getting k wrong swaps in the
k=2 cell, which is a different trade.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

import numpy as np
import pandas as pd

P = close_panel(["GDX", "GLD"]).dropna(subset=["GDX"])
g = P["GDX"]
idx = g.index
ASOF = idx[-1]
rk5 = pct_rank(g, 5)

cpi = pd.Timestamp("2026-08-12")
ppi = pd.Timestamp("2026-08-13")
# forward sessions: the cache ends at ASOF, so build the calendar by hand
fwd = pd.bdate_range(ASOF, periods=12)
print("session calendar from the signal bar:")
for i, d in enumerate(fwd):
    tag = []
    if d == cpi:
        tag.append("CPI")
    if d == ppi:
        tag.append("PPI")
    if i == 0:
        tag.append("SIGNAL BAR (D)")
    if i == 1:
        tag.append("<< MOC ENTRY (D+1)")
    if i == 4:
        tag.append("<< h=3 exit")
    if i == 6:
        tag.append("<< h=5 exit")
    if i == 11:
        tag.append("<< h=10 exit")
    print(f"  D+{i:<2d} {d.date()} {d.day_name()[:3]}  {' '.join(tag)}")
k = int(np.busday_count(ASOF.date(), cpi.date()))
print(f"\nbusiness days from signal bar to CPI = {k}  -> anchor k={k} (need k=3)")

print(f"\nTRIGGER STATE at {ASOF.date()}:")
print(f"  GDX rank5 = {rk5.loc[ASOF]:.1f}   (gate: >= 80)  -> {rk5.loc[ASOF] >= 80}")
print(f"  GDX 5d ret = {100*g.pct_change(5).loc[ASOF]:+.2f}%")
print(f"  GDX close = {g.loc[ASOF]:.2f}")
d = load_prices(["GDX"])["GDX"]
atr = pd.Series(wilder_atr(d["High"], d["Low"], d["Close"], 14), index=d.index)
print(f"  GDX Wilder-14 ATR = {atr.iloc[-1]:.3f} ({100*atr.iloc[-1]/g.iloc[-1]:.2f}% of price)")

# cluster depth on the live trigger
m = (rk5 >= 80).fillna(False)
p = list(idx).index(ASOF)
dep = 0
while p - dep >= 0 and bool(m.iloc[p - dep]):
    dep += 1
print(f"  cluster depth (consecutive rank5>=80 sessions incl. today) = {dep}")

# the pitched cell, restated once from scratch as a final check
LAG, K = 1, 3
ev = load_events(["cpi"])["date"]
A = []
for dt in ev:
    q = int(np.searchsorted(idx.values, np.datetime64(dt)))
    if q - K < 0 or q >= len(idx):
        continue
    A.append(idx[q - K])
A = pd.DatetimeIndex(sorted(set(A)))
for H in (3, 5, 10):
    fw = fwd_lag(g, H, LAG)
    ok = fw.notna()
    T = pd.DatetimeIndex(A).intersection(idx[m.values & ok.values])
    e = declusters(T, H, idx[ok.values])
    v = fw.loc[e].values
    print(f"\nh={H}: N={len(v)} mean={100*v.mean():+.3f}% hit={100*(v>0).mean():.1f}% "
          f"t={v.mean()/(v.std(ddof=1)/np.sqrt(len(v))):.2f} "
          f"sign_p={sign_test(int((v>0).sum()), len(v)):.4f} "
          f"worst={100*v.min():+.2f}% vs GDX drift {100*fw[ok].mean():+.3f}%")
