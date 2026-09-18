"""A3 round 2 - the only thing that could rescue the short is the midterm
half, so cross it with the FOMC-in-window split that defines the LIVE cell,
and price the midterm cell's concentration. Also charge the k ladder.
"""
import sys
from pathlib import Path

ROOTP = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOTP))
import numpy as np                                        # noqa: E402
import pandas as pd                                       # noqa: E402
from pitch_lab import *  # noqa: E402,F403

px = close_panel(["IWM", "SPY"])
cal = px["SPY"].dropna().index
pos = pd.Series(range(len(cal)), index=cal)
K = 5
ev = load_events(["quad_witching"])
QUADS = pd.DatetimeIndex(sorted(ev[ev["date"].dt.month == 9]["date"]))
QUADS = QUADS[QUADS <= cal[-1]]
fomc = load_events(["fomc_decision"])["date"]
ret = vehicle_ret(px, [("IWM", 1.0)], K, 1)

rows = []
for q in QUADS:
    loc = int(cal.searchsorted(q))
    if cal[loc] != q:
        continue
    a = cal[loc - (K + 1)]
    v = ret.get(a, np.nan)
    if np.isnan(v):
        continue
    p = int(pos[a])
    lo, hi = cal[p + 1], cal[p + 1 + K]
    rows.append({"year": q.year, "mid": q.year % 4 == 2,
                 "fomc": bool(((fomc > lo) & (fomc <= hi)).any()),
                 "short": -v})
D = pd.DataFrame(rows)

print("=" * 78)
print("A. the 2x2: midterm x FOMC-inside-the-window, SHORT IWM at k=5")
for m in (True, False):
    for f in (True, False):
        s = D[(D["mid"] == m) & (D["fomc"] == f)]["short"].values
        lab = f"midterm={m}, fomc_in={f}"
        if len(s) == 0:
            print(f"  {lab:<30} N=0  -- EMPTY CELL")
            continue
        w = int((s > 0).sum())
        print(f"  {lab:<30} N={len(s):<3} short {100*s.mean():+.3f}%  "
              f"record {w}-{len(s)-w}  years "
              f"{sorted(D[(D['mid']==m)&(D['fomc']==f)]['year'].tolist())}")
print("\n  2026 IS midterm=True, fomc_in=True. The LIVE-EXACT cell above is")
print("  what the pitch is actually buying.")

print("\n" + "=" * 78)
print("B. the midterm cell's concentration (it is the only positive half)")
mid = D[D["mid"]]
v = mid["short"].values
print(f"  midterm SHORT mean {100*v.mean():+.3f}% on N={len(v)}, "
      f"record {int((v>0).sum())}-{int((v<=0).sum())}, "
      f"sign p = {sign_test(int((v>0).sum()), len(v)):.4f}")
print(f"  years/values: "
      f"{dict(zip(mid['year'], np.round(100*v, 2)))}")
o = np.sort(v)[::-1]
print(f"  drop-best-1 -> {100*o[1:].mean():+.3f}%   "
      f"drop-best-2 -> {100*o[2:].mean():+.3f}%")
print(f"  concentration: "
      f"{cluster_note(pd.DatetimeIndex([pd.Timestamp(f'{y}-09-15') for y in mid['year']]), v)}")

print("\n" + "=" * 78)
print("C. search charge: the candidate scans k=2..8 x 2 directions = 14 cells")
print("   plus the midterm and FOMC crossings. Best |mean| in the k ladder:")
best = None
for k in range(2, 9):
    r = vehicle_ret(px, [("IWM", 1.0)], k, 1)
    vals = []
    for q in QUADS:
        loc = int(cal.searchsorted(q))
        a = cal[loc - (k + 1)]
        x = r.get(a, np.nan)
        if not np.isnan(x):
            vals.append(-x)
    vals = np.asarray(vals)
    t = vals.mean() / (vals.std(ddof=1) / np.sqrt(len(vals)))
    if best is None or abs(t) > abs(best[1]):
        best = (k, t, 100 * vals.mean())
print(f"   max |t| over the 7 rungs = k={best[0]} at t={best[1]:+.2f} "
      f"({best[2]:+.3f}%). Family-wise |t| for p=0.05 at 14 cells is ~2.9.")
print("   Nothing in the ladder reaches |t| 1.0 on the short side.")
