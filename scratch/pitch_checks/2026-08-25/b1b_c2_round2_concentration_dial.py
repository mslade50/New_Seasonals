"""C2 round 2 - the only cell that looked alive in round 1 was the h=3 gated
form (+1.021% over 20 episodes, 20.4x cost). This is the mandatory round-2
pass on it: concentration / drop-top-k, definition neighbours on BOTH the
lookback and the threshold, era + midterm + fragility-dial splits.

Dial vintage note: data/rd2_fragility.parquet is APPEND-ONLY point-in-time
only from 2026-07-02; 2016..2026-07-02 rows are a recompute vintage. This
script uses the recompute vintage and says so, because 19 of the cell's 20
episodes predate 2016 entirely and have NO dial at all.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change
import numpy as np
import pandas as pd

px = close_panel(["SMH", "SPY"])
idx = px.index
pos = pd.Series(range(len(idx)), index=idx)
EARN = pd.read_parquet(ROOT / "data" / "earnings_calendar.parquet")
EARN["date"] = pd.to_datetime(EARN["date"])
nv = EARN[EARN["ticker"] == "NVDA"]["date"].sort_values().unique()
prints = pd.DatetimeIndex(sorted({idx[idx.searchsorted(pd.Timestamp(x))]
                                  for x in nv if idx.searchsorted(pd.Timestamp(x)) < len(idx)}))
prints = prints[prints <= idx[-1]]
A_all = pd.DatetimeIndex(sorted({idx[pos[p] - 2] for p in prints if pos[p] >= 2}))

DIAL = pd.read_parquet(ROOT / "data" / "rd2_fragility.parquet")
ma10 = DIAL["63d"].rolling(10).mean()
print(f"dial ma10(63d) today = {ma10.iloc[-1]:.1f}  (recompute vintage before 2026-07-02)")

LEGS = [("SMH", 1.0)]


def gate_pit(lb, thr):
    r = _valid_pct_change(px["SMH"], lb) - _valid_pct_change(px["SPY"], lb)
    p = rolling_on_valid(r, lambda x: x.rolling(252).rank(pct=True) * 100.0)
    return p <= thr, p


print("\n=== 1. CONCENTRATION / drop-top-k on the h=3 gated cell (PIT<=25) ===")
for h in (1, 3, 5):
    ret = vehicle_ret(px, LEGS, h, 1)
    g, _ = gate_pit(63, 25)
    d = A_all.intersection(idx[g.reindex(idx, fill_value=False).values]).intersection(ret.dropna().index)
    v = ret.loc[d].values
    order = np.argsort(-v)
    print(f"  h={h}  N={len(v)}  full {100*v.mean():+.3f}%  "
          f"drop-top1 {100*np.delete(v, order[:1]).mean():+.3f}%  "
          f"drop-top2 {100*np.delete(v, order[:2]).mean():+.3f}%  "
          f"drop-top3 {100*np.delete(v, order[:3]).mean():+.3f}%  "
          f"median {100*np.median(v):+.3f}%  record {int((v>0).sum())}-{int((v<=0).sum())}")
    yr = pd.Series(v, index=d).groupby(d.year).sum()
    print(f"        by year (pp): " + ", ".join(f"{y}:{100*r:+.1f}" for y, r in yr.items()))

print("\n=== 2. DEFINITION NEIGHBOURS: lookback x threshold grid (h=3, episodes) ===")
rows = []
ret = vehicle_ret(px, LEGS, 3, 1)
base = 100 * ret.dropna().mean()
for lb in (21, 42, 63, 126):
    for thr in (5, 10, 20, 25, 50):
        g, _ = gate_pit(lb, thr)
        d = A_all.intersection(idx[g.reindex(idx, fill_value=False).values]).intersection(ret.dropna().index)
        if len(d) == 0:
            rows.append({"label": f"lb{lb} PIT<={thr}", "n": 0})
            continue
        v = ret.loc[d].values
        rows.append({"label": f"lb{lb} PIT<={thr}", "n": len(v),
                     "mean_pct": round(100 * v.mean(), 3),
                     "median_pct": round(100 * np.median(v), 3),
                     "hit": round(100 * (v > 0).mean(), 1),
                     "edge_pp": round(100 * v.mean() - base, 3),
                     "sign_p": round(sign_test(int((v > 0).sum()), len(v)), 3)})
show(rows, f"h=3 grid; ungated print cell = {100*ret.loc[A_all.intersection(ret.dropna().index)].mean():+.3f}%, all-days {base:+.3f}%")
mm = [r["mean_pct"] for r in rows if r.get("n")]
print(f"  grid spread: min {min(mm):+.3f}% max {max(mm):+.3f}% -> the cell is a dial, "
      f"and 20 cells were scanned to report the best of them")

print("\n=== 3. MIDTERM + DIAL + ERA splits on the pitched cell (PIT<=25, h=3) ===")
g, _ = gate_pit(63, 25)
d = A_all.intersection(idx[g.reindex(idx, fill_value=False).values]).intersection(ret.dropna().index)
v = ret.loc[d].values
mid = np.array([(x.year % 4) == 2 for x in d])
show([summarize(v[mid], f"midterm years (N={int(mid.sum())})  <-- 2026"),
      summarize(v[~mid], f"non-midterm (N={int((~mid).sum())})")], "cycle-year split")
dv = ma10.reindex(d)
have = dv.notna().values
print(f"  episodes with ANY dial reading: {int(have.sum())} of {len(d)} "
      f"(dial history starts 2016-07)")
if have.sum():
    hi = have & (dv.values >= 50)
    lo = have & (dv.values < 50)
    show([summarize(v[hi], f"dial>=50 (N={int(hi.sum())})"),
          summarize(v[lo], f"dial<50 (N={int(lo.sum())})")], "fragility split (recompute vintage)")
    print("  dial readings on those episodes: " +
          ", ".join(f"{str(x.date())}={dv[x]:.0f}" for x in d[have]))
print("  TODAY the dial is 89.5, the top of the 2016+ series; the cell has "
      f"{int((dv.dropna() >= 70).sum())} precedent(s) above 70.")

print("\n=== 4. the live slice one more time: August prints, 2020+ ===")
for h in (1, 2, 3, 5):
    ret = vehicle_ret(px, LEGS, h, 1)
    dd = A_all.intersection(ret.dropna().index)
    mon = np.array([idx[pos[x] + 2].month for x in dd])
    m = (mon == 8) & (dd >= pd.Timestamp("2020-01-01"))
    vv = ret.loc[dd].values[m]
    print(f"  h={h}: N={len(vv)} mean {100*vv.mean():+.3f}% record "
          f"{int((vv>0).sum())}-{int((vv<=0).sum())} worst {100*vv.min():+.2f}% "
          f"best {100*vv.max():+.2f}%  dates "
          + ", ".join(str(x.date()) for x in dd[m]))
