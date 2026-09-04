"""The sweep's only BH-passing opex cell: natural gas on the session two trading
days before a monthly equity opex, 129-181 down, sign p 0.0023. Ask whether the
cell is opex or just the third week of the month, and whether it is one seasonal
month in disguise."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, cluster_note, era_split, fwd_ret, sign_test, summarize,
)

px = close_panel(["NG=F", "CL=F", "SPY"]).dropna(subset=["NG=F"])
ng = px["NG=F"]
idx = ng.index
r1 = fwd_ret(ng, 1)

# monthly opex = third Friday of each month, mapped to the nearest session
opex = []
for (y, m), grp in ng.groupby([idx.year, idx.month]):
    fri = [d for d in grp.index if d.weekday() == 4]
    if len(fri) >= 3:
        opex.append(fri[2])
opex = pd.DatetimeIndex(sorted(opex))
pos = {d: i for i, d in enumerate(idx)}
anch = pd.DatetimeIndex([idx[pos[d] - 2] for d in opex if pos[d] >= 2])
anch = anch[anch < idx[-1]]
print("NG=F", idx[0].date(), "->", idx[-1].date(), "n", len(idx))
print("opex months", len(opex), "| k2 anchors", len(anch))
print("today is the k2 anchor for opex", opex[-1].date() if opex[-1] > idx[-1]
      else "2026-08-21 (next third Friday)")


def line(lab, dates, series=r1):
    f = series.reindex(pd.DatetimeIndex(dates)).dropna()
    if len(f) == 0:
        print(f"  {lab:<38} n=0")
        return None
    s = summarize(f.values)
    nup = int((f > 0).sum())
    print(f"  {lab:<38} n={s['n']:<4} mean={s['mean_pct']:+.3f}%  "
          f"med={s['median_pct']:+.3f}%  {nup}-{len(f)-nup} up  t={s['t']:+.2f}  "
          f"sign_p={sign_test(len(f)-nup, len(f)):.4f} (down)")
    return f


print("\n=== the cell and its neighbours ===")
f_cell = line("k2 before opex (tomorrow's analogue)", anch)
line("all NG sessions", idx[:-1])
for k in (1, 3, 4, 5):
    a = pd.DatetimeIndex([idx[pos[d] - k] for d in opex if pos[d] >= k])
    line(f"k{k} before opex", a[a < idx[-1]])
line("opex session itself", pd.DatetimeIndex([d for d in opex if d < idx[-1]]))

print("\n=== is it opex, or the third week? ===")
# every Wednesday, every Wednesday in the third week, every other Wednesday
wed = idx[(idx.weekday == 2)]
wed = wed[wed < idx[-1]]
third_wk = pd.DatetimeIndex([d for d in wed if 15 <= d.day <= 21])
other_wed = pd.DatetimeIndex([d for d in wed if d not in set(third_wk)])
line("all Wednesdays", wed)
line("Wednesdays, day 15-21 of month", third_wk)
line("Wednesdays, rest of month", other_wed)
non_anch_third = pd.DatetimeIndex([d for d in third_wk if d not in set(anch)])
line("third-week Wed that are NOT the k2", non_anch_third)

print("\n=== era and concentration on the cell ===")
print("  era:", [(e["label"], e["n"], round(e["mean_pct"], 3))
                 for e in era_split(f_cell.index, f_cell.values)])
print(" ", cluster_note(f_cell.index, f_cell.values))
by_month = {}
for d, v in f_cell.items():
    by_month.setdefault(d.month, []).append(v)
print("  by calendar month (mean %, n, down-up):")
for m in sorted(by_month):
    v = np.array(by_month[m])
    print(f"    {m:>2}  {v.mean()*100:+7.3f}%  n={len(v):<3} "
          f"{int((v<0).sum())}-{int((v>0).sum())} down")
aug = pd.DatetimeIndex([d for d in f_cell.index if d.month == 8])
line("August only", aug)
non_aug = pd.DatetimeIndex([d for d in f_cell.index if d.month != 8])
line("every month except August", non_aug)

print("\n=== by year, the cell ===")
yr = {}
for d, v in f_cell.items():
    yr.setdefault(d.year, []).append(v)
neg = sum(1 for y in yr if np.mean(yr[y]) < 0)
print(f"  negative in {neg} of {len(yr)} years")

print("\n=== crude, the same cell, as a sanity control ===")
cl = px["CL=F"].dropna()
r1c = fwd_ret(cl, 1)
line("CL=F k2 before opex", anch, r1c)
line("CL=F all sessions", cl.index[:-1], r1c)
