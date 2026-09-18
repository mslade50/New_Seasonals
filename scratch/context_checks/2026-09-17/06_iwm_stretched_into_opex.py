"""IWM goes into September witching already beaten up.

IWM 21d return -4.68%, trailing-252 rank 7.9, 63d rank 4.8, and it added only
+0.53% today against QQQ's +1.73%. The post-September-witching week is the
worst equity cell on the calendar for small caps (02: 6-20 up, -1.79%). Does
arriving oversold make it worse, or is an oversold small-cap tape the one that
bounces?
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (  # noqa: E402
    close_panel, load_events, summarize, show, sign_test, pct_rank, cluster_note,
)

px = close_panel(["IWM", "SPY", "QQQ"])
idx = px.index
iwm, spy = px["IWM"].dropna(), px["SPY"].dropna()

r21 = iwm / iwm.shift(21) - 1.0
rank21 = pct_rank(r21, 21, 252)          # trailing-252 percentile of the 21d return
print(f"IWM 21d return today {100 * r21.iloc[-1]:+.2f}%, "
      f"trailing-252 rank {rank21.iloc[-1]:.1f}")

qw = load_events(["quad_witching"])["date"]
qw = pd.DatetimeIndex([d for d in qw if d in set(idx)])
sep = pd.DatetimeIndex([d for d in qw if d.month == 9])

f5 = iwm.shift(-5) / iwm - 1.0
f5s = spy.shift(-5) / spy - 1.0
sep_live = sep.intersection(f5.dropna().index).intersection(rank21.dropna().index)
print(f"September witchings with a usable 21d rank: {len(sep_live)} "
      f"({sep_live[0].year}..{sep_live[-1].year})")

rk = rank21.reindex(sep_live)
tbl = pd.DataFrame({"year": sep_live.year,
                    "iwm_rank21": rk.round(1).values,
                    "iwm_r21_pct": (100 * r21.reindex(sep_live)).round(2).values,
                    "iwm_next5_pct": (100 * f5.loc[sep_live]).round(2).values,
                    "spy_next5_pct": (100 * f5s.loc[sep_live]).round(2).values})
tbl["iwm_minus_spy"] = (tbl.iwm_next5_pct - tbl.spy_next5_pct).round(2)
print(tbl.to_string(index=False))

rows = []
for lab, m in [("rank21 <= 25 (beaten up)", rk <= 25),
               ("rank21 25-75", (rk > 25) & (rk <= 75)),
               ("rank21 > 75 (strong)", rk > 75)]:
    d = sep_live[m.values]
    r = summarize(f5.loc[d].values, lab)
    if r["n"]:
        k = int((f5.loc[d] > 0).sum())
        r["record"] = f"{k}-{r['n'] - k} up"
        r["sign_p_down"] = round(sign_test(r["n"] - k, r["n"]), 4)
    rows.append(r)
rows.append(summarize(f5.loc[sep_live].values, "all September witchings"))
show(rows, "IWM, 5 sessions after September witching, by how stretched it arrived")

print("\n=== the beaten-up cell, year by year ===")
d = sep_live[(rk <= 25).values]
print(tbl[tbl.year.isin(d.year)].to_string(index=False))
v = f5.loc[d]
k = int((v > 0).sum())
print(f"  record {k}-{len(v) - k} up, mean {100 * v.mean():+.2f}%, "
      f"median {100 * v.median():+.2f}%, worst {100 * v.min():+.2f}%, "
      f"best {100 * v.max():+.2f}%, sign p(down) = {sign_test(len(v) - k, len(v)):.4f}")
print(f"  concentration: {cluster_note(d, v.values, k=2)}")
sp = (f5.loc[d] - f5s.loc[d])
k2 = int((sp > 0).sum())
print(f"  IWM vs SPY: beat it {k2} of {len(sp)}, mean {100 * sp.mean():+.2f}%")

print("\n=== control: does a low 21d rank predict a bad 5 sessions on ANY date? ===")
allrk = rank21.dropna()
f = f5.dropna()
common = allrk.index.intersection(f.index)
rows = []
for lab, m in [("rank21 <= 25", allrk.reindex(common) <= 25),
               ("rank21 > 25", allrk.reindex(common) > 25)]:
    dd = common[m.values]
    r = summarize(f.loc[dd].values, lab)
    k = int((f.loc[dd] > 0).sum())
    r["record"] = f"{k}-{r['n'] - k} up"
    rows.append(r)
rows.append(summarize(f.loc[common].values, "CTL every session"))
show(rows, "IWM 5-session forward return by 21d rank, ALL dates")
