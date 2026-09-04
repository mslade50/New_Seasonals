"""The strongest cell in tonight's sweep, and the one last night did not spend.

E:turn_of_month|EEM: n=560, +0.288%, 341-215 up, t 4.02, era-stable, BH pass.
Last night's brief spent the SPY/^GSPC version of turn-of-month (as the
September arm, which is negative), so the equity index turn is off the table.
EEM has never been published here.

What has to be established:
  1. Does it survive the SEPTEMBER arm, which is where the SPY version died?
  2. Is it EEM specifically, or is EEM just a higher-beta wrapper on the same
     SPY turn? The test is the SPY-relative leg.
  3. Controls, eras, concentration.

Convention: lag=0 close-to-close from the anchor close (tonight's close), so
h=1 is 2026-09-01.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, cluster_note, era_split, fwd_ret,  # noqa: E402
                       show, sign_test, summarize)

px = close_panel(["EEM", "SPY", "^GSPC", "HYG"])
px = px[px.index >= "2003-05-01"]  # EEM inception 2003-04
sub = px[["EEM", "SPY"]].dropna()
idx = sub.index

month = pd.Series(idx.month, index=idx)
is_month_end = (month != month.shift(-1)).values
pos = np.arange(len(idx))
# turn-of-month anchor set, matching the engine: the next session is one of
# the month's first two. That is the month-end bar and the first bar after it.
me_pos = pos[is_month_end][:-1]
anchor_pos = np.unique(np.concatenate([me_pos, me_pos + 1]))
anchor_pos = anchor_pos[anchor_pos < len(idx) - 1]
anchors = idx[anchor_pos]
print(f"turn-of-month anchors: {len(anchors)} since {idx[0].date()}")

f1e = fwd_ret(sub["EEM"], 1)
f1s = fwd_ret(sub["SPY"], 1)
rel = f1e - f1s

sep_anchors = anchors[[(d.month == 8 and i in me_pos) or (d.month == 9)
                       for d, i in zip(anchors, anchor_pos)]]
# simpler and unambiguous: the anchor whose NEXT session falls in September
nxt = {idx[p]: idx[p + 1] for p in anchor_pos}
sep_anchors = pd.DatetimeIndex([d for d in anchors if nxt[d].month == 9])
oth_anchors = anchors.difference(sep_anchors)
# the exact slot tomorrow occupies: month-end bar, next session is Sep 1
sep_first = pd.DatetimeIndex([d for d in sep_anchors if d.month == 8])

rows = []
for label, sel, series in [
    ("EEM all turns", anchors, f1e),
    ("EEM September turns", sep_anchors, f1e),
    ("EEM Sep FIRST session only", sep_first, f1e),
    ("EEM other 11 months", oth_anchors, f1e),
    ("EEM all non-turn days", idx.difference(anchors), f1e),
]:
    d = pd.DatetimeIndex(sel).intersection(series.dropna().index)
    v = series.loc[d].values
    r = summarize(v, label)
    r["record"] = f"{int((v>0).sum())}-{int((v<0).sum())}"
    r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(r)
show(rows, "EEM h=1")

rows = []
for label, sel in [("all turns", anchors), ("September turns", sep_anchors),
                   ("Sep FIRST session only", sep_first),
                   ("other 11 months", oth_anchors)]:
    d = pd.DatetimeIndex(sel).intersection(rel.dropna().index)
    v = rel.loc[d].values
    r = summarize(v, label)
    r["record"] = f"{int((v>0).sum())}-{int((v<0).sum())}"
    r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 4)
    rows.append(r)
show(rows, "EEM minus SPY h=1 (is it EM, or is it beta?)")

print("\n=== SPY on the same anchors, for the comparison ===")
rows = []
for label, sel in [("all turns", anchors), ("September turns", sep_anchors),
                   ("Sep FIRST session only", sep_first)]:
    d = pd.DatetimeIndex(sel).intersection(f1s.dropna().index)
    v = f1s.loc[d].values
    r = summarize(v, label)
    r["record"] = f"{int((v>0).sum())}-{int((v<0).sum())}"
    rows.append(r)
show(rows, "SPY h=1")

print("\n=== era split, EEM all turns and September turns ===")
for label, sel in [("all turns", anchors), ("September turns", sep_anchors)]:
    d = pd.DatetimeIndex(sel).intersection(f1e.dropna().index)
    show(era_split(d, f1e.loc[d].values), f"EEM {label}")
    print("  ", cluster_note(d, f1e.loc[d].values, k=2))

print("\n=== the Sep-first-session slot, year by year (EEM, then EEM-SPY) ===")
for d in sep_first:
    if d in f1e.index and np.isfinite(f1e.loc[d]):
        print(f"  {d.date()} -> {nxt[d].date()}  EEM {100*f1e.loc[d]:+6.2f}%  "
              f"vs SPY {100*rel.loc[d]:+6.2f}pp")
