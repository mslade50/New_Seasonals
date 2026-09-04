"""Era stability and concentration for the surviving arm of drill 05.

05 found that the ABSOLUTE turn-of-month bid dies in September for EEM exactly
as it does for SPY (24-22, +0.057%), but the RELATIVE leg, EEM minus SPY on the
same anchors, does not: 329-229 across all turns at t 3.55, and 16-7 on the 23
occasions the anchor was a month-end whose next session opened September.

A relative cell needs the same honesty checks as an absolute one before it can
carry a tag: era split at 2018, concentration in the top two episodes, and the
non-turn control so the claim is an EDGE and not just EM beta.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, cluster_note, era_split, fwd_ret,  # noqa: E402
                       show, sign_test, summarize)

px = close_panel(["EEM", "SPY"]).dropna()
px = px[px.index >= "2003-05-01"]
idx = px.index
month = pd.Series(idx.month, index=idx)
is_me = (month != month.shift(-1)).values
pos = np.arange(len(idx))
me_pos = pos[is_me][:-1]
anchor_pos = np.unique(np.concatenate([me_pos, me_pos + 1]))
anchor_pos = anchor_pos[anchor_pos < len(idx) - 1]
anchors = idx[anchor_pos]

rel = fwd_ret(px["EEM"], 1) - fwd_ret(px["SPY"], 1)
rel = rel.dropna()
non = idx.difference(anchors)

print("=== control: turn days vs every other day ===")
rows = []
for label, sel in (("turn-of-month anchors", anchors), ("all other sessions", non)):
    d = pd.DatetimeIndex(sel).intersection(rel.index)
    v = rel.loc[d].values
    r = summarize(v, label)
    r["record"] = f"{int((v>0).sum())}-{int((v<0).sum())}"
    r["sign_p"] = round(sign_test(int((v > 0).sum()), len(v)), 6)
    rows.append(r)
show(rows, "EEM minus SPY, h=1, percentage points")

print("\n=== era split, all turn anchors ===")
d = pd.DatetimeIndex(anchors).intersection(rel.index)
show(era_split(d, rel.loc[d].values))
for label, m in (("pre-2018", d < pd.Timestamp("2018-01-01")),
                 ("2018+", d >= pd.Timestamp("2018-01-01"))):
    v = rel.loc[d[m]].values
    up = int((v > 0).sum())
    print(f"  {label}: {up}-{len(v)-up}, sign_p {sign_test(up, len(v)):.5f}")
print(" ", cluster_note(d, rel.loc[d].values, k=2))

print("\n=== per-year record, so a single decade cannot be carrying it ===")
s = pd.Series(rel.loc[d].values, index=d)
by = s.groupby(s.index.year).agg(n="size", up=lambda x: int((x > 0).sum()),
                                 mean=lambda x: 100 * x.mean())
by["mean"] = by["mean"].round(3)
print(by.to_string())
print(f"\n  years with a positive mean: {int((by['mean'] > 0).sum())} of {len(by)}")

print("\n=== the September slot only (anchor = a month-end, next session in Sep) ===")
nxt = {idx[p]: idx[p + 1] for p in anchor_pos}
sep_first = pd.DatetimeIndex([a for a in anchors if nxt[a].month == 9 and a.month == 8])
d2 = sep_first.intersection(rel.index)
v = rel.loc[d2].values
up = int((v > 0).sum())
print(f"  n={len(v)} {up}-{len(v)-up}, mean {100*v.mean():+.3f}pp, "
      f"median {100*np.median(v):+.3f}pp, sign_p {sign_test(up, len(v)):.4f}")
show(era_split(d2, v), "Sep slot")
print(" ", cluster_note(d2, v, k=2))
