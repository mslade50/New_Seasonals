"""September Tuesdays: the VIX cell, and the copper cell that cleared BH.

Two E:weekday_month members are live for tomorrow (Tuesday, September):

  ^VIX  n=110, +1.797%, 66-44 up, t 2.32, era-stable, BH FAIL.
  HG=F  n=110, -0.101%, 41-67 down, sign p 0.0139, era-stable, BH PASS. The
        record is significant while the mean is flat, which is its own fact.

The VIX cell is on probation before it starts: last night's brief published
"August Mondays off a vol floor", so a weekday-by-month VIX cell two nights
running is the same trick twice unless the September arm says something the
August arm did not. The decisive test is whether this is TUESDAY or whether it
is simply SEPTEMBER, which is the month everyone already knows about.

Convention: lag=0 close-to-close, h=1 is 2026-09-01.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import (close_panel, cluster_note, era_split, fwd_ret,  # noqa: E402
                       show, sign_test, summarize)

px = close_panel(["^VIX", "HG=F", "SPY"])
px = px[px.index >= "1999-01-01"]


def cell(series, mask, label):
    f = fwd_ret(series, 1).dropna()
    d = f.index[mask.reindex(f.index).fillna(False).values]
    v = f.loc[d].values
    r = summarize(v, label)
    if r.get("n"):
        up = int((v > 0).sum())
        r["record"] = f"{up}-{len(v)-up}"
        r["sign_p"] = round(sign_test(up, len(v)), 4)
    return r, pd.DatetimeIndex(d), v


for tkr in ["^VIX", "HG=F"]:
    s = px[tkr].dropna()
    idx = s.index
    wd = pd.Series(idx.weekday, index=idx)
    mo = pd.Series(idx.month, index=idx)
    # the anchor is the session BEFORE the Tuesday, so h=1 is the Tuesday
    nxt_wd = wd.shift(-1)
    nxt_mo = mo.shift(-1)
    tue_sep = (nxt_wd == 1) & (nxt_mo == 9)
    tue_oth = (nxt_wd == 1) & (nxt_mo != 9)
    sep_any = (nxt_mo == 9)
    sep_nontue = (nxt_mo == 9) & (nxt_wd != 1)

    rows = []
    for label, m in [("-> Tuesday in September", tue_sep),
                     ("-> Tuesday, other 11 months", tue_oth),
                     ("-> any September session", sep_any),
                     ("-> September, NOT Tuesday", sep_nontue),
                     ("-> every session", pd.Series(True, index=idx))]:
        r, _, _ = cell(s, m, label)
        rows.append(r)
    show(rows, f"{tkr} h=1: is it Tuesday, or is it September?")

    r, d, v = cell(s, tue_sep, "tue_sep")
    print(f"  era split, {tkr} -> September Tuesdays:")
    show(era_split(d, v))
    print("  ", cluster_note(d, v, k=2))
    print()

print("=== the September-Tuesday VIX cell vs last night's August-Monday cell ===")
s = px["^VIX"].dropna()
idx = s.index
wd = pd.Series(idx.weekday, index=idx)
mo = pd.Series(idx.month, index=idx)
nxt_wd, nxt_mo = wd.shift(-1), mo.shift(-1)
lo3 = s <= s.rolling(252, min_periods=252).min() * 1.0 + (
    s.rolling(252, min_periods=252).max() - s.rolling(252, min_periods=252).min()) / 3.0
print(f"  VIX today {s.iloc[-1]:.2f}; in the bottom third of its 52w range: {bool(lo3.iloc[-1])}")
rows = []
for label, m in [("-> Sep Tuesday, ANY vol level", (nxt_wd == 1) & (nxt_mo == 9)),
                 ("-> Sep Tuesday, from bottom third", (nxt_wd == 1) & (nxt_mo == 9) & lo3),
                 ("-> Aug Monday, from bottom third (last night)",
                  (nxt_wd == 0) & (nxt_mo == 8) & lo3)]:
    r, _, _ = cell(s, m, label)
    rows.append(r)
show(rows, "^VIX h=1")

print("\n=== copper: today's state ===")
h = px["HG=F"].dropna()
print(f"  HG=F close {h.iloc[-1]:.4f}, 1d {100*(h.iloc[-1]/h.iloc[-2]-1):+.2f}%, "
      f"21d {100*(h.iloc[-1]/h.iloc[-22]-1):+.2f}%")
f = fwd_ret(h, 1).dropna()
idxh = h.index
nxt_wdh = pd.Series(idxh.weekday, index=idxh).shift(-1)
nxt_moh = pd.Series(idxh.month, index=idxh).shift(-1)
m = ((nxt_wdh == 1) & (nxt_moh == 9)).reindex(f.index).fillna(False)
v = f.loc[f.index[m.values]].values
print(f"  September Tuesdays: n={len(v)}, {int((v>0).sum())}-{int((v<0).sum())}, "
      f"mean {100*v.mean():+.3f}%, median {100*np.median(v):+.3f}%, "
      f"best {100*v.max():+.2f}%, worst {100*v.min():+.2f}%")
print(f"  share of the total that sits in the 3 best days: "
      f"{100*np.sort(v)[-3:].sum()/abs(v.sum()) if v.sum() else float('nan'):.0f}%")
