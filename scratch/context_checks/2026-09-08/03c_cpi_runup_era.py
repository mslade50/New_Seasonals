"""The half of 03/03b that survived: the three-session run-up into CPI, split by
whether the print lands on a Friday.

k3 anchor (which is TODAY for Friday's print) -> h=3 is the print session close.
  non-Friday CPI: n=246, 154-92 up (62.6%), sign p 0.00003
  Friday CPI:     n= 73,  39-34 up (53.4%), sign p 0.32
Before publishing: is the non-Friday drift alive after 2018, and what is the
unconditional 3-session base rate it has to beat?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (close_panel, load_events, fwd_ret, anchor_positions,
                       summarize, era_split, sign_test, cluster_note, show)

px = close_panel(["^GSPC", "SPY"])
cpi = load_events(["cpi"])
cpi = cpi[cpi["date"] <= "2026-09-11"].copy()
cpi["dow"] = cpi["date"].dt.day_name()
fri = pd.DatetimeIndex(cpi[cpi["dow"] == "Friday"]["date"])
non = pd.DatetimeIndex(cpi[cpi["dow"] != "Friday"]["date"])

f3 = fwd_ret(px["^GSPC"], 3)
valid = f3.dropna().index

def line(v, lab):
    v = np.asarray(v, float); v = v[~np.isnan(v)]
    if len(v) == 0:
        print(f"  {lab:38s} (empty)"); return
    up = int((v > 0).sum()); n = len(v)
    s = summarize(v, lab)
    print(f"  {lab:38s} n={n:5d} mean={s['mean_pct']:+.3f}% med={s['median_pct']:+.3f}% "
          f"hit={s['hit']:5.1f}% t={s['t']:+.2f} rec {up}-{n-up} p={sign_test(up, n):.5f}")

def anchors_for(dd, k=3):
    pos, _ = anchor_positions(px.index, dd, offset=-k)
    return px.index[pos]

a_non = pd.DatetimeIndex(anchors_for(non)).intersection(valid)
a_fri = pd.DatetimeIndex(anchors_for(fri)).intersection(valid)

print("BASE RATE the drift has to beat: unconditional 3-session forward return")
line(f3.loc[valid].values, "all sessions h3")
print("  (that is the number 62.6% is measured against)")

print("\nTHE TWO CELLS, k3 anchor -> h3 = the print session close")
line(f3.loc[a_non].values, "non-Friday CPI")
line(f3.loc[a_fri].values, "Friday CPI")

print("\nERA STABILITY of the non-Friday run-up")
v = f3.loc[a_non].values
show(era_split(a_non, v), "non-Friday CPI k3->h3")
for lab, lo, hi in [("2000-2012", "2000-01-01", "2013-01-01"),
                    ("2013-2017", "2013-01-01", "2018-01-01"),
                    ("2018+",     "2018-01-01", "2027-01-01"),
                    ("2022+",     "2022-01-01", "2027-01-01")]:
    m = (a_non >= lo) & (a_non < hi)
    if m.sum():
        line(v[m], f"non-Friday CPI {lab}")
print("  " + cluster_note(a_non, v, k=2))

print("\nERA STABILITY of the Friday run-up (the live cell)")
vf = f3.loc[a_fri].values
show(era_split(a_fri, vf), "Friday CPI k3->h3")
for lab, lo, hi in [("2000-2012", "2000-01-01", "2013-01-01"),
                    ("2013-2017", "2013-01-01", "2018-01-01"),
                    ("2018+",     "2018-01-01", "2027-01-01")]:
    m = (a_fri >= lo) & (a_fri < hi)
    if m.sum():
        line(vf[m], f"Friday CPI {lab}")

print("\nIs the gap between the two just the print session, or the whole run-up?")
print("Decompose: the two sessions BEFORE the print (k3 -> h2), then the print day alone.")
f2 = fwd_ret(px["^GSPC"], 2)
line(f2.loc[pd.DatetimeIndex(a_non).intersection(f2.dropna().index)].values,
     "non-Friday, 2 sessions pre-print")
line(f2.loc[pd.DatetimeIndex(a_fri).intersection(f2.dropna().index)].values,
     "Friday, 2 sessions pre-print")
r1 = px["^GSPC"].pct_change(fill_method=None)
line(r1.loc[r1.dropna().index.intersection(non)].values, "non-Friday, print session only")
line(r1.loc[r1.dropna().index.intersection(fri)].values, "Friday, print session only")

print("\nSame decomposition restricted to 2018+ (the era that matters)")
for lab, dd in [("non-Friday", non), ("Friday", fri)]:
    d = r1.dropna().index.intersection(dd)
    d = d[d >= "2018-01-01"]
    line(r1.loc[d].values, f"{lab} print session, 2018+")
    a = pd.DatetimeIndex(anchors_for(dd)).intersection(f2.dropna().index)
    a = a[a >= "2018-01-01"]
    line(f2.loc[a].values, f"{lab} 2 sessions pre-print, 2018+")

print("\nMIDTERM-YEAR September CPI, for the record (small N by construction)")
sep = cpi[(cpi["date"].dt.month == 9) & (cpi["date"].dt.year % 4 == 2)]
print("  ", [str(d.date()) for d in sep["date"]])
d = r1.dropna().index.intersection(pd.DatetimeIndex(sep["date"]))
line(r1.loc[d].values, "Sept CPI in midterm years, print session")
