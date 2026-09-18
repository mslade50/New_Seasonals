"""Friday CPI, the controls that decide whether it survives.

From 03: ^GSPC on a Friday CPI print session -0.273% / 49.3% hit (n=71) against
+0.108% / 58.2% (n=244) on every other CPI. Before that is a sentence it has to
beat THREE controls:
  1. all Fridays (maybe Fridays are just worse)
  2. era stability at 2018 (Friday CPIs are heavily pre-2013 by BLS scheduling)
  3. concentration (2008-2009 sits inside this sample)
And the live question is the RUN-UP, since tomorrow is the k2 session, not the print.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (close_panel, load_events, fwd_ret, anchor_positions,
                       summarize, era_split, sign_test, cluster_note, show)

px = close_panel(["^GSPC", "SPY", "^VIX", "TLT"])
cpi = load_events(["cpi"])
cpi = cpi[cpi["date"] <= "2026-09-11"].copy()
cpi["dow"] = cpi["date"].dt.day_name()
fri = pd.DatetimeIndex(cpi[cpi["dow"] == "Friday"]["date"])
non = pd.DatetimeIndex(cpi[cpi["dow"] != "Friday"]["date"])

r1 = px["^GSPC"].pct_change()
valid = r1.dropna().index

def line(v, lab):
    up = int((v > 0).sum())
    s = summarize(v, lab)
    if not s["n"]:
        return
    print(f"  {lab:34s} n={s['n']:5d} mean={s['mean_pct']:+.3f}% "
          f"med={s['median_pct']:+.3f}% hit={s['hit']:5.1f}% t={s['t']:+.2f} "
          f"rec {up}-{s['n']-up} sign_p={sign_test(up, s['n']):.4f}")

print("CONTROL 1: is it just Fridays?")
allfri = valid[valid.dayofweek == 4]
line(r1.loc[valid.intersection(fri)].values, "CPI on a Friday")
line(r1.loc[allfri.difference(fri)].values, "every OTHER Friday")
line(r1.loc[valid.intersection(non)].values, "CPI not on a Friday")
line(r1.loc[valid].values, "all sessions")

print("\nCONTROL 2: era stability of the Friday-CPI print session")
d = valid.intersection(fri)
v = r1.loc[d].values
show(era_split(d, v), "^GSPC Friday CPI print session")
for lab, lo, hi in [("2000-2012", "2000-01-01", "2013-01-01"),
                    ("2013-2017", "2013-01-01", "2018-01-01"),
                    ("2018+", "2018-01-01", "2027-01-01")]:
    m = (d >= lo) & (d < hi)
    if m.sum():
        line(v[m], f"Friday CPI {lab}")

print("\nCONTROL 3: concentration")
print("  " + cluster_note(d, v, k=2))
print("  " + cluster_note(d, v, k=4))
noGFC = d[~((d >= "2008-06-01") & (d < "2009-07-01"))]
line(r1.loc[noGFC].values, "Friday CPI ex Jun08-Jun09")

print("\n" + "=" * 74)
print("THE LIVE QUESTION: the run-up. Anchor = k td before the print.")
print("Today is k3. Tomorrow is the k2 session, Thursday is k1 (PPI), Friday is the print.")
print("=" * 74)
for k in (3, 2, 1):
    print(f"\n-- anchor k={k} --")
    for lab, dd in [("Friday CPI", fri), ("non-Friday CPI", non)]:
        pos, _ = anchor_positions(px.index, dd, offset=-k)
        anch = px.index[pos]
        for h in (1, k):
            f = fwd_ret(px["^GSPC"], h)
            a = pd.DatetimeIndex(anch).intersection(f.dropna().index)
            line(f.loc[a].values, f"{lab} k{k} -> h={h}")

print("\n" + "=" * 74)
print("Cumulative path from TODAY'S anchor (k3) for the Friday-CPI cell")
print("=" * 74)
pos, _ = anchor_positions(px.index, fri, offset=-3)
anch = px.index[pos]
for h in (1, 2, 3, 4, 5):
    f = fwd_ret(px["^GSPC"], h)
    a = pd.DatetimeIndex(anch).intersection(f.dropna().index)
    line(f.loc[a].values, f"h={h} (h3 = the print session)")
print("  -- same horizons, non-Friday CPI, for contrast --")
pos2, _ = anchor_positions(px.index, non, offset=-3)
anch2 = px.index[pos2]
for h in (1, 2, 3, 4, 5):
    f = fwd_ret(px["^GSPC"], h)
    a = pd.DatetimeIndex(anch2).intersection(f.dropna().index)
    line(f.loc[a].values, f"h={h} non-Friday")
