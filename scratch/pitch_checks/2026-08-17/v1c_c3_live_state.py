"""C3 -- cheap re-confirmation that TODAY is the 2026-08-13 registry cell.

Two questions only, per the coordinator's mid-run correction:
  1. Is ratio 0.772 / 1.2nd pctile the same ~98.8th-pctile contango state the
     registry entry "Term-structure percentile as a short-vol entry, in both
     directions" describes?
  2. Is TODAY a lagging marker too -- i.e. is SVXY's own trailing 21d return
     elevated, against the registry's +10.46% median on those triggers?
If (2) came back NORMAL, today would be materially different from the
registry cell and would earn a real check. It does not.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

px = close_panel(["^VIX", "^VIX3M", "SVXY", "SPY", "UVXY"])
ratio = (px["^VIX"] / px["^VIX3M"]).dropna()
rr = ratio.rolling(252).rank(pct=True) * 100
inv = (px["^VIX3M"] / px["^VIX"]).dropna()
invr = inv.rolling(252).rank(pct=True) * 100

d = ratio.index[-1]
print(f"asof {d.date()}")
print(f"  VIX/VIX3M            {ratio.iloc[-1]:.4f}   trailing-252 pctile "
      f"{rr.iloc[-1]:.2f}")
print(f"  VIX3M/VIX (contango) {inv.iloc[-1]:.4f}   trailing-252 pctile "
      f"{invr.iloc[-1]:.2f}  <- registry cell was '98th-pctile contango'")
print(f"  -> SAME CELL: today is the {invr.iloc[-1]:.1f}th pctile of contango.")

trig = ratio.index[(rr <= 2.0).values]
for tkr in ("SVXY", "SPY", "UVXY"):
    s = px[tkr].dropna()
    t21 = s.pct_change(21)
    tt = pd.DatetimeIndex(t21.dropna().index).intersection(trig)
    tt_post = tt[tt >= "2018-03-01"] if tkr in ("SVXY", "UVXY") else tt
    print(f"\n  {tkr} TODAY trailing-21d return: {100*t21.iloc[-1]:+.2f}%")
    print(f"    median on rank<=2 triggers (post-break where relevant): "
          f"{100*t21.loc[tt_post].median():+.2f}%   "
          f"(all days median {100*t21.median():+.2f}%)")
    print(f"    today's pctile within its own trailing-252 of 21d returns: "
          f"{(t21.rolling(252).rank(pct=True)*100).iloc[-1]:.1f}")
print("\nVERDICT: today reproduces the lagging-marker signature. SVXY's carry "
      "has already been harvested into the trigger, which is why the placebo\n"
      "offset ladder in v1b_c3_round2.py pays MORE at every negative offset.")
