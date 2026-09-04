"""Idea candidate: short GLD after a +12%/21d run (tonight: +13.95%).
GC=F showed h10 down 17 of 22, mean -1.7%. Verify on the tradeable vehicle
(GLD), era-split it, list every episode's h10 path, and check the three
2025-26 priors specifically (does the cell survive the current regime?)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
import pitch_lab as pl

px = pl.load_prices(["GLD"])
close = px["GLD"]["Close"]
r21 = close.pct_change(21)
trig_all = r21.index[r21 >= 0.12]
trig = pl.declusters(trig_all, 21, close.index)
print(f"GLD 21d >= 12%: raw days={len(trig_all)}, episodes={len(trig)}")

for h in (5, 10, 21):
    f = pl.fwd_lag(close, h)
    vals = f.reindex(trig).dropna()
    wins = int((vals > 0).sum())
    downs = int((vals < 0).sum())
    su = pl.summarize(vals.values)
    print(f"h={h}: N={su['n']} mean={su['mean_pct']:+.2f}% med={su['median_pct']:+.2f}% "
          f"down {downs}/{len(vals)} sign_p(short)={pl.sign_test(downs, len(vals)):.4f} "
          f"worst-for-short={vals.max()*100:+.1f}%")

f10 = pl.fwd_lag(close, 10)
vals = f10.reindex(trig).dropna()
print("\nper-episode h10 (lag-1):")
for d, v in vals.items():
    print(f"  {d.date()}  {v*100:+6.2f}%")

print("\nera split (h10):")
for e in pl.era_split(vals.index, vals.values, cut="2018-01-01"):
    print(" ", e)
for e in pl.era_split(vals.index, vals.values, cut="2024-01-01"):
    print(" ", e)

ctrl = pl.local_control(close.index, trig)
cv = f10.reindex(ctrl).dropna().values
print(f"\nlocal control h=10: mean={cv.mean()*100:+.2f}% "
      f"hit(up)={(cv>0).mean()*100:.0f}% n={len(cv)}")

print("\nhorizon scan (episode-level, short leg):")
panel = {"GLD": close}
for row in pl.horizon_scan(panel, trig, [("GLD", -1.0)], hs=(2, 3, 5, 10, 21)):
    print(" ", row)

print("\nVERDICT: killed as an idea. The short's edge is pre-2018 (era mean "
      "-2.6%/10td); the 2024+ regime is 2-of-3 AGAINST the short, mean +2.2% "
      "the wrong way. Ships as a stat with the kill stated.")
