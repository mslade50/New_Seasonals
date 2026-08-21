"""C9 round 2c — the last thing standing is HYG's 17-2 record at JH-5, h=8/10.

The kill this file is looking for is a WRONG NULL. `sign_test(17, 19)` = 0.0004
under a COIN. HYG is a credit instrument that grinds up: its unconditional
10-session up-rate is nowhere near 50%, so p=0.5 is the wrong null and the
headline p is an artifact of the instrument, not the anchor. pitch_lab's
sign_test takes p; use it.

Also finishes the JH-session table b2b cut off, and prices the h=8/10 cell
against the one control that matters: SPY, whose JH cell the registry CLOSED
on 2026-08-18.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

VEH = ["HYG", "LQD", "EFA", "EEM", "FXI", "EWJ", "SPY"]
px = close_panel(VEH)
idx = px.index
jh = load_events(["jackson_hole"])["date"]
JHPOS = [int(idx.searchsorted(d)) for d in jh if int(idx.searchsorted(d)) < len(idx)]
aJH = pd.DatetimeIndex([idx[p - 6] for p in JHPOS if p - 6 >= 0])
JHDAY = pd.DatetimeIndex([idx[p] for p in JHPOS])
tdom = pd.Series(pd.Series(idx, index=idx).groupby([idx.year, idx.month])
                 .cumcount().values + 1, index=idx)
ATD = sorted(tdom.loc[aJH].unique())

print("===== JH SESSION own return, full table =====")
for v in VEH:
    s = px[v].dropna()
    day = s / s.shift(1) - 1.0
    a = pd.DatetimeIndex(JHDAY).intersection(day.dropna().index)
    ad = day.dropna()
    ad = ad[(ad.index >= a[0]) & (ad.index <= a[-1])]
    p0 = float((ad > 0).mean())
    w = int((day.loc[a] > 0).sum())
    print(f"{v:4s}: JH day {100*day.loc[a].mean():+.3f}% ({w}-{len(a)-w}) vs "
          f"all-days {100*ad.mean():+.3f}% | excess "
          f"{100*(day.loc[a].mean()-ad.mean()):+.3f}pp | sign p vs coin "
          f"{sign_test(w, len(a)):.4f} | sign p vs OWN up-rate {p0:.3f}: "
          f"{sign_test(w, len(a), p0):.4f}")

print("\n\n===== THE WRONG-NULL TEST: sign test against the instrument's OWN "
      "unconditional up-rate =====")
for v in VEH:
    s = px[v].dropna()
    for h in (5, 8, 10):
        r = fwd_lag(s, h, 1).dropna()
        a = pd.DatetimeIndex(aJH).intersection(r.index)
        base = r[(r.index >= a[0]) & (r.index <= a[-1])]
        aug = r.index[(r.index.month == 8) & (tdom.reindex(r.index).isin(ATD))
                      & (r.index >= a[0]) & (r.index <= a[-1])].difference(a)
        p_all = float((base > 0).mean())
        p_aug = float((r.loc[aug] > 0).mean())
        w = int((r.loc[a] > 0).sum())
        n = len(a)
        print(f"{v:4s} h={h:2d}: {w}-{n-w} ({100*w/n:.0f}%) | coin p "
              f"{sign_test(w, n):.4f} | own all-days up-rate {100*p_all:.1f}% "
              f"-> p {sign_test(w, n, p_all):.4f} | AUG-tdom up-rate "
              f"{100*p_aug:.1f}% -> p {sign_test(w, n, p_aug):.4f}")

print("\n\n===== HYG vs SPY head to head at the same anchor =====")
print("SPY's JH cell was CLOSED in the registry on 2026-08-18. If credit is a")
print("new class it has to beat that leg, not the coin.")
for h in (5, 8, 10):
    rh = fwd_lag(px["HYG"].dropna(), h, 1).dropna()
    rs = fwd_lag(px["SPY"].dropna(), h, 1).dropna()
    a = pd.DatetimeIndex(aJH).intersection(rh.index).intersection(rs.index)
    aug_h = rh.index[(rh.index.month == 8) & (tdom.reindex(rh.index).isin(ATD))
                     & (rh.index >= a[0]) & (rh.index <= a[-1])].difference(a)
    aug_s = rs.index[(rs.index.month == 8) & (tdom.reindex(rs.index).isin(ATD))
                     & (rs.index >= a[0]) & (rs.index <= a[-1])].difference(a)
    xh = 100 * (rh.loc[a].mean() - rh.loc[aug_h].mean())
    xs = 100 * (rs.loc[a].mean() - rs.loc[aug_s].mean())
    print(f"h={h:2d}: HYG excess {xh:+.3f}pp vs SPY excess {xs:+.3f}pp over the "
          f"SAME 19 anchors -> credit adds {xh-xs:+.3f}pp")
    print(f"      risk-adjusted: HYG {100*rh.loc[a].mean():+.3f}% / sd "
          f"{100*rh.loc[a].std(ddof=1):.2f}% = "
          f"{rh.loc[a].mean()/rh.loc[a].std(ddof=1):.3f} | SPY "
          f"{100*rs.loc[a].mean():+.3f}% / sd "
          f"{100*rs.loc[a].std(ddof=1):.2f}% = "
          f"{rs.loc[a].mean()/rs.loc[a].std(ddof=1):.3f}")

print("\n\n===== live-state honesty on HYG =====")
s = px["HYG"].dropna()
hi = rolling_on_valid(s, lambda x: x.rolling(252).max())
print(f"HYG 2026-08-20 close {s.iloc[-1]:.2f}, 252d high {hi.iloc[-1]:.2f} "
      f"({100*(s.iloc[-1]/hi.iloc[-1]-1):+.2f}%), 21d "
      f"{100*(s.iloc[-1]/s.iloc[-22]-1):+.2f}%, "
      f"63d {100*(s.iloc[-1]/s.iloc[-64]-1):+.2f}%")
print("A 10-session long in a 4%-vol credit ETF for a +0.08pp beta-adjusted")
print("edge is a leverage question, not an alpha question: quote the sd.")
r10 = fwd_lag(s, 10, 1).dropna()
a = pd.DatetimeIndex(aJH).intersection(r10.index)
print(f"cell sd {100*r10.loc[a].std(ddof=1):.2f}%, worst {100*r10.loc[a].min():.2f}%, "
      f"mean/sd {r10.loc[a].mean()/r10.loc[a].std(ddof=1):.2f}")
