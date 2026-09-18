"""Breadth version of the bond story: the whole US duration complex pinned at
52-week lows on the same session.

Live 2026-09-08: TLT 1.43% off its 52w low, IEF 0.34% off, LQD 0.25% off,
^TNX and ^FVX both AT their 252d highs. 01 showed the ^TNX-at-a-high cell is
dead post-2018 for equities and duration; this asks the breadth question instead,
which is a different and rarer state.

Also lands the 2-session unconditional base rate the CPI run-up cell needs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (close_panel, rolling_on_valid, fwd_ret, declusters,
                       local_control, summarize, era_split, sign_test,
                       cluster_note, show)

px = close_panel(["TLT", "IEF", "LQD", "^GSPC", "SPY", "^TNX", "GC=F"])

print("=" * 74)
print("BASE RATES the CPI run-up cell is measured against")
print("=" * 74)
for h in (1, 2, 3):
    f = fwd_ret(px["^GSPC"], h)
    v = f.dropna().values
    up = int((v > 0).sum())
    print(f"  all sessions h={h}: n={len(v)} mean={100*v.mean():+.3f}% "
          f"hit={100*(v>0).mean():.1f}% record {up}-{len(v)-up}")

print("\n" + "=" * 74)
print("DURATION BREADTH: TLT, IEF and LQD all within 2% of their 52-week lows")
print("=" * 74)
near = {}
for t in ("TLT", "IEF", "LQD"):
    s = px[t]
    lo = rolling_on_valid(s, lambda x: x.rolling(252).min())
    near[t] = (s <= lo * 1.02) & s.notna() & lo.notna()
    d = px.index[near[t].fillna(False)]
    print(f"  {t}: {len(d)} sessions within 2% of a 52w low, "
          f"{d.min().date()} -> {d.max().date()}")

allthree = (near["TLT"].fillna(False) & near["IEF"].fillna(False)
            & near["LQD"].fillna(False))
dates = px.index[allthree]
print(f"\nALL THREE at once: {len(dates)} sessions")
print("  by year:", dict(pd.Series(1, index=dates).groupby(dates.year).sum()))
print("  2026-09-08 in set:", pd.Timestamp("2026-09-08") in dates)

valid = px["^GSPC"].dropna().index
epi = declusters(pd.DatetimeIndex(dates), 21, valid)
print(f"  declustered at 21td: {len(epi)} episodes -> "
      f"{[str(d.date()) for d in epi]}")

def cell(sub, dd, h, gap, lab):
    f = fwd_ret(px[sub], h)
    v0 = f.dropna()
    d = pd.DatetimeIndex(dd).intersection(v0.index)
    e = declusters(d, gap, v0.index)
    v = f.loc[e].values
    if len(v) == 0:
        print(f"  {lab}: empty"); return
    up = int((v > 0).sum())
    s = summarize(v, lab)
    base = v0.mean()
    print(f"  {lab:30s} n={s['n']:4d} mean={s['mean_pct']:+.3f}% med={s['median_pct']:+.3f}% "
          f"hit={s['hit']:5.1f}% t={s['t']:+.2f} rec {up}-{s['n']-up} "
          f"p={sign_test(up, s['n']):.4f} ctl={100*base:+.3f}% edge={s['mean_pct']-100*base:+.3f}pp")
    return e, v

print("\n-- forward returns from an all-three-at-lows session --")
for sub in ["^GSPC", "TLT", "IEF", "GC=F"]:
    print(f"  [{sub}]")
    for h in (1, 5, 21):
        cell(sub, dates, h, max(h, 5), f"{sub} h={h}")

print("\n-- era split + concentration on the equity cell, h=21 --")
r = cell("^GSPC", dates, 21, 21, "^GSPC h=21")
if r:
    e, v = r
    show(era_split(e, v), "^GSPC h21 era split")
    print("  " + cluster_note(e, v, k=2))
    f = fwd_ret(px["^GSPC"], 21)
    ctl = local_control(f.dropna().index, pd.DatetimeIndex(dates), 126)
    print("  local +/-126td control:", summarize(f.loc[ctl].values, "local"))

print("\n-- and the same for TLT h=21, the 'does duration bounce' question --")
r = cell("TLT", dates, 21, 21, "TLT h=21")
if r:
    e, v = r
    show(era_split(e, v), "TLT h21 era split")
    print("  " + cluster_note(e, v, k=2))
