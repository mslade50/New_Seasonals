"""Which reference class does TODAY (2026-08-06 close) belong to for each candidate?
Consecutive-trigger-day run length decides whether the first-of-cluster stats apply.
Also prints the exact live values one more time for the record.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _study import *  # noqa

import numpy as np
import pandas as pd

P = close_panel(["GLD", "SLV", "GDX", "UNG"])
ASOF = P.index[-1]


def run_len(mask: pd.Series) -> int:
    m = mask.fillna(False).values
    i, n = len(m) - 1, 0
    while i >= 0 and m[i]:
        n += 1
        i -= 1
    return n


Pg = P[["GLD", "GDX"]].dropna()
c5m = ((Pg["GDX"].pct_change(21) - Pg["GLD"].pct_change(21)) * 100) >= 8.0

Ps = P[["SLV", "GLD"]].dropna()
c6m = (((Ps["SLV"].pct_change(63) - Ps["GLD"].pct_change(63)) * 100) <= -8.0) & (Ps["SLV"].pct_change(5) > 0)

u = P["UNG"].dropna()
c7m = ((u / u.rolling(252).min() - 1) * 100 <= 1.0) & pd.Series(u.index.month.isin([7, 8, 9]), index=u.index)

gg = Pg["GDX"]
c8m = (gg.pct_change(21) * 100 >= 12.0) & (pct_rank(gg, 63) < 30.0)

for nm, m, gap in (("C5", c5m, 5), ("C6", c6m, 10), ("C7", c7m, 10), ("C8", c8m, 10)):
    r = run_len(m)
    print(f"{nm}: fires_today={bool(m.iloc[-1])}  consecutive_trigger_days={r}  "
          f"-> {'FIRST of cluster (day-1 stats apply)' if r == 1 else f'day {r} of an ongoing cluster'}")
    prior = m.index[m.fillna(False)]
    prior = prior[prior < ASOF - pd.Timedelta(days=1)]
    if len(prior):
        print(f"      previous trigger day before this run: {prior[-1].date()} "
              f"({(ASOF - prior[-1]).days} calendar days ago)")

print(f"\nas-of {ASOF.date()}")
print(f"  C5 spread(GDX21-GLD21) = {((Pg['GDX'].pct_change(21)-Pg['GLD'].pct_change(21))*100).iloc[-1]:+.2f}pp  [need >= +8]")
print(f"  C6 spread(SLV63-GLD63) = {((Ps['SLV'].pct_change(63)-Ps['GLD'].pct_change(63))*100).iloc[-1]:+.2f}pp  [need <= -8]"
      f"  SLV r5 = {(Ps['SLV'].pct_change(5)*100).iloc[-1]:+.2f}%  [need > 0]")
print(f"  C7 UNG above 52w low   = {((u/u.rolling(252).min()-1)*100).iloc[-1]:.3f}%  [need <= 1.0]  month={ASOF.month}")
print(f"  C8 GDX r21 = {(gg.pct_change(21)*100).iloc[-1]:+.2f}%  [need >= +12]  "
      f"rank63 = {pct_rank(gg,63).iloc[-1]:.1f}  [need < 30]")

# how often do C5 and C8 fire on the SAME day, and what did the book do?
both = (c5m.reindex(Pg.index).fillna(False) & c8m.reindex(Pg.index).fillna(False))
print(f"\nC5 & C8 co-fire on {int(both.sum())} of {int(c8m.sum())} C8 days ({100*both.sum()/max(c8m.sum(),1):.0f}%), "
      f"today included = {bool(both.iloc[-1])}. They are opposite bets on GDX.")
