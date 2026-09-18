"""KC=F printed -10.58% today, the largest move anywhere in the 87-name universe.

Two questions, both descriptive: how historic is that session, and does a crash
of that size in a soft commodity carry any next-session information. The engine's
own KC cells were coin flips (P7b down-streak 90-96, P6 2-ATR 27-25), so the prior
is that this is a magnitude fact, not a signal.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (close_panel, fwd_ret, declusters, summarize,
                       era_split, sign_test, cluster_note, show)

px = close_panel(["KC=F", "CC=F", "SB=F", "CT=F"])
kc = px["KC=F"].dropna()
r = kc.pct_change(fill_method=None).dropna()
print(f"KC=F history: {r.index.min().date()} -> {r.index.max().date()}, {len(r)} sessions")

today = r.loc["2026-09-08"]
print(f"\n2026-09-08 return: {100*today:.2f}%")
worse = (r < today).sum()
print(f"sessions with a LARGER decline: {worse}  "
      f"(so today is the {worse+1}th worst of {len(r)}, "
      f"{100*(worse+1)/len(r):.2f}th percentile)")
print("\nten worst KC=F sessions on record:")
for d, v in r.nsmallest(10).items():
    print(f"   {d.date()}  {100*v:+.2f}%")
print("\nworst sessions by year, last 12 years:")
byyr = r.groupby(r.index.year).min()
print("   ", {int(y): round(100*v, 1) for y, v in byyr.tail(12).items()})

print("\n-- how often does KC=F fall 8%+ in a session? --")
big = r[r <= -0.08]
print(f"   {len(big)} sessions since {r.index.min().date()}; by year:",
      dict(pd.Series(1, index=big.index).groupby(big.index.year).sum()))

print("\n-- forward returns after a -8% or worse KC=F session --")
valid = r.index
for h in (1, 5, 21):
    f = fwd_ret(kc, h)
    d = pd.DatetimeIndex(big.index).intersection(f.dropna().index)
    e = declusters(d, max(h, 5), f.dropna().index)
    v = f.loc[e].values
    if len(v) == 0:
        continue
    up = int((v > 0).sum())
    s = summarize(v, f"h={h}")
    base = f.dropna()
    print(f"   h={h:2d}: n={s['n']:3d} mean={s['mean_pct']:+.3f}% med={s['median_pct']:+.3f}% "
          f"hit={s['hit']:5.1f}% t={s['t']:+.2f} rec {up}-{s['n']-up} "
          f"p={sign_test(up, s['n']):.4f} ctl={100*base.mean():+.3f}% "
          f"edge={s['mean_pct']-100*base.mean():+.3f}pp")

f1 = fwd_ret(kc, 1)
d1 = pd.DatetimeIndex(big.index).intersection(f1.dropna().index)
v1 = f1.loc[d1].values
show(era_split(d1, v1), "KC=F h1 after an 8%+ drop, era split")
print("   " + cluster_note(d1, v1, k=2))

print("\n-- context: the softs complex today --")
for t in ("KC=F", "CC=F", "SB=F", "CT=F"):
    s = px[t].dropna()
    rr = s.pct_change(fill_method=None)
    print(f"   {t}: today {100*rr.loc['2026-09-08']:+.2f}%, "
          f"5d {100*(s.loc['2026-09-08']/s.iloc[-6]-1):+.2f}%, "
          f"21d {100*(s.loc['2026-09-08']/s.iloc[-22]-1):+.2f}%")
