"""r5 - the live-state kicker attack 4a implied but did not spell out: under
the STUDY'S OWN declustering rule, is today even the tradeable date?

Depth 3 means the current cluster STARTED two sessions ago. The 16-episode
statistic is measured on cluster-FIRST days (15 of 16 are depth 1). If the
episode-first day of the live cluster was 2026-08-10, an entry tomorrow is
the 3rd-day chase, and part of the measured move is already spent.

Also: the depth-matched and magnitude-matched cells, restated as the honest
live-analogue N, and one summary block of every number the verdict rests on.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

H = 5
c = load_prices(["IHI"])["IHI"]["Close"].dropna()
r21 = pct_rank(c, 21)
dd = c / c.rolling(252).max() - 1.0
m = ((r21 >= 99) & (dd <= -0.10)).fillna(False)

print("=== 1. the LIVE cluster: every trigger day since 2026-06-01 ===")
recent = c.index[(c.index >= "2026-06-01") & m.reindex(c.index).fillna(False).values]
pos = pd.Series(range(len(c.index)), index=c.index)
for d in recent:
    p = pos[d]
    print(f"  {d.date()}  close {c.iloc[p]:7.2f}  r21 {r21.loc[d]:5.1f}  "
          f"ret21 {100*(c.iloc[p]/c.iloc[p-21]-1):+6.2f}%  dd {100*dd.loc[d]:+6.2f}%")
print(f"\n  ALL trigger days incl. today: n={int(m.sum())}")
epi_all = declusters(c.index[m.reindex(c.index).fillna(False).values], 5, c.index)
print(f"  declustered episode-first days since 2026-01-01: "
      f"{[str(d.date()) for d in epi_all if d.year == 2026]}")
print(f"  LAST episode-first day = {epi_all[-1].date()};  today = {c.index[-1].date()}")
gap = pos[c.index[-1]] - pos[epi_all[-1]]
print(f"  today is {gap} td after that episode-first day.")
if gap > 0:
    p0 = pos[epi_all[-1]]
    print(f"  IHI since the episode-first close {c.iloc[p0]:.2f} -> "
          f"{c.iloc[-1]:.2f} = {100*(c.iloc[-1]/c.iloc[p0]-1):+.2f}% ALREADY BANKED")
    ex_i = p0 + 1 + H
    ex_lbl = (str(c.index[ex_i].date()) if ex_i < len(c)
              else f"~{H - (len(c)-1 - (p0+1))} sessions from the last bar (STILL OPEN)")
    print(f"  the study's tradeable entry for this cluster was the close of "
          f"{c.index[p0+1].date()} at {c.iloc[p0+1]:.2f}; from there IHI is "
          f"{100*(c.iloc[-1]/c.iloc[p0+1]-1):+.2f}%, and its h=5 exit close is {ex_lbl}")
    print(f"  -> an entry tomorrow is a {gap}-session-late DUPLICATE of a "
          f"position the rule already holds, not a fresh episode.")

    print("\n  --- and today's HEADLINE MAGNITUDE is a denominator roll ---")
    for i in (-1, -2, -3):
        print(f"    {c.index[i].date()} close {c.iloc[i]:6.2f}  21d base "
              f"{c.index[i-21].date()} {c.iloc[i-21]:6.2f}  ret21 "
              f"{100*(c.iloc[i]/c.iloc[i-21]-1):+6.2f}%")
    d_price = 100 * (c.iloc[-1] / c.iloc[-2] - 1)
    d_base = 100 * (c.iloc[-22] / c.iloc[-23] - 1)
    print(f"    ret21 jumped +9.04% -> +13.94% (+4.90pp) on a session where "
          f"PRICE moved {d_price:+.2f}%.")
    print(f"    cause: the 21d reference rolled {c.index[-23].date()} "
          f"{c.iloc[-23]:.2f} -> {c.index[-22].date()} {c.iloc[-22]:.2f} "
          f"({d_base:+.2f}%). The thrust is a lookback artefact, not new buying.")

print("\n=== 2. honest live-analogue N, three ways ===")
ret = fwd_lag(c, H)
trig = c.index[m.reindex(c.index).fillna(False).values & ret.notna().values]
epi = declusters(trig, 5, c.index)
epi = epi[ret.reindex(epi).notna().values]
v = ret.loc[epi].values
span = (c.index >= trig[0]) & (c.index <= trig[-1]) & ret.notna().values
ctrl = ret[span].values
base = float((ctrl > 0).mean())
mv = m.reindex(c.index).fillna(False).values
dep = np.zeros(len(mv), int)
for i in range(len(mv)):
    dep[i] = (dep[i-1] + 1) if (mv[i] and i > 0) else (1 if mv[i] else 0)
dser = pd.Series(dep, index=c.index)
mag = c.pct_change(21)
rows = []
for lbl, sel in [
    ("pitched cell: 16 episodes (depth-1 by construction)", epi),
    ("depth>=3 trigger days (today's depth)",
     trig[dser.loc[trig].values >= 3]),
    ("ret21 >= today's 13.94% (magnitude-matched)",
     trig[mag.loc[trig].values >= mag.iloc[-1]]),
    ("depth>=3 AND ret21>=10%",
     trig[(dser.loc[trig].values >= 3) & (mag.loc[trig].values >= 0.10)]),
]:
    sel = pd.DatetimeIndex(sel)
    sel = sel[ret.reindex(sel).notna().values]
    if len(sel) == 0:
        rows.append({"cell": lbl, "n": 0})
        continue
    vv = ret.loc[sel].values
    w = int((vv > 0).sum())
    rows.append({"cell": lbl, "n": len(vv), "mean_pct": round(100*vv.mean(), 3),
                 "hit": round(100*(vv > 0).mean(), 1),
                 "excess_pp": round(100*(vv.mean()-ctrl.mean()), 3),
                 "record": f"{w}-{len(vv)-w}",
                 "sign_p_base": round(sign_test(w, len(vv), base), 4),
                 "worst_pct": round(100*vv.min(), 2)})
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== 3. VERDICT LEDGER ===")
led = [
    ("A1 cross-section: pooled excess over 27 sector ETFs", "-0.231pp, 11/27 positive"),
    ("A1 Cochran Q on the 27 excess estimates", "Q=24.56/26df p=0.544, I^2=0.0%"),
    ("A1 fixed-effect COMMON excess across the 27", "-0.035pp (zero)"),
    ("A1 observed cross-sec sd / mean sampling SE", "0.936 / 1.054 = 0.89"),
    ("A1 family-wise max-stat p (IHI vs 27-ticker null max)", "0.9330"),
    ("A1 null max-of-27 median vs IHI's observed", "+1.92pp vs +1.21pp (SUB-MEDIAN)"),
    ("A1 tickers with p<0.05 one-sided", "1 of 27 (null expects 1.4)"),
    ("A2 sign p vs IHI's own base rate", "0.1198"),
    ("A2 2-look Sidak on the base-rate p", "0.2252 (> 0.20)"),
    ("A2 h=5 is argmax of 10 horizons?", "No (h=8 is; h=5 sign_p_base 0.1198)"),
    ("A2 min_gap 21td episodes", "excess +0.484pp, boot 0.194"),
    ("A2 year-cluster bootstrap", "0.0000 (degenerate: 9/9 years positive)"),
    ("A3 era fence", "ABSENT (pre-2018 +1.62 N=9, 2018+ +1.34 N=7)"),
    ("A3 2013-2019", "ZERO firings - 12-year hole"),
    ("A3 2008-2012 share of total R", "60.9% on 9/16 episodes"),
    ("A3 LOYO floor / drop-2-best-year", "+1.322% / +1.158%"),
    ("A3 definition fragility: thrust window 26d", "-0.036% (SIGN FLIP)"),
    ("A3 definition fragility: thrust window 10d", "+0.198%, excess -0.080pp"),
    ("A3 rank lookback 504d", "+0.504%, excess +0.174pp"),
    ("A3 magnitude gate ret21>=10% (no rank)", "+0.800%, excess +0.469pp"),
    ("A3 9-of-9 years unique?", "1 of 27 tickers; null expects ~0.3"),
    ("A4 today's cluster depth", "3; depth-2 days excess +0.113pp (p 0.68)"),
    ("A4 depth of the 16 episodes", "15 of 16 are depth 1"),
    ("A4 magnitude-matched cell (ret21>=13.94%)", "N=2, 1-1"),
    ("A4 corr(ret21 magnitude, fwd5)", "+0.080"),
    ("A4 holdings breadth today vs sample", "1.00 vs episode median 0.40, hist p95 0.40"),
    ("A4 breadth-only trigger", "+0.639%, excess +0.350pp (not a breadth effect)"),
    ("A4 stale trigger", f"episode-first day was {epi_all[-1].date()}, {gap} td ago"),
]
for k, x in led:
    print(f"  {k:55s} : {x}")
