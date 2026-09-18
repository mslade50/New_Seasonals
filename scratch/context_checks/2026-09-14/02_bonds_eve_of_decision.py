"""E:fomc_decision k2: TLT +0.151% t 2.57, IEF t 2.53, ^TNX -0.31% on the eve session (h1 from the k2
anchor = the session immediately before the decision). Tonight bonds enter it at the bottom:
IEF 252d low, IEF 5d rank 1.2, 5 straight down closes, 10y 4.961% near a 252d high, ^IRX at a 252d high.

Questions: is the eve bond bid real against its controls (all sessions, all Tuesdays, same state
without a decision)? Does it survive or grow when bonds enter oversold? Era? Concentration?
Yields measured in bp (level differences).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import (load_prices, load_events, summarize, era_split, sign_test, cluster_note,
                       declusters, local_control)

px = load_prices(['SPY', 'TLT', 'IEF', '^TNX', '^IRX', '^GSPC'])
nyse = px['SPY']['Close'].dropna().index
nyse = nyse[nyse >= '2002-08-01']
d = nyse
pos = pd.Series(range(len(d)), index=d)
TLT = px['TLT']['Close'].reindex(d)
IEF = px['IEF']['Close'].reindex(d)
TNX = px['^TNX']['Close'].reindex(d).ffill(limit=2)
IRX = px['^IRX']['Close'].reindex(d).ffill(limit=2)

r1 = {'TLT': TLT.shift(-1) / TLT - 1, 'IEF': IEF.shift(-1) / IEF - 1}
r2 = {'TLT': TLT.shift(-2) / TLT - 1, 'IEF': IEF.shift(-2) / IEF - 1}
tnx1 = 100 * (TNX.shift(-1) - TNX)

ief5 = IEF.pct_change(5)
rank5 = ief5.rolling(252, min_periods=200).apply(lambda w: (w[:-1] < w[-1]).mean() * 100, raw=True)
low252 = IEF <= IEF.rolling(252, min_periods=200).min() * 1.0025
near_low = IEF <= IEF.rolling(252, min_periods=200).min() * 1.01
dn = (IEF.diff() < 0).astype(int)
streak = dn.groupby((dn == 0).cumsum()).cumsum()

print("tonight", d[-1].date(), "IEF rank5", round(rank5.iloc[-1], 1), "at low(0.25%)", bool(low252.iloc[-1]),
      "streak", int(streak.iloc[-1]), "TNX", TNX.iloc[-1])

ev = load_events(['fomc_decision'])
fomc = pd.DatetimeIndex(sorted(ev['date'].unique()))
fomc = fomc[(fomc >= d[3]) & (fomc <= d[-1])]
k2 = pd.DatetimeIndex([d[pos[f] - 2] for f in fomc if f in pos.index and pos[f] >= 2 and pos[f] < len(d)])
k2 = k2[k2 < d[-1]]


def rep(label, a):
    a = pd.DatetimeIndex([x for x in a if pos[x] + 2 < len(d)])
    out = {}
    print(f"\n--- {label}: n={len(a)}")
    if len(a) == 0:
        return a
    for t in ['TLT', 'IEF']:
        v = r1[t].reindex(a).values
        s = summarize(v)
        w = int((v > 0).sum())
        print(f"  {t} h1(eve): mean {s['mean_pct']:+.3f}% med {s['median_pct']:+.3f}% up {w}/{s['n']} "
              f"t {s['t']:+.2f} signp {sign_test(w, s['n']):.4f}")
        out[t] = v
    v = tnx1.reindex(a).values
    lo = int((v < 0).sum())
    print(f"  TNX h1: mean {np.nanmean(v):+.2f}bp lower {lo}/{len(v)}")
    v2 = r2['TLT'].reindex(a).values
    s2 = summarize(v2)
    print(f"  TLT h2(through decision): mean {s2['mean_pct']:+.3f}% up {int((v2 > 0).sum())}/{s2['n']} t {s2['t']:+.2f}")
    return a


print("\n=== controls ===")
alld = d[:-2]
rep("all sessions", alld)
rep("all Mondays as anchor (Tuesday h1)", alld[alld.weekday == 0])

print("\n=== FOMC k2 anchors ===")
a_all = rep("all scheduled decisions, k2 anchor", k2)
v = r1['TLT'].reindex(a_all).values
print("  era:", [(e['label'], e['n'], round(e['mean_pct'], 3), round(e['hit'], 1), round(e['t'], 2)) for e in era_split(a_all, v)])
print("  conc:", cluster_note(a_all, v))
print("  midterm:", summarize(v[(a_all.year % 4 == 2)])['mean_pct'], summarize(v[(a_all.year % 4 == 2)])['n'])
print("  september:", [(str(x.date()), round(100 * r1['TLT'][x], 2)) for x in a_all if x.month == 9])

for lab, m in [("IEF 5d rank <= 10", rank5 <= 10), ("IEF 5d rank <= 20", rank5 <= 20),
               ("IEF within 0.25% of 252d low", low252), ("IEF within 1% of 252d low", near_low),
               ("IEF 4+ straight down closes", streak >= 4), ("IEF 5d rank > 20", rank5 > 20)]:
    sel = k2[m.reindex(k2).fillna(False).values]
    a = rep(f"k2 with {lab}", sel)
    if len(a) and len(a) <= 30:
        print("  episodes:", [(str(x.date()), round(100 * r1['TLT'][x], 2), round(100 * r2['TLT'][x], 2)) for x in a])
    if len(a) >= 5:
        vv = r1['TLT'].reindex(a).values
        print("  era:", [(e['label'], e['n'], round(e.get('mean_pct', np.nan), 3), round(e.get('hit', np.nan), 1)) for e in era_split(a, vv)])
        print("  conc:", cluster_note(a, vv))

print("\n=== same state, no decision within 5 sessions (declustered 5td) ===")
near = set()
for f in fomc:
    if f in pos.index:
        for k in range(0, 6):
            if pos[f] - k >= 0:
                near.add(d[pos[f] - k])
for lab, m in [("IEF 5d rank <= 10", rank5 <= 10), ("IEF within 1% of 252d low", near_low)]:
    st = d[m.reindex(d).fillna(False).values]
    st = pd.DatetimeIndex([x for x in st if x not in near and pos[x] + 2 < len(d)])
    rep(f"non-FOMC {lab}, all days", st)
    rep(f"non-FOMC {lab}, declustered 5td", declusters(st, 5, d))
