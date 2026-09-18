"""Crude closed 101.89, +23.65% over 21 sessions, z10 +2.51, two sessions before a decision, with the
10-year at a 252-day high. FOMC decisions that arrive after a 15%+/20%+ crude month: decision-day and
following-week S&P, 10-year (bp) and crude itself. Expect anecdote-scale N.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, load_events, summarize, sign_test

px = load_prices(['SPY', '^GSPC', 'CL=F', '^TNX', 'TLT'])
d = px['SPY']['Close'].dropna().index
d = d[d >= '1999-01-01']
pos = pd.Series(range(len(d)), index=d)
S = px['^GSPC']['Close'].reindex(d)
C = px['CL=F']['Close'].reindex(d).ffill(limit=2)
T = px['^TNX']['Close'].reindex(d).ffill(limit=2)
c21 = C / C.shift(21) - 1
print("tonight crude 21d", round(100 * c21.iloc[-1], 2))

ev = load_events(['fomc_decision'])
fo = pd.DatetimeIndex(sorted(ev['date'].unique()))
k2 = pd.DatetimeIndex([d[pos[f] - 2] for f in fo if f in pos.index and pos[f] >= 2 and pos[f] + 5 < len(d)])


def fwd(s, a, h):
    return np.array([s.iloc[pos[x] + h] / s[x] - 1 for x in a])


def rep(lab, a):
    print(f"\n--- {lab}: n={len(a)}")
    for h, nm in [(2, 'decision close'), (7, 'decision +5')]:
        v = fwd(S, a, h)
        s = summarize(v)
        w = int((v > 0).sum())
        bp = np.array([100 * (T.iloc[pos[x] + h] - T[x]) for x in a])
        cv = fwd(C, a, h)
        print(f"  h{h} ({nm}): S&P mean {s['mean_pct']:+.2f}% up {w}/{s['n']} t {s['t']:+.2f} | "
              f"10y {np.nanmean(bp):+.1f}bp higher {int((bp > 0).sum())}/{len(bp)} | crude mean {100 * np.nanmean(cv):+.2f}% up {int((cv > 0).sum())}")
    return a


rep("all k2 anchors", k2)
for thr in (0.15, 0.20):
    a = k2[(c21.reindex(k2) >= thr).values]
    rep(f"k2 with crude 21d >= {int(thr * 100)}%", a)
    print("  episodes (anchor, crude21d, S&P h2, 10y bp h2, S&P h7):",
          [(str(x.date()), round(100 * c21[x], 1), round(100 * (S.iloc[pos[x] + 2] / S[x] - 1), 2),
            round(100 * (T.iloc[pos[x] + 2] - T[x]), 1), round(100 * (S.iloc[pos[x] + 7] / S[x] - 1), 2)) for x in a])
a = k2[((c21.reindex(k2) >= 0.15) & (C.reindex(k2) >= 80)).values]
rep("k2 with crude 21d >= 15% and crude >= $80", a)
