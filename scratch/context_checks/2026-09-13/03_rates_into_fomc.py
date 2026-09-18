"""Front-end yields arrive at the k3 FOMC anchor at an extreme: ^FVX 5d rank 99.6 (percent basis),
^TNX and ^FVX at 252-day highs, ^IRX 5d rank 100. What do yields do through the decision?

Measured in basis points (yield level differences), not percent, so low-yield eras do not
explode the scale. Rank of the 5d bp change over a trailing 252-session window.
h3 from the k3 anchor = decision-day close; h5 = two sessions after.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, load_events, summarize, era_split, sign_test, cluster_note, declusters

px = load_prices(['SPY', '^FVX', '^TNX', '^IRX', 'TLT', 'IEF', '^GSPC', '^VIX'])
nyse = px['SPY']['Close'].dropna().index
nyse = nyse[nyse >= '1999-01-01']
Y = {t: px[t]['Close'].reindex(nyse).ffill(limit=2) for t in ['^FVX', '^TNX', '^IRX']}
P = {t: px[t]['Close'].reindex(nyse) for t in ['TLT', 'IEF', '^GSPC', '^VIX']}
d = nyse
pos = pd.Series(range(len(d)), index=d)

fvx = Y['^FVX']
ch5 = fvx - fvx.shift(5)                                   # yield points (4.79 = 4.79%)
rank5 = ch5.rolling(252, min_periods=200).apply(lambda w: (w[:-1] < w[-1]).mean() * 100, raw=True)
pct5 = fvx.pct_change(5)
rank5p = pct5.rolling(252, min_periods=200).apply(lambda w: (w[:-1] < w[-1]).mean() * 100, raw=True)
hi252 = fvx >= fvx.rolling(252, min_periods=200).max() - 1e-9

print("tonight:", d[-1].date(), "FVX", fvx.iloc[-1], "5d bp", round(100 * ch5.iloc[-1], 1),
      "rank5(bp)", round(rank5.iloc[-1], 1), "rank5(pct)", round(rank5p.iloc[-1], 1), "252d hi", bool(hi252.iloc[-1]))

ev = load_events(['fomc_decision'])
fomc = pd.DatetimeIndex(sorted(ev['date'].unique()))
k3 = pd.DatetimeIndex([d[pos[f] - 3] for f in fomc if f in pos.index and pos[f] >= 3 and pos[f] + 2 < len(d)])


def fwd_bp(s, a, h):
    return np.array([100 * (s.iloc[pos[x] + h] - s[x]) for x in a])


def fwd_pct(s, a, h):
    return np.array([s.iloc[pos[x] + h] / s[x] - 1 for x in a])


def report(label, a):
    a = pd.DatetimeIndex([x for x in a if pos[x] + 5 < len(d)])
    print(f"\n--- {label}: n={len(a)}")
    if len(a) == 0:
        return
    for t in ['^FVX', '^TNX']:
        for h in (3, 5):
            v = fwd_bp(Y[t], a, h)
            dn = int((v < 0).sum())
            print(f"  {t} h{h}: mean {v.mean():+6.2f}bp median {np.median(v):+6.2f}bp lower {dn}/{len(v)} "
                  f"signp_down={sign_test(dn, len(v)):.4f}")
    for t in ['TLT', 'IEF', '^GSPC']:
        v = fwd_pct(P[t], a, 3)
        st = summarize(v)
        w = int((v > 0).sum())
        print(f"  {t} h3: mean {st['mean_pct']:+.3f}% up {w}/{len(v)} t {st['t']:+.2f}")
    return a


print("\n=== all k3 anchors (control) ===")
report("all FOMC k3 anchors", k3)

for thr in (90, 95):
    sel = k3[(rank5.reindex(k3) >= thr).values]
    a = report(f"k3 anchors with FVX 5d bp-change rank >= {thr}", sel)
    if thr == 90 and a is not None:
        v = fwd_bp(Y['^FVX'], a, 3)
        print("  episodes (anchor, 5d bp in, h3 bp):", [(str(x.date()), round(100 * ch5[x], 1), round(vv, 1)) for x, vv in zip(a, v)])
        print("  era:", [(e['label'], e['n'], round(e['mean_pct'] / 100, 2), round(100 - e['hit'], 1)) for e in era_split(a, v / 100)],
              "(mean bp, pct lower)")

sel = k3[(rank5p.reindex(k3) >= 95).values]
report("k3 anchors with FVX 5d PERCENT rank >= 95 (engine basis)", sel)
sel = k3[hi252.reindex(k3).fillna(False).values]
report("k3 anchors with FVX at a 252-day high", sel)

print("\n=== non-FOMC control: same state, no decision in the next 5 sessions ===")
near = set()
for f in fomc:
    if f in pos.index:
        for k in range(0, 6):
            if pos[f] - k >= 0:
                near.add(d[pos[f] - k])
state = d[(rank5 >= 90).values]
state = pd.DatetimeIndex([x for x in state if x not in near])
state = declusters(state, 10, d)
report("FVX 5d bp rank >= 90, declustered 10td, no FOMC within 5", state)
