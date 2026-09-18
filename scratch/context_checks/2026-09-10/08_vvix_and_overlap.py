"""(1) ^VVIX 5d return in the top 5% of its year -- the sweep's only [solid] cell (n=239,
    h1 -1.578%, t -2.78, BH pass). Transfer it to the S&P, and cross it with tonight's
    second VVIX state: a first 200-day cross in 63+ sessions.
(2) Overlap between the two CPI-eve conditionings from drills 04 and 07, so the brief
    does not publish the same 65 sessions twice.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np, pandas as pd
from pitch_lab import load_prices, load_events, summarize, era_split, sign_test, cluster_note, declusters, local_control

px = load_prices(['^VVIX','^VIX','^GSPC','SPY','^VIX3M'])
cl = {t: px[t]['Close'].dropna() for t in px}
vv = cl['^VVIX']; d = vv.index
r5 = vv.pct_change(5)
rank5 = r5.rolling(252).rank(pct=True) * 100
sma200 = vv.rolling(200).mean()

print("tonight  ^VVIX", round(vv.iloc[-1], 2), "| 5d return", f"{100*r5.iloc[-1]:+.2f}%",
      "| 5d rank", round(rank5.iloc[-1], 1), "| vs 200d", f"{100*(vv.iloc[-1]/sma200.iloc[-1]-1):+.2f}%")

trig = d[(rank5 >= 95).reindex(d).fillna(False)]
trig = trig[trig >= pd.Timestamp('2007-01-01')]
dec = declusters(trig, 5, d)
print(f"\n5d rank >= 95: {len(trig)} sessions, {len(dec)} declustered episodes")

def fwd(sel, sym, h):
    s = cl[sym]; out = []
    for a in sel:
        if a not in s.index: continue
        i = s.index.get_loc(a)
        if i + h < len(s): out.append((a, s.iloc[i+h]/s.iloc[i]-1))
    return pd.DatetimeIndex([o[0] for o in out]), np.array([o[1] for o in out])

def show(lab, sel, sym, h, verbose=False):
    dd, v = fwd(sel, sym, h)
    if len(v) < 4: print(f"  {lab:42} {sym:6} h{h:<2} n={len(v)}"); return None
    st = summarize(v); w = int((v > 0).sum())
    print(f"  {lab:42} {sym:6} h{h:<2} n={len(v):4} mean={st['mean_pct']:+7.3f}% up={w:3}/{len(v):3} "
          f"t={st['t']:+6.2f} signp={sign_test(w,len(v)):.4f}")
    if verbose:
        print("      era:", [(e['label'], e['n'], round(e['mean_pct'],3), round(e['hit'],1)) for e in era_split(dd, v)])
        print("      concentration:", cluster_note(dd, v))
    return dd, v

print("\n=== raw (overlapping) and declustered ===")
for sel, lab in [(trig, 'raw sessions'), (dec, 'declustered 5td')]:
    for h in (1, 5, 21):
        show(lab, sel, '^GSPC', h)
    for h in (1, 5):
        show(lab, sel, '^VVIX', h)
    print()

print("=== declustered, S&P legs in detail ===")
show('VVIX 5d rank >= 95', dec, '^GSPC', 1, verbose=True)
show('VVIX 5d rank >= 95', dec, '^GSPC', 5, verbose=True)

print("\n=== cross with the 200-day: VVIX above its 200d mean vs below ===")
above = dec[(vv.reindex(dec) > sma200.reindex(dec)).fillna(False)]
below = dec[(vv.reindex(dec) <= sma200.reindex(dec)).fillna(False)]
print(f"  above 200d: {len(above)}   below: {len(below)}")
for h in (1, 5, 21):
    show('rank>=95 AND above the 200d', above, '^GSPC', h)
for h in (1, 5, 21):
    show('rank>=95 AND below the 200d', below, '^GSPC', h)

print("\n=== CONTROLS ===")
dd = d[d >= pd.Timestamp('2007-01-01')]
for h in (1, 5, 21):
    _, v = fwd(dd, '^GSPC', h)
    print(f"  baseline ^GSPC h{h:<2} n={len(v):5} mean={summarize(v)['mean_pct']:+7.3f}% up={100*(v>0).mean():5.1f}%")
lc = local_control(dd, dec, 126)
for h in (1, 5):
    _, v = fwd(lc, '^GSPC', h)
    print(f"  local +/-126td control h{h}: n={len(v)} mean={summarize(v)['mean_pct']:+.3f}% up={100*(v>0).mean():.1f}%")

# ---------------- overlap of the two CPI conditionings --------------------
print("\n\n=== overlap check: CPI-eve conditionings ===")
ev = load_events()
cpi = pd.DatetimeIndex(sorted(pd.to_datetime(ev.loc[ev['event']=='cpi','date'].unique())))
px2 = load_prices(['^VIX','^GSPC','^TNX','QQQ'])
c2 = {t: px2[t]['Close'].dropna() for t in px2}
dd2 = c2['^GSPC'].index
pairs = [(dd2[dd2 < c][-1], c) for c in cpi if c in dd2 and len(dd2[dd2 < c])]
pairs = [(a, c) for a, c in pairs if a >= pd.Timestamp('1999-01-01')]
A = pd.DatetimeIndex([a for a, _ in pairs])
vixup = set(A[(c2['^VIX'].pct_change().reindex(A) > 0).fillna(False)])
risky = set(A[((c2['^GSPC'].pct_change().reindex(A) < 0) & (c2['^TNX'].pct_change().reindex(A) > 0)).fillna(False)])
print(f"  VIX-up eves: {len(vixup)}   S&P-down+yield-up eves: {len(risky)}   both: {len(vixup & risky)}")
print(f"  share of the S&P-down+yield-up set inside the VIX-up set: {100*len(vixup & risky)/len(risky):.0f}%")
only = risky - vixup
print(f"  S&P-down+yield-up but VIX NOT up: {len(only)}")
rows = [(a, c2['QQQ'].loc[c]/c2['QQQ'].loc[a]-1) for a, c in pairs if a in risky and a in c2['QQQ'].index and c in c2['QQQ'].index]
v = np.array([r[1] for r in rows]); print(f"  QQQ on the S&P-down+yield-up set: n={len(v)} mean={summarize(v)['mean_pct']:+.3f}%")
