"""Quantify the live states named in the surface map. Numbers only, no verdicts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa

SECT = ['XLK','XLF','XLV','XLY','XLP','XLI','XLE','XLU','XLB','XLRE','XLC']
px = close_panel(SECT + ['SPY','QQQ','SMH','TLT','IEF','LQD','GLD','GDX','SLV','USO','UUP','DX-Y.NYB','^TNX','^VIX','^SKEW','IWM','EFA','EEM','HYG'])
px = px.dropna(how='all')
asof = pd.Timestamp('2026-08-18')
r1 = px.pct_change()

print("== one-day sector dispersion, 2026-08-18")
d = r1.loc[asof, SECT].sort_values()
print((d*100).round(2).to_string())
spread = d.max() - d.min()
hist = (r1[SECT].max(axis=1) - r1[SECT].min(axis=1)).dropna()
print(f"max-min sector spread {spread*100:.2f}pp, pctile of history {(hist<=spread).mean()*100:.1f} (n={len(hist)})")

xlv_xlk = (r1['XLV'] - r1['XLK']).dropna()
v = xlv_xlk.loc[asof]
print(f"XLV-XLK 1d {v*100:.2f}pp, pctile {(xlv_xlk<=v).mean()*100:.1f} (n={len(xlv_xlk)}), trailing-252d pctile {(xlv_xlk.loc[:asof].tail(252)<=v).mean()*100:.1f}")

spy1 = r1['SPY'].loc[asof]
print(f"SPY 1d {spy1*100:.2f}%  QQQ {r1['QQQ'].loc[asof]*100:.2f}%  SMH {r1['SMH'].loc[asof]*100:.2f}%")

print("\n== breadth on 2026-08-18: how many liquid names up while SPY down")
import json
tape = json.load(open('data/pitch_tape.json'))['tickers']
ups = [k for k,v in tape.items() if v.get('ret_1d') is not None and v['ret_1d']>0 and not k.startswith('^')]
tot = [k for k,v in tape.items() if v.get('ret_1d') is not None and not k.startswith('^')]
print(f"tape names up {len(ups)}/{len(tot)} = {len(ups)/len(tot)*100:.1f}% while SPY -0.68%")

print("\n== 52w-high / drawdown states")
for t in ['XLV','XLE','XLK','SPY','QQQ','SMH']:
    s = px[t].dropna()
    hi = s.loc[:asof].tail(252).max()
    print(f"{t:5s} last {s.loc[asof]:8.2f} 52wh {hi:8.2f} dist {100*(s.loc[asof]/hi-1):6.2f}%")

print("\n== rates vs dollar divergence")
for t,n in [('^TNX',21),('DX-Y.NYB',21),('UUP',21),('TLT',21)]:
    s = px[t].dropna()
    r = s.pct_change(n).loc[asof]
    pr = pct_rank(s, n).loc[asof]
    print(f"{t:10s} {n}d ret {r*100:7.2f}% trailing-252d rank {pr:5.1f}")

print("\n== TLT distance from its 52w low")
s = px['TLT'].dropna(); lo = s.loc[:asof].tail(252).min()
print(f"TLT {s.loc[asof]:.2f} 52wlow {lo:.2f} dist +{100*(s.loc[asof]/lo-1):.2f}%")

print("\n== SKEW / VIX watchlist legs")
sk = px['^SKEW'].dropna(); vx = px['^VIX'].dropna()
print(f"SKEW 5d rank {pct_rank(sk,5).loc[asof]:.1f} (needs >=95)")
print(f"VIX 21d rank {pct_rank(vx,21).loc[asof]:.1f}")
sp = px['SPY'].dropna(); print(f"SPY off 52w high {100*(sp.loc[asof]/sp.loc[:asof].tail(252).max()-1):.2f}% (needs < -1%)")

print("\n== calendar distances")
ev = load_events()
ev = ev[(ev['date']>='2026-08-01') & (ev['date']<='2026-09-30')]
print(ev.to_string())
