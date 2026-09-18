"""A. NG=F September Mondays (+1.11% vs +0.04% other months). The top episode, 2025-09-29, is the
   Monday after the October contract expired, where the continuous series jumps into a steeper
   winter contract. Screen roll seams by the bar itself (gap vs prior close, volume vs 20d median)
   and by calendar (late-September Mondays), then re-measure. Tomorrow, Sep 14, is not a roll day.

B. Friday VIX -10% -> Monday, by era, against the all-Monday base IN THE SAME ERA.

C. TLT/IEF surge-anchor counts with NaN (pre-inception) anchors removed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices, load_events, summarize, era_split, sign_test, cluster_note

px = load_prices(['SPY', 'NG=F', '^VIX', '^FVX', 'TLT', 'IEF'])
nyse = px['SPY']['Close'].dropna().index
d = nyse[nyse >= '1999-01-01']
pos = pd.Series(range(len(d)), index=d)
wk_next = pd.Series(d.weekday, index=d).shift(-1)
fri_mon = d[(d.weekday == 4) & (wk_next == 0).values]

print("=== A. NG=F September Mondays ===")
ng = px['NG=F'].reindex(d)
c, o, vol = ng['Close'], ng['Open'], ng['Volume']
nxt = c.shift(-1) / c - 1
gap_next = o.shift(-1) / c - 1
volr_next = (vol / vol.rolling(20, min_periods=10).median().shift(1)).shift(-1)
rows = []
for x in fri_mon[fri_mon.month == 9]:
    if np.isnan(nxt.get(x, np.nan)):
        continue
    mon = d[pos[x] + 1]
    rows.append((x, mon, nxt[x], gap_next[x], volr_next[x]))
df = pd.DataFrame(rows, columns=['fri', 'mon', 'ret', 'gap', 'volr']).set_index('fri')
print(f"September Mondays n={len(df)} mean {100*df['ret'].mean():+.3f}% up {int((df['ret']>0).sum())}")
late = df['mon'].dt.day >= 22
print(f"  Monday on/after the 22nd: n={late.sum()} mean {100*df.loc[late,'ret'].mean():+.3f}% up {int((df.loc[late,'ret']>0).sum())}")
early = df[~late]
w = int((early['ret'] > 0).sum())
print(f"  Monday before the 22nd:   n={len(early)} mean {100*early['ret'].mean():+.3f}% med {100*early['ret'].median():+.3f}% up {w}")
print("  largest |ret| Mondays (mon, ret%, gap%, volratio):")
for fri, r in df.reindex(df['ret'].abs().sort_values(ascending=False).index).head(10).iterrows():
    print(f"     {r['mon'].date()} {100*r['ret']:+6.2f} gap {100*r['gap']:+6.2f} volr {r['volr']:.2f}")
seam = (df['volr'] > 5) | (df['mon'].dt.day >= 22)
clean = df[~seam]
v = clean['ret'].values
w = int((v > 0).sum())
oth = nxt.reindex(fri_mon[fri_mon.month != 9]).dropna()
oth_e = oth[[not (d[pos[x] + 1].day >= 22) for x in oth.index]]
p0 = float((oth_e > 0).mean())
st = summarize(v)
print(f"  clean (vol<=5x, day<22): n={len(v)} mean {st['mean_pct']:+.3f}% med {st['median_pct']:+.3f}% up {w}/{len(v)} t {st['t']:+.2f}")
print(f"  other months, Mondays before the 22nd: n={len(oth_e)} mean {100*oth_e.mean():+.3f}% up {100*p0:.1f}%; sign p {sign_test(w, len(v), p0):.4f}")
diff = 100 * (v.mean() - oth_e.mean())
se = 100 * np.sqrt(v.var(ddof=1) / len(v) + oth_e.var(ddof=1) / len(oth_e))
print(f"  diff {diff:+.3f}pp t {diff/se:+.2f}")
print("  era:", [(e['label'], e['n'], round(e['mean_pct'], 2), round(e['hit'], 1)) for e in era_split(clean.index, v)])
print("  concentration:", cluster_note(clean.index, v))
for m in range(1, 13):
    s = nxt.reindex(fri_mon[fri_mon.month == m]).dropna()
    s = s[[d[pos[x] + 1].day < 22 for x in s.index]]
    print(f"     month {m:2} early Mondays n={len(s):3} mean {100*s.mean():+6.2f}% up {100*(s>0).mean():5.1f}%")
# Is it Mondays, or September sessions in general (early month)?
sep_all = d[(d.month == 9) & (d.day < 22)][:-1]
s = nxt.reindex(sep_all).dropna()
print(f"  every Sept session before the 22nd (h1): n={len(s)} mean {100*s.mean():+.3f}% up {100*(s>0).mean():.1f}%")

print("\n=== B. Friday VIX -10% -> Monday, era-matched base ===")
vix = px['^VIX']['Close'].reindex(d)
r1 = vix.pct_change(fill_method=None)
vn = vix.shift(-1) / vix - 1
for lab, lo, hi in [('pre-2018', '1999-01-01', '2017-12-31'), ('2018+', '2018-01-01', '2100-01-01')]:
    fm = fri_mon[(fri_mon >= lo) & (fri_mon <= hi)]
    base = vn.reindex(fm).dropna()
    sel = fm[(r1.reindex(fm) <= -0.10).values]
    v = vn.reindex(sel).dropna()
    w = int((v > 0).sum())
    p0 = float((base > 0).mean())
    print(f"  {lab}: all Mondays up {100*p0:.1f}% med {100*base.median():+.2f}% (n={len(base)}) | after -10% Friday up {w}/{len(v)} med {100*v.median():+.2f}% mean {100*v.mean():+.2f}%"
          f" | P(<= {w} up | base) {1 - sign_test(w + 1, len(v), p0):.4f}")
fm = fri_mon
for thr in (-0.06, -0.08, -0.10, -0.12):
    sel = fm[(r1.reindex(fm) <= thr).values]
    v = vn.reindex(sel).dropna()
    v18 = v[v.index >= '2018-01-01']
    print(f"  Friday <= {int(100*thr)}%: all-era up {int((v>0).sum())}/{len(v)} ({100*(v>0).mean():.1f}%) | 2018+ up {int((v18>0).sum())}/{len(v18)} ({100*(v18>0).mean():.1f}%)")

print("\n=== C. TLT/IEF surge anchors, NaN removed ===")
fvx = px['^FVX']['Close'].reindex(d).ffill(limit=2)
ch5 = fvx - fvx.shift(5)
rank5 = ch5.rolling(252, min_periods=200).apply(lambda x: (x[:-1] < x[-1]).mean() * 100, raw=True)
fomc = pd.DatetimeIndex(sorted(load_events(['fomc_decision'])['date'].unique())).intersection(d)
k3 = pd.DatetimeIndex([d[pos[f] - 3] for f in fomc if pos[f] >= 3 and pos[f] + 2 < len(d)])
surge = k3[(rank5.reindex(k3) >= 90).values]
rest = k3.difference(surge)
for t in ['TLT', 'IEF']:
    s = px[t]['Close'].reindex(d)
    f3 = s.shift(-3) / s - 1
    a, b = f3.reindex(surge).dropna(), f3.reindex(rest).dropna()
    w = int((a > 0).sum())
    p0 = float((b > 0).mean())
    diff = 100 * (a.mean() - b.mean())
    se = 100 * np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    print(f"  {t} h3: surge up {w}/{len(a)} mean {100*a.mean():+.3f}% | rest up {int((b>0).sum())}/{len(b)} ({100*p0:.1f}%) mean {100*b.mean():+.3f}% "
          f"t(rest vs 0) {b.mean()/(b.std(ddof=1)/np.sqrt(len(b))):+.2f} | diff {diff:+.3f}pp t {diff/se:+.2f} | P(<= {w} up) {1 - sign_test(w + 1, len(a), p0):.4f}")
# non-midterm split for the S&P leg
spx = px['SPY']['Close'].reindex(d)
