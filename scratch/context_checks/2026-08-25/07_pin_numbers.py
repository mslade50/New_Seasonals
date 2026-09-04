"""Pin every number that goes in the 2026-08-25 brief, in one place."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from pitch_lab import (load_prices, load_events, fwd_ret, summarize, sign_test,
                       declusters, local_control, era_split, cluster_note)

px = load_prices(['HYG', 'IWM', '^RUT', '^GSPC', 'ZC=F', 'ZW=F', 'LE=F', 'KC=F', 'CT=F'])

print('#' * 70)
print('# 1. IWM on the symposium session')
c = px['IWM']['Close'].astype(float)
ev = pd.DatetimeIndex(load_events(['jackson_hole'])['date'])
dates = c.index
posn = pd.Series(range(len(dates)), index=dates)
anch = []
for e in ev:
    nxt = dates[dates >= e]
    if len(nxt):
        p = posn[nxt[0]] - 1
        if p >= 0:
            anch.append(dates[p])
anch = pd.DatetimeIndex(anch)
v = fwd_ret(c, 1).reindex(anch).dropna()
up = int((v.values > 0).sum())
print('   n=%d  %d-%d up  hit %.1f%%  mean %+.3f%%  t %.2f  sign p %.4f'
      % (len(v), up, len(v) - up, 100 * up / len(v), 100 * v.mean(),
         summarize(v.values)['t'], sign_test(up, len(v))))
print('   worst %+.2f%% (%s)  best %+.2f%% (%s)'
      % (100 * v.min(), v.idxmin().date(), 100 * v.max(), v.idxmax().date()))
lc = local_control(c.index, v.index, 126)
print('   local +-126 baseline %+.3f%%' % (100 * fwd_ret(c, 1).reindex(lc).dropna().mean()))
print('   concentration:', cluster_note(v.index, v.values))
post = v[v.index.year >= 2018]
print('   2018+: n=%d mean %+.3f%% %d-%d up'
      % (len(post), 100 * post.mean(), int((post > 0).sum()), int((post <= 0).sum())))
mid = v[[d.year % 4 == 2 for d in v.index]]
print('   midterm: n=%d mean %+.3f%% %d-%d up  sign p %.4f'
      % (len(mid), 100 * mid.mean(), int((mid > 0).sum()), int((mid <= 0).sum()),
         sign_test(int((mid > 0).sum()), len(mid))))
non = v[[d.year % 4 != 2 for d in v.index]]
print('   non-midterm: n=%d mean %+.3f%% %d-%d up'
      % (len(non), 100 * non.mean(), int((non > 0).sum()), int((non <= 0).sum())))
r = px['^RUT']['Close'].astype(float)
vr = fwd_ret(r, 1).reindex(anch).dropna()
print('   ^RUT cross-check: %d-%d up, mean %+.3f%%'
      % (int((vr > 0).sum()), int((vr <= 0).sum()), 100 * vr.mean()))

print('#' * 70)
print('# 2. ^GSPC three-session run-in (anchor = today, h3 = symposium close)')
s = px['^GSPC']['Close'].astype(float)
dts = s.index
pn = pd.Series(range(len(dts)), index=dts)
a3 = []
for e in ev:
    nxt = dts[dts >= e]
    if len(nxt):
        p = pn[nxt[0]] - 3
        if p >= 0:
            a3.append(dts[p])
a3 = pd.DatetimeIndex(a3)
v3 = fwd_ret(s, 3).reindex(a3).dropna()
up3 = int((v3.values > 0).sum())
print('   n=%d  %d-%d up  mean %+.3f%%  t %.2f  sign p %.4f'
      % (len(v3), up3, len(v3) - up3, 100 * v3.mean(),
         summarize(v3.values)['t'], sign_test(up3, len(v3))))
print('   concentration:', cluster_note(v3.index, v3.values))
p18 = v3[v3.index.year >= 2018]
print('   2018+: n=%d mean %+.3f%% %d-%d up'
      % (len(p18), 100 * p18.mean(), int((p18 > 0).sum()), int((p18 <= 0).sum())))
m3 = v3[[d.year % 4 == 2 for d in v3.index]]
print('   midterm: n=%d mean %+.3f%% %d-%d up' % (len(m3), 100 * m3.mean(),
      int((m3 > 0).sum()), int((m3 <= 0).sum())))
print('   all-days h3 baseline %+.3f%%' % (100 * fwd_ret(s, 3).dropna().mean()))
v1 = fwd_ret(s, 1).reindex(a3).dropna()
print('   h1 (tomorrow alone): n=%d %d-%d up mean %+.3f%% sign p %.4f'
      % (len(v1), int((v1 > 0).sum()), int((v1 <= 0).sum()), 100 * v1.mean(),
         sign_test(int((v1 > 0).sum()), len(v1))))

print('#' * 70)
print('# 3. HYG, the last seven calendar days of August')
h = px['HYG']['Close'].astype(float)
f5 = fwd_ret(h, 5)
w = f5[(f5.index.month == 8) & (f5.index.day >= 20)].dropna()
print('   n=%d overlapping sessions from %d Augusts (%s)'
      % (len(w), w.index.year.nunique(), sorted(set(w.index.year))[0]))
print('   mean %+.3f%%  hit %.1f%%  t %.2f  median %+.3f%%'
      % (100 * w.mean(), 100 * (w > 0).mean(), summarize(w.values)['t'], 100 * w.median()))
print('   unconditional h5 %+.3f%% hit %.1f%%'
      % (100 * f5.dropna().mean(), 100 * (f5.dropna() > 0).mean()))
print('   all-August h5 %+.3f%%' % (100 * f5[f5.index.month == 8].dropna().mean()))
byyr = w.groupby(w.index.year).mean()
print('   per-August mean: %s' % ', '.join('%d %+.2f%%' % (y, 100 * x) for y, x in byyr.items()))
print('   Augusts positive on the window mean: %d of %d'
      % (int((byyr > 0).sum()), len(byyr)))
print('   sign p on the per-year record %.4f' % sign_test(int((byyr > 0).sum()), len(byyr)))
pre = byyr[byyr.index < 2018]; pos2 = byyr[byyr.index >= 2018]
print('   pre-2018 %+.3f%% (%d of %d up) | 2018+ %+.3f%% (%d of %d up)'
      % (100 * pre.mean(), int((pre > 0).sum()), len(pre),
         100 * pos2.mean(), int((pos2 > 0).sum()), len(pos2)))
midy = byyr[[y % 4 == 2 for y in byyr.index]]
print('   midterm Augusts: %d of %d up, mean %+.3f%%'
      % (int((midy > 0).sum()), len(midy), 100 * midy.mean()))

print('#' * 70)
print('# 4. corn, 52w-high close on a top-5% five-day run')
cc = px['ZC=F']['Close'].astype(float)
r5 = cc.pct_change(5, fill_method=None)
rank5 = r5.rolling(252).rank(pct=True) * 100
at_high = cc >= cc.rolling(252).max() * 0.9999
mask = (at_high & (rank5 >= 95)).fillna(False)
ep = declusters(cc.index[mask.values], 10, cc.index)
prior = ep[:-1]
for hz in (1, 5, 10, 21, 42):
    vv = fwd_ret(cc, hz).reindex(prior).dropna()
    lcx = local_control(cc.index, ep, 126)
    print('   h%-2d n=%d mean %+.3f%% hit %.1f%% t %.2f sign p %.4f | local %+.3f%%'
          % (hz, len(vv), 100 * vv.mean(), 100 * (vv > 0).mean(),
             summarize(vv.values)['t'], sign_test(int((vv > 0).sum()), len(vv)),
             100 * fwd_ret(cc, hz).reindex(lcx).dropna().mean()))
v5c = fwd_ret(cc, 5).reindex(prior).dropna()
print('   h5 concentration:', cluster_note(v5c.index, v5c.values))
p18c = v5c[v5c.index.year >= 2018]
print('   h5 2018+: n=%d mean %+.3f%% %d-%d up'
      % (len(p18c), 100 * p18c.mean(), int((p18c > 0).sum()), int((p18c <= 0).sum())))
augdays = cc.index[(mask & (cc.index.month == 8)).values]
print('   August occurrences: %s' % [str(d.date()) for d in augdays])
print('   -> prior August YEARS: %s' % sorted({d.year for d in augdays if d.year != 2026}))
print('   tonight: 5d %+.2f%% (rank %.1f), 21d %+.2f%%, 200d dist %+.1f%%'
      % (100 * r5.iloc[-1], rank5.iloc[-1], 100 * cc.pct_change(21).iloc[-1],
         100 * (cc.iloc[-1] / cc.rolling(200).mean().iloc[-1] - 1)))

print('#' * 70)
print('# 5. the roll / bad-bar note')
for t in ['LE=F']:
    d = px[t]
    g = (d['Open'] / d['Close'].shift(1) - 1) * 100
    yrs = []
    for yr in range(2001, 2027):
        w2 = g[(g.index >= f'{yr}-08-15') & (g.index <= f'{yr}-09-05')].dropna()
        if len(w2):
            i = w2.abs().idxmax()
            if abs(w2.loc[i]) > 3:
                yrs.append((yr, str(i.date()), round(float(w2.loc[i]), 2)))
    print('   %s years with a >3%% gap in Aug15-Sep05: %d' % (t, len(yrs)))
    print('   ', yrs)
ct = px['CT=F']
print('   CT=F last bar: O %.2f H %.2f L %.2f C %.2f'
      % (ct['Open'].iloc[-1], ct['High'].iloc[-1], ct['Low'].iloc[-1], ct['Close'].iloc[-1]))
kc = px['KC=F']
print('   KC=F gap %.2f%%, session %.2f%%'
      % (100 * (kc['Open'].iloc[-1] / kc['Close'].iloc[-2] - 1),
         100 * (kc['Close'].iloc[-1] / kc['Close'].iloc[-2] - 1)))
