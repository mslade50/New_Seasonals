"""IWM on the symposium session: 21-5 up, sign p 0.001. Does it survive?

Post-selection warning: I looked at 7 subjects x 4 horizons plus a k1 pass
before this surfaced. Treat as suggestive and check era, midterm and
concentration before writing anything.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from pitch_lab import (load_prices, load_events, fwd_ret, summarize,
                       era_split, sign_test, cluster_note, local_control, show)

TK = ['IWM', '^GSPC', '^RUT']
px = load_prices(TK)
c = {t: px[t]['Close'].astype(float) for t in TK}
ev = pd.DatetimeIndex(load_events(['jackson_hole'])['date'])
dates = pd.DatetimeIndex(sorted(c['IWM'].index))
pos = pd.Series(range(len(dates)), index=dates)

anch, sess = [], []
for e in ev:
    nxt = dates[dates >= e]
    if len(nxt) == 0:
        continue
    p = pos[nxt[0]] - 1
    if p >= 0:
        anch.append(dates[p]); sess.append(nxt[0])
anch = pd.DatetimeIndex(anch); sess = pd.DatetimeIndex(sess)

for t in TK:
    f = fwd_ret(c[t], 1)
    v = f.reindex(anch).dropna()
    print('\n=== %s, the symposium session ===' % t)
    print(summarize(v.values, t))
    print('record %d-%d up, sign p %.4f'
          % (int((v.values > 0).sum()), int((v.values <= 0).sum()),
             sign_test(int((v.values > 0).sum()), len(v))))
    print('concentration:', cluster_note(v.index, v.values))
    print('era:', era_split(v.index, v.values)[-1])
    lc = local_control(c[t].index, v.index, 126)
    print('CTRL local +-126:', summarize(f.reindex(lc).dropna().values, 'local'))
    print('CTRL all days   :', summarize(f.dropna().values, 'all'))
    mid = np.array([d.year % 4 == 2 for d in v.index])
    if mid.sum():
        print('midterm n=%d: %s' % (mid.sum(), summarize(v.values[mid], 'midterm')))
        print('  sign p %.4f' % sign_test(int((v.values[mid] > 0).sum()), int(mid.sum())))
        print('non-midterm n=%d: %s' % ((~mid).sum(), summarize(v.values[~mid], 'non')))

print('\n--- IWM per symposium session ---')
f = fwd_ret(c['IWM'], 1)
fs = fwd_ret(c['^GSPC'], 1)
for a, s in zip(anch, sess):
    if a not in f.index:
        continue
    print('  %s symposium %s  IWM %+6.2f%%  SPX %+6.2f%%  %s'
          % (a.date(), s.date(), 100 * f.get(a, np.nan), 100 * fs.get(a, np.nan),
             'MIDTERM' if a.year % 4 == 2 else ''))
