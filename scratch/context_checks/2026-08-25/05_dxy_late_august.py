"""The dollar in the last week of August.

Engine seasonal_doy cell for Aug 26: DX-Y.NYB h5 is 21-5 up, mean +0.50%,
sign p 0.0012. That is the strongest clean seasonal cell in tonight's sweep.
2026 enters it with DXY's 21d return in the 0.8th percentile of its trailing
year, so the interaction is the question worth asking.

The doy cell WAS found by the sweep, so it owes multiplicity a correction:
BH crit p tonight is 0.0038.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from pitch_lab import (load_prices, fwd_ret, summarize, era_split,
                       sign_test, cluster_note, show)

px = load_prices(['DX-Y.NYB', 'EURUSD=X', 'GC=F'])
c = px['DX-Y.NYB']['Close'].astype(float)
print('DX-Y.NYB', c.index[0].date(), '->', c.index[-1].date(), len(c), 'bars')

r21 = c.pct_change(21, fill_method=None)
rank21 = r21.rolling(252).rank(pct=True) * 100
print('tonight: close %.2f, 21d %+.2f%% (rank %.1f), 5d %+.2f%%'
      % (c.iloc[-1], 100 * r21.iloc[-1], rank21.iloc[-1], 100 * c.pct_change(5).iloc[-1]))

# --- the doy cell: one pick per prior year at the trading day matching Aug 26
target = pd.Timestamp('2026-08-26')
anch = []
for yr in range(1999, 2026):
    want = pd.Timestamp(year=yr, month=target.month, day=target.day)
    # the session before the matching session (anchor convention: h1 = that day)
    prior = c.index[c.index < want]
    if len(prior) == 0:
        continue
    anch.append(prior[-1])
anch = pd.DatetimeIndex(anch)
print('\nanchors (session before Aug 26 each year): %d, %s .. %s'
      % (len(anch), anch[0].date(), anch[-1].date()))

rows = []
for h in (1, 3, 5, 10):
    f = fwd_ret(c, h)
    v = f.reindex(anch).dropna()
    r = summarize(v.values, 'DXY h%d' % h)
    r['sign_p'] = round(sign_test(int((v.values > 0).sum()), len(v)), 4)
    rows.append(r)
    rows.append(summarize(f.dropna().values, '  CTRL all days h%d' % h))
    aug = f[(f.index.month == 8)].dropna()
    rows.append(summarize(aug.values, '  CTRL all August h%d' % h))
show(rows, 'dollar: the Aug-26 trading day of year')

f5 = fwd_ret(c, 5)
v5 = f5.reindex(anch).dropna()
print('\nh5 record %d-%d up, sign p %.4f'
      % (int((v5.values > 0).sum()), int((v5.values <= 0).sum()),
         sign_test(int((v5.values > 0).sum()), len(v5))))
print('h5 concentration:', cluster_note(v5.index, v5.values))
print('h5 era:', era_split(v5.index, v5.values)[-1])
pre = v5[v5.index < '2018-01-01']
print('h5 pre-2018: n=%d mean %+.3f%% record %d-%d'
      % (len(pre), 100 * pre.mean(), int((pre > 0).sum()), int((pre <= 0).sum())))

print('\nper-year h5 from the anchor:')
for d in v5.index:
    print('  %s  entry 21d rank %5.1f  ->  h5 %+6.2f%%  %s'
          % (d.date(), rank21.get(d, np.nan), 100 * v5.get(d),
             'MIDTERM' if d.year % 4 == 2 else ''))

# --- the interaction: entering the window weak
rk = np.array([rank21.get(d, np.nan) for d in v5.index])
ok = ~np.isnan(rk)
low = ok & (rk < 30)
print('\nentered with 21d rank < 30 (2026 enters at 0.8): n=%d' % low.sum())
print('  ', summarize(v5.values[low], 'h5 weak entry'))
print('   sign p %.4f' % sign_test(int((v5.values[low] > 0).sum()), int(low.sum())))
print('entered with 21d rank >= 30: n=%d' % (ok & ~low).sum())
print('  ', summarize(v5.values[ok & ~low], 'h5 other'))

mid = np.array([d.year % 4 == 2 for d in v5.index])
print('\nmidterm years n=%d: %s' % (mid.sum(), summarize(v5.values[mid], 'h5 midterm')))
print('  sign p %.4f' % sign_test(int((v5.values[mid] > 0).sum()), int(mid.sum())))
