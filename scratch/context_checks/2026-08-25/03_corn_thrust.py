"""Corn: a 52-week-high close on a top-5% five-day run.

Tonight ZC=F closed +6.66% at a 252-day high, 5d +13.17% and 21d +16.05%
(both the 100th percentile of the trailing year), z10 2.95, 18.6% above its
200-day mean. What is that cell, how rare, and what follows?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from pitch_lab import (load_prices, fwd_ret, declusters, local_control,
                       summarize, era_split, sign_test, cluster_note, show)

px = load_prices(['ZC=F', 'ZW=F', 'ZS=F'])
d = px['ZC=F'].copy()
c = d['Close'].astype(float)
print('ZC=F', d.index[0].date(), '->', d.index[-1].date(), len(d), 'bars')

r5 = c.pct_change(5, fill_method=None)
r21 = c.pct_change(21, fill_method=None)
rank5 = r5.rolling(252).rank(pct=True) * 100
rank21 = r21.rolling(252).rank(pct=True) * 100
hi252 = c.rolling(252).max()
sma200 = c.rolling(200).mean()
at_high = c >= hi252 * 0.9999

print('\ntonight: close %.2f  5d %+.2f%% (rank %.1f)  21d %+.2f%% (rank %.1f)'
      % (c.iloc[-1], 100 * r5.iloc[-1], rank5.iloc[-1], 100 * r21.iloc[-1], rank21.iloc[-1]))
print('        at 252d high: %s   above 200d mean by %+.1f%%'
      % (bool(at_high.iloc[-1]), 100 * (c.iloc[-1] / sma200.iloc[-1] - 1)))

# --- the cell: 52w high close AND 5d return in the top 5% of its year
mask = (at_high & (rank5 >= 95)).fillna(False)
trig = c.index[mask.values]
ep = declusters(trig, 10, c.index)
print('\nCELL 52w-high close + top-5%% 5d run: %d sessions, %d episodes at 10td'
      % (len(trig), len(ep)))
print('by year:', dict(pd.Series(1, index=ep).groupby(ep.year).size()))
print('by month:', dict(pd.Series(1, index=ep).groupby(ep.month).size()))
print('tonight qualifies:', bool(mask.iloc[-1]))
print('prior episodes:', [str(x.date()) for x in ep[:-1]][-12:])

rows = []
for h in (1, 3, 5, 10, 21, 42):
    f = fwd_ret(c, h)
    v = f.reindex(ep[:-1] if mask.iloc[-1] else ep).dropna()
    r = summarize(v.values, 'h%d' % h)
    r['sign_p'] = round(sign_test(int((v.values > 0).sum()), len(v)), 4)
    rows.append(r)
    rows.append(summarize(f.dropna().values, '  CTRL all days h%d' % h))
    lc = local_control(c.index, ep, 126)
    rows.append(summarize(f.reindex(lc).dropna().values, '  CTRL local+-126 h%d' % h))
show(rows, 'corn 52w high + top-5%% 5d -> forward')

use = ep[:-1] if mask.iloc[-1] else ep
for h in (5, 21):
    v = fwd_ret(c, h).reindex(use).dropna()
    if len(v) >= 4:
        print('\nh%d concentration: %s' % (h, cluster_note(v.index, v.values)))
        print('h%d era: %s' % (h, era_split(v.index, v.values)[-1]))

print('\nper-episode:')
f5, f10, f21 = fwd_ret(c, 5), fwd_ret(c, 10), fwd_ret(c, 21)
for x in use:
    print('  %s  5d-in %+6.2f%%  ->  h5 %+7.2f%%  h10 %+7.2f%%  h21 %+7.2f%%'
          % (x.date(), 100 * r5.get(x, np.nan), 100 * f5.get(x, np.nan),
             100 * f10.get(x, np.nan), 100 * f21.get(x, np.nan)))

# --- how extended is corn vs its own history, and how rare is a 21d +16%
print('\n--- context on the magnitude ---')
print('21d return percentile in FULL history: %.2f  (%d of %d sessions >= +16.05%%)'
      % (100 * (r21 <= r21.iloc[-1]).mean(), int((r21 >= r21.iloc[-1]).sum()), r21.notna().sum()))
big = c.index[(r21 >= 0.16).fillna(False).values]
bigep = declusters(big, 21, c.index)
print('21d >= +16%%: %d sessions, %d episodes at 21td, years %s'
      % (len(big), len(bigep), sorted(set(bigep.year))))
print('above-200d-mean percentile: %.1f'
      % (100 * ((c / sma200) <= (c / sma200).iloc[-1]).mean()))

# --- August specifically
aug = mask & (c.index.month == 8)
print('\nsame cell in AUGUST: %d sessions, dates %s'
      % (int(aug.sum()), [str(x.date()) for x in c.index[aug.values]]))
