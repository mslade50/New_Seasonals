"""C6 round 1 -- commodity EM ripping while China breaks.

Rule 9: legs before the spread. Rule 8: reference-class max-of-K for the
single-country pick. Every percentile recomputed PIT on valid sessions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

CLASS = ['EWZ', 'EWJ', 'EWY', 'EWT', 'INDA', 'FXI', 'EEM', 'EFA', 'EWW', 'KWEB']
px = close_panel(CLASS + ['DBC', 'SPY']).dropna(subset=['EWZ', 'FXI', 'EEM', 'EFA', 'SPY'])
print('panel', px.index[0].date(), '..', px.index[-1].date(), len(px))

r5 = {t: pct_rank(px[t], 5, 252) for t in px.columns}
r21 = {t: pct_rank(px[t], 21, 252) for t in px.columns}
print('LIVE 09-08: EWZ 5d rank %.1f  z10 %.2f | FXI 21d rank %.1f  FXI 1d %+.2f%%'
      % (r5['EWZ'].iloc[-1], zscore(px['EWZ'], 10).iloc[-1], r21['FXI'].iloc[-1],
         100 * (px['FXI'].iloc[-1] / px['FXI'].iloc[-2] - 1)))

cond = (r5['EWZ'] >= 95) & (r21['FXI'] <= 30)
d_all = px.index[cond.values]
print('\nRARITY: %d days, episodes(gap=10) %d, years %s'
      % (int(cond.sum()), len(declusters(d_all, 10, px.index)),
         sorted(set(pd.DatetimeIndex(declusters(d_all, 10, px.index)).year))))

print('\n### (a) LEGS ALONE, episode mean %% (gap=h) and edge vs own all-day drift')
LEGS = {'EWZ long': [('EWZ', 1.0)], 'FXI short': [('FXI', -1.0)],
        'EEM long': [('EEM', 1.0)], 'EFA short': [('EFA', -1.0)],
        'SPY long': [('SPY', 1.0)]}
for name, legs in LEGS.items():
    row = {'leg': name}
    for h in (1, 2, 3, 5, 10):
        r = horizon_scan(px, d_all, legs, hs=(h,), lag=1)[0]
        if r['n']:
            row['h%d' % h] = round(r['mean_pct'], 3)
            row['h%dedge' % h] = round(r['edge_pct'], 3)
            row['h%dN' % h] = r['n']
    print(row)

print('\n### (b) SPREADS, only after the legs')
SPR = {'EWZ - FXI': [('EWZ', 1.0), ('FXI', -1.0)],
       'EEM - EFA': [('EEM', 1.0), ('EFA', -1.0)],
       'FXI - EWZ (reversal side)': [('FXI', 1.0), ('EWZ', -1.0)]}
for name, legs in SPR.items():
    row = {'spread': name}
    for h in (1, 2, 3, 5, 10):
        r = horizon_scan(px, d_all, legs, hs=(h,), lag=1)[0]
        if r['n']:
            row['h%d' % h] = round(r['mean_pct'], 3)
            row['h%dedge' % h] = round(r['edge_pct'], 3)
            row['h%dN' % h] = r['n']
    print(row)

print('\n### (c) GATE ATTRIBUTION -- complement test first')
CELLS = {'COND EWZ5>=95 & FXI21<=30': cond,
         'PARENT EWZ 5d rank>=95 alone': (r5['EWZ'] >= 95),
         'COMPLEMENT EWZ5>=95 & FXI21>30': (r5['EWZ'] >= 95) & (r21['FXI'] > 30),
         'PARENT FXI 21d rank<=30 alone': (r21['FXI'] <= 30),
         'COMPLEMENT FXI21<=30 & EWZ5<95': (r21['FXI'] <= 30) & (r5['EWZ'] < 95)}
for vn in ('EWZ long', 'FXI short', 'EWZ - FXI'):
    legs = LEGS.get(vn) or SPR[vn]
    rows = []
    for lbl, m in CELLS.items():
        rec = {'cell': lbl}
        for h in (3, 5, 10):
            r = horizon_scan(px, px.index[m.values], legs, hs=(h,), lag=1)[0]
            if r['n']:
                rec['h%d' % h] = round(r['mean_pct'], 3)
                rec['h%dN' % h] = r['n']
        rows.append(rec)
    show(rows, vn)

print('\n### (d) REFERENCE CLASS: same cell with each country as the long leg')
print('  (X 5d rank>=95 & FXI 21d rank<=30) -> long X, h=5, episodes gap=5')
res = []
for t in CLASS:
    if t == 'FXI':
        continue
    s = px[t].dropna()
    m = (pct_rank(px[t], 5, 252) >= 95) & (r21['FXI'] <= 30)
    d = px.index[m.fillna(False).values]
    r = horizon_scan(px, d, [(t, 1.0)], hs=(5,), lag=1)[0]
    if r['n']:
        res.append((t, r['n'], r['mean_pct'], r['edge_pct']))
res.sort(key=lambda x: -x[3])
for i, (t, n, m_, e) in enumerate(res, 1):
    print('   %2d. %-5s N=%2d mean %+0.3f%% edge %+0.3f pp %s' % (i, t, n, m_, e, '<== EWZ' if t == 'EWZ' else ''))
k = [i for i, r_ in enumerate(res, 1) if r_[0] == 'EWZ']
print('   EWZ rank %s of %d  -> max-of-K permutation P >= %.4f'
      % (k, len(res), (k[0] / len(res)) if k else np.nan))
