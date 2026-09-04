"""Reproduce the engine's DXY seasonal_doy h5 cell and stress its alignment.

05 anchored on the CALENDAR date (session before Aug 26) and got 15-11.
The engine anchors on the TRADING day-of-year nearest asof, +/-2. If the
cell's 21-5 record depends on which alignment you pick, it is an artifact
of the alignment and does not go in the brief.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from pitch_lab import load_prices, sign_test
from seasonal_edge import seasonal_window_returns

px = load_prices(['DX-Y.NYB', 'HYG', 'TLT'])
asof = pd.Timestamp('2026-08-25')

for tk in ['DX-Y.NYB', 'HYG', 'TLT']:
    df = px[tk]
    print('\n======', tk)
    for h in (1, 5):
        st = seasonal_window_returns(df, asof, h)
        if not st or st.get('insufficient'):
            print('  h%d insufficient' % h); continue
        rets = np.array(st['rets']) if 'rets' in st else None
        print('  h%d: n=%d mean %+.3f%% median %+.3f%% up %d down %d sign_p %.4f'
              % (h, st['n'], 100 * st['mean'], 100 * st['median'],
                 st['n_up'], st['n_down'],
                 sign_test(max(st['n_up'], st['n_down']), st['n'])))
        if rets is not None and len(rets):
            yrs = st.get('years', [])
            order = np.argsort(-np.abs(rets))
            top = order[:2]
            print('     top2 |ret| years %s = %+.2fpp of %+.2fpp total (%.0f%%)'
                  % ([yrs[i] for i in top] if len(yrs) == len(rets) else 'n/a',
                     100 * rets[top].sum(), 100 * rets.sum(),
                     100 * rets[top].sum() / rets.sum() if rets.sum() else float('nan')))
            if len(yrs) == len(rets):
                post = np.array([y >= 2018 for y in yrs])
                print('     2018+: n=%d mean %+.3f%% up %d down %d'
                      % (post.sum(), 100 * rets[post].mean(),
                         int((rets[post] > 0).sum()), int((rets[post] <= 0).sum())))
                print('     per year:', ', '.join('%s %+.2f%%' % (y, 100 * r)
                                                  for y, r in zip(yrs, rets)))

# --- alignment stress: shift asof by +/-1 and +/-2 sessions
print('\n\n--- alignment stress on DX-Y.NYB h5 (does the record survive?) ---')
df = px['DX-Y.NYB']
idx = df['Close'].dropna().index
p = int(np.searchsorted(idx.values, np.datetime64(asof), side='right')) - 1
for shift in (-3, -2, -1, 0, 1):
    if p + shift < 0:
        continue
    a = idx[p + shift]
    st = seasonal_window_returns(df, a, 5)
    if st and not st.get('insufficient'):
        print('  asof %s (shift %+d): n=%d mean %+.3f%% up %d down %d sign_p %.4f'
              % (a.date(), shift, st['n'], 100 * st['mean'], st['n_up'],
                 st['n_down'], sign_test(max(st['n_up'], st['n_down']), st['n'])))
for tol in (0, 1, 2, 3):
    st = seasonal_window_returns(df, asof, 5, doy_tol=tol)
    if st and not st.get('insufficient'):
        print('  doy_tol %d: n=%d mean %+.3f%% up %d down %d sign_p %.4f'
              % (tol, st['n'], 100 * st['mean'], st['n_up'], st['n_down'],
                 sign_test(max(st['n_up'], st['n_down']), st['n'])))
