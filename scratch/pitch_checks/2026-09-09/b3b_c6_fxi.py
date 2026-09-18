"""C6 round 1b -- the pair's whole edge is the SHORT FXI leg (h=5 +1.509%
against EWZ long -0.699%). Does short FXI stand alone, or is it just short
global beta wearing a China label?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

px = close_panel(['EWZ', 'FXI', 'EEM', 'EFA', 'SPY', 'DBC']).dropna(
    subset=['EWZ', 'FXI', 'EEM', 'EFA', 'SPY'])
r5 = pct_rank(px['EWZ'], 5, 252)
r21 = pct_rank(px['FXI'], 21, 252)
cond = (r5 >= 95) & (r21 <= 30)

variants = {
    'EWZ5>=90': (pct_rank(px['EWZ'], 5, 252) >= 90) & (r21 <= 30),
    'EWZ5>=98': (pct_rank(px['EWZ'], 5, 252) >= 98) & (r21 <= 30),
    'FXI21<=20': (r5 >= 95) & (r21 <= 20),
    'FXI21<=40': (r5 >= 95) & (r21 <= 40),
    'EWZ 10d rank>=95': (pct_rank(px['EWZ'], 10, 252) >= 95) & (r21 <= 30),
    'EWZ5 lookback 126': (pct_rank(px['EWZ'], 5, 126) >= 95) & (r21 <= 30),
    'PARENT FXI21<=30 alone': (r21 <= 30),
    'COMPLEMENT FXI21<=30 & EWZ5<95': (r21 <= 30) & (r5 < 95),
}
battery(px, cond, [('FXI', -1.0)], 5, 'C6 SHORT FXI h=5 | EWZ 5d rank>=95 & FXI 21d rank<=30',
        cost_bps=3.0, variants=variants, lag=1, min_gap=10)

ret = vehicle_ret(px, [('FXI', -1.0)], 5, 1)
sig = px.index[cond.values & ret.notna().values]
epi = declusters(sig, 10, px.index)
v = ret.loc[epi].values
yrs = pd.DatetimeIndex(epi).year
show([summarize(v[yrs % 4 == 2], 'midterm'), summarize(v[yrs % 4 != 2], 'non-midterm')],
     'cycle split')
by = pd.DataFrame({'yr': yrs, 'r': 100 * v}).groupby('yr')['r'].agg(['count', 'sum', 'mean'])
print('\nper-year (pp):'); print(by.round(2).to_string())
order = np.argsort(-v)
for k in (1, 2, 3):
    keep = np.ones(len(v), bool); keep[order[:k]] = False
    print('drop-best-%d: N=%d mean %+0.3f%% hit %.0f%%'
          % (k, keep.sum(), 100 * v[keep].mean(), 100 * (v[keep] > 0).mean()))

# is it just short global beta?
print('\n### beta decomposition, h=5 short leg')
for pair in (('FXI', 'SPY'), ('FXI', 'EEM'), ('FXI', 'EFA')):
    y = fwd_lag(px[pair[0]], 5, 1); x = fwd_lag(px[pair[1]], 5, 1)
    ok = y.notna() & x.notna()
    b = np.polyfit(x[ok].values, y[ok].values, 1)[0]
    resid = -(y - b * x)   # short FXI, hedged with the benchmark
    e = epi.intersection(resid.dropna().index)
    print('  short %s hedged by %s (beta %.2f): COND %+0.3f%% (N=%d) vs all-day %+0.3f%%'
          % (pair[0], pair[1], b, 100 * resid.loc[e].mean(), len(e), 100 * resid.dropna().mean()))

print('\n### everything-falls check: the same cell on other shorts, h=5 episodes')
for t in ('SPY', 'EFA', 'EEM', 'EWZ', 'FXI'):
    r = horizon_scan(px, sig, [(t, -1.0)], hs=(5,), lag=1, min_gap=10)[0]
    print('  short %-4s N=%d mean %+0.3f%% edge %+0.3f pp' % (t, r['n'], r['mean_pct'], r['edge_pct']))
