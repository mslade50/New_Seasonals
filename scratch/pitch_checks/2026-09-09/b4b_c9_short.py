"""C9 round 1b -- the premise is false today (PIT 84.5 tape / 42.3 sectors,
gate wants >=90). Is there a SHORT cell worth parking, and is the SPY-near-high
gate a filter or an anti-filter? Complement test.
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

TAPE = json.load(open(Path(__file__).resolve().parents[3] / 'data' / 'pitch_tape.json'))['tickers']
idxpx = close_panel(['SPY', 'IWM']).dropna()
spy = idxpx['SPY']
spy_dd = spy / rolling_on_valid(spy, lambda x: x.rolling(252).max()) - 1.0

px = close_panel(TAPE).reindex(spy.index)
R = pd.DataFrame({t: pct_rank(px[t], 21, 252) for t in px.columns
                  if px[t].notna().sum() >= 273})
frac = (R <= 2.5).sum(axis=1) / R.notna().sum(axis=1).replace(0, np.nan)
pit = rolling_on_valid(frac, lambda x: x.rolling(252).rank(pct=True) * 100.0)
print('today frac %.2f%%  PIT %.1f' % (100 * frac.iloc[-1], pit.iloc[-1]))
print('frac needed for PIT>=90 today: %.2f%% => %d of %d names'
      % (100 * frac.dropna().tail(252).quantile(0.90),
         int(np.ceil(frac.dropna().tail(252).quantile(0.90) * R.notna().sum(axis=1).iloc[-1])),
         R.notna().sum(axis=1).iloc[-1]))

near = spy_dd > -0.02
cells = {
    'COND pit>=90 & SPY within 2% (LIVE FORM)': (pit >= 90) & near,
    'COMPLEMENT pit>=90 & SPY >2% off': (pit >= 90) & (~near),
    'PARENT pit>=90 alone': (pit >= 90),
    'PARENT SPY within 2% alone': near,
    'pit>=95 & SPY within 2%': (pit >= 95) & near,
    'pit>=80 & SPY within 2%': (pit >= 80) & near,
}
print('\n### SHORT SPY h=3, episodes gap=10')
for lbl, m in cells.items():
    d = pit.index[m.fillna(False).values]
    r = horizon_scan(idxpx, d, [('SPY', -1.0)], hs=(3,), lag=1, min_gap=10)[0]
    print('  %-42s N=%3d  mean %+0.3f%%  edge %+0.3f pp' % (lbl, r['n'], r['mean_pct'], r['edge_pct']))

m = (pit >= 90) & near
battery(idxpx, m.fillna(False), [('SPY', -1.0)], 3,
        'C9 SHORT SPY h=3 | tape breadth PIT>=90 & SPY within 2% of high',
        cost_bps=1.0, variants={k: v.fillna(False) for k, v in cells.items()},
        lag=1, min_gap=10)

ret = vehicle_ret(idxpx, [('SPY', -1.0)], 3, 1)
sig = pit.index[m.fillna(False).values & ret.notna().reindex(pit.index, fill_value=False).values]
epi = declusters(sig, 10, idxpx.index)
v = ret.loc[epi].values
yrs = pd.DatetimeIndex(epi).year
show([summarize(v[yrs % 4 == 2], 'midterm'), summarize(v[yrs % 4 != 2], 'non-midterm')], 'cycle')
by = pd.DataFrame({'yr': yrs, 'r': 100 * v}).groupby('yr')['r'].agg(['count', 'sum'])
print('\nper-year (pp):'); print(by.round(2).to_string())
o = np.argsort(-v)
for k in (1, 2, 3):
    keep = np.ones(len(v), bool); keep[o[:k]] = False
    print('drop-best-%d: N=%d mean %+0.3f%%' % (k, keep.sum(), 100 * v[keep].mean()))
