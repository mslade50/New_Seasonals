"""C2 round 1b -- the only surviving direction is SHORT XLE at h<=3.
Battery + era split + the [1.5,2.0)% band that the 2% cut discards.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

px = close_panel(['USO', 'GLD', '^TNX', 'XLE', 'SPY', 'TLT', 'DBC'])
px = px.dropna(subset=['USO', 'GLD', '^TNX', 'XLE', 'SPY'])
r1 = {t: px[t] / px[t].shift(1) - 1.0 for t in px.columns}
uso, gld, tnx = r1['USO'], r1['GLD'], r1['^TNX']

cond = (uso >= 0.02) & (gld < 0) & (tnx > 0)
variants = {
    'USO>=1.5%': (uso >= 0.015) & (gld < 0) & (tnx > 0),
    'USO>=2.5%': (uso >= 0.025) & (gld < 0) & (tnx > 0),
    'USO in [1.5,2.0)%': (uso >= 0.015) & (uso < 0.02) & (gld < 0) & (tnx > 0),
    'GLD<-0.5% (deeper)': (uso >= 0.02) & (gld < -0.005) & (tnx > 0),
    'no rates leg': (uso >= 0.02) & (gld < 0),
    'no gold leg (PARENT)': (uso >= 0.02) & (tnx > 0),
    'COMPLEMENT GLD>0': (uso >= 0.02) & (tnx > 0) & (gld > 0),
}
battery(px, cond, [('XLE', -1.0)], 3, 'C2 SHORT XLE h=3 | USO>=2%, GLD<0, TNX>0',
        cost_bps=3.0, variants=variants, lag=1, event_kinds=('cpi', 'ppi'))

# midterm vs non-midterm, and a year-by-year table
ret = vehicle_ret(px, [('XLE', -1.0)], 3, 1)
sig = px.index[cond.values & ret.notna().values]
epi = declusters(sig, 3, px.index)
v = ret.loc[epi].values
yrs = pd.DatetimeIndex(epi).year
mid = (yrs % 4 == 2)
show([summarize(v[mid], 'midterm yrs'), summarize(v[~mid], 'non-midterm')],
     'cycle split (episodes)')
by = pd.DataFrame({'yr': yrs, 'r': 100 * v}).groupby('yr')['r'].agg(['count', 'sum', 'mean'])
print('\nper-year episode totals (pp):')
print(by.round(2).to_string())

# drop-best-N
order = np.argsort(v)  # most negative = best for a short
for k in (1, 2, 3):
    keep = np.ones(len(v), bool); keep[order[:k]] = False
    print('drop-best-%d: %s' % (k, summarize(v[keep], 'rest')))

# 2018 cut is in battery; add a 2021 cut because energy regime changed
show(era_split(epi, v, cut='2021-01-01'), 'era split at 2021')
