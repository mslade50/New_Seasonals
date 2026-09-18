"""C2 round 1c -- USO long was the ONE vehicle where the gold gate was worth
anything (+0.386pp at h=5). Price it against the mandated USO roll cost:
-8.8 bps per 3-session hold, i.e. -2.93 bps/session.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

px = close_panel(['USO', 'GLD', '^TNX', 'XLE']).dropna()
r1 = {t: px[t] / px[t].shift(1) - 1.0 for t in px.columns}
cond = (r1['USO'] >= 0.02) & (r1['GLD'] < 0) & (r1['^TNX'] > 0)
sig = px.index[cond.values]

ROLL_BPS_PER_SESSION = 8.8 / 3.0
for h in (3, 5, 10):
    ret = vehicle_ret(px, [('USO', 1.0)], h, 1)
    epi = declusters(sig.intersection(ret.dropna().index), h, px.index)
    v = ret.loc[epi].values
    s = summarize(v, 'USO h=%d' % h)
    gross = s['mean_pct'] * 100          # bps
    roll = ROLL_BPS_PER_SESSION * h
    net = gross - roll - 5.0             # 5 bps USO round trip
    w = int((v > 0).sum())
    print('h=%2d N=%3d gross %+6.1f bps | roll -%.1f | trade -5.0 | NET %+6.1f bps '
          '= %.1fx the 5 bps trade cost | rec %d-%d sign p %.4f'
          % (h, len(v), gross, roll, net, max(net, 0) / 5.0, w, len(v) - w,
             sign_test(w, len(v))))
    print('    %s' % cluster_note(epi, v))
    o = np.argsort(-v)
    for k in (1, 2):
        keep = np.ones(len(v), bool); keep[o[:k]] = False
        print('    drop-best-%d gross %+0.1f bps' % (k, 100 * 100 * v[keep].mean()))
    yrs = pd.DatetimeIndex(epi).year
    print('    midterm %+0.3f%% (N=%d) vs non-midterm %+0.3f%% (N=%d)'
          % (100 * v[yrs % 4 == 2].mean(), (yrs % 4 == 2).sum(),
             100 * v[yrs % 4 != 2].mean(), (yrs % 4 != 2).sum()))
