"""C1 round 2c: the one residual worth pricing before the kill.

a1b's vol rung had a real plateau (rv<=10 5-0, <=15 8-1, <=20 9-1 at +1.018pp
t 2.70). If the LOOSER rung also has a horizon plateau then the h=5-only spike
was a small-N artefact and C1 is a near-miss rather than a kill. If the looser
rung is ALSO one horizon wide, the definition is the finding.

Also: does the offered mechanism (index vol suppressed by rotation -> the rates
leg correlates it -> SPY down) hold in its own window? The registry says high
dispersion is followed by SPY UP.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd


def level_rank(s, lookback=252):
    return rolling_on_valid(s, lambda x: x.rolling(lookback).rank(pct=True) * 100.0)


px_d = load_prices(['SPY', 'TLT', 'IEF', '^TNX', '^VIX'])
spy = px_d['SPY']['Close']
tnx = px_d['^TNX']['Close'].reindex(spy.index)
tlt = px_d['TLT']['Close'].reindex(spy.index)

rv21 = rolling_on_valid(spy, lambda x: x.pct_change().rolling(21).std() * np.sqrt(252) * 100)
rv_pct = level_rank(rv21, 252)
tnx_gap = tnx / rolling_on_valid(tnx, lambda x: x.rolling(252).max()) - 1.0
hi = tnx_gap >= -0.0025


def prof(mask, label, veh):
    s = {'TLT': tlt, 'SPY': spy}[veh]
    out = []
    for h in (1, 2, 3, 5, 7, 10):
        rr = fwd_lag(s, h, 1)
        val = rr.dropna().index
        tr = pd.DatetimeIndex(spy.index[mask.fillna(False).values]).intersection(val)
        e = declusters(tr, 21, val)
        v = rr.loc[e].values
        base = rr.loc[val]
        w = int((v > 0).sum())
        out.append({'label': '%s h=%d' % (label, h), 'n': len(e),
                    'mean_pct': round(100 * v.mean(), 3),
                    'edge_pct': round(100 * (v.mean() - base.mean()), 3),
                    'record': '%d-%d' % (w, len(e) - w),
                    'hit': round(100 * w / len(e), 1),
                    't': round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2),
                    'sign_p': round(sign_test(w, len(e)), 4)})
    return out


print('=== TLT long: horizon profile at each vol rung (does the plateau survive?) ===')
for rung in (10, 15, 20, 25):
    show(prof(hi & (rv_pct <= rung), 'rv<=%d' % rung, 'TLT'), 'vol rung <= %d' % rung)

print('\n\n=== the OFFERED mechanism: SPY should be DOWN. Is it? ===')
for rung in (10, 20):
    show(prof(hi & (rv_pct <= rung), 'SPY rv<=%d' % rung, 'SPY'), 'SPY long, vol rung <= %d' % rung)
