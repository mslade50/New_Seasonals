"""C2 round 1 -- crude-led inflation impulse gold refuses to confirm.

COMPLEMENT TEST FIRST. The gold leg is the whole candidate: if "USO up >=2%
and TNX up" already pays what the gold-gated cell pays, the gold condition is
a filter that does not filter.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

TK = ['USO', 'GLD', '^TNX', 'XLE', 'SPY', 'TLT', 'DBC', 'SLV', 'GDX']
px = close_panel(TK)
px = px.dropna(subset=['USO', 'GLD', '^TNX', 'XLE', 'SPY'])
print('panel', px.index[0].date(), '..', px.index[-1].date(), len(px))

r1 = {t: px[t] / px[t].shift(1) - 1.0 for t in px.columns}
uso, gld, tnx = r1['USO'], r1['GLD'], r1['^TNX']

# live state 2026-09-08
print('\nLIVE 09-08: USO %+.2f%%  GLD %+.2f%%  TNX %+.2f%%'
      % (100 * uso.iloc[-1], 100 * gld.iloc[-1], 100 * tnx.iloc[-1]))

MASKS = {
    'COND  USO>=2 & GLD<0 & TNX>0': (uso >= 0.02) & (gld < 0) & (tnx > 0),
    'PARENT USO>=2 & TNX>0 (no gold)': (uso >= 0.02) & (tnx > 0),
    'COMPL USO>=2 & TNX>0 & GLD>0': (uso >= 0.02) & (tnx > 0) & (gld > 0),
    'USO>=2 alone': (uso >= 0.02),
    'USO>=2 & GLD<0 (no rates)': (uso >= 0.02) & (gld < 0),
}
for k, m in MASKS.items():
    print('%-34s n=%d  last=%s' % (k, int(m.sum()), str(px.index[m][-1].date()) if m.sum() else '-'))

VEH = {'XLE': [('XLE', 1.0)], 'USO': [('USO', 1.0)], 'GLD': [('GLD', 1.0)],
       'SPY': [('SPY', 1.0)], 'TLT': [('TLT', 1.0)], 'DBC': [('DBC', 1.0)]}

print('\n### gate attribution: episode mean %% by horizon, lag=1 (declustered gap=h)')
for vname, legs in VEH.items():
    rows = []
    for k, m in MASKS.items():
        d = px.index[m.values]
        hs = horizon_scan(px, d, legs, hs=(1, 2, 3, 5, 10), lag=1)
        rec = {'cell': k}
        for r in hs:
            if r['n']:
                rec[r['label'] + ' N'] = r['n']
                rec[r['label']] = round(r['mean_pct'], 3)
        rows.append(rec)
    # all-days control
    base = {'cell': 'CTRL all days'}
    for h in (1, 2, 3, 5, 10):
        s = vehicle_ret(px, legs, h, 1).dropna()
        base['h=%d' % h] = round(100 * s.mean(), 3)
        base['h=%d N' % h] = len(s)
    rows.append(base)
    show(rows, 'vehicle %s' % vname)

# gate WORTH: cond minus parent, per horizon, on XLE / USO / GLD
print('\n### gate worth (COND minus PARENT), pp, episode level')
for vname, legs in VEH.items():
    out = {'vehicle': vname}
    for h in (1, 2, 3, 5, 10):
        c = horizon_scan(px, px.index[MASKS['COND  USO>=2 & GLD<0 & TNX>0'].values],
                         legs, hs=(h,), lag=1)[0]
        p = horizon_scan(px, px.index[MASKS['PARENT USO>=2 & TNX>0 (no gold)'].values],
                         legs, hs=(h,), lag=1)[0]
        if c['n'] and p['n']:
            out['h=%d' % h] = round(c['mean_pct'] - p['mean_pct'], 3)
    print(out)

# dose response on the USO threshold (with gold+rates gate on)
print('\n### dose response on the USO threshold, gate ON, XLE and USO h=3/h=5')
for thr in (0.010, 0.015, 0.020, 0.025, 0.030, 0.040):
    m = (uso >= thr) & (gld < 0) & (tnx > 0)
    d = px.index[m.values]
    line = 'USO>=%.1f%%  ndays=%3d' % (100 * thr, int(m.sum()))
    for vn in ('XLE', 'USO', 'GLD'):
        for h in (3, 5):
            r = horizon_scan(px, d, VEH[vn], hs=(h,), lag=1)[0]
            if r['n']:
                line += '  %s h%d %+0.3f%%(N%d)' % (vn, h, r['mean_pct'], r['n'])
    print(line)
