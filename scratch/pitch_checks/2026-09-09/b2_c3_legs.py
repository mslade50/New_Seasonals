"""C3 round 1 -- DBC at a 252d high with its metals leg in a deep drawdown.

Rule 9: price EVERY leg against its own drift BEFORE any spread.
Rule 2: complement test -- DBC at a high with metals NOT deep, and metals deep
with DBC NOT at a high ("buy what fell", the short-term reversal explanation).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

TK = ['DBC', 'GLD', 'SLV', 'GDX', 'USO', 'XLE', 'SPY']
px = close_panel(TK).dropna()
print('panel', px.index[0].date(), '..', px.index[-1].date(), len(px))


def dd_from_high(s, lb=252):
    hi = rolling_on_valid(s, lambda x: x.rolling(lb).max())
    return s / hi - 1.0


dbc_hi = px['DBC'] >= rolling_on_valid(px['DBC'], lambda x: x.rolling(252).max()) - 1e-9
gdd, sdd = dd_from_high(px['GLD']), dd_from_high(px['SLV'])
print('LIVE 09-08: DBC at 252d high? %s | GLD dd %+.2f%% | SLV dd %+.2f%%'
      % (bool(dbc_hi.iloc[-1]), 100 * gdd.iloc[-1], 100 * sdd.iloc[-1]))

# ---- (a) how rare is it
cond = dbc_hi & (gdd <= -0.10) & (sdd <= -0.20)
print('\n(a) RARITY')
print('DBC at 252d high alone: %d days, %d episodes(gap=21)'
      % (int(dbc_hi.sum()), len(declusters(px.index[dbc_hi.values], 21, px.index))))
for lbl, m in [('DBC hi & GLD<=-5%', dbc_hi & (gdd <= -0.05)),
               ('DBC hi & GLD<=-10%', dbc_hi & (gdd <= -0.10)),
               ('DBC hi & GLD<=-15%', dbc_hi & (gdd <= -0.15)),
               ('DBC hi & GLD<=-10% & SLV<=-20%', cond),
               ('DBC hi & GLD<=-15% & SLV<=-35%', dbc_hi & (gdd <= -0.15) & (sdd <= -0.35))]:
    d = px.index[m.values]
    e = declusters(d, 21, px.index)
    print('  %-32s days=%4d episodes=%3d  yrs=%s' % (
        lbl, int(m.sum()), len(e), sorted(set(pd.DatetimeIndex(e).year))))

# ---- (b) price EVERY leg alone against its own drift
LEGS = {'GLD': [('GLD', 1.0)], 'SLV': [('SLV', 1.0)], 'GDX': [('GDX', 1.0)],
        'USO': [('USO', 1.0)], 'XLE': [('XLE', 1.0)], 'DBC': [('DBC', 1.0)],
        'SPY': [('SPY', 1.0)]}
print('\n(b) SINGLE LEGS -- conditional episode mean vs own all-day drift, lag=1')
for name, legs in LEGS.items():
    row = {'leg': name}
    for h in (1, 2, 3, 5, 10):
        r = horizon_scan(px, px.index[cond.values], legs, hs=(h,), lag=1)[0]
        if r['n']:
            row['h%d' % h] = round(r['mean_pct'], 3)
            row['h%d edge' % h] = round(r['edge_pct'], 3)
            row['h%dN' % h] = r['n']
    print(row)

# ---- (c) spreads, only after the legs
print('\n(c) SPREADS (metals long vs energy short) -- episode mean, lag=1')
SPREADS = {
    'GLD - USO': [('GLD', 1.0), ('USO', -1.0)],
    'GLD - XLE': [('GLD', 1.0), ('XLE', -1.0)],
    'GLD+SLV /2 - DBC': [('GLD', 0.5), ('SLV', 0.5), ('DBC', -1.0)],
    'SLV - USO': [('SLV', 1.0), ('USO', -1.0)],
    'CONTINUATION: USO - GLD': [('USO', 1.0), ('GLD', -1.0)],
    'CONTINUATION: XLE - SLV': [('XLE', 1.0), ('SLV', -1.0)],
}
for name, legs in SPREADS.items():
    row = {'spread': name}
    for h in (1, 2, 3, 5, 10):
        r = horizon_scan(px, px.index[cond.values], legs, hs=(h,), lag=1)[0]
        if r['n']:
            row['h%d' % h] = round(r['mean_pct'], 3)
            row['h%d edge' % h] = round(r['edge_pct'], 3)
    print(row)

# ---- (d) COMPLEMENT / gate attribution -- is it just "buy what fell"?
print('\n(d) GATE ATTRIBUTION, GLD long h=5 and h=10 (episodes, gap=h)')
CELLS = {
    'COND (DBC hi & GLD<=-10 & SLV<=-20)': cond,
    'PARENT metals only: GLD<=-10 & SLV<=-20 (no DBC)': (gdd <= -0.10) & (sdd <= -0.20),
    'COMPLEMENT: metals deep, DBC NOT at high': (~dbc_hi) & (gdd <= -0.10) & (sdd <= -0.20),
    'PARENT DBC only: DBC at 252d high': dbc_hi,
    'COMPLEMENT: DBC hi, metals NOT deep': dbc_hi & (gdd > -0.10),
}
for vn in ('GLD', 'SLV', 'GDX'):
    rows = []
    for lbl, m in CELLS.items():
        rec = {'cell': lbl}
        for h in (3, 5, 10):
            r = horizon_scan(px, px.index[m.values], LEGS[vn], hs=(h,), lag=1)[0]
            if r['n']:
                rec['h%d' % h] = round(r['mean_pct'], 3)
                rec['h%dN' % h] = r['n']
        rows.append(rec)
    show(rows, 'gate attribution, long %s' % vn)
