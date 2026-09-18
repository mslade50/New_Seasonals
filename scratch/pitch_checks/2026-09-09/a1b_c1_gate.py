"""C1 round 2 (run FIRST, it is the fastest kill): does the realized-vol gate filter?

The grid in a1 put TLT LONG h=5 at 6-for-6, +1.546%, t 3.78. Two of its three
components are separable, so:

  PARENT-A   TNX within 0.25% of its 252d high, NO vol gate
  COMPLEMENT TNX at the high AND rv pctile > 10  (the discarded half)
  PARENT-B   rv pctile <= 10, NO yield gate
  CELL       both

Kill #2 says: if the COMPLEMENT keeps the parent's edge, the gate is a lucky
subset and nothing may be attributed to it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd


def level_rank(s, lookback=252):
    return rolling_on_valid(s, lambda x: x.rolling(lookback).rank(pct=True) * 100.0)


TK = ['SPY', 'IWM', 'TLT', 'IEF', '^TNX']
px_d = load_prices(TK)
spy = px_d['SPY']['Close']
tnx = px_d['^TNX']['Close'].reindex(spy.index)
tlt = px_d['TLT']['Close'].reindex(spy.index)
ief = px_d['IEF']['Close'].reindex(spy.index)

rv21 = rolling_on_valid(spy, lambda x: x.pct_change().rolling(21).std() * np.sqrt(252) * 100)
rv_pct = level_rank(rv21, 252)
tnx_hi = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
tnx_gap = tnx / tnx_hi - 1.0

panel = pd.DataFrame({'SPY': spy, 'TLT': tlt, 'IEF': ief,
                      'IWM': px_d['IWM']['Close'].reindex(spy.index)})

H = 5
LAG = 1


def cell(mask, label, veh='TLT', h=H, gap=21, quiet=False):
    r = fwd_lag(panel[veh], h, LAG)
    valid = r.dropna().index
    trig = pd.DatetimeIndex(spy.index[mask.fillna(False).values]).intersection(valid)
    if len(trig) == 0:
        return {'label': label, 'n': 0, 'n_days': 0}
    epi = declusters(trig, gap, valid)
    d = summarize(r.loc[epi].values, label)
    d['n_days'] = len(trig)
    base = r.loc[valid]
    d['edge_pct'] = round(d['mean_pct'] - 100 * base.mean(), 3)
    w = int((r.loc[epi].values > 0).sum())
    d['record'] = '%d-%d' % (w, len(epi) - w)
    d['sign_p'] = round(sign_test(w, len(epi)), 4)
    if not quiet:
        d['dates'] = ', '.join(str(x.date()) for x in epi)
    return d


hi = (tnx_gap >= -0.0025)
lo_vol = (rv_pct <= 10.0)

print('TLT is the vehicle unless stated. h=%d, lag=%d, decluster gap 21td.' % (H, LAG))
for veh in ('TLT', 'IEF'):
    rows = [
        cell(hi & lo_vol, 'CELL  yield-high AND rv<=10', veh, quiet=True),
        cell(hi & ~lo_vol, 'COMPLEMENT yield-high AND rv>10', veh, quiet=True),
        cell(hi, 'PARENT-A  yield-high alone', veh, quiet=True),
        cell(lo_vol, 'PARENT-B  rv<=10 alone', veh, quiet=True),
        cell(pd.Series(True, index=spy.index), 'ALL DAYS', veh, quiet=True),
    ]
    show(rows, '%s long, h=%d  --- gate attribution ---' % (veh, H))

print('\n### the number that decides kill #2 ###')
c = cell(hi & lo_vol, 'cell', 'TLT', quiet=True)
k = cell(hi & ~lo_vol, 'compl', 'TLT', quiet=True)
p = cell(hi, 'parent', 'TLT', quiet=True)
print('TLT h=5  cell edge %+.3fpp (n=%d) | COMPLEMENT edge %+.3fpp (n=%d) | parent edge %+.3fpp (n=%d)'
      % (c['edge_pct'], c['n'], k['edge_pct'], k['n'], p['edge_pct'], p['n']))
print('gate is worth  cell - complement = %+.3fpp' % (c['edge_pct'] - k['edge_pct']))

# ---- definition neighbours -------------------------------------------------
print('\n=== definition neighbours: the yield-gate rung (vol gate held at <=10) ===')
rows = []
for g in (0.0, -0.0010, -0.0025, -0.0050, -0.0100, -0.0200, -0.0300):
    rows.append(cell((tnx_gap >= g) & lo_vol, 'yield gap >= %+.2f%%' % (100 * g), 'TLT', quiet=True))
show(rows)

print('\n=== definition neighbours: the vol-gate rung (yield gate held at 0.25%) ===')
rows = []
for v in (5, 10, 15, 20, 25, 33, 50):
    rows.append(cell(hi & (rv_pct <= v), 'rv pctile <= %d' % v, 'TLT', quiet=True))
show(rows)

print('\n=== definition neighbours: rv LOOKBACK and WINDOW ===')
rows = []
for lb in (126, 252, 504):
    rp = level_rank(rv21, lb)
    rows.append(cell(hi & (rp <= 10), 'rv rank lookback %d' % lb, 'TLT', quiet=True))
for w in (10, 21, 42, 63):
    rvw = rolling_on_valid(spy, lambda x: x.pct_change().rolling(w).std() * np.sqrt(252) * 100)
    rows.append(cell(hi & (level_rank(rvw, 252) <= 10), 'rv window %dd' % w, 'TLT', quiet=True))
show(rows)

print('\n=== horizon profile of the CELL vs the COMPLEMENT (TLT long) ===')
rows = []
for h in (1, 2, 3, 5, 7, 10):
    a = cell(hi & lo_vol, 'CELL h=%d' % h, 'TLT', h=h, quiet=True)
    b = cell(hi & ~lo_vol, 'COMPL h=%d' % h, 'TLT', h=h, quiet=True)
    rows += [a, b]
show(rows)

print('\n=== era split of PARENT-A (yield high alone), TLT long h=5, episodes ===')
r = fwd_lag(panel['TLT'], H, LAG)
valid = r.dropna().index
trig = pd.DatetimeIndex(spy.index[hi.fillna(False).values]).intersection(valid)
epi = declusters(trig, 21, valid)
show(era_split(epi, r.loc[epi].values), 'parent-A eras')
print('parent-A episodes: %d, by year: %s' % (len(epi), pd.Series(1, index=epi).groupby(epi.year).sum().to_dict()))
