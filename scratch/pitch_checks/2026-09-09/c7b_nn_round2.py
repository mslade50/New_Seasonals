"""C7 round 2 — the two cells the pre-declared k=25 grid produced:
  SPY h=1 short  = -0.394% (16-9 down), the k-stable, largest-|t| occupant
  SVXY h=10 long = +3.474% (16-5), the largest |mean|

Round 2 kills:
 (a) charged permutation on the DEFENDED value of each (not the grid max).
 (b) k stability across k = 10,15,20,25,30,40,50.
 (c) the calendar-leakage trap: drop every neighbour with a CPI or PPI print
     inside the hold. Does the cell survive on the remainder?
 (d) era split and per-neighbour table.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

SEC = ['XLK', 'XLV', 'XLU', 'XLP', 'XLI', 'XLF', 'XLY', 'XLB', 'XLE']
TRADE = ['SPY', 'TLT', 'GLD', 'XLE', 'SVXY']
raw = load_prices(sorted(set(SEC + TRADE + ['DBC', '^TNX', '^VIX'])))
IDX = raw['SPY']['Close'].index
px = pd.DataFrame({t: raw[t]['Close'] for t in TRADE}).reindex(IDX)
spy = raw['SPY']['Close']
vix = raw['^VIX']['Close'].reindex(IDX)
tnx = raw['^TNX']['Close'].reindex(IDX)
dbc = raw['DBC']['Close'].reindex(IDX)
rv = spy.pct_change().rolling(21).std() * np.sqrt(252) * 100
sp = pd.DataFrame({t: raw[t]['Close'] for t in SEC}).reindex(IDX)
r21 = sp.apply(lambda s: _valid_pct_change(s, 21))
sprd = (r21.max(axis=1) - r21.min(axis=1)) * 100
F = pd.DataFrame({
    'rvol_pct': rolling_on_valid(rv, lambda x: x.rolling(252).rank(pct=True) * 100),
    'spy_off_hi': 100 * (spy / rolling_on_valid(spy, lambda x: x.rolling(252).max()) - 1),
    'tnx_off_hi': 100 * (tnx / rolling_on_valid(tnx, lambda x: x.rolling(252).max()) - 1),
    'dbc_off_hi': 100 * (dbc / rolling_on_valid(dbc, lambda x: x.rolling(252).max()) - 1),
    'disp_pct': rolling_on_valid(sprd, lambda x: x.rolling(252).rank(pct=True) * 100),
    'vix': vix}).dropna()
Z = ((F - F.expanding(504).mean()) / F.expanding(504).std()).dropna()
dist = pd.Series(np.sqrt(((Z.values[:-1] - Z.iloc[-1].values) ** 2).sum(axis=1)),
                 index=Z.index[:-1]).sort_values()
usable = IDX[:-12]
posmap = pd.Series(range(len(IDX)), index=IDX)


def neighbours(k, gap=21):
    keep = []
    for d in dist.index:
        if d not in usable:
            continue
        p = posmap[d]
        if any(abs(p - posmap[q]) < gap for q in keep):
            continue
        keep.append(d)
        if len(keep) == k:
            break
    return pd.DatetimeIndex(keep)


CELLS = [('SPY short h=1', [('SPY', -1.0)], 1, 1.0),
         ('SVXY long h=10', [('SVXY', 1.0)], 10, 15.0)]

print('### (b) k STABILITY of the two defended cells ###')
for name, legs, h, cost in CELLS:
    ret = vehicle_ret(px, legs, h, 1)
    valid = ret.dropna().index
    print('\n%s' % name)
    for k in (10, 15, 20, 25, 30, 40, 50):
        nb = neighbours(k).intersection(valid)
        v = ret.loc[nb].values
        w = int((v > 0).sum())
        print('  k=%2d  n=%2d  mean %+.3f%%  hit %.1f%%  t %+.2f  record %d-%d  sign p %.4f'
              % (k, len(v), 100 * v.mean(), 100 * (v > 0).mean(),
                 v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), w, len(v) - w,
                 sign_test(w, len(v))))

print('\n\n### (a) CHARGED PERMUTATION on each DEFENDED value ###')
retmap = {(t, h): vehicle_ret(px, [(t, 1.0)], h, 1) for t in TRADE for h in (1, 2, 3, 5, 10)}
rng = np.random.default_rng(23)
for name, legs, h, cost in CELLS:
    ret = vehicle_ret(px, legs, h, 1)
    valid = ret.dropna().index
    nb = neighbours(25).intersection(valid)
    OBS = 100 * ret.loc[nb].mean()
    n = len(nb)
    pool = valid[:-12]
    nulls, solo = [], []
    for _ in range(4000):
        picks = pd.DatetimeIndex(rng.choice(pool, size=n, replace=False))
        b = -np.inf
        for key, r in retmap.items():
            rr = r.reindex(picks).dropna()
            if len(rr) >= n - 2:
                b = max(b, abs(100 * rr.mean()))
        nulls.append(b)
        solo.append(abs(100 * ret.reindex(picks).dropna().mean()))
    print('%-16s defended |%.3f%%| on n=%d: charged P(null grid max >= it) = %.4f ; '
          'uncharged single-cell P = %.4f ; edge/cost = %.1fx'
          % (name, OBS, n, (np.asarray(nulls) >= abs(OBS)).mean(),
             (np.asarray(solo) >= abs(OBS)).mean(), abs(OBS) * 100 / cost))

print('\n\n### (c) CALENDAR-LEAKAGE STRESS: drop print-window neighbours ###')
for name, legs, h, cost in CELLS:
    ret = vehicle_ret(px, legs, h, 1)
    valid = ret.dropna().index
    nb = neighbours(25).intersection(valid)
    fl = event_in_window(nb, IDX, h, 1, ('cpi', 'ppi'))
    show([summarize(ret.loc[nb].values, 'all neighbours'),
          summarize(ret.loc[nb[fl]].values, 'PRINT in hold'),
          summarize(ret.loc[nb[~fl]].values, 'NO print in hold')], name)
    a = ret.loc[nb[fl]].values
    b = ret.loc[nb[~fl]].values
    print('  print-window share of neighbours: %d/%d = %.0f%% (all-day base rate below)'
          % (int(fl.sum()), len(nb), 100 * fl.mean()))
    base = event_in_window(usable, IDX, h, 1, ('cpi', 'ppi'))
    print('  all-day base rate for a print inside an h=%d hold: %.0f%%' % (h, 100 * base.mean()))
    if len(b) > 1:
        w = int((b > 0).sum())
        print('  EX-PRINT remainder: n=%d mean %+.3f%% record %d-%d sign p %.4f'
              % (len(b), 100 * b.mean(), w, len(b) - w, sign_test(w, len(b))))

print('\n\n### (d) ERA + per-neighbour table ###')
for name, legs, h, cost in CELLS:
    ret = vehicle_ret(px, legs, h, 1)
    valid = ret.dropna().index
    nb = neighbours(25).intersection(valid)
    v = ret.loc[nb].values
    show(era_split(nb, v), name)
    print('  ', cluster_note(nb, v))
    s = np.sort(v)[::-1] if 'SVXY' in name else np.sort(v)
    print('  drop-best-2 (in the pitched direction): %+.3f%% on n=%d'
          % (100 * (s[2:].mean() if 'SVXY' in name else s[:-2].mean()), len(s) - 2))
    print('  ', ', '.join('%s %+.2f%%' % (d.date(), 100 * r) for d, r in zip(nb, ret.loc[nb].values)))
