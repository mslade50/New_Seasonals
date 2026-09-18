"""C12 addendum — the cell the placebo ladder actually found, priced honestly.

The ladder's top occupant at EVERY horizon and both vehicles is k=+1: enter at
the CLOSE OF THE PPI RELEASE SESSION (not before it) with ^TNX at a 252d high.
Since that came out of MY OWN 11-rung ladder x 2 vehicles x 3 horizons, it is
charged a permutation on ITS OWN observed value.

Also: the decluster-ORDER fragility of the round-1 headline
(decluster-then-filter +0.406% vs filter-then-decluster +0.218%).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import rolling_on_valid
import numpy as np, pandas as pd

raw = load_prices(['TLT', 'IEF', '^TNX'])
px = pd.DataFrame({t: raw[t]['Close'] for t in ('TLT', 'IEF')}).dropna()
IDX = px.index
tnx = raw['^TNX']['Close'].reindex(IDX)
tnx_off = tnx / rolling_on_valid(tnx, lambda x: x.rolling(252).max()) - 1
LIVE = (tnx_off >= -0.001).fillna(False)
ppi = load_events(['ppi'])['date']

print('### decluster-ORDER fragility of the round-1 headline ###', flush=True)
for tkr in ('IEF', 'TLT'):
    for h in (3, 5):
        ret = vehicle_ret(px, [(tkr, 1.0)], h, 1)
        valid = ret.dropna().index
        st = IDX[LIVE.values].intersection(valid)
        e1 = declusters(st, max(h, 5), valid)
        fl = event_in_window(e1, IDX, h, 1, ('ppi',))
        a = ret.loc[e1[fl]].values
        pos, _ = anchor_positions(IDX, ppi, -2)
        anc = pd.DatetimeIndex(sorted(set(IDX[p] for p in pos))).intersection(valid)
        e2 = declusters(anc.intersection(IDX[LIVE.values]), max(h, 5), valid)
        b = ret.loc[e2].values
        print('  long %s h=%d: decluster-then-filter %+.3f%% (n=%d) vs '
              'filter-then-decluster %+.3f%% (n=%d)  -> %.3fpp apart'
              % (tkr, h, 100 * a.mean(), len(a), 100 * b.mean(), len(b),
                 100 * (a.mean() - b.mean())), flush=True)

print('\n### k=+1 (post-release entry) charged for the ladder search ###', flush=True)
print('search = 11 ladder rungs x {IEF,TLT} x h in {1,3,5} = 66 cells', flush=True)

# dense arrays: rows = trading days, cols = the 6 vehicle/horizon series
cols = [(t, h) for t in ('IEF', 'TLT') for h in (1, 3, 5)]
M = np.column_stack([vehicle_ret(px, [(t, 1.0)], h, 1).values for t, h in cols])
ok = np.isfinite(M).all(axis=1)
Mok = M[ok]
dates_ok = IDX[ok]
rng = np.random.default_rng(31)
NB = 4000

for tkr, h in (('TLT', 1), ('IEF', 3), ('TLT', 3), ('IEF', 5), ('TLT', 5)):
    ci = cols.index((tkr, h))
    ret = vehicle_ret(px, [(tkr, 1.0)], h, 1)
    valid = ret.dropna().index
    pos, _ = anchor_positions(IDX, ppi, -1)          # k=+1 -> offset -2+1
    a = pd.DatetimeIndex(sorted(set(IDX[p] for p in pos)))
    a = a.intersection(valid).intersection(IDX[LIVE.values])
    epi = declusters(a, max(h, 5), valid)
    v = ret.loc[epi].values
    n = len(v); OBS = 100 * v.mean(); w = int((v > 0).sum())
    # null: n random days, 11 ladder rungs are 11 independent draws of the same
    # size, so the family max is over 11 x 6 = 66 independent n-day means.
    idxs = rng.integers(0, len(Mok), size=(NB, 11, n))
    fam = 100 * Mok[idxs].mean(axis=2)               # (NB, 11, 6)
    nulls = fam.max(axis=(1, 2))
    solo = 100 * Mok[rng.integers(0, len(Mok), size=(NB, n)), ci].mean(axis=1)
    print('long %s h=%d: %+.3f%% n=%d record %d-%d sign p %.4f | charged P %.4f '
          '| uncharged P %.4f | %.1fx a 3 bp round trip'
          % (tkr, h, OBS, n, w, n - w, sign_test(w, n),
             float((nulls >= OBS).mean()), float((solo >= OBS).mean()), OBS * 100 / 3.0),
          flush=True)
    print('   dates:', ', '.join('%s(%+.2f%%)' % (d.date(), 100 * r) for d, r in zip(epi, v)),
          flush=True)
    print('   ', cluster_note(epi, v), flush=True)
    show(era_split(epi, v), 'era')
    mt = np.array([d.year % 4 == 2 for d in epi])
    show([summarize(v[mt], 'midterm'), summarize(v[~mt], 'non-midterm')], 'cycle')
