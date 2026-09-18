"""C1 round 2b: per-episode paths, era, cycle, drop-best-N, and the permutation
charge for the 5-vehicle x 5-horizon x 2-direction grid the candidate handed me.

a1b showed the vol gate DOES filter (complement +0.075pp vs cell +1.173pp), so
kill #2 misses. What it also showed is a one-horizon spike (5-0 at h=5, 3-2 at
h=3 and h=7 on the SAME five episodes) and a yield rung that flips negative at
its first real neighbour. This quantifies both.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd


def level_rank(s, lookback=252):
    return rolling_on_valid(s, lambda x: x.rolling(lookback).rank(pct=True) * 100.0)


TK = ['SPY', 'IWM', 'TLT', 'IEF', 'SVXY', '^TNX']
px_d = load_prices(TK)
spy = px_d['SPY']['Close']
tnx = px_d['^TNX']['Close'].reindex(spy.index)
panel = pd.DataFrame({t: px_d[t]['Close'].reindex(spy.index)
                      for t in ['SPY', 'IWM', 'TLT', 'IEF', 'SVXY']})

rv21 = rolling_on_valid(spy, lambda x: x.pct_change().rolling(21).std() * np.sqrt(252) * 100)
rv_pct = level_rank(rv21, 252)
tnx_hi = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
tnx_gap = tnx / tnx_hi - 1.0
mask = ((tnx_gap >= -0.0025) & (rv_pct <= 10.0)).fillna(False)

trig = spy.index[mask.values]
print('mask days (%d):' % len(trig), ', '.join(str(d.date()) for d in trig))

r5 = fwd_lag(panel['TLT'], 5, 1)
valid = r5.dropna().index
epi = declusters(pd.DatetimeIndex(trig).intersection(valid), 21, valid)
print('\nTLT h=5 episodes (%d): ' % len(epi), ', '.join(str(d.date()) for d in epi))

# per-episode across horizons -- is h=5 a spike or a plateau?
print('\n=== per-episode TLT long return by horizon (%) ===')
tab = {}
for h in (1, 2, 3, 5, 7, 10):
    rr = fwd_lag(panel['TLT'], h, 1)
    tab['h=%d' % h] = (100 * rr.loc[epi]).round(2)
t = pd.DataFrame(tab)
t.index = [str(d.date()) for d in epi]
print(t.to_string())
print('\nmean by horizon:', t.mean().round(3).to_dict())
print('wins by horizon:', (t > 0).sum().to_dict(), 'of', len(t))

print('\n=== concentration / drop-best-N (TLT h=5 episodes) ===')
v = r5.loc[epi].values
print(cluster_note(epi, v, k=2))
order = np.argsort(-v)
for k in (0, 1, 2):
    keep = np.ones(len(v), bool)
    keep[order[:k]] = False
    if keep.sum():
        print('  drop-best-%d: n=%d mean %+.3f%% record %d-%d' % (
            k, keep.sum(), 100 * v[keep].mean(), int((v[keep] > 0).sum()),
            int((v[keep] <= 0).sum())))

print('\n=== era + cycle (2026 is MIDTERM: year %% 4 == 2) ===')
for lbl, m in [('pre-2018', epi.year < 2018), ('2018+', epi.year >= 2018),
               ('midterm yrs', (epi.year % 4) == 2), ('non-midterm', (epi.year % 4) != 2)]:
    if m.sum():
        print('  %-12s n=%d mean %+.3f%% record %d-%d  dates %s' % (
            lbl, m.sum(), 100 * v[m].mean(), int((v[m] > 0).sum()), int((v[m] <= 0).sum()),
            [str(d.date()) for d in epi[m]]))
    else:
        print('  %-12s n=0  *** the live configuration has NO precedent in this bucket ***' % lbl)

# ---- multiplicity: max-of-K permutation over the handed grid ----------------
print('\n=== permutation: best occupant of the 5 vehicle x 5 horizon x 2 direction grid ===')
rng = np.random.default_rng(42)
GRID = [(veh, h) for veh in ['SPY', 'IWM', 'TLT', 'IEF', 'SVXY'] for h in (1, 2, 3, 5, 10)]

# precompute forward returns + valid indices once
pre = {}
for veh, h in GRID:
    rr = fwd_lag(panel[veh], h, 1)
    pre[(veh, h)] = (rr, rr.dropna().index)


def best_abs_t(trigger_days):
    best = 0.0
    for veh, h in GRID:
        rr, val = pre[(veh, h)]
        tt = pd.DatetimeIndex(trigger_days).intersection(val)
        if len(tt) < 3:
            continue
        e = declusters(tt, max(h, 5), val)
        x = rr.loc[e].values
        x = x[~np.isnan(x)]
        if len(x) < 3 or x.std(ddof=1) == 0:
            continue
        best = max(best, abs(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))))
    return best


obs = best_abs_t(trig)
print('observed max |t| over the grid = %.3f  (TLT long h=5)' % obs)

# permutation: place the SAME number of trigger days at random, preserving the
# episode structure (draw n_epi anchors, keep the same intra-episode day counts)
n_epi = len(declusters(pd.DatetimeIndex(trig), 21, spy.index))
pool = spy.index[(rv_pct.notna()) & (tnx_gap.notna())]
pool = pool[(pool >= trig[0]) & (pool <= trig[-1])]
cnt = 0
NPERM = 2000
draws = []
for i in range(NPERM):
    anchors = pd.DatetimeIndex(rng.choice(pool, size=n_epi, replace=False)).sort_values()
    b = best_abs_t(anchors)
    draws.append(b)
    cnt += (b >= obs)
draws = np.array(draws)
print('permutation P(max|t| >= observed) = %.4f  over %d draws  (median draw %.2f, 95th %.2f)'
      % (cnt / NPERM, NPERM, np.median(draws), np.percentile(draws, 95)))
