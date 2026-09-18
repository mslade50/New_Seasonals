"""C11 round 2 — the only thing that survived round 1 is the EQUITY leg:
SPY long h=3 on CELL = (HYG within 0.5% of its 252d high) & (^TNX at a 252d
high). 7 episodes, +0.558%, 6-1, gate worth +0.727pp, complement -0.364%.

Round 2 must kill it: depth-matched split (watchlist 24's killer), drop-best-N,
decluster sensitivity, midterm, fragility, per-episode table, and a permutation
charged for MY OWN 5-vehicle x 6-horizon search testing THIS cell's +0.558%.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

TK = ['HYG', 'LQD', 'IEF', 'TLT', 'SPY']
px = close_panel(TK).dropna()
IDX = px.index
tnx = load_prices(['^TNX'])['^TNX']['Close'].reindex(IDX)

hyg_off = px['HYG'] / rolling_on_valid(px['HYG'], lambda x: x.rolling(252).max()) - 1
tnx_off = tnx / rolling_on_valid(tnx, lambda x: x.rolling(252).max()) - 1
spy_off = px['SPY'] / rolling_on_valid(px['SPY'], lambda x: x.rolling(252).max()) - 1
CRED = (hyg_off >= -0.005).fillna(False)
RATE = (tnx_off >= -0.001).fillna(False)
CELL = CRED & RATE
H = 3
ret = vehicle_ret(px, [('SPY', 1.0)], H, 1)
valid = ret.dropna().index
cell_days = IDX[CELL.values].intersection(valid)
epi = declusters(cell_days, 10, valid)

print('live SPY off 252d high = %+.3f%%' % (100 * spy_off.iloc[-1]))
print('\n--- per-episode table (SPY long h=3) ---')
frag_path = Path(__file__).resolve().parents[3] / 'data' / 'rd2_fragility.parquet'
fr = pd.read_parquet(frag_path)
fr.index = pd.to_datetime(fr.index)
ma = fr['63d'].rolling(10).mean()
rows = []
for d in epi:
    rows.append({'date': str(d.date()), 'ret_pct': round(100 * ret.loc[d], 3),
                 'spy_off_high_pct': round(100 * spy_off.loc[d], 2),
                 'hyg_off_high_pct': round(100 * hyg_off.loc[d], 3),
                 'tnx': round(float(tnx.loc[d]), 3),
                 'dial_ma10_63d': round(float(ma.reindex([d]).iloc[0]), 1) if d in ma.index else np.nan,
                 'midterm': d.year % 4 == 2})
print(pd.DataFrame(rows).to_string(index=False))

v = ret.loc[epi].values
print('\ncell mean %.3f%%  n=%d  record %d-%d  sign p %.4f  bootstrap P(<=0) %.3f'
      % (100 * v.mean(), len(v), int((v > 0).sum()), int((v <= 0).sum()),
         sign_test(int((v > 0).sum()), len(v)), bootstrap_p_le0(v)))
s = np.sort(v)[::-1]
for k in (1, 2, 3):
    print('  drop-best-%d: mean %.3f%% on n=%d' % (k, 100 * s[k:].mean(), len(s) - k))
print('  ', cluster_note(epi, v))

# ---- DEPTH-MATCHED SPLIT (the watchlist-24 killer) -------------------------
print('\n\n### DEPTH-MATCHED SPLIT on SPY distance from its own 252d high ###')
bands = [(-0.005, 0.0, '0.0-0.5% off'), (-0.010, -0.005, '0.5-1.0% off'),
         (-0.020, -0.010, '1.0-2.0% off  <-- LIVE 1.53%'),
         (-0.050, -0.020, '2.0-5.0% off'), (-1.0, -0.050, '>5% off')]
for lo, hi, lbl in bands:
    m = (spy_off > lo) & (spy_off <= hi)
    c = IDX[(CELL & m).values].intersection(valid)
    r = IDX[(RATE & ~CRED & m).values].intersection(valid)
    ec = declusters(c, 10, valid); er = declusters(r, 10, valid)
    sc = summarize(ret.loc[ec].values, 'CELL ' + lbl)
    sr = summarize(ret.loc[er].values, 'RATE&~CRED ' + lbl)
    gate = (sc.get('mean_pct', np.nan) - sr.get('mean_pct', np.nan))
    print('%-30s CELL n=%-3s %8s   COMPLEMENT n=%-3s %8s   credit gate %+.3fpp'
          % (lbl, sc.get('n'), ('%.3f%%' % sc['mean_pct']) if sc.get('n') else '--',
             sr.get('n'), ('%.3f%%' % sr['mean_pct']) if sr.get('n') else '--', gate))

# ---- decluster sensitivity + horizon neighbours ----------------------------
print('\n### decluster sensitivity (SPY h=3) ###')
for g in (1, 5, 10, 21, 63, 126):
    e = declusters(cell_days, g, valid)
    r = summarize(ret.loc[e].values, 'gap=%d' % g)
    print('  gap %3d td: n=%2d mean %+.3f%% hit %.1f%% t %+.2f'
          % (g, r['n'], r['mean_pct'], r['hit'], r['t']))

print('\n### horizon neighbours (the pitched h must not be an isolated spike) ###')
show(horizon_scan(px, cell_days, [('SPY', 1.0)], hs=tuple(range(1, 11)), lag=1, min_gap=10),
     'SPY long, CELL episodes')

# ---- PERMUTATION charged for MY search -------------------------------------
print('\n### PERMUTATION: 5 vehicles x 6 horizons = 30 cells searched ###')
print('    tested statistic = SPY long h=3 episode mean = +0.558%')
d = px.pct_change().dropna()
X = np.column_stack([np.ones(len(d)), d['IEF'].values, d['SPY'].values])
beta, *_ = np.linalg.lstsq(X, d['HYG'].values, rcond=None)
VEH = {'HYG': [('HYG', 1.0)], 'LQD': [('LQD', 1.0)], 'SPY': [('SPY', 1.0)],
       'HYG-IEF': [('HYG', 1.0), ('IEF', -1.0)],
       'RESID': [('HYG', 1.0), ('IEF', -round(float(beta[1]), 3)),
                 ('SPY', -round(float(beta[2]), 3))]}
HS = (1, 2, 3, 5, 7, 10)
retmap = {(k, h): vehicle_ret(px, legs, h, 1) for k, legs in VEH.items() for h in HS}
n_ep = len(epi)
rng = np.random.default_rng(7)
pool = valid[:-11]
OBS = 100 * v.mean()
nulls = []
for _ in range(4000):
    picks = pd.DatetimeIndex(rng.choice(pool, size=n_ep, replace=False))
    best = -np.inf
    for k in VEH:
        for h in HS:
            r = retmap[(k, h)].reindex(picks).dropna()
            if len(r) >= n_ep - 1:
                best = max(best, 100 * r.mean())
    nulls.append(best)
nulls = np.asarray(nulls)
print('    P(null max-over-search >= +0.558%%) = %.4f   (null max median %.3f%%, 95th %.3f%%)'
      % ((nulls >= OBS).mean(), np.median(nulls), np.percentile(nulls, 95)))
solo = []
for _ in range(4000):
    picks = pd.DatetimeIndex(rng.choice(pool, size=n_ep, replace=False))
    r = retmap[('SPY', 3)].reindex(picks).dropna()
    solo.append(100 * r.mean())
print('    single-cell P(null >= +0.558%%) = %.4f  (uncharged, for contrast)'
      % (np.asarray(solo) >= OBS).mean())

# ---- is the cell just "SPY quiet + near its high"? -------------------------
print('\n### matched-tape control: SPY-depth + realized-vol matched non-cell days ###')
rv = px['SPY'].pct_change().rolling(21).std() * np.sqrt(252) * 100
rvp = rolling_on_valid(rv, lambda x: x.rolling(252).rank(pct=True) * 100)
print('  live SPY 21d rvol pctile = %.1f' % rvp.iloc[-1])
for d in epi:
    print('   %s  spy_off %+.2f%%  rvol_pctile %.1f' % (d.date(), 100 * spy_off.loc[d], rvp.loc[d]))
mask_match = ((spy_off > -0.030) & (spy_off <= 0.0) & (rvp <= 40)).fillna(False)
mm = IDX[(mask_match & ~CELL).values].intersection(valid)
show([summarize(ret.loc[declusters(mm, 10, valid)].values, 'matched non-cell (shallow dip, calm)'),
      summarize(v, 'CELL')], 'matched control')
