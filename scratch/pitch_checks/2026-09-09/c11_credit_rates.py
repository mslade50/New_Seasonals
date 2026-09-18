"""C11 round 1 — HYG at/near a 252d high on a day the ten-year prints one too.

Live 2026-09-08: HYG -0.46% off its 252d high, ^TNX AT a 252d high.

Kills to run: complement test FIRST (does the credit gate filter, or is the
rates gate the whole cell?), then the credit-specific RESIDUAL after regressing
HYG on IEF and SPY, then era/midterm split and concentration.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

TK = ['HYG', 'LQD', 'IEF', 'TLT', 'SPY']
px = close_panel(TK).dropna()          # HYG inception 2007-04 bounds it
IDX = px.index
tnx = load_prices(['^TNX'])['^TNX']['Close'].reindex(IDX)

hyg = px['HYG']
hyg_hi = rolling_on_valid(hyg, lambda x: x.rolling(252).max())
hyg_off = hyg / hyg_hi - 1.0                       # <=0, 0 = at the high
tnx_hi = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
tnx_off = tnx / tnx_hi - 1.0

CRED = (hyg_off >= -0.005).fillna(False)           # within 0.5% of 252d high
RATE = (tnx_off >= -0.001).fillna(False)           # ^TNX at a 252d high
CELL = CRED & RATE

print('panel %d rows %s..%s' % (len(IDX), IDX[0].date(), IDX[-1].date()))
print('live: HYG off high %+.3f%%   TNX off high %+.4f%%   CRED=%s RATE=%s CELL=%s'
      % (100 * hyg_off.iloc[-1], 100 * tnx_off.iloc[-1],
         bool(CRED.iloc[-1]), bool(RATE.iloc[-1]), bool(CELL.iloc[-1])))
print('day counts: CRED %d  RATE %d  CELL %d  RATE&~CRED %d  CRED&~RATE %d'
      % (CRED.sum(), RATE.sum(), CELL.sum(), (RATE & ~CRED).sum(), (CRED & ~RATE).sum()))
print('CELL episodes(gap10):', ', '.join(str(d.date()) for d in declusters(IDX[CELL.values], 10, IDX)))

# ---- standing decomposition, recomputed on this panel ----------------------
d = px.pct_change().dropna()
X = np.column_stack([np.ones(len(d)), d['IEF'].values, d['SPY'].values])
beta, *_ = np.linalg.lstsq(X, d['HYG'].values, rcond=None)
print('\nHYG daily = %+.4f%% + %.3f*IEF + %.3f*SPY   (n=%d, R2=%.3f)'
      % (100 * beta[0], beta[1], beta[2], len(d),
         1 - ((d['HYG'].values - X @ beta) ** 2).sum() / ((d['HYG'].values - d['HYG'].mean()) ** 2).sum()))
B_IEF, B_SPY = round(float(beta[1]), 3), round(float(beta[2]), 3)

VEH = [('HYG long', [('HYG', 1.0)], 3.0),
       ('LQD long', [('LQD', 1.0)], 3.0),
       ('SPY long', [('SPY', 1.0)], 1.0),
       ('HYG-IEF spread', [('HYG', 1.0), ('IEF', -1.0)], 3.0),
       ('HYG residual (ex IEF %.3f / SPY %.3f)' % (B_IEF, B_SPY),
        [('HYG', 1.0), ('IEF', -B_IEF), ('SPY', -B_SPY)], 3.0)]

# ---- 1. horizon scan of the cell, all vehicles -----------------------------
print('\n\n################ 1. HORIZON SCAN, CELL episodes ################')
cell_days = IDX[CELL.values]
for name, legs, _c in VEH:
    show(horizon_scan(px, cell_days, legs, hs=(1, 2, 3, 5, 7, 10), lag=1),
         'CELL %s' % name)

# ---- 2. COMPLEMENT TEST (run first, per doctrine) --------------------------
print('\n\n################ 2. COMPLEMENT / GATE ATTRIBUTION ################')
for name, legs, _c in VEH:
    for h in (3, 5, 10):
        ret = vehicle_ret(px, legs, h, 1)
        valid = ret.dropna().index
        rows = []
        for lbl, m in [('CELL (RATE&CRED)', CELL), ('RATE only (parent)', RATE),
                       ('RATE & ~CRED (complement)', RATE & ~CRED),
                       ('CRED only (parent)', CRED),
                       ('CRED & ~RATE (complement)', CRED & ~RATE),
                       ('ALL DAYS', pd.Series(True, index=IDX))]:
            dd = IDX[m.reindex(IDX, fill_value=False).values].intersection(valid)
            epi = declusters(dd, max(h, 10), valid)
            r = summarize(ret.loc[epi].values, lbl)
            r['n_days'] = len(dd)
            rows.append(r)
        show(rows, '%s  h=%d' % (name, h))
        c = rows[0].get('mean_pct', np.nan)
        p = rows[1].get('mean_pct', np.nan)
        comp = rows[2].get('mean_pct', np.nan)
        print('  credit-gate attribution: CELL %.3f%% vs RATE-parent %.3f%% '
              '(gate worth %+.3fpp); COMPLEMENT RATE&~CRED = %.3f%%'
              % (c, p, c - p, comp))

# ---- 3. full battery on the strongest-looking horizon ----------------------
print('\n\n################ 3. BATTERY ################')
variants = {'HYG within 0.25%%': (hyg_off >= -0.0025).fillna(False) & RATE,
            'HYG within 0.5%% (base)': CELL,
            'HYG within 1.0%%': (hyg_off >= -0.010).fillna(False) & RATE,
            'HYG within 2.0%%': (hyg_off >= -0.020).fillna(False) & RATE,
            'TNX within 0.5%% of hi': CRED & (tnx_off >= -0.005).fillna(False),
            'TNX within 2.0%% of hi': CRED & (tnx_off >= -0.020).fillna(False)}
for name, legs, cost in VEH:
    for h in (3, 5, 10):
        battery(px, CELL, legs, h, 'C11 %s' % name, cost, variants=variants,
                min_gap=10, event_kinds=('cpi', 'ppi'))

# ---- 4. era + midterm + fragility -----------------------------------------
print('\n\n################ 4. ERA / MIDTERM / FRAGILITY ################')
for name, legs, _c in VEH:
    for h in (3, 5, 10):
        ret = vehicle_ret(px, legs, h, 1)
        valid = ret.dropna().index
        epi = declusters(cell_days.intersection(valid), max(h, 10), valid)
        mt = np.array([y % 4 == 2 for y in epi.year])
        show(era_split(epi, ret.loc[epi].values) +
             [summarize(ret.loc[epi[mt]].values, 'midterm'),
              summarize(ret.loc[epi[~mt]].values, 'non-midterm')],
             '%s h=%d' % (name, h))
        print('  ', cluster_note(epi, ret.loc[epi].values))

frag = Path(__file__).resolve().parents[3] / 'data' / 'rd2_fragility.parquet'
if frag.exists():
    f = pd.read_parquet(frag)
    ma = f['63d'].rolling(10).mean()
    ma.index = pd.to_datetime(f.index)
    j = ma.reindex(cell_days).dropna()
    print('\nfragility ma10(63d) on CELL days: n=%d  min %.1f  max %.1f  '
          'median %.1f  days>=70: %d  days>=85: %d   (today 87.9)'
          % (len(j), j.min(), j.max(), j.median(), int((j >= 70).sum()), int((j >= 85).sum())))
