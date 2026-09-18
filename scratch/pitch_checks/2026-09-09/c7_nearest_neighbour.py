"""C7 round 1 — nearest-neighbour tapes to 2026-09-08.

PRE-REGISTERED BEFORE ANY RETURN IS COMPUTED (stated here, in this order):

  features (6, all point-in-time, no calendar information by construction):
    f1 SPY 21d realized vol, trailing-252 LEVEL percentile     live   6.3
    f2 SPY distance to its 252d high, %                        live  -1.53
    f3 ^TNX distance to its 252d high, %                       live   0.00
    f4 DBC distance to its 252d high, %                        live   0.00
    f5 sector 21d max-min return spread, trailing-252 pctile   live  81.3
    f6 ^VIX level                                              live  15.72
  standardisation: EXPANDING mean/sd through the scored day (min 504 obs),
    so no future information enters a neighbour's coordinates.
  metric: Euclidean in that 6-d space.
  k = 25, DECLARED BEFORE RETURNS, declustered at a 21 td minimum gap.
  horizons 1..10, lag=1 (MOC tomorrow), instruments SPY TLT GLD XLE SVXY.
  charge: k in {10, 25, 50} is a search -> permutation on the k=25 statistic.

The known trap this must answer: a calendar-blind neighbour lane can still
rediscover a calendar effect. So the neighbour DATES are printed, and the
month/year distribution of the neighbour set is reported.
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
px = pd.DataFrame({t: raw[t]['Close'] for t in TRADE})
IDX = raw['SPY']['Close'].index
px = px.reindex(IDX)

spy = raw['SPY']['Close']
vix = raw['^VIX']['Close'].reindex(IDX)
tnx = raw['^TNX']['Close'].reindex(IDX)
dbc = raw['DBC']['Close'].reindex(IDX)

rv = spy.pct_change().rolling(21).std() * np.sqrt(252) * 100
f1 = rolling_on_valid(rv, lambda x: x.rolling(252).rank(pct=True) * 100)
f2 = 100 * (spy / rolling_on_valid(spy, lambda x: x.rolling(252).max()) - 1)
f3 = 100 * (tnx / rolling_on_valid(tnx, lambda x: x.rolling(252).max()) - 1)
f4 = 100 * (dbc / rolling_on_valid(dbc, lambda x: x.rolling(252).max()) - 1)
sp = pd.DataFrame({t: raw[t]['Close'] for t in SEC}).reindex(IDX)
r21 = sp.apply(lambda s: _valid_pct_change(s, 21))
sprd = (r21.max(axis=1) - r21.min(axis=1)) * 100
f5 = rolling_on_valid(sprd, lambda x: x.rolling(252).rank(pct=True) * 100)
f6 = vix

F = pd.DataFrame({'rvol_pct': f1, 'spy_off_hi': f2, 'tnx_off_hi': f3,
                  'dbc_off_hi': f4, 'disp_pct': f5, 'vix': f6}).dropna()
print('feature panel %d rows %s..%s' % (len(F), F.index[0].date(), F.index[-1].date()))
print('LIVE 2026-09-08 features:\n', F.iloc[-1].round(2).to_string())

# expanding PIT standardisation
mu = F.expanding(504).mean()
sd = F.expanding(504).std()
Z = ((F - mu) / sd).dropna()
print('\nstandardised panel %d rows from %s' % (len(Z), Z.index[0].date()))
target = Z.iloc[-1].values
D = np.sqrt(((Z.values[:-1] - target) ** 2).sum(axis=1))
dist = pd.Series(D, index=Z.index[:-1]).sort_values()

MAXH = 10
usable = IDX[:-(MAXH + 2)]


def neighbours(k, gap=21):
    keep, kept_d = [], []
    posmap = pd.Series(range(len(IDX)), index=IDX)
    for d, v in dist.items():
        if d not in posmap.index or d not in usable:
            continue
        p = posmap[d]
        if any(abs(p - posmap[q]) < gap for q in keep):
            continue
        keep.append(d); kept_d.append(v)
        if len(keep) == k:
            break
    return pd.DatetimeIndex(keep), np.asarray(kept_d)


for K in (10, 25, 50):
    nb, dd = neighbours(K)
    tag = '  <-- PRE-DECLARED k' if K == 25 else ''
    print('\n\n############ k=%d neighbours (21td declustered)%s ############' % (K, tag))
    print('distance range %.2f .. %.2f (6-d standardised units)' % (dd.min(), dd.max()))
    print('dates:', ', '.join('%s[%.2f]' % (d.date(), v) for d, v in zip(nb, dd)))
    yrs = pd.Series(nb.year).value_counts().sort_index()
    mons = pd.Series(nb.month).value_counts().sort_index()
    print('by year: ', dict(yrs))
    print('by month:', dict(mons))
    top2y = yrs.sort_values(ascending=False).head(2)
    print('top-2 years hold %d of %d neighbours (%.0f%%)'
          % (top2y.sum(), K, 100 * top2y.sum() / K))
    rows = []
    for t in TRADE:
        legs = [(t, 1.0)]
        for h in (1, 2, 3, 5, 10):
            ret = vehicle_ret(px, legs, h, 1)
            valid = ret.dropna().index
            n2 = nb.intersection(valid)
            r = summarize(ret.loc[n2].values, '%s h=%d' % (t, h))
            base = ret.loc[valid]
            if r.get('n'):
                r['ctl_pct'] = round(100 * base.mean(), 3)
                r['edge_pct'] = round(r['mean_pct'] - 100 * base.mean(), 3)
                r['sign_p'] = round(sign_test(int((ret.loc[n2].values > 0).sum()), r['n']), 4)
            rows.append(r)
    show(rows, 'k=%d forward returns' % K)

# ---- charge the k search ----------------------------------------------------
print('\n\n############ MULTIPLICITY CHARGE ############')
nb25, _ = neighbours(25)
best = None
for t in TRADE:
    for h in (1, 2, 3, 5, 10):
        ret = vehicle_ret(px, [(t, 1.0)], h, 1)
        valid = ret.dropna().index
        n2 = nb25.intersection(valid)
        if len(n2) < 5:
            continue
        m = 100 * ret.loc[n2].mean()
        if best is None or abs(m) > abs(best[2]):
            best = (t, h, m, len(n2))
print('best |mean| occupant of the pre-declared k=25 grid: %s h=%d = %+.3f%% on n=%d'
      % best)
t, h, OBS, n_used = best
retmap = {(a, b): vehicle_ret(px, [(a, 1.0)], b, 1) for a in TRADE for b in (1, 2, 3, 5, 10)}
rng = np.random.default_rng(3)
pool = retmap[(t, h)].dropna().index
nulls, solo = [], []
for _ in range(4000):
    picks = pd.DatetimeIndex(rng.choice(pool, size=n_used, replace=False))
    bb = -np.inf
    for key, r in retmap.items():
        rr = r.reindex(picks).dropna()
        if len(rr) >= n_used - 2:
            bb = max(bb, abs(100 * rr.mean()))
    nulls.append(bb)
    solo.append(abs(100 * retmap[(t, h)].reindex(picks).dropna().mean()))
print('P(null max|mean| over the 25-cell grid >= |%.3f%%|) = %.4f  (charged)'
      % (OBS, (np.asarray(nulls) >= abs(OBS)).mean()))
print('P(single-cell null >= |%.3f%%|) = %.4f  (uncharged)'
      % (OBS, (np.asarray(solo) >= abs(OBS)).mean()))

# ---- calendar leakage check -------------------------------------------------
print('\n\n############ CALENDAR LEAKAGE CHECK ############')
ev = load_events(['cpi', 'ppi', 'nfp', 'fomc_decision', 'opex'])
for K in (25,):
    nb, _ = neighbours(K)
    for kind in ('cpi', 'ppi', 'nfp', 'fomc_decision', 'opex'):
        e = pd.DatetimeIndex(ev[ev['event'] == kind]['date'])
        fl = event_in_window(nb, IDX, 3, 1, (kind,))
        base = event_in_window(usable, IDX, 3, 1, (kind,))
        print('  %-14s in an h=3 hold: neighbours %2d/%d = %.0f%%   all days %.0f%%'
              % (kind, int(fl.sum()), len(nb), 100 * fl.mean(), 100 * base.mean()))
