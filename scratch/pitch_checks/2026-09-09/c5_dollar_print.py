"""C5 round 1 — dollar at a 63d rank floor into an inflation print.

Live state 2026-09-08: DX-Y.NYB 63d rank 14.7, 5d rank 10.7; ^TNX at a 252d
high. PPI 09-10 (2 td after the signal close), CPI 09-11 (3 td).

Anchor convention: signal close D, entry MOC D+1, exit D+1+h. The TRUE event
anchor is D = print - 2 td, i.e. entry the session before the release, so an
h=1 hold exits on the release close. Placebo ladder shifts that offset by
k = -5..+5.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

ETF = ['SPY', 'UUP', 'GLD', 'EEM']
px = close_panel(ETF)
IDX = px.index

raw = load_prices(['DX-Y.NYB', '^TNX'])
dxs = raw['DX-Y.NYB']['Close']
tnx_raw = raw['^TNX']['Close']


def state_on(index, dx, tnx, rank_max=20.0, hi_tol=0.001):
    """dollar 63d return rank <= rank_max AND ^TNX within hi_tol of a 252d high."""
    dxr = pct_rank(dx, 63, 252).reindex(index)
    tnx = tnx.reindex(index)
    hi = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
    tnx_hi = tnx >= hi * (1 - hi_tol)
    return (dxr <= rank_max).fillna(False) & tnx_hi.fillna(False), dxr


# ------------------------------------------------------------------ ETF side
st_etf, dxr_etf = state_on(IDX, dxs, tnx_raw)
print('ETF calendar: %d rows %s..%s' % (len(IDX), IDX[0].date(), IDX[-1].date()))
print('live dx 63d rank = %.1f   state today = %s' % (dxr_etf.iloc[-1], bool(st_etf.iloc[-1])))
print('joint state (dx63rank<=20 & TNX 252d high) day count = %d' % int(st_etf.sum()))
print('bare dx63rank<=20 day count = %d' % int((dxr_etf <= 20).sum()))

prints = load_events(['ppi'])['date']
cpis = load_events(['cpi'])['date']
both = pd.DatetimeIndex(sorted(set(prints) | set(cpis)))
print('ppi=%d cpi=%d union=%d' % (len(prints), len(cpis), len(both)))


def anchor_set(index, ev_dates, offset):
    pos, kept = anchor_positions(index, ev_dates, offset)
    return pd.DatetimeIndex(sorted(set(index[p] for p in pos)))


# --- 1. bare price state, no event, for the complement test -----------------
for legs, cost, name in [([('UUP', -1.0)], 3.0, 'short UUP'),
                         ([('GLD', 1.0)], 2.0, 'long GLD'),
                         ([('EEM', 1.0)], 3.0, 'long EEM')]:
    for h in (1, 3, 5):
        battery(px, st_etf, legs, h, 'C5-state %s h=%d (no event gate)' % (name, h),
                cost, event_kinds=('ppi', 'cpi'))

# --- 2. event anchor x state ------------------------------------------------
print('\n\n################ C5 EVENT ANCHOR x STATE ################')
for evname, evd in [('ppi', prints), ('cpi', cpis), ('ppi+cpi', both)]:
    a0 = anchor_set(IDX, evd, -2)
    st_days = IDX[st_etf.values]
    inter = a0.intersection(st_days)
    print('\n%s true anchor (print-2td): %d anchor days, %d also in dollar/TNX state'
          % (evname, len(a0), len(inter)))
    if len(inter) >= 3:
        print('  state+anchor dates:', ', '.join(str(d.date()) for d in inter))


# --- 3. placebo ladder -------------------------------------------------------
def ladder(legs, evd, evname, h, restrict=None, label=''):
    rows = []
    ret = vehicle_ret(px, legs, h, 1)
    valid = ret.dropna().index
    for k in range(-5, 6):
        a = anchor_set(IDX, evd, -2 + k)
        a = a.intersection(valid)
        if restrict is not None:
            a = a.intersection(IDX[restrict.values])
        epi = declusters(a, max(h, 5), valid)
        r = summarize(ret.loc[epi].values, 'k=%+d%s' % (k, ' <TRUE>' if k == 0 else ''))
        rows.append(r)
    show(rows, 'PLACEBO LADDER %s %s h=%d %s' % (label, evname, h, legs))
    vals = [r.get('mean_pct', np.nan) for r in rows]
    order = np.argsort(-np.asarray([v if v == v else -1e9 for v in vals]))
    rank = int(np.where(order == 5)[0][0]) + 1
    print('  TRUE anchor (k=0) mean %.3f%% ranks %d of 11' % (vals[5], rank))
    return rank


for legs, name in [([('UUP', -1.0)], 'short UUP'), ([('GLD', 1.0)], 'long GLD')]:
    for evd, evname in [(prints, 'ppi'), (both, 'ppi+cpi')]:
        for h in (1, 3):
            ladder(legs, evd, evname, h, restrict=st_etf, label='STATE+' + name)
            ladder(legs, evd, evname, h, restrict=None, label='BARE ' + name)

# --- 4. DX futures on its own calendar --------------------------------------
print('\n\n################ C5 DX-Y.NYB (own calendar) ################')
dpx = pd.DataFrame({'DX': dxs}).dropna()
DIDX = dpx.index
tnx_d = tnx_raw.reindex(DIDX).ffill(limit=5)
st_dx, dxr_dx = state_on(DIDX, dxs, tnx_d)
print('DX rows %d, state days %d, live rank %.1f state %s'
      % (len(DIDX), int(st_dx.sum()), dxr_dx.iloc[-1], bool(st_dx.iloc[-1])))
for h in (1, 3, 5):
    battery(dpx, st_dx, [('DX', -1.0)], h, 'C5 short DX state h=%d' % h, 1.5,
            event_kinds=('ppi', 'cpi'))

# --- 5. midterm split (2026 is midterm) -------------------------------------
print('\n\n################ C5 MIDTERM SPLIT ################')
for legs, name, cost in [([('UUP', -1.0)], 'short UUP', 3.0), ([('GLD', 1.0)], 'long GLD', 2.0)]:
    for h in (1, 3, 5):
        ret = vehicle_ret(px, legs, h, 1)
        valid = ret.dropna().index
        d = IDX[st_etf.values].intersection(valid)
        epi = declusters(d, max(h, 5), valid)
        mt = np.array([y % 4 == 2 for y in epi.year])
        show([summarize(ret.loc[epi[mt]].values, 'midterm'),
              summarize(ret.loc[epi[~mt]].values, 'non-midterm')],
             '%s h=%d state episodes' % (name, h))
