"""C8 round 1: a VIX settlement landing ON an FOMC decision date.

HONESTY NOTE, stated up front as the candidate requires: this repo has NO
dealer-gamma history, NO options open interest history and NO futures roll
positioning. data/option_*.csv accrue only since 2026-08-05. The stated
mechanism (expiring-settle hedging unwind colliding with the event risk
premium release) is therefore NOT DIRECTLY MEASURABLE here. All that can be
measured is the PRICE consequence of the calendar collision. A mechanism that
cannot be falsified has not been verified, and that is graded as such.

Structure:
  1. how often does vix_expiry == fomc_decision?
  2. the RUN-IN: anchor 6 td before the collision, entry lag 1, h=5 lands ON it
     (exactly today's geometry: D=09-08, entry 09-09, exit 09-16)
  3. the REDUCTION TEST (kill #2): is the collision cell any different from the
     plain pre-FOMC run-in? If not, C8 IS the pre-FOMC drift, which is
     midterm-inverted and is already the Event Sleeve's T2 short.
  4. the settle session itself
  5. placebo ladder k=-5..+5 (kill #8)
  6. midterm split -- 2026 is a midterm year
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

ev = load_events()
print('event kinds:', sorted(ev['event'].unique()))
vx = pd.DatetimeIndex(ev[ev['event'] == 'vix_expiry']['date'])
fo = pd.DatetimeIndex(ev[ev['event'] == 'fomc_decision']['date'])
op = pd.DatetimeIndex(ev[ev['event'] == 'opex']['date'])
print('vix_expiry n=%d (%s..%s) | fomc_decision n=%d (%s..%s)'
      % (len(vx), vx[0].date(), vx[-1].date(), len(fo), fo[0].date(), fo[-1].date()))

coll = vx.intersection(fo)
print('\n=== 1. COLLISIONS: vix_expiry == fomc_decision ===')
print('n = %d  (%.1f%% of all FOMC decisions)' % (len(coll), 100 * len(coll) / len(fo)))
print('dates:', ', '.join(str(d.date()) for d in coll))
print('by month:', pd.Series(1, index=coll).groupby(coll.month).sum().to_dict())
print('by year :', pd.Series(1, index=coll).groupby(coll.year).sum().to_dict())
print('midterm years (yr%%4==2):', [str(d.date()) for d in coll if d.year % 4 == 2])

TK = ['SPY', 'IWM', 'SVXY', 'QQQ']
px_d = load_prices(TK)
spy = px_d['SPY']['Close']
idx = spy.index
panel = pd.DataFrame({t: px_d[t]['Close'].reindex(idx) for t in TK})

LAG, H = 1, 5


def runin(anchor_dates, veh, offset=-6, h=H, label=''):
    """Entry at close of (event + offset + lag), exit h sessions later."""
    pos, kept = anchor_positions(idx, anchor_dates, offset)
    d = idx[pos]
    r = fwd_lag(panel[veh], h, LAG)
    val = r.dropna().index
    d = pd.DatetimeIndex(d).intersection(val)
    if len(d) == 0:
        return {'label': label, 'n': 0}
    v = r.loc[d].values
    base = r.loc[val]
    w = int((v > 0).sum())
    return {'label': label, 'n': len(v), 'mean_pct': round(100 * v.mean(), 3),
            'edge_pct': round(100 * (v.mean() - base.mean()), 3),
            'median_pct': round(100 * np.median(v), 3),
            'hit': round(100 * w / len(v), 1),
            't': round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2) if len(v) > 1 else np.nan,
            'record': '%d-%d' % (w, len(v) - w),
            'sign_p': round(sign_test(w, len(v)), 4),
            'worst_pct': round(100 * v.min(), 2)}


print('\n=== 2+3. RUN-IN (anchor -6td, entry lag1, h=5 lands ON the date) ===')
print('    THE REDUCTION TEST: collision vs every FOMC vs FOMC-without-collision')
nocoll = fo.difference(vx)
for veh in ['SPY', 'IWM', 'SVXY']:
    rows = [runin(coll, veh, label='COLLISION fomc==vix_exp (n_cal=%d)' % len(coll)),
            runin(fo, veh, label='ALL fomc_decision'),
            runin(nocoll, veh, label='FOMC without a collision'),
            runin(vx.difference(fo), veh, label='vix_expiry without an FOMC')]
    show(rows, '%s run-in, h=5' % veh)
    a = runin(coll, veh)
    b = runin(nocoll, veh)
    if a['n'] and b['n']:
        print('  >> collision gate is worth  %+.3fpp  (collision %+.3f vs no-collision %+.3f)'
              % (a['edge_pct'] - b['edge_pct'], a['edge_pct'], b['edge_pct']))

print('\n=== 4. THE SETTLE SESSION ITSELF (entry prior close, h=1 across the date) ===')
for veh in ['SPY', 'IWM', 'SVXY']:
    rows = [runin(coll, veh, offset=-2, h=1, label='COLLISION settle day'),
            runin(nocoll, veh, offset=-2, h=1, label='FOMC no-collision settle day'),
            runin(vx.difference(fo), veh, offset=-2, h=1, label='vix_expiry alone, that day')]
    show(rows, '%s, the day itself' % veh)

print('\n=== 5. PLACEBO LADDER on the collision anchor (SPY, h=5 run-in) ===')
rows = []
for k in range(-5, 6):
    rows.append(runin(coll, 'SPY', offset=-6 + k, h=H, label='k=%+d' % k))
show(rows)
r = pd.DataFrame(rows)
if 'mean_pct' in r:
    r2 = r.dropna(subset=['mean_pct']).copy()
    r2['rank'] = r2['mean_pct'].rank(ascending=False)
    tr = r2[r2['label'] == 'k=+0']
    print('  TRUE anchor (k=+0) ranks %s of %d placebo offsets by mean'
          % (tr['rank'].iloc[0] if len(tr) else 'n/a', len(r2)))

print('\n=== 6. MIDTERM SPLIT -- 2026 IS A MIDTERM YEAR ===')
for veh in ['SPY', 'IWM']:
    mt = pd.DatetimeIndex([d for d in coll if d.year % 4 == 2])
    nm = pd.DatetimeIndex([d for d in coll if d.year % 4 != 2])
    rows = [runin(mt, veh, label='collision, MIDTERM (live cell)'),
            runin(nm, veh, label='collision, non-midterm')]
    show(rows, '%s run-in h=5 by cycle' % veh)

print('\n=== 7. is the collision anchor just OPEX? (registry: expiry/opex are ONE anchor) ===')
print('collision dates that are within 2 td of an opex:',
      sum(1 for d in coll if (abs((op - d).days) <= 4).any()), 'of', len(coll))
print('opex-minus-collision day gaps:',
      [int((op[op >= d][0] - d).days) if len(op[op >= d]) else None for d in coll])
