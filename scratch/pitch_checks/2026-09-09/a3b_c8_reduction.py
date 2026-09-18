"""C8 round 2: THE REDUCTION TEST.

a3 found the collision run-in splits +1.504% non-midterm (24-8) against
-1.753% midterm (4-6), and 2026 IS midterm. That is the pre-FOMC drift's known
midterm inversion (event-sleeve T2 prereg). The question that decides the
verdict: does the "vix_expiry lands on it" gate add ANYTHING once the cycle
bucket is held fixed? If not, C8 is the pre-FOMC drift in costume and the
candidate's own stated kill condition fires.

Also: the settle-session SVXY cell (+1.574%, 25-9, sign p 0.004) needs the
mandatory residual against SPY, and needs to be checked for tradeability from
TODAY (it is a 09-15 entry, not a 09-09 one).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

ev = load_events()
vx = pd.DatetimeIndex(ev[ev['event'] == 'vix_expiry']['date'])
fo = pd.DatetimeIndex(ev[ev['event'] == 'fomc_decision']['date'])
coll = vx.intersection(fo)
nocoll = fo.difference(vx)

px_d = load_prices(['SPY', 'IWM', 'SVXY'])
idx = px_d['SPY']['Close'].index
panel = pd.DataFrame({t: px_d[t]['Close'].reindex(idx) for t in ['SPY', 'IWM', 'SVXY']})


def vals(dates, veh, offset=-6, h=5, lag=1):
    pos, kept = anchor_positions(idx, dates, offset)
    d = idx[pos]
    r = fwd_lag(panel[veh], h, lag)
    val = r.dropna().index
    d = pd.DatetimeIndex(d).intersection(val)
    return r.loc[d].values, d


def welch(a, b, la, lb):
    if len(a) < 2 or len(b) < 2:
        return
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    print('    %-34s %+7.3f%% (n=%2d, %d-%d)  vs %-30s %+7.3f%% (n=%3d)   diff %+7.3fpp  welch t %+.2f'
          % (la, 100 * a.mean(), len(a), int((a > 0).sum()), int((a <= 0).sum()),
             lb, 100 * b.mean(), len(b), 100 * (a.mean() - b.mean()),
             (a.mean() - b.mean()) / se))


print('=== THE REDUCTION TEST: collision gate held against cycle bucket ===')
for veh in ['SPY', 'IWM']:
    print('\n--- %s run-in h=5 ---' % veh)
    for cyc, lbl in [(lambda y: y % 4 == 2, 'MIDTERM (2026 is here)'),
                     (lambda y: y % 4 != 2, 'non-midterm')]:
        c = pd.DatetimeIndex([d for d in coll if cyc(d.year)])
        n = pd.DatetimeIndex([d for d in nocoll if cyc(d.year)])
        a, _ = vals(c, veh)
        b, _ = vals(n, veh)
        print('  %s' % lbl)
        welch(a, b, 'COLLISION', 'FOMC no collision', )
        print('      >> collision gate worth %+.3fpp inside this bucket' % (100 * (a.mean() - b.mean())))

print('\n=== is the midterm inversion a property of ALL FOMC run-ins? (the reduction) ===')
for veh in ['SPY', 'IWM']:
    am, _ = vals(pd.DatetimeIndex([d for d in fo if d.year % 4 == 2]), veh)
    an, _ = vals(pd.DatetimeIndex([d for d in fo if d.year % 4 != 2]), veh)
    cm, _ = vals(pd.DatetimeIndex([d for d in coll if d.year % 4 == 2]), veh)
    cn, _ = vals(pd.DatetimeIndex([d for d in coll if d.year % 4 != 2]), veh)
    print('  %-4s ALL FOMC       midterm %+7.3f%% (n=%2d, %d-%d)  non-midterm %+7.3f%% (n=%3d)  gap %+.3fpp'
          % (veh, 100 * am.mean(), len(am), int((am > 0).sum()), int((am <= 0).sum()),
             100 * an.mean(), len(an), 100 * (am.mean() - an.mean())))
    print('  %-4s COLLISION only midterm %+7.3f%% (n=%2d, %d-%d)  non-midterm %+7.3f%% (n=%3d)  gap %+.3fpp'
          % (veh, 100 * cm.mean(), len(cm), int((cm > 0).sum()), int((cm <= 0).sum()),
             100 * cn.mean(), len(cn), 100 * (cm.mean() - cn.mean())))

print('\n=== midterm collision episodes, one line each (SPY run-in h=5) ===')
a, d = vals(pd.DatetimeIndex([x for x in coll if x.year % 4 == 2]), 'SPY')
for dt, v in zip(d, a):
    print('   anchor %s -> %+7.2f%%' % (dt.date(), 100 * v))

print('\n=== the settle-session SVXY cell: MANDATORY residual against SPY ===')
sv, dsv = vals(coll, 'SVXY', offset=-2, h=1)
sp, dsp = vals(coll, 'SPY', offset=-2, h=1)
common = pd.DatetimeIndex(dsv).intersection(dsp)
rsv = pd.Series(sv, index=dsv).loc[common].values
rsp = pd.Series(sp, index=dsp).loc[common].values
beta, alpha = np.polyfit(rsp, rsv, 1)
resid = rsv - (alpha + beta * rsp)
ss = 1 - resid.var() / rsv.var()
print('  n=%d  SVXY = %+.4f%% + %.3f * SPY   R^2 %.3f' % (len(common), 100 * alpha, beta, ss))
print('  alpha (the volatility-specific residual) = %+.4f%% ; t = %+.2f'
      % (100 * alpha, alpha / (resid.std(ddof=2) / np.sqrt(len(resid)))))
print('  raw SVXY mean %+.3f%%  |  beta*SPY explains %+.3f%%'
      % (100 * rsv.mean(), 100 * beta * rsp.mean()))
print('  registry comparison: SVXY = -0.293%% + 1.62*SPY, R^2 0.648 (2026-09-08)')
print('\n  TRADEABILITY: the settle-session cell enters at the 2026-09-15 close,')
print('  not today. It is not an order this morning can place.')
