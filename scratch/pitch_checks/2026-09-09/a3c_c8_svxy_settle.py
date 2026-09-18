"""C8 round 2b: the ONE genuine residual in C8, priced for the watchlist.

a3b: on the collision SETTLE SESSION, SVXY runs +1.574% (25-9, sign p 0.004)
with alpha +1.628% at t 5.91 against SPY and beta*SPY explaining -0.055% of it.
That is a real volatility-specific residual -- it clears the standing rule that
"any SVXY cell that survives owes a residual against SPY".

It is NOT pitchable today (entry is the 2026-09-15 close, not 2026-09-09), so
this exists to fix the arm for a watchlist entry. It has to survive: the
midterm split that killed the run-in, a placebo ladder, concentration, and the
era cut.
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

px_d = load_prices(['SPY', 'SVXY', 'IWM'])
idx = px_d['SPY']['Close'].index
panel = pd.DataFrame({t: px_d[t]['Close'].reindex(idx) for t in ['SPY', 'SVXY', 'IWM']})


def cell(dates, veh, offset=-2, h=1, lag=1):
    pos, kept = anchor_positions(idx, dates, offset)
    d = idx[pos]
    r = fwd_lag(panel[veh], h, lag)
    val = r.dropna().index
    d = pd.DatetimeIndex(d).intersection(val)
    return r.loc[d].values, d


def line(lbl, v):
    if len(v) == 0:
        return {'label': lbl, 'n': 0}
    w = int((v > 0).sum())
    return {'label': lbl, 'n': len(v), 'mean_pct': round(100 * v.mean(), 3),
            'median_pct': round(100 * np.median(v), 3), 'hit': round(100 * w / len(v), 1),
            't': round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2) if len(v) > 1 else np.nan,
            'record': '%d-%d' % (w, len(v) - w), 'sign_p': round(sign_test(w, len(v)), 4),
            'worst_pct': round(100 * v.min(), 2)}


v_all, d_all = cell(coll, 'SVXY')
print('=== SVXY on the collision settle session (entry prior close, exit that close) ===')
show([line('COLLISION settle', v_all),
      line('FOMC no-collision settle', cell(nocoll, 'SVXY')[0]),
      line('vix_expiry alone settle', cell(vx.difference(fo), 'SVXY')[0])])

print('\n--- episodes, one line each ---')
for dt, v in zip(d_all, v_all):
    print('   %s  %+7.2f%%   %s' % (dt.date(), 100 * v, 'MIDTERM' if dt.year % 4 == 2 else ''))

print('\n=== THE TEST THAT KILLED THE RUN-IN: midterm split ===')
show([line('settle, MIDTERM', v_all[np.array([d.year % 4 == 2 for d in d_all])]),
      line('settle, non-midterm', v_all[np.array([d.year % 4 != 2 for d in d_all])])])

print('\n=== era split ===')
show([line('pre-2018', v_all[np.array([d.year < 2018 for d in d_all])]),
      line('2018+', v_all[np.array([d.year >= 2018 for d in d_all])])])

print('\n=== concentration / drop-best-N ===')
print(' ', cluster_note(d_all, v_all, k=2))
order = np.argsort(-v_all)
for k in (1, 2, 3):
    keep = np.ones(len(v_all), bool); keep[order[:k]] = False
    print('  drop-best-%d: n=%d mean %+.3f%% record %d-%d' % (
        k, keep.sum(), 100 * v_all[keep].mean(), int((v_all[keep] > 0).sum()), int((v_all[keep] <= 0).sum())))

print('\n=== PLACEBO LADDER k=-5..+5 around the settle session (SVXY, h=1) ===')
rows = []
for k in range(-5, 6):
    v, _ = cell(coll, 'SVXY', offset=-2 + k, h=1)
    rows.append(line('k=%+d' % k, v))
show(rows)
df = pd.DataFrame(rows).dropna(subset=['mean_pct'])
df['rank'] = df['mean_pct'].rank(ascending=False)
print('  TRUE settle session (k=+0) ranks %s of %d'
      % (df[df.label == 'k=+0']['rank'].iloc[0], len(df)))

print('\n=== residual against SPY, by cycle bucket (the mandatory vol check) ===')
sp_all, dsp = cell(coll, 'SPY')
com = pd.DatetimeIndex(d_all).intersection(dsp)
rsv = pd.Series(v_all, index=d_all).loc[com].values
rsp = pd.Series(sp_all, index=dsp).loc[com].values
for lbl, m in [('all', np.ones(len(com), bool)),
               ('midterm', np.array([d.year % 4 == 2 for d in com])),
               ('non-midterm', np.array([d.year % 4 != 2 for d in com]))]:
    if m.sum() < 5:
        print('  %-12s n=%d too few' % (lbl, m.sum())); continue
    b, a = np.polyfit(rsp[m], rsv[m], 1)
    res = rsv[m] - (a + b * rsp[m])
    print('  %-12s n=%2d  SVXY = %+.3f%% + %.2f*SPY  R^2 %.3f  alpha t %+.2f'
          % (lbl, m.sum(), 100 * a, b, 1 - res.var() / rsv[m].var(),
             a / (res.std(ddof=2) / np.sqrt(m.sum()))))

print('\n=== cost: SVXY round trip ~4-8 bps. edge = %.0f bps -> %.1fx to %.1fx cost ==='
      % (100 * 100 * v_all.mean(), 100 * 100 * v_all.mean() / 8, 100 * 100 * v_all.mean() / 4))
