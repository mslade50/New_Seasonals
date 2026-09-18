"""C1 round 1a: yields at a 252d high while SPY 21d realized vol is bottom-decile.

Establish the mask, count episodes, and run the FULL handed grid
(5 vehicles x 5 horizons x 2 directions) so the direction is picked from a
table rather than assumed. Anything selected here is charged a max-of-K
permutation in a1c.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd



def level_rank(s, lookback=252):
    return rolling_on_valid(s, lambda x: x.rolling(lookback).rank(pct=True) * 100.0)


TK = ['SPY', 'IWM', 'TLT', 'IEF', 'SVXY', '^TNX', '^VIX', 'QQQ']
px_d = load_prices(TK)
for t in TK:
    print('%-10s %s .. %s  n=%d' % (t, px_d[t].index[0].date(), px_d[t].index[-1].date(), len(px_d[t])))

# Build the mask on SPY's own calendar (the state is an SPY-vol + TNX state).
spy = px_d['SPY']['Close']
tnx = px_d['^TNX']['Close'].reindex(spy.index)

rv21 = rolling_on_valid(spy, lambda x: x.pct_change().rolling(21).std() * np.sqrt(252) * 100)
rv_pct = level_rank(rv21, 252)

tnx_hi = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
tnx_gap = tnx / tnx_hi - 1.0          # 0 at the high, negative below

print('\nlive: TNX %.3f  gap-to-252d-high %+.4f%%   rv21 %.2f%%  rv pctile %.2f' % (
    tnx.iloc[-1], 100 * tnx_gap.iloc[-1], rv21.iloc[-1], rv_pct.iloc[-1]))

mask = (tnx_gap >= -0.0025) & (rv_pct <= 10.0)
mask = mask.fillna(False)
print('mask days: %d   first %s  last %s   live? %s' % (
    int(mask.sum()), mask[mask].index[0].date(), mask[mask].index[-1].date(), bool(mask.iloc[-1])))

trig = spy.index[mask.values]
for gap in (5, 10, 21):
    print('  episodes(gap=%d): %d' % (gap, len(declusters(trig, gap, spy.index))))
epi21 = declusters(trig, 21, spy.index)
print('  episode dates (gap21):', ', '.join(str(d.date()) for d in epi21))
print('  by year:', pd.Series(1, index=epi21).groupby(epi21.year).sum().to_dict())

# ---- the handed grid -------------------------------------------------------
panel = pd.DataFrame({t: px_d[t]['Close'] for t in ['SPY', 'IWM', 'TLT', 'IEF', 'SVXY', 'QQQ']})
panel = panel.reindex(spy.index)

rows = []
for veh in ['SPY', 'IWM', 'TLT', 'IEF', 'SVXY']:
    s = panel[veh]
    for h in (1, 2, 3, 5, 10):
        r = fwd_lag(s, h, 1)
        valid = r.dropna().index
        t_epi = declusters(pd.DatetimeIndex(trig).intersection(valid), max(h, 5), valid)
        if len(t_epi) == 0:
            continue
        v = r.loc[t_epi].values
        base = r.loc[valid]
        for sgn, lbl in ((1, 'LONG'), (-1, 'SHORT')):
            d = summarize(sgn * v, '%s %s h=%d' % (veh, lbl, h))
            d['edge_pct'] = round(d['mean_pct'] - 100 * sgn * base.mean(), 3)
            d['ctl_pct'] = round(100 * sgn * base.mean(), 3)
            rows.append(d)

df = pd.DataFrame(rows)
df = df[['label', 'n', 'mean_pct', 'edge_pct', 'ctl_pct', 'median_pct', 'hit', 't', 'worst_pct']]
print('\n=== FULL GRID, episode level (gap=max(h,5)), lag=1 ===')
print(df.round(3).to_string(index=False))

print('\n--- top 6 by edge ---')
print(df.sort_values('edge_pct', ascending=False).head(6).round(3).to_string(index=False))
print('\n--- top 6 by |t| ---')
print(df.reindex(df['t'].abs().sort_values(ascending=False).index).head(6).round(3).to_string(index=False))
