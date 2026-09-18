"""C4 round 1: VIX3M bid while spot VIX stays dead, the session before a print cluster.

Live: 2026-09-08 VIX3M +4.43% vs VIX +2.75% (spread +1.68pp), VIX 15.72,
ratio 0.855, PPI +1 td and CPI +2 td.

The candidate demands the margin be tested as a CONTINUOUS DOSE RESPONSE, not a
mined threshold, and demands the "3M vol always bids into a known calendar"
explanation be interrogated. So:

  step 1  what IS the spread's unconditional distribution, and where does today sit
  step 2  the calendar explanation: is the spread systematically higher before a
          print cluster than on all other days? If yes the "signal" is a calendar
          effect with a price label.
  step 3  dose response of forward returns on the spread DECILE, print-conditioned
          and not, for SVXY / SPY / short-^VIX.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd


def level_rank(s, lookback=252):
    return rolling_on_valid(s, lambda x: x.rolling(lookback).rank(pct=True) * 100.0)


TK = ['SPY', 'SVXY', '^VIX', '^VIX3M', 'QQQ', 'IWM']
px_d = load_prices(TK)
for t in TK:
    print('%-8s %s .. %s n=%d' % (t, px_d[t].index[0].date(), px_d[t].index[-1].date(), len(px_d[t])))

spy = px_d['SPY']['Close']
idx = spy.index
vix = px_d['^VIX']['Close'].reindex(idx)
v3m = px_d['^VIX3M']['Close'].reindex(idx)
svxy = px_d['SVXY']['Close'].reindex(idx)

# spread = 1-day pct change of VIX3M minus that of VIX, in percentage POINTS
d_v3 = rolling_on_valid(v3m, lambda x: x.pct_change()) * 100
d_vx = rolling_on_valid(vix, lambda x: x.pct_change()) * 100
spread = (d_v3 - d_vx)
ratio = vix / v3m
vix_pct = level_rank(vix, 252)

print('\nlive 2026-09-08: dVIX3M %+.2f%%  dVIX %+.2f%%  spread %+.2fpp  ratio %.3f  VIX %.2f pctile %.1f'
      % (d_v3.iloc[-1], d_vx.iloc[-1], spread.iloc[-1], ratio.iloc[-1], vix.iloc[-1], vix_pct.iloc[-1]))

s = spread.dropna()
print('spread distribution: n=%d mean %+.3f sd %.3f | today is the %.1fth pctile'
      % (len(s), s.mean(), s.std(), 100 * (spread.iloc[-1] > s).mean()))
print('  deciles:', np.round(np.percentile(s, [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]), 2).tolist())

# ---- step 2: is the spread just the calendar? -------------------------------
ev = load_events(['cpi', 'ppi'])
ev_dates = pd.DatetimeIndex(ev['date'])
pos = pd.Series(range(len(idx)), index=idx)
# "print cluster ahead": a cpi or ppi print on any of the next 1..3 sessions
ahead = np.zeros(len(idx), bool)
evpos = sorted({int(pos.get(d)) for d in ev_dates if d in pos.index})
for p in evpos:
    for k in (1, 2, 3):
        if p - k >= 0:
            ahead[p - k] = True
ahead = pd.Series(ahead, index=idx)
# tighter: BOTH a ppi and a cpi inside the next 3 sessions (today's exact shape)
ppi_pos = {int(pos.get(d)) for d in load_events(['ppi'])['date'] if d in pos.index}
cpi_pos = {int(pos.get(d)) for d in load_events(['cpi'])['date'] if d in pos.index}
pair = np.zeros(len(idx), bool)
for i in range(len(idx)):
    nxt = set(range(i + 1, i + 4))
    if (nxt & ppi_pos) and (nxt & cpi_pos):
        pair[i] = True
pair = pd.Series(pair, index=idx)

print('\n=== step 2: THE CALENDAR EXPLANATION ===')
print('is the VIX3M-minus-VIX spread higher before a print? (if yes, the "signal" is the calendar)')
rows = []
for lbl, m in [('print in next 1-3 td', ahead), ('PPI+CPI pair in next 3 td', pair),
               ('no print next 1-3 td', ~ahead)]:
    x = spread[m.values & spread.notna().values]
    rows.append({'label': lbl, 'n': len(x), 'mean_spread_pp': round(x.mean(), 4),
                 'median': round(x.median(), 4), 'sd': round(x.std(), 3),
                 'P(spread>=1.68)': round(100 * (x >= 1.68).mean(), 2)})
show(rows)
a = spread[ahead.values & spread.notna().values]
b = spread[(~ahead).values & spread.notna().values]
se = np.sqrt(a.var() / len(a) + b.var() / len(b))
print('  pre-print minus other days = %+.4f pp, welch t = %+.2f' % (a.mean() - b.mean(), (a.mean() - b.mean()) / se))

# ---- step 3: dose response --------------------------------------------------
print('\n=== step 3: DOSE RESPONSE of forward return on the spread decile ===')
print('(vehicle long, lag=1 MOC, episode-declustered at gap=max(h,5))')

vehicles = {'SVXY': svxy, 'SPY': spy, 'VIX_SHORT': None}


def fwd(veh, h):
    if veh == 'VIX_SHORT':
        return -fwd_lag(vix, h, 1)
    return fwd_lag(vehicles[veh], h, 1)


for veh in ['SVXY', 'SPY', 'VIX_SHORT']:
    for h in (1, 2, 3):
        rr = fwd(veh, h)
        val = rr.dropna().index
        sp = spread.reindex(val)
        ok = sp.notna()
        val2 = val[ok.values]
        q = pd.qcut(sp[ok], 10, labels=False, duplicates='drop')
        rows = []
        for dec in sorted(pd.Series(q).dropna().unique()):
            dts = val2[(q == dec).values]
            e = declusters(pd.DatetimeIndex(dts), max(h, 5), val)
            v = rr.loc[e].values
            v = v[~np.isnan(v)]
            if len(v) < 3:
                continue
            rows.append({'decile': int(dec) + 1,
                         'spread_lo': round(sp[ok][(q == dec).values].min(), 2),
                         'spread_hi': round(sp[ok][(q == dec).values].max(), 2),
                         'n_epi': len(v), 'mean_pct': round(100 * v.mean(), 3),
                         'hit': round(100 * (v > 0).mean(), 1),
                         't': round(v.mean() / (v.std(ddof=1) / np.sqrt(len(v))), 2)})
        df = pd.DataFrame(rows)
        # monotonicity: spearman of decile vs mean
        rho = df['decile'].corr(df['mean_pct'], method='spearman') if len(df) > 2 else np.nan
        print('\n--- %s long, h=%d --- spearman(decile, mean) = %+.3f' % (veh, h, rho))
        print(df.to_string(index=False))
