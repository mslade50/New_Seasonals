"""C4 round 2: the cell as SPECIFIED, measured -- so the kill is not only
"today does not qualify".

a2c established today's state is the LEFT tail (spread -3.76pp, 11.4th pctile;
the +1.68pp reading came from a ^VIX bar dated 2026-09-07, LABOR DAY, that
^VIX3M and SPY do not have). This script does two separate things:

  PART A  contamination scope -- how many holiday phantom bars does ^VIX carry,
          and does anything else in the vol complex share them?
  PART B  build the cell EXACTLY as C4 specifies (3M bid over spot, spot VIX in
          a low band, a print inside the next few sessions) and measure it on
          the common calendar. If it has no edge even when it fires, the kill
          does not depend on today's reading at all.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import rolling_on_valid
import numpy as np, pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


def level_rank(s, lookback=252):
    return rolling_on_valid(s, lambda x: x.rolling(lookback).rank(pct=True) * 100.0)


print('=' * 72)
print('PART A -- contamination scope of the phantom holiday bars')
print('=' * 72)
mp = pd.read_parquet(PRICES_PATH, columns=['date', 'ticker'])
mp['date'] = pd.to_datetime(mp['date'])
hol = USFederalHolidayCalendar().holidays(start='2000-01-01', end='2026-12-31')
spy_days = set(mp[mp['ticker'] == 'SPY']['date'])
for t in ['^VIX', '^VIX3M', '^SKEW', '^MOVE', 'SPY', '^GSPC']:
    d = set(mp[mp['ticker'] == t]['date'])
    ph = sorted(x for x in d if x in set(hol))
    extra = sorted(x for x in d - spy_days if x >= pd.Timestamp('2000-01-03'))
    print('%-8s bars=%5d  on a federal holiday: %3d  not-in-SPY-calendar: %3d  last 3 such: %s'
          % (t, len(d), len(ph), len(extra), [str(x.date()) for x in extra[-3:]]))

v = load_prices(['^VIX'])['^VIX']['Close']
ph = sorted(x for x in v.index if x in set(hol))
print('\n^VIX holiday bars (all): %s' % [str(x.date()) for x in ph])
print('  -> any 1-day ^VIX change computed on ^VIX\'s OWN calendar is wrong on the')
print('     session AFTER each of these. Reindex to SPY before differencing.')

print('\n' + '=' * 72)
print('PART B -- the C4 cell as specified, measured on the COMMON calendar')
print('=' * 72)
TK = ['SPY', 'SVXY', '^VIX', '^VIX3M']
px_d = load_prices(TK)
spy = px_d['SPY']['Close']
idx = spy.index                      # SPY calendar = the real session calendar
vix = px_d['^VIX']['Close'].reindex(idx)
v3m = px_d['^VIX3M']['Close'].reindex(idx)
svxy = px_d['SVXY']['Close'].reindex(idx)
panel = pd.DataFrame({'SPY': spy, 'SVXY': svxy})

d_v3 = rolling_on_valid(v3m, lambda x: x.pct_change()) * 100
d_vx = rolling_on_valid(vix, lambda x: x.pct_change()) * 100
spread = d_v3 - d_vx
vix_pct = level_rank(vix, 252)

# print inside the next 1-3 sessions
pos = pd.Series(range(len(idx)), index=idx)
evp = {int(pos.get(d)) for d in load_events(['cpi', 'ppi'])['date'] if d in pos.index}
ahead = pd.Series([any((i + k) in evp for k in (1, 2, 3)) for i in range(len(idx))], index=idx)

print('live readings on the real calendar: spread %+.2fpp | VIX pctile %.1f | print ahead %s'
      % (spread.iloc[-1], vix_pct.iloc[-1], bool(ahead.iloc[-1])))

for sp_thr in (1.0, 1.5, 2.0, 3.0):
    for vlo in (25, 33, 50):
        m = (spread >= sp_thr) & (vix_pct <= vlo) & ahead
        m = m.fillna(False)
        rows = []
        for veh in ['SVXY', 'SPY']:
            for h in (1, 2, 3):
                r = fwd_lag(panel[veh], h, 1)
                val = r.dropna().index
                tr = pd.DatetimeIndex(idx[m.values]).intersection(val)
                if len(tr) < 5:
                    continue
                e = declusters(tr, max(h, 5), val)
                x = r.loc[e].values
                base = r.loc[val]
                w = int((x > 0).sum())
                rows.append({'veh': veh, 'h': h, 'n': len(x),
                             'mean_pct': round(100 * x.mean(), 3),
                             'edge_pct': round(100 * (x.mean() - base.mean()), 3),
                             'rec': '%d-%d' % (w, len(x) - w),
                             't': round(x.mean() / (x.std(ddof=1) / np.sqrt(len(x))), 2),
                             'sign_p': round(sign_test(w, len(x)), 3)})
        if rows:
            print('\n--- spread >= %.1fpp, VIX pctile <= %d, print in next 1-3 td  (%d days) ---'
                  % (sp_thr, vlo, int(m.sum())))
            print(pd.DataFrame(rows).to_string(index=False))

print('\n=== and the COMPLEMENT of the print gate (kill #2): same vol state, NO print ahead ===')
for sp_thr in (1.5, 2.0):
    m_on = ((spread >= sp_thr) & (vix_pct <= 33) & ahead).fillna(False)
    m_off = ((spread >= sp_thr) & (vix_pct <= 33) & ~ahead).fillna(False)
    for veh in ['SVXY', 'SPY']:
        r = fwd_lag(panel[veh], 1, 1)
        val = r.dropna().index
        out = []
        for lbl, mm in [('print AHEAD', m_on), ('NO print', m_off)]:
            tr = pd.DatetimeIndex(idx[mm.values]).intersection(val)
            e = declusters(tr, 5, val)
            x = r.loc[e].values
            w = int((x > 0).sum())
            out.append('%s n=%d %+.3f%% (%d-%d)' % (lbl, len(x), 100 * x.mean(), w, len(x) - w))
        print('  spread>=%.1f %-5s h=1: %s | %s' % (sp_thr, veh, out[0], out[1]))
