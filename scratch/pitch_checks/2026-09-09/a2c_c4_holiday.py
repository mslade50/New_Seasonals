"""C4 round 1c: is 2026-09-07 a real session, or a phantom bar in ^VIX?

a2b found ^VIX carries a 2026-09-07 close of 15.30 while ^VIX3M and SPY jump
2026-09-04 -> 2026-09-08. 2026-09-07 is the first Monday of September 2026 =
LABOR DAY, a US market holiday. If ^VIX is close to alone in having that bar,
then the candidate's "+2.75% VIX" is measured against a bar that does not
exist, and the real session move is 14.53 -> 15.72 = +8.19%.

That inverts the premise: spot vol bid HARDER than 3M vol, curve FLATTENED.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

mp = pd.read_parquet(PRICES_PATH, columns=['date', 'ticker', 'Close'])
mp['date'] = pd.to_datetime(mp['date'])

for d in ['2026-09-03', '2026-09-04', '2026-09-07', '2026-09-08']:
    sub = mp[mp['date'] == d]
    print('%s : %5d tickers with a bar' % (d, len(sub)))

sub = mp[mp['date'] == '2026-09-07']
print('\nthe FULL list of tickers carrying a 2026-09-07 bar:')
print(sorted(sub['ticker'].unique()))

# US market holiday check, independent of the cache
from pandas.tseries.holiday import USFederalHolidayCalendar
cal = USFederalHolidayCalendar()
hol = cal.holidays(start='2026-01-01', end='2026-12-31')
print('\n2026-09-07 in the US federal holiday calendar? %s' % (pd.Timestamp('2026-09-07') in hol))
print('  (Labor Day 2026 = %s)' % [str(h.date()) for h in hol if h.month == 9][:1])

# the number that decides C4
v = load_prices(['^VIX'])['^VIX']['Close']
v3 = load_prices(['^VIX3M'])['^VIX3M']['Close']
print('\n=== the premise, recomputed over the REAL last session (09-04 -> 09-08) ===')
dv = 100 * (v.loc['2026-09-08'] / v.loc['2026-09-04'] - 1)
dv3 = 100 * (v3.loc['2026-09-08'] / v3.loc['2026-09-04'] - 1)
print('  VIX   14.53 -> 15.72 = %+.2f%%' % dv)
print('  VIX3M 17.61 -> 18.39 = %+.2f%%' % dv3)
print('  spread (dVIX3M - dVIX) = %+.2f pp   [candidate claimed +1.68pp]' % (dv3 - dv))
print('  ratio  %.4f -> %.4f  = %+.4f  -> the curve %s'
      % (v.loc['2026-09-04'] / v3.loc['2026-09-04'], v.loc['2026-09-08'] / v3.loc['2026-09-08'],
         v.loc['2026-09-08'] / v3.loc['2026-09-08'] - v.loc['2026-09-04'] / v3.loc['2026-09-04'],
         'FLATTENED (spot bid harder)' if v.loc['2026-09-08'] / v3.loc['2026-09-08'] >
         v.loc['2026-09-04'] / v3.loc['2026-09-04'] else 'steepened'))

# where the true reading sits in the spread distribution, computed on the
# COMMON calendar of the two indices (the only honest basis)
common = v.index.intersection(v3.index)
sp = (100 * v3.loc[common].pct_change()) - (100 * v.loc[common].pct_change())
sp = sp.dropna()
tv = dv3 - dv
print('\nspread distribution on the COMMON ^VIX/^VIX3M calendar: n=%d mean %+.3f sd %.3f'
      % (len(sp), sp.mean(), sp.std()))
print('  today %+.2f pp = %.1fth pctile   -> %s tail'
      % (tv, 100 * (tv > sp).mean(), 'RIGHT' if tv > sp.median() else 'LEFT'))
print('  the candidate needed the RIGHT tail (3M bid, spot dead).')
