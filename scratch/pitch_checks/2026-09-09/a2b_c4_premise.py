"""C4 round 1b: VERIFY THE PREMISE against the raw bars.

The candidate (and the surface map it came from) states: "On 2026-09-08 ^VIX3M
rose +4.43% while ^VIX rose only +2.75%". a2 computed dVIX = +8.19% on each
instrument's OWN calendar, which would make today the 11.4th percentile of the
spread -- the OPPOSITE tail from the one the idea is built on.

One of the two is wrong. This reads the raw per-ticker frames with no reindex,
no ffill and no panel alignment, and prints the actual closes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

px = load_prices(['^VIX', '^VIX3M', 'SPY'])
for t in ['^VIX', '^VIX3M']:
    s = px[t]['Close']
    print('\n--- %s raw tail (own calendar, no reindex) ---' % t)
    tail = pd.DataFrame({'close': s.tail(8), 'pct_chg': (100 * s.pct_change()).tail(8)})
    print(tail.round(4).to_string())

v = px['^VIX']['Close']
v3 = px['^VIX3M']['Close']
print('\nVIX  last two closes: %s = %.4f , %s = %.4f  -> %+.2f%%'
      % (v.index[-2].date(), v.iloc[-2], v.index[-1].date(), v.iloc[-1],
         100 * (v.iloc[-1] / v.iloc[-2] - 1)))
print('VIX3M last two closes: %s = %.4f , %s = %.4f  -> %+.2f%%'
      % (v3.index[-2].date(), v3.iloc[-2], v3.index[-1].date(), v3.iloc[-1],
         100 * (v3.iloc[-1] / v3.iloc[-2] - 1)))

r_prev = v.iloc[-2] / v3.iloc[-2]
r_now = v.iloc[-1] / v3.iloc[-1]
print('\nVIX/VIX3M ratio  %s -> %s :  %.4f -> %.4f  (change %+.4f)'
      % (v.index[-2].date(), v.index[-1].date(), r_prev, r_now, r_now - r_prev))
print('spread (dVIX3M - dVIX) = %+.3f pp'
      % (100 * (v3.iloc[-1] / v3.iloc[-2] - 1) - 100 * (v.iloc[-1] / v.iloc[-2] - 1)))

print('\nINTERPRETATION:')
if r_now > r_prev:
    print('  the ratio ROSE -> SPOT vol was bid MORE than 3M vol -> the curve FLATTENED.')
    print('  That is the OPPOSITE of "three-month vol bid while spot vol stays dead".')
else:
    print('  the ratio FELL -> 3M bid more than spot -> the curve steepened, premise holds.')

# where does the true spread sit historically?
sp = (v3.pct_change() * 100).reindex(v.index) - (v.pct_change() * 100)
sp = sp.dropna()
today = sp.iloc[-1]
print('\ntrue spread today %+.3f pp = %.1fth pctile of %d sessions (mean %+.3f)'
      % (today, 100 * (today > sp).mean(), len(sp), sp.mean()))
print('  the idea needs the RIGHT tail; today is in the %s tail.'
      % ('right' if today > sp.median() else 'LEFT'))
