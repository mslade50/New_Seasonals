"""C8 round 2c: is the surviving DIRECTION of C8 already a live book trade?

The collision cell in a midterm year is SHORT (-1.753%, 4-6). The Event Sleeve
carries T2 FOMC_MIDTERM_SHORT: short SPY, entry N sessions before a scheduled
FOMC decision, midterm years only, gated on SPY's 21d-return rank. If that gate
passes for the 2026-09-16 FOMC, C8's only non-dead direction is an order the
book is already placing, which is a duplicate rather than a novel idea.

Read the config rather than trusting the doc.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
import numpy as np, pandas as pd

root = Path(__file__).resolve().parents[3]
src = (root / 'event_sleeve.py').read_text(encoding='utf-8', errors='replace')

import re
i = src.find('EVENT_SLEEVE')
print('--- event_sleeve.py: EVENT_SLEEVE dict + FOMC entry constant ---')
print(src[i:i + 2200])
m = re.search(r'FOMC_ENTRY_TD_BEFORE\s*=\s*(\d+)', src)
print('\nFOMC_ENTRY_TD_BEFORE =', m.group(1) if m else 'NOT FOUND')

px = load_prices(['SPY'])['SPY']['Close']
r21 = pct_rank(px, 21, 252)
print('\nSPY 21d-return rank(252): 09-08 = %.1f  (lag-1, 09-04 = %.1f)'
      % (r21.iloc[-1], r21.iloc[-2]))
print('T2 gate is "rank < 50" -> %s' % ('PASSES, T2 is live for the 09-16 FOMC'
                                        if r21.iloc[-1] < 50 else 'FAILS'))

ev = load_events(['fomc_decision'])
nxt = ev[ev['date'] > pd.Timestamp('2026-09-08')]['date'].iloc[0]
print('next FOMC decision: %s   midterm year? %s' % (nxt.date(), nxt.year % 4 == 2))
