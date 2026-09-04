"""Is the late-August corn/wheat gap a continuous-contract roll artifact?

Decisive test: for each year, find the largest single overnight gap in
ZC=F / ZW=F inside Aug 15 - Sep 05. A roll shows up as a same-sized,
same-signed gap in the same calendar window EVERY year.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import pandas as pd
from pitch_lab import load_prices

TK = ['ZC=F', 'ZW=F', 'ZS=F', 'KC=F', 'LE=F']
px = load_prices(TK)

for t in TK:
    d = px[t].copy()
    d['gap'] = (d['Open'] / d['Close'].shift(1) - 1) * 100
    d['sess'] = d['Close'].pct_change() * 100
    print('===', t, ' bars', len(d), d.index[0].date(), '->', d.index[-1].date())
    rows = []
    for yr in range(1999, 2027):
        w = d[(d.index >= f'{yr}-08-15') & (d.index <= f'{yr}-09-05')]
        w = w.dropna(subset=['gap'])
        if w.empty:
            continue
        i = w['gap'].abs().idxmax()
        rows.append((yr, str(i.date()), w.loc[i, 'gap'], w.loc[i, 'sess']))
    for yr, dt, g, s in rows:
        flag = '  <== |gap|>3%' if abs(g) > 3 else ''
        print('   %d  biggest Aug15-Sep05 gap %s  %7.2f%%  (session %7.2f%%)%s' % (yr, dt, g, s, flag))
    gaps = np.array([r[2] for r in rows])
    print('   -> median biggest-gap %.2f%%, years with |gap|>3%%: %d of %d, years with gap>+3%%: %d'
          % (np.median(gaps), int((np.abs(gaps) > 3).sum()), len(gaps), int((gaps > 3).sum())))

    # how unusual is a >3% gap anywhere in the year for this ticker
    allg = d['gap'].dropna()
    print('   -> all-history: %d of %d sessions gap>+3%% (%.2f%%); by month:'
          % (int((allg > 3).sum()), len(allg), 100 * (allg > 3).mean()))
    bm = allg[allg > 3].groupby(allg[allg > 3].index.month).size()
    print('      ', dict(bm))
