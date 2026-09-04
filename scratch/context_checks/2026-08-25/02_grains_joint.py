"""Corn, wheat and beans stretched at the same time.

Tonight: ZC z10 2.95 (5d +13.2%, 21d +16.1%, both 100th pctile, 52w high),
ZW z10 2.00, ZS z10 2.21. How often is the whole grain complex extended
together, and what follows?

z10 uses the ENGINE definition (10d return / 21d vol scaled to 10d), not
pitch_lab.zscore, so the trigger matches the payload's tape block.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import pandas as pd
from pitch_lab import (load_prices, fwd_ret, declusters, local_control,
                       summarize, era_split, sign_test, cluster_note, show)

TK = ['ZC=F', 'ZW=F', 'ZS=F']


def z10(close: pd.Series) -> pd.Series:
    r10 = close.pct_change(10, fill_method=None)
    vol21 = close.pct_change(fill_method=None).rolling(21).std()
    return r10 / (vol21 * np.sqrt(10))


px = load_prices(TK)
cl = pd.DataFrame({t: px[t]['Close'] for t in TK}).dropna()
print('panel', cl.index[0].date(), '->', cl.index[-1].date(), len(cl), 'sessions')

z = pd.DataFrame({t: z10(cl[t]) for t in TK})
print('tonight z10:', {t: round(float(z[t].iloc[-1]), 2) for t in TK})

joint = ((z['ZC=F'] >= 2) & (z['ZW=F'] >= 2) & (z['ZS=F'] >= 2)).fillna(False)
trig = cl.index[joint.values]
print('\nall three grains z10 >= 2 on the same close: %d of %d sessions (%.2f%%)'
      % (len(trig), len(cl), 100 * len(trig) / len(cl)))
ep = declusters(trig, 10, cl.index)
print('declustered at 10 td: %d episodes' % len(ep))
print('episodes:', [str(d.date()) for d in ep])
print('by year:', dict(pd.Series(1, index=ep).groupby(ep.year).size()))
print('by month:', dict(pd.Series(1, index=ep).groupby(ep.month).size()))
print('tonight in trigger set:', bool(joint.iloc[-1]))

cx = cl.pct_change(fill_method=None).mean(axis=1)
cxlvl = (1 + cx.fillna(0)).cumprod()

rows = []
for h in (1, 3, 5, 10, 21):
    f = fwd_ret(cxlvl, h)
    v = f.reindex(ep).dropna()
    r = summarize(v.values, 'complex h%d' % h)
    r['sign_p'] = round(sign_test(int((v.values > 0).sum()), len(v)), 4)
    rows.append(r)
    rows.append(summarize(f.dropna().values, '  CTRL all days h%d' % h))
    lc = local_control(cl.index, ep, 126)
    rows.append(summarize(f.reindex(lc).dropna().values, '  CTRL local+-126 h%d' % h))
show(rows, 'JOINT grain stretch -> equal-weight complex, with controls')

rows = []
for t in TK:
    for h in (1, 5, 10, 21):
        f = fwd_ret(cl[t], h)
        v = f.reindex(ep).dropna()
        if len(v) < 3:
            continue
        r = summarize(v.values, '%s h%d' % (t, h))
        r['sign_p'] = round(sign_test(int((v.values > 0).sum()), len(v)), 4)
        rows.append(r)
show(rows, 'JOINT grain stretch -> per leg')

for h in (5, 21):
    f = fwd_ret(cxlvl, h)
    v = f.reindex(ep).dropna()
    if len(v) >= 4:
        print('\nh%d concentration: %s' % (h, cluster_note(v.index, v.values)))
        print('h%d era: %s' % (h, era_split(v.index, v.values)[-1]))

# per-episode detail, since N is small
print('\nper-episode forward moves in the complex:')
f5, f21 = fwd_ret(cxlvl, 5), fwd_ret(cxlvl, 21)
for d in ep:
    print('  %s  z10 ZC %5.2f ZW %5.2f ZS %5.2f   h5 %7.2f%%  h21 %7.2f%%'
          % (d.date(), z['ZC=F'].get(d, np.nan), z['ZW=F'].get(d, np.nan),
             z['ZS=F'].get(d, np.nan),
             100 * f5.get(d, np.nan), 100 * f21.get(d, np.nan)))
