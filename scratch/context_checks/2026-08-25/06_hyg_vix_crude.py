"""Three leftovers: HYG's late-August drift with controls, the VIX doy cell
in midterm years, and crude's -4.59% intraday session against the grains.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))

import numpy as np
import pandas as pd
from pitch_lab import (load_prices, fwd_ret, summarize, sign_test, declusters,
                       local_control, era_split, cluster_note)
from seasonal_edge import seasonal_window_returns

asof = pd.Timestamp('2026-08-25')
px = load_prices(['HYG', '^VIX', 'CL=F', 'ZC=F', 'ZW=F', 'ZS=F', 'SPY'])

# ---------------- HYG: control the late-August drift ----------------
print('=' * 70)
print('HYG late-August 5-session cell vs its own baseline')
h = px['HYG']['Close'].astype(float)
f5 = fwd_ret(h, 5)
print('  unconditional h5: %s' % summarize(f5.dropna().values, 'all days'))
aug = f5[f5.index.month == 8].dropna()
print('  all August h5   : %s' % summarize(aug.values, 'August'))
lastweek = f5[(f5.index.month == 8) & (f5.index.day >= 20)].dropna()
print('  Aug 20-31 h5    : %s' % summarize(lastweek.values, 'late Aug'))
st = seasonal_window_returns(px['HYG'], asof, 5)
print('  the doy cell    : n=%d mean %+.3f%% up %d down %d sign_p %.4f'
      % (st['n'], 100 * st['mean'], st['n_up'], st['n_down'],
         sign_test(max(st['n_up'], st['n_down']), st['n'])))
yrs, rets = st['years'], np.array(st['rets'])
mid = np.array([y % 4 == 2 for y in yrs])
print('  midterm years n=%d mean %+.3f%% up %d down %d'
      % (mid.sum(), 100 * rets[mid].mean(), int((rets[mid] > 0).sum()),
         int((rets[mid] <= 0).sum())))
print('  tonight HYG closed at its 252d high (dist 0.00%), 21d rank 92.9')

# ---------------- VIX: the doy cell in midterm years ----------------
print('=' * 70)
print('VIX at this trading day of year')
for lab, filt in (('all years', None), ('midterm', 2)):
    for hz in (1, 5):
        st = seasonal_window_returns(px['^VIX'], asof, hz, cycle_phase_filter=filt)
        if not st or st.get('insufficient'):
            print('  %s h%d insufficient (n=%s)' % (lab, hz, (st or {}).get('n')))
            continue
        r = np.array(st['rets'])
        print('  %-9s h%d: n=%d mean %+.2f%% median %+.2f%% up %d down %d sign_p %.4f'
              % (lab, hz, st['n'], 100 * st['mean'], 100 * st['median'],
                 st['n_up'], st['n_down'],
                 sign_test(max(st['n_up'], st['n_down']), st['n'])))
        print('        years:', ', '.join('%s %+.1f%%' % (y, 100 * v)
                                          for y, v in zip(st['years'], r)))

# ---------------- crude down hard while the grains rally ----------------
print('=' * 70)
print('CL=F -4%+ on a session the grain complex rallies')
cl = px['CL=F']['Close'].astype(float)
gr = pd.DataFrame({t: px[t]['Close'].astype(float) for t in ['ZC=F', 'ZW=F', 'ZS=F']}).dropna()
grx = gr.pct_change(fill_method=None).mean(axis=1)
clr = cl.pct_change(fill_method=None)
al = pd.DataFrame({'cl': clr, 'gr': grx}).dropna()
print('  tonight: CL %+.2f%%, grain complex %+.2f%%'
      % (100 * al['cl'].iloc[-1], 100 * al['gr'].iloc[-1]))
mask = (al['cl'] <= -0.04) & (al['gr'] >= 0.02)
trig = al.index[mask.values]
print('  cell (CL <= -4%% and grains >= +2%%): %d sessions of %d' % (len(trig), len(al)))
print('  dates:', [str(d.date()) for d in trig])
if len(trig) >= 3:
    ep = declusters(trig, 5, al.index)
    print('  %d episodes at 5td' % len(ep))
    for hz in (1, 5, 10):
        for name, s in (('CL=F', cl), ('SPY', px['SPY']['Close'].astype(float))):
            v = fwd_ret(s, hz).reindex(ep).dropna()
            if len(v) >= 3:
                print('   %s h%d: %s sign_p %.4f'
                      % (name, hz, summarize(v.values, ''),
                         sign_test(int((v.values > 0).sum()), len(v))))

# looser: crude's own -4% intraday sessions
print('\n  looser cell: CL=F session <= -4%%')
m2 = (clr <= -0.04).fillna(False)
t2 = cl.index[m2.values]
ep2 = declusters(t2, 5, cl.index)
print('  %d sessions, %d episodes' % (len(t2), len(ep2)))
for hz in (1, 5, 10):
    v = fwd_ret(cl, hz).reindex(ep2).dropna()
    r = summarize(v.values, 'CL h%d' % hz)
    print('   %s sign_p %.4f' % (r, sign_test(int((v.values > 0).sum()), len(v))))
    lc = local_control(cl.index, ep2, 126)
    print('     CTRL local:', summarize(fwd_ret(cl, hz).reindex(lc).dropna().values, ''))
v5 = fwd_ret(cl, 5).reindex(ep2).dropna()
print('  h5 era:', era_split(v5.index, v5.values)[-1])
print('  h5 concentration:', cluster_note(v5.index, v5.values))
