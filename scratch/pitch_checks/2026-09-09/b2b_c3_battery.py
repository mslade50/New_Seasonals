"""C3 round 1b -- the catch-up leg is wrong-signed, so only GDX-long and the
CONTINUATION (long energy / short metals) are left. Decluster properly at
gap=21 and find out how many independent episodes there actually are.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

px = close_panel(['DBC', 'GLD', 'SLV', 'GDX', 'USO', 'XLE', 'SPY']).dropna()


def dd(s, lb=252):
    return s / rolling_on_valid(s, lambda x: x.rolling(lb).max()) - 1.0


dbc_hi = px['DBC'] >= rolling_on_valid(px['DBC'], lambda x: x.rolling(252).max()) - 1e-9
gdd, sdd = dd(px['GLD']), dd(px['SLV'])
cond = dbc_hi & (gdd <= -0.10) & (sdd <= -0.20)

print('### independent episodes (gap=21td) and WHEN they are')
epi21 = declusters(px.index[cond.values], 21, px.index)
print('N=%d :' % len(epi21), [str(d.date()) for d in epi21])
live = dbc_hi & (gdd <= -0.15) & (sdd <= -0.35)
print('LIVE-EXACT (GLD<=-15, SLV<=-35): %d days, episodes %s'
      % (int(live.sum()), [str(d.date()) for d in declusters(px.index[live.values], 21, px.index)]))
print('   all live-exact days:', [str(d.date()) for d in px.index[live.values]])

CAND = {'GDX long': [('GDX', 1.0)],
        'CONT: XLE - SLV': [('XLE', 1.0), ('SLV', -1.0)],
        'CONT: USO - GLD': [('USO', 1.0), ('GLD', -1.0)],
        'CATCHUP: GLD+SLV/2 - DBC': [('GLD', 0.5), ('SLV', 0.5), ('DBC', -1.0)]}

for name, legs in CAND.items():
    for h in (5, 10):
        ret = vehicle_ret(px, legs, h, 1)
        v = ret.loc[epi21.intersection(ret.dropna().index)].values
        if len(v) == 0:
            continue
        s = summarize(v, '%s h=%d gap21' % (name, h))
        w = int((v > 0).sum())
        print('\n%-26s h=%2d  N=%d  mean %+0.3f%%  hit %.0f%%  record %d-%d sign p %.4f'
              % (name, h, len(v), s['mean_pct'], s['hit'], w, len(v) - w, sign_test(w, len(v))))
        print('   per-episode: %s' % ', '.join(
            '%s %+0.2f' % (str(d.date()), 100 * x)
            for d, x in zip(epi21.intersection(ret.dropna().index), v)))
        base = ret.dropna()
        print('   all-day drift %+0.3f%%  edge %+0.3f pp  bootstrapP(<=0) %.3f'
              % (100 * base.mean(), s['mean_pct'] - 100 * base.mean(), bootstrap_p_le0(v)))

print('\n### definition neighbours, GDX long h=5, gap=21 episodes')
for lbl, m in [('GLD<=-5,SLV<=-10', dbc_hi & (gdd <= -0.05) & (sdd <= -0.10)),
               ('GLD<=-5,SLV<=-20', dbc_hi & (gdd <= -0.05) & (sdd <= -0.20)),
               ('GLD<=-10,SLV<=-20 (COND)', cond),
               ('GLD<=-12,SLV<=-25', dbc_hi & (gdd <= -0.12) & (sdd <= -0.25)),
               ('GLD<=-15,SLV<=-30', dbc_hi & (gdd <= -0.15) & (sdd <= -0.30)),
               ('DBC hi within 1% (not exact)',
                (px['DBC'] / rolling_on_valid(px['DBC'], lambda x: x.rolling(252).max()) >= 0.99)
                & (gdd <= -0.10) & (sdd <= -0.20)),
               ('DBC 126d high instead of 252d',
                (px['DBC'] >= rolling_on_valid(px['DBC'], lambda x: x.rolling(126).max()) - 1e-9)
                & (gdd <= -0.10) & (sdd <= -0.20))]:
    e = declusters(px.index[m.values], 21, px.index)
    ret = vehicle_ret(px, [('GDX', 1.0)], 5, 1)
    v = ret.loc[e.intersection(ret.dropna().index)].values
    if len(v):
        w = int((v > 0).sum())
        print('  %-30s N=%2d mean %+0.3f%% rec %d-%d p %.4f yrs %s'
              % (lbl, len(v), 100 * v.mean(), w, len(v) - w, sign_test(w, len(v)),
                 sorted(set(pd.DatetimeIndex(e).year))))
    else:
        print('  %-30s N=0' % lbl)

print('\n### is GDX just miner beta to a gold bounce? GDX residual of GLD')
h = 5
rg = fwd_lag(px['GLD'], h, 1)
rx = fwd_lag(px['GDX'], h, 1)
ok = rg.notna() & rx.notna()
beta = np.polyfit(rg[ok].values, rx[ok].values, 1)[0]
resid = rx - beta * rg
e5 = epi21.intersection(resid.dropna().index)
print('  beta(GDX~GLD, h=5) = %.2f ; GDX residual on COND episodes = %+0.3f%% (N=%d) vs all-day %+0.3f%%'
      % (beta, 100 * resid.loc[e5].mean(), len(e5), 100 * resid.dropna().mean()))
