"""C12 round 2 — long duration into a PPI print with ^TNX at/near a 252d high.

Round 1 left: RATE(0.1%) state, PPI in hold, long IEF h=3 = +0.406% (8-of-10);
h=5 +0.327% (13-of-18). Complement (state, no print) +0.022% / +0.071%.
Direction is LONG, opposite watchlist 41's short. Jaccard vs the commodity
mask 0.036, so not a re-skin.

Round 2 must kill it:
 (a) the RATES-gate complement — does long duration into PPI pay with NO rates
     condition at all? If yes the gate is decoration.
 (b) placebo ladder on TODAY's tight gate.
 (c) session decomposition — where in the hold does it accrue? The release
     session is the claimed mechanism.
 (d) concentration / era / midterm / September / pair.
 (e) permutation charged for the 4-vehicle x 3-horizon x 3-gate search.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

raw = load_prices(['TLT', 'IEF', 'SPY', '^TNX'])
px = pd.DataFrame({t: raw[t]['Close'] for t in ('TLT', 'IEF', 'SPY')}).dropna()
IDX = px.index
tnx = raw['^TNX']['Close'].reindex(IDX)
tnx_off = tnx / rolling_on_valid(tnx, lambda x: x.rolling(252).max()) - 1
GATES = {'RATE 0.1% (LIVE)': (tnx_off >= -0.001).fillna(False),
         'RATE 0.5%': (tnx_off >= -0.005).fillna(False),
         'RATE 2%': (tnx_off >= -0.02).fillna(False),
         'NO GATE (parent)': pd.Series(True, index=IDX)}
ppi = load_events(['ppi'])['date']
cpi = load_events(['cpi'])['date']

# ---- (a) RATES-GATE COMPLEMENT --------------------------------------------
print('################ (a) RATES-GATE COMPLEMENT ################')
print('does long duration into a PPI print need the rates state at all?\n')
for tkr, cost in (('IEF', 3.0), ('TLT', 3.0)):
    for h in (1, 2, 3, 5):
        ret = vehicle_ret(px, [(tkr, 1.0)], h, 1)
        valid = ret.dropna().index
        rows = []
        for gname, gm in GATES.items():
            st = IDX[gm.reindex(IDX, fill_value=False).values].intersection(valid)
            epi = declusters(st, max(h, 5), valid)
            fl = event_in_window(epi, IDX, h, 1, ('ppi',))
            a = summarize(ret.loc[epi[fl]].values, '%s | PPI in hold' % gname)
            b = summarize(ret.loc[epi[~fl]].values, '%s | no PPI' % gname)
            a['print_gate_pp'] = round(a.get('mean_pct', np.nan) - b.get('mean_pct', np.nan), 3)
            rows += [a, b]
        show(rows, 'long %s h=%d' % (tkr, h))
        base = [r for r in rows if r['label'].startswith('NO GATE') and 'in hold' in r['label']][0]
        live = [r for r in rows if r['label'].startswith('RATE 0.1%') and 'in hold' in r['label']][0]
        print('  RATES gate worth %+.3fpp over the ungated PPI parent '
              '(%.3f%% on n=%d vs %.3f%% on n=%d);  parent print gate %+.3fpp'
              % (live['mean_pct'] - base['mean_pct'], live['mean_pct'], live['n'],
                 base['mean_pct'], base['n'], base['print_gate_pp']))
        print('  cost: %.1f bps edge vs %.1f bps round trip = %.1fx'
              % (100 * live['mean_pct'], cost, 100 * live['mean_pct'] / cost))

# ---- (b) PLACEBO LADDER on the live gate ----------------------------------
print('\n\n################ (b) PLACEBO LADDER, live RATE 0.1% gate ############')
LIVE = GATES['RATE 0.1% (LIVE)']


def ladder(tkr, h, gm, gname):
    ret = vehicle_ret(px, [(tkr, 1.0)], h, 1)
    valid = ret.dropna().index
    rows = []
    for k in range(-5, 6):
        pos, _ = anchor_positions(IDX, ppi, -2 + k)
        a = pd.DatetimeIndex(sorted(set(IDX[p] for p in pos)))
        a = a.intersection(valid).intersection(IDX[gm.reindex(IDX, fill_value=False).values])
        epi = declusters(a, max(h, 5), valid)
        rows.append(summarize(ret.loc[epi].values, 'k=%+d%s' % (k, ' <TRUE>' if k == 0 else '')))
    show(rows, 'LADDER long %s h=%d gate=%s' % (tkr, h, gname))
    vv = np.asarray([r.get('mean_pct', np.nan) if r.get('n') else -1e9 for r in rows], float)
    vv = np.where(np.isnan(vv), -1e9, vv)
    rank = int(np.where(np.argsort(-vv) == 5)[0][0]) + 1
    print('  TRUE k=0 = %.3f%% (n=%s) ranks %d of 11'
          % (rows[5].get('mean_pct', np.nan), rows[5].get('n'), rank))


for tkr in ('IEF', 'TLT'):
    for h in (1, 3, 5):
        ladder(tkr, h, LIVE, 'RATE 0.1%')

# ---- (c) SESSION DECOMPOSITION --------------------------------------------
print('\n\n################ (c) SESSION DECOMPOSITION of the h=3 hold ##########')
print('entry = close D+1 (session before the release). D+2 = release session.')
pos = pd.Series(range(len(IDX)), index=IDX)
for gname in ('RATE 0.1% (LIVE)', 'RATE 0.5%', 'RATE 2%', 'NO GATE (parent)'):
    gm = GATES[gname]
    for tkr in ('IEF', 'TLT'):
        ret3 = vehicle_ret(px, [(tkr, 1.0)], 3, 1)
        valid = ret3.dropna().index
        st = IDX[gm.reindex(IDX, fill_value=False).values].intersection(valid)
        epi = declusters(st, 5, valid)
        fl = event_in_window(epi, IDX, 3, 1, ('ppi',))
        ep = epi[fl]
        gaps, d1, d2, d3, tot = [], [], [], [], []
        c = px[tkr]; o = raw[tkr]['Open'].reindex(IDX)
        for d in ep:
            p = pos[d]
            if p + 4 >= len(IDX):
                continue
            e = c.iloc[p + 1]
            gaps.append(o.iloc[p + 2] / e - 1)
            d1.append(c.iloc[p + 2] / e - 1)
            d2.append(c.iloc[p + 3] / c.iloc[p + 2] - 1)
            d3.append(c.iloc[p + 4] / c.iloc[p + 3] - 1)
            tot.append(c.iloc[p + 4] / e - 1)
        if not tot:
            continue
        g, D1, D2, D3, T = map(lambda x: np.asarray(x), (gaps, d1, d2, d3, tot))
        print('%-20s %s n=%d | overnight gap into release %+.4f%% | RELEASE SESSION '
              'close-to-close %+.4f%% (%.0f%% of hold) | day2 %+.4f%% | day3 %+.4f%% '
              '| hold %+.4f%%'
              % (gname, tkr, len(T), 100 * g.mean(), 100 * D1.mean(),
                 100 * D1.mean() / T.mean() if T.mean() else np.nan,
                 100 * D2.mean(), 100 * D3.mean(), 100 * T.mean()))

# ---- (d) concentration / era / midterm / September -------------------------
print('\n\n################ (d) CONCENTRATION / ERA / MIDTERM / SEPTEMBER ######')
for tkr in ('IEF', 'TLT'):
    for h in (3, 5):
        ret = vehicle_ret(px, [(tkr, 1.0)], h, 1)
        valid = ret.dropna().index
        st = IDX[LIVE.values].intersection(valid)
        epi = declusters(st, max(h, 5), valid)
        fl = event_in_window(epi, IDX, h, 1, ('ppi',))
        ep = epi[fl]; v = ret.loc[ep].values
        print('\nlong %s h=%d  n=%d  mean %+.3f%%  record %d-%d sign p %.4f  boot P(<=0) %.3f'
              % (tkr, h, len(v), 100 * v.mean(), int((v > 0).sum()), int((v <= 0).sum()),
                 sign_test(int((v > 0).sum()), len(v)), bootstrap_p_le0(v)))
        print('  ', cluster_note(ep, v))
        s = np.sort(v)[::-1]
        print('  drop-best-1 %+.3f%% (n=%d)  drop-best-2 %+.3f%% (n=%d)'
              % (100 * s[1:].mean(), len(s) - 1, 100 * s[2:].mean(), len(s) - 2))
        mt = np.array([d.year % 4 == 2 for d in ep])
        show(era_split(ep, v) + [summarize(v[mt], 'midterm'), summarize(v[~mt], 'non-midterm')],
             'long %s h=%d splits' % (tkr, h))
        print('  dates:', ', '.join('%s(%+.2f%%)' % (d.date(), 100 * r) for d, r in zip(ep, v)))
        mon = pd.Series(v).groupby(ep.month).agg(['size', 'mean'])
        mon['mean'] = (100 * mon['mean']).round(3)
        print('  by month:\n', mon.to_string())

# ---- (e) permutation charged for my own search ----------------------------
print('\n\n################ (e) PERMUTATION for the C12 search ################')
print('grid searched: {IEF,TLT} x {long,short} x h in {1,2,3,5} x 3 gates = 48 cells')
ret3 = vehicle_ret(px, [('IEF', 1.0)], 3, 1)
valid3 = ret3.dropna().index
st = IDX[LIVE.values].intersection(valid3)
epi = declusters(st, 5, valid3)
fl = event_in_window(epi, IDX, 3, 1, ('ppi',))
OBS = 100 * ret3.loc[epi[fl]].mean()
n_ep = int(fl.sum())
print('tested statistic = long IEF h=3, live gate, PPI in hold = %+.3f%% on n=%d' % (OBS, n_ep))
retmap = {}
for tkr in ('IEF', 'TLT'):
    for sgn in (1.0, -1.0):
        for h in (1, 2, 3, 5):
            retmap[(tkr, sgn, h)] = vehicle_ret(px, [(tkr, sgn)], h, 1)
rng = np.random.default_rng(11)
pool = valid3[:-11]
nulls, solo = [], []
for _ in range(4000):
    picks = pd.DatetimeIndex(rng.choice(pool, size=n_ep, replace=False))
    best = -np.inf
    for key, r in retmap.items():
        rr = r.reindex(picks).dropna()
        if len(rr) >= n_ep - 1:
            best = max(best, 100 * rr.mean())
    nulls.append(best)
    solo.append(100 * retmap[('IEF', 1.0, 3)].reindex(picks).dropna().mean())
nulls = np.asarray(nulls); solo = np.asarray(solo)
print('P(null max-over-16-vehicle-horizon cells >= %+.3f%%) = %.4f  (null max median %.3f%%)'
      % (OBS, (nulls >= OBS).mean(), np.median(nulls)))
print('single-cell P(null >= %+.3f%%) = %.4f  (uncharged, for contrast)'
      % (OBS, (solo >= OBS).mean()))
