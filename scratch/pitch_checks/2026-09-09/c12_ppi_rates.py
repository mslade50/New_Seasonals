"""C12 round 1 — a PPI print landing with the ten-year already at a 252d high.

Anchor: signal close D = print - 2 td, entry MOC D+1 (the session before the
08:30 release), so h=1 exits on the release close. Placebo ladder k=-5..+5.
Mandatory extras: gap-share (does the release itself move it?) and the Jaccard
against watchlist 41's COMMODITY-high mask (is C12 a re-skin?).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from pitch_lab import *  # noqa
from pitch_lab import _valid_pct_change, rolling_on_valid
import numpy as np, pandas as pd

TK = ['TLT', 'IEF', 'LQD', 'SPY', 'DBC']
raw = load_prices(TK + ['^TNX'])
px = pd.DataFrame({t: raw[t]['Close'] for t in TK}).dropna(subset=['TLT', 'IEF'])
px = px[['TLT', 'IEF', 'LQD', 'SPY', 'DBC']]
IDX = px.index
tnx = raw['^TNX']['Close'].reindex(IDX)

tnx_hi = rolling_on_valid(tnx, lambda x: x.rolling(252).max())
tnx_off = tnx / tnx_hi - 1.0
RATE = (tnx_off >= -0.001).fillna(False)
RATE05 = (tnx_off >= -0.005).fillna(False)
RATE2 = (tnx_off >= -0.02).fillna(False)

dbc = px['DBC']
dbc_hi = rolling_on_valid(dbc, lambda x: x.rolling(252).max())
COMMOD = (dbc / dbc_hi - 1.0 >= -0.001).fillna(False)     # watchlist 41's state

print('panel %d rows %s..%s' % (len(IDX), IDX[0].date(), IDX[-1].date()))
print('live: TNX off 252d high %+.4f%%  RATE=%s   DBC off high %+.3f%%  COMMOD=%s'
      % (100 * tnx_off.iloc[-1], bool(RATE.iloc[-1]),
         100 * (dbc.iloc[-1] / dbc_hi.iloc[-1] - 1), bool(COMMOD.iloc[-1])))
print('RATE days %d | RATE05 %d | RATE2 %d | COMMOD days %d'
      % (RATE.sum(), RATE05.sum(), RATE2.sum(), COMMOD.sum()))

# ---- 0. JACCARD vs watchlist 41 -------------------------------------------
ov = int((RATE & COMMOD).sum())
un = int((RATE | COMMOD).sum())
print('\n0. RE-SKIN TEST vs watchlist 41 (commodity-high anchor):')
print('   |RATE & COMMOD| = %d, |RATE | COMMOD| = %d, JACCARD = %.4f' % (ov, un, ov / un))
print('   P(COMMOD | RATE) = %.3f   P(RATE | COMMOD) = %.3f'
      % (ov / max(RATE.sum(), 1), ov / max(COMMOD.sum(), 1)))

ppi = load_events(['ppi'])['date']
cpi = load_events(['cpi'])['date']


def anchor_days(offset, ev):
    pos, _ = anchor_positions(IDX, ev, offset)
    return pd.DatetimeIndex(sorted(set(IDX[p] for p in pos)))


A0 = anchor_days(-2, ppi)
print('\nPPI anchors (print-2td) inside panel: %d' % len(A0))
for lbl, m in [('RATE(0.1%)', RATE), ('RATE(0.5%)', RATE05), ('RATE(2%)', RATE2)]:
    inter = A0.intersection(IDX[m.values])
    print('  %s & PPI anchor: %d days -> %s' % (lbl, len(inter),
          ', '.join(str(d.date()) for d in inter[-14:])))

LEGS = [('long TLT', [('TLT', 1.0)], 3.0), ('short TLT', [('TLT', -1.0)], 3.0),
        ('long IEF', [('IEF', 1.0)], 3.0), ('short IEF', [('IEF', -1.0)], 3.0)]

# ---- 1. horizon scan on the gated anchor ----------------------------------
print('\n\n################ 1. HORIZON SCAN (RATE gate x PPI anchor) #############')
for gname, gm in [('RATE 0.1%', RATE), ('RATE 0.5%', RATE05), ('RATE 2%', RATE2)]:
    dd = A0.intersection(IDX[gm.values])
    if len(dd) < 3:
        print('\n%s: only %d anchors, skipping scan' % (gname, len(dd)))
        continue
    for name, legs, _c in LEGS[:3:2] + [LEGS[2]]:
        show(horizon_scan(px, dd, legs, hs=(1, 2, 3, 5), lag=1, min_gap=5),
             '%s  %s  (n_anchor_days=%d)' % (gname, name, len(dd)))

# ---- 2. print gate complement: rates state WITHOUT a print in the hold ----
print('\n\n################ 2. PRINT GATE COMPLEMENT ################')
for gname, gm in [('RATE 0.1%', RATE), ('RATE 0.5%', RATE05), ('RATE 2%', RATE2)]:
    for name, legs, _c in LEGS:
        for h in (1, 3, 5):
            ret = vehicle_ret(px, legs, h, 1)
            valid = ret.dropna().index
            st = IDX[gm.values].intersection(valid)
            epi = declusters(st, max(h, 5), valid)
            fl = event_in_window(epi, IDX, h, 1, ('ppi',))
            fl2 = event_in_window(epi, IDX, h, 1, ('ppi', 'cpi'))
            rows = [summarize(ret.loc[epi].values, 'state, all'),
                    summarize(ret.loc[epi[fl]].values, 'PPI in hold'),
                    summarize(ret.loc[epi[~fl]].values, 'no PPI (COMPLEMENT)'),
                    summarize(ret.loc[epi[fl2]].values, 'PPI or CPI in hold'),
                    summarize(ret.loc[epi[~fl2]].values, 'no print (COMPLEMENT)')]
            show(rows, '%s %s h=%d' % (gname, name, h))

# ---- 3. PLACEBO LADDER -----------------------------------------------------
print('\n\n################ 3. PLACEBO LADDER k=-5..+5 ################')


def ladder(legs, gm, gname, h, name):
    ret = vehicle_ret(px, legs, h, 1)
    valid = ret.dropna().index
    rows = []
    for k in range(-5, 6):
        a = anchor_days(-2 + k, ppi).intersection(valid).intersection(IDX[gm.values])
        epi = declusters(a, max(h, 5), valid)
        rows.append(summarize(ret.loc[epi].values,
                              'k=%+d%s' % (k, ' <TRUE>' if k == 0 else '')))
    show(rows, 'LADDER %s %s h=%d' % (gname, name, h))
    v = [r.get('mean_pct', np.nan) for r in rows]
    a = np.asarray([x if x == x else -1e9 for x in v])
    rank = int(np.where(np.argsort(-a) == 5)[0][0]) + 1
    print('  TRUE k=0 mean %.3f%% (n=%s) ranks %d of 11'
          % (v[5], rows[5].get('n'), rank))


for gname, gm in [('RATE 0.5%', RATE05), ('RATE 2%', RATE2)]:
    for name, legs, _c in LEGS:
        for h in (1, 3, 5):
            ladder(legs, gm, gname, h, name)

# ---- 4. GAP SHARE ----------------------------------------------------------
print('\n\n################ 4. GAP SHARE (does the release move it?) ########')
opens = {t: raw[t]['Open'].reindex(IDX) for t in ('TLT', 'IEF')}
pos = pd.Series(range(len(IDX)), index=IDX)
for gname, gm in [('RATE 0.5%', RATE05), ('RATE 2%', RATE2)]:
    for tkr in ('TLT', 'IEF'):
        for h in (1, 3, 5):
            a = anchor_days(-2, ppi).intersection(IDX[gm.values])
            epi = declusters(a, max(h, 5), IDX)
            g, tot = [], []
            for d in epi:
                p = pos[d]
                if p + 1 + h >= len(IDX):
                    continue
                c_entry = px[tkr].iloc[p + 1]
                o_print = opens[tkr].iloc[p + 2] if p + 2 < len(IDX) else np.nan
                c_exit = px[tkr].iloc[p + 1 + h]
                if not np.isfinite(o_print):
                    continue
                g.append(o_print / c_entry - 1.0)
                tot.append(c_exit / c_entry - 1.0)
            if not tot:
                continue
            g, tot = np.asarray(g), np.asarray(tot)
            share = g.sum() / tot.sum() if tot.sum() != 0 else np.nan
            print('%s %s h=%d n=%d: gap mean %+.4f%%, hold mean %+.4f%%, '
                  'GAP SHARE %.1f%% of total; |gap|/|hold| mean %.2f'
                  % (gname, tkr, h, len(g), 100 * g.mean(), 100 * tot.mean(),
                     100 * share, np.mean(np.abs(g)) / max(np.mean(np.abs(tot)), 1e-9)))

# ---- 5. September + pair configuration ------------------------------------
print('\n\n################ 5. SEPTEMBER / PAIR CONFIG ################')
pair_dates = []
pp = set(pd.DatetimeIndex(ppi).date)
cc = set(pd.DatetimeIndex(cpi).date)
for d in pd.DatetimeIndex(ppi):
    nxt = [x for x in pd.DatetimeIndex(cpi) if 0 < (x - d).days <= 3]
    if nxt:
        pair_dates.append(d)
pair_dates = pd.DatetimeIndex(pair_dates)
print('PPI-then-CPI within 3 days: %d prints (live config)' % len(pair_dates))
for name, legs, _c in LEGS:
    for h in (1, 3, 5):
        ret = vehicle_ret(px, legs, h, 1)
        valid = ret.dropna().index
        a_all = anchor_days(-2, ppi).intersection(valid).intersection(IDX[RATE2.values])
        a_pair = anchor_days(-2, pair_dates).intersection(valid).intersection(IDX[RATE2.values])
        e_all = declusters(a_all, max(h, 5), valid)
        e_pair = declusters(a_pair, max(h, 5), valid)
        sep = pd.DatetimeIndex([d for d in e_pair if d.month == 9])
        show([summarize(ret.loc[e_all].values, 'RATE2 x any PPI'),
              summarize(ret.loc[e_pair].values, 'RATE2 x PPI-then-CPI pair'),
              summarize(ret.loc[sep].values, 'RATE2 x pair, SEPTEMBER')],
             '%s h=%d' % (name, h))
