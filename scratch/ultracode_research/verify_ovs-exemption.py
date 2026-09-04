"""Adversarial recompute of the ovs-exemption findings. Fresh code, vectorized bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(20260702)
NBOOT = 10_000

# ---------- join ----------
tr = pd.read_parquet('data/backtest_trades_full.parquet')
ovs = tr[tr['Strategy'].str.contains('Overbot Vol', na=False)].copy()
ovs['Signal Date'] = pd.to_datetime(ovs['Signal Date'])

fr = pd.read_parquet('data/rd2_fragility.parquet')
live = fr['63d'].rolling(10, min_periods=1).mean().rename('score').reset_index()
live.columns = ['Date', 'score']

ovs = ovs.sort_values('Signal Date')
joined = pd.merge_asof(ovs, live.sort_values('Date'),
                       left_on='Signal Date', right_on='Date',
                       direction='backward', tolerance=pd.Timedelta(days=5))
j = joined.dropna(subset=['score']).copy()
print('=== JOIN ===')
print('joined N:', len(j), '| window:', j['Signal Date'].min().date(),
      '->', j['Signal Date'].max().date())

def path_of(m: float) -> str:
    if pd.isna(m):
        return 'NA'
    return 'P1' if m >= 0.7 else 'P2'

j['path'] = j['Size_Mult'].map(path_of)
j['year'] = j['Signal Date'].dt.year
j['midterm'] = (j['year'] % 4 == 2)
print('P1:', (j['path'] == 'P1').sum(), 'P2:', (j['path'] == 'P2').sum(),
      'NA:', (j['path'] == 'NA').sum(), '| midterm:', int(j['midterm'].sum()))

g = j.dropna(subset=['T+1 Open', 'Signal Close', 'ATR']).copy()
g = g[g['path'] != 'NA']
gap_p1 = (g['T+1 Open'] - g['Signal Close']) / g['ATR'] > 0.25
agree = (gap_p1 == (g['path'] == 'P1')).mean()
print(f'Size_Mult vs gap agreement: {agree:.4f} on N={len(g)}')
ovs['path'] = ovs['Size_Mult'].map(path_of)
print('full-history P1/P2/NA:', ovs['path'].value_counts().to_dict())

# ---------- bands ----------
edges = [0, 3, 21, 44, 55, 100.0001]
labels = ['0-3', '3-21', '21-44', '44-55', '55+']
j['band'] = pd.cut(j['score'], bins=edges, labels=labels, right=False)

print('\n=== BAND TABLE ===')
bt = j.groupby('band', observed=True)['R_Multiple'].agg(
    N='size', avgR='mean', medR='median', totR='sum',
    win=lambda s: (s > 0).mean() * 100)
print(bt.round(3))

print('\nby path x band (avgR / N):')
print(j.groupby(['path', 'band'], observed=True)['R_Multiple']
       .agg(['mean', 'size']).round(3))

# ---------- tests ----------
j['ym'] = j['Signal Date'].dt.to_period('M')

def ew_monthly_t(a, b):
    ma = a.groupby('ym')['R_Multiple'].mean()
    mb = b.groupby('ym')['R_Multiple'].mean()
    t, p = stats.ttest_ind(ma, mb, equal_var=False)
    return ma.mean(), len(ma), mb.mean(), len(mb), t, p

def block_boot(a: pd.DataFrame, b: pd.DataFrame, nboot=NBOOT, rng=rng):
    """Trade-weighted monthly block bootstrap of mean(a)-mean(b), vectorized."""
    months = pd.Index(sorted(set(a['ym']) | set(b['ym'])))
    idx = {m: i for i, m in enumerate(months)}
    M = len(months)
    sa = np.zeros(M); na = np.zeros(M); sb = np.zeros(M); nb = np.zeros(M)
    for m, grp in a.groupby('ym'):
        sa[idx[m]] = grp['R_Multiple'].sum(); na[idx[m]] = len(grp)
    for m, grp in b.groupby('ym'):
        sb[idx[m]] = grp['R_Multiple'].sum(); nb[idx[m]] = len(grp)
    obs = a['R_Multiple'].mean() - b['R_Multiple'].mean()
    draws = rng.integers(0, M, size=(nboot, M))
    SA = sa[draws].sum(axis=1); NA_ = na[draws].sum(axis=1)
    SB = sb[draws].sum(axis=1); NB_ = nb[draws].sum(axis=1)
    ok = (NA_ > 0) & (NB_ > 0)
    diffs = SA[ok] / NA_[ok] - SB[ok] / NB_[ok]
    se = diffs.std()
    z = obs / se
    p_emp = 2 * min((diffs >= 0).mean(), (diffs <= 0).mean())
    return obs, z, p_emp, ok.sum()

mid = j[j['band'] == '21-44']
below21 = j[j['score'] < 21]
calm = j[j['band'] == '0-3']

print('\n=== EQUAL-WEIGHT MONTHLY t (21-44 vs <21) ===')
ma, na_, mb, nb_, t, p = ew_monthly_t(mid, below21)
print(f'mid monthly-mean {ma:+.3f} ({na_} mo) vs <21 {mb:+.3f} ({nb_} mo): '
      f't={t:.2f} p={p:.3f}')

print('\n=== TRADE-WEIGHTED BLOCK BOOTSTRAP ===')
obs, z, pe, nb = block_boot(mid, below21)
print(f'21-44 vs <21: diff {obs:+.3f}R  z={z:.2f}  p_emp={pe:.4f}')
obs3, z3, pe3, _ = block_boot(mid, calm)
print(f'21-44 vs 0-3: diff {obs3:+.3f}R  z={z3:.2f}  p_emp={pe3:.4f}')

print('\nLOYO (drop signal-year, 21-44 vs <21):')
zs_all = {}
for y in sorted(j['year'].unique()):
    a = mid[mid['year'] != y]
    b = below21[below21['year'] != y]
    if len(a) < 10 or len(b) < 10:
        continue
    o, zz, _, _ = block_boot(a, b)
    zs_all[y] = zz
    print(f'  drop {y}: diff {o:+.3f} z={zz:.2f} (Nmid={len(a)})')
print(f'LOYO worst z (closest to 0): {max(zs_all.values()):.2f}')

nm_mid = mid[~mid['midterm']]; nm_b = below21[~below21['midterm']]
o, znm, _, _ = block_boot(nm_mid, nm_b)
print(f'non-midterm only: diff {o:+.3f} z={znm:.2f} (Nmid={len(nm_mid)})')
d56 = j[~j['year'].isin([2025, 2026])]
o, z56, _, _ = block_boot(d56[d56['band'] == '21-44'], d56[d56['score'] < 21])
print(f'drop 2025+2026: diff {o:+.3f} z={z56:.2f}')

print('\n=== EDGE SENSITIVITY ===')
for lo, hi in [(15, 40), (18, 42), (21, 44), (25, 44), (21, 50), (25, 50), (30, 50)]:
    a = j[(j['score'] >= lo) & (j['score'] < hi)]
    b = j[j['score'] < lo]
    o, zz, _, _ = block_boot(a, b)
    print(f'  [{lo},{hi}): avgR {a["R_Multiple"].mean():+.3f} (N={len(a)}) '
          f'diff {o:+.3f} z={zz:.2f}')
sliver = j[(j['score'] >= 21) & (j['score'] < 25)]
print(f'  21-25 sliver: N={len(sliver)} avgR {sliver["R_Multiple"].mean():+.3f}')

print('\n=== P1 / P2 ===')
p1 = j[j['path'] == 'P1']; p2 = j[j['path'] == 'P2']
p1mid, p1calm = p1[p1['band'] == '21-44'], p1[p1['band'] == '0-3']
o, zp1, _, _ = block_boot(p1mid, p1calm)
print(f'P1 mid {p1mid["R_Multiple"].mean():+.3f} (N={len(p1mid)}) vs calm '
      f'{p1calm["R_Multiple"].mean():+.3f} (N={len(p1calm)}): diff {o:+.3f} z={zp1:.2f}')
zs = []
for y in sorted(j['year'].unique()):
    a = p1mid[p1mid['year'] != y]; b = p1calm[p1calm['year'] != y]
    if len(a) < 10 or len(b) < 10:
        continue
    _, zz, _, _ = block_boot(a, b, nboot=5000)
    zs.append(zz)
print(f'P1 LOYO z range: {min(zs):.2f} .. {max(zs):.2f}')
p2mid, p2calm = p2[p2['band'] == '21-44'], p2[p2['band'] == '0-3']
o2, zp2, _, _ = block_boot(p2mid, p2calm)
print(f'P2 mid {p2mid["R_Multiple"].mean():+.3f} (N={len(p2mid)}) vs calm '
      f'{p2calm["R_Multiple"].mean():+.3f} (N={len(p2calm)}): diff {o2:+.3f} z={zp2:.2f}')

print('\n=== MIDTERM CONFOUND ===')
print(f'mid-band midterm share: {mid["midterm"].mean()*100:.1f}% '
      f'vs base {j["midterm"].mean()*100:.1f}%')
mt_mid = mid[mid['midterm']]
print(f'midterm mid-band avgR {mt_mid["R_Multiple"].mean():+.3f} (N={len(mt_mid)}); '
      f'non-midterm mid-band {nm_mid["R_Multiple"].mean():+.3f} (N={len(nm_mid)})')

print('\n=== 55+ BY YEAR ===')
hi = j[j['band'] == '55+']
print(hi.groupby('year')['R_Multiple'].agg(['mean', 'size']).round(3))
ex22 = hi[hi['year'] != 2022]
print(f'ex-2022: avgR {ex22["R_Multiple"].mean():+.3f} (N={len(ex22)})')
o, zh, _, _ = block_boot(ex22, mid)
print(f'55+ ex-2022 vs 21-44: diff {o:+.3f} z={zh:.2f}')
o, zha, _, _ = block_boot(hi, mid)
print(f'55+ all vs 21-44: diff {o:+.3f} z={zha:.2f}')

print('\n=== THROTTLE REPLAY (bookkeeping) ===')
def dd_of(r: np.ndarray) -> float:
    c = np.cumsum(r)
    return float((c - np.maximum.accumulate(c)).min())

jj = j.sort_values('Exit Date')
base_r = jj['R_Multiple'].to_numpy()
mult = np.where(jj['band'] == '21-44', 0.5, 1.0)
thr_r = base_r * mult
print(f'totR: base {base_r.sum():+.1f} -> 0.5x mid {thr_r.sum():+.1f} '
      f'(delta {thr_r.sum()-base_r.sum():+.1f})')
print(f'avgR/unit-risk: base {base_r.sum()/len(jj):.3f} -> {thr_r.sum()/mult.sum():.3f}')
print(f'worst R-DD (exit order): base {dd_of(base_r):.1f} -> {dd_of(thr_r):.1f}')
m075 = np.where(jj['band'] == '21-44', 0.75, 1.0)
print(f'0.75x cost: {(base_r*m075).sum()-base_r.sum():+.1f}R')
print(f'mid-band totR: {mid["R_Multiple"].sum():+.1f} over {len(mid)} trades '
      f'({len(mid)/len(j)*100:.0f}% of OVS)')
