"""v3e study: v3d, plus the same tables restricted to dates near the index high.

Most components only fire when SPY is close to its trailing high, so the dial is
built to warn while the index still looks healthy. This adds:
  * the band table and the 55 split over dates whose close is less than 2% below
    the trailing 252-session closing high;
  * the share of those dates followed by a 21-day loss;
  * the same split over the dates 2% or more below the high, for contrast.
Window, dial basis, component histories, band edges, display windows, the SPY
total-return basis and the averaging conventions are v3d's. Read-only apart from
the CSV/JSON artifacts it writes here.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))

RUNTIME_FRAG = Path(r'C:\Users\McKinley Slade\dev\New_Seasonals-automation-runtime-v9'
                    r'\data\rd2_fragility.parquet')
RECON = ROOT / 'v3c_reconstruction.parquet'
STARTS = ROOT / 'v3c_component_starts.json'
HS = [5, 10, 21]
WARMUP_SESSIONS = 525
CUTOFF = 55.0
NEAR_HIGH = 0.02
BANDS = [(0, 20), (20, 40), (40, 55), (55, 65), (65, 80), (80, np.inf)]
MIN_BAND = 100

import nyse_risk
from fragility_core import load_horizon_stats

spy = pd.read_parquet(REPO / 'data/master_prices.parquet',
                      filters=[('ticker', '==', 'SPY')], columns=['ticker', 'date', 'Close'])
spy = spy.set_index('date')['Close'].sort_index()
spy.index = pd.to_datetime(spy.index).tz_localize(None)

rec = pd.read_parquet(RECON)
rec.index = pd.to_datetime(rec.index).tz_localize(None)
starts = json.loads(STARTS.read_text(encoding='utf-8'))
NAMES = ['Distribution Dominance', 'Defensive Leadership', 'VIX Range Compression',
         'Low Absorption Ratio', 'Seasonal Rank Divergence', 'Dispersion', 'NYSE Net Highs']
PRICE_NAMES = NAMES[:-1]

frag = pd.read_parquet(RUNTIME_FRAG)
frag.index = pd.to_datetime(frag.index).tz_localize(None)
stored = frag['63d'].dropna().rolling(10, min_periods=1).mean()

stored_first = stored.index.min()
pos = spy.index.get_indexer([stored_first])[0]
stored_usable_from = spy.index[pos + WARMUP_SESSIONS]
base = rec['base_recon'].reindex(spy.index)
use_stored = stored.reindex(spy.index).notna() & (spy.index >= stored_usable_from)
base = base.where(~use_stored, stored.reindex(spy.index))

net = rec['nyse_net'].reindex(spy.index)
nyse = nyse_risk.compute_nyse_main(base, spy, net, load_horizon_stats())
main = nyse[nyse_risk.MAIN_COLUMN].copy()
saved = frag[nyse_risk.MAIN_COLUMN].dropna() if nyse_risk.MAIN_COLUMN in frag else pd.Series(dtype=float)
main.loc[main.index.intersection(saved.index)] = saved.reindex(main.index.intersection(saved.index))

fires = {n: rec[f'fire_{n}'].reindex(spy.index).fillna(False).astype(bool) for n in PRICE_NAMES}
fires[NAMES[-1]] = nyse['nyse_severity'].fillna(0).gt(0)
avail = pd.concat([rec[f'avail_{n}'].reindex(spy.index).fillna(False).astype(bool)
                   for n in PRICE_NAMES], axis=1).all(axis=1)

# Distance below the trailing 252-session closing high, on the same adjusted series.
distance = 1 - spy / spy.rolling(252, min_periods=252).max()

df = pd.DataFrame({'dial': main, 'base': base, 'nyse_net': net,
                   'distance_from_high': distance,
                   'severity': nyse['nyse_severity'],
                   'nyse_available': nyse['nyse_available'].fillna(False).astype(bool),
                   'components_formed': avail}, index=spy.index)
for h in HS + [63]:
    df[f'fwd{h}'] = spy.shift(-h) / spy - 1

mask = df.components_formed & df.dial.notna()
sample = df[mask].dropna(subset=[f'fwd{h}' for h in HS + [63]])
for n in NAMES:
    sample[f'fire_{n}'] = fires[n].reindex(sample.index).fillna(False).astype(bool)
sample.to_csv(ROOT / 'v3e_observations.csv')

nyse_sample = sample[sample.nyse_available]
near = sample[sample.distance_from_high < NEAR_HIGH]
off = sample[sample.distance_from_high >= NEAR_HIGH]


def pct(v):
    return f'{v:+.2%}'


def returns(g):
    return [pct(g[f'fwd{h}'].mean()) for h in HS]


def stats(g):
    return {'n': len(g), **{f'mean{h}': float(g[f'fwd{h}'].mean()) for h in HS}}


def band_table(pool):
    rows = []
    for lo, hi in BANDS:
        g = pool[(pool.dial >= lo) & (pool.dial < hi)]
        rows.append({'range': f'{lo}–<{hi}' if np.isfinite(hi) else f'{lo}+', 'n': len(g),
                     **{f'mean{h}': g[f'fwd{h}'].mean() for h in HS}})
    return pd.DataFrame(rows)


components = []
for n in NAMES:
    pool = nyse_sample if n == NAMES[-1] else sample
    g = pool[pool[f'fire_{n}']]
    components.append({'component': n, 'n': len(g), 'pool_n': len(pool),
                       **{f'mean{h}': g[f'fwd{h}'].mean() for h in HS}})
    if n != NAMES[-1]:
        components[-1]['near_high_share'] = float((g.distance_from_high < NEAR_HIGH).mean()) if len(g) else float('nan')
comp_df = pd.DataFrame(components)
comp_df.to_csv(ROOT / 'v3e_component_returns.csv', index=False)

band_df = band_table(sample)
band_df.to_csv(ROOT / 'v3e_main_dial_ranges.csv', index=False)
near_band_df = band_table(near)
near_band_df.to_csv(ROOT / 'v3e_near_high_ranges.csv', index=False)
thin = [b for b, n in zip(near_band_df['range'], near_band_df['n']) if n < MIN_BAND]

on = sample.dial >= CUTOFF
episodes = int((on.astype(int).diff().fillna(on.iloc[0].astype(int)) == 1).sum())
cutoff_pctile = float((sample.dial < CUTOFF).mean() * 100)
p85 = float(sample.dial.quantile(.85))


def loss_share(g):
    return float((g.fwd21 < 0).mean()) if len(g) else float('nan')


manifest = {
    'rows': len(sample), 'start': str(sample.index.min().date()),
    'end': str(sample.index.max().date()),
    'prices_through': str(spy.index.max().date()),
    'display_windows': HS,
    'cutoff': CUTOFF,
    'cutoff_percentile': round(cutoff_pctile, 2),
    'dial_85th_percentile': round(p85, 2),
    'share_at_or_above_cutoff': round(float(on.mean() * 100), 2),
    'episodes_at_or_above_cutoff': episodes,
    'band_edges': [b[0] for b in BANDS],
    'bands_under_min': thin,
    'near_high_rule': ('close less than 2% below the trailing 252-session closing high, '
                       'rolling max with 252 minimum periods on the same adjusted SPY series'),
    'near_high': {
        'n': len(near),
        'share_of_sample': round(float(len(near) / len(sample) * 100), 2),
        'baseline': stats(near),
        'below_cutoff': stats(near[near.dial < CUTOFF]),
        'at_or_above_cutoff': stats(near[near.dial >= CUTOFF]),
        'loss21_share_below': round(loss_share(near[near.dial < CUTOFF]), 4),
        'loss21_share_at_or_above': round(loss_share(near[near.dial >= CUTOFF]), 4),
        'loss21_share_80plus': round(loss_share(near[near.dial >= 80]), 4),
        'loss21_share_all': round(loss_share(near), 4),
    },
    'off_high': {
        'n': len(off),
        'baseline': stats(off),
        'below_cutoff': stats(off[off.dial < CUTOFF]),
        'at_or_above_cutoff': stats(off[off.dial >= CUTOFF]),
    },
    'window_rule': ('from the first session on which all six price components have their full '
                    'trailing windows, to the last session with a complete 63-session outcome'),
    'components_formed_from': starts['_meta']['all_components_formed'],
    'component_starts': {n: starts[n]['ready'] for n in PRICE_NAMES},
    'component_near_high_share': {r['component']: round(r['near_high_share'], 4)
                                  for r in components if 'near_high_share' in r},
    'nyse_breadth_first': starts['_meta']['breadth_first_on_spy_calendar'],
    'nyse_breadth_missing': starts['_meta']['breadth_missing_sessions'],
    'nyse_sample_rows': len(nyse_sample),
    'nyse_sample_start': str(nyse_sample.index.min().date()),
    'nyse_baseline': {f'mean{h}': float(nyse_sample[f'fwd{h}'].mean()) for h in HS},
    'dial_floored_sessions': int(sample.nyse_available.sum()),
    'dial_unfloored_sessions': int((~sample.nyse_available).sum()),
    'dial_basis': ('the current rules applied to the full price cache, with the stored '
                   'point-in-time record taking precedence from '
                   f'{stored_usable_from.date()} (the stored file begins '
                   f'{stored_first.date()}; its first {WARMUP_SESSIONS} sessions are its own '
                   'warm-up), and the shipped EMA5 NYSE recovery-reset floor applied on top'),
    'stored_record_first': str(stored_first.date()),
    'stored_usable_from': str(stored_usable_from.date()),
    'warmup_sessions': WARMUP_SESSIONS,
    'nyse_model_version': nyse_risk.MODEL_VERSION,
    'baseline': {f'mean{h}': float(sample[f'fwd{h}'].mean()) for h in HS},
    'below_cutoff': stats(sample[sample.dial < CUTOFF]),
    'at_or_above_cutoff': stats(sample[sample.dial >= CUTOFF]),
    'under20_mean21': float(sample[sample.dial < 20].fwd21.mean()),
    'sources': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in [RUNTIME_FRAG, REPO / 'data/market_breadth.parquet',
                          REPO / 'data/master_prices.parquet',
                          REPO / 'data/signal_horizon_stats.json', RECON]},
}
(ROOT / 'v3e_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

print('COHORT', manifest['start'], '->', manifest['end'], 'N', len(sample))
print('near-high dates', len(near), f"({manifest['near_high']['share_of_sample']}%)",
      '| off-high dates', len(off))
print('\nFULL BANDS')
print(band_df.assign(**{f'mean{h}': lambda d, h=h: (d[f'mean{h}'] * 100).round(3) for h in HS})
      .to_string(index=False))
print('\nNEAR-HIGH BANDS')
print(near_band_df.assign(**{f'mean{h}': lambda d, h=h: (d[f'mean{h}'] * 100).round(3) for h in HS})
      .to_string(index=False))
print('bands below the', MIN_BAND, 'date floor:', thin or 'none')
print('near-high all dates', returns(near))
print('near-high below', CUTOFF, returns(near[near.dial < CUTOFF]),
      'n', int(manifest['near_high']['below_cutoff']['n']))
print('near-high at/above', CUTOFF, returns(near[near.dial >= CUTOFF]),
      'n', int(manifest['near_high']['at_or_above_cutoff']['n']))
print('21d loss share: below', manifest['near_high']['loss21_share_below'],
      '| at/above', manifest['near_high']['loss21_share_at_or_above'],
      '| 80+', manifest['near_high']['loss21_share_80plus'])
print('\nOFF-HIGH at/above', CUTOFF, returns(off[off.dial >= CUTOFF]),
      'n', int(manifest['off_high']['at_or_above_cutoff']['n']),
      '| off-high all', returns(off), 'n', len(off))
print('\ncomponent share of warnings landing near the high:')
for k, v in manifest['component_near_high_share'].items():
    print(f'  {k:30s} {v:.1%}')
