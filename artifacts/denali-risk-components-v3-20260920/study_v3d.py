"""v3d study: the v3c window and basis, with the warning line moved to 55.

Differences from v3c, and only these:
  * the working warning threshold is 55 rather than 50, chosen as the 85th
    percentile of the dial over this window;
  * the band edges move so that no band straddles 55: 0, 20, 40, 55, 65, 80.
Window, dial basis, component histories, the 5/10/21-day windows, the SPY
total-return basis and the averaging conventions are v3c's. Read-only apart from
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

df = pd.DataFrame({'dial': main, 'base': base, 'nyse_net': net,
                   'severity': nyse['nyse_severity'],
                   'nyse_available': nyse['nyse_available'].fillna(False).astype(bool),
                   'components_formed': avail}, index=spy.index)
for h in HS + [63]:
    df[f'fwd{h}'] = spy.shift(-h) / spy - 1

mask = df.components_formed & df.dial.notna()
sample = df[mask].dropna(subset=[f'fwd{h}' for h in HS + [63]])
for n in NAMES:
    sample[f'fire_{n}'] = fires[n].reindex(sample.index).fillna(False).astype(bool)
sample.to_csv(ROOT / 'v3d_observations.csv')

nyse_sample = sample[sample.nyse_available]


def pct(v):
    return f'{v:+.2%}'


def returns(g):
    return [pct(g[f'fwd{h}'].mean()) for h in HS]


components = []
for n in NAMES:
    pool = nyse_sample if n == NAMES[-1] else sample
    g = pool[pool[f'fire_{n}']]
    components.append({'component': n, 'n': len(g), 'pool_n': len(pool),
                       **{f'mean{h}': g[f'fwd{h}'].mean() for h in HS}})
comp_df = pd.DataFrame(components)
comp_df.to_csv(ROOT / 'v3d_component_returns.csv', index=False)

bands = []
for lo, hi in BANDS:
    g = sample[(sample.dial >= lo) & (sample.dial < hi)]
    bands.append({'range': f'{lo}–<{hi}' if np.isfinite(hi) else f'{lo}+', 'n': len(g),
                  **{f'mean{h}': g[f'fwd{h}'].mean() for h in HS}})
band_df = pd.DataFrame(bands)
band_df.to_csv(ROOT / 'v3d_main_dial_ranges.csv', index=False)
thin = [b['range'] for b in bands if b['n'] < MIN_BAND]

low = sample[sample.dial < CUTOFF]
high = sample[sample.dial >= CUTOFF]
on = sample.dial >= CUTOFF
episodes = int((on.astype(int).diff().fillna(on.iloc[0].astype(int)) == 1).sum())
cutoff_pctile = float((sample.dial < CUTOFF).mean() * 100)
p85 = float(sample.dial.quantile(.85))

nyse_baseline = {f'mean{h}': float(nyse_sample[f'fwd{h}'].mean()) for h in HS}
floored = int(sample.nyse_available.sum())

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
    'window_rule': ('from the first session on which all six price components have their full '
                    'trailing windows, to the last session with a complete 63-session outcome'),
    'components_formed_from': starts['_meta']['all_components_formed'],
    'component_starts': {n: starts[n]['ready'] for n in PRICE_NAMES},
    'component_first_fire': {n: starts[n]['first_fire'] for n in PRICE_NAMES},
    'nyse_breadth_first': starts['_meta']['breadth_first_on_spy_calendar'],
    'nyse_breadth_missing': starts['_meta']['breadth_missing_sessions'],
    'nyse_sample_rows': len(nyse_sample),
    'nyse_sample_start': str(nyse_sample.index.min().date()),
    'nyse_baseline': nyse_baseline,
    'dial_floored_sessions': floored,
    'dial_unfloored_sessions': len(sample) - floored,
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
    'below_cutoff': {'n': len(low), **{f'mean{h}': float(low[f'fwd{h}'].mean()) for h in HS}},
    'at_or_above_cutoff': {'n': len(high),
                           **{f'mean{h}': float(high[f'fwd{h}'].mean()) for h in HS}},
    'under20_mean21': float(sample[sample.dial < 20].fwd21.mean()),
    'sources': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in [RUNTIME_FRAG, REPO / 'data/market_breadth.parquet',
                          REPO / 'data/master_prices.parquet',
                          REPO / 'data/signal_horizon_stats.json', RECON]},
}
(ROOT / 'v3d_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')

print('COHORT', manifest['start'], '->', manifest['end'], 'N', len(sample))
print(f'CUTOFF {CUTOFF:.0f} sits at the {cutoff_pctile:.2f}th percentile; the 85th percentile '
      f'of the dial is {p85:.2f}')
print(f'at or above {CUTOFF:.0f}: {len(high)} dates ({on.mean():.2%}), {episodes} episodes')
print('BASELINE', returns(sample))
print('\nCOMPONENTS')
print(comp_df.assign(**{f'mean{h}': lambda d, h=h: (d[f'mean{h}'] * 100).round(3) for h in HS})
      .to_string(index=False))
print('\nBANDS')
print(band_df.assign(**{f'mean{h}': lambda d, h=h: (d[f'mean{h}'] * 100).round(3) for h in HS})
      .to_string(index=False))
print('bands below the', MIN_BAND, 'date floor:', thin or 'none')
print('\nbelow', CUTOFF, 'n', len(low), returns(low), '| at/above n', len(high), returns(high))

prior = ROOT / 'v3c_observations.csv'
if prior.exists():
    base_obs = pd.read_csv(prior, index_col=0, parse_dates=True)
    shared = sample.index.intersection(base_obs.index)
    d = (base_obs.dial.reindex(shared) - sample.dial.reindex(shared))
    print(f'\nvs the v3c sample: shared {len(shared)} of {len(base_obs)} dates; dial max abs diff '
          f'{d.abs().max():.4f} (the sample and the dial are unchanged; only the cutoff moved)')
