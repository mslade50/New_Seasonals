"""v3b study: the 17 September team-introduction design, recomputed on current code.

Differences from the base study, and only these:
  * the main dial's NYSE layer is the shipped EMA5 trigger (2026-09-18), not the
    one-day print the base was built on;
  * the base composite comes from the production point-in-time parquet in the
    pinned runtime (through 2026-09-18) rather than the workbook reconstruction;
  * SPY adjusted closes come from data/master_prices.parquet, so the cohort
    extends to the newest date with a complete 63-session outcome.
Component fire histories are the same file the base used. Read-only.
"""
from pathlib import Path
import hashlib
import json
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))

RUNTIME_FRAG = Path(r'C:\Users\McKinley Slade\dev\New_Seasonals-automation-runtime-v9'
                    r'\data\rd2_fragility.parquet')
FIRES_PKL = REPO / 'scratch/ultracode_sizing_2026-09-02/dd_pit/pit_signals_extended.pkl'
BASE_OBS = REPO / 'artifacts/denali-risk-introduction-20260917/report_observations.csv'
START = '2019-01-02'
HS = [5, 10, 21]

import nyse_risk
from fragility_core import load_horizon_stats

spy = pd.read_parquet(REPO / 'data/master_prices.parquet',
                      filters=[('ticker', '==', 'SPY')], columns=['ticker', 'date', 'Close'])
spy = spy.set_index('date')['Close'].sort_index()
spy.index = pd.to_datetime(spy.index).tz_localize(None)

breadth = pd.read_parquet(REPO / 'data/market_breadth.parquet')
breadth.index = pd.to_datetime(breadth.index).tz_localize(None)
net = breadth['nyse_net'].reindex(spy.index)

frag = pd.read_parquet(RUNTIME_FRAG)
frag.index = pd.to_datetime(frag.index).tz_localize(None)
base_dial = nyse_risk.main_dial_from_frame(frag).reindex(spy.index)
legacy_base = frag['63d'].dropna().rolling(10, min_periods=1).mean().reindex(spy.index)

nyse = nyse_risk.compute_nyse_main(legacy_base, spy, net, load_horizon_stats())
# Explicit saved point-in-time scores win where production wrote them.
main = nyse[nyse_risk.MAIN_COLUMN].copy()
saved = frag[nyse_risk.MAIN_COLUMN].dropna() if nyse_risk.MAIN_COLUMN in frag else pd.Series(dtype=float)
main.loc[main.index.intersection(saved.index)] = saved.reindex(main.index.intersection(saved.index))

raw = pickle.load(FIRES_PKL.open('rb'))
NAMES = ['Distribution Dominance', 'Defensive Leadership', 'VIX Range Compression',
         'Low Absorption Ratio', 'Seasonal Rank Divergence', 'Dispersion', 'NYSE Net Highs']
fires = {n: raw['fires'][n].reindex(spy.index).fillna(False).astype(bool) for n in NAMES[:-1]}
fires[NAMES[-1]] = nyse['nyse_severity'].fillna(0).gt(0)
fires_available = pd.Series(spy.index.isin(raw['fires'][NAMES[0]].index), index=spy.index)

df = pd.DataFrame({'dial': main, 'base_dial': legacy_base,
                   'distance_from_high': nyse['distance_from_high'],
                   'nyse_net': net, 'severity': nyse['nyse_severity'],
                   'eligible': nyse['nyse_available'].astype(bool)}, index=spy.index)
for h in HS + [63]:
    df[f'fwd{h}'] = spy.shift(-h) / spy - 1

mask = (df.index >= START) & df.eligible & df.nyse_net.notna() & df.dial.notna() & fires_available
sample = df[mask].dropna(subset=[f'fwd{h}' for h in HS + [63]])
for n in NAMES:
    sample[f'fire_{n}'] = fires[n].reindex(sample.index)
sample.to_csv(ROOT / 'v3b_observations.csv')


def pct(v):
    return f'{v:+.2%}'


def returns(g):
    return [pct(g[f'fwd{h}'].mean()) for h in HS]


components = []
for n in NAMES:
    g = sample[sample[f'fire_{n}']]
    components.append({'component': n, 'n': len(g),
                       **{f'mean{h}': g[f'fwd{h}'].mean() for h in HS}})
comp_df = pd.DataFrame(components)
comp_df.to_csv(ROOT / 'v3b_component_returns.csv', index=False)

BANDS = [(0, 20), (20, 40), (40, 50), (50, 60), (60, 80), (80, np.inf)]
bands = []
for lo, hi in BANDS:
    g = sample[(sample.dial >= lo) & (sample.dial < hi)]
    bands.append({'range': f'{lo}–<{hi}' if np.isfinite(hi) else '80+', 'n': len(g),
                  **{f'mean{h}': g[f'fwd{h}'].mean() for h in HS}})
band_df = pd.DataFrame(bands)
band_df.to_csv(ROOT / 'v3b_main_dial_ranges.csv', index=False)

low = sample[sample.dial < 50]
high = sample[sample.dial >= 50]
near = high[high.distance_from_high < .02]
away = high[high.distance_from_high >= .02]

manifest = {
    'rows': len(sample), 'start': str(sample.index.min().date()),
    'end': str(sample.index.max().date()),
    'prices_through': str(spy.index.max().date()),
    'display_windows': HS,
    'dial_basis': ('production point-in-time composite (runtime rd2_fragility.parquet, '
                   'explicit main_score where written, else the ten-session mean of the 63d '
                   'column) with the shipped EMA5 NYSE recovery-reset floor applied'),
    'nyse_model_version': nyse_risk.MODEL_VERSION,
    'cohort_rule': ('from 2019-01-02; NYSE completeness gate satisfied, breadth reading present, '
                    'dial present, component fire history present, complete 63-session outcome'),
    'baseline': {f'mean{h}': float(sample[f'fwd{h}'].mean()) for h in HS},
    'below50': {'n': len(low), **{f'mean{h}': float(low[f'fwd{h}'].mean()) for h in HS}},
    'atleast50': {'n': len(high), **{f'mean{h}': float(high[f'fwd{h}'].mean()) for h in HS}},
    'dual_filter_split': {
        'near': {'n': len(near), **{f'mean{h}': float(near[f'fwd{h}'].mean()) for h in HS}},
        'away': {'n': len(away), **{f'mean{h}': float(away[f'fwd{h}'].mean()) for h in HS}}},
    'under20_mean21': float(sample[sample.dial < 20].fwd21.mean()),
    'sources': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in [RUNTIME_FRAG, REPO / 'data/market_breadth.parquet', FIRES_PKL,
                          REPO / 'data/signal_horizon_stats.json']},
}
(ROOT / 'v3b_manifest.json').write_text(json.dumps(manifest, indent=2))

print('COHORT', manifest['start'], '->', manifest['end'], 'N', len(sample),
      '| prices through', manifest['prices_through'])
print('BASELINE', returns(sample))
print('\nCOMPONENTS')
print(comp_df.assign(**{f'mean{h}': lambda d, h=h: (d[f'mean{h}'] * 100).round(3) for h in HS})
      .to_string(index=False))
print('\nBANDS')
print(band_df.assign(**{f'mean{h}': lambda d, h=h: (d[f'mean{h}'] * 100).round(3) for h in HS})
      .to_string(index=False))
print('\nbelow 50 n', len(low), returns(low), '| 50+ n', len(high), returns(high))
print('dual split: near n', len(near), returns(near), '| away n', len(away), returns(away))

# --- reconciliation against the 17 September base -------------------------
base = pd.read_csv(BASE_OBS, index_col=0, parse_dates=True)
print('\nRECONCILIATION vs the 17 September introduction study')
print('  base cohort', base.index.min().date(), '->', base.index.max().date(), 'n', len(base))
shared = sample.index.intersection(base.index)
print('  shared dates', len(shared), '| in base not v3b', len(base.index.difference(sample.index)),
      '| in v3b not base', len(sample.index.difference(base.index)))
cmp = pd.DataFrame({'base_floor': base.reset_floor.reindex(shared),
                    'v3b_dial': sample.dial.reindex(shared),
                    'base_base': base.base_main63.reindex(shared),
                    'v3b_base': sample.base_dial.reindex(shared)})
print('  dial max abs diff', round(float((cmp.base_floor - cmp.v3b_dial).abs().max()), 4),
      '| mean abs', round(float((cmp.base_floor - cmp.v3b_dial).abs().mean()), 4))
print('  base-composite max abs diff', round(float((cmp.base_base - cmp.v3b_base).abs().max()), 4))
worst = (cmp.base_base - cmp.v3b_base).abs().sort_values(ascending=False).head(5)
print('  largest base-composite gaps:')
print(worst.round(3).to_string())
print('  NYSE fire days: base', int(base.severity.gt(0).sum()),
      '-> v3b', int(sample[f'fire_{NAMES[-1]}'].sum()))
