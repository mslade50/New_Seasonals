"""Reconcile the 17 September introduction study with the production dial."""
from pathlib import Path
import pickle
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))
BASE = REPO / 'artifacts/denali-risk-introduction-20260917'
STUDY = REPO / 'artifacts/net-new-highs-workbook-20260917'
RUNTIME_FRAG = Path(r'C:\Users\McKinley Slade\dev\New_Seasonals-automation-runtime-v9'
                    r'\data\rd2_fragility.parquet')

obs = pd.read_csv(BASE / 'report_observations.csv', index_col=0, parse_dates=True)
print('base cohort:', obs.index.min().date(), '->', obs.index.max().date(), 'n', len(obs))
print('columns:', list(obs.columns))
print(obs[['distance_from_high', 'nyse_net', 'severity', 'base_main63', 'reset_floor',
           'eligible_main63']].describe().round(4).to_string())

print('\nworkbook files:')
for p in sorted(STUDY.rglob('*')):
    if p.is_file():
        print('  ', p.relative_to(STUDY), p.stat().st_size)

pkl = REPO / 'scratch/ultracode_sizing_2026-09-02/dd_pit/pit_signals_extended.pkl'
raw = pickle.load(pkl.open('rb'))
print('\npkl keys:', list(raw.keys()))
fires = raw['fires']
print('fire names:', list(fires.keys()))
for k, v in fires.items():
    print(f'  {k}: {v.index.min().date()} -> {v.index.max().date()} n={len(v)} true={int(v.sum())}')

# Does the runtime parquet's legacy base reproduce the study's base_main63?
frag = pd.read_parquet(RUNTIME_FRAG)
base10 = frag['63d'].dropna().rolling(10, min_periods=1).mean()
base10.index = pd.to_datetime(base10.index)
joined = pd.DataFrame({'study': obs.base_main63, 'runtime': base10.reindex(obs.index)}).dropna()
print('\nbase_main63 vs runtime 10-session mean of 63d: n', len(joined))
print('  max abs diff', round(float((joined.study - joined.runtime).abs().max()), 6))
print('  corr', round(float(joined.study.corr(joined.runtime)), 6))
print(joined.tail(3).round(4).to_string())
