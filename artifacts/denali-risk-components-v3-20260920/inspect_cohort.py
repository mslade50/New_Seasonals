"""Check whether the frozen component cohort carries reusable fire flags."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parents[1] / 'artifacts/denali-risk-components-v2-20260917/component_data'

cohort = pd.read_parquet(PRIOR / 'audit_daily_cohort.parquet')
print('cohort columns:', list(cohort.columns))
print('index', cohort.index.min(), '->', cohort.index.max(), 'rows', len(cohort))
print(cohort.head(3).to_string())
print(cohort.tail(3).to_string())
print('\nmanifest:')
print((PRIOR / 'manifest.json').read_text()[:2500])
