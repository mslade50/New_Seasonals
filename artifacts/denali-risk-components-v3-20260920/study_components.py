"""Component forward returns on the frozen cohort, priced from master_prices.

Re-prices the 17 September cohort's saved fire history so every table in the
v3 report shares one price source and one all-date baseline. Read-only.
"""
from pathlib import Path
import hashlib
import json

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PRIOR = REPO / 'artifacts/denali-risk-components-v2-20260917/component_data'
MASTER = REPO / 'data/master_prices.parquet'
HORIZONS = [5, 10, 21, 42, 63]

cohort = pd.read_parquet(PRIOR / 'audit_daily_cohort.parquet').sort_index()

spy = pd.read_parquet(MASTER, filters=[('ticker', '==', 'SPY')],
                      columns=['ticker', 'date', 'Close'])
spy = spy.set_index('date')['Close'].sort_index()
spy.index = pd.to_datetime(spy.index).tz_localize(None)

fwd = pd.DataFrame(index=spy.index)
for h in HORIZONS:
    fwd[h] = spy.shift(-h) / spy - 1
fwd = fwd.reindex(cohort.index)
assert fwd.notna().all().all(), 'cohort dates must all carry complete outcomes'

rows = []
baseline = {h: fwd[h].mean() for h in HORIZONS}
baseline_neg = {h: float((fwd[h] < 0).mean()) for h in HORIZONS}
for col in [c for c in cohort.columns if c.startswith('signal_')]:
    name = col[len('signal_'):]
    mask = cohort[col].astype(bool)
    sub = fwd[mask.values]
    runs = (mask.astype(int).diff().fillna(mask.iloc[0].astype(int)) == 1).sum()
    for h in HORIZONS:
        rows.append({'component': name, 'horizon': f'{h}d', 'n': int(mask.sum()),
                     'episodes': int(runs),
                     'mean_pct': float(sub[h].mean() * 100),
                     'median_pct': float(sub[h].median() * 100),
                     'negative_pct': float((sub[h] < 0).mean() * 100),
                     'benchmark_n': int(len(fwd)),
                     'benchmark_mean_pct': float(baseline[h] * 100),
                     'benchmark_negative_pct': baseline_neg[h] * 100})
out = pd.DataFrame(rows)
out.to_csv(ROOT / 'component_forward_returns.csv', index=False)

manifest = {
    'cohort_start': str(cohort.index.min().date()),
    'cohort_end': str(cohort.index.max().date()),
    'cohort_sessions': int(len(cohort)),
    'price_source': 'data/master_prices.parquet, SPY adjusted close',
    'return_definition': '100*(SPY adjusted close[t+h]/adjusted close[t]-1), close to close',
    'fire_history_source': str(PRIOR / 'audit_daily_cohort.parquet'),
    'fire_history_sha256': hashlib.sha256((PRIOR / 'audit_daily_cohort.parquet').read_bytes()).hexdigest(),
    'baseline_mean_pct': {f'{h}d': round(baseline[h] * 100, 4) for h in HORIZONS},
    'baseline_negative_pct': {f'{h}d': round(baseline_neg[h] * 100, 4) for h in HORIZONS},
}
(ROOT / 'component_manifest.json').write_text(json.dumps(manifest, indent=2))

print('cohort', manifest['cohort_start'], '->', manifest['cohort_end'], 'N', len(cohort))
print('baseline %', manifest['baseline_mean_pct'])
piv = out.pivot_table(index=['component', 'n', 'episodes'], columns='horizon', values='mean_pct')
print(piv[['5d', '10d', '21d', '42d', '63d']].round(3).to_string())

prior = pd.read_csv(PRIOR / 'component_forward_returns_primary.csv')
prior = prior[prior['sample'] == 'active_days'][['component', 'horizon', 'n', 'mean_pct']]
cmp = out.merge(prior, on=['component', 'horizon'], suffixes=('_new', '_old'), how='inner')
cmp['delta_pp'] = cmp.mean_pct_new - cmp.mean_pct_old
print('\nmax |delta| vs 17 Sep study (pp):', round(cmp.delta_pp.abs().max(), 4))
print('n mismatches:', int((cmp.n_new != cmp.n_old).sum()))
