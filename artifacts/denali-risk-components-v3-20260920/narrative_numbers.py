"""Pull the exact figures quoted in the v3 report prose."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
bands = pd.read_csv(ROOT / 'dial_ranges.csv')
th = pd.read_csv(ROOT / 'thresholds.csv')

d = bands[(bands.subset == 'all') & (bands.bins == 'display')]
print('display bands, near-high share + 63-spaced starts')
print(d[['lo', 'hi', 'days', 'starts63', 'near_high_share', 'spaced_mean63', 'spaced_n63']].round(4).to_string(index=False))

print('\nthresholds by era at 40 / 60 / 50')
for subset in ['all', '2018_2021', '2022_2026', 'excluding_2020', 'near_high']:
    sub = th[(th.subset == subset) & (th.threshold.isin([40, 50, 60]))]
    print(subset)
    print(sub[['threshold', 'side', 'days', 'mean5', 'mean21', 'mean63',
               'negative63', 'spaced_mean63', 'spaced_n63']].round(4).to_string(index=False))

print('\n40+ spaced-observation sensitivity (all)')
row = th[(th.subset == 'all') & (th.threshold == 40) & (th.side == 'at_or_above')].iloc[0]
print({k: round(row[k], 4) for k in ['days', 'mean63', 'spaced_mean63', 'spaced_n63',
                                     'spaced_mean21', 'spaced_n21', 'negative63', 'near_high_share']})
print('\n80+ detail')
row = th[(th.subset == 'all') & (th.threshold == 80) & (th.side == 'at_or_above')].iloc[0]
print({k: round(row[k], 4) for k in ['days', 'mean5', 'mean21', 'mean63', 'negative63',
                                     'spaced_n63', 'spaced_mean63', 'near_high_share']})
