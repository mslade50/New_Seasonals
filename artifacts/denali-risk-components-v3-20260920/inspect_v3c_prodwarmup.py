"""When does the stored record stop being its own start-up?

The stored parquet begins 2016-07-05. Components whose definitions need years of
trailing history cannot fire until that much history has accumulated inside the
stored window, so the first stretch of the record is a warm-up rather than a
reading. This prints the monthly picture and the first month the stored series
tracks the full-history reconstruction. Read-only.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
rec = pd.read_parquet(ROOT / 'v3c_reconstruction.parquet')
both = rec[['base_recon', 'base_prod']].dropna()
both = both[both.index < '2019-07-01']

print('month    n  prod_mean prod_zero%  recon_mean  mean_abs  max_abs')
for month, g in both.groupby(both.index.to_period('M')):
    d = (g.base_prod - g.base_recon).abs()
    print(f'{month}  {len(g):3d}  {g.base_prod.mean():8.2f}  {(g.base_prod < .01).mean():8.1%}  '
          f'{g.base_recon.mean():9.2f}  {d.mean():8.2f} {d.max():8.2f}')

full = pd.read_parquet(ROOT / 'v3c_reconstruction.parquet')[['base_recon', 'base_prod']].dropna()
gap = (full.base_prod - full.base_recon).abs()
roll = gap.rolling(63).mean()
settled = roll[roll < 3.0]
print('\nfirst session where the trailing 63-session mean gap stays under 3 points:',
      settled.index.min().date() if len(settled) else 'never')
for cut in [2.0, 3.0, 5.0]:
    ok = roll[roll < cut]
    if len(ok):
        # first date after which it never again exceeds the cut for a full year
        print(f'  gap<{cut}: first {ok.index.min().date()}, last breach '
              f'{roll[roll >= cut].index.max().date() if (roll >= cut).any() else "none"}')
