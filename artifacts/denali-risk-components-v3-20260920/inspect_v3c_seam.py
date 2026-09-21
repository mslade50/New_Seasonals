"""Where the reconstruction and the production record disagree, year by year.

The v3c window splices the two, so the size and the location of the gap decides
what the closing note has to say. Read-only.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
rec = pd.read_parquet(ROOT / 'v3c_reconstruction.parquet')
both = rec[['base_recon', 'base_prod']].dropna()
diff = both.base_prod - both.base_recon

print('overlap', both.index.min().date(), '->', both.index.max().date(), 'n', len(both))
print('\nyear   n   prod_mean recon_mean  mean_abs  max_abs  side50_agree')
for year, g in both.groupby(both.index.year):
    d = (g.base_prod - g.base_recon)
    agree = ((g.base_prod >= 50) == (g.base_recon >= 50)).mean()
    print(f'{year}  {len(g):4d}   {g.base_prod.mean():7.2f}  {g.base_recon.mean():8.2f}  '
          f'{d.abs().mean():8.2f} {d.abs().max():8.2f}  {agree:11.1%}')

print('\nlargest ten gaps:')
print(both.assign(diff=diff).reindex(diff.abs().sort_values(ascending=False).index[:10])
      .round(2).to_string())

zero = (diff.abs() < .01).mean()
print(f'\nexactly equal to two decimals on {zero:.1%} of overlap sessions')
print('quantiles of |gap|:', diff.abs().quantile([.5, .75, .9, .95, .99]).round(2).to_dict())
