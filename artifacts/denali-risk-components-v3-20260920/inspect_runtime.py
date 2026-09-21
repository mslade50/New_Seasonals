"""Compare the dev checkout's dial parquet with the pinned runtime copy."""
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PATHS = {
    'dev': Path(r'C:\Users\McKinley Slade\dev\New_Seasonals\data\rd2_fragility.parquet'),
    'runtime': Path(r'C:\Users\McKinley Slade\dev\New_Seasonals-automation-runtime-v9\data\rd2_fragility.parquet'),
}
for label, path in PATHS.items():
    frag = pd.read_parquet(path)
    md = pq.read_schema(path).metadata or {}
    basis = md.get(b'main_score_basis', b'?').decode()
    print(f'== {label}: rows {len(frag)} last {frag.index.max().date()} basis {basis}')
    if 'main_score' in frag:
        saved = frag[frag['main_score'].notna()]
        print('   main_score rows:', len(saved))
        print(saved[['63d', 'main_score']].round(4).to_string())

mp = Path(r'C:\Users\McKinley Slade\dev\New_Seasonals-automation-runtime-v9\data\master_prices.parquet')
spy = pd.read_parquet(mp, filters=[('ticker', '==', 'SPY')], columns=['ticker', 'date', 'Close'])
print('RUNTIME SPY rows', len(spy), spy['date'].min().date(), '->', spy['date'].max().date())

mp2 = Path(r'C:\Users\McKinley Slade\dev\New_Seasonals\data\master_prices.parquet')
spy2 = pd.read_parquet(mp2, filters=[('ticker', '==', 'SPY')], columns=['ticker', 'date', 'Close'])
print('DEV     SPY rows', len(spy2), spy2['date'].min().date(), '->', spy2['date'].max().date())
