"""Read-only inspection of the v3 inputs. No production file is touched."""
from pathlib import Path
import sys

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))

frag = pd.read_parquet(REPO / 'data/rd2_fragility.parquet')
print('FRAG columns:', list(frag.columns))
print('FRAG index:', frag.index.min(), '->', frag.index.max(), 'rows', len(frag))
md = pq.read_schema(REPO / 'data/rd2_fragility.parquet').metadata or {}
for k, v in md.items():
    try:
        ks = k.decode()
    except Exception:
        ks = str(k)
    if ks.startswith('pandas'):
        continue
    print('  META', ks, '=', v.decode()[:160])
print('main_score non-null:', frag['main_score'].notna().sum() if 'main_score' in frag else 'ABSENT')
if 'main_score' in frag:
    print(frag[frag['main_score'].notna()][['63d', 'main_score']].tail(10))

from fragility_core import load_main_dial_series
main = load_main_dial_series()
print('MAIN dial:', main.index.min(), '->', main.index.max(), 'n', len(main),
      'last', round(float(main.iloc[-1]), 3))
print(main.tail(8).round(3).to_string())

mp = pd.read_parquet(REPO / 'data/master_prices.parquet')
print('MASTER cols:', list(mp.columns)[:12], 'rows', len(mp))
print(mp.head(2))

br = pd.read_parquet(REPO / 'data/market_breadth.parquet')
print('BREADTH cols:', list(br.columns), 'rows', len(br))
print(br.tail(5))
