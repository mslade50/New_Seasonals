"""What bounds the v3b window, and how far back each input actually reaches.

Read-only. Prints the first and last usable date of every series the v3b study
consumed, plus the research-basis full recompute the repo keeps for exactly this
kind of long-window question.
"""
from pathlib import Path
import pickle
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))

RUNTIME_FRAG = Path(r'C:\Users\McKinley Slade\dev\New_Seasonals-automation-runtime-v9'
                    r'\data\rd2_fragility.parquet')
FIRES_PKL = REPO / 'scratch/ultracode_sizing_2026-09-02/dd_pit/pit_signals_extended.pkl'


def span(name: str, s: pd.Series) -> None:
    s = s.dropna()
    if s.empty:
        print(f'{name:38s} EMPTY')
        return
    print(f'{name:38s} {s.index.min().date()} -> {s.index.max().date()}  n={len(s)}')


spy = pd.read_parquet(REPO / 'data/master_prices.parquet',
                      filters=[('ticker', '==', 'SPY')], columns=['ticker', 'date', 'Close'])
spy = spy.set_index('date')['Close'].sort_index()
spy.index = pd.to_datetime(spy.index).tz_localize(None)
span('master_prices SPY close', spy)

breadth = pd.read_parquet(REPO / 'data/market_breadth.parquet')
breadth.index = pd.to_datetime(breadth.index).tz_localize(None)
print('breadth columns:', list(breadth.columns))
for col in breadth.columns:
    if breadth[col].dtype.kind in 'fiu':
        span(f'  breadth {col}', breadth[col])

for label, path in [('runtime rd2_fragility', RUNTIME_FRAG),
                    ('repo rd2_fragility', REPO / 'data/rd2_fragility.parquet'),
                    ('repo rd2_fragility_ts', REPO / 'data/rd2_fragility_ts.parquet'),
                    ('repo fragility_63d_history', REPO / 'data/fragility_63d_history.parquet')]:
    if not Path(path).exists():
        print(f'{label:38s} MISSING {path}')
        continue
    frame = pd.read_parquet(path)
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    print(f'-- {label}: columns {list(frame.columns)}')
    for col in frame.columns:
        if frame[col].dtype.kind in 'fiu':
            span(f'  {label} {col}', frame[col])

raw = pickle.load(FIRES_PKL.open('rb'))
print('-- pit_signals_extended keys:', list(raw.keys()))
for name, series in raw['fires'].items():
    s = pd.Series(series)
    s.index = pd.to_datetime(s.index).tz_localize(None)
    print(f'  fires {name:28s} {s.index.min().date()} -> {s.index.max().date()} '
          f'n={len(s)} fired={int(s.astype(bool).sum())}')
for key, value in raw.items():
    if key != 'fires':
        print(f'  meta {key}: {str(value)[:400]}')
