"""Why does the saved 2026-09-18 main score equal the unfloored base?

Hypothesis: the post-close run scores a session before that session's breadth
has been collected (the collector runs the next morning), so severity is
unknown for the newest row and the completeness gate falls back to the base
dial. Read-only reconstruction; nothing is written to production.
"""
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))

RUNTIME_FRAG = Path(r'C:\Users\McKinley Slade\dev\New_Seasonals-automation-runtime-v9'
                    r'\data\rd2_fragility.parquet')

import nyse_risk
from fragility_core import load_horizon_stats

spy = pd.read_parquet(REPO / 'data/master_prices.parquet',
                      filters=[('ticker', '==', 'SPY')], columns=['ticker', 'date', 'Close'])
spy = spy.set_index('date')['Close'].sort_index()
spy.index = pd.to_datetime(spy.index).tz_localize(None)

breadth = pd.read_parquet(REPO / 'data/market_breadth.parquet')
breadth.index = pd.to_datetime(breadth.index).tz_localize(None)

frag = pd.read_parquet(RUNTIME_FRAG)
base = frag['63d'].dropna().rolling(10, min_periods=1).mean().reindex(spy.index)
stats = load_horizon_stats()
saved = frag[frag['main_score'].notna()]['main_score']

for cutoff in ['2026-09-18', '2026-09-17', '2026-09-16']:
    net = breadth.loc[:cutoff, 'nyse_net'].reindex(spy.index)
    out = nyse_risk.compute_nyse_main(base, spy, net, stats)
    got = out['main_score'].reindex(saved.index)
    print(f'breadth through {cutoff}:',
          {str(ix.date()): round(float(got.loc[ix]), 4) for ix in saved.index},
          '| saved:', {str(ix.date()): round(float(v), 4) for ix, v in saved.items()})
