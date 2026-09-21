"""Reconstruct the NYSE component around the Aug-Sep 2026 episode, read-only."""
from pathlib import Path
import json
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
net = breadth['nyse_net'].reindex(spy.index)

frag = pd.read_parquet(RUNTIME_FRAG)
base = frag['63d'].dropna().rolling(10, min_periods=1).mean()
base = base.reindex(spy.index)

out = nyse_risk.compute_nyse_main(base, spy, net, load_horizon_stats())
ema = nyse_risk.smooth_nyse_net(net)
dist = (1 - spy / spy.rolling(252).max()).clip(lower=0)

view = pd.DataFrame({
    'spy': spy, 'pct_below_high': dist * 100, 'raw_net': net, 'ema5': ema,
    'severity': nyse_risk.warning_severity(net, dist),
    'base': out['base_main'], 'contribution': out['nyse_contribution'],
    'effective': out['nyse_effective'], 'main': out['main_score'],
}).loc['2026-08-10':'2026-09-18']
print(view.round(3).to_string())

saved = frag[frag['main_score'].notna()][['63d', 'main_score']]
print('\nSAVED point-in-time rows')
print(saved.round(4).to_string())
print('\nbase (10-session mean of saved 63d) on those dates')
print(base.reindex(saved.index).round(4).to_string())

flick = view.loc['2026-08-14':'2026-09-02', ['raw_net', 'ema5']]
print('\nraw non-negative prints in the episode:',
      [str(d.date()) for d in flick.index[flick.raw_net >= 0]])
print('ema5 non-negative prints in the episode:',
      [str(d.date()) for d in flick.index[flick.ema5 >= 0]])
first_raw = view.index[(view.raw_net < 0) & (view.pct_below_high <= 3)][0]
first_ema = view.index[(view.severity > 0)][0]
print('first raw-trigger fire:', first_raw.date(), 'first ema5 fire:', first_ema.date())

facts = {
    'saved_rows': {str(ix.date()): {'dial_63d': round(float(saved.loc[ix, '63d']), 3),
                                    'base_10session': round(float(base.loc[ix]), 3),
                                    'published_main': round(float(saved.loc[ix, 'main_score']), 3)}
                   for ix in saved.index},
    'reconstruction': {str(ix.date()): {k: (None if pd.isna(view.loc[ix, k]) else round(float(view.loc[ix, k]), 3))
                                        for k in ['spy', 'pct_below_high', 'raw_net', 'ema5',
                                                  'severity', 'base', 'contribution', 'main']}
                       for ix in view.index[-8:]},
}
(ROOT / 'nyse_example.json').write_text(json.dumps(facts, indent=2))
