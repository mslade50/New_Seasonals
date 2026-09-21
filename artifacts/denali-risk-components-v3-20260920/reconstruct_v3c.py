"""Full-history reconstruction of the seven components and the main dial.

Reuses the validated reconstruction path in scripts/build_atr_downside_stats.py:
the production compute_* functions from pages/risk_dashboard_v2 fed the 25-year
master_prices cache, so no signal logic is rewritten here. The composite then
goes through fragility_core.compute_fragility_timeseries, the five-session mean
the production writer stores, and the ten-session mean live consumers read, which
is the same chain daily_risk_report runs each evening.

Writes v3c_reconstruction.parquet (per-component fires, availability, base dial)
and prints the overlap checks against the production point-in-time parquet and
against the frozen fire history the v3b study used. Read-only apart from the two
artifacts it writes into this folder.
"""
from pathlib import Path
import json
import pickle
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO))

import build_atr_downside_stats as bads  # installs the streamlit shim + page path

from fragility_core import compute_fragility_timeseries, load_horizon_stats

RUNTIME_FRAG = Path(r'C:\Users\McKinley Slade\dev\New_Seasonals-automation-runtime-v9'
                    r'\data\rd2_fragility.parquet')
FIRES_PKL = REPO / 'scratch/ultracode_sizing_2026-09-02/dd_pit/pit_signals_extended.pkl'
PRICE_NAMES = ['Distribution Dominance', 'Defensive Leadership', 'VIX Range Compression',
               'Low Absorption Ratio', 'Seasonal Rank Divergence', 'Dispersion']
# The series each component's firing test reads. A component cannot fire before
# its own driver is defined, so the latest of these is where the composite is
# first fully formed.
DRIVERS = {'Distribution Dominance': 'da_ratio', 'Defensive Leadership': 'spread',
           'VIX Range Compression': 'compression_pctile', 'Low Absorption Ratio': 'ar_pctile',
           'Seasonal Rank Divergence': 'spread', 'Dispersion': 'composite_pctile'}

print('building 25y inputs from master_prices ...')
spy_df, closes, sp500_closes = bads.build_inputs_from_master()
spy_close = spy_df['Close']
spy_close.index = pd.to_datetime(spy_close.index).tz_localize(None)
spy_df.index = spy_close.index
print(f'  SPY {spy_close.index.min().date()} -> {spy_close.index.max().date()} '
      f'({len(spy_close)} sessions); sp500 cols {sp500_closes.shape[1]}; '
      f'close cols {list(closes.columns)}')

masks = bads.compute_signal_masks(spy_df, closes, sp500_closes)

out = pd.DataFrame(index=spy_close.index)
starts = {}
print('\ncomponent                       history first    driver first   first fire   fires')
for name in PRICE_NAMES:
    hist = masks[name]['signal_history'].dropna().astype(bool)
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    driver = masks[name][DRIVERS[name]].dropna()
    driver.index = pd.to_datetime(driver.index).tz_localize(None)
    ready = max(hist.index.min(), driver.index.min())
    fire = hist.reindex(out.index).fillna(False).astype(bool)
    out[f'fire_{name}'] = fire
    out[f'avail_{name}'] = pd.Series(out.index >= ready, index=out.index)
    starts[name] = {'history_first': str(hist.index.min().date()),
                    'driver': DRIVERS[name],
                    'driver_first': str(driver.index.min().date()),
                    'ready': str(ready.date()),
                    'first_fire': str(hist[hist].index.min().date()),
                    'fires_full_history': int(hist.sum())}
    print(f'  {name:30s} {hist.index.min().date()}      {driver.index.min().date()}     '
          f'{hist[hist].index.min().date()}   {int(hist.sum()):5d}')
ready_all = max(pd.Timestamp(v['ready']) for v in starts.values())
print(f'  all six components formed from {ready_all.date()}')

stats = load_horizon_stats()
frag_raw = compute_fragility_timeseries(masks, spy_close, stats)
frag_smoothed = frag_raw.rolling(5, min_periods=1).mean()
base_recon = frag_smoothed['63d'].rolling(10, min_periods=1).mean()
out['base_recon'] = base_recon
first_recon = base_recon.dropna().index.min()
print(f'\nreconstructed base dial {first_recon.date()} -> '
      f'{base_recon.dropna().index.max().date()} n={int(base_recon.notna().sum())}')

frag = pd.read_parquet(RUNTIME_FRAG)
frag.index = pd.to_datetime(frag.index).tz_localize(None)
base_prod = frag['63d'].dropna().rolling(10, min_periods=1).mean()
out['base_prod'] = base_prod.reindex(out.index)
shared = base_prod.index.intersection(base_recon.dropna().index)
diff = (base_prod.reindex(shared) - base_recon.reindex(shared))
print(f'overlap with production point-in-time base: {len(shared)} sessions '
      f'{shared.min().date()} -> {shared.max().date()}')
print(f'  corr {base_prod.reindex(shared).corr(base_recon.reindex(shared)):.4f} '
      f'| mean abs {diff.abs().mean():.2f} | median abs {diff.abs().median():.2f} '
      f'| p95 abs {diff.abs().quantile(.95):.2f} | max abs {diff.abs().max():.2f}')
for cut in [50]:
    agree = ((base_prod.reindex(shared) >= cut) == (base_recon.reindex(shared) >= cut)).mean()
    print(f'  side-of-{cut} agreement {agree:.3%}')

raw = pickle.load(FIRES_PKL.open('rb'))
print('\nfire reconstruction vs the frozen history the v3b study used:')
for name in PRICE_NAMES:
    old = pd.Series(raw['fires'][name]).astype(bool)
    old.index = pd.to_datetime(old.index).tz_localize(None)
    idx = old.index.intersection(out.index)
    new = out.loc[idx, f'fire_{name}']
    agree = (new == old.reindex(idx)).mean()
    print(f'  {name:30s} overlap {len(idx):5d} agree {agree:.4%} '
          f'| frozen fires {int(old.reindex(idx).sum()):4d} new {int(new.sum()):4d}')

breadth = pd.read_parquet(REPO / 'data/market_breadth.parquet')
breadth.index = pd.to_datetime(breadth.index).tz_localize(None)
net = breadth['nyse_net'].reindex(out.index)
out['nyse_net'] = net
print(f'\nbreadth on the SPY calendar: {int(net.notna().sum())} of {len(net)} sessions covered; '
      f'first {net.dropna().index.min().date()} last {net.dropna().index.max().date()}')
missing = net[net.isna()]
if len(missing):
    print(f'  missing sessions: {len(missing)}; first few {[str(d.date()) for d in missing.index[:6]]}'
          f' last few {[str(d.date()) for d in missing.index[-6:]]}')
    by_year = missing.groupby(missing.index.year).size()
    print('  missing by year:', by_year.to_dict())

prod = frag['63d'].dropna()
gaps = out.index[(out.index >= prod.index.min()) & (out.index <= prod.index.max())].difference(prod.index)
continuous_from = gaps.max() if len(gaps) else prod.index.min()
if len(gaps):
    continuous_from = out.index[out.index > continuous_from].min()
print(f'\nstored record {prod.index.min().date()} -> {prod.index.max().date()}: '
      f'{len(gaps)} trading days missing inside its own span; continuous from '
      f'{continuous_from.date()}')

starts['_meta'] = {
    'spy_first': str(spy_close.index.min().date()),
    'spy_last': str(spy_close.index.max().date()),
    'all_components_formed': str(ready_all.date()),
    'base_recon_first': str(first_recon.date()),
    'stored_record_first': str(prod.index.min().date()),
    'stored_record_gaps_inside_span': int(len(gaps)),
    'stored_record_continuous_from': str(continuous_from.date()),
    'breadth_first_on_spy_calendar': str(net.dropna().index.min().date()),
    'breadth_missing_sessions': [str(d.date()) for d in missing.index],
}
(ROOT / 'v3c_component_starts.json').write_text(json.dumps(starts, indent=2), encoding='utf-8')

out.to_parquet(ROOT / 'v3c_reconstruction.parquet')
frag_smoothed.to_parquet(ROOT / 'v3c_frag_smoothed.parquet')
print(f'\nwrote {ROOT / "v3c_reconstruction.parquet"}')
