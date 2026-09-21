"""Read-only research on the single main risk dial. Frozen inputs, no trading.

The dial series is the production point-in-time record written by the pinned
runtime (`main_dial_from_frame`: explicit saved `main_score` where present,
otherwise the ten-session mean of the legacy 63d column). SPY adjusted closes
come from the shared master price cache. Nothing here writes a production file.
"""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path.insert(0, str(REPO))

RUNTIME_FRAG = Path(r'C:\Users\McKinley Slade\dev\New_Seasonals-automation-runtime-v9'
                    r'\data\rd2_fragility.parquet')
MASTER = REPO / 'data/master_prices.parquet'

# Frozen cohort, identical to the 17 September study so component tables and
# dial tables describe one sample.
START = '2018-03-01'
PRICE_THROUGH = '2026-09-01'
HORIZONS = [5, 10, 21, 42, 63]

from fragility_core import load_main_dial_series, ACTIVE_RISK_SIGNALS, _signal_edge, load_horizon_stats
from nyse_risk import main_dial_from_frame

spy = pd.read_parquet(MASTER, filters=[('ticker', '==', 'SPY')],
                      columns=['ticker', 'date', 'Close'])
spy = spy.set_index('date')['Close'].sort_index()
spy.index = pd.to_datetime(spy.index).tz_localize(None)
price_all = spy.copy()
spy = spy.loc[:PRICE_THROUGH]

main_full = load_main_dial_series(str(RUNTIME_FRAG))
frag = pd.read_parquet(RUNTIME_FRAG)

df = pd.DataFrame(index=spy.index)
df['dial'] = main_full.reindex(spy.index)
for h in HORIZONS:
    df[f'f{h}'] = spy.shift(-h) / spy - 1
high = spy.rolling(252, min_periods=252).max()
df['dd'] = spy / high - 1
df['session'] = np.arange(len(df))
df = df.loc[START:].dropna(subset=['dial', 'f63'])
df.to_parquet(ROOT / 'range_observations.parquet')


def spaced(g, gap=63):
    keep, last = [], -9999
    for ix, r in g.iterrows():
        if r.session - last >= gap:
            keep.append(ix)
            last = r.session
    return g.loc[keep]


def summarize(g):
    ans = {'days': len(g), 'starts63': len(spaced(g)),
           'near_high_share': float((g.dd >= -.02).mean()) if len(g) else float('nan')}
    for h in HORIZONS:
        x = g[f'f{h}']
        z = spaced(g, h)
        ans.update({f'mean{h}': x.mean(), f'median{h}': x.median(),
                    f'negative{h}': (x < 0).mean(),
                    f'spaced_mean{h}': z[f'f{h}'].mean(), f'spaced_n{h}': len(z)})
    return ans


coarse = [0, 20, 40, 60, 80, np.inf]
fine = list(range(0, 101, 10)) + [np.inf]
rows = []
subsets = [('all', df), ('near_high', df[df.dd >= -.02]),
           ('2018_2021', df.loc[:'2021-12-31']), ('2022_2026', df.loc['2022-01-01':]),
           ('excluding_2020', df[df.index.year != 2020])]
for subset, sub in subsets:
    for bname, bins in [('display', coarse), ('ten_point', fine)]:
        for lo, hi in zip(bins, bins[1:]):
            g = sub[(sub.dial >= lo) & (sub.dial < hi)]
            rows.append({'subset': subset, 'bins': bname, 'lo': lo, 'hi': hi, **summarize(g)})
bands = pd.DataFrame(rows)
bands.to_csv(ROOT / 'dial_ranges.csv', index=False)

rows = []
for subset, sub in subsets:
    for threshold in [20, 30, 40, 50, 60, 70, 80]:
        for side in ['below', 'at_or_above']:
            g = sub[sub.dial < threshold] if side == 'below' else sub[sub.dial >= threshold]
            rows.append({'subset': subset, 'threshold': threshold, 'side': side, **summarize(g)})
pd.DataFrame(rows).to_csv(ROOT / 'thresholds.csv', index=False)

baseline = summarize(df)

# --- NYSE component arithmetic for the report text -------------------------
stats = load_horizon_stats()
weights = {name: _signal_edge(stats, name, '63d') for name in ACTIVE_RISK_SIGNALS}
w_lar = weights['Low Absorption Ratio']
total = sum(weights.values())
alpha = w_lar / (total + w_lar)

md = pq.read_schema(RUNTIME_FRAG).metadata or {}
meta = {k.decode(): v.decode() for k, v in md.items() if not k.decode().startswith('pandas')}

legacy_base = frag['63d'].dropna().rolling(10, min_periods=1).mean()
saved = frag[frag['main_score'].notna()]
floor_example = [{'date': str(ix.date()),
                  'dial_63d_column': round(float(frag.loc[ix, '63d']), 3),
                  'base_10session_mean': round(float(legacy_base.loc[ix]), 3),
                  'main_score': round(float(frag.loc[ix, 'main_score']), 3)}
                 for ix in saved.index]

manifest = {
    'report_date': '2026-09-20',
    'cohort_start': START,
    'last_signal_date': str(df.index.max().date()),
    'prices_through': str(spy.index.max().date()),
    'prices_available_through': str(price_all.index.max().date()),
    'n': len(df),
    'dial_series_first': str(main_full.index.min().date()),
    'dial_series_last': str(main_full.index.max().date()),
    'dial_basis': ('production point-in-time main dial: explicit saved main_score '
                   'where present, otherwise the ten-session mean of the saved '
                   '5-session-smoothed 63d column'),
    'return_basis': 'SPY adjusted close, close t to t+h; every date has a complete 63-session outcome.',
    'parquet_metadata': meta,
    'saved_main_score_rows': floor_example,
    'weights_63d': {k: round(v, 2) for k, v in weights.items()},
    'weights_63d_total': round(total, 2),
    'nyse_borrowed_weight': round(w_lar, 2),
    'nyse_alpha': round(alpha, 6),
    'baseline': baseline,
    'sources': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                for p in [RUNTIME_FRAG, REPO / 'data/signal_horizon_stats.json']},
}
(ROOT / 'study_manifest.json').write_text(json.dumps(manifest, indent=2, default=str))

print('COHORT', START, '->', manifest['last_signal_date'], 'N', len(df))
print('BASELINE', {f'mean{h}': round(baseline[f'mean{h}'] * 100, 3) for h in HORIZONS},
      'neg63', round(baseline['negative63'], 4))
print('ALPHA', round(alpha, 6), 'LAR weight', w_lar, 'total', total)
print('SAVED MAIN SCORES', floor_example)
show = ['lo', 'hi', 'days', 'starts63'] + [f'mean{h}' for h in HORIZONS] + ['negative63']
print(bands[(bands.subset == 'all') & (bands.bins == 'display')][show].round(4).to_string(index=False))
print('--- ten point ---')
print(bands[(bands.subset == 'all') & (bands.bins == 'ten_point')][show].round(4).to_string(index=False))
print('--- era split, display bins ---')
for label in ['2018_2021', '2022_2026', 'near_high']:
    sub = bands[(bands.subset == label) & (bands.bins == 'display')]
    print(label)
    print(sub[show].round(4).to_string(index=False))
print('--- thresholds (all) ---')
th = pd.read_csv(ROOT / 'thresholds.csv')
print(th[th.subset == 'all'][['threshold', 'side', 'days', 'mean5', 'mean21', 'mean63', 'negative63']].round(4).to_string(index=False))
