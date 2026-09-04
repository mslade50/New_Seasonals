"""PIT step 2: expanding-window edge weights -> vintage fragility series ->
re-run the headline sizing cells.

Method:
- estimate_stats(end): event study using ONLY data through `end` — per signal,
  per horizon h, diff_mean = mean forward-h SPY return on fire days minus the
  unconditional mean (percentage points), fire days capped at end - h so the
  forward window is fully realized by the vintage date.
- Vintages at each year-end 2017..2025; scores for calendar year Y use vintage
  Y-1. The composite is rebuilt per vintage with the PRODUCTION function
  (pages.risk_dashboard_v2.compute_fragility_timeseries), then the year-Y rows
  are stitched into the PIT series. Basis matching the live chain exactly:
  raw -> rolling(5).mean() [parquet basis] -> 63d -> rolling(10).mean() ->
  daily ffill(limit=5).
- Cells on 2018+ signals (first honest vintage year), PIT vs current-weights
  series side by side on the SAME window.

What this cures: weight-calibration lookahead. What it cannot cure: the
signal definitions/parameters themselves are today's code.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats as sstats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _NoOp:
    def __getattr__(self, name): return self
    def __call__(self, *a, **k): return self
    def __bool__(self): return False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cache_data(self, *a, **k):
        def deco(fn): return fn
        return deco
    cache_resource = cache_data


sys.modules['streamlit'] = _NoOp()
from pages.risk_dashboard_v2 import compute_fragility_timeseries

with open('scratch/pit_signals.pkl', 'rb') as f:
    raw = pickle.load(f)
fires: pd.DataFrame = raw['fires']
spy: pd.Series = raw['spy_close']
spy.index = pd.to_datetime(spy.index)
fires.index = pd.to_datetime(fires.index)

HORIZONS = {'5d': 5, '21d': 21, '63d': 63}


def estimate_stats(end_date) -> dict:
    end = pd.Timestamp(end_date)
    out = {}
    for name in fires.columns:
        hor = {}
        for hl, h in HORIZONS.items():
            fwd = (spy.shift(-h) / spy - 1.0) * 100.0
            valid = fwd.index[fwd.index <= end - pd.Timedelta(days=int(h * 1.6))]
            fwd = fwd.reindex(valid).dropna()
            f = fires[name].reindex(fwd.index).fillna(False)
            if f.sum() >= 10:
                dm = float(fwd[f].mean() - fwd.mean())
            else:
                dm = None  # too few events by this vintage -> edge 0
            hor[hl] = {'diff_mean': dm}
        out[name] = {'horizons': hor}
    return {'signals': out}


def signals_shape():
    return {name: {'signal_history': fires[name]} for name in fires.columns}


# --- validation: full-sample recompute vs the shipped JSON ---
import json
shipped = json.load(open('data/signal_horizon_stats.json'))['signals']
mine = estimate_stats(fires.index.max())['signals']
print("=== validation: my full-sample diff_mean vs shipped JSON (63d) ===")
for name in fires.columns:
    s = shipped.get(name, {}).get('horizons', {}).get('63d', {}).get('diff_mean')
    m = mine[name]['horizons']['63d']['diff_mean']
    print(f"  {name:<38} shipped {s if s is not None else 'n/a':>7} | mine "
          f"{m:+.2f}" if m is not None else f"  {name:<38} shipped {s} | mine n/a")

# --- vintage series ---
sig_dict = signals_shape()
pit_frames = []
for year in range(2018, 2027):
    vint = estimate_stats(f"{year - 1}-12-31")
    frame = compute_fragility_timeseries(sig_dict, spy, vint)
    yr_rows = frame[frame.index.year == year]
    pit_frames.append(yr_rows)
    edges63 = {n: round(max(0.0, -(v['horizons']['63d']['diff_mean'] or 0)), 2)
               for n, v in vint['signals'].items()}
    print(f"vintage {year-1}: 63d edges {edges63}")
pit_raw = pd.concat(pit_frames).sort_index()

cur_raw = compute_fragility_timeseries(sig_dict, spy, {'signals': mine})
cur_raw = cur_raw[cur_raw.index.year >= 2018]


def live_basis(frame):
    s = frame['63d'].rolling(5, min_periods=1).mean()       # parquet basis
    s = s.dropna().rolling(10, min_periods=1).mean()         # scan MA
    s.index = pd.to_datetime(s.index).normalize()
    grid = pd.date_range(s.index.min(), s.index.max(), freq='D')
    return s.reindex(grid).ffill(limit=5)


pit = live_basis(pit_raw)
cur = live_basis(cur_raw)
both = pd.concat([pit.rename('pit'), cur.rename('cur')], axis=1).dropna()
print(f"\nPIT vs current-weights series (2018+): corr {both.pit.corr(both.cur):.3f}, "
      f"rank-corr {both.pit.corr(both.cur, method='spearman'):.3f}")
agree = ((both.pit >= 50) == (both.cur >= 50)).mean()
print(f">=50 day-classification agreement: {agree*100:.1f}%  "
      f"(days >=50: PIT {(both.pit>=50).mean()*100:.1f}% vs cur {(both.cur>=50).mean()*100:.1f}%)")

# --- re-run the sizing cells on 2018+ trades, both series ---
led = pd.read_parquet('scratch/ledger_pre_frag_bands.parquet')
led['Signal Date'] = pd.to_datetime(led['Signal Date']).dt.normalize()
led = led[led['Signal Date'] >= '2018-01-01'].dropna(subset=['R_Multiple']).copy()
led['ym'] = led['Signal Date'].dt.to_period('M')
is_ovs = led.Strategy.str.contains('Overbot Vol|OVS', case=False, na=False)
FAMILY4 = {"Weak Close Decent Sznls", "SPY QQQ MonFri Reversion",
           "Monday Dip", "Indices Oversold Bounce"}
is_fam = led.Strategy.isin(FAMILY4)


def cell(d, mask_hi, label):
    hi, lo = d[mask_hi], d[~mask_hi]
    if len(hi) < 10:
        return f"  {label:<26} N too small ({len(hi)})"
    hm = hi.groupby('ym')['R_Multiple'].mean()
    lm = lo.groupby('ym')['R_Multiple'].mean()
    t, p = sstats.ttest_ind(hm, lm, equal_var=False)
    return (f"  {label:<26} hi {hi.R_Multiple.mean():+.3f} (N={len(hi):>3}) vs "
            f"{lo.R_Multiple.mean():+.3f} (N={len(lo):>4})  t={t:+.2f} p={p:.3f}")


for tag, series in [('CURRENT weights', cur), ('PIT weights', pit)]:
    led['score'] = series.reindex(led['Signal Date']).values
    d = led.dropna(subset=['score'])
    nb, fam = d[~is_ovs.reindex(d.index)], d[is_fam.reindex(d.index)]
    rest = d[(~is_ovs.reindex(d.index)) & (~is_fam.reindex(d.index))]
    ovs = d[is_ovs.reindex(d.index)]
    print(f"\n=== {tag} (2018+ trades, N={len(d)}) ===")
    print(cell(nb, nb.score >= 50, 'book ex-OVS >=50'))
    print(cell(fam, fam.score >= 50, 'FAMILY4 >=50'))
    print(cell(rest, rest.score >= 50, 'rest ex-OVS >=50'))
    print(cell(ovs, (ovs.score >= 21) & (ovs.score < 44), 'OVS mid-band [21,44)'))
