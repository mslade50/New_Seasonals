"""PIT step 1: run the production signal pipeline once and persist the raw
materials — per-signal boolean fire histories + SPY closes — so the vintage
re-weighting can run offline. Uses the exact code path the risk report uses
(daily_risk_report.download_data + compute_all_signals)."""
import os
import pickle
import sys

import pandas as pd

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

from daily_risk_report import download_data, compute_all_signals

spy_df, closes, sp500_closes = download_data()
computed = compute_all_signals(spy_df, closes, sp500_closes)

signals_ordered = computed['signals_ordered'] if 'signals_ordered' in computed else None
if signals_ordered is None:
    # compute_all_signals returns a dict — find the signals container
    for k, v in computed.items():
        if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
            if any('signal_history' in x for x in v.values()):
                signals_ordered = v
                print(f"(signals found under key '{k}')")
                break
assert signals_ordered, f"no signals container; keys={list(computed.keys())}"

fires = {}
for name, sig in signals_ordered.items():
    h = sig.get('signal_history')
    if h is not None and hasattr(h, 'empty') and not h.empty:
        fires[name] = h.astype(bool)

spy_close = computed['spy_close']
out = {
    'fires': pd.DataFrame(fires).reindex(pd.to_datetime(spy_close.index)).fillna(False),
    'spy_close': spy_close,
    'frag_df_current': computed.get('frag_df'),
    'horizon_stats_current': computed.get('horizon_stats'),
}
with open('scratch/pit_signals.pkl', 'wb') as f:
    pickle.dump(out, f)
print(f"saved: {out['fires'].shape[1]} signals x {len(out['fires'])} days "
      f"({out['fires'].index.min().date()} -> {out['fires'].index.max().date()})")
print("fire counts:", out['fires'].sum().to_dict())
