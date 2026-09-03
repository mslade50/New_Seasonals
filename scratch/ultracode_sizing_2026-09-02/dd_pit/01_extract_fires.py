"""dd_pit step 1: re-run the production risk-signal pipeline through the latest
close and persist per-signal fire histories + SPY closes, exactly the way
scratch/pit_extract_signals.py did on 2026-07-03 (whose pickle stops at
2026-07-02 and is the only reason the study's PIT dial stops there).

The rd2_* parquet caches are REDIRECTED into this folder so the repo's own
risk caches are never overwritten. Start date is pinned to 2016-07-05, the
first row of the 2026-07-03 pickle, so the two fire histories share their
burn-in window and can be diffed over the overlap.
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))


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


sys.modules["streamlit"] = _NoOp()
import pages.risk_dashboard_v2 as rd2  # noqa: E402

rd2.CACHE_SPY_OHLC = str(HERE / "rd2_spy_ohlc.parquet")
rd2.CACHE_CLOSES = str(HERE / "rd2_closes.parquet")
rd2.CACHE_SP500 = str(HERE / "rd2_sp500_closes.parquet")

from daily_risk_report import compute_all_signals  # noqa: E402

START = "2016-07-05"
if not Path(rd2.CACHE_SPY_OHLC).exists() or "--refresh" in sys.argv:
    print(f"downloading risk data from {START} into {HERE} ...")
    rd2.refresh_all_data(START, progress_callback=lambda t, p: print(f"  {p:5.0%} {t}"))
spy_df, closes, sp500_closes = rd2.load_cached_data()
print(f"spy_df {spy_df.index.min().date()} .. {spy_df.index.max().date()} n={len(spy_df)}; "
      f"closes {closes.shape}; sp500 {None if sp500_closes is None else sp500_closes.shape}")

computed = compute_all_signals(spy_df, closes, sp500_closes)
signals_ordered = None
for k, v in computed.items():
    if isinstance(v, dict) and v and all(isinstance(x, dict) for x in v.values()):
        if any("signal_history" in x for x in v.values()):
            signals_ordered = v
            print(f"(signals found under key '{k}')")
            break
assert signals_ordered, f"no signals container; keys={list(computed.keys())}"

fires = {}
for name, sig in signals_ordered.items():
    h = sig.get("signal_history")
    if h is not None and hasattr(h, "empty") and not h.empty:
        fires[name] = h.astype(bool)
spy_close = computed["spy_close"]
fires_df = pd.DataFrame(fires).reindex(pd.to_datetime(spy_close.index)).fillna(False)
out = {"fires": fires_df, "spy_close": spy_close,
       "frag_df_current": computed.get("frag_df"), "horizon_stats_current": computed.get("horizon_stats")}
with open(HERE / "pit_signals_extended.pkl", "wb") as f:
    pickle.dump(out, f)
print(f"saved: {fires_df.shape[1]} signals x {len(fires_df)} days ({fires_df.index.min().date()} -> {fires_df.index.max().date()})")
print("fire counts:", fires_df.sum().to_dict())

# --- diff against the 2026-07-03 pickle over the overlap ---
old = pickle.load(open(ROOT / "scratch/pit_signals.pkl", "rb"))
of = old["fires"]; of.index = pd.to_datetime(of.index)
common = of.index.intersection(fires_df.index)
print(f"\noverlap with 2026-07-03 pickle: {common.min().date()} .. {common.max().date()} n={len(common)}")
for c in of.columns:
    if c in fires_df.columns:
        a = of.loc[common, c].astype(bool); b = fires_df.loc[common, c].astype(bool)
        diff = (a != b)
        print(f"  {c:<28} old fires {int(a.sum()):4d} new fires {int(b.sum()):4d} disagreeing days {int(diff.sum()):4d}"
              + (f"  (first {diff[diff].index.min().date()}, last {diff[diff].index.max().date()})" if diff.any() else ""))
os_ = old["spy_close"]; os_.index = pd.to_datetime(os_.index)
sc = spy_close.copy(); sc.index = pd.to_datetime(sc.index)
cm = os_.index.intersection(sc.index)
print(f"SPY close max abs rel diff over overlap: {((os_.loc[cm] / sc.loc[cm]) - 1).abs().max():.2e}")
