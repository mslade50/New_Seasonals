"""merge_fragility_history: the fragility cache must be append-only.

Guards the 2026-07-02 change: daily_risk_report recomputes the full fragility
series in memory each run, but the on-disk cache (data/rd2_fragility.parquet)
freezes history — a recompute whose historical values drifted must not rewrite
rows before the run date.
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Stub streamlit before importing daily_risk_report (it transitively imports
# risk_dashboard_v2). Chainable + falsy so the page's import-time
# `st.runtime.exists()` gate evaluates False and its main() never runs.
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

from daily_risk_report import merge_fragility_history

COLS = ["5d", "21d", "63d"]


def _frame(dates, base):
    idx = pd.to_datetime(dates)
    return pd.DataFrame({c: [base + i for i in range(len(idx))] for c in COLS}, index=idx)


def test_nightly_append_freezes_history():
    existing = _frame(["2026-06-29", "2026-06-30", "2026-07-01"], base=10)
    # recompute drifted: same dates, different values, plus today
    recomputed = _frame(["2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02"], base=99)
    merged, frozen_through = merge_fragility_history(
        recomputed, existing, pd.Timestamp("2026-07-02"))
    assert frozen_through == pd.Timestamp("2026-07-01")
    # history untouched despite the drifted recompute
    pd.testing.assert_frame_equal(merged.loc[:"2026-07-01"], existing)
    # only today appended, from the recompute
    assert merged.index[-1] == pd.Timestamp("2026-07-02")
    assert merged.loc["2026-07-02", "63d"] == recomputed.loc["2026-07-02", "63d"]


def test_same_day_rerun_refreshes_todays_row():
    existing = _frame(["2026-06-30", "2026-07-01", "2026-07-02"], base=10)
    recomputed = _frame(["2026-06-30", "2026-07-01", "2026-07-02"], base=50)
    merged, frozen_through = merge_fragility_history(
        recomputed, existing, pd.Timestamp("2026-07-02"))
    assert frozen_through == pd.Timestamp("2026-07-01")
    pd.testing.assert_frame_equal(merged.loc[:"2026-07-01"], existing.loc[:"2026-07-01"])
    # today's row comes from the fresh recompute
    assert merged.loc["2026-07-02", "5d"] == recomputed.loc["2026-07-02", "5d"]
    assert len(merged) == 3


def test_missed_days_backfilled_from_recompute():
    existing = _frame(["2026-06-26", "2026-06-29"], base=10)  # producer missed 6/30-7/1
    recomputed = _frame(["2026-06-26", "2026-06-29", "2026-06-30", "2026-07-01", "2026-07-02"], base=70)
    merged, _ = merge_fragility_history(recomputed, existing, pd.Timestamp("2026-07-02"))
    assert list(merged.index) == list(recomputed.index)
    pd.testing.assert_frame_equal(merged.loc[:"2026-06-29"], existing)
    assert merged.loc["2026-06-30", "63d"] == recomputed.loc["2026-06-30", "63d"]


def test_bootstrap_when_no_history():
    recomputed = _frame(["2026-07-01", "2026-07-02"], base=5)
    merged, frozen_through = merge_fragility_history(
        recomputed, recomputed.iloc[0:0], pd.Timestamp("2026-07-02"))
    assert frozen_through is None
    pd.testing.assert_frame_equal(merged, recomputed)


def test_append_keeps_existing_column_schema():
    existing = _frame(["2026-07-01"], base=10)
    recomputed = _frame(["2026-07-01", "2026-07-02"], base=20)
    recomputed["extra"] = 1.0  # future code adds a column — schema must not change
    merged, _ = merge_fragility_history(recomputed, existing, pd.Timestamp("2026-07-02"))
    assert list(merged.columns) == COLS
