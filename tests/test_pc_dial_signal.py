"""Equity P/C Complacency dial signal (2026-08-05) — 5d-only invariants.

The signal joins the fragility composite at the 5d horizon ONLY. These tests
freeze the three contracts that make that safe:
  1. the stats JSON entry carries no 21d/63d horizons (zero edge there),
  2. adding the signal leaves the 21d/63d composite series BYTE-IDENTICAL
     (the sizing 63d column and the exposure-leg 21d input are untouched),
  3. the pre-registered simple-dial shadow stays a 7-signal sum.
"""
import json
import os
import sys

import numpy as np
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


sys.modules.setdefault('streamlit', _NoOp())

from fragility_core import (compute_fragility_timeseries,
                            compute_horizon_fragility, _signal_edge)
from fragility_simple import SIMPLE_SIGNALS, compute_simple_dial

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATS_PATH = os.path.join(_ROOT, "data", "signal_horizon_stats.json")
NAME = "Equity P/C Complacency"


def _stats():
    with open(STATS_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_stats_entry_is_5d_only():
    entry = _stats()["signals"][NAME]
    assert list(entry["horizons"]) == ["5d"]
    assert entry["horizons"]["5d"]["diff_mean"] < 0
    assert _signal_edge(_stats(), NAME, "5d") > 0
    assert _signal_edge(_stats(), NAME, "21d") == 0.0
    assert _signal_edge(_stats(), NAME, "63d") == 0.0


def test_generator_restricts_signal_to_5d():
    from scripts.build_signal_horizon_stats import HORIZON_RESTRICT
    assert HORIZON_RESTRICT.get(NAME) == {"5d"}


def _synth_world():
    idx = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.RandomState(7)
    spy = pd.Series(400 * np.cumprod(1 + rng.normal(0, 0.008, len(idx))),
                    index=idx)
    disp_hist = pd.Series(False, index=idx)
    disp_hist.iloc[100:110] = True
    pcc_hist = pd.Series(False, index=idx)
    pcc_hist.iloc[150:200] = True
    base = {
        "Dispersion": {"on": False, "signal_history": disp_hist},
    }
    withpcc = dict(base)
    withpcc[NAME] = {"on": True, "signal_history": pcc_hist}
    stats = {"signals": {
        "Dispersion": {"horizons": {
            "5d": {"diff_mean": -0.5}, "21d": {"diff_mean": -1.0},
            "63d": {"diff_mean": -2.9}}},
        NAME: {"horizons": {"5d": {"diff_mean": -0.18}}},
    }}
    return spy, base, withpcc, stats


def test_21d_63d_series_byte_identical_with_signal_added():
    spy, base, withpcc, stats = _synth_world()
    a = compute_fragility_timeseries(base, spy, stats)
    b = compute_fragility_timeseries(withpcc, spy, stats)
    pd.testing.assert_series_equal(a["63d"], b["63d"])
    pd.testing.assert_series_equal(a["21d"], b["21d"])
    # and the 5d series DOES move where the new signal is active
    active = withpcc[NAME]["signal_history"]
    assert not a["5d"][active].equals(b["5d"][active])


def test_scalar_scores_21d_63d_unchanged():
    spy, base, withpcc, stats = _synth_world()
    ctx = {"drawdown": -0.01}
    a = compute_horizon_fragility(base, 1.0, stats, ctx, spy_close=None)
    b = compute_horizon_fragility(withpcc, 1.0, stats, ctx, spy_close=None)
    assert a["63d"] == b["63d"]
    assert a["21d"] == b["21d"]
    assert b["5d"] != a["5d"]  # ON signal must move the 5d dial


def test_simple_dial_shadow_pinned_to_registered_seven():
    # The production caller (daily_risk_report) filters signals_ordered to
    # SIMPLE_SIGNALS before calling compute_simple_dial — replicate that
    # filter here and assert an added 8th signal cannot change the shadow.
    assert NAME not in SIMPLE_SIGNALS and len(SIMPLE_SIGNALS) == 7
    spy, base, withpcc, _ = _synth_world()
    seven = {n: {"on": False, "signal_history":
                 pd.Series(True, index=spy.index[:50]).reindex(spy.index, fill_value=False)}
             for n in SIMPLE_SIGNALS}
    eight = dict(seven)
    eight[NAME] = withpcc[NAME]
    filtered = {n: eight.get(n, {}) for n in SIMPLE_SIGNALS}
    a = compute_simple_dial(seven, spy.index)
    b = compute_simple_dial(filtered, spy.index)
    pd.testing.assert_frame_equal(a, b)
    assert b.attrs["n_signals"] == 7
    # and the source-of-truth caller really does filter
    import inspect, daily_risk_report
    src = inspect.getsource(daily_risk_report)
    assert "_simple_inputs" in src and "SIMPLE_SIGNALS" in src


def test_compute_signal_smoke_on_real_cache():
    from pages.risk_dashboard_v2 import compute_pc_complacency_signal
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=400)
    spy = pd.Series(500.0, index=idx)
    sig = compute_pc_complacency_signal(spy)
    assert set(sig) >= {"on", "detail", "summary", "signal_history", "pc_pctile"}
    assert isinstance(sig["on"], bool)
    if not sig["pc_pctile"].empty:  # real cache present
        assert sig["signal_history"].index.equals(spy.index)
        assert sig["signal_history"].dtype == bool
