"""A3 extraction guard: the scoring core moved to fragility_core.py verbatim.

Two protections: (1) the page re-exports ARE the core objects (no fork can
drift), (2) golden values on a synthetic case lock the scoring math across
future edits — computed by hand from the documented formula:
score = (active_weight / max_weight) * 80 * regime_mult * calm_mult.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

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

import fragility_core as core


def test_page_reexports_are_core_objects():
    import pages.risk_dashboard_v2 as page
    for name in ("load_horizon_stats", "_signal_edge",
                 "_compute_calm_multiplier_scalar",
                 "_compute_calm_multiplier_series", "_days_since_last_fire",
                 "_signal_decay_weight", "_compute_decay_metadata",
                 "compute_horizon_fragility", "compute_fragility_timeseries",
                 "compute_fragility_bundle"):
        assert getattr(page, name) is getattr(core, name), name


# compute_horizon_fragility looks up the seven CANONICAL signal names —
# synthetic names score zero by construction. Use two canonical ones.
SIG_ON = "Distribution Dominance"
SIG_NEVER = "VIX Range Compression"


def _stats(edge_on=2.0, edge_never=1.0):
    def h(diff):
        return {hz: {"diff_mean": -diff} for hz in ("5d", "21d", "63d")}
    return {"signals": {SIG_ON: {"horizons": h(edge_on)},
                        SIG_NEVER: {"horizons": h(edge_never)}}}


def _flat_spy(n=300, price=100.0):
    # Dead-flat SPY, long enough for every 252d lookback to resolve:
    # ret_12m = 0, extension_200d = 0, drawdown = 0 -> regime vector
    # = 1.0 + 0.10 (drawdown > -0.02 bonus) = 1.10 on resolved rows.
    return pd.Series(price, index=pd.bdate_range("2025-01-02", periods=n))


def test_horizon_fragility_golden_value():
    spy = _flat_spy(300)
    idx = spy.index
    hist_on = pd.Series(False, index=idx)
    hist_on.iloc[-1] = True
    signals = {SIG_ON: {"on": True, "signal_history": hist_on},
               SIG_NEVER: {"on": False,
                           "signal_history": pd.Series(False, index=idx)}}

    scores = core.compute_horizon_fragility(
        signals, regime_mult=1.0, horizon_stats=_stats(),
        price_ctx={"drawdown": 0.0}, spy_close=spy)

    # No correction ever in 300 flat days -> calm streaks = 300:
    # 5% streak 300 >= p95 152 -> 1.20; 10% streak 300 >= p90 292 -> 1.10.
    calm = 1.20 * 1.10
    # ON signal edge 2 over denominator 3 (never-fired still in max_weight):
    for hz in ("5d", "21d", "63d"):
        assert scores[hz] == pytest.approx(2.0 / 3.0 * 80.0 * 1.0 * calm)


def test_timeseries_golden_and_decay():
    spy = _flat_spy(300)
    idx = spy.index
    hist_on = pd.Series(False, index=idx)
    hist_on.iloc[260:266] = True         # ON for 6 sessions, then decays
    signals = {SIG_ON: {"signal_history": hist_on},
               SIG_NEVER: {"signal_history": pd.Series(False, index=idx)}}

    ts = core.compute_fragility_timeseries(signals, spy, _stats())
    calm = core._compute_calm_multiplier_series(spy)
    # hand-check the calm series at the probed rows (streak = row index):
    assert calm.iloc[265] == pytest.approx(1.20 * 1.05)   # 265 in [203, 292)
    assert calm.iloc[275] == pytest.approx(1.20 * 1.05)

    on_expected = 2.0 / 3.0 * 80.0 * 1.10 * calm.iloc[265]
    assert ts["63d"].iloc[265] == pytest.approx(on_expected)
    # 10 sessions after off: weight = (63-10)/63, spy at high -> factor 1
    decay_expected = (2.0 * (63 - 10) / 63) / 3.0 * 80.0 * 1.10 * calm.iloc[275]
    assert ts["63d"].iloc[275] == pytest.approx(decay_expected)
    # before first fire: exactly 0
    assert ts["63d"].iloc[255] == pytest.approx(0.0)


def test_bundle_without_stats_is_none_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "HORIZON_STATS_PATH",
                        str(tmp_path / "missing.json"))
    spy = _flat_spy(60)
    out = core.compute_fragility_bundle({}, 1.0, {}, spy)
    assert out == {"horizon_stats": None, "h_scores": None,
                   "h_scores_10d": None, "frag_df": None}


def test_bundle_ts_write_stamps_vintage(tmp_path):
    spy = _flat_spy(180)
    hist = pd.Series(False, index=spy.index)
    hist.iloc[100:106] = True
    signals = {"A": {"on": False, "signal_history": hist}}
    stats_path = tmp_path / "stats.json"
    import json
    stats_path.write_text(json.dumps(_stats()))
    orig = core.HORIZON_STATS_PATH
    core.HORIZON_STATS_PATH = str(stats_path)
    try:
        out_path = tmp_path / "ts.parquet"
        bundle = core.compute_fragility_bundle(
            signals, 1.0, {"drawdown": 0.0}, spy, ts_write_path=str(out_path))
        assert out_path.exists()
        assert bundle["frag_df"] is not None
        import pyarrow.parquet as pq
        md = pq.read_schema(str(out_path)).metadata or {}
        assert md.get(b"fragility_basis") == b"raw_recompute"
    finally:
        core.HORIZON_STATS_PATH = orig
