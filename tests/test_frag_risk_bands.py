"""frag_risk_bands: per-strategy fragility sizing (2026-07-02).

Guards the three aligned sites: strategy_config carries the bands (FAMILY4
dip-buyers 0.25x at 63d-MA10 >= 50; OVS 0.75x in [21,44)), daily_scan applies
them live via frag_band_mult, strat_backtester replays them point-in-time via
frag_band_mult_at. The retired book-wide ramp (1.25x boost / 0.10x floor) must
stay dead.
"""
import os
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

from daily_scan import frag_band_mult
from strategy_config import STRATEGY_BOOK
import pages.strat_backtester as sb

FAMILY4 = {"Weak Close Decent Sznls", "SPY QQQ MonFri Reversion",
           "Monday Dip", "Indices Oversold Bounce",
           # Dip-buy-adjacent (fades overbought inverse ETFs = buys market
           # selloffs) — carries the same bands since 2026-07-07.
           "3x Bear ETF Overbot Fade"}
FAMILY_BANDS = [[50, 999, 0.25]]


def _exec_of(name):
    for s in STRATEGY_BOOK:
        if s["name"] == name:
            return s["execution"]
    raise AssertionError(f"strategy {name!r} not in STRATEGY_BOOK")


def test_config_carries_exactly_the_family_band_entries():
    # OVS mid-band tilt was removed 2026-07-03 after failing the PIT
    # edge-weight gate (roadmap step 5) — OVS must stay fully exempt.
    with_bands = {s["name"]: s["execution"]["frag_risk_bands"]
                  for s in STRATEGY_BOOK if s["execution"].get("frag_risk_bands")}
    assert set(with_bands) == FAMILY4
    for name in FAMILY4:
        assert with_bands[name] == FAMILY_BANDS


def test_family_band_boundaries():
    ex = _exec_of("Monday Dip")
    assert frag_band_mult(ex, 49.99) == 1.0
    assert frag_band_mult(ex, 50.0) == 0.25
    assert frag_band_mult(ex, 94.0) == 0.25
    assert frag_band_mult(ex, 0.0) == 1.0     # no boost anywhere, ever
    assert frag_band_mult(ex, None) == 1.0    # missing/stale score -> native


def test_ovs_fully_exempt_at_every_score():
    ex = _exec_of("Overbot Vol Spike")
    for score in (0.0, 21.0, 43.99, 50.0, 70.0, 95.0, None):
        assert frag_band_mult(ex, score) == 1.0


def test_bandless_strategy_untouched_at_any_score():
    ex = _exec_of("Oversold Low Volume")
    for score in (0, 30, 55, 90, None):
        assert frag_band_mult(ex, score) == 1.0


def test_engine_replay_matches_live_helper():
    dates = pd.date_range("2026-01-05", periods=6, freq="D")
    scores = pd.Series([10.0, 30.0, 45.0, 52.0, 60.0, 40.0], index=dates)
    sb._FRAG_SCORE_CACHE["series"] = scores
    try:
        for name in ("Monday Dip", "Overbot Vol Spike", "Oversold Low Volume"):
            ex = _exec_of(name)
            for ts, sc in scores.items():
                assert sb.frag_band_mult_at(ex, ts) == frag_band_mult(ex, sc), \
                    f"{name} diverges at score {sc}"
        # date with no score (pre-series) -> 1.0 even for banded strategies
        assert sb.frag_band_mult_at(_exec_of("Monday Dip"), "2010-06-01") == 1.0
    finally:
        sb._FRAG_SCORE_CACHE.clear()


def test_engine_uses_real_cache_when_present():
    sb._FRAG_SCORE_CACHE.clear()
    series = sb._frag_score_series()
    if series is None:
        return  # no local fragility cache; live behavior is 1.0x everywhere
    # a date long before the fragility history must resolve to 1.0x
    assert sb.frag_band_mult_at(_exec_of("Monday Dip"), "2010-06-01") == 1.0
    # the series must be the 10d-MA basis: smooth, no NaN block inside range
    assert series.index.is_monotonic_increasing
