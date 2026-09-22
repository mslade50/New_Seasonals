"""VIX duration/direction trigger boundaries, independent of the Streamlit UI."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def signal_module():
    path = Path(__file__).parents[1] / "pages" / "risk_dashboard_v2.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in {"_rolling_percentile", "compute_vix_range_compression"}]
    namespace = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def case(namespace, days=10):
    index = pd.bdate_range("2023-01-03", periods=540)
    prices = pd.Series(20. - np.arange(len(index)) * .001, index=index)
    percentile = pd.Series(50., index=index)
    percentile.iloc[-days:] = 14.
    namespace["_rolling_percentile"] = lambda metric, lookback: percentile
    return prices, percentile, namespace["compute_vix_range_compression"]


@pytest.mark.parametrize("days,expected", [(9, False), (10, True), (11, True)])
def test_requires_ten_consecutive_observations_without_ma_filter(signal_module, days, expected):
    prices, _, compute = case(signal_module, days)
    assert prices.iloc[-1] < prices.rolling(20).mean().iloc[-1]
    result = compute(prices)
    assert result["on"] is expected
    assert result["compression_age"].iloc[-1] == days
    assert result["signal_history"].iloc[-1] == expected


@pytest.mark.parametrize("percentile,expected", [(14.999, True), (15., False)])
def test_percentile_boundary_is_strict(signal_module, percentile, expected):
    prices, rank, compute = case(signal_module)
    rank.iloc[-1] = percentile
    assert compute(prices)["on"] is expected


@pytest.mark.parametrize("level,expected", [(13., False), (13.001, True)])
def test_vix_floor_is_strict(signal_module, level, expected):
    prices, _, compute = case(signal_module)
    prices.iloc[-1] = level
    assert compute(prices)["on"] is expected


@pytest.mark.parametrize("change,expected", [(-.01, True), (0., False), (.01, False)])
def test_five_observation_change_must_be_negative(signal_module, change, expected):
    prices, _, compute = case(signal_module)
    prices.iloc[-6] = prices.iloc[-1] - change
    result = compute(prices)
    assert result["on"] is expected
    assert result["vix_change5"].iloc[-1] == pytest.approx(change)


@pytest.mark.parametrize("interruption", [15., np.nan])
def test_compression_interruption_resets_age(signal_module, interruption):
    prices, rank, compute = case(signal_module, days=20)
    rank.iloc[-10] = interruption
    result = compute(prices)
    assert not result["on"]
    assert result["compression_age"].iloc[-10] == 0
    assert result["compression_age"].iloc[-1] == 9


def test_floor_and_direction_failures_do_not_reset_compression_age(signal_module):
    prices, _, compute = case(signal_module)
    prices.iloc[-8] = 12.
    prices.iloc[-2] = 30.
    result = compute(prices)
    assert not result["signal_history"].iloc[-8]
    assert not result["signal_history"].iloc[-2]
    assert result["compression_age"].iloc[-1] == 10
    assert result["on"]


def test_short_or_missing_data_cannot_fire(signal_module):
    compute = signal_module["compute_vix_range_compression"]
    for prices in [pd.Series(dtype=float), pd.Series(20., index=range(503)),
                   pd.Series(np.nan, index=range(540))]:
        result = compute(prices)
        assert not result["on"]
        assert not result["signal_history"].any()
        assert result["rule_version"] == "vix-compression-duration10-fall5-v1"
    prices = pd.Series(20. + np.sin(np.arange(800) / 20))
    prices.iloc[-5] = np.nan
    result = compute(prices)
    assert not result["on"]
    assert result["compression_age"].iloc[-1] == 0


def test_future_prices_do_not_change_past_features(signal_module):
    rng = np.random.default_rng(27)
    prices = pd.Series(18. + np.cumsum(rng.normal(0, .08, 950)),
                       index=pd.bdate_range("2020-01-02", periods=950))
    compute = signal_module["compute_vix_range_compression"]
    full, prefix = compute(prices), compute(prices.iloc[:800])
    for key in ["compression_pctile", "compression_age", "vix_change5", "signal_history"]:
        pd.testing.assert_series_equal(full[key].iloc[:800], prefix[key])


def test_summary_describes_all_trigger_conditions(signal_module):
    prices, _, compute = case(signal_module)
    result = compute(prices)
    assert "10" in result["summary"] and "15" in result["summary"]
    assert "13" in result["summary"] and "5d" in result["summary"]
    assert "violent" not in result["detail"]


def test_site_signal_detail_retains_current_rule_version(signal_module):
    from scripts.build_risk_json import _build_signal_detail, SIGNAL_METRICS
    prices, _, compute = case(signal_module)
    result = compute(prices)
    detail = _build_signal_detail({"VIX Range Compression": result}, prices.index,
                                  lambda history: [])
    assert detail["VIX Range Compression"]["rule_version"] == result["rule_version"]
    assert detail["VIX Range Compression"]["current"]["summary"] == result["summary"]
    assert SIGNAL_METRICS["VIX Range Compression"]["thresholds"][0]["label"] != "Fire"
