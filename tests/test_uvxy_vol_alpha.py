import numpy as np
import pandas as pd

from research.uvxy_vol_alpha.backtest import (
    StrategyConfig,
    build_signals,
    forward_trade_return,
    remove_vrc_from_simple_dial,
    rolling_percentile,
    run_trades,
    vrc_activation,
)


def test_rolling_percentile_uses_only_prior_observations():
    index = pd.bdate_range("2024-01-02", periods=12)
    series = pd.Series(np.arange(12, dtype=float), index=index)

    ranked = rolling_percentile(series, lookback=5, min_fraction=1.0)
    assert ranked.iloc[4] != ranked.iloc[4]  # warm-up is unavailable
    assert ranked.iloc[5] == 100.0

    changed_future = series.copy()
    changed_future.iloc[9:] = -999.0
    reranked = rolling_percentile(changed_future, lookback=5, min_fraction=1.0)
    pd.testing.assert_series_equal(ranked.iloc[:9], reranked.iloc[:9])


def test_rolling_percentile_matches_dashboard_minimum_history_floor():
    index = pd.bdate_range("2024-01-02", periods=11)
    series = pd.Series(
        [*np.arange(7, dtype=float), np.nan, np.nan, np.nan, 10.0], index=index
    )

    ranked = rolling_percentile(series, lookback=10, min_fraction=0.80)
    assert ranked.iloc[-1] == 100.0


def test_remove_vrc_from_equal_weight_simple_dial():
    index = pd.bdate_range("2024-01-02", periods=3)
    six_signal_mean = pd.Series([0.40, 0.55, 0.70], index=index)
    vrc_weight = pd.Series([0.80, 0.25, 1.00], index=index)
    simple = (six_signal_mean * 6.0 + vrc_weight) / 7.0 * 100.0

    recovered = remove_vrc_from_simple_dial(simple, vrc_weight, signal_count=7)
    np.testing.assert_allclose(recovered, six_signal_mean * 100.0)


def test_remove_vrc_surfaces_source_vintage_mismatch_instead_of_clipping():
    index = pd.bdate_range("2024-01-02", periods=2)
    simple = pd.Series([0.0, 100.0], index=index)
    incompatible_vrc = pd.Series([1.0, 0.0], index=index)

    recovered = remove_vrc_from_simple_dial(simple, incompatible_vrc, signal_count=7)
    assert recovered.iloc[0] < 0.0
    assert recovered.iloc[1] > 100.0


def test_vrc_activation_requires_five_prior_off_sessions():
    index = pd.bdate_range("2024-01-02", periods=16)
    on = pd.Series(
        [False] * 5 + [True] + [False] * 2 + [True] + [False] * 5 + [True, False],
        index=index,
    )
    activation = vrc_activation(on, off_sessions=5)

    assert activation.iloc[5]
    assert not activation.iloc[8]
    assert activation.iloc[14]


def test_primary_signal_uses_prior_day_fragility():
    index = pd.bdate_range("2024-01-02", periods=12)
    features = pd.DataFrame(
        {
            "compression_pctile": [50.0] * 6 + [5.0] + [50.0] * 5,
            "vix": [15.0] * 12,
            "vix_sma": [14.0] * 12,
            "fragility_rank": [10.0] * 5 + [80.0, 1.0] + [10.0] * 5,
            "incumbent_rank": [10.0] * 5 + [80.0, 1.0] + [10.0] * 5,
            "incumbent_ma10": [10.0] * 5 + [60.0, 1.0] + [10.0] * 5,
        },
        index=index,
    )
    config = StrategyConfig(
        percentile_lookback=5,
        rearm_off_sessions=5,
        fragility_rank_threshold=67.0,
    )

    signals = build_signals(features, config)
    assert signals["primary"].iloc[6]

    # Same-close fragility is deliberately irrelevant; the prior close owns the gate.
    mutated = features.copy()
    mutated.loc[index[6], "fragility_rank"] = 99.0
    mutated_signals = build_signals(mutated, config)
    assert mutated_signals["primary"].iloc[6]


def test_forward_return_is_next_open_to_fifth_close_with_costs():
    index = pd.bdate_range("2024-01-02", periods=10)
    features = pd.DataFrame(
        {
            "uvxy_open": np.arange(100.0, 110.0),
            "uvxy_close": np.arange(100.5, 110.5),
        },
        index=index,
    )
    record = forward_trade_return(
        features,
        signal_position=1,
        hold_sessions=5,
        round_trip_cost_bps=12.0,
    )

    assert record is not None
    assert record["entry_date"] == index[2]
    assert record["exit_date"] == index[6]
    expected = (106.5 * (1 - 0.0006)) / (102.0 * (1 + 0.0006)) - 1.0
    assert abs(record["net_return"] - expected) < 1e-12


def test_run_trades_refuses_overlapping_positions():
    index = pd.bdate_range("2024-01-02", periods=14)
    features = pd.DataFrame(
        {
            "uvxy_open": np.linspace(100.0, 113.0, len(index)),
            "uvxy_close": np.linspace(100.5, 113.5, len(index)),
            "vix": 15.0,
            "vix3m": 17.0,
            "compression_pctile": 5.0,
            "fragility_ma10": 70.0,
            "fragility_rank": 80.0,
            "incumbent_ma10": 70.0,
            "incumbent_rank": 80.0,
            "term_ratio": 15 / 17,
            "term_rank": 40.0,
        },
        index=index,
    )
    signal = pd.Series(False, index=index)
    signal.iloc[[1, 3, 8]] = True
    config = StrategyConfig(start_date="2024-01-01", hold_sessions=5)

    trades = run_trades(signal, features, config, label="test")
    assert trades["signal_date"].tolist() == [index[1], index[8]]
