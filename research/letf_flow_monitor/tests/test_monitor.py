from __future__ import annotations

import numpy as np
import pandas as pd

from research.letf_flow_monitor.monitor import (
    first_episode_event,
    prior_quantile,
    split_adjusted_share_flow,
    trailing_zscore,
)


def test_split_adjustment_does_not_create_phantom_flow():
    shares = pd.Series([100.0, 10.0, 11.0])
    nav = pd.Series([10.0, 101.0, 103.0])
    prior_nav = pd.Series([np.nan, 100.0, 101.0])
    result = split_adjusted_share_flow(shares, nav, prior_nav)
    assert result.loc[1, "split_detected"]
    assert result.loc[1, "delta_shares_k"] == 0.0
    assert result.loc[2, "delta_shares_k"] == 1.0


def test_ordinary_nav_move_does_not_rescale_previous_shares():
    shares = pd.Series([100.0, 105.0])
    nav = pd.Series([10.0, 10.5])
    prior_nav = pd.Series([np.nan, 10.0])
    result = split_adjusted_share_flow(shares, nav, prior_nav)
    assert not result.loc[1, "split_detected"]
    assert result.loc[1, "delta_shares_k"] == 5.0


def test_point_in_time_quantile_excludes_current_observation():
    x = pd.Series(list(range(130)) + [10_000.0])
    threshold = prior_quantile(x, 0.90, window=130)
    assert threshold.iloc[-1] < 200


def test_point_in_time_zscore_excludes_current_observation():
    x = pd.Series(list(range(130)) + [10_000.0])
    z = trailing_zscore(x, window=130)
    expected = (10_000 - np.mean(range(130))) / np.std(range(130), ddof=1)
    assert np.isclose(z.iloc[-1], expected)


def test_episode_filter_keeps_only_first_event_after_quiet_window():
    frame = pd.DataFrame(
        {"benchmark": ["A"] * 9, "date": pd.date_range("2020-01-01", periods=9)}
    )
    raw = pd.Series([False, True, True, False, False, False, False, False, True])
    result = first_episode_event(frame, raw, cooldown=5)
    assert result.tolist() == [False, True, False, False, False, False, False, False, True]
