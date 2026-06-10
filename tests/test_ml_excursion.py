"""Arithmetic correctness of MAE-in-R labels (synthetic prices)."""

import numpy as np
import pandas as pd

from ml.excursion import trade_mae_r

IDX = pd.date_range("2020-01-06", periods=5, freq="B")
BARS = pd.DataFrame({
    "Open":  [100, 99, 97, 98, 101],
    "High":  [101, 100, 98, 102, 104],
    "Low":   [99, 96, 95, 97, 100],
    "Close": [100, 97, 96, 101, 103],
}, index=IDX)


def test_long_mae():
    # entry 100 on day1, exit day4; worst low in window = 95; risk/share = 2.5
    mae = trade_mae_r(BARS, IDX[0], IDX[3], 100.0, 2.5, "Long")
    assert np.isclose(mae, (100 - 95) / 2.5)


def test_short_mae():
    # short entry 99 on day2, exit day5; max high in window = 104
    mae = trade_mae_r(BARS, IDX[1], IDX[4], 99.0, 2.0, "Short")
    assert np.isclose(mae, (104 - 99) / 2.0)


def test_mae_floor_at_zero():
    # long that never goes offside: entry below every low in window
    mae = trade_mae_r(BARS.assign(Low=BARS["Low"] + 50), IDX[0], IDX[2], 90.0, 2.0, "Long")
    assert mae == 0.0


def test_window_respects_exit_date():
    # exit on day3 -> day4 spike high (102/104) must not count for the short
    mae = trade_mae_r(BARS, IDX[0], IDX[2], 101.0, 1.0, "Short")
    assert np.isclose(mae, max(0.0, 101 - 101))


def test_bad_inputs_are_nan():
    assert np.isnan(trade_mae_r(BARS, IDX[0], IDX[3], 100.0, 0.0, "Long"))
    assert np.isnan(trade_mae_r(BARS, pd.NaT, IDX[3], 100.0, 2.0, "Long"))
    empty = trade_mae_r(BARS, IDX[4] + pd.Timedelta(days=5),
                        IDX[4] + pd.Timedelta(days=9), 100.0, 2.0, "Long")
    assert np.isnan(empty)
