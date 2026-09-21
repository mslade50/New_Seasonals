import numpy as np
import pandas as pd

from scripts.analyze_legend_ema_nq_short_filters import (
    build_roll_neutral_daily,
    filter_masks,
)


def _daily_fixture(rows: list[dict[str, object]]) -> pd.DataFrame:
    dates = pd.date_range("2026-01-02", periods=len(rows), freq="B")
    frame = pd.DataFrame(rows, index=dates)
    frame["minute_count"] = 390
    frame["instrument_count"] = 1
    frame["first_ts"] = [
        pd.Timestamp(date).tz_localize("America/New_York") + pd.Timedelta(hours=9, minutes=30)
        for date in dates
    ]
    return frame


def test_roll_neutral_series_removes_additive_contract_splice():
    daily = _daily_fixture(
        [
            {"open": 99.0, "close": 100.0, "instrument_id": 1},
            {"open": 101.0, "close": 102.0, "instrument_id": 1},
            {"open": 200.0, "close": 203.0, "instrument_id": 2},
            {"open": 203.0, "close": 202.0, "instrument_id": 2},
        ]
    )

    result = build_roll_neutral_daily(daily, trusted_start="2026-01-01")

    assert result["additive_close"].tolist() == [100.0, 102.0, 105.0, 104.0]
    assert np.allclose(
        result["ratio_close"].tolist(),
        [100.0, 102.0, 102.0 * 203.0 / 200.0, 102.0 * 203.0 / 200.0 * 202.0 / 203.0],
    )
    assert result["contract_changed"].tolist() == [True, False, True, False]


def test_moving_averages_are_point_in_time_and_respect_warmups():
    rows = [
        {"open": float(value), "close": float(value), "instrument_id": 1}
        for value in range(100, 320)
    ]
    daily = _daily_fixture(rows)

    result = build_roll_neutral_daily(daily, trusted_start="2026-01-01")

    assert result["additive_ema21"].iloc[:20].isna().all()
    assert result["additive_sma50"].iloc[:49].isna().all()
    assert result["additive_sma200"].iloc[:199].isna().all()
    assert result["additive_sma50"].iloc[49] == np.mean(range(100, 150))
    assert result["additive_sma200"].iloc[199] == np.mean(range(100, 300))


def test_filter_branches_are_complements_when_indicator_exists():
    rows = [
        {"open": float(value), "close": float(value), "instrument_id": 1}
        for value in range(100, 320)
    ]
    trend = build_roll_neutral_daily(_daily_fixture(rows), trusted_start="2026-01-01")

    masks = filter_masks(trend, "additive")
    ready = trend["additive_sma200"].notna()

    assert (masks["above_sma200"].loc[ready] ^ masks["below_sma200"].loc[ready]).all()
    assert masks["above_all_three"].loc[ready].all()
    assert not masks["below_all_three"].loc[ready].any()
