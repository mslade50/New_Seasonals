import numpy as np
import pandas as pd

from scripts.build_risk_json import build_similar_reading_drawdown_iv


def _sample_inputs():
    dates = pd.bdate_range("2026-01-02", periods=32)
    close = np.full(len(dates), 100.0)
    high = np.full(len(dates), 101.0)
    low = np.full(len(dates), 99.0)
    # Wilder ATR(14) is exactly 2.0 at the first analog anchor. Its realized
    # path then correctly feeds the rolling ATR observed at the later anchor.
    low[15:20] = [98.0, 96.0, 94.0, 97.0, 99.0]
    low[21:26] = [99.0, 98.0, 99.0, 99.0, 99.0]
    spy = pd.DataFrame({"High": high, "Low": low, "Close": close}, index=dates)
    vix_close = pd.Series(15.0, index=dates)
    vix_close.iloc[14] = 14.0
    vix_close.iloc[20] = 16.0
    vix_high = pd.Series(16.0, index=dates)
    vix_high.iloc[15:20] = [18.0, 23.0, 31.0, 25.0, 20.0]
    vix_high.iloc[21:26] = [17.0, 19.0, 18.0, 17.0, 17.0]
    similar = {
        "episode_dates": [dates[14], dates[20]],
        "n_episodes": 2,
        "current_score": 44.0,
        "band_low": 39.0,
        "band_high": 49.0,
    }
    return dates, spy, vix_close, vix_high, similar


def test_drawdown_iv_uses_only_similar_reading_anchors_and_atr_low_touch_logic():
    dates, spy, vix_close, vix_high, similar = _sample_inputs()

    result = build_similar_reading_drawdown_iv(
        spy, vix_close, similar, vix_high,
        horizons={"5d": 5}, thresholds=[1, 2, 3, 5],
    )

    assert result["sample_basis"].startswith("exact declustered anchors")
    assert result["current_score"] == 44
    assert result["iv_basis"] == "VIX intraday high"
    assert result["eligible_by_horizon"]["5d"] == 2
    assert len(result["rows_by_horizon"]["5d"]) == 2

    newest, oldest = result["rows_by_horizon"]["5d"]
    assert newest["anchor_date"] == dates[20].strftime("%Y-%m-%d")
    assert newest["max_drawdown_atr"] == 0.764
    assert oldest["anchor_date"] == dates[14].strftime("%Y-%m-%d")
    assert oldest["worst_low_date"] == dates[17].strftime("%Y-%m-%d")
    assert oldest["max_drawdown_atr"] == 3.0
    assert oldest["max_drawdown_pct"] == -0.06
    assert oldest["iv_start_close"] == 14
    assert oldest["iv_peak"] == 31
    assert oldest["iv_change_points"] == 17
    assert oldest["iv_peak_date"] == dates[17].strftime("%Y-%m-%d")
    assert result["counts"]["5d"] == {"1": 1, "2": 1, "3": 1, "5": 0}


def test_drawdown_iv_falls_back_to_vix_closes_and_excludes_incomplete_windows():
    dates, spy, vix_close, _, similar = _sample_inputs()
    similar["episode_dates"].append(dates[-2])
    similar["n_episodes"] = 3
    vix_close.iloc[15:18] = [18.0, 23.0, 29.0]

    result = build_similar_reading_drawdown_iv(
        spy, vix_close, similar, horizons={"5d": 5}, thresholds=[1, 2],
    )

    assert result["iv_basis"] == "VIX daily close (high unavailable)"
    assert result["n_episodes"] == 3
    assert result["eligible_by_horizon"]["5d"] == 2
    assert len(result["rows_by_horizon"]["5d"]) == 2
