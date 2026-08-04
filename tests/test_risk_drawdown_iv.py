import pandas as pd

from scripts.build_risk_json import build_drawdown_iv_episodes


def test_drawdown_iv_episodes_use_peak_close_and_intraday_vix_high():
    dates = pd.bdate_range("2026-01-02", periods=10)
    spy = pd.Series([100, 98, 94, 90, 95, 100, 101, 95, 80, 102], index=dates)
    vix_close = pd.Series([15, 18, 24, 28, 20, 16, 14, 22, 35, 15], index=dates)
    vix_high = pd.Series([16, 20, 27, 31, 22, 18, 15, 25, 41, 17], index=dates)

    result = build_drawdown_iv_episodes(spy, vix_close, vix_high)

    assert result["iv_basis"] == "VIX intraday high"
    assert len(result["episodes"]) == 2

    newest, oldest = result["episodes"]
    assert newest["peak_date"] == dates[6].strftime("%Y-%m-%d")
    assert newest["trough_date"] == dates[8].strftime("%Y-%m-%d")
    assert newest["max_drawdown"] == round(80 / 101 - 1, 5)
    assert newest["iv_start_close"] == 14
    assert newest["iv_peak"] == 41
    assert newest["iv_change_points"] == 27
    assert newest["iv_peak_date"] == dates[8].strftime("%Y-%m-%d")

    assert oldest["max_drawdown"] == -0.10
    assert oldest["iv_start_close"] == 15
    assert oldest["iv_peak"] == 31
    assert oldest["iv_change_points"] == 16
    assert result["counts"]["0.05"] == 2
    assert result["counts"]["0.1"] == 2
    assert result["counts"]["0.15"] == 1


def test_drawdown_iv_includes_unrecovered_episode_and_falls_back_to_closes():
    dates = pd.bdate_range("2026-02-02", periods=5)
    spy = pd.Series([100, 99, 94, 91, 92], index=dates)
    vix = pd.Series([12, 14, 20, 29, 25], index=dates)

    result = build_drawdown_iv_episodes(spy, vix)

    assert len(result["episodes"]) == 1
    episode = result["episodes"][0]
    assert episode["recovery_date"] is None
    assert episode["iv_peak"] == 29
    assert result["iv_basis"] == "VIX daily close (high unavailable)"
