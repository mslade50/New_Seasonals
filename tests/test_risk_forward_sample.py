import pandas as pd

from pages.risk_dashboard_v2 import compute_similar_reading_returns


def test_four_matches_withhold_returns_but_preserve_completed_counts():
    dates = pd.bdate_range("2025-01-01", periods=100)
    frag = pd.Series(0.0, index=dates)
    frag.iloc[[0, 12, 80, 99]] = 85.0
    price = pd.Series(range(100, 200), index=dates, dtype=float)
    result = compute_similar_reading_returns(frag, price, 85)
    assert result["status"] == "insufficient_sample"
    assert result["min_samples"] == 5
    assert result["n_episodes"] == 4
    assert result["sample_counts"] == {5: 3, 10: 3, 21: 2, 42: 2, 63: 2}
    assert all(value is None for value in result["returns"].values())


def test_five_completed_matches_still_produce_statistics():
    dates = pd.bdate_range("2025-01-01", periods=130)
    frag = pd.Series(0.0, index=dates)
    frag.iloc[[0, 12, 24, 36, 48]] = 85.0
    result = compute_similar_reading_returns(frag, pd.Series(range(100, 230), index=dates), 85)
    assert result["status"] == "ok"
    assert all(value == 5 for value in result["sample_counts"].values())
    assert all(stats["n"] == 5 for stats in result["returns"].values())
