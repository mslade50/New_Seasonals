"""
tests/test_indicators.py — Core test suite for the trading system
=================================================================
Covers:
    1. Indicator calculation correctness (indicators.py)
    2. Signal logic ("given this state, does the strategy fire?")
    3. Edge cases that have caused production bugs

Run with: pytest tests/ -v
Run quick: pytest tests/ -v -x  (stop on first failure)

NOTE: These tests use synthetic data. No yfinance calls, no network needed.
      They run in < 2 seconds total.
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add repo root to path so we can import indicators
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indicators import calculate_indicators, apply_first_instance_filter, get_sznl_val_series


# =============================================================================
# FIXTURES: Synthetic price data generators
# =============================================================================

def make_ohlcv(
    n_days: int = 300,
    start_price: float = 100.0,
    volatility: float = 0.02,
    start_date: str = "2022-01-03",
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data that looks realistic enough for indicator tests.
    Deterministic via seed for reproducibility.
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range(start=start_date, periods=n_days, freq='B')

    closes = [start_price]
    for _ in range(n_days - 1):
        ret = rng.normal(0.0003, volatility)
        closes.append(closes[-1] * (1 + ret))
    closes = np.array(closes)

    # Generate realistic OHLC from close
    highs = closes * (1 + rng.uniform(0.001, 0.03, n_days))
    lows = closes * (1 - rng.uniform(0.001, 0.03, n_days))
    opens = closes * (1 + rng.uniform(-0.01, 0.01, n_days))
    volumes = rng.randint(500_000, 5_000_000, n_days).astype(float)

    df = pd.DataFrame({
        'Open': opens, 'High': highs, 'Low': lows,
        'Close': closes, 'Volume': volumes,
    }, index=dates)
    return df


def make_empty_sznl_map() -> dict:
    """Empty seasonal map — returns 50 for everything."""
    return {}


def make_simple_sznl_map(ticker: str, value: float = 75.0, start_date: str = "2020-01-01", periods: int = 2000) -> dict:
    """
    Seasonal map that returns a fixed value for a range of dates.
    Matches production format: {ticker: pd.Series} indexed by date.
    """
    dates = pd.bdate_range(start=start_date, periods=periods, freq='B')
    series = pd.Series(value, index=dates, dtype=float)
    return {ticker: series}


def make_partial_sznl_map(ticker: str, start_date: str = "2024-06-01", periods: int = 10, value: float = 85.0) -> dict:
    """
    Seasonal map with only a few dates populated.
    Tests that ffill works correctly for dates between entries.
    """
    dates = pd.bdate_range(start=start_date, periods=periods, freq='B')
    series = pd.Series(value, index=dates, dtype=float)
    return {ticker: series}


@pytest.fixture
def sample_df():
    """300 trading days of synthetic data."""
    return make_ohlcv(n_days=300, seed=42)


@pytest.fixture
def long_df():
    """500 trading days — enough for 252-period indicators to warm up."""
    return make_ohlcv(n_days=500, seed=99)


@pytest.fixture
def sznl_map():
    return make_empty_sznl_map()


# =============================================================================
# TEST GROUP 1: Indicator Calculation Correctness
# =============================================================================

class TestIndicatorColumns:
    """Verify that calculate_indicators produces all expected columns."""

    def test_core_columns_present(self, long_df, sznl_map):
        """Every consumer expects these columns. If any are missing, something broke."""
        result = calculate_indicators(long_df, sznl_map, "AAPL")

        required_columns = [
            # MAs
            'SMA10', 'SMA20', 'SMA50', 'SMA100', 'SMA200',
            'EMA8', 'EMA11', 'EMA21',
            # Perf ranks
            'ret_2d', 'ret_5d', 'ret_10d', 'ret_21d',
            'rank_ret_2d', 'rank_ret_5d', 'rank_ret_10d', 'rank_ret_21d',
            # ATR
            'ATR', 'ATR_Pct', 'today_return_atr',
            # Volume
            'vol_ma', 'vol_ratio', 'Vol_Spike',
            'AccCount_21', 'AccCount_5', 'AccCount_10', 'AccCount_42',
            'DistCount_21', 'DistCount_5', 'DistCount_10', 'DistCount_42',
            'vol_ratio_10d', 'vol_ratio_10d_rank',
            # Candle
            'RangePct',
            # Gaps
            'GapCount_21', 'GapCount_10', 'GapCount_5', 'GapCount',
            # Seasonal
            'Sznl', 'Mkt_Sznl_Ref',
            # Age
            'age_years',
            # 52w / ATH
            'is_52w_high', 'is_52w_low', 'High_52w',
            'is_ath', 'prior_ath', 'ATH_Level',
            # VIX
            'VIX_Value',
            # Convenience
            'DayOfWeekVal', 'PrevHigh', 'PrevLow', 'NextOpen',
            # Pivot
            'is_pivot_high', 'is_pivot_low', 'LastPivotHigh', 'LastPivotLow',
        ]
        missing = [c for c in required_columns if c not in result.columns]
        assert missing == [], f"Missing columns: {missing}"

    def test_custom_sma_added(self, long_df, sznl_map):
        """Backtester can request arbitrary SMA lengths."""
        result = calculate_indicators(long_df, sznl_map, "AAPL", custom_sma_lengths=[150, 30])
        assert 'SMA150' in result.columns
        assert 'SMA30' in result.columns

    def test_custom_sma_no_duplicate(self, long_df, sznl_map):
        """Requesting SMA200 as custom shouldn't create a second column or error."""
        result = calculate_indicators(long_df, sznl_map, "AAPL", custom_sma_lengths=[200])
        # Should still have exactly one SMA200
        assert 'SMA200' in result.columns
        sma200_cols = [c for c in result.columns if c == 'SMA200']
        assert len(sma200_cols) == 1


class TestIndicatorValues:
    """Verify indicator calculations produce correct values."""

    def test_sma200_is_200_period_mean(self, long_df, sznl_map):
        """SMA200 on row 250 should equal the rolling mean of the 200 preceding Close values."""
        result = calculate_indicators(long_df, sznl_map, "AAPL")
        idx = 250
        expected = result['Close'].iloc[idx - 199:idx + 1].mean()
        actual = result['SMA200'].iloc[idx]
        assert abs(actual - expected) < 0.01, f"SMA200 mismatch: {actual} vs {expected}"

    def test_atr_is_14_period(self, long_df, sznl_map):
        """ATR should be NaN for first 13 rows, then valid."""
        result = calculate_indicators(long_df, sznl_map, "AAPL")
        assert pd.isna(result['ATR'].iloc[12])  # Not enough data yet
        assert not pd.isna(result['ATR'].iloc[14])  # Should have value

    def test_rank_min_periods_252(self, sample_df, sznl_map):
        """
        CRITICAL: This test catches the old daily_scan bug where min_periods=50
        would produce ranks on tickers with < 1 year of data.
        ret_2d is NaN for first 2 rows, so 252 non-NaN values are available
        at row index 253. Rows before that must be NaN.
        """
        result = calculate_indicators(sample_df, sznl_map, "AAPL")
        # Row 250: only 249 non-NaN values available — should still be NaN
        assert pd.isna(result['rank_ret_2d'].iloc[250]), \
            "rank_ret_2d should be NaN before 252 non-NaN periods (was the min_periods bug reintroduced?)"
        # Row 253: exactly 252 non-NaN values — should have a rank
        assert not pd.isna(result['rank_ret_2d'].iloc[253])

    def test_range_pct_bounds(self, long_df, sznl_map):
        """RangePct should always be between 0 and 1 (inclusive)."""
        result = calculate_indicators(long_df, sznl_map, "AAPL")
        valid = result['RangePct'].dropna()
        assert (valid >= 0).all() and (valid <= 1).all()

    def test_ath_uses_gte(self, sznl_map):
        """
        CRITICAL: ATH should fire on exact match (>=), not just strict new high (>).
        This was a divergence between daily_scan and backtester.
        """
        df = make_ohlcv(n_days=100, seed=1)
        # Force an exact ATH tie: set today's high equal to the prior max
        prior_max = df['High'].iloc[:50].max()
        df.iloc[55, df.columns.get_loc('High')] = prior_max
        # Make sure nothing between 50 and 55 exceeded it
        df.iloc[50:55, df.columns.get_loc('High')] = prior_max - 1

        result = calculate_indicators(df, sznl_map, "TEST")
        assert result['is_ath'].iloc[55] == True, \
            "is_ath should be True on exact ATH match (>= not >)"

    def test_vol_spike_logic(self, sznl_map):
        """Vol Spike = volume above 63d MA AND above previous day's volume."""
        df = make_ohlcv(n_days=100, seed=7)
        result = calculate_indicators(df, sznl_map, "TEST")

        # Manually check a row where Vol_Spike is True
        spikes = result[result['Vol_Spike'] == True]
        if len(spikes) > 0:
            row_idx = spikes.index[0]
            loc = result.index.get_loc(row_idx)
            assert result['Volume'].iloc[loc] > result['vol_ma'].iloc[loc]
            assert result['Volume'].iloc[loc] > result['Volume'].iloc[loc - 1]


# =============================================================================
# TEST GROUP 2: Edge Cases & Defensive Handling
# =============================================================================

class TestEdgeCases:
    """Tests for the specific bugs that have bitten us in production."""

    def test_multiindex_columns_handled(self, sznl_map):
        """
        CRITICAL: yfinance sometimes returns MultiIndex columns.
        The function must flatten them silently — not crash on .capitalize().
        """
        df = make_ohlcv(n_days=100, seed=3)
        # Simulate yfinance MultiIndex: ('Close', 'AAPL'), etc.
        mi_cols = pd.MultiIndex.from_tuples(
            [(c, 'AAPL') for c in df.columns]
        )
        df.columns = mi_cols

        # This should NOT raise
        result = calculate_indicators(df, sznl_map, "AAPL")
        assert 'Close' in result.columns
        assert 'ATR' in result.columns

    def test_tz_aware_index_handled(self, sznl_map):
        """yfinance can return tz-aware index. Must be stripped."""
        df = make_ohlcv(n_days=100, seed=4)
        df.index = df.index.tz_localize('US/Eastern')

        result = calculate_indicators(df, sznl_map, "TEST")
        assert result.index.tz is None

    def test_lowercase_columns_handled(self, sznl_map):
        """Some data sources return lowercase column names."""
        df = make_ohlcv(n_days=100, seed=5)
        df.columns = [c.lower() for c in df.columns]

        result = calculate_indicators(df, sznl_map, "TEST")
        assert 'Close' in result.columns

    def test_short_dataframe_no_crash(self, sznl_map):
        """A ticker with only 20 rows shouldn't crash, just produce NaN indicators."""
        df = make_ohlcv(n_days=20, seed=6)
        result = calculate_indicators(df, sznl_map, "TINY")
        assert len(result) == 20
        # SMA200 should be all NaN (not enough data)
        assert result['SMA200'].isna().all()

    def test_empty_dataframe_no_crash(self, sznl_map):
        """Empty df should return empty df, not crash."""
        df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        df.index = pd.DatetimeIndex([])
        result = calculate_indicators(df, sznl_map, "EMPTY")
        assert len(result) == 0


# =============================================================================
# TEST GROUP 3: Signal Logic Primitives
# =============================================================================

class TestFirstInstanceFilter:
    """Test the apply_first_instance_filter helper."""

    def test_first_instance_blocks_repeat(self):
        """If signal fires on day 5 and 6, lookback=5 should block day 6."""
        signals = pd.Series([False]*5 + [True, True] + [False]*3)
        filtered = apply_first_instance_filter(signals, lookback=5)
        assert filtered.iloc[5] == True   # First fire — allowed
        assert filtered.iloc[6] == False  # Within lookback — blocked

    def test_first_instance_allows_after_gap(self):
        """Signal on day 5, then day 15 with lookback=5 — day 15 should fire."""
        signals = pd.Series([False]*5 + [True] + [False]*9 + [True] + [False]*5)
        filtered = apply_first_instance_filter(signals, lookback=5)
        assert filtered.iloc[5] == True
        assert filtered.iloc[15] == True  # Gap > lookback, so allowed

    def test_lookback_1_passes_everything(self):
        """lookback <= 1 means no filtering."""
        signals = pd.Series([True, True, True, False, True])
        filtered = apply_first_instance_filter(signals, lookback=1)
        assert filtered.tolist() == signals.tolist()


class TestSeasonalLookup:
    """Test the seasonal map helper — uses date-indexed pd.Series format."""

    def test_empty_map_returns_50(self):
        dates = pd.bdate_range("2024-01-02", periods=5)
        result = get_sznl_val_series("AAPL", dates, {})
        assert (result == 50.0).all()

    def test_fixed_value_map(self):
        """Map with known dates should return the stored value."""
        smap = make_simple_sznl_map("AAPL", value=80.0, start_date="2024-01-01", periods=500)
        dates = pd.bdate_range("2024-06-03", periods=5)
        result = get_sznl_val_series("AAPL", dates, smap)
        assert (result == 80.0).all()

    def test_unknown_ticker_returns_50(self):
        smap = make_simple_sznl_map("AAPL", value=80.0)
        dates = pd.bdate_range("2024-01-02", periods=5)
        result = get_sznl_val_series("MSFT", dates, smap)
        assert (result == 50.0).all()

    def test_ffill_covers_gaps(self):
        """Dates not in the map should get forward-filled from the last known value."""
        smap = make_partial_sznl_map("AAPL", start_date="2024-06-03", periods=5, value=85.0)
        # Query dates that extend past the map's coverage — ffill should carry forward
        dates = pd.bdate_range("2024-06-03", periods=10)
        result = get_sznl_val_series("AAPL", dates, smap)
        # First 5 dates are exact matches, next 5 should be ffilled to 85.0
        assert (result == 85.0).all()

    def test_dates_before_map_get_default(self):
        """Dates before the map's first entry should get 50 (fillna default)."""
        smap = make_simple_sznl_map("AAPL", value=90.0, start_date="2024-06-01", periods=100)
        dates = pd.bdate_range("2024-01-02", periods=5)
        result = get_sznl_val_series("AAPL", dates, smap)
        # These dates are before the map starts — ffill has nothing to fill from
        assert (result == 50.0).all()

    def test_seasonal_values_in_calculate_indicators(self):
        """End-to-end: seasonal values should flow through calculate_indicators correctly."""
        df = make_ohlcv(n_days=100, start_date="2024-06-03", seed=42)
        smap = make_simple_sznl_map("TEST", value=72.0, start_date="2024-01-01", periods=500)
        result = calculate_indicators(df, smap, "TEST")
        # All Sznl values should be 72.0 (not the default 50.0)
        assert (result['Sznl'] == 72.0).all(), \
            f"Sznl should be 72.0 but got unique values: {result['Sznl'].unique()}"


# =============================================================================
# TEST GROUP 4: Parity Checks
# =============================================================================

class TestParityGuards:
    """
    Tests that ensure backtest and production indicator outputs are identical.
    These would catch re-introduction of the divergences we fixed.
    """

    def test_acc_dist_standard_windows_always_present(self, long_df, sznl_map):
        """
        daily_scan needs AccCount_5/10/21/42 regardless of acc_window param.
        Even when backtester passes acc_window=10, the other windows must exist.
        """
        result = calculate_indicators(long_df, sznl_map, "AAPL", acc_window=10, dist_window=10)
        for w in [5, 10, 21, 42]:
            assert f'AccCount_{w}' in result.columns, f"AccCount_{w} missing"
            assert f'DistCount_{w}' in result.columns, f"DistCount_{w} missing"

    def test_gap_count_standard_windows_always_present(self, long_df, sznl_map):
        """Gap counts for 5/10/21 must always exist, plus the configurable GapCount."""
        result = calculate_indicators(long_df, sznl_map, "AAPL", gap_window=15)
        assert 'GapCount_5' in result.columns
        assert 'GapCount_10' in result.columns
        assert 'GapCount_21' in result.columns
        assert 'GapCount' in result.columns
        assert not result['GapCount'].isna().all()

    def test_vix_default_zero_when_not_provided(self, long_df, sznl_map):
        """When no vix_series is passed, VIX_Value should be 0, not missing."""
        result = calculate_indicators(long_df, sznl_map, "AAPL")
        assert (result['VIX_Value'] == 0).all()

    def test_market_regime_missing_when_no_series(self, long_df, sznl_map):
        """If market_series is None, Market_Above_SMA200 should not be in columns."""
        result = calculate_indicators(long_df, sznl_map, "AAPL", market_series=None)
        assert 'Market_Above_SMA200' not in result.columns


# =============================================================================
# TEST GROUP 5: Pivot Columns
# =============================================================================

class TestPivotColumns:
    """Verify pivot high/low columns are computed correctly."""

    def test_pivot_columns_exist(self, long_df, sznl_map):
        result = calculate_indicators(long_df, sznl_map, "AAPL")
        for col in ['is_pivot_high', 'is_pivot_low', 'LastPivotHigh', 'LastPivotLow']:
            assert col in result.columns, f"Missing pivot column: {col}"

    def test_last_pivot_high_valid(self, long_df, sznl_map):
        """LastPivotHigh should be NaN initially, then forward-filled with valid prices."""
        result = calculate_indicators(long_df, sznl_map, "AAPL")
        valid = result['LastPivotHigh'].dropna()
        if len(valid) > 0:
            assert (valid > 0).all(), "LastPivotHigh should be positive prices"

    def test_last_pivot_low_valid(self, long_df, sznl_map):
        """LastPivotLow should be NaN initially, then forward-filled with valid prices."""
        result = calculate_indicators(long_df, sznl_map, "AAPL")
        valid = result['LastPivotLow'].dropna()
        if len(valid) > 0:
            assert (valid > 0).all(), "LastPivotLow should be positive prices"


# =============================================================================
# TEST GROUP 6: get_sznl_val_series Enhancements
# =============================================================================

class TestSznlEnhancements:
    """Test the .upper(), ^GSPC fallback, and dict-of-dicts compatibility."""

    def test_uppercase_lookup(self):
        """Passing lowercase ticker should match uppercase key in map."""
        smap = make_simple_sznl_map("AAPL", value=80.0, start_date="2024-01-01", periods=500)
        dates = pd.bdate_range("2024-06-03", periods=5)
        result = get_sznl_val_series("aapl", dates, smap)
        assert (result == 80.0).all(), "Lowercase ticker should match uppercase key"

    def test_gspc_spy_fallback(self):
        """^GSPC should fall back to SPY if ^GSPC key doesn't exist."""
        smap = make_simple_sznl_map("SPY", value=70.0, start_date="2024-01-01", periods=500)
        dates = pd.bdate_range("2024-06-03", periods=5)
        result = get_sznl_val_series("^GSPC", dates, smap)
        assert (result == 70.0).all(), "^GSPC should fall back to SPY"

    def test_dict_of_dicts_format(self):
        """Backtester's load_seasonal_map can return {ticker: dict} format."""
        dates = pd.bdate_range("2024-06-03", periods=5)
        # Simulate backtester format: {ticker: {date: value}}
        inner_dict = {d: 85.0 for d in dates}
        smap = {"AAPL": inner_dict}
        result = get_sznl_val_series("AAPL", dates, smap)
        assert (result == 85.0).all(), "dict-of-dicts format should work"


# =============================================================================
# TEST GROUP 7: Index Normalization
# =============================================================================

class TestIndexNormalization:
    """Verify that calculate_indicators normalizes the datetime index."""

    def test_non_normalized_index_gets_normalized(self, sznl_map):
        """Pass a non-normalized index (with time component), verify output is normalized."""
        df = make_ohlcv(n_days=100, seed=10)
        # Add time component to simulate non-normalized timestamps
        df.index = df.index + pd.Timedelta(hours=16)
        assert df.index[0].hour == 16, "Precondition: index should have time component"

        result = calculate_indicators(df, sznl_map, "TEST")
        assert result.index[0].hour == 0, "Index should be normalized (midnight)"
        assert result.index[0].minute == 0


# =============================================================================
# TEST GROUP 8: Cross-Consumer Parity
# =============================================================================

class TestCrossConsumerParity:
    """
    Verify that calling calculate_indicators with backtester-style args
    vs daily_scan-style args produces identical shared columns.
    """

    def test_shared_columns_identical(self):
        """Core indicator columns must be identical regardless of caller arguments."""
        df = make_ohlcv(n_days=500, seed=77)
        smap = make_empty_sznl_map()

        # Backtester-style call (custom windows, market_sznl_series)
        market_sznl = pd.Series(60.0, index=df.index)
        bt_result = calculate_indicators(
            df.copy(), smap, "AAPL",
            market_sznl_series=market_sznl,
            gap_window=15,
            acc_window=30,
            dist_window=30,
            custom_sma_lengths=[150],
        )

        # Daily-scan-style call (defaults only, with ref_ticker_ranks)
        ref_ranks = {5: pd.Series(50.0, index=df.index)}
        ds_result = calculate_indicators(
            df.copy(), smap, "AAPL",
            ref_ticker_ranks=ref_ranks,
        )

        # Shared columns that MUST be identical
        shared_cols = [
            'SMA10', 'SMA20', 'SMA50', 'SMA100', 'SMA200',
            'EMA8', 'EMA11', 'EMA21',
            'ret_2d', 'ret_5d', 'ret_10d', 'ret_21d',
            'rank_ret_2d', 'rank_ret_5d', 'rank_ret_10d', 'rank_ret_21d',
            'ATR', 'ATR_Pct', 'today_return_atr',
            'vol_ma', 'vol_ratio', 'Vol_Spike',
            'is_acc_day', 'is_dist_day',
            'AccCount_5', 'AccCount_10', 'AccCount_21', 'AccCount_42',
            'DistCount_5', 'DistCount_10', 'DistCount_21', 'DistCount_42',
            'vol_ratio_10d', 'vol_ratio_10d_rank',
            'RangePct',
            'GapCount_5', 'GapCount_10', 'GapCount_21',
            'Sznl', 'Mkt_Sznl_Ref',
            'age_years',
            'is_52w_high', 'is_52w_low', 'High_52w',
            'is_ath', 'prior_ath', 'ATH_Level',
            'VIX_Value',
            'DayOfWeekVal', 'PrevHigh', 'PrevLow', 'NextOpen',
            'is_pivot_high', 'is_pivot_low', 'LastPivotHigh', 'LastPivotLow',
        ]

        for col in shared_cols:
            assert col in bt_result.columns, f"Backtester missing: {col}"
            assert col in ds_result.columns, f"Daily-scan missing: {col}"
            bt_vals = bt_result[col].fillna(-999)
            ds_vals = ds_result[col].fillna(-999)
            if bt_vals.dtype == float:
                assert np.allclose(bt_vals.values, ds_vals.values, equal_nan=True), \
                    f"Column {col} differs between backtester and daily_scan calls"
            else:
                assert (bt_vals == ds_vals).all(), \
                    f"Column {col} differs between backtester and daily_scan calls"

    def test_backtester_only_columns_present(self):
        """Backtester-specific columns should exist when custom args are passed."""
        df = make_ohlcv(n_days=500, seed=77)
        smap = make_empty_sznl_map()
        market_sznl = pd.Series(60.0, index=df.index)

        result = calculate_indicators(
            df.copy(), smap, "AAPL",
            market_sznl_series=market_sznl,
            gap_window=15,
            acc_window=30,
            dist_window=30,
            custom_sma_lengths=[150],
        )

        assert 'Market_Sznl' in result.columns, "Market_Sznl should be present"
        assert 'SMA150' in result.columns, "Custom SMA150 should be present"
        assert 'AccCount_30' in result.columns, "Custom AccCount_30 should be present"
        assert 'DistCount_30' in result.columns, "Custom DistCount_30 should be present"

    def test_daily_scan_ref_ticker_columns_present(self):
        """Daily-scan specific ref ticker columns should exist when ref_ticker_ranks passed."""
        df = make_ohlcv(n_days=500, seed=77)
        smap = make_empty_sznl_map()
        ref_ranks = {5: pd.Series(50.0, index=df.index), 10: pd.Series(60.0, index=df.index)}

        result = calculate_indicators(
            df.copy(), smap, "AAPL",
            ref_ticker_ranks=ref_ranks,
        )

        assert 'Ref_rank_ret_5d' in result.columns
        assert 'Ref_rank_ret_10d' in result.columns
