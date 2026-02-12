"""
Test suite for abs_return_dispersion.py
Simulates yfinance MultiIndex behavior and validates calculation correctness.

Run: python test_dispersion.py
"""
import pandas as pd
import numpy as np
import sys

# Import the module under test
from abs_return_dispersion import (
    compute_dispersion_series,
    compute_dispersion_from_dict,
    _extract_close_from_yf_download,
    _clean_single_ticker_df,
)

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✅ {name}")
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}")


def make_dates(n=500):
    return pd.bdate_range("2020-01-01", periods=n, freq="B")


def make_price_series(dates, start=100, drift=0.0003, vol=0.02, seed=None):
    """Generate a realistic random-walk price series."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    returns = drift + vol * rng.randn(len(dates))
    prices = start * np.exp(np.cumsum(returns))
    return pd.Series(prices, index=dates)


# ============================================================================
# TEST 1: yfinance MultiIndex handling — _extract_close_from_yf_download
# ============================================================================
print("\n" + "=" * 60)
print("TEST 1: yfinance MultiIndex extraction")
print("=" * 60)

dates = make_dates(100)

# Simulate yfinance multi-ticker download: MultiIndex (Price, Ticker)
# This is EXACTLY what yf.download(["AAPL", "MSFT", "SPY"], ...) returns
arrays = [
    ["Close", "Close", "Close", "Open", "Open", "Open", "Volume", "Volume", "Volume"],
    ["AAPL",  "MSFT",  "SPY",   "AAPL", "MSFT", "SPY",  "AAPL",   "MSFT",   "SPY"],
]
tuples = list(zip(*arrays))
multi_index = pd.MultiIndex.from_tuples(tuples, names=["Price", "Ticker"])

data = np.random.rand(100, 9) * 100
yf_multi = pd.DataFrame(data, index=dates, columns=multi_index)

result = _extract_close_from_yf_download(yf_multi, ["AAPL", "MSFT", "SPY"])
check("MultiIndex → extracted 3 tickers", len(result.columns) == 3)
check("MultiIndex → columns are uppercase", set(result.columns) == {"AAPL", "MSFT", "SPY"})
check("MultiIndex → shape matches", result.shape == (100, 3))
check("MultiIndex → values match Close slice",
      np.allclose(result["AAPL"].values, yf_multi[("Close", "AAPL")].values))


# Simulate single-ticker download: flat columns
yf_single = pd.DataFrame({
    "Close": np.random.rand(100) * 100,
    "Open": np.random.rand(100) * 100,
    "Volume": np.random.rand(100) * 1e6,
}, index=dates)

result_single = _extract_close_from_yf_download(yf_single, ["SPY"])
check("Single ticker → extracted 1 column", len(result_single.columns) == 1)
check("Single ticker → column named SPY", result_single.columns[0] == "SPY")


# Simulate yfinance with lowercase "close" (some versions)
yf_lower = yf_multi.copy()
yf_lower.columns = pd.MultiIndex.from_tuples(
    [(p.lower(), t) for p, t in yf_lower.columns],
    names=["Price", "Ticker"]
)
result_lower = _extract_close_from_yf_download(yf_lower, ["AAPL", "MSFT", "SPY"])
check("Lowercase 'close' → still extracts", len(result_lower.columns) == 3)


# Timezone-aware index (yfinance sometimes returns this)
tz_dates = dates.tz_localize("UTC")
yf_tz = pd.DataFrame(data[:, :3], index=tz_dates,
                      columns=pd.MultiIndex.from_tuples(
                          [("Close", "AAPL"), ("Close", "MSFT"), ("Close", "SPY")],
                          names=["Price", "Ticker"]))
result_tz = _extract_close_from_yf_download(yf_tz, ["AAPL", "MSFT", "SPY"])
check("TZ-aware index → stripped to naive", result_tz.index.tz is None)


# ============================================================================
# TEST 2: _clean_single_ticker_df (used in compute_dispersion_from_dict)
# ============================================================================
print("\n" + "=" * 60)
print("TEST 2: Single ticker DataFrame cleaning")
print("=" * 60)

# Normal flat DataFrame (most common in data_dict)
flat_df = pd.DataFrame({"Close": [100, 101, 102], "Open": [99, 100, 101]},
                        index=make_dates(3))
series = _clean_single_ticker_df(flat_df)
check("Flat DF → returns Series", isinstance(series, pd.Series))
check("Flat DF → correct values", list(series.values) == [100, 101, 102])

# MultiIndex DataFrame (if someone passes raw yfinance slice)
multi_df = pd.DataFrame(
    {("Close", "AAPL"): [100, 101], ("Open", "AAPL"): [99, 100]},
    index=make_dates(2)
)
series_multi = _clean_single_ticker_df(multi_df)
check("MultiIndex DF → returns Series", isinstance(series_multi, pd.Series))

# Lowercase columns
lower_df = pd.DataFrame({"close": [100, 101], "open": [99, 100]}, index=make_dates(2))
series_lower = _clean_single_ticker_df(lower_df)
check("Lowercase 'close' → still works", series_lower is not None and len(series_lower) == 2)

# Empty / None
check("None input → returns None", _clean_single_ticker_df(None) is None)
check("Empty DF → returns None", _clean_single_ticker_df(pd.DataFrame()) is None)

# TZ-aware
tz_df = flat_df.copy()
tz_df.index = tz_df.index.tz_localize("UTC")
series_tz = _clean_single_ticker_df(tz_df)
check("TZ-aware → stripped", series_tz.index.tz is None)


# ============================================================================
# TEST 3: Core dispersion calculation correctness
# ============================================================================
print("\n" + "=" * 60)
print("TEST 3: Dispersion calculation correctness")
print("=" * 60)

dates = make_dates(300)

# Scenario: Index flat (0% return), stocks each move ±10% → dispersion ≈ 10%
# Create 10 stocks that move 10% over 21 days, but in opposite directions
# so the index (if equal-weighted) stays flat.
np.random.seed(42)
n_stocks = 20
stock_prices = {}
for i in range(n_stocks):
    # Half go up 10%, half go down 10% over each 21-day window
    direction = 1 if i < n_stocks // 2 else -1
    daily_ret = direction * 0.10 / 21  # ~10% over 21 days
    noise = np.random.randn(300) * 0.005  # small noise
    prices = 100 * np.exp(np.cumsum(daily_ret + noise))
    stock_prices[f"STOCK_{i}"] = prices

# Index: average of all stocks (should be roughly flat since half up, half down)
index_prices = np.mean(list(stock_prices.values()), axis=0)
stock_prices["IDX"] = index_prices

price_matrix = pd.DataFrame(stock_prices, index=dates)
result = compute_dispersion_series(price_matrix, index_col="IDX", window=21, rank_min_periods=50)

check("Returns DataFrame", isinstance(result, pd.DataFrame))
check("Has all expected columns",
      set(result.columns) == {"avg_abs_ret", "index_abs_ret", "dispersion", "dispersion_rank", "n_constituents"})
check("n_constituents = 20", result["n_constituents"].iloc[-1] == 20)

# Dispersion should be positive (stocks moved more than index)
last_disp = result["dispersion"].iloc[-1]
check(f"Dispersion is positive ({last_disp:.4f})", last_disp > 0,
      f"got {last_disp}")

# avg_abs_ret should be meaningfully > index_abs_ret
last_avg = result["avg_abs_ret"].iloc[-1]
last_idx = result["index_abs_ret"].iloc[-1]
check(f"Avg stock move ({last_avg:.4f}) > index move ({last_idx:.4f})",
      last_avg > last_idx)


# Scenario: Everything moves together (high correlation) → low dispersion
# All stocks move +5% over 21 days
correlated_prices = {}
base_returns = np.random.randn(300) * 0.01 + 0.002
base_cumret = np.exp(np.cumsum(base_returns))
for i in range(20):
    # Tiny noise around the same path
    noise = np.random.randn(300) * 0.001
    correlated_prices[f"STOCK_{i}"] = 100 * base_cumret * np.exp(np.cumsum(noise))
correlated_prices["IDX"] = 100 * base_cumret

corr_matrix = pd.DataFrame(correlated_prices, index=dates)
corr_result = compute_dispersion_series(corr_matrix, index_col="IDX", window=21, rank_min_periods=50)
corr_disp = corr_result["dispersion"].dropna().iloc[-1]

check(f"Correlated regime → low dispersion ({corr_disp:.4f})",
      abs(corr_disp) < 0.02,
      f"expected near 0, got {corr_disp}")


# ============================================================================
# TEST 4: compute_dispersion_from_dict (the daily_scan integration path)
# ============================================================================
print("\n" + "=" * 60)
print("TEST 4: Dict-based interface (daily_scan.py path)")
print("=" * 60)

dates = make_dates(300)
data_dict = {}

# Simulate data_dict format: {ticker: DataFrame with Close column}
for i in range(10):
    ticker = f"STOCK_{i}"
    prices = make_price_series(dates, seed=i)
    data_dict[ticker] = pd.DataFrame({"Close": prices, "Volume": np.random.rand(300) * 1e6})

# Add SPY as index
data_dict["SPY"] = pd.DataFrame({"Close": make_price_series(dates, seed=99)})

result = compute_dispersion_from_dict(
    data_dict, index_ticker="SPY",
    constituent_tickers=[f"STOCK_{i}" for i in range(10)],
    window=21, rank_min_periods=50,
)

check("Dict interface → returns DataFrame", isinstance(result, pd.DataFrame))
check("Dict interface → has dispersion col", "dispersion" in result.columns)
check("Dict interface → non-empty", len(result.dropna(subset=["dispersion"])) > 200)

# Test with MultiIndex DataFrames in the dict (simulating raw yfinance slices)
data_dict_multi = {}
for i in range(5):
    ticker = f"STOCK_{i}"
    prices = make_price_series(dates, seed=i)
    # Create MultiIndex DF like yfinance returns when you slice df[ticker]
    mi_df = pd.DataFrame(
        {("Close", ticker): prices.values, ("Volume", ticker): np.random.rand(300) * 1e6},
        index=dates
    )
    mi_df.columns = pd.MultiIndex.from_tuples(mi_df.columns)
    data_dict_multi[ticker] = mi_df

data_dict_multi["SPY"] = pd.DataFrame({"Close": make_price_series(dates, seed=99)})

result_multi = compute_dispersion_from_dict(
    data_dict_multi, index_ticker="SPY",
    constituent_tickers=[f"STOCK_{i}" for i in range(5)],
    window=21, rank_min_periods=50,
)

check("MultiIndex dict → still works", isinstance(result_multi, pd.DataFrame))
check("MultiIndex dict → has data",
      len(result_multi.dropna(subset=["dispersion"])) > 200)


# Test with missing tickers (some constituents not in dict)
result_partial = compute_dispersion_from_dict(
    data_dict, index_ticker="SPY",
    constituent_tickers=[f"STOCK_{i}" for i in range(20)],  # only 10 exist
    window=21, rank_min_periods=50,
)
check("Partial coverage → still works", isinstance(result_partial, pd.DataFrame))
check("Partial coverage → n_constituents = 10",
      result_partial["n_constituents"].iloc[-1] == 10,
      f"got {result_partial['n_constituents'].iloc[-1]}")


# ============================================================================
# TEST 5: Edge cases
# ============================================================================
print("\n" + "=" * 60)
print("TEST 5: Edge cases")
print("=" * 60)

# Missing index ticker
try:
    compute_dispersion_from_dict(data_dict, index_ticker="NOPE")
    check("Missing index → raises error", False, "no error raised")
except ValueError as e:
    check("Missing index → raises ValueError", "NOPE" in str(e))

# Empty data_dict
try:
    compute_dispersion_from_dict({}, index_ticker="SPY")
    check("Empty dict → raises error", False, "no error raised")
except (ValueError, KeyError):
    check("Empty dict → raises error", True)

# Single constituent
data_dict_one = {
    "AAPL": pd.DataFrame({"Close": make_price_series(dates, seed=1)}),
    "SPY": pd.DataFrame({"Close": make_price_series(dates, seed=99)}),
}
result_one = compute_dispersion_from_dict(
    data_dict_one, index_ticker="SPY",
    constituent_tickers=["AAPL"], window=21, rank_min_periods=50,
)
check("Single constituent → works", len(result_one.dropna(subset=["dispersion"])) > 0)
check("Single constituent → n_constituents=1", result_one["n_constituents"].iloc[-1] == 1)


# ============================================================================
# SUMMARY
# ============================================================================
print(f"\n{'=' * 60}")
print(f"RESULTS: {PASS} passed, {FAIL} failed")
print(f"{'=' * 60}")

if FAIL > 0:
    sys.exit(1)
else:
    print("\nAll tests passed. Module is safe to deploy.\n")
    sys.exit(0)