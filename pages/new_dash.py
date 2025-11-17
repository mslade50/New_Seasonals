# pages/sector_trends.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt


SECTOR_ETFS = [
    "IBB", "IHI", "ITA", "ITB", "IYR", "KRE", "OIH", "SMH", "VNQ",
    "XBI", "XHB", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP",
    "XLU", "XLV", "XLY", "XME", "XOP", "XRT", "GLD", "CEF", "SLV", "BTC-USD",
    "ETH-USD", "UNG", "UVXY",
    "SPY", "QQQ", "IWM", "DIA",
]


def get_current_cycle_year():
    """Determine which presidential cycle year we're in (Election, Post, Midterm, Pre)"""
    current_year = dt.date.today().year
    cycle_position = (current_year - 1952) % 4
    
    cycle_map = {
        0: "Election",      # 1952, 1956, ..., 2024
        1: "Post-Election", # 1953, 1957, ..., 2025
        2: "Midterm",       # 1954, 1958, ..., 2026
        3: "Pre-Election"   # 1955, 1959, ..., 2027
    }
    return cycle_map[cycle_position]


def get_cycle_years(cycle_type):
    """Get all years for a given cycle type"""
    base_years = {
        "Election": 1952,
        "Post-Election": 1953,
        "Midterm": 1954,
        "Pre-Election": 1955
    }
    
    start = base_years[cycle_type]
    years = [start + i * 4 for i in range(20)]  # Get ~20 cycles worth
    return [y for y in years if y <= dt.date.today().year]


def calculate_seasonal_rank(df, current_trading_day, current_cycle_type):
    """
    Calculate seasonal rank based on forward returns at current trading day position.
    Optimized for speed with vectorized operations.
    
    Parameters:
    - df: DataFrame with OHLC data and Date index
    - current_trading_day: int, current trading day of year (1-252)
    - current_cycle_type: str, current presidential cycle year type
    
    Returns:
    - float: Seasonal rank percentile (0-100)
    """
    if df.empty or len(df) < 252:
        return np.nan
    
    # Use Close price
    if "Adj Close" in df.columns:
        close = df["Adj Close"]
    elif "Close" in df.columns:
        close = df["Close"]
    else:
        return np.nan
    
    # Ensure close is a Series, not DataFrame
    if isinstance(close, pd.DataFrame):
        close = close.squeeze()
    
    close = close.dropna()
    if len(close) < 252:
        return np.nan
    
    # Add year and trading day columns - use numpy for speed
    data = pd.DataFrame({"close": close.values}, index=close.index)
    data["year"] = data.index.year
    
    # Vectorized trading day calculation
    year_groups = data.groupby("year").cumcount().values + 1
    data["trading_day"] = year_groups
    
    # Calculate forward returns (5, 10, 21 days) - vectorized
    close_vals = data["close"].values
    fwd_5d = np.full(len(close_vals), np.nan)
    fwd_10d = np.full(len(close_vals), np.nan)
    fwd_21d = np.full(len(close_vals), np.nan)
    
    fwd_5d[:-5] = close_vals[5:] / close_vals[:-5] - 1
    fwd_10d[:-10] = close_vals[10:] / close_vals[:-10] - 1
    fwd_21d[:-21] = close_vals[21:] / close_vals[:-21] - 1
    
    # Average of forward returns
    data["avg_fwd_return"] = np.nanmean(np.column_stack([fwd_5d, fwd_10d, fwd_21d]), axis=1)
    
    # Get cycle years for weighting
    cycle_years = set(get_cycle_years(current_cycle_type))
    
    # Apply 3x weight to current cycle years - vectorized
    data["weight"] = np.where(data["year"].isin(cycle_years), 3.0, 1.0)
    
    # Filter to only rows with valid forward returns and limit to trading days 1-252
    valid_data = data[
        (data["avg_fwd_return"].notna()) & 
        (data["trading_day"] <= 252)
    ].copy()
    
    if valid_data.empty or current_trading_day is None:
        return np.nan
    
    # Define window around current day
    window_start = max(1, current_trading_day - 2)
    window_end = min(252, current_trading_day + 2)
    window_days = range(window_start, window_end + 1)
    
    current_year = dt.date.today().year
    percentile_ranks = []
    
    # Pre-filter data for current year to avoid repeated filtering
    current_year_data = valid_data[valid_data["year"] == current_year]
    
    for day in window_days:
        # Get all observations for this trading day
        day_mask = valid_data["trading_day"] == day
        day_data = valid_data[day_mask]
        
        if day_data.empty:
            continue
        
        # Get current year's value for this day
        current_day_data = current_year_data[current_year_data["trading_day"] == day]
        
        if current_day_data.empty:
            continue
        
        current_val = current_day_data["avg_fwd_return"].iloc[0]
        
        # Vectorized weighted percentile calculation
        weights = day_data["weight"].values
        returns = day_data["avg_fwd_return"].values
        
        weighted_below = np.sum(weights[returns <= current_val])
        total_weight = np.sum(weights)
        
        if total_weight > 0:
            pct_rank = (weighted_below / total_weight) * 100.0
            percentile_ranks.append(pct_rank)
    
    if not percentile_ranks:
        return np.nan
    
    # Return average of percentile ranks across the window
    return float(np.mean(percentile_ranks))


def percentile_rank(series: pd.Series, value) -> float:
    """Return percentile rank (0–100) of value within series."""
    s = series.dropna().values
    if s.size == 0:
        return np.nan

    # Normalize value to a scalar float
    if isinstance(value, (pd.Series, np.ndarray, list, tuple)):
        arr = np.asarray(value).ravel()
        if arr.size == 0:
            return np.nan
        v = float(arr[-1])
    else:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return np.nan

    if np.isnan(v):
        return np.nan

    pct = (s <= v).sum() / s.size * 100.0
    return float(pct)


@st.cache_data(show_spinner=True)
def load_sector_metrics(tickers):
    rows = []
    current_cycle = get_current_cycle_year()
    
    # Get current trading day of year
    today = dt.date.today()
    start_of_year = dt.date(today.year, 1, 1)
    spy_ytd = yf.download("SPY", start=start_of_year, end=today, progress=False)
    current_trading_day = len(spy_ytd) if not spy_ytd.empty else None

    for t in tickers:
        try:
            df = yf.download(
                t,
                period="max",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            st.write(f"⚠️ Failed download for {t}: {e}")
            continue

        if df.empty:
            st.write(f"⚠️ No data for {t}, skipping")
            continue

        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Use Adj Close if present, otherwise Close
        if "Adj Close" in df.columns:
            close = df["Adj Close"]
        elif "Close" in df.columns:
            close = df["Close"]
        else:
            st.write(f"⚠️ No Close/Adj Close for {t}, skipping")
            continue

        # Ensure close is a Series, not DataFrame
        if isinstance(close, pd.DataFrame):
            close = close.squeeze()
        
        close = close.dropna()
        if close.empty:
            continue

        # Moving averages
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        # Distances to MA in %
        dist20 = (close - ma20) / ma20 * 100.0
        dist50 = (close - ma50) / ma50 * 100.0
        dist200 = (close - ma200) / ma200 * 100.0

        # Today's distance values
        d20_today = dist20.dropna().iloc[-1] if not dist20.dropna().empty else np.nan
        d50_today = dist50.dropna().iloc[-1] if not dist50.dropna().empty else np.nan
        d200_today = dist200.dropna().iloc[-1] if not dist200.dropna().empty else np.nan

        # Percentile ranks over life of each distance series
        p20 = percentile_rank(dist20, d20_today)
        p50 = percentile_rank(dist50, d50_today)
        p200 = percentile_rank(dist200, d200_today)

        # Calculate seasonal rank
        seasonal_rank = np.nan
        if current_trading_day is not None:
            seasonal_rank = calculate_seasonal_rank(df, current_trading_day, current_cycle)

        rows.append(
            {
                "Ticker": t,
                "Price": float(close.iloc[-1]),
                "SeasonalRank": seasonal_rank,
                "PctRank20": p20,
                "PctRank50": p50,
                "PctRank200": p200,
            }
        )

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)

    # Ensure numeric dtypes before rounding
    num_cols = ["Price", "SeasonalRank", "PctRank20", "PctRank50", "PctRank200"]
    for col in num_cols:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

    # Rounding / formatting
    if "Price" in df_out.columns:
        df_out["Price"] = df_out["Price"].round(2)
    for col in ["SeasonalRank", "PctRank20", "PctRank50", "PctRank200"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].round(1)

    # Sort by SeasonalRank descending
    if "SeasonalRank" in df_out.columns:
        df_out = df_out.sort_values("SeasonalRank", ascending=False, ignore_index=True)
    else:
        df_out = df_out.sort_values("Ticker", ignore_index=True)

    return df_out


def main():
    st.title("Sector ETF Trend Dashboard with Seasonal Rank")

    current_cycle = get_current_cycle_year()
    today = dt.date.today()
    
    st.write(
        f"**Current Date:** {today.strftime('%B %d, %Y')} | "
        f"**Presidential Cycle Year:** {current_cycle}"
    )
    
    st.write(
        "**Seasonal Rank:** Percentile rank of average forward returns (5/10/21-day) "
        f"at current position vs all historical days. {current_cycle} years weighted 3x."
    )
    
    st.write(
        "**Trend Ranks:** Distance to moving averages and percentile rank vs full history."
    )

    # Optional: refresh button to bust cache
    if st.button("Refresh data"):
        load_sector_metrics.clear()

    with st.spinner("Loading sector ETF data from Yahoo Finance..."):
        table = load_sector_metrics(sorted(set(SECTOR_ETFS)))

    if table.empty:
        st.error("No data available. Try refreshing or checking the ticker list.")
        return

    st.subheader("Sector & Index ETFs")

    def highlight_pct(val):
        if pd.isna(val):
            return ""
        if val > 90:
            return "background-color: #ffcccc; color: #8b0000;"
        if val < 15:
            return "background-color: #ccffcc; color: #006400;"
        return ""

    styled = (
        table.style
        .format(
            {
                "Price": "{:.2f}",
                "SeasonalRank": "{:.1f}",
                "PctRank20": "{:.1f}",
                "PctRank50": "{:.1f}",
                "PctRank200": "{:.1f}",
            }
        )
        .applymap(
            highlight_pct,
            subset=["SeasonalRank", "PctRank20", "PctRank50", "PctRank200"],
        )
    )

    st.dataframe(styled, use_container_width=True)


if __name__ == "__main__":
    main()
