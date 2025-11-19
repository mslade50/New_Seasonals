# pages/sector_trends.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import sqlite3
import datetime

SECTOR_ETFS = [
    # From screenshot
    "IBB", "IHI", "ITA", "ITB", "IYR", "KRE", "OIH", "SMH", "VNQ",
    "XBI", "XHB", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP",
    "XLU", "XLV", "XLY", "XME", "XOP", "XRT", "GLD", "CEF", "SLV", "BTC-USD",
    "ETH-USD", "UNG", "UVXY",
    # Extra index/sector leaders
    "SPY", "QQQ", "IWM", "DIA", "SMH",
]

CORE_TICKERS = ["SPY", "QQQ", "IWM", "SMH", "DIA"]
DB_PATH = "past_sznl.db"

# Helper to clear all relevant caches
def clear_all_caches():
    load_sector_metrics.clear()
    load_core_distance_frame.clear()
    load_seasonal_map.clear()
    load_spy_ohlc.clear()

# -----------------------------------------------------------------------------
# DATABASE / SEASONAL HELPERS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_seasonal_map():
    """
    Loads the DB and creates a lookup dictionary:
    {
       'TICKER': { (Month, Day): rank_value, ... },
       ...
    }
    This allows us to map any date in history to a seasonal rank based on M/D.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        # Select relevant columns. Assuming table name is 'seasonal_ranks' based on screenshot
        df = pd.read_sql("SELECT Date, seasonal_rank, ticker FROM seasonal_ranks", conn)
        conn.close()
    except Exception as e:
        st.error(f"Could not load database: {e}")
        return {}

    if df.empty:
        return {}

    # Convert DB date string to datetime objects
    df["Date"] = pd.to_datetime(df["Date"])

    # Create a (Month, Day) tuple column for matching
    df["MD"] = df["Date"].apply(lambda x: (x.month, x.day))

    # Build nested dictionary
    # Result: output_map["SPY"][(1, 3)] = 13.5
    output_map = {}
    
    # Group by ticker to build the inner dicts
    for ticker, group in df.groupby("ticker"):
        # Create dict: {(1, 3): 13.5, (1, 4): 16.1, ...}
        output_map[ticker] = pd.Series(
            group.seasonal_rank.values, index=group.MD
        ).to_dict()

    return output_map


def get_sznl_val(ticker, target_date, sznl_map):
    """Retuns the rank for a specific ticker and date object."""
    if ticker not in sznl_map:
        return np.nan
    
    md = (target_date.month, target_date.day)
    return sznl_map[ticker].get(md, np.nan)

# -----------------------------------------------------------------------------
# METRICS & CALCULATIONS
# -----------------------------------------------------------------------------

def percentile_rank(series: pd.Series, value) -> float:
    """Return percentile rank (0–100) of value within series."""
    s = series.dropna().values
    if s.size == 0:
        return np.nan

    # Normalize value to a scalar float if it's array-like / Series
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
    # 1. Load Seasonal Data first
    sznl_map = load_seasonal_map()
    today = datetime.datetime.now()
    
    rows = []

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

        # Use Adj Close if present, otherwise Close
        if "Adj Close" in df.columns:
            close = df["Adj Close"]
        elif "Close" in df.columns:
            close = df["Close"]
        else:
            st.write(f"⚠️ No Close/Adj Close for {t}, skipping")
            continue

        close = close.dropna()
        if close.empty:
            continue

        # Moving averages
        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        # Distances to MA in %
        dist5 = (close - ma5) / ma5 * 100.0
        dist20 = (close - ma20) / ma20 * 100.0
        dist50 = (close - ma50) / ma50 * 100.0
        dist200 = (close - ma200) / ma200 * 100.0

        # Today's distance values
        d5_today = dist5.dropna().iloc[-1] if not dist5.dropna().empty else np.nan
        d20_today = dist20.dropna().iloc[-1] if not dist20.dropna().empty else np.nan
        d50_today = dist50.dropna().iloc[-1] if not dist50.dropna().empty else np.nan
        d200_today = dist200.dropna().iloc[-1] if not dist200.dropna().empty else np.nan

        # Percentile ranks over life of each distance series
        p5 = percentile_rank(dist5, d5_today)
        p20 = percentile_rank(dist20, d20_today)
        p50 = percentile_rank(dist50, d50_today)
        p200 = percentile_rank(dist200, d200_today)
        
        # Lookup Seasonal Value for TODAY
        current_sznl = get_sznl_val(t, today, sznl_map)

        rows.append(
            {
                "Ticker": t,
                "Price": float(close.iloc[-1]),
                "Sznl": current_sznl,  # Added Column
                "PctRank5": p5,
                "PctRank20": p20,
                "PctRank50": p50,
                "PctRank200": p200,
            }
        )

    if not rows:
        return pd.DataFrame()

    df_out = pd.DataFrame(rows)

    # Ensure numeric dtypes before rounding
    num_cols = ["Price", "Sznl", "PctRank5", "PctRank20", "PctRank50", "PctRank200"]
    for col in num_cols:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

    # Rounding / formatting
    if "Price" in df_out.columns:
        df_out["Price"] = df_out["Price"].round(2)
    
    if "Sznl" in df_out.columns:
        df_out["Sznl"] = df_out["Sznl"].round(1)
        
    for col in ["PctRank5", "PctRank20", "PctRank50", "PctRank200"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].round(1)

    # Sort (e.g., by PctRank200 descending)
    if "PctRank200" in df_out.columns:
        df_out = df_out.sort_values("PctRank200", ascending=False, ignore_index=True)
    else:
        df_out = df_out.sort_values("Ticker", ignore_index=True)

    return df_out


@st.cache_data(show_spinner=True)
def load_spy_ohlc():
    """Daily SPY OHLC for candlestick charts."""
    df = yf.download(
        "SPY",
        period="max",
        interval="1d",
        auto_adjust=False,  # keep real OHLC
        progress=False,
    )
    if df.empty:
        return pd.DataFrame()

    # 1. Check if the columns are a MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    cols = ["Open", "High", "Low", "Close"]
    existing_cols = [c for c in cols if c in df.columns]
    
    if len(existing_cols) < 4:
        st.warning(f"Could not find all required columns (Open, High, Low, Close) in SPY data.")
        return pd.DataFrame()

    return df[existing_cols].dropna()

@st.cache_data(show_spinner=True)
def load_core_distance_frame():
    """
    Build a daily feature frame for SPY/QQQ/IWM/SMH/DIA using
    distance-to-MA (5/20/50/200) AND Seasonal Rank for each.
    """
    sznl_map = load_seasonal_map()
    
    all_feats = []
    close_series_map = {} 

    for t in CORE_TICKERS:
        try:
            df = yf.download(
                t,
                period="max",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            st.write(f"⚠️ Failed core download for {t}: {e}")
            continue

        if df.empty:
            continue

        if "Adj Close" in df.columns:
            close = df["Adj Close"].dropna()
        elif "Close" in df.columns:
            close = df["Close"].dropna()
        else:
            continue

        if close.empty:
            continue

        close_series_map[t] = close.copy() 

        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        dist5 = (close - ma5) / ma5 * 100.0
        dist20 = (close - ma20) / ma20 * 100.0
        dist50 = (close - ma50) / ma50 * 100.0
        dist200 = (close - ma200) / ma200 * 100.0

        # Build feature frame
        feats = pd.DataFrame(index=close.index)
        feats[f"{t}_dist5"] = dist5
        feats[f"{t}_dist20"] = dist20
        feats[f"{t}_dist50"] = dist50
        feats[f"{t}_dist200"] = dist200
        
        # --- ADD SEASONAL PREDICTOR ---
        # 1. Extract month/day from index
        idx_months = feats.index.month
        idx_days = feats.index.day
        
        # 2. Get the dictionary for this specific ticker
        t_sznl_dict = sznl_map.get(t, {})
        
        # 3. Map (month, day) to value. 
        # We use a list comprehension for speed, defaulting to NaN if date not in DB
        sznl_values = [t_sznl_dict.get((m, d), np.nan) for m, d in zip(idx_months, idx_days)]
        
        feats[f"{t}_sznl"] = sznl_values
        # ------------------------------

        all_feats.append(feats)

    if not all_feats:
        return pd.DataFrame()

    # Inner join on dates across all core tickers
    core_df = all_feats[0]
    for feats in all_feats[1:]:
        core_df = core_df.join(feats, how="inner")

    core_df = core_df.dropna()
    core_df = core_df.sort_index()
    core_df.index = pd.to_datetime(core_df.index)
    core_df.index.name = "Date"

    # Add forward returns
    horizons = [2, 5, 10, 21, 63, 126, 252]
    
    for t in CORE_TICKERS:
        close = close_series_map.get(t)
        if close is not None:
            close = close.reindex(core_df.index).dropna() 
            close = close.reindex(core_df.index)
        
            for h in horizons:
                core_df[f"{t}_fwd_{h}d"] = close.shift(-h) / close - 1.0

    return core_df


def compute_distance_matches(core_df: pd.DataFrame,
                             n_matches: int = 20,
                             exclude_last_n: int = 63,
                             min_spacing_days: int = 21) -> pd.DataFrame:
    """
    Uses Euclidean distance on MA distances AND Seasonal Ranks ("sznl").
    """
    if core_df.empty:
        return pd.DataFrame()

    df = core_df.copy().sort_index()

    # Feature columns: Include 'dist' AND 'sznl' columns
    feature_cols = [c for c in df.columns if "dist" in c or "sznl" in c]
    
    fwd_cols = [c for c in df.columns if "_fwd_" in c] 

    # Require features + SPY returns
    spy_fwd_cols = [c for c in fwd_cols if c.startswith("SPY_fwd_")]
    df = df.dropna(subset=feature_cols + spy_fwd_cols)

    if len(df) <= exclude_last_n + 1:
        return pd.DataFrame()

    target = df.iloc[-1][feature_cols].astype(float).values
    hist = df.iloc[:-exclude_last_n].copy()

    # Restrict to post-1997 and exclude 2020
    hist = hist[hist.index.year > 1997]
    hist = hist[hist.index.year != 2020]

    if hist.empty:
        return pd.DataFrame()

    X = hist[feature_cols].astype(float).values
    dists = np.sqrt(((X - target) ** 2).sum(axis=1))
    hist["distance"] = dists

    hist = hist.sort_values("distance").copy()

    selected_dates = []
    selected_idx = []

    for dt, row in hist.iterrows():
        if all(abs((dt - prev).days) > min_spacing_days for prev in selected_dates):
            selected_dates.append(dt)
            selected_idx.append(dt)
            if len(selected_idx) >= n_matches:
                break

    if not selected_idx:
        return pd.DataFrame()

    matches = hist.loc[selected_idx].copy()
    matches = matches.reset_index() 
    if "index" in matches.columns and "Date" not in matches.columns:
        matches = matches.rename(columns={"index": "Date"})

    keep_cols = ["Date", "distance"] + feature_cols + fwd_cols
    matches = matches[keep_cols]

    return matches


def main():
    st.title("Sector ETF Trend Dashboard")

    st.write(
        "Distance to 5/20/50/200-day moving averages and the percentile rank of "
        "today's distance vs each ETF's full trading history."
    )

    if st.button("Refresh data"):
        clear_all_caches()
        st.experimental_rerun()

    with st.spinner("Loading sector ETF data from Yahoo Finance & DB..."):
        table = load_sector_metrics(sorted(set(SECTOR_ETFS)))

    if table.empty:
        st.error("No data available. Try refreshing or relaxing the ticker list.")
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

    # Updated format dict to include Sznl
    format_dict = {
        "Price": "{:.2f}",
        "Sznl": "{:.1f}",  # Format Sznl
        "PctRank5": "{:.1f}",
        "PctRank20": "{:.1f}",
        "PctRank50": "{:.1f}",
        "PctRank200": "{:.1f}",
    }

    styled = (
        table.style
        .format(format_dict)
        .applymap(
            highlight_pct,
            subset=["PctRank5", "PctRank20", "PctRank50", "PctRank200"],
        )
    )

    st.dataframe(styled, use_container_width=True)

    # -------- Distance-matching section --------
    st.subheader("Historical Distance Matches (SPY/QQQ/IWM/SMH/DIA)")
    st.info("Matches are now calculated using MA Distances + Seasonal Ranks.")

    with st.spinner("Computing distance matches vs history..."):
        core_df = load_core_distance_frame()
        match_table_raw = compute_distance_matches(core_df, n_matches=20, exclude_last_n=63)

    if match_table_raw.empty:
        st.warning("Not enough historical data to compute distance matches.")
    else:
        st.write(
            "20 closest historical dates to today's joint MA-distance and Seasonality profile."
        )

        raw_matches = match_table_raw.copy()
        horizons = [2, 5, 10, 21, 63, 126, 252]
        display_fwd_cols = [f"Fwd {h}d" for h in horizons]
        final_cols = ["Date", "distance"] + display_fwd_cols
        
        avg_rows_data = []
        for t in CORE_TICKERS:
            t_fwd_cols = [f"{t}_fwd_{h}d" for h in horizons]
            avg_t_fwd_vals = raw_matches[t_fwd_cols].mean(numeric_only=True)
            
            avg_row = {"Date": f"Avg {t}", "distance": np.nan}
            for i, display_col in enumerate(display_fwd_cols):
                raw_col = t_fwd_cols[i]
                avg_row[display_col] = avg_t_fwd_vals[raw_col]
                
            avg_rows_data.append(avg_row)

        avg_df = pd.DataFrame(avg_rows_data)
        avg_df = avg_df[final_cols] 

        for c in display_fwd_cols:
            avg_df[c] = (avg_df[c] * 100.0).round(2)

        spy_avg_row_idx = avg_df[avg_df["Date"] == "Avg SPY"].index[0]
        avg_df.loc[spy_avg_row_idx, "Date"] = "Average"
        avg_df.loc[spy_avg_row_idx, "distance"] = raw_matches["distance"].mean().round(4)
        
        avg_df_sorted = avg_df.set_index("Date").reindex([
            "Average", "Avg QQQ", "Avg IWM", "Avg SMH", "Avg DIA"
        ]).reset_index().rename(columns={'index':'Date'}).copy()

        match_rows_for_display = raw_matches[["Date", "distance"]].copy()
        spy_fwd_cols = [f"SPY_fwd_{h}d" for h in horizons]
        match_rows_for_display[display_fwd_cols] = raw_matches[spy_fwd_cols].values
        
        for c in display_fwd_cols:
            match_rows_for_display[c] = (match_rows_for_display[c] * 100.0).round(2)
        match_rows_for_display["distance"] = match_rows_for_display["distance"].round(4)
        match_rows_for_display["Date"] = match_rows_for_display["Date"].dt.strftime("%b %d %Y")

        match_with_avg = pd.concat([avg_df_sorted, match_rows_for_display], ignore_index=True)

        for c in display_fwd_cols:
            match_with_avg[c] = match_with_avg[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
        match_with_avg["distance"] = match_with_avg["distance"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        
        match_with_avg.loc[match_with_avg["Date"].str.startswith("Avg ") & (match_with_avg["Date"] != "Average"), "distance"] = "" 
        match_with_avg = match_with_avg.astype(str)

        st.dataframe(match_with_avg, use_container_width=True)

        # ---- Candlestick charts ----
        st.subheader("SPY Candles Around Top 10 Match Dates")

        spy_ohlc = load_spy_ohlc()
        
        if not spy_ohlc.empty:
            ohlc_cols = ["Open", "High", "Low", "Close"]
            for col in ohlc_cols:
                if col in spy_ohlc.columns:
                    spy_ohlc[col] = pd.to_numeric(spy_ohlc[col], errors='coerce')
            
            spy_ohlc = spy_ohlc.dropna(subset=ohlc_cols)
            spy_ohlc.index = pd.to_datetime(spy_ohlc.index).normalize().tz_localize(None)

            top10 = raw_matches.head(10).copy() 
            if not pd.api.types.is_datetime64_any_dtype(top10["Date"]):
                top10["Date"] = pd.to_datetime(top10["Date"])
            top10["Date"] = top10["Date"].dt.normalize().dt.tz_localize(None)

            for i, row in top10.iterrows():
                center = row["Date"]
                start_dt = center - pd.Timedelta(days=90)
                end_dt = center + pd.Timedelta(days=90)
                start_date_str = start_dt.strftime("%Y-%m-%d")
                end_date_str = end_dt.strftime("%Y-%m-%d")

                window = spy_ohlc.loc[start_date_str:end_date_str].copy()

                if window.empty:
                    continue
                if window[ohlc_cols].isnull().any().any():
                     continue

                center_date_norm = center.normalize()
                sparse_labels = [""] * len(window.index)
                
                try:
                    center_loc = window.index.get_loc(center_date_norm)
                    sparse_labels[center_loc] = center.strftime('%b %d %Y')
                except KeyError:
                    pass
                
                fig = go.Figure(data=[go.Candlestick(
                    x=window.index,
                    open=window["Open"], high=window["High"],
                    low=window["Low"], close=window["Close"],
                )])
                
                fig.add_vline(x=center, line_width=1, line_dash="dot", line_color="gray")
                fig.update_layout(
                    title=f"SPY $\\pm$3 Months Around {center.date()}",
                    xaxis_title="Date", yaxis_title="Price", height=400,
                    xaxis={'type': 'category', 'rangeslider': {'visible': False},
                           'tickvals': window.index, 'ticktext': sparse_labels, 'tickangle': 0}
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
