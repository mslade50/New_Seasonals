# pages/sector_trends.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go


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

# Helper to clear all relevant caches
def clear_all_caches():
    load_sector_metrics.clear()
    load_core_distance_frame.clear()
    
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

        rows.append(
            {
                "Ticker": t,
                "Price": float(close.iloc[-1]),
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
    num_cols = ["Price", "PctRank5", "PctRank20", "PctRank50", "PctRank200"]
    for col in num_cols:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

    # Rounding / formatting
    if "Price" in df_out.columns:
        df_out["Price"] = df_out["Price"].round(2)
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

    # --- START OF FIX ---
    # 1. Check if the columns are a MultiIndex (the main problem you identified)
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten the MultiIndex: keep the first level (e.g., 'Open', 'High')
        # while discarding the ticker level (e.g., 'SPY').
        df.columns = df.columns.get_level_values(0)
    # --- END OF FIX ---

    # Ensure expected columns exist and drop any row with missing data
    cols = ["Open", "High", "Low", "Close"]
    
    # 2. Defensive check: only select columns that actually exist
    existing_cols = [c for c in cols if c in df.columns]
    
    if len(existing_cols) < 4:
        # If we can't find the necessary columns, return empty
        st.warning(f"Could not find all required columns (Open, High, Low, Close) in SPY data.")
        return pd.DataFrame()

    return df[existing_cols].dropna()

def load_core_distance_frame():
    """
    Build a daily feature frame for SPY/QQQ/IWM/SMH/DIA using
    distance-to-MA (5/20/50/200) for each, plus SPY forward returns.
    """
    all_feats = []
    spy_close = None

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
            st.write(f"⚠️ No data for {t} in core set, skipping")
            continue

        if "Adj Close" in df.columns:
            close = df["Adj Close"].dropna()
        elif "Close" in df.columns:
            close = df["Close"].dropna()
        else:
            st.write(f"⚠️ No Close/Adj Close for {t}, skipping")
            continue

        if close.empty:
            continue

        if t == "SPY":
            spy_close = close.copy()

        ma5 = close.rolling(5).mean()
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        ma200 = close.rolling(200).mean()

        dist5 = (close - ma5) / ma5 * 100.0
        dist20 = (close - ma20) / ma20 * 100.0
        dist50 = (close - ma50) / ma50 * 100.0
        dist200 = (close - ma200) / ma200 * 100.0

        # ---- FIX: always build with an explicit index so pandas never thinks these are scalars ----
        feats = pd.DataFrame(index=close.index)
        feats[f"{t}_dist5"] = dist5
        feats[f"{t}_dist20"] = dist20
        feats[f"{t}_dist50"] = dist50
        feats[f"{t}_dist200"] = dist200

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

    # Add forward returns for SPY
    if spy_close is not None:
        spy_close = spy_close.reindex(core_df.index).dropna()
        spy_close = spy_close.reindex(core_df.index)  # align index exactly

        horizons = [2, 5, 10, 21, 63, 126, 252]
        for h in horizons:
            core_df[f"SPY_fwd_{h}d"] = spy_close.shift(-h) / spy_close - 1.0

    return core_df


def compute_distance_matches(core_df: pd.DataFrame,
                             n_matches: int = 20,
                             exclude_last_n: int = 63,
                             min_spacing_days: int = 21) -> pd.DataFrame:
    """
    Use equal-weight Euclidean distance on all distance features
    across SPY/QQQ/IWM/SMH/DIA to find closest historical dates.

    min_spacing_days: minimum calendar-day gap between any two
    selected dates (e.g., 21 means no other match within +/-21 days).
    """
    if core_df.empty:
        return pd.DataFrame()

    df = core_df.copy().sort_index()

    # Feature columns are all dist* columns for the 5 core tickers
    feature_cols = [c for c in df.columns if "dist" in c]
    fwd_cols = [c for c in df.columns if c.startswith("SPY_fwd_")]

    # Require all features + forward returns to be present
    df = df.dropna(subset=feature_cols + fwd_cols)

    if len(df) <= exclude_last_n + 1:
        return pd.DataFrame()

    # Target vector = latest row
    target = df.iloc[-1][feature_cols].astype(float).values

    # Exclude most recent N rows for matching
    hist = df.iloc[:-exclude_last_n].copy()

    # Optional: restrict to post-1997 and exclude 2020
    hist = hist[hist.index.year > 1997]
    hist = hist[hist.index.year != 2020]

    if hist.empty:
        return pd.DataFrame()

    X = hist[feature_cols].astype(float).values
    dists = np.sqrt(((X - target) ** 2).sum(axis=1))
    hist["distance"] = dists

    # -------- spacing logic: only 1 date per +/- min_spacing_days window --------
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
    matches = matches.reset_index()  # Date becomes a column named index or Date
    if "index" in matches.columns and "Date" not in matches.columns:
        matches = matches.rename(columns={"index": "Date"})

    # Keep just Date, distance, and SPY forward returns
    keep_cols = ["Date", "distance"] + fwd_cols
    matches = matches[keep_cols]

    # Format returns as percents
    for c in fwd_cols:
        matches[c] = (matches[c] * 100.0).round(2)

    matches["distance"] = matches["distance"].round(4)

    return matches


def main():
    st.title("Sector ETF Trend Dashboard")

    st.write(
        "Distance to 5/20/50/200-day moving averages and the percentile rank of "
        "today's distance vs each ETF's full trading history."
    )

    # Optional: refresh button to bust cache
    if st.button("Refresh data"):
        clear_all_caches()
        st.experimental_rerun()

    with st.spinner("Loading sector ETF data from Yahoo Finance..."):
        table = load_sector_metrics(sorted(set(SECTOR_ETFS)))

    if table.empty:
        st.error("No data available. Try refreshing or relaxing the ticker list.")
        return

    st.subheader("Sector & Index ETFs")

    def highlight_pct(val):
        if pd.isna(val):
            return ""
        if val > 90:
            # light red bg, dark red text
            return "background-color: #ffcccc; color: #8b0000;"
        if val < 15:
            # light green bg, dark green text
            return "background-color: #ccffcc; color: #006400;"
        return ""

    styled = (
        table.style
        .format(
            {
                "Price": "{:.2f}",
                "PctRank5": "{:.1f}",
                "PctRank20": "{:.1f}",
                "PctRank50": "{:.1f}",
                "PctRank200": "{:.1f}",
            }
        )
        .applymap(
            highlight_pct,
            subset=["PctRank5", "PctRank20", "PctRank50", "PctRank200"],
        )
    )

    st.dataframe(styled, use_container_width=True)

    # -------- Distance-matching section --------
    st.subheader("Historical Distance Matches (SPY/QQQ/IWM/SMH/DIA)")

    with st.spinner("Computing distance matches vs history..."):
        core_df = load_core_distance_frame()
        match_table = compute_distance_matches(core_df, n_matches=20, exclude_last_n=63)

    if match_table.empty:
        st.warning("Not enough historical data to compute distance matches.")
    else:
        st.write(
            "20 closest historical dates to today's joint MA-distance profile "
            "using equal-weight distances across SPY, QQQ, IWM, SMH, and DIA. "
            "Returns are forward % changes in SPY."
        )

        # Keep a copy with real datetimes for charting
        raw_matches = match_table.copy() 

        # --- EXTREME FIX: Force all columns to simple string format ---
        
        # 1. Start with the raw match data
        match_display = raw_matches.copy()
        
        # 2. Convert the 'Date' column to the desired string format (e.g., 'Jun 05 2007')
        match_display["Date"] = match_display["Date"].dt.strftime("%b %d %Y")
        
        # Identify numeric columns for calculation and string formatting
        num_cols = [c for c in match_display.columns if c != "Date"]

        # 3. Calculate Average row values (from the raw numeric data)
        avg_vals = raw_matches[num_cols].mean(numeric_only=True)

        # 4. Format all numeric columns in match_display to strings
        for c in num_cols:
            if c.startswith("SPY_fwd_"):
                # Format to string with 2 decimals
                match_display[c] = match_display[c].round(2).apply(lambda x: f"{x:.2f}")
            elif c == "distance":
                # Format to string with 4 decimals
                match_display[c] = match_display[c].round(4).apply(lambda x: f"{x:.4f}")
        
        # 5. Create the Average row with formatted strings
        avg_row = {"Date": "Average"}
        for c in num_cols:
            if c.startswith("SPY_fwd_"):
                avg_row[c] = f"{avg_vals[c]:.2f}"
            elif c == "distance":
                avg_row[c] = f"{avg_vals[c]:.4f}"
        
        avg_df = pd.DataFrame([avg_row])
        avg_df = avg_df[match_display.columns]

        # 6. Concatenate the average row and the match rows
        match_with_avg = pd.concat([avg_df, match_display], ignore_index=True)

        # 7. CRITICAL: Convert the entire DataFrame to the 'object' (string) dtype one last time.
        # This is the final, definitive step to prevent datetime interpretation.
        match_with_avg = match_with_avg.astype(str)

        st.dataframe(match_with_avg, use_container_width=True)
        # ---------------------------------------------------------------------------------

        # ---- 3.2 Candlestick charts for top 10 dates (uses the original raw_matches) ----
        st.subheader("SPY Candles Around Top 10 Match Dates")
        
        # ... (rest of the plotting code remains unchanged, using raw_matches)
        # --------------------------------------------------------------------------

        # ---- 3.2 Candlestick charts for top 10 dates (uses the original raw_matches) ----
        # ... (Rest of the code remains the same)
        # --------------------------------------------------------------------------

        # ---- 3.2 Candlestick charts for top 10 dates (uses the original raw_matches) ----
        st.subheader("SPY Candles Around Top 10 Match Dates")

        spy_ohlc = load_spy_ohlc()
        
        # CRITICAL NEW CHECK
        if spy_ohlc.empty:
            st.warning("Could not load SPY OHLC data. Check `load_spy_ohlc` or internet connection.")
            return

        # Ensure OHLC columns are numeric right after loading/before slicing
        ohlc_cols = ["Open", "High", "Low", "Close"]
        for col in ohlc_cols:
            if col in spy_ohlc.columns:
                spy_ohlc[col] = pd.to_numeric(spy_ohlc[col], errors='coerce')
        
        spy_ohlc = spy_ohlc.dropna(subset=ohlc_cols)

        if spy_ohlc.empty:
            st.warning("SPY OHLC data was lost after cleaning (NaNs or bad types).")
            return

        # Normalize SPY index to date-only and strip any timezone info
        spy_ohlc = spy_ohlc.copy()
        spy_ohlc.index = pd.to_datetime(spy_ohlc.index).normalize().tz_localize(None)

        top10 = raw_matches.head(10).copy()

        # Normalize Date column in top10 (which uses the original datetime raw_matches)
        if not pd.api.types.is_datetime64_any_dtype(top10["Date"]):
            top10["Date"] = pd.to_datetime(top10["Date"])
        top10["Date"] = top10["Date"].dt.normalize().dt.tz_localize(None)


        for i, row in top10.iterrows():
            center = row["Date"]
            
            start_dt = center - pd.Timedelta(days=90)
            end_dt = center + pd.Timedelta(days=90)

            start_date_str = start_dt.strftime("%Y-%m-%d")
            end_date_str = end_dt.strftime("%Y-%m-%d")

            # Robust date-based slice
            window = spy_ohlc.loc[start_date_str:end_date_str].copy()

            if window.empty:
                st.warning(f"Slice empty for center date: {center.date()}")
                continue
            
            if window[ohlc_cols].isnull().any().any():
                 st.warning(f"Skipping chart for {center.date()}: contains NaN OHLC data.")
                 continue


            # Get the exact index value of the center date in the current window
            center_date_norm = center.normalize()
            
            # --- FIX: Create sparse labels list (only center date labeled) ---
            sparse_labels = [""] * len(window.index)
            
            try:
                center_loc = window.index.get_loc(center_date_norm)
                
                # Insert the short-form date string only at that position
                formatted_center_date = center.strftime('%b %d %Y')
                sparse_labels[center_loc] = formatted_center_date
            except KeyError:
                st.warning(f"Center date {center.date()} not found in window index for plotting.")
                pass
            # --- END FIX ---
            
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=window.index,
                        open=window["Open"],
                        high=window["High"],
                        low=window["Low"],
                        close=window["Close"],
                    )
                ]
            )
            
            # Add vertical dotted gray line at the match date (no annotation text)
            fig.add_vline(
                x=center,
                line_width=1,
                line_dash="dot",
                line_color="gray",
            )

            fig.update_layout(
                title=f"SPY ±3 Months Around {center.date()}",
                xaxis_title="Date",
                yaxis_title="Price",
                height=400,
                
                xaxis={
                    'type': 'category',
                    'rangeslider': {'visible': False},
                    
                    # 1. Use ALL dates as tick positions (tickvals)
                    'tickvals': window.index,  
                    
                    # 2. Use the sparse list (ticktext)
                    'ticktext': sparse_labels,  
                    
                    # Ensure the label is readable if it were visible
                    'tickangle': 0 
                }
            )

            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
