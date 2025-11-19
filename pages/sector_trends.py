import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px  # Added for the contribution chart
import datetime

SECTOR_ETFS = [
    "IBB", "IHI", "ITA", "ITB", "IYR", "KRE", "OIH", "SMH", "VNQ",
    "XBI", "XHB", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP",
    "XLU", "XLV", "XLY", "XME", "XOP", "XRT", "GLD", "CEF", "SLV", "BTC-USD",
    "ETH-USD", "UNG", "UVXY",
    "SPY", "QQQ", "IWM", "DIA", "SMH",
]

CORE_TICKERS = ["SPY", "QQQ", "IWM", "SMH", "DIA"]
CSV_PATH = "seasonal_ranks.csv"

def clear_all_caches():
    load_sector_metrics.clear()
    load_core_distance_frame.clear()
    load_seasonal_map.clear()
    load_spy_ohlc.clear()

# -----------------------------------------------------------------------------
# CSV / SEASONAL HELPERS
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_seasonal_map():
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        # Create a dummy dataframe if file missing to prevent crash during copy-paste testing
        return {}

    if df.empty:
        return {}

    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])
    df["MD"] = df["Date"].apply(lambda x: (x.month, x.day))
    
    output_map = {}
    for ticker, group in df.groupby("ticker"):
        output_map[ticker] = pd.Series(
            group.seasonal_rank.values, index=group.MD
        ).to_dict()

    return output_map

def get_sznl_val(ticker, target_date, sznl_map):
    if ticker not in sznl_map:
        return np.nan
    md = (target_date.month, target_date.day)
    return sznl_map[ticker].get(md, np.nan)

# -----------------------------------------------------------------------------
# METRICS & CALCULATIONS
# -----------------------------------------------------------------------------

def percentile_rank(series: pd.Series, value) -> float:
    s = series.dropna().values
    if s.size == 0: return np.nan
    if isinstance(value, (pd.Series, np.ndarray, list, tuple)):
        arr = np.asarray(value).ravel()
        v = float(arr[-1]) if arr.size > 0 else np.nan
    else:
        try: v = float(value)
        except: return np.nan
    if np.isnan(v): return np.nan
    return float((s <= v).sum() / s.size * 100.0)

@st.cache_data(show_spinner=True)
def load_sector_metrics(tickers):
    sznl_map = load_seasonal_map()
    today = datetime.datetime.now()
    rows = []

    for t in tickers:
        try:
            df = yf.download(t, period="max", interval="1d", auto_adjust=True, progress=False)
        except: continue
        if df.empty: continue
        
        col_name = "Adj Close" if "Adj Close" in df.columns else "Close"
        if col_name not in df.columns: continue
        close = df[col_name].dropna()
        if close.empty: continue

        ma5, ma20, ma50, ma200 = [close.rolling(w).mean() for w in [5, 20, 50, 200]]
        
        dist5 = (close - ma5) / ma5 * 100.0
        dist20 = (close - ma20) / ma20 * 100.0
        dist50 = (close - ma50) / ma50 * 100.0
        dist200 = (close - ma200) / ma200 * 100.0

        vals = [d.dropna().iloc[-1] if not d.dropna().empty else np.nan 
                for d in [dist5, dist20, dist50, dist200]]
        
        ranks = [percentile_rank(d, v) for d, v in zip([dist5, dist20, dist50, dist200], vals)]

        rows.append({
            "Ticker": t,
            "Price": float(close.iloc[-1]),
            "Sznl": get_sznl_val(t, today, sznl_map),
            "PctRank5": ranks[0],
            "PctRank20": ranks[1],
            "PctRank50": ranks[2],
            "PctRank200": ranks[3],
        })

    if not rows: return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    for col in df_out.columns:
        if col != "Ticker": df_out[col] = pd.to_numeric(df_out[col], errors="coerce")

    if "Price" in df_out.columns: df_out["Price"] = df_out["Price"].round(2)
    for col in ["Sznl", "PctRank5", "PctRank20", "PctRank50", "PctRank200"]:
        if col in df_out.columns: df_out[col] = df_out[col].round(1)

    if "PctRank200" in df_out.columns:
        df_out = df_out.sort_values("PctRank200", ascending=False, ignore_index=True)
    
    return df_out

@st.cache_data(show_spinner=True)
def load_spy_ohlc():
    df = yf.download("SPY", period="max", interval="1d", auto_adjust=False, progress=False)
    if df.empty: return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    cols = ["Open", "High", "Low", "Close"]
    return df[[c for c in cols if c in df.columns]].dropna()

@st.cache_data(show_spinner=True)
def load_core_distance_frame():
    sznl_map = load_seasonal_map()
    all_feats = []
    close_series_map = {} 

    for t in CORE_TICKERS:
        try:
            df = yf.download(t, period="max", interval="1d", auto_adjust=True, progress=False)
        except: continue
        
        col_name = "Adj Close" if "Adj Close" in df.columns else "Close"
        if col_name not in df.columns: continue
        close = df[col_name].dropna()
        if close.empty: continue

        close_series_map[t] = close.copy() 

        # 1. Calculate Raw Distances
        dists = pd.DataFrame(index=close.index)
        for w in [5, 20, 50, 200]:
            ma = close.rolling(w).mean()
            raw_dist = (close - ma) / ma * 100.0
            dists[f"raw_dist{w}"] = raw_dist
            
        # 2. Convert Distances to Percentile Ranks (0.0 to 1.0)
        ranked_dists = dists.rank(pct=True)
        
        feats = pd.DataFrame(index=close.index)
        for w in [5, 20, 50, 200]:
            feats[f"{t}_dist{w}"] = ranked_dists[f"raw_dist{w}"]

        # 3. Add Seasonal Predictor (Raw 0-100)
        t_sznl_dict = sznl_map.get(t, {})
        feats[f"{t}_sznl"] = [t_sznl_dict.get((m, d), np.nan) 
                              for m, d in zip(feats.index.month, feats.index.day)]
        
        all_feats.append(feats)

    if not all_feats: return pd.DataFrame()

    core_df = all_feats[0]
    for feats in all_feats[1:]:
        core_df = core_df.join(feats, how="inner")

    core_df = core_df.dropna().sort_index()
    core_df.index = pd.to_datetime(core_df.index)
    core_df.index.name = "Date"

    horizons = [2, 5, 10, 21, 63, 126, 252]
    for t in CORE_TICKERS:
        close = close_series_map.get(t)
        if close is not None:
            close = close.reindex(core_df.index)
            for h in horizons:
                core_df[f"{t}_fwd_{h}d"] = close.shift(-h) / close - 1.0

    return core_df

def compute_distance_matches(core_df: pd.DataFrame,
                             n_matches: int = 20,
                             exclude_last_n: int = 63,
                             min_spacing_days: int = 21):
    """
    Returns:
        (matches_df, contribution_df)
        contribution_df contains the Squared Errors per feature for the selected matches
        allowing the user to see which feature drove the distance calculation.
    """
    if core_df.empty: return pd.DataFrame(), pd.DataFrame()

    df = core_df.copy().sort_index()
    
    feature_cols = [c for c in df.columns if "dist" in c or "sznl" in c]
    sznl_cols = [c for c in feature_cols if "sznl" in c]
    
    spy_fwd_cols = [c for c in df.columns if c.startswith("SPY_fwd_")]
    df = df.dropna(subset=feature_cols + spy_fwd_cols)

    if len(df) <= exclude_last_n + 1: return pd.DataFrame(), pd.DataFrame()

    X_full = df[feature_cols].astype(float).values
    
    # --- SCALING LOGIC ---
    sznl_indices = [df[feature_cols].columns.get_loc(c) for c in sznl_cols]
    
    X_scaled = X_full.copy()
    # Scale seasonality (0-100) down to (0-1) to match MA Percentile Ranks
    X_scaled[:, sznl_indices] /= 100.0 
    # ---------------------

    target_vector = X_scaled[-1]
    history_matrix = X_scaled[:-exclude_last_n]
    
    hist_index = df.index[:-exclude_last_n]
    valid_mask = (hist_index.year > 1997) & (hist_index.year != 2020)
    
    if not valid_mask.any(): return pd.DataFrame(), pd.DataFrame()

    history_matrix = history_matrix[valid_mask]
    hist_index = hist_index[valid_mask]

    # Euclidean Distance
    diffs = history_matrix - target_vector
    squared_diffs = diffs ** 2
    dists = np.sqrt(squared_diffs.sum(axis=1))
    
    results = df.loc[hist_index].copy()
    results["distance"] = dists
    results = results.sort_values("distance")

    selected_dates = []
    for dt in results.index:
        if all(abs((dt - prev).days) > min_spacing_days for prev in selected_dates):
            selected_dates.append(dt)
            if len(selected_dates) >= n_matches: break

    if not selected_dates: return pd.DataFrame(), pd.DataFrame()

    matches = results.loc[selected_dates].copy().reset_index()
    if "index" in matches.columns: matches = matches.rename(columns={"index": "Date"})
    
    # --- CALCULATE CONTRIBUTION (SQUARED ERROR) FOR VISUALIZATION ---
    # We need to grab the scaled vectors for the selected dates to show true contribution
    # Find indices in the *original* valid history matrix that correspond to selected dates
    
    # Map dates to integer locations in hist_index
    locs = [hist_index.get_loc(d) for d in selected_dates]
    
    # Get the scaled vectors for these dates
    selected_scaled_vectors = history_matrix[locs]
    
    # Calculate squared difference per feature: (x_i - y_i)^2
    contrib_matrix = (selected_scaled_vectors - target_vector) ** 2
    
    contrib_df = pd.DataFrame(
        contrib_matrix, 
        index=selected_dates, 
        columns=feature_cols
    )
    
    return matches, contrib_df

def main():
    st.title("Sector ETF Trend Dashboard")
    st.write("Matching based on **Percentile Rank** of MA Extensions + Seasonality.")

    if st.button("Refresh data"):
        clear_all_caches()
        st.rerun()

    with st.spinner("Loading data..."):
        table = load_sector_metrics(sorted(set(SECTOR_ETFS)))

    if table.empty:
        st.error("No data available.")
        return

    st.subheader("Sector & Index ETFs")

    def highlight_pct_rank(val):
        if pd.isna(val): return ""
        if val > 90: return "background-color: #ffcccc; color: #8b0000;"
        if val < 15: return "background-color: #ccffcc; color: #006400;"
        return ""

    def highlight_sznl(val):
        if pd.isna(val): return ""
        if val > 85: return "background-color: #ccffcc; color: #006400;"
        if val < 15: return "background-color: #ffcccc; color: #8b0000;"
        return ""

    format_dict = {
        "Price": "{:.2f}", "Sznl": "{:.1f}",
        "PctRank5": "{:.1f}", "PctRank20": "{:.1f}", 
        "PctRank50": "{:.1f}", "PctRank200": "{:.1f}",
    }

    styled = (
        table.style
        .format(format_dict)
        .map(highlight_pct_rank, subset=["PctRank5", "PctRank20", "PctRank50", "PctRank200"])
        .map(highlight_sznl, subset=["Sznl"])
    )

    st.dataframe(styled, use_container_width=True)

    # -------- Distance-matching section --------
    st.subheader("Historical Matches (MA Rank + Seasonality)")
    
    with st.spinner("Computing matches..."):
        core_df = load_core_distance_frame()
        match_table_raw, contrib_df = compute_distance_matches(core_df)

    if match_table_raw.empty:
        st.warning("Not enough data.")
    else:
        st.write("20 closest historical dates based on Euclidean distance of ranks (0-1 scale).")
        
        horizons = [2, 5, 10, 21, 63, 126, 252]
        display_fwd_cols = [f"Fwd {h}d" for h in horizons]
        
        avg_rows = []
        for t in CORE_TICKERS:
            vals = match_table_raw[[f"{t}_fwd_{h}d" for h in horizons]].mean()
            row = {"Date": f"Avg {t}", "distance": ""}
            for i, h in enumerate(horizons):
                row[f"Fwd {h}d"] = vals[i]
            avg_rows.append(row)
        
        avg_rows[0]["Date"] = "Average"
        avg_rows[0]["distance"] = match_table_raw["distance"].mean()

        matches_disp = match_table_raw[["Date", "distance"]].copy()
        spy_fwds = match_table_raw[[f"SPY_fwd_{h}d" for h in horizons]].values
        matches_disp[display_fwd_cols] = spy_fwds
        
        final_df = pd.concat([pd.DataFrame(avg_rows), matches_disp], ignore_index=True)
        final_df["Date"] = final_df["Date"].apply(lambda x: x.strftime("%b %d %Y") if isinstance(x, pd.Timestamp) else x)
        
        for c in display_fwd_cols:
            final_df[c] = pd.to_numeric(final_df[c]).apply(lambda x: f"{x*100:.2f}" if pd.notna(x) else "")
            
        final_df["distance"] = pd.to_numeric(final_df["distance"]).apply(lambda x: f"{x:.4f}" if pd.notna(x) else "")
        
        st.dataframe(final_df, use_container_width=True)

        # ---------------------------------------------------------------------
        # 2. FACTOR CONTRIBUTION VISUALIZER
        # ---------------------------------------------------------------------
        st.write("---")
        st.subheader("Why these matches?")
        st.markdown("""
        **Feature Contribution (Squared Error)**
        
        The chart below breaks down the Euclidean distance. 
        * **Taller Bars** = Worse match (higher distance).
        * **Colors** = How much each specific factor (MA Rank or Seasonality) contributed to the mismatch.
        * If a specific color dominates the bar, that factor is the "outlier" driving the distance.
        """)
        
        if not contrib_df.empty:
            # Melt for Plotly Express
            contrib_reset = contrib_df.reset_index().rename(columns={"index": "Match Date"})
            contrib_melted = contrib_reset.melt(id_vars="Match Date", var_name="Feature", value_name="Squared Error")
            
            # Shorten Date for axis
            contrib_melted["Match Date"] = contrib_melted["Match Date"].dt.strftime('%Y-%m-%d')
            
            fig_contrib = px.bar(
                contrib_melted, 
                x="Match Date", 
                y="Squared Error", 
                color="Feature",
                title="Contribution to Euclidean Distance (Squared Error per Feature)",
                height=500
            )
            fig_contrib.update_layout(xaxis={'categoryorder':'total ascending'}) # Sort by total distance (best matches first)
            st.plotly_chart(fig_contrib, use_container_width=True)

        # ---------------------------------------------------------------------
        # 1. REVERTED CANDLESTICK LOGIC
        # ---------------------------------------------------------------------
        st.subheader("SPY Candles Around Top 10 Match Dates")

        spy_ohlc = load_spy_ohlc()
        
        if spy_ohlc.empty:
            st.warning("Could not load SPY OHLC data. Check `load_spy_ohlc` or internet connection.")
        else:
            # Ensure OHLC columns are numeric right after loading/before slicing
            ohlc_cols = ["Open", "High", "Low", "Close"]
            for col in ohlc_cols:
                if col in spy_ohlc.columns:
                    spy_ohlc[col] = pd.to_numeric(spy_ohlc[col], errors='coerce')
            
            spy_ohlc = spy_ohlc.dropna(subset=ohlc_cols)

            if spy_ohlc.empty:
                st.warning("SPY OHLC data was lost after cleaning (NaNs or bad types).")
            else:
                # Normalize SPY index to date-only and strip any timezone info
                spy_ohlc = spy_ohlc.copy()
                spy_ohlc.index = pd.to_datetime(spy_ohlc.index).normalize().tz_localize(None)

                # The raw matches still hold the datetime objects needed for slicing
                # Note: match_table_raw is the variable from compute_distance_matches
                top10 = match_table_raw.head(10).copy() 

                # Normalize Date column in top10 
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
                    
                    # --- Create sparse labels list (only center date labeled) ---
                    sparse_labels = [""] * len(window.index)
                    
                    try:
                        center_loc = window.index.get_loc(center_date_norm)
                        
                        # Insert the short-form date string only at that position
                        formatted_center_date = center.strftime('%b %d %Y')
                        sparse_labels[center_loc] = formatted_center_date
                    except KeyError:
                        # If exact center date isn't trading day, try to find closest or pass
                        pass
                    
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
