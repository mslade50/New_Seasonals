import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import date, timedelta

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def compute_atr(df, window=14):
    df = df.copy()
    df["previous_close"] = df["Close"].shift(1)
    df["TR"] = df[["High", "previous_close"]].max(axis=1) - df[["Low", "previous_close"]].min(axis=1)
    df["ATR"] = df["TR"].rolling(window=window).mean()
    df["ATR%"] = (df["ATR"] / df["Close"]) * 100
    return df

def get_default_cycle_index():
    year = dt.date.today().year
    rem = year % 4
    if rem == 0: return 1 # Election
    if rem == 1: return 3 # Post-Election
    if rem == 2: return 4 # Midterm
    if rem == 3: return 2 # Pre-Election
    return 0

# -----------------------------------------------------------------------------
# SEASONAL RANK CALCULATION (Correct Forward Returns)
# -----------------------------------------------------------------------------

def calculate_seasonal_rank(df, cycle_label, cycle_start_mapping, cutoff_year):
    """
    Calculate 0-100 seasonal rank for each day_count using correct forward log returns.
    Uses data strictly prior to cutoff_year (walk-forward safe).
    Returns a Series indexed by day_count.
    """
    hist = df[df['year'] < cutoff_year].copy()
    if hist.empty:
        return pd.Series(dtype=float)

    # Correct forward return: ln(Close_t+w / Close_t)
    for w in [5, 10, 21]:
        hist[f'Fwd_{w}d'] = np.log(hist['Close'].shift(-w) / hist['Close'])

    fwd_cols = [f'Fwd_{w}d' for w in [5, 10, 21]]

    # All-years profile
    stats_all = hist.groupby('day_count')[fwd_cols].mean()
    rank_all = stats_all.rank(pct=True) * 100

    # Cycle-specific profile
    if cycle_label != "All Years":
        start_yr = cycle_start_mapping[cycle_label]
        valid_years = [start_yr + i * 4 for i in range(30)]
        cycle_data = hist[hist['year'].isin(valid_years)]
        if not cycle_data.empty:
            stats_cycle = cycle_data.groupby('day_count')[fwd_cols].mean()
            rank_cycle = stats_cycle.rank(pct=True) * 100
        else:
            rank_cycle = rank_all.copy()
    else:
        rank_cycle = rank_all.copy()

    # Reindex to cover full range of possible day_counts
    max_day = max(hist['day_count'].max(), 253)
    full_idx = pd.RangeIndex(start=1, stop=max_day + 1)
    rank_all = rank_all.reindex(full_idx).interpolate(method='nearest').fillna(50)
    rank_cycle = rank_cycle.reindex(full_idx).interpolate(method='nearest').fillna(50)

    # Weighted average: 25% all years, 75% cycle
    avg_all = rank_all.mean(axis=1)
    avg_cycle = rank_cycle.mean(axis=1)
    final = (avg_all + 3 * avg_cycle) / 4

    # Smooth
    return final.rolling(5, center=True, min_periods=1).mean()


# -----------------------------------------------------------------------------
# MAIN CHART LOGIC
# -----------------------------------------------------------------------------

def calculate_path(df, cycle_label, cycle_start_mapping):
    if df.empty: return pd.Series()

    if cycle_label == "All Years":
        cycle_data = df.copy()
    else:
        cycle_start = cycle_start_mapping[cycle_label]
        years_in_cycle = [cycle_start + i * 4 for i in range(30)] 
        cycle_data = df[df["year"].isin(years_in_cycle)].copy()

    if "week_of_month_5day" in cycle_data.columns:
        cycle_data.loc[cycle_data["week_of_month_5day"] > 4, "week_of_month_5day"] = 4

    avg_path = (
        cycle_data.groupby("day_count")["log_return"]
        .mean()
        .cumsum()
        .apply(np.exp) - 1
    )
    return avg_path

# -----------------------------------------------------------------------------
# RECENT PERFORMANCE ANALYSIS (Table Only - Weighted Logic)
# -----------------------------------------------------------------------------
def recent_performance_analysis(df, cycle_label, cycle_start_mapping):
    """
    Calculates percentile ranks for returns and seasonality for the last 21 days.
    Seasonality uses a weighted average: (All_Years + 3 * Cycle_Years) / 4.
    """
    df = df.copy()
    
    # 1. Calculate Historical Returns Windows (Standard Overbought/Oversold Ranks)
    for w in [5, 10, 21]:
        df[f'Ret_{w}d'] = df['Close'].pct_change(w)
        df[f'Rank_{w}d'] = df[f'Ret_{w}d'].expanding(min_periods=252).rank(pct=True) * 100
        
    df['Daily_Ret'] = df['Close'].pct_change() * 100

    # 2. Calculate Weighted Seasonal Rank
    # A. Get Baseline (All Years)
    daily_seasonality_all = df.groupby('day_count')['log_return'].mean()

    # B. Get Cycle Specific
    if cycle_label == "All Years":
        combined_seasonality = daily_seasonality_all
    else:
        start_yr = cycle_start_mapping[cycle_label]
        valid_years = [start_yr + i * 4 for i in range(30)]
        cycle_df = df[df['year'].isin(valid_years)].copy()
        daily_seasonality_cycle = cycle_df.groupby('day_count')['log_return'].mean()
        combined_seasonality = (daily_seasonality_all + (daily_seasonality_cycle * 3)) / 4

    seasonal_score_map = combined_seasonality.rank(pct=True) * 100
    df['Seasonal_Rank'] = df['day_count'].map(seasonal_score_map)

    # 3. Filter Last 21 Days
    recent = df.tail(21).copy()
    
    # -------------------------------------------------------------------------
    # DISPLAY TABLE
    # -------------------------------------------------------------------------
    st.markdown(f"### 📋 Recent Performance Data")
    
    display_cols = ['Close', 'Daily_Ret', 'Volume', 'Rank_5d', 'Rank_10d', 'Rank_21d', 'Seasonal_Rank']
    table_df = recent[display_cols].sort_index(ascending=False)
    
    def color_ret_rank(val):
        if pd.isna(val): return ''
        if val >= 90: return 'color: #ff4444; font-weight: bold;'
        if val >= 80: return 'color: #ff8888;'
        if val <= 10: return 'color: #00ff00; font-weight: bold;'
        if val <= 20: return 'color: #90ee90;'
        return 'color: #cccccc;'

    def color_seasonal_rank(val):
        if pd.isna(val): return ''
        if val >= 85: return 'color: #00ff00; font-weight: bold;'
        if val >= 65: return 'color: #90ee90;'
        if val <= 15: return 'color: #ff4444; font-weight: bold;'
        if val <= 35: return 'color: #ff8888;'
        return 'color: #cccccc;'
    
    def color_ret(val):
        color = '#ff6666' if val < 0 else '#66ff66'
        return f'color: {color}'

    styler = table_df.style.format({
        "Close": "{:.2f}",
        "Daily_Ret": "{:+.2f}%",
        "Volume": "{:,.0f}",
        "Rank_5d": "{:.0f}",
        "Rank_10d": "{:.0f}",
        "Rank_21d": "{:.0f}",
        "Seasonal_Rank": "{:.0f}"
    })
    
    styler = styler.map(color_ret_rank, subset=['Rank_5d', 'Rank_10d', 'Rank_21d'])
    styler = styler.map(color_seasonal_rank, subset=['Seasonal_Rank'])
    styler = styler.map(color_ret, subset=['Daily_Ret'])
    styler = styler.bar(subset=['Volume'], color='#444444') 

    st.caption(f"**Seasonal Rank** is based on a weighted average: 75% {cycle_label} data, 25% All Years data. >85 (Green) indicates historically strong days.")
    st.dataframe(styler, use_container_width=True, height=600)


# -----------------------------------------------------------------------------
# MAIN LOGIC
# -----------------------------------------------------------------------------
def seasonals_chart(ticker, cycle_label, enable_time_travel, reference_year, show_all_years_line):
    cycle_start_mapping = {
        "Election": 1952,
        "Pre-Election": 1951,
        "Post-Election": 1953,
        "Midterm": 1950
    }

    # Data Fetching
    end_date_fetch = dt.datetime.now() + timedelta(days=5)
    spx = yf.download(ticker, period="max", end=end_date_fetch, progress=False)

    if spx.empty:
        st.error(f"No data found for {ticker}.")
        return
    
    if isinstance(spx.columns, pd.MultiIndex):
        spx.columns = spx.columns.get_level_values(0)

    # Global ATR & Feature Engineering
    spx = compute_atr(spx)
    
    current_atr_pct = 0.0
    if not spx["ATR%"].dropna().empty:
        current_atr_pct = spx["ATR%"].dropna().iloc[-1]

    spx["log_return"] = np.log(spx["Close"] / spx["Close"].shift(1))
    spx["year"] = spx.index.year
    spx["month"] = spx.index.month
    spx["day_count"] = spx.groupby("year").cumcount() + 1
    
    current_year = dt.date.today().year

    # -----------------------------------------------------------------
    # SEASONAL RANK PROFILES
    # -----------------------------------------------------------------
    # Current model rank (uses data < current_year)
    rank_profile_current = calculate_seasonal_rank(spx, cycle_label, cycle_start_mapping, current_year)

    # All-years rank (for the optional overlay)
    rank_profile_all_years = pd.Series(dtype=float)
    if show_all_years_line:
        rank_profile_all_years = calculate_seasonal_rank(spx, "All Years", cycle_start_mapping, current_year)

    # Time travel rank (uses data < reference_year)
    rank_profile_historical = pd.Series(dtype=float)
    if enable_time_travel:
        rank_profile_historical = calculate_seasonal_rank(spx, cycle_label, cycle_start_mapping, reference_year)

    # -----------------------------------------------------------------
    # PATH CALCULATIONS (unchanged logic)
    # -----------------------------------------------------------------
    # 1. CURRENT MODEL CONSTRUCTION
    df_all_history = spx[spx["year"] < current_year].copy()
    path_current_avg = calculate_path(df_all_history, cycle_label, cycle_start_mapping)
    
    path_current_all_years = pd.Series()
    if show_all_years_line:
        path_current_all_years = calculate_path(df_all_history, "All Years", cycle_start_mapping)

    df_current_year = spx[spx["year"] == current_year].copy()
    path_current_realized = pd.Series()
    if not df_current_year.empty:
        path_current_realized = df_current_year["log_return"].cumsum().apply(np.exp) - 1

    # 2. TIME TRAVEL CONSTRUCTION
    path_historical_avg = pd.Series()
    path_ref_realized = pd.Series()
    
    if enable_time_travel:
        df_historical_pool = spx[spx["year"] < reference_year].copy()
        path_historical_avg = calculate_path(df_historical_pool, cycle_label, cycle_start_mapping)
        
        df_ref_year = spx[spx["year"] == reference_year].copy()
        if not df_ref_year.empty:
            path_ref_realized = df_ref_year["log_return"].cumsum().apply(np.exp) - 1

    # -------------------------------------------------------------------------
    # DATE MAPPING LOGIC
    # -------------------------------------------------------------------------
    map_year = reference_year if enable_time_travel else current_year
    theoretical_dates = pd.bdate_range(start=f"{map_year}-01-01", end=f"{map_year}-12-31")
    date_map = {i+1: d.strftime("%b %d") for i, d in enumerate(theoretical_dates)}

    def get_date_labels(series_index):
        return [date_map.get(i, f"Day {i}") for i in series_index]

    # -------------------------------------------------------------------------
    # HELPER: Build customdata array with date + seasonal rank
    # -------------------------------------------------------------------------
    def build_customdata(day_indices, rank_profile, date_labels=None):
        """
        Returns a list of [date_label, seasonal_rank] for each day index.
        """
        if date_labels is None:
            date_labels = get_date_labels(day_indices)
        result = []
        for i, day in enumerate(day_indices):
            label = date_labels[i] if i < len(date_labels) else f"Day {day}"
            rank = rank_profile.get(day, np.nan) if not rank_profile.empty else np.nan
            result.append([label, rank])
        return result

    HOVER_WITH_RANK = (
        "<b>%{customdata[0]}</b><br>"
        "Day: %{x}<br>"
        "Return: %{y:.2%}<br>"
        "Seasonal Rank: %{customdata[1]:.0f}"
        "<extra></extra>"
    )

    # -------------------------------------------------------------------------
    # PLOTTING
    # -------------------------------------------------------------------------
    fig = go.Figure()

    # A. Current Cycle Model (Orange)
    fig.add_trace(go.Scatter(
        x=path_current_avg.index,
        y=path_current_avg.values,
        mode="lines",
        name=f"Current Model ({cycle_label})",
        line=dict(color="#FF8C00", width=3),
        customdata=build_customdata(path_current_avg.index, rank_profile_current),
        hovertemplate=HOVER_WITH_RANK
    ))

    # B. Current All Years Model (Light Blue - Optional)
    if not path_current_all_years.empty:
        fig.add_trace(go.Scatter(
            x=path_current_all_years.index,
            y=path_current_all_years.values,
            mode="lines",
            name="Current Model (All Years)",
            line=dict(color="lightblue", width=1, dash='dot'),
            customdata=build_customdata(path_current_all_years.index, rank_profile_all_years),
            hovertemplate=HOVER_WITH_RANK
        ))

    # C. Historical Model (Gold Dashed - Time Travel)
    if enable_time_travel and not path_historical_avg.empty:
        fig.add_trace(go.Scatter(
            x=path_historical_avg.index,
            y=path_historical_avg.values,
            mode="lines",
            name=f"Model in {reference_year} (Pre-{reference_year} Data)",
            line=dict(color="#FCD12A", width=2, dash='dash'),
            customdata=build_customdata(path_historical_avg.index, rank_profile_historical),
            hovertemplate=HOVER_WITH_RANK
        ))

    # D. Current Realized (Green)
    if not path_current_realized.empty:
        realized_dates = [d.strftime("%b %d") for d in df_current_year.index]
        realized_day_counts = df_current_year["day_count"].values
        cdata = build_customdata(realized_day_counts, rank_profile_current, date_labels=realized_dates)

        fig.add_trace(go.Scatter(
            x=np.arange(1, len(path_current_realized) + 1),
            y=path_current_realized.values,
            mode="lines",
            name=f"{current_year} Realized (YTD)",
            line=dict(color="#39FF14", width=2),
            customdata=cdata,
            hovertemplate=HOVER_WITH_RANK
        ))

    # E. Historical Realized (Cyan - Time Travel)
    if enable_time_travel and not path_ref_realized.empty:
        realized_dates_ref = [d.strftime("%b %d") for d in df_ref_year.index]
        ref_day_counts = df_ref_year["day_count"].values
        cdata_ref = build_customdata(ref_day_counts, rank_profile_historical, date_labels=realized_dates_ref)

        fig.add_trace(go.Scatter(
            x=np.arange(1, len(path_ref_realized) + 1),
            y=path_ref_realized.values,
            mode="lines",
            name=f"{reference_year} Realized",
            line=dict(color="#00FFFF", width=2),
            customdata=cdata_ref,
            hovertemplate=HOVER_WITH_RANK
        ))

    # -------------------------------------------------------------------------
    # MARKERS & DATES
    # -------------------------------------------------------------------------
    today = dt.date.today()
    
    if enable_time_travel:
        target_year = reference_year
        try:
            target_date_start = dt.date(target_year, today.month, today.day)
        except ValueError:
            target_date_start = dt.date(target_year, 2, 28)
        df_context = spx[spx["year"] == target_year]
        model_to_plot_on = path_historical_avg if not path_historical_avg.empty else path_current_avg
        marker_color = "#FCD12A" 
    else:
        target_year = current_year
        target_date_start = today
        df_context = spx[spx["year"] == target_year]
        model_to_plot_on = path_current_avg
        marker_color = "white"

    day_count_marker = None
    if not df_context.empty:
        closest_idx = df_context.index.searchsorted(dt.datetime.combine(target_date_start, dt.time.min))
        if closest_idx >= len(df_context):
            closest_idx = len(df_context) - 1
        
        day_count_marker = df_context.iloc[closest_idx]["day_count"]

        # --- PLOT MARKERS ON MAIN CYCLE LINE ---
        if day_count_marker in model_to_plot_on.index:
            y_val = model_to_plot_on.get(day_count_marker)
            
            fig.add_trace(go.Scatter(
                x=[day_count_marker],
                y=[y_val],
                mode="markers",
                name=f"Current Date ({target_date_start.strftime('%b %d')})",
                marker=dict(color=marker_color, size=8, line=dict(width=1, color="black")),
                showlegend=False,
                hoverinfo="skip"
            ))

            future_dates = pd.bdate_range(start=target_date_start, periods=30)
            offsets = [5, 10, 21]
            
            for offset in offsets:
                target_idx = day_count_marker + offset
                if target_idx in model_to_plot_on.index:
                    proj_y = model_to_plot_on.get(target_idx)
                    d_label = future_dates[offset].strftime("%b %d") if offset < len(future_dates) else "N/A"
                        
                    fig.add_trace(go.Scatter(
                        x=[target_idx],
                        y=[proj_y],
                        mode="markers",
                        name=f"T+{offset} ({d_label})", 
                        marker=dict(color=marker_color, size=5, symbol="diamond"),
                        hovertemplate=f"<b>{d_label}</b><br>T+{offset}<extra></extra>"
                    ))

        # --- PLOT MARKERS ON 'ALL YEARS' LINE (If Visible) ---
        if not path_current_all_years.empty:
            if day_count_marker in path_current_all_years.index:
                y_val_all = path_current_all_years.get(day_count_marker)
                fig.add_trace(go.Scatter(
                    x=[day_count_marker],
                    y=[y_val_all],
                    mode="markers",
                    marker=dict(color="white", size=5, line=dict(width=1, color="white")),
                    showlegend=False,
                    hoverinfo="skip"
                ))

                for offset in offsets:
                    target_idx = day_count_marker + offset
                    if target_idx in path_current_all_years.index:
                        proj_y_all = path_current_all_years.get(target_idx)
                        fig.add_trace(go.Scatter(
                            x=[target_idx],
                            y=[proj_y_all],
                            mode="markers",
                            marker=dict(color="white", size=5, symbol="diamond"), 
                            showlegend=False,
                            hoverinfo="skip"
                        ))

    # Layout
    title_suffix = f"vs {reference_year}" if enable_time_travel else ""
    fig.update_layout(
        height=800,
        title=f"Seasonal Analysis: {ticker} {title_suffix}",
        xaxis_title="Trading Day of Year",
        yaxis_title="Cumulative Return",
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        legend=dict(
            bgcolor="rgba(20,20,20,0.8)", 
            font=dict(color="white"),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode="x unified"
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # DETAILED HISTORY TABLE
    # -------------------------------------------------------------------------
    st.divider()
    
    if enable_time_travel:
        st.subheader(f"📜 Historical Returns (Pre-{reference_year})")
        st.caption(f"Stats exclude {reference_year} and later. Color scale based on **Today's** ATR ({current_atr_pct:.2f}%).")
        cutoff_year = reference_year
    else:
        st.subheader(f"📜 Historical Returns (Pre-{current_year})")
        st.caption(f"Stats exclude {current_year}. Color scale based on **Current** ATR ({current_atr_pct:.2f}%).")
        cutoff_year = current_year

    if day_count_marker:
        spx_full = spx.copy()
        spx_full['Fwd_5d'] = spx_full['Close'].shift(-5) / spx_full['Close'] - 1
        spx_full['Fwd_10d'] = spx_full['Close'].shift(-10) / spx_full['Close'] - 1
        spx_full['Fwd_21d'] = spx_full['Close'].shift(-21) / spx_full['Close'] - 1

        spx_full['Daily_Pct'] = spx_full['Close'].pct_change()

        spx_full['rv_5'] = spx_full['Daily_Pct'].rolling(window=5).std().shift(-5) * np.sqrt(252) * 100
        spx_full['rv_10'] = spx_full['Daily_Pct'].rolling(window=10).std().shift(-10) * np.sqrt(252) * 100
        spx_full['rv_21'] = spx_full['Daily_Pct'].rolling(window=21).std().shift(-21) * np.sqrt(252) * 100

        daily_snapshots = spx_full[spx_full['day_count'] == day_count_marker].copy()
        display_df = daily_snapshots[daily_snapshots['year'] < cutoff_year].copy()

        if not display_df.empty:
            display_df = display_df[['year', 'Fwd_5d', 'Fwd_10d', 'Fwd_21d', 'rv_5', 'rv_10', 'rv_21']]
            display_df['Fwd_5d'] = display_df['Fwd_5d'] * 100
            display_df['Fwd_10d'] = display_df['Fwd_10d'] * 100
            display_df['Fwd_21d'] = display_df['Fwd_21d'] * 100
            display_df = display_df.sort_values('year', ascending=False)

            if cycle_label != "All Years":
                start_yr = cycle_start_mapping.get(cycle_label)
                highlight_years = [start_yr + i * 4 for i in range(30)] 
            else:
                highlight_years = []

            # --- STATS ---
            st.markdown(f"##### 🎯 Fwd Return Statistics")
            
            def calculate_stats_row(sub_df):
                if sub_df.empty:
                    return {k: np.nan for k in ["n", "5_median", "5_mean", "5_pospct", "rv_5", 
                                                "10_median", "10_mean", "10_pospct", "rv_10", 
                                                "21_median", "21_mean", "21_pospct", "rv_21"]}
                
                res = {"n": int(len(sub_df))}
                for d in [5, 10, 21]:
                    ret_col = f"Fwd_{d}d"
                    rv_col = f"rv_{d}"
                    res[f"{d}_median"] = sub_df[ret_col].median()
                    res[f"{d}_mean"] = sub_df[ret_col].mean()
                    res[f"{d}_pospct"] = (sub_df[ret_col] > 0).mean() * 100
                    res[rv_col] = sub_df[rv_col].mean()
                return res

            stats_all = calculate_stats_row(display_df)
            
            if cycle_label != "All Years":
                df_cycle = display_df[display_df['year'].isin(highlight_years)]
                stats_cycle = calculate_stats_row(df_cycle)
                cycle_row_name = f"{cycle_label} Cycle"
            else:
                stats_cycle = stats_all
                cycle_row_name = f"All Years"

            stats_df = pd.DataFrame([stats_all, stats_cycle], index=[f"All History (<{cutoff_year})", cycle_row_name])
            
            ordered_cols = ["n"]
            for d in [5, 10, 21]:
                ordered_cols.extend([f"{d}_median", f"{d}_mean", f"{d}_pospct", f"rv_{d}"])
            stats_df = stats_df[ordered_cols]

            def color_pos_pct(val):
                if pd.isna(val): return ''
                if val > 80: return 'color: #90ee90; font-weight: bold;' 
                elif val < 25: return 'color: #ffcccb; font-weight: bold;' 
                return ''

            styler = stats_df.style.format({
                "n": "{:.0f}",
                "5_median": "{:.2f}%", "5_mean": "{:.2f}%", "5_pospct": "{:.1f}%", "rv_5": "{:.2f}%",
                "10_median": "{:.2f}%", "10_mean": "{:.2f}%", "10_pospct": "{:.1f}%", "rv_10": "{:.2f}%",
                "21_median": "{:.2f}%", "21_mean": "{:.2f}%", "21_pospct": "{:.1f}%", "rv_21": "{:.2f}%",
            }).map(color_pos_pct, subset=["5_pospct", "10_pospct", "21_pospct"])

            if current_atr_pct > 0:
                for d in [5, 10, 21]:
                    vol_scale = current_atr_pct * np.sqrt(d)
                    
                    mean_limit = 1.5 * vol_scale
                    styler = styler.background_gradient(
                        subset=[f"{d}_mean"], 
                        cmap="RdYlGn", 
                        vmin=-mean_limit, 
                        vmax=mean_limit
                    )
                    
                    median_limit = 1.0 * vol_scale
                    styler = styler.background_gradient(
                        subset=[f"{d}_median"], 
                        cmap="RdYlGn", 
                        vmin=-median_limit, 
                        vmax=median_limit
                    )

            st.dataframe(styler, use_container_width=True)

            # --- MAIN DATAFRAME RENDER ---
            def highlight_year_cell(val):
                if val in highlight_years:
                    return 'background-color: #d4af37; color: black; font-weight: bold;' 
                return ''

            st.dataframe(
                display_df.style
                .format({
                    "year": "{:.0f}",
                    "Fwd_5d": "{:+.2f}%",
                    "Fwd_10d": "{:+.2f}%",
                    "Fwd_21d": "{:+.2f}%"
                })
                .background_gradient(subset=["Fwd_5d", "Fwd_10d", "Fwd_21d"], cmap="RdYlGn", vmin=-5, vmax=5)
                .map(highlight_year_cell, subset=['year']),
                use_container_width=True,
                height=400,
                hide_index=True
            )
        else:
            st.warning(f"No historical data available prior to {cutoff_year}.")

    # -------------------------------------------------------------------------
    # RECENT PERFORMANCE ANALYSIS
    # -------------------------------------------------------------------------
    st.divider()
    recent_performance_analysis(spx, cycle_label, cycle_start_mapping)

# -----------------------------------------------------------------------------
# APP ENTRY POINT
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Seasonality Analysis")
    st.title("📊 Presidential Cycle Seasonality")

    col1, col2, col3, col4 = st.columns([1, 1, 1.5, 1])
    
    with col1:
        ticker = st.text_input("Ticker", value="SPY").upper()
    
    with col2:
        default_index = get_default_cycle_index()
        cycle_label = st.selectbox(
            "Cycle Type",
            ["All Years", "Election", "Pre-Election", "Post-Election", "Midterm"],
            index=default_index
        )
    
    with col3:
        c3_1, c3_2 = st.columns([1, 1])
        with c3_1:
            enable_time_travel = st.checkbox("Enable Time Travel", value=False)
        with c3_2:
            current_year = dt.date.today().year
            if enable_time_travel:
                reference_year = st.number_input("Compare vs Year", min_value=1950, max_value=current_year, value=current_year-1)
            else:
                reference_year = current_year
                st.write("")

    with col4:
        st.write("")
        show_all_years_line = st.checkbox("Overlay 'All Years' Avg", value=False)

    if st.button("Run Analysis", type="primary", use_container_width=True):
        try:
            with st.spinner("Calculating cycle stats..."):
                seasonals_chart(ticker, cycle_label, enable_time_travel, reference_year, show_all_years_line)
        except Exception as e:
            st.error(f"Error generating chart: {e}")

if __name__ == "__main__":
    main()
