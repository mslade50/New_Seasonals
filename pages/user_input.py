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
# RECENCY WEIGHTING (exponential decay by year)
# -----------------------------------------------------------------------------

def recency_weights(years, anchor_year, half_life):
    """Exponential decay: weight = 0.5 ** ((anchor_year - year) / half_life)."""
    if half_life is None or half_life <= 0:
        return np.ones(len(years))
    return np.power(0.5, (anchor_year - np.asarray(years, dtype=float)) / half_life)

def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values)
    if not mask.any():
        return np.nan
    v, w = values[mask], weights[mask]
    tot = w.sum()
    return float((v * w).sum() / tot) if tot > 0 else np.nan

def weighted_median(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = ~np.isnan(values)
    if not mask.any():
        return np.nan
    v, w = values[mask], weights[mask]
    order = np.argsort(v)
    v_sorted, w_sorted = v[order], w[order]
    cum_w = np.cumsum(w_sorted)
    tot = cum_w[-1]
    if tot <= 0:
        return np.nan
    idx = np.searchsorted(cum_w, tot / 2.0)
    return float(v_sorted[min(idx, len(v_sorted) - 1)])

# -----------------------------------------------------------------------------
# SEASONAL RANK CALCULATION (Correct Forward Returns)
# -----------------------------------------------------------------------------

def calculate_seasonal_rank(df, cycle_label, cycle_start_mapping, cutoff_year, half_life=None):
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
    hist['_weight'] = recency_weights(hist['year'].values, cutoff_year, half_life)

    def _wmean_by_daycount(sub):
        out = {}
        for col in fwd_cols:
            valid = sub[col].notna()
            s = sub.loc[valid]
            num = (s[col] * s['_weight']).groupby(s['day_count']).sum()
            den = s['_weight'].groupby(s['day_count']).sum()
            out[col] = num / den
        return pd.DataFrame(out)

    # All-years profile (weighted)
    stats_all = _wmean_by_daycount(hist)
    rank_all = stats_all.rank(pct=True) * 100

    # Cycle-specific profile (weighted)
    if cycle_label != "All Years":
        start_yr = cycle_start_mapping[cycle_label]
        valid_years = [start_yr + i * 4 for i in range(30)]
        cycle_data = hist[hist['year'].isin(valid_years)]
        if not cycle_data.empty:
            stats_cycle = _wmean_by_daycount(cycle_data)
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

def calculate_path(df, cycle_label, cycle_start_mapping, use_atr=False,
                   half_life=None, anchor_year=None):
    if df.empty: return pd.Series()

    if cycle_label == "All Years":
        cycle_data = df.copy()
    else:
        cycle_start = cycle_start_mapping[cycle_label]
        years_in_cycle = [cycle_start + i * 4 for i in range(30)]
        cycle_data = df[df["year"].isin(years_in_cycle)].copy()

    if "week_of_month_5day" in cycle_data.columns:
        cycle_data.loc[cycle_data["week_of_month_5day"] > 4, "week_of_month_5day"] = 4

    ret_col = "atr_return" if use_atr else "log_return"

    if half_life is None or anchor_year is None:
        avg_daily = cycle_data.groupby("day_count")[ret_col].mean()
    else:
        cycle_data = cycle_data.copy()
        cycle_data["_weight"] = recency_weights(cycle_data["year"].values, anchor_year, half_life)
        valid = cycle_data[ret_col].notna()
        sub = cycle_data.loc[valid]
        num = (sub[ret_col] * sub["_weight"]).groupby(sub["day_count"]).sum()
        den = sub["_weight"].groupby(sub["day_count"]).sum()
        avg_daily = num / den

    if use_atr:
        # ATR returns are additive — just cumsum
        avg_path = avg_daily.cumsum()
    else:
        avg_path = avg_daily.cumsum().apply(np.exp) - 1

    return avg_path

# -----------------------------------------------------------------------------
# MAIN LOGIC
# -----------------------------------------------------------------------------
def seasonals_chart(ticker, cycle_label, enable_time_travel, reference_year, show_all_years_line, show_pct=True, show_atr=True, half_life=20):
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

    current_price = float(spx["Close"].iloc[-1])
    atr_nonan_series = spx["ATR"].dropna()
    current_atr_dollars = float(atr_nonan_series.iloc[-1]) if not atr_nonan_series.empty else 0.0

    spx["log_return"] = np.log(spx["Close"] / spx["Close"].shift(1))
    spx["atr_return"] = (spx["Close"] - spx["Close"].shift(1)) / spx["ATR"]
    spx["year"] = spx.index.year
    spx["month"] = spx.index.month
    spx["day_count"] = spx.groupby("year").cumcount() + 1
    
    current_year = dt.date.today().year

    # -----------------------------------------------------------------
    # SEASONAL RANK PROFILES
    # -----------------------------------------------------------------
    # Current model rank (uses data < current_year)
    rank_profile_current = calculate_seasonal_rank(spx, cycle_label, cycle_start_mapping, current_year, half_life=half_life)

    # All-years rank (for the optional overlay)
    rank_profile_all_years = pd.Series(dtype=float)
    if show_all_years_line:
        rank_profile_all_years = calculate_seasonal_rank(spx, "All Years", cycle_start_mapping, current_year, half_life=half_life)

    # Time travel rank (uses data < reference_year)
    rank_profile_historical = pd.Series(dtype=float)
    if enable_time_travel:
        rank_profile_historical = calculate_seasonal_rank(spx, cycle_label, cycle_start_mapping, reference_year, half_life=half_life)

    # -----------------------------------------------------------------
    # PATH CALCULATIONS — compute both % and ATR sets
    # -----------------------------------------------------------------
    df_all_history = spx[spx["year"] < current_year].copy()
    df_current_year = spx[spx["year"] == current_year].copy()

    # % paths
    pct_cycle = calculate_path(df_all_history, cycle_label, cycle_start_mapping, use_atr=False,
                               half_life=half_life, anchor_year=current_year) if show_pct else pd.Series()
    pct_all = calculate_path(df_all_history, "All Years", cycle_start_mapping, use_atr=False,
                             half_life=half_life, anchor_year=current_year) if (show_pct and show_all_years_line) else pd.Series()
    pct_realized = pd.Series()
    if show_pct and not df_current_year.empty:
        pct_realized = df_current_year["log_return"].cumsum().apply(np.exp) - 1

    # ATR paths
    atr_cycle = calculate_path(df_all_history, cycle_label, cycle_start_mapping, use_atr=True,
                               half_life=half_life, anchor_year=current_year) if show_atr else pd.Series()
    atr_all = calculate_path(df_all_history, "All Years", cycle_start_mapping, use_atr=True,
                             half_life=half_life, anchor_year=current_year) if (show_atr and show_all_years_line) else pd.Series()
    atr_realized = pd.Series()
    if show_atr and not df_current_year.empty:
        atr_realized = df_current_year["atr_return"].cumsum()

    # Time travel paths
    pct_hist_avg = pd.Series()
    pct_ref_realized = pd.Series()
    atr_hist_avg = pd.Series()
    atr_ref_realized = pd.Series()

    if enable_time_travel:
        df_historical_pool = spx[spx["year"] < reference_year].copy()
        df_ref_year = spx[spx["year"] == reference_year].copy()
        if show_pct:
            pct_hist_avg = calculate_path(df_historical_pool, cycle_label, cycle_start_mapping, use_atr=False,
                                          half_life=half_life, anchor_year=reference_year)
            if not df_ref_year.empty:
                pct_ref_realized = df_ref_year["log_return"].cumsum().apply(np.exp) - 1
        if show_atr:
            atr_hist_avg = calculate_path(df_historical_pool, cycle_label, cycle_start_mapping, use_atr=True,
                                          half_life=half_life, anchor_year=reference_year)
            if not df_ref_year.empty:
                atr_ref_realized = df_ref_year["atr_return"].cumsum()

    # -------------------------------------------------------------------------
    # DATE MAPPING LOGIC (uses actual trading dates, not bdate_range)
    # -------------------------------------------------------------------------
    map_year = reference_year if enable_time_travel else current_year
    map_year_data = spx[spx["year"] == map_year]
    if not map_year_data.empty:
        date_map = {row["day_count"]: idx.strftime("%b %d") for idx, row in map_year_data.iterrows()}
        # Extend with business days for the rest of the year
        last_date = map_year_data.index[-1]
        last_day_count = int(map_year_data["day_count"].iloc[-1])
        remaining = pd.bdate_range(start=last_date + timedelta(days=1), end=f"{map_year}-12-31")
        for i, d in enumerate(remaining):
            date_map[last_day_count + i + 1] = d.strftime("%b %d")
    else:
        # Fallback to bdate_range if no data for the year
        theoretical_dates = pd.bdate_range(start=f"{map_year}-01-01", end=f"{map_year}-12-31")
        date_map = {i+1: d.strftime("%b %d") for i, d in enumerate(theoretical_dates)}

    def get_date_labels(series_index):
        return [date_map.get(i, f"Day {i}") for i in series_index]

    # -------------------------------------------------------------------------
    # MARKER / ANCHOR DAY (needed for theoretical price projection)
    # -------------------------------------------------------------------------
    today = dt.date.today()

    if enable_time_travel:
        target_year = reference_year
        try:
            target_date_start = dt.date(target_year, today.month, today.day)
        except ValueError:
            target_date_start = dt.date(target_year, 2, 28)
        df_context = spx[spx["year"] == target_year]
        marker_color = "#FCD12A"
    else:
        target_year = current_year
        target_date_start = today
        df_context = spx[spx["year"] == target_year]
        marker_color = "white"

    day_count_marker = None
    if not df_context.empty:
        closest_idx = df_context.index.searchsorted(dt.datetime.combine(target_date_start, dt.time.min))
        if closest_idx >= len(df_context):
            closest_idx = len(df_context) - 1
        day_count_marker = df_context.iloc[closest_idx]["day_count"]

    # -------------------------------------------------------------------------
    # HELPER: Build customdata array with date + seasonal rank (+ theoretical price)
    # -------------------------------------------------------------------------
    def build_customdata(day_indices, rank_profile, date_labels=None, atr_path=None, anchor_day=None):
        """
        Returns a list of [date_label, seasonal_rank, theo_price_str] for each day index.
        Theoretical price uses ATR delta from anchor_day on atr_path, scaled by today's $-ATR.
        """
        if date_labels is None:
            date_labels = get_date_labels(day_indices)
        anchor_val = None
        if (atr_path is not None and not atr_path.empty
                and anchor_day is not None and anchor_day in atr_path.index):
            anchor_val = atr_path.loc[anchor_day]
        result = []
        for i, day in enumerate(day_indices):
            label = date_labels[i] if i < len(date_labels) else f"Day {day}"
            rank = rank_profile.get(day, np.nan) if not rank_profile.empty else np.nan
            if anchor_val is not None and day in atr_path.index:
                theo = current_price + (atr_path.loc[day] - anchor_val) * current_atr_dollars
                theo_str = f"${theo:,.2f}"
            else:
                theo_str = ""
            result.append([label, rank, theo_str])
        return result

    HOVER_PCT = (
        "<b>%{customdata[0]}</b><br>"
        "Day: %{x}<br>"
        "Return: %{y:.2%}<br>"
        "Seasonal Rank: %{customdata[1]:.0f}"
        "<extra></extra>"
    )
    HOVER_ATR = (
        "<b>%{customdata[0]}</b><br>"
        "Day: %{x}<br>"
        "Cumulative ATR: %{y:.2f}<br>"
        "Seasonal Rank: %{customdata[1]:.0f}"
        "<extra></extra>"
    )
    HOVER_ATR_PROJ = (
        "<b>%{customdata[0]}</b><br>"
        "Day: %{x}<br>"
        "Cumulative ATR: %{y:.2f}<br>"
        "Theoretical: %{customdata[2]}<br>"
        "Seasonal Rank: %{customdata[1]:.0f}"
        "<extra></extra>"
    )

    # -------------------------------------------------------------------------
    # PLOTTING (dual y-axis: % left, ATR right)
    # -------------------------------------------------------------------------
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # --- % Return traces (left y-axis, solid lines) ---
    if not pct_cycle.empty:
        fig.add_trace(go.Scatter(
            x=pct_cycle.index, y=pct_cycle.values, mode="lines",
            name=f"{cycle_label} %", line=dict(color="#FF8C00", width=3),
            customdata=build_customdata(pct_cycle.index, rank_profile_current),
            hovertemplate=HOVER_PCT
        ), secondary_y=False)

    if not pct_all.empty:
        fig.add_trace(go.Scatter(
            x=pct_all.index, y=pct_all.values, mode="lines",
            name="All Years %", line=dict(color="lightblue", width=1, dash='dot'),
            customdata=build_customdata(pct_all.index, rank_profile_all_years),
            hovertemplate=HOVER_PCT
        ), secondary_y=False)

    if enable_time_travel and not pct_hist_avg.empty:
        fig.add_trace(go.Scatter(
            x=pct_hist_avg.index, y=pct_hist_avg.values, mode="lines",
            name=f"Model {reference_year} %", line=dict(color="#FCD12A", width=2),
            customdata=build_customdata(pct_hist_avg.index, rank_profile_historical),
            hovertemplate=HOVER_PCT
        ), secondary_y=False)

    if not pct_realized.empty:
        realized_dates = [d.strftime("%b %d") for d in df_current_year.index]
        cdata = build_customdata(df_current_year["day_count"].values, rank_profile_current, date_labels=realized_dates)
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(pct_realized) + 1), y=pct_realized.values, mode="lines",
            name=f"{current_year} %", line=dict(color="#39FF14", width=2),
            customdata=cdata, hovertemplate=HOVER_PCT
        ), secondary_y=False)

    if enable_time_travel and not pct_ref_realized.empty:
        realized_dates_ref = [d.strftime("%b %d") for d in df_ref_year.index]
        cdata_ref = build_customdata(df_ref_year["day_count"].values, rank_profile_historical, date_labels=realized_dates_ref)
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(pct_ref_realized) + 1), y=pct_ref_realized.values, mode="lines",
            name=f"{reference_year} %", line=dict(color="#00FFFF", width=2),
            customdata=cdata_ref, hovertemplate=HOVER_PCT
        ), secondary_y=False)

    # --- ATR traces (right y-axis, dashed lines) ---
    if not atr_cycle.empty:
        fig.add_trace(go.Scatter(
            x=atr_cycle.index, y=atr_cycle.values, mode="lines",
            name=f"{cycle_label} ATR", line=dict(color="#FF8C00", width=2),
            customdata=build_customdata(atr_cycle.index, rank_profile_current,
                                        atr_path=atr_cycle, anchor_day=day_count_marker),
            hovertemplate=HOVER_ATR_PROJ
        ), secondary_y=True)

    if not atr_all.empty:
        fig.add_trace(go.Scatter(
            x=atr_all.index, y=atr_all.values, mode="lines",
            name="All Years ATR", line=dict(color="lightblue", width=1, dash='dot'),
            customdata=build_customdata(atr_all.index, rank_profile_all_years,
                                        atr_path=atr_all, anchor_day=day_count_marker),
            hovertemplate=HOVER_ATR_PROJ
        ), secondary_y=True)

    if enable_time_travel and not atr_hist_avg.empty:
        fig.add_trace(go.Scatter(
            x=atr_hist_avg.index, y=atr_hist_avg.values, mode="lines",
            name=f"Model {reference_year} ATR", line=dict(color="#FCD12A", width=2),
            customdata=build_customdata(atr_hist_avg.index, rank_profile_historical,
                                        atr_path=atr_hist_avg, anchor_day=day_count_marker),
            hovertemplate=HOVER_ATR_PROJ
        ), secondary_y=True)

    if not atr_realized.empty:
        realized_dates = [d.strftime("%b %d") for d in df_current_year.index]
        cdata = build_customdata(df_current_year["day_count"].values, rank_profile_current, date_labels=realized_dates)
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(atr_realized) + 1), y=atr_realized.values, mode="lines",
            name=f"{current_year} ATR", line=dict(color="#39FF14", width=2),
            customdata=cdata, hovertemplate=HOVER_ATR
        ), secondary_y=True)

    if enable_time_travel and not atr_ref_realized.empty:
        realized_dates_ref = [d.strftime("%b %d") for d in df_ref_year.index]
        cdata_ref = build_customdata(df_ref_year["day_count"].values, rank_profile_historical, date_labels=realized_dates_ref)
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(atr_ref_realized) + 1), y=atr_ref_realized.values, mode="lines",
            name=f"{reference_year} ATR", line=dict(color="#00FFFF", width=2),
            customdata=cdata_ref, hovertemplate=HOVER_ATR
        ), secondary_y=True)

    # -------------------------------------------------------------------------
    # MARKERS
    # -------------------------------------------------------------------------
    if day_count_marker is not None:
        # Marker on % cycle path
        if show_pct and not pct_cycle.empty and day_count_marker in pct_cycle.index:
            fig.add_trace(go.Scatter(
                x=[day_count_marker], y=[pct_cycle.get(day_count_marker)],
                mode="markers", marker=dict(color=marker_color, size=8, line=dict(width=1, color="black")),
                showlegend=False, hoverinfo="skip"
            ), secondary_y=False)

        # Marker on ATR cycle path
        if show_atr and not atr_cycle.empty and day_count_marker in atr_cycle.index:
            fig.add_trace(go.Scatter(
                x=[day_count_marker], y=[atr_cycle.get(day_count_marker)],
                mode="markers", marker=dict(color=marker_color, size=8, symbol="diamond",
                                            line=dict(width=1, color="black")),
                showlegend=False, hoverinfo="skip"
            ), secondary_y=True)

    # Layout
    title_suffix = f"vs {reference_year}" if enable_time_travel else ""
    fig.update_layout(
        height=800,
        title=f"Seasonal Analysis: {ticker} {title_suffix}",
        xaxis_title=None,
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        legend=dict(
            bgcolor="rgba(20,20,20,0.8)",
            font=dict(color="white"),
            orientation="h",
            yanchor="bottom", y=-0.05,
            xanchor="left", x=0.01
        ),
        hovermode="x unified",
        yaxis=dict(showgrid=False, title="Return" if show_pct else None),
        yaxis2=dict(showgrid=False, title="Cumulative ATR" if show_atr else None,
                    tickformat=".1f"),
    )
    fig.update_xaxes(showgrid=False)

    st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------------------------------
    # DETAILED HISTORY TABLE
    # -------------------------------------------------------------------------
    st.divider()
    
    if enable_time_travel:
        st.subheader(f"📜 Historical Returns (Pre-{reference_year})")
        st.caption(f"Stats exclude {reference_year} and later. Forward returns in ATR units. "
                   f"Recency weighting: {half_life}yr half-life.")
        cutoff_year = reference_year
    else:
        st.subheader(f"📜 Historical Returns (Pre-{current_year})")
        st.caption(f"Stats exclude {current_year}. Forward returns in ATR units. "
                   f"Recency weighting: {half_life}yr half-life.")
        cutoff_year = current_year

    if day_count_marker:
        spx_full = spx.copy()
        spx_full['Fwd_5d'] = (spx_full['Close'].shift(-5) - spx_full['Close']) / spx_full['ATR']
        spx_full['Fwd_10d'] = (spx_full['Close'].shift(-10) - spx_full['Close']) / spx_full['ATR']
        spx_full['Fwd_21d'] = (spx_full['Close'].shift(-21) - spx_full['Close']) / spx_full['ATR']

        spx_full['Daily_Pct'] = spx_full['Close'].pct_change()

        spx_full['rv_5'] = spx_full['Daily_Pct'].rolling(window=5).std().shift(-5) * np.sqrt(252) * 100
        spx_full['rv_10'] = spx_full['Daily_Pct'].rolling(window=10).std().shift(-10) * np.sqrt(252) * 100
        spx_full['rv_21'] = spx_full['Daily_Pct'].rolling(window=21).std().shift(-21) * np.sqrt(252) * 100

        daily_snapshots = spx_full[spx_full['day_count'] == day_count_marker].copy()
        display_df = daily_snapshots[daily_snapshots['year'] < cutoff_year].copy()

        if not display_df.empty:
            display_df = display_df[['year', 'Fwd_5d', 'Fwd_10d', 'Fwd_21d', 'rv_5', 'rv_10', 'rv_21']]
            display_df = display_df.sort_values('year', ascending=False)

            if cycle_label != "All Years":
                start_yr = cycle_start_mapping.get(cycle_label)
                highlight_years = [start_yr + i * 4 for i in range(30)]
            else:
                highlight_years = []

            # --- STATS ---
            st.markdown(f"##### 🎯 Fwd Return Statistics (ATR units)")

            def calculate_stats_row(sub_df):
                if sub_df.empty:
                    return {k: np.nan for k in ["n", "5_median", "5_mean", "5_pospct", "rv_5",
                                                "10_median", "10_mean", "10_pospct", "rv_10",
                                                "21_median", "21_mean", "21_pospct", "rv_21"]}

                weights = recency_weights(sub_df['year'].values, cutoff_year, half_life)
                res = {"n": int(len(sub_df))}
                for d in [5, 10, 21]:
                    ret_col = f"Fwd_{d}d"
                    rv_col = f"rv_{d}"
                    vals = sub_df[ret_col].values
                    rv_vals = sub_df[rv_col].values
                    res[f"{d}_median"] = weighted_median(vals, weights)
                    res[f"{d}_mean"] = weighted_mean(vals, weights)
                    # Weighted positive-percentage
                    mask_ret = ~np.isnan(vals)
                    if mask_ret.any():
                        w_valid = weights[mask_ret]
                        tot_w = w_valid.sum()
                        if tot_w > 0:
                            pos_w = w_valid[vals[mask_ret] > 0].sum()
                            res[f"{d}_pospct"] = pos_w / tot_w * 100
                        else:
                            res[f"{d}_pospct"] = np.nan
                    else:
                        res[f"{d}_pospct"] = np.nan
                    res[rv_col] = weighted_mean(rv_vals, weights)
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
                "5_median": "{:+.2f}", "5_mean": "{:+.2f}", "5_pospct": "{:.1f}%", "rv_5": "{:.2f}%",
                "10_median": "{:+.2f}", "10_mean": "{:+.2f}", "10_pospct": "{:.1f}%", "rv_10": "{:.2f}%",
                "21_median": "{:+.2f}", "21_mean": "{:+.2f}", "21_pospct": "{:.1f}%", "rv_21": "{:.2f}%",
            }).map(color_pos_pct, subset=["5_pospct", "10_pospct", "21_pospct"])

            for d in [5, 10, 21]:
                # Expected dispersion ~ sqrt(N) ATRs; scale gradient accordingly
                vol_scale = np.sqrt(d)
                mean_limit = 0.75 * vol_scale
                styler = styler.background_gradient(
                    subset=[f"{d}_mean"],
                    cmap="RdYlGn",
                    vmin=-mean_limit,
                    vmax=mean_limit
                )
                median_limit = 0.5 * vol_scale
                styler = styler.background_gradient(
                    subset=[f"{d}_median"],
                    cmap="RdYlGn",
                    vmin=-median_limit,
                    vmax=median_limit
                )

            st.dataframe(styler, use_container_width=True)
        else:
            st.warning(f"No historical data available prior to {cutoff_year}.")

    # -------------------------------------------------------------------------
    # SECONDARY CHART: Individual cycle-year paths since 2000 (ATR)
    # -------------------------------------------------------------------------
    if cycle_label != "All Years":
        cycle_start = cycle_start_mapping[cycle_label]
        cycle_years_all = [cycle_start + i * 4 for i in range(30)]
        years_in_data = set(spx['year'].unique())
        modern_cycle_years = sorted([y for y in cycle_years_all if y >= 2000 and y in years_in_data])

        if modern_cycle_years:
            st.divider()
            st.subheader(f"📈 {cycle_label} Years Since 2000 (Cumulative ATR)")

            fig2 = go.Figure()
            palette = ["#FF8C00", "#00FFFF", "#FCD12A", "#FF4DFF", "#87CEFA",
                       "#FF6347", "#ADFF2F", "#FF69B4", "#DDA0DD"]

            for i, y in enumerate(modern_cycle_years):
                yr_data = spx[spx['year'] == y]
                if yr_data.empty:
                    continue
                path = yr_data['atr_return'].cumsum()
                is_current = (y == current_year)
                color = "#39FF14" if is_current else palette[i % len(palette)]
                fig2.add_trace(go.Scatter(
                    x=yr_data['day_count'].values,
                    y=path.values,
                    mode='lines',
                    name=f"{y}" + (" (YTD)" if is_current else ""),
                    line=dict(color=color, width=3 if is_current else 1.5),
                    hovertemplate=f"<b>{y}</b><br>Day: %{{x}}<br>Cum ATR: %{{y:.2f}}<extra></extra>"
                ))

            # Overlay weighted cycle average
            if not atr_cycle.empty:
                fig2.add_trace(go.Scatter(
                    x=atr_cycle.index, y=atr_cycle.values,
                    mode='lines',
                    name=f"{cycle_label} Avg (weighted)",
                    line=dict(color='white', width=2.5, dash='dot'),
                    hovertemplate="Day: %{x}<br>Avg Cum ATR: %{y:.2f}<extra></extra>"
                ))

            fig2.update_layout(
                height=500,
                plot_bgcolor="black",
                paper_bgcolor="black",
                font=dict(color="white"),
                legend=dict(bgcolor="rgba(20,20,20,0.8)", font=dict(color="white"),
                            orientation="h", yanchor="bottom", y=-0.15,
                            xanchor="left", x=0.01),
                hovermode="x unified",
                yaxis=dict(showgrid=False, title="Cumulative ATR", tickformat=".1f"),
            )
            fig2.update_xaxes(showgrid=False, title=None)

            st.plotly_chart(fig2, use_container_width=True)

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
        show_all_years_line = st.checkbox("Overlay 'All Years' Avg", value=False)
        show_pct = st.checkbox("Show % Return", value=False)
        show_atr = True
        half_life = st.slider(
            "Recency half-life (yrs)",
            min_value=5, max_value=100, value=20, step=1,
            help="Exponential decay applied to all averages. e.g. 20 → a year 20 ago has half the weight of today."
        )

    if st.button("Run Analysis", type="primary", use_container_width=True):
        try:
            with st.spinner("Calculating cycle stats..."):
                seasonals_chart(ticker, cycle_label, enable_time_travel, reference_year,
                                show_all_years_line, show_pct, show_atr, half_life=half_life)
        except Exception as e:
            st.error(f"Error generating chart: {e}")

if __name__ == "__main__":
    main()
