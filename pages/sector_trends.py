import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import datetime as dt
import os
from datetime import timedelta

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
SECTOR_ETFS = [
    "IBB", "IHI", "ITA", "ITB", "IYR", "KRE", "OIH", "SMH", "VNQ",
    "XBI", "XHB", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP",
    "XLU", "XLV", "XLY", "XME", "XOP", "XRT", "GLD", "CEF", "SLV", "BTC-USD",
    "ETH-USD", "UNG", "UVXY",'EURUSD=X', 'JPY=X', 'GBPUSD=X', 'AUDUSD=X', 'NZDUSD=X',
    'CAD=X', 'CHF=X', 'DX-Y.NYB','^GSPC','^NDX','^RUT','^DJI','CL=F','NG=F','GC=F','HG=F',
    "SPY", "QQQ", "IWM", "DIA",'KC=F','PL=F','ZC=F','ZW=F','CC=F','SB=F','PA=F','ZS=F',
    # International equity indices (15+ years on yfinance)
    "^FTSE", "^GDAXI", "^FCHI", "^N225", "^HSI", "^STI",
    "^AXJO", "^KS11", "^TWII", "^BSESN", "^GSPTSE", "^MXX",
    "^BVSP", "^STOXX50E",
    # Fixed income
    "TLT", "IEF", "TIP", "LQD", "HYG", "AGG",
    # Volatility
    "^VIX",
    # Additional commodities
    "CT=F", "SI=F",
]

CSV_PATH = "sznl_sector_forecast.csv"
ATR_SZNL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "atr_seasonal_ranks.parquet")
ATR_SZNL_WINDOWS = [5, 10, 21, 63, 126, 252]
ATR_SZNL_COLS = [f"atr_sznl_{w}d" for w in ATR_SZNL_WINDOWS]

# -----------------------------------------------------------------------------
# CACHE MANAGEMENT
# -----------------------------------------------------------------------------
def clear_all_caches():
    st.cache_data.clear()

# -----------------------------------------------------------------------------
# HELPER: DETERMINE CYCLE
# -----------------------------------------------------------------------------
def get_current_cycle_label():
    year = dt.date.today().year
    rem = year % 4
    if rem == 0: return "Election"
    if rem == 1: return "Post-Election"
    if rem == 2: return "Midterm"
    if rem == 3: return "Pre-Election"
    return "All Years"

# -----------------------------------------------------------------------------
# DATA LOADING (TOP TABLE)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_seasonal_map():
    """
    Loads the CSV and creates a dictionary of TimeSeries for each ticker.
    Structure: { 'SPY': pd.Series(index=Datetime, data=Rank) }
    """
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        return {}

    if df.empty: return {}

    # Ensure valid dates
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])
    
    # Normalize to midnight (remove time component if present)
    df["Date"] = df["Date"].dt.normalize()
    
    output_map = {}
    # Group by ticker and create a sorted Series for each
    for ticker, group in df.groupby("ticker"):
        # We set the index to Date and ensure it is sorted for 'asof' lookups
        series = group.set_index("Date")["seasonal_rank"].sort_index()
        output_map[ticker] = series

    return output_map

def get_sznl_val(ticker, target_date, sznl_map):
    """
    Smart lookup: Finds the rank for the target date.
    If target date is missing (e.g., weekend), finds the most recent previous value.
    """
    if ticker not in sznl_map: return np.nan
    
    series = sznl_map[ticker]
    target = pd.Timestamp(target_date).normalize()
    
    # 'asof' finds the last valid value up to (and including) the target date.
    # It handles weekends/holidays automatically by looking back.
    try:
        val = series.asof(target)
        return val
    except:
        return np.nan

@st.cache_data(show_spinner=False)
def load_atr_seasonal_map():
    """Load ATR seasonal ranks. Returns {ticker: DataFrame with 6 rank columns}."""
    if not os.path.exists(ATR_SZNL_PATH):
        return {}
    try:
        df = pd.read_parquet(ATR_SZNL_PATH)
        df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
        output = {}
        for ticker, group in df.groupby('ticker'):
            output[ticker] = group.set_index('Date')[ATR_SZNL_COLS].sort_index()
        return output
    except Exception:
        return {}


def get_atr_sznl_vals(ticker, target_date, atr_sznl_map):
    """Look up ATR seasonal ranks for a ticker at a date. Returns dict of {col: value}."""
    if not atr_sznl_map or ticker not in atr_sznl_map:
        return {col: np.nan for col in ATR_SZNL_COLS}
    df = atr_sznl_map[ticker]
    target = pd.Timestamp(target_date).normalize()
    result = {}
    for col in ATR_SZNL_COLS:
        try:
            result[col] = df[col].asof(target)
        except Exception:
            result[col] = np.nan
    return result


def percentile_rank(series: pd.Series, value) -> float:
    s = series.dropna().values
    if s.size == 0: return np.nan
    try: v = float(value)
    except: return np.nan
    if np.isnan(v): return np.nan
    return float((s <= v).sum() / s.size * 100.0)

@st.cache_data(show_spinner=True)
def load_sector_metrics(tickers):
    sznl_map = load_seasonal_map()
    atr_sznl_map = load_atr_seasonal_map()
    today = dt.datetime.now()
    rows = []

    for t in tickers:
        try:
            df = yf.download(t, period="2y", interval="1d", auto_adjust=True, progress=False)
        except: continue
        if df.empty: continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

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

        row = {
            "Ticker": t,
            "Price": float(close.iloc[-1]),
            "Sznl": get_sznl_val(t, today, sznl_map),
            "PctRank5": ranks[0],
            "PctRank20": ranks[1],
            "PctRank50": ranks[2],
            "PctRank200": ranks[3],
        }

        # Add ATR seasonal ranks
        atr_vals = get_atr_sznl_vals(t, today, atr_sznl_map)
        row.update(atr_vals)

        rows.append(row)

    if not rows: return pd.DataFrame()

    df_out = pd.DataFrame(rows)
    if "Sznl" in df_out.columns:
        df_out = df_out.sort_values("Sznl", ascending=False, ignore_index=True)

    return df_out

# -----------------------------------------------------------------------------
# CHARTING LOGIC (SEASONALITY)
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def get_chart_data(ticker):
    # Fetch Max History for the Seasonal Chart
    end_date_fetch = dt.datetime.now() + timedelta(days=5)
    df = yf.download(ticker, period="max", end=end_date_fetch, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def calculate_path(df, cycle_label, use_atr=False):
    cycle_start_mapping = {
        "Election": 1952, "Pre-Election": 1951,
        "Post-Election": 1953, "Midterm": 1950
    }

    if df.empty: return pd.Series()

    if cycle_label == "All Years":
        cycle_data = df.copy()
    else:
        cycle_start = cycle_start_mapping.get(cycle_label, 1953)
        years_in_cycle = [cycle_start + i * 4 for i in range(30)]
        cycle_data = df[df["year"].isin(years_in_cycle)].copy()

    if "week_of_month_5day" in cycle_data.columns:
        cycle_data.loc[cycle_data["week_of_month_5day"] > 4, "week_of_month_5day"] = 4

    ret_col = "atr_return" if use_atr else "log_return"
    avg_daily = cycle_data.groupby("day_count")[ret_col].mean()

    if use_atr:
        avg_path = avg_daily.cumsum()
    else:
        avg_path = avg_daily.cumsum().apply(np.exp) - 1

    return avg_path

def render_seasonal_chart(ticker, show_pct=True, show_atr=True):
    cycle_label = get_current_cycle_label()
    spx = get_chart_data(ticker)

    if spx.empty:
        st.warning(f"No data for {ticker}")
        return

    # Feature Engineering
    spx = spx.copy()
    spx["log_return"] = np.log(spx["Close"] / spx["Close"].shift(1))
    prev_close = spx["Close"].shift(1)
    tr = pd.concat([
        spx["High"] - spx["Low"],
        (spx["High"] - prev_close).abs(),
        (spx["Low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    spx["ATR"] = tr.rolling(14).mean()
    spx["atr_return"] = (spx["Close"] - prev_close) / spx["ATR"]

    spx["year"] = spx.index.year
    spx["month"] = spx.index.month
    spx["day_count"] = spx.groupby("year").cumcount() + 1

    current_year = dt.date.today().year
    df_history = spx[spx["year"] < current_year].copy()
    df_current = spx[spx["year"] == current_year].copy()

    # Compute both path sets
    path_cycle_pct = calculate_path(df_history, cycle_label, use_atr=False) if show_pct else pd.Series()
    path_all_pct = calculate_path(df_history, "All Years", use_atr=False) if show_pct else pd.Series()
    path_current_pct = pd.Series()
    if show_pct and not df_current.empty:
        path_current_pct = df_current["log_return"].cumsum().apply(np.exp) - 1

    path_cycle_atr = calculate_path(df_history, cycle_label, use_atr=True) if show_atr else pd.Series()
    path_all_atr = calculate_path(df_history, "All Years", use_atr=True) if show_atr else pd.Series()
    path_current_atr = pd.Series()
    if show_atr and not df_current.empty:
        path_current_atr = df_current["atr_return"].cumsum()

    # Date mapping: use actual trading dates from this year's data
    map_year_data = spx[spx["year"] == current_year]
    if not map_year_data.empty:
        date_map = {row["day_count"]: idx.strftime("%b %d") for idx, row in map_year_data.iterrows()}
        # Extend with business days for the rest of the year
        last_date = map_year_data.index[-1]
        last_day_count = int(map_year_data["day_count"].iloc[-1])
        remaining = pd.bdate_range(start=last_date + timedelta(days=1), end=f"{current_year}-12-31")
        for i, d in enumerate(remaining):
            date_map[last_day_count + i + 1] = d.strftime("%b %d")
    else:
        theoretical_dates = pd.bdate_range(start=f"{current_year}-01-01", end=f"{current_year}-12-31")
        date_map = {i+1: d.strftime("%b %d") for i, d in enumerate(theoretical_dates)}

    def get_date_labels(day_indices):
        return [date_map.get(int(i), f"Day {i}") for i in day_indices]

    HOVER_PCT = (
        "<b>%{customdata[0]}</b><br>"
        "Day: %{x}<br>"
        "Return: %{y:.2%}"
        "<extra></extra>"
    )
    HOVER_ATR = (
        "<b>%{customdata[0]}</b><br>"
        "Day: %{x}<br>"
        "Cumulative ATR: %{y:.2f}"
        "<extra></extra>"
    )

    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # --- % Return traces (left y-axis) ---
    if not path_cycle_pct.empty:
        cycle_dates = get_date_labels(path_cycle_pct.index)
        fig.add_trace(go.Scatter(
            x=path_cycle_pct.index, y=path_cycle_pct.values,
            mode="lines", name=f"{cycle_label} %",
            line=dict(color="#FF8C00", width=2),
            customdata=[[d] for d in cycle_dates],
            hovertemplate=HOVER_PCT
        ), secondary_y=False)

    if not path_all_pct.empty:
        all_dates = get_date_labels(path_all_pct.index)
        fig.add_trace(go.Scatter(
            x=path_all_pct.index, y=path_all_pct.values,
            mode="lines", name="All Years %",
            line=dict(color="lightblue", width=1, dash='dot'),
            customdata=[[d] for d in all_dates],
            hovertemplate=HOVER_PCT
        ), secondary_y=False)

    if not path_current_pct.empty:
        realized_dates = [d.strftime("%b %d") for d in df_current.index]
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(path_current_pct) + 1),
            y=path_current_pct.values,
            mode="lines", name=f"{current_year} %",
            line=dict(color="#39FF14", width=2),
            customdata=[[d] for d in realized_dates],
            hovertemplate=HOVER_PCT
        ), secondary_y=False)

    # --- ATR traces (right y-axis) ---
    if not path_cycle_atr.empty:
        cycle_dates_atr = get_date_labels(path_cycle_atr.index)
        fig.add_trace(go.Scatter(
            x=path_cycle_atr.index, y=path_cycle_atr.values,
            mode="lines", name=f"{cycle_label} ATR",
            line=dict(color="#FF8C00", width=2),
            customdata=[[d] for d in cycle_dates_atr],
            hovertemplate=HOVER_ATR
        ), secondary_y=True)

    if not path_all_atr.empty:
        all_dates_atr = get_date_labels(path_all_atr.index)
        fig.add_trace(go.Scatter(
            x=path_all_atr.index, y=path_all_atr.values,
            mode="lines", name="All Years ATR",
            line=dict(color="lightblue", width=1, dash='dot'),
            customdata=[[d] for d in all_dates_atr],
            hovertemplate=HOVER_ATR
        ), secondary_y=True)

    if not path_current_atr.empty:
        realized_dates = [d.strftime("%b %d") for d in df_current.index]
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(path_current_atr) + 1),
            y=path_current_atr.values,
            mode="lines", name=f"{current_year} ATR",
            line=dict(color="#39FF14", width=2),
            customdata=[[d] for d in realized_dates],
            hovertemplate=HOVER_ATR
        ), secondary_y=True)

    # Markers
    today = dt.date.today()
    df_current_ctx = spx[spx["year"] == current_year]
    
    if not df_current_ctx.empty:
        # Find today's index in the dataframe
        closest_idx = df_current_ctx.index.searchsorted(dt.datetime.combine(today, dt.time.min))
        if closest_idx >= len(df_current_ctx): closest_idx = len(df_current_ctx) - 1
        
        day_count_marker = df_current_ctx.iloc[closest_idx]["day_count"]

        # --- Markers on % cycle path ---
        if show_pct and not path_cycle_pct.empty and day_count_marker in path_cycle_pct.index:
            fig.add_trace(go.Scatter(
                x=[day_count_marker], y=[path_cycle_pct.get(day_count_marker)],
                mode="markers",
                marker=dict(color="white", size=8, line=dict(width=1, color="black")),
                showlegend=False, hoverinfo="skip"
            ), secondary_y=False)

        # --- Markers on ATR cycle path ---
        if show_atr and not path_cycle_atr.empty and day_count_marker in path_cycle_atr.index:
            fig.add_trace(go.Scatter(
                x=[day_count_marker], y=[path_cycle_atr.get(day_count_marker)],
                mode="markers",
                marker=dict(color="white", size=8, symbol="diamond",
                            line=dict(width=1, color="black")),
                showlegend=False, hoverinfo="skip"
            ), secondary_y=True)

    fig.update_layout(
        title=f"{ticker} Seasonality ({cycle_label})",
        margin=dict(l=10, r=10, t=40, b=10),
        height=600,
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        showlegend=True,
        legend=dict(
            bgcolor="rgba(20,20,20,0.8)",
            font=dict(color="white"),
            orientation="h",
            yanchor="bottom", y=-0.05,
            xanchor="left", x=0.01
        ),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, title="Return" if show_pct else None),
        yaxis2=dict(showgrid=False, title="Cumulative ATR" if show_atr else None,
                    tickformat=".1f")
    )
    
    st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------
def main():
    st.set_page_config(layout="wide", page_title="Sector Trend & Seasonals")
    
    st.title("Sector ETF Trend Dashboard")
    st.write("Top: MA Extension Ranks. Bottom: Individual Seasonal Charts.")

    if st.button("Refresh data"):
        clear_all_caches()
        st.rerun()

    # 1. TABLE SECTION
    with st.spinner("Loading Sector Metrics..."):
        table = load_sector_metrics(sorted(set(SECTOR_ETFS)))

    if table.empty:
        st.error("No data available.")
        return

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
    for col in ATR_SZNL_COLS:
        if col in table.columns:
            format_dict[col] = "{:.1f}"

    atr_cols_present = [c for c in ATR_SZNL_COLS if c in table.columns]

    styled = (
        table.style
        .format(format_dict)
        .map(highlight_pct_rank, subset=["PctRank5", "PctRank20", "PctRank50", "PctRank200"])
        .map(highlight_sznl, subset=["Sznl"])
        .map(highlight_sznl, subset=atr_cols_present)
    )

    st.dataframe(styled, use_container_width=True)

    # 2. CHARTS SECTION
    st.divider()
    st.subheader("Seasonal Charts")
    c1, c2 = st.columns([3, 1])
    with c1:
        st.caption(f"ATR-normalized path (default). Check box to overlay raw % return.")
    with c2:
        show_pct = st.checkbox("Show % Return", value=False)
    show_atr = True

    tickers_to_plot = table["Ticker"].tolist()
    cols = st.columns(2)

    for i, ticker in enumerate(tickers_to_plot):
        with cols[i % 2]:
            render_seasonal_chart(ticker, show_pct=show_pct, show_atr=show_atr)

if __name__ == "__main__":
    main()
