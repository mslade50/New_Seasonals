import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import datetime as dt
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
    "SPY", "QQQ", "IWM", "DIA",'KC=F','PL=F','ZC=F','ZW=F','CC=F','SB=F','PA=F','ZS=F'
]

CSV_PATH = "seasonal_ranks.csv"

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
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        return {}

    if df.empty: return {}

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
    if ticker not in sznl_map: return np.nan
    md = (target_date.month, target_date.day)
    return sznl_map[ticker].get(md, np.nan)

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
    today = dt.datetime.now()
    rows = []

    for t in tickers:
        try:
            # We fetch a smaller window for the table to keep it fast
            df = yf.download(t, period="2y", interval="1d", auto_adjust=True, progress=False)
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
    # SORT BY SEASONAL RANK (High to Low)
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

def calculate_path(df, cycle_label):
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

    avg_path = (
        cycle_data.groupby("day_count")["log_return"]
        .mean()
        .cumsum()
        .apply(np.exp) - 1
    )
    return avg_path

def render_seasonal_chart(ticker):
    cycle_label = get_current_cycle_label()
    spx = get_chart_data(ticker)

    if spx.empty:
        st.warning(f"No data for {ticker}")
        return

    # Feature Engineering
    spx = spx.copy()
    spx["log_return"] = np.log(spx["Close"] / spx["Close"].shift(1))
    spx["year"] = spx.index.year
    spx["month"] = spx.index.month
    spx["day_count"] = spx.groupby("year").cumcount() + 1
    
    current_year = dt.date.today().year

    # 1. Historical Pools (Exclude Current Year)
    df_history = spx[spx["year"] < current_year].copy()
    
    # Path A: Selected Cycle
    path_cycle = calculate_path(df_history, cycle_label)
    
    # Path B: All Years
    path_all = calculate_path(df_history, "All Years")

    # Path C: Current Realized
    df_current = spx[spx["year"] == current_year].copy()
    path_current = pd.Series()
    if not df_current.empty:
        path_current = df_current["log_return"].cumsum().apply(np.exp) - 1

    # Plotting
    fig = go.Figure()

    # Cycle Average (Orange)
    fig.add_trace(go.Scatter(
        x=path_cycle.index, y=path_cycle.values,
        mode="lines", name=f"{cycle_label} Avg",
        line=dict(color="#FF8C00", width=2)
    ))

    # All Years Average (Blue Dashed)
    if not path_all.empty:
        fig.add_trace(go.Scatter(
            x=path_all.index, y=path_all.values,
            mode="lines", name="All Years Avg",
            line=dict(color="lightblue", width=1, dash='dot')
        ))

    # Current Realized (Green)
    if not path_current.empty:
        fig.add_trace(go.Scatter(
            x=np.arange(1, len(path_current) + 1),
            y=path_current.values,
            mode="lines", name=f"{current_year} Realized",
            line=dict(color="#39FF14", width=2)
        ))

    # Markers
    today = dt.date.today()
    df_current_ctx = spx[spx["year"] == current_year]
    
    if not df_current_ctx.empty:
        # Find today's index in the dataframe
        closest_idx = df_current_ctx.index.searchsorted(dt.datetime.combine(today, dt.time.min))
        if closest_idx >= len(df_current_ctx): closest_idx = len(df_current_ctx) - 1
        
        day_count_marker = df_current_ctx.iloc[closest_idx]["day_count"]
        future_dates = pd.bdate_range(start=today, periods=30)
        offsets = [5, 10, 21]

        # --- Plot on Cycle Path (ORANGE) ---
        if day_count_marker in path_cycle.index:
            y_val = path_cycle.get(day_count_marker)
            fig.add_trace(go.Scatter(
                x=[day_count_marker], y=[y_val],
                mode="markers", name="Today",
                marker=dict(color="white", size=8, line=dict(width=1, color="black")),
                showlegend=False
            ))
            
            for offset in offsets:
                target_idx = day_count_marker + offset
                if target_idx in path_cycle.index:
                    proj_y = path_cycle.get(target_idx)
                    d_label = future_dates[offset].strftime("%b %d") if offset < len(future_dates) else ""
                    
                    fig.add_trace(go.Scatter(
                        x=[target_idx], y=[proj_y],
                        mode="markers", name=f"T+{offset}",
                        marker=dict(color="yellow", size=6),
                        hovertemplate=f"T+{offset}: {d_label}<extra></extra>",
                        showlegend=False
                    ))

        # --- Plot on All Years Path (BLUE) ---
        if day_count_marker in path_all.index:
            y_val_all = path_all.get(day_count_marker)
            fig.add_trace(go.Scatter(
                x=[day_count_marker], y=[y_val_all],
                mode="markers", 
                marker=dict(color="lightblue", size=6, line=dict(width=1, color="white")),
                showlegend=False, hoverinfo="skip"
            ))
            
            for offset in offsets:
                target_idx = day_count_marker + offset
                if target_idx in path_all.index:
                    proj_y_all = path_all.get(target_idx)
                    fig.add_trace(go.Scatter(
                        x=[target_idx], y=[proj_y_all],
                        mode="markers", 
                        marker=dict(color="lightblue", size=5, symbol="circle"),
                        showlegend=False, hoverinfo="skip"
                    ))

    fig.update_layout(
        title=f"{ticker} Seasonality ({cycle_label})",
        margin=dict(l=10, r=10, t=40, b=10),
        height=600, # <--- UPDATED HEIGHT (600px)
        plot_bgcolor="black",
        paper_bgcolor="black",
        font=dict(color="white"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="right", x=1
        ),
        xaxis=dict(showgrid=False, title="Day of Year"),
        yaxis=dict(showgrid=False, title="Return")
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

    styled = (
        table.style
        .format(format_dict)
        .map(highlight_pct_rank, subset=["PctRank5", "PctRank20", "PctRank50", "PctRank200"])
        .map(highlight_sznl, subset=["Sznl"])
    )

    st.dataframe(styled, use_container_width=True)

    # 2. CHARTS SECTION
    st.divider()
    st.subheader("Seasonal Charts")
    st.caption(f"Current Cycle: {get_current_cycle_label()} (Orange). All Years (Blue Dashed). Current Year Realized (Green).")
    
    # Grid layout for charts (2 per row for better visibility)
    tickers_to_plot = table["Ticker"].tolist()
    
    # Create columns for grid layout
    cols = st.columns(2)
    
    for i, ticker in enumerate(tickers_to_plot):
        with cols[i % 2]:
            render_seasonal_chart(ticker)

if __name__ == "__main__":
    main()
