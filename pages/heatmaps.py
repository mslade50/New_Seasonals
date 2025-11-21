import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import datetime
import random
import time
import re 

# -----------------------------------------------------------------------------
# CONFIG / CONSTANTS
# -----------------------------------------------------------------------------
SECTOR_ETFS = [
    "IBB", "IHI", "ITA", "ITB", "IYR", "KRE", "OIH", "SMH", "VNQ",
    "XBI", "XHB", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP",
    "XLU", "XLV", "XLY", "XME", "XOP", "XRT", "GLD", "CEF", "SLV",
]

INDEX_ETFS = ["SPY", "QQQ", "IWM", "DIA", "SMH"]

INTERNATIONAL_ETFS = [
    "EWZ", "EWC", "ECH", "ECOL", "EWW", "ARGT",
    "EWQ", "EWG", "EWI", "EWU", "EWP", "EWK", "EWO", "EWN", "EWD", "EWL",
    "EWJ", "EWH", "MCHI", "INDA", "EWY", "EWT", "EWA", "EWS", "EWM", "THD", "EIDO", "VNM", "EPHE",
    "EZA", "TUR", "EGPT"
]

CSV_PATH = "seasonal_ranks.csv"

# -----------------------------------------------------------------------------
# DATA UTILS
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

def get_sznl_val_series(ticker, dates, sznl_map):
    t_map = sznl_map.get(ticker, {})
    if not t_map:
        return pd.Series(50.0, index=dates)
    
    mds = dates.map(lambda x: (x.month, x.day))
    return mds.map(t_map).fillna(50.0)

@st.cache_data(show_spinner=True)
def download_universe_data(tickers, fetch_start_date):
    """
    Downloads data in chunks using a specific Fetch Date.
    """
    if not tickers: return {} 
    
    clean_tickers = [str(t).strip().upper() for t in tickers if str(t).strip() != '']
    if not clean_tickers: return {}

    if isinstance(fetch_start_date, datetime.date):
        start_str = fetch_start_date.strftime('%Y-%m-%d')
    else:
        start_str = fetch_start_date 

    data_dict = {}
    CHUNK_SIZE = 50 
    total_tickers = len(clean_tickers)
    
    download_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, total_tickers, CHUNK_SIZE):
        chunk = clean_tickers[i:i + CHUNK_SIZE]
        status_text.text(f"Downloading batch {i} to {min(i+CHUNK_SIZE, total_tickers)} of {total_tickers}...")
        download_bar.progress(min((i + CHUNK_SIZE) / total_tickers, 1.0))
        
        try:
            df = yf.download(chunk, start=start_str, group_by='ticker', auto_adjust=True, progress=False, threads=True)
            if df.empty: continue

            if len(chunk) == 1:
                t = chunk[0]
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]
                if not df.empty:
                    data_dict[t] = df
            else:
                for t in chunk:
                    try:
                        if t not in df.columns.levels[0]: continue
                        t_df = df[t].copy()
                        col_to_check = 'Close' if 'Close' in t_df.columns else 'close'
                        if col_to_check in t_df.columns:
                            t_df = t_df.dropna(subset=[col_to_check])
                        if not t_df.empty:
                            data_dict[t] = t_df
                    except: continue
            time.sleep(0.1)

        except Exception as e:
            err_msg = str(e).lower()
            if "rate limited" in err_msg or "too many requests" in err_msg:
                st.warning(f"⚠️ Rate limit hit. Stopping download.")
                break
            continue
    
    download_bar.empty()
    status_text.empty()
    return data_dict

def get_cycle_year(year):
    rem = year % 4
    if rem == 0: return "4. Election Year"
    if rem == 1: return "1. Post-Election"
    if rem == 2: return "2. Midterm Year"
    if rem == 3: return "3. Pre-Election"
    return "Unknown"

def get_age_bucket(years):
    if years < 3: return "< 3 Years"
    if years < 5: return "3-5 Years"
    if years < 10: return "5-10 Years"
    if years < 20: return "10-20 Years"
    return "> 20 Years"

# -----------------------------------------------------------------------------
# HEATMAP UTILS (Smoothing & Binning)
# -----------------------------------------------------------------------------

def smooth_display(Z, sigma=1.2):
    """Mask-aware Gaussian smoothing for pretty plots."""
    Z = np.asarray(Z, float)
    mask = np.isfinite(Z)
    Z_fill = np.where(mask, Z, 0.0)

    # Smooth data and a weight mask, then renormalize
    w = gaussian_filter(mask.astype(float), sigma=sigma, mode="nearest")
    z = gaussian_filter(Z_fill, sigma=sigma, mode="nearest")
    out = z / np.maximum(w, 1e-9)

    # Keep original NaN holes invisible (optional, currently filling them visually)
    # out[~mask] = np.nan 
    return out

def build_bins_quantile(x, y, nx=30, ny=30):
    """Quantile-based edges to minimize empty cells."""
    x = pd.Series(x, dtype=float).dropna()
    y = pd.Series(y, dtype=float).dropna()

    if len(x) < 10 or len(y) < 10:
        return np.linspace(0,100,nx+1), np.linspace(0,100,ny+1)

    x_edges = np.unique(np.quantile(x, np.linspace(0, 1, nx + 1)))
    y_edges = np.unique(np.quantile(y, np.linspace(0, 1, ny + 1)))

    # Fallback if data is too discrete
    if len(x_edges) < 3: x_edges = np.linspace(x.min(), x.max(), max(3, nx + 1))
    if len(y_edges) < 3: y_edges = np.linspace(y.min(), y.max(), max(3, ny + 1))
    
    return x_edges, y_edges

def grid_mean(df_sub, xcol, ycol, zcol, x_edges, y_edges):
    x = df_sub[xcol].to_numpy(dtype=float)
    y = df_sub[ycol].to_numpy(dtype=float)
    z = df_sub[zcol].to_numpy(dtype=float)
    m = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    x = x[m]; y = y[m]; z = z[m]

    if len(x) == 0:
        return np.array([]), np.array([]), np.full((len(y_edges)-1, len(x_edges)-1), np.nan)

    xi = np.digitize(x, x_edges) - 1
    yi = np.digitize(y, y_edges) - 1
    nx = len(x_edges) - 1; ny = len(y_edges) - 1

    sums   = np.zeros((ny, nx), float)
    counts = np.zeros((ny, nx), float)

    valid = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    xi = xi[valid]; yi = yi[valid]; zv = z[valid]
    
    # Simple loop is often fast enough for display grids, could vectorise with bincount for huge data
    for X, Y, Z in zip(xi, yi, zv):
        sums[Y, X]  += Z
        counts[Y, X]+= 1.0

    with np.errstate(invalid='ignore', divide='ignore'):
        mean = sums / counts
    mean[counts == 0] = np.nan

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    return x_centers, y_centers, mean

def nan_neighbor_fill(Z, k=3):
    Z = np.asarray(Z, float)
    kernel = np.ones((k, k), dtype=float)
    valid = np.isfinite(Z).astype(float)
    sum_vals   = convolve2d(np.nan_to_num(Z), kernel, mode='same', boundary='symm')
    count_vals = convolve2d(valid, kernel, mode='same', boundary='symm')
    Z_out = Z.copy()
    mask = np.isnan(Z) & (count_vals > 0)
    Z_out[mask] = sum_vals[mask] / count_vals[mask]
    return Z_out

# -----------------------------------------------------------------------------
# HEATMAP VARIABLE FACTORY
# -----------------------------------------------------------------------------

def calculate_heatmap_variables(df, sznl_map, ticker):
    """Calculates advanced vars and ranks them 0-100 for the heatmap axes."""
    df = df.copy()
    
    # Basic Returns (for Z-axis)
    for w in [5, 10, 21, 63]:
        df[f'FwdRet_{w}d'] = df['Close'].pct_change(periods=-w) * 100.0  # Forward returns %

    # --- X/Y VARIABLES (Raw Calculation) ---
    
    # 1. Trailing Returns
    for w in [5, 10, 21, 63, 126, 252]:
        df[f'Ret_{w}d'] = df['Close'].pct_change(w)

    # 2. Realized Volatility (Annualized)
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    for w in [21, 63]:
        df[f'RealVol_{w}d'] = df['LogRet'].rolling(w).std() * np.sqrt(252) * 100.0
    
    # 3. Change in Realized Vol (21d vs 63d)
    df['VolChange'] = df['RealVol_21d'] - df['RealVol_63d']

    # 4. Volume Ratios
    for w in [5, 10, 21]:
        # Volume relative to 63d baseline
        df[f'VolRatio_{w}d'] = df['Volume'].rolling(w).mean() / df['Volume'].rolling(63).mean()

    # 5. Seasonality
    df['Seasonal'] = get_sznl_val_series(ticker, df.index, sznl_map)

    # --- RANK TRANSFORMATION (Percentiles 0-100) ---
    # We use expanding rank to prevent lookahead bias if used for backtesting, 
    # but for pure descriptive heatmap analysis, full period rank is often preferred.
    # Given the prompt implies "variables to test", we will use Rolling or Expanding Rank 
    # to normalize everything to 0-100 scale. Let's use Expanding (min 252) for robustness.
    
    vars_to_rank = [
        'Ret_5d', 'Ret_10d', 'Ret_21d', 'Ret_63d', 'Ret_126d', 'Ret_252d',
        'RealVol_21d', 'RealVol_63d', 'VolChange',
        'VolRatio_5d', 'VolRatio_10d', 'VolRatio_21d'
    ]
    
    for v in vars_to_rank:
        # Create a column name suitable for the dropdown (e.g., "5d Return %ile")
        # Rank 0 to 100
        df[v + '_Rank'] = df[v].expanding(min_periods=252).rank(pct=True) * 100.0

    return df

# -----------------------------------------------------------------------------
# BACKTEST ENGINE (Copied from previous request)
# -----------------------------------------------------------------------------
def calculate_backtest_indicators(df, sznl_map, ticker, spy_series=None):
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    df['SMA200'] = df['Close'].rolling(200).mean()
    for window in [5, 10, 21]:
        df[f'ret_{window}d'] = df['Close'].pct_change(window)
        df[f'rank_ret_{window}d'] = df[f'ret_{window}d'].expanding(min_periods=252).rank(pct=True) * 100.0
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    df['Sznl'] = get_sznl_val_series(ticker, df.index, sznl_map)
    rolling_high = df['High'].shift(1).rolling(252).max()
    rolling_low = df['Low'].shift(1).rolling(252).min()
    df['is_52w_high'] = df['High'] > rolling_high
    df['is_52w_low'] = df['Low'] < rolling_low
    vol_ma = df['Volume'].rolling(63).mean()
    df['vol_ma'] = vol_ma
    df['vol_ratio'] = df['Volume'] / vol_ma
    if not df.empty:
        start_ts = df.index[0]
        df['age_years'] = (df.index - start_ts).days / 365.25
    else:
        df['age_years'] = 0.0
    if spy_series is not None:
        df['SPY_Above_SMA200'] = spy_series.reindex(df.index, method='ffill').fillna(False)
    return df

def run_backtest_engine(universe_dict, params, sznl_map, spy_series=None):
    trades = []
    total = len(universe_dict)
    bt_start_ts = pd.to_datetime(params['backtest_start_date'])
    progress_bar = st.progress(0)
    
    for i, (ticker, df_raw) in enumerate(universe_dict.items()):
        progress_bar.progress((i+1)/total)
        if len(df_raw) < 100: continue
        try:
            df = calculate_backtest_indicators(df_raw, sznl_map, ticker, spy_series)
            df = df[df.index >= bt_start_ts]
            if df.empty: continue
            conditions = []
            trend_opt = params.get('trend_filter', 'None')
            if trend_opt == "Price > 200 SMA": conditions.append(df['Close'] > df['SMA200'])
            elif trend_opt == "Price > Rising 200 SMA": conditions.append((df['Close'] > df['SMA200']) & (df['SMA200'] > df['SMA200'].shift(1)))
            elif trend_opt == "SPY > 200 SMA" and 'SPY_Above_SMA200' in df.columns: conditions.append(df['SPY_Above_SMA200'])
            curr_age = df['age_years'].fillna(0)
            curr_vol = df['vol_ma'].fillna(0)
            curr_close = df['Close'].fillna(0)
            gate = (curr_close >= params['min_price']) & (curr_vol >= params['min_vol']) & (curr_age >= params['min_age']) & (curr_age <= params['max_age'])
            conditions.append(gate)
            if params['use_perf_rank']:
                col = f"rank_ret_{params['perf_window']}d"
                if params['perf_logic'] == '<': cond = df[col] < params['perf_thresh']
                else: cond = df[col] > params['perf_thresh']
                if params['perf_first_instance']:
                    prev = cond.shift(1).rolling(params['perf_lookback']).sum()
                    cond = cond & (prev == 0)
                conditions.append(cond)
            if params['use_sznl']:
                if params['sznl_logic'] == '<': cond = df['Sznl'] < params['sznl_thresh']
                else: cond = df['Sznl'] > params['sznl_thresh']
                if params['sznl_first_instance']:
                    prev = cond.shift(1).rolling(params['sznl_lookback']).sum()
                    cond = cond & (prev == 0)
                conditions.append(cond)
            if params['use_52w']:
                if params['52w_type'] == 'New 52w High': cond = df['is_52w_high']
                else: cond = df['is_52w_low']
                if params['52w_first_instance']:
                    prev = cond.shift(1).rolling(params['52w_lookback']).sum()
                    cond = cond & (prev == 0)
                conditions.append(cond)
            if params['use_vol']:
                cond = df['vol_ratio'] > params['vol_thresh']
                conditions.append(cond)
            if not conditions: continue
            final_signal = conditions[0]
            for c in conditions[1:]: final_signal = final_signal & c
            signal_dates = df.index[final_signal]
            for signal_date in signal_dates:
                try:
                    sig_idx = df.index.get_loc(signal_date)
                    if sig_idx + params['holding_days'] + 2 >= len(df): continue
                    atr = df['ATR'].iloc[sig_idx]
                    if np.isnan(atr) or atr == 0: continue
                    if params['entry_type'] == 'Signal Close':
                        entry_price = df['Close'].iloc[sig_idx]
                        start_idx = sig_idx + 1
                    elif params['entry_type'] == 'T+1 Open':
                        entry_price = df['Open'].iloc[sig_idx + 1]
                        start_idx = sig_idx + 1
                    else:
                        entry_price = df['Close'].iloc[sig_idx + 1]
                        start_idx = sig_idx + 2
                    stop_price = entry_price - (atr * params['stop_atr'])
                    tgt_price = entry_price + (atr * params['tgt_atr'])
                    exit_price = entry_price
                    exit_type = "Hold"
                    exit_date = None
                    future = df.iloc[start_idx : start_idx + params['holding_days']]
                    if not params['time_exit_only']:
                        for f_date, f_row in future.iterrows():
                            if f_row['Low'] <= stop_price:
                                exit_price = f_row['Open'] if f_row['Open'] < stop_price else stop_price
                                exit_type = "Stop"
                                exit_date = f_date
                                break
                            if f_row['High'] >= tgt_price:
                                exit_price = f_row['Open'] if f_row['Open'] > tgt_price else tgt_price
                                exit_type = "Target"
                                exit_date = f_date
                                break
                    if exit_type == "Hold":
                        exit_price = future['Close'].iloc[-1]
                        exit_date = future.index[-1]
                        exit_type = "Time"
                    risk_unit = entry_price - stop_price
                    if risk_unit <= 0: risk_unit = 0.001
                    pnl = exit_price - entry_price
                    r = pnl / risk_unit
                    trades.append({
                        "Ticker": ticker, "SignalDate": signal_date, "Entry": entry_price,
                        "Exit": exit_price, "ExitDate": exit_date, "Type": exit_type, "R": r,
                        "Age": df['age_years'].iloc[sig_idx], "AvgVol": df['vol_ma'].iloc[sig_idx]
                    })
                except: continue
        except: continue
    progress_bar.empty()
    return pd.DataFrame(trades)

def grade_strategy(pf, sqn, win_rate, total_trades):
    score = 0
    reasons = []
    if pf >= 2.0: score += 4
    elif pf >= 1.5: score += 3
    elif pf >= 1.2: score += 2
    elif pf >= 1.0: score += 1
    else: score -= 5 
    if sqn >= 3.0: score += 4
    elif sqn >= 2.0: score += 3
    elif sqn >= 1.5: score += 2
    elif sqn > 0: score += 1
    if total_trades < 30: score -= 2
    if score >= 7: return "A", "Excellent", reasons
    if score >= 5: return "B", "Good", reasons
    if score >= 3: return "C", "Marginal", reasons
    if score >= 0: return "D", "Poor", reasons
    return "F", "Uninvestable", reasons

# -----------------------------------------------------------------------------
# UI: RENDER BACKTESTER
# -----------------------------------------------------------------------------
def render_backtester():
    st.subheader("1. Universe & Data")
    col_u1, col_u2, col_u3 = st.columns([1, 1, 2])
    sample_pct = 100 
    use_full_history = False
    
    with col_u1:
        univ_choice = st.selectbox("Choose Universe", 
            ["Sector ETFs", "Indices", "International ETFs", "Sector + Index ETFs", "All CSV Tickers", "Custom (Upload CSV)"])
    with col_u2:
        default_start = datetime.date(2000, 1, 1)
        start_date = st.date_input("Backtest Start Date", value=default_start)
    
    custom_tickers = []
    if univ_choice == "Custom (Upload CSV)":
        with col_u3:
            sample_pct = st.slider("Random Sample %", 1, 100, 100)
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file:
                try:
                    c_df = pd.read_csv(uploaded_file)
                    if "Ticker" in c_df.columns:
                        c_df["Ticker"] = c_df["Ticker"].astype(str).str.strip().str.upper()
                        c_df = c_df[~c_df["Ticker"].isin(["NAN", "NONE", "NULL", ""])]
                        custom_tickers = c_df["Ticker"].unique().tolist()
                        if len(custom_tickers) > 0: st.success(f"Loaded {len(custom_tickers)} valid tickers.")
                except: st.error("Invalid CSV.")
    
    st.write("")
    use_full_history = st.checkbox("⚠️ Download Full History (1950+) for Accurate 'Age'", value=False)

    st.markdown("---")
    st.subheader("2. Execution & Risk")
    time_exit_only = st.checkbox("Time Exit Only (Disable Stop/Target)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: entry_type = st.selectbox("Entry Price", ["Signal Close", "T+1 Open", "T+1 Close"])
    with c2: stop_atr = st.number_input("Stop Loss (ATR)", value=3.0, step=0.1)
    with c3: tgt_atr = st.number_input("Target (ATR)", value=8.0, step=0.1, disabled=time_exit_only)
    with c4: hold_days = st.number_input("Max Holding Days", value=10, step=1)
    with c5: risk_per_trade = st.number_input("Risk Amount ($)", value=1000, step=100)

    st.markdown("---")
    st.subheader("3. Signal Criteria")

    with st.expander("Liquidity & Data History Filters", expanded=True):
        l1, l2, l3, l4 = st.columns(4)
        with l1: min_price = st.number_input("Min Price ($)", value=10.0, step=1.0)
        with l2: min_vol = st.number_input("Min Avg Volume", value=100000, step=50000)
        with l3: min_age = st.number_input("Min True Age (Yrs)", value=0.25, step=0.25)
        with l4: max_age = st.number_input("Max True Age (Yrs)", value=100.0, step=1.0)
        
    with st.expander("Trend Filter (NEW)", expanded=True):
        t1, _ = st.columns([1, 3])
        with t1:
            trend_filter = st.selectbox("Trend Condition", 
                ["None", "Price > 200 SMA", "Price > Rising 200 SMA", "SPY > 200 SMA"],
                help="Requires 200 days of data prior to signal. 'SPY > 200 SMA' checks the broad market regime.")

    with st.expander("Performance Percentile Rank", expanded=False):
        use_perf = st.checkbox("Enable Performance Filter", value=False)
        p1, p2, p3, p4, p5 = st.columns(5)
        with p1: perf_window = st.selectbox("Window", [5, 10, 21], disabled=not use_perf)
        with p2: perf_logic = st.selectbox("Logic", ["<", ">"], disabled=not use_perf)
        with p3: perf_thresh = st.number_input("Threshold (%)", 0.0, 100.0, 15.0, disabled=not use_perf)
        with p4: perf_first = st.checkbox("First Instance Only", value=True, disabled=not use_perf)
        with p5: perf_lookback = st.number_input("Instance Lookback (Days)", 1, 100, 21, disabled=not use_perf)

    with st.expander("Seasonal Rank", expanded=False):
        use_sznl = st.checkbox("Enable Seasonal Filter", value=False)
        s1, s2, s3, s4 = st.columns(4)
        with s1: sznl_logic = st.selectbox("Seasonality", ["<", ">"], key="sl", disabled=not use_sznl)
        with s2: sznl_thresh = st.number_input("Seasonal Rank Threshold", 0.0, 100.0, 15.0, key="st", disabled=not use_sznl)
        with s3: sznl_first = st.checkbox("First Instance Only", value=True, key="sf", disabled=not use_sznl)
        with s4: sznl_lookback = st.number_input("Instance Lookback (Days)", 1, 100, 21, key="slb", disabled=not use_sznl)

    with st.expander("52-Week High/Low", expanded=False):
        use_52w = st.checkbox("Enable 52w High/Low Filter", value=False)
        h1, h2, h3 = st.columns(3)
        with h1: type_52w = st.selectbox("Condition", ["New 52w High", "New 52w Low"], disabled=not use_52w)
        with h2: first_52w = st.checkbox("First Instance Only", value=True, key="hf", disabled=not use_52w)
        with h3: lookback_52w = st.number_input("Instance Lookback (Days)", 1, 252, 21, key="hlb", disabled=not use_52w)

    with st.expander("Volume Spike", expanded=False):
        use_vol = st.checkbox("Enable Volume Spike Filter", value=False)
        v1, _ = st.columns([1, 3])
        with v1: vol_thresh = st.number_input("Vol Multiple (> X * 63d Avg)", 1.0, 10.0, 1.5, disabled=not use_vol)

    st.markdown("---")
    
    if st.button("Run Backtest", type="primary", use_container_width=True):
        tickers_to_run = []
        sznl_map = load_seasonal_map()
        
        if univ_choice == "Sector ETFs": tickers_to_run = SECTOR_ETFS
        elif univ_choice == "Indices": tickers_to_run = INDEX_ETFS
        elif univ_choice == "International ETFs": tickers_to_run = INTERNATIONAL_ETFS
        elif univ_choice == "Sector + Index ETFs": tickers_to_run = list(set(SECTOR_ETFS + INDEX_ETFS))
        elif univ_choice == "All CSV Tickers": tickers_to_run = [t for t in list(sznl_map.keys()) if t not in ["BTC-USD", "ETH-USD"]]
        elif univ_choice == "Custom (Upload CSV)": tickers_to_run = custom_tickers
            
        if not tickers_to_run:
            st.error("No tickers found.")
            return

        fetch_start = "1950-01-01" if use_full_history else start_date
        st.info(f"Downloading data ({len(tickers_to_run)} tickers)...")
        data_dict = download_universe_data(tickers_to_run, fetch_start)
        if not data_dict: return
        
        spy_series = None
        if trend_filter == "SPY > 200 SMA":
            if "SPY" not in data_dict:
                st.info("Fetching SPY data for regime filter...")
                spy_dict = download_universe_data(["SPY"], fetch_start)
                if "SPY" in spy_dict:
                    spy_df = spy_dict["SPY"]
                    spy_df['SMA200'] = spy_df['Close'].rolling(200).mean()
                    spy_series = spy_df['Close'] > spy_df['SMA200']
            else:
                spy_df = data_dict["SPY"]
                spy_df['SMA200'] = spy_df['Close'].rolling(200).mean()
                spy_series = spy_df['Close'] > spy_df['SMA200']

        params = {
            'backtest_start_date': start_date,
            'time_exit_only': time_exit_only,
            'stop_atr': stop_atr, 'tgt_atr': tgt_atr, 'holding_days': hold_days, 'entry_type': entry_type,
            'min_price': min_price, 'min_vol': min_vol, 'min_age': min_age, 'max_age': max_age,
            'trend_filter': trend_filter,
            'use_perf_rank': use_perf, 'perf_window': perf_window, 'perf_logic': perf_logic, 
            'perf_thresh': perf_thresh, 'perf_first_instance': perf_first, 'perf_lookback': perf_lookback,
            'use_sznl': use_sznl, 'sznl_logic': sznl_logic, 'sznl_thresh': sznl_thresh, 
            'sznl_first_instance': sznl_first, 'sznl_lookback': sznl_lookback,
            'use_52w': use_52w, '52w_type': type_52w, '52w_first_instance': first_52w, '52w_lookback': lookback_52w,
            'use_vol': use_vol, 'vol_thresh': vol_thresh
        }
        
        trades_df = run_backtest_engine(data_dict, params, sznl_map, spy_series)
        if trades_df.empty:
            st.warning("No signals generated.")
            return

        trades_df = trades_df.sort_values("ExitDate")
        trades_df['PnL_Dollar'] = trades_df['R'] * risk_per_trade
        trades_df['CumPnL'] = trades_df['PnL_Dollar'].cumsum()
        trades_df['SignalDate'] = pd.to_datetime(trades_df['SignalDate'])
        trades_df['Year'] = trades_df['SignalDate'].dt.year
        trades_df['Month'] = trades_df['SignalDate'].dt.strftime('%b')
        trades_df['MonthNum'] = trades_df['SignalDate'].dt.month
        trades_df['DayOfWeek'] = trades_df['SignalDate'].dt.day_name()
        trades_df['CyclePhase'] = trades_df['Year'].apply(get_cycle_year)
        trades_df['AgeBucket'] = trades_df['Age'].apply(get_age_bucket)
        
        if len(trades_df) >= 10:
            try: trades_df['VolDecile'] = pd.qcut(trades_df['AvgVol'], 10, labels=False, duplicates='drop') + 1
            except: trades_df['VolDecile'] = 1
        else: trades_df['VolDecile'] = 1

        wins = trades_df[trades_df['R'] > 0]
        losses = trades_df[trades_df['R'] <= 0]
        win_rate = len(wins) / len(trades_df) * 100
        pf = wins['PnL_Dollar'].sum() / abs(losses['PnL_Dollar'].sum()) if not losses.empty else 999
        r_series = trades_df['R']
        sqn = np.sqrt(len(trades_df)) * (r_series.mean() / r_series.std()) if len(trades_df) > 1 else 0
        
        grade, verdict, notes = grade_strategy(pf, sqn, win_rate, len(trades_df))
        
        st.success("Backtest Complete!")
        
        st.markdown(f"""
        <div style="background-color: #0e1117; padding: 20px; border-radius: 10px; border: 1px solid #444;">
            <h2 style="margin-top:0; color: #ffffff;">Strategy Grade: <span style="color: {'#00ff00' if grade in ['A','B'] else '#ffaa00' if grade=='C' else '#ff0000'};">{grade}</span> ({verdict})</h2>
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                <div><h3>Profit Factor: {pf:.2f}</h3></div>
                <div><h3>SQN: {sqn:.2f}</h3></div>
                <div><h3>Win Rate: {win_rate:.1f}%</h3></div>
                <div><h3>Expectancy: ${trades_df['PnL_Dollar'].mean():.2f}</h3></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        if notes: st.warning("Notes: " + ", ".join(notes))

        fig = px.line(trades_df, x="ExitDate", y="CumPnL", title=f"Cumulative Equity (Risk: ${risk_per_trade}/trade)", markers=True)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Performance Breakdowns")
        b1, b2 = st.columns(2)
        b1.plotly_chart(px.bar(trades_df.groupby('Year')['PnL_Dollar'].sum().reset_index(), x='Year', y='PnL_Dollar', title="PnL by Year", text_auto='.2s'), use_container_width=True)
        b2.plotly_chart(px.bar(trades_df.groupby('CyclePhase')['PnL_Dollar'].sum().reset_index().sort_values('CyclePhase'), x='CyclePhase', y='PnL_Dollar', title="PnL by Cycle", text_auto='.2s'), use_container_width=True)
        
        st.subheader("Trade Log")
        st.dataframe(trades_df.style.format({
            "Entry": "{:.2f}", "Exit": "{:.2f}", "R": "{:.2f}", "PnL_Dollar": "${:,.2f}",
            "Age": "{:.1f}y", "AvgVol": "{:,.0f}"
        }), use_container_width=True)

        st.markdown("---")
        st.subheader("Configuration & Results (Copy Code)")
        st.info("Copy the dictionary below and paste it into your `STRATEGY_BOOK` list in the Screener.")

        dict_str = f"""{{
    "id": "STRAT_{int(time.time())}",
    "name": "Generated Strategy ({grade})",
    "description": "Universe: {univ_choice}. Filter: {trend_filter}. PF: {pf:.2f}. SQN: {sqn:.2f}.",
    "universe_tickers": {tickers_to_run}, 
    "settings": {{
        "use_perf_rank": {use_perf}, "perf_window": {perf_window}, "perf_logic": "{perf_logic}", "perf_thresh": {perf_thresh},
        "use_sznl": {use_sznl}, "sznl_logic": "{sznl_logic}", "sznl_thresh": {sznl_thresh},
        "use_52w": {use_52w}, "52w_type": "{type_52w}",
        "use_vol": {use_vol}, "vol_thresh": {vol_thresh},
        "trend_filter": "{trend_filter}",
        "min_price": {min_price}, "min_vol": {min_vol},
        "min_age": {min_age}, "max_age": {max_age}
    }},
    "execution": {{
        "risk_per_trade": {risk_per_trade},
        "stop_atr": {stop_atr},
        "tgt_atr": {tgt_atr},
        "hold_days": {hold_days}
    }},
    "stats": {{
        "grade": "{grade} ({verdict})",
        "win_rate": "{win_rate:.1f}%",
        "expectancy": "${trades_df['PnL_Dollar'].mean():.2f}",
        "profit_factor": "{pf:.2f}"
    }}
}},"""
        st.code(dict_str, language="python")

# -----------------------------------------------------------------------------
# UI: RENDER HEATMAP
# -----------------------------------------------------------------------------
def render_heatmap():
    st.subheader("Heatmap Analysis")
    
    # 1. SELECTION
    col1, col2, col3 = st.columns(3)
    with col1:
        # Variable Mapping for Display -> Column Name
        var_options = {
            "Seasonality Rank": "Seasonal",
            "5d Trailing Return Rank": "Ret_5d_Rank",
            "10d Trailing Return Rank": "Ret_10d_Rank",
            "21d Trailing Return Rank": "Ret_21d_Rank",
            "63d Trailing Return Rank": "Ret_63d_Rank",
            "126d Trailing Return Rank": "Ret_126d_Rank",
            "252d Trailing Return Rank": "Ret_252d_Rank",
            "21d Realized Vol Rank": "RealVol_21d_Rank",
            "63d Realized Vol Rank": "RealVol_63d_Rank",
            "Change in Vol (21d-63d) Rank": "VolChange_Rank",
            "5d Rel. Volume Rank": "VolRatio_5d_Rank",
            "10d Rel. Volume Rank": "VolRatio_10d_Rank",
            "21d Rel. Volume Rank": "VolRatio_21d_Rank"
        }
        
        x_axis_label = st.selectbox("X-Axis Variable", list(var_options.keys()), index=0)
        y_axis_label = st.selectbox("Y-Axis Variable", list(var_options.keys()), index=3)
    
    with col2:
        target_options = {
            "5d Forward Return": "FwdRet_5d",
            "10d Forward Return": "FwdRet_10d",
            "21d Forward Return": "FwdRet_21d",
            "63d Forward Return": "FwdRet_63d",
        }
        z_axis_label = st.selectbox("Target (Z-Axis)", list(target_options.keys()), index=2)
        ticker = st.text_input("Ticker", value="SPY").upper()
    
    with col3:
        # Settings
        smooth_sigma = st.slider("Smoothing (Sigma)", 0.5, 3.0, 1.2, 0.1)
        bins = st.slider("Grid Resolution (Bins)", 10, 50, 28)
        
    if st.button("Generate Heatmap", type="primary"):
        st.info(f"Downloading full history for {ticker}...")
        
        # We need MAX history for the heatmap to fill the buckets effectively
        data = yf.download(ticker, period="max", progress=False, auto_adjust=True)
        
        if data.empty:
            st.error("No data found for ticker.")
            return
            
        # flatten if needed
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0] for c in data.columns]
            
        sznl_map = load_seasonal_map()
        
        # Calculate Variables
        df = calculate_heatmap_variables(data, sznl_map, ticker)
        
        # Get cols
        xcol = var_options[x_axis_label]
        ycol = var_options[y_axis_label]
        zcol = target_options[z_axis_label]
        
        # Build Edges (Quantile based)
        x_edges, y_edges = build_bins_quantile(df[xcol], df[ycol], nx=bins, ny=bins)
        
        # Grid Mean
        x_centers, y_centers, Z = grid_mean(df, xcol, ycol, zcol, x_edges, y_edges)
        
        # Fill & Smooth
        Z_filled = nan_neighbor_fill(Z)
        Z_smooth = smooth_display(Z_filled, sigma=smooth_sigma)
        
        # Plot
        # Determine color scale limits centered at 0 for returns
        z_max = np.nanmax(np.abs(Z_smooth))
        z_min = -z_max
        
        fig = go.Figure(data=go.Heatmap(
            z=Z_smooth,
            x=x_centers,
            y=y_centers,
            colorscale='RdBu', # Red to Blue (Seismic-like)
            zmin=z_min, zmax=z_max,
            reversescale=False, # Red=Low, Blue=High usually. Check context.
                                # Usually Red=Neg Returns, Blue=Pos. 
                                # RdBu: Red is low (neg), Blue is high (pos).
            colorbar=dict(title="Fwd Return %")
        ))
        
        fig.update_layout(
            title=f"{ticker}: {z_axis_label} Heatmap",
            xaxis_title=x_axis_label + " (0-100)",
            yaxis_title=y_axis_label + " (0-100)",
            width=800, height=700
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats
        st.write("### Distribution Statistics")
        st.write(df[[xcol, ycol, zcol]].describe())


def main():
    st.set_page_config(layout="wide", page_title="Quantitative Suite")
    st.title("Quantitative Strategy Suite")
    
    tab1, tab2 = st.tabs(["Backtester", "Heatmap Analysis"])
    
    with tab1:
        render_backtester()
    with tab2:
        render_heatmap()

if __name__ == "__main__":
    main()
