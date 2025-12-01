import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import matplotlib
import datetime as dt
import itertools
from collections import defaultdict
from dateutil.easter import easter
from dateutil.relativedelta import relativedelta, MO, TH

# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
SEASONAL_PATH = "seasonal_ranks.csv"
METRICS_PATH = "market_metrics_full_export.csv"

# --- HOLIDAY LOGIC ---
def get_nyse_holidays(start_year, end_year):
    holidays = []
    for year in range(start_year, end_year + 1):
        fixed_dates = {
            "New Years": dt.date(year, 1, 1),
            "Juneteenth": dt.date(year, 6, 19),
            "Independence": dt.date(year, 7, 4),
            "Christmas": dt.date(year, 12, 25)
        }
        for name, date in fixed_dates.items():
            if date.weekday() == 5: holidays.append(date - dt.timedelta(days=1))
            elif date.weekday() == 6: holidays.append(date + dt.timedelta(days=1))
            else: holidays.append(date)

        holidays.append(dt.date(year, 1, 1) + relativedelta(day=1, weekday=MO(3))) 
        holidays.append(dt.date(year, 2, 1) + relativedelta(day=1, weekday=MO(3))) 
        holidays.append(dt.date(year, 5, 31) + relativedelta(weekday=MO(-1)))      
        holidays.append(dt.date(year, 9, 1) + relativedelta(day=1, weekday=MO(1))) 
        holidays.append(dt.date(year, 11, 1) + relativedelta(day=1, weekday=TH(4)))
        holidays.append(easter(year) - dt.timedelta(days=2)) 

    return sorted(list(set(holidays)))

current_year = dt.date.today().year
generated_holidays = get_nyse_holidays(current_year, current_year + 1)
nyse_holidays_dt = np.array(generated_holidays)

def get_trading_days(expiry_date):
    start = dt.date.today()
    if isinstance(expiry_date, dt.datetime): end = expiry_date.date()
    else: end = expiry_date
    if end <= start: return 0
    bus_days = np.busday_count(start, end)
    mask = (nyse_holidays_dt > start) & (nyse_holidays_dt <= end)
    return max(0, bus_days - np.sum(mask))

# -----------------------------------------------------------------------------
# CORE DATA LOADING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_seasonal_map():
    try:
        df = pd.read_csv(SEASONAL_PATH)
    except Exception:
        return {}
    if df.empty: return {}
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])
    df["MD"] = df["Date"].apply(lambda x: (x.month, x.day))
    output_map = {}
    for ticker, group in df.groupby("ticker"):
        output_map[ticker] = pd.Series(group.seasonal_rank.values, index=group.MD).to_dict()
    return output_map

@st.cache_data(show_spinner=False)
def load_market_metrics():
    try:
        df = pd.read_csv(METRICS_PATH)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
    except Exception:
        return pd.DataFrame()
    if df.empty: return pd.DataFrame()
    pivoted = df.pivot_table(index='date', columns='exchange', values='net_new_highs', aggfunc='sum')
    pivoted['Total'] = pivoted.get('NYSE', 0) + pivoted.get('NASDAQ', 0)
    windows = [5, 21] 
    categories = ['Total']
    results = pd.DataFrame(index=pivoted.index)
    for cat in categories:
        if cat not in pivoted.columns: continue
        series = pivoted[cat]
        for w in windows:
            ma_col = series.rolling(window=w).mean()
            results[f"Mkt_{cat}_NH_{w}d_Rank"] = ma_col.expanding(min_periods=126).rank(pct=True) * 100.0
    return results

def get_sznl_val_series(ticker, dates, sznl_map, df_hist=None):
    t_map = sznl_map.get(ticker, {})
    if t_map:
        mds = dates.map(lambda x: (x.month, x.day))
        return mds.map(t_map).fillna(50.0)
    return pd.Series(50.0, index=dates)

@st.cache_data(show_spinner=True)
def download_data(ticker):
    if not ticker: return pd.DataFrame(), None
    try:
        df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex): df.columns = [c[0] for c in df.columns]
        fetch_time = pd.Timestamp.now(tz='America/New_York')
        return df, fetch_time
    except:
        return pd.DataFrame(), None

# -----------------------------------------------------------------------------
# FEATURE ENGINEERING
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def calculate_heatmap_variables(df, sznl_map, market_metrics_df, ticker):
    df = df.copy()
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    
    for w in [1, 2, 3, 5, 10, 21, 63, 126, 252]:
        df[f'FwdRet_{w}d'] = (df['Close'].shift(-w) / df['Close'] - 1.0) * 100.0

    for w in [5, 10, 21, 252]: df[f'Ret_{w}d'] = df['Close'].pct_change(w)
    for w in [21, 63]: df[f'RealVol_{w}d'] = df['LogRet'].rolling(w).std() * np.sqrt(252) * 100.0
    for w in [10, 21]: df[f'VolRatio_{w}d'] = df['Volume'].rolling(w).mean() / df['Volume'].rolling(63).mean()
    df['Seasonal'] = get_sznl_val_series(ticker, df.index, sznl_map, df)

    vars_to_rank = ['Ret_5d', 'Ret_10d', 'Ret_21d', 'Ret_252d', 'RealVol_21d', 'RealVol_63d', 'VolRatio_10d', 'VolRatio_21d']
    rank_cols = []
    for v in vars_to_rank:
        col_name = v + '_Rank'
        df[col_name] = df[v].expanding(min_periods=252).rank(pct=True) * 100.0
        rank_cols.append(col_name)

    if not market_metrics_df.empty:
        df = df.join(market_metrics_df, how='left')
        df.update(df.filter(regex='^Mkt_').ffill(limit=3))

    return df, rank_cols

# -----------------------------------------------------------------------------
# EUCLIDEAN ENGINE
# -----------------------------------------------------------------------------
def get_euclidean_neighbors(df, rank_cols, market_cols, n_neighbors=50):
    current_row = df.iloc[-1]
    all_feats = rank_cols + ['Seasonal'] + market_cols
    valid_feats = [f for f in all_feats if f in df.columns and not pd.isna(current_row[f])]
    if not valid_feats: return pd.DataFrame()
    
    target_vec = current_row[valid_feats].astype(float).values
    history = df.iloc[:-1].dropna(subset=valid_feats).copy()
    if history.empty: return pd.DataFrame()
    
    diff = history[valid_feats].values - target_vec
    dist_sq = np.sum(diff**2, axis=1)
    history['Euclidean_Dist'] = np.sqrt(dist_sq)
    
    n_take = min(len(history), n_neighbors)
    return history.nsmallest(n_take, 'Euclidean_Dist').copy()

# -----------------------------------------------------------------------------
# OPTION PRICING & VISUALIZATION LOGIC
# -----------------------------------------------------------------------------
def option_chain_prices(ticker, price_type="midpoint"):
    stock = yf.Ticker(ticker)
    spot = None
    try: spot = float(stock.fast_info.get("last_price", np.nan))
    except: spot = np.nan
    
    if not np.isfinite(spot):
        try: 
            hist = stock.history(period="5d")
            if not hist.empty: spot = hist["Close"].dropna().iloc[-1]
        except: spot = np.nan

    available_expiries = []
    if stock.options:
        for d in stock.options:
            try: available_expiries.append(dt.datetime.strptime(d, "%Y-%m-%d").date())
            except: continue

    all_data = []
    if not np.isfinite(spot) or not available_expiries:
        return {"Options_Data": [], "Available_Expiries": [], "Current_Price": spot}

    for expiry in available_expiries[:12]:
        try: chain = stock.option_chain(date=expiry.strftime("%Y-%m-%d"))
        except: continue

        calls = chain.calls.dropna(subset=["bid", "ask", "strike"]).copy().query("ask >= bid and bid > 0.05")
        puts = chain.puts.dropna(subset=["bid", "ask", "strike"]).copy().query("ask >= bid and bid > 0.05")
        
        calls = calls[calls["strike"] > spot]
        puts  = puts[puts["strike"]  < spot]

        calls = calls.sort_values("strike")
        puts  = puts.sort_values("strike")

        def quote_price(row):
            return (row["bid"] + row["ask"]) / 2.0 + 0.02 if price_type == "midpoint" else row.get("lastPrice", np.nan) + 0.02

        for _, row in calls.iterrows():
            px = quote_price(row)
            if not np.isfinite(px): continue
            all_data.append({
                "Ticker": ticker, "Expiry": expiry, "Strike": row["strike"], 
                "Type": "Call", "Market_Price": float(px), "IV": row.get('impliedVolatility', 0)
            })

        for _, row in puts.iterrows():
            px = quote_price(row)
            if not np.isfinite(px): continue
            all_data.append({
                "Ticker": ticker, "Expiry": expiry, "Strike": row["strike"], 
                "Type": "Put", "Market_Price": float(px), "IV": row.get('impliedVolatility', 0)
            })

    return {"Options_Data": all_data, "Available_Expiries": available_expiries, "Current_Price": spot}

def get_simulated_prices(df_hist, neighbors, days_forward, current_price):
    future_returns = []
    for idx in neighbors.index:
        loc = df_hist.index.get_loc(idx)
        future_loc = loc + days_forward
        if future_loc < len(df_hist):
            ret = df_hist['Close'].iloc[future_loc] / df_hist['Close'].iloc[loc] - 1
            future_returns.append(ret)
    if not future_returns: return []
    sim_prices = [current_price * (1 + r) for r in future_returns]
    return np.array(sim_prices)

def calculate_option_fair_value(simulated_prices, strike, opt_type):
    if len(simulated_prices) == 0: return 0
    if opt_type == 'Call': payoffs = np.maximum(0, simulated_prices - strike)
    else: payoffs = np.maximum(0, strike - simulated_prices)
    return np.mean(payoffs)

# --- VISUALIZATION FUNCTION ---
def visualize_expiry_analysis(ticker, expiry, forward_prices, df_options, current_price, trading_days):
    sigma = np.std(forward_prices)
    lower_bound = current_price - (2 * sigma)
    upper_bound = current_price + (2 * sigma)

    subset_options = df_options[
        (df_options["Strike"] >= lower_bound) & 
        (df_options["Strike"] <= upper_bound)
    ].copy()
    
    # Sort for plotting
    subset_options = subset_options.sort_values("Strike")

    if subset_options.empty: return

    # Metrics
    pct_positive = np.mean(forward_prices > current_price) * 100
    
    # Model IV
    log_returns = np.log(forward_prices / current_price)
    period_vol = np.std(log_returns)
    annualization_factor = np.sqrt(252 / max(1, trading_days))
    model_iv = period_vol * annualization_factor

    # Market ATM IV
    window_pct = 0.025
    iv_window_df = df_options[
        (df_options['Strike'] >= current_price * (1 - window_pct)) & 
        (df_options['Strike'] <= current_price * (1 + window_pct)) &
        (df_options['IV'] > 0.001)
    ]
    if not iv_window_df.empty: mkt_atm_iv = iv_window_df['IV'].mean()
    else: 
        closest_idx = (df_options['Strike'] - current_price).abs().idxmin()
        mkt_atm_iv = df_options.loc[closest_idx, 'IV']

    # Plot
    fig = make_subplots(
        rows=2, cols=1, 
        subplot_titles=(f"Forward Prices ({expiry})", f"OTM Options (+/- 2 Std Dev): Theo vs Market"),
        vertical_spacing=0.15,
        row_heights=[0.4, 0.6]
    )

    fig.add_trace(go.Histogram(x=forward_prices, name='Forward Prices', marker_color='green', opacity=0.6, nbinsx=50), row=1, col=1)
    fig.add_vline(x=current_price, line_dash="dot", line_color="black", annotation_text="Current", row=1, col=1)
    fig.add_vline(x=np.mean(forward_prices), line_dash="dash", line_color="blue", annotation_text="Mean", row=1, col=1)

    fig.add_annotation(
        xref="x domain", yref="y domain", 
        x=0.98, y=0.95,
        text=f"<b>Model IV:</b> {model_iv:.2%} | <b>Mkt ATM IV:</b> {mkt_atm_iv:.2%}<br>Mean: {np.mean(forward_prices):.2f} | % Pos: {pct_positive:.1f}%",
        showarrow=False, 
        bgcolor="#444444", bordercolor="white", borderwidth=1,
        font=dict(color="white", size=10),
        align="right", row=1, col=1
    )

    fig.add_trace(go.Bar(x=subset_options["Strike"], y=subset_options["Theo"], name="Theoretical", marker_color="pink", opacity=0.7), row=2, col=1)
    fig.add_trace(go.Bar(x=subset_options["Strike"], y=subset_options["Market_Price"], name="Market Price", marker_color="blue", opacity=0.7), row=2, col=1)

    fig.update_xaxes(range=[lower_bound, upper_bound], row=1, col=1)
    fig.update_xaxes(range=[lower_bound, upper_bound], row=2, col=1)
    fig.update_layout(title=f"{ticker} Analysis - Expiry {expiry} ({trading_days} days)", height=700, barmode='group', template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

def generate_strategies(df_chain, sim_prices, spot):
    strategies = []
    
    # 1. Independent Longs (Naked OTM)
    for _, row in df_chain.iterrows():
        fair_val = calculate_option_fair_value(sim_prices, row['Strike'], row['Type'])
        mkt_val = row['Market_Price']
        edge = fair_val - mkt_val
        
        if edge > mkt_val * 0.1 and mkt_val > 0.05:
            strategies.append({
                "Type": "Long " + row['Type'], "Strikes": f"{row['Strike']}", "Expiry": row['Expiry'],
                "Cost": mkt_val, "Fair Value": fair_val, "Edge": edge, "ROI": (edge / mkt_val) * 100
            })
            
    # 2. Vertical Spreads (Using OTM Legs)
    for (expiry, otype), group in df_chain.groupby(['Expiry', 'Type']):
        strikes = sorted(group['Strike'].unique())
        for s1, s2 in itertools.combinations(strikes, 2):
            if (s2 - s1) < (spot * 0.005): continue
            leg1, leg2 = group[group['Strike'] == s1].iloc[0], group[group['Strike'] == s2].iloc[0]
            
            if otype == 'Call':
                cost = leg1['Market_Price'] - leg2['Market_Price']
                if cost <= 0: continue
                fv1 = calculate_option_fair_value(sim_prices, s1, 'Call')
                fv2 = calculate_option_fair_value(sim_prices, s2, 'Call')
                fair_spread = fv1 - fv2
                edge = fair_spread - cost
                
                if edge > cost * 0.15:
                    strategies.append({
                        "Type": "Bull Call Spread", "Strikes": f"{s1}/{s2}", "Expiry": expiry,
                        "Cost": cost, "Fair Value": fair_spread, "Edge": edge, "ROI": (edge / cost) * 100
                    })
            else: # Put
                cost = leg2['Market_Price'] - leg1['Market_Price'] 
                if cost <= 0: continue
                fv_long = calculate_option_fair_value(sim_prices, s2, 'Put')
                fv_short = calculate_option_fair_value(sim_prices, s1, 'Put')
                fair_spread = fv_long - fv_short
                edge = fair_spread - cost
                
                if edge > cost * 0.15:
                    strategies.append({
                        "Type": "Bear Put Spread", "Strikes": f"{s2}/{s1}", "Expiry": expiry,
                        "Cost": cost, "Fair Value": fair_spread, "Edge": edge, "ROI": (edge / cost) * 100
                    })

    return pd.DataFrame(strategies)

# -----------------------------------------------------------------------------
# UI RENDER
# -----------------------------------------------------------------------------
def render_heatmap():
    st.subheader("Heatmap Analytics & Option Lab")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_axis_label = st.selectbox("X-Axis Variable", ["Ret_5d_Rank", "Ret_21d_Rank", "Seasonal", "VolRatio_10d_Rank"], index=0)
        y_axis_label = st.selectbox("Y-Axis Variable", ["Ret_5d_Rank", "Ret_21d_Rank", "Seasonal", "VolRatio_10d_Rank"], index=2)
    with col2:
        z_axis_label = st.selectbox("Target (Z-Axis)", ["FwdRet_5d", "FwdRet_21d", "FwdRet_63d"], index=0)
        ticker = st.text_input("Ticker", value="SMH").upper()
    with col3:
        ensemble_tol = st.slider("Pairwise Tolerance", 1, 25, 5, 1)
        k_neighbors = st.number_input("Euclidean Neighbors", 10, 250, 50)
        analysis_start = st.date_input("Start Date", dt.date(2000, 1, 1))

    if st.button("Run Analysis", type="primary"):
        st.session_state['run_main'] = True

    if st.session_state.get('run_main'):
        with st.spinner(f"Processing {ticker}..."):
            data, _ = download_data(ticker)
            if data.empty: 
                st.error("No data")
                return
            
            sznl_map = load_seasonal_map()
            mkt_metrics = load_market_metrics()
            df, rank_cols = calculate_heatmap_variables(data, sznl_map, mkt_metrics, ticker)
            
            neighbors = get_euclidean_neighbors(df, rank_cols, [c for c in df.columns if c.startswith("Mkt_")], n_neighbors=k_neighbors)
            st.success(f"Analysis Complete. Found {len(neighbors)} Euclidean matches.")
            
            st.session_state['full_df'] = df
            st.session_state['neighbors'] = neighbors
            st.session_state['ticker'] = ticker

    if 'full_df' in st.session_state:
        st.divider()
        st.header("🧪 Option Strategy Lab")
        st.markdown("""
        **Method:** Scrapes live **OTM** option chains (filtered for liquidity) and compares Market Prices vs. Theoretical Prices derived from the **Euclidean Neighbor** distribution.
        """)
        
        c_opt1, c_opt2 = st.columns([1, 3])
        with c_opt1:
            min_dte = st.number_input("Min DTE", 0, 60, 0)
            max_dte = st.number_input("Max DTE", 0, 90, 21)
            run_opt = st.button("🔎 Scan Options Chain")
            
        if run_opt:
            with st.spinner("Fetching live options & calculating fair values..."):
                data_pkg = option_chain_prices(st.session_state['ticker'])
                df_opts = pd.DataFrame(data_pkg["Options_Data"])
                spot = data_pkg["Current_Price"]
                expiries = data_pkg["Available_Expiries"]
                
                if df_opts.empty:
                    st.error("Could not fetch options data (or no OTM options found).")
                else:
                    valid_expiries = []
                    all_strategies = []
                    processed_count = 0
                    
                    # Store visualization data for the FIRST processed expiry
                    viz_expiry = None
                    
                    for exp in expiries:
                        days = get_trading_days(exp)
                        if days < min_dte or days > max_dte: continue
                        if processed_count >= 3: break 
                        
                        processed_count += 1
                        valid_expiries.append(exp)
                        
                        sim_prices = get_simulated_prices(st.session_state['full_df'], st.session_state['neighbors'], days, spot)
                        
                        if len(sim_prices) > 0:
                            subset_chain = df_opts[df_opts['Expiry'] == exp].copy()
                            
                            # Calculate Theo for Visualization (All Strikes in this expiry)
                            subset_chain['Theo'] = subset_chain.apply(lambda row: calculate_option_fair_value(sim_prices, row['Strike'], row['Type']), axis=1)
                            
                            # Render chart for the first valid expiry we encounter
                            if viz_expiry is None:
                                viz_expiry = exp
                                visualize_expiry_analysis(st.session_state['ticker'], exp, sim_prices, subset_chain, spot, days)
                            
                            strats = generate_strategies(subset_chain, sim_prices, spot)
                            if not strats.empty:
                                all_strategies.append(strats)

                    if all_strategies:
                        master_df = pd.concat(all_strategies)
                        # SORT BY ROI
                        spreads = master_df[master_df['Type'].str.contains("Spread")].sort_values("ROI", ascending=False).head(5)
                        naked = master_df[~master_df['Type'].str.contains("Spread")].sort_values("ROI", ascending=False).head(3)
                        
                        st.subheader(f"🏆 Top Trade Ideas (Next {processed_count} Expiries)")
                        
                        col_tbl1, col_tbl2 = st.columns(2)
                        with col_tbl1:
                            st.write("**Top 5 Vertical Spreads (OTM) by ROI**")
                            st.dataframe(spreads[['Type', 'Expiry', 'Strikes', 'Cost', 'Fair Value', 'Edge', 'ROI']].style.format({
                                'Cost': '${:.2f}', 'Fair Value': '${:.2f}', 'Edge': '${:.2f}', 'ROI': '{:.1f}%'
                            }).background_gradient(subset=['ROI'], cmap='Greens'), use_container_width=True)
                            
                        with col_tbl2:
                            st.write("**Top 3 Independent Longs (OTM) by ROI**")
                            st.dataframe(naked[['Type', 'Expiry', 'Strikes', 'Cost', 'Fair Value', 'Edge', 'ROI']].style.format({
                                'Cost': '${:.2f}', 'Fair Value': '${:.2f}', 'Edge': '${:.2f}', 'ROI': '{:.1f}%'
                            }).background_gradient(subset=['ROI'], cmap='Greens'), use_container_width=True)
                    else:
                        st.warning("No trades found with positive edge criteria.")

def main():
    st.set_page_config(layout="wide", page_title="Heatmap & Option Lab")
    render_heatmap()

if __name__ == "__main__":
    main()
