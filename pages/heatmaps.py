import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
import matplotlib
import datetime
import itertools
from collections import defaultdict
from dateutil.easter import easter
from dateutil.relativedelta import relativedelta, MO, TH

# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
SEASONAL_PATH = "seasonal_ranks.csv"
METRICS_PATH = "market_metrics_full_export.csv"

# --- HOLIDAY LOGIC (FOR OPTION PRICING) ---
def get_nyse_holidays(start_year, end_year):
    holidays = []
    for year in range(start_year, end_year + 1):
        fixed_dates = {
            "New Years": datetime.date(year, 1, 1),
            "Juneteenth": datetime.date(year, 6, 19),
            "Independence": datetime.date(year, 7, 4),
            "Christmas": datetime.date(year, 12, 25)
        }
        for name, date in fixed_dates.items():
            if date.weekday() == 5: holidays.append(date - datetime.timedelta(days=1))
            elif date.weekday() == 6: holidays.append(date + datetime.timedelta(days=1))
            else: holidays.append(date)

        holidays.append(datetime.date(year, 1, 1) + relativedelta(day=1, weekday=MO(3))) # MLK
        holidays.append(datetime.date(year, 2, 1) + relativedelta(day=1, weekday=MO(3))) # Pres
        holidays.append(datetime.date(year, 5, 31) + relativedelta(weekday=MO(-1)))      # Memorial
        holidays.append(datetime.date(year, 9, 1) + relativedelta(day=1, weekday=MO(1))) # Labor
        holidays.append(datetime.date(year, 11, 1) + relativedelta(day=1, weekday=TH(4)))# Thx
        holidays.append(easter(year) - datetime.timedelta(days=2)) # Good Fri

    return sorted(list(set(holidays)))

current_year = datetime.date.today().year
generated_holidays = get_nyse_holidays(current_year, current_year + 1)
nyse_holidays_dt = np.array(generated_holidays)

def get_trading_days(expiry_date):
    start = datetime.date.today()
    if isinstance(expiry_date, datetime.datetime): end = expiry_date.date()
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
    
    # Calculate enough targets for the heatmap/ensemble
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
# OPTION PRICING & STRATEGY LOGIC
# -----------------------------------------------------------------------------
def fetch_option_chain(ticker):
    stock = yf.Ticker(ticker)
    try:
        spot = stock.history(period='1d')['Close'].iloc[-1]
    except:
        return None, [], 0
        
    expiries = stock.options
    all_opts = []
    
    # Only grab next 6 expiries to save time/bandwidth
    for date_str in expiries[:6]:
        try:
            chain = stock.option_chain(date_str)
            calls = chain.calls
            calls['Type'] = 'call'
            puts = chain.puts
            puts['Type'] = 'put'
            
            combined = pd.concat([calls, puts])
            combined['Expiry'] = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
            all_opts.append(combined)
        except: pass
        
    if not all_opts: return None, [], spot
    
    df_opts = pd.concat(all_opts)
    # Basic cleaning
    df_opts = df_opts[(df_opts['bid'] > 0.01) & (df_opts['ask'] > 0.01)]
    df_opts['Mid'] = (df_opts['bid'] + df_opts['ask']) / 2
    
    # Filter for reasonable strikes (+/- 30% of spot)
    df_opts = df_opts[
        (df_opts['strike'] > spot * 0.7) & 
        (df_opts['strike'] < spot * 1.3)
    ]
    
    return df_opts, sorted(list(set(df_opts['Expiry']))), spot

def get_simulated_prices(df_hist, neighbors, days_forward, current_price):
    """
    Calculates what the price WOULD be if the returns from the neighbors
    occurred starting today.
    """
    # Look forward 'days_forward' in the full history for each neighbor
    future_returns = []
    
    for idx in neighbors.index:
        loc = df_hist.index.get_loc(idx)
        future_loc = loc + days_forward
        
        if future_loc < len(df_hist):
            # Calculate exact return over this window
            # Future Close / Neighbor Close - 1
            ret = df_hist['Close'].iloc[future_loc] / df_hist['Close'].iloc[loc] - 1
            future_returns.append(ret)
            
    if not future_returns: return []
    
    # Apply to current price
    sim_prices = [current_price * (1 + r) for r in future_returns]
    return np.array(sim_prices)

def calculate_option_fair_value(simulated_prices, strike, opt_type):
    if len(simulated_prices) == 0: return 0
    if opt_type == 'call':
        payoffs = np.maximum(0, simulated_prices - strike)
    else:
        payoffs = np.maximum(0, strike - simulated_prices)
    return np.mean(payoffs)

def generate_strategies(df_chain, sim_prices, spot):
    strategies = []
    
    # 1. Independent Longs (Naked)
    for _, row in df_chain.iterrows():
        fair_val = calculate_option_fair_value(sim_prices, row['strike'], row['Type'])
        mkt_val = row['Mid']
        edge = fair_val - mkt_val
        
        # Filter for minimal edge
        if edge > mkt_val * 0.1 and mkt_val > 0.05:
            strategies.append({
                "Type": "Long " + row['Type'].capitalize(),
                "Strikes": f"{row['strike']}",
                "Expiry": row['Expiry'],
                "Cost": mkt_val,
                "Fair Value": fair_val,
                "Edge": edge,
                "ROI": (edge / mkt_val) * 100
            })
            
    # 2. Vertical Spreads
    # Group by Expiry and Type
    for (expiry, otype), group in df_chain.groupby(['Expiry', 'Type']):
        strikes = sorted(group['strike'].unique())
        # Iterate combinations
        for s1, s2 in itertools.combinations(strikes, 2):
            # Ensure width is decent (at least 0.5% of spot)
            if (s2 - s1) < (spot * 0.005): continue
            
            # Identify legs
            leg1 = group[group['strike'] == s1].iloc[0]
            leg2 = group[group['strike'] == s2].iloc[0]
            
            # Call Spread (Long lower, Short higher)
            if otype == 'call':
                cost = leg1['Mid'] - leg2['Mid']
                if cost <= 0: continue
                
                # Payoff = Max(0, Min(S - K1, K2 - K1)) - Cost
                # We calculate Fair Value of the spread = FV(Long) - FV(Short)
                fv1 = calculate_option_fair_value(sim_prices, s1, 'call')
                fv2 = calculate_option_fair_value(sim_prices, s2, 'call')
                fair_spread = fv1 - fv2
                edge = fair_spread - cost
                
                if edge > cost * 0.15:
                    strategies.append({
                        "Type": "Bull Call Spread",
                        "Strikes": f"{s1}/{s2}",
                        "Expiry": expiry,
                        "Cost": cost,
                        "Fair Value": fair_spread,
                        "Edge": edge,
                        "ROI": (edge / cost) * 100
                    })
            
            # Put Spread (Long higher, Short lower)
            else: # Put
                cost = leg2['Mid'] - leg1['Mid'] # Long 2 (high), Short 1 (low)
                if cost <= 0: continue
                
                fv_long = calculate_option_fair_value(sim_prices, s2, 'put')
                fv_short = calculate_option_fair_value(sim_prices, s1, 'put')
                fair_spread = fv_long - fv_short
                edge = fair_spread - cost
                
                if edge > cost * 0.15:
                    strategies.append({
                        "Type": "Bear Put Spread",
                        "Strikes": f"{s2}/{s1}",
                        "Expiry": expiry,
                        "Cost": cost,
                        "Fair Value": fair_spread,
                        "Edge": edge,
                        "ROI": (edge / cost) * 100
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
        analysis_start = st.date_input("Start Date", datetime.date(2000, 1, 1))

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
            
            # --- EUCLIDEAN NEIGHBORS ---
            neighbors = get_euclidean_neighbors(df, rank_cols, [c for c in df.columns if c.startswith("Mkt_")], n_neighbors=k_neighbors)
            st.success(f"Analysis Complete. Found {len(neighbors)} Euclidean matches.")
            
            # Store in session state for the option lab to use
            st.session_state['full_df'] = df
            st.session_state['neighbors'] = neighbors
            st.session_state['ticker'] = ticker

    # --- OPTION LAB SECTION ---
    if 'full_df' in st.session_state:
        st.divider()
        st.header("🧪 Option Strategy Lab")
        st.markdown("""
        **Method:** Scrapes live option chains and compares Market Prices vs. Theoretical Prices derived from the **Euclidean Neighbor** distribution.
        """)
        
        c_opt1, c_opt2 = st.columns([1, 3])
        with c_opt1:
            min_dte = st.number_input("Min DTE", 0, 60, 0)
            max_dte = st.number_input("Max DTE", 0, 90, 21)
            run_opt = st.button("🔎 Scan Options Chain")
            
        if run_opt:
            with st.spinner("Fetching live options & calculating fair values..."):
                df_opts, expiries, spot = fetch_option_chain(st.session_state['ticker'])
                
                if df_opts is None or df_opts.empty:
                    st.error("Could not fetch options data.")
                else:
                    valid_expiries = []
                    all_strategies = []
                    
                    # Process Front 3 Valid Expiries
                    processed_count = 0
                    
                    # Store data for the very front expiry visualization
                    viz_data = None
                    
                    for exp in expiries:
                        days = get_trading_days(exp)
                        if days < min_dte or days > max_dte: continue
                        if processed_count >= 3: break # Limit to 3 expiries to be fast
                        
                        processed_count += 1
                        valid_expiries.append(exp)
                        
                        # Get Simulated Prices for this specific time horizon
                        sim_prices = get_simulated_prices(st.session_state['full_df'], st.session_state['neighbors'], days, spot)
                        
                        if len(sim_prices) > 0:
                            # Generate Strategies
                            subset_chain = df_opts[df_opts['Expiry'] == exp]
                            strats = generate_strategies(subset_chain, sim_prices, spot)
                            if not strats.empty:
                                all_strategies.append(strats)
                                
                            # Capture data for the FIRST expiry for the chart
                            if viz_data is None:
                                viz_data = {
                                    'expiry': exp,
                                    'days': days,
                                    'sim_prices': sim_prices,
                                    'spot': spot
                                }

                    # --- RESULTS ---
                    if all_strategies:
                        master_df = pd.concat(all_strategies)
                        
                        # Top 5 Spreads
                        spreads = master_df[master_df['Type'].str.contains("Spread")].sort_values("Edge", ascending=False).head(5)
                        
                        # Top 3 Naked
                        naked = master_df[~master_df['Type'].str.contains("Spread")].sort_values("Edge", ascending=False).head(3)
                        
                        st.subheader(f"🏆 Top Trade Ideas (Next {processed_count} Expiries)")
                        
                        col_tbl1, col_tbl2 = st.columns(2)
                        with col_tbl1:
                            st.write("**Top 5 Vertical Spreads**")
                            st.dataframe(spreads[['Type', 'Expiry', 'Strikes', 'Cost', 'Fair Value', 'Edge', 'ROI']].style.format({
                                'Cost': '${:.2f}', 'Fair Value': '${:.2f}', 'Edge': '${:.2f}', 'ROI': '{:.1f}%'
                            }).background_gradient(subset=['Edge'], cmap='Greens'), use_container_width=True)
                            
                        with col_tbl2:
                            st.write("**Top 3 Independent Longs**")
                            st.dataframe(naked[['Type', 'Expiry', 'Strikes', 'Cost', 'Fair Value', 'Edge', 'ROI']].style.format({
                                'Cost': '${:.2f}', 'Fair Value': '${:.2f}', 'Edge': '${:.2f}', 'ROI': '{:.1f}%'
                            }).background_gradient(subset=['Edge'], cmap='Greens'), use_container_width=True)
                            
                        # --- VISUALIZATION (FRONT EXPIRY) ---
                        if viz_data:
                            st.divider()
                            st.subheader(f"📊 Front Expiry Distribution Analysis: {viz_data['expiry']} ({viz_data['days']} Trading Days)")
                            
                            sims = viz_data['sim_prices']
                            curr = viz_data['spot']
                            mean_sim = np.mean(sims)
                            
                            fig = go.Figure()
                            # Forecast Dist
                            fig.add_trace(go.Histogram(
                                x=sims, nbinsx=50, name='Model Distribution', marker_color='rgba(0, 0, 255, 0.4)', opacity=0.6
                            ))
                            
                            fig.add_vline(x=curr, line_width=3, line_color="black", annotation_text="Current Spot")
                            fig.add_vline(x=mean_sim, line_width=3, line_dash="dot", line_color="blue", annotation_text=f"Model Mean (${mean_sim:.2f})")
                            
                            # Add B/E lines for the #1 Spread if available
                            if not spreads.empty:
                                best_spread = spreads.iloc[0]
                                if best_spread['Expiry'] == viz_data['expiry']:
                                    strikes = [float(x) for x in best_spread['Strikes'].split('/')]
                                    fig.add_vline(x=strikes[0], line_color="green", line_dash="dash", annotation_text="Long Strike")
                                    fig.add_vline(x=strikes[1], line_color="red", line_dash="dash", annotation_text="Short Strike")
                                    st.caption(f"Chart includes strikes for top spread: {best_spread['Type']} ({best_spread['Strikes']})")

                            fig.update_layout(
                                title=f"Price Distribution Forecast vs. Spot",
                                xaxis_title="Price",
                                yaxis_title="Frequency",
                                template="plotly_white",
                                bargap=0.1
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                    else:
                        st.warning("No trades found with positive edge criteria.")

def main():
    st.set_page_config(layout="wide", page_title="Heatmap & Option Lab")
    render_heatmap()

if __name__ == "__main__":
    main()
