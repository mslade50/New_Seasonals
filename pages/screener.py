import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from pandas.tseries.offsets import BusinessDay

# -----------------------------------------------------------------------------
# 1. THE STRATEGY BOOK (STATIC BATCH)
# -----------------------------------------------------------------------------
# To add a new strategy, copy this dictionary structure and paste it into the list.

STRATEGY_BOOK = [
    {
        "id": "IND_OS_SZNL",
        "name": "Oversold Indices + Bullish Seasonality",
        "description": "Buying major indices when short-term momentum is washed out but seasonal tailwinds are strong.",
        "universe_tickers": ["SPY", "QQQ", "IWM", "DIA", "SMH"], 
        "settings": {
            "use_perf_rank": True, "perf_window": 5, "perf_logic": "<", "perf_thresh": 15.0,
            "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 80.0,
            "use_52w": False, "52w_type": None,
            "use_vol": False, "vol_thresh": None,
            "min_price": 10.0, "min_vol": 100000,
        },
        "execution": {
            "risk_per_trade": 1000,
            "stop_atr": 3.0,
            "tgt_atr": 8.0,
            "hold_days": 10
        },
        "stats": {
            "grade": "B (Good)",
            "win_rate": "88.2%",
            "expectancy": "$995.68",
            "profit_factor": "8.38"
        }
    },
    {
    "id": "STRAT_1763673538",
    "name": "Generated Strategy (A)",
    "description": "Universe: All CSV Tickers. Sznl >80 5d perf < 15 (3d lookback) Filter: SPY > 200 SMA.",
    "universe_tickers": ['AAPL', 'AMGN', 'AMZN', 'AVGO', 'AXP', 'BA', 'CAT', 'CEF', 'CRM', 'CSCO', 'CVX', 'DIA', 'DIS', 'GLD', 'GOOG', 'GS', 'HD', 'HON', 'IBB', 'IBM', 'IHI', 'INTC', 'ITA', 'ITB', 'IWM', 'IYR', 'JNJ', 'JPM', 'KO', 'KRE', 'MCD', 'META', 'MMM', 'MRK', 'MSFT', 'NKE', 'NVDA', 'OIH', 'PG', 'QQQ', 'SLV', 'SMH', 'SPY', 'TRV', 'UNG', 'UNH', 'UVXY', 'V', 'VNQ', 'VZ', 'WMT', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOP', 'XRT'], 
    "settings": {
        "use_perf_rank": True, "perf_window": 5, "perf_logic": "<", "perf_thresh": 15.0,
        "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 80.0,
        "use_52w": False, "52w_type": "New 52w High",
        "use_vol": False, "vol_thresh": 1.5,
        "trend_filter": "SPY > 200 SMA",
        "min_price": 10.0, "min_vol": 100000,
        "min_age": 0.25, "max_age": 100.0
    },
    "execution": {
        "risk_per_trade": 1000,
        "stop_atr": 2.0,
        "tgt_atr": 8.0,
        "hold_days": 21
    },
    "stats": {
        "grade": "A (Excellent)",
        "win_rate": "63.9%",
        "expectancy": "$646.42",
        "profit_factor": "2.85"
    }
},
    # ... You can add Strategy #2, Strategy #3 here ...
]

# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
CSV_PATH = "seasonal_ranks.csv"

@st.cache_data(show_spinner=False)
def load_seasonal_map():
    """Loads CSV for seasonality lookups."""
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
    if not t_map: return pd.Series(50.0, index=dates)
    mds = dates.map(lambda x: (x.month, x.day))
    return mds.map(t_map).fillna(50.0)

# -----------------------------------------------------------------------------
# ENGINE
# -----------------------------------------------------------------------------

def calculate_indicators(df, sznl_map, ticker):
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    
    # Perf Ranks
    for window in [5, 10, 21]:
        df[f'ret_{window}d'] = df['Close'].pct_change(window)
        # We use a smaller min_periods here to ensure the screener works on recent downloads
        df[f'rank_ret_{window}d'] = df[f'ret_{window}d'].expanding(min_periods=50).rank(pct=True) * 100.0
        
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    
    # Seasonality
    df['Sznl'] = get_sznl_val_series(ticker, df.index, sznl_map)
    
    # 52w High/Low
    rolling_high = df['High'].shift(1).rolling(252).max()
    rolling_low = df['Low'].shift(1).rolling(252).min()
    df['is_52w_high'] = df['High'] > rolling_high
    df['is_52w_low'] = df['Low'] < rolling_low
    
    # Volume
    vol_ma = df['Volume'].rolling(63).mean()
    df['vol_ratio'] = df['Volume'] / vol_ma
    df['vol_ma'] = vol_ma
    
    return df

def check_signal(row, params):
    """Checks if the specific row matches the strategy parameters."""
    # Liquidity Gates
    if row['Close'] < params['min_price']: return False
    if row['vol_ma'] < params['min_vol']: return False

    # 1. Performance Rank
    if params['use_perf_rank']:
        col = f"rank_ret_{params['perf_window']}d"
        val = row[col]
        if params['perf_logic'] == '<': 
            if not (val < params['perf_thresh']): return False
        else:
            if not (val > params['perf_thresh']): return False

    # 2. Seasonal
    if params['use_sznl']:
        if params['sznl_logic'] == '<':
            if not (row['Sznl'] < params['sznl_thresh']): return False
        else:
            if not (row['Sznl'] > params['sznl_thresh']): return False

    # 3. 52w
    if params['use_52w']:
        if params['52w_type'] == 'New 52w High':
            if not row['is_52w_high']: return False
        else:
            if not row['is_52w_low']: return False

    # 4. Volume
    if params['use_vol']:
        if not (row['vol_ratio'] > params['vol_thresh']): return False
        
    return True

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="Production Strategy Screener")
    st.title("⚡ Daily Strategy Screener")
    st.markdown("---")
    
    sznl_map = load_seasonal_map()
    
    # Container for results
    results_container = st.container()
    
    if st.button("Run All Strategies", type="primary", use_container_width=True):
        
        st.info(f"Scanning {len(STRATEGY_BOOK)} strategies against current market data...")
        
        # Collect unique tickers to download in one batch (efficient)
        all_tickers = set()
        for strat in STRATEGIES:
            all_tickers.update(strat['universe_tickers'])
        all_tickers = list(all_tickers)
        
        # Download last ~400 days (enough for indicators)
        start_date = datetime.date.today() - datetime.timedelta(days=400)
        try:
            raw_data = yf.download(all_tickers, start=start_date, group_by='ticker', progress=False, threads=True)
        except Exception as e:
            st.error(f"Data download failed: {e}")
            return

        # Iterate through the Strategy Book
        for strat in STRATEGIES:
            
            # Create visuals for the Strategy Header
            with st.expander(f"Strategy: {strat['name']} (Grade: {strat['stats']['grade']})", expanded=True):
                
                # 1. Display Stats
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric("Win Rate", strat['stats']['win_rate'])
                s2.metric("Expectancy", strat['stats']['expectancy'])
                s3.metric("Profit Factor", strat['stats']['profit_factor'])
                s5.metric("Risk Unit", f"${strat['execution']['risk_per_trade']}")
                
                st.caption(strat['description'])
                
                # 2. Find Signals
                signals = []
                
                for ticker in strat['universe_tickers']:
                    try:
                        # Extract Ticker Data
                        if len(all_tickers) > 1:
                            if ticker not in raw_data.columns.levels[0]: continue
                            df = raw_data[ticker].copy()
                        else:
                            df = raw_data.copy()

                        df = df.dropna(subset=['Close'])
                        if df.empty: continue
                        
                        # Calc Indicators
                        df = calculate_indicators(df, sznl_map, ticker)
                        
                        # Check Last Row (Today/Yesterday Close)
                        last_row = df.iloc[-1]
                        
                        if check_signal(last_row, strat['settings']):
                            # Calculate Execution Plan
                            atr = last_row['ATR']
                            risk = strat['execution']['risk_per_trade']
                            entry = last_row['Close']
                            
                            stop_dist = atr * strat['execution']['stop_atr']
                            tgt_dist = atr * strat['execution']['tgt_atr']
                            
                            stop_price = entry - stop_dist
                            tgt_price = entry + tgt_dist
                            
                            # Share Sizing: Risk / (Entry - Stop)
                            shares = int(risk / stop_dist) if stop_dist > 0 else 0
                            
                            # Exit Date (Business Days)
                            exit_date = (last_row.name + BusinessDay(strat['execution']['hold_days'])).date()
                            
                            signals.append({
                                "Ticker": ticker,
                                "Signal Date": last_row.name.date(),
                                "Action": "BUY",
                                "Shares": shares,
                                "Entry Price": entry,
                                "Stop Loss": stop_price,
                                "Target": tgt_price,
                                "Est. Exit": exit_date,
                                "ATR": atr
                            })
                            
                    except Exception as e:
                        continue
                
                # 3. Display Signals
                if signals:
                    st.success(f"✅ Found {len(signals)} Actionable Signals")
                    sig_df = pd.DataFrame(signals)
                    
                    st.dataframe(
                        sig_df.style.format({
                            "Entry Price": "${:.2f}",
                            "Stop Loss": "${:.2f}", 
                            "Target": "${:.2f}",
                            "ATR": "{:.2f}"
                        }), 
                        use_container_width=True
                    )
                    
                    # Copyable Text for Execution
                    clip_text = ""
                    for s in signals:
                        clip_text += f"{s['Action']} {s['Shares']} {s['Ticker']} @ MKT. Stop: {s['Stop Loss']:.2f}. Target: {s['Target']:.2f}. Time Exit: {s['Est. Exit']}.\n"
                    st.text_area("Execution Clipboard", clip_text, height=100)
                    
                else:
                    st.info("No signals found for this strategy today.")

# Hack to map the previous variable name to the new list if needed
STRATEGIES = STRATEGY_BOOK

if __name__ == "__main__":
    main()
