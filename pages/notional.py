import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np

# --- CONFIGURATION ---
FUTURES_SPECS = {
    # --- INDICES ---
    'E-mini S&P 500':      {'yf': 'ES=F',  'mult': 50,   'tick': 0.25, 'sector': 'Index'},
    'Micro E-mini S&P':    {'yf': 'MES=F', 'mult': 5,    'tick': 0.25, 'sector': 'Index'},
    'E-mini Nasdaq 100':   {'yf': 'NQ=F',  'mult': 20,   'tick': 0.25, 'sector': 'Index'},
    'Micro E-mini NQ':     {'yf': 'MNQ=F', 'mult': 2,    'tick': 0.25, 'sector': 'Index'},
    'E-mini Russell 2000': {'yf': 'RTY=F', 'mult': 50,   'tick': 0.10, 'sector': 'Index'},
    'E-mini Dow ($5)':     {'yf': 'YM=F',  'mult': 5,    'tick': 1.00, 'sector': 'Index'},
    'VIX Futures':         {'yf': 'VX=F',  'mult': 1000, 'tick': 0.05, 'sector': 'Index'},
    'Micro VIX Futures':   {'yf': 'VXM=F', 'mult': 100,  'tick': 0.05, 'sector': 'Index'}, # Added

    # --- CURRENCIES ---
    'Euro FX':         {'yf': '6E=F', 'mult': 125000,   'tick': 0.00005,   'sector': 'Currency'},
    'Japanese Yen':    {'yf': '6J=F', 'mult': 12500000, 'tick': 0.0000005, 'sector': 'Currency'},
    'British Pound':   {'yf': '6B=F', 'mult': 62500,    'tick': 0.0001,    'sector': 'Currency'},
    'Canadian Dollar': {'yf': '6C=F', 'mult': 100000,   'tick': 0.00005,   'sector': 'Currency'},
    'Australian Dollar':{'yf': '6A=F', 'mult': 100000,   'tick': 0.0001,    'sector': 'Currency'},

    # --- ENERGY ---
    'Crude Oil (WTI)': {'yf': 'CL=F',  'mult': 1000,  'tick': 0.01,   'sector': 'Energy'},
    'Micro Crude Oil': {'yf': 'MCL=F', 'mult': 100,   'tick': 0.01,   'sector': 'Energy'},
    'Natural Gas':     {'yf': 'NG=F',  'mult': 10000, 'tick': 0.001,  'sector': 'Energy'},
    'Heating Oil':     {'yf': 'HO=F',  'mult': 42000, 'tick': 0.0001, 'sector': 'Energy'},
    'RBOB Gasoline':   {'yf': 'RB=F',  'mult': 42000, 'tick': 0.0001, 'sector': 'Energy'},

    # --- METALS ---
    'Gold (100oz)':        {'yf': 'GC=F',  'mult': 100,  'tick': 0.10,  'sector': 'Metals'},
    'E-mini Gold (50oz)':  {'yf': 'QO=F',  'mult': 50,   'tick': 0.25,  'sector': 'Metals'},
    'Micro Gold (10oz)':   {'yf': 'MGC=F', 'mult': 10,   'tick': 0.10,  'sector': 'Metals'},
    'Silver (5000oz)':     {'yf': 'SI=F',  'mult': 5000, 'tick': 0.005, 'sector': 'Metals'},
    'Micro Silver':        {'yf': 'SIL=F', 'mult': 1000, 'tick': 0.01,  'sector': 'Metals'},
    'Copper':              {'yf': 'HG=F',  'mult': 25000,'tick': 0.0005,'sector': 'Metals'},

    # --- RATES ---
    '10-Year T-Note':     {'yf': 'ZN=F',  'mult': 1000, 'tick': 0.015625, 'sector': 'Rates'},
    'Ultra 10-Year':      {'yf': 'TN=F',  'mult': 1000, 'tick': 0.015625, 'sector': 'Rates'},
    '30-Year T-Bond':     {'yf': 'ZB=F',  'mult': 1000, 'tick': 0.03125,  'sector': 'Rates'},
    '2-Year T-Note':      {'yf': 'ZT=F',  'mult': 2000, 'tick': 0.0078125,'sector': 'Rates'},
    'Micro 10Y Yield':    {'yf': '10Y=F', 'mult': 100,  'tick': 0.001,    'sector': 'Rates'},

    # --- GRAINS / SOFTS ---
    'Corn (5000bu)':       {'yf': 'ZC=F', 'mult': 5000,  'tick': 0.25,  'sector': 'Ags'},
    'Mini Corn (1000bu)':  {'yf': 'XC=F', 'mult': 1000,  'tick': 0.125, 'sector': 'Ags'},
    'Soybeans (5000bu)':   {'yf': 'ZS=F', 'mult': 5000,  'tick': 0.25,  'sector': 'Ags'},
    'Mini Soybeans (1k)':  {'yf': 'XK=F', 'mult': 1000,  'tick': 0.125, 'sector': 'Ags'},
    'Wheat (5000bu)':      {'yf': 'ZW=F', 'mult': 5000,  'tick': 0.25,  'sector': 'Ags'},
    'Mini Wheat (1000bu)': {'yf': 'XW=F', 'mult': 1000,  'tick': 0.125, 'sector': 'Ags'},
    'Coffee': {'yf': 'KC=F', 'mult': 37500,  'tick': 0.05, 'sector': 'Ags'}, 
    'Sugar':  {'yf': 'SB=F', 'mult': 112000, 'tick': 0.01, 'sector': 'Ags'}, 
}

def get_market_data():
    """
    Fetches 1 month of history to calculate ATR and get current Price.
    Returns a dict: {ticker: {'price': float, 'atr': float}}
    """
    tickers = [spec['yf'] for spec in FUTURES_SPECS.values()]
    
    # We need ~1 month to get a valid 14-period ATR
    try:
        df = yf.download(tickers, period="1mo", progress=False, group_by='ticker')
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return {}
        
    data_map = {}
    
    # yf.download with group_by='ticker' returns a MultiIndex if len(tickers) > 1
    # Structure: df[Ticker][Open/High/Low/Close]
    
    for ticker in tickers:
        try:
            # Handle single ticker vs multi-ticker structure
            if len(tickers) == 1:
                ticker_df = df
            else:
                ticker_df = df[ticker]
            
            # Check if empty (sometimes YF returns empty cols for bad tickers)
            if ticker_df.empty or ticker_df['Close'].isnull().all():
                data_map[ticker] = {'price': 0.0, 'atr': 0.0}
                continue

            # 1. Get Current Price (Last Close)
            last_price = ticker_df['Close'].iloc[-1]
            if pd.isna(last_price): last_price = 0.0
            
            # 2. Calculate ATR (14)
            # TR = max(High - Low, abs(High - PrevClose), abs(Low - PrevClose))
            high = ticker_df['High']
            low = ticker_df['Low']
            close = ticker_df['Close']
            prev_close = close.shift(1)
            
            tr1 = high - low
            tr2 = (high - prev_close).abs()
            tr3 = (low - prev_close).abs()
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_series = tr.rolling(window=14).mean()
            last_atr = atr_series.iloc[-1]
            
            if pd.isna(last_atr): last_atr = 0.0
            
            data_map[ticker] = {'price': last_price, 'atr': last_atr}
            
        except Exception as e:
            # Fail gracefully for individual tickers
            data_map[ticker] = {'price': 0.0, 'atr': 0.0}
            
    return data_map

def futures_dashboard():
    st.set_page_config(layout="wide", page_title="Futures Specs")
    st.title("🚜 Futures Specs & Risk Analysis")
    st.markdown("Liquid contracts with Notional Value, ATR, and Implied Risk.")

    if st.button("Refresh Data"):
        st.rerun()

    # 1. Fetch Data
    with st.spinner("Fetching OHLC data for ATR calculation..."):
        market_data = get_market_data()

    # 2. Build Table
    table_data = []
    
    for name, spec in FUTURES_SPECS.items():
        yf_tick = spec['yf']
        sector = spec['sector']
        mult = spec['mult']
        
        # Retrieve data
        data = market_data.get(yf_tick, {'price': 0.0, 'atr': 0.0})
        price = data['price']
        atr = data['atr']
        
        # --- MULTIPLIER LOGIC (Cents vs Dollars) ---
        calc_mult = mult 
        
        if sector == 'Ags':
            if name in ['Corn (5000bu)', 'Soybeans (5000bu)', 'Wheat (5000bu)']:
                calc_mult = 50 
            elif name in ['Mini Corn (1000bu)', 'Mini Soybeans (1k)', 'Mini Wheat (1000bu)']:
                calc_mult = 10
            elif name == 'Sugar': calc_mult = 1120
            elif name == 'Coffee': calc_mult = 375

        # --- CALCULATIONS ---
        notional = price * calc_mult
        
        # User Formula: ATR Implied ($) = (ATR / Price) * 0.75 * Notional
        # Note: If price is 0, avoid division by zero
        if price > 0:
            atr_implied = (atr / price) * 0.75 * notional
        else:
            atr_implied = 0.0
            
        tick_val_dollar = spec['tick'] * calc_mult

        table_data.append({
            "Contract": name,
            "Sector": sector,
            "Ticker": yf_tick,
            "Price": price,
            "Multiplier": f"x{mult}",
            "Notional ($)": notional,
            "ATR (14d)": atr,
            "ATR Implied ($)": atr_implied
        })

    df = pd.DataFrame(table_data)
    
    # 3. Formatting
    df = df.sort_values(by=['Sector', 'Notional ($)'], ascending=[True, False])

    st.dataframe(
        df,
        column_config={
            "Price": st.column_config.NumberColumn(format="%.2f"),
            "Notional ($)": st.column_config.NumberColumn(format="$%.2f"),
            "ATR (14d)": st.column_config.NumberColumn(format="%.2f"),
            "ATR Implied ($)": st.column_config.NumberColumn(format="$%.2f", help="(ATR/Price) * 0.75 * Notional"),
        },
        use_container_width=True,
        hide_index=True,
        height=1000
    )

if __name__ == "__main__":
    futures_dashboard()
