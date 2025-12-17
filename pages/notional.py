import streamlit as st
import pandas as pd
import yfinance as yf

# --- CONFIGURATION: EXPANDED UNIVERSE ---
# Format: 'Label': {'yf_ticker': '...', 'multiplier': ..., 'tick_size': ...}
FUTURES_SPECS = {
    # --- INDICES ---
    'E-mini S&P 500':      {'yf': 'ES=F',  'mult': 50,   'tick': 0.25, 'sector': 'Index'},
    'Micro E-mini S&P':    {'yf': 'MES=F', 'mult': 5,    'tick': 0.25, 'sector': 'Index'},
    'E-mini Nasdaq 100':   {'yf': 'NQ=F',  'mult': 20,   'tick': 0.25, 'sector': 'Index'},
    'Micro E-mini NQ':     {'yf': 'MNQ=F', 'mult': 2,    'tick': 0.25, 'sector': 'Index'},
    'E-mini Russell 2000': {'yf': 'RTY=F', 'mult': 50,   'tick': 0.10, 'sector': 'Index'},
    'E-mini Dow ($5)':     {'yf': 'YM=F',  'mult': 5,    'tick': 1.00, 'sector': 'Index'},
    'VIX Futures':         {'yf': 'VX=F',  'mult': 1000, 'tick': 0.05, 'sector': 'Index'},

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

def get_futures_prices():
    tickers = [spec['yf'] for spec in FUTURES_SPECS.values()]
    try:
        data = yf.download(tickers, period="1d", progress=False)['Close']
        if isinstance(data, pd.DataFrame):
            return data.iloc[-1]
        else:
            return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.Series()

def futures_dashboard():
    st.set_page_config(layout="wide", page_title="Futures Specs")
    st.title("🚜 Futures Contract Specs & Notional Values")
    st.markdown("Liquid contracts tradeable on IBKR, including Mini/Micro variations.")

    if st.button("Refresh Prices"):
        st.rerun()

    # 1. Fetch Live Prices
    with st.spinner("Fetching live quotes from Yahoo Finance..."):
        prices = get_futures_prices()

    # 2. Build the Table
    table_data = []
    
    for name, spec in FUTURES_SPECS.items():
        yf_tick = spec['yf']
        sector = spec['sector']
        mult = spec['mult']
        
        # Get Price
        try:
            price = prices[yf_tick] if yf_tick in prices else 0.0
            if pd.isna(price): price = 0.0
        except:
            price = 0.0
            
        # --- CALCULATION LOGIC ---
        # "calc_mult" handles the conversion from Cents-to-Dollars for Notional calc.
        # Standard Ags (Corn/Soy/Wheat) are 5000 bu. Quote is cents. Notional = Price * 50.
        # Mini Ags (XC, XK, XW) are 1000 bu. Quote is cents. Notional = Price * 10.
        
        calc_mult = mult # Default assumption: Price is in dollars
        
        if sector == 'Ags':
            if name in ['Corn (5000bu)', 'Soybeans (5000bu)', 'Wheat (5000bu)']:
                calc_mult = 50 
            elif name in ['Mini Corn (1000bu)', 'Mini Soybeans (1k)', 'Mini Wheat (1000bu)']:
                calc_mult = 10
            elif name == 'Sugar': 
                # Quote 22.00 (cents/lb). 112,000 lbs. Notional = 0.22 * 112,000 = 24,640.
                # Calculation: 22.00 * 1120 = 24,640.
                calc_mult = 1120
            elif name == 'Coffee':
                # Quote 250.00 (cents/lb). 37,500 lbs. Notional = 2.50 * 37,500.
                # Calculation: 250 * 375 = 93,750.
                calc_mult = 375
        
        notional = price * calc_mult
        tick_val_dollar = spec['tick'] * calc_mult

        table_data.append({
            "Contract": name,
            "Sector": sector,
            "Ticker (YF)": yf_tick,
            "Price": price,
            "Multiplier": f"x{mult}",
            "Tick Size": spec['tick'],
            "Tick Value ($)": tick_val_dollar,
            "Notional Value ($)": notional
        })

    df = pd.DataFrame(table_data)
    
    # 3. Formatting
    df = df.sort_values(by=['Sector', 'Notional Value ($)'], ascending=[True, False])

    st.dataframe(
        df,
        column_config={
            "Price": st.column_config.NumberColumn(format="%.2f"),
            "Tick Size": st.column_config.NumberColumn(format="%.5f"),
            "Tick Value ($)": st.column_config.NumberColumn(format="$%.2f"),
            "Notional Value ($)": st.column_config.NumberColumn(format="$%.2f"),
        },
        use_container_width=True,
        hide_index=True,
        height=1000
    )

if __name__ == "__main__":
    futures_dashboard()
