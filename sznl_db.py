import pandas as pd
import yfinance as yf
import numpy as np
import sqlite3
import datetime as dt
from pandas.tseries.offsets import CustomBusinessDay

# --- CONFIGURATION ---
DB_NAME = 'seasonal_ranks_2025_2026.db'
TICKERS = ['^GSPC', '^NDX']
FORECAST_YEARS = [2025, 2026]

# Hardcoded NYSE Holidays for 2025 and 2026
# Ensures we don't generate forecast rows for days the market is closed.
NYSE_HOLIDAYS = [
    # 2025
    '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18', # Good Friday
    '2025-05-26', '2025-06-19', '2025-07-04', '2025-09-01', 
    '2025-11-27', '2025-12-25',
    # 2026
    '2026-01-01', '2026-01-19', '2026-02-16', '2026-04-03', # Good Friday
    '2026-05-25', '2026-06-19', '2026-07-03', # Independence Day Observed
    '2026-09-07', '2026-11-26', '2026-12-25',
]

def get_forward_returns(df, windows=[5, 10, 21]):
    """Pre-calculates forward returns for the entire dataset."""
    df = df.copy()
    # Log returns: ln(Close / PrevClose) * 100
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1)) * 100
    
    for w in windows:
        df[f'Fwd_{w}d'] = df['log_ret'].shift(-w).rolling(w).sum()
    
    df['Year'] = df.index.year
    df['DayOfYear'] = df.index.dayofyear
    return df

def calculate_seasonal_profile(df, target_year):
    """
    Calculates the 0-100 rank profile based on history PRIOR to target_year.
    """
    # 1. Knowledge Cutoff: Use only past data
    past_data = df[df['Year'] < target_year].copy()
    
    # Use 365 days of data if calculating for specific year, 
    # but use full dataset if calculating forecast for future (simulated by passing year > max)
    if past_data.empty: 
        # If target_year is in the far future, 'past_data' might be empty if we filtered too aggressively.
        # However, usually we pass the full df.
        pass

    # 2. Identify Cycle
    cycle_remainder = target_year % 4
    
    # 3. Create Profiles
    # A. All Years
    stats_all = past_data.groupby('DayOfYear')[[f'Fwd_{w}d' for w in [5, 10, 21]]].mean()
    
    # B. Cycle Specific
    cycle_data = past_data[past_data['Year'] % 4 == cycle_remainder]
    
    if cycle_data.empty:
        stats_cycle = stats_all
    else:
        stats_cycle = cycle_data.groupby('DayOfYear')[[f'Fwd_{w}d' for w in [5, 10, 21]]].mean()
        
    # 4. Rank
    rank_all = stats_all.rank(pct=True) * 100
    rank_cycle = stats_cycle.rank(pct=True) * 100
    
    # Reindex/Fill for missing days (leap years etc)
    full_idx = pd.RangeIndex(start=1, stop=367) # Day 1 to 366
    rank_all = rank_all.reindex(full_idx).interpolate(method='nearest').fillna(50)
    rank_cycle = rank_cycle.reindex(full_idx).interpolate(method='nearest').fillna(50)

    # 5. Weighted Avg
    avg_rank_all = rank_all.mean(axis=1)
    avg_rank_cycle = rank_cycle.mean(axis=1)
    
    final_profile = (avg_rank_all + 3 * avg_rank_cycle) / 4
    
    # 6. Smooth
    return final_profile.rolling(5, center=True, min_periods=1).mean()

def generate_walk_forward_ranks(df):
    """Generates HISTORICAL ranks using expanding window."""
    years = sorted(df['Year'].unique())
    results = []
    start_idx = 4 
    
    for i in range(start_idx, len(years)):
        target_year = years[i]
        profile = calculate_seasonal_profile(df, target_year)
        
        if profile is not None:
            target_data = df[df['Year'] == target_year].copy()
            target_data['seasonal_rank'] = target_data['DayOfYear'].map(profile)
            results.append(target_data[['seasonal_rank']])

    if not results: return pd.DataFrame()
    return pd.concat(results)

def generate_forecast(df, forecast_years):
    """Generates FUTURE ranks using custom market calendar."""
    results = []
    
    # Define our Custom Market Calendar to skip holidays
    market_freq = CustomBusinessDay(holidays=NYSE_HOLIDAYS)
    
    # Calculate profile using ALL available data (simulating we are at end of data)
    # We pass a year far in the future to ensure all 'df' is treated as 'past_data'
    # inside the helper, or we can just pass the max year + 1.
    max_year = df['Year'].max()
    
    for year in forecast_years:
        # Calculate profile specifically for the cycle of the forecast year
        # We assume we have knowledge up to the end of the current dataset.
        # To reuse the helper, we temporarily modify 'df' or just re-implement logic briefly:
        
        # Logic: We use ALL data in 'df' to predict 'year'.
        # Since 'df' is all historical, we just use it directly.
        cycle_remainder = year % 4
        
        stats_all = df.groupby('DayOfYear')[[f'Fwd_{w}d' for w in [5, 10, 21]]].mean()
        cycle_data = df[df['Year'] % 4 == cycle_remainder]
        
        if cycle_data.empty: stats_cycle = stats_all
        else: stats_cycle = cycle_data.groupby('DayOfYear')[[f'Fwd_{w}d' for w in [5, 10, 21]]].mean()
            
        rank_all = stats_all.rank(pct=True) * 100
        rank_cycle = stats_cycle.rank(pct=True) * 100
        
        full_idx = pd.RangeIndex(start=1, stop=367)
        rank_all = rank_all.reindex(full_idx).interpolate(method='nearest').fillna(50)
        rank_cycle = rank_cycle.reindex(full_idx).interpolate(method='nearest').fillna(50)
        
        final_profile = ((rank_all.mean(axis=1) + 3 * rank_cycle.mean(axis=1)) / 4).rolling(5, center=True, min_periods=1).mean()
        
        # Generate Dates
        start = f"{year}-01-01"
        end = f"{year}-12-31"
        dates = pd.date_range(start=start, end=end, freq=market_freq)
        
        future_df = pd.DataFrame(index=dates)
        future_df['DayOfYear'] = future_df.index.dayofyear
        future_df['seasonal_rank'] = future_df['DayOfYear'].map(final_profile)
        
        results.append(future_df[['seasonal_rank']])
        
    return pd.concat(results)

def process_tickers(ticker_list):
    conn = sqlite3.connect(DB_NAME)
    
    # Initialize DB
    conn.execute('''CREATE TABLE IF NOT EXISTS seasonal_ranks (
                    Date TIMESTAMP,
                    seasonal_rank REAL,
                    ticker TEXT,
                    PRIMARY KEY (Date, ticker)
                )''')
    
    for ticker in ticker_list:
        print(f"Processing {ticker}...")
        try:
            # 1. Download
            raw_df = yf.download(ticker, period="max", progress=False, auto_adjust=True)
            if isinstance(raw_df.columns, pd.MultiIndex): 
                raw_df.columns = raw_df.columns.get_level_values(0)
            
            if raw_df.empty:
                print(f"  No data for {ticker}")
                continue

            # 2. Process Returns
            df_processed = get_forward_returns(raw_df)
            
            # 3. Generate Historical Ranks (Backtest)
            hist_ranks = generate_walk_forward_ranks(df_processed)
            
            # 4. Generate Future Forecast
            future_ranks = generate_forecast(df_processed, FORECAST_YEARS)
            
            # 5. Merge and Deduplicate
            # We prefer Historical data (realized) over Forecast data for dates that overlap.
            combined = pd.concat([hist_ranks, future_ranks])
            
            # Remove duplicates based on Index (Date), keeping the first occurrence (Historical)
            combined = combined[~combined.index.duplicated(keep='first')]
            
            # 6. Format and Save
            combined['ticker'] = ticker
            combined = combined.reset_index().rename(columns={'index': 'Date'})
            
            # Filter for Year > 2000
            combined = combined[combined['Date'].dt.year > 2000]
            
            conn.execute("DELETE FROM seasonal_ranks WHERE ticker = ?", (ticker,))
            combined.to_sql('seasonal_ranks', conn, if_exists='append', index=False)
            print(f"  Saved {len(combined)} rows (History + Forecast).")
            
        except Exception as e:
            print(f"  Error processing {ticker}: {e}")
            
    conn.commit()
    conn.close()
    print("Done.")

if __name__ == "__main__":
    process_tickers(TICKERS)