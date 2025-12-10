import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import gspread
from pandas.tseries.offsets import BusinessDay

# -----------------------------------------------------------------------------
# 1. THE STRATEGY BOOK
# -----------------------------------------------------------------------------
STRATEGY_BOOK = [
    # 1. INDEX SEASONALS
    {
        "id": "Indx sznl > 85, 21dr < 15",
        "name": "Index Seasonals",
        "description": "Start: 2000-01-01. Universe: Indices. Dir: Long.",
        "universe_tickers": ['SPY', 'QQQ', 'IWM', 'DIA', 'SMH'], 
        "settings": {
            "trade_direction": "Long", "entry_type": "T+1 Open", "max_one_pos": False, "max_daily_entries": 5, "max_total_positions": 10,
            "use_perf_rank": True, "perf_window": 21, "perf_logic": "<", "perf_thresh": 15.0, "perf_first_instance": False, "perf_lookback": 21, "perf_consecutive": 1,
            "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 85.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000, "min_age": 0.25, "max_age": 100.0
        },
        "execution": { "risk_per_trade": 1000, "stop_atr": 3.0, "tgt_atr": 8.0, "hold_days": 21 },
        "stats": { "grade": "A", "win_rate": "64.1%", "expectancy": "$471.36", "profit_factor": "4.51" }
    },
    # 2. BOTTOM OF RANGE REVERSION
    {
        "id": "Lower 10% of range <50% ile 5dr",
        "name": "Bottom of Range Reversion",
        "description": "Start: 2000-01-01. Universe: Indices. Dir: Long.",
        "universe_tickers": ['SPY', 'QQQ', 'IWM', 'DIA', 'SMH', '^GSPC', '^NDX'], 
        "settings": {
            "trade_direction": "Long", "entry_type": "Signal Close", "max_one_pos": True, "allow_same_day_reentry": False, "max_daily_entries": 2, "max_total_positions": 10,
            "perf_filters": [{'window': 5, 'logic': '<', 'thresh': 50.0, 'consecutive': 1}], "perf_first_instance": False, "perf_lookback": 21,
            "use_sznl": False, "sznl_logic": "<", "sznl_thresh": 15.0, "sznl_first_instance": True, "sznl_lookback": 21,
            "use_market_sznl": True, "market_sznl_logic": ">", "market_sznl_thresh": 33.0, "market_ticker": "^GSPC",
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_range_filter": True, 
            "range_min": 0.0,
            "range_max": 10.0,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000, "min_age": 0.25, "max_age": 100.0, "entry_conf_bps": 0,
            "use_dist_filter": False, "dist_ma_type": "SMA 10", "dist_logic": "Greater Than (>)", "dist_min": 0.0, "dist_max": 2.0,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": ">", "gap_thresh": 3
        },
        "execution": { "risk_per_trade": 300, "slippage_bps": 2, "stop_atr": 1.0, "tgt_atr": 2.0, "hold_days": 2 },
        "stats": { "grade": "B", "win_rate": "54.9%", "expectancy": "$174.76", "profit_factor": "1.49" }
    },
    # 3. OVERBOUGHT LIQUID NAMES (SHORT)
    {
        "id": "21dr > 85 3 consec, 5dr > 85, SPX sznl <50",
        "name": "Overbot Liquid Names",
        "description": "Start: 2000-01-01. Universe: All CSV Tickers. Dir: Short.",
        "universe_tickers": ['AAPL', 'ABT', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEP', 'AIG', 'ALL', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'AON', 'APD', 'AVGO', 'AXP', 'BA', 'BAC', 'BAX', 'BDX', 'BK', 'BMY', 'C', 'CAG', 'CAT', 'CEF', 'CL', 'CMCSA', 'CMS', 'CNP', 'COP', 'COST', 'CPB', 'CRM', 'CSCO', 'CSX', 'CVS', 'CVX', 'D', 'DE', 'DIA', 'DIS', 'DOV', 'DTE', 'DUK', 'ECL', 'ED', 'EIX', 'EMR', 'EOG', 'ETR', 'EXC', 'F', 'FCX', 'FDX', 'FE', 'GD', 'GE', 'GILD', 'GIS', 'GLD', 'GLW', 'GOOG', 'GPC', 'GS', 'HAL', 'HD', 'HIG', 'HON', 'HPQ', 'HRL', 'HSY', 'HUM', 'IBB', 'IBM', 'IHI', 'INTC', 'IP', 'ITA', 'ITB', 'ITW', 'IWM', 'IYR', 'JNJ', 'JPM', 'K', 'KEY', 'KMB', 'KO', 'KR', 'KRE', 'LEG', 'LIN', 'LLY', 'LMT', 'LOW', 'LUV', 'MAS', 'MCD', 'MDT', 'MET', 'META', 'MMC', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'MU', 'NEE', 'NEM', 'NKE', 'NOC', 'NSC', 'NUE', 'NVDA', 'OIH', 'ORCL', 'OXY', 'PAYX', 'PCG', 'PEG', 'PEP', 'PFE', 'PG', 'PGR', 'PH', 'PNW', 'PPG', 'PPL', 'PSA', 'QCOM', 'QQQ', 'REGN', 'RF', 'RHI', 'ROK', 'ROST', 'RTX', 'SBUX', 'SCHW', 'SHW', 'SLB', 'SLV', 'SMH', 'SNA', 'SO', 'SPG', 'SPY', 'SRE', 'STT', 'SWK', 'SYK', 'SYY', 'T', 'TAP', 'TGT', 'TJX', 'TMO', 'TRV', 'TSN', 'TXN', 'UNG', 'UNH', 'UNP', 'USB', 'USO', 'UVXY', 'V', 'VFC', 'VLO', 'VMC', 'VNQ', 'VZ', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOM', 'XOP', 'XRT', '^GSPC', '^NDX'], 
        "settings": {
            "trade_direction": "Short", "entry_type": "Signal Close", "max_one_pos": False, "allow_same_day_reentry": False, "max_daily_entries": 3, "max_total_positions": 10,
            "perf_filters": [{'window': 5, 'logic': '>', 'thresh': 85.0, 'consecutive': 1}, {'window': 21, 'logic': '>', 'thresh': 85.0, 'consecutive': 3}], "perf_first_instance": False, "perf_lookback": 21,
            "use_sznl": False, "sznl_logic": "<", "sznl_thresh": 65.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_market_sznl": True, "market_sznl_logic": "<", "market_sznl_thresh": 50.0, "market_ticker": "^GSPC",
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": False, "52w_lookback": 21,
            "use_vol": True, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": ">", "vol_rank_thresh": 40.0,
            "trend_filter": "None", "min_price": 10.0, "min_vol": 100000, "min_age": 0.25, "max_age": 100.0, "entry_conf_bps": 0
        },
        "execution": { "risk_per_trade": 1000, "slippage_bps": 2, "stop_atr": 1.0, "tgt_atr": 1.0, "hold_days": 3 },
        "stats": { "grade": "A", "win_rate": "60.0%", "expectancy": "$191.71", "profit_factor": "1.57" }
    },
    # 4. LIQUID SEASONALS (SHORT TERM)
    {
        "id": "Sznl > 90, 5d <15 for 3d consec",
        "name": "Liquid Seasonals (short term)",
        "description": "Start: 2000-01-01. Universe: All CSV Tickers. Dir: Long.",
        "universe_tickers": ['AAPL', 'ABT', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEP', 'AIG', 'ALL', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'AON', 'APD', 'AVGO', 'AXP', 'BA', 'BAC', 'BAX', 'BDX', 'BK', 'BMY', 'C', 'CAG', 'CAT', 'CEF', 'CL', 'CMCSA', 'CMS', 'CNP', 'COP', 'COST', 'CPB', 'CRM', 'CSCO', 'CSX', 'CVS', 'CVX', 'D', 'DE', 'DIA', 'DIS', 'DOV', 'DTE', 'DUK', 'ECL', 'ED', 'EIX', 'EMR', 'EOG', 'ETR', 'EXC', 'F', 'FCX', 'FDX', 'FE', 'GD', 'GE', 'GILD', 'GIS', 'GLD', 'GLW', 'GOOG', 'GPC', 'GS', 'HAL', 'HD', 'HIG', 'HON', 'HPQ', 'HRL', 'HSY', 'HUM', 'IBB', 'IBM', 'IHI', 'INTC', 'IP', 'ITA', 'ITB', 'ITW', 'IWM', 'IYR', 'JNJ', 'JPM', 'K', 'KEY', 'KMB', 'KO', 'KR', 'KRE', 'LEG', 'LIN', 'LLY', 'LMT', 'LOW', 'LUV', 'MAS', 'MCD', 'MDT', 'MET', 'META', 'MMC', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'MU', 'NEE', 'NEM', 'NKE', 'NOC', 'NSC', 'NUE', 'NVDA', 'OIH', 'ORCL', 'OXY', 'PAYX', 'PCG', 'PEG', 'PEP', 'PFE', 'PG', 'PGR', 'PH', 'PNW', 'PPG', 'PPL', 'PSA', 'QCOM', 'QQQ', 'REGN', 'RF', 'RHI', 'ROK', 'ROST', 'RTX', 'SBUX', 'SCHW', 'SHW', 'SLB', 'SLV', 'SMH', 'SNA', 'SO', 'SPG', 'SPY', 'SRE', 'STT', 'SWK', 'SYK', 'SYY', 'T', 'TAP', 'TGT', 'TJX', 'TMO', 'TRV', 'TSN', 'TXN', 'UNG', 'UNH', 'UNP', 'USB', 'USO', 'UVXY', 'V', 'VFC', 'VLO', 'VMC', 'VNQ', 'VZ', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOM', 'XOP', 'XRT'], 
        "settings": {
            "trade_direction": "Long", "entry_type": "Signal Close", "max_one_pos": True, "max_daily_entries": 3, "max_total_positions": 10,
            "use_perf_rank": True, "perf_window": 5, "perf_logic": "<", "perf_thresh": 15.0, "perf_first_instance": False, "perf_lookback": 21, "perf_consecutive": 3,
            "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 90.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None", "min_price": 10.0, "min_vol": 100000, "min_age": 0.25, "max_age": 100.0
        },
        "execution": { "risk_per_trade": 500, "stop_atr": 2.0, "tgt_atr": 3.0, "hold_days": 5 },
        "stats": { "grade": "A", "win_rate": "61.1%", "expectancy": "$316.29", "profit_factor": "2.80" }
    },
    # 5. LIQUID SEASONALS (INTERMEDIATE)
    {
        "id": "Sznl > 85, 21dr < 15 3 consec",
        "name": "Liquid Seasonals (intermediate)",
        "description": "Start: 2000-01-01. Universe: All CSV Tickers. Dir: Long.",
        "universe_tickers": ['AAPL', 'ABT', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEP', 'AIG', 'ALL', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'AON', 'APD', 'AVGO', 'AXP', 'BA', 'BAC', 'BAX', 'BDX', 'BK', 'BMY', 'C', 'CAG', 'CAT', 'CEF', 'CL', 'CMCSA', 'CMS', 'CNP', 'COP', 'COST', 'CPB', 'CRM', 'CSCO', 'CSX', 'CVS', 'CVX', 'D', 'DE', 'DIA', 'DIS', 'DOV', 'DTE', 'DUK', 'ECL', 'ED', 'EIX', 'EMR', 'EOG', 'ETR', 'EXC', 'F', 'FCX', 'FDX', 'FE', 'GD', 'GE', 'GILD', 'GIS', 'GLD', 'GLW', 'GOOG', 'GPC', 'GS', 'HAL', 'HD', 'HIG', 'HON', 'HPQ', 'HRL', 'HSY', 'HUM', 'IBB', 'IBM', 'IHI', 'INTC', 'IP', 'ITA', 'ITB', 'ITW', 'IWM', 'IYR', 'JNJ', 'JPM', 'K', 'KEY', 'KMB', 'KO', 'KR', 'KRE', 'LEG', 'LIN', 'LLY', 'LMT', 'LOW', 'LUV', 'MAS', 'MCD', 'MDT', 'MET', 'META', 'MMC', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'MU', 'NEE', 'NEM', 'NKE', 'NOC', 'NSC', 'NUE', 'NVDA', 'OIH', 'ORCL', 'OXY', 'PAYX', 'PCG', 'PEG', 'PEP', 'PFE', 'PG', 'PGR', 'PH', 'PNW', 'PPG', 'PPL', 'PSA', 'QCOM', 'QQQ', 'REGN', 'RF', 'RHI', 'ROK', 'ROST', 'RTX', 'SBUX', 'SCHW', 'SHW', 'SLB', 'SLV', 'SMH', 'SNA', 'SO', 'SPG', 'SPY', 'SRE', 'STT', 'SWK', 'SYK', 'SYY', 'T', 'TAP', 'TGT', 'TJX', 'TMO', 'TRV', 'TSN', 'TXN', 'UNG', 'UNH', 'UNP', 'USB', 'USO', 'UVXY', 'V', 'VFC', 'VLO', 'VMC', 'VNQ', 'VZ', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOM', 'XOP', 'XRT'], 
        "settings": {
            "trade_direction": "Long", "entry_type": "Signal Close", "max_one_pos": True, "max_daily_entries": 3, "max_total_positions": 10,
            "use_perf_rank": True, "perf_window": 21, "perf_logic": "<", "perf_thresh": 15.0, "perf_first_instance": False, "perf_lookback": 21, "perf_consecutive": 3,
            "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 85.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None", "min_price": 10.0, "min_vol": 100000, "min_age": 0.25, "max_age": 100.0
        },
        "execution": { "risk_per_trade": 750, "stop_atr": 3.0, "tgt_atr": 8.0, "hold_days": 21 },
        "stats": { "grade": "A", "win_rate": "60.3%", "expectancy": "$265.02", "profit_factor": "1.97" }
    },
    # 6. UGLY MONDAY CLOSE
    {
        "id": "Lower 10% of Range 5d perf < 50%ile",
        "name": "Ugly Monday Close",
        "description": "Start: 2000-01-01. Universe: Indices. Dir: Long.",
        "universe_tickers": ['SPY', 'QQQ', 'IWM', 'DIA', 'SMH'], 
        "settings": {
            "trade_direction": "Long", "entry_type": "T+1 Open", "max_one_pos": True, "max_daily_entries": 5, "max_total_positions": 10,
            "use_dow_filter": True, "allowed_days": [0], "use_range_filter": True, "range_min": 0.0, "range_max": 10.0,
            "use_perf_rank": True, "perf_window": 5, "perf_logic": "<", "perf_thresh": 50.0, "perf_first_instance": False, "perf_lookback": 21, "perf_consecutive": 1,
            "use_sznl": False, "sznl_logic": ">", "sznl_thresh": 50.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None", "min_price": 10.0, "min_vol": 100000, "min_age": 0.25, "max_age": 100.0
        },
        "execution": { "risk_per_trade": 1000, "stop_atr": 2.0, "tgt_atr": 8.0, "hold_days": 4 },
        "stats": { "grade": "A", "win_rate": "68.8%", "expectancy": "$237.24", "profit_factor": "2.36" }
    },
    # 7. OVERSOLD LOW VOLUME
    {
        "id": "21dr < 15 3 consec, 5dr < 33, rel vol < 15, SPY > 200d",
        "name": "Oversold Low Volume",
        "description": "Start: 2000-01-01. Universe: All CSV Tickers. Dir: Long. Filter: SPY > 200 SMA.",
        "universe_tickers": ['AAPL', 'ABT', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEP', 'AIG', 'ALL', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'AON', 'APD', 'AVGO', 'AXP', 'BA', 'BAC', 'BAX', 'BDX', 'BK', 'BMY', 'C', 'CAG', 'CAT', 'CEF', 'CL', 'CMCSA', 'CMS', 'CNP', 'COP', 'COST', 'CPB', 'CRM', 'CSCO', 'CSX', 'CVS', 'CVX', 'D', 'DE', 'DIA', 'DIS', 'DOV', 'DTE', 'DUK', 'ECL', 'ED', 'EIX', 'EMR', 'EOG', 'ETR', 'EXC', 'F', 'FCX', 'FDX', 'FE', 'GD', 'GE', 'GILD', 'GIS', 'GLD', 'GLW', 'GOOG', 'GPC', 'GS', 'HAL', 'HD', 'HIG', 'HON', 'HPQ', 'HRL', 'HSY', 'HUM', 'IBB', 'IBM', 'IHI', 'INTC', 'IP', 'ITA', 'ITB', 'ITW', 'IWM', 'IYR', 'JNJ', 'JPM', 'K', 'KEY', 'KMB', 'KO', 'KR', 'KRE', 'LEG', 'LIN', 'LLY', 'LMT', 'LOW', 'LUV', 'MAS', 'MCD', 'MDT', 'MET', 'META', 'MMC', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'MU', 'NEE', 'NEM', 'NKE', 'NOC', 'NSC', 'NUE', 'NVDA', 'OIH', 'ORCL', 'OXY', 'PAYX', 'PCG', 'PEG', 'PEP', 'PFE', 'PG', 'PGR', 'PH', 'PNW', 'PPG', 'PPL', 'PSA', 'QCOM', 'QQQ', 'REGN', 'RF', 'RHI', 'ROK', 'ROST', 'RTX', 'SBUX', 'SCHW', 'SHW', 'SLB', 'SLV', 'SMH', 'SNA', 'SO', 'SPG', 'SPY', 'SRE', 'STT', 'SWK', 'SYK', 'SYY', 'T', 'TAP', 'TGT', 'TJX', 'TMO', 'TRV', 'TSN', 'TXN', 'UNG', 'UNH', 'UNP', 'USB', 'USO', 'UVXY', 'V', 'VFC', 'VLO', 'VMC', 'VNQ', 'VZ', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOM', 'XOP', 'XRT'], 
        "settings": {
            "trade_direction": "Long", "entry_type": "Signal Close", "max_one_pos": True, "max_daily_entries": 2, "max_total_positions": 5,
            "perf_filters": [{'window': 5, 'logic': '<', 'thresh': 33.0, 'consecutive': 1}, {'window': 21, 'logic': '<', 'thresh': 15.0, 'consecutive': 3}], "perf_first_instance": False, "perf_lookback": 21,
            "use_sznl": False, "sznl_logic": "<", "sznl_thresh": 15.0, "sznl_first_instance": True, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": True, "vol_rank_logic": "<", "vol_rank_thresh": 15.0,
            "trend_filter": "SPY > 200 SMA", "min_price": 10.0, "min_vol": 100000, "min_age": 0.25, "max_age": 100.0
        },
        "execution": { "risk_per_trade": 1000, "stop_atr": 2.0, "tgt_atr": 8.0, "hold_days": 21 },
        "stats": { "grade": "A", "win_rate": "66.3%", "expectancy": "$670.59", "profit_factor": "2.64" }
    },
    # 8. OVERBOUGHT VOLUME SPIKE
    {
        "id": "5dr > 90, 10dr > 90, 21d > 85 x5, vol >1.25x",
        "name": "Overbot Volume Spike",
        "description": "Start: 2000-01-01. Universe: Custom (Upload CSV). Dir: Short.",
        "universe_tickers": ['AA', 'AAL', 'AAOI', 'AAON', 'AAP', 'AAPL', 'AAUC', 'ABNB', 'ACGL', 'ACHC', 'ACI', 'ACIW', 'ACLS', 'ACM', 'ACMR', 'ACN', 'ADBE', 'ADEA', 'ADI', 'ADM', 'ADNT', 'ADP', 'ADSK', 'AEE', 'AEHR', 'AEIS', 'AEM', 'AEO', 'AEP', 'AER', 'AES', 'AEVA', 'AFL', 'AFRM', 'AG', 'AGCO', 'AGI', 'AHMA', 'AI', 'AIG', 'AIP', 'AJG', 'AKAM', 'ALAB', 'ALB', 'ALGM', 'ALH', 'ALHC', 'ALK', 'ALKT', 'ALL', 'ALLE', 'ALLY', 'ALSN', 'ALV', 'AM', 'AMAT', 'AMBA', 'AMD', 'AME', 'AMKR', 'AMN', 'AMP', 'AMPL', 'AMPX', 'AMRC', 'AMRZ', 'AMSC', 'AMTM', 'AMX', 'AMZN', 'ANET', 'ANF', 'ANGI', 'AON', 'AOS', 'APA', 'APD', 'APG', 'APH', 'APLD', 'APO', 'APP', 'APPN', 'APTV', 'AR', 'ARCC', 'ARES', 'ARHS', 'ARLO', 'ARM', 'ARMK', 'ARMN', 'AROC', 'ARQQ', 'ARW', 'ARX', 'AS', 'ASAN', 'ASB', 'ASC', 'ASGN', 'ASH', 'ASML', 'ASO', 'ASTH', 'ASTS', 'ASX', 'ATAT', 'ATEN', 'ATGE', 'ATI', 'ATKR', 'ATMU', 'ATO', 'ATR', 'ATRO', 'AU', 'AUB', 'AUGO', 'AVA', 'AVAV', 'AVGO', 'AVNT', 'AVPT', 'AVT', 'AVY', 'AWK', 'AXIA', 'AXON', 'AXP', 'AXS', 'AXTA', 'AXTI', 'AZTA', 'B', 'BA', 'BABA', 'BAC', 'BACQ', 'BAH', 'BALL', 'BAM', 'BANC', 'BBAR', 'BBT', 'BBVA', 'BBWI', 'BBY', 'BC', 'BCE', 'BCS', 'BE', 'BEKE', 'BEN', 'BEP', 'BEPC', 'BETR', 'BF.B', 'BFAM', 'BFH', 'BG', 'BHF', 'BHP', 'BIDU', 'BILI', 'BILL', 'BIPC', 'BIRK', 'BJ', 'BJRI', 'BK', 'BKD', 'BKH', 'BKKT', 'BKR', 'BKSY', 'BKU', 'BKV', 'BL', 'BLDR', 'BLK', 'BLSH', 'BMA', 'BMNR', 'BMO', 'BN', 'BNS', 'BOOT', 'BOX', 'BP', 'BPOP', 'BR', 'BRK.B', 'BRO', 'BROS', 'BRSL', 'BRZE', 'BSY', 'BTDR', 'BTI', 'BTSG', 'BTU', 'BUD', 'BURL', 'BV', 'BVN', 'BWA', 'BWIN', 'BWXT', 'BX', 'BXSL', 'BYD', 'BZ', 'C', 'CADE', 'CAE', 'CAG', 'CAH', 'CAKE', 'CAL', 'CALM', 'CALX', 'CAMT', 'CAR', 'CARG', 'CARR', 'CARS', 'CART', 'CAT', 'CAVA', 'CB', 'CBRE', 'CBRL', 'CBSH', 'CBZ', 'CC', 'CCCX', 'CCEP', 'CCHH', 'CCJ', 'CCK', 'CCL', 'CCOI', 'CDE', 'CDNS', 'CDW', 'CE', 'CEG', 'CELH', 'CENX', 'CEPT', 'CEVA', 'CF', 'CFG', 'CFLT', 'CG', 'CGAU', 'CGBD', 'CGNX', 'CHA', 'CHAC', 'CHD', 'CHDN', 'CHH', 'CHKP', 'CHOW', 'CHRD', 'CHRW', 'CHTR', 'CHWY', 'CHYM', 'CI', 'CIEN', 'CIFR', 'CINF', 'CIVI', 'CL', 'CLB', 'CLBT', 'CLF', 'CLH', 'CLMT', 'CLS', 'CLSK', 'CLX', 'CM', 'CMA', 'CMBT', 'CMC', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMPO', 'CMRE', 'CMS', 'CNC', 'CNI', 'CNK', 'CNM', 'CNNE', 'CNO', 'CNP', 'CNQ', 'CNR', 'CNX', 'CNXC', 'COCO', 'COF', 'COHR', 'COHU', 'COIN', 'COLB', 'COLM', 'COMM', 'COMP', 'CON', 'COP', 'COR', 'CORZ', 'COST', 'CP', 'CPAY', 'CPB', 'CPNG', 'CPRI', 'CPRT', 'CRBG', 'CRC', 'CRCL', 'CRDO', 'CRH', 'CRI', 'CRK', 'CRL', 'CRM', 'CRNC', 'CROX', 'CRS', 'CRUS', 'CRWD', 'CRWV', 'CSCO', 'CSGP', 'CSGS', 'CSIQ', 'CSL', 'CSTM', 'CSWC', 'CSX', 'CTAS', 'CTLP', 'CTRA', 'CTRI', 'CTSH', 'CTVA', 'CUK', 'CVBF', 'CVE', 'CVI', 'CVLT', 'CVNA', 'CVS', 'CVX', 'CWAN', 'CWEN', 'CWH', 'CWK', 'CWST', 'CX', 'CZR', 'D', 'DAL', 'DAN', 'DAR', 'DASH', 'DAVE', 'DB', 'DBX', 'DCI', 'DD', 'DDOG', 'DE', 'DECK', 'DELL', 'DEO', 'DG', 'DGNX', 'DGX', 'DHI', 'DHT', 'DIN', 'DINO', 'DIS', 'DJT', 'DK', 'DKNG', 'DKS', 'DLB', 'DLO', 'DLTR', 'DNOW', 'DOCN', 'DOCS', 'DOCU', 'DOLE', 'DOV', 'DOW', 'DOX', 'DPZ', 'DQ', 'DRD', 'DRI', 'DRS', 'DRVN', 'DSGX', 'DT', 'DTE', 'DTM', 'DUK', 'DUOL', 'DV', 'DVA', 'DVN', 'DXC', 'EAT', 'EBAY', 'EBC', 'ECG', 'ECL', 'ED', 'EDU', 'EEFT', 'EFX', 'EFXT', 'EGBN', 'EGO', 'EH', 'EHC', 'EIX', 'EL', 'ELF', 'ELV', 'EMBJ', 'EMN', 'EMR', 'ENB', 'ENPH', 'ENR', 'ENTG', 'EOG', 'EOSE', 'EPAM', 'EPC', 'EPD', 'EQH', 'EQNR', 'EQT', 'EQX', 'ERII', 'ERO', 'ES', 'ESAB', 'ESI', 'ESNT', 'ESTC', 'ET', 'ETN', 'ETOR', 'ETR', 'ETSY', 'EVER', 'EVRG', 'EVTC', 'EWBC', 'EXC', 'EXE', 'EXLS', 'EXPD', 'EXPE', 'EXPI', 'EXTR', 'EYE', 'EZPW', 'F', 'FA', 'FAF', 'FANG', 'FAST', 'FBIN', 'FBP', 'FCF', 'FCX', 'FDS', 'FDX', 'FE', 'FER', 'FERG', 'FFIN', 'FFIV', 'FHB', 'FHI', 'FHN', 'FIBK', 'FIG', 'FIGR', 'FIGS', 'FIS', 'FISV', 'FITB', 'FIVE', 'FIVN', 'FIX', 'FLEX', 'FLG', 'FLNC', 'FLO', 'FLR', 'FLS', 'FLUT', 'FLY', 'FLYE', 'FLYW', 'FMC', 'FN', 'FNB', 'FND', 'FNF', 'FNV', 'FORM', 'FOUR', 'FOX', 'FOXA', 'FOXF', 'FRGE', 'FRMI', 'FRO', 'FROG', 'FRPT', 'FRSH', 'FSK', 'FSLR', 'FSLY', 'FSS', 'FTAI', 'FTDR', 'FTI', 'FTNT', 'FTRE', 'FTS', 'FTV', 'FULT', 'FUN', 'FUTU', 'FVRR', 'FWONK', 'FWRD', 'FWRG', 'G', 'GAP', 'GBCI', 'GBDC', 'GCMG', 'GCT', 'GD', 'GDDY', 'GDEN', 'GDOT', 'GDS', 'GE', 'GEN', 'GENI', 'GES', 'GEV', 'GFI', 'GFL', 'GFS', 'GGAL', 'GGG', 'GH', 'GIL', 'GILT', 'GIS', 'GL', 'GLBE', 'GLIBK', 'GLNG', 'GLOB', 'GLW', 'GLXY', 'GM', 'GME', 'GNK', 'GNRC', 'GNTX', 'GO', 'GOLD', 'GOOG', 'GOOGL', 'GOOS', 'GPC', 'GPK', 'GPN', 'GPRE', 'GRMN', 'GRND', 'GRPN', 'GRRR', 'GS', 'GSAT', 'GTES', 'GTLB', 'GTM', 'GTX', 'GVA', 'GWRE', 'GXO', 'H', 'HAL', 'HAS', 'HASI', 'HAYW', 'HBAN', 'HBM', 'HCA', 'HCC', 'HCSG', 'HD', 'HDB', 'HE', 'HELE', 'HESM', 'HGV', 'HI', 'HIG', 'HIMS', 'HL', 'HLF', 'HLNE', 'HLT', 'HMC', 'HMY', 'HNGE', 'HNI', 'HNRG', 'HOG', 'HOMB', 'HON', 'HOOD', 'HOPE', 'HOUS', 'HP', 'HPE', 'HPQ', 'HQY', 'HRB', 'HRL', 'HSAI', 'HSBC', 'HSIC', 'HSY', 'HTFL', 'HTGC', 'HTHT', 'HUBB', 'HUBG', 'HUBS', 'HUM', 'HUN', 'HUT', 'HWC', 'HWM', 'HXL', 'HYMC', 'IAC', 'IAG', 'IBKR', 'IBM', 'IBN', 'ICE', 'ICHR', 'ICLR', 'IDR', 'IE', 'IEX', 'IFF', 'IMAX', 'IMO', 'INFY', 'ING', 'INGR', 'INOD', 'INTA', 'INTC', 'INTU', 'IONQ', 'IOT', 'IP', 'IR', 'IRDM', 'IREN', 'IT', 'ITRI', 'ITT', 'ITW', 'IVZ', 'J', 'JACK', 'JAMF', 'JBHT', 'JBL', 'JBS', 'JBTM', 'JCI', 'JD', 'JEF', 'JHG', 'JHX', 'JKHY', 'JKS', 'JMIA', 'JOBY', 'JPM', 'JXN', 'KAR', 'KBH', 'KBR', 'KC', 'KD', 'KDP', 'KEX', 'KEY', 'KEYS', 'KGC', 'KGS', 'KHC', 'KKR', 'KLAC', 'KLAR', 'KLIC', 'KMB', 'KMI', 'KMPR', 'KMT', 'KMX', 'KN', 'KNF', 'KNTK', 'KNX', 'KO', 'KR', 'KRMN', 'KRNT', 'KRP', 'KSPI', 'KSS', 'KT', 'KTB', 'KTOS', 'KVUE', 'KVYO', 'KYIV', 'L', 'LASR', 'LAUR', 'LAZ', 'LBRDK', 'LBRT', 'LBTYA', 'LBTYK', 'LC', 'LCID', 'LDOS', 'LEA', 'LEG', 'LEN', 'LEU', 'LEVI', 'LGN', 'LH', 'LHX', 'LI', 'LIF', 'LII', 'LIN', 'LITE', 'LKQ', 'LMND', 'LMT', 'LNC', 'LNG', 'LNT', 'LOAR', 'LOGI', 'LOMA', 'LOVE', 'LOW', 'LPLA', 'LPX', 'LRCX', 'LRN', 'LSCC', 'LSPD', 'LTBR', 'LTH', 'LTM', 'LULU', 'LUV', 'LVS', 'LW', 'LYB', 'LYFT', 'LYV', 'LZB', 'M', 'MA', 'MAGN', 'MAN', 'MANH', 'MAR', 'MARA', 'MAS', 'MAT', 'MAX', 'MBC', 'MBLY', 'MC', 'MCD', 'MCHP', 'MCK', 'MCO', 'MD', 'MDB', 'MDLZ', 'MDU', 'MELI', 'MEOH', 'MET', 'META', 'METC', 'MFC', 'MGA', 'MGM', 'MGY', 'MH', 'MHK', 'MIDD', 'MIR', 'MKC', 'MKSI', 'MKTX', 'MLI', 'MLKN', 'MMC', 'MMM', 'MMS', 'MMYT', 'MNDY', 'MNRO', 'MNSO', 'MNST', 'MNTN', 'MO', 'MOD', 'MODG', 'MOH', 'MOS', 'MP', 'MPC', 'MPLX', 'MPWR', 'MRCY', 'MRTN', 'MRVL', 'MRX', 'MS', 'MSCI', 'MSFT', 'MSI', 'MSM', 'MSTR', 'MT', 'MTB', 'MTCH', 'MTDR', 'MTG', 'MTH', 'MTN', 'MTSI', 'MTZ', 'MU', 'MUFG', 'MUR', 'MUX', 'MWA', 'MXL', 'NAVI', 'NBIS', 'NCLH', 'NCNO', 'NDAQ', 'NE', 'NEE', 'NEGG', 'NEM', 'NEO', 'NESR', 'NET', 'NFG', 'NFLX', 'NGG', 'NI', 'NICE', 'NIQ', 'NJR', 'NKE', 'NMRK', 'NN', 'NNE', 'NOC', 'NOG', 'NOMD', 'NOV', 'NOVT', 'NOW', 'NPKI', 'NRDS', 'NRG', 'NSC', 'NSIT', 'NSP', 'NTAP', 'NTCT', 'NTES', 'NTNX', 'NTR', 'NTRS', 'NTSK', 'NU', 'NUE', 'NVDA', 'NVRI', 'NVT', 'NWBI', 'NWG', 'NWS', 'NWSA', 'NX', 'NXPI', 'NXT', 'NYT', 'OBDC', 'OC', 'OCSL', 'ODD', 'ODFL', 'OGE', 'OI', 'OII', 'OKE', 'OKLO', 'OKTA', 'OLED', 'OLLI', 'OLN', 'OMC', 'OMDA', 'OMF', 'ON', 'ONB', 'ONON', 'ONTO', 'OPCH', 'OPRA', 'OR', 'ORA', 'ORCL', 'ORI', 'ORLA', 'ORLY', 'OS', 'OSCR', 'OSK', 'OSPN', 'OSW', 'OTEX', 'OTF', 'OTIS', 'OUST', 'OVV', 'OWL', 'OXY', 'OZK', 'PAA', 'PAAS', 'PACS', 'PAGP', 'PAGS', 'PANW', 'PAR', 'PARR', 'PATH', 'PAY', 'PAYC', 'PAYX', 'PB', 'PBA', 'PBF', 'PBR', 'PBR.A', 'PCAR', 'PCG', 'PCOR', 'PCTY', 'PD', 'PDD', 'PEG', 'PEGA', 'PENG', 'PENN', 'PEP', 'PFG', 'PFGC', 'PFS', 'PG', 'PGNY', 'PGR', 'PGY', 'PH', 'PHM', 'PHR', 'PI', 'PII', 'PINS', 'PKG', 'PL', 'PLAB', 'PLAY', 'PLNT', 'PLTR', 'PM', 'PNC', 'PNFP', 'PNR', 'PNW', 'PONY', 'POOL', 'POR', 'POST', 'POWI', 'PPC', 'PPG', 'PPL', 'PPTA', 'PR', 'PRDO', 'PRGS', 'PRIM', 'PRKS', 'PRM', 'PRMB', 'PRU', 'PRVA', 'PSIX', 'PSKY', 'PSN', 'PSNL', 'PSO', 'PSTG', 'PSX', 'PTC', 'PTRN', 'PUK', 'PVH', 'PWP', 'PWR', 'PYPL', 'PZZA', 'QBTS', 'QCOM', 'QFIN', 'QNST', 'QRVO', 'QS', 'QSR', 'QTWO', 'QUBT', 'QXO', 'RACE', 'RAL', 'RAMP', 'RBA', 'RBLX', 'RBRK', 'RCI', 'RCL', 'RDDT', 'RDN', 'RDNT', 'REAL', 'RELX', 'RELY', 'REVG', 'REYN', 'REZI', 'RF', 'RGLD', 'RGTI', 'RH', 'RHI', 'RIO', 'RIOT', 'RIVN', 'RJF', 'RKLB', 'RKT', 'RL', 'RLI', 'RMBS', 'RNG', 'RNST', 'ROK', 'ROKU', 'ROL', 'ROOT', 'ROP', 'ROST', 'RPD', 'RPM', 'RPRX', 'RRC', 'RRR', 'RRX', 'RSG', 'RSI', 'RTO', 'RTX', 'RUN', 'RUSHA', 'RVLV', 'RXO', 'RXST', 'RY', 'RYAAY', 'RYAN', 'S', 'SA', 'SAIA', 'SAIC', 'SAIL', 'SAN', 'SANM', 'SAP', 'SARO', 'SATS', 'SBCF', 'SBGI', 'SBH', 'SBLK', 'SBS', 'SBSW', 'SBUX', 'SCCO', 'SCHW', 'SCI', 'SCS', 'SDGR', 'SDRL', 'SE', 'SEDG', 'SEE', 'SEI', 'SEIC', 'SEM', 'SEMR', 'SEZL', 'SF', 'SFD', 'SFM', 'SFNC', 'SGHC', 'SGI', 'SGML', 'SGRY', 'SHAK', 'SHEL', 'SHOO', 'SHOP', 'SHW', 'SIG', 'SIGI', 'SIRI', 'SITE', 'SJM', 'SKE', 'SKM', 'SKY', 'SKYT', 'SLB', 'SLDE', 'SLGN', 'SLM', 'SM', 'SMCI', 'SMFG', 'SMG', 'SMPL', 'SMR', 'SMTC', 'SMX', 'SN', 'SNCY', 'SNDK', 'SNDR', 'SNOW', 'SNPS', 'SNV', 'SNX', 'SO', 'SOBO', 'SOFI', 'SOLS', 'SOLV', 'SON', 'SONO', 'SONY', 'SOUN', 'SPGI', 'SPHR', 'SPNT', 'SPOT', 'SPR', 'SPSC', 'SPT', 'SQM', 'SRAD', 'SRE', 'SSB', 'SSNC', 'SSRM', 'ST', 'STEP', 'STLA', 'STLD', 'STM', 'STNE', 'STNG', 'STRL', 'STT', 'STUB', 'STX', 'STZ', 'SU', 'SUN', 'SUPV', 'SW', 'SWK', 'SWKS', 'SYF', 'SYM', 'SYY', 'T', 'TAC', 'TAL', 'TALO', 'TAP', 'TBBB', 'TBBK', 'TCOM', 'TD', 'TDC', 'TDS', 'TDW', 'TEAM', 'TECK', 'TECX', 'TEL', 'TEM', 'TENB', 'TER', 'TEX', 'TFC', 'TFPM', 'TGNA', 'TGT', 'THC', 'THO', 'THS', 'TIGO', 'TJX', 'TKO', 'TKR', 'TLK', 'TLN', 'TME', 'TMHC', 'TMUS', 'TNL', 'TOL', 'TOST', 'TPC', 'TPG', 'TPH', 'TPR', 'TREX', 'TRGP', 'TRI', 'TRIN', 'TRIP', 'TRMB', 'TRMD', 'TRN', 'TROW', 'TRP', 'TRS', 'TRU', 'TRV', 'TS', 'TSCO', 'TSEM', 'TSLA', 'TSM', 'TSN', 'TT', 'TTAN', 'TTC', 'TTD', 'TTE', 'TTEK', 'TTMI', 'TTWO', 'TU', 'TW', 'TWLO', 'TXN', 'TXRH', 'TXT', 'U', 'UAL', 'UBER', 'UBS', 'UBSI', 'UCB', 'UCTT', 'UEC', 'UGI', 'UHAL.B', 'UHS', 'UL', 'ULS', 'ULTA', 'UNFI', 'UNH', 'UNM', 'UNP', 'UPBD', 'UPS', 'UPST', 'UPWK', 'URBN', 'URI', 'USAR', 'USB', 'USFD', 'UTI', 'UUUU', 'V', 'VAC', 'VAL', 'VALE', 'VECO', 'VEEV', 'VERX', 'VFC', 'VIAV', 'VICR', 'VIK', 'VIPS', 'VIRT', 'VIST', 'VITL', 'VIV', 'VLO', 'VLTO', 'VLY', 'VMC', 'VNOM', 'VNT', 'VOD', 'VOYA', 'VOYG', 'VRNS', 'VRRM', 'VRSK', 'VRSN', 'VRT', 'VSAT', 'VSCO', 'VSH', 'VST', 'VTLE', 'VVV', 'VZ', 'W', 'WAB', 'WAFD', 'WAL', 'WAY', 'WB', 'WBD', 'WBS', 'WBTN', 'WCC', 'WCN', 'WDAY', 'WDC', 'WDS', 'WEC', 'WERN', 'WES', 'WFC', 'WFRD', 'WGO', 'WGS', 'WH', 'WHD', 'WHR', 'WING', 'WIX', 'WK', 'WKC', 'WLK', 'WM', 'WMB', 'WMG', 'WMS', 'WMT', 'WOLF', 'WPM', 'WPP', 'WRB', 'WRBY', 'WSC', 'WSM', 'WT', 'WTRG', 'WTTR', 'WTW', 'WULF', 'WWD', 'WWW', 'WYFI', 'WYNN', 'XEL', 'XMTR', 'XOM', 'XP', 'XPEV', 'XPO', 'XPRO', 'XYL', 'XYZ', 'YELP', 'YETI', 'YMM', 'YOU', 'YPF', 'YUM', 'YUMC', 'Z', 'ZBRA', 'ZD', 'ZETA', 'ZG', 'ZGN', 'ZIM', 'ZION', 'ZM', 'ZS', 'ZTO', 'ZWS'], 
        "settings": {
            "trade_direction": "Short", "entry_type": "T+1 Close", "max_one_pos": False, "allow_same_day_reentry": False, "max_daily_entries": 2, "max_total_positions": 4,
            "perf_filters": [{'window': 5, 'logic': '>', 'thresh': 90.0, 'consecutive': 1}, {'window': 10, 'logic': '>', 'thresh': 90.0, 'consecutive': 1}, {'window': 21, 'logic': '>', 'thresh': 85.0, 'consecutive': 5}], "perf_first_instance": False, "perf_lookback": 21,
            "use_range_filter": True, "range_min": 90.0, "range_max": 100.0,
            "use_dow_filter": True, "allowed_days": [0, 4],
            "use_sznl": False, "sznl_logic": "<", "sznl_thresh": 15.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_market_sznl": False, "market_sznl_logic": "<", "market_sznl_thresh": 65.0, "market_ticker": "^GSPC",
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.25,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 30.0,
            "trend_filter": "None", "min_price": 10.0, "min_vol": 3000000, "min_age": 0.5, "max_age": 100.0, "entry_conf_bps": 0,
            "use_dist_filter": False, "dist_ma_type": "SMA 50", "dist_logic": "Between", "dist_min": 7.0, "dist_max": 30.0,
            "use_gap_filter": True, "gap_lookback": 5, "gap_logic": ">", "gap_thresh": 0
        },
        "execution": { "risk_per_trade": 1000, "slippage_bps": 0, "stop_atr": 1.0, "tgt_atr": 1.0, "hold_days": 3 },
        "stats": { "grade": "A", "win_rate": "61.6%", "expectancy": "$227.88", "profit_factor": "1.76" }
    },
]

# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
CSV_PATH = "sznl_ranks.csv" 

@st.cache_resource 
def load_seasonal_map():
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        return {}
    if df.empty: return {}
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])
    df["Date"] = df["Date"].dt.normalize()
    output_map = {}
    for ticker, group in df.groupby("ticker"):
        series = group.set_index("Date")["seasonal_rank"].sort_index()
        output_map[ticker] = series
    return output_map

def get_sznl_val_series(ticker, dates, sznl_map):
    ticker = ticker.upper()
    t_series = sznl_map.get(ticker)
    if t_series is None and ticker == "^GSPC":
        t_series = sznl_map.get("SPY")
    if t_series is None:
        return pd.Series(50.0, index=dates)
    return dates.map(t_series).fillna(50.0)

def save_signals_to_gsheet(new_dataframe, sheet_name='Trade_Signals_Log'):
    if new_dataframe.empty: return
    df_new = new_dataframe.copy()
    cols_to_round = ['Entry', 'Stop', 'Target', 'ATR']
    existing_cols = [c for c in cols_to_round if c in df_new.columns]
    df_new[existing_cols] = df_new[existing_cols].astype(float).round(2)
    df_new['Date'] = df_new['Date'].astype(str) 
    df_new["Scan_Timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cols = ['Scan_Timestamp'] + [c for c in df_new.columns if c != 'Scan_Timestamp']
    df_new = df_new[cols]
    try:
        if "gcp_service_account" in st.secrets:
            creds_dict = st.secrets["gcp_service_account"]
            gc = gspread.service_account_from_dict(creds_dict)
        else:
            gc = gspread.service_account(filename='credentials.json')
        sh = gc.open(sheet_name)
        worksheet = sh.sheet1 
        existing_data = worksheet.get_all_values()
        if existing_data:
            headers = existing_data[0]
            df_existing = pd.DataFrame(existing_data[1:], columns=headers)
        else:
            df_existing = pd.DataFrame()
        if not df_existing.empty:
            df_existing = df_existing.reindex(columns=df_new.columns)
            combined = pd.concat([df_existing, df_new])
        else:
            combined = df_new
        combined = combined.drop_duplicates(subset=['Ticker', 'Date', 'Strategy_ID'], keep='last')
        worksheet.clear()
        data_to_write = [combined.columns.tolist()] + combined.astype(str).values.tolist()
        worksheet.update(values=data_to_write)
        st.toast(f"✅ Synced! Sheet now has {len(combined)} total rows.")
    except Exception as e:
        st.error(f"❌ Google Sheet Error: {e}")

# -----------------------------------------------------------------------------
# ENGINE
# -----------------------------------------------------------------------------

def calculate_indicators(df, sznl_map, ticker, market_series=None):
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean() 
    df['SMA200'] = df['Close'].rolling(200).mean()
    
    is_open_gap = (df['Low'] > df['High'].shift(1)).astype(int)
    df['GapCount_21'] = is_open_gap.rolling(21).sum() 
    df['GapCount_10'] = is_open_gap.rolling(10).sum()
    df['GapCount_5'] = is_open_gap.rolling(5).sum() 

    denom = (df['High'] - df['Low'])
    df['RangePct'] = np.where(denom == 0, 0.5, (df['Close'] - df['Low']) / denom)

    for window in [5, 10, 21]:
        df[f'ret_{window}d'] = df['Close'].pct_change(window)
        df[f'rank_ret_{window}d'] = df[f'ret_{window}d'].expanding(min_periods=50).rank(pct=True) * 100.0
        
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
    df['vol_ratio'] = df['Volume'] / vol_ma
    df['vol_ma'] = vol_ma
    
    vol_ma_10 = df['Volume'].rolling(10).mean()
    df['vol_ratio_10d'] = vol_ma_10 / vol_ma
    df['vol_ratio_10d_rank'] = df['vol_ratio_10d'].expanding(min_periods=50).rank(pct=True) * 100.0
    df['Sznl'] = get_sznl_val_series(ticker, df.index, sznl_map)
    df['Mkt_Sznl_Ref'] = get_sznl_val_series("^GSPC", df.index, sznl_map)
    
    if not df.empty:
        start_ts = df.index[0]
        df['age_years'] = (df.index - start_ts).days / 365.25
    else:
        df['age_years'] = 0.0

    if market_series is not None:
        df['Market_Above_SMA200'] = market_series.reindex(df.index, method='ffill').fillna(False)
        
    return df

def get_historical_mask(df, params, sznl_map):
    """ Vectorized version of check_signal for historical backtesting """
    mask = pd.Series(True, index=df.index)

    if params.get('use_dow_filter', False):
        allowed = params.get('allowed_days', [])
        mask &= df.index.dayofweek.isin(allowed)

    mask &= (df['Close'] >= params.get('min_price', 0))
    mask &= (df['vol_ma'] >= params.get('min_vol', 0))
    mask &= (df['age_years'] >= params.get('min_age', 0))
    mask &= (df['age_years'] <= params.get('max_age', 100))

    trend_opt = params.get('trend_filter', 'None')
    if trend_opt == "Price > 200 SMA":
        mask &= (df['Close'] > df['SMA200'])
    elif trend_opt == "Price > Rising 200 SMA":
        mask &= (df['Close'] > df['SMA200']) & (df['SMA200'] > df['SMA200'].shift(1))
    elif "Market" in trend_opt or "SPY" in trend_opt:
        if 'Market_Above_SMA200' in df.columns:
            is_above = df['Market_Above_SMA200']
            if ">" in trend_opt: mask &= is_above
            elif "<" in trend_opt: mask &= ~is_above

    if params.get('use_range_filter', False):
        rn_val = df['RangePct'] * 100
        mask &= (rn_val >= params.get('range_min', 0)) & (rn_val <= params.get('range_max', 100))

    if 'perf_filters' in params:
        for pf in params['perf_filters']:
            col = f"rank_ret_{pf['window']}d"
            consec = pf.get('consecutive', 1)
            if pf['logic'] == '<': cond_f = df[col] < pf['thresh']
            else: cond_f = df[col] > pf['thresh']
            if consec > 1: cond_f = (cond_f.rolling(consec).sum() == consec)
            mask &= cond_f
        if params.get('perf_first_instance', False):
            lookback = params.get('perf_lookback', 21)
            prev_inst = mask.shift(1).rolling(lookback).sum()
            mask &= (prev_inst == 0)

    elif params.get('use_perf_rank', False):
        col = f"rank_ret_{params['perf_window']}d"
        if params['perf_logic'] == '<': raw = df[col] < params['perf_thresh']
        else: raw = df[col] > params['perf_thresh']
        consec = params.get('perf_consecutive', 1)
        if consec > 1: persist = (raw.rolling(consec).sum() == consec)
        else: persist = raw
        mask &= persist
        if params.get('perf_first_instance', False):
            lookback = params.get('perf_lookback', 21)
            prev_inst = mask.shift(1).rolling(lookback).sum()
            mask &= (prev_inst == 0)

    if params.get('use_gap_filter', False):
        lookback = params.get('gap_lookback', 21)
        col_name = f'GapCount_{lookback}' if f'GapCount_{lookback}' in df.columns else 'GapCount_21'
        g_val = df.get(col_name, 0)
        g_thresh = params.get('gap_thresh', 0)
        g_logic = params.get('gap_logic', '>')
        if g_logic == ">": mask &= (g_val > g_thresh)
        elif g_logic == "<": mask &= (g_val < g_thresh)
        elif g_logic == "=": mask &= (g_val == g_thresh)

    if params.get('use_dist_filter', False):
        ma_type = params.get('dist_ma_type', 'SMA 200')
        ma_col = ma_type.replace(" ", "") 
        if ma_col in df.columns:
            atr = df['ATR'].replace(0, np.nan)
            dist_units = (df['Close'] - df[ma_col]) / atr
            d_logic = params.get('dist_logic', 'Between')
            d_min = params.get('dist_min', 0)
            d_max = params.get('dist_max', 0)
            if d_logic == "Greater Than (>)": mask &= (dist_units > d_min)
            elif d_logic == "Less Than (<)": mask &= (dist_units < d_max)
            elif d_logic == "Between": mask &= (dist_units >= d_min) & (dist_units <= d_max)

    if params['use_sznl']:
        if params['sznl_logic'] == '<': raw_sznl = df['Sznl'] < params['sznl_thresh']
        else: raw_sznl = df['Sznl'] > params['sznl_thresh']
        sznl_cond = raw_sznl
        if params.get('sznl_first_instance', False):
            lookback = params.get('sznl_lookback', 21)
            prev = sznl_cond.shift(1).rolling(lookback).sum()
            sznl_cond &= (prev == 0)
        mask &= sznl_cond

    if params.get('use_market_sznl', False):
        mkt_ranks = df['Mkt_Sznl_Ref']
        if params['market_sznl_logic'] == '<': mask &= (mkt_ranks < params['market_sznl_thresh'])
        else: mask &= (mkt_ranks > params['market_sznl_thresh'])

    if params['use_52w']:
        if params['52w_type'] == 'New 52w High': cond_52 = df['is_52w_high']
        else: cond_52 = df['is_52w_low']
        if params.get('52w_first_instance', True):
            lookback = params.get('52w_lookback', 21)
            prev = cond_52.shift(1).rolling(lookback).sum()
            cond_52 &= (prev == 0)
        mask &= cond_52

    if params['use_vol']:
        mask &= (df['vol_ratio'] > params['vol_thresh'])
    if params.get('use_vol_rank'):
        val = df['vol_ratio_10d_rank']
        if params['vol_rank_logic'] == '<': mask &= (val < params['vol_rank_thresh'])
        else: mask &= (val > params['vol_rank_thresh'])

    return mask.fillna(False)

def check_signal(df, params, sznl_map):
    """ Single row version for Live Scan (legacy support for execution logic) """
    mask = get_historical_mask(df, params, sznl_map)
    return mask.iloc[-1]

def calculate_trade_result(df, signal_date, action, entry_price, stop_price, target_price, hold_days, risk_per_trade):
    """ Simulates the trade outcome by looking forward in the dataframe """
    
    # Slice dataframe from day after signal to hold days
    start_idx = df.index.searchsorted(signal_date)
    if start_idx >= len(df) - 1:
        return 0, signal_date # Not enough data yet
    
    # Look at future window
    window = df.iloc[start_idx+1 : start_idx+1+hold_days].copy()
    if window.empty: return 0, signal_date

    pnl = 0
    exit_date = window.index[-1]

    if action == "BUY":
        # Check if stop or target hit
        # We assume if Low < Stop, we stopped out. If High > Target, we hit target.
        # Conservative assumption: If both happen on same day, Stop hits first.
        
        hit_stop = window[window['Low'] <= stop_price]
        hit_tgt = window[window['High'] >= target_price]
        
        first_stop = hit_stop.index[0] if not hit_stop.empty else None
        first_tgt = hit_tgt.index[0] if not hit_tgt.empty else None
        
        if first_stop and first_tgt:
            if first_stop <= first_tgt:
                # Stopped out
                pnl = -risk_per_trade
                exit_date = first_stop
            else:
                # Target hit
                # Calculate R multiple. If Target is 2R, we make 2 * Risk
                r_mult = (target_price - entry_price) / (entry_price - stop_price)
                pnl = risk_per_trade * r_mult
                exit_date = first_tgt
        elif first_stop:
            pnl = -risk_per_trade
            exit_date = first_stop
        elif first_tgt:
            r_mult = (target_price - entry_price) / (entry_price - stop_price)
            pnl = risk_per_trade * r_mult
            exit_date = first_tgt
        else:
            # Time Exit
            exit_price = window.iloc[-1]['Close']
            dist_risk = entry_price - stop_price
            if dist_risk == 0: dist_risk = 0.01
            r_realized = (exit_price - entry_price) / dist_risk
            pnl = risk_per_trade * r_realized
            
    elif action == "SELL SHORT":
        hit_stop = window[window['High'] >= stop_price]
        hit_tgt = window[window['Low'] <= target_price]
        
        first_stop = hit_stop.index[0] if not hit_stop.empty else None
        first_tgt = hit_tgt.index[0] if not hit_tgt.empty else None
        
        if first_stop and first_tgt:
            if first_stop <= first_tgt:
                pnl = -risk_per_trade
                exit_date = first_stop
            else:
                r_mult = (stop_price - entry_price) / (target_price - stop_price) # Risk dist is absolute
                # Actually simpler: (Entry - Target) / (Stop - Entry)
                r_mult = (entry_price - target_price) / (stop_price - entry_price)
                pnl = risk_per_trade * r_mult
                exit_date = first_tgt
        elif first_stop:
            pnl = -risk_per_trade
            exit_date = first_stop
        elif first_tgt:
            r_mult = (entry_price - target_price) / (stop_price - entry_price)
            pnl = risk_per_trade * r_mult
            exit_date = first_tgt
        else:
            # Time Exit
            exit_price = window.iloc[-1]['Close']
            dist_risk = stop_price - entry_price
            if dist_risk == 0: dist_risk = 0.01
            r_realized = (entry_price - exit_price) / dist_risk
            pnl = risk_per_trade * r_realized

    return pnl, exit_date

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="Strategy Screener & Backtest")
    st.title("⚡ Strategy Screener & Lab")
    st.markdown("---")
    
    sznl_map = load_seasonal_map()
    
    # --- UI CONTROLS ---
    col1, col2 = st.columns([1, 1])
    with col1:
        run_live = st.button("Run Live Scan (Today)", type="primary", use_container_width=True)
    with col2:
        run_hist = st.button("Run 3-Month Historical Backtest", type="secondary", use_container_width=True)

    if run_live or run_hist:
        
        is_backtest = True if run_hist else False
        mode_text = "Historical Backtest (90 Days)" if is_backtest else "Live Scan"
        st.info(f"Running {mode_text} on {len(STRATEGY_BOOK)} strategies...")
        
        # 1. Consolidate Tickers
        all_tickers = set()
        market_tickers = set()
        market_tickers.add("SPY") 
        
        for strat in STRATEGY_BOOK:
            all_tickers.update(strat['universe_tickers'])
            s = strat['settings']
            if s.get('use_market_sznl'): market_tickers.add(s.get('market_ticker', '^GSPC'))
            if "Market" in s.get('trend_filter', ''): market_tickers.add(s.get('market_ticker', 'SPY'))
            if "SPY" in s.get('trend_filter', ''): market_tickers.add("SPY")
        
        final_download_list = list(all_tickers.union(market_tickers))
        final_download_list = [t.replace('.', '-') for t in final_download_list]
        
        # 2. Download Data
        start_date = datetime.date.today() - datetime.timedelta(days=400)
        try:
            raw_data = yf.download(final_download_list, start=start_date, group_by='ticker', progress=False, threads=True)
        except Exception as e:
            st.error(f"Data download failed: {e}")
            return

        # Prepare Market Data
        market_series = None
        try:
            if len(final_download_list) > 1:
                if "SPY" in raw_data.columns.levels[0]: mkt_df = raw_data["SPY"].copy()
                else: mkt_df = raw_data.iloc[:, :5].copy()
            else: mkt_df = raw_data.copy()
            
            if isinstance(mkt_df.columns, pd.MultiIndex):
                mkt_df.columns = [c if isinstance(c, str) else c[0] for c in mkt_df.columns]

            mkt_df['SMA200'] = mkt_df['Close'].rolling(200).mean()
            market_series = mkt_df['Close'] > mkt_df['SMA200']
        except: pass

        all_signals = []
        progress_bar = st.progress(0)
        
        # 3. Iterate Strategies
        for i, strat in enumerate(STRATEGY_BOOK):
            progress_bar.progress((i + 1) / len(STRATEGY_BOOK))
            
            for ticker in strat['universe_tickers']:
                try:
                    if len(final_download_list) > 1:
                        if ticker not in raw_data.columns.levels[0]: continue
                        df = raw_data[ticker].copy()
                    else: df = raw_data.copy()

                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]

                    df = df.dropna(subset=['Close'])
                    if len(df) < 250: continue
                    
                    df = calculate_indicators(df, sznl_map, ticker, market_series)

                    # --- LOGIC ---
                    if is_backtest:
                        # HISTORICAL MODE
                        mask = get_historical_mask(df, strat['settings'], sznl_map)
                        cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=90)
                        mask = mask[mask.index >= cutoff_date]
                        true_dates = mask[mask].index
                        
                        for d in true_dates:
                            row = df.loc[d]
                            atr = row['ATR']
                            risk = strat['execution']['risk_per_trade']
                            entry = row['Close']
                            direction = strat['settings'].get('trade_direction', 'Long')
                            
                            if direction == 'Long':
                                stop_price = entry - (atr * strat['execution']['stop_atr'])
                                tgt_price = entry + (atr * strat['execution']['tgt_atr'])
                                action = "BUY"
                            else:
                                stop_price = entry + (atr * strat['execution']['stop_atr'])
                                tgt_price = entry - (atr * strat['execution']['tgt_atr'])
                                action = "SELL SHORT"
                            
                            # CALCULATE PNL
                            pnl, exit_date = calculate_trade_result(
                                df, d, action, entry, stop_price, tgt_price, 
                                strat['execution']['hold_days'], risk
                            )

                            all_signals.append({
                                "Date": d.date(),
                                "Exit Date": exit_date.date(),
                                "Strategy": strat['name'],
                                "Ticker": ticker,
                                "Action": action,
                                "Price": entry,
                                "PnL": pnl
                            })
                            
                    else:
                        # LIVE MODE
                        if check_signal(df, strat['settings'], sznl_map):
                            last_row = df.iloc[-1]
                            
                            # Entry Conf Check
                            entry_conf_bps = strat['settings'].get('entry_conf_bps', 0)
                            if strat['settings'].get('entry_type') == 'Signal Close' and entry_conf_bps > 0:
                                threshold = last_row['Open'] * (1 + entry_conf_bps/10000.0)
                                if last_row['High'] < threshold: continue

                            atr = last_row['ATR']
                            risk = strat['execution']['risk_per_trade']
                            entry = last_row['Close']
                            direction = strat['settings'].get('trade_direction', 'Long')
                            
                            if direction == 'Long':
                                stop_price = entry - (atr * strat['execution']['stop_atr'])
                                tgt_price = entry + (atr * strat['execution']['tgt_atr'])
                                dist = entry - stop_price
                                action = "BUY"
                            else:
                                stop_price = entry + (atr * strat['execution']['stop_atr'])
                                tgt_price = entry - (atr * strat['execution']['tgt_atr'])
                                dist = stop_price - entry
                                action = "SELL SHORT"
                            
                            shares = int(risk / dist) if dist > 0 else 0
                            exit_date = (last_row.name + BusinessDay(strat['execution']['hold_days'])).date()

                            all_signals.append({
                                "Strategy_ID": strat['id'],
                                "Ticker": ticker,
                                "Date": last_row.name.date(),
                                "Action": action,
                                "Shares": shares,
                                "Entry": entry,
                                "Stop": stop_price,
                                "Target": tgt_price,
                                "Time Exit": exit_date,
                                "ATR": atr
                            })

                except Exception as e:
                    continue
        
        progress_bar.empty()
        
        # --- DISPLAY RESULTS ---
        if all_signals:
            sig_df = pd.DataFrame(all_signals)
            
            if is_backtest:
                # 1. Equity Curve
                st.subheader("📈 Cumulative PnL (Rolling 90 Days)")
                sig_df = sig_df.sort_values(by="Exit Date") # Sort by exit for realizd pnl
                
                # Group by Exit Date to sum up daily PnL
                daily_pnl = sig_df.groupby("Exit Date")['PnL'].sum().cumsum()
                st.line_chart(daily_pnl)
                
                total_pnl = sig_df['PnL'].sum()
                st.metric("Total PnL (Simulated)", f"${total_pnl:,.2f}")

                # 2. Detailed Table
                st.subheader("📜 Historical Signal Log")
                st.dataframe(
                    sig_df.sort_values(by="Date", ascending=False).style.format({
                        "Price": "${:.2f}", "PnL": "${:.2f}"
                    }), 
                    use_container_width=True,
                    height=600 
                )
            else:
                st.success(f"✅ Found {len(sig_df)} Actionable Signals Today")
                save_signals_to_gsheet(sig_df, sheet_name='Trade_Signals_Log')
                st.dataframe(sig_df.style.format({"Entry": "${:.2f}", "Stop": "${:.2f}", "Target": "${:.2f}", "ATR": "{:.2f}"}), use_container_width=True)
                
                clip_text = ""
                for index, s in sig_df.iterrows():
                     clip_text += f"{s['Action']} {s['Shares']} {s['Ticker']} @ MKT. Stop: {s['Stop']:.2f}. Target: {s['Target']:.2f}. Exit Date: {s['Time Exit']}.\n"
                st.text_area("Execution Clipboard", clip_text, height=100)
        else:
            st.info("No signals found.")

if __name__ == "__main__":
    main()
