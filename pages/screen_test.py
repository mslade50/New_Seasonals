import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import time
from pandas.tseries.offsets import BusinessDay
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# 1. THE STRATEGY BOOK
# -----------------------------------------------------------------------------
STRATEGY_BOOK = [
    # 1. INDEX SEASONALS
    {
        "id": "Indx sznl > 85, 21dr < 15",
        "name": "Index Seasonals",
        "description": "Start: 2000-01-01. Universe: Indices. Dir: Long. Filter: None.",
        "universe_tickers": ['SPY', 'QQQ', 'IWM', 'DIA', 'SMH'], 
        "settings": {
            "trade_direction": "Long", "entry_type": "T+1 Open", "max_one_pos": False,
            "max_daily_entries": 5, "max_total_positions": 10,
            "use_perf_rank": True, "perf_window": 21, "perf_logic": "<", "perf_thresh": 15.0,
            "perf_first_instance": False, "perf_lookback": 21, "perf_consecutive": 1,
            "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 85.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000, "min_age": 0.25, "max_age": 100.0
        },
        "execution": { "risk_per_trade": 1000, "stop_atr": 2, "tgt_atr": 8.0, "hold_days": 21 },
        "stats": { "grade": "A (Excellent)", "win_rate": "64.1%", "expectancy": "$471.36", "profit_factor": "4.51" }
    },
    # 2. NO ACCUMULATION DAYS
    {
        "id": "5+10+21d > 85%ile, 0 acc days in last 21, sell t+1 open + 0.5 ATR 10d time stop",
        "name": "No Accumulation Days",
        "description": "Start: 2000-01-01. Universe: All CSV Tickers. Dir: Short. Filter: None. PF: 5.15. SQN: 3.82.",
        "universe_tickers": ['AAPL', 'ABT', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEP', 'AIG', 'ALL', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'AON', 'APD', 'AVGO', 'AXP', 'BA', 'BAC', 'BAX', 'BDX', 'BK', 'BMY', 'C', 'CAG', 'CAT', 'CEF', 'CL', 'CMCSA', 'CMS', 'CNP', 'COP', 'COST', 'CPB', 'CRM', 'CSCO', 'CSX', 'CVS', 'CVX', 'D', 'DE', 'DIA', 'DIS', 'DOV', 'DTE', 'DUK', 'ECL', 'ED', 'EIX', 'EMR', 'EOG', 'ETR', 'EXC', 'F', 'FCX', 'FDX', 'FE', 'GD', 'GE', 'GILD', 'GIS', 'GLD', 'GLW', 'GOOG', 'GPC', 'GS', 'HAL', 'HD', 'HIG', 'HON', 'HPQ', 'HRL', 'HSY', 'HUM', 'IBB', 'IBM', 'IHI', 'INTC', 'IP', 'ITA', 'ITB', 'ITW', 'IWM', 'IYR', 'JNJ', 'JPM', 'K', 'KEY', 'KMB', 'KO', 'KR', 'KRE', 'LEG', 'LIN', 'LLY', 'LMT', 'LOW', 'LUV', 'MAS', 'MCD', 'MDT', 'MET', 'META', 'MMC', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'MU', 'NEE', 'NEM', 'NKE', 'NOC', 'NSC', 'NUE', 'NVDA', 'OIH', 'ORCL', 'OXY', 'PAYX', 'PCG', 'PEG', 'PEP', 'PFE', 'PG', 'PGR', 'PH', 'PNW', 'PPG', 'PPL', 'PSA', 'QCOM', 'QQQ', 'REGN', 'RF', 'RHI', 'ROK', 'ROST', 'RTX', 'SBUX', 'SCHW', 'SHW', 'SLB', 'SLV', 'SMH', 'SNA', 'SO', 'SPG', 'SPY', 'SRE', 'STT', 'SWK', 'SYK', 'SYY', 'T', 'TAP', 'TGT', 'TJX', 'TMO', 'TRV', 'TSN', 'TXN', 'UNG', 'UNH', 'UNP', 'USB', 'USO', 'UVXY', 'V', 'VFC', 'VLO', 'VMC', 'VNQ', 'VZ', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOM', 'XOP', 'XRT', '^GSPC', '^NDX'], 
        "settings": {
            "trade_direction": "Short",
            "entry_type": "Limit (Open +/- 0.5 ATR)",
            "max_one_pos": True,
            "allow_same_day_reentry": False,
            "max_daily_entries": 2,
            "max_total_positions": 10,
            "perf_filters": [{'window': 5, 'logic': '>', 'thresh': 85.0, 'consecutive': 1}, {'window': 10, 'logic': '>', 'thresh': 85.0, 'consecutive': 1}, {'window': 21, 'logic': '>', 'thresh': 85.0, 'consecutive': 1}],
            "perf_first_instance": False, "perf_lookback": 21,
            "use_sznl": True, "sznl_logic": "<", "sznl_thresh": 33.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_market_sznl": False, "market_sznl_logic": "<", "market_sznl_thresh": 40.0,
            "market_ticker": "^GSPC",
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21, "52w_lag": 0,
            "breakout_mode": "None",
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 15.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0,
            "min_atr_pct": 0.0,"max_atr_pct": 10.0,
            "entry_conf_bps": 0,
            "use_ma_dist_filter": False, "dist_ma_type": "SMA 10", 
            "dist_logic": "Greater Than (>)", "dist_min": 0.0, "dist_max": 2.0,
            "use_gap_filter": False, "gap_lookback": 21, 
            "gap_logic": ">", "gap_thresh": 3,
            "use_acc_count_filter": True, "acc_count_window": 21, "acc_count_logic": "=", "acc_count_thresh": 0,
            "use_dist_count_filter": True, "dist_count_window": 21, "dist_count_logic": ">", "dist_count_thresh": 0
        },
        "execution": {
            "risk_per_trade": 500,
            "slippage_bps": 5,
            "stop_atr": 2.0,
            "tgt_atr": 5.0,
            "hold_days": 10
        },
        "stats": {
            "grade": "A (Excellent)",
            "win_rate": "66.7%",
            "expectancy": "$609.33",
            "profit_factor": "5.15"
        }
    },
    # 3. BOTTOM OF RANGE REVERSION
    {
        "id": "Lower 10% of range",
        "name": "Bottom of Range Reversion",
        "description": "Start: 2000-01-01. Universe: Indices. Dir: Long. Filter: None. PF: 1.49. SQN: 8.83.",
        "universe_tickers": ['SPY', 'QQQ', 'IWM', 'DIA', 'SMH'], 
        "settings": {
            "trade_direction": "Long", "entry_type": "Signal Close", "max_one_pos": True,
            "allow_same_day_reentry": False, "max_daily_entries": 2, "max_total_positions": 10,
            "use_range_filter": True, "range_min": 0.0, "range_max": 10.0,
            "perf_filters": [{'window': 5, 'logic': '<', 'thresh': 50.0, 'consecutive': 1}],
            "perf_first_instance": False, "perf_lookback": 21,
            "use_sznl": False, "sznl_logic": "<", "sznl_thresh": 15.0, "sznl_first_instance": True, "sznl_lookback": 21,
            "use_market_sznl": True, "market_sznl_logic": ">", "market_sznl_thresh": 33.0, "market_ticker": "^GSPC",
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000, "min_age": 0.25, "max_age": 100.0,
            "entry_conf_bps": 0,
            "use_dist_filter": False, "dist_ma_type": "SMA 10", "dist_logic": "Greater Than (>)", "dist_min": 0.0, "dist_max": 2.0,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": ">", "gap_thresh": 3
        },
        "execution": { "risk_per_trade": 100, "slippage_bps": 2, "stop_atr": 1.0, "tgt_atr": 2.0, "hold_days": 2 },
        "stats": { "grade": "B", "win_rate": "54.9%", "expectancy": "$174.76", "profit_factor": "1.49" }
    },
    # 5. LIQUID SEASONALS (SHORT TERM)
    {
        "id": "Sznl > 90, 5d <15 for 3d consec",
        "name": "Liquid Seasonals (short term)",
        "description": "Start: 2000-01-01. Universe: All CSV Tickers. Dir: Long. Filter: None.",
        "universe_tickers": ['AAPL', 'ABT', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEP', 'AIG', 'ALL', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'AON', 'APD', 'AVGO', 'AXP', 'BA', 'BAC', 'BAX', 'BDX', 'BK', 'BMY', 'C', 'CAG', 'CAT', 'CEF', 'CL', 'CMCSA', 'CMS', 'CNP', 'COP', 'COST', 'CPB', 'CRM', 'CSCO', 'CSX', 'CVS', 'CVX', 'D', 'DE', 'DIA', 'DIS', 'DOV', 'DTE', 'DUK', 'ECL', 'ED', 'EIX', 'EMR', 'EOG', 'ETR', 'EXC', 'F', 'FCX', 'FDX', 'FE', 'GD', 'GE', 'GILD', 'GIS', 'GLD', 'GLW', 'GOOG', 'GPC', 'GS', 'HAL', 'HD', 'HIG', 'HON', 'HPQ', 'HRL', 'HSY', 'HUM', 'IBB', 'IBM', 'IHI', 'INTC', 'IP', 'ITA', 'ITB', 'ITW', 'IWM', 'IYR', 'JNJ', 'JPM', 'K', 'KEY', 'KMB', 'KO', 'KR', 'KRE', 'LEG', 'LIN', 'LLY', 'LMT', 'LOW', 'LUV', 'MAS', 'MCD', 'MDT', 'MET', 'META', 'MMC', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'MU', 'NEE', 'NEM', 'NKE', 'NOC', 'NSC', 'NUE', 'NVDA', 'OIH', 'ORCL', 'OXY', 'PAYX', 'PCG', 'PEG', 'PEP', 'PFE', 'PG', 'PGR', 'PH', 'PNW', 'PPG', 'PPL', 'PSA', 'QCOM', 'QQQ', 'REGN', 'RF', 'RHI', 'ROK', 'ROST', 'RTX', 'SBUX', 'SCHW', 'SHW', 'SLB', 'SLV', 'SMH', 'SNA', 'SO', 'SPG', 'SPY', 'SRE', 'STT', 'SWK', 'SYK', 'SYY', 'T', 'TAP', 'TGT', 'TJX', 'TMO', 'TRV', 'TSN', 'TXN', 'UNG', 'UNH', 'UNP', 'USB', 'USO', 'UVXY', 'V', 'VFC', 'VLO', 'VMC', 'VNQ', 'VZ', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOM', 'XOP', 'XRT'], 
        "settings": {
            "trade_direction": "Long", "entry_type": "Signal Close", "max_one_pos": True,
            "max_daily_entries": 3, "max_total_positions": 10,
            "use_perf_rank": True, "perf_window": 5, "perf_logic": "<", "perf_thresh": 15.0,
            "perf_first_instance": False, "perf_lookback": 21, "perf_consecutive": 3,
            "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 90.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000, "min_age": 0.25, "max_age": 100.0
        },
        "execution": { "risk_per_trade": 500, "stop_atr": 2.0, "tgt_atr": 3.0, "hold_days": 5 },
        "stats": { "grade": "A (Excellent)", "win_rate": "61.1%", "expectancy": "$316.29", "profit_factor": "2.80" }
    },
    # 6. LIQUID SEASONALS (INTERMEDIATE)
    {
        "id": "Sznl > 85, 21dr < 15 3 consec",
        "name": "Liquid Seasonals (intermediate)",
        "description": "Start: 2000-01-01. Universe: All CSV Tickers. Dir: Long. Filter: None.",
        "universe_tickers": ['AAPL', 'ABT', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEP', 'AIG', 'ALL', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'AON', 'APD', 'AVGO', 'AXP', 'BA', 'BAC', 'BAX', 'BDX', 'BK', 'BMY', 'C', 'CAG', 'CAT', 'CEF', 'CL', 'CMCSA', 'CMS', 'CNP', 'COP', 'COST', 'CPB', 'CRM', 'CSCO', 'CSX', 'CVS', 'CVX', 'D', 'DE', 'DIA', 'DIS', 'DOV', 'DTE', 'DUK', 'ECL', 'ED', 'EIX', 'EMR', 'EOG', 'ETR', 'EXC', 'F', 'FCX', 'FDX', 'FE', 'GD', 'GE', 'GILD', 'GIS', 'GLD', 'GLW', 'GOOG', 'GPC', 'GS', 'HAL', 'HD', 'HIG', 'HON', 'HPQ', 'HRL', 'HSY', 'HUM', 'IBB', 'IBM', 'IHI', 'INTC', 'IP', 'ITA', 'ITB', 'ITW', 'IWM', 'IYR', 'JNJ', 'JPM', 'K', 'KEY', 'KMB', 'KO', 'KR', 'KRE', 'LEG', 'LIN', 'LLY', 'LMT', 'LOW', 'LUV', 'MAS', 'MCD', 'MDT', 'MET', 'META', 'MMC', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'MU', 'NEE', 'NEM', 'NKE', 'NOC', 'NSC', 'NUE', 'NVDA', 'OIH', 'ORCL', 'OXY', 'PAYX', 'PCG', 'PEG', 'PEP', 'PFE', 'PG', 'PGR', 'PH', 'PNW', 'PPG', 'PPL', 'PSA', 'QCOM', 'QQQ', 'REGN', 'RF', 'RHI', 'ROK', 'ROST', 'RTX', 'SBUX', 'SCHW', 'SHW', 'SLB', 'SLV', 'SMH', 'SNA', 'SO', 'SPG', 'SPY', 'SRE', 'STT', 'SWK', 'SYK', 'SYY', 'T', 'TAP', 'TGT', 'TJX', 'TMO', 'TRV', 'TSN', 'TXN', 'UNG', 'UNH', 'UNP', 'USB', 'USO', 'UVXY', 'V', 'VFC', 'VLO', 'VMC', 'VNQ', 'VZ', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOM', 'XOP', 'XRT'], 
        "settings": {
            "trade_direction": "Long", "entry_type": "Signal Close", "max_one_pos": True,
            "max_daily_entries": 3, "max_total_positions": 10,
            "use_perf_rank": True, "perf_window": 21, "perf_logic": "<", "perf_thresh": 15.0,
            "perf_first_instance": False, "perf_lookback": 21, "perf_consecutive": 3,
            "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 85.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000, "min_age": 0.25, "max_age": 100.0
        },
        "execution": { "risk_per_trade": 750, "stop_atr": 3.0, "tgt_atr": 8.0, "hold_days": 21 },
        "stats": { "grade": "A", "win_rate": "60.3%", "expectancy": "$265.02", "profit_factor": "1.97" }
    },
    # 7. UGLY MONDAY CLOSE
    {
        "id": "5dr < 50, sznl > 33, close < 20% range, close > 20d",
        "name": "Weak Close Decent Sznls",
        "description": "Start: 2000-01-01. Universe: Sector + Index ETFs. Dir: Long. Filter: None. PF: 2.19. SQN: 6.06.",
        "universe_tickers": ['SPY', 'OIH', 'XHB', 'DIA', 'XRT', 'IBB', 'ITA', 'XLF', 'XLY', 'XME', 'XLE', 'XLK', 'ITB', 'KRE', 'XLU', 'XLI', 'XLP', 'QQQ', 'IHI', 'VNQ', 'SMH', 'XBI', 'IWM', 'XLC', 'XLB', 'XOP', 'XLV', 'IYR'], 
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Signal Close",
            "max_one_pos": True,
            "allow_same_day_reentry": False,
            "max_daily_entries": 2,
            "max_total_positions": 10,
            "perf_filters": [{'window': 5, 'logic': '<', 'thresh': 50.0, 'consecutive': 1}],
            "perf_first_instance": False, "perf_lookback": 21,
            "ma_consec_filters": [{'length': 20, 'logic': 'Above', 'consec': 1}],
            "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 33.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_market_sznl": False, "market_sznl_logic": "<", "market_sznl_thresh": 15.0,
            "market_ticker": "^GSPC",
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21, "52w_lag": 0,
            "exclude_52w_high": False,
            "breakout_mode": "None",
            "use_range_filter": True, 
            "range_min": 0, 
            "range_max": 20,
            "use_dow_filter": True, 
            "allowed_days": [0, 2, 3, 4],
            "use_vix_filter": False, "vix_min": 0.0, "vix_max": 20.0,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0,
            "min_atr_pct": 0.2,"max_atr_pct": 10.0,
            "entry_conf_bps": 0,
            "use_ma_dist_filter": False, "dist_ma_type": "SMA 10", 
            "dist_logic": "Greater Than (>)", "dist_min": 0.0, "dist_max": 2.0,
            "use_gap_filter": False, "gap_lookback": 21, 
            "gap_logic": ">", "gap_thresh": 3,
            "use_acc_count_filter": True, "acc_count_window": 21, "acc_count_logic": ">", "acc_count_thresh": 3,
            "use_dist_count_filter": True, "dist_count_window": 21, "dist_count_logic": "<", "dist_count_thresh": 3
        },
        "execution": {
            "risk_per_trade": 400,
            "slippage_bps": 2,
            "stop_atr": 1.0,
            "tgt_atr": 8.0,
            "hold_days": 4
        },
        "stats": {
            "grade": "A (Excellent)",
            "win_rate": "63.2%",
            "expectancy": "0.44r",
            "profit_factor": "2.19"
        }
    },
    # 8. OVERSOLD LOW VOLUME
    {
        "id": "21dr < 15 3 consec, 5dr < 33, rel vol < 15, SPY > 200d",
        "name": "Oversold Low Volume",
        "description": "Start: 2000-01-01. Universe: All CSV Tickers. Dir: Long. Filter: SPY > 200 SMA.",
        "universe_tickers": ['AAPL', 'ABT', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEP', 'AIG', 'ALL', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'AON', 'APD', 'AVGO', 'AXP', 'BA', 'BAC', 'BAX', 'BDX', 'BK', 'BMY', 'C', 'CAG', 'CAT', 'CEF', 'CL', 'CMCSA', 'CMS', 'CNP', 'COP', 'COST', 'CPB', 'CRM', 'CSCO', 'CSX', 'CVS', 'CVX', 'D', 'DE', 'DIA', 'DIS', 'DOV', 'DTE', 'DUK', 'ECL', 'ED', 'EIX', 'EMR', 'EOG', 'ETR', 'EXC', 'F', 'FCX', 'FDX', 'FE', 'GD', 'GE', 'GILD', 'GIS', 'GLD', 'GLW', 'GOOG', 'GPC', 'GS', 'HAL', 'HD', 'HIG', 'HON', 'HPQ', 'HRL', 'HSY', 'HUM', 'IBB', 'IBM', 'IHI', 'INTC', 'IP', 'ITA', 'ITB', 'ITW', 'IWM', 'IYR', 'JNJ', 'JPM', 'K', 'KEY', 'KMB', 'KO', 'KR', 'KRE', 'LEG', 'LIN', 'LLY', 'LMT', 'LOW', 'LUV', 'MAS', 'MCD', 'MDT', 'MET', 'META', 'MMC', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'MU', 'NEE', 'NEM', 'NKE', 'NOC', 'NSC', 'NUE', 'NVDA', 'OIH', 'ORCL', 'OXY', 'PAYX', 'PCG', 'PEG', 'PEP', 'PFE', 'PG', 'PGR', 'PH', 'PNW', 'PPG', 'PPL', 'PSA', 'QCOM', 'QQQ', 'REGN', 'RF', 'RHI', 'ROK', 'ROST', 'RTX', 'SBUX', 'SCHW', 'SHW', 'SLB', 'SLV', 'SMH', 'SNA', 'SO', 'SPG', 'SPY', 'SRE', 'STT', 'SWK', 'SYK', 'SYY', 'T', 'TAP', 'TGT', 'TJX', 'TMO', 'TRV', 'TSN', 'TXN', 'UNG', 'UNH', 'UNP', 'USB', 'USO', 'UVXY', 'V', 'VFC', 'VLO', 'VMC', 'VNQ', 'VZ', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOM', 'XOP', 'XRT'], 
        "settings": {
            "trade_direction": "Long", "entry_type": "Signal Close", "max_one_pos": True,
            "max_daily_entries": 2, "max_total_positions": 5,
            "perf_filters": [{'window': 5, 'logic': '<', 'thresh': 33.0, 'consecutive': 1}, {'window': 21, 'logic': '<', 'thresh': 15.0, 'consecutive': 3}],
            "perf_first_instance": False, "perf_lookback": 21,
            "use_sznl": False, "sznl_logic": "<", "sznl_thresh": 15.0, "sznl_first_instance": True, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": True, "vol_rank_logic": "<", "vol_rank_thresh": 15.0,
            "trend_filter": "SPY > 200 SMA",
            "min_price": 10.0, "min_vol": 100000, "min_age": 0.25, "max_age": 100.0
        },
        "execution": { "risk_per_trade": 500, "stop_atr": 2.0, "tgt_atr": 8.0, "hold_days": 21 },
        "stats": { "grade": "A", "win_rate": "66.3%", "expectancy": "$670.59", "profit_factor": "2.64" }
    },
    # 9. OVERBOT VOLUME SPIKE
    {
        "id": "5+10+21d > 85, 21d 3x, vol >1.25x, >0 dist day, sell open +0.5 atr",
        "name": "Overbot Vol Spike",
        "description": "Start: 2000-01-01. Universe: All CSV Tickers. Dir: Short. Filter: None. PF: 2.46. SQN: 4.44.",
        "universe_tickers": ['AAPL', 'ABT', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEP', 'AIG', 'ALL', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'AON', 'APD', 'AVGO', 'AXP', 'BA', 'BAC', 'BAX', 'BDX', 'BK', 'BMY', 'C', 'CAG', 'CAT', 'CEF', 'CL', 'CMCSA', 'CMS', 'CNP', 'COP', 'COST', 'CPB', 'CRM', 'CSCO', 'CSX', 'CVS', 'CVX', 'D', 'DE', 'DIA', 'DIS', 'DOV', 'DTE', 'DUK', 'ECL', 'ED', 'EIX', 'EMR', 'EOG', 'ETR', 'EXC', 'F', 'FCX', 'FDX', 'FE', 'GD', 'GE', 'GILD', 'GIS', 'GLD', 'GLW', 'GOOG', 'GPC', 'GS', 'HAL', 'HD', 'HIG', 'HON', 'HPQ', 'HRL', 'HSY', 'HUM', 'IBB', 'IBM', 'IHI', 'INTC', 'IP', 'ITA', 'ITB', 'ITW', 'IWM', 'IYR', 'JNJ', 'JPM', 'K', 'KEY', 'KMB', 'KO', 'KR', 'KRE', 'LEG', 'LIN', 'LLY', 'LMT', 'LOW', 'LUV', 'MAS', 'MCD', 'MDT', 'MET', 'META', 'MMC', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'MU', 'NEE', 'NEM', 'NKE', 'NOC', 'NSC', 'NUE', 'NVDA', 'OIH', 'ORCL', 'OXY', 'PAYX', 'PCG', 'PEG', 'PEP', 'PFE', 'PG', 'PGR', 'PH', 'PNW', 'PPG', 'PPL', 'PSA', 'QCOM', 'QQQ', 'REGN', 'RF', 'RHI', 'ROK', 'ROST', 'RTX', 'SBUX', 'SCHW', 'SHW', 'SLB', 'SLV', 'SMH', 'SNA', 'SO', 'SPG', 'SPY', 'SRE', 'STT', 'SWK', 'SYK', 'SYY', 'T', 'TAP', 'TGT', 'TJX', 'TMO', 'TRV', 'TSN', 'TXN', 'UNG', 'UNH', 'UNP', 'USB', 'USO', 'UVXY', 'V', 'VFC', 'VLO', 'VMC', 'VNQ', 'VZ', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOM', 'XOP', 'XRT', '^GSPC', '^NDX'], 
        "settings": {
            "trade_direction": "Short",
            "entry_type": "Limit (Open +/- 0.5 ATR)",
            "max_one_pos": True,
            "allow_same_day_reentry": True,
            "max_daily_entries": 2,
            "max_total_positions": 10,
            "perf_filters": [{'window': 5, 'logic': '>', 'thresh': 85.0, 'consecutive': 1}, {'window': 10, 'logic': '>', 'thresh': 85.0, 'consecutive': 1}, {'window': 21, 'logic': '>', 'thresh': 85.0, 'consecutive': 3}],
            "perf_first_instance": False, "perf_lookback": 21,
            "use_sznl": False, "sznl_logic": ">", "sznl_thresh": 85.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_market_sznl": True, "market_sznl_logic": "<", "market_sznl_thresh": 40.0,
            "market_ticker": "^GSPC",
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": True, "vol_thresh": 1.25,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0,
            "min_atr_pct": 0.0,"max_atr_pct": 100.0,
            "entry_conf_bps": 0,
            "use_ma_dist_filter": False, "dist_ma_type": "SMA 10", 
            "dist_logic": "Greater Than (>)", "dist_min": 0.0, "dist_max": 2.0,
            "use_gap_filter": False, "gap_lookback": 21, 
            "gap_logic": ">", "gap_thresh": 3,
            "use_acc_count_filter": False, "acc_count_window": 21, "acc_count_logic": ">", "acc_count_thresh": 3,
            "use_dist_count_filter": True, "dist_count_window": 21, "dist_count_logic": ">", "dist_count_thresh": 0
        },
        "execution": {
            "risk_per_trade": 400,
            "slippage_bps": 2,
            "stop_atr": 1.0,
            "tgt_atr": 8.0,
            "hold_days": 3
        },
        "stats": {
            "grade": "A (Excellent)",
            "win_rate": "64.0%",
            "expectancy": "$411.09",
            "profit_factor": "2.46"
        }
    },
]

# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
PRIMARY_SZNL_PATH = "sznl_ranks.csv"      # Your main high-quality data
BACKUP_SZNL_PATH = "seasonal_ranks.csv"   # The backup data to fill gaps

@st.cache_resource 
def load_seasonal_map():
    """
    Loads two CSV sources and merges them. 
    Priority is given to PRIMARY_SZNL_PATH. 
    BACKUP_SZNL_PATH is used only where Primary has no data for that specific Ticker/Date.
    """
    
    # 1. Helper to load and clean a specific file
    def load_raw_csv(path):
        try:
            df = pd.read_csv(path)
            # Standardize column names if they differ between files
            # Assuming columns are 'ticker', 'Date', 'seasonal_rank'
            if 'ticker' not in df.columns or 'Date' not in df.columns:
                return pd.DataFrame()
            
            df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
            df = df.dropna(subset=["Date", "ticker"])
            df["Date"] = df["Date"].dt.normalize()
            df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
            return df
        except Exception:
            return pd.DataFrame()

    # 2. Load both
    df_primary = load_raw_csv(PRIMARY_SZNL_PATH)
    df_backup = load_raw_csv(BACKUP_SZNL_PATH)

    # 3. Merge Logic
    if df_primary.empty and df_backup.empty:
        return {}
    elif df_primary.empty:
        final_df = df_backup
    elif df_backup.empty:
        final_df = df_primary
    else:
        # Stack Primary on top of Backup
        # We assume both CSVs have the column 'seasonal_rank'
        final_df = pd.concat([df_primary, df_backup], axis=0)
        
        # Remove Duplicates:
        # We keep='first'. Since Primary is on top, if a Ticker+Date exists in both,
        # the Primary value is kept and the Backup value is dropped.
        final_df = final_df.drop_duplicates(subset=['ticker', 'Date'], keep='first')

    # 4. Build Dictionary Map
    output_map = {}
    # Optimization: Sort by date once to ensure monotonicity
    final_df = final_df.sort_values(by="Date")
    
    for ticker, group in final_df.groupby("ticker"):
        # Create a series mapping Date -> Rank
        series = group.set_index("Date")["seasonal_rank"]
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

# -----------------------------------------------------------------------------
# HELPER: BATCH DOWNLOADER (Optimized)
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def download_historical_data(tickers, start_date="2000-01-01"):
    if not tickers: return {}
    
    clean_tickers = list(set([str(t).strip().upper().replace('.', '-') for t in tickers]))
    
    data_dict = {}
    CHUNK_SIZE = 50 
    total = len(clean_tickers)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(0, total, CHUNK_SIZE):
        chunk = clean_tickers[i : i + CHUNK_SIZE]
        current_progress = min((i + CHUNK_SIZE) / total, 1.0)
        status_text.text(f"📥 Downloading batch {i+1}-{min(i+CHUNK_SIZE, total)} of {total}...")
        progress_bar.progress(current_progress)
        
        try:
            df = yf.download(
                chunk, 
                start=start_date, 
                group_by='ticker', 
                auto_adjust=False, 
                progress=False, 
                threads=True
            )
            
            if df.empty: continue
            
            if len(chunk) == 1:
                ticker = chunk[0]
                if 'Close' in df.columns:
                    df.index = df.index.tz_localize(None)
                    data_dict[ticker] = df
            else:
                available_tickers = df.columns.levels[0]
                for t in available_tickers:
                    try:
                        t_df = df[t].copy()
                        if t_df.empty or 'Close' not in t_df.columns: continue
                        t_df.index = t_df.index.tz_localize(None)
                        data_dict[t] = t_df
                    except Exception:
                        continue
            
            time.sleep(0.25) # Throttle
            
        except Exception:
            continue

    progress_bar.empty()
    status_text.empty()
    return data_dict

# -----------------------------------------------------------------------------
# ENGINE
# -----------------------------------------------------------------------------

def calculate_indicators(df, sznl_map, ticker, market_series=None):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    
    # --- MAs ---
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean() 
    df['SMA200'] = df['Close'].rolling(200).mean()
    
    # --- Gap Count ---
    is_open_gap = (df['Low'] > df['High'].shift(1)).astype(int)
    df['GapCount_21'] = is_open_gap.rolling(21).sum() 
    df['GapCount_10'] = is_open_gap.rolling(10).sum()
    df['GapCount_5'] = is_open_gap.rolling(5).sum() 

    # --- Candle Range Location % ---
    denom = (df['High'] - df['Low'])
    df['RangePct'] = np.where(denom == 0, 0.5, (df['Close'] - df['Low']) / denom)

    # --- Perf Ranks (Must be expanding from start of data) ---
    for window in [5, 10, 21]:
        df[f'ret_{window}d'] = df['Close'].pct_change(window)
        df[f'rank_ret_{window}d'] = df[f'ret_{window}d'].expanding(min_periods=50).rank(pct=True) * 100.0
        
    # --- ATR ---
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    
    # --- Volume Logic ---
    vol_ma = df['Volume'].rolling(63).mean()
    df['vol_ratio'] = df['Volume'] / vol_ma
    df['vol_ma'] = vol_ma
    
    # Spike logic
    cond_vol_ma = df['Volume'] > vol_ma
    cond_vol_up = df['Volume'] > df['Volume'].shift(1)
    df['Vol_Spike'] = cond_vol_ma & cond_vol_up
    
    # Accumulation
    cond_green = df['Close'] > df['Open']
    is_accumulation = (df['Vol_Spike'] & cond_green).astype(int)
    df['AccCount_21'] = is_accumulation.rolling(21).sum()
    
    # Distribution
    cond_red = df['Close'] < df['Open']
    is_distribution = (df['Vol_Spike'] & cond_red).astype(int)
    df['DistCount_21'] = is_distribution.rolling(21).sum()
    
    # Volume Rank
    vol_ma_10 = df['Volume'].rolling(10).mean()
    df['vol_ratio_10d'] = vol_ma_10 / vol_ma
    df['vol_ratio_10d_rank'] = df['vol_ratio_10d'].expanding(min_periods=50).rank(pct=True) * 100.0
    
    # Seasonality
    df['Sznl'] = get_sznl_val_series(ticker, df.index, sznl_map)
    df['Mkt_Sznl_Ref'] = get_sznl_val_series("^GSPC", df.index, sznl_map)
    
    # Age
    if not df.empty:
        start_ts = df.index[0]
        df['age_years'] = (df.index - start_ts).days / 365.25
    else:
        df['age_years'] = 0.0

    if market_series is not None:
        df['Market_Above_SMA200'] = market_series.reindex(df.index, method='ffill').fillna(False)
    
    # 52w
    rolling_high = df['High'].shift(1).rolling(252).max()
    rolling_low = df['Low'].shift(1).rolling(252).min()
    df['is_52w_high'] = df['High'] > rolling_high
    df['is_52w_low'] = df['Low'] < rolling_low
        
    return df

def get_historical_mask(df, params, sznl_map):
    mask = pd.Series(True, index=df.index)

    # 0. Day of Week Filter
    if params.get('use_dow_filter', False):
        allowed = params.get('allowed_days', [])
        mask &= df.index.dayofweek.isin(allowed)

    # 1. Liquidity & Age Gates
    mask &= (df['Close'] >= params.get('min_price', 0))
    mask &= (df['vol_ma'] >= params.get('min_vol', 0))
    mask &= (df['age_years'] >= params.get('min_age', 0))
    mask &= (df['age_years'] <= params.get('max_age', 100))

    # 1b. ATR % Filter (MISSING IN ORIGINAL)
    if 'ATR' in df.columns:
        # Calculate ATR % (ATR / Close * 100)
        atr_pct = (df['ATR'] / df['Close']) * 100
        min_atr = params.get('min_atr_pct', 0.0)
        max_atr = params.get('max_atr_pct', 1000.0)
        mask &= (atr_pct >= min_atr) & (atr_pct <= max_atr)

    # 2. Trend Filter (Global)
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

    # 2b. MA Consecutive Filters (MISSING IN ORIGINAL - Crucial for Ugly Monday)
    if 'ma_consec_filters' in params:
        for maf in params['ma_consec_filters']:
            length = maf['length']
            col_name = f"SMA{length}"
            if col_name not in df.columns: continue
            
            if maf['logic'] == 'Above':
                cond = df['Close'] > df[col_name]
            elif maf['logic'] == 'Below':
                cond = df['Close'] < df[col_name]
            else:
                continue

            consec = maf.get('consec', 1)
            if consec > 1:
                # Rolling sum of boolean (True=1) must equal window size
                cond = (cond.rolling(consec).sum() == consec)
            
            mask &= cond

    # 3. Candle Range Filter
    if params.get('use_range_filter', False):
        rn_val = df['RangePct'] * 100
        mask &= (rn_val >= params.get('range_min', 0)) & (rn_val <= params.get('range_max', 100))

    # 4. Perf Rank Filters
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
            # Check if sum of previous 'lookback' days is 0
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

    # 5. Gap/Acc/Dist Filters
    if params.get('use_gap_filter', False):
        lookback = params.get('gap_lookback', 21)
        col_name = f'GapCount_{lookback}' if f'GapCount_{lookback}' in df.columns else 'GapCount_21'
        g_val = df.get(col_name, 0)
        g_thresh = params.get('gap_thresh', 0)
        g_logic = params.get('gap_logic', '>')
        if g_logic == ">": mask &= (g_val > g_thresh)
        elif g_logic == "<": mask &= (g_val < g_thresh)
        elif g_logic == "=": mask &= (g_val == g_thresh)

    if params.get('use_acc_count_filter', False):
        window = params.get('acc_count_window', 21)
        col_name = f'AccCount_{window}'
        if col_name in df.columns:
            acc_thresh = params.get('acc_count_thresh', 0)
            acc_logic = params.get('acc_count_logic', '=')
            if acc_logic == '=': mask &= (df[col_name] == acc_thresh)
            elif acc_logic == '>': mask &= (df[col_name] > acc_thresh)
            elif acc_logic == '<': mask &= (df[col_name] < acc_thresh)

    if params.get('use_dist_count_filter', False):
        window = params.get('dist_count_window', 21)
        col_name = f'DistCount_{window}'
        if col_name in df.columns:
            dist_thresh = params.get('dist_count_thresh', 0)
            dist_logic = params.get('dist_count_logic', '>')
            if dist_logic == '=': mask &= (df[col_name] == dist_thresh)
            elif dist_logic == '>': mask &= (df[col_name] > dist_thresh)
            elif dist_logic == '<': mask &= (df[col_name] < dist_thresh)

    # 6. Distance Filter
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

    # 7. Seasonality
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

    # 8. 52w High/Low
    if params['use_52w']:
        if params['52w_type'] == 'New 52w High': cond_52 = df['is_52w_high']
        else: cond_52 = df['is_52w_low']
        
        if params.get('52w_first_instance', True):
            lookback = params.get('52w_lookback', 21)
            prev = cond_52.shift(1).rolling(lookback).sum()
            cond_52 &= (prev == 0)
        mask &= cond_52

    # 8b. Exclude 52w High (MISSING IN ORIGINAL)
    if params.get('exclude_52w_high', False):
        mask &= (~df['is_52w_high'])

    # 9. Volume (Ratio ONLY)
    if params['use_vol']:
        mask &= (df['vol_ratio'] > params['vol_thresh'])

    if params.get('use_vol_rank'):
        val = df['vol_ratio_10d_rank']
        if params['vol_rank_logic'] == '<': mask &= (val < params['vol_rank_thresh'])
        else: mask &= (val > params['vol_rank_thresh'])

    return mask.fillna(False)

def calculate_trade_result(df, signal_date, action, shares, entry_price, hold_days):
    start_idx = df.index.searchsorted(signal_date)
    if start_idx >= len(df) - 1:
        return 0, signal_date
    
    # We slice strictly by days ahead to respect time stop
    window = df.iloc[start_idx+1 : start_idx+1+hold_days].copy()
    
    if window.empty: 
        return 0, signal_date

    exit_row = window.iloc[-1]
    exit_price = exit_row['Close']
    exit_date = exit_row.name
    
    if action == "BUY":
        pnl = (exit_price - entry_price) * shares
    elif action == "SELL SHORT":
        pnl = (entry_price - exit_price) * shares
    else:
        pnl = 0

    return pnl, exit_date

def calculate_daily_exposure(sig_df):
    if sig_df.empty: return pd.DataFrame()
    min_date = sig_df['Date'].min()
    max_date = sig_df['Exit Date'].max()
    all_dates = pd.date_range(start=min_date, end=max_date)
    exposure_df = pd.DataFrame(0.0, index=all_dates, columns=['Long Exposure ($)', 'Short Exposure ($)'])
    for idx, row in sig_df.iterrows():
        trade_dates = pd.date_range(start=row['Date'], end=row['Exit Date'])
        dollar_val = row['Price'] * row['Shares']
        if row['Action'] == 'BUY':
            exposure_df.loc[exposure_df.index.isin(trade_dates), 'Long Exposure ($)'] += dollar_val
        elif row['Action'] == 'SELL SHORT':
            exposure_df.loc[exposure_df.index.isin(trade_dates), 'Short Exposure ($)'] += dollar_val
    exposure_df['Net Exposure ($)'] = exposure_df['Long Exposure ($)'] - exposure_df['Short Exposure ($)']
    return exposure_df

def calculate_performance_stats(sig_df):
    stats = []
    def get_metrics(df, name):
        if df.empty: return None
        count = len(df)
        total_pnl = df['PnL'].sum()
        gross_profit = df[df['PnL'] > 0]['PnL'].sum()
        gross_loss = abs(df[df['PnL'] < 0]['PnL'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan
        avg_pnl = df['PnL'].mean()
        std_pnl = df['PnL'].std()
        sqn = (avg_pnl / std_pnl * np.sqrt(count)) if std_pnl != 0 else 0
        daily_pnl = df.groupby("Exit Date")['PnL'].sum()
        daily_mean = daily_pnl.mean()
        daily_std = daily_pnl.std()
        sharpe = (daily_mean / daily_std * np.sqrt(252)) if daily_std != 0 else 0
        return {
            "Strategy": name,
            "Trades": count,
            "Total PnL": total_pnl,
            "Sharpe Ratio": sharpe,
            "Profit Factor": profit_factor,
            "SQN": sqn
        }
    strategies = sig_df['Strategy'].unique()
    for strat in strategies:
        strat_df = sig_df[sig_df['Strategy'] == strat]
        m = get_metrics(strat_df, strat)
        if m: stats.append(m)
    total_m = get_metrics(sig_df, "TOTAL PORTFOLIO")
    if total_m: stats.append(total_m)
    return pd.DataFrame(stats)

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="Strategy Backtest Lab")
    
    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("⚙️ Backtest Settings")

    # 1. HARD REFRESH BUTTON (Crucial for fixing cache issues)
    if st.sidebar.button("🔴 Force Clear Cache & Data"):
        st.cache_data.clear()
        st.cache_resource.clear()
        if 'backtest_data' in st.session_state:
            del st.session_state['backtest_data']
        st.rerun()
    
    current_year = datetime.date.today().year
    selected_year = st.sidebar.slider("Select Start Year", 2000, current_year, current_year - 2)
    default_date = datetime.date(selected_year, 1, 1)
    
    with st.sidebar.form("backtest_form"):
        user_start_date = st.date_input(
            "Backtest Start Date", 
            value=default_date, 
            min_value=datetime.date(2000, 1, 1)
        )
        st.caption(f"Data download buffer: 365 days prior to {user_start_date}.")
        run_btn = st.form_submit_button("⚡ Run Backtest")

    st.title("⚡ Strategy Backtest Lab")
    st.markdown(f"**Selected Start Date:** {user_start_date}")
    st.markdown("---")

    if run_btn:
        sznl_map = load_seasonal_map()
        
        if 'backtest_data' not in st.session_state:
            st.session_state['backtest_data'] = {}

        # -------------------------------------------------------------------------
        # 1. TICKER COLLECTION
        # -------------------------------------------------------------------------
        long_term_tickers = set()
        
        for strat in STRATEGY_BOOK:
            # Universe
            for t in strat['universe_tickers']:
                long_term_tickers.add(t)
            
            # Market Refs
            s = strat['settings']
            if s.get('use_market_sznl'): long_term_tickers.add(s.get('market_ticker', '^GSPC'))
            if "Market" in s.get('trend_filter', ''): long_term_tickers.add(s.get('market_ticker', 'SPY'))
            if "SPY" in s.get('trend_filter', ''): long_term_tickers.add("SPY")

        # Clean formatting (Standardize to Yahoo format, e.g., BRK.B -> BRK-B)
        long_term_list = [t.replace('.', '-') for t in long_term_tickers]

        # -------------------------------------------------------------------------
        # 2. BATCH DOWNLOAD
        # -------------------------------------------------------------------------
        existing_keys = set(st.session_state['backtest_data'].keys())
        missing_long = list(set(long_term_list) - existing_keys)
        
        if missing_long:
            st.write(f"📥 Downloading **Deep History (from 2000)** for {len(missing_long)} tickers...")
            data_long = download_historical_data(missing_long, start_date="2000-01-01")
            st.session_state['backtest_data'].update(data_long)
            st.success("✅ Download Batch Complete.")

        # --- DATA INTEGRITY CHECK (NEW) ---
        # Verify that everything we asked for is actually in the dictionary
        master_dict = st.session_state['backtest_data']
        downloaded_keys = set(master_dict.keys())
        requested_set = set(long_term_list)
        
        failed_tickers = requested_set - downloaded_keys
        
        if failed_tickers:
            st.error(f"⚠️ {len(failed_tickers)} Tickers failed to download or returned no data.")
            with st.expander("View Failed Tickers"):
                st.write(list(failed_tickers))
                st.caption("These tickers will be skipped in the backtest.")
        # -----------------------------------

        all_signals = []
        progress_bar = st.progress(0)
        
        # -------------------------------------------------------------------------
        # 3. STRATEGY EXECUTION LOOP
        # -------------------------------------------------------------------------
        for i, strat in enumerate(STRATEGY_BOOK):
            progress_bar.progress((i + 1) / len(STRATEGY_BOOK))
            
            strat_mkt_ticker = strat['settings'].get('market_ticker', 'SPY')
            mkt_df = master_dict.get(strat_mkt_ticker)
            if mkt_df is None: mkt_df = master_dict.get('SPY')
            
            market_series = None
            if mkt_df is not None:
                temp_mkt = mkt_df.copy()
                temp_mkt['SMA200'] = temp_mkt['Close'].rolling(200).mean()
                market_series = temp_mkt['Close'] > temp_mkt['SMA200']

            # --- STRATEGY COVERAGE CHECK (NEW) ---
            # Check which tickers in this strategy's universe are missing data
            strat_universe_clean = [t.replace('.', '-') for t in strat['universe_tickers']]
            missing_in_strat = [t for t in strat_universe_clean if t not in master_dict]
            
            if missing_in_strat:
                st.warning(f"Strategy **'{strat['name']}'** is skipping {len(missing_in_strat)} tickers due to missing data.")
                # Optional: print them to console or expander if needed
                # print(f"Missing for {strat['name']}: {missing_in_strat}")

            for ticker in strat['universe_tickers']:
                t_clean = ticker.replace('.', '-')
                
                df = master_dict.get(t_clean)
                if df is None: continue # Skip if download failed
                if len(df) < 200: continue
                
                try:
                    # Calculate indicators
                    df = calculate_indicators(df, sznl_map, t_clean, market_series)

                    # Get mask
                    mask = get_historical_mask(df, strat['settings'], sznl_map)
                    
                    # --- BACKTEST FILTER ---
                    cutoff_ts = pd.Timestamp(user_start_date)
                    mask = mask[mask.index >= cutoff_ts]
                    
                    if not mask.any(): continue

                    true_dates = mask[mask].index
                    last_exit_date = None
                    
                    for d in true_dates:
                        if last_exit_date is not None and d <= last_exit_date:
                            continue
                            
                        row = df.loc[d]
                        atr = row['ATR']
                        risk = strat['execution']['risk_per_trade']
                        
                        # Dynamic Risk Adjustments
                        if strat['name'] == "Overbot Vol Spike":
                            vol_ratio = row.get('vol_ratio', 0)
                            if vol_ratio > 2.0: risk = 675
                            elif vol_ratio > 1.5: risk = 525
                        
                        if strat['name'] == "Weak Close Decent Sznls":
                            sznl_val = row.get('Sznl', 0)
                            if sznl_val >= 65: risk = risk * 1.5
                            elif sznl_val >= 50: risk = risk * 1.0
                            elif sznl_val >= 33: risk = risk * 0.66
                        
                        entry_type = strat['settings'].get('entry_type', 'Signal Close')
                        entry_idx = df.index.get_loc(d)
                        
                        if entry_idx + 1 >= len(df): continue 
                        entry_row = df.iloc[entry_idx + 1]

                        if entry_type == 'T+1 Close':
                            entry = entry_row['Close']
                            entry_date = entry_row.name 
                        elif entry_type == 'T+1 Open':
                            entry = entry_row['Open']
                            entry_date = entry_row.name 
                        elif entry_type == "Limit (Open +/- 0.5 ATR)":
                            limit_offset = 0.5 * atr
                            if strat['settings']['trade_direction'] == 'Short':
                                limit_price = entry_row['Open'] + limit_offset
                                if entry_row['High'] < limit_price: continue 
                                entry = limit_price
                            else:
                                limit_price = entry_row['Open'] - limit_offset
                                if entry_row['Low'] > limit_price: continue
                                entry = limit_price
                            entry_date = entry_row.name
                        else: 
                            entry = row['Close']
                            entry_date = d 
                        
                        direction = strat['settings'].get('trade_direction', 'Long')
                        
                        if direction == 'Long':
                            stop_price = entry - (atr * strat['execution']['stop_atr'])
                            dist = entry - stop_price
                            action = "BUY"
                        else:
                            stop_price = entry + (atr * strat['execution']['stop_atr'])
                            dist = stop_price - entry
                            action = "SELL SHORT"
                        
                        shares = int(risk / dist) if dist > 0 else 0
                        
                        pnl, exit_date = calculate_trade_result(
                            df, entry_date, action, shares, entry,
                            strat['execution']['hold_days']
                        )

                        hold_days = strat['execution']['hold_days']
                        try:
                            e_idx = df.index.get_loc(entry_date)
                            ts_idx = e_idx + hold_days
                            
                            if ts_idx < len(df):
                                time_stop_date = df.index[ts_idx]
                            else:
                                time_stop_date = entry_date + BusinessDay(hold_days)
                        except:
                            time_stop_date = pd.NaT

                        all_signals.append({
                            "Date": d.date(), 
                            "Exit Date": exit_date.date(),
                            "Time Stop": time_stop_date, 
                            "Strategy": strat['name'],
                            "Ticker": ticker,
                            "Action": action,
                            "Entry Criteria": entry_type,
                            "Price": entry,
                            "Shares": shares,
                            "PnL": pnl,
                            "ATR": atr,
                            "Range %": row['RangePct'] * 100
                        })
                        last_exit_date = exit_date
                    
                except Exception:
                    continue
        
        progress_bar.empty()
        
        if all_signals:
            sig_df = pd.DataFrame(all_signals)
            sig_df['Date'] = pd.to_datetime(sig_df['Date'])
            sig_df['Exit Date'] = pd.to_datetime(sig_df['Exit Date'])
            sig_df['Time Stop'] = pd.to_datetime(sig_df['Time Stop'])
            sig_df = sig_df.sort_values(by="Exit Date")

            # =================================================================
            # NEW: CURRENT EXPOSURE SECTION
            # =================================================================
            today = pd.Timestamp(datetime.date.today())
            
            # 1. Filter for Open Positions (Time Stop >= Today)
            open_mask = sig_df['Time Stop'] >= today
            open_df = sig_df[open_mask].copy()

            if not open_df.empty:
                # 2. Calculate Open PnL & Current Exposure
                current_prices = []
                open_pnls = []
                current_values = []

                for idx, row in open_df.iterrows():
                    ticker = row['Ticker']
                    t_df = master_dict.get(ticker.replace('.', '-'))
                    
                    if t_df is not None and not t_df.empty:
                        last_close = t_df.iloc[-1]['Close']
                    else:
                        last_close = row['Price']

                    if row['Action'] == 'BUY':
                        pnl = (last_close - row['Price']) * row['Shares']
                        val = last_close * row['Shares']
                    else:
                        pnl = (row['Price'] - last_close) * row['Shares']
                        val = last_close * row['Shares']

                    current_prices.append(last_close)
                    open_pnls.append(pnl)
                    current_values.append(val)

                open_df['Current Price'] = current_prices
                open_df['Open PnL'] = open_pnls
                open_df['Mkt Value'] = current_values

                total_long = open_df[open_df['Action'] == 'BUY']['Mkt Value'].sum()
                total_short = open_df[open_df['Action'] == 'SELL SHORT']['Mkt Value'].sum()
                net_exposure = total_long - total_short
                total_open_pnl = open_df['Open PnL'].sum()
                num_positions = len(open_df)

                st.divider()
                st.subheader("💼 Current Exposure (Active Positions)")
                
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("# Positions", num_positions)
                m2.metric("Total Long", f"${total_long:,.0f}")
                m3.metric("Total Short", f"${total_short:,.0f}")
                m4.metric("Net Exposure", f"${net_exposure:,.0f}")
                m5.metric("Total Open PnL", f"${total_open_pnl:,.2f}", 
                          delta_color="normal", 
                          delta=f"{total_open_pnl:,.2f}")

                st.dataframe(open_df.style.format({
                    "Date": "{:%Y-%m-%d}", 
                    "Time Stop": "{:%Y-%m-%d}",
                    "Price": "${:.2f}", 
                    "Current Price": "${:.2f}",
                    "Open PnL": "${:,.2f}",
                    "Range %": "{:.1f}%"
                }), use_container_width=True)
            else:
                st.info("No active positions (Time Stop >= Today).")

            st.divider()
            st.subheader("📊 Strategy Performance Metrics")
            stats_df = calculate_performance_stats(sig_df)
            st.dataframe(stats_df.style.format({
                "Total PnL": "${:,.2f}", "Sharpe Ratio": "{:.2f}", 
                "Profit Factor": "{:.2f}", "SQN": "{:.2f}"
            }), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📈 Total Portfolio PnL")
                
                # 1. Use your existing logic to get the Equity Curve
                total_daily_pnl = sig_df.groupby("Exit Date")['PnL'].sum().cumsum()
                
                # 2. Convert to DataFrame for calculations
                df_eq = total_daily_pnl.to_frame(name='Equity')
                
                # 3. Calculate Indicators (SMA 20 + Bollinger Bands)
                df_eq['SMA20'] = df_eq['Equity'].rolling(window=20).mean()
                df_eq['StdDev'] = df_eq['Equity'].rolling(window=20).std()
                df_eq['Upper'] = df_eq['SMA20'] + (2 * df_eq['StdDev'])
                df_eq['Lower'] = df_eq['SMA20'] - (2 * df_eq['StdDev'])
                
                # 4. Plot with Plotly (Standard Line Chart doesn't support bands well)
                fig = go.Figure()

                # Upper Band (Hidden line for fill reference)
                fig.add_trace(go.Scatter(
                    x=df_eq.index, y=df_eq['Upper'],
                    mode='lines', line=dict(width=0),
                    showlegend=False, hoverinfo='skip'
                ))

                # Lower Band (Fill up to Upper)
                fig.add_trace(go.Scatter(
                    x=df_eq.index, y=df_eq['Lower'],
                    mode='lines', line=dict(width=0),
                    fill='tonexty', fillcolor='rgba(255, 255, 255, 0.1)', # Subtle grey shading
                    name='Bollinger Band (20, 2)',
                    hoverinfo='skip'
                ))

                # SMA 20 (Orange Dashed)
                fig.add_trace(go.Scatter(
                    x=df_eq.index, y=df_eq['SMA20'],
                    mode='lines', name='SMA 20',
                    line=dict(color='orange', width=1, dash='dot')
                ))

                # Main Equity Curve (Green)
                fig.add_trace(go.Scatter(
                    x=df_eq.index, y=df_eq['Equity'],
                    mode='lines', name='Total PnL',
                    line=dict(color='#00FF00', width=2)
                ))

                fig.update_layout(
                    margin=dict(l=10, r=10, t=30, b=10),
                    height=400,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified",
                    xaxis_title="Date",
                    yaxis_title="Cumulative PnL ($)"
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.subheader("📉 Cumulative PnL by Strategy")
                strat_pnl = sig_df.pivot_table(index='Exit Date', columns='Strategy', values='PnL', aggfunc='sum').fillna(0)
                st.line_chart(strat_pnl.cumsum())

            st.subheader("⚖️ Portfolio Exposure Over Time")
            exposure_df = calculate_daily_exposure(sig_df)
            if not exposure_df.empty: st.line_chart(exposure_df)

            st.subheader("📜 Historical Signal Log")
            st.dataframe(sig_df.sort_values(by="Date", ascending=False).style.format({
                "Price": "${:.2f}", "PnL": "${:.2f}", "Date": "{:%Y-%m-%d}", "Exit Date": "{:%Y-%m-%d}", "Time Stop": "{:%Y-%m-%d}", "Range %": "{:.1f}%"
            }), use_container_width=True, height=400)
        else:
            st.warning(f"No signals found in the backtest period starting from {user_start_date}.")
    else:
        st.info("👈 Please select a start year/date and click 'Run Backtest' in the sidebar to begin.")

if __name__ == "__main__":
    main()
