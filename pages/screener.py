import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import gspread
from pandas.tseries.offsets import BusinessDay

# -----------------------------------------------------------------------------
# 1. THE STRATEGY BOOK (STATIC BATCH)
# -----------------------------------------------------------------------------
STRATEGY_BOOK = [
    # STRATEGY 1: OVERSOLD INDICES
    {
        "id": "IND_OS_SZNL",
        "name": "Oversold Indices + Bullish Seasonality",
        "description": "Major index etfs, sznl > 80, 5d trailing < 15",
        "universe_tickers": ["SPY", "QQQ", "IWM", "DIA", "SMH"], 
        "settings": {
            "trade_direction": "Long",
            "max_one_pos": False,
            "use_perf_rank": True, "perf_window": 5, "perf_logic": "<", "perf_thresh": 15.0,
            "perf_first_instance": True, "perf_lookback": 21, "perf_consecutive": 1,
            "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 80.0, "sznl_first_instance": True, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": False, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0
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
        "id": "Sznl > 85, 5dr < 15, 5d time stop",
        "name": "Generated Strategy (A)",
        "description": "Start: 2000-01-01. Universe: All CSV Tickers. Dir: Long. Filter: None. PF: 2.38. SQN: 12.92.",
        "universe_tickers": ['AAPL', 'ABT', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEP', 'AIG', 'ALL', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'AON', 'APD', 'AVGO', 'AXP', 'BA', 'BAC', 'BAX', 'BDX', 'BK', 'BMY', 'C', 'CAG', 'CAT', 'CEF', 'CL', 'CMCSA', 'CMS', 'CNP', 'COP', 'COST', 'CPB', 'CRM', 'CSCO', 'CSX', 'CVS', 'CVX', 'D', 'DE', 'DIA', 'DIS', 'DOV', 'DTE', 'DUK', 'ECL', 'ED', 'EIX', 'EMR', 'EOG', 'ETR', 'EXC', 'F', 'FCX', 'FDX', 'FE', 'GD', 'GE', 'GILD', 'GIS', 'GLD', 'GLW', 'GOOG', 'GPC', 'GS', 'HAL', 'HD', 'HIG', 'HON', 'HPQ', 'HRL', 'HSY', 'HUM', 'IBB', 'IBM', 'IHI', 'INTC', 'IP', 'ITA', 'ITB', 'ITW', 'IWM', 'IYR', 'JNJ', 'JPM', 'K', 'KEY', 'KMB', 'KO', 'KR', 'KRE', 'LEG', 'LIN', 'LLY', 'LMT', 'LOW', 'LUV', 'MAS', 'MCD', 'MDT', 'MET', 'META', 'MMC', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'MU', 'NEE', 'NEM', 'NKE', 'NOC', 'NSC', 'NUE', 'NVDA', 'OIH', 'ORCL', 'OXY', 'PAYX', 'PCG', 'PEG', 'PEP', 'PFE', 'PG', 'PGR', 'PH', 'PNW', 'PPG', 'PPL', 'PSA', 'QCOM', 'QQQ', 'REGN', 'RF', 'RHI', 'ROK', 'ROST', 'RTX', 'SBUX', 'SCHW', 'SHW', 'SLB', 'SLV', 'SMH', 'SNA', 'SO', 'SPG', 'SPY', 'SRE', 'STT', 'SWK', 'SYK', 'SYY', 'T', 'TAP', 'TGT', 'TJX', 'TMO', 'TRV', 'TSN', 'TXN', 'UNG', 'UNH', 'UNP', 'USB', 'USO', 'UVXY', 'V', 'VFC', 'VLO', 'VMC', 'VNQ', 'VZ', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOM', 'XOP', 'XRT'], 
        "settings": {
            "trade_direction": "Long",
            "entry_type": "T+1 Open",
            "max_one_pos": True,
            "max_daily_entries": 2,
            "max_total_positions": 10,
            "use_perf_rank": True, "perf_window": 5, "perf_logic": "<", "perf_thresh": 15.0,
            "perf_first_instance": False, "perf_lookback": 21, "perf_consecutive": 1,
            "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 85.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0
        },
        "execution": {
            "risk_per_trade": 300,
            "stop_atr": 2.0,
            "tgt_atr": 8.0,
            "hold_days": 5
        },
        "stats": {
            "grade": "A (Excellent)",
            "win_rate": "65.8%",
            "expectancy": "$270.09",
            "profit_factor": "2.38"
        }
    },
    {
        "id": "STRAT_1764794945",
        "name": "Seasonal < 10, 21d t > 70 for 3 consec days. 5d Time Stop",
        "description": "Start: 2000-01-01. Universe: All CSV Tickers. Dir: Short. Filter: None. PF: 1.70. SQN: 5.23.",
        "universe_tickers": ['AAPL', 'ABT', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEP', 'AIG', 'ALL', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'AON', 'APD', 'AVGO', 'AXP', 'BA', 'BAC', 'BAX', 'BDX', 'BK', 'BMY', 'C', 'CAG', 'CAT', 'CEF', 'CL', 'CMCSA', 'CMS', 'CNP', 'COP', 'COST', 'CPB', 'CRM', 'CSCO', 'CSX', 'CVS', 'CVX', 'D', 'DE', 'DIA', 'DIS', 'DOV', 'DTE', 'DUK', 'ECL', 'ED', 'EIX', 'EMR', 'EOG', 'ETR', 'EXC', 'F', 'FCX', 'FDX', 'FE', 'GD', 'GE', 'GILD', 'GIS', 'GLD', 'GLW', 'GOOG', 'GPC', 'GS', 'HAL', 'HD', 'HIG', 'HON', 'HPQ', 'HRL', 'HSY', 'HUM', 'IBB', 'IBM', 'IHI', 'INTC', 'IP', 'ITA', 'ITB', 'ITW', 'IWM', 'IYR', 'JNJ', 'JPM', 'K', 'KEY', 'KMB', 'KO', 'KR', 'KRE', 'LEG', 'LIN', 'LLY', 'LMT', 'LOW', 'LUV', 'MAS', 'MCD', 'MDT', 'MET', 'META', 'MMC', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'MU', 'NEE', 'NEM', 'NKE', 'NOC', 'NSC', 'NUE', 'NVDA', 'OIH', 'ORCL', 'OXY', 'PAYX', 'PCG', 'PEG', 'PEP', 'PFE', 'PG', 'PGR', 'PH', 'PNW', 'PPG', 'PPL', 'PSA', 'QCOM', 'QQQ', 'REGN', 'RF', 'RHI', 'ROK', 'ROST', 'RTX', 'SBUX', 'SCHW', 'SHW', 'SLB', 'SLV', 'SMH', 'SNA', 'SO', 'SPG', 'SPY', 'SRE', 'STT', 'SWK', 'SYK', 'SYY', 'T', 'TAP', 'TGT', 'TJX', 'TMO', 'TRV', 'TSN', 'TXN', 'UNG', 'UNH', 'UNP', 'USB', 'USO', 'UVXY', 'V', 'VFC', 'VLO', 'VMC', 'VNQ', 'VZ', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOM', 'XOP', 'XRT'], 
        "settings": {
            "trade_direction": "Short",
            "entry_type": "Signal Close",
            "max_one_pos": True,
            "max_daily_entries": 2,
            "max_total_positions": 5,
            "use_perf_rank": True, "perf_window": 21, "perf_logic": ">", "perf_thresh": 70.0,
            "perf_first_instance": False, "perf_lookback": 21, "perf_consecutive": 3,
            "use_sznl": True, "sznl_logic": "<", "sznl_thresh": 10.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0
        },
        "execution": {
            "risk_per_trade": 1000,
            "stop_atr": 2.0,
            "tgt_atr": 8.0,
            "hold_days": 5
        },
        "stats": {
            "grade": "A (Excellent)",
            "win_rate": "55.0%",
            "expectancy": "$165.83",
            "profit_factor": "1.70"
        }
    },
    # STRATEGY 2: LARGE CAP MEAN REVERSION (A)
    {
        "id": "Sznl > 80, 5d perf < 15, SPY > 200d",
        "name": "Large Cap Mean Reversion (A)",
        "description": "Universe: Liquid Large Caps. Sznl >80, 5d perf < 15. Filter: SPY > 200 SMA.",
        "universe_tickers": ['AAPL', 'AMGN', 'AMZN', 'AVGO', 'AXP', 'BA', 'CAT', 'CEF', 'CRM', 'CSCO', 'CVX', 'DIA', 'DIS', 'GLD', 'GOOG', 'GS', 'HD', 'HON', 'IBB', 'IBM', 'IHI', 'INTC', 'ITA', 'ITB', 'IWM', 'IYR', 'JNJ', 'JPM', 'KO', 'KRE', 'MCD', 'META', 'MMM', 'MRK', 'MSFT', 'NKE', 'NVDA', 'OIH', 'PG', 'QQQ', 'SLV', 'SMH', 'SPY', 'TRV', 'UNG', 'UNH', 'UVXY', 'V', 'VNQ', 'VZ', 'WMT', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOP', 'XRT'], 
        "settings": {
            "trade_direction": "Long",
            "max_one_pos": True,
            "use_perf_rank": True, "perf_window": 5, "perf_logic": "<", "perf_thresh": 15.0,
            "perf_first_instance": True, "perf_lookback": 3, "perf_consecutive": 1,
            "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 80.0, "sznl_first_instance": True, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": False, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
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
    {
        "id": "Lower 10% of Range 5d perf < 50%ile",
        "name": "Ugly Monday Close",
        "description": "Start: 2000-01-01. Universe: Indices. Dir: Long. Filter: None. PF: 2.36. SQN: 6.97.",
        "universe_tickers": ['SPY', 'QQQ', 'IWM', 'DIA', 'SMH'], 
        "settings": {
            "trade_direction": "Long",
            "entry_type": "T+1 Open",
            "max_one_pos": True,
            "max_daily_entries": 5,
            "max_total_positions": 10,
            "use_perf_rank": True, "perf_window": 5, "perf_logic": "<", "perf_thresh": 50.0,
            "perf_first_instance": False, "perf_lookback": 21, "perf_consecutive": 1,
            "use_sznl": False, "sznl_logic": ">", "sznl_thresh": 50.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0
        },
        "execution": {
            "risk_per_trade": 1000,
            "stop_atr": 2.0,
            "tgt_atr": 8.0,
            "hold_days": 4
        },
        "stats": {
            "grade": "A (Excellent)",
            "win_rate": "68.8%",
            "expectancy": "$237.24",
            "profit_factor": "2.36"
        }
    },
    # STRATEGY 3: LARGE CAP MEAN REVERSION (B) - HIGHER PRECISION
    {
        "id": "5d of 21d perf < 15, 10d Rel Vol < 15",
        "name": "Large Cap Mean Reversion (B)",
        "description": "Universe: Liquid Large Caps. 21d perf < 15 (5 consec days). Low Rel Vol (<15 rank). No Trend Filter.",
        "universe_tickers": ['AAPL', 'AMGN', 'AMZN', 'AVGO', 'AXP', 'BA', 'CAT', 'CEF', 'CRM', 'CSCO', 'CVX', 'DIA', 'DIS', 'GLD', 'GOOG', 'GS', 'HD', 'HON', 'IBB', 'IBM', 'IHI', 'INTC', 'ITA', 'ITB', 'IWM', 'IYR', 'JNJ', 'JPM', 'KO', 'KRE', 'MCD', 'META', 'MMM', 'MRK', 'MSFT', 'NKE', 'NVDA', 'OIH', 'PG', 'QQQ', 'SLV', 'SMH', 'SPY', 'TRV', 'UNG', 'UNH', 'UVXY', 'V', 'VNQ', 'VZ', 'WMT', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOP', 'XRT'], 
        "settings": {
            "trade_direction": "Long",
            "max_one_pos": True,
            "use_perf_rank": True, "perf_window": 21, "perf_logic": "<", "perf_thresh": 15.0,
            "perf_first_instance": True, "perf_lookback": 21, "perf_consecutive": 5,
            "use_sznl": False, "sznl_logic": "<", "sznl_thresh": 15.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": False, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": True, "vol_rank_logic": "<", "vol_rank_thresh": 15.0,
            "trend_filter": "None",
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
            "win_rate": "73.1%",
            "expectancy": "$726.57",
            "profit_factor": "3.71"
        }
    },
    {
        "id": "5d of 21d perf < 15, sznl > 85",
        "name": "Generated Strategy (A)",
        "description": "Start: 2000-01-01. Universe: All CSV Tickers. Dir: Long. Filter: None. PF: 2.16. SQN: 9.97.",
        "universe_tickers": ['AAPL', 'ABT', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEP', 'AIG', 'ALL', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'AON', 'APD', 'AVGO', 'AXP', 'BA', 'BAC', 'BAX', 'BDX', 'BK', 'BMY', 'C', 'CAG', 'CAT', 'CEF', 'CL', 'CMCSA', 'CMS', 'CNP', 'COP', 'COST', 'CPB', 'CRM', 'CSCO', 'CSX', 'CVS', 'CVX', 'D', 'DE', 'DIA', 'DIS', 'DOV', 'DTE', 'DUK', 'ECL', 'ED', 'EIX', 'EMR', 'EOG', 'ETR', 'EXC', 'F', 'FCX', 'FDX', 'FE', 'GD', 'GE', 'GILD', 'GIS', 'GLD', 'GLW', 'GOOG', 'GPC', 'GS', 'HAL', 'HD', 'HIG', 'HON', 'HPQ', 'HRL', 'HSY', 'HUM', 'IBB', 'IBM', 'IHI', 'INTC', 'IP', 'ITA', 'ITB', 'ITW', 'IWM', 'IYR', 'JNJ', 'JPM', 'K', 'KEY', 'KMB', 'KO', 'KR', 'KRE', 'LEG', 'LIN', 'LLY', 'LMT', 'LOW', 'LUV', 'MAS', 'MCD', 'MDT', 'MET', 'META', 'MMC', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'MU', 'NEE', 'NEM', 'NKE', 'NOC', 'NSC', 'NUE', 'NVDA', 'OIH', 'ORCL', 'OXY', 'PAYX', 'PCG', 'PEG', 'PEP', 'PFE', 'PG', 'PGR', 'PH', 'PNW', 'PPG', 'PPL', 'PSA', 'QCOM', 'QQQ', 'REGN', 'RF', 'RHI', 'ROK', 'ROST', 'RTX', 'SBUX', 'SCHW', 'SHW', 'SLB', 'SLV', 'SMH', 'SNA', 'SO', 'SPG', 'SPY', 'SRE', 'STT', 'SWK', 'SYY', 'T', 'TAP', 'TGT', 'TJX', 'TMO', 'TRV', 'TSN', 'TXN', 'UNG', 'UNH', 'UNP', 'USB', 'UVXY', 'V', 'VFC', 'VLO', 'VMC', 'VNQ', 'VZ', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOM', 'XOP', 'XRT'], 
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Signal Close",
            "max_one_pos": True,
            "use_perf_rank": True, "perf_window": 21, "perf_logic": "<", "perf_thresh": 15.0,
            "perf_first_instance": False, "perf_lookback": 21, "perf_consecutive": 5,
            "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 85.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0
        },
        "execution": {
            "risk_per_trade": 1000,
            "stop_atr": 3.0,
            "tgt_atr": 8.0,
            "hold_days": 21
        },
        "stats": {
            "grade": "A (Excellent)",
            "win_rate": "63.5%",
            "expectancy": "$289.39",
            "profit_factor": "2.16"
        }
    },
    {
        "id": "3d of 5d perf < 10, sznl > 85",
        "name": "Generated Strategy (A)",
        "description": "Start: 2000-01-01. Universe: All CSV Tickers. Dir: Long. Filter: None. PF: 3.23. SQN: 9.71.",
        "universe_tickers": ['AAPL', 'ABT', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEP', 'AIG', 'ALL', 'AMAT', 'AMD', 'AMGN', 'AMZN', 'AON', 'APD', 'AVGO', 'AXP', 'BA', 'BAC', 'BAX', 'BDX', 'BK', 'BMY', 'C', 'CAG', 'CAT', 'CEF', 'CL', 'CMCSA', 'CMS', 'CNP', 'COP', 'COST', 'CPB', 'CRM', 'CSCO', 'CSX', 'CVS', 'CVX', 'D', 'DE', 'DIA', 'DIS', 'DOV', 'DTE', 'DUK', 'ECL', 'ED', 'EIX', 'EMR', 'EOG', 'ETR', 'EXC', 'F', 'FCX', 'FDX', 'FE', 'GD', 'GE', 'GILD', 'GIS', 'GLD', 'GLW', 'GOOG', 'GPC', 'GS', 'HAL', 'HD', 'HIG', 'HON', 'HPQ', 'HRL', 'HSY', 'HUM', 'IBB', 'IBM', 'IHI', 'INTC', 'IP', 'ITA', 'ITB', 'ITW', 'IWM', 'IYR', 'JNJ', 'JPM', 'K', 'KEY', 'KMB', 'KO', 'KR', 'KRE', 'LEG', 'LIN', 'LLY', 'LMT', 'LOW', 'LUV', 'MAS', 'MCD', 'MDT', 'MET', 'META', 'MMC', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'MU', 'NEE', 'NEM', 'NKE', 'NOC', 'NSC', 'NUE', 'NVDA', 'OIH', 'ORCL', 'OXY', 'PAYX', 'PCG', 'PEG', 'PEP', 'PFE', 'PG', 'PGR', 'PH', 'PNW', 'PPG', 'PPL', 'PSA', 'QCOM', 'QQQ', 'REGN', 'RF', 'RHI', 'ROK', 'ROST', 'RTX', 'SBUX', 'SCHW', 'SHW', 'SLB', 'SLV', 'SMH', 'SNA', 'SO', 'SPG', 'SPY', 'SRE', 'STT', 'SWK', 'SYK', 'SYY', 'T', 'TAP', 'TGT', 'TJX', 'TMO', 'TRV', 'TSN', 'TXN', 'UNG', 'UNH', 'UNP', 'USB', 'USO', 'UVXY', 'V', 'VFC', 'VLO', 'VMC', 'VNQ', 'VZ', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOM', 'XOP', 'XRT'], 
        "settings": {
            "trade_direction": "Long",
            "entry_type": "T+1 Open",
            "max_one_pos": True,
            "max_daily_entries": 1,
            "max_total_positions": 5,
            "use_perf_rank": True, "perf_window": 5, "perf_logic": "<", "perf_thresh": 10.0,
            "perf_first_instance": False, "perf_lookback": 21, "perf_consecutive": 3,
            "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 85.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0
        },
        "execution": {
            "risk_per_trade": 1000,
            "stop_atr": 3.0,
            "tgt_atr": 8.0,
            "hold_days": 21
        },
        "stats": {
            "grade": "A (Excellent)",
            "win_rate": "70.5%",
            "expectancy": "$446.40",
            "profit_factor": "3.23"
        }
    },
]

# -----------------------------------------------------------------------------
# CONSTANTS & SETUP
# -----------------------------------------------------------------------------
CSV_PATH = "seasonal_ranks.csv"

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
    if not t_map: return pd.Series(50.0, index=dates)
    mds = dates.map(lambda x: (x.month, x.day))
    return mds.map(t_map).fillna(50.0)

# -----------------------------------------------------------------------------
# DATA LOGGING (GOOGLE SHEETS)
# -----------------------------------------------------------------------------
def save_signals_to_gsheet(new_dataframe, sheet_name='Trade_Signals_Log'):
    """
    Rounds data, reads existing sheet, merges with new data, removes duplicates 
    (updates old rows), and writes the clean dataset back to Google Sheets.
    """
    if new_dataframe.empty:
        return

    # 1. Prepare New Data
    df_new = new_dataframe.copy()
    
    # --- ROUNDING LOGIC ---
    # Ensure these are floats first, then round to 2 decimals
    cols_to_round = ['Entry', 'Stop', 'Target', 'ATR']
    # We check if columns exist just to be safe
    existing_cols = [c for c in cols_to_round if c in df_new.columns]
    df_new[existing_cols] = df_new[existing_cols].astype(float).round(2)
    # ----------------------

    # Ensure Date is string for accurate comparison
    df_new['Date'] = df_new['Date'].astype(str) 
    
    # Add/Update Scan Timestamp
    df_new["Scan_Timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Reorder columns to ensure Timestamp is first
    cols = ['Scan_Timestamp'] + [c for c in df_new.columns if c != 'Scan_Timestamp']
    df_new = df_new[cols]

    try:
        # 2. Authenticate
        if "gcp_service_account" in st.secrets:
            creds_dict = st.secrets["gcp_service_account"]
            gc = gspread.service_account_from_dict(creds_dict)
        else:
            gc = gspread.service_account(filename='credentials.json')

        sh = gc.open(sheet_name)
        worksheet = sh.sheet1 
        
        # 3. Get Existing Data
        existing_data = worksheet.get_all_values()
        
        if existing_data:
            # Convert to DataFrame (First row is header)
            headers = existing_data[0]
            df_existing = pd.DataFrame(existing_data[1:], columns=headers)
        else:
            df_existing = pd.DataFrame()

        # 4. Merge & Deduplicate
        if not df_existing.empty:
            # Align columns: Ensure existing DF has same columns as new DF
            # (This handles cases where you might add new columns to your strat later)
            df_existing = df_existing.reindex(columns=df_new.columns)
            
            # Combine: Old on top, New on bottom
            combined = pd.concat([df_existing, df_new])
        else:
            combined = df_new

        # 5. THE UPSERT: Drop Duplicates
        # We define a "Unique Signal" as same Ticker + Date + Strategy_ID
        # keep='last' means we keep the new version we just generated
        combined = combined.drop_duplicates(
            subset=['Ticker', 'Date', 'Strategy_ID'], 
            keep='last'
        )
        
        # 6. Write Back
        worksheet.clear()
        
        # Add headers back
        data_to_write = [combined.columns.tolist()] + combined.astype(str).values.tolist()
        
        worksheet.update(values=data_to_write)
        
        st.toast(f"✅ Synced! Sheet now has {len(combined)} total rows.")
        
    except FileNotFoundError:
        st.error("❌ credentials.json not found.")
    except Exception as e:
        st.error(f"❌ Google Sheet Error: {e}")

# -----------------------------------------------------------------------------
# ENGINE
# -----------------------------------------------------------------------------

def calculate_indicators(df, sznl_map, ticker, spy_series=None):
    df = df.copy()
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None: df.index = df.index.tz_localize(None)
    
    # SMA 200 (Trend)
    df['SMA200'] = df['Close'].rolling(200).mean()

    # Perf Ranks
    for window in [5, 10, 21]:
        df[f'ret_{window}d'] = df['Close'].pct_change(window)
        # Min periods 50 ensures we get a rank if we downloaded 400 days
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
    
    # Volume Rank (10d Relative)
    vol_ma_10 = df['Volume'].rolling(10).mean()
    df['vol_ratio_10d'] = vol_ma_10 / vol_ma
    df['vol_ratio_10d_rank'] = df['vol_ratio_10d'].expanding(min_periods=50).rank(pct=True) * 100.0

    # Age
    if not df.empty:
        start_ts = df.index[0]
        df['age_years'] = (df.index - start_ts).days / 365.25
    else:
        df['age_years'] = 0.0

    # SPY Regime
    if spy_series is not None:
        df['SPY_Above_SMA200'] = spy_series.reindex(df.index, method='ffill').fillna(False)
        
    return df

def check_signal(df, params):
    """
    Checks conditions on the DataFrame. 
    Returns True/False for the LAST ROW only.
    """
    # We calculate conditions on the whole DF to handle 'consecutive' logic
    last_idx = df.index[-1]
    last_row = df.iloc[-1]
    
    # 1. Liquidity Gates (Check last row only)
    if last_row['Close'] < params.get('min_price', 0): return False
    if last_row['vol_ma'] < params.get('min_vol', 0): return False
    if last_row['age_years'] < params.get('min_age', 0): return False
    if last_row['age_years'] > params.get('max_age', 100): return False

    # 2. Trend Filter
    trend_opt = params.get('trend_filter', 'None')
    # Long Logic
    if trend_opt == "Price > 200 SMA":
        if not (last_row['Close'] > last_row['SMA200']): return False
    elif trend_opt == "Price > Rising 200 SMA":
        prev_row = df.iloc[-2]
        if not ((last_row['Close'] > last_row['SMA200']) and (last_row['SMA200'] > prev_row['SMA200'])): return False
    elif trend_opt == "SPY > 200 SMA":
        if 'SPY_Above_SMA200' in df.columns and not last_row['SPY_Above_SMA200']: return False
    # Short Logic
    elif trend_opt == "Price < 200 SMA":
        if not (last_row['Close'] < last_row['SMA200']): return False
    elif trend_opt == "Price < Falling 200 SMA":
        prev_row = df.iloc[-2]
        if not ((last_row['Close'] < last_row['SMA200']) and (last_row['SMA200'] < prev_row['SMA200'])): return False
    elif trend_opt == "SPY < 200 SMA":
        if 'SPY_Above_SMA200' in df.columns and last_row['SPY_Above_SMA200']: return False

    # 3. Perf Rank (with Consecutive logic)
    if params['use_perf_rank']:
        col = f"rank_ret_{params['perf_window']}d"
        # Calc raw condition for whole column
        if params['perf_logic'] == '<': 
            raw_cond = df[col] < params['perf_thresh']
        else: 
            raw_cond = df[col] > params['perf_thresh']
            
        consec = params.get('perf_consecutive', 1)
        if consec > 1:
            # Rolling sum of Trues must equal window size
            persist_cond = raw_cond.rolling(consec).sum() == consec
            if not persist_cond.iloc[-1]: return False
        else:
            if not raw_cond.iloc[-1]: return False

    # 4. Seasonal
    if params['use_sznl']:
        if params['sznl_logic'] == '<':
            if not (last_row['Sznl'] < params['sznl_thresh']): return False
        else:
            if not (last_row['Sznl'] > params['sznl_thresh']): return False

    # 5. 52w
    if params['use_52w']:
        if params['52w_type'] == 'New 52w High':
            if not last_row['is_52w_high']: return False
        else:
            if not last_row['is_52w_low']: return False

    # 6. Volume Spike
    if params['use_vol']:
        if not (last_row['vol_ratio'] > params['vol_thresh']): return False

    # 7. Volume Rank
    if params.get('use_vol_rank'):
        val = last_row['vol_ratio_10d_rank']
        if params['vol_rank_logic'] == '<':
            if not (val < params['vol_rank_thresh']): return False
        else:
            if not (val > params['vol_rank_thresh']): return False
        
    return True

# -----------------------------------------------------------------------------
# MAIN APP
# -----------------------------------------------------------------------------

def main():
    st.set_page_config(layout="wide", page_title="Production Strategy Screener")
    st.title("⚡ Daily Strategy Screener")
    st.markdown("---")
    
    sznl_map = load_seasonal_map()
    
    if st.button("Run All Strategies", type="primary", use_container_width=True):
        
        st.info(f"Scanning {len(STRATEGY_BOOK)} strategies against current market data...")
        
        # 1. Consolidate Tickers
        all_tickers = set()
        needs_spy = False
        for strat in STRATEGY_BOOK:
            all_tickers.update(strat['universe_tickers'])
            if "SPY" in strat['settings'].get('trend_filter', ''):
                needs_spy = True
        
        all_tickers = list(all_tickers)
        if needs_spy and "SPY" not in all_tickers:
            all_tickers.append("SPY")
        
        # 2. Download Data (400 days for indicators)
        start_date = datetime.date.today() - datetime.timedelta(days=400)
        try:
            raw_data = yf.download(all_tickers, start=start_date, group_by='ticker', progress=False, threads=True)
        except Exception as e:
            st.error(f"Data download failed: {e}")
            return

        # 3. Process SPY Regime if needed
        spy_series = None
        if needs_spy:
            try:
                if len(all_tickers) > 1 and "SPY" in raw_data.columns.levels[0]:
                    spy_df = raw_data["SPY"].copy()
                elif len(all_tickers) == 1:
                    spy_df = raw_data.copy()
                
                # Flatten cols if needed
                if isinstance(spy_df.columns, pd.MultiIndex):
                    spy_df.columns = [c if isinstance(c, str) else c[0] for c in spy_df.columns]
                
                spy_df['SMA200'] = spy_df['Close'].rolling(200).mean()
                spy_series = spy_df['Close'] > spy_df['SMA200']
            except: pass

        # 4. Iterate Strategies
        for strat in STRATEGY_BOOK:
            
            with st.expander(f"Strategy: {strat['name']} (Grade: {strat['stats']['grade']})", expanded=True):
                
                # Header Stats
                s1, s2, s3, s4, s5 = st.columns(5)
                s1.metric("Win Rate", strat['stats']['win_rate'])
                s2.metric("Expectancy", strat['stats']['expectancy'])
                s3.metric("Profit Factor", strat['stats']['profit_factor'])
                s4.metric("Direction", strat['settings'].get('trade_direction', 'Long'))
                s5.metric("Risk Unit", f"${strat['execution']['risk_per_trade']}")
                
                st.caption(strat['description'])
                
                signals = []
                
                for ticker in strat['universe_tickers']:
                    try:
                        # Extract Ticker DF
                        if len(all_tickers) > 1:
                            if ticker not in raw_data.columns.levels[0]: continue
                            df = raw_data[ticker].copy()
                        else:
                            df = raw_data.copy()

                        # Clean columns
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = [c if isinstance(c, str) else c[0] for c in df.columns]

                        df = df.dropna(subset=['Close'])
                        if len(df) < 250: continue # Need history for ranks
                        
                        # Calc Indicators
                        df = calculate_indicators(df, sznl_map, ticker, spy_series)
                        
                        # Check Logic (Pass whole DF for rolling checks)
                        if check_signal(df, strat['settings']):
                            
                            last_row = df.iloc[-1]
                            atr = last_row['ATR']
                            risk = strat['execution']['risk_per_trade']
                            entry = last_row['Close']
                            direction = strat['settings'].get('trade_direction', 'Long')
                            
                            # Calc Stops/Targets based on Direction
                            stop_atr = strat['execution']['stop_atr']
                            tgt_atr = strat['execution']['tgt_atr']
                            
                            if direction == 'Long':
                                stop_price = entry - (atr * stop_atr)
                                tgt_price = entry + (atr * tgt_atr)
                                dist = entry - stop_price
                                action = "BUY"
                            else: # Short
                                stop_price = entry + (atr * stop_atr)
                                tgt_price = entry - (atr * tgt_atr)
                                dist = stop_price - entry
                                action = "SELL SHORT"
                            
                            shares = int(risk / dist) if dist > 0 else 0
                            exit_date = (last_row.name + BusinessDay(strat['execution']['hold_days'])).date()
                            
                            signals.append({
                                "Strategy_ID": strat['id'], # Added to log which strategy triggered
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
                
                # Output
                if signals:
                    st.success(f"✅ Found {len(signals)} Actionable Signals")
                    sig_df = pd.DataFrame(signals)
                    
                    # -----------------------------------------------
                    # SAVE TO GOOGLE SHEETS
                    # -----------------------------------------------
                    save_signals_to_gsheet(sig_df, sheet_name='Trade_Signals_Log')
                    # -----------------------------------------------

                    # For Display, format numbers nicely but don't save the formatted strings
                    st.dataframe(
                        sig_df.style.format({
                            "Entry": "${:.2f}", "Stop": "${:.2f}", "Target": "${:.2f}", "ATR": "{:.2f}"
                        }), use_container_width=True
                    )
                    
                    clip_text = ""
                    for s in signals:
                        clip_text += f"{s['Action']} {s['Shares']} {s['Ticker']} @ MKT. Stop: {s['Stop']:.2f}. Target: {s['Target']:.2f}. Exit Date: {s['Time Exit']}.\n"
                    st.text_area("Execution Clipboard", clip_text, height=100)
                    
                else:
                    st.info("No signals found for this strategy today.")

if __name__ == "__main__":
    main()
