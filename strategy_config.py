# strategy_config.py
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# CRITICAL WARNING FOR AI AGENTS & DEVELOPERS
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 1. Strategy Names are KEYS. If you change a 'name' field below (e.g., 
#    "Overbot Vol Spike"), you MUST check `daily_scan.py`. That script uses 
#    string matching on these names to apply custom risk multipliers.
#    Renaming without updating `daily_scan.py` will break risk sizing.
#
# 2. This file is updated via MANUAL COPY-PASTE from the Backtester UI.
#    Do not change the schema of the strategy dictionaries, or the copy-paste
#    workflow will fail.
#
# 3. SCHEMA UPDATE (Phase 1): Added 'setup' and 'exit_summary' blocks for
#    email clarity. The 'description' field is now deprecated but kept for
#    backwards compatibility.
#
# 4. TICKER UNIVERSES: Shared ticker lists are defined at the top of this file.
#    Strategies reference these by variable name (e.g., LIQUID_UNIVERSE).
#    To add/remove tickers globally, edit the universe definitions below.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

# ============================================
# ACCOUNT CONFIGURATION
# ============================================
ACCOUNT_VALUE = 750000  # Adjust this to your current account size

# Global aggregate daily risk cap (bps of account) across all strategies' signals.
# If total new-risk from today's staged signals exceeds this, ALL signals are
# proportionally scaled down so aggregate == cap. Applied in daily_scan.py,
# local_overflow_scan.py, and the backtester (strat_backtester.py).
DAILY_RISK_CAP_BPS = 150

# ============================================
# TICKER UNIVERSES
# ============================================
# Define shared ticker lists here. Strategies reference these by variable name.
# This eliminates duplication and makes global ticker changes easy.

# Core index ETFs (5 tickers)
INDEX_ETFS = ['DIA', 'IWM', 'QQQ', 'SMH', 'SPY']

# Sector + Index ETFs for rotation strategies (26 tickers)
SECTOR_INDEX_ETFS = [
    'DIA', 'IBB', 'IHI', 'ITA', 'ITB', 'IWM', 'IYR', 'KRE', 'QQQ', 'SMH',
    'SPY', 'VNQ', 'XBI', 'XHB', 'XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK',
    'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XRT'
]

# Main liquid universe - large caps + sector ETFs + indices (190 tickers)
LIQUID_UNIVERSE = [
    'AAPL', 'ABT', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEP', 'AIG', 'ALL',
    'AMAT', 'AMD', 'AMGN', 'AMZN', 'AON', 'APD', 'AVGO', 'AXP', 'BA', 'BAC',
    'BAX', 'BDX', 'BK', 'BMY', 'C', 'CAG', 'CAT', 'CL', 'CMCSA', 'CMS',
    'CNP', 'COP', 'COST', 'CPB', 'CRM', 'CSCO', 'CSX', 'CVS', 'CVX', 'D',
    'DE', 'DIA', 'DIS', 'DOV', 'DTE', 'DUK', 'ECL', 'ED', 'EIX', 'EMR',
    'EOG', 'ETR', 'EXC', 'F', 'FCX', 'FDX', 'FE', 'GD', 'GE', 'GILD',
    'GIS', 'GLW', 'GOOG', 'GPC', 'GS', 'HAL', 'HD', 'HIG', 'HON', 'HPQ',
    'HRL', 'HSY', 'HUM', 'IBB', 'IBM', 'IHI', 'INTC', 'IP', 'ITA', 'ITB',
    'ITW', 'IWM', 'IYR', 'JNJ', 'JPM', 'KEY', 'KMB', 'KO', 'KR',
    'KRE', 'LEG', 'LIN', 'LLY', 'LMT', 'LOW', 'LUV', 'MAS', 'MCD', 'MDT',
    'MET', 'META', 'MMC', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'MU', 'NEE',
    'NEM', 'NKE', 'NOC', 'NSC', 'NUE', 'NVDA', 'ORCL', 'OXY', 'PAYX', 'PCG',
    'PEG', 'PEP', 'PFE', 'PG', 'PGR', 'PH', 'PNW', 'PPG', 'PPL', 'PSA',
    'QCOM', 'QQQ', 'REGN', 'RF', 'RHI', 'ROK', 'ROST', 'RTX', 'SBUX', 'SCHW',
    'SHW', 'SLB', 'SMH', 'SNA', 'SO', 'SPG', 'SPY', 'SRE', 'STT', 'SWK',
    'SYK', 'SYY', 'T', 'TAP', 'TGT', 'TJX', 'TMO', 'TRV', 'TSN', 'TXN',
    'UNH', 'UNP', 'USB', 'V', 'VFC', 'VLO', 'VMC', 'VNQ', 'VZ', 'WFC',
    'WHR', 'WM', 'WMB', 'WMT', 'XBI', 'XHB', 'XLB', 'XLE', 'XLF', 'XLI',
    'XLK', 'XLP', 'XLU', 'XLV', 'XLY', 'XME', 'XOM', 'XRT', '^GSPC', '^NDX'
]

# Liquid universe without index tickers - for strategies that shouldn't trade indices (188 tickers)
LIQUID_NO_INDEX = [t for t in LIQUID_UNIVERSE if t not in ['^GSPC', '^NDX']]

# Liquid universe + commodity ETFs for broader coverage (198 tickers)
LIQUID_PLUS_COMMODITIES = LIQUID_UNIVERSE + ['CEF', 'GLD', 'OIH', 'SLV', 'UNG', 'USO', 'UVXY', 'XOP']

# 3x Leveraged ETFs — broad + sector equities, bonds, commodities (bull + bear)
# Must stay in sync with LEV3X_ALL in pages/backtester.py
LEV3X_ALL = [
    # Broad equity bull
    'SPXL', 'TQQQ', 'UDOW', 'TNA', 'MIDU',
    # Broad equity bear
    'SPXS', 'SQQQ', 'SDOW', 'TZA',
    # Sector equity bull
    'SOXL', 'FAS', 'TECL', 'LABU', 'CURE', 'ERX', 'DPST',
    'DRN', 'NAIL', 'RETL', 'WEBL', 'DFEN', 'YINN', 'BRZU', 'EDC', 'MEXX',
    # Sector equity bear
    'SOXS', 'FAZ', 'TECS', 'LABD', 'ERY', 'DRV', 'WEBS', 'YANG', 'EDZ',
    # Bonds
    'TMF', 'TMV',
    # Commodities
    'NUGT', 'JNUG', 'GUSH', 'DUST', 'JDST', 'DRIP',
]

# All CSV tickers from sznl_ranks.csv (~1062 tickers)
import os as _os, pandas as _pd
_csv_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'sznl_ranks.csv')
try:
    CSV_UNIVERSE = sorted(_pd.read_csv(_csv_path)['ticker'].unique().tolist())
except Exception:
    CSV_UNIVERSE = LIQUID_UNIVERSE  # fallback

# ============================================
# STRATEGY DEFINITIONS
# ============================================
# Note: risk_bps replaces fixed dollar risk (100 bps = 1% of account)
# Examples at $750k: 10 bps = $750, 20 bps = $1500, 35 bps = $2625

_STRATEGY_BOOK_RAW = [
    {
        "id": "5d <50, 21d > 50 3x, lower 20% range, sznl > 33, 5d time stop",
        "name": "Weak Close Reversion",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Swing",
            "thesis": "Oversold bounce in names with relative strength and positive seasonality",
            "key_filters": [
                "5D rank < 50th %ile (short-term weakness)",
                "21D rank > 50th %ile for 3 consecutive days (underlying strength)",
                "Close in lower 20% of daily range (weak close)",
                "Ticker seasonal rank > 33 (not fighting seasonality)"
            ]
        },
        "exit_summary": {
            "primary_exit": "5-day time stop",
            "stop_logic": "2.0 ATR below entry",
            "target_logic": "8.0 ATR above entry",
            "notes": None
        },
        "description": "Start: 2000-01-01. Universe: Indices. Dir: Long. Filter: None. PF: 2.51. SQN: 8.88.",
        "universe_tickers": INDEX_ETFS,
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Signal Close",
            "max_one_pos": False,
            "allow_same_day_reentry": False,
            "max_daily_entries": 30,
            "max_total_positions": 50,
            "perf_filters": [{'window': 5, 'logic': '<', 'thresh': 50.0, 'consecutive': 1}, {'window': 21, 'logic': '>', 'thresh': 50.0, 'consecutive': 3}],
            "perf_first_instance": False, "perf_lookback": 21,
            "ma_consec_filters": [],
            "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 33.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_market_sznl": False, "market_sznl_logic": "<", "market_sznl_thresh": 15.0,
            "market_ticker": "^GSPC",
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21, "52w_lag": 0,
            "exclude_52w_high": False,
            "breakout_mode": "None",
            "use_range_filter": True, "range_min": 0, "range_max": 20,
            "use_dow_filter": True, "allowed_days": [0, 1, 2, 3, 4],
            "allowed_cycles": [1, 3, 0],
            "use_vix_filter": False, "vix_min": 0.0, "vix_max": 20.0,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 15.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0,
            "min_atr_pct": 0.2, "max_atr_pct": 10.0,
            "entry_conf_bps": 0,
            "use_ma_dist_filter": False, "dist_ma_type": "SMA 10", "dist_logic": "Greater Than (>)", "dist_min": 0.0, "dist_max": 2.0,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": ">", "gap_thresh": 3,
            "use_acc_count_filter": False, "acc_count_window": 21, "acc_count_logic": ">", "acc_count_thresh": 3,
            "use_dist_count_filter": False, "dist_count_window": 21, "dist_count_logic": "<", "dist_count_thresh": 3
        },
        "execution": {"risk_bps": 35, "slippage_bps": 2, "stop_atr": 2.0, "tgt_atr": 8.0, "hold_days": 5,"use_stop_loss": False, "use_take_profit": False},
        "stats": {"grade": "A (Excellent)", "win_rate": "66.6%", "expectancy": "0.28r", "profit_factor": "2.51"}
    },
    {
        "id": "Indx sznl > 85, 21dr < 15 (add on additional sigs)",
        "name": "Index Seasonals",
        "setup": {
            "type": "Seasonal",
            "timeframe": "Position",
            "thesis": "Buying index ETFs during historically strong seasonal windows when oversold",
            "key_filters": [
                "Ticker seasonal rank > 85 (strong seasonal tailwind)",
                "21D rank < 15th %ile (oversold entry)",
                "Allows adding to position on repeated signals"
            ]
        },
        "exit_summary": {
            "primary_exit": "21-day time stop",
            "stop_logic": "2.0 ATR below entry",
            "target_logic": "8.0 ATR above entry",
            "notes": "Allows multiple positions per ticker (pyramiding)"
        },
        "description": "Start: 2000-01-01. Universe: Indices. Dir: Long. Filter: None. PF: 4.51. SQN: 4.85.",
        "universe_tickers": INDEX_ETFS,
        "settings": {
            "trade_direction": "Long",
            "entry_type": "T+1 Open",
            "max_one_pos": False,
            "max_daily_entries": 5,
            "max_total_positions": 10,
            "use_perf_rank": True, "perf_window": 21, "perf_logic": "<", "perf_thresh": 15.0,
            "perf_first_instance": False, "perf_lookback": 21, "perf_consecutive": 1,
            "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 85.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0
        },
        "execution": {"risk_bps": 33, "stop_atr": 2, "tgt_atr": 8.0, "hold_days": 21,"use_stop_loss": False, "use_take_profit": False},
        "stats": {"grade": "A (Excellent)", "win_rate": "64.1%", "expectancy": "0.47r", "profit_factor": "4.51"}
    },
    {
        "id": "5+10+21d<15, SPX sznl > 20, lower 20% range, 2d time stop 1.5 atr tgt",
        "name": "Deep Oversold Weak Close",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Overnight",
            "thesis": "Extreme oversold bounce with weak close, filtered by uptrend and market seasonality",
            "key_filters": [
                "5D + 10D + 21D ranks ALL < 15th %ile (deeply oversold)",
                "Market seasonal > 20 (not fighting macro headwind)",
                "Close in lower 20% of daily range (capitulation close)",
                "Price > Rising 200 SMA (uptrend filter)",
                "ATR% > 3% (volatile names only, exempt: SPY/QQQ/IWM/DIA)",
                "No Tuesday signals"
            ]
        },
        "exit_summary": {
            "primary_exit": "2-day time stop (quick bounce or cut)",
            "stop_logic": "1.0 ATR below entry",
            "target_logic": "1.5 ATR above entry",
            "notes": "Tight stop/target for quick mean reversion"
        },
        "description": "Start: 2000-01-01. Universe: All CSV Tickers. Dir: Long. Filter: Price > 200 SMA. PF: 2.20. SQN: 7.35.",
        "universe_tickers": LIQUID_UNIVERSE,
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Signal Close",
            "max_one_pos": False,
            "allow_same_day_reentry": False,
            "max_daily_entries": 2,
            "max_total_positions": 10,
            "perf_filters": [{'window': 5, 'logic': '<', 'thresh': 15.0, 'consecutive': 1}, {'window': 10, 'logic': '<', 'thresh': 15.0, 'consecutive': 1}, {'window': 21, 'logic': '<', 'thresh': 15.0, 'consecutive': 1}],
            "perf_first_instance": False, "perf_lookback": 21,
            "use_sznl": False, "sznl_logic": "<", "sznl_thresh": 15.0, "sznl_first_instance": True, "sznl_lookback": 21,
            "use_market_sznl": True, "market_sznl_logic": ">", "market_sznl_thresh": 20.0,
            "market_ticker": "^GSPC",
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21, "52w_lag": 1,
            "breakout_mode": "None",
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": ">", "vol_rank_thresh": 75.0,
            "trend_filter": "Price > Rising 200 SMA",
            "use_dow_filter": True, "allowed_days": [0, 2, 3, 4],
            "use_range_filter": True, "range_min": 0, "range_max": 20,
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0,
            "min_atr_pct": 3.0, "max_atr_pct": 10.0,
            "entry_conf_bps": 0,
            "use_ma_dist_filter": False, "dist_ma_type": "SMA 10", "dist_logic": "Greater Than (>)", "dist_min": 0.0, "dist_max": 2.0,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": ">", "gap_thresh": 3,
            "use_acc_count_filter": False, "acc_count_window": 21, "acc_count_logic": ">", "acc_count_thresh": 3,
            "use_dist_count_filter": False, "dist_count_window": 21, "dist_count_logic": ">", "dist_count_thresh": 3
        },
        "execution": {"risk_bps": 40, "slippage_bps": 2, "stop_atr": 1.0, "tgt_atr": 1.5, "hold_days": 2,"use_stop_loss": False, "use_take_profit": True},
        "stats": {"grade": "A (Excellent)", "win_rate": "67.1%", "expectancy": "0.30r", "profit_factor": "2.20"}
    },
    {
        "id": "5+10+21d > 85%ile, 0 acc days in last 21, sell t+1 open + 0.5 ATR 10d time stop",
        "name": "No Accumulation Days",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Swing",
            "thesis": "Fading overbought names with zero institutional accumulation and weak seasonality",
            "key_filters": [
                "5D + 10D + 21D ranks ALL > 85th %ile (extremely overbought)",
                "ZERO accumulation days in last 21 (no institutional support)",
                "At least 1 distribution day in last 21",
                "Ticker seasonal < 33 (fighting seasonal headwind)"
            ]
        },
        "exit_summary": {
            "primary_exit": "10-day time stop",
            "stop_logic": "2.0 ATR above entry (short)",
            "target_logic": "5.0 ATR below entry (short)",
            "notes": "Limit entry at Open + 0.5 ATR for better fill"
        },
        "description": "Start: 2000-01-01. Universe: All CSV Tickers. Dir: Short. Filter: None. PF: 5.15. SQN: 3.82.",
        "universe_tickers": LIQUID_UNIVERSE,
        "settings": {
            "trade_direction": "Short",
            "entry_type": "Limit (Open +/- 0.5 ATR)",
            "max_one_pos": False,
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
            "min_atr_pct": 0.0, "max_atr_pct": 10.0,
            "entry_conf_bps": 0,
            "use_ma_dist_filter": False, "dist_ma_type": "SMA 10", "dist_logic": "Greater Than (>)", "dist_min": 0.0, "dist_max": 2.0,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": ">", "gap_thresh": 3,
            "use_acc_count_filter": True, "acc_count_window": 21, "acc_count_logic": "=", "acc_count_thresh": 0,
            "use_dist_count_filter": True, "dist_count_window": 21, "dist_count_logic": ">", "dist_count_thresh": 0
        },
        "execution": {"risk_bps": 20, "slippage_bps": 5, "stop_atr": 2.0, "tgt_atr": 5.0, "hold_days": 10,"use_stop_loss": False, "use_take_profit": False},
        "stats": {"grade": "A (Excellent)", "win_rate": "66.7%", "expectancy": "0.61r", "profit_factor": "5.15"}
    },
    {
        "id": "Sznl > 90, 5d <15 for 3d consec, 5d time stop",
        "name": "Liquid Seasonals (short term)",
        "setup": {
            "type": "Seasonal",
            "timeframe": "Swing",
            "thesis": "Buying oversold names during peak seasonal windows for quick bounce",
            "key_filters": [
                "Ticker seasonal rank > 90 (top decile seasonal strength)",
                "5D rank < 15th %ile for 3 consecutive days (persistent oversold)"
            ]
        },
        "exit_summary": {
            "primary_exit": "5-day time stop",
            "stop_logic": "2.0 ATR below entry",
            "target_logic": "3.0 ATR above entry",
            "notes": None
        },
        "description": "Start: 2000-01-01. Universe: All CSV Tickers. Dir: Long. Filter: None. PF: 2.80. SQN: 4.76.",
        "universe_tickers": LIQUID_NO_INDEX,
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Signal Close",
            "max_one_pos": False,
            "max_daily_entries": 3,
            "max_total_positions": 10,
            "use_perf_rank": True, "perf_window": 5, "perf_logic": "<", "perf_thresh": 15.0,
            "perf_first_instance": False, "perf_lookback": 21, "perf_consecutive": 3,
            "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 90.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0
        },
        "execution": {"risk_bps": 35, "stop_atr": 2.0, "tgt_atr": 3.0, "hold_days": 5,"use_stop_loss": False, "use_take_profit": False},
        "stats": {"grade": "A (Excellent)", "win_rate": "61.1%", "expectancy": "0.32r", "profit_factor": "2.80"}
    },
    {
        "id": "52wh 15d lookback, 1.5x volume, spx sznl > 50, >50%rng > yday high, no midterms",
        "name": "52wh Breakout",
        "setup": {
            "type": "Breakout",
            "timeframe": "Position",
            "thesis": "Momentum continuation after new 52-week high with volume confirmation",
            "key_filters": [
                "New 52-week high (first in 15 days)",
                "Volume > 1.5x 63-day average (institutional participation)",
                "Market seasonal > 50 (favorable macro backdrop)",
                "Close > previous day high (breakout confirmation)",
                "Close in upper 50% of daily range (strong close)",
                "Excludes midterm election years"
            ]
        },
        "exit_summary": {
            "primary_exit": "63-day time stop (quarterly hold)",
            "stop_logic": "3.0 ATR below entry",
            "target_logic": "10.0 ATR above entry",
            "notes": "Persistent limit order at -0.5 ATR for better entry"
        },
        "description": "Start: 2000-01-01. Universe: All CSV Tickers. Dir: Long. Filter: None. PF: 2.14. SQN: 5.71.",
        "universe_tickers": LIQUID_UNIVERSE,
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Limit Order -0.5 ATR Persistent",
            "max_one_pos": False,
            "allow_same_day_reentry": False,
            "max_daily_entries": 2,
            "max_total_positions": 10,
            "perf_filters": [],
            "perf_first_instance": True, "perf_lookback": 21,
            "ma_consec_filters": [],
            "use_sznl": False, "sznl_logic": ">", "sznl_thresh": 65.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_market_sznl": True, "market_sznl_logic": ">", "market_sznl_thresh": 50.0,
            "market_ticker": "^GSPC",
            "use_52w": True, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 15, "52w_lag": 0,
            "exclude_52w_high": False,
            "breakout_mode": "Close > Prev Day High",
            "use_range_filter": True, "range_min": 50, "range_max": 100,
            "use_dow_filter": True, "allowed_days": [0, 1, 2, 3, 4],
            "allowed_cycles": [3, 0, 1],
            "use_vix_filter": False, "vix_min": 0.0, "vix_max": 15.0,
            "use_vol": True, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 15.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 500000,
            "min_age": 0.25, "max_age": 100.0,
            "min_atr_pct": 0.2, "max_atr_pct": 10.0,
            "entry_conf_bps": 0,
            "use_ma_dist_filter": False, "dist_ma_type": "SMA 10", "dist_logic": "Greater Than (>)", "dist_min": 0.0, "dist_max": 2.0,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": ">", "gap_thresh": 3,
            "use_acc_count_filter": False, "acc_count_window": 21, "acc_count_logic": ">", "acc_count_thresh": 3,
            "use_dist_count_filter": False, "dist_count_window": 21, "dist_count_logic": ">", "dist_count_thresh": 3,
            "use_ma_touch": False, "ma_touch_type": "SMA 200", "ma_slope_days": 50, "ma_untested_days": 30
        },
        "execution": {"risk_bps": 10, "slippage_bps": 2, "stop_atr": 3.0, "tgt_atr": 10.0, "hold_days": 63,"use_stop_loss": False, "use_take_profit": False},
        "stats": {"grade": "A (Excellent)", "win_rate": "62.0%", "expectancy": "$138.18", "profit_factor": "2.14"}
    },
    {
        "id": "5+10+21d r <15, 21d 3 consec + sznl > 85, 21d time stop",
        "name": "Liquid Seasonals (1 Month)",
        "setup": {
            "type": "Seasonal",
            "timeframe": "Position",
            "thesis": "Buying deeply oversold names with strong seasonal tailwind for month-long hold",
            "key_filters": [
                "5D + 10D + 21D ranks ALL < 15th %ile (multi-timeframe oversold)",
                "21D < 15th %ile for 3 consecutive days (persistent weakness)",
                "Ticker seasonal rank > 85 (strong seasonal support)",
                "ATR% > 2.5% (volatile enough for mean reversion)"
            ]
        },
        "exit_summary": {
            "primary_exit": "21-day time stop",
            "stop_logic": "3.0 ATR below entry",
            "target_logic": "8.0 ATR above entry",
            "notes": None
        },
        "description": "Start: 2000-01-01. Universe: All CSV Tickers. Dir: Long. Filter: None. PF: 3.28. SQN: 7.41.",
        "universe_tickers": LIQUID_UNIVERSE,
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Signal Close",
            "max_one_pos": False,
            "allow_same_day_reentry": False,
            "max_daily_entries": 2,
            "max_total_positions": 10,
            "perf_filters": [{'window': 5, 'logic': '<', 'thresh': 15.0, 'consecutive': 1}, {'window': 10, 'logic': '<', 'thresh': 15.0, 'consecutive': 1}, {'window': 21, 'logic': '<', 'thresh': 15.0, 'consecutive': 3}],
            "perf_first_instance": False, "perf_lookback": 21,
            "use_sznl": True, "sznl_logic": ">", "sznl_thresh": 85.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_market_sznl": False, "market_sznl_logic": ">", "market_sznl_thresh": 25.0,
            "market_ticker": "^GSPC",
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0,
            "min_atr_pct": 2.5,
            "entry_conf_bps": 0,
            "use_ma_dist_filter": False, "dist_ma_type": "SMA 10", "dist_logic": "Greater Than (>)", "dist_min": 0.0, "dist_max": 2.0,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": ">", "gap_thresh": 3,
            "use_acc_count_filter": False, "acc_count_window": 21, "acc_count_logic": ">", "acc_count_thresh": 3,
            "use_dist_count_filter": False, "dist_count_window": 21, "dist_count_logic": ">", "dist_count_thresh": 3
        },
        "execution": {"risk_bps": 15, "slippage_bps": 5, "stop_atr": 3.0, "tgt_atr": 8.0, "hold_days": 21,"use_stop_loss": False, "use_take_profit": False},
        "stats": {"grade": "A (Excellent)", "win_rate": "65.4%", "expectancy": "0.395r", "profit_factor": "3.28"}
    },
    {
        "id": "5d < 50, 5D ATR sznl > 50, close > 20d, close in 0-20% range, 3 acc > + 3 dist < (21d), Feb-May & Sep-Dec only, Entry: Signal Close, 5d hold",
        "name": "Weak Close Decent Sznls",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Swing",
            "thesis": "Weak-close pullback in sector ETFs with ATR-normalized seasonal tailwind (5D) and accumulation dominance, scoped to the historically strongest months",
            "key_filters": [
                "5D rank < 50th %ile (short-term pullback)",
                "5D ATR seasonal rank > 50th %ile (in favorable seasonal window)",
                "Close in lower 20% of daily range (weak close)",
                "Close > 20 SMA (trend support intact)",
                "Accumulation days > 3 in last 21",
                "Distribution days < 3 in last 21",
                "Only fires in Feb-May and Sep-Dec"
            ]
        },
        "exit_summary": {
            "primary_exit": "5-day time stop",
            "stop_logic": "None (time exit only)",
            "target_logic": "None (time exit only)",
            "notes": None
        },
        "description": "Start: 2000-01-01. Universe: Sector + Index ETFs. 5d hold, MOC entry, month-filtered Feb-May + Sep-Dec.",
        "universe_tickers": SECTOR_INDEX_ETFS,
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Signal Close",
            "max_one_pos": True,
            "allow_same_day_reentry": False,
            "max_daily_entries": 20,
            "max_total_positions": 99,
            "entry_conf_bps": 0,
            "perf_filters": [{'window': 5, 'logic': '<', 'thresh': 50.0, 'thresh_max': 100.0, 'consecutive': 1}],
            "perf_first_instance": False, "perf_lookback": 21,
            "ma_consec_filters": [{'length': 20, 'logic': 'Above', 'consec': 1}],
            "use_sznl": False, "sznl_logic": '<', "sznl_thresh": 15.0,
            "sznl_first_instance": False, "sznl_lookback": 21,
            "use_market_sznl": False, "market_sznl_logic": '<', "market_sznl_thresh": 15.0,
            "market_ticker": "^GSPC",
            "use_52w": False, "52w_type": 'New 52w High', "52w_first_instance": False,
            "52w_lookback": 21, "52w_lag": 0,
            "exclude_52w_high": False,
            "use_ath": False, "ath_type": 'Today is ATH',
            "use_recent_ath": False, "recent_ath_invert": False, "ath_lookback_days": 21,
            "use_recent_52w": False, "recent_52w_invert": False, "recent_52w_lookback": 21,
            "use_recent_52w_low": False, "recent_52w_low_invert": False, "recent_52w_low_lookback": 21,
            "breakout_mode": 'None',
            "require_close_gt_open": False,
            "use_range_filter": True, "range_min": 0, "range_max": 20,
            "use_atr_ret_filter": False, "atr_ret_min": 0.0, "atr_ret_max": 1.0,
            "use_range_atr_filter": False, "range_atr_logic": '>', "range_atr_min": 1.0, "range_atr_max": 3.0,
            "price_action_filters": [],
            "use_ma_dist_filter": False, "dist_ma_type": 'SMA 10', "dist_logic": 'Greater Than (>)', "dist_min": 0.0, "dist_max": 2.0,
            "use_weekly_ma_pullback": False, "wma_type": 'EMA', "wma_period": 8, "wma_min_ext_pct": 30.0, "wma_lookback_months": 6, "wma_touch_logic": 'Low <= MA',
            "vol_gt_prev": False,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": '<', "vol_rank_thresh": 50.0,
            "use_acc_count_filter": True, "acc_count_window": 21, "acc_count_logic": '>', "acc_count_thresh": 3,
            "use_dist_count_filter": True, "dist_count_window": 21, "dist_count_logic": '<', "dist_count_thresh": 3,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": '>', "gap_thresh": 3,
            "trend_filter": 'None',
            "use_vix_filter": False, "vix_min": 0.0, "vix_max": 20.0,
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0,
            "min_atr_pct": 0.2, "max_atr_pct": 10.0,
            "use_dow_filter": False, "allowed_days": [0, 1, 2, 3, 4],
            "use_month_filter": True, "allowed_months": [2, 3, 4, 5, 9, 10, 11, 12],
            "use_ref_ticker_filter": False, "ref_ticker": 'IWM', "ref_filters": [],
            "use_t1_open_filter": False, "t1_open_filters": [],
            "use_xsec_filter": False, "xsec_filters": [],
            "atr_sznl_filters": [{'window': 5, 'logic': '>', 'thresh': 50.0, 'thresh_max': 100.0, 'consecutive': 1}]
        },
        "execution": {"risk_bps": 40, "slippage_bps": 2, "stop_atr": 1.0, "tgt_atr": 8.0, "hold_days": 5, "use_stop_loss": False, "use_take_profit": False},
        "stats": {"grade": "A (Excellent)", "win_rate": "64.8%", "expectancy": "0.49r", "profit_factor": "2.21"}
    },
    {
        "id": "21dr < 15 3 consec, 5dr < 33, 252dr > 50, rel vol < 15, MOC + LOC companion, 20d cooldown",
        "name": "Oversold Low Volume",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Position",
            "thesis": "Buying oversold names during low-volume selloffs, filtered to uptrending names (252d > 50th pctile)",
            "key_filters": [
                "21D rank < 15th %ile for 3 consecutive days (persistent oversold)",
                "5D rank < 33rd %ile (recent weakness)",
                "252D rank > 50th %ile (in longer-term uptrend)",
                "10D volume rank < 15th %ile (low volume = lack of conviction selling)",
                "20 trading day cooldown per ticker"
            ]
        },
        "exit_summary": {
            "primary_exit": "21-day time stop",
            "stop_logic": "3.0 ATR below entry",
            "target_logic": "8.0 ATR above entry",
            "notes": "MOC primary (today's close) + LOC companion (T+1 close if < signal close)"
        },
        "description": "Start: 2000-01-01. Universe: Liquid + commodities. Dir: Long. MOC + LOC companion. 20d cooldown.",
        "universe_tickers": LIQUID_PLUS_COMMODITIES,
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Signal Close",
            "max_one_pos": False,
            "allow_same_day_reentry": False,
            "max_daily_entries": 10,
            "max_total_positions": 20,
            "perf_filters": [
                {'window': 5, 'logic': '<', 'thresh': 33.0, 'consecutive': 1},
                {'window': 21, 'logic': '<', 'thresh': 15.0, 'consecutive': 3},
                {'window': 252, 'logic': '>', 'thresh': 50.0, 'consecutive': 1}
            ],
            "perf_first_instance": False, "perf_lookback": 21,
            "ma_consec_filters": [],
            "use_sznl": False, "sznl_logic": "<", "sznl_thresh": 15.0, "sznl_first_instance": True, "sznl_lookback": 21,
            "use_market_sznl": False, "market_sznl_logic": "<", "market_sznl_thresh": 15.0,
            "market_ticker": "^GSPC",
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21, "52w_lag": 0,
            "exclude_52w_high": False,
            "breakout_mode": "None",
            "use_range_filter": False, "range_min": 0, "range_max": 100,
            "use_dow_filter": False, "allowed_days": [0, 1, 2, 3, 4],
            "use_vix_filter": False, "vix_min": 0.0, "vix_max": 20.0,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": True, "vol_rank_logic": "<", "vol_rank_thresh": 15.0,
            "trend_filter": "None",
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0,
            "min_atr_pct": 0.0, "max_atr_pct": 100.0,
            "entry_conf_bps": 0,
            "use_ma_dist_filter": False, "dist_ma_type": "SMA 10", "dist_logic": "Greater Than (>)", "dist_min": 0.0, "dist_max": 2.0,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": ">", "gap_thresh": 3,
            "use_acc_count_filter": False, "acc_count_window": 21, "acc_count_logic": ">", "acc_count_thresh": 3,
            "use_dist_count_filter": False, "dist_count_window": 21, "dist_count_logic": ">", "dist_count_thresh": 3,
            "use_recent_52w_low": False, "recent_52w_low_invert": True, "recent_52w_low_lookback": 10
        },
        "execution": {"risk_bps": 15, "slippage_bps": 2, "stop_atr": 3.0, "tgt_atr": 8.0, "hold_days": 21,"use_stop_loss": False, "use_take_profit": False,
                      "ladder_multipliers": [0.85, 1.00, 1.15], "loc_companion_multiplier": 0.85},
        "stats": {"grade": "A (Excellent)", "win_rate": "69.0%", "expectancy": "0.48r", "profit_factor": "2.82"}
    },
    {
        "id": "2+5+10+21d > 85, >0 dist day, sell open +0.75 atr, 2 ATR tgt",
        "name": "Overbot Vol Spike",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Overnight",
            "thesis": "Fading multi-horizon overbought names — pure short-term overbought fade (no long-term-leader filter, no seasonal gate)",
            "key_filters": [
                "2D + 5D + 10D + 21D ranks ALL > 85th %ile (extremely overbought)",
                "21D > 85th %ile for 3 consecutive days",
                "At least 1 distribution day in last 21",
                "Today's return > 0.25 ATR (up day)"
            ]
        },
        "exit_summary": {
            "primary_exit": "2-day time stop OR 2.0 ATR target (whichever first)",
            "stop_logic": "None (time/target exit only)",
            "target_logic": "2.0 ATR below entry (short)",
            "notes": "LOC companion retired. Target reduced from 8 ATR → 2 ATR per path-analysis give-back profile."
        },
        "description": "Start: 2000-01-01. Universe: LIQUID_PLUS_COMMODITIES. Dir: Short. Pure multi-horizon overbought fade, no 126d/252d leader filter, no market seasonal filter.",
        "universe_tickers": LIQUID_PLUS_COMMODITIES,
        "settings": {
            "trade_direction": "Short",
            "entry_type": "Limit (Open +/- 0.75 ATR)",
            "max_one_pos": False,
            "allow_same_day_reentry": True,
            "max_daily_entries": 2,
            "max_total_positions": 10,
            "perf_filters": [
                {'window': 2, 'logic': '>', 'thresh': 85.0, 'consecutive': 1},
                {'window': 5, 'logic': '>', 'thresh': 85.0, 'consecutive': 1},
                {'window': 10, 'logic': '>', 'thresh': 85.0, 'consecutive': 1},
                {'window': 21, 'logic': '>', 'thresh': 85.0, 'consecutive': 3},
            ],
            "perf_first_instance": False, "perf_lookback": 21,
            "use_sznl": False, "sznl_logic": ">", "sznl_thresh": 85.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_market_sznl": False, "market_sznl_logic": "<", "market_sznl_thresh": 75.0,
            "market_ticker": "^GSPC",
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21,
            "use_vol": False, "vol_thresh": 1.25,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "trend_filter": "None",
            "use_today_return": True, "return_min": 0.25, "return_max": 100,
            "use_range_filter": False, "range_min": 50, "range_max": 100,
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0,
            "min_atr_pct": 0.0, "max_atr_pct": 100.0,
            "entry_conf_bps": 0,
            "use_ma_dist_filter": False, "dist_ma_type": "SMA 10", "dist_logic": "Greater Than (>)", "dist_min": 0.0, "dist_max": 2.0,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": ">", "gap_thresh": 3,
            "use_acc_count_filter": False, "acc_count_window": 21, "acc_count_logic": ">", "acc_count_thresh": 3,
            "use_dist_count_filter": True, "dist_count_window": 21, "dist_count_logic": ">", "dist_count_thresh": 0,
            "use_xsec_filter": True, "xsec_filters": []
        },
        "execution": {"risk_bps": 25, "slippage_bps": 2, "stop_atr": 1.0, "tgt_atr": 2.0, "hold_days": 2,"use_stop_loss": False, "use_take_profit": True,
                      "ladder_multipliers": [0.85, 1.00, 1.15]},
        "stats": {"grade": "A (Excellent)", "win_rate": "58.0%", "expectancy": "0.28r", "profit_factor": "1.96"}
    },
    {
        "id": "2d+5d+10d+21d < 15%ile, 126d+252d between 65-90, vol > 1.25x (1.0x intraday), 2 ATR tgt, 2d hold",
        "name": "LT Trend ST OS",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Overnight",
            "thesis": "Oversold bounce in long-term uptrenders — but NOT extreme leaders (126D/252D capped at 90th %ile) to avoid buying climactic tops",
            "key_filters": [
                "2D rank < 15th %ile",
                "5D rank < 15th %ile",
                "10D rank < 15th %ile",
                "21D rank < 15th %ile",
                "126D rank between 65-90th %ile",
                "252D rank between 65-90th %ile",
                "Close in 0-25% of daily range",
                "Volume > 1.25x 63-day avg (relaxed to 1.0x for intraday scan runs)",
                "63d dial (10d avg) < 65 (not in extreme fragile regime)"
            ]
        },
        "exit_summary": {
            "primary_exit": "2-day time stop OR 2.0 ATR target (whichever first)",
            "stop_logic": "None (time/target exit only)",
            "target_logic": "2.0 ATR above entry",
            "notes": None
        },
        "description": "Start: 2000-01-01. Universe: LIQUID_PLUS_COMMODITIES. Dir: Long. WR 68.4% / PF 2.91 / Exp 0.40r.",
        "universe_tickers": LIQUID_PLUS_COMMODITIES,
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Signal Close",
            "max_one_pos": False,
            "allow_same_day_reentry": False,
            "max_daily_entries": 20,
            "max_total_positions": 99,
            "entry_conf_bps": 0,
            "perf_filters": [
                {'window': 2, 'logic': '<', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 5, 'logic': '<', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 10, 'logic': '<', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 21, 'logic': '<', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 126, 'logic': 'Between', 'thresh': 65.0, 'thresh_max': 90.0, 'consecutive': 1},
                {'window': 252, 'logic': 'Between', 'thresh': 65.0, 'thresh_max': 90.0, 'consecutive': 1},
            ],
            "perf_atr_filters": [],
            "perf_first_instance": False, "perf_lookback": 21,
            "ma_consec_filters": [],
            "use_sznl": False, "sznl_logic": "<", "sznl_thresh": 15.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_market_sznl": False, "market_sznl_logic": ">", "market_sznl_thresh": 30.0,
            "market_ticker": "^GSPC",
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": False, "52w_lookback": 21, "52w_lag": 0, "exclude_52w_high": False,
            "use_ath": False, "ath_type": "Today is NOT ATH",
            "use_recent_ath": False, "recent_ath_invert": False, "ath_lookback_days": 21,
            "use_recent_52w": False, "recent_52w_invert": False, "recent_52w_lookback": 21,
            "use_recent_52w_low": False, "recent_52w_low_invert": False, "recent_52w_low_lookback": 21,
            "breakout_mode": "None",
            "require_close_gt_open": False,
            "use_range_filter": True, "range_min": 0, "range_max": 25,
            "use_atr_ret_filter": False, "atr_ret_min": -10.0, "atr_ret_max": 0.0,
            "use_range_atr_filter": False, "range_atr_logic": ">", "range_atr_min": 1.0, "range_atr_max": 3.0,
            "price_action_filters": [],
            "use_ma_dist_filter": False, "dist_ma_type": "SMA 10", "dist_logic": "Greater Than (>)", "dist_min": 0.0, "dist_max": 2.0,
            "use_weekly_ma_pullback": False, "wma_type": "EMA", "wma_period": 8,
            "wma_min_ext_pct": 30.0, "wma_lookback_months": 6, "wma_touch_logic": "Low <= MA",
            "vol_gt_prev": False,
            "use_vol": True, "vol_logic": ">", "vol_thresh": 1.25, "vol_thresh_max": 10.0,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "use_acc_count_filter": False, "acc_count_window": 21, "acc_count_logic": ">", "acc_count_thresh": 3,
            "use_dist_count_filter": False, "dist_count_window": 21, "dist_count_logic": ">", "dist_count_thresh": 0,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": ">", "gap_thresh": 3,
            "trend_filter": "None",
            "use_vix_filter": False, "vix_min": 0.0, "vix_max": 20.0,
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0,
            "min_atr_pct": 0.2, "max_atr_pct": 10.0,
            "use_dow_filter": False, "allowed_days": [0, 1, 2, 3, 4],
            "allowed_cycles": [1, 2, 3, 0],
            "excluded_years": [],
            "use_ref_ticker_filter": False, "ref_ticker": "SPY", "ref_filters": [],
            "use_t1_open_filter": False, "t1_open_filters": [],
            "use_xsec_filter": True, "xsec_filters": [],
            "atr_sznl_filters": [],
            "dial_filters": [{'dial': '63d', 'window': 10, 'logic': '<', 'thresh': 65.0}]
        },
        "execution": {
            "risk_bps": 35,
            "risk_per_trade": "[EDIT: calculated from account size]",
            "slippage_bps": 2,
            "stop_atr": 1.0,
            "tgt_atr": 2.0,
            "hold_days": 2,
            "use_stop_loss": False,
            "use_take_profit": True
        },
        "stats": {"grade": "A (Excellent)", "win_rate": "68.4%", "expectancy": "0.40r", "profit_factor": "2.91"}
    },
    {
        "id": "2d+5d+10d+21d < 15%ile, 252d between 50-90, 5D ATR sznl > 90, vol_rank < 65, range 0-25, dial 63d 10ma < 65, Entry: signal close -0.25 ATR GTC, 1.5 ATR tgt, 5d hold",
        "name": "St OS Sznl",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Swing",
            "thesis": "Short-term oversold bounce in long-term-uptrenders-but-not-leaders during top-decile seasonal windows. Persistent limit at signal close - 0.25 ATR catches dips within the holding window for a slight price improvement vs MOC.",
            "key_filters": [
                "2D rank < 15th %ile",
                "5D rank < 15th %ile",
                "10D rank < 15th %ile",
                "21D rank < 15th %ile",
                "252D rank between 50-90th %ile",
                "5D ATR seasonal rank > 90th %ile",
                "10D vol rank < 65th %ile",
                "Close in 0-25% of daily range",
                "63d dial (10d avg) < 65 (not in extreme fragile regime)"
            ]
        },
        "exit_summary": {
            "primary_exit": "5-day time stop OR 1.5 ATR target (whichever first)",
            "stop_logic": "None (time/target exit only)",
            "target_logic": "1.5 ATR above entry",
            "notes": None
        },
        "description": "Short-term oversold + seasonal tailwind. Universe: LIQUID_PLUS_COMMODITIES. 5d hold, MOC entry.",
        "universe_tickers": LIQUID_PLUS_COMMODITIES,
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Limit Order -0.25 ATR (Persistent)",
            "max_one_pos": True,
            "allow_same_day_reentry": False,
            "max_daily_entries": 20,
            "max_total_positions": 99,
            "entry_conf_bps": 0,
            "perf_filters": [
                {'window': 2, 'logic': '<', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 5, 'logic': '<', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 10, 'logic': '<', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 21, 'logic': '<', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 252, 'logic': 'Between', 'thresh': 50.0, 'thresh_max': 90.0, 'consecutive': 1},
            ],
            "perf_first_instance": False,
            "perf_lookback": 21,
            "ma_consec_filters": [],
            "use_sznl": False, "sznl_logic": '<', "sznl_thresh": 15.0,
            "sznl_first_instance": False, "sznl_lookback": 21,
            "use_market_sznl": False, "market_sznl_logic": '<', "market_sznl_thresh": 15.0,
            "market_ticker": "^GSPC",
            "use_52w": False, "52w_type": 'New 52w High', "52w_first_instance": False,
            "52w_lookback": 21, "52w_lag": 0,
            "exclude_52w_high": False,
            "use_ath": False, "ath_type": 'Today is ATH',
            "use_recent_ath": False, "recent_ath_invert": False, "ath_lookback_days": 21,
            "use_recent_52w": False, "recent_52w_invert": False, "recent_52w_lookback": 21,
            "use_recent_52w_low": False, "recent_52w_low_invert": False, "recent_52w_low_lookback": 21,
            "breakout_mode": 'None',
            "require_close_gt_open": False,
            "use_range_filter": True, "range_min": 0, "range_max": 25,
            "use_atr_ret_filter": False, "atr_ret_min": 0.0, "atr_ret_max": 1.0,
            "use_range_atr_filter": False, "range_atr_logic": '>', "range_atr_min": 1.0, "range_atr_max": 3.0,
            "price_action_filters": [],
            "use_ma_dist_filter": False, "dist_ma_type": 'SMA 10', "dist_logic": 'Greater Than (>)', "dist_min": 0.0, "dist_max": 2.0,
            "use_weekly_ma_pullback": False, "wma_type": 'EMA', "wma_period": 8, "wma_min_ext_pct": 30.0, "wma_lookback_months": 6, "wma_touch_logic": 'Low <= MA',
            "vol_gt_prev": False,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": True, "vol_rank_logic": '<', "vol_rank_thresh": 65.0,
            "use_acc_count_filter": False, "acc_count_window": 21, "acc_count_logic": '>', "acc_count_thresh": 3,
            "use_dist_count_filter": False, "dist_count_window": 21, "dist_count_logic": '>', "dist_count_thresh": 3,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": '>', "gap_thresh": 3,
            "trend_filter": 'None',
            "use_vix_filter": False, "vix_min": 0.0, "vix_max": 20.0,
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0,
            "min_atr_pct": 0.2, "max_atr_pct": 10.0,
            "use_dow_filter": False, "allowed_days": [0, 1, 2, 3, 4],
            "use_ref_ticker_filter": False, "ref_ticker": 'IWM', "ref_filters": [],
            "use_t1_open_filter": False, "t1_open_filters": [],
            "use_xsec_filter": False, "xsec_filters": [],
            "atr_sznl_filters": [{'window': 5, 'logic': '>', 'thresh': 90.0, 'thresh_max': 100.0, 'consecutive': 1}],
            "dial_filters": [{'dial': '63d', 'window': 10, 'logic': '<', 'thresh': 65.0}]
        },
        "execution": {
            "risk_bps": 40,
            "risk_per_trade": "[EDIT: calculated from account size]",
            "slippage_bps": 2,
            "stop_atr": 1.0,
            "tgt_atr": 1.5,
            "hold_days": 5,
            "use_stop_loss": False,
            "use_take_profit": True
        },
        "stats": {"grade": "A (Excellent)", "win_rate": "64.5%", "expectancy": "0.45r", "profit_factor": "2.17"}
    },
    {
        "id": "5d < 15%ile+10d < 15%ile+21d < 15%ile+252d > 60%ile, 252D ATR sznl > 90, Entry: Signal Close, 252d hold",
        "name": "LT OS Sznl",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Position",
            "thesis": "Long-term oversold bounce filtered to names in the best 252D seasonal windows (ATR-normalized rank > 90th %ile) and still in longer-term uptrend (252D rank > 60)",
            "key_filters": [
                "5D rank < 15th %ile",
                "10D rank < 15th %ile",
                "21D rank < 15th %ile",
                "252D rank > 60th %ile",
                "252D ATR seasonal rank > 90th %ile"
            ]
        },
        "exit_summary": {
            "primary_exit": "252-day time stop",
            "stop_logic": "None (time exit only)",
            "target_logic": "None (time exit only)",
            "notes": None
        },
        "description": "Long-term oversold + strong long-horizon seasonal. Universe: LIQUID_PLUS_COMMODITIES. 252d hold, MOC entry.",
        "universe_tickers": LIQUID_PLUS_COMMODITIES,
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Signal Close",
            "max_one_pos": True,
            "allow_same_day_reentry": False,
            "max_daily_entries": 20,
            "max_total_positions": 99,
            "entry_conf_bps": 0,
            "perf_filters": [
                {'window': 5, 'logic': '<', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 10, 'logic': '<', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 21, 'logic': '<', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 252, 'logic': '>', 'thresh': 60.0, 'thresh_max': 100.0, 'consecutive': 1},
            ],
            "perf_first_instance": False,
            "perf_lookback": 21,
            "ma_consec_filters": [],
            "use_sznl": False, "sznl_logic": '<', "sznl_thresh": 15.0,
            "sznl_first_instance": False, "sznl_lookback": 21,
            "use_market_sznl": False, "market_sznl_logic": '<', "market_sznl_thresh": 15.0,
            "market_ticker": "^GSPC",
            "use_52w": False, "52w_type": 'New 52w High', "52w_first_instance": False,
            "52w_lookback": 21, "52w_lag": 0,
            "exclude_52w_high": False,
            "use_ath": False, "ath_type": 'Today is ATH',
            "use_recent_ath": False, "recent_ath_invert": False, "ath_lookback_days": 21,
            "use_recent_52w": False, "recent_52w_invert": False, "recent_52w_lookback": 21,
            "use_recent_52w_low": False, "recent_52w_low_invert": False, "recent_52w_low_lookback": 21,
            "breakout_mode": 'None',
            "require_close_gt_open": False,
            "use_range_filter": False, "range_min": 0, "range_max": 100,
            "use_atr_ret_filter": False, "atr_ret_min": 0.0, "atr_ret_max": 1.0,
            "use_range_atr_filter": False, "range_atr_logic": '>', "range_atr_min": 1.0, "range_atr_max": 3.0,
            "price_action_filters": [],
            "use_ma_dist_filter": False, "dist_ma_type": 'SMA 10', "dist_logic": 'Greater Than (>)', "dist_min": 0.0, "dist_max": 2.0,
            "use_weekly_ma_pullback": False, "wma_type": 'EMA', "wma_period": 8, "wma_min_ext_pct": 30.0, "wma_lookback_months": 6, "wma_touch_logic": 'Low <= MA',
            "vol_gt_prev": False,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": False, "vol_rank_logic": '<', "vol_rank_thresh": 50.0,
            "use_acc_count_filter": False, "acc_count_window": 21, "acc_count_logic": '>', "acc_count_thresh": 3,
            "use_dist_count_filter": False, "dist_count_window": 21, "dist_count_logic": '>', "dist_count_thresh": 3,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": '>', "gap_thresh": 3,
            "trend_filter": 'None',
            "use_vix_filter": False, "vix_min": 0.0, "vix_max": 20.0,
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0,
            "min_atr_pct": 0.2, "max_atr_pct": 10.0,
            "use_dow_filter": False, "allowed_days": [0, 1, 2, 3, 4],
            "use_ref_ticker_filter": False, "ref_ticker": 'IWM', "ref_filters": [],
            "use_t1_open_filter": False, "t1_open_filters": [],
            "use_xsec_filter": False, "xsec_filters": [],
            "atr_sznl_filters": [{'window': 252, 'logic': '>', 'thresh': 90.0, 'thresh_max': 100.0, 'consecutive': 1}]
        },
        "execution": {
            "risk_bps": 35,
            "risk_per_trade": "[EDIT: calculated from account size]",
            "slippage_bps": 2,
            "stop_atr": 4.0,
            "tgt_atr": 8.0,
            "hold_days": 252,
            "use_stop_loss": False,
            "use_take_profit": False
        },
        "stats": {"grade": "A (Excellent)", "win_rate": "72.9%", "expectancy": "1.36r", "profit_factor": "4.14"}
    },
    {
        "id": "2d > 85%ile+5d > 85%ile+10d > 85%ile+21d > 85%ile+126d < 65%ile+252d < 65%ile, Entry: Limit (Open +/- 0.5 ATR), 2d hold",
        "name": "3x ETF Overbot Fade",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Overnight",
            "thesis": "Overbought fade on 3x leveraged ETFs that are NOT medium/long-term leaders — pure multi-horizon overbought fade (no volume or range requirement)",
            "key_filters": [
                "2D rank > 85th %ile",
                "5D rank > 85th %ile",
                "10D rank > 85th %ile",
                "21D rank > 85th %ile (3d consecutive)",
                "126D rank < 65th %ile",
                "252D rank < 65th %ile"
            ]
        },
        "exit_summary": {
            "primary_exit": "2-day time stop",
            "stop_logic": "None (time exit only)",
            "target_logic": "None (time exit only)",
            "notes": None
        },
        "description": "Backtest: 2000-01-01 to present. Universe: 3x Leveraged (All) — 42 tickers.",
        "universe_tickers": LEV3X_ALL,
        "settings": {
            "trade_direction": "Short",
            "entry_type": "Limit (Open +/- 0.5 ATR)",
            "max_one_pos": True,
            "allow_same_day_reentry": False,
            "max_daily_entries": 20,
            "max_total_positions": 99,
            "entry_conf_bps": 0,
            "perf_filters": [
                {'window': 2, 'logic': '>', 'thresh': 85.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 5, 'logic': '>', 'thresh': 85.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 10, 'logic': '>', 'thresh': 85.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 21, 'logic': '>', 'thresh': 85.0, 'thresh_max': 100.0, 'consecutive': 3},
                {'window': 126, 'logic': '<', 'thresh': 65.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 252, 'logic': '<', 'thresh': 65.0, 'thresh_max': 100.0, 'consecutive': 1}
            ],
            "perf_first_instance": False,
            "perf_lookback": 21,
            "ma_consec_filters": [],
            "use_sznl": False,
            "sznl_logic": "<",
            "sznl_thresh": 15.0,
            "sznl_first_instance": False,
            "sznl_lookback": 21,
            "use_market_sznl": False,
            "market_sznl_logic": "<",
            "market_sznl_thresh": 15.0,
            "market_ticker": "^GSPC",
            "use_52w": False,
            "52w_type": "New 52w High",
            "52w_first_instance": False,
            "52w_lookback": 21,
            "52w_lag": 0,
            "exclude_52w_high": False,
            "use_ath": False,
            "ath_type": "Today is ATH",
            "use_recent_ath": False,
            "recent_ath_invert": False,
            "ath_lookback_days": 21,
            "use_recent_52w": False,
            "recent_52w_invert": False,
            "recent_52w_lookback": 21,
            "use_recent_52w_low": False,
            "recent_52w_low_invert": False,
            "recent_52w_low_lookback": 21,
            "breakout_mode": "None",
            "require_close_gt_open": False,
            "use_range_filter": False,
            "range_min": 50,
            "range_max": 100,
            "use_atr_ret_filter": False,
            "atr_ret_min": 0.0,
            "atr_ret_max": 1.0,
            "use_range_atr_filter": False,
            "range_atr_logic": ">",
            "range_atr_min": 1.0,
            "range_atr_max": 3.0,
            "price_action_filters": [],
            "use_ma_dist_filter": False,
            "dist_ma_type": "SMA 10",
            "dist_logic": "Greater Than (>)",
            "dist_min": 0.0,
            "dist_max": 2.0,
            "use_weekly_ma_pullback": False,
            "wma_type": "EMA",
            "wma_period": 8,
            "wma_min_ext_pct": 30.0,
            "wma_lookback_months": 6,
            "wma_touch_logic": "Low <= MA",
            "vol_gt_prev": False,
            "use_vol": False,
            "vol_thresh": 1.2,
            "use_vol_rank": False,
            "vol_rank_logic": "<",
            "vol_rank_thresh": 50.0,
            "use_acc_count_filter": False,
            "acc_count_window": 21,
            "acc_count_logic": "=",
            "acc_count_thresh": 0,
            "use_dist_count_filter": False,
            "dist_count_window": 21,
            "dist_count_logic": ">",
            "dist_count_thresh": 3,
            "use_gap_filter": False,
            "gap_lookback": 21,
            "gap_logic": ">",
            "gap_thresh": 3,
            "trend_filter": "None",
            "use_vix_filter": False,
            "vix_min": 0.0,
            "vix_max": 20.0,
            "min_price": 10.0,
            "min_vol": 100000,
            "min_age": 0.25,
            "max_age": 100.0,
            "min_atr_pct": 0.2,
            "max_atr_pct": 10.0,
            "use_dow_filter": False,
            "allowed_days": [0, 1, 2, 3, 4],
            "allowed_cycles": [1, 2, 3, 0],
            "use_ref_ticker_filter": False,
            "ref_ticker": "IWM",
            "ref_filters": [],
            "use_t1_open_filter": False,
            "t1_open_filters": [],
            "use_xsec_filter": False,
            "xsec_filters": [],
            "atr_sznl_filters": []
        },
        "execution": {
            "risk_bps": 40,
            "risk_per_trade": "[EDIT: calculated from account size]",
            "slippage_bps": 2,
            "stop_atr": 1.0,
            "tgt_atr": 8.0,
            "hold_days": 2,
            "use_stop_loss": False,
            "use_take_profit": False
        },
        "stats": {"grade": "A (Excellent)", "win_rate": "79.6%", "expectancy": "0.87r", "profit_factor": "7.58"}
    },
]


# ============================================
# RISK CALCULATION FUNCTIONS
# ============================================
def calculate_dollar_risk(risk_bps, account_value=None):
    """
    Convert basis points to dollar risk.
    
    Args:
        risk_bps: Risk in basis points (100 bps = 1%)
        account_value: Account size in dollars (defaults to ACCOUNT_VALUE)
    
    Returns:
        Dollar risk amount (rounded to nearest dollar)
    """
    if account_value is None:
        account_value = ACCOUNT_VALUE
    return round(account_value * risk_bps / 10000)


def build_strategy_book(account_value=None):
    """
    Build strategy book with calculated dollar risks.
    
    Args:
        account_value: Account size in dollars (defaults to ACCOUNT_VALUE)
    
    Returns:
        List of strategy dicts with risk_per_trade populated
    """
    import copy
    if account_value is None:
        account_value = ACCOUNT_VALUE
    
    strategies = copy.deepcopy(_STRATEGY_BOOK_RAW)
    for strategy in strategies:
        risk_bps = strategy["execution"]["risk_bps"]
        strategy["execution"]["risk_per_trade"] = calculate_dollar_risk(risk_bps, account_value)
    return strategies


def get_strategy_by_name(name, account_value=None):
    """
    Get a single strategy by name with calculated dollar risk.
    
    Args:
        name: Strategy name to find
        account_value: Account size in dollars (defaults to ACCOUNT_VALUE)
    
    Returns:
        Strategy dict or None if not found
    """
    import copy
    if account_value is None:
        account_value = ACCOUNT_VALUE
    
    for strategy in _STRATEGY_BOOK_RAW:
        if strategy["name"] == name:
            strat = copy.deepcopy(strategy)
            strat["execution"]["risk_per_trade"] = calculate_dollar_risk(
                strat["execution"]["risk_bps"], account_value
            )
            return strat
    return None


def list_strategies():
    """List all strategy names and their risk in bps."""
    return [(s["name"], s["execution"]["risk_bps"]) for s in _STRATEGY_BOOK_RAW]


# ============================================
# DEFAULT EXPORT
# ============================================
# Uses ACCOUNT_VALUE at top of file - change that value to adjust all risks
STRATEGY_BOOK = build_strategy_book()
