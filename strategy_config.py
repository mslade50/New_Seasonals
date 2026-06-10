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

# Global risk multiplier — scales every strategy's per-trade risk uniformly
# across the whole book (daily_scan, strat_backtester, daily_portfolio_report).
# Affects: execution['risk_bps'], OVS path1_bps/path2_bps and
# path2_daily_cap_pct, the OLV earnings_size_override.risk_bps, and the
# OVERFLOW_RISK_OVERRIDES in daily_scan / daily_portfolio_report. Set to 1.0
# for prod-default sizing; raise to lever up, lower to throttle the whole book.
GLOBAL_RISK_MULTIPLIER = 1.5

# ============================================
# TICKER UNIVERSES
# ============================================
# Define shared ticker lists here. Strategies reference these by variable name.
# This eliminates duplication and makes global ticker changes easy.

# Core index ETFs (5 tickers)
INDEX_ETFS = ['DIA', 'IWM', 'QQQ', 'SMH', 'SPY']

# Spot index tickers — purer price representation than ETFs (no dividend/tracking drag).
# Strategies use these for signal detection, but daily_scan substitutes the tradeable
# ETF (via SPOT_TO_TRADEABLE) when staging orders, since spot indices aren't tradeable.
INDICES_SPOT = ['^GSPC', '^NDX']

# Mapping from spot index → tradeable ETF used for order staging.
# When a signal fires on a key, daily_scan recomputes calc_df against the value
# (use ETF's ATR, close, etc.) and stages the order on the ETF as a 1:1 alias.
SPOT_TO_TRADEABLE = {'^GSPC': 'SPY', '^NDX': 'QQQ'}

# Cross-strategy risk clamps. When two strategies in `strategies` fire on the
# same signal date AND the same tradeable ticker (compared after
# SPOT_TO_TRADEABLE substitution), each side's per-trade risk is reduced to
# `risk_bps_when_overlapping`. Prevents same-day, same-tradeable double-up
# across structurally similar strategies that would otherwise compete for the
# same dollar of capital under the aggregate daily risk cap.
CROSS_STRATEGY_OVERLAP_OVERRIDES = [
    {
        'strategies': ('Indices Oversold Bounce', 'SPY QQQ MonFri Reversion'),
        'risk_bps_when_overlapping': 20,
    },
]

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
        "id": "252d Between 50-90, New 52w High, Today is ATH, vol > 2.5x, Market > 200 SMA, 63d dial 10ma < 30, Entry: Limit -0.5 ATR Persistent, 63d hold",
        "name": "52wh Breakout",
        "setup": {
            "type": "Breakout",
            "timeframe": "Position",
            "thesis": "Momentum continuation after new highs in uptrending names, gated to calm regimes (63d dial < 30) to avoid buying breakouts into panic-vol chop",
            "key_filters": [
                "252D rank between 50-90th %ile",
                "New 52w High",
                "Today is ATH",
                "Volume > 2.5x 63-day avg",
                "Trend: Market > 200 SMA",
                "63d dial (10d avg) < 30 (calm-regime gate)"
            ]
        },
        "exit_summary": {
            "primary_exit": "Target, Stop, or 63-day time stop (whichever first)",
            "stop_logic": "2.0 ATR below entry",
            "target_logic": "8.0 ATR above entry",
            "notes": None
        },
        "description": "Backtest: 2016-01-01 to present. Universe: LIQUID_UNIVERSE. Dir: Long. WR 47.3% / Exp 0.77r / PF 2.43.",
        "universe_tickers": LIQUID_UNIVERSE,
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Limit Order -0.5 ATR (Persistent)",
            "max_one_pos": True,
            "allow_same_day_reentry": False,
            "entry_conf_bps": 0,
            "perf_filters": [
                {'window': 252, 'logic': 'Between', 'thresh': 50.0, 'thresh_max': 90.0, 'consecutive': 1},
            ],
            "perf_atr_filters": [],
            "perf_first_instance": False, "perf_lookback": 21,
            "ma_consec_filters": [],
            "use_sznl": False, "sznl_logic": "<", "sznl_thresh": 15.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_market_sznl": False, "market_sznl_logic": "<", "market_sznl_thresh": 15.0,
            "market_ticker": "^GSPC",
            "use_52w": True, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 63, "52w_lag": 0,
            "exclude_52w_high": False,
            "use_ath": True, "ath_type": "Today is ATH",
            "use_recent_ath": False, "recent_ath_invert": False, "ath_lookback_days": 21,
            "use_recent_52w": False, "recent_52w_invert": False, "recent_52w_lookback": 21,
            "use_recent_52w_low": False, "recent_52w_low_invert": False, "recent_52w_low_lookback": 21,
            "breakout_mode": "None",
            "require_close_gt_open": False,
            "use_range_filter": False, "range_min": 0, "range_max": 100,
            "use_atr_ret_filter": False, "atr_ret_min": 0.0, "atr_ret_max": 1.0,
            "use_range_atr_filter": False, "range_atr_logic": ">", "range_atr_min": 1.0, "range_atr_max": 3.0,
            "price_action_filters": [],
            "use_ma_dist_filter": False, "dist_ma_type": "SMA 10", "dist_logic": "Greater Than (>)", "dist_min": 0.0, "dist_max": 2.0,
            "use_weekly_ma_pullback": False, "wma_type": "EMA", "wma_period": 8, "wma_min_ext_pct": 30.0, "wma_lookback_months": 6, "wma_touch_logic": "Low <= MA",
            "vol_gt_prev": False,
            "use_vol": True, "vol_logic": ">", "vol_thresh": 2.5, "vol_thresh_max": 10.0,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "use_acc_count_filter": False, "acc_count_window": 21, "acc_count_logic": ">", "acc_count_thresh": 3,
            "use_dist_count_filter": False, "dist_count_window": 21, "dist_count_logic": ">", "dist_count_thresh": 3,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": ">", "gap_thresh": 3,
            "trend_filter": "Market > 200 SMA",
            "use_vix_filter": False, "vix_min": 0.0, "vix_max": 20.0,
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0,
            "min_atr_pct": 0.2, "max_atr_pct": 10.0,
            "use_dow_filter": False, "allowed_days": [0, 1, 2, 3, 4],
            "allowed_cycles": [1, 3, 0, 2],
            "excluded_years": [],
            "use_ref_ticker_filter": False, "ref_ticker": "IWM", "ref_filters": [],
            "use_t1_open_filter": False, "t1_open_filters": [],
            "use_xsec_filter": False, "xsec_filters": [],
            "atr_sznl_filters": [
                {'window': 5,  'logic': '>', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 10, 'logic': '>', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 21, 'logic': '>', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 63, 'logic': '>', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1},
            ],
            "dial_filters": [{'dial': '63d', 'window': 10, 'logic': '<', 'thresh': 30.0}],
            # Earnings-quality + analyst-grade filters. Schema is wired in so
            # the production scanner / portfolio report can adopt these once
            # we validate thresholds in the backtester. All use_*_filter flags
            # are False until then — the runtime treats the entire block as a
            # no-op. Source parquets:
            #   data/earnings_calendar.parquet (derived: eps_surprise_pct,
            #     rev_surprise_pct, eps_yoy, rev_yoy)
            #   data/analyst_grades.parquet     (FMP /stable/grades event log)
            "use_eps_surp_filter": False, "eps_surp_logic": ">", "eps_surp_min": 0.0, "eps_surp_max": 1.0,
            "use_rev_surp_filter": False, "rev_surp_logic": ">", "rev_surp_min": 0.0, "rev_surp_max": 1.0,
            "use_eps_yoy_filter":  False, "eps_yoy_logic":  ">", "eps_yoy_min":  0.0, "eps_yoy_max":  5.0,
            "use_rev_yoy_filter":  False, "rev_yoy_logic":  ">", "rev_yoy_min":  0.0, "rev_yoy_max":  5.0,
            "use_grades_filter":   False, "grades_window_days": 30, "grades_logic": ">=", "grades_thresh": 1,
        },
        "execution": {
            "risk_bps": 35,
            "risk_per_trade": "[EDIT: calculated from account size]",
            "slippage_bps": 2,
            "stop_atr": 2.0,
            "tgt_atr": 8.0,
            "hold_days": 63,
            "use_stop_loss": True,
            "use_take_profit": True
        },
        "stats": {"grade": "A (Excellent)", "win_rate": "47.3%", "expectancy": "0.77r", "profit_factor": "2.43"}
    },
    {'id': '2d Between 5%ile+5d < 50%ile, Entry: Limit (Open +/- 0.25 ATR), 2d hold',
     'name': 'Weak Close Decent Sznls',
     'setup': {'type': 'MeanReversion',
               'timeframe': 'Overnight',
               'thesis': 'Weak-close pullback in trending index/sector ETFs (above 20 + 50 SMA for 10d) with elevated 5D ATR seasonality. Faded the next morning via a limit at open +/- 0.25 ATR for a 2-day mean-reversion play.',
               'key_filters': ['2D rank between 5-50th %ile',
                               '5D rank < 50th %ile',
                               '5D ATR seasonal rank > 65th %ile',
                               'Close above 20 SMA (10d consecutive)',
                               'Close above 50 SMA (10d consecutive)',
                               'Close in 0-15% of daily range']},
     'exit_summary': {'primary_exit': 'Target, Stop, or 2-day time stop',
                      'stop_logic': '1.0 ATR below entry',
                      'target_logic': '2.0 ATR above entry',
                      'notes': None},
     'description': 'Backtest: 2000-01-01 to present. Tested on 28 tickers.',
     'universe_tickers': SECTOR_INDEX_ETFS,
     'settings': {'trade_direction': 'Long',
                  'entry_type': 'Limit (Open +/- 0.25 ATR)',
                  'max_one_pos': True,
                  'allow_same_day_reentry': False,
                  'max_daily_entries': 20,
                  'max_total_positions': 99,
                  'entry_conf_bps': 0,
                  'perf_filters': [{'window': 2, 'logic': 'Between', 'thresh': 5.0, 'thresh_max': 50.0, 'consecutive': 1},
                                   {'window': 5, 'logic': '<', 'thresh': 50.0, 'thresh_max': 100.0, 'consecutive': 1}],
                  'perf_atr_filters': [],
                  'perf_first_instance': False,
                  'perf_lookback': 21,
                  'ma_consec_filters': [{'length': 20, 'logic': 'Above', 'consec': 10},
                                        {'length': 50, 'logic': 'Above', 'consec': 10}],
                  'use_sznl': False,
                  'sznl_logic': '<',
                  'sznl_thresh': 15.0,
                  'sznl_first_instance': False,
                  'sznl_lookback': 21,
                  'use_market_sznl': False,
                  'market_sznl_logic': '<',
                  'market_sznl_thresh': 15.0,
                  'market_ticker': '^GSPC',
                  'use_52w': False,
                  '52w_type': 'New 52w High',
                  '52w_first_instance': False,
                  '52w_lookback': 21,
                  '52w_lag': 0,
                  'exclude_52w_high': False,
                  'use_ath': False,
                  'ath_type': 'Today is ATH',
                  'use_recent_ath': False,
                  'recent_ath_invert': False,
                  'ath_lookback_days': 21,
                  'use_recent_52w': False,
                  'recent_52w_invert': False,
                  'recent_52w_lookback': 21,
                  'use_recent_52w_low': False,
                  'recent_52w_low_invert': False,
                  'recent_52w_low_lookback': 21,
                  'breakout_mode': 'None',
                  'require_close_gt_open': False,
                  'use_range_filter': True,
                  'range_min': 0,
                  'range_max': 15,
                  'use_atr_ret_filter': False,
                  'atr_ret_min': 0.25,
                  'atr_ret_max': 10.0,
                  'use_range_atr_filter': False,
                  'range_atr_logic': '>',
                  'range_atr_min': 1.0,
                  'range_atr_max': 3.0,
                  'use_open_gap_atr_filter': False,
                  'open_gap_atr_logic': '>',
                  'open_gap_atr_min': 0.0,
                  'open_gap_atr_max': 1.0,
                  'price_action_filters': [],
                  'use_ma_dist_filter': False,
                  'dist_ma_type': 'SMA 10',
                  'dist_logic': 'Greater Than (>)',
                  'dist_min': 0.0,
                  'dist_max': 2.0,
                  'use_weekly_ma_pullback': False,
                  'wma_type': 'EMA',
                  'wma_period': 8,
                  'wma_min_ext_pct': 30.0,
                  'wma_lookback_months': 6,
                  'wma_touch_logic': 'Low <= MA',
                  'vol_gt_prev': False,
                  'use_vol': False,
                  'vol_logic': '>',
                  'vol_thresh': 1.5,
                  'vol_thresh_max': 10.0,
                  'use_vol_rank': False,
                  'vol_rank_logic': '<',
                  'vol_rank_thresh': 50.0,
                  'use_acc_count_filter': False,
                  'acc_count_window': 21,
                  'acc_count_logic': '>',
                  'acc_count_thresh': 3,
                  'use_dist_count_filter': False,
                  'dist_count_window': 21,
                  'dist_count_logic': '<',
                  'dist_count_thresh': 3,
                  'use_gap_filter': False,
                  'gap_lookback': 21,
                  'gap_logic': '>',
                  'gap_thresh': 3,
                  'trend_filter': 'None',
                  'use_vix_filter': False,
                  'vix_min': 0.0,
                  'vix_max': 20.0,
                  'min_price': 10.0,
                  'min_vol': 100000,
                  'min_age': 0.0,
                  'max_age': 100.0,
                  'min_atr_pct': 0.0,
                  'max_atr_pct': 10.0,
                  'use_dow_filter': False,
                  'allowed_days': [0, 1, 2, 3, 4],
                  'allowed_cycles': [1, 2, 3, 0],
                  'excluded_years': [],
                  'use_ref_ticker_filter': False,
                  'ref_ticker': 'IWM',
                  'ref_filters': [],
                  'use_t1_open_filter': False,
                  't1_open_filters': [],
                  'use_xsec_filter': False,
                  'xsec_filters': [],
                  'atr_sznl_filters': [{'window': 5, 'logic': '>', 'thresh': 65.0, 'thresh_max': 100.0, 'consecutive': 1}]},
     'execution': {'risk_bps': 35,
                   'risk_per_trade': '[EDIT: calculated from account size]',
                   'slippage_bps': 2,
                   'stop_atr': 1.0,
                   'tgt_atr': 2.0,
                   'hold_days': 2,
                   'use_stop_loss': True,
                   'use_take_profit': True,
                   'use_trailing_stop': False,
                   'trail_atr': 2.0,
                   'trail_anchor': 'Peak High'},
     'stats': {'grade': 'A (Excellent)', 'win_rate': '61.3%', 'expectancy': '0.28r', 'profit_factor': '1.78'}},
    {
        "id": "21dr < 15 3 consec, 5dr < 33, 2dr < 25, 252dr 50-90, rel vol < 15, market > 200 SMA, age >= 5y, pre-earnings -> 10 bps, GTC limit close-0.25 ATR, 10d hold, 1.25 ATR stop, 2.5 ATR tgt",
        "name": "Oversold Low Volume",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Position",
            "thesis": "Buying oversold names during low-volume selloffs in uptrenders (252d 50-90), gated to a market uptrend regime (SPY > 200 SMA) and minimum 5y of trading history. Persistent limit at close - 0.25 ATR lets overflow signals (post-close scan) still get filled overnight or intraday. Pre-earnings signals (signal_date in [-10, 0] TD relative to earnings) are still allowed but sized at 10 bps instead of the default 35 bps to dampen binary-event risk.",
            "key_filters": [
                "21D rank < 15th %ile for 3 consecutive days (persistent oversold)",
                "5D rank < 33rd %ile (recent weakness)",
                "2D rank < 25th %ile (acute today/yesterday weakness)",
                "252D rank between 50-90th %ile (uptrending but not extreme leader)",
                "10D volume rank < 15th %ile (low volume = lack of conviction selling)",
                "Market (SPY) > 200 SMA (uptrend regime)",
                "Min 5 years of price history (mature liquid name)",
                "Pre-earnings (-10..0 TD) signals: sized 10 bps instead of default"
            ]
        },
        "exit_summary": {
            "primary_exit": "10-day time stop OR 2.5 ATR target OR 1.25 ATR stop (whichever first)",
            "stop_logic": "1.25 ATR below entry",
            "target_logic": "2.5 ATR above entry",
            "notes": "Persistent limit at close - 0.25 ATR; GTC for the hold window. No cooldown — consecutive signals on same ticker allowed. Earnings handling: signals 10 TD before through earnings day get sized at 10 bps (vs. default 35 bps liquid / 25 bps overflow); commodity ETFs / indices / futures with no earnings data pass through at default sizing."
        },
        "description": "Start: 2000-01-01. Universe: Liquid + commodities + overflow tier (CSV_UNIVERSE via OVERFLOW_ELIGIBLE). Dir: Long. Entry: limit at close-0.25 ATR (GTC). 10d hold, 2.5 ATR target, 1.25 ATR stop. Liquid 35 bps / overflow 25 bps; pre-earnings window sizes both at 10 bps.",
        "universe_tickers": LIQUID_PLUS_COMMODITIES,
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Limit Order -0.25 ATR (Persistent)",
            "max_one_pos": False,
            "allow_same_day_reentry": False,
            "perf_filters": [
                {'window': 2, 'logic': '<', 'thresh': 25.0, 'consecutive': 1},
                {'window': 5, 'logic': '<', 'thresh': 33.0, 'consecutive': 1},
                {'window': 21, 'logic': '<', 'thresh': 15.0, 'consecutive': 3},
                {'window': 252, 'logic': 'Between', 'thresh': 50.0, 'thresh_max': 90.0, 'consecutive': 1}
            ],
            "atr_sznl_filters": [],
            "perf_first_instance": False, "perf_lookback": 21,
            "ma_consec_filters": [],
            "use_sznl": False, "sznl_logic": "<", "sznl_thresh": 15.0, "sznl_first_instance": True, "sznl_lookback": 21,
            "use_market_sznl": False, "market_sznl_logic": "<", "market_sznl_thresh": 15.0,
            "market_ticker": "SPY",
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21, "52w_lag": 0,
            "exclude_52w_high": False,
            "breakout_mode": "None",
            "use_range_filter": False, "range_min": 0, "range_max": 100,
            "use_dow_filter": False, "allowed_days": [0, 1, 2, 3, 4],
            "use_vix_filter": False, "vix_min": 0.0, "vix_max": 20.0,
            "use_vol": False, "vol_thresh": 1.5,
            "use_vol_rank": True, "vol_rank_logic": "<", "vol_rank_thresh": 15.0,
            "trend_filter": "Market > 200 SMA",
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 5.0, "max_age": 100.0,
            "min_atr_pct": 0.0, "max_atr_pct": 100.0,
            "entry_conf_bps": 0,
            "use_ma_dist_filter": False, "dist_ma_type": "SMA 10", "dist_logic": "Greater Than (>)", "dist_min": 0.0, "dist_max": 2.0,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": ">", "gap_thresh": 3,
            "use_acc_count_filter": False, "acc_count_window": 21, "acc_count_logic": ">", "acc_count_thresh": 3,
            "use_dist_count_filter": False, "dist_count_window": 21, "dist_count_logic": ">", "dist_count_thresh": 3,
            "use_recent_52w_low": False, "recent_52w_low_invert": True, "recent_52w_low_lookback": 10,
            "dial_filters": []
        },
        "execution": {"risk_bps": 35, "slippage_bps": 2, "stop_atr": 1.25, "tgt_atr": 2.5, "hold_days": 10, "use_stop_loss": True, "use_take_profit": True,
                      "ladder_multipliers": [0.85, 1.00, 1.15],
                      # Earnings size override: when signal_date sits in the
                      # offset range [min_td, max_td] (trading days relative to
                      # earnings, negative = before), reduce risk to risk_bps
                      # instead of using the strategy's default. NaN offsets
                      # (commodity ETFs / indices / futures with no earnings
                      # data) bypass the override — they keep default sizing.
                      "earnings_size_override": {"min_td": -10, "max_td": 0, "risk_bps": 10}},
        "stats": {"grade": "A (Excellent)", "win_rate": "69.0%", "expectancy": "0.48r", "profit_factor": "2.82"}
    },
    {
        "id": "2+5+10+21d > 85, sell open +0.75 atr, 2 ATR tgt",
        "name": "Overbot Vol Spike",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Overnight",
            "thesis": "Fading multi-horizon overbought names — short-term overbought fade with a 252D barbell (skip mediocre 65-95th %ile) and a 5D seasonal headwind gate",
            "key_filters": [
                "2D + 5D + 10D + 21D ranks ALL > 85th %ile (extremely overbought)",
                "21D > 85th %ile for 3 consecutive days",
                "252D rank NOT between 65-95th %ile (avoid mediocre LT names)",
                "5D ATR seasonal rank < 85 (skip strong 5d seasonal windows)",
                "Today's return > 0.25 ATR (up day)"
            ]
        },
        "exit_summary": {
            "primary_exit": "2-day time stop OR 2.0 ATR target (whichever first)",
            "stop_logic": "None (time/target exit only)",
            "target_logic": "2.0 ATR below entry (short)",
            "notes": "Two-path execution. Path 1 (decisive): T+1 open > signal close + 0.25 ATR → flat 40 bps. Path 2 (mild): signal close < T+1 open ≤ close + 0.25 ATR → 8 bps with 1% aggregate path-2 cap (pro-rata scale-down). Open ≤ close → skip. ±10 trading-day earnings blackout applied at scan time (NaN passes through for tickers without earnings data). Same scheme for liquid and overflow universes."
        },
        "description": "Start: 2000-01-01. Universe: LIQUID_PLUS_COMMODITIES. Dir: Short. Multi-horizon overbought fade with 252D barbell + 5D seasonal headwind gate. Two-path sizing (40 bps decisive / 8 bps mild + 1% aggregate cap) keyed off T+1 open vs close+0.25 ATR. ±10 TD earnings blackout.",
        "universe_tickers": LIQUID_PLUS_COMMODITIES,
        "settings": {
            "trade_direction": "Short",
            "entry_type": "Limit (Open +/- 0.75 ATR)",
            "max_one_pos": False,
            "allow_same_day_reentry": True,
            "perf_filters": [
                {'window': 2, 'logic': '>', 'thresh': 85.0, 'consecutive': 1},
                {'window': 5, 'logic': '>', 'thresh': 85.0, 'consecutive': 1},
                {'window': 10, 'logic': '>', 'thresh': 85.0, 'consecutive': 1},
                {'window': 21, 'logic': '>', 'thresh': 85.0, 'consecutive': 3},
                {'window': 252, 'logic': 'Not Between', 'thresh': 65.0, 'thresh_max': 95.0, 'consecutive': 1},
            ],
            "atr_sznl_filters": [
                {'window': 5, 'logic': '<', 'thresh': 85.0, 'thresh_max': 100.0, 'consecutive': 1},
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
            "min_atr_pct": 0.2, "max_atr_pct": 10.0,
            "entry_conf_bps": 0,
            "use_ma_dist_filter": False, "dist_ma_type": "SMA 10", "dist_logic": "Greater Than (>)", "dist_min": 0.0, "dist_max": 2.0,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": ">", "gap_thresh": 3,
            "use_acc_count_filter": False, "acc_count_window": 21, "acc_count_logic": ">", "acc_count_thresh": 3,
            "use_dist_count_filter": False, "dist_count_window": 21, "dist_count_logic": ">", "dist_count_thresh": 0,
            "use_xsec_filter": True, "xsec_filters": []
        },
        "execution": {"risk_bps": 40, "slippage_bps": 2, "stop_atr": 1.0, "tgt_atr": 2.0, "hold_days": 2, "use_stop_loss": False, "use_take_profit": True,
                      "path1_bps": 40, "path2_bps": 8, "path2_daily_cap_pct": 0.75,
                      "earnings_blackout_td": 10,
                      "eod_dd_atr": 0.25, "eod_dd_weekdays": [4],
                      # Cycle-year risk tilt (2026-06-10): midterm years (year%4==2)
                      # run OVS at 0.75x. Evidence: all 6 midterm years 2006-2026
                      # underperform (avgR +0.19 vs +0.49 baseline), leave-one-year-
                      # out stable (-0.28..-0.37R gap), damage concentrated in P1
                      # decisive-gap entries (+0.63 -> +0.23 avgR). ~1.5 sigma after
                      # episode clustering, so 0.75x (shrunk-Kelly), not full 0.4x.
                      # Mirrored: strat_backtester sizing 3b2, daily_scan sizing 2e,
                      # order_staging OVS_CYCLE_MULTS (P1 fixed-dollar target).
                      "cycle_risk_mults": {2: 0.75}},
        "stats": {"grade": "A (Excellent)", "win_rate": "58.0%", "expectancy": "0.28r", "profit_factor": "1.96"}
    },
    {
        "id": "2d+5d+10d+21d < 15%ile, 252d between 65-90, range 0-15, today ret -10..-0.25 ATR, 100sma 20 consec above, 200sma 50 consec above, age >= 5y, ±10 earnings blackout, GTC limit close-0.25 ATR, 2 ATR tgt, 1d hold",
        "name": "LT Trend ST OS",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Overnight",
            "thesis": "Oversold bounce in long-term uptrenders — but NOT extreme leaders (252D capped at 90th %ile) to avoid buying climactic tops. Persistent uptrend confirmed via consecutive closes above 100D SMA (>=20) and 200D SMA (>=50). Demand a sharp red bar today (close in lower 15% of range AND today return <= -0.25 ATR) for the oversold setup. Persistent limit at close - 0.25 ATR for entry. ±10 TD earnings blackout to avoid catching a knife into a binary event.",
            "key_filters": [
                "2D rank < 15th %ile",
                "5D rank < 15th %ile",
                "10D rank < 15th %ile",
                "21D rank < 15th %ile",
                "252D rank between 65-90th %ile",
                "Close in 0-15% of daily range",
                "Today return between -10 ATR and -0.25 ATR (decisive red day)",
                "Close above 100D SMA for 20+ consecutive days",
                "Close above 200D SMA for 50+ consecutive days",
                "Min 5 years of price history",
                "No earnings within ±10 trading days"
            ]
        },
        "exit_summary": {
            "primary_exit": "1-day time stop OR 2.0 ATR target (whichever first)",
            "stop_logic": "None (time/target exit only)",
            "target_logic": "2.0 ATR above entry",
            "notes": "Entry changed from Signal Close (MOC) to Limit Order -0.25 ATR (Persistent GTC). No longer a MOC strategy — won't be picked up by the intraday --moc-only GHA runs."
        },
        "description": "Start: 2000-01-01. Universe: LIQUID_PLUS_COMMODITIES + overflow tier. Dir: Long. Entry: limit at close-0.25 ATR (GTC). 1d hold, 2 ATR target, no stop. 40 bps risk. WR 68.4% / PF 2.91 / Exp 0.40r (pre-changes).",
        "universe_tickers": LIQUID_PLUS_COMMODITIES,
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Limit Order -0.25 ATR (Persistent)",
            "max_one_pos": False,
            "allow_same_day_reentry": False,
            "entry_conf_bps": 0,
            "perf_filters": [
                {'window': 2, 'logic': '<', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 5, 'logic': '<', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 10, 'logic': '<', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 21, 'logic': '<', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 252, 'logic': 'Between', 'thresh': 65.0, 'thresh_max': 90.0, 'consecutive': 1},
            ],
            "perf_atr_filters": [],
            "perf_first_instance": False, "perf_lookback": 21,
            "ma_consec_filters": [
                {'length': 100, 'logic': 'Above', 'consec': 20},
                {'length': 200, 'logic': 'Above', 'consec': 50},
            ],
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
            "use_range_filter": True, "range_min": 0, "range_max": 15,
            "use_atr_ret_filter": True, "atr_ret_min": -10.0, "atr_ret_max": -0.25,
            "use_range_atr_filter": False, "range_atr_logic": ">", "range_atr_min": 1.0, "range_atr_max": 3.0,
            "price_action_filters": [],
            "use_ma_dist_filter": False, "dist_ma_type": "SMA 10", "dist_logic": "Greater Than (>)", "dist_min": 0.0, "dist_max": 2.0,
            "use_weekly_ma_pullback": False, "wma_type": "EMA", "wma_period": 8,
            "wma_min_ext_pct": 30.0, "wma_lookback_months": 6, "wma_touch_logic": "Low <= MA",
            "vol_gt_prev": False,
            "use_vol": False, "vol_logic": ">", "vol_thresh": 1.25, "vol_thresh_max": 10.0,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "use_acc_count_filter": False, "acc_count_window": 21, "acc_count_logic": ">", "acc_count_thresh": 3,
            "use_dist_count_filter": False, "dist_count_window": 21, "dist_count_logic": ">", "dist_count_thresh": 0,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": ">", "gap_thresh": 3,
            "trend_filter": "None",
            "use_vix_filter": False, "vix_min": 0.0, "vix_max": 20.0,
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 5.0, "max_age": 100.0,
            "min_atr_pct": 0.2, "max_atr_pct": 10.0,
            "use_dow_filter": False, "allowed_days": [0, 1, 2, 3, 4],
            "allowed_cycles": [1, 2, 3, 0],
            "excluded_years": [],
            "use_ref_ticker_filter": False, "ref_ticker": "SPY", "ref_filters": [],
            "use_t1_open_filter": False, "t1_open_filters": [],
            "use_xsec_filter": True, "xsec_filters": [],
            "atr_sznl_filters": [],
            "dial_filters": []
        },
        "execution": {
            "risk_bps": 40,
            "risk_per_trade": "[EDIT: calculated from account size]",
            "slippage_bps": 2,
            "stop_atr": 1.0,
            "tgt_atr": 2.0,
            "hold_days": 1,
            "use_stop_loss": False,
            "use_take_profit": True,
            # Symmetric earnings blackout: skip if signal_date is within ±10
            # trading days of an earnings announcement. NaN (commodities /
            # ETFs / futures with no earnings data) passes through.
            "earnings_blackout_td": 10
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
    {
        "id": "2d < 25%ile, Entry: Limit Order -0.25 ATR (Persistent), 2d hold",
        "name": "Indices Oversold Bounce",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Overnight",
            "thesis": "Short-horizon oversold bounce on the spot indices (^GSPC, ^NDX) — staged 1:1 in SPY/QQQ via spot-tradeable alias",
            "key_filters": [
                "2D rank < 25th %ile",
                "5D ATR seasonal rank > 50th %ile",
                "Close in 0-15% of daily range",
                "Net change between -10.0 and -0.25 ATR"
            ]
        },
        "exit_summary": {
            "primary_exit": "Target or 2-day time stop",
            "stop_logic": "None (time exit only)",
            "target_logic": "2.0 ATR above entry",
            "notes": "Detection on ^GSPC/^NDX (purer price); orders staged 1:1 on SPY/QQQ (recomputed ATR/close)."
        },
        "description": "Backtest: 2000-01-01 to present. Universe: INDICES_SPOT (^GSPC, ^NDX). Dir: Long. WR 64.6% / PF 1.90 / Exp 0.34r.",
        "universe_tickers": INDICES_SPOT,
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Limit Order -0.25 ATR (Persistent)",
            "max_one_pos": False,
            "allow_same_day_reentry": False,
            "max_daily_entries": 50,
            "max_total_positions": 99,
            "entry_conf_bps": 0,
            "perf_filters": [{'window': 2, 'logic': '<', 'thresh': 25.0, 'thresh_max': 100.0, 'consecutive': 1}],
            "perf_atr_filters": [],
            "perf_first_instance": False,
            "perf_lookback": 21,
            "ma_consec_filters": [],
            "use_sznl": False, "sznl_logic": '<', "sznl_thresh": 15.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_market_sznl": False, "market_sznl_logic": '<', "market_sznl_thresh": 15.0, "market_ticker": "^GSPC",
            "use_52w": False, "52w_type": "New 52w High", "52w_first_instance": False, "52w_lookback": 21, "52w_lag": 0,
            "exclude_52w_high": False,
            "use_ath": False, "ath_type": "Today is ATH",
            "use_recent_ath": False, "recent_ath_invert": False, "ath_lookback_days": 21,
            "use_recent_52w": False, "recent_52w_invert": False, "recent_52w_lookback": 21,
            "use_recent_52w_low": False, "recent_52w_low_invert": False, "recent_52w_low_lookback": 21,
            "breakout_mode": "None",
            "require_close_gt_open": False,
            "use_range_filter": True, "range_min": 0, "range_max": 15,
            "use_atr_ret_filter": True, "atr_ret_min": -10.0, "atr_ret_max": -0.25,
            "use_range_atr_filter": False, "range_atr_logic": '>', "range_atr_min": 1.0, "range_atr_max": 3.0,
            "price_action_filters": [],
            "use_ma_dist_filter": False, "dist_ma_type": "SMA 10", "dist_logic": "Greater Than (>)", "dist_min": 0.0, "dist_max": 2.0,
            "use_weekly_ma_pullback": False, "wma_type": "EMA", "wma_period": 8, "wma_min_ext_pct": 30.0, "wma_lookback_months": 6, "wma_touch_logic": "Low <= MA",
            "vol_gt_prev": False,
            "use_vol": False, "vol_logic": '>', "vol_thresh": 1.5, "vol_thresh_max": 10.0,
            "use_vol_rank": False, "vol_rank_logic": '<', "vol_rank_thresh": 15.0,
            "use_acc_count_filter": False, "acc_count_window": 21, "acc_count_logic": '>', "acc_count_thresh": 0,
            "use_dist_count_filter": False, "dist_count_window": 21, "dist_count_logic": '=', "dist_count_thresh": 0,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": '>', "gap_thresh": 3,
            "trend_filter": "None",
            "use_vix_filter": False, "vix_min": 0.0, "vix_max": 20.0,
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0,
            "min_atr_pct": 0.2, "max_atr_pct": 10.0,
            "use_dow_filter": False, "allowed_days": [0, 1, 2, 3, 4],
            "allowed_cycles": [1, 2, 3, 0],
            "excluded_years": [],
            "use_ref_ticker_filter": False,
            "ref_ticker": "IWM",
            "ref_filters": [],
            "use_t1_open_filter": False,
            "t1_open_filters": [],
            "use_xsec_filter": False,
            "xsec_filters": [{'window': 252, 'logic': '>', 'thresh': 85.0, 'thresh_max': 100.0, 'consecutive': 1}],
            "atr_sznl_filters": [{'window': 5, 'logic': '>', 'thresh': 50.0, 'thresh_max': 100.0, 'consecutive': 1}]
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
        "stats": {"grade": "A (Excellent)", "win_rate": "64.6%", "expectancy": "0.34r", "profit_factor": "1.90"}
    },
    {
        "id": "2d < 85%ile, Close 0-15% range, VIX >= 13, Mon/Fri only, Entry: Limit (Open +/- 0.25 ATR), 2d hold",
        "name": "SPY QQQ MonFri Reversion",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Overnight",
            "thesis": "Short-horizon mean-reversion harvest on SPY/QQQ — fades closes that finished in the lower 15% of the daily range, but only on Mondays and Fridays where weekly seasonality tends to amplify reversal odds. VIX >= 13 gate ensures there's enough realized vol for the 2-day mean reversion drift to be worth harvesting. Time stop captures the bulk of the edge; 1 ATR stop / 2 ATR target are path bounds that net to roughly zero.",
            "key_filters": [
                "2D rank < 85th %ile",
                "Close in 0-15% of daily range",
                "VIX >= 13",
                "Entry days: Mon, Fri",
            ]
        },
        "exit_summary": {
            "primary_exit": "Target, Stop, or 2-day time stop",
            "stop_logic": "1.0 ATR below entry",
            "target_logic": "2.0 ATR above entry",
            "notes": "Mon/Fri-only mean-reversion harvest on liquid index ETFs. Time exit is the dominant PnL contributor — stops and targets approximately cancel."
        },
        "description": "Backtest: 2000-01-01 to present. Universe: SPY, QQQ. Dir: Long. WR 62.9% / PF 2.21 / Exp 0.40r. 35 bps, 2d hold, 1 ATR stop, 2 ATR target.",
        "universe_tickers": ['SPY', 'QQQ'],
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Limit (Open +/- 0.25 ATR)",
            "max_one_pos": False,
            "allow_same_day_reentry": False,
            "max_daily_entries": 20,
            "max_total_positions": 99,
            "entry_conf_bps": 0,
            "perf_filters": [{'window': 2, 'logic': '<', 'thresh': 85.0, 'thresh_max': 100.0, 'consecutive': 1}],
            "perf_atr_filters": [],
            "perf_first_instance": False, "perf_lookback": 21,
            "ma_consec_filters": [],
            "use_sznl": False, "sznl_logic": '<', "sznl_thresh": 15.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_market_sznl": False, "market_sznl_logic": '<', "market_sznl_thresh": 15.0, "market_ticker": "^GSPC",
            "use_52w": False, "52w_type": "New High", "52w_first_instance": False, "52w_lookback": 21, "52w_lag": 0, "52w_window": 252,
            "exclude_52w_high": False,
            "use_ath": False, "ath_type": "Today is ATH",
            "use_recent_ath": False, "recent_ath_invert": False, "ath_lookback_days": 21,
            "use_recent_52w": False, "recent_52w_invert": False, "recent_52w_lookback": 21,
            "use_recent_52w_low": False, "recent_52w_low_invert": False, "recent_52w_low_lookback": 21,
            "breakout_mode": "None",
            "require_close_gt_open": False,
            "use_range_filter": True, "range_min": 0, "range_max": 15,
            "use_atr_ret_filter": False, "atr_ret_min": 0.5, "atr_ret_max": 10.0,
            "use_range_atr_filter": False, "range_atr_logic": '>', "range_atr_min": 1.0, "range_atr_max": 3.0,
            "use_open_gap_atr_filter": False, "open_gap_atr_logic": '>', "open_gap_atr_min": 0.0, "open_gap_atr_max": 1.0,
            "price_action_filters": [],
            "use_ma_dist_filter": False, "dist_ma_type": "SMA 50", "dist_logic": "Greater Than (>)", "dist_min": 6.0, "dist_max": 20.0,
            "use_weekly_ma_pullback": False, "wma_type": "EMA", "wma_period": 8, "wma_min_ext_pct": 30.0, "wma_lookback_months": 6, "wma_touch_logic": "Low <= MA",
            "use_volret_delta": False, "vrd_method": "Z-score diff", "vrd_rank_window": "Expanding", "vrd_vol_halflife": 20, "vrd_ret_horizon": 20, "vrd_delta_n": 5, "vrd_min_periods": 252, "vrd_pctile_min": 70.0, "vrd_pctile_max": 90.0,
            "use_tr_vcr_filter": False, "tr_vcr_metric": "Trend Ratio (TR)", "tr_vcr_window": 20, "tr_vcr_sample_freq": 5, "tr_vcr_min_periods": 252, "tr_vcr_rank_window": "Expanding", "tr_vcr_filter_mode": "Percentile rank", "tr_vcr_pctile_min": 70.0, "tr_vcr_pctile_max": 100.0, "tr_vcr_raw_min": 1.0, "tr_vcr_raw_max": 5.0, "tr_vcr_raw_logic": "Between", "tr_vcr_regime_quadrants": ('grinding_trend',), "tr_vcr_min_consec": 1, "tr_vcr_consec_first": False,
            "vol_gt_prev": False,
            "use_vol": False, "vol_logic": '>', "vol_thresh": 1.5, "vol_thresh_max": 10.0,
            "use_vol_rank": False, "vol_rank_logic": '<', "vol_rank_thresh": 50.0,
            "use_acc_count_filter": False, "acc_count_window": 21, "acc_count_logic": '>', "acc_count_thresh": 3,
            "use_dist_count_filter": False, "dist_count_window": 21, "dist_count_logic": '>', "dist_count_thresh": 3,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": '>', "gap_thresh": 3,
            "trend_filter": "None",
            "use_vix_filter": True, "vix_min": 13.0, "vix_max": 100.0,
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0,
            "min_atr_pct": 0.2, "max_atr_pct": 10.0,
            "use_dow_filter": True, "allowed_days": [0, 4],
            "allowed_cycles": [1, 2, 3, 0],
            "excluded_years": [],
            "use_ref_ticker_filter": False, "ref_ticker": "IWM", "ref_filters": [],
            "use_t1_open_filter": False, "t1_open_filters": [],
            # Monday-gap kill: drop a signal when its T+1 open gaps more than
            # t1_gap_kill_atr ATR in the kill direction vs the signal close, but
            # ONLY for signals whose weekday is in t1_gap_kill_signal_weekdays.
            # [4] = Friday signals (T+1 open lands on Monday); a Monday signal
            # (T+1 = Tuesday) is untouched. A gap UP kills the long mean-reversion
            # edge (the bounce already happened at the open). Enforced in the
            # backtest by get_historical_mask (drops the candidate); enforced live
            # by order_staging.py via the MonGapKill_* staging stamps.
            "use_t1_gap_kill": True, "t1_gap_kill_atr": 0.5, "t1_gap_kill_dir": "up", "t1_gap_kill_signal_weekdays": [4],
            "use_xsec_filter": False, "xsec_filters": [],
            "atr_sznl_filters": []
        },
        "execution": {
            "risk_bps": 35,
            "risk_per_trade": "[EDIT: calculated from account size]",
            "slippage_bps": 2,
            "stop_atr": 1.0,
            "tgt_atr": 2.0,
            "hold_days": 2,
            "use_stop_loss": True,
            "use_take_profit": True,
            "use_trailing_stop": False,
            "trail_atr": 2.0,
            "trail_anchor": "Peak High"
        },
        "stats": {"grade": "A (Excellent)", "win_rate": "62.9%", "expectancy": "0.40r", "profit_factor": "2.21"}
    },
    {
        "id": "252d Between 65-90%ile, New 52wH first in 21d, XSec 252d > 85%ile, Entry: Limit Order -0.5 ATR (Persistent), 63d hold",
        "name": "Sector BO",
        "setup": {
            "type": "Breakout",
            "timeframe": "Position",
            "thesis": "Momentum continuation in sector / index ETFs after a fresh 52w high, with cross-sectional 252D rank > 85th to ensure the breakout is from a genuine leader rather than a laggard catching up.",
            "key_filters": [
                "252D rank between 65-90th %ile (strong but not climactic)",
                "New 52w High (first in 21d)",
                "XSec 252D rank > 85th %ile (cross-sectional leader)"
            ]
        },
        "exit_summary": {
            "primary_exit": "Target, Stop, or 63-day time stop",
            "stop_logic": "1.0 ATR below entry",
            "target_logic": "8.0 ATR above entry",
            "notes": "Persistent GTC limit at signal close - 0.5 ATR; 63d hold for the full continuation move."
        },
        "description": "Backtest: 2000-01-01 to present. Universe: SECTOR_INDEX_ETFS. Dir: Long. WR 28.9% / Exp 1.17r / PF 2.53 — long-tail position trade.",
        "universe_tickers": SECTOR_INDEX_ETFS,
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Limit Order -0.5 ATR (Persistent)",
            "max_one_pos": True,
            "allow_same_day_reentry": False,
            "max_daily_entries": 20,
            "max_total_positions": 99,
            "entry_conf_bps": 0,
            "perf_filters": [
                {'window': 252, 'logic': 'Between', 'thresh': 65.0, 'thresh_max': 90.0, 'consecutive': 1},
            ],
            "perf_atr_filters": [],
            "perf_first_instance": False, "perf_lookback": 21,
            "ma_consec_filters": [],
            "use_sznl": False, "sznl_logic": "<", "sznl_thresh": 15.0, "sznl_first_instance": False, "sznl_lookback": 21,
            "use_market_sznl": False, "market_sznl_logic": "<", "market_sznl_thresh": 15.0,
            "market_ticker": "^GSPC",
            "use_52w": True, "52w_type": "New 52w High", "52w_first_instance": True, "52w_lookback": 21, "52w_lag": 0, "exclude_52w_high": False,
            "use_ath": False, "ath_type": "Today is ATH",
            "use_recent_ath": False, "recent_ath_invert": False, "ath_lookback_days": 21,
            "use_recent_52w": False, "recent_52w_invert": False, "recent_52w_lookback": 21,
            "use_recent_52w_low": False, "recent_52w_low_invert": False, "recent_52w_low_lookback": 21,
            "breakout_mode": "None",
            "require_close_gt_open": False,
            "use_range_filter": False, "range_min": 60, "range_max": 100,
            "use_atr_ret_filter": False, "atr_ret_min": 0.5, "atr_ret_max": 10.0,
            "use_range_atr_filter": False, "range_atr_logic": ">", "range_atr_min": 1.0, "range_atr_max": 3.0,
            "price_action_filters": [],
            "use_ma_dist_filter": False, "dist_ma_type": "SMA 10", "dist_logic": "Greater Than (>)", "dist_min": 0.0, "dist_max": 2.0,
            "use_weekly_ma_pullback": False, "wma_type": "EMA", "wma_period": 8,
            "wma_min_ext_pct": 30.0, "wma_lookback_months": 6, "wma_touch_logic": "Low <= MA",
            "vol_gt_prev": False,
            "use_vol": False, "vol_logic": ">", "vol_thresh": 1.5, "vol_thresh_max": 10.0,
            "use_vol_rank": False, "vol_rank_logic": "<", "vol_rank_thresh": 50.0,
            "use_acc_count_filter": False, "acc_count_window": 21, "acc_count_logic": ">", "acc_count_thresh": 3,
            "use_dist_count_filter": False, "dist_count_window": 21, "dist_count_logic": ">", "dist_count_thresh": 3,
            "use_gap_filter": False, "gap_lookback": 21, "gap_logic": ">", "gap_thresh": 3,
            "trend_filter": "None",
            "use_vix_filter": False, "vix_min": 0.0, "vix_max": 20.0,
            "min_price": 10.0, "min_vol": 100000,
            "min_age": 0.25, "max_age": 100.0,
            "min_atr_pct": 0.2, "max_atr_pct": 10.0,
            "use_dow_filter": False, "allowed_days": [0, 1, 2, 3, 4],
            "allowed_cycles": [1, 2, 3, 0],
            "excluded_years": [],
            "use_ref_ticker_filter": False, "ref_ticker": "IWM", "ref_filters": [],
            "use_t1_open_filter": False, "t1_open_filters": [],
            "use_xsec_filter": True,
            "xsec_filters": [
                {'window': 252, 'logic': '>', 'thresh': 85.0, 'thresh_max': 100.0, 'consecutive': 1},
            ],
            "atr_sznl_filters": [],
            "dial_filters": []
        },
        "execution": {
            "risk_bps": 25,
            "slippage_bps": 2,
            "stop_atr": 1.0,
            "tgt_atr": 8.0,
            "hold_days": 63,
            "use_stop_loss": True,
            "use_take_profit": True,
            "use_trailing_stop": False,
            "trail_atr": 2.0,
            "trail_anchor": "Peak High"
        },
        "stats": {"grade": "A (Excellent)", "win_rate": "28.9%", "expectancy": "1.17r", "profit_factor": "2.53"}
    },
    {'id': '2d < 50%ile, VIX >= 13, Entry: Limit (Open +/- 0.25 ATR), 2d hold',
     'name': 'Monday Dip',
     'setup': {'type': 'Custom',
               'timeframe': 'Overnight',
               'thesis': 'Short term oversold, closing near the lows of the day. Expecting mean reversion. Universe trimmed to IWM/DIA/SMH (SPY/QQQ carved out to the SPY QQQ MonFri Reversion strat to avoid same-day cross-fire). VIX >= 13 floor ensures enough realized vol for the 2-day drift to be worth harvesting.',
               'key_filters': ['2D rank < 50th %ile',
                               '5D ATR seasonal rank > 15th %ile',
                               'Close above 200 SMA (15d consecutive)',
                               'Close in 0-15% of daily range',
                               'VIX >= 13',
                               'Entry days: Mon']},
     'exit_summary': {'primary_exit': 'Target, Stop, or 2-day time stop',
                      'stop_logic': '1.0 ATR below entry',
                      'target_logic': '2.0 ATR above entry',
                      'notes': 'SPY/QQQ excluded — handled by SPY QQQ MonFri Reversion to prevent same-date overlap.'},
     'description': 'Backtest: 2000-01-01 to present. Universe: IWM, DIA, SMH (SPY/QQQ excluded). VIX >= 13 gate added. Stats below are pre-change (5-ticker, no VIX gate) — re-run to refresh.',
     'universe_tickers': ['IWM', 'DIA', 'SMH'],
     'settings': {'trade_direction': 'Long',
                  'entry_type': 'Limit (Open +/- 0.25 ATR)',
                  'max_one_pos': True,
                  'allow_same_day_reentry': False,
                  'max_daily_entries': 20,
                  'max_total_positions': 99,
                  'entry_conf_bps': 0,
                  'perf_filters': [{'window': 2, 'logic': '<', 'thresh': 50.0, 'thresh_max': 100.0, 'consecutive': 1}],
                  'perf_atr_filters': [],
                  'perf_first_instance': False,
                  'perf_lookback': 21,
                  'ma_consec_filters': [{'length': 200, 'logic': 'Above', 'consec': 15}],
                  'use_sznl': False,
                  'sznl_logic': '<',
                  'sznl_thresh': 15.0,
                  'sznl_first_instance': False,
                  'sznl_lookback': 21,
                  'use_market_sznl': False,
                  'market_sznl_logic': '<',
                  'market_sznl_thresh': 15.0,
                  'market_ticker': '^GSPC',
                  'use_52w': False,
                  '52w_type': 'New 52w High',
                  '52w_first_instance': False,
                  '52w_lookback': 21,
                  '52w_lag': 0,
                  'exclude_52w_high': False,
                  'use_ath': False,
                  'ath_type': 'Today is ATH',
                  'use_recent_ath': False,
                  'recent_ath_invert': False,
                  'ath_lookback_days': 21,
                  'use_recent_52w': False,
                  'recent_52w_invert': False,
                  'recent_52w_lookback': 21,
                  'use_recent_52w_low': False,
                  'recent_52w_low_invert': False,
                  'recent_52w_low_lookback': 21,
                  'breakout_mode': 'None',
                  'require_close_gt_open': False,
                  'use_range_filter': True,
                  'range_min': 0,
                  'range_max': 15,
                  'use_atr_ret_filter': False,
                  'atr_ret_min': 0.0,
                  'atr_ret_max': 1.0,
                  'use_range_atr_filter': False,
                  'range_atr_logic': '>',
                  'range_atr_min': 1.0,
                  'range_atr_max': 3.0,
                  'price_action_filters': [],
                  'use_ma_dist_filter': False,
                  'dist_ma_type': 'SMA 10',
                  'dist_logic': 'Greater Than (>)',
                  'dist_min': 0.0,
                  'dist_max': 2.0,
                  'use_weekly_ma_pullback': False,
                  'wma_type': 'EMA',
                  'wma_period': 8,
                  'wma_min_ext_pct': 30.0,
                  'wma_lookback_months': 6,
                  'wma_touch_logic': 'Low <= MA',
                  'vol_gt_prev': False,
                  'use_vol': False,
                  'vol_logic': '>',
                  'vol_thresh': 1.5,
                  'vol_thresh_max': 10.0,
                  'use_vol_rank': False,
                  'vol_rank_logic': '<',
                  'vol_rank_thresh': 50.0,
                  'use_acc_count_filter': False,
                  'acc_count_window': 21,
                  'acc_count_logic': '>',
                  'acc_count_thresh': 3,
                  'use_dist_count_filter': False,
                  'dist_count_window': 21,
                  'dist_count_logic': '>',
                  'dist_count_thresh': 3,
                  'use_gap_filter': False,
                  'gap_lookback': 21,
                  'gap_logic': '>',
                  'gap_thresh': 3,
                  'trend_filter': 'None',
                  'use_vix_filter': True,
                  'vix_min': 13.0,
                  'vix_max': 100.0,
                  'min_price': 10.0,
                  'min_vol': 100000,
                  'min_age': 0.25,
                  'max_age': 100.0,
                  'min_atr_pct': 0.2,
                  'max_atr_pct': 10.0,
                  'use_dow_filter': True,
                  'allowed_days': [0],
                  'allowed_cycles': [1, 2, 3, 0],
                  'excluded_years': [],
                  'use_ref_ticker_filter': False,
                  'ref_ticker': 'IWM',
                  'ref_filters': [],
                  'use_t1_open_filter': False,
                  't1_open_filters': [],
                  'use_xsec_filter': False,
                  'xsec_filters': [],
                  'atr_sznl_filters': [{'window': 5, 'logic': '>', 'thresh': 15.0, 'thresh_max': 100.0, 'consecutive': 1}]},
     'execution': {'risk_bps': 30,
                   'risk_per_trade': '[EDIT: calculated from account size]',
                   'slippage_bps': 2,
                   'stop_atr': 1.0,
                   'tgt_atr': 2.0,
                   'hold_days': 2,
                   'use_stop_loss': True,
                   'use_take_profit': True,
                   'use_trailing_stop': False,
                   'trail_atr': 2.0,
                   'trail_anchor': 'Peak High'},
     'stats': {'grade': 'A (Excellent)', 'win_rate': '64.1%', 'expectancy': '0.40r', 'profit_factor': '2.17'}},
    {'id': 'T+1 Open > Close +0.5 ATR, Entry: Limit (Open +/- 0.75 ATR), 2d hold',
     'name': 'ATR Extended Gap Up',
     'setup': {'type': 'MeanReversion',
               'timeframe': 'Overnight',
               'thesis': 'Fade exhaustion in names that have stretched extremely far from their 50d SMA (>10 ATR) on a high-conviction volume spike (>2x 63d avg) and then continued gapping up on the T+1 open. The combination of parabolic extension, demand exhaustion (volume climax), and a final gap reach signals a blow-off top that historically mean-reverts. Short the open via a limit at +0.75 ATR; hold 2 days to a 4 ATR target with no hard stop (time exit absorbs adverse days).',
               'key_filters': ['Distance from SMA 50 > 10.0 ATR (parabolic extension)',
                               'Volume > 2.0x 63-day avg (conviction spike)',
                               'T+1 Open > Signal Close + 0.5 ATR (gap-up confirmation)']},
     'exit_summary': {'primary_exit': '4.0 ATR target or 2-day time stop',
                      'stop_logic': 'None (time exit only)',
                      'target_logic': '4.0 ATR below entry (short)',
                      'notes': 'Limit short at T+1 Open + 0.75 ATR. No stop loss - relies on the 2-day time stop to bound adverse moves. Long-tail risk is real here (one extension can keep extending) so size carefully.'},
     'description': 'Backtest: 2000-01-01 to present. Universe: LIQUID_PLUS_COMMODITIES. Dir: Short. Fade parabolic blow-off tops (>10 ATR from 50d SMA) with volume + gap-up confirmation. 2d hold, 4 ATR tgt, no stop. 40 bps risk.',
     'universe_tickers': LIQUID_PLUS_COMMODITIES,
     'settings': {'trade_direction': 'Short',
                  'entry_type': 'Limit (Open +/- 0.75 ATR)',
                  'max_one_pos': False,
                  'allow_same_day_reentry': False,
                  'max_daily_entries': 20,
                  'max_total_positions': 99,
                  'entry_conf_bps': 0,
                  'perf_filters': [],
                  'perf_atr_filters': [],
                  'perf_first_instance': False,
                  'perf_lookback': 21,
                  'ma_consec_filters': [],
                  'use_sznl': False,
                  'sznl_logic': '<',
                  'sznl_thresh': 15.0,
                  'sznl_first_instance': False,
                  'sznl_lookback': 21,
                  'use_market_sznl': False,
                  'market_sznl_logic': '<',
                  'market_sznl_thresh': 15.0,
                  'market_ticker': '^GSPC',
                  'use_52w': False,
                  '52w_type': 'New High',
                  '52w_first_instance': False,
                  '52w_lookback': 21,
                  '52w_lag': 0,
                  '52w_window': 252,
                  'exclude_52w_high': False,
                  'use_ath': False,
                  'ath_type': 'Today is ATH',
                  'use_recent_ath': False,
                  'recent_ath_invert': False,
                  'ath_lookback_days': 21,
                  'use_recent_52w': False,
                  'recent_52w_invert': False,
                  'recent_52w_lookback': 21,
                  'use_recent_52w_low': False,
                  'recent_52w_low_invert': False,
                  'recent_52w_low_lookback': 21,
                  'breakout_mode': 'None',
                  'require_close_gt_open': False,
                  'use_range_filter': False,
                  'range_min': 0,
                  'range_max': 100,
                  'use_atr_ret_filter': False,
                  'atr_ret_min': 0.0,
                  'atr_ret_max': 1.0,
                  'use_range_atr_filter': False,
                  'range_atr_logic': '>',
                  'range_atr_min': 1.0,
                  'range_atr_max': 3.0,
                  'use_open_gap_atr_filter': False,
                  'open_gap_atr_logic': '>',
                  'open_gap_atr_min': 0.0,
                  'open_gap_atr_max': 1.0,
                  'price_action_filters': [],
                  'use_ma_dist_filter': True,
                  'dist_ma_type': 'SMA 50',
                  'dist_logic': 'Greater Than (>)',
                  'dist_min': 10.0,
                  'dist_max': 50.0,
                  'use_weekly_ma_pullback': False,
                  'wma_type': 'EMA',
                  'wma_period': 8,
                  'wma_min_ext_pct': 30.0,
                  'wma_lookback_months': 6,
                  'wma_touch_logic': 'Low <= MA',
                  'use_volret_delta': False,
                  'vrd_method': 'Z-score diff',
                  'vrd_rank_window': 'Expanding',
                  'vrd_vol_halflife': 20,
                  'vrd_ret_horizon': 20,
                  'vrd_delta_n': 5,
                  'vrd_min_periods': 252,
                  'vrd_pctile_min': 70.0,
                  'vrd_pctile_max': 90.0,
                  'use_tr_vcr_filter': False,
                  'tr_vcr_metric': 'Trend Ratio (TR)',
                  'tr_vcr_window': 20,
                  'tr_vcr_sample_freq': 5,
                  'tr_vcr_min_periods': 252,
                  'tr_vcr_rank_window': 'Expanding',
                  'tr_vcr_filter_mode': 'Percentile rank',
                  'tr_vcr_pctile_min': 70.0,
                  'tr_vcr_pctile_max': 100.0,
                  'tr_vcr_raw_min': 1.0,
                  'tr_vcr_raw_max': 5.0,
                  'tr_vcr_raw_logic': 'Between',
                  'tr_vcr_regime_quadrants': ('grinding_trend',),
                  'tr_vcr_min_consec': 1,
                  'tr_vcr_consec_first': False,
                  'vol_gt_prev': False,
                  'use_vol': True,
                  'vol_logic': '>',
                  'vol_thresh': 2.0,
                  'vol_thresh_max': 10.0,
                  'use_vol_rank': False,
                  'vol_rank_logic': '<',
                  'vol_rank_thresh': 50.0,
                  'use_acc_count_filter': False,
                  'acc_count_window': 21,
                  'acc_count_logic': '>',
                  'acc_count_thresh': 3,
                  'use_dist_count_filter': False,
                  'dist_count_window': 21,
                  'dist_count_logic': '>',
                  'dist_count_thresh': 3,
                  'use_gap_filter': False,
                  'gap_lookback': 21,
                  'gap_logic': '>',
                  'gap_thresh': 3,
                  'trend_filter': 'None',
                  'use_vix_filter': False,
                  'vix_min': 0.0,
                  'vix_max': 20.0,
                  'min_price': 10.0,
                  'min_vol': 100000,
                  'min_age': 0.25,
                  'max_age': 100.0,
                  'min_atr_pct': 0.2,
                  'max_atr_pct': 10.0,
                  'use_dow_filter': False,
                  'allowed_days': [0, 1, 2, 3, 4],
                  'allowed_cycles': [1, 2, 3, 0],
                  'excluded_years': [],
                  'use_ref_ticker_filter': False,
                  'ref_ticker': 'IWM',
                  'ref_filters': [],
                  'use_t1_open_filter': True,
                  't1_open_filters': [{'logic': '>', 'reference': 'Close', 'atr_offset': 0.5}],
                  'use_xsec_filter': False,
                  'xsec_filters': [],
                  'atr_sznl_filters': []},
     'execution': {'risk_bps': 40,
                   'slippage_bps': 2,
                   'stop_atr': 1.0,
                   'tgt_atr': 4.0,
                   'hold_days': 2,
                   'use_stop_loss': False,
                   'use_take_profit': True,
                   'use_trailing_stop': False,
                   'trail_atr': 2.0,
                   'trail_anchor': 'Peak High'},
     'stats': {'grade': 'A (Excellent)', 'win_rate': '65.2%', 'expectancy': '0.80r', 'profit_factor': '3.25'}},
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
# APPLY GLOBAL RISK MULTIPLIER (in-place on _STRATEGY_BOOK_RAW)
# ============================================
# Done before STRATEGY_BOOK = build_strategy_book() so both the raw book
# (imported by strat_backtester) and the public book see scaled bps.
if GLOBAL_RISK_MULTIPLIER != 1.0:
    for _s in _STRATEGY_BOOK_RAW:
        _exe = _s.get('execution', {})
        for _k in ('risk_bps', 'path1_bps', 'path2_bps', 'path2_daily_cap_pct'):
            if _k in _exe:
                _exe[_k] = _exe[_k] * GLOBAL_RISK_MULTIPLIER
        _eo = _exe.get('earnings_size_override')
        if _eo and 'risk_bps' in _eo:
            _eo['risk_bps'] = _eo['risk_bps'] * GLOBAL_RISK_MULTIPLIER

# ============================================
# DEFAULT EXPORT
# ============================================
# Uses ACCOUNT_VALUE at top of file - change that value to adjust all risks
STRATEGY_BOOK = build_strategy_book()
