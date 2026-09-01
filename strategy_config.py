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

# Per-strategy bps overrides for the overflow tier. NOMINAL — consumers scale
# by GLOBAL_RISK_MULTIPLIER at use. Single source for daily_scan,
# daily_portfolio_report and the engine's overflow_active path. OVS uses
# path-1 nominal (40 bps) for both universes — see order_staging.py.
OVERFLOW_RISK_OVERRIDES = {
    "Oversold Low Volume": 25,  # vs liquid 35 (signal-recency ladder applies on both tiers)
}

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
    'BAX', 'BDX', 'BNY', 'BMY', 'C', 'CAG', 'CAT', 'CL', 'CMCSA', 'CMS',
    'CNP', 'COP', 'COST', 'CPB', 'CRM', 'CSCO', 'CSX', 'CVS', 'CVX', 'D',
    'DE', 'DIA', 'DIS', 'DOV', 'DTE', 'DUK', 'ECL', 'ED', 'EIX', 'EMR',
    'EOG', 'ETR', 'EXC', 'F', 'FCX', 'FDX', 'FE', 'GD', 'GE', 'GILD',
    'GIS', 'GLW', 'GOOG', 'GPC', 'GS', 'HAL', 'HD', 'HIG', 'HON', 'HPQ',
    'HRL', 'HSY', 'HUM', 'IBB', 'IBM', 'IHI', 'INTC', 'IP', 'ITA', 'ITB',
    'ITW', 'IWM', 'IYR', 'JNJ', 'JPM', 'KEY', 'KMB', 'KO', 'KR',
    'KRE', 'LEG', 'LIN', 'LLY', 'LMT', 'LOW', 'LUV', 'MAS', 'MCD', 'MDT',
    'MET', 'META', 'MRSH', 'MMM', 'MO', 'MRK', 'MS', 'MSFT', 'MU', 'NEE',
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

# ETFs exempt from the OLV per-ticker notional cap (2026-07-20): diversified
# instruments where single-name gap risk doesn't apply. Union of the
# sector/index ETF lists, the commodity add-ons, and ETFs seen in the
# overflow tier. A future ETF missing from this list simply gets capped —
# fail-safe in the conservative direction.
OLV_CAP_EXEMPT_ETFS = sorted(set(
    SECTOR_INDEX_ETFS + INDEX_ETFS + INDICES_SPOT
    + ['CEF', 'GLD', 'OIH', 'SLV', 'UNG', 'USO', 'UVXY', 'XOP']
    + ['DBC', 'EEM', 'EFA', 'EWZ', 'FXI', 'GDX', 'GDXJ', 'HYG', 'IEF',
       'LQD', 'TLT']
))

# Fear-conditioned fragility band tables (2026-08-05, McKinley's chosen form —
# shipped ahead of the prereg gates as an explicit appetite decision, recorded
# in scratch/ultracode_research/family_pc_fear_band_prereg_2026-08-05.md).
# Carried by the 6 frag_risk_bands strategies via execution['pc_fear_bands'];
# the fear state (pc_fear.py: lag-1 trailing-252d pctile of the 10d-MA CBOE
# equity put/call > 85) SELECTS which band table applies to the 10d-MA 63d
# dial score. Stale/missing P/C (> 3 bd) fails CLOSED to the strategy's plain
# frag_risk_bands — the incumbent 0.25x book. Multiplier set is closed per the
# prereg: {1.25, 1.0, 0.25 (stale incumbent), 0.0}. NOT GRM-scaled (pure
# multipliers). Evidence + caveats (19-trade fear-ON hi-frag cell, appetite
# legs B/C): the prereg doc. (An OLV-specific table pair lived here for one
# session, 2026-08-24 -> 25; see the OLV execution comment.)
PC_FEAR_BANDS = {
    'on':  [[0, 50, 1.25], [50, 999, 1.0]],
    'off': [[0, 50, 1.0],  [50, 999, 0.0]],
}

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

# Bear-equity 3x names — carved out of the generic 3x ETF Overbot Fade into
# the looser bear-only fade (2026-07-07 study: scratch/lev3x_fade_class_study.py
# + lev3x_fade_bear_sizing_rule.py). Universes must stay disjoint so the two
# fades can never fire the same ticker on the same day.
LEV3X_BEAR_EQ = [
    'SPXS', 'SQQQ', 'SDOW', 'TZA',
    'SOXS', 'FAZ', 'TECS', 'LABD', 'ERY', 'DRV', 'WEBS', 'YANG', 'EDZ',
]

# Equity-BULL 3x names (broad + sector; excludes bonds and the commodity
# bull names NUGT/JNUG/GUSH). Excluded from the 3x Leader Gap Fade: shorting
# an overbought 252d-LEADER bull ETF fades momentum leadership and loses
# pervasively — every selectivity layer makes it worse (strictest cell
# 0-for-7, avgR -1.28; losses span 2018/2020/2021/2023/2024/2026). Evidence:
# scratch/lev3x_fade_leader_bulleq_clusters.py + _bulleq_strict.py.
LEV3X_BULL_EQ = [
    'SPXL', 'TQQQ', 'UDOW', 'TNA', 'MIDU',
    'SOXL', 'FAS', 'TECL', 'LABU', 'CURE', 'ERX', 'DPST',
    'DRN', 'NAIL', 'RETL', 'WEBL', 'DFEN', 'YINN', 'BRZU', 'EDC', 'MEXX',
]

# All CSV tickers from sznl_ranks.csv (~1062 tickers)
import os as _os, pandas as _pd
_csv_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'sznl_ranks.csv')
try:
    # Futures (=F) excluded 2026-07-06: not executable through the equity
    # order path (IBKR share orders), yfinance continuous-contract bars are
    # roll-distorted, and their holiday-session bars break equity-calendar
    # assumptions downstream. 14 ledger trades / -1.4R over 18y at removal.
    # Crypto pairs (-USD) excluded 2026-07-11 for the same reasons: no equity
    # order path, and 7-day/week bars book weekend fills the live book can't
    # take (ledger had BTC exits on a Sat/Sun). 19 ledger trades / -3.0R
    # (OVS 15 / -6.8R, OLV 4 / +3.8R) at removal.
    # Caret indices (^) excluded 2026-07-11 unless they carry a
    # SPOT_TO_TRADEABLE alias (^GSPC/^NDX -> SPY/QQQ): the other 18 (foreign
    # indices, ^RUT/^DJI/^IXIC, ^VIX, ^TNX-style yield levels) stage as raw
    # index symbols no broker can fill, and foreign/yield series break the
    # US-equity calendar assumptions. They reached the book only via the
    # overflow tier (^GSPC/^NDX sit in LIQUID and are subtracted there).
    # 25 ledger trades / -0.25R, all Overflow, at removal.
    # Corporate-action exclusions: names whose price action is pinned by a
    # deal, so rank-based signals on them are artifacts, not edges.
    # CBZ excluded 2026-07-29 (merger announced that day — the pop is
    # deal-arb, and fading/buying an arb-pinned name never mean-reverts).
    # 0 ledger trades at removal. Overflow tier only. Re-add if a deal breaks.
    UNIVERSE_CORP_ACTION_EXCLUSIONS = {'CBZ'}
    # Known-dead names, catalogued 2026-08-07 by scripts/diagnose_stale_tickers
    # after a cache audit found 14 symbols that had silently stopped updating.
    # No live successor in FMP's symbol-change feed and no recent yfinance bar.
    #
    # THIS SET IS DOCUMENTATION AND TELEMETRY, NOT A UNIVERSE FILTER.
    # It was briefly used to filter CSV_UNIVERSE and that was wrong: the
    # universe feeds the BACKTEST as well as the live scan, so excluding these
    # names deleted 25 historical trades on companies that were genuinely
    # tradeable at the time, deepening the very survivorship bias CLAUDE.md
    # flags on the 23-year ledger.
    #
    # Forward signalling is blocked where it belongs, in daily_scan's
    # per-ticker staleness drop: a symbol whose newest bar predates the
    # expected session is dropped from THAT RUN. That is general (it catches
    # the next delisting with nobody maintaining a list), it is date-aware
    # (the name still trades in history, up to its last real bar), and it is
    # loud. Keep this set current anyway: verify_universe_access reports on it
    # and it explains why those tickers linger in the cache.
    # All overflow-tier.
    UNIVERSE_DELISTED = {
        'TGNA', 'SEE', 'CTRA', 'CUK', 'CSGS', 'CTLP', 'GDEN', 'SEM', 'RSX',
        'THS',
    }
    # THS is the one case where excluding from the universe IS right: it has
    # ZERO rows in master_prices, so there is no history to preserve and
    # nothing for the staleness drop to catch (a ticker with no data never
    # enters the scan's dict at all). Keeping it in the universe would just
    # leave a permanent no-data entry that every consumer has to skip.
    # Anything here must have no usable price history, ever.
    UNIVERSE_NO_DATA = {'THS'}
    CSV_UNIVERSE = sorted(
        t for t in _pd.read_csv(_csv_path)['ticker'].unique().tolist()
        if t not in UNIVERSE_CORP_ACTION_EXCLUSIONS
        and t not in UNIVERSE_NO_DATA
        and not str(t).endswith('=F') and not str(t).endswith('-USD')
        and (not str(t).startswith('^') or t in SPOT_TO_TRADEABLE)
    )
except Exception:
    CSV_UNIVERSE = LIQUID_UNIVERSE  # fallback

# ============================================
# STRATEGY DEFINITIONS
# ============================================
# Note: risk_bps replaces fixed dollar risk (100 bps = 1% of account)
# Examples at $750k: 10 bps = $750, 20 bps = $1500, 35 bps = $2625

_STRATEGY_BOOK_RAW = [
    {
        "id": "252d Between 50-90, New 52w High, Today is ATH, vol > 2.5x, Market > 200 SMA, 63d dial 10ma < 30, Entry: Limit -0.25 ATR Persistent, 63d hold",
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
        "description": "Backtest: 2016-01-01 to present. Universe: LIQUID_UNIVERSE. Dir: Long. WR 47.3% / Exp 0.77r / PF 2.43. Entry moved -0.5 -> -0.25 ATR 2026-07-07 (entry-lab sweep: -0.25 wins every era, totR +131.8 vs +98.3, fill 87% vs 76% — shallower limit misses fewer runners; scratch/entry_lab_era_split.csv). Stats above are pre-change — re-run to refresh.",
        "universe_tickers": LIQUID_UNIVERSE,
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Limit Order -0.25 ATR (Persistent)",
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
            "fill_window_days": 10,
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
                   'trail_anchor': 'Peak High',
                   # Fragility risk bands (2026-07-02): [lo, hi, mult] on the
                   # 63d 10d-MA risk-dial score as of signal date; mult applies
                   # when lo <= score < hi, else 1.0 (also 1.0 when the score
                   # is missing/stale). The dip-buy FAMILY4 (this, MonFri
                   # Reversion, Monday Dip, Indices OS Bounce) runs 0.25x at
                   # >= 50: family avgR -0.283 (N=74) above 50 vs +0.607 below,
                   # clustered p=0.032, survives every single-year exclusion;
                   # the rest of the book shows NO degradation there (p=0.47)
                   # and stays 1.0x. Replaces the retired book-wide ramp (1.25x
                   # boost -> 0.10x floor). Aligned sites: daily_scan sizing 2b,
                   # strat_backtester sizing 3b3. Evidence:
                   # scratch/ultracode_research/PORTFOLIO_RESEARCH_2026-07-02.md
                   'frag_risk_bands': [[50, 999, 0.25]],
                   # P/C-fear table selection (2026-08-05) — see PC_FEAR_BANDS.
                   'pc_fear_bands': PC_FEAR_BANDS},
     'stats': {'grade': 'A (Excellent)', 'win_rate': '61.3%', 'expectancy': '0.28r', 'profit_factor': '1.78'}},
    {
        "id": "21dr < 15 3 consec, 5dr < 33, 2dr < 25, 252dr 50-90, rel vol < 15, market > 200 SMA, age >= 5y, pre-earnings -> 10 bps, GTC limit close-0.25 ATR, 10d hold, vol-confirmed 1.25 ATR stop (next open), 2.5 ATR tgt",
        "name": "Oversold Low Volume",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Position",
            "thesis": "Buying oversold names during low-volume selloffs in uptrenders (252d 50-90), gated to a market uptrend regime (SPY > 200 SMA) and minimum 5y of trading history. The persistent limit defaults to close - 0.25 ATR, but prior-resistance retests use causal 40/40 closing pivots no more than 252 ticker sessions old: -0.50 ATR when 2-3 ATR above the nearest confirmed pivot high, -0.75 ATR at 4-5 ATR, and no order above 5 ATR. Pre-earnings signals (signal_date in [-10, 0] TD relative to earnings) are still allowed but sized at 10 bps instead of the default 35 bps to dampen binary-event risk. First entry in a ticker sizes at 0.5x (ladder [0.5,1,1], 2026-07-29); stacked adds at full size.",
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
            "primary_exit": "10-day time stop OR 2.5 ATR target OR volume-confirmed stop (whichever first)",
            "stop_logic": "Vol-confirmed (2026-07-20): no resting stop. If a session CLOSES at/below entry - 1.25 ATR AND that day's volume >= 1.5x the trailing 20d median, exit MOO at the next open. Quiet closes below the level are held (low-volume weakness is the thesis, not its failure). stop_atr 1.25 still defines the risk unit for sizing.",
            "target_logic": "2.5 ATR above entry",
            "notes": "Persistent limit defaults to close - 0.25 ATR and uses the causal 40/40 closing-pivot entry policy in execution['pivot_entry_policy']; pivot sources expire after 252 ticker sessions and the GTC entry expires after T+3. No cooldown — consecutive signals on same ticker allowed. No ladder (removed 2026-07-20, all legs 1.0x) and no sector loss gate (removed 2026-07-20 with the vol-confirmed stop + notional cap package). Per-ticker concurrent notional capped at 50% of NAV for single stocks (ETFs exempt). Earnings handling: signals 10 TD before through earnings day get sized at 10 bps (vs. default 35 bps liquid / 25 bps overflow); commodity ETFs / indices / futures with no earnings data pass through at default sizing. First-entry half-size ladder [0.5,1,1] since 2026-07-29 (footprint trim on the weakest leg; adds stay full size). No fragility band and no book-level cap (a 0.5x dial>=65 band and an EOD 100%-NAV trim ran for one session, 2026-08-24, then were retired 2026-08-25 in favour of a manual one-off hedge; olv_book_cap.py stays in OneDrive, task disabled)."
        },
        "description": "Start: 2000-01-01. Universe: Liquid + commodities + overflow tier (CSV_UNIVERSE via OVERFLOW_ELIGIBLE). Dir: Long. Entry: persistent close-anchored limit, normally -0.25 ATR; causal 40/40 nearest-high retests use only pivot sources <=252 ticker sessions old, entering -0.50 ATR at 2-3 ATR above, -0.75 ATR at 4-5 ATR, and skipping above 5 ATR. 10d hold, 2.5 ATR target, 1.25 ATR stop. Liquid 35 bps / overflow 25 bps; first entry in a ticker 0.5x (ladder [0.5,1,1], 2026-07-29), adds full size; pre-earnings window sizes at 10 bps flat.",
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
        # Signal-recency ladder (2026-07-30, McKinley): rung = count of this
        # ticker's OLV SIGNALS (mask days, fill-independent) in the trailing
        # window_td sessions before the signal day. 0 prior -> 0.5x, 1 prior
        # -> 0.7x, 2+ -> 1.0x. Replaced the one-day-old open-position-count
        # ladder [0.5, 1, 1] (2026-07-29): same first-iteration half-size
        # intent, but (a) a re-signal within 21td of a closed chain no longer
        # resets to 0.5x, (b) a signal fired while yesterday's limit is still
        # unfilled counts it (the open-count ladder was blind to working
        # orders, like the notional cap), (c) second leg grades 0.7x instead
        # of jumping straight to full. Leg-order stats (2016+, dial-era):
        # leg-1 avgR +0.56-0.82 vs leg-3+ +1.1-1.4 — initial entries are
        # OLV's weakest legs, deep adds its strongest. Context: OLV open
        # notional ran $365k/day in 2026 vs the $160-180k 2018-2020 norm and
        # carried 49% of the book's 2026 intraday trough dollars.
        # Risk-appetite call, NOT an edge call (leg-1 is still positive; this
        # costs expectancy by design). NOTE the INVERSE old [0.85,1,1] ladder
        # (removed 2026-07-20) measured a first-rung discount as a drag —
        # this deeper cut is deliberate. The earnings override COMPOSES with
        # this mult since 2026-07-30 (override replaces the base, recency
        # mult still applies): pre-earnings first-iteration = 10 x 0.5 bps.
        "execution": {"risk_bps": 35, "slippage_bps": 2, "stop_atr": 1.25, "tgt_atr": 2.5, "hold_days": 10, "use_stop_loss": True, "use_take_profit": True,
                      "signal_recency_ladder": {"window_td": 21, "mults": [0.5, 0.7, 1.0]},
                      # NO fragility band (deliberate). A 0.5x cut at dial >= 65
                      # (lifted in P/C fear) shipped 2026-08-24 as an appetite
                      # decision and was RETIRED 2026-08-25 after one session:
                      # McKinley chose to hedge the Aug-2026 high-dial episode
                      # manually as a one-off rather than carry standing
                      # machinery. Evidence for the record (unchanged): 21 OLV
                      # trades at dial >= 70 ran +8R with zero stop-outs; OLV's
                      # worst drawdowns came at dial ~20 and were idiosyncratic
                      # (a SPY hedge would not have touched them). The generic
                      # band-composes-with-earnings-override path in daily_scan
                      # 2d / strat_backtester 3b3b stays (carrier-less for OLV).
                      # History: CLAUDE.md "Fragility Risk Bands".
                      # Entry-order live window (2026-06-24): the persistent
                      # close-0.25 ATR limit is cancelled if unfilled after 3
                      # trading days (T+1..T+3), NOT the full 10-day hold. 89% of
                      # OLV fills land by T+3; the day 4-10 fills add ~0 total R
                      # (+211 -> +211) while diluting per-trade edge (avgR +0.637
                      # T+3 vs +0.566 T+10, win 62.8% vs 60.6%, PF 2.90 vs 2.65).
                      # Defaults to hold_days when absent, so other persistent
                      # strategies are unchanged. Aligned: strat_backtester fill
                      # loop, daily_scan Fill_Window_Days stamp, order_staging
                      # GTC cancel-after-N. Evidence: scratch/olv_fill_window.py.
                      "fill_window_days": 3,
                      # Causal closing-pivot entry policy (2026-08-31). For each
                      # signal, compare the latest eligible 40/40 close pivot high
                      # with the latest eligible low and select the nearer PRICE
                      # level. A pivot at p is eligible only at p+40, so scanner and
                      # engine see identical point-in-time context. High-nearest
                      # signals 2<d<=3 ATR above resistance enter 0.50 ATR below the
                      # signal close; 4<d<=5 enter 0.75 ATR below; d>5 is not staged.
                      # High and low expire independently once their SOURCE bar
                      # is >252 ticker sessions old, then nearest is reselected.
                      # If neither survives, the ordinary -0.25 ATR entry applies.
                      # Strict replay on the 359 completed-fill research sample:
                      # 23 classifications / 12 policy assignments changed;
                      # >2/>4/>5 ATR degradation held and policy total improved
                      # +8.68R with unchanged max drawdown (2026-09-01).
                      # The non-monotonic 3<d<=4 pocket stays at the 0.25 ATR base
                      # because deeper entries degraded in that band. One-switch
                      # rollback: enabled=False restores 0.25 ATR and stages all.
                      "pivot_entry_policy": {
                          "enabled": True,
                          "version": "olv_close_pivot_40_v2_20260901",
                          "left_bars": 40,
                          "right_bars": 40,
                          "max_source_age_bars": 252,
                          "default_offset_atr": 0.25,
                          "rules": [
                              {"name": "above_high_gt5", "min_exclusive": 5.0,
                               "action": "skip"},
                              {"name": "above_high_4_5", "min_exclusive": 4.0,
                               "max_inclusive": 5.0, "offset_atr": 0.75},
                              {"name": "above_high_2_3", "min_exclusive": 2.0,
                               "max_inclusive": 3.0, "offset_atr": 0.50},
                          ],
                      },
                      # Vol-confirmed stop (2026-07-20): replaces the resting
                      # 1.25 ATR STP. Exit MOO at the NEXT open iff a session
                      # CLOSES <= entry - stop_atr*ATR AND that day's volume
                      # >= stop_vol_mult x the trailing 20d median (ex-today).
                      # Quiet closes below the level are HELD — low-volume
                      # weakness is the entry thesis, not its failure; the
                      # T+10 time exit still bounds everything. Evidence
                      # (scratch/olv_stop_condition_study.py + _nextopen_test):
                      # intraday stop touches that recover by the close end
                      # -0.32R avg vs -1.10R for confirmed closes; the rule
                      # adds ~+35R/21y, cuts same-day stop+rebuy churn 39 -> ~0
                      # (a volume-spike exit and a fresh low-volume-rank signal
                      # are near mutually exclusive by construction), clustered
                      # t=2.44, LOYO floor +22R. Costs: per-leg tail widens to
                      # occasional -2..-3R; no resting stop overnight (gaps are
                      # evaluated at the next close). stop_atr still defines
                      # the risk unit for sizing. The sector loss gate and the
                      # ladder_multipliers were REMOVED with this change
                      # (2026-07-20): the gate's drop list flipped to +10R
                      # after it blocked the entire late-June-2026 oil recovery,
                      # and flat 1.0x sizing beat both ladder variants.
                      # Aligned: strat_backtester exit branch, daily_scan
                      # Use_Stop stamp + OLV_Exits staging, order_staging /
                      # eq_order_entry MOO-exit path (OneDrive).
                      "stop_mode": "vol_confirm_close",
                      "stop_vol_mult": 1.5,
                      # Per-ticker concurrent notional cap (2026-07-20):
                      # stacked OLV legs in ONE single-stock ticker may not
                      # exceed pct_nav of account value at entry; later legs
                      # are scaled down / skipped to fit. ETFs (diversified
                      # gap profile) are exempt via OLV_CAP_EXEMPT_ETFS.
                      # Catastrophe insurance for the no-resting-stop world:
                      # historical cost ~4% of OLV PnL (binds 8 legs in 21y,
                      # every one a winner — the balloon stacks are low-ATR
                      # names that mean-revert best), in exchange for bounding
                      # the survivorship-blind single-name overnight tail at
                      # half of NAV. Evidence: scratch/olv_notional_cap_*.py.
                      "ticker_notional_cap": {"pct_nav": 0.50,
                                              "exempt": OLV_CAP_EXEMPT_ETFS},
                      # Earnings size override: when signal_date sits in the
                      # offset range [min_td, max_td] (trading days relative to
                      # earnings, negative = before), reduce risk to risk_bps
                      # instead of using the strategy's default. NaN offsets
                      # (commodity ETFs / indices / futures with no earnings
                      # data) bypass the override — they keep default sizing.
                      # Since 2026-07-30 the override COMPOSES with the
                      # signal_recency_ladder mult (it replaces the BASE bps
                      # only): first-iteration pre-earnings = 10 x 0.5 bps.
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
                      # Scale-out (live 2026-06-17, engine-modeled 2026-07-16):
                      # order_staging splits every OVS P1/P2 row into two
                      # independent single-target brackets — near_frac of the
                      # shares target near_tgt_atr ATR, the remainder targets
                      # the full tgt_atr. Deliberate short-book VARIANCE
                      # SMOOTHING, not PnL-maximizing (measured -R vs full-size
                      # 2 ATR — accepted trade-off, McKinley 2026-07-16). MUST
                      # match order_staging.py OVS_SCALEOUT_NEAR_FRAC /
                      # OVS_PROFIT_TAKER_ATR_MULT (PA is never split). Engine:
                      # strat_backtester books two tranche rows per fill.
                      # Guard: tests/test_ovs_scaleout.py. NOT GRM-scaled.
                      "scaleout_near_frac": 0.40, "scaleout_near_tgt_atr": 1.0,
                      # Cycle-year risk tilt (2026-06-10): midterm years (year%4==2)
                      # run OVS at 0.75x. Evidence: all 6 midterm years 2006-2026
                      # underperform (avgR +0.19 vs +0.49 baseline), leave-one-year-
                      # out stable (-0.28..-0.37R gap), damage concentrated in P1
                      # decisive-gap entries (+0.63 -> +0.23 avgR). ~1.5 sigma after
                      # episode clustering, so 0.75x (shrunk-Kelly), not full 0.4x.
                      # Mirrored: strat_backtester sizing 3b2, daily_scan sizing 2e,
                      # order_staging OVS_CYCLE_MULTS (P1 fixed-dollar target).
                      "cycle_risk_mults": {2: 0.75},
                      # Fragility mid-band tilt REMOVED (2026-07-03, PIT gate).
                      # A 0.75x tilt in [21,44) shipped 2026-07-02 on full-sample
                      # z=-3.0, but the point-in-time edge-weight re-estimation
                      # (roadmap step 5, scratch/pit_reestimate.py) failed it on
                      # the testable window: 2018+ clustered t=-1.34 PIT /
                      # t=-0.63 even with current weights — the evidence lived in
                      # 2016-17 trades no honest vintage can grade. Unwound per
                      # the pre-agreed gate. OVS remains fully EXEMPT from
                      # fragility sizing (its 55+ strength was affirmative and
                      # is untouched by this). Do not re-add without fresh
                      # out-of-window evidence.
                      },
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
            "risk_bps": 30,
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
            "use_take_profit": True,
            # Imminent-earnings derate (2026-07-30): the no-stop 5d hold can
            # straddle the print. Ledger -5..-1 TD cell: N=9, avgR -0.50, all
            # tail losers (AAP -5.96R, LYV -3.05R) vs +0.32 outside. Small-N
            # risk-appetite haircut (~15% of normal), NOT a fitted edge rule.
            "earnings_size_override": {"min_td": -5, "max_td": -1, "risk_bps": 6}
        },
        "stats": {"grade": "A (Excellent)", "win_rate": "64.5%", "expectancy": "0.45r", "profit_factor": "2.17"}
    },
    {
        "id": "2d > 85%ile+5d > 85%ile+10d > 85%ile+21d > 85%ile+126d < 65%ile+252d < 65%ile, Entry: Limit (Open +/- 0.5 ATR), 2d hold",
        "name": "3x ETF Overbot Fade",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Overnight",
            "thesis": "Overbought fade on 3x leveraged ETFs that are NOT medium/long-term leaders — pure multi-horizon overbought fade (no volume or range requirement). Bear-equity names carved out to the looser 3x Bear ETF Overbot Fade (2026-07-07) — universes disjoint to prevent same-day cross-fire.",
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
        "description": "Backtest: 2000-01-01 to present. Universe: 3x Leveraged minus bear-equity names (29 tickers; bear-equity carved out to 3x Bear ETF Overbot Fade 2026-07-07). Stats below are pre-carve-out (42-ticker) — re-run to refresh.",
        "universe_tickers": [t for t in LEV3X_ALL if t not in LEV3X_BEAR_EQ],
        "settings": {
            "trade_direction": "Short",
            "entry_type": "Limit (Open +/- 0.5 ATR)",
            # False since 2026-07-28: consecutive-day re-fires stack (live never
            # enforced one-pos; model aligned to live). Evidence:
            # scratch/lev3x_fade_stacking_study.py (+12.7R/23y, depth<=3).
            "max_one_pos": False,
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
        "id": "2d > 80%ile+5d > 80%ile+10d > 80%ile+21d > 80%ile+126d < 65%ile+252d < 65%ile, Entry: Limit (Open +/- 0.5 ATR), 2d hold",
        "name": "3x Bear ETF Overbot Fade",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Overnight",
            "thesis": "Looser overbought fade restricted to bear-equity 3x ETFs (carved out of the generic 3x fade 2026-07-07). Fading an overbought inverse ETF is a long-market dip-buy plus 3x vol-drag harvest, so it runs FAMILY4-style fragility bands and a same-day signal de-rate (multiple inverse names lighting up together = violent selloff, degraded per-trade edge). The 126/252d < 65 leader exclusion is LOAD-BEARING: it is what keeps this from shorting sustained bear markets (dropping it collapsed avgR +0.66 -> +0.28 with -14R/-16R years in 2020/2022) — do not relax it. Evidence: scratch/lev3x_fade_class_study.py, lev3x_fade_bear_episodes.py, lev3x_fade_bear_sizing_rule.py.",
            "key_filters": [
                "2D rank > 80th %ile",
                "5D rank > 80th %ile",
                "10D rank > 80th %ile",
                "21D rank > 80th %ile",
                "126D rank < 65th %ile",
                "252D rank < 65th %ile"
            ]
        },
        "exit_summary": {
            "primary_exit": "2-day time stop",
            "stop_logic": "None (time exit only)",
            "target_logic": "None (time exit only)",
            "notes": "Same execution as the parent 3x fade; only the entry thresholds (85->80, 21d consec 3->1), universe, and sizing overlays differ."
        },
        "description": "Backtest: 2003-01-01 to present (signals all land 2020+, one-regime sample — hence reduced bps). Universe: 13 bear-equity 3x ETFs. WR 66.7% / Exp 0.66r / PF 2.80, ~9 tr/yr, episode t +3.5.",
        "universe_tickers": LEV3X_BEAR_EQ,
        "settings": {
            "trade_direction": "Short",
            "entry_type": "Limit (Open +/- 0.5 ATR)",
            # False since 2026-07-28: consecutive-day re-fires stack (live never
            # enforced one-pos; model aligned to live). Stacking widens this
            # strat's DD profile ~1.5x for +10.2R — accepted. Evidence:
            # scratch/lev3x_fade_stacking_study.py.
            "max_one_pos": False,
            "allow_same_day_reentry": False,
            "entry_conf_bps": 0,
            "perf_filters": [
                {'window': 2, 'logic': '>', 'thresh': 80.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 5, 'logic': '>', 'thresh': 80.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 10, 'logic': '>', 'thresh': 80.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 21, 'logic': '>', 'thresh': 80.0, 'thresh_max': 100.0, 'consecutive': 1},
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
            "risk_bps": 25,
            "risk_per_trade": "[EDIT: calculated from account size]",
            "slippage_bps": 2,
            "stop_atr": 1.0,
            "tgt_atr": 8.0,
            "hold_days": 2,
            "use_stop_loss": False,
            "use_take_profit": False,
            # Dip-buy-adjacent: fading overbought inverse ETFs = buying market
            # selloffs, so it inherits the FAMILY4 fragility throttle.
            "frag_risk_bands": [[50, 999, 0.25]],
            "pc_fear_bands": PC_FEAR_BANDS,
            # Same-day de-rate: each staged signal scaled by
            # max(floor, 1 - derate x (n_signals_today - 1)). Count is ex-ante
            # (staged signals, not fills). See same_day_derate_mult().
            "same_day_signal_derate": 0.10,
            "same_day_derate_floor": 0.30,
            # Pilot promotion review (2026-07-16). No kill criteria by
            # McKinley's call ("we aren't gonna kill those") — the pilot
            # stays regardless; the review is only about earning the
            # size-up. Guard: tests/test_pilot_governance.py asserts every
            # 'B (Pilot)' strategy carries this block.
            "pilot": {
                "start": "2026-07-07",
                "review_by": "2027-01-15 or +15 live fills, whichever first",
                "promote_if": "live avgR > +0.3 across >=15 fills spanning >=2 "
                              "quarters incl. at least one SPY<200SMA episode "
                              "-> consider 40 bps",
            },
        },
        "stats": {"grade": "B (Pilot)", "win_rate": "66.7%", "expectancy": "0.66r", "profit_factor": "2.80"}
    },
    {
        "id": "2d > 80%ile+5d > 80%ile+10d > 80%ile+21d > 80%ile+252d > 95%ile+T1 gap up 0.25 ATR, Entry: Limit (Open +/- 0.75 ATR), 2d hold",
        "name": "3x Leader Gap Fade",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Overnight",
            "thesis": "Capitulation fade on 3x ETFs whose UNDERLYING is spiking on fear: 252d rank > 95 (leader REQUIRED — the inverse of the other fades' <65 exclusion, so no same-day cross-fire is possible on shared tickers by construction), short-horizon overbought, AND still gapping up 0.25 ATR at the T+1 open. The demanding entry (gap gate + 0.75 ATR limit above the open) IS the risk control: it replaces both the stop (tested 1.0-2.0 ATR, all destroyed the edge — adverse excursion > 1 ATR is the normal path before the reversal; worst no-stop trade -2.95R) and the fragility throttle (deliberately EXEMPT from frag_risk_bands: the edge lives on exactly the high-fragility days the FAMILY4 throttle would quarter). Universe = LEV3X_ALL minus LEV3X_BULL_EQ (13 bear-eq + TMF/TMV + 6 cmdty; bull-eq excluded structurally, see LEV3X_BULL_EQ comment). Tail risk is bounded by the engine/order_staging per-strategy 2.5% daily aggregate cap (would have trimmed only 2022-09-22 and 2025-04-04, ~5% each). Evidence: scratch/lev3x_fade_leader_{expansion,stops,entries,ovs_entry,class_split,validation}.py.",
            "key_filters": [
                "2D rank > 80th %ile",
                "5D rank > 80th %ile",
                "10D rank > 80th %ile",
                "21D rank > 80th %ile",
                "252D rank > 95th %ile (leader required)",
                "T+1 open > close + 0.25 ATR (gap gate, resolved at open)"
            ]
        },
        "exit_summary": {
            "primary_exit": "2-day time stop",
            "stop_logic": "None (time exit only — stops tested and rejected, see thesis)",
            "target_logic": "None (time exit only)",
            "notes": "Gap gate + 0.75 ATR limit resolve live at the T+1 open in order_staging (OVS-style), not at scan time."
        },
        "description": "Backtest: 2003-01-01 to present (first signal 2011). 31 trades / 15 episodes, ~2.3 tr/yr. Validation 2026-07-10: episode-clustered t=2.17, LOYO floor 1.55, drop-best-episode leaves +9.4R at t=1.79, bootstrap P(<=0)=2.1%. Sized one tier below conviction (25 bps, bear-fade parity): multi-regime sample offsets the lower t vs the bear fade's one-regime 2020+ sample.",
        "universe_tickers": [t for t in LEV3X_ALL if t not in LEV3X_BULL_EQ],
        "settings": {
            "trade_direction": "Short",
            "entry_type": "Limit (Open +/- 0.75 ATR)",
            "max_one_pos": True,
            "allow_same_day_reentry": False,
            "entry_conf_bps": 0,
            "perf_filters": [
                {'window': 2, 'logic': '>', 'thresh': 80.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 5, 'logic': '>', 'thresh': 80.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 10, 'logic': '>', 'thresh': 80.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 21, 'logic': '>', 'thresh': 80.0, 'thresh_max': 100.0, 'consecutive': 1},
                {'window': 252, 'logic': '>', 'thresh': 95.0, 'thresh_max': 100.0, 'consecutive': 1}
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
            "use_t1_open_filter": True,
            "t1_open_filters": [
                {'reference': 'Close', 'atr_offset': 0.25, 'logic': '>'}
            ],
            "use_xsec_filter": False,
            "xsec_filters": [],
            "atr_sznl_filters": []
        },
        "execution": {
            "risk_bps": 25,
            "risk_per_trade": "[EDIT: calculated from account size]",
            "slippage_bps": 2,
            "stop_atr": 1.0,
            "tgt_atr": 8.0,
            "hold_days": 2,
            "use_stop_loss": False,
            "use_take_profit": False,
            # Deliberately NO frag_risk_bands and NO same_day_signal_derate
            # (decided 2026-07-10): the gap gate + 0.75 ATR limit already
            # select for the high-fragility capitulation days those overlays
            # would throttle, and the generic per-strategy 2.5% daily cap
            # (engine post-loop cap_bps=250 default + order_staging's live
            # 2.5% cap) bounds the many-signal days. Do not add them without
            # re-running scratch/lev3x_fade_leader_validation.py.
            # Pilot promotion review (2026-07-16). From the validation notes
            # ('consider 40 bps only after clean out-of-sample quarters';
            # ~2 signals/yr historically, so the clock is long). No kill
            # criteria by McKinley's call — the review is size-up only.
            "pilot": {
                "start": "2026-07-10",
                "review_by": "2027-07-10 or +6 live fills, whichever first",
                "promote_if": "clean OOS: >=4 of first 6 live fills positive "
                              "AND live totR > 0 spanning >=2 quarters "
                              "-> consider 40 bps",
            },
        },
        "stats": {"grade": "B (Pilot)", "win_rate": "54.8%", "expectancy": "0.80r", "profit_factor": "2.82"}
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
            "use_take_profit": True,
            # Dip-buy FAMILY4 fragility throttle — see Weak Close Decent Sznls
            # for the full evidence note.
            "frag_risk_bands": [[50, 999, 0.25]],
            "pc_fear_bands": PC_FEAR_BANDS
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
            "trail_anchor": "Peak High",
            # Dip-buy FAMILY4 fragility throttle — see Weak Close Decent Sznls
            # for the full evidence note.
            "frag_risk_bands": [[50, 999, 0.25]],
            "pc_fear_bands": PC_FEAR_BANDS,
            # Large-gap-up size derate (2026-07-21). When the T+1 open gaps more
            # than threshold_atr * ATR ABOVE the signal close, the dip-buy edge is
            # roughly halved (the bounce partly played out at the open, and the
            # limit fills at a worse price), so trade at `mult` size. This LAYERS
            # ON TOP of the Friday-only t1_gap_kill above: a Friday signal gapping
            # > 0.5 ATR is dropped entirely by the kill (filter, runs first); this
            # derate then half-sizes everything the kill leaves standing that still
            # gaps > 0.25 ATR (Mon/Tue/etc. signals + Friday 0.25-0.5 ATR gaps).
            # Sizing overlay, not a filter — engine applies it in strat_backtester
            # step 3b5; live it is stamped (GapDerate_*) for order_staging to apply
            # at the IBKR open. See CLAUDE.md "Large-Gap-Up Size Derate".
            "gap_size_derate": {"threshold_atr": 0.25, "mult": 0.5, "dir": "up"}
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
            "fill_window_days": 10,
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
                   'trail_anchor': 'Peak High',
                   # Dip-buy FAMILY4 fragility throttle — see Weak Close Decent
                   # Sznls for the full evidence note.
                   'frag_risk_bands': [[50, 999, 0.25]],
                   'pc_fear_bands': PC_FEAR_BANDS,
                   # Large-gap-up size derate (2026-07-21) — half size when the
                   # T+1 open gaps > threshold_atr * ATR above the signal close.
                   # Same overlay as SPY QQQ MonFri Reversion; see that strat's
                   # note + CLAUDE.md "Large-Gap-Up Size Derate". Monday Dip has
                   # no t1_gap_kill, so this is its only gap-up size response.
                   'gap_size_derate': {'threshold_atr': 0.25, 'mult': 0.5, 'dir': 'up'}},
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
    {
        "id": "month close in lower 15% of monthly range on month-end day, Price > 200 SMA, Entry: Limit Order -0.25 ATR (Persistent, 2d window), 5d hold, 2 ATR tgt, no stop",
        "name": "Monthly Weak Close",
        "setup": {
            "type": "MeanReversion",
            "timeframe": "Position",
            "thesis": "A month that closes in the bottom 15% of its own high-low range while the uptrend is intact (close > 200d SMA, ~10-month MA equivalent) marks a capitulative month-end in a bull regime; the mean reversion plays out over the following weeks. The trend gate is load-bearing: unfiltered, QQQ 2000-2001 and Aug-2022 style signals ride the next bear leg down (worst -17.9%); gated, the 26y sample has one losing fill. The persistent limit (close - 0.25 ATR, live T+1..T+2) fills ~half the signals but historically captured ~90% of the close-entry total PnL with materially better per-fill stats — the missed half bounces immediately and never looks back.",
            "key_filters": [
                "Month's close in lower 15% of the month's high-low range",
                "Signal fires only on the month's last trading day",
                "Price > 200 SMA (uptrend regime, ~= the 10-month MA gate)"
            ]
        },
        "exit_summary": {
            "primary_exit": "2.0 ATR target or 5-day time stop",
            "stop_logic": "None (time exit bounds everything; stop_atr 1.0 defines the sizing risk unit only)",
            "target_logic": "2.0 ATR above entry",
            "notes": "Persistent limit at signal close - 0.25 ATR, cancelled if unfilled after 2 trading days (fill_window_days). ~1.1 signals/yr at the gated rate; SPY+QQQ same-month signals are near-duplicates (cluster stats treat them as one obs)."
        },
        "description": "Backtest: 2000-01-01 to present (scratch/monthly_weak_close_mr*.py, 2026-07-31). Gated close-entry cell: N=30, 77% win, avg +1.55%, worst -1.74%, clustered t=4.08; limit-entry cell N=15, 15-for-15, avg +2.79%, t=8.58 (same population, half the fills). LOYO avg range [+4.02,+4.50] on the h21 research variant; cluster bootstrap P(<=0)=0.0000. Pilot at 30 bps (manual-seasonal parity).",
        "universe_tickers": ['SPY', 'QQQ'],
        "settings": {
            "trade_direction": "Long",
            "entry_type": "Limit Order -0.25 ATR (Persistent)",
            "max_one_pos": True,
            "allow_same_day_reentry": False,
            "entry_conf_bps": 0,
            "perf_filters": [],
            "perf_first_instance": False,
            "perf_lookback": 21,
            "ma_consec_filters": [],
            "use_month_range_pos": True,
            "month_range_pos_max": 0.15,
            "use_sznl": False,
            "sznl_logic": "<",
            "sznl_thresh": 15.0,
            "sznl_first_instance": False,
            "sznl_lookback": 21,
            "use_market_sznl": False,
            "market_sznl_logic": "<",
            "market_sznl_thresh": 15.0,
            "market_ticker": "SPY",
            "use_52w": False,
            "52w_type": "New 52w High",
            "52w_first_instance": False,
            "52w_lookback": 21,
            "52w_lag": 0,
            "exclude_52w_high": False,
            "breakout_mode": "None",
            "use_range_filter": False,
            "range_min": 0,
            "range_max": 100,
            "use_dow_filter": False,
            "allowed_days": [0, 1, 2, 3, 4],
            "use_vix_filter": False,
            "vix_min": 0.0,
            "vix_max": 20.0,
            "use_vol": False,
            "vol_thresh": 1.5,
            "use_vol_rank": False,
            "vol_rank_logic": "<",
            "vol_rank_thresh": 50.0,
            "trend_filter": "Price > 200 SMA",
            "min_price": 10.0,
            "min_vol": 100000,
            "min_age": 1.0,
            "max_age": 100.0,
            "min_atr_pct": 0.0,
            "max_atr_pct": 100.0,
            "use_ma_dist_filter": False,
            "dist_ma_type": "SMA 10",
            "dist_logic": "Greater Than (>)",
            "dist_min": 0.0,
            "dist_max": 2.0,
            "use_gap_filter": False,
            "gap_lookback": 21,
            "gap_logic": ">",
            "gap_thresh": 3,
            "use_acc_count_filter": False,
            "acc_count_window": 21,
            "acc_count_logic": ">",
            "acc_count_thresh": 3,
            "use_dist_count_filter": False,
            "dist_count_window": 21,
            "dist_count_logic": ">",
            "dist_count_thresh": 3,
            "use_recent_52w_low": False,
            "recent_52w_low_invert": True,
            "recent_52w_low_lookback": 10,
            "dial_filters": []
        },
        "execution": {
            "risk_bps": 30,
            "slippage_bps": 2,
            "stop_atr": 1.0,
            "tgt_atr": 2.0,
            "hold_days": 5,
            "use_stop_loss": False,
            "use_take_profit": True,
            # Entry-order live window: the persistent limit is cancelled if
            # unfilled after 2 trading days (the researched GTC-2d spec).
            "fill_window_days": 2,
            # FAMILY4 band by analogy (2026-07-31, McKinley's call): this is
            # the same dip-buy exposure at monthly scale, and the one gated
            # loser (SPY 2020-01-31 into COVID) is exactly the high-dial-
            # above-trend case the band exists for. Same convention as the
            # 3x Bear Fade's band (family analogy, no own-sample fit).
            "frag_risk_bands": [[50, 999, 0.25]],
            "pc_fear_bands": PC_FEAR_BANDS,
            "pilot": {
                "start": "2026-07-31",
                "review_by": "2028-07-31 or +4 live fills, whichever first",
                "promote_if": "clean OOS fills consistent with the 15-for-15 "
                              "limit-entry cell -> consider 40 bps",
            },
        },
        "stats": {"grade": "B (Pilot)", "win_rate": "100% (limit cell, N=15)", "expectancy": "+2.79%/fill", "profit_factor": "inf (no losing fills in sample)"}
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


def same_day_derate_mult(execution, n_signals):
    """Same-day signal de-rate (3x Bear ETF Overbot Fade, 2026-07-07).

    When a strategy sets execution['same_day_signal_derate'] = d, every signal
    it stages on a day with n same-strategy signals is sized at
    max(floor, 1 - d*(n-1)), floor = execution['same_day_derate_floor']
    (default 0.30). n counts STAGED SIGNALS (known pre-market), not fills —
    several inverse-3x names overbought at once marks a violent selloff where
    per-trade edge degrades (scratch/lev3x_fade_bear_sizing_rule.py). Shared
    by daily_scan (post-pass 5c) and strat_backtester (sizing 3b4).
    """
    d = execution.get('same_day_signal_derate')
    if not d or n_signals <= 1:
        return 1.0
    floor = float(execution.get('same_day_derate_floor', 0.30))
    return max(floor, 1.0 - float(d) * (n_signals - 1))


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
    # Overlap clamp is nominal in source like every other bps value
    for _ovr in CROSS_STRATEGY_OVERLAP_OVERRIDES:
        _ovr['risk_bps_when_overlapping'] = (
            _ovr['risk_bps_when_overlapping'] * GLOBAL_RISK_MULTIPLIER)

# ============================================
# DEFAULT EXPORT
# ============================================
# Uses ACCOUNT_VALUE at top of file - change that value to adjust all risks
STRATEGY_BOOK = build_strategy_book()
