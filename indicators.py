"""
indicators.py — Unified Indicator Calculation Engine
=====================================================
Single source of truth for all indicator logic across the trading system.

IMPORTED BY:
    - daily_scan.py         (production scanner)
    - pages/backtester.py   (single-strategy research)
    - pages/strat_backtester.py  (portfolio simulation)
    - pages/walk_forward.py (walk-forward analysis)

HISTORY:
    Before this file existed, calculate_indicators() was duplicated across
    all four scripts above. Known divergences that existed:
        1. min_periods for perf rank: daily_scan used 50, backtesters used 252
        2. daily_scan passed fill_method=None to pct_change; backtester did not
        3. daily_scan hardcoded gap/acc/dist windows; backtester made them configurable
        4. ATH check: daily_scan used >= (correct), backtester used > (missed exact ties)
        5. Column naming: 'Change_in_ATR' vs 'today_return_atr' for same calc
        6. vol_ratio_10d_rank min_periods: 50 vs 252
    All of these are now resolved in THIS file. The authoritative choices are
    documented inline.

RULES:
    - ALL indicator changes go here. Never add a local calculate_indicators().
    - If you need a new indicator, add it here with a comment and date.
    - The backtester's signal logic (conditions list) stays in each consumer.
      This file only computes raw indicator columns.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict


# =============================================================================
# HELPERS (also shared — import these instead of re-implementing)
# =============================================================================

def get_election_cycle(year: int) -> str:
    """Map year to presidential election cycle phase."""
    rem = year % 4
    if rem == 0: return "0. Election Year"
    if rem == 1: return "1. Post-Election"
    if rem == 2: return "2. Midterm Year"
    if rem == 3: return "3. Pre-Election"
    return "Unknown"


def get_age_bucket(years: float) -> str:
    """Categorize ticker age for analysis bucketing."""
    if years < 3: return "< 3 Years"
    if years < 5: return "3-5 Years"
    if years < 10: return "5-10 Years"
    if years < 20: return "10-20 Years"
    return "> 20 Years"


def compute_weekly_ma(df: pd.DataFrame, ma_type: str = 'EMA', period: int = 8) -> pd.Series:
    """Resample daily Close to weekly, compute MA, forward-fill back to daily index."""
    weekly = df['Close'].resample('W-FRI').last().dropna()
    if len(weekly) < period:
        return pd.Series(np.nan, index=df.index)
    if ma_type == 'EMA':
        weekly_ma = weekly.ewm(span=period, adjust=False).mean()
    else:  # SMA
        weekly_ma = weekly.rolling(period).mean()
    return weekly_ma.reindex(df.index, method='ffill')


def apply_first_instance_filter(condition_series: pd.Series, lookback: int) -> pd.Series:
    """
    Only fire a signal if it hasn't fired in the prior (lookback - 1) bars.
    Used for "first new 52w high in 21 days" type filters.
    """
    if lookback <= 1:
        return condition_series
    condition_shifted = condition_series.shift(1)
    condition_shifted = condition_shifted.fillna(False).infer_objects(copy=False)
    rolling_sum = condition_shifted.rolling(window=lookback - 1, min_periods=1).sum()
    return condition_series & (rolling_sum == 0)


# =============================================================================
# CORE: Unified Indicator Calculator
# =============================================================================

def calculate_indicators(
    df: pd.DataFrame,
    sznl_map: dict,
    ticker: str,
    market_series: Optional[pd.Series] = None,
    vix_series: Optional[pd.Series] = None,
    # --- Optional params (backtester passes these; daily_scan uses defaults) ---
    market_sznl_series: Optional[pd.Series] = None,
    gap_window: int = 21,
    custom_sma_lengths: Optional[List[int]] = None,
    acc_window: Optional[int] = None,
    dist_window: Optional[int] = None,
    ref_ticker_ranks: Optional[Dict[int, pd.Series]] = None,
    weekly_ma_configs: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """
    Calculate all technical indicators for a single ticker.

    Parameters
    ----------
    df : DataFrame with columns: Open, High, Low, Close, Volume (any casing).
    sznl_map : Seasonal rank lookup (ticker -> date -> rank).
    ticker : Ticker symbol (used for seasonal lookup).
    market_series : Bool series — True when market (e.g. SPY) is above its 200 SMA.
    vix_series : VIX Close series aligned to trading dates.
    market_sznl_series : Pre-computed market seasonal series (backtester provides this).
    gap_window : Lookback for open gap count. Default 21.
    custom_sma_lengths : Extra SMA periods beyond the standard 10/20/50/100/200.
    acc_window : If set, compute AccCount for this specific window (backtester mode).
    dist_window : If set, compute DistCount for this specific window (backtester mode).
    ref_ticker_ranks : Dict of {window: rank_series} for reference ticker filter.

    Returns
    -------
    DataFrame with all indicator columns appended. Index is tz-naive, normalized.
    """
    df = df.copy()

    # -------------------------------------------------------------------------
    # 0. DEFENSIVE CLEANUP (yfinance MultiIndex trap)
    # -------------------------------------------------------------------------
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.capitalize() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.normalize()
    df.sort_index(inplace=True)

    # -------------------------------------------------------------------------
    # 1. MOVING AVERAGES
    # -------------------------------------------------------------------------
    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['SMA100'] = df['Close'].rolling(100).mean()
    df['SMA200'] = df['Close'].rolling(200).mean()

    # Custom SMAs (backtester may request e.g. SMA150 for a specific strategy)
    if custom_sma_lengths:
        for length in custom_sma_lengths:
            col_name = f"SMA{length}"
            if col_name not in df.columns:
                df[col_name] = df['Close'].rolling(length).mean()

    df['EMA8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['EMA11'] = df['Close'].ewm(span=11, adjust=False).mean()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()

    # -------------------------------------------------------------------------
    # 2. PERFORMANCE RANKS
    #    RESOLVED DIVERGENCE: Using min_periods=252 everywhere.
    #    Rationale: 252 ensures a full year of data before ranking, which
    #    prevents noisy percentiles on young tickers. daily_scan previously
    #    used 50, which could produce misleading ranks.
    # -------------------------------------------------------------------------
    RANK_MIN_PERIODS = 252
    for window in [2, 5, 10, 21]:
        df[f'ret_{window}d'] = df['Close'].pct_change(window, fill_method=None)
        df[f'rank_ret_{window}d'] = (
            df[f'ret_{window}d']
            .expanding(min_periods=RANK_MIN_PERIODS)
            .rank(pct=True) * 100.0
        )

    # -------------------------------------------------------------------------
    # 3. ATR (14-period, standard True Range)
    # -------------------------------------------------------------------------
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    df['ATR'] = ranges.max(axis=1).rolling(14).mean()
    df['ATR_Pct'] = (df['ATR'] / df['Close']) * 100

    # Today's net move in ATR units (vs yesterday's close)
    # RESOLVED DIVERGENCE: Standardized column name to 'today_return_atr'.
    # backtester.py previously used 'Change_in_ATR' — update backtester import.
    df['today_return_atr'] = (df['Close'] - df['Close'].shift(1)) / df['ATR']

    # Today's range in ATR units (for range_atr filter)
    df['range_in_atr'] = (df['High'] - df['Low']) / df['ATR']

    # -------------------------------------------------------------------------
    # 4. CANDLE RANGE LOCATION %
    #    Where did Close land within today's High-Low range?
    #    0% = closed at low, 100% = closed at high
    # -------------------------------------------------------------------------
    denom = df['High'] - df['Low']
    df['RangePct'] = np.where(denom == 0, 0.5, (df['Close'] - df['Low']) / denom)

    # -------------------------------------------------------------------------
    # 5. GAP COUNTS (open gaps = today's Low > yesterday's High)
    # -------------------------------------------------------------------------
    is_open_gap = (df['Low'] > df['High'].shift(1)).astype(int)
    # Always compute the standard windows (daily_scan needs these)
    df['GapCount_21'] = is_open_gap.rolling(21).sum()
    df['GapCount_10'] = is_open_gap.rolling(10).sum()
    df['GapCount_5'] = is_open_gap.rolling(5).sum()
    # Also store the configurable-window version for strat_backtester
    df['GapCount'] = is_open_gap.rolling(gap_window).sum()

    # -------------------------------------------------------------------------
    # 6. VOLUME, ACCUMULATION & DISTRIBUTION
    # -------------------------------------------------------------------------
    vol_ma = df['Volume'].rolling(63).mean()
    df['vol_ma'] = vol_ma
    df['vol_ratio'] = df['Volume'] / vol_ma

    # Volume rank (10d MA vs 63d MA, percentile over history)
    vol_ma_10 = df['Volume'].rolling(10).mean()
    df['vol_ratio_10d'] = vol_ma_10 / vol_ma
    # RESOLVED DIVERGENCE: Using min_periods=252 (was 50 in daily_scan)
    df['vol_ratio_10d_rank'] = (
        df['vol_ratio_10d']
        .expanding(min_periods=RANK_MIN_PERIODS)
        .rank(pct=True) * 100.0
    )

    # Vol Spike = volume above 63d MA AND above previous day
    cond_vol_ma = df['Volume'] > vol_ma
    cond_vol_up = df['Volume'] > df['Volume'].shift(1)
    df['Vol_Spike'] = cond_vol_ma & cond_vol_up

    # Accumulation days (green candle + vol spike)
    cond_green = df['Close'] > df['Open']
    is_accumulation = (df['Vol_Spike'] & cond_green).astype(int)
    df['is_acc_day'] = is_accumulation

    # Distribution days (red candle + vol spike)
    cond_red = df['Close'] < df['Open']
    is_distribution = (df['Vol_Spike'] & cond_red).astype(int)
    df['is_dist_day'] = is_distribution

    # Always compute standard windows (daily_scan needs all four)
    for w in [5, 10, 21, 42]:
        df[f'AccCount_{w}'] = is_accumulation.rolling(w).sum()
        df[f'DistCount_{w}'] = is_distribution.rolling(w).sum()

    # Backtester may request a specific non-standard window
    if acc_window and acc_window not in [5, 10, 21, 42]:
        df[f'AccCount_{acc_window}'] = is_accumulation.rolling(acc_window).sum()
    if dist_window and dist_window not in [5, 10, 21, 42]:
        df[f'DistCount_{dist_window}'] = is_distribution.rolling(dist_window).sum()

    # -------------------------------------------------------------------------
    # 7. SEASONALITY
    # -------------------------------------------------------------------------
    df['Sznl'] = get_sznl_val_series(ticker, df.index, sznl_map)
    # Market seasonal: prefer pre-computed series if provided, else compute from map
    if market_sznl_series is not None:
        df['Market_Sznl'] = market_sznl_series.reindex(df.index, method='ffill').fillna(50.0)
    # Always compute Mkt_Sznl_Ref from the map (daily_scan uses this column name)
    df['Mkt_Sznl_Ref'] = get_sznl_val_series("^GSPC", df.index, sznl_map)

    # -------------------------------------------------------------------------
    # 8. AGE (years since first row in the dataframe)
    # -------------------------------------------------------------------------
    if not df.empty:
        start_ts = df.index[0]
        df['age_years'] = (df.index - start_ts).days / 365.25
    else:
        df['age_years'] = 0.0

    # -------------------------------------------------------------------------
    # 9. MARKET REGIME
    # -------------------------------------------------------------------------
    if market_series is not None:
        df['Market_Above_SMA200'] = market_series.reindex(df.index, method='ffill').fillna(False)

    # -------------------------------------------------------------------------
    # 10. VIX
    # -------------------------------------------------------------------------
    if vix_series is not None:
        df['VIX_Value'] = vix_series.reindex(df.index, method='ffill').fillna(0)
    else:
        df['VIX_Value'] = 0.0

    # -------------------------------------------------------------------------
    # 11. 52-WEEK HIGH / LOW
    # -------------------------------------------------------------------------
    rolling_high = df['High'].shift(1).rolling(252).max()
    rolling_low = df['Low'].shift(1).rolling(252).min()
    df['is_52w_high'] = df['High'] > rolling_high
    df['is_52w_low'] = df['Low'] < rolling_low
    df['High_52w'] = df['High'].rolling(252).max()

    # -------------------------------------------------------------------------
    # 12. ALL-TIME HIGH
    #     RESOLVED DIVERGENCE: Using >= (daily_scan's approach).
    #     Rationale: If today's high exactly matches the prior ATH, that IS
    #     still an all-time high. Using > would miss exact-match ATH days.
    # -------------------------------------------------------------------------
    df['prior_ath'] = df['High'].shift(1).expanding().max()
    df['is_ath'] = df['High'] >= df['prior_ath']
    df['ATH_Level'] = df['High'].expanding().max()

    # -------------------------------------------------------------------------
    # 13. REFERENCE TICKER RANKS
    # -------------------------------------------------------------------------
    if ref_ticker_ranks is not None:
        for window, series in ref_ticker_ranks.items():
            df[f'Ref_rank_ret_{window}d'] = series.reindex(df.index, method='ffill').fillna(50.0)

    # -------------------------------------------------------------------------
    # 14. CONVENIENCE COLUMNS (used by strat_backtester for vectorized signals)
    # -------------------------------------------------------------------------
    df['DayOfWeekVal'] = df.index.dayofweek
    df['PrevHigh'] = df['High'].shift(1)
    df['PrevLow'] = df['Low'].shift(1)
    # NextOpen is look-ahead — only safe for backtesting, not live scanning
    df['NextOpen'] = df['Open'].shift(-1)

    # -------------------------------------------------------------------------
    # 15. PIVOT HIGHS / LOWS (used by backtester limit entry mode)
    # -------------------------------------------------------------------------
    piv_len = 20
    roll_max = df['High'].rolling(window=piv_len * 2 + 1, center=True).max()
    df['is_pivot_high'] = (df['High'] == roll_max)
    roll_min = df['Low'].rolling(window=piv_len * 2 + 1, center=True).min()
    df['is_pivot_low'] = (df['Low'] == roll_min)
    df['LastPivotHigh'] = np.where(df['is_pivot_high'], df['High'], np.nan)
    df['LastPivotHigh'] = df['LastPivotHigh'].shift(piv_len).ffill()
    df['LastPivotLow'] = np.where(df['is_pivot_low'], df['Low'], np.nan)
    df['LastPivotLow'] = df['LastPivotLow'].shift(piv_len).ffill()

    # -------------------------------------------------------------------------
    # 16. WEEKLY MOVING AVERAGES (only when backtester requests them)
    # -------------------------------------------------------------------------
    if weekly_ma_configs:
        for cfg in weekly_ma_configs:
            ma_type = cfg.get('type', 'EMA')
            period = cfg.get('period', 8)
            col_name = f"Weekly_{ma_type}{period}"
            if col_name not in df.columns:
                df[col_name] = compute_weekly_ma(df, ma_type=ma_type, period=period)

    return df


# =============================================================================
# SEASONAL HELPER — shared by all consumers
# =============================================================================
# NOTE: This function should be defined once here. If it already exists in a
# shared utils.py, import it instead. The implementation below is the canonical
# version extracted from the existing codebase.

def get_sznl_val_series(ticker: str, dates: pd.DatetimeIndex, sznl_map: dict) -> pd.Series:
    """
    Look up seasonal rank for a ticker across a date range.
    sznl_map is {ticker: pd.Series} or {ticker: dict} where keys are dates.
    Returns a Series aligned to the input dates with default value 50.
    """
    ticker = ticker.upper()

    val = sznl_map.get(ticker)
    if val is None and ticker == "^GSPC":
        val = sznl_map.get("SPY")

    if val is None:
        return pd.Series(50.0, index=dates)

    # Support dict-of-dicts format (backtester's load_seasonal_map legacy)
    if isinstance(val, dict):
        val = pd.Series(val)

    return val.reindex(dates, method='ffill').fillna(50.0)
