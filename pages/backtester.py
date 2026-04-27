import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import datetime
import random
import time
import uuid
import json

from indicators import calculate_indicators, apply_first_instance_filter, get_sznl_val_series

MARKET_TICKER = "^GSPC" 
VIX_TICKER = "^VIX"

SECTOR_ETFS = ["IBB", "IHI", "ITA", "ITB", "IYR", "KRE", "OIH", "SMH", "VNQ",
    "XBI", "XHB", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP",
    "XLU", "XLV", "XLY", "XME", "XOP", "XRT"]
SPX=['^GSPC','SPY']
INDEX_ETFS = ["SPY", "QQQ", "IWM", "DIA", "SMH"]
INTERNATIONAL_ETFS = ["EWZ", "EWC", "ECH", "ECOL", "EWW", "ARGT", "EWQ", "EWG", "EWI", "EWU", "EWP", "EWK", "EWO", "EWN", "EWD", "EWL",
    "EWJ", "EWH", "MCHI", "INDA", "EWY", "EWT", "EWA", "EWS", "EWM", "THD", "EIDO", "VNM", "EPHE", "EZA", "TUR", "EGPT"]

# 3x Leveraged ETFs (most-liquid pick per underlying; dedupe SPXL/UPRO, TNA/URTY, etc.)
LEV3X_EQUITY_BULL_BROAD  = ["SPXL", "TQQQ", "UDOW", "TNA", "MIDU"]
LEV3X_EQUITY_BEAR_BROAD  = ["SPXS", "SQQQ", "SDOW", "TZA"]
LEV3X_EQUITY_BULL_SECTOR = ["SOXL", "FAS", "TECL", "LABU", "CURE", "ERX", "DPST",
                            "DRN", "NAIL", "RETL", "WEBL", "DFEN", "YINN", "BRZU", "EDC", "MEXX"]
LEV3X_EQUITY_BEAR_SECTOR = ["SOXS", "FAZ", "TECS", "LABD", "ERY", "DRV", "WEBS", "YANG", "EDZ"]
LEV3X_BOND_BULL          = ["TMF"]
LEV3X_BOND_BEAR          = ["TMV"]
LEV3X_COMMODITY_BULL     = ["NUGT", "JNUG", "GUSH"]
LEV3X_COMMODITY_BEAR     = ["DUST", "JDST", "DRIP"]

LEV3X_EQUITY_BULL_ALL = LEV3X_EQUITY_BULL_BROAD + LEV3X_EQUITY_BULL_SECTOR
LEV3X_EQUITY_BEAR_ALL = LEV3X_EQUITY_BEAR_BROAD + LEV3X_EQUITY_BEAR_SECTOR
LEV3X_EQUITY_ALL      = LEV3X_EQUITY_BULL_ALL + LEV3X_EQUITY_BEAR_ALL
LEV3X_EQUITY_BROAD    = LEV3X_EQUITY_BULL_BROAD + LEV3X_EQUITY_BEAR_BROAD
LEV3X_ALL             = (LEV3X_EQUITY_ALL + LEV3X_BOND_BULL + LEV3X_BOND_BEAR
                         + LEV3X_COMMODITY_BULL + LEV3X_COMMODITY_BEAR)
CSV_PATH = "seasonal_ranks.csv"
ATR_SZNL_PATH = "atr_seasonal_ranks.parquet"
ATR_SZNL_WINDOWS = [5, 10, 21, 63, 126, 252]
ATR_SZNL_COLS = [f"atr_sznl_{w}d" for w in ATR_SZNL_WINDOWS]

@st.cache_resource
def load_seasonal_map():
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        return {}
    if df.empty: return {}
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce').dt.normalize().dt.tz_localize(None)
    df = df.dropna(subset=["Date"])
    output_map = {}
    for ticker, group in df.groupby("ticker"):
        output_map[str(ticker).upper()] = pd.Series(group.seasonal_rank.values, index=group.Date)
    return output_map


@st.cache_resource
def load_atr_seasonal_map():
    """Load ATR-normalized seasonal ranks. Returns {ticker: DataFrame with 6 rank columns}."""
    try:
        df = pd.read_parquet(ATR_SZNL_PATH)
    except Exception:
        return {}
    if df.empty:
        return {}
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    output = {}
    for ticker, group in df.groupby('ticker'):
        output[str(ticker).upper()] = group.set_index('Date')[ATR_SZNL_COLS].sort_index()
    return output


@st.cache_resource
def load_fragility_dials():
    """Load daily fragility dial history (5d/21d/63d scores) from the Risk Dashboard.

    Returns a DataFrame indexed by date with columns '5d', '21d', '63d' (0-100).
    Returns None if the parquet is missing or unreadable. Available 2016-04-25+
    for most backtests; trades before that date will fail any dial-based filter
    (NaN > threshold evaluates False).
    """
    path = "data/rd2_fragility.parquet"
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df.empty:
        return None
    df.index = pd.to_datetime(df.index).normalize()
    try:
        df.index = df.index.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return df.sort_index()

def clean_ticker_df(df):
    if df.empty: return df
    if isinstance(df.columns, pd.MultiIndex):
        if 'Close' in df.columns.get_level_values(1) or 'Adj Close' in df.columns.get_level_values(1):
             df.columns = df.columns.get_level_values(1)
        else:
             df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().capitalize() for c in df.columns]
    if 'Close' not in df.columns and 'Adj close' in df.columns:
        df.rename(columns={'Adj close': 'Close'}, inplace=True)
    if 'Close' not in df.columns:
        return pd.DataFrame() 
    df = df.dropna(subset=['Close'])
    return df

def build_xsec_rank_matrices(data_dict, windows=[5, 10, 21]):
    """Build cross-sectional percentile rank matrices from downloaded data.

    For each return window, first computes each ticker's temporal percentile
    rank (expanding, min 252 days — same as indicators.py), then ranks those
    percentiles across tickers on each date. This normalizes for volatility:
    a stock at its 95th %ile of its own history is treated the same whether
    it's a low-vol utility or a high-vol biotech.

    Returns dict {window: DataFrame} where each DataFrame has Date index and
    ticker columns with values 0-100.
    """
    RANK_MIN_PERIODS = 252
    rank_dict = {}
    for ticker, df in data_dict.items():
        if 'Close' not in df.columns or len(df) < 50:
            continue
        for w in windows:
            ret = df['Close'].pct_change(w)
            # Temporal percentile: where is this return vs ticker's own history
            temporal_pctile = ret.expanding(min_periods=RANK_MIN_PERIODS).rank(pct=True) * 100.0
            rank_dict.setdefault(w, {})[ticker] = temporal_pctile

    result = {}
    for w in windows:
        if not rank_dict.get(w):
            continue
        mat = pd.DataFrame(rank_dict[w])
        # Rank the temporal percentiles across tickers on each date
        result[w] = mat.rank(axis=1, pct=True) * 100.0
    return result

def build_mtm_curves(trades_df, data_dict, starting_equity, risk_bps, mode='flat'):
    """Build daily MTM equity curves with intraday H/L envelopes.

    Chronologically walks trades in EntryDate order, sizing each at either a
    flat fraction of starting_equity or a dynamic fraction of running equity.
    For each day, computes the aggregate close-based equity and the best/worst
    intraday equity bounds using daily H/L of every open position.

    Returns a DataFrame indexed by Date with columns:
        Equity_Close, Equity_High, Equity_Low, InMarket (bool)
    """
    if trades_df.empty:
        return pd.DataFrame(columns=['Equity_Close', 'Equity_High', 'Equity_Low', 'InMarket'])

    trades = trades_df.copy()
    trades['EntryDate'] = pd.to_datetime(trades['EntryDate'])
    trades['ExitDate']  = pd.to_datetime(trades['ExitDate'])
    trades = trades[trades['Ticker'].isin(data_dict.keys())].sort_values('EntryDate').reset_index(drop=True)
    if trades.empty:
        return pd.DataFrame(columns=['Equity_Close', 'Equity_High', 'Equity_Low', 'InMarket'])

    # Pre-slice OHLC for each trade's holding period; override exit-day close with
    # the realized exit price so stop/target fills are honored.
    trade_prices = {}
    for idx, tr in trades.iterrows():
        df = data_dict[tr['Ticker']]
        mask = (df.index >= tr['EntryDate']) & (df.index <= tr['ExitDate'])
        sub = df.loc[mask, ['Open', 'High', 'Low', 'Close']].copy() if all(c in df.columns for c in ['Open','High','Low','Close']) else None
        if sub is None or sub.empty:
            trade_prices[idx] = None
            continue
        exit_dt = tr['ExitDate']
        if exit_dt in sub.index:
            sub.loc[exit_dt, 'Close'] = tr['Exit']
            if tr['Exit'] > sub.loc[exit_dt, 'High']: sub.loc[exit_dt, 'High'] = tr['Exit']
            if tr['Exit'] < sub.loc[exit_dt, 'Low']:  sub.loc[exit_dt, 'Low']  = tr['Exit']
        trade_prices[idx] = sub

    # Build calendar: ALL trading days in [min_entry, max_exit] from the full data_dict,
    # not just days spanned by positions. Without this, gap days (no active positions)
    # are absent from the index entirely — which silently forces Time-in-Market to 100%
    # and reduces the denominator for Sharpe/Parkinson vol.
    min_dt = trades['EntryDate'].min()
    max_dt = trades['ExitDate'].max()
    cal_set = set()
    for _df in data_dict.values():
        if _df is None or _df.empty:
            continue
        cal_set.update(_df.index[(_df.index >= min_dt) & (_df.index <= max_dt)])
    if not cal_set:
        return pd.DataFrame(columns=['Equity_Close', 'Equity_High', 'Equity_Low', 'InMarket'])
    all_dates = sorted(cal_set)

    entries_by_date = {}
    for idx, tr in trades.iterrows():
        entries_by_date.setdefault(tr['EntryDate'], []).append(idx)

    flat_risk_dollars = starting_equity * risk_bps / 10000.0
    equity = float(starting_equity)

    eq_close, eq_high, eq_low, in_mkt = [], [], [], []
    active = {}  # idx -> {'shares','entry_price','direction','prev_close_val'}

    for date in all_dates:
        # 1. Open today's new trades, sized against equity at prior close
        for idx in entries_by_date.get(date, []):
            sub = trade_prices.get(idx)
            if sub is None or sub.empty:
                continue
            tr = trades.iloc[idx]
            tech_risk = tr['TechRisk']
            if not (tech_risk and tech_risk > 0):
                continue
            risk_scale = float(tr['RiskScale']) if 'RiskScale' in trades.columns and pd.notna(tr.get('RiskScale')) else 1.0
            risk_dollars = (flat_risk_dollars if mode == 'flat' else equity * risk_bps / 10000.0) * risk_scale
            shares = risk_dollars / tech_risk
            if shares <= 0:
                continue
            active[idx] = {
                'shares': shares,
                'entry_price': tr['Entry'],
                'direction': tr['Direction'],
                'prev_close_val': 0.0,
            }

        # 2. Walk each active position and accumulate daily P&L deltas
        chg_close, chg_best, chg_worst = 0.0, 0.0, 0.0
        closing_today = []
        for idx, pos in active.items():
            sub = trade_prices[idx]
            if date not in sub.index:
                continue
            c = sub.at[date, 'Close']
            h = sub.at[date, 'High']
            l = sub.at[date, 'Low']
            if pos['direction'] == 'Long':
                val_c = (c - pos['entry_price']) * pos['shares']
                val_h = (h - pos['entry_price']) * pos['shares']
                val_l = (l - pos['entry_price']) * pos['shares']
            else:  # Short
                val_c = (pos['entry_price'] - c) * pos['shares']
                val_h = (pos['entry_price'] - l) * pos['shares']  # best case short = low print
                val_l = (pos['entry_price'] - h) * pos['shares']  # worst case short = high print
            prev = pos['prev_close_val']
            chg_close += (val_c - prev)
            chg_best  += (val_h - prev)
            chg_worst += (val_l - prev)
            pos['prev_close_val'] = val_c
            if date >= trades.iloc[idx]['ExitDate']:
                closing_today.append(idx)

        start_of_day = equity
        equity += chg_close
        eq_close.append(equity)
        eq_high.append(start_of_day + chg_best)
        eq_low.append(start_of_day + chg_worst)
        in_mkt.append(len(active) > 0)

        for idx in closing_today:
            del active[idx]

    return pd.DataFrame(
        {'Equity_Close': eq_close, 'Equity_High': eq_high, 'Equity_Low': eq_low, 'InMarket': in_mkt},
        index=pd.DatetimeIndex(all_dates, name='Date'),
    )


def compute_portfolio_stats(equity_df, starting_equity, risk_bps=None, trades_df=None):
    """Portfolio-level stats from an MTM equity curve with H/L envelope.

    Uses close-to-close daily returns for Sharpe/Sortino, Parkinson estimator
    on daily H/L for annualized vol (captures intraday swings), and
    peak-to-trough Close for max drawdown. Also reports MaxDD @ Lows which
    uses the low envelope (worst intraday mark) vs close-based running peak,
    with R-unit conversion and time-to-recover metrics.
    """
    empty = {'CAGR_Pct': 0, 'TotalReturn_Pct': 0, 'MaxDD_Pct': 0, 'MaxDD_Low_Pct': 0,
             'MaxDD_R': 0, 'MaxDD_Low_R': 0, 'Sharpe': 0, 'Sharpe_Active': 0,
             'Sortino': 0, 'Sortino_Active': 0, 'Calmar': 0, 'ParkinsonVol_Pct': 0,
             'TimeInMarket_Pct': 0, 'FinalEquity': starting_equity,
             'UnderwaterDays': 0, 'TradesDuringDD': None, 'DDStillOngoing': False,
             'PeakDate': None, 'TroughDate': None, 'RecoveryDate': None}
    if equity_df.empty or len(equity_df) < 2:
        return empty

    close = equity_df['Equity_Close']
    high = equity_df['Equity_High']
    low = equity_df['Equity_Low']

    years = max((equity_df.index[-1] - equity_df.index[0]).days / 365.25, 1e-9)
    total_ret = close.iloc[-1] / starting_equity
    cagr = (total_ret ** (1 / years) - 1) * 100 if total_ret > 0 else -100

    peak = close.cummax()
    dd_series = (close - peak) / peak
    max_dd = dd_series.min() * 100

    # Max DD @ Lows: intraday low vs close-based running peak. More conservative
    # (deeper) than C-C drawdown — reflects the worst mark you actually saw.
    dd_low_series = (low - peak) / peak
    max_dd_low = dd_low_series.min() * 100

    # R-unit conversion: DD_pct / risk_bps_as_pct = # of R. Equivalent to
    # dd_dollars (at starting-equity basis) / risk-per-trade.
    if risk_bps and risk_bps > 0:
        max_dd_R = abs(max_dd) * 100.0 / risk_bps
        max_dd_low_R = abs(max_dd_low) * 100.0 / risk_bps
    else:
        max_dd_R = 0.0
        max_dd_low_R = 0.0

    # Underwater duration + time-to-recover for the C-C max drawdown
    peak_date = trough_date = recovery_date = None
    underwater_days = 0
    trades_during_dd = None
    still_underwater = False
    if dd_series.min() < 0:
        trough_date = dd_series.idxmin()
        # Peak date = last date at-or-before trough where close equals its running peak
        _peak_mask = (equity_df.index <= trough_date) & (close == peak)
        peak_date = equity_df.index[_peak_mask][-1] if _peak_mask.any() else equity_df.index[0]
        peak_value = close.loc[peak_date]
        # Recovery = first date after trough where close reaches back to peak value
        _post = close.loc[trough_date:]
        _recov_mask = _post >= peak_value
        if _recov_mask.any():
            recovery_date = _post[_recov_mask].index[0]
            underwater_days = (recovery_date - peak_date).days
        else:
            # Still underwater at the end of the backtest
            still_underwater = True
            recovery_date = None
            underwater_days = (equity_df.index[-1] - peak_date).days

        if trades_df is not None and not trades_df.empty and 'EntryDate' in trades_df.columns:
            _tc = trades_df.copy()
            _ed = pd.to_datetime(_tc['EntryDate'], errors='coerce')
            end_cut = recovery_date if recovery_date is not None else equity_df.index[-1]
            trades_during_dd = int(((_ed >= peak_date) & (_ed <= end_cut)).sum())

    daily_ret = close.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    downside = daily_ret[daily_ret < 0]
    sortino = (daily_ret.mean() / downside.std() * np.sqrt(252)) if len(downside) > 1 and downside.std() > 0 else 0

    # Active-period Sharpe/Sortino: compute on in-market days only. More relevant
    # for ensemble-book sizing — answers "what's the quality of returns *when deployed*",
    # ignoring the cost of idle days (which other strategies in the book will fill).
    # Relationship: sharpe_calendar ≈ sharpe_active * sqrt(TIM).
    if 'InMarket' in equity_df.columns:
        # Align InMarket to daily_ret (both drop first NaN row of pct_change)
        active_mask = equity_df['InMarket'].iloc[1:].reset_index(drop=True)
        active_ret = daily_ret.reset_index(drop=True)[active_mask.values]
        if len(active_ret) > 1 and active_ret.std() > 0:
            sharpe_active = active_ret.mean() / active_ret.std() * np.sqrt(252)
            ds_active = active_ret[active_ret < 0]
            sortino_active = (active_ret.mean() / ds_active.std() * np.sqrt(252)) if len(ds_active) > 1 and ds_active.std() > 0 else 0
        else:
            sharpe_active, sortino_active = 0, 0
    else:
        sharpe_active, sortino_active = sharpe, sortino

    # Parkinson vol from aggregate daily H/L envelope of the equity curve itself
    valid = (high > 0) & (low > 0) & (high >= low)
    if valid.any():
        log_hl = np.log(high[valid] / low[valid])
        park_daily = np.sqrt((1.0 / (4 * np.log(2))) * (log_hl ** 2).mean())
        park_annual = park_daily * np.sqrt(252) * 100
    else:
        park_annual = 0

    calmar = (cagr / abs(max_dd)) if max_dd != 0 else 0
    tim_pct = equity_df['InMarket'].sum() / len(equity_df) * 100 if 'InMarket' in equity_df.columns else 0

    return {
        'CAGR_Pct': cagr,
        'TotalReturn_Pct': (total_ret - 1) * 100,
        'MaxDD_Pct': max_dd,
        'MaxDD_Low_Pct': max_dd_low,
        'MaxDD_R': max_dd_R,
        'MaxDD_Low_R': max_dd_low_R,
        'Sharpe': sharpe,
        'Sharpe_Active': sharpe_active,
        'Sortino': sortino,
        'Sortino_Active': sortino_active,
        'Calmar': calmar,
        'ParkinsonVol_Pct': park_annual,
        'TimeInMarket_Pct': tim_pct,
        'FinalEquity': close.iloc[-1],
        'UnderwaterDays': underwater_days,
        'TradesDuringDD': trades_during_dd,
        'DDStillOngoing': still_underwater,
        'PeakDate': peak_date,
        'TroughDate': trough_date,
        'RecoveryDate': recovery_date,
    }


def compute_trade_path_stats(trades_df, data_dict):
    """Compute per-trade MAE/MFE/Give-back/Capture using daily H/L.

    Entry-day handling is pessimistic: only adverse excursion counts
    (Long -> day's Low; Short -> day's High). Favorable side is zeroed
    because we don't know where in the bar the fill happened. For MOC
    entries (entry_price == entry-day close) the entry day is skipped
    entirely — the bar was complete before we got in. Middle days use
    full H/L both sides. Exit day uses full H/L (peak-after-exit is a
    minor overstatement flagged in the UI).

    Returns the trades_df with four new columns: MAE_R, MFE_R,
    GiveBack_R, Capture_Pct.
    """
    if trades_df.empty:
        return trades_df

    trades = trades_df.copy()
    # Dates may have been stringified already by upstream; normalize
    e_dt = pd.to_datetime(trades['EntryDate'])
    x_dt = pd.to_datetime(trades['ExitDate'])

    mae_R = np.full(len(trades), np.nan)
    mfe_R = np.full(len(trades), np.nan)

    for i, (_, tr) in enumerate(trades.iterrows()):
        ticker = tr['Ticker']
        if ticker not in data_dict:
            continue
        df = data_dict[ticker]
        if not all(c in df.columns for c in ['High', 'Low', 'Close']):
            continue
        entry_dt = e_dt.iloc[i]
        exit_dt  = x_dt.iloc[i]
        entry_px = float(tr['Entry'])
        tech_risk = float(tr['TechRisk']) if tr['TechRisk'] else 0.0
        if tech_risk <= 0:
            continue
        direction = tr['Direction']

        mask = (df.index >= entry_dt) & (df.index <= exit_dt)
        hold = df.loc[mask, ['High', 'Low', 'Close']]
        if hold.empty:
            continue

        worst_excursion = 0.0  # most adverse (negative)
        best_excursion  = 0.0  # most favorable (positive)
        for j, (dt, row) in enumerate(hold.iterrows()):
            h, l, c = float(row['High']), float(row['Low']), float(row['Close'])
            is_entry_day = (dt == entry_dt)
            is_moc_entry = is_entry_day and abs(entry_px - c) < 1e-9

            if is_moc_entry:
                continue  # bar completed before entry

            if direction == 'Long':
                adverse = l - entry_px   # negative when worse
                favor   = h - entry_px   # positive when better
            else:  # Short
                adverse = entry_px - h   # negative when worse (stock rallied)
                favor   = entry_px - l   # positive when better

            if is_entry_day:
                # Pessimistic: count adverse, zero favorable
                worst_excursion = min(worst_excursion, adverse)
            else:
                worst_excursion = min(worst_excursion, adverse)
                best_excursion  = max(best_excursion, favor)

        mae_R[i] = worst_excursion / tech_risk
        mfe_R[i] = best_excursion / tech_risk

    trades['MAE_R'] = mae_R
    trades['MFE_R'] = mfe_R
    trades['GiveBack_R'] = trades['MFE_R'] - trades['R']
    trades['Capture_Pct'] = np.where(
        trades['MFE_R'] > 0,
        trades['R'] / trades['MFE_R'] * 100,
        np.nan,
    )
    return trades


@st.cache_data(show_spinner=True)
def download_universe_data(tickers, fetch_start_date):
    if not tickers: return {} 
    clean_tickers = [str(t).strip().upper() for t in tickers if str(t).strip() != '']
    if not clean_tickers: return {}
    if isinstance(fetch_start_date, datetime.date):
        start_str = fetch_start_date.strftime('%Y-%m-%d')
    else:
        start_str = fetch_start_date 
    data_dict = {}
    CHUNK_SIZE = 50 
    total_tickers = len(clean_tickers)
    download_bar = st.progress(0)
    status_text = st.empty()
    for i in range(0, total_tickers, CHUNK_SIZE):
        chunk = clean_tickers[i:i + CHUNK_SIZE]
        status_text.text(f"Downloading batch {i} to {min(i+CHUNK_SIZE, total_tickers)} of {total_tickers}...")
        download_bar.progress(min((i + CHUNK_SIZE) / total_tickers, 1.0))
        try:
            df = yf.download(chunk, start=start_str, group_by='ticker', auto_adjust=True, progress=False, threads=True)
            if df.empty: continue
            if len(chunk) == 1:
                t = chunk[0]
                t_df = clean_ticker_df(df)
                if not t_df.empty:
                    t_df.index = t_df.index.normalize().tz_localize(None)
                    data_dict[t] = t_df
            else:
                for t in chunk:
                    try:
                        if t in df.columns.levels[0]:
                            t_df = df[t].copy()
                            t_df = clean_ticker_df(t_df)
                            if not t_df.empty:
                                t_df.index = t_df.index.normalize().tz_localize(None)
                                data_dict[t] = t_df
                    except: continue
            time.sleep(0.1)
        except Exception as e:
            if "rate limited" in str(e).lower(): break
            continue
    download_bar.empty()
    status_text.empty()
    return data_dict

def get_cycle_year(year):
    rem = year % 4
    if rem == 0: return "4. Election Year"
    if rem == 1: return "1. Post-Election"
    if rem == 2: return "2. Midterm Year"
    if rem == 3: return "3. Pre-Election"
    return "Unknown"

def get_age_bucket(years):
    if years < 3: return "< 3 Years"
    if years < 5: return "3-5 Years"
    if years < 10: return "5-10 Years"
    if years < 20: return "10-20 Years"
    return "> 20 Years"

def grade_strategy(pf, sqn, win_rate, total_trades):
    score = 0
    reasons = []
    if pf >= 2.0: score += 4
    elif pf >= 1.5: score += 3
    elif pf >= 1.2: score += 2
    elif pf >= 1.0: score += 1
    else: score -= 5 
    if sqn >= 3.0: score += 4
    elif sqn >= 2.0: score += 3
    elif sqn >= 1.5: score += 2
    elif sqn > 0: score += 1
    if total_trades < 30: score -= 2
    if score >= 7: return "A", "Excellent", reasons
    if score >= 5: return "B", "Good", reasons
    if score >= 3: return "C", "Marginal", reasons
    if score >= 0: return "D", "Poor", reasons
    return "F", "Uninvestable", reasons

def _infer_strategy_type(params):
    direction = params.get('trade_direction', 'Long')
    perf_filters = params.get('perf_filters', [])
    has_oversold = any(f['logic'] == '<' and f['thresh'] <= 33 for f in perf_filters)
    has_overbought = any(f['logic'] == '>' and f['thresh'] >= 67 for f in perf_filters)
    has_52w = params.get('use_52w', False)
    has_breakout = params.get('breakout_mode', 'None') != 'None'
    has_sznl = params.get('use_sznl', False) or params.get('use_market_sznl', False)
    
    if has_52w and params.get('52w_type') == 'New 52w High':
        strat_type = "Breakout"
        thesis = "Momentum continuation after new highs"
    elif has_breakout and "High" in params.get('breakout_mode', ''):
        strat_type = "Breakout"
        thesis = "Breakout above previous range"
    elif has_oversold and direction == 'Long':
        strat_type = "MeanReversion"
        thesis = "Oversold bounce setup"
    elif has_overbought and direction == 'Short':
        strat_type = "MeanReversion"
        thesis = "Overbought fade setup"
    elif has_sznl and not (has_oversold or has_overbought):
        strat_type = "Seasonal"
        thesis = "Seasonal tendency play"
    elif has_overbought and direction == 'Long':
        strat_type = "Momentum"
        thesis = "Momentum continuation in strong names"
    else:
        strat_type = "Custom"
        thesis = "[EDIT: Add thesis]"
    
    if has_sznl and strat_type != "Seasonal":
        sznl_logic = params.get('sznl_logic', '>')
        sznl_thresh = params.get('sznl_thresh', 50)
        if sznl_logic == '>' and sznl_thresh >= 65:
            thesis += " with strong seasonal tailwind"
        elif sznl_logic == '<' and sznl_thresh <= 33:
            thesis += " despite weak seasonality (contrarian)"
    
    trend = params.get('trend_filter', 'None')
    if "200 SMA" in trend and ">" in trend:
        thesis += " in uptrending names"
    elif "200 SMA" in trend and "<" in trend:
        thesis += " in downtrending names"
    
    return strat_type, thesis

def _generate_key_filters(params):
    filters = []
    direction = params.get('trade_direction', 'Long')
    
    for pf in params.get('perf_filters', []):
        consec_str = f" ({pf['consecutive']}d consecutive)" if pf.get('consecutive', 1) > 1 else ""
        if pf['logic'] == 'Between':
            filters.append(f"{pf['window']}D rank between {pf['thresh']:.0f}-{pf.get('thresh_max', 100):.0f}th %ile{consec_str}")
        elif pf['logic'] == 'Not Between':
            filters.append(f"{pf['window']}D rank NOT between {pf['thresh']:.0f}-{pf.get('thresh_max', 100):.0f}th %ile{consec_str}")
        else:
            filters.append(f"{pf['window']}D rank {pf['logic']} {pf['thresh']:.0f}th %ile{consec_str}")

    for paf in params.get('perf_atr_filters', []):
        consec_str = f" ({paf['consecutive']}d consecutive)" if paf.get('consecutive', 1) > 1 else ""
        if paf['logic'] == 'Between':
            filters.append(f"{paf['window']}D ATR rank between {paf['thresh']:.0f}-{paf.get('thresh_max', 100):.0f}th %ile{consec_str}")
        elif paf['logic'] == 'Not Between':
            filters.append(f"{paf['window']}D ATR rank NOT between {paf['thresh']:.0f}-{paf.get('thresh_max', 100):.0f}th %ile{consec_str}")
        else:
            filters.append(f"{paf['window']}D ATR rank {paf['logic']} {paf['thresh']:.0f}th %ile{consec_str}")

    for asf in params.get('atr_sznl_filters', []):
        consec_str = f" ({asf['consecutive']}d consecutive)" if asf.get('consecutive', 1) > 1 else ""
        if asf['logic'] == 'Between':
            filters.append(f"{asf['window']}D ATR seasonal rank between {asf['thresh']:.0f}-{asf.get('thresh_max', 100):.0f}th %ile{consec_str}")
        elif asf['logic'] == 'Not Between':
            filters.append(f"{asf['window']}D ATR seasonal rank NOT between {asf['thresh']:.0f}-{asf.get('thresh_max', 100):.0f}th %ile{consec_str}")
        else:
            filters.append(f"{asf['window']}D ATR seasonal rank {asf['logic']} {asf['thresh']:.0f}th %ile{consec_str}")
    
    for maf in params.get('ma_consec_filters', []):
        filters.append(f"Close {maf['logic'].lower()} {maf['length']} SMA ({maf['consec']}d consecutive)")
    
    if params.get('use_sznl'):
        filters.append(f"Ticker seasonal rank {params['sznl_logic']} {params['sznl_thresh']:.0f}")
    if params.get('use_market_sznl'):
        filters.append(f"Market seasonal {params['market_sznl_logic']} {params['market_sznl_thresh']:.0f}")
    
    if params.get('use_52w'):
        first_str = " (first in {0}d)".format(params.get('52w_lookback', 21)) if params.get('52w_first_instance') else ""
        filters.append(f"{params['52w_type']}{first_str}")
    if params.get('exclude_52w_high'):
        filters.append("NOT at 52-week high")
    if params.get('use_ath'):
        filters.append(params.get('ath_type', 'Today is ATH'))
    if params.get('use_recent_ath'):
        prefix = "Has NOT made" if params.get('recent_ath_invert') else "Made"
        filters.append(f"{prefix} ATH in last {params['ath_lookback_days']} days")
    if params.get('use_recent_52w'):
        prefix = "Has NOT made" if params.get('recent_52w_invert') else "Made"
        filters.append(f"{prefix} 52w high in last {params['recent_52w_lookback']} days")
    if params.get('use_recent_52w_low'):
        prefix = "Has NOT made" if params.get('recent_52w_low_invert') else "Made"
        filters.append(f"{prefix} 52w low in last {params['recent_52w_low_lookback']} days")
    if params.get('breakout_mode', 'None') != 'None':
        filters.append(params['breakout_mode'])
    
    if params.get('require_close_gt_open'):
        filters.append("Close > Open (green candle)")
    
    if params.get('use_range_filter'):
        filters.append(f"Close in {params['range_min']}-{params['range_max']}% of daily range")
    
    if params.get('use_atr_ret_filter'):
        filters.append(f"Net change between {params['atr_ret_min']:.1f} and {params['atr_ret_max']:.1f} ATR")
    
    if params.get('use_range_atr_filter'):
        logic = params.get('range_atr_logic', 'Between')
        if logic == '>':
            filters.append(f"Today's range > {params['range_atr_min']:.1f} ATR")
        elif logic == '<':
            filters.append(f"Today's range < {params['range_atr_max']:.1f} ATR")
        else:
            filters.append(f"Today's range between {params['range_atr_min']:.1f}-{params['range_atr_max']:.1f} ATR")
    day_label = {0: "Signal day", 1: "T-1", 2: "T-2", 3: "T-3", 4: "T-4", 5: "T-5"}
    for pa in params.get('price_action_filters', []):
        dl = day_label.get(pa.get('lag', 0), f"T-{pa.get('lag', 0)}")
        pa_type = pa['type']
        if pa_type == 'range_pct':
            filters.append(f"{dl}: Close in {pa['min']:.0f}-{pa['max']:.0f}% of range")
        elif pa_type == 'atr_ret':
            filters.append(f"{dl}: Net change {pa['min']:.1f} to {pa['max']:.1f} ATR")
        elif pa_type == 'range_atr':
            logic = pa.get('logic', 'Between')
            if logic == '>':
                filters.append(f"{dl}: Range > {pa['min']:.1f} ATR")
            elif logic == '<':
                filters.append(f"{dl}: Range < {pa['max']:.1f} ATR")
            else:
                filters.append(f"{dl}: Range {pa['min']:.1f}-{pa['max']:.1f} ATR")
        elif pa_type == 'close_gt_open':
            filters.append(f"{dl}: Close > Open (green)")
        elif pa_type == 'close_lt_open':
            filters.append(f"{dl}: Close < Open (red)")
        elif pa_type == 'close_gt_prev_high':
            filters.append(f"{dl}: Close > Prev High")
        elif pa_type == 'close_lt_prev_low':
            filters.append(f"{dl}: Close < Prev Low")
    if params.get('use_ma_dist_filter'):
        ma_type = params.get('dist_ma_type', 'SMA 200')
        logic = params.get('dist_logic', 'Between')
        if logic == "Greater Than (>)":
            filters.append(f"Distance from {ma_type} > {params['dist_min']:.1f} ATR")
        elif logic == "Less Than (<)":
            filters.append(f"Distance from {ma_type} < {params['dist_max']:.1f} ATR")
        else:
            filters.append(f"Distance from {ma_type} between {params['dist_min']:.1f}-{params['dist_max']:.1f} ATR")
    
    if params.get('use_vol'):
        if params.get('vol_logic', '>') == 'Between':
            filters.append(f"Volume between {params['vol_thresh']:.1f}x and {params.get('vol_thresh_max', 10.0):.1f}x 63-day avg")
        else:
            filters.append(f"Volume {params.get('vol_logic', '>')} {params['vol_thresh']:.1f}x 63-day avg")
    if params.get('use_vol_rank'):
        filters.append(f"10D vol rank {params['vol_rank_logic']} {params['vol_rank_thresh']:.0f}th %ile")
    if params.get('vol_gt_prev'):
        filters.append("Volume > previous day")
    
    if params.get('use_acc_count_filter'):
        filters.append(f"Acc days {params['acc_count_logic']} {params['acc_count_thresh']} in last {params['acc_count_window']}d")
    if params.get('use_dist_count_filter'):
        filters.append(f"Dist days {params['dist_count_logic']} {params['dist_count_thresh']} in last {params['dist_count_window']}d")
    
    if params.get('use_gap_filter'):
        filters.append(f"Gap count {params['gap_logic']} {params['gap_thresh']} in last {params['gap_lookback']}d")
    
    trend = params.get('trend_filter', 'None')
    if trend != 'None':
        filters.append(f"Trend: {trend}")
    
    if params.get('use_vix_filter'):
        filters.append(f"VIX between {params['vix_min']:.0f}-{params['vix_max']:.0f}")
    
    if params.get('use_ref_ticker_filter') and params.get('ref_filters'):
        ref_ticker = params.get('ref_ticker', 'IWM')
        for rf in params['ref_filters']:
            filters.append(f"{ref_ticker} {rf['window']}D rank {rf['logic']} {rf['thresh']:.0f}th %ile")

    if params.get('use_xsec_filter') and params.get('xsec_filters'):
        for xf in params['xsec_filters']:
            consec_str = f" ({xf['consecutive']}d consecutive)" if xf.get('consecutive', 1) > 1 else ""
            if xf['logic'] == 'Between':
                filters.append(f"XSec {xf['window']}D rank between {xf['thresh']:.0f}-{xf.get('thresh_max', 100):.0f}th %ile{consec_str}")
            elif xf['logic'] == 'Not Between':
                filters.append(f"XSec {xf['window']}D rank NOT between {xf['thresh']:.0f}-{xf.get('thresh_max', 100):.0f}th %ile{consec_str}")
            else:
                filters.append(f"XSec {xf['window']}D rank {xf['logic']} {xf['thresh']:.0f}th %ile{consec_str}")

    if params.get('use_weekly_ma_pullback'):
        filters.append(f"First touch of Weekly {params['wma_type']}{params['wma_period']} after {params['wma_min_ext_pct']:.0f}%+ extension (lookback {params['wma_lookback_months']}mo, {params['wma_touch_logic']})")

    if params.get('use_dow_filter'):
        day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
        days_str = ", ".join([day_names.get(d, '?') for d in params.get('allowed_days', [])])
        filters.append(f"Entry days: {days_str}")

    if 'allowed_cycles' in params and len(params.get('allowed_cycles', [])) < 4:
        cycle_names = {0: 'Election', 1: 'Post-Election', 2: 'Midterm', 3: 'Pre-Election'}
        cycles_str = ", ".join([cycle_names.get(c, '?') for c in params['allowed_cycles']])
        filters.append(f"Cycle years: {cycles_str}")

    if params.get('excluded_years'):
        yrs = sorted(params['excluded_years'])
        filters.append(f"Excluded years: {', '.join(str(y) for y in yrs)}")
    
    if params.get('use_t1_open_filter'):
        for f in params.get('t1_open_filters', []):
            offset_str = f" {'+' if f['atr_offset'] >= 0 else ''}{f['atr_offset']} ATR" if f['atr_offset'] != 0 else ""
            filters.append(f"T+1 Open {f['logic']} {f['reference']}{offset_str}")
    
    return filters if filters else ["[EDIT: Add key filters]"]

def _generate_exit_summary(params):
    direction = params.get('trade_direction', 'Long')
    stop_atr = params.get('stop_atr', 2.0)
    tgt_atr = params.get('tgt_atr', 5.0)
    hold_days = params.get('holding_days', 10)
    use_stop = params.get('use_stop_loss', True)
    use_tgt = params.get('use_take_profit', True)
    
    use_partial = params.get('use_partial_exits', False) and use_tgt
    tgt_frac = float(params.get('partial_target_fraction', 0.5)) if use_partial else 0.0

    if not use_stop and not use_tgt:
        primary = f"{hold_days}-day time stop"
    elif use_partial and use_stop:
        primary = f"{int(tgt_frac*100)}% at target, {int((1-tgt_frac)*100)}% at {hold_days}d time stop (or stop if first)"
    elif use_partial:
        primary = f"{int(tgt_frac*100)}% at target, {int((1-tgt_frac)*100)}% at {hold_days}d time stop"
    elif use_tgt and use_stop:
        primary = f"Target, Stop, or {hold_days}-day time stop"
    elif use_tgt:
        primary = f"Target or {hold_days}-day time stop"
    else:
        primary = f"Stop or {hold_days}-day time stop"
    
    if use_stop:
        if direction == 'Long':
            stop_logic = f"{stop_atr:.1f} ATR below entry"
        else:
            stop_logic = f"{stop_atr:.1f} ATR above entry"
    else:
        stop_logic = "None (time exit only)"
    
    if use_tgt:
        if direction == 'Long':
            tgt_logic = f"{tgt_atr:.1f} ATR above entry"
        else:
            tgt_logic = f"{tgt_atr:.1f} ATR below entry"
    else:
        tgt_logic = "None (time exit only)"
    
    return {
        "primary_exit": primary,
        "stop_logic": stop_logic,
        "target_logic": tgt_logic,
        "notes": None
    }

def _infer_timeframe(params):
    hold_days = params.get('holding_days', 10)
    entry_type = params.get('entry_type', '')
    
    if "Overnight" in entry_type or "Intraday" in entry_type or "Day Trade" in entry_type:
        return "Intraday"
    elif hold_days <= 2:
        return "Overnight"
    elif hold_days <= 10:
        return "Swing"
    else:
        return "Position"

def build_strategy_dict(params, tickers_to_run, pf, sqn, win_rate, expectancy_r):
    id_parts = []
    if params.get('perf_filters'):
        perf_str = "+".join([f"{f['window']}d {f['logic']} {f['thresh']:.0f}%ile" for f in params['perf_filters']])
        id_parts.append(perf_str)
    if params.get('perf_atr_filters'):
        patr_str = "+".join([f"{f['window']}d ATR {f['logic']} {f['thresh']:.0f}%ile" for f in params['perf_atr_filters']])
        id_parts.append(patr_str)
    if params.get('use_sznl'): id_parts.append(f"Sznl {params['sznl_logic']} {params['sznl_thresh']:.0f}")
    if params.get('use_acc_count_filter'): id_parts.append(f"{params['acc_count_thresh']} acc {params['acc_count_logic']} in {params['acc_count_window']}d")
    if params.get('use_dist_count_filter'): id_parts.append(f"{params['dist_count_thresh']} dist {params['dist_count_logic']} in {params['dist_count_window']}d")
    if params.get('use_ref_ticker_filter') and params.get('ref_filters'):
        ref_ticker = params.get('ref_ticker', 'IWM')
        ref_str = "+".join([f"{ref_ticker} {f['window']}d {f['logic']} {f['thresh']:.0f}%ile" for f in params['ref_filters']])
        id_parts.append(ref_str)
    if params.get('use_xsec_filter') and params.get('xsec_filters'):
        xsec_str = "+".join([f"XSec {f['window']}d {f['logic']} {f['thresh']:.0f}%ile" for f in params['xsec_filters']])
        id_parts.append(xsec_str)
    if params.get('use_t1_open_filter') and params.get('t1_open_filters'):
        for f in params['t1_open_filters']:
            t1_str = f"T+1 Open {f['logic']} {f['reference']}"
            if f['atr_offset'] != 0: t1_str += f" {'+' if f['atr_offset'] > 0 else ''}{f['atr_offset']} ATR"
            id_parts.append(t1_str)
    id_parts.append(f"Entry: {params['entry_type']}")
    id_parts.append(f"{params['holding_days']}d hold")
    strategy_id = ", ".join(id_parts) if id_parts else "Custom Strategy"
    
    grade, verdict, _ = grade_strategy(pf, sqn, win_rate, 100)
    strat_type, thesis = _infer_strategy_type(params)
    key_filters = _generate_key_filters(params)
    timeframe = _infer_timeframe(params)
    exit_summary = _generate_exit_summary(params)
    
    strategy = {
        "id": strategy_id,
        "name": "[EDIT: Strategy Name]",
        "setup": {
            "type": strat_type,
            "timeframe": timeframe,
            "thesis": thesis,
            "key_filters": key_filters
        },
        "exit_summary": exit_summary,
        "description": f"Backtest: {params['backtest_start_date']} to present. Tested on {len(tickers_to_run)} tickers.",
        "universe_tickers": "[EDIT: LIQUID_UNIVERSE or other]",
        "settings": {
            "trade_direction": params.get('trade_direction', 'Long'),
            "entry_type": params.get('entry_type', 'T+1 Open'),
            "max_one_pos": params.get('max_one_pos', True),
            "allow_same_day_reentry": params.get('allow_same_day_reentry', False),
            "max_daily_entries": params.get('max_daily_entries', 2),
            "max_total_positions": params.get('max_total_positions', 10),
            "entry_conf_bps": params.get('entry_conf_bps', 0),
            # Performance rank filters
            "perf_filters": params.get('perf_filters', []),
            "perf_atr_filters": params.get('perf_atr_filters', []),
            "perf_first_instance": params.get('perf_first_instance', False),
            "perf_lookback": params.get('perf_lookback', 21),
            # MA consecutive filters
            "ma_consec_filters": params.get('ma_consec_filters', []),
            # Seasonal
            "use_sznl": params.get('use_sznl', False), "sznl_logic": params.get('sznl_logic', '<'), "sznl_thresh": params.get('sznl_thresh', 33.0),
            "sznl_first_instance": params.get('sznl_first_instance', False), "sznl_lookback": params.get('sznl_lookback', 21),
            "use_market_sznl": params.get('use_market_sznl', False), "market_sznl_logic": params.get('market_sznl_logic', '<'), "market_sznl_thresh": params.get('market_sznl_thresh', 40.0),
            "market_ticker": "^GSPC",
            # 52-week / ATH
            "use_52w": params.get('use_52w', False), "52w_type": params.get('52w_type', 'New 52w High'), "52w_first_instance": params.get('52w_first_instance', True),
            "52w_lookback": params.get('52w_lookback', 21), "52w_lag": params.get('52w_lag', 0),
            "exclude_52w_high": params.get('exclude_52w_high', False),
            "use_ath": params.get('use_ath', False), "ath_type": params.get('ath_type', 'Today is ATH'),
            "use_recent_ath": params.get('use_recent_ath', False), "recent_ath_invert": params.get('recent_ath_invert', False), "ath_lookback_days": params.get('ath_lookback_days', 21),
            "use_recent_52w": params.get('use_recent_52w', False), "recent_52w_invert": params.get('recent_52w_invert', False), "recent_52w_lookback": params.get('recent_52w_lookback', 21),
            "use_recent_52w_low": params.get('use_recent_52w_low', False), "recent_52w_low_invert": params.get('recent_52w_low_invert', False), "recent_52w_low_lookback": params.get('recent_52w_low_lookback', 21),
            # Price action
            "breakout_mode": params.get('breakout_mode', 'None'),
            "require_close_gt_open": params.get('require_close_gt_open', False),
            "use_range_filter": params.get('use_range_filter', False), "range_min": params.get('range_min', 0), "range_max": params.get('range_max', 100),
            "use_atr_ret_filter": params.get('use_atr_ret_filter', False), "atr_ret_min": params.get('atr_ret_min', 0.0), "atr_ret_max": params.get('atr_ret_max', 1.0),
            "use_range_atr_filter": params.get('use_range_atr_filter', False), "range_atr_logic": params.get('range_atr_logic', 'Between'), "range_atr_min": params.get('range_atr_min', 1.0), "range_atr_max": params.get('range_atr_max', 3.0),
            # Multi-day price action
            "price_action_filters": params.get('price_action_filters', []),
            # Distance from MA
            "use_ma_dist_filter": params.get('use_ma_dist_filter', False), "dist_ma_type": params.get('dist_ma_type', 'SMA 200'), "dist_logic": params.get('dist_logic', 'Between'), "dist_min": params.get('dist_min', 0.0), "dist_max": params.get('dist_max', 2.0),
            # Weekly MA Pullback
            "use_weekly_ma_pullback": params.get('use_weekly_ma_pullback', False), "wma_type": params.get('wma_type', 'EMA'), "wma_period": params.get('wma_period', 8),
            "wma_min_ext_pct": params.get('wma_min_ext_pct', 30.0), "wma_lookback_months": params.get('wma_lookback_months', 6), "wma_touch_logic": params.get('wma_touch_logic', 'Low <= MA'),
            # Volume
            "vol_gt_prev": params.get('vol_gt_prev', False),
            "use_vol": params.get('use_vol', False), "vol_logic": params.get('vol_logic', '>'), "vol_thresh": params.get('vol_thresh', 1.5), "vol_thresh_max": params.get('vol_thresh_max', 10.0),
            "use_vol_rank": params.get('use_vol_rank', False), "vol_rank_logic": params.get('vol_rank_logic', '<'), "vol_rank_thresh": params.get('vol_rank_thresh', 15.0),
            # Acc/Dist counts
            "use_acc_count_filter": params.get('use_acc_count_filter', False), "acc_count_window": params.get('acc_count_window', 21), "acc_count_logic": params.get('acc_count_logic', '>'), "acc_count_thresh": params.get('acc_count_thresh', 3),
            "use_dist_count_filter": params.get('use_dist_count_filter', False), "dist_count_window": params.get('dist_count_window', 21), "dist_count_logic": params.get('dist_count_logic', '>'), "dist_count_thresh": params.get('dist_count_thresh', 3),
            # Gap filter
            "use_gap_filter": params.get('use_gap_filter', False), "gap_lookback": params.get('gap_lookback', 21), "gap_logic": params.get('gap_logic', '>'), "gap_thresh": params.get('gap_thresh', 3),
            # Trend & regime
            "trend_filter": params.get('trend_filter', 'None'),
            "use_vix_filter": params.get('use_vix_filter', False), "vix_min": params.get('vix_min', 0.0), "vix_max": params.get('vix_max', 20.0),
            # Liquidity & data
            "min_price": params.get('min_price', 10.0), "min_vol": params.get('min_vol', 100000),
            "min_age": params.get('min_age', 0.25), "max_age": params.get('max_age', 100.0),
            "min_atr_pct": params.get('min_atr_pct', 0.0), "max_atr_pct": params.get('max_atr_pct', 10.0),
            # Time & cycle
            "use_dow_filter": params.get('use_dow_filter', False), "allowed_days": params.get('allowed_days', [0, 1, 2, 3, 4]),
            "allowed_cycles": params.get('allowed_cycles', [0, 1, 2, 3]),
            "excluded_years": params.get('excluded_years', []),
            # Reference ticker
            "use_ref_ticker_filter": params.get('use_ref_ticker_filter', False), "ref_ticker": params.get('ref_ticker', 'IWM'), "ref_filters": params.get('ref_filters', []),
            # T+1 open filter
            "use_t1_open_filter": params.get('use_t1_open_filter', False), "t1_open_filters": params.get('t1_open_filters', []),
            # Cross-sectional rank
            "use_xsec_filter": params.get('use_xsec_filter', False), "xsec_filters": params.get('xsec_filters', []),
            # ATR seasonal ranks
            "atr_sznl_filters": params.get('atr_sznl_filters', [])
        },
        "execution": {
            "risk_bps": 35,
            "risk_per_trade": "[EDIT: calculated from account size]",
            "slippage_bps": params.get('slippage_bps', 5),
            "stop_atr": params.get('stop_atr', 2.0),
            "tgt_atr": params.get('tgt_atr', 5.0),
            "hold_days": params.get('holding_days', 10),
            "use_stop_loss": params.get('use_stop_loss', True),
            "use_take_profit": params.get('use_take_profit', True)
        },
        "stats": {
            "grade": f"{grade} ({verdict})",
            "win_rate": f"{win_rate:.1f}%",
            "expectancy": f"{expectancy_r:.2f}r",
            "profit_factor": f"{pf:.2f}"
        }
    }
    return strategy
    
def run_engine(universe_dict, params, sznl_map, market_series=None, vix_series=None, market_sznl_series=None, ref_ticker_ranks=None, xsec_rank_matrices=None, atr_sznl_map=None, fragility_df=None, market_sma_not_declining_series=None):
    all_potential_trades = []
    total = len(universe_dict)
    bt_start_ts = pd.to_datetime(params['backtest_start_date'])
    direction = params.get('trade_direction', 'Long')
    max_one_pos_per_ticker = params.get('max_one_pos', True)
    slippage_bps = params.get('slippage_bps', 5)
    
    # --- ENTRY MODES ---
    entry_mode = params['entry_type']
    is_pullback = "Pullback" in entry_mode
    is_limit_pers_025 = "Limit Order -0.25 ATR" in entry_mode
    is_limit_pers_05 = "Limit Order -0.5 ATR" in entry_mode
    is_limit_pers_10 = "Limit Order -1 ATR" in entry_mode
    is_limit_atr = entry_mode == "Limit (Close -0.5 ATR)"
    is_limit_prev = entry_mode == "Limit (Prev Close)"
    is_limit_open_atr = entry_mode == "Limit (Open +/- 0.5 ATR)"
    is_limit_open_atr_075 = entry_mode == "Limit (Open +/- 0.75 ATR)"
    is_limit_open_atr_gtc = entry_mode == "Limit (Open +/- 0.5 ATR) GTC"
    is_day_trade_limit = entry_mode == "Day Trade (Limit Open +/- 0.5 ATR, Exit Close)"
    is_limit_pivot = entry_mode == "Limit (Untested Pivot)"
    is_gap_up = "Gap Up Only" in entry_mode
    is_overnight = "Overnight" in entry_mode
    is_intraday = "Intraday" in entry_mode
    is_cond_close_lower = "T+1 Close if < Signal Close" in entry_mode
    is_cond_close_higher = "T+1 Close if > Signal Close" in entry_mode
    
    pullback_col = None
    if "10 SMA" in entry_mode: pullback_col = "SMA10"
    elif "21 EMA" in entry_mode: pullback_col = "EMA21"
    pullback_use_level = "Level" in entry_mode
    
    gap_window = params.get('gap_lookback', 21)
    total_signals_generated = 0
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    req_custom_mas = list(set([f['length'] for f in params.get('ma_consec_filters', [])]))
    acc_win = params['acc_count_window'] if params.get('use_acc_count_filter', False) else None
    dist_win = params['dist_count_window'] if params.get('use_dist_count_filter', False) else None
    use_t1_open_filter = params.get('use_t1_open_filter', False)
    t1_open_filters = params.get('t1_open_filters', [])

    # Weekly MA Pullback configs
    weekly_ma_configs = None
    if params.get('use_weekly_ma_pullback', False):
        weekly_ma_configs = [{'type': params['wma_type'], 'period': params['wma_period']}]

    # Reference Ticker Filter params
    use_ref_ticker_filter = params.get('use_ref_ticker_filter', False)
    ref_filters = params.get('ref_filters', [])

    # Cross-Sectional Rank Filter params
    use_xsec_filter = params.get('use_xsec_filter', False)
    xsec_filters = params.get('xsec_filters', [])

    for i, (ticker, df_raw) in enumerate(universe_dict.items()):
        status_text.text(f"Scanning signals for {ticker}...")
        progress_bar.progress((i+1)/total)

        if ticker == MARKET_TICKER and MARKET_TICKER not in params.get('universe_tickers', []): continue
        if ticker == VIX_TICKER: continue
        if len(df_raw) < 100: continue

        ticker_last_exit = pd.Timestamp.min

        try:
            df = calculate_indicators(df_raw, sznl_map, ticker, market_series, vix_series, market_sznl_series, gap_window, req_custom_mas, acc_win, dist_win, ref_ticker_ranks, weekly_ma_configs, xsec_rank_matrices)
            if market_sma_not_declining_series is not None:
                df['Market_SMA200_Not_Declining'] = market_sma_not_declining_series.reindex(df.index, method='ffill').fillna(False)
            df = df[df.index >= bt_start_ts]
            if df.empty: continue

            # Merge ATR seasonal ranks (if available) — 6 rank columns joined by date
            if atr_sznl_map and ticker in atr_sznl_map:
                atr_ranks = atr_sznl_map[ticker]
                df_dates = df.index.normalize()
                for col in ATR_SZNL_COLS:
                    if col in atr_ranks.columns:
                        df[col] = atr_ranks[col].reindex(df_dates).values
            
            # --- SIGNAL GENERATION ---
            conditions = []
            trend_opt = params.get('trend_filter', 'None')
            
            if trend_opt == "Price > 200 SMA": conditions.append(df['Close'] > df['SMA200'])
            elif trend_opt == "Price > Rising 200 SMA": conditions.append((df['Close'] > df['SMA200']) & (df['SMA200'] > df['SMA200'].shift(1)))
            elif trend_opt == "Not Below Declining 200 SMA": conditions.append(~((df['Close'] < df['SMA200']) & (df['SMA200'] < df['SMA200'].shift(1))))
            elif trend_opt == "Price < 200 SMA": conditions.append(df['Close'] < df['SMA200'])
            elif trend_opt == "Price < Falling 200 SMA": conditions.append((df['Close'] < df['SMA200']) & (df['SMA200'] < df['SMA200'].shift(1)))
            elif "Market" in trend_opt and 'Market_Above_SMA200' in df.columns:
                if trend_opt == "Market > 200 SMA": conditions.append(df['Market_Above_SMA200'])
                elif trend_opt == "Market < 200 SMA": conditions.append(~df['Market_Above_SMA200'])
                elif trend_opt == "Market Not Declining 200 SMA" and 'Market_SMA200_Not_Declining' in df.columns:
                    conditions.append(df['Market_SMA200_Not_Declining'])
                
            conditions.append((df['Close'] >= params['min_price']) & (df['vol_ma'] >= params['min_vol']) & (df['age_years'] >= params['min_age']) & (df['age_years'] <= params['max_age']) & (df['ATR_Pct'] >= params['min_atr_pct']) & (df['ATR_Pct'] <= params['max_atr_pct']))
            
            if params.get('require_close_gt_open', False): conditions.append(df['Close'] > df['Open'])
            
            bk_mode = params.get('breakout_mode', 'None')
            if bk_mode == "Close > Prev Day High": conditions.append(df['Close'] > df['High'].shift(1))
            elif bk_mode == "Close < Prev Day Low": conditions.append(df['Close'] < df['Low'].shift(1))
            
            if params.get('use_range_filter', False): conditions.append((df['RangePct'] * 100 >= params['range_min']) & (df['RangePct'] * 100 <= params['range_max']))
            
            if params.get('use_atr_ret_filter', False):
                conditions.append((df['today_return_atr'] >= params['atr_ret_min']) & (df['today_return_atr'] <= params['atr_ret_max']))
            if params.get('use_range_atr_filter', False):
                range_in_atr = (df['High'] - df['Low']) / df['ATR']
                if params['range_atr_logic'] == '>':
                    conditions.append(range_in_atr > params['range_atr_min'])
                elif params['range_atr_logic'] == '<':
                    conditions.append(range_in_atr < params['range_atr_max'])
                elif params['range_atr_logic'] == 'Between':
                    conditions.append((range_in_atr >= params['range_atr_min']) & (range_in_atr <= params['range_atr_max']))
            # --- MULTI-DAY PRICE ACTION FILTERS ---
            for pa in params.get('price_action_filters', []):
                pa_lag = pa.get('lag', 0)
                pa_type = pa['type']
                if pa_type == 'range_pct':
                    series = df['RangePct'].shift(pa_lag) * 100
                    conditions.append((series >= pa['min']) & (series <= pa['max']))
                elif pa_type == 'atr_ret':
                    series = df['today_return_atr'].shift(pa_lag)
                    conditions.append((series >= pa['min']) & (series <= pa['max']))
                elif pa_type == 'range_atr':
                    series = ((df['High'] - df['Low']) / df['ATR']).shift(pa_lag)
                    pa_logic = pa.get('logic', 'Between')
                    if pa_logic == '>':
                        conditions.append(series > pa['min'])
                    elif pa_logic == '<':
                        conditions.append(series < pa['max'])
                    else:
                        conditions.append((series >= pa['min']) & (series <= pa['max']))
                elif pa_type == 'close_gt_open':
                    conditions.append(df['Close'].shift(pa_lag) > df['Open'].shift(pa_lag))
                elif pa_type == 'close_lt_open':
                    conditions.append(df['Close'].shift(pa_lag) < df['Open'].shift(pa_lag))
                elif pa_type == 'close_gt_prev_high':
                    conditions.append(df['Close'].shift(pa_lag) > df['High'].shift(pa_lag + 1))
                elif pa_type == 'close_lt_prev_low':
                    conditions.append(df['Close'].shift(pa_lag) < df['Low'].shift(pa_lag + 1))
            if params.get('use_dow_filter', False): conditions.append(df['DayOfWeekVal'].isin(params['allowed_days']))

            if params.get('use_month_filter', False):
                allowed_months = params.get('allowed_months', list(range(1, 13)))
                conditions.append(pd.Series(df.index.month, index=df.index).isin(allowed_months))

            if 'allowed_cycles' in params and len(params['allowed_cycles']) < 4:
                year_rems = df.index.year % 4
                conditions.append(pd.Series(year_rems, index=df.index).isin(params['allowed_cycles']))

            _excluded_years = params.get('excluded_years', [])
            if _excluded_years:
                conditions.append(~pd.Series(df.index.year, index=df.index).isin(_excluded_years))
                
            if params.get('use_gap_filter', False):
                if params['gap_logic'] == ">": conditions.append(df['GapCount'] > params['gap_thresh'])
                elif params['gap_logic'] == "<": conditions.append(df['GapCount'] < params['gap_thresh'])
                elif params['gap_logic'] == "=": conditions.append(df['GapCount'] == params['gap_thresh'])

            if params.get('use_acc_count_filter', False):
                v2_prefix = "AccCount_v2_" if params.get('use_acc_dist_v2', False) else "AccCount_"
                col = f"{v2_prefix}{params['acc_count_window']}"
                if col in df.columns:
                    r_acc = df[col]
                    if params['acc_count_logic'] == ">": conditions.append(r_acc > params['acc_count_thresh'])
                    elif params['acc_count_logic'] == "<": conditions.append(r_acc < params['acc_count_thresh'])
                    elif params['acc_count_logic'] == "=": conditions.append(r_acc == params['acc_count_thresh'])

            if params.get('use_dist_count_filter', False):
                v2_prefix = "DistCount_v2_" if params.get('use_acc_dist_v2', False) else "DistCount_"
                col = f"{v2_prefix}{params['dist_count_window']}"
                if col in df.columns:
                    r_dist = df[col]
                    if params['dist_count_logic'] == ">": conditions.append(r_dist > params['dist_count_thresh'])
                    elif params['dist_count_logic'] == "<": conditions.append(r_dist < params['dist_count_thresh'])
                    elif params['dist_count_logic'] == "=": conditions.append(r_dist == params['dist_count_thresh'])
            
            for pf in params.get('perf_filters', []):
                col = f"rank_ret_{pf['window']}d"
                if pf['logic'] == '<': c_f = (df[col] < pf['thresh'])
                elif pf['logic'] == '>': c_f = (df[col] > pf['thresh'])
                elif pf['logic'] == 'Between': c_f = (df[col] >= pf['thresh']) & (df[col] <= pf.get('thresh_max', 100.0))
                elif pf['logic'] == 'Not Between': c_f = (df[col] < pf['thresh']) | (df[col] > pf.get('thresh_max', 100.0))
                else: continue
                if pf['consecutive'] > 1: c_f = c_f.rolling(pf['consecutive']).sum() == pf['consecutive']
                conditions.append(c_f)

            for paf in params.get('perf_atr_filters', []):
                col = f"rank_ret_atr_{paf['window']}d"
                if col not in df.columns: continue
                if paf['logic'] == '<': c_f = (df[col] < paf['thresh'])
                elif paf['logic'] == '>': c_f = (df[col] > paf['thresh'])
                elif paf['logic'] == 'Between': c_f = (df[col] >= paf['thresh']) & (df[col] <= paf.get('thresh_max', 100.0))
                elif paf['logic'] == 'Not Between': c_f = (df[col] < paf['thresh']) | (df[col] > paf.get('thresh_max', 100.0))
                else: continue
                if paf.get('consecutive', 1) > 1: c_f = c_f.rolling(paf['consecutive']).sum() == paf['consecutive']
                conditions.append(c_f)

            for asf in params.get('atr_sznl_filters', []):
                col = f"atr_sznl_{asf['window']}d"
                if col not in df.columns: continue
                if asf['logic'] == '<': c_f = (df[col] < asf['thresh'])
                elif asf['logic'] == '>': c_f = (df[col] > asf['thresh'])
                elif asf['logic'] == 'Between': c_f = (df[col] >= asf['thresh']) & (df[col] <= asf.get('thresh_max', 100.0))
                elif asf['logic'] == 'Not Between': c_f = (df[col] < asf['thresh']) | (df[col] > asf.get('thresh_max', 100.0))
                else: continue
                if asf.get('consecutive', 1) > 1: c_f = c_f.rolling(asf['consecutive']).sum() == asf['consecutive']
                conditions.append(c_f)

            # --- CROSS-SECTIONAL RANK FILTER ---
            if use_xsec_filter and xsec_filters:
                for xf in xsec_filters:
                    col = f"xsec_rank_ret_{xf['window']}d"
                    if col in df.columns:
                        if xf['logic'] == '<': c_f = (df[col] < xf['thresh'])
                        elif xf['logic'] == '>': c_f = (df[col] > xf['thresh'])
                        elif xf['logic'] == 'Between': c_f = (df[col] >= xf['thresh']) & (df[col] <= xf.get('thresh_max', 100.0))
                        elif xf['logic'] == 'Not Between': c_f = (df[col] < xf['thresh']) | (df[col] > xf.get('thresh_max', 100.0))
                        else: continue
                        if xf.get('consecutive', 1) > 1: c_f = c_f.rolling(xf['consecutive']).sum() == xf['consecutive']
                        conditions.append(c_f)

            # --- OR FILTER GROUPS (at least one condition in each group must be true) ---
            for group in params.get('or_filter_groups', []):
                group_conds = []
                for cond in group:
                    ctype = cond.get('type', 'perf')
                    window = cond['window']
                    if ctype == 'perf':
                        col = f"rank_ret_{window}d"
                    elif ctype == 'xsec':
                        col = f"xsec_rank_ret_{window}d"
                    else:
                        continue
                    if col not in df.columns:
                        continue
                    if cond['logic'] == '<':
                        group_conds.append(df[col] < cond['thresh'])
                    elif cond['logic'] == '>':
                        group_conds.append(df[col] > cond['thresh'])
                if group_conds:
                    or_result = group_conds[0]
                    for gc in group_conds[1:]:
                        or_result = or_result | gc
                    conditions.append(or_result)

            for f in params.get('ma_consec_filters', []):
                col = f"SMA{f['length']}"
                mask = (df['Close'] > df[col]) if f['logic'] == 'Above' else (df['Close'] < df[col])
                if f['consec'] > 1: mask = mask.rolling(f['consec']).sum() == f['consec']
                conditions.append(mask)

            if params.get('use_sznl', False):
                sznl_cond = (df['Sznl'] < params['sznl_thresh']) if params['sznl_logic'] == '<' else (df['Sznl'] > params['sznl_thresh'])
                if params.get('sznl_first_instance', False): sznl_cond = apply_first_instance_filter(sznl_cond, params.get('sznl_lookback', 21))
                conditions.append(sznl_cond)

            if params.get('use_market_sznl', False) and 'Market_Sznl' in df.columns:
                market_sznl_cond = (df['Market_Sznl'] < params['market_sznl_thresh']) if params['market_sznl_logic'] == '<' else (df['Market_Sznl'] > params['market_sznl_thresh'])
                conditions.append(market_sznl_cond)
                
            if params.get('use_52w', False):
                c_52_raw = df['is_52w_high'] if params['52w_type'] == 'New 52w High' else df['is_52w_low']
                lag_days = params.get('52w_lag', 0)
                if lag_days > 0: c_52_raw = c_52_raw.shift(lag_days).fillna(False)
                if params.get('52w_first_instance', False): c_52 = apply_first_instance_filter(c_52_raw, params.get('52w_lookback', 21))
                else: c_52 = c_52_raw
                conditions.append(c_52)
                
            if params.get('exclude_52w_high', False): 
                conditions.append(~df['is_52w_high'])
                
            if params.get('use_ath', False):
                if params['ath_type'] == 'Today is ATH':
                    conditions.append(df['is_ath'])
                else:  # Today is NOT ATH
                    conditions.append(~df['is_ath'])
            if params.get('use_recent_ath', False):
                ath_lookback = params.get('ath_lookback_days', 21)
                recent_ath_mask = df['is_ath'].rolling(window=ath_lookback, min_periods=1).max().astype(bool)
                if params.get('recent_ath_invert', False):
                    conditions.append(~recent_ath_mask)
                else:
                    conditions.append(recent_ath_mask)
            if params.get('use_recent_52w', False):
                r52w_lookback = params.get('recent_52w_lookback', 21)
                recent_52w_mask = df['is_52w_high'].rolling(window=r52w_lookback, min_periods=1).max().astype(bool)
                if params.get('recent_52w_invert', False):
                    conditions.append(~recent_52w_mask)
                else:
                    conditions.append(recent_52w_mask)
            if params.get('use_recent_52w_low', False):
                r52w_low_lookback = params.get('recent_52w_low_lookback', 21)
                recent_52w_low_mask = df['is_52w_low'].rolling(window=r52w_low_lookback, min_periods=1).max().astype(bool)
                if params.get('recent_52w_low_invert', False):
                    conditions.append(~recent_52w_low_mask)
                else:
                    conditions.append(recent_52w_low_mask)
            if params.get('vol_gt_prev', False): 
                conditions.append(df['Volume'] > df['Volume'].shift(1))
            if params.get('use_vol', False):
                _vl = params.get('vol_logic', '>')
                if _vl == '<':
                    conditions.append(df['vol_ratio'] < params['vol_thresh'])
                elif _vl == 'Between':
                    conditions.append((df['vol_ratio'] >= params['vol_thresh']) & (df['vol_ratio'] <= params.get('vol_thresh_max', 10.0)))
                else:
                    conditions.append(df['vol_ratio'] > params['vol_thresh'])
            
            if params.get('use_vol_rank', False):
                vr_col = 'vol_ratio_10d_rank'
                if params['vol_rank_logic'] == ">": conditions.append(df[vr_col] > params['vol_rank_thresh'])
                elif params['vol_rank_logic'] == "<": conditions.append(df[vr_col] < params['vol_rank_thresh'])

            if params.get('use_ma_dist_filter', False):
                ma_col_map = {"SMA 10": "SMA10", "SMA 20": "SMA20", "SMA 50": "SMA50", "SMA 100": "SMA100", "SMA 200": "SMA200", "EMA 8": "EMA8", "EMA 11": "EMA11", "EMA 21": "EMA21","52-Week High": "High_52w", "All-Time High": "ATH_Level"}
                ma_target = ma_col_map.get(params['dist_ma_type'])
                if ma_target and ma_target in df.columns:
                    dist_val = (df['Close'] - df[ma_target]) / df['ATR']
                    if params['dist_logic'] == "Greater Than (>)": conditions.append(dist_val > params['dist_min'])
                    elif params['dist_logic'] == "Less Than (<)": conditions.append(dist_val < params['dist_max'])
                    elif params['dist_logic'] == "Between": conditions.append((dist_val >= params['dist_min']) & (dist_val <= params['dist_max']))
                    
            if params.get('use_vix_filter', False) and 'VIX_Value' in df.columns:
                vix_val = df['VIX_Value']
                conditions.append((vix_val >= params['vix_min']) & (vix_val <= params['vix_max']))

            # --- WEEKLY MA PULLBACK FILTER (first touch after extension) ---
            if params.get('use_weekly_ma_pullback', False):
                wma_col = f"Weekly_{params['wma_type']}{params['wma_period']}"
                if wma_col in df.columns:
                    lookback_days = params['wma_lookback_months'] * 21
                    pct_above = (df['High'] / df[wma_col] - 1) * 100

                    # Staleness guard: extension must have occurred within lookback window
                    max_ext = pct_above.rolling(lookback_days, min_periods=21).max()
                    recently_extended = max_ext >= params['wma_min_ext_pct']

                    # Touch condition
                    if params.get('wma_touch_logic', 'Low <= MA') == 'Low <= MA':
                        is_touch = df['Low'] <= df[wma_col]
                    else:
                        is_touch = df['Close'] <= df[wma_col]

                    # First-touch-after-extension: only fire on the FIRST day
                    # the stock crosses below the MA since it was extended.
                    # Uses cumsum trick: extension days increment a counter;
                    # a touch is "new" only if that counter advanced since the
                    # previous touch.
                    is_extended = pct_above >= params['wma_min_ext_pct']
                    ext_cumsum = is_extended.astype(int).cumsum()
                    touch_ext_count = ext_cumsum.where(is_touch)
                    prev_touch_ext = touch_ext_count.ffill().shift(1).fillna(0)
                    first_touch = is_touch & (ext_cumsum > prev_touch_ext) & recently_extended

                    conditions.append(first_touch)

            # --- REFERENCE TICKER FILTER ---
            if use_ref_ticker_filter and ref_filters:
                for rf in ref_filters:
                    col = f"Ref_rank_ret_{rf['window']}d"
                    if col in df.columns:
                        if rf['logic'] == '<':
                            conditions.append(df[col] < rf['thresh'])
                        elif rf['logic'] == '>':
                            conditions.append(df[col] > rf['thresh'])
                
            if use_t1_open_filter and t1_open_filters:
                for t1_filter in t1_open_filters:
                    ref_col = t1_filter['reference']
                    atr_offset = t1_filter['atr_offset']
                    logic = t1_filter['logic']
                    threshold = df[ref_col] + (atr_offset * df['ATR'])
                    if logic == '>': t1_cond = df['NextOpen'] > threshold
                    elif logic == '>=': t1_cond = df['NextOpen'] >= threshold
                    elif logic == '<': t1_cond = df['NextOpen'] < threshold
                    elif logic == '<=': t1_cond = df['NextOpen'] <= threshold
                    else: continue
                    conditions.append(t1_cond)

            # Risk dial filters — apply fragility score thresholds on signal date
            dial_filters = params.get('dial_filters', [])
            if dial_filters and fragility_df is not None and not fragility_df.empty:
                for df_filter in dial_filters:
                    dial_col = df_filter.get('dial')
                    if dial_col not in fragility_df.columns:
                        continue
                    win = max(1, int(df_filter.get('window', 1)))
                    dial_series = fragility_df[dial_col]
                    if win > 1:
                        dial_series = dial_series.rolling(win, min_periods=win).mean()
                    dial_on_date = dial_series.reindex(df.index, method='ffill')
                    thresh = float(df_filter.get('thresh', 0))
                    logic = df_filter.get('logic', '>')
                    vals = dial_on_date.values
                    if logic == '>':    cond = (vals > thresh)
                    elif logic == '<':  cond = (vals < thresh)
                    elif logic == '>=': cond = (vals >= thresh)
                    elif logic == '<=': cond = (vals <= thresh)
                    else: continue
                    # NaN dial (pre-2016 or missing) → pass through, treat as if
                    # the dial filter doesn't apply for those trades. Lets the
                    # full historical sample contribute to backtest stats while
                    # still gating modern-era trades on the dial criterion.
                    cond = np.where(np.isnan(vals), True, cond)
                    conditions.append(pd.Series(cond, index=df.index))

            if not conditions: continue

            final_signal = conditions[0]
            for c in conditions[1:]: final_signal = final_signal & c
            
            if params.get('perf_first_instance', False) and len(params.get('perf_filters', [])) > 0:
                final_signal = apply_first_instance_filter(final_signal, params.get('perf_lookback', 21))
                
            signal_dates = df.index[final_signal]
            total_signals_generated += len(signal_dates)
            
            # --- EXECUTION LOGIC ---
            for signal_date in signal_dates:
                if max_one_pos_per_ticker and signal_date <= ticker_last_exit: continue
                
                sig_idx = df.index.get_loc(signal_date)
                if sig_idx + 1 >= len(df): continue
                
                found_entry, actual_entry_idx, actual_entry_price = False, -1, 0.0
                
                # ========== ENTRY LOGIC SECTION ==========
                if is_overnight:
                    found_entry, actual_entry_idx, actual_entry_price = True, sig_idx, df['Close'].iloc[sig_idx]
                elif is_intraday:
                    found_entry, actual_entry_idx, actual_entry_price = True, sig_idx + 1, df['Open'].iloc[sig_idx + 1]
                elif is_pullback and pullback_col:
                    for wait_i in range(1, params['holding_days'] + 1):
                        curr_idx = sig_idx + wait_i
                        if curr_idx >= len(df): break
                        if df.iloc[curr_idx]['Low'] <= df.iloc[curr_idx][pullback_col]:
                            found_entry, actual_entry_idx = True, curr_idx
                            actual_entry_price = df.iloc[curr_idx][pullback_col] if pullback_use_level else df.iloc[curr_idx]['Close']
                            break
                elif is_gap_up:
                    if df['Open'].iloc[sig_idx + 1] > df['High'].iloc[sig_idx]:
                        found_entry, actual_entry_idx, actual_entry_price = True, sig_idx + 1, df['Open'].iloc[sig_idx + 1]
                elif is_limit_pers_025 or is_limit_pers_05 or is_limit_pers_10:
                    atr_mult = 0.25 if is_limit_pers_025 else (0.5 if is_limit_pers_05 else 1.0)
                    sig_close, sig_atr = df['Close'].iloc[sig_idx], df['ATR'].iloc[sig_idx]
                    limit_price = (sig_close - (sig_atr * atr_mult)) if direction == 'Long' else (sig_close + (sig_atr * atr_mult))
                    for wait_i in range(1, params['holding_days'] + 1):
                        curr_idx = sig_idx + wait_i
                        if curr_idx >= len(df): break
                        day_low, day_high, day_open = df['Low'].iloc[curr_idx], df['High'].iloc[curr_idx], df['Open'].iloc[curr_idx]
                        filled, fill_px = False, limit_price
                        if direction == 'Long':
                            if day_low <= limit_price: filled = True; fill_px = day_open if day_open < limit_price else limit_price
                        else:
                            if day_high >= limit_price: filled = True; fill_px = day_open if day_open > limit_price else limit_price
                        if filled: found_entry, actual_entry_idx, actual_entry_price = True, curr_idx, fill_px; break
                elif is_limit_atr:
                    sig_close, sig_atr = df['Close'].iloc[sig_idx], df['ATR'].iloc[sig_idx]
                    limit_price = (sig_close - (sig_atr * 0.5)) if direction == 'Long' else (sig_close + (sig_atr * 0.5))
                    next_idx = sig_idx + 1
                    if next_idx < len(df):
                        day_low, day_high, day_open = df['Low'].iloc[next_idx], df['High'].iloc[next_idx], df['Open'].iloc[next_idx]
                        if direction == 'Long' and day_low <= limit_price:
                            found_entry, actual_entry_idx, actual_entry_price = True, next_idx, min(limit_price, day_open) if day_open < limit_price else limit_price
                        elif direction == 'Short' and day_high >= limit_price:
                            found_entry, actual_entry_idx, actual_entry_price = True, next_idx, max(limit_price, day_open) if day_open > limit_price else limit_price
                elif is_limit_prev:
                    limit_price = df['Close'].iloc[sig_idx]
                    next_idx = sig_idx + 1
                    if next_idx < len(df):
                        day_low, day_high, day_open = df['Low'].iloc[next_idx], df['High'].iloc[next_idx], df['Open'].iloc[next_idx]
                        if direction == 'Long' and day_low <= limit_price:
                            found_entry, actual_entry_idx, actual_entry_price = True, next_idx, min(limit_price, day_open) if day_open < limit_price else limit_price
                        elif direction == 'Short' and day_high >= limit_price:
                            found_entry, actual_entry_idx, actual_entry_price = True, next_idx, max(limit_price, day_open) if day_open > limit_price else limit_price
                elif is_limit_open_atr:
                    next_idx = sig_idx + 1
                    if next_idx < len(df):
                        sig_atr, day_open = df['ATR'].iloc[sig_idx], df['Open'].iloc[next_idx]
                        day_low, day_high = df['Low'].iloc[next_idx], df['High'].iloc[next_idx]
                        limit_price = (day_open - (sig_atr * 0.5)) if direction == 'Long' else (day_open + (sig_atr * 0.5))
                        if direction == 'Long' and day_low <= limit_price: found_entry, actual_entry_idx, actual_entry_price = True, next_idx, limit_price
                        elif direction == 'Short' and day_high >= limit_price: found_entry, actual_entry_idx, actual_entry_price = True, next_idx, limit_price
                elif is_limit_open_atr_075:
                    next_idx = sig_idx + 1
                    if next_idx < len(df):
                        sig_atr, day_open = df['ATR'].iloc[sig_idx], df['Open'].iloc[next_idx]
                        day_low, day_high = df['Low'].iloc[next_idx], df['High'].iloc[next_idx]
                        limit_price = (day_open - (sig_atr * 0.75)) if direction == 'Long' else (day_open + (sig_atr * 0.75))
                        if direction == 'Long' and day_low <= limit_price: found_entry, actual_entry_idx, actual_entry_price = True, next_idx, limit_price
                        elif direction == 'Short' and day_high >= limit_price: found_entry, actual_entry_idx, actual_entry_price = True, next_idx, limit_price
                elif is_limit_open_atr_gtc:
                    next_idx = sig_idx + 1
                    if next_idx < len(df):
                        sig_atr = df['ATR'].iloc[sig_idx]
                        base_price = df['Open'].iloc[next_idx]
                        limit_price = (base_price - (sig_atr * 0.5)) if direction == 'Long' else (base_price + (sig_atr * 0.5))
                        for wait_i in range(1, params['holding_days'] + 1):
                            curr_idx = sig_idx + wait_i
                            if curr_idx >= len(df): break
                            day_low, day_high, day_open = df['Low'].iloc[curr_idx], df['High'].iloc[curr_idx], df['Open'].iloc[curr_idx]
                            filled, fill_px = False, limit_price
                            if direction == 'Long':
                                if day_open < limit_price: filled, fill_px = True, day_open
                                elif day_low <= limit_price: filled, fill_px = True, limit_price
                            else:
                                if day_open > limit_price: filled, fill_px = True, day_open
                                elif day_high >= limit_price: filled, fill_px = True, limit_price
                            if filled: found_entry, actual_entry_idx, actual_entry_price = True, curr_idx, fill_px; break
                elif is_day_trade_limit:
                    next_idx = sig_idx + 1
                    if next_idx < len(df):
                        sig_atr = df['ATR'].iloc[sig_idx]
                        day_open = df['Open'].iloc[next_idx]
                        day_low, day_high = df['Low'].iloc[next_idx], df['High'].iloc[next_idx]
                        limit_price = (day_open - (sig_atr * 0.5)) if direction == 'Long' else (day_open + (sig_atr * 0.5))
                        if direction == 'Long' and day_low <= limit_price:
                            found_entry, actual_entry_idx, actual_entry_price = True, next_idx, limit_price
                        elif direction == 'Short' and day_high >= limit_price:
                            found_entry, actual_entry_idx, actual_entry_price = True, next_idx, limit_price
                elif is_limit_pivot:
                    sig_close = df['Close'].iloc[sig_idx]
                    sig_atr = df['ATR'].iloc[sig_idx]
                    if direction == 'Long' and 'LastPivotLow' in df.columns:
                        limit_price = df['LastPivotLow'].iloc[sig_idx]
                        # Pivot must be below close and at least 0.5 ATR away
                        if pd.notna(limit_price) and limit_price < sig_close and (sig_close - limit_price) >= 0.5 * sig_atr:
                            for wait_i in range(1, params['holding_days'] + 1):
                                curr_idx = sig_idx + wait_i
                                if curr_idx >= len(df): break
                                day_low, day_open = df['Low'].iloc[curr_idx], df['Open'].iloc[curr_idx]
                                if day_open <= limit_price:
                                    found_entry, actual_entry_idx, actual_entry_price = True, curr_idx, day_open
                                    break
                                elif day_low <= limit_price:
                                    found_entry, actual_entry_idx, actual_entry_price = True, curr_idx, limit_price
                                    break
                    elif direction == 'Short' and 'LastPivotHigh' in df.columns:
                        limit_price = df['LastPivotHigh'].iloc[sig_idx]
                        # Pivot must be above close and at least 0.5 ATR away
                        if pd.notna(limit_price) and limit_price > sig_close and (limit_price - sig_close) >= 0.5 * sig_atr:
                            for wait_i in range(1, params['holding_days'] + 1):
                                curr_idx = sig_idx + wait_i
                                if curr_idx >= len(df): break
                                day_high, day_open = df['High'].iloc[curr_idx], df['Open'].iloc[curr_idx]
                                if day_open >= limit_price:
                                    found_entry, actual_entry_idx, actual_entry_price = True, curr_idx, day_open
                                    break
                                elif day_high >= limit_price:
                                    found_entry, actual_entry_idx, actual_entry_price = True, curr_idx, limit_price
                                    break
                elif is_cond_close_lower:
                    atr_mult = 0.5 if "-0.5 ATR" in entry_mode else (1.0 if "-1 ATR" in entry_mode else 0.0)
                    sig_val, sig_atr = df['Close'].iloc[sig_idx], df['ATR'].iloc[sig_idx]
                    limit_price = sig_val - (atr_mult * sig_atr)
                    next_idx = sig_idx + 1
                    if next_idx < len(df):
                        t1_close = df['Close'].iloc[next_idx]
                        if t1_close < limit_price: found_entry, actual_entry_idx, actual_entry_price = True, next_idx, t1_close
                elif is_cond_close_higher:
                    atr_mult = 0.5 if "+0.5 ATR" in entry_mode else (1.0 if "+1 ATR" in entry_mode else 0.0)
                    sig_val, sig_atr = df['Close'].iloc[sig_idx], df['ATR'].iloc[sig_idx]
                    limit_price = sig_val + (atr_mult * sig_atr)
                    next_idx = sig_idx + 1
                    if next_idx < len(df):
                        t1_close = df['Close'].iloc[next_idx]
                        if t1_close > limit_price: found_entry, actual_entry_idx, actual_entry_price = True, next_idx, t1_close
                else:
                    found_entry = True
                    actual_entry_idx = sig_idx if entry_mode == 'Signal Close' else sig_idx + 1
                    if actual_entry_idx >= len(df): found_entry = False
                    else: actual_entry_price = df['Close'].iloc[actual_entry_idx] if 'Close' in entry_mode else df['Open'].iloc[actual_entry_idx]

                if not found_entry: continue

                atr = df['ATR'].iloc[actual_entry_idx]
                if np.isnan(atr) or atr == 0: continue
                
                # ========== EXIT LOGIC SECTION ==========
                if is_overnight:
                    exit_idx = sig_idx + 1
                    if exit_idx >= len(df): continue
                    exit_date = df.index[exit_idx]
                    exit_price = df['Open'].iloc[exit_idx]
                    exit_type = "Time"
                elif is_intraday:
                    exit_idx = actual_entry_idx
                    exit_date = df.index[exit_idx]
                    exit_price = df['Close'].iloc[exit_idx]
                    exit_type = "Time"
                elif is_day_trade_limit:
                    exit_idx = actual_entry_idx
                    exit_date = df.index[exit_idx]
                    exit_price = df['Close'].iloc[exit_idx]
                    exit_type = "Time (EOD)"
                    day_low, day_high = df['Low'].iloc[exit_idx], df['High'].iloc[exit_idx]
                    stop_price = actual_entry_price - (atr * params['stop_atr']) if direction == 'Long' else actual_entry_price + (atr * params['stop_atr'])
                    tgt_price = actual_entry_price + (atr * params['tgt_atr']) if direction == 'Long' else actual_entry_price - (atr * params['tgt_atr'])
                    if direction == 'Long':
                        if params['use_take_profit'] and day_high >= tgt_price: exit_price, exit_type = tgt_price, "Target"
                        elif params['use_stop_loss'] and day_low <= stop_price: exit_price, exit_type = stop_price, "Stop"
                    else:
                        if params['use_take_profit'] and day_low <= tgt_price: exit_price, exit_type = tgt_price, "Target"
                        elif params['use_stop_loss'] and day_high >= stop_price: exit_price, exit_type = stop_price, "Stop"
                else:
                    fixed_exit_idx = min(actual_entry_idx + params['holding_days'], len(df) - 1)
                    future = df.iloc[actual_entry_idx + 1 : fixed_exit_idx + 1]
                    if future.empty: continue
                    stop_price = actual_entry_price - (atr * params['stop_atr']) if direction == 'Long' else actual_entry_price + (atr * params['stop_atr'])
                    tgt_price = actual_entry_price + (atr * params['tgt_atr']) if direction == 'Long' else actual_entry_price - (atr * params['tgt_atr'])

                    use_partial = params.get('use_partial_exits', False) and params.get('use_take_profit', False)
                    tgt_frac = float(params.get('partial_target_fraction', 0.5)) if use_partial else 0.0

                    exit_price, exit_type, exit_date = actual_entry_price, "Hold", None
                    target_filled = False
                    target_fill_px = None

                    for f_date, f_row in future.iterrows():
                        if direction == 'Long':
                            # Stop check — closes remaining fraction (or full pos if no partial target hit yet)
                            if params['use_stop_loss'] and f_row['Low'] <= stop_price:
                                actual_stop_px = min(f_row['Open'], stop_price) if f_row['Open'] <= stop_price else stop_price
                                if use_partial and target_filled:
                                    exit_price = tgt_frac * target_fill_px + (1 - tgt_frac) * actual_stop_px
                                    exit_type, exit_date = "Target→Stop", f_date
                                else:
                                    exit_price, exit_type, exit_date = actual_stop_px, "Stop", f_date
                                break
                            # Target check
                            if params['use_take_profit'] and f_row['High'] >= tgt_price:
                                actual_tgt_px = max(f_row['Open'], tgt_price) if f_row['Open'] >= tgt_price else tgt_price
                                if not use_partial:
                                    exit_price, exit_type, exit_date = actual_tgt_px, "Target", f_date
                                    break
                                elif not target_filled:
                                    # Book the target leg; remaining fraction keeps walking forward
                                    target_filled = True
                                    target_fill_px = actual_tgt_px
                        else:
                            # Stop check
                            if params['use_stop_loss'] and f_row['High'] >= stop_price:
                                actual_stop_px = max(f_row['Open'], stop_price) if f_row['Open'] >= stop_price else stop_price
                                if use_partial and target_filled:
                                    exit_price = tgt_frac * target_fill_px + (1 - tgt_frac) * actual_stop_px
                                    exit_type, exit_date = "Target→Stop", f_date
                                else:
                                    exit_price, exit_type, exit_date = actual_stop_px, "Stop", f_date
                                break
                            # Target check
                            if params['use_take_profit'] and f_row['Low'] <= tgt_price:
                                actual_tgt_px = min(f_row['Open'], tgt_price) if f_row['Open'] <= tgt_price else tgt_price
                                if not use_partial:
                                    exit_price, exit_type, exit_date = actual_tgt_px, "Target", f_date
                                    break
                                elif not target_filled:
                                    target_filled = True
                                    target_fill_px = actual_tgt_px

                    # No stop fired — if partial target was booked, blend with time-stop close;
                    # otherwise close full position at time stop.
                    if exit_type == "Hold":
                        time_exit_px = future['Close'].iloc[-1]
                        time_exit_date = future.index[-1]
                        if use_partial and target_filled:
                            exit_price = tgt_frac * target_fill_px + (1 - tgt_frac) * time_exit_px
                            exit_type, exit_date = "Target→Time", time_exit_date
                        else:
                            exit_price, exit_date, exit_type = time_exit_px, time_exit_date, "Time"
                    
                ticker_last_exit = exit_date
                slip = slippage_bps / 10000.0
                tech_risk = atr * params['stop_atr']
                if tech_risk <= 0: tech_risk = 0.001
                pnl = (exit_price*(1-slip) - actual_entry_price*(1+slip)) if direction == 'Long' else (actual_entry_price*(1-slip) - exit_price*(1+slip))
                
                all_potential_trades.append({"Ticker": ticker, "SignalDate": signal_date, "EntryDate": df.index[actual_entry_idx], "Direction": direction, "Entry": actual_entry_price, "Exit": exit_price, "ExitDate": exit_date, "Type": exit_type, "R": pnl / tech_risk, "TechRisk": tech_risk, "Age": df['age_years'].iloc[sig_idx], "AvgVol": df['vol_ma'].iloc[sig_idx], "Status": "Valid Signal", "Reason": "Executed"})
        except Exception: continue

    progress_bar.empty()
    status_text.empty()
    
    if not all_potential_trades: return pd.DataFrame(), pd.DataFrame(), 0
    
    pot_df = pd.DataFrame(all_potential_trades).sort_values(by=["EntryDate", "Ticker"])
    final_log, rejected_log, active_pos, daily_count = [], [], [], {}
    max_total, max_daily = params.get('max_total_positions', 100), params.get('max_daily_entries', 100)
    
    for _, trade in pot_df.iterrows():
        active_pos = [t for t in active_pos if t['ExitDate'] > trade['EntryDate']]
        today_num = daily_count.get(trade['EntryDate'], 0)
        if len(active_pos) >= max_total or today_num >= max_daily:
            trade_copy = trade.copy()
            trade_copy['Status'], trade_copy['Reason'] = "Portfolio Rejected", "Constraints"
            rejected_log.append(trade_copy)
        else:
            final_log.append(trade)
            active_pos.append(trade)
            daily_count[trade['EntryDate']] = today_num + 1
            
    return pd.DataFrame(final_log), pd.DataFrame(rejected_log), total_signals_generated
    
def main():
    st.set_page_config(layout="wide", page_title="Quantitative Backtester")
    st.title("Quantitative Strategy Backtester")
    st.markdown("---")
    st.subheader("1. Universe & Data")
    col_u1, col_u2, col_u3 = st.columns([1, 1, 2])
    sample_pct = 100; use_full_history = False
    with col_u1: univ_choice = st.selectbox("Choose Universe", ["All CSV Tickers", "All CSV + Overflow Extras", "Sector ETFs","SPX", "Indices", "International ETFs", "Sector + Index ETFs", "All CSV (Equities Only)", "3x Leveraged (All)", "3x Leveraged Equities", "3x Leveraged Equities (Bull)", "3x Leveraged Equities (Bear)", "3x Leveraged Equities (Broad Only)", "Custom (Upload CSV)"])
    with col_u2:
        default_start = datetime.date(2000, 1, 1)
        start_date = st.date_input("Backtest Start Date", value=default_start, min_value=datetime.date(1950, 1, 1), max_value=datetime.date.today())
        end_date = st.date_input("Backtest End Date", value=datetime.date.today(), min_value=start_date, max_value=datetime.date.today(),
                                 help="Truncate the backtest at this date. Useful for out-of-sample testing — set start to e.g. 2000 and end to 2020 to leave 2020-present untouched as a holdout.")
    custom_tickers = []
    extras_tickers = []
    if univ_choice == "Custom (Upload CSV)":
        with col_u3:
            sample_pct = st.slider("Random Sample %", 1, 100, 100)
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file:
                try:
                    c_df = pd.read_csv(uploaded_file)
                    if "Ticker" in c_df.columns:
                        c_df["Ticker"] = c_df["Ticker"].astype(str).str.strip().str.upper()
                        c_df = c_df[~c_df["Ticker"].isin(["NAN", "NONE", "NULL", ""])]
                        custom_tickers = c_df["Ticker"].unique().tolist()
                        if len(custom_tickers) > 0: st.success(f"Loaded {len(custom_tickers)} valid tickers.")
                except: st.error("Invalid CSV.")
    elif univ_choice == "All CSV + Overflow Extras":
        with col_u3:
            ex_c1, ex_c2 = st.columns([2, 1])
            with ex_c1:
                n_extras = st.number_input(
                    "# Random extras from overflow pool", min_value=0, max_value=2000,
                    value=300, step=50, key="n_overflow_extras",
                    help="Randomly drawn from sznl_ranks.csv (overflow universe), excluding tickers already in the All CSV base. Sample persists across backtests in this session so the cache stays warm — click Reshuffle to regenerate.",
                )
            with ex_c2:
                reshuffle = st.button("🎲 Reshuffle", help="Generate a new random sample from the overflow pool")

            # Build / refresh the sample. Cached in session_state so repeated backtests
            # hit the same ticker list (and thus the same @st.cache_data key).
            _regen = (
                reshuffle
                or 'overflow_extras_sample' not in st.session_state
                or st.session_state.get('_overflow_n') != n_extras
            )
            if _regen:
                try:
                    _of_df = pd.read_csv("sznl_ranks.csv")
                    _of_pool = [str(t).strip().upper() for t in _of_df['ticker'].dropna().unique().tolist()]
                    _base_keys = set(load_seasonal_map().keys())
                    _of_pool = [t for t in _of_pool if t and t not in _base_keys and t not in {"NAN", "NONE", "NULL"}]
                    if n_extras == 0 or n_extras >= len(_of_pool):
                        st.session_state.overflow_extras_sample = sorted(_of_pool)
                    else:
                        st.session_state.overflow_extras_sample = sorted(random.sample(_of_pool, n_extras))
                    st.session_state._overflow_n = n_extras
                except Exception as e:
                    st.error(f"Couldn't read sznl_ranks.csv: {e}")
                    st.session_state.overflow_extras_sample = []
            extras_tickers = st.session_state.get('overflow_extras_sample', [])
            if extras_tickers:
                st.success(f"Sample: {len(extras_tickers)} extras (pinned for this session — reshuffle resets)")
            run_scope = st.radio(
                "Run scope (subset — doesn't affect cache)",
                ["All (base + extras)", "Base only (All CSV)", "Extras only (overflow)"],
                horizontal=True,
                key="overflow_run_scope",
                help="Both base and extras are still downloaded so their caches stay warm. Scope only filters which tickers the strategy is tested on.",
            )
    st.write("")
    use_full_history = st.checkbox("Download Full History (1950+) for Accurate 'Age'", value=False)
    st.markdown("---")
    st.subheader("2. Execution & Risk")
    r_c1, r_c2, r_c3 = st.columns(3)
    with r_c1: trade_direction = st.selectbox("Trade Direction", ["Long", "Short"])
    with r_c2: 
        exit_mode = st.selectbox("Exit Mode", ["Time Only (Hold)", "Standard (Stop & Target)", "No Stop (Target + Time)"])
        use_stop_loss = (exit_mode == "Standard (Stop & Target)")
        use_take_profit = (exit_mode != "Time Only (Hold)")
        time_exit_only = (exit_mode == "Time Only (Hold)")
    with r_c3: max_one_pos = st.checkbox("Max 1 Position/Ticker", value=True)
    p_c1, p_c2, p_c3 = st.columns(3)
    with p_c1: max_daily_entries = st.number_input("Max New Trades Per Day", 1, 100, 20)
    with p_c2: max_total_positions = st.number_input("Max Total Positions", 1, 200, 99)
    with p_c3: slippage_bps = st.number_input("Slippage (bps)", value=2)
    c_re, c_conf = st.columns(2)
    with c_re: allow_same_day_reentry = st.checkbox("Allow Same-Day Re-entry", value=False)
    with c_conf: entry_conf_bps = st.number_input("Entry Confirmation (bps)", value=0)
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: 
        entry_type = st.selectbox("Entry Price", [
            "Limit (Open +/- 0.5 ATR)",
            "Limit (Open +/- 0.75 ATR)",
            "Signal Close", "T+1 Open", "T+1 Close",
            "Overnight (Buy Close, Sell T+1 Open)", "Intraday (Buy Open, Sell Close)",
            "Day Trade (Limit Open +/- 0.5 ATR, Exit Close)",
            "Gap Up Only (Open > Prev High)",
            "Limit Order -0.25 ATR (Persistent)", "Limit Order -0.5 ATR (Persistent)", "Limit Order -1 ATR (Persistent)",
            "Limit (Close -0.5 ATR)", "Limit (Prev Close)",
            "Limit (Open +/- 0.5 ATR) GTC",
            "Limit (Untested Pivot)", 
            "Pullback 10 SMA (Entry: Close)", "Pullback 10 SMA (Entry: Level)", 
            "Pullback 21 EMA (Entry: Close)", "Pullback 21 EMA (Entry: Level)", 
            "T+1 Close if < Signal Close", "T+1 Close if < Signal Close -0.5 ATR", 
            "T+1 Close if < Signal Close -1 ATR", "T+1 Close if > Signal Close", 
            "T+1 Close if > Signal Close +0.5 ATR", "T+1 Close if > Signal Close +1 ATR"
        ])
        use_ma_entry_filter = st.checkbox("Filter: Close > MA - 0.25*ATR", value=False) if "Pullback" in entry_type else False
    with c2: stop_atr = st.number_input("Stop Loss (ATR)", value=1.0, step=0.1, disabled=not use_stop_loss)
    with c3: tgt_atr = st.number_input("Target (ATR)", value=8.0, step=0.1, disabled=not use_take_profit)
    with c4: hold_days = st.number_input("Max Holding Days", min_value=1, value=10, step=1)
    with c5:
        starting_portfolio = st.number_input("Starting Portfolio ($)", value=100000, step=1000, min_value=100)
        risk_bps_input = st.number_input("Risk per Trade (bps)", value=25, step=5, min_value=1, max_value=500,
                                         help="Basis points of starting portfolio risked per trade. 25 bps on $100k = $250/trade.")
        use_max_daily_risk = st.checkbox(
            "Cap Total Daily Risk", value=False,
            help="If a day's signals would risk more than the cap in aggregate, scale every trade that day down pro-rata so total risk = cap. No effect on days under the cap.",
        )
        max_daily_risk_pct = st.number_input(
            "Max Daily Risk (% of equity)", value=3.0, step=0.25, min_value=0.05, max_value=100.0,
            disabled=not use_max_daily_risk,
            help="E.g. 3.0%. With 25bps/trade, 12 trades fit under 3%; a 25-signal day would scale each down by 0.48x.",
        )
    risk_per_trade = starting_portfolio * risk_bps_input / 10000.0

    # --- Partial exits: scale out at target, let remainder ride ---
    with st.expander("Partial Exits (scale out at target)", expanded=False):
        st.markdown(
            "Take profit on a fraction of the position when the target prints, "
            "then let the remainder ride to the time stop. If the stop triggers "
            "before the remaining fraction exits, the remainder exits at the stop."
        )
        pe_c1, pe_c2 = st.columns([1, 2])
        with pe_c1:
            use_partial_exits = st.checkbox(
                "Enable partial profit-taking",
                value=False,
                disabled=not use_take_profit,
            )
            if use_partial_exits and not use_take_profit:
                st.caption("⚠ Requires a profit target (change Exit Mode).")
        with pe_c2:
            partial_target_fraction = st.slider(
                "Fraction exiting at target",
                min_value=0.10, max_value=0.90, value=0.50, step=0.05,
                disabled=not (use_partial_exits and use_take_profit),
                help="The remaining fraction exits at the time stop or, if it fires first, the stop.",
            )
    st.markdown("---")
    st.subheader("3. Signal Criteria")
    with st.expander("Liquidity & Data History Filters", expanded=True):
        l1, l2, l3, l4, l5, l6 = st.columns(6) 
        with l1: min_price = st.number_input("Min Price ($)", value=10.0, step=1.0)
        with l2: min_vol = st.number_input("Min Avg Volume", value=100000, step=50000)
        with l3: min_age = st.number_input("Min True Age (Yrs)", value=0.25, step=0.25)
        with l4: max_age = st.number_input("Max True Age (Yrs)", value=100.0, step=1.0)
        with l5: min_atr_pct = st.number_input("Min ATR %", value=0.2, step=0.1)
        with l6: max_atr_pct = st.number_input("Max ATR %", value=10.0, step=0.1)
    with st.expander("T+1 Open Filter (Next Day Open vs Today's OHLC)", expanded=False):
        st.markdown("**Filter signals based on how the next day opens relative to today's price action.**")
        use_t1_open_filter = st.checkbox("Enable T+1 Open Filter", value=False)
        t1_open_filters = []
        if use_t1_open_filter:
            st.markdown("---\n**Configure up to 3 T+1 Open conditions (AND logic)**")
            t1_c1, t1_c2, t1_c3 = st.columns(3)
            with t1_c1:
                st.markdown("**Condition 1**")
                use_t1_1 = st.checkbox("Enable", key="use_t1_1", value=True)
                t1_logic_1 = st.selectbox("T+1 Open is", [">", ">=", "<", "<="], key="t1_logic_1", disabled=not use_t1_1)
                t1_ref_1 = st.selectbox("Today's", ["High", "Low", "Open", "Close"], key="t1_ref_1", disabled=not use_t1_1)
                t1_atr_1 = st.number_input("ATR Offset (+/-)", value=0.5, step=0.25, min_value=-5.0, max_value=5.0, key="t1_atr_1", disabled=not use_t1_1)
                if use_t1_1: t1_open_filters.append({'logic': t1_logic_1, 'reference': t1_ref_1, 'atr_offset': t1_atr_1})
            with t1_c2:
                st.markdown("**Condition 2**")
                use_t1_2 = st.checkbox("Enable", key="use_t1_2", value=False)
                t1_logic_2 = st.selectbox("T+1 Open is", [">", ">=", "<", "<="], key="t1_logic_2", disabled=not use_t1_2)
                t1_ref_2 = st.selectbox("Today's", ["High", "Low", "Open", "Close"], key="t1_ref_2", index=1, disabled=not use_t1_2)
                t1_atr_2 = st.number_input("ATR Offset (+/-)", value=0.0, step=0.25, min_value=-5.0, max_value=5.0, key="t1_atr_2", disabled=not use_t1_2)
                if use_t1_2: t1_open_filters.append({'logic': t1_logic_2, 'reference': t1_ref_2, 'atr_offset': t1_atr_2})
            with t1_c3:
                st.markdown("**Condition 3**")
                use_t1_3 = st.checkbox("Enable", key="use_t1_3", value=False)
                t1_logic_3 = st.selectbox("T+1 Open is", [">", ">=", "<", "<="], key="t1_logic_3", disabled=not use_t1_3)
                t1_ref_3 = st.selectbox("Today's", ["High", "Low", "Open", "Close"], key="t1_ref_3", index=3, disabled=not use_t1_3)
                t1_atr_3 = st.number_input("ATR Offset (+/-)", value=0.0, step=0.25, min_value=-5.0, max_value=5.0, key="t1_atr_3", disabled=not use_t1_3)
                if use_t1_3: t1_open_filters.append({'logic': t1_logic_3, 'reference': t1_ref_3, 'atr_offset': t1_atr_3})
    with st.expander("Accumulation/Distribution Counts", expanded=False):
        st.markdown("**Filters are additive (AND logic).**")
        acc_dist_version = st.radio("Signal Version", ["v1 (Vol > Prev & > Avg)", "v2 (Vol > 1.25x Prev & > Avg)"], horizontal=True, key="acc_dist_ver")
        use_acc_dist_v2 = acc_dist_version.startswith("v2")
        c_acc, c_dist = st.columns(2)
        with c_acc:
            st.markdown("#### Accumulation Filter")
            use_acc_count_filter = st.checkbox("Enable Acc Count", value=False)
            acc_count_window = st.selectbox("Acc Window", [5, 10, 21, 42], index=2, disabled=not use_acc_count_filter)
            acc_count_logic = st.selectbox("Acc Logic", [">", "<", "="], disabled=not use_acc_count_filter)
            acc_count_thresh = st.number_input("Acc Threshold", 0, 42, 3, disabled=not use_acc_count_filter)
        with c_dist:
            st.markdown("#### Distribution Filter")
            use_dist_count_filter = st.checkbox("Enable Dist Count", value=False)
            dist_count_window = st.selectbox("Dist Window", [5, 10, 21, 42], index=2, disabled=not use_dist_count_filter)
            dist_count_logic = st.selectbox("Dist Logic", [">", "<", "="], disabled=not use_dist_count_filter)
            dist_count_thresh = st.number_input("Dist Threshold", 0, 42, 3, disabled=not use_dist_count_filter)
    with st.expander("Gap Filter", expanded=False):
        use_gap_filter = st.checkbox("Enable Open Gap Filter", value=False)
        g1, g2, g3 = st.columns(3)
        with g1: gap_lookback = st.number_input("Lookback Window (Days)", 1, 252, 21, disabled=not use_gap_filter)
        with g2: gap_logic = st.selectbox("Gap Count Logic", [">", "<", "="], disabled=not use_gap_filter)
        with g3: gap_thresh = st.number_input("Count Threshold", 0, 50, 3, disabled=not use_gap_filter)
    with st.expander("Distance from MA Filter", expanded=False):
        use_ma_dist_filter = st.checkbox("Enable Distance Filter", value=False)
        d1, d2, d3, d4 = st.columns(4)
        with d1: dist_ma_type = st.selectbox("Select MA", ["SMA 10", "SMA 20", "SMA 50", "SMA 100", "SMA 200", "EMA 8", "EMA 11", "EMA 21","52-Week High", "All-Time High"], disabled=not use_ma_dist_filter)
        with d2: dist_logic = st.selectbox("Logic", ["Greater Than (>)", "Less Than (<)", "Between"], disabled=not use_ma_dist_filter)
        with d3: dist_min = st.number_input("Min ATR Dist", -50.0, 50.0, 0.0, step=0.5, disabled=not use_ma_dist_filter)
        with d4: dist_max = st.number_input("Max ATR Dist", -50.0, 50.0, 2.0, step=0.5, disabled=not use_ma_dist_filter)
    with st.expander("Weekly MA Pullback", expanded=False):
        use_weekly_ma_pullback = st.checkbox("Enable Weekly MA Pullback Filter", value=False)
        st.caption("Buy pullbacks to a weekly MA after the stock was significantly extended above it.")
        wma_c1, wma_c2, wma_c3, wma_c4 = st.columns(4)
        with wma_c1: wma_type = st.selectbox("MA Type", ["EMA", "SMA"], key="wma_type", disabled=not use_weekly_ma_pullback)
        with wma_c2: wma_period = st.number_input("MA Period", 2, 100, 8, key="wma_period", disabled=not use_weekly_ma_pullback)
        with wma_c3: wma_min_ext_pct = st.number_input("Min Extension %", 1.0, 200.0, 30.0, step=5.0, key="wma_min_ext", disabled=not use_weekly_ma_pullback, help="Stock must have been at least this % above the weekly MA at some point in the lookback window")
        with wma_c4: wma_lookback_months = st.number_input("Lookback Months", 1, 24, 6, key="wma_lookback_mo", disabled=not use_weekly_ma_pullback, help="How far back to check for the extension")
        wma_touch_logic = st.selectbox("Touch Logic", ["Low <= MA", "Close <= MA"], key="wma_touch", disabled=not use_weekly_ma_pullback, help="How to define 'touching' the weekly MA on the signal day")
    with st.expander("Price Action", expanded=False):
        pa1, pa2 = st.columns(2)
        with pa1: 
            req_green_candle = st.checkbox("Require Close > Open", value=False)
            breakout_mode = st.selectbox("Close vs Prev Range", ["None", "Close > Prev Day High", "Close < Prev Day Low"])
        with pa2:
            use_range_filter = st.checkbox("Filter by Daily Candle Range %", value=False)
            r1, r2 = st.columns(2)
            with r1: range_min = st.number_input("Min % (0=Low)", 0, 100, 0, disabled=not use_range_filter)
            with r2: range_max = st.number_input("Max % (100=High)", 0, 100, 100, disabled=not use_range_filter)
        st.markdown("---")
        use_atr_ret_filter = st.checkbox("Filter by Today's Net Change (in ATR units)", value=False)
        st.caption("Calculates (Close - Prev Close) / ATR.")
        ar1, ar2 = st.columns(2)
        with ar1: atr_ret_min = st.number_input("Min Return (ATR)", -10.0, 10.0, 0.0, step=0.1, disabled=not use_atr_ret_filter)
        with ar2: atr_ret_max = st.number_input("Max Return (ATR)", -10.0, 10.0, 1.0, step=0.1, disabled=not use_atr_ret_filter)
        st.markdown("---")
        use_range_atr_filter = st.checkbox("Filter by Today's Range (in ATR units)", value=False)
        st.caption("Calculates (High - Low) / ATR.")
        ra1, ra2, ra3 = st.columns(3)
        with ra1: range_atr_logic = st.selectbox("Logic", [">", "<", "Between"], key="range_atr_logic", disabled=not use_range_atr_filter)
        with ra2: range_atr_min = st.number_input("Min Range (ATR)", 0.0, 20.0, 1.0, step=0.1, key="range_atr_min", disabled=not use_range_atr_filter)
        with ra3: range_atr_max = st.number_input("Max Range (ATR)", 0.0, 20.0, 3.0, step=0.1, key="range_atr_max", disabled=not use_range_atr_filter)
        st.markdown("---")
        st.markdown("#### Multi-Day Price Action Conditions")
        st.caption("Add conditions on prior days' candles. Lag 0 = signal day, 1 = day before signal, etc. All conditions are AND logic.")
        num_pa_conditions = st.number_input("Number of conditions", 0, 6, 0, key="num_pa_cond")
        price_action_filters = []
        for pa_i in range(num_pa_conditions):
            st.markdown(f"**Condition {pa_i + 1}**")
            pa_cols = st.columns([2, 1, 1, 1, 1])
            with pa_cols[0]:
                pa_type = st.selectbox("Type", [
                    "range_pct", "atr_ret", "range_atr",
                    "close_gt_open", "close_lt_open",
                    "close_gt_prev_high", "close_lt_prev_low"
                ], key=f"pa_type_{pa_i}")
            with pa_cols[1]:
                pa_lag = st.number_input("Day Offset", 0, 5, 0, key=f"pa_lag_{pa_i}")
            pa_dict = {"type": pa_type, "lag": pa_lag}
            if pa_type in ("range_pct", "atr_ret", "range_atr"):
                with pa_cols[2]:
                    pa_dict["min"] = st.number_input("Min", -10.0, 100.0, 0.0, step=0.1, key=f"pa_min_{pa_i}")
                with pa_cols[3]:
                    pa_dict["max"] = st.number_input("Max", -10.0, 100.0, 100.0, step=0.1, key=f"pa_max_{pa_i}")
                if pa_type == "range_atr":
                    with pa_cols[4]:
                        pa_dict["logic"] = st.selectbox("Logic", [">", "<", "Between"], key=f"pa_ra_logic_{pa_i}")
            price_action_filters.append(pa_dict)
    with st.expander("Time & Cycle Filters", expanded=False):
        t_c1, t_c2 = st.columns(2)
        with t_c1:
            use_dow_filter = st.checkbox("Enable Day of Week Filter", value=False)
            c_mon, c_tue, c_wed, c_thu, c_fri = st.columns(5)
            valid_days = []
            with c_mon: 
                if st.checkbox("Mon", value=True, disabled=not use_dow_filter): valid_days.append(0)
            with c_tue: 
                if st.checkbox("Tue", value=True, disabled=not use_dow_filter): valid_days.append(1)
            with c_wed: 
                if st.checkbox("Wed", value=True, disabled=not use_dow_filter): valid_days.append(2)
            with c_thu: 
                if st.checkbox("Thu", value=True, disabled=not use_dow_filter): valid_days.append(3)
            with c_fri: 
                if st.checkbox("Fri", value=True, disabled=not use_dow_filter): valid_days.append(4)
        with t_c2:
            cycle_options = {"1. Post-Election": 1, "2. Midterm Year": 2, "3. Pre-Election": 3, "4. Election Year": 0}
            sel_cycles = st.multiselect("Include Years:", options=list(cycle_options.keys()), default=list(cycle_options.keys()))
            allowed_cycles = [cycle_options[x] for x in sel_cycles]
            _year_options = list(range(2000, datetime.datetime.now().year + 1))
            excluded_years = st.multiselect(
                "Exclude Years (drop trades entered in these years):",
                options=_year_options, default=[],
                help="Signals that would enter in these years are skipped. Useful for seeing how the strategy looks ex-2008, ex-2020, etc."
            )
    with st.expander("Trend Filter", expanded=False):
        trend_filter = st.selectbox("Trend Condition", ["None", "Price > 200 SMA", "Not Below Declining 200 SMA", "Price > Rising 200 SMA", "Market > 200 SMA", "Market Not Declining 200 SMA", "Price < 200 SMA", "Price < Falling 200 SMA", "Market < 200 SMA"])
    with st.expander("Performance Percentile Rank", expanded=False):
        col_p_config, col_p_seq = st.columns([3, 1])
        perf_filters = []
        with col_p_config:
            c2d, c5d, c10d, c21d, c126d, c252d = st.columns(6)
            with c2d:
                use_2d = st.checkbox("Enable 2D Rank")
                logic_2d = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="l2d", disabled=not use_2d)
                l_2d_txt = "Min %ile" if logic_2d in ("Between", "Not Between") else "Threshold"
                thresh_2d = st.number_input(l_2d_txt, 0.0, 100.0, 85.0, key="t2d", disabled=not use_2d)
                thresh_2d_max = 100.0
                if logic_2d in ("Between", "Not Between"): thresh_2d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="t2d_max")
                consec_2d = st.number_input("Consec Days", 1, 10, 1, key="c2d_days", disabled=not use_2d)
                if use_2d: perf_filters.append({'window': 2, 'logic': logic_2d, 'thresh': thresh_2d, 'thresh_max': thresh_2d_max, 'consecutive': consec_2d})
            with c5d:
                use_5d = st.checkbox("Enable 5D Rank")
                logic_5d = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="l5d", disabled=not use_5d)
                l_5d_txt = "Min %ile" if logic_5d in ("Between", "Not Between") else "Threshold"
                thresh_5d = st.number_input(l_5d_txt, 0.0, 100.0, 85.0, key="t5d", disabled=not use_5d)
                thresh_5d_max = 100.0
                if logic_5d in ("Between", "Not Between"): thresh_5d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="t5d_max")
                consec_5d = st.number_input("Consec Days", 1, 10, 1, key="c5d_days", disabled=not use_5d)
                if use_5d: perf_filters.append({'window': 5, 'logic': logic_5d, 'thresh': thresh_5d, 'thresh_max': thresh_5d_max, 'consecutive': consec_5d})
            with c10d:
                use_10d = st.checkbox("Enable 10D Rank")
                logic_10d = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="l10d", disabled=not use_10d)
                l_10d_txt = "Min %ile" if logic_10d in ("Between", "Not Between") else "Threshold"
                thresh_10d = st.number_input(l_10d_txt, 0.0, 100.0, 85.0, key="t10d", disabled=not use_10d)
                thresh_10d_max = 100.0
                if logic_10d in ("Between", "Not Between"): thresh_10d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="t10d_max")
                consec_10d = st.number_input("Consec Days", 1, 10, 1, key="c10d_days", disabled=not use_10d)
                if use_10d: perf_filters.append({'window': 10, 'logic': logic_10d, 'thresh': thresh_10d, 'thresh_max': thresh_10d_max, 'consecutive': consec_10d})
            with c21d:
                use_21d = st.checkbox("Enable 21D Rank")
                logic_21d = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="l21d", disabled=not use_21d)
                l_21d_txt = "Min %ile" if logic_21d in ("Between", "Not Between") else "Threshold"
                thresh_21d = st.number_input(l_21d_txt, 0.0, 100.0, 85.0, key="t21d", disabled=not use_21d)
                thresh_21d_max = 100.0
                if logic_21d in ("Between", "Not Between"): thresh_21d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="t21d_max")
                consec_21d = st.number_input("Consec Days", 1, 10, 1, key="c21d_days", disabled=not use_21d)
                if use_21d: perf_filters.append({'window': 21, 'logic': logic_21d, 'thresh': thresh_21d, 'thresh_max': thresh_21d_max, 'consecutive': consec_21d})
            with c126d:
                use_126d = st.checkbox("Enable 126D Rank")
                logic_126d = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="l126d", disabled=not use_126d)
                l_126d_txt = "Min %ile" if logic_126d in ("Between", "Not Between") else "Threshold"
                thresh_126d = st.number_input(l_126d_txt, 0.0, 100.0, 85.0, key="t126d", disabled=not use_126d)
                thresh_126d_max = 100.0
                if logic_126d in ("Between", "Not Between"): thresh_126d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="t126d_max")
                consec_126d = st.number_input("Consec Days", 1, 10, 1, key="c126d_days", disabled=not use_126d)
                if use_126d: perf_filters.append({'window': 126, 'logic': logic_126d, 'thresh': thresh_126d, 'thresh_max': thresh_126d_max, 'consecutive': consec_126d})
            with c252d:
                use_252d = st.checkbox("Enable 252D Rank")
                logic_252d = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="l252d", disabled=not use_252d)
                l_252d_txt = "Min %ile" if logic_252d in ("Between", "Not Between") else "Threshold"
                thresh_252d = st.number_input(l_252d_txt, 0.0, 100.0, 85.0, key="t252d", disabled=not use_252d)
                thresh_252d_max = 100.0
                if logic_252d in ("Between", "Not Between"): thresh_252d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="t252d_max")
                consec_252d = st.number_input("Consec Days", 1, 10, 1, key="c252d_days", disabled=not use_252d)
                if use_252d: perf_filters.append({'window': 252, 'logic': logic_252d, 'thresh': thresh_252d, 'thresh_max': thresh_252d_max, 'consecutive': consec_252d})
        with col_p_seq:
            perf_first = st.checkbox("First Instance", value=False)
            perf_lookback = st.number_input("Lookback (Days)", 1, 100, 21, disabled=not perf_first)

    perf_atr_filters = []
    with st.expander("ATR-Normalized Performance Percentile Rank", expanded=False):
        st.markdown("**Same as Performance Percentile Rank, but N-day moves are measured in ATR units before ranking.** Better for comparing moves across vol regimes within the same ticker.")
        _atr_windows = [2, 5, 10, 21, 126, 252]
        _atr_cols = st.columns(6)
        for _col, _w in zip(_atr_cols, _atr_windows):
            with _col:
                _use = st.checkbox(f"Enable {_w}D ATR Rank", key=f"use_atrp_{_w}")
                _logic = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key=f"atrp_l_{_w}", disabled=not _use)
                _lbl = "Min %ile" if _logic in ("Between", "Not Between") else "Threshold"
                _thresh = st.number_input(_lbl, 0.0, 100.0, 85.0, key=f"atrp_t_{_w}", disabled=not _use)
                _thresh_max = 100.0
                if _logic in ("Between", "Not Between"):
                    _thresh_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key=f"atrp_tmax_{_w}", disabled=not _use)
                _consec = st.number_input("Consec Days", 1, 10, 1, key=f"atrp_c_{_w}", disabled=not _use)
                if _use:
                    perf_atr_filters.append({
                        'window': _w, 'logic': _logic, 'thresh': _thresh,
                        'thresh_max': _thresh_max, 'consecutive': _consec
                    })

    atr_sznl_filters = []
    with st.expander("ATR Seasonal Rank Filter", expanded=False):
        st.markdown("**Rank historical ATR-normalized forward returns per day-of-year.** Walk-forward safe. 100 = best seasonal window, 0 = worst. Requires `atr_seasonal_ranks.parquet`.")
        asz_cols = st.columns(6)
        _asz_windows = [5, 10, 21, 63, 126, 252]
        for _col, _w in zip(asz_cols, _asz_windows):
            with _col:
                _use = st.checkbox(f"Enable {_w}D", key=f"use_asz_{_w}")
                _logic = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key=f"asz_l_{_w}", disabled=not _use)
                _lbl = "Min %ile" if _logic in ("Between", "Not Between") else "Threshold"
                _thresh = st.number_input(_lbl, 0.0, 100.0, 80.0, key=f"asz_t_{_w}", disabled=not _use)
                _thresh_max = 100.0
                if _logic in ("Between", "Not Between"):
                    _thresh_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key=f"asz_tmax_{_w}", disabled=not _use)
                _consec = st.number_input("Consec Days", 1, 10, 1, key=f"asz_c_{_w}", disabled=not _use)
                if _use:
                    atr_sznl_filters.append({
                        'window': _w, 'logic': _logic, 'thresh': _thresh,
                        'thresh_max': _thresh_max, 'consecutive': _consec
                    })

    xsec_filters = []
    with st.expander("Cross-Sectional Rank (vs Universe Peers)", expanded=False):
        st.markdown("**Rank this ticker's return percentile vs all other tickers in the universe on each date.** Normalized for volatility. 100 = most overbought vs peers, 0 = most oversold.")
        use_xsec_filter = st.checkbox("Enable Cross-Sectional Rank Filter", value=False)
        xc5d, xc10d, xc21d, xc126d, xc252d = st.columns(5)
        with xc5d:
            st.markdown("**5-Day Rank**")
            use_xsec_5d = st.checkbox("Enable", key="use_xsec_5d", value=False, disabled=not use_xsec_filter)
            xsec_5d_logic = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="xsec_l5d", disabled=not (use_xsec_filter and use_xsec_5d))
            xsec_5d_lbl = "Min %ile" if xsec_5d_logic in ("Between", "Not Between") else "Threshold"
            xsec_5d_thresh = st.number_input(xsec_5d_lbl, 0.0, 100.0, 85.0, key="xsec_t5d", disabled=not (use_xsec_filter and use_xsec_5d))
            xsec_5d_max = 100.0
            if xsec_5d_logic in ("Between", "Not Between"): xsec_5d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="xsec_t5d_max")
            xsec_5d_consec = st.number_input("Consec Days", 1, 10, 1, key="xsec_c5d", disabled=not (use_xsec_filter and use_xsec_5d))
            if use_xsec_5d: xsec_filters.append({'window': 5, 'logic': xsec_5d_logic, 'thresh': xsec_5d_thresh, 'thresh_max': xsec_5d_max, 'consecutive': xsec_5d_consec})
        with xc10d:
            st.markdown("**10-Day Rank**")
            use_xsec_10d = st.checkbox("Enable", key="use_xsec_10d", value=False, disabled=not use_xsec_filter)
            xsec_10d_logic = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="xsec_l10d", disabled=not (use_xsec_filter and use_xsec_10d))
            xsec_10d_lbl = "Min %ile" if xsec_10d_logic in ("Between", "Not Between") else "Threshold"
            xsec_10d_thresh = st.number_input(xsec_10d_lbl, 0.0, 100.0, 85.0, key="xsec_t10d", disabled=not (use_xsec_filter and use_xsec_10d))
            xsec_10d_max = 100.0
            if xsec_10d_logic in ("Between", "Not Between"): xsec_10d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="xsec_t10d_max")
            xsec_10d_consec = st.number_input("Consec Days", 1, 10, 1, key="xsec_c10d", disabled=not (use_xsec_filter and use_xsec_10d))
            if use_xsec_10d: xsec_filters.append({'window': 10, 'logic': xsec_10d_logic, 'thresh': xsec_10d_thresh, 'thresh_max': xsec_10d_max, 'consecutive': xsec_10d_consec})
        with xc21d:
            st.markdown("**21-Day Rank**")
            use_xsec_21d = st.checkbox("Enable", key="use_xsec_21d", value=False, disabled=not use_xsec_filter)
            xsec_21d_logic = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="xsec_l21d", disabled=not (use_xsec_filter and use_xsec_21d))
            xsec_21d_lbl = "Min %ile" if xsec_21d_logic in ("Between", "Not Between") else "Threshold"
            xsec_21d_thresh = st.number_input(xsec_21d_lbl, 0.0, 100.0, 85.0, key="xsec_t21d", disabled=not (use_xsec_filter and use_xsec_21d))
            xsec_21d_max = 100.0
            if xsec_21d_logic in ("Between", "Not Between"): xsec_21d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="xsec_t21d_max")
            xsec_21d_consec = st.number_input("Consec Days", 1, 10, 1, key="xsec_c21d", disabled=not (use_xsec_filter and use_xsec_21d))
            if use_xsec_21d: xsec_filters.append({'window': 21, 'logic': xsec_21d_logic, 'thresh': xsec_21d_thresh, 'thresh_max': xsec_21d_max, 'consecutive': xsec_21d_consec})
        with xc126d:
            st.markdown("**126-Day Rank**")
            use_xsec_126d = st.checkbox("Enable", key="use_xsec_126d", value=False, disabled=not use_xsec_filter)
            xsec_126d_logic = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="xsec_l126d", disabled=not (use_xsec_filter and use_xsec_126d))
            xsec_126d_lbl = "Min %ile" if xsec_126d_logic in ("Between", "Not Between") else "Threshold"
            xsec_126d_thresh = st.number_input(xsec_126d_lbl, 0.0, 100.0, 85.0, key="xsec_t126d", disabled=not (use_xsec_filter and use_xsec_126d))
            xsec_126d_max = 100.0
            if xsec_126d_logic in ("Between", "Not Between"): xsec_126d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="xsec_t126d_max")
            xsec_126d_consec = st.number_input("Consec Days", 1, 10, 1, key="xsec_c126d", disabled=not (use_xsec_filter and use_xsec_126d))
            if use_xsec_126d: xsec_filters.append({'window': 126, 'logic': xsec_126d_logic, 'thresh': xsec_126d_thresh, 'thresh_max': xsec_126d_max, 'consecutive': xsec_126d_consec})
        with xc252d:
            st.markdown("**252-Day Rank**")
            use_xsec_252d = st.checkbox("Enable", key="use_xsec_252d", value=False, disabled=not use_xsec_filter)
            xsec_252d_logic = st.selectbox("Logic", [">", "<", "Between", "Not Between"], key="xsec_l252d", disabled=not (use_xsec_filter and use_xsec_252d))
            xsec_252d_lbl = "Min %ile" if xsec_252d_logic in ("Between", "Not Between") else "Threshold"
            xsec_252d_thresh = st.number_input(xsec_252d_lbl, 0.0, 100.0, 85.0, key="xsec_t252d", disabled=not (use_xsec_filter and use_xsec_252d))
            xsec_252d_max = 100.0
            if xsec_252d_logic in ("Between", "Not Between"): xsec_252d_max = st.number_input("Max %ile", 0.0, 100.0, 99.0, key="xsec_t252d_max")
            xsec_252d_consec = st.number_input("Consec Days", 1, 10, 1, key="xsec_c252d", disabled=not (use_xsec_filter and use_xsec_252d))
            if use_xsec_252d: xsec_filters.append({'window': 252, 'logic': xsec_252d_logic, 'thresh': xsec_252d_thresh, 'thresh_max': xsec_252d_max, 'consecutive': xsec_252d_consec})
    ma_consec_filters = []
    with st.expander("Consecutive Closes vs SMA", expanded=False):
        c_ma1, c_ma2, c_ma3 = st.columns(3)
        with c_ma1:
            use_ma1 = st.checkbox("Enable MA 1", key="use_ma1")
            len_ma1 = st.number_input("Length", 2, 500, 10, key="len_ma1", disabled=not use_ma1)
            logic_ma1 = st.selectbox("Close vs MA", ["Above", "Below"], key="logic_ma1", disabled=not use_ma1)
            consec_ma1 = st.number_input("Consecutive Days", 1, 50, 1, key="consec_ma1", disabled=not use_ma1)
            if use_ma1: ma_consec_filters.append({'length': len_ma1, 'logic': logic_ma1, 'consec': consec_ma1})
        with c_ma2:
            use_ma2 = st.checkbox("Enable MA 2", key="use_ma2")
            len_ma2 = st.number_input("Length", 2, 500, 20, key="len_ma2", disabled=not use_ma2)
            logic_ma2 = st.selectbox("Close vs MA", ["Above", "Below"], key="logic_ma2", disabled=not use_ma2)
            consec_ma2 = st.number_input("Consecutive Days", 1, 50, 1, key="consec_ma2", disabled=not use_ma2)
            if use_ma2: ma_consec_filters.append({'length': len_ma2, 'logic': logic_ma2, 'consec': consec_ma2})
        with c_ma3:
            use_ma3 = st.checkbox("Enable MA 3", key="use_ma3")
            len_ma3 = st.number_input("Length", 2, 500, 50, key="len_ma3", disabled=not use_ma3)
            logic_ma3 = st.selectbox("Close vs MA", ["Above", "Below"], key="logic_ma3", disabled=not use_ma3)
            consec_ma3 = st.number_input("Consecutive Days", 1, 50, 1, key="consec_ma3", disabled=not use_ma3)
            if use_ma3: ma_consec_filters.append({'length': len_ma3, 'logic': logic_ma3, 'consec': consec_ma3})
    with st.expander("Seasonal Rank", expanded=False):
        use_sznl = st.checkbox("Enable Ticker Seasonal Filter", value=False)
        s1, s2, s3, s4 = st.columns(4)
        with s1: sznl_logic = st.selectbox("Logic", ["<", ">"], key="sl", disabled=not use_sznl)
        with s2: sznl_thresh = st.number_input("Threshold", 0.0, 100.0, 15.0, key="st", disabled=not use_sznl)
        with s3: sznl_first = st.checkbox("First Instance Only", value=False, key="sf", disabled=not use_sznl)
        with s4: sznl_lookback = st.number_input("Instance Lookback (Days)", 1, 100, 21, key="slb", disabled=not use_sznl)
        st.markdown("---")
        use_market_sznl = st.checkbox("Enable Market Seasonal Filter", value=False)
        spy1, spy2 = st.columns(2)
        with spy1: market_sznl_logic = st.selectbox("Market Logic", ["<", ">"], key="spy_sl", disabled=not use_market_sznl)
        with spy2: market_sznl_thresh = st.number_input("Market Threshold", 0.0, 100.0, 15.0, key="spy_st", disabled=not use_market_sznl)
    with st.expander("52-Week High/Low", expanded=False):
        use_52w = st.checkbox("Enable 52w High/Low Filter", value=False)
        h1, h2, h3, h4 = st.columns(4) 
        with h1: type_52w = st.selectbox("Condition", ["New 52w High", "New 52w Low"], disabled=not use_52w)
        with h2: first_52w = st.checkbox("First Instance Only", value=False, key="hf", disabled=not use_52w)
        with h3: lookback_52w = st.number_input("Instance Lookback (Days)", 1, 252, 21, key="hlb", disabled=not use_52w)
        with h4: lag_52w = st.number_input("Lag (Days)", 0, 10, 0, disabled=not use_52w)
        exclude_52w_high = st.checkbox("Exclude if Today IS a 52w High", value=False)
        recent_52w_mode = st.selectbox("Trailing 52w High Filter", ["Disabled", "Made 52w High in Last N Days", "Has NOT Made 52w High in Last N Days"])
        use_recent_52w = recent_52w_mode != "Disabled"
        recent_52w_invert = recent_52w_mode == "Has NOT Made 52w High in Last N Days"
        recent_52w_lookback = st.number_input("52w High Lookback (Days)", 1, 252, 21, disabled=not use_recent_52w)
        recent_52w_low_mode = st.selectbox("Trailing 52w Low Filter", ["Disabled", "Made 52w Low in Last N Days", "Has NOT Made 52w Low in Last N Days"], key="recent_52w_low_mode")
        use_recent_52w_low = recent_52w_low_mode != "Disabled"
        recent_52w_low_invert = recent_52w_low_mode == "Has NOT Made 52w Low in Last N Days"
        recent_52w_low_lookback = st.number_input("52w Low Lookback (Days)", 1, 252, 21, disabled=not use_recent_52w_low, key="recent_52w_low_lb")
        st.markdown("---")
        use_ath = st.checkbox("Enable All-Time High Filter", value=False)
        ath_type = st.selectbox("ATH Condition", ["Today is ATH", "Today is NOT ATH"], disabled=not use_ath)
        recent_ath_mode = st.selectbox("Trailing ATH Filter", ["Disabled", "Made ATH in Last N Days", "Has NOT Made ATH in Last N Days"])
        use_recent_ath = recent_ath_mode != "Disabled"
        recent_ath_invert = recent_ath_mode == "Has NOT Made ATH in Last N Days"
        ath_lookback_days = st.number_input("ATH Lookback (Days)", 1, 252, 21, disabled=not use_recent_ath)
    with st.expander("Market Regime (VIX)", expanded=False):
        use_vix_filter = st.checkbox(f"Enable {VIX_TICKER} Filter", value=False)
        v1, v2 = st.columns(2)
        with v1: vix_min = st.number_input("Min VIX Value", 0.0, 200.0, 0.0, disabled=not use_vix_filter)
        with v2: vix_max = st.number_input("Max VIX Value", 0.0, 200.0, 20.0, disabled=not use_vix_filter)

    with st.expander("Risk Dial Filters (Fragility Score, 2016+)", expanded=False):
        _frag_df = load_fragility_dials()
        if _frag_df is None or _frag_df.empty:
            st.warning("data/rd2_fragility.parquet not found — run the Risk Dashboard once to generate it.")
            dial_filters = []
        else:
            st.caption(
                f"Fragility dial (0-100) from Risk Dashboard. History: {_frag_df.index.min().strftime('%Y-%m-%d')} to "
                f"{_frag_df.index.max().strftime('%Y-%m-%d')}. Pre-history trades fail any enabled dial filter."
            )
            dial_filters = []
            for i in range(1, 4):
                fc1, fc2, fc3, fc4, fc5 = st.columns([1, 1, 1.1, 0.9, 1])
                with fc1:
                    _use = st.checkbox(f"Filter {i}", key=f"dial_use_{i}")
                with fc2:
                    _dial = st.selectbox(
                        "Dial", ["5d", "21d", "63d"], key=f"dial_col_{i}",
                        index=(i - 1) if i <= 3 else 0, disabled=not _use,
                    )
                with fc3:
                    _win = st.number_input(
                        "Rolling avg (days)", 1, 63, 10, step=1,
                        key=f"dial_win_{i}", disabled=not _use,
                        help="1 = raw dial value; higher = smoothed over N days",
                    )
                with fc4:
                    _log = st.selectbox(
                        "Logic", [">", "<", ">=", "<="], key=f"dial_log_{i}", disabled=not _use,
                    )
                with fc5:
                    _thr = st.number_input(
                        "Threshold", 0.0, 100.0, 40.0 if i == 1 else 30.0 if i == 2 else 20.0,
                        step=5.0, key=f"dial_thr_{i}", disabled=not _use,
                    )
                if _use:
                    dial_filters.append({
                        'dial': _dial, 'window': int(_win),
                        'logic': _log, 'thresh': float(_thr),
                    })
            if dial_filters:
                st.success(
                    "Active dial filters: "
                    + "; ".join(f"{f['dial']} dial ({f['window']}d avg) {f['logic']} {f['thresh']:.0f}" for f in dial_filters)
                )
    with st.expander("Volume Filters", expanded=False):
        use_vol_gt_prev = st.checkbox("Require Volume > Prev Day Volume", value=False)
        c1, c2 = st.columns(2)
        with c1:
            use_vol = st.checkbox("Enable Spike Filter", value=False)
            vs_col1, vs_col2 = st.columns(2)
            with vs_col1: vol_logic = st.selectbox("Logic", [">", "<", "Between"], key="vs_logic", disabled=not use_vol)
            vs_lbl = "Min Multiple" if vol_logic == "Between" else "Vol Multiple (vs 63d Avg)"
            with vs_col2: vol_thresh = st.number_input(vs_lbl, 0.1, 10.0, 1.5, disabled=not use_vol)
            vol_thresh_max = 10.0
            if vol_logic == "Between":
                vol_thresh_max = st.number_input("Max Multiple", 0.1, 10.0, 3.0, key="vs_tmax", disabled=not use_vol)
        with c2:
            use_vol_rank = st.checkbox("Enable Regime Filter", value=False)
            v_col1, v_col2 = st.columns(2)
            with v_col1: vol_rank_logic = st.selectbox("Logic", ["<", ">"], key="vrl", disabled=not use_vol_rank)
            with v_col2: vol_rank_thresh = st.number_input("Percentile (0-100)", 0.0, 100.0, 50.0, key="vrt", disabled=not use_vol_rank)
    # =====================================================
    # NEW: REFERENCE TICKER FILTER
    # =====================================================
    with st.expander("Reference Ticker Filter (e.g., IWM, SPY regime)", expanded=False):
        st.markdown("**Filter signals based on another ticker's performance rank.** Useful for regime filtering (e.g., only take longs when IWM is oversold).")
        use_ref_ticker_filter = st.checkbox("Enable Reference Ticker Filter", value=False)
        ref_ticker_input = st.text_input("Reference Ticker", value="IWM", disabled=not use_ref_ticker_filter).strip().upper()
        st.markdown("---")
        ref_filters = []
        ref_c1, ref_c2, ref_c3 = st.columns(3)
        with ref_c1:
            st.markdown("**5-Day Rank**")
            use_ref_5d = st.checkbox("Enable", key="use_ref_5d", value=False, disabled=not use_ref_ticker_filter)
            ref_5d_logic = st.selectbox("Logic", ["<", ">"], key="ref_5d_logic", disabled=not (use_ref_ticker_filter and use_ref_5d))
            ref_5d_thresh = st.number_input("Threshold %ile", 0.0, 100.0, 33.0, key="ref_5d_thresh", disabled=not (use_ref_ticker_filter and use_ref_5d))
            if use_ref_5d: ref_filters.append({'window': 5, 'logic': ref_5d_logic, 'thresh': ref_5d_thresh})
        with ref_c2:
            st.markdown("**10-Day Rank**")
            use_ref_10d = st.checkbox("Enable", key="use_ref_10d", value=False, disabled=not use_ref_ticker_filter)
            ref_10d_logic = st.selectbox("Logic", ["<", ">"], key="ref_10d_logic", disabled=not (use_ref_ticker_filter and use_ref_10d))
            ref_10d_thresh = st.number_input("Threshold %ile", 0.0, 100.0, 33.0, key="ref_10d_thresh", disabled=not (use_ref_ticker_filter and use_ref_10d))
            if use_ref_10d: ref_filters.append({'window': 10, 'logic': ref_10d_logic, 'thresh': ref_10d_thresh})
        with ref_c3:
            st.markdown("**21-Day Rank**")
            use_ref_21d = st.checkbox("Enable", key="use_ref_21d", value=False, disabled=not use_ref_ticker_filter)
            ref_21d_logic = st.selectbox("Logic", ["<", ">"], key="ref_21d_logic", disabled=not (use_ref_ticker_filter and use_ref_21d))
            ref_21d_thresh = st.number_input("Threshold %ile", 0.0, 100.0, 33.0, key="ref_21d_thresh", disabled=not (use_ref_ticker_filter and use_ref_21d))
            if use_ref_21d: ref_filters.append({'window': 21, 'logic': ref_21d_logic, 'thresh': ref_21d_thresh})
    st.markdown("---")
    if st.button("Run Backtest", type="primary", use_container_width=True):
        tickers_to_run = []
        sznl_map = load_seasonal_map()
        if univ_choice == "Sector ETFs": tickers_to_run = SECTOR_ETFS
        elif univ_choice == "Indices": tickers_to_run = INDEX_ETFS
        elif univ_choice == "SPX": tickers_to_run = SPX
        elif univ_choice == "International ETFs": tickers_to_run = INTERNATIONAL_ETFS
        elif univ_choice == "Sector + Index ETFs": tickers_to_run = list(set(SECTOR_ETFS + INDEX_ETFS))
        elif univ_choice == "All CSV Tickers": tickers_to_run = [t for t in list(sznl_map.keys())]
        elif univ_choice == "All CSV + Overflow Extras":
            _base = [t for t in list(sznl_map.keys())]
            _base_set = set(_base)
            _new_extras = [t for t in extras_tickers if t not in _base_set]
            _scope = st.session_state.get('overflow_run_scope', 'All (base + extras)')
            if _scope == "Base only (All CSV)":
                tickers_to_run = list(_base)
            elif _scope == "Extras only (overflow)":
                tickers_to_run = list(_new_extras)
            else:
                tickers_to_run = _base + _new_extras
            if extras_tickers:
                st.info(f"Universe cached: {len(_base)} base + {len(_new_extras)} extras. **Running on {len(tickers_to_run)}** ({_scope}).")
        elif univ_choice == "All CSV (Equities Only)": tickers_to_run = [t for t in list(sznl_map.keys()) if t not in ["BTC-USD", "ETH-USD", "SLV", "GLD", "USO", "UVXY", "CEF", "UNG", "XOP"] + SECTOR_ETFS + INDEX_ETFS + INTERNATIONAL_ETFS + SPX]
        elif univ_choice == "3x Leveraged (All)": tickers_to_run = LEV3X_ALL
        elif univ_choice == "3x Leveraged Equities": tickers_to_run = LEV3X_EQUITY_ALL
        elif univ_choice == "3x Leveraged Equities (Bull)": tickers_to_run = LEV3X_EQUITY_BULL_ALL
        elif univ_choice == "3x Leveraged Equities (Bear)": tickers_to_run = LEV3X_EQUITY_BEAR_ALL
        elif univ_choice == "3x Leveraged Equities (Broad Only)": tickers_to_run = LEV3X_EQUITY_BROAD
        elif univ_choice == "Custom (Upload CSV)": tickers_to_run = custom_tickers
        if tickers_to_run and sample_pct < 100:
            count = max(1, int(len(tickers_to_run) * (sample_pct / 100)))
            tickers_to_run = random.sample(tickers_to_run, count)
            st.info(f"Randomly selected {len(tickers_to_run)} tickers.")
        if not tickers_to_run: st.error("No tickers found."); return
        fetch_start = "1950-01-01" if use_full_history else start_date - datetime.timedelta(days=365)
        # Split download: in hybrid mode, fetch base and extras separately so the 1k-ticker
        # base keeps its cache key across different extras uploads (only the delta re-fetches).
        if univ_choice == "All CSV + Overflow Extras" and extras_tickers:
            st.info(f"Downloading base ({len(_base)}, cached) + extras ({len(_new_extras)} new)...")
            base_data = download_universe_data(_base, fetch_start)
            extras_data = download_universe_data(_new_extras, fetch_start) if _new_extras else {}
            _full_data = {**base_data, **extras_data}
            _run_set = set(tickers_to_run)
            data_dict = {t: df for t, df in _full_data.items() if t in _run_set}
        else:
            st.info(f"Downloading data ({len(tickers_to_run)} tickers)...")
            data_dict = download_universe_data(tickers_to_run, fetch_start)
        if not data_dict: return

        # Truncate to end_date — drops any data after the chosen end so signals
        # can't fire and trades can't extend past the backtest window. Useful
        # for out-of-sample testing.
        if end_date < datetime.date.today():
            _end_ts = pd.Timestamp(end_date)
            _truncated = 0
            for _t, _df in list(data_dict.items()):
                if _df is None or _df.empty:
                    continue
                _sliced = _df[_df.index <= _end_ts]
                if len(_sliced) < len(_df):
                    _truncated += 1
                if _sliced.empty:
                    del data_dict[_t]
                else:
                    data_dict[_t] = _sliced
            st.info(f"Truncated {_truncated} ticker(s) to end date {end_date}")
        market_series, market_sznl_series = None, None
        market_sma_not_declining_series = None
        need_market_data = ("Market" in trend_filter) or use_market_sznl
        if need_market_data:
            market_df = data_dict.get(MARKET_TICKER)
            if market_df is None:
                st.info(f"Fetching {MARKET_TICKER} data...")
                market_dict_temp = download_universe_data([MARKET_TICKER], fetch_start)
                market_df = market_dict_temp.get(MARKET_TICKER, None)
            if market_df is not None and not market_df.empty:
                if market_df.index.tz is not None: market_df.index = market_df.index.tz_localize(None)
                market_df.index = market_df.index.normalize()
                market_df['SMA200'] = market_df['Close'].rolling(200).mean()
                market_series = market_df['Close'] > market_df['SMA200']
                market_sma_not_declining_series = market_df['SMA200'] >= market_df['SMA200'].shift(1)
                if use_market_sznl: market_sznl_series = get_sznl_val_series(MARKET_TICKER, market_df.index, sznl_map)
        vix_series = None
        if use_vix_filter:
            vix_df = data_dict.get(VIX_TICKER)
            if vix_df is None:
                st.info(f"Fetching {VIX_TICKER} data...")
                vix_dict_temp = download_universe_data([VIX_TICKER], fetch_start)
                vix_df = vix_dict_temp.get(VIX_TICKER, None)
            if vix_df is not None and not vix_df.empty:
                if vix_df.index.tz is not None: vix_df.index = vix_df.index.tz_localize(None)
                vix_df.index = vix_df.index.normalize()
                vix_series = vix_df['Close']
        
        # =====================================================
        # NEW: PREPARE REFERENCE TICKER DATA
        # =====================================================
        ref_ticker_ranks = None
        if use_ref_ticker_filter and ref_filters and ref_ticker_input:
            st.info(f"Preparing reference ticker data for {ref_ticker_input}...")
            ref_ticker_clean = ref_ticker_input.replace('.', '-')
            ref_df = data_dict.get(ref_ticker_clean)
            if ref_df is None:
                ref_dict_temp = download_universe_data([ref_ticker_clean], fetch_start)
                ref_df = ref_dict_temp.get(ref_ticker_clean, None)
            if ref_df is not None and not ref_df.empty:
                # Calculate indicators for the reference ticker
                ref_df_calc = calculate_indicators(ref_df.copy(), sznl_map, ref_ticker_clean, market_series, vix_series, market_sznl_series)
                # Extract the rank series we need
                ref_ticker_ranks = {}
                for rf in ref_filters:
                    window = rf['window']
                    col = f'rank_ret_{window}d'
                    if col in ref_df_calc.columns:
                        ref_ticker_ranks[window] = ref_df_calc[col]
                st.success(f"✓ Reference ticker {ref_ticker_input} loaded with {len(ref_df_calc)} rows")
            else:
                st.warning(f"Could not load data for reference ticker {ref_ticker_input}. Filter will be skipped.")
        
        params = {
            'use_max_daily_risk': use_max_daily_risk, 'max_daily_risk_pct': max_daily_risk_pct,
            'backtest_start_date': start_date, 'trade_direction': trade_direction, 'max_one_pos': max_one_pos, 'allow_same_day_reentry': allow_same_day_reentry,
            'max_daily_entries': max_daily_entries, 'max_total_positions': max_total_positions, 'use_stop_loss': use_stop_loss, 'use_take_profit': use_take_profit, 'time_exit_only': time_exit_only,
            'use_partial_exits': use_partial_exits, 'partial_target_fraction': partial_target_fraction,
            'stop_atr': stop_atr, 'tgt_atr': tgt_atr, 'holding_days': hold_days, 'entry_type': entry_type, 'use_ma_entry_filter': use_ma_entry_filter, 'require_close_gt_open': req_green_candle,
            'breakout_mode': breakout_mode, 'use_range_filter': use_range_filter, 'range_min': range_min, 'range_max': range_max, 'use_dow_filter': use_dow_filter, 'allowed_days': valid_days,
            'allowed_cycles': allowed_cycles, 'excluded_years': excluded_years, 'min_price': min_price, 'min_vol': min_vol, 'min_age': min_age, 'max_age': max_age, 'min_atr_pct': min_atr_pct, 'max_atr_pct': max_atr_pct,
            'trend_filter': trend_filter, 'universe_tickers': tickers_to_run, 'slippage_bps': slippage_bps, 'entry_conf_bps': entry_conf_bps, 'perf_filters': perf_filters, 'perf_atr_filters': perf_atr_filters, 'perf_first_instance': perf_first,
            'use_atr_ret_filter': use_atr_ret_filter, 'atr_ret_min': atr_ret_min, 'atr_ret_max': atr_ret_max,
            'use_range_atr_filter': use_range_atr_filter, 'range_atr_logic': range_atr_logic, 'range_atr_min': range_atr_min, 'range_atr_max': range_atr_max,
            'price_action_filters': price_action_filters,
            'perf_lookback': perf_lookback, 'ma_consec_filters': ma_consec_filters, 'use_sznl': use_sznl, 'sznl_logic': sznl_logic, 'sznl_thresh': sznl_thresh, 'sznl_first_instance': sznl_first,
            'sznl_lookback': sznl_lookback, 'use_market_sznl': use_market_sznl, 'market_sznl_logic': market_sznl_logic, 'market_sznl_thresh': market_sznl_thresh, 'use_52w': use_52w, '52w_type': type_52w,
            'use_ath': use_ath, 'ath_type': ath_type,
            '52w_first_instance': first_52w, '52w_lookback': lookback_52w, '52w_lag': lag_52w, 'exclude_52w_high': exclude_52w_high, 'use_vix_filter': use_vix_filter, 'vix_min': vix_min, 'vix_max': vix_max,
            'use_recent_52w': use_recent_52w, 'recent_52w_invert': recent_52w_invert, 'recent_52w_lookback': recent_52w_lookback,
            'use_recent_52w_low': use_recent_52w_low, 'recent_52w_low_invert': recent_52w_low_invert, 'recent_52w_low_lookback': recent_52w_low_lookback,
            'vol_gt_prev': use_vol_gt_prev, 'use_vol': use_vol, 'vol_logic': vol_logic, 'vol_thresh': vol_thresh, 'vol_thresh_max': vol_thresh_max, 'use_vol_rank': use_vol_rank, 'vol_rank_logic': vol_rank_logic, 'vol_rank_thresh': vol_rank_thresh,
            'use_ma_dist_filter': use_ma_dist_filter, 'dist_ma_type': dist_ma_type, 'dist_logic': dist_logic, 'dist_min': dist_min, 'dist_max': dist_max,
            'use_weekly_ma_pullback': use_weekly_ma_pullback, 'wma_type': wma_type, 'wma_period': wma_period, 'wma_min_ext_pct': wma_min_ext_pct, 'wma_lookback_months': wma_lookback_months, 'wma_touch_logic': wma_touch_logic,
            'use_gap_filter': use_gap_filter, 'gap_lookback': gap_lookback, 'gap_logic': gap_logic, 'gap_thresh': gap_thresh,
            'use_acc_count_filter': use_acc_count_filter, 'acc_count_window': acc_count_window, 'acc_count_logic': acc_count_logic, 'acc_count_thresh': acc_count_thresh,
            'use_dist_count_filter': use_dist_count_filter, 'dist_count_window': dist_count_window, 'dist_count_logic': dist_count_logic, 'dist_count_thresh': dist_count_thresh,
            'use_acc_dist_v2': use_acc_dist_v2,
            'use_t1_open_filter': use_t1_open_filter, 't1_open_filters': t1_open_filters,
            'use_recent_ath': use_recent_ath, 'recent_ath_invert': recent_ath_invert, 'ath_lookback_days': ath_lookback_days,
            'use_ref_ticker_filter': use_ref_ticker_filter, 'ref_ticker': ref_ticker_input, 'ref_filters': ref_filters,
            'use_xsec_filter': use_xsec_filter, 'xsec_filters': xsec_filters,
            'atr_sznl_filters': atr_sznl_filters,
            'dial_filters': dial_filters
        }

        # Build cross-sectional rank matrices if enabled
        xsec_rank_matrices = None
        if use_xsec_filter and xsec_filters:
            xsec_windows = list(set(f['window'] for f in xsec_filters))
            st.info(f"Computing cross-sectional ranks ({len(data_dict)} tickers, windows: {xsec_windows})...")
            xsec_rank_matrices = build_xsec_rank_matrices(data_dict, xsec_windows)
            st.success(f"Cross-sectional ranks computed.")

        atr_sznl_map = load_atr_seasonal_map() if atr_sznl_filters else None
        if atr_sznl_filters and not atr_sznl_map:
            st.warning("ATR Seasonal Rank filter is enabled but atr_seasonal_ranks.parquet could not be loaded. Filter will be skipped.")

        fragility_df = load_fragility_dials() if dial_filters else None
        trades_df, rejected_df, total_signals = run_engine(data_dict, params, sznl_map, market_series, vix_series, market_sznl_series, ref_ticker_ranks, xsec_rank_matrices, atr_sznl_map, fragility_df=fragility_df, market_sma_not_declining_series=market_sma_not_declining_series)
        if trades_df.empty: st.warning("No executed signals.")
        if not trades_df.empty:
            trades_df = trades_df.sort_values("ExitDate")

            # Per-trade RiskScale: pro-rata down on days where aggregate raw risk
            # would exceed the daily cap. Always present (1.0 when feature off).
            if use_max_daily_risk and risk_bps_input > 0:
                per_trade_pct = risk_bps_input / 100.0  # bps → %
                _entry_dt = pd.to_datetime(trades_df['EntryDate'])
                day_counts = _entry_dt.value_counts()
                day_raw_pct = day_counts * per_trade_pct
                day_scale = (max_daily_risk_pct / day_raw_pct).clip(upper=1.0)
                trades_df['RiskScale'] = _entry_dt.map(day_scale).astype(float)
                _capped_days = int((day_scale < 1.0).sum())
                _capped_trades = int((trades_df['RiskScale'] < 1.0).sum())
                if _capped_days > 0:
                    _avg_scale = float(trades_df.loc[trades_df['RiskScale'] < 1.0, 'RiskScale'].mean())
                    st.info(f"Daily risk cap engaged on **{_capped_days}** day(s), scaling **{_capped_trades}** trades (avg scale {_avg_scale:.2f}x).")
            else:
                trades_df['RiskScale'] = 1.0

            trades_df['PnL_Dollar'] = trades_df['R'] * risk_per_trade * trades_df['RiskScale']
            trades_df['CumPnL'] = trades_df['PnL_Dollar'].cumsum()
            # Build MTM curves (flat + dynamic) before dates get stringified
            mtm_flat = build_mtm_curves(trades_df, data_dict, starting_portfolio, risk_bps_input, mode='flat')
            mtm_dyn  = build_mtm_curves(trades_df, data_dict, starting_portfolio, risk_bps_input, mode='dynamic')
            portfolio_stats = compute_portfolio_stats(mtm_dyn, starting_portfolio, risk_bps=risk_bps_input, trades_df=trades_df)
            trades_df = compute_trade_path_stats(trades_df, data_dict)
            trades_df['SignalDate'] = pd.to_datetime(trades_df['SignalDate'])
            trades_df['EntryDate'] = pd.to_datetime(trades_df['EntryDate'])
            trades_df['DayOfWeek'] = trades_df['EntryDate'].dt.day_name()
            trades_df['Year'] = trades_df['SignalDate'].dt.year
            trades_df['Month'] = trades_df['SignalDate'].dt.strftime('%b')
            trades_df['SignalDate'] = trades_df['SignalDate'].dt.strftime('%Y-%m-%d')
            trades_df['EntryDate'] = trades_df['EntryDate'].dt.strftime('%Y-%m-%d')
            trades_df['ExitDate'] = pd.to_datetime(trades_df['ExitDate']).dt.strftime('%Y-%m-%d')
            trades_df['CyclePhase'] = trades_df['Year'].apply(get_cycle_year)
            trades_df['AgeBucket'] = trades_df['Age'].apply(get_age_bucket)
            if len(trades_df) >= 10:
                try: trades_df['VolDecile'] = pd.qcut(trades_df['AvgVol'], 10, labels=False, duplicates='drop') + 1
                except: trades_df['VolDecile'] = 1
            else: trades_df['VolDecile'] = 1
            wins = trades_df[trades_df['R'] > 0]
            losses = trades_df[trades_df['R'] <= 0]
            win_rate = len(wins) / len(trades_df) * 100
            pf = wins['PnL_Dollar'].sum() / abs(losses['PnL_Dollar'].sum()) if not losses.empty else 999
            r_series = trades_df['R']
            sqn = np.sqrt(len(trades_df)) * (r_series.mean() / r_series.std()) if len(trades_df) > 1 else 0
            expectancy_r = r_series.mean()
        else:
            win_rate, pf, sqn, expectancy_r = 0, 0, 0, 0
        fill_rate = (len(trades_df) / total_signals * 100) if total_signals > 0 else 0
        grade, verdict, notes = grade_strategy(pf, sqn, win_rate, len(trades_df))
        st.success("Backtest Complete!")
        st.markdown(f"""
        <div style="background-color: #0e1117; padding: 20px; border-radius: 10px; border: 1px solid #444;">
            <h2 style="margin-top:0; color: #ffffff;">Strategy Grade: <span style="color: {'#00ff00' if grade in ['A','B'] else '#ffaa00' if grade=='C' else '#ff0000'};">{grade}</span> ({verdict})</h2>
            <div style="display: flex; justify-content: space-between; flex-wrap: wrap;">
                <div><h3>Profit Factor: {pf:.2f}</h3></div>
                <div><h3>SQN: {sqn:.2f}</h3></div>
                <div><h3>Win Rate: {win_rate:.1f}%</h3></div>
                <div><h3>Expectancy: ${trades_df['PnL_Dollar'].mean() if not trades_df.empty else 0:.2f}</h3></div>
            </div>
            <div style="margin-top: 10px; color: #aaa; font-size: 14px;">
               Fill Rate: {fill_rate:.1f}% ({len(trades_df)} executed vs {total_signals} raw signals) | Rejection/Missed: {len(rejected_df)} trades
            </div>
        </div>
        """, unsafe_allow_html=True)
        if notes: st.warning("Notes: " + ", ".join(notes))
        if not trades_df.empty:
            strategy_dict = build_strategy_dict(params, tickers_to_run, pf, sqn, win_rate, expectancy_r)
            with st.expander("Export Strategy Configuration", expanded=False):
                st.markdown("**Copy this dictionary to `strategy_config.py`:**")
                import pprint
                py_code = pprint.pformat(strategy_dict, width=120, sort_dicts=False)
                st.code(py_code, language="python")
                st.download_button("Download as Python", py_code, file_name="strategy_export.py", mime="text/x-python")
        if not trades_df.empty:
            def _mtm_fig(df, title):
                d = df.reset_index()
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=d['Date'], y=d['Equity_High'], mode='lines',
                                         line=dict(width=0), hoverinfo='skip', showlegend=False))
                fig.add_trace(go.Scatter(x=d['Date'], y=d['Equity_Low'], mode='lines',
                                         line=dict(width=0), fill='tonexty',
                                         fillcolor='rgba(100,149,237,0.18)',
                                         name='Intraday H/L band', hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=d['Date'], y=d['Equity_Close'], mode='lines',
                                         line=dict(width=1.8, color='rgb(100,149,237)'),
                                         name='Close MTM'))
                fig.update_layout(title=title, yaxis_title='Equity ($)', xaxis_title='Date',
                                  hovermode='x unified', height=420, margin=dict(t=50, b=40, l=40, r=20))
                return fig

            # ========== SECTION 1: FLAT STAKING ==========
            st.subheader(f"Flat Staking — ${risk_per_trade:,.0f} per trade ({risk_bps_input} bps of ${starting_portfolio:,})")
            if not mtm_flat.empty:
                st.plotly_chart(_mtm_fig(mtm_flat, "Mark-to-Market Equity — Flat Risk"), use_container_width=True)

            st.subheader("Performance Breakdowns (Trade-level, Flat Risk)")
            y1, y2 = st.columns(2)
            y1.plotly_chart(px.bar(trades_df.groupby('Year')['PnL_Dollar'].sum().reset_index(), x='Year', y='PnL_Dollar', title="Net Profit ($) by Year", text_auto='.2s'), use_container_width=True)
            trades_per_year = trades_df.groupby('Year').size().reset_index(name='Count')
            y2.plotly_chart(px.bar(trades_per_year, x='Year', y='Count', title="Total Trades by Year", text_auto=True), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.plotly_chart(px.bar(trades_df.groupby('CyclePhase')['PnL_Dollar'].sum().reset_index().sort_values('CyclePhase'), x='CyclePhase', y='PnL_Dollar', title="PnL by Cycle", text_auto='.2s'), use_container_width=True)
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_pnl = trades_df.groupby("Month")["PnL_Dollar"].sum().reindex(month_order).reset_index()
            c2.plotly_chart(px.bar(monthly_pnl, x="Month", y="PnL_Dollar", title="PnL by Month", text_auto='.2s'), use_container_width=True)
            dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            dow_pnl = trades_df.groupby("DayOfWeek")["PnL_Dollar"].sum().reindex(dow_order).reset_index()
            c3.plotly_chart(px.bar(dow_pnl, x="DayOfWeek", y="PnL_Dollar", title="PnL by Entry Day", text_auto='.2s'), use_container_width=True)
            ticker_pnl = trades_df.groupby("Ticker")["PnL_Dollar"].sum().reset_index()
            ticker_pnl = ticker_pnl.sort_values("PnL_Dollar", ascending=False).head(75)
            st.plotly_chart(px.bar(ticker_pnl, x="Ticker", y="PnL_Dollar", title="Cumulative PnL by Ticker (Top 75)", text_auto='.2s'), use_container_width=True)

            # ========== SECTION 2: DYNAMIC SIZING (COMPOUNDING) ==========
            st.markdown("---")
            st.subheader(f"Dynamic Sizing — {risk_bps_input} bps of running equity (starting ${starting_portfolio:,})")
            if not mtm_dyn.empty:
                st.plotly_chart(_mtm_fig(mtm_dyn, "Mark-to-Market Equity — Dynamic Risk (Compounded)"), use_container_width=True)

                ps = portfolio_stats
                dd_color = '#00ff00' if ps['MaxDD_Pct'] > -15 else '#ffaa00' if ps['MaxDD_Pct'] > -30 else '#ff0000'
                dd_low_color = '#00ff00' if ps['MaxDD_Low_Pct'] > -20 else '#ffaa00' if ps['MaxDD_Low_Pct'] > -35 else '#ff0000'
                sh_color = '#00ff00' if ps['Sharpe'] >= 1.0 else '#ffaa00' if ps['Sharpe'] >= 0.5 else '#ff0000'
                sha_color = '#00ff00' if ps['Sharpe_Active'] >= 1.0 else '#ffaa00' if ps['Sharpe_Active'] >= 0.5 else '#ff0000'

                # Build underwater / recovery display strings
                _pd_str = ps['PeakDate'].strftime('%Y-%m-%d') if ps.get('PeakDate') is not None else '—'
                _td_str = ps['TroughDate'].strftime('%Y-%m-%d') if ps.get('TroughDate') is not None else '—'
                if ps.get('DDStillOngoing'):
                    _rd_str = f"<span style='color:#ffaa00;'>still underwater</span>"
                elif ps.get('RecoveryDate') is not None:
                    _rd_str = ps['RecoveryDate'].strftime('%Y-%m-%d')
                else:
                    _rd_str = '—'
                _uw_days = ps.get('UnderwaterDays', 0)
                _uw_trades = ps.get('TradesDuringDD')
                _uw_trades_str = f"{_uw_trades:,}" if _uw_trades is not None else '—'

                st.markdown(f"""
                <div style="background-color: #0e1117; padding: 20px; border-radius: 10px; border: 1px solid #444; margin-top: 10px;">
                    <h3 style="margin-top:0; color:#ffffff;">Portfolio Stats (Dynamic, MTM)</h3>
                    <div style="display: flex; flex-wrap: wrap; gap: 24px;">
                        <div><strong>Final Equity:</strong> ${ps['FinalEquity']:,.0f}</div>
                        <div><strong>Total Return:</strong> {ps['TotalReturn_Pct']:.1f}%</div>
                        <div><strong>CAGR:</strong> {ps['CAGR_Pct']:.2f}%</div>
                        <div><strong>Calmar:</strong> {ps['Calmar']:.2f}</div>
                        <div><strong>Parkinson Vol (ann.):</strong> {ps['ParkinsonVol_Pct']:.2f}%</div>
                        <div><strong>Time in Market:</strong> {ps['TimeInMarket_Pct']:.1f}%</div>
                    </div>
                    <div style="margin-top: 14px; padding-top: 12px; border-top: 1px dashed #333; display: flex; flex-wrap: wrap; gap: 24px;">
                        <div>
                            <div style="color:#aaa; font-size:11px;">CALENDAR (standalone view)</div>
                            <div><strong style="color:{sh_color};">Sharpe:</strong> {ps['Sharpe']:.2f} &nbsp; <strong>Sortino:</strong> {ps['Sortino']:.2f}</div>
                        </div>
                        <div>
                            <div style="color:#aaa; font-size:11px;">ACTIVE-PERIOD (ensemble-relevant)</div>
                            <div><strong style="color:{sha_color};">Sharpe:</strong> {ps['Sharpe_Active']:.2f} &nbsp; <strong>Sortino:</strong> {ps['Sortino_Active']:.2f}</div>
                        </div>
                    </div>
                    <div style="margin-top: 14px; padding-top: 12px; border-top: 1px dashed #333; display: flex; flex-wrap: wrap; gap: 24px;">
                        <div>
                            <div style="color:#aaa; font-size:11px;">MAX DD (close-to-close)</div>
                            <div><strong style="color:{dd_color};">{ps['MaxDD_Pct']:.2f}%</strong> &nbsp; / &nbsp; <strong>{ps['MaxDD_R']:.1f} R</strong></div>
                        </div>
                        <div>
                            <div style="color:#aaa; font-size:11px;">MAX DD @ LOWS (worst intraday mark)</div>
                            <div><strong style="color:{dd_low_color};">{ps['MaxDD_Low_Pct']:.2f}%</strong> &nbsp; / &nbsp; <strong>{ps['MaxDD_Low_R']:.1f} R</strong></div>
                        </div>
                        <div>
                            <div style="color:#aaa; font-size:11px;">UNDERWATER</div>
                            <div><strong>{_uw_days:,}</strong> calendar days &nbsp; / &nbsp; <strong>{_uw_trades_str}</strong> trades</div>
                        </div>
                        <div>
                            <div style="color:#aaa; font-size:11px;">PEAK → TROUGH → RECOVERY</div>
                            <div style="font-size:13px;">{_pd_str} → {_td_str} → {_rd_str}</div>
                        </div>
                    </div>
                    <div style="margin-top: 10px; color:#aaa; font-size: 13px;">
                        Calendar Sharpe/Sortino use all trading days (idle days = 0 return). Active-period uses only days with an open position — more relevant when this strategy is one block in a multi-strategy book. Relationship: Sharpe_calendar ≈ Sharpe_active × √(TIM). Max DD @ Lows uses the intraday low envelope vs close-based running peak — reflects the worst mark you actually felt. R-units use starting-equity basis: DD_R = |DD_pct| / risk_bps × 100 (so 4% DD at 25 bps = 16 R). Underwater = days & trades between the pre-trough peak and the eventual recovery to new highs.
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ========== SECTION 3: INTRA-TRADE PATH ANALYSIS ==========
            if 'MAE_R' in trades_df.columns and trades_df['MAE_R'].notna().any():
                st.markdown("---")
                st.subheader("Intra-Trade Path (MAE / MFE / Give-back)")

                path = trades_df.dropna(subset=['MAE_R', 'MFE_R']).copy()
                wins_p = path[path['R'] > 0]
                losses_p = path[path['R'] <= 0]

                def _avg(s): return s.mean() if len(s) else 0.0

                avg_mae_all = _avg(path['MAE_R'])
                avg_mae_win = _avg(wins_p['MAE_R'])
                avg_mae_los = _avg(losses_p['MAE_R'])
                avg_mfe_all = _avg(path['MFE_R'])
                avg_mfe_win = _avg(wins_p['MFE_R'])
                avg_mfe_los = _avg(losses_p['MFE_R'])
                avg_gb_win  = _avg(wins_p['GiveBack_R']) if len(wins_p) else 0.0
                # Capture efficiency: total R on winners / total MFE on winners
                mfe_sum = wins_p['MFE_R'].sum()
                capture_eff = (wins_p['R'].sum() / mfe_sum * 100) if mfe_sum > 0 else 0.0

                st.markdown(f"""
                <div style="background-color: #0e1117; padding: 18px; border-radius: 10px; border: 1px solid #444; margin-top: 6px;">
                    <div style="display: flex; flex-wrap: wrap; gap: 28px;">
                        <div>
                            <div style="color:#aaa; font-size:12px;">AVG MAE (R)</div>
                            <div style="color:#ff6b6b;"><strong>All:</strong> {avg_mae_all:.2f} &nbsp; <strong>Win:</strong> {avg_mae_win:.2f} &nbsp; <strong>Loss:</strong> {avg_mae_los:.2f}</div>
                        </div>
                        <div>
                            <div style="color:#aaa; font-size:12px;">AVG MFE (R)</div>
                            <div style="color:#6bcf7f;"><strong>All:</strong> {avg_mfe_all:.2f} &nbsp; <strong>Win:</strong> {avg_mfe_win:.2f} &nbsp; <strong>Loss:</strong> {avg_mfe_los:.2f}</div>
                        </div>
                        <div>
                            <div style="color:#aaa; font-size:12px;">GIVE-BACK (winners, MFE − R)</div>
                            <div><strong>{avg_gb_win:.2f} R</strong> avg per winner</div>
                        </div>
                        <div>
                            <div style="color:#aaa; font-size:12px;">CAPTURE EFFICIENCY (winners)</div>
                            <div><strong>{capture_eff:.1f}%</strong> of MFE locked in</div>
                        </div>
                        <div>
                            <div style="color:#aaa; font-size:12px;">TRADES ANALYZED</div>
                            <div><strong>{len(path):,}</strong> ({len(wins_p):,} W / {len(losses_p):,} L)</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Build binned bar charts
                mae_edges = [-float('inf'), -2.0, -1.0, -0.5, -0.25, 0.0001]
                mae_labels = ['≤ -2R', '-2 to -1R', '-1 to -0.5R', '-0.5 to -0.25R', '0 to -0.25R']
                mfe_edges = [-0.0001, 0.25, 0.5, 1.0, 2.0, float('inf')]
                mfe_labels = ['0 to 0.25R', '0.25 to 0.5R', '0.5 to 1R', '1 to 2R', '≥ 2R']

                path['MAE_Bucket'] = pd.cut(path['MAE_R'], bins=mae_edges, labels=mae_labels, include_lowest=True)
                path['MFE_Bucket'] = pd.cut(path['MFE_R'], bins=mfe_edges, labels=mfe_labels, include_lowest=True)

                mae_group = path.groupby('MAE_Bucket', observed=True).agg(
                    avg_R=('R', 'mean'),
                    win_rate=('R', lambda s: (s > 0).mean() * 100),
                    count=('R', 'size'),
                ).reindex(mae_labels).fillna(0).reset_index()
                mfe_group = path.groupby('MFE_Bucket', observed=True).agg(
                    avg_R=('R', 'mean'),
                    win_rate=('R', lambda s: (s > 0).mean() * 100),
                    count=('R', 'size'),
                ).reindex(mfe_labels).fillna(0).reset_index()

                def _dual_axis_bar(df, x_col, title, x_title):
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=df[x_col], y=df['avg_R'], name='Avg Final R',
                        marker_color=['#ff6b6b' if v < 0 else '#6bcf7f' for v in df['avg_R']],
                        text=[f"n={int(c)}" for c in df['count']], textposition='outside',
                    ))
                    fig.add_trace(go.Scatter(
                        x=df[x_col], y=df['win_rate'], name='Win Rate (%)',
                        mode='lines+markers', yaxis='y2',
                        line=dict(color='rgba(200,200,200,0.9)', width=2, dash='dot'),
                        marker=dict(size=8),
                    ))
                    fig.update_layout(
                        title=title, xaxis_title=x_title,
                        yaxis=dict(title='Avg Final R', zeroline=True, zerolinecolor='rgba(255,255,255,0.3)'),
                        yaxis2=dict(title='Win Rate (%)', overlaying='y', side='right', range=[0, 100]),
                        height=380, margin=dict(t=60, b=60, l=50, r=50),
                        legend=dict(orientation='h', y=1.12, x=0.5, xanchor='center'),
                    )
                    return fig

                colA, colB = st.columns(2)
                colA.plotly_chart(
                    _dual_axis_bar(mae_group, 'MAE_Bucket',
                                   "Did trades that suffered more still tend to win?",
                                   'MAE bucket (worst intra-trade drawdown, R)'),
                    use_container_width=True,
                )
                colB.plotly_chart(
                    _dual_axis_bar(mfe_group, 'MFE_Bucket',
                                   "Did trades that ran further close strong, or fade?",
                                   'MFE bucket (best intra-trade run, R)'),
                    use_container_width=True,
                )

                st.caption(
                    "MAE/MFE computed from daily H/L. Entry day is pessimistic — only adverse "
                    "excursion counts (we don't know where in the bar our fill happened); MOC entries "
                    "skip entry day entirely. Exit day uses full H/L, which can slightly overstate MFE "
                    "if the peak printed after the exit fill. Intraday ordering within a bar is unknown, "
                    "so a trade that got stopped intraday and recovered to close profitable cannot be "
                    "detected at this resolution."
                )
        st.subheader("Trade Logs")
        tab1, tab2 = st.tabs(["Executed Trades", "Missed Trades"])
        with tab1:
            if not trades_df.empty:
                st.dataframe(trades_df.style.format({"Entry": "{:.2f}", "Exit": "{:.2f}", "R": "{:.2f}", "PnL_Dollar": "${:,.2f}", "Age": "{:.1f}y", "AvgVol": "{:,.0f}"}), use_container_width=True)
            else: st.info("No Executed Trades.")
        with tab2:
            if not rejected_df.empty:
                if 'PnL_Dollar' not in rejected_df.columns: rejected_df['PnL_Dollar'] = rejected_df['R'] * risk_per_trade
                st.dataframe(rejected_df.style.format({"Entry": "{:.2f}", "Exit": "{:.2f}", "R": "{:.2f}", "PnL_Dollar": "${:,.2f}", "Age": "{:.1f}y", "AvgVol": "{:,.0f}"}), use_container_width=True)
            else: st.info("No signals were rejected.")

if __name__ == "__main__":
    main()
