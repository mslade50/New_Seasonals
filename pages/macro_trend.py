"""Macro Trend Dashboard.

For every ticker in SECTOR_ETFS (the macro_seasonality universe), runs the
Higher-Low / Lower-High pullback-swing indicator on daily and weekly bars and
classifies the current trend state.

Classification rules (per timeframe):

    Most recent printed signal vs current close drives the state. The signal
    BEFORE it (if any) breaks ties between Definitive and Weak:

    Last HL, close > level
        prior signal was HL  -> Weak Uptrend   (HL printed, broke, no LH, new HL above us)
        otherwise            -> Definitive Uptrend
    Last LH, close < level
        prior signal was LH  -> Weak Downtrend (mirror)
        otherwise            -> Definitive Downtrend
    Last HL, close < level   -> Neutral (HL just broke, nothing new yet)
    Last LH, close > level   -> Neutral
    No signals               -> Neutral

This is a faithful Python port of the Pine Script v5 indicator dropped in
hh_hl.txt — regime filters (dual ROC + dual MA), wick-based 20-bar extremes,
5-bar pullback, locked-per-leg signaling, 50-bar cool-off after invalidation.
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime as dt
from datetime import timedelta
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from macro_seasonality import SECTOR_ETFS, TICKER_INFO
except Exception:
    # Fallback copy — keep in sync with pages/macro_seasonality.py if that module
    # cannot be imported in this context.
    SECTOR_ETFS = [
        "^GSPC", "^NDX", "^IXIC", "^DJI", "^DJT", "^RUT", "^MID", "^SOX",
        "GLD", "CEF", "SLV", "BTC-USD", "ETH-USD", "UNG", "UVXY",
        "EURUSD=X", "JPY=X", "GBPUSD=X", "AUDUSD=X", "NZDUSD=X",
        "CAD=X", "CHF=X", "DX-Y.NYB",
        "CL=F", "NG=F", "GC=F", "HG=F",
        "KC=F", "PL=F", "ZC=F", "ZW=F", "CC=F", "SB=F", "PA=F", "ZS=F",
        "CT=F", "SI=F",
        "^FTSE", "^GDAXI", "^FCHI", "^N225", "^HSI", "^STI",
        "^AXJO", "^KS11", "^TWII", "^BSESN", "^GSPTSE", "^MXX",
        "^BVSP", "^STOXX50E",
        "TLT", "IEF", "TIP", "LQD", "HYG", "AGG",
        "^VIX",
    ]
    TICKER_INFO = {}


# ---------------------------------------------------------------------------
# Pine Script port: HL/LH state machine
# ---------------------------------------------------------------------------

def detect_swings(df,
                  extreme_lookback: int = 20,
                  streak_len: int = 5,
                  cool_off_bars: int = 50,
                  use_regime: bool = True,
                  use_ma_filter: bool = True,
                  roc_fast_len: int = 20,
                  roc_slow_len: int = 50,
                  ma_fast_len: int = 20,
                  ma_slow_len: int = 50,
                  one_signal_per_leg: bool = True,
                  recovery_lookback: int = 63):
    """Bar-by-bar simulation matching the Pine Script in hh_hl.txt.

    Returns a chronologically ordered list of event dicts:
        {'idx', 'date', 'type', 'level'}
    For invalidation events ('HL_INV', 'LH_INV') additional keys are set:
        'recovery_level' — trailing `recovery_lookback`-bar high (HL_INV) or
                           low (LH_INV) measured at the violation bar.
        'cleared'        — True once a later close has crossed beyond
                           recovery_level (above for HL_INV, below for LH_INV).
        'cleared_idx'    — bar where clearing happened (only set if cleared).
    """
    n = len(df)
    min_history = max(roc_slow_len, ma_slow_len, extreme_lookback) + 5
    if n < min_history:
        return []

    o = df['Open'].to_numpy(dtype=float)
    h = df['High'].to_numpy(dtype=float)
    l = df['Low'].to_numpy(dtype=float)
    c = df['Close'].to_numpy(dtype=float)
    idx_arr = df.index

    c_series = pd.Series(c)
    ma_fast = c_series.rolling(ma_fast_len).mean().to_numpy()
    ma_slow = c_series.rolling(ma_slow_len).mean().to_numpy()
    roc_fast = c_series.pct_change(roc_fast_len).to_numpy() * 100.0
    roc_slow = c_series.pct_change(roc_slow_len).to_numpy() * 100.0
    prior_high = pd.Series(h).rolling(extreme_lookback).max().shift(1).to_numpy()
    prior_low = pd.Series(l).rolling(extreme_lookback).min().shift(1).to_numpy()

    events = []
    last_hl = np.nan
    last_lh = np.nan
    invalid_hl = np.nan
    invalid_lh = np.nan
    hl_pause_end = -1
    lh_pause_end = -1
    hl_locked = False
    lh_locked = False

    new_high_arr = np.zeros(n, dtype=bool)
    new_low_arr = np.zeros(n, dtype=bool)
    bsh_arr = np.full(n, np.inf)
    bsl_arr = np.full(n, np.inf)
    hl_setup_arr = np.zeros(n, dtype=bool)
    lh_setup_arr = np.zeros(n, dtype=bool)

    bsh = np.inf
    bsl = np.inf

    # Pending invalidation events waiting for their recovery_level to be crossed.
    pending_hl_invs = []
    pending_lh_invs = []

    for i in range(n):
        # Check whether any pending invalidations have been cleared this bar
        if pending_hl_invs:
            remaining = []
            for ev in pending_hl_invs:
                if c[i] > ev['recovery_level']:
                    ev['cleared'] = True
                    ev['cleared_idx'] = i
                else:
                    remaining.append(ev)
            pending_hl_invs = remaining
        if pending_lh_invs:
            remaining = []
            for ev in pending_lh_invs:
                if c[i] < ev['recovery_level']:
                    ev['cleared'] = True
                    ev['cleared_idx'] = i
                else:
                    remaining.append(ev)
            pending_lh_invs = remaining

        if np.isnan(prior_high[i]) or np.isnan(prior_low[i]):
            bsh_arr[i] = bsh
            bsl_arr[i] = bsl
            continue

        new_high_arr[i] = h[i] > prior_high[i]
        new_low_arr[i] = l[i] < prior_low[i]
        bsh = 0 if new_high_arr[i] else bsh + 1
        bsl = 0 if new_low_arr[i] else bsl + 1
        bsh_arr[i] = bsh
        bsl_arr[i] = bsl

        if use_regime and not np.isnan(roc_fast[i]) and not np.isnan(roc_slow[i]):
            up_roc = roc_fast[i] > 0 and roc_slow[i] > 0
            down_roc = roc_fast[i] < 0 and roc_slow[i] < 0
        elif use_regime:
            up_roc = False
            down_roc = False
        else:
            up_roc = True
            down_roc = True

        if (use_ma_filter and i > 0
                and not (np.isnan(ma_fast[i]) or np.isnan(ma_slow[i])
                         or np.isnan(ma_fast[i-1]) or np.isnan(ma_slow[i-1]))):
            ma_bullish = (c[i] > ma_fast[i] and c[i] > ma_slow[i]
                          and ma_fast[i] > ma_fast[i-1] and ma_slow[i] > ma_slow[i-1])
            ma_bearish = (c[i] < ma_fast[i] and c[i] < ma_slow[i]
                          and ma_fast[i] < ma_fast[i-1] and ma_slow[i] < ma_slow[i-1])
        elif use_ma_filter:
            ma_bullish = False
            ma_bearish = False
        else:
            ma_bullish = True
            ma_bearish = True

        uptrend = up_roc and ma_bullish
        downtrend = down_roc and ma_bearish

        green = c[i] > o[i]
        red = c[i] < o[i]
        hl_setup_arr[i] = green and bsh >= streak_len and uptrend
        lh_setup_arr[i] = red and bsl >= streak_len and downtrend

        hl_confirm = i > 0 and hl_setup_arr[i-1] and c[i] > h[i-1]
        lh_confirm = i > 0 and lh_setup_arr[i-1] and c[i] < l[i-1]

        if new_high_arr[i]:
            hl_locked = False
        if new_low_arr[i]:
            lh_locked = False

        if hl_pause_end >= 0 and i >= hl_pause_end:
            invalid_hl = np.nan
            hl_pause_end = -1
        if lh_pause_end >= 0 and i >= lh_pause_end:
            invalid_lh = np.nan
            lh_pause_end = -1

        hl_invalidated = False
        if not np.isnan(last_hl) and c[i] < last_hl:
            invalid_hl = last_hl
            look_start = max(0, i - recovery_lookback + 1)
            recovery_high = float(np.max(h[look_start:i + 1]))
            inv_ev = {
                'idx': i, 'date': idx_arr[i], 'type': 'HL_INV',
                'level': float(last_hl), 'recovery_level': recovery_high,
                'cleared': False,
            }
            events.append(inv_ev)
            pending_hl_invs.append(inv_ev)
            last_hl = np.nan
            hl_pause_end = i + cool_off_bars
            hl_locked = False
            hl_invalidated = True

        if not hl_invalidated and not np.isnan(invalid_hl) and c[i] > invalid_hl:
            invalid_hl = np.nan
            hl_pause_end = -1

        lh_invalidated = False
        if not np.isnan(last_lh) and c[i] > last_lh:
            invalid_lh = last_lh
            look_start = max(0, i - recovery_lookback + 1)
            recovery_low = float(np.min(l[look_start:i + 1]))
            inv_ev = {
                'idx': i, 'date': idx_arr[i], 'type': 'LH_INV',
                'level': float(last_lh), 'recovery_level': recovery_low,
                'cleared': False,
            }
            events.append(inv_ev)
            pending_lh_invs.append(inv_ev)
            last_lh = np.nan
            lh_pause_end = i + cool_off_bars
            lh_locked = False
            lh_invalidated = True

        if not lh_invalidated and not np.isnan(invalid_lh) and c[i] < invalid_lh:
            invalid_lh = np.nan
            lh_pause_end = -1

        hl_free = (np.isnan(invalid_hl) and hl_pause_end < 0
                   and (not one_signal_per_leg or not hl_locked))
        lh_free = (np.isnan(invalid_lh) and lh_pause_end < 0
                   and (not one_signal_per_leg or not lh_locked))

        if hl_confirm and hl_free:
            prev_bsh = bsh_arr[i-1]
            pb_len = int(prev_bsh) if np.isfinite(prev_bsh) else 1
            pb_len = max(1, pb_len)
            min_low = l[i-1]
            min_off = 1
            for k in range(1, min(500, pb_len) + 1):
                if i - k < 0:
                    break
                if l[i - k] < min_low:
                    min_low = l[i - k]
                    min_off = k
            if np.isnan(last_hl) or min_low > last_hl:
                last_hl = min_low
                hl_locked = True
                swing_i = i - min_off
                events.append({
                    'idx': i, 'date': idx_arr[i],
                    'swing_idx': swing_i, 'swing_date': idx_arr[swing_i],
                    'type': 'HL', 'level': float(min_low),
                })

        if lh_confirm and lh_free:
            prev_bsl = bsl_arr[i-1]
            pb_len = int(prev_bsl) if np.isfinite(prev_bsl) else 1
            pb_len = max(1, pb_len)
            max_high = h[i-1]
            max_off = 1
            for k in range(1, min(500, pb_len) + 1):
                if i - k < 0:
                    break
                if h[i - k] > max_high:
                    max_high = h[i - k]
                    max_off = k
            if np.isnan(last_lh) or max_high < last_lh:
                last_lh = max_high
                lh_locked = True
                swing_i = i - max_off
                events.append({
                    'idx': i, 'date': idx_arr[i],
                    'swing_idx': swing_i, 'swing_date': idx_arr[swing_i],
                    'type': 'LH', 'level': float(max_high),
                })

    return events


def detect_reversals(df,
                     rev_len: int = 21,
                     rev_len2: int = 42,
                     rev_atr_len: int = 14,
                     rev_decline_atr: float = 5.0,
                     rev_bounce_atr: float = 1.5):
    """Port of the Pine Script reversal-triangle signal in hh_hl.txt.

    Fires on a 2-bar bullish-engulfing-style confirmation when the prior bar
    or the bar before it printed a trailing rev_len low AND the decline into
    that low was at least rev_decline_atr × ATR over either 21 or 42 bars AND
    the 2-bar bounce off the close 2 bars back is at least rev_bounce_atr × ATR.

    Returns events with type='REV', plotted at the bottom bar's low.
    """
    n = len(df)
    if n < max(rev_len, rev_len2, rev_atr_len) + 5:
        return []

    h = df['High'].to_numpy(dtype=float)
    l = df['Low'].to_numpy(dtype=float)
    c = df['Close'].to_numpy(dtype=float)
    o = df['Open'].to_numpy(dtype=float)
    idx_arr = df.index

    prev_close = pd.Series(c).shift(1).to_numpy()
    tr = np.maximum.reduce([
        h - l,
        np.abs(h - prev_close),
        np.abs(l - prev_close),
    ])
    atr = pd.Series(tr).rolling(rev_atr_len).mean().to_numpy()
    rev_low_window = pd.Series(l).rolling(rev_len).min().to_numpy()
    is_rev_low = l <= rev_low_window

    events = []
    min_i = max(rev_len, rev_len2, rev_atr_len) + 2
    for i in range(min_i, n):
        if not (c[i-1] > o[i-1]):
            continue
        if not (c[i] > h[i-1]):
            continue

        if is_rev_low[i-1]:
            bottom_off = 1
        elif is_rev_low[i-2]:
            bottom_off = 2
        else:
            continue

        bottom_idx = i - bottom_off
        rev_low = l[bottom_idx]
        rev_atr_bot = atr[bottom_idx]
        if np.isnan(rev_atr_bot) or np.isnan(atr[i]):
            continue

        idx21 = bottom_idx - rev_len
        idx42 = bottom_idx - rev_len2
        decline_ok = False
        if idx21 >= 0 and (rev_low - c[idx21]) <= -rev_decline_atr * rev_atr_bot:
            decline_ok = True
        if not decline_ok and idx42 >= 0 and (rev_low - c[idx42]) <= -rev_decline_atr * rev_atr_bot:
            decline_ok = True
        if not decline_ok:
            continue

        if (c[i] - c[i-2]) < rev_bounce_atr * atr[i]:
            continue

        if events and events[-1]['idx'] == bottom_idx:
            continue
        events.append({
            'idx': bottom_idx,
            'date': idx_arr[bottom_idx],
            'type': 'REV',
            'level': float(rev_low),
        })

    return events


def classify_trend(events, current_close):
    """Return one of {Definitive/Weak Uptrend, Definitive/Weak Downtrend, Neutral}.

    "Weak" requires the precise sequence the user described:
        prior signal same direction, an invalidation event between them, AND
        that invalidation's recovery_level has not been crossed since.
    Two consecutive same-direction signals with NO invalidation between them is
    a continued trend → Definitive.
    """
    if not events or current_close is None or np.isnan(current_close):
        return 'Neutral'
    signals = [e for e in events if e['type'] in ('HL', 'LH')]
    if not signals:
        return 'Neutral'
    last = signals[-1]
    last_type = last['type']
    inv_type = 'HL_INV' if last_type == 'HL' else 'LH_INV'

    # If the most recent signal was itself invalidated and nothing new has
    # printed since, we're between signals → Neutral.
    if any(e['type'] == inv_type and e['idx'] > last['idx'] for e in events):
        return 'Neutral'

    def is_weak():
        if len(signals) < 2 or signals[-2]['type'] != last_type:
            return False
        prior = signals[-2]
        invs_between = [e for e in events
                        if e['type'] == inv_type
                        and prior['idx'] < e['idx'] < last['idx']]
        uncleared = [e for e in invs_between if not e.get('cleared')]
        return bool(uncleared)

    if last_type == 'LH':
        if current_close >= last['level']:
            return 'Neutral'
        return 'Weak Downtrend' if is_weak() else 'Definitive Downtrend'
    if current_close <= last['level']:
        return 'Neutral'
    return 'Weak Uptrend' if is_weak() else 'Definitive Uptrend'


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_price_panel(tickers, years_history=20):
    """Returns {ticker: {'daily': DataFrame, 'weekly': DataFrame}}."""
    end = dt.datetime.now() + timedelta(days=5)
    start = end - timedelta(days=years_history * 365 + 60)
    raw = yf.download(
        tickers, start=start, end=end, interval='1d',
        auto_adjust=True, group_by='ticker', progress=False, threads=True,
    )

    out = {}
    for t in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                top_level = raw.columns.get_level_values(0)
                if t not in top_level:
                    continue
                df = raw[t].copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
            else:
                df = raw.copy()
            df = df.dropna(subset=['Close'])
            if df.empty:
                continue
            for col in ('Open', 'High', 'Low', 'Close'):
                if col not in df.columns:
                    df = pd.DataFrame()
                    break
            if df.empty:
                continue
            weekly = df.resample('W-FRI').agg({
                'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
            }).dropna()
            out[t] = {'daily': df, 'weekly': weekly}
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Styling + UI
# ---------------------------------------------------------------------------

TREND_ORDER = {
    'Definitive Uptrend': 5,
    'Weak Uptrend': 4,
    'Neutral': 3,
    'Weak Downtrend': 2,
    'Definitive Downtrend': 1,
}

TREND_STYLE = {
    'Definitive Uptrend': 'background-color: #006400; color: white;',
    'Weak Uptrend': 'background-color: #98FB98; color: #003300;',
    'Neutral': 'background-color: #2b2b2b; color: #cccccc;',
    'Weak Downtrend': 'background-color: #FFB6B6; color: #5b0000;',
    'Definitive Downtrend': 'background-color: #8B0000; color: white;',
}


def _last_signal_summary(events):
    signals = [e for e in events if e['type'] in ('HL', 'LH')]
    if not signals:
        return ''
    last = signals[-1]
    date_str = pd.Timestamp(last['date']).strftime('%Y-%m-%d')
    return f"{last['type']} @ {last['level']:.2f} ({date_str})"


def _missing_date_gaps(df):
    """Return list of timestamps in [df.start, df.end] that are NOT in df.index.
    Handles both weekday-only series (stocks: skips weekends + holidays) and
    7-day series (crypto: skips nothing). Empty list if df is empty."""
    if df.empty:
        return []
    full = pd.date_range(start=df.index[0].normalize(),
                         end=df.index[-1].normalize(), freq='D')
    present = pd.DatetimeIndex(df.index.normalize().unique())
    return list(full.difference(present))


def render_signal_chart(ticker, df, events, title, default_view_years=None):
    import plotly.graph_objects as go

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name='Price', increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
    ))

    hl_x, hl_y = [], []
    lh_x, lh_y = [], []
    hl_inv_x, hl_inv_y = [], []
    lh_inv_x, lh_inv_y = [], []
    rev_x, rev_y = [], []
    for e in events:
        if e['type'] == 'HL':
            hl_x.append(e.get('swing_date', e['date'])); hl_y.append(e['level'])
        elif e['type'] == 'LH':
            lh_x.append(e.get('swing_date', e['date'])); lh_y.append(e['level'])
        elif e['type'] == 'HL_INV':
            hl_inv_x.append(e['date']); hl_inv_y.append(e['level'])
        elif e['type'] == 'LH_INV':
            lh_inv_x.append(e['date']); lh_inv_y.append(e['level'])
        elif e['type'] == 'REV':
            rev_x.append(e['date']); rev_y.append(e['level'])

    if hl_x:
        fig.add_trace(go.Scatter(x=hl_x, y=hl_y, mode='markers', name='HL',
                                 marker=dict(symbol='triangle-up', size=12, color='#39FF14',
                                             line=dict(width=1, color='black'))))
    if lh_x:
        fig.add_trace(go.Scatter(x=lh_x, y=lh_y, mode='markers', name='LH',
                                 marker=dict(symbol='triangle-down', size=12, color='#ff4d4d',
                                             line=dict(width=1, color='black'))))
    if rev_x:
        fig.add_trace(go.Scatter(x=rev_x, y=rev_y, mode='markers', name='Reversal',
                                 marker=dict(symbol='triangle-up', size=14, color='#3b82f6',
                                             line=dict(width=1, color='black'))))
    if hl_inv_x:
        fig.add_trace(go.Scatter(x=hl_inv_x, y=hl_inv_y, mode='markers', name='HL broken',
                                 marker=dict(symbol='x', size=8, color='rgba(57,255,20,0.5)')))
    if lh_inv_x:
        fig.add_trace(go.Scatter(x=lh_inv_x, y=lh_inv_y, mode='markers', name='LH broken',
                                 marker=dict(symbol='x', size=8, color='rgba(255,77,77,0.5)')))

    xaxis_kwargs = {}
    yaxis_kwargs = {}
    if default_view_years is not None and not df.empty:
        end = df.index[-1]
        start = end - pd.Timedelta(days=int(default_view_years * 365))
        if start <= df.index[0]:
            start = df.index[0]
        xaxis_kwargs['range'] = [start, end]
        # Auto-fit y-axis to candles + any markers within the visible window.
        view = df.loc[start:end]
        if not view.empty:
            y_lo = float(view['Low'].min())
            y_hi = float(view['High'].max())
            # Include any HL/LH/inv markers that fall in the window
            for marker_x, marker_y in (
                list(zip(hl_x, hl_y)) + list(zip(lh_x, lh_y))
                + list(zip(hl_inv_x, hl_inv_y)) + list(zip(lh_inv_x, lh_inv_y))
                + list(zip(rev_x, rev_y))
            ):
                if start <= pd.Timestamp(marker_x) <= end:
                    y_lo = min(y_lo, marker_y)
                    y_hi = max(y_hi, marker_y)
            pad = (y_hi - y_lo) * 0.08 if y_hi > y_lo else abs(y_hi) * 0.05 + 1
            yaxis_kwargs['range'] = [y_lo - pad, y_hi + pad]

    gaps = _missing_date_gaps(df)
    if gaps:
        xaxis_kwargs['rangebreaks'] = [dict(values=gaps)]

    fig.update_layout(
        title=title, height=500, plot_bgcolor='black', paper_bgcolor='black',
        font=dict(color='white'), margin=dict(l=10, r=10, t=40, b=10),
        xaxis_rangeslider_visible=False,
        xaxis=xaxis_kwargs,
        yaxis=yaxis_kwargs,
        legend=dict(bgcolor='rgba(20,20,20,0.8)', orientation='h',
                    yanchor='bottom', y=-0.15, xanchor='left', x=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    st.set_page_config(layout='wide', page_title='Macro Trend')
    st.title('Macro Trend')
    st.caption(
        'Higher-low / lower-high pullback swings across the macro universe. '
        'Daily and weekly columns reflect the trend state implied by the most '
        'recent printed signal vs current close.'
    )

    with st.sidebar:
        st.header('Detection params')
        extreme_lookback = st.number_input('New-high/low lookback', 5, 60, 20)
        streak_len = st.number_input('Min pullback bars', 1, 20, 4)
        cool_off = st.number_input('Cool-off after invalidation', 1, 200, 50)
        use_regime = st.checkbox('ROC regime filter', value=True)
        use_ma = st.checkbox('MA trend filter', value=True)
        one_per_leg = st.checkbox('One signal per swing leg', value=True)
        enable_reversal = st.checkbox('Reversal signals (blue)', value=True)
        st.divider()
        st.caption('Weak→Definitive: once price crosses the trailing N-bar '
                   'high (HL) / low (LH) at the violation, the prior break is '
                   'considered fully recovered.')
        recov_daily = st.number_input('Recovery lookback (daily bars)', 5, 252, 63)
        recov_weekly = st.number_input('Recovery lookback (weekly bars)', 2, 52, 13)
        st.divider()
        show_detail_cols = st.checkbox('Show last-signal columns', value=False)

        if st.button('Refresh data'):
            st.cache_data.clear()
            st.rerun()

    tickers = sorted(set(SECTOR_ETFS))
    with st.spinner(f'Fetching {len(tickers)} tickers...'):
        panel = fetch_price_panel(tickers)

    base_params = dict(
        extreme_lookback=extreme_lookback, streak_len=streak_len,
        cool_off_bars=cool_off, use_regime=use_regime, use_ma_filter=use_ma,
        one_signal_per_leg=one_per_leg,
    )

    rows = []
    events_cache = {}
    for t in tickers:
        if t not in panel:
            continue
        daily_df = panel[t]['daily']
        weekly_df = panel[t]['weekly']
        if daily_df.empty:
            continue

        daily_events = detect_swings(daily_df, recovery_lookback=recov_daily, **base_params)
        weekly_events = detect_swings(weekly_df, recovery_lookback=recov_weekly, **base_params)
        if enable_reversal:
            daily_events = sorted(daily_events + detect_reversals(daily_df), key=lambda e: e['idx'])
            weekly_events = sorted(weekly_events + detect_reversals(weekly_df), key=lambda e: e['idx'])
        events_cache[t] = {'daily': daily_events, 'weekly': weekly_events}

        cur_daily_close = float(daily_df['Close'].iloc[-1])
        cur_weekly_close = float(weekly_df['Close'].iloc[-1]) if not weekly_df.empty else cur_daily_close
        daily_state = classify_trend(daily_events, cur_daily_close)
        weekly_state = classify_trend(weekly_events, cur_weekly_close)

        info = TICKER_INFO.get(t, ('', ''))
        row = {
            'Ticker': t,
            'Name': info[0] if info else '',
            'IBKR': info[1] if info else '',
            'Price': cur_daily_close,
            'Daily': daily_state,
            'Weekly': weekly_state,
        }
        if show_detail_cols:
            row['Daily Last'] = _last_signal_summary(daily_events)
            row['Weekly Last'] = _last_signal_summary(weekly_events)
        rows.append(row)

    if not rows:
        st.error('No data returned for any ticker.')
        return

    df = pd.DataFrame(rows)
    df['_w'] = df['Weekly'].map(TREND_ORDER).fillna(3)
    df['_d'] = df['Daily'].map(TREND_ORDER).fillna(3)
    df = df.sort_values(['_w', '_d'], ascending=False).drop(columns=['_w', '_d']).reset_index(drop=True)

    styled = (df.style
              .format({'Price': '{:.2f}'})
              .map(lambda v: TREND_STYLE.get(v, ''), subset=['Daily', 'Weekly']))
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ---------- Per-ticker inspection ----------
    st.divider()
    st.subheader('Inspect ticker')
    available = [t for t in tickers if t in events_cache]
    if available:
        sel = st.selectbox('Ticker', options=available, index=0)
        if sel:
            daily_df = panel[sel]['daily']
            weekly_df = panel[sel]['weekly']
            c1, c2 = st.columns(2)
            with c1:
                render_signal_chart(sel, daily_df,
                                    events_cache[sel]['daily'],
                                    f'{sel} — Daily',
                                    default_view_years=1)
            with c2:
                render_signal_chart(sel, weekly_df,
                                    events_cache[sel]['weekly'],
                                    f'{sel} — Weekly',
                                    default_view_years=5)


if __name__ == '__main__':
    main()
