"""
Exposure Leg — fragility-gated buy-and-hold overlay rendered at the top of
the AM daily-scan email.

Base allocation: 25% of ACCOUNT_VALUE split 50% VOO / 50% QQQ
(so 12.5% / 12.5% of total account at the 1.0x multiplier).

Two rule layers stacked multiplicatively (final_weight =
base_weight × global_mult × ticker_mult):

GLOBAL rules (apply to ALL tickers, fragility-driven, "ALL ×" semantics).
Zero precedence over boost — if a zero-rule fires, it wins:
    1. ALL × 0.00 when Raw 21D fragility > 50
    2. ALL × 1.25 when Raw 21D < 5 AND Raw 63D < 5
    3. ALL × 0.00 when 10d-MA 63D fragility > 50
    else 1.00

PER-TICKER rules (zero out a single ticker's weight independently):
    4. ticker × 0.00 when self 2D rank > 85 AND self 5D rank > 85
                          AND self 21D rank > 85 (multi-horizon overbought)
    5. ticker × 0.00 when self 200-SMA distance in [-10%, -3%]
                          (3-10% below the 200 SMA — "knife-catching" zone
                          before deeper-pullback bounce probabilities improve)

"Raw" = the values stored directly in data/rd2_fragility.parquet (which
daily_risk_report writes after a 5d rolling smooth — same convention used
across exposure_backtester.py and daily_scan's sizing block).

State across runs is persisted in data/exposure_state.json so the email can
diff today's targets vs yesterday's and emit "necessary orders today" only
when any per-ticker target weight flips.
"""
from __future__ import annotations

import json
import os
import datetime
from typing import Dict, Optional

import pandas as pd

BASE_WEIGHTS: Dict[str, float] = {
    'VOO': 0.125,  # 50% of 25%
    'QQQ': 0.125,  # 50% of 25%
}
EXPOSURE_TICKERS = list(BASE_WEIGHTS.keys())

# Per-ticker rule constants (centralized so the backtester preset stays in sync).
PT_OVERBOUGHT_WINDOWS = (2, 5, 21)
PT_OVERBOUGHT_THRESHOLD = 85.0
PT_SMA_PERIOD = 200
PT_SMA_DIST_BAND = (-10.0, -3.0)  # ticker × 0 when distance falls in this band
RANK_MIN_PERIODS = 252  # matches exposure_backtester.compute_return_rank

# File paths are resolved relative to the daily_scan working directory.
FRAGILITY_PATH = os.path.join('data', 'rd2_fragility.parquet')
STATE_PATH = os.path.join('data', 'exposure_state.json')


# ---------------------------------------------------------------------------
# DIAL / RULE EVALUATION
# ---------------------------------------------------------------------------
def _read_dial_readings(frag_path: str) -> Optional[Dict[str, float]]:
    """Return today's dial readings: raw 21d, raw 63d, 10d-MA 63d.

    Returns None if the parquet is missing or unreadable.
    """
    if not os.path.exists(frag_path):
        return None
    try:
        df = pd.read_parquet(frag_path)
    except Exception:
        return None
    if df is None or df.empty:
        return None
    if '21d' not in df.columns or '63d' not in df.columns:
        return None
    df = df.sort_index()
    raw_21d = float(df['21d'].dropna().iloc[-1])
    raw_63d = float(df['63d'].dropna().iloc[-1])
    ma10_63d = float(df['63d'].dropna().rolling(10, min_periods=1).mean().iloc[-1])
    return {
        'raw_21d': raw_21d,
        'raw_63d': raw_63d,
        'ma10_63d': ma10_63d,
        'asof': df.index[-1].strftime('%Y-%m-%d'),
    }


def _apply_global_rules(readings: Dict[str, float]) -> Dict:
    """Fragility-driven multiplier applied to all tickers. Zero wins over boost."""
    raw_21d = readings['raw_21d']
    raw_63d = readings['raw_63d']
    ma10_63d = readings['ma10_63d']

    if raw_21d > 50:
        return {'mult': 0.0, 'active_rule': 'Rule 1',
                'reason': f'Raw 21D fragility {raw_21d:.1f} > 50'}
    if ma10_63d > 50:
        return {'mult': 0.0, 'active_rule': 'Rule 3',
                'reason': f'10d-MA 63D fragility {ma10_63d:.1f} > 50'}
    if raw_21d < 5 and raw_63d < 5:
        return {'mult': 1.25, 'active_rule': 'Rule 2',
                'reason': f'Raw 21D {raw_21d:.1f} < 5 AND Raw 63D {raw_63d:.1f} < 5'}
    return {'mult': 1.0, 'active_rule': None,
            'reason': 'No global rule active — base allocation'}


# ---------------------------------------------------------------------------
# PER-TICKER INDICATORS + RULES
# ---------------------------------------------------------------------------
def _compute_return_rank(close: pd.Series, window: int,
                         min_periods: int = RANK_MIN_PERIODS) -> pd.Series:
    """Expanding percentile rank of N-day returns (0-100). Matches
    exposure_backtester.compute_return_rank semantics so a 21D rank > 85
    here means the same thing as in the backtester preset.
    """
    ret = close.pct_change(window, fill_method=None)
    return ret.expanding(min_periods=min_periods).rank(pct=True) * 100.0


def _compute_sma_distance(close: pd.Series, period: int = PT_SMA_PERIOD) -> pd.Series:
    """% distance from N-day SMA. Positive = above MA, negative = below."""
    sma = close.rolling(period, min_periods=max(2, period // 4)).mean()
    return (close - sma) / sma * 100.0


def _evaluate_per_ticker_rules_at(rank_2d, rank_5d, rank_21d, sma_dist) -> Dict:
    """Apply rules 4 + 5 to a single ticker's latest readings. Used by both
    the live snapshot and the backtest loop (which calls it per row)."""
    readings = {
        'rank_2d': float(rank_2d) if pd.notna(rank_2d) else None,
        'rank_5d': float(rank_5d) if pd.notna(rank_5d) else None,
        'rank_21d': float(rank_21d) if pd.notna(rank_21d) else None,
        'sma200_dist': float(sma_dist) if pd.notna(sma_dist) else None,
    }
    # Rule 4: multi-horizon overbought
    if (pd.notna(rank_2d) and pd.notna(rank_5d) and pd.notna(rank_21d)
            and rank_2d > PT_OVERBOUGHT_THRESHOLD
            and rank_5d > PT_OVERBOUGHT_THRESHOLD
            and rank_21d > PT_OVERBOUGHT_THRESHOLD):
        return {
            'mult': 0.0, 'active_rule': 'Rule 4',
            'reason': (f'2D rank {rank_2d:.0f} / 5D rank {rank_5d:.0f} / '
                       f'21D rank {rank_21d:.0f} all > {PT_OVERBOUGHT_THRESHOLD:.0f}'),
            'readings': readings,
        }
    # Rule 5: 200 SMA distance in the knife-catch band
    lo, hi = PT_SMA_DIST_BAND
    if pd.notna(sma_dist) and lo <= sma_dist <= hi:
        return {
            'mult': 0.0, 'active_rule': 'Rule 5',
            'reason': f'200 SMA distance {sma_dist:+.1f}% in [{lo:.0f}%, {hi:.0f}%]',
            'readings': readings,
        }
    return {
        'mult': 1.0, 'active_rule': None,
        'reason': 'No per-ticker rule active',
        'readings': readings,
    }


def _evaluate_per_ticker_rules(close_series: Optional[pd.Series]) -> Dict:
    """Evaluate per-ticker rules 4 + 5 against today's readings."""
    if close_series is None or close_series.empty:
        return {'mult': 1.0, 'active_rule': None, 'reason': 'No price data',
                'readings': {}}
    rank_2d = _compute_return_rank(close_series, 2).iloc[-1]
    rank_5d = _compute_return_rank(close_series, 5).iloc[-1]
    rank_21d = _compute_return_rank(close_series, 21).iloc[-1]
    sma_dist = _compute_sma_distance(close_series, PT_SMA_PERIOD).iloc[-1]
    return _evaluate_per_ticker_rules_at(rank_2d, rank_5d, rank_21d, sma_dist)


# ---------------------------------------------------------------------------
# PRICE / HISTORY RESOLUTION
# ---------------------------------------------------------------------------
def _resolve_close_series(prices_dict: Optional[dict], ticker: str) -> Optional[pd.Series]:
    """Pull a close-price Series from prices_dict (master_dict-style) or yfinance.

    Returns a tz-naive normalized Series, or None.
    """
    if prices_dict is not None:
        df = prices_dict.get(ticker)
        if df is not None and not df.empty:
            cols = {c.lower(): c for c in df.columns}
            cc = cols.get('close')
            if cc is not None:
                s = df[cc].copy()
                s.index = pd.to_datetime(s.index)
                if s.index.tz is not None:
                    s.index = s.index.tz_localize(None)
                s.index = s.index.normalize()
                return s.dropna()
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period='max', auto_adjust=False)
        if hist is None or hist.empty:
            return None
        s = hist['Close'].copy()
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s.index = s.index.normalize()
        return s.dropna()
    except Exception:
        return None


# ---------------------------------------------------------------------------
# LIVE SNAPSHOT
# ---------------------------------------------------------------------------
def compute_exposure_targets(
    account_value: float,
    master_dict: Optional[dict] = None,
    frag_path: str = FRAGILITY_PATH,
) -> Optional[Dict]:
    """Build today's exposure target snapshot.

    Returns None if fragility data is unavailable. Otherwise returns:
      {
        'asof', 'mult' (global), 'active_rule' (global), 'reason' (global),
        'readings': {raw_21d, raw_63d, ma10_63d},
        'targets': {
            ticker: {
                weight, dollars, shares, price,
                ticker_mult, ticker_rule, ticker_reason,
                ticker_readings: {rank_2d, rank_5d, rank_21d, sma200_dist},
            },
        },
        'account_value': float,
      }
    """
    readings = _read_dial_readings(frag_path)
    if readings is None:
        return None

    global_rule = _apply_global_rules(readings)
    global_mult = global_rule['mult']

    targets: Dict[str, Dict] = {}
    for tkr, base_w in BASE_WEIGHTS.items():
        close_s = _resolve_close_series(master_dict, tkr)
        pt_rule = _evaluate_per_ticker_rules(close_s)
        ticker_mult = pt_rule['mult']
        target_w = base_w * global_mult * ticker_mult
        dollars = account_value * target_w
        price = float(close_s.iloc[-1]) if (close_s is not None and not close_s.empty) else None
        shares = int(round(dollars / price)) if (price and price > 0 and target_w > 0) else (0 if target_w == 0 else None)
        targets[tkr] = {
            'weight': target_w,
            'dollars': dollars,
            'price': price,
            'shares': shares,
            'ticker_mult': ticker_mult,
            'ticker_rule': pt_rule['active_rule'],
            'ticker_reason': pt_rule['reason'],
            'ticker_readings': pt_rule['readings'],
        }

    return {
        'asof': readings['asof'],
        'computed_at': datetime.datetime.utcnow().isoformat() + 'Z',
        'mult': global_mult,
        'active_rule': global_rule['active_rule'],
        'reason': global_rule['reason'],
        'readings': {
            'raw_21d': readings['raw_21d'],
            'raw_63d': readings['raw_63d'],
            'ma10_63d': readings['ma10_63d'],
        },
        'targets': {k: {kk: vv for kk, vv in v.items()} for k, v in targets.items()},
        'account_value': account_value,
    }


# ---------------------------------------------------------------------------
# STATE PERSISTENCE
# ---------------------------------------------------------------------------
def load_prior_state(path: str = STATE_PATH) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def save_state(snapshot: Dict, path: str = STATE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(snapshot, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# BACKTEST
# ---------------------------------------------------------------------------
def compute_exposure_leg_backtest(
    start_date,
    end_date,
    prices_dict,
    starting_equity: float,
    frag_path: str = FRAGILITY_PATH,
):
    """Daily $ P&L series for the exposure leg over [start_date, end_date].

    Sizes the leg at base × global_mult × per-ticker_mult of `starting_equity`.
    BASE_WEIGHTS sums to 0.25 (the 25% production allocation), so max gross is
    25% of equity at 1.0x global and 31.25% at 1.25x global. Per-ticker rules
    can independently zero out individual tickers.

    Uses prior-day target weights (lagged 1 trading day) so today's PnL
    reflects yesterday's close-of-business decision — same convention as
    the strat backtester's daily MTM.

    Returns:
      {
        'daily_pnl': Series of $ P&L per day,
        'global_mult': Series of global multipliers per day,
        'ticker_mult': DataFrame of per-ticker multipliers per day,
        'target_weights': DataFrame of per-ticker fractional weights per day,
        'close_df': DataFrame of close prices for VOO/QQQ,
        'errors': list[str],
      }
    Empty 'daily_pnl' Series if data is missing.
    """
    errors = []
    empty = {
        'daily_pnl': pd.Series(dtype=float),
        'global_mult': pd.Series(dtype=float),
        'ticker_mult': pd.DataFrame(),
        'target_weights': pd.DataFrame(),
        'close_df': pd.DataFrame(),
        'errors': errors,
    }

    if not os.path.exists(frag_path):
        errors.append(f"Fragility parquet missing: {frag_path}")
        return empty
    try:
        frag = pd.read_parquet(frag_path)
    except Exception as e:
        errors.append(f"Failed to read fragility parquet: {e}")
        return empty
    if frag is None or frag.empty or '21d' not in frag.columns or '63d' not in frag.columns:
        errors.append("Fragility parquet missing 21d/63d columns")
        return empty

    frag = frag.sort_index()
    frag.index = pd.to_datetime(frag.index)
    if frag.index.tz is not None:
        frag.index = frag.index.tz_localize(None)
    frag.index = frag.index.normalize()

    # Global multiplier time series
    raw_21d = frag['21d']
    raw_63d = frag['63d']
    ma10_63d = raw_63d.rolling(10, min_periods=1).mean()
    rule1_active = raw_21d > 50
    rule3_active = ma10_63d > 50
    rule2_active = (raw_21d < 5) & (raw_63d < 5)
    global_mult = pd.Series(1.0, index=frag.index)
    global_mult[rule2_active] = 1.25
    global_mult[rule1_active | rule3_active] = 0.0

    # Per-ticker price + indicator + multiplier
    closes = {}
    for tk in EXPOSURE_TICKERS:
        s = _resolve_close_series(prices_dict, tk)
        if s is None or s.empty:
            errors.append(f"Price data missing for {tk}")
            return empty
        closes[tk] = s

    close_df = pd.DataFrame(closes).sort_index()
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    cal = close_df.index[(close_df.index >= start) & (close_df.index <= end)]
    if len(cal) < 2:
        errors.append("Not enough overlapping price history in window")
        return empty

    # Compute indicators on full history (needs lookback before start), then
    # reindex to the backtest calendar.
    ticker_mult_full = pd.DataFrame(index=close_df.index, columns=EXPOSURE_TICKERS, dtype=float)
    ticker_mult_full[:] = 1.0
    for tk in EXPOSURE_TICKERS:
        cs = close_df[tk]
        rk2 = _compute_return_rank(cs, 2)
        rk5 = _compute_return_rank(cs, 5)
        rk21 = _compute_return_rank(cs, 21)
        sma_d = _compute_sma_distance(cs, PT_SMA_PERIOD)
        # Rule 4: all three ranks > threshold
        r4 = (rk2 > PT_OVERBOUGHT_THRESHOLD) & (rk5 > PT_OVERBOUGHT_THRESHOLD) & (rk21 > PT_OVERBOUGHT_THRESHOLD)
        # Rule 5: 200 SMA distance in band
        lo, hi = PT_SMA_DIST_BAND
        r5 = (sma_d >= lo) & (sma_d <= hi)
        zero_mask = r4.fillna(False) | r5.fillna(False)
        ticker_mult_full.loc[zero_mask, tk] = 0.0

    close_df_window = close_df.reindex(cal).ffill()
    ret_df = close_df_window.pct_change().fillna(0.0)
    ticker_mult_w = ticker_mult_full.reindex(cal).ffill().fillna(1.0)
    global_mult_w = global_mult.reindex(cal).ffill().fillna(1.0)

    target_w = pd.DataFrame(index=cal, columns=EXPOSURE_TICKERS, dtype=float)
    for tk in EXPOSURE_TICKERS:
        target_w[tk] = BASE_WEIGHTS[tk] * global_mult_w * ticker_mult_w[tk]

    target_w_lagged = target_w.shift(1).fillna(0.0)
    daily_pnl_pct = (target_w_lagged * ret_df).sum(axis=1)
    daily_pnl_dollars = daily_pnl_pct * float(starting_equity)

    return {
        'daily_pnl': daily_pnl_dollars,
        'global_mult': global_mult_w,
        'ticker_mult': ticker_mult_w,
        'target_weights': target_w,
        'close_df': close_df_window,
        'errors': errors,
    }


# ---------------------------------------------------------------------------
# EMAIL RENDERING
# ---------------------------------------------------------------------------
def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def _fmt_dollars(x: float) -> str:
    if abs(x) >= 1000:
        return f"${x:,.0f}"
    return f"${x:.0f}"


def build_exposure_email_html(today: Dict, prior: Optional[Dict]) -> str:
    """Render the HTML block for the top of the daily-scan email."""
    if today is None:
        return (
            '<div style="margin: 0 0 18px 0; padding: 14px 16px; background: #fff8e1; '
            'border-left: 4px solid #f9a825; border-radius: 4px; font-family: Arial, sans-serif;">'
            '<div style="font-weight: bold; color: #f57f17; font-size: 14px;">Exposure Leg</div>'
            '<div style="font-size: 12px; color: #666; margin-top: 4px;">'
            'Fragility cache missing — exposure leg not computed.</div></div>'
        )

    mult = today['mult']
    rule_label = today['active_rule'] or 'Default (1.00x)'
    reason = today['reason']
    r = today['readings']
    targets = today['targets']
    av = today['account_value']

    # Determine effective gross (accounts for both global mult AND per-ticker zeros)
    total_target_weight = sum(t['weight'] for t in targets.values())
    has_per_ticker_zero = any(t.get('ticker_mult', 1.0) == 0.0 for t in targets.values())

    if total_target_weight == 0:
        banner_bg = '#ffebee'; banner_border = '#c62828'; banner_color = '#b71c1c'
        if mult == 0.0:
            banner_text = f'EXPOSURE OFF — {rule_label}'
        else:
            banner_text = 'EXPOSURE OFF — all tickers gated by per-ticker rules'
    elif mult > 1.0:
        banner_bg = '#e8f5e9'; banner_border = '#2e7d32'; banner_color = '#1b5e20'
        banner_text = f'EXPOSURE BOOST {mult:.2f}x — {rule_label}'
    elif has_per_ticker_zero:
        banner_bg = '#fff3e0'; banner_border = '#ef6c00'; banner_color = '#e65100'
        banner_text = f'EXPOSURE PARTIAL {mult:.2f}x — some tickers gated by per-ticker rules'
    else:
        banner_bg = '#e3f2fd'; banner_border = '#1565c0'; banner_color = '#0d47a1'
        banner_text = f'EXPOSURE BASE {mult:.2f}x — {rule_label}'

    # Global readings strip
    readings_html = (
        f'<div style="font-size: 12px; color: #555; margin-top: 6px;">'
        f'<strong>Global Dials</strong> &nbsp; '
        f'Raw 21D: <strong>{r["raw_21d"]:.1f}</strong> &nbsp;|&nbsp; '
        f'Raw 63D: <strong>{r["raw_63d"]:.1f}</strong> &nbsp;|&nbsp; '
        f'10d-MA 63D: <strong>{r["ma10_63d"]:.1f}</strong> &nbsp;'
        f'<span style="color: #888;">(asof {today["asof"]})</span>'
        f'</div>'
        f'<div style="font-size: 12px; color: #666; margin-top: 4px;"><em>Global:</em> {reason}</div>'
    )

    # Targets table — now with per-ticker rule status column
    rows = []
    rows.append(
        '<tr style="background: #f5f5f5;">'
        '<th style="text-align: left; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">Ticker</th>'
        '<th style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">Weight</th>'
        '<th style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">$ Amount</th>'
        '<th style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">Shares</th>'
        '<th style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">Last Close</th>'
        '<th style="text-align: left; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">Per-ticker rule</th>'
        '</tr>'
    )
    for tkr in EXPOSURE_TICKERS:
        t = targets.get(tkr, {})
        shares_str = f"{t['shares']:,}" if t.get('shares') is not None else '—'
        price_str = f"${t['price']:.2f}" if t.get('price') is not None else '—'
        if t.get('ticker_mult', 1.0) == 0.0:
            pt_rule_str = (f"<span style='color:#c62828; font-weight:bold;'>"
                           f"{t.get('ticker_rule', 'Per-ticker')}: 0x</span>")
            row_bg = ' style="background: #ffebee;"'
        else:
            pt_rule_str = '<span style="color:#666;">none</span>'
            row_bg = ''
        rows.append(
            f'<tr{row_bg}>'
            f'<td style="padding: 6px 10px; border-bottom: 1px solid #eee; font-weight: bold;">{tkr}</td>'
            f'<td style="padding: 6px 10px; border-bottom: 1px solid #eee; text-align: right;">{_fmt_pct(t.get("weight", 0))}</td>'
            f'<td style="padding: 6px 10px; border-bottom: 1px solid #eee; text-align: right;">{_fmt_dollars(t.get("dollars", 0))}</td>'
            f'<td style="padding: 6px 10px; border-bottom: 1px solid #eee; text-align: right;">{shares_str}</td>'
            f'<td style="padding: 6px 10px; border-bottom: 1px solid #eee; text-align: right; color: #666;">{price_str}</td>'
            f'<td style="padding: 6px 10px; border-bottom: 1px solid #eee; font-size: 11px;">{pt_rule_str}</td>'
            f'</tr>'
        )
    targets_table = (
        '<table style="width: 100%; border-collapse: collapse; margin-top: 12px; font-family: Arial, sans-serif;">'
        + ''.join(rows) +
        '</table>'
    )

    # Per-ticker readings detail (collapsible visual block — shown only for zeroed tickers)
    pt_detail_rows = []
    for tkr in EXPOSURE_TICKERS:
        t = targets.get(tkr, {})
        if t.get('ticker_mult', 1.0) != 0.0:
            continue
        rd = t.get('ticker_readings', {})
        rk2 = rd.get('rank_2d')
        rk5 = rd.get('rank_5d')
        rk21 = rd.get('rank_21d')
        smad = rd.get('sma200_dist')
        pieces = []
        if rk2 is not None: pieces.append(f"2D rank {rk2:.0f}")
        if rk5 is not None: pieces.append(f"5D rank {rk5:.0f}")
        if rk21 is not None: pieces.append(f"21D rank {rk21:.0f}")
        if smad is not None: pieces.append(f"200 SMA dist {smad:+.1f}%")
        pt_detail_rows.append(
            f'<div style="font-size: 11px; color: #555; margin-top: 2px;">'
            f'<strong>{tkr}</strong>: {t.get("ticker_reason", "")} '
            f'<span style="color:#888;">({" | ".join(pieces)})</span></div>'
        )
    pt_detail_html = ''
    if pt_detail_rows:
        pt_detail_html = (
            '<div style="margin-top: 10px; padding: 8px 10px; background: #fafafa; '
            'border-left: 3px solid #ef6c00; border-radius: 3px;">'
            '<div style="font-size: 11px; color: #ef6c00; font-weight: bold; margin-bottom: 4px;">'
            'Per-ticker rule details</div>'
            + ''.join(pt_detail_rows) +
            '</div>'
        )

    # Orders table — diff vs prior session, only when something flipped
    orders_html = ''
    if prior is not None:
        prior_targets = prior.get('targets', {})
        order_rows = []
        for tkr in EXPOSURE_TICKERS:
            cur = targets.get(tkr, {})
            old = prior_targets.get(tkr, {'weight': 0, 'dollars': 0, 'shares': 0})
            d_w = cur.get('weight', 0) - old.get('weight', 0)
            if abs(d_w) < 1e-9:
                continue
            d_d = cur.get('dollars', 0) - old.get('dollars', 0)
            d_sh = (cur.get('shares') or 0) - (old.get('shares') or 0)
            action = 'BUY' if d_d > 0 else 'SELL'
            action_color = '#2e7d32' if action == 'BUY' else '#c62828'
            order_rows.append(
                '<tr>'
                f'<td style="padding: 6px 10px; border-bottom: 1px solid #eee; font-weight: bold;">{tkr}</td>'
                f'<td style="padding: 6px 10px; border-bottom: 1px solid #eee; color: {action_color}; font-weight: bold;">{action}</td>'
                f'<td style="padding: 6px 10px; border-bottom: 1px solid #eee; text-align: right;">{d_w * 100:+.2f}%</td>'
                f'<td style="padding: 6px 10px; border-bottom: 1px solid #eee; text-align: right;">{d_d:+,.0f}</td>'
                f'<td style="padding: 6px 10px; border-bottom: 1px solid #eee; text-align: right;">{d_sh:+,}</td>'
                '</tr>'
            )
        if order_rows:
            header = (
                '<tr style="background: #fafafa;">'
                '<th style="text-align: left; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">Ticker</th>'
                '<th style="text-align: left; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">Action</th>'
                '<th style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">&Delta; Weight</th>'
                '<th style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">&Delta; $</th>'
                '<th style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">&Delta; Shares</th>'
                '</tr>'
            )
            orders_html = (
                '<div style="margin-top: 14px; font-size: 13px; font-weight: bold; color: #333;">'
                'Orders today</div>'
                '<table style="width: 100%; border-collapse: collapse; margin-top: 6px; font-family: Arial, sans-serif;">'
                + header + ''.join(order_rows) +
                '</table>'
            )
    elif prior is None:
        orders_html = (
            '<div style="margin-top: 14px; font-size: 12px; color: #888;">'
            'No prior state — first run, treat full target as opening trade.</div>'
        )

    block = (
        f'<div style="margin: 0 0 22px 0; padding: 14px 18px; background: {banner_bg}; '
        f'border-left: 5px solid {banner_border}; border-radius: 4px; font-family: Arial, sans-serif;">'
        f'<div style="font-weight: bold; color: {banner_color}; font-size: 14px;">Exposure Leg &nbsp;'
        f'<span style="font-weight: normal; color: #444; font-size: 12px;">'
        f'(account ${av:,.0f}, base 25% allocation)</span></div>'
        f'<div style="margin-top: 4px; font-size: 13px; color: {banner_color}; font-weight: bold;">{banner_text}</div>'
        f'{readings_html}'
        f'{targets_table}'
        f'{pt_detail_html}'
        f'{orders_html}'
        f'</div>'
    )
    return block
