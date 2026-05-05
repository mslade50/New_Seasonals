"""
Exposure Leg — fragility-gated buy-and-hold overlay rendered at the top of
the AM daily-scan email.

Base allocation: 25% of ACCOUNT_VALUE split 40% VOO / 40% VGK / 20% VTI
(so 10% / 10% / 5% of total account at the 1.0x multiplier).

Rule precedence (zero wins over boost):
    1. ALL x 0.00 when Raw 21D fragility > 50
    2. ALL x 0.00 when 10d-MA 63D fragility > 50
    3. ALL x 1.25 when Raw 21D < 5 AND Raw 63D < 5
    else 1.00

"Raw" = the values stored directly in data/rd2_fragility.parquet (which
daily_risk_report writes after a 5d rolling smooth — same convention used
across exposure_backtester.py and daily_scan's sizing block).

State across runs is persisted in data/exposure_state.json so the email can
diff today's targets vs yesterday's and emit "necessary orders today" only
when the multiplier flips.
"""
from __future__ import annotations

import json
import os
import datetime
from typing import Dict, Optional

import pandas as pd

BASE_WEIGHTS: Dict[str, float] = {
    'VOO': 0.10,  # 40% of 25%
    'VGK': 0.10,  # 40% of 25%
    'VTI': 0.05,  # 20% of 25%
}
EXPOSURE_TICKERS = list(BASE_WEIGHTS.keys())

# File paths are resolved relative to the daily_scan working directory.
FRAGILITY_PATH = os.path.join('data', 'rd2_fragility.parquet')
STATE_PATH = os.path.join('data', 'exposure_state.json')


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


def _apply_rules(readings: Dict[str, float]) -> Dict:
    """Apply rule precedence. Returns dict with mult, active_rule, reason."""
    raw_21d = readings['raw_21d']
    raw_63d = readings['raw_63d']
    ma10_63d = readings['ma10_63d']

    # Zero rules check first (precedence: zero wins).
    if raw_21d > 50:
        return {
            'mult': 0.0,
            'active_rule': 'Rule 1',
            'reason': f'Raw 21D fragility {raw_21d:.1f} > 50',
        }
    if ma10_63d > 50:
        return {
            'mult': 0.0,
            'active_rule': 'Rule 3',
            'reason': f'10d-MA 63D fragility {ma10_63d:.1f} > 50',
        }
    if raw_21d < 5 and raw_63d < 5:
        return {
            'mult': 1.25,
            'active_rule': 'Rule 2',
            'reason': f'Raw 21D {raw_21d:.1f} < 5 AND Raw 63D {raw_63d:.1f} < 5',
        }
    return {
        'mult': 1.0,
        'active_rule': None,
        'reason': 'No rule active — base allocation',
    }


def _resolve_price(ticker: str, master_dict: Optional[dict]) -> Optional[float]:
    """Pull last close from master_dict if available, else yfinance, else None."""
    if master_dict is not None:
        df = master_dict.get(ticker)
        if df is not None and not df.empty:
            try:
                cols = {c.lower(): c for c in df.columns}
                close_col = cols.get('close') or cols.get('Close')
                if close_col is not None:
                    return float(df[close_col].dropna().iloc[-1])
            except Exception:
                pass
    try:
        import yfinance as yf
        hist = yf.Ticker(ticker).history(period='5d', auto_adjust=False)
        if hist is not None and not hist.empty:
            return float(hist['Close'].dropna().iloc[-1])
    except Exception:
        pass
    return None


def compute_exposure_targets(
    account_value: float,
    master_dict: Optional[dict] = None,
    frag_path: str = FRAGILITY_PATH,
) -> Optional[Dict]:
    """Build today's exposure target snapshot.

    Returns None if fragility data is unavailable. Otherwise returns:
      {
        'asof', 'mult', 'active_rule', 'reason',
        'readings': {raw_21d, raw_63d, ma10_63d},
        'targets': {ticker: {weight, dollars, shares, price}},
        'account_value': float,
      }
    """
    readings = _read_dial_readings(frag_path)
    if readings is None:
        return None

    rule = _apply_rules(readings)
    mult = rule['mult']

    targets: Dict[str, Dict] = {}
    for tkr, base_w in BASE_WEIGHTS.items():
        target_w = base_w * mult
        dollars = account_value * target_w
        price = _resolve_price(tkr, master_dict)
        shares = int(round(dollars / price)) if (price and price > 0) else None
        targets[tkr] = {
            'weight': target_w,
            'dollars': dollars,
            'price': price,
            'shares': shares,
        }

    return {
        'asof': readings['asof'],
        'computed_at': datetime.datetime.utcnow().isoformat() + 'Z',
        'mult': mult,
        'active_rule': rule['active_rule'],
        'reason': rule['reason'],
        'readings': {
            'raw_21d': readings['raw_21d'],
            'raw_63d': readings['raw_63d'],
            'ma10_63d': readings['ma10_63d'],
        },
        'targets': {k: {kk: vv for kk, vv in v.items()} for k, v in targets.items()},
        'account_value': account_value,
    }


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


def _resolve_close_series(prices_dict: dict, ticker: str):
    """Pull a close-price Series from prices_dict (master_dict-style) or yfinance.

    Returns a Series indexed by tz-naive normalized DatetimeIndex, or None.
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


def compute_exposure_leg_backtest(
    start_date,
    end_date,
    prices_dict,
    starting_equity: float,
    frag_path: str = FRAGILITY_PATH,
):
    """Daily $ P&L series for the exposure leg over [start_date, end_date].

    Sizes the leg at BASE_WEIGHTS × mult of `starting_equity`. Since
    BASE_WEIGHTS already sums to 0.25 (the 25% production allocation),
    the leg's max gross is 25% of equity at 1.0x and 31.25% at 1.25x.

    Uses prior-day target weights (lagged 1 trading day) so today's PnL
    reflects yesterday's close-of-business decision — same convention as
    the strat backtester's daily MTM.

    Returns:
      {
        'daily_pnl': Series of $ P&L per day,
        'mult': Series of multipliers per day,
        'target_weights': DataFrame of per-ticker fractional weights per day,
        'close_df': DataFrame of close prices for VOO/VGK/VTI,
        'errors': list[str],
      }
    Empty 'daily_pnl' Series if data is missing.
    """
    errors = []
    empty = {
        'daily_pnl': pd.Series(dtype=float),
        'mult': pd.Series(dtype=float),
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

    raw_21d = frag['21d']
    raw_63d = frag['63d']
    ma10_63d = raw_63d.rolling(10, min_periods=1).mean()

    rule1_active = raw_21d > 50
    rule3_active = ma10_63d > 50
    rule2_active = (raw_21d < 5) & (raw_63d < 5)

    mult = pd.Series(1.0, index=frag.index)
    mult[rule2_active] = 1.25
    # Zero wins over boost (rule precedence).
    mult[rule1_active | rule3_active] = 0.0

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
    close_df = close_df.reindex(cal).ffill()
    ret_df = close_df.pct_change().fillna(0.0)

    mult_aligned = mult.reindex(cal).ffill().fillna(1.0)

    target_w = pd.DataFrame(index=cal, columns=EXPOSURE_TICKERS, dtype=float)
    for tk in EXPOSURE_TICKERS:
        target_w[tk] = BASE_WEIGHTS[tk] * mult_aligned

    target_w_lagged = target_w.shift(1).fillna(0.0)
    daily_pnl_pct = (target_w_lagged * ret_df).sum(axis=1)
    daily_pnl_dollars = daily_pnl_pct * float(starting_equity)

    return {
        'daily_pnl': daily_pnl_dollars,
        'mult': mult_aligned,
        'target_weights': target_w,
        'close_df': close_df,
        'errors': errors,
    }


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

    if mult == 0.0:
        banner_bg = '#ffebee'
        banner_border = '#c62828'
        banner_color = '#b71c1c'
        banner_text = f'EXPOSURE OFF — {rule_label}'
    elif mult > 1.0:
        banner_bg = '#e8f5e9'
        banner_border = '#2e7d32'
        banner_color = '#1b5e20'
        banner_text = f'EXPOSURE BOOST {mult:.2f}x — {rule_label}'
    else:
        banner_bg = '#e3f2fd'
        banner_border = '#1565c0'
        banner_color = '#0d47a1'
        banner_text = f'EXPOSURE BASE 1.00x — {rule_label}'

    # Dial readings strip
    readings_html = (
        f'<div style="font-size: 12px; color: #555; margin-top: 6px;">'
        f'<strong>Dials</strong> &nbsp; '
        f'Raw 21D: <strong>{r["raw_21d"]:.1f}</strong> &nbsp;|&nbsp; '
        f'Raw 63D: <strong>{r["raw_63d"]:.1f}</strong> &nbsp;|&nbsp; '
        f'10d-MA 63D: <strong>{r["ma10_63d"]:.1f}</strong> &nbsp;'
        f'<span style="color: #888;">(asof {today["asof"]})</span>'
        f'</div>'
        f'<div style="font-size: 12px; color: #666; margin-top: 4px;">{reason}</div>'
    )

    # Targets table
    rows = []
    rows.append(
        '<tr style="background: #f5f5f5;">'
        '<th style="text-align: left; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">Ticker</th>'
        '<th style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">Target Weight</th>'
        '<th style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">$ Amount</th>'
        '<th style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">Shares</th>'
        '<th style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">Last Close</th>'
        '</tr>'
    )
    for tkr in EXPOSURE_TICKERS:
        t = targets[tkr]
        shares_str = f"{t['shares']:,}" if t['shares'] is not None else '—'
        price_str = f"${t['price']:.2f}" if t['price'] is not None else '—'
        rows.append(
            '<tr>'
            f'<td style="padding: 6px 10px; border-bottom: 1px solid #eee; font-weight: bold;">{tkr}</td>'
            f'<td style="padding: 6px 10px; border-bottom: 1px solid #eee; text-align: right;">{_fmt_pct(t["weight"])}</td>'
            f'<td style="padding: 6px 10px; border-bottom: 1px solid #eee; text-align: right;">{_fmt_dollars(t["dollars"])}</td>'
            f'<td style="padding: 6px 10px; border-bottom: 1px solid #eee; text-align: right;">{shares_str}</td>'
            f'<td style="padding: 6px 10px; border-bottom: 1px solid #eee; text-align: right; color: #666;">{price_str}</td>'
            '</tr>'
        )
    targets_table = (
        '<table style="width: 100%; border-collapse: collapse; margin-top: 12px; font-family: Arial, sans-serif;">'
        + ''.join(rows) +
        '</table>'
    )

    # Orders table — only when multiplier flipped vs prior session
    orders_html = ''
    if prior is not None and prior.get('mult') != mult:
        order_rows = []
        order_rows.append(
            '<tr style="background: #fafafa;">'
            '<th style="text-align: left; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">Ticker</th>'
            '<th style="text-align: left; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">Action</th>'
            '<th style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">&Delta; Weight</th>'
            '<th style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">&Delta; $</th>'
            '<th style="text-align: right; padding: 6px 10px; border-bottom: 1px solid #ddd; font-size: 12px;">&Delta; Shares</th>'
            '</tr>'
        )
        prior_targets = prior.get('targets', {})
        for tkr in EXPOSURE_TICKERS:
            cur = targets[tkr]
            old = prior_targets.get(tkr, {'weight': 0, 'dollars': 0, 'shares': 0})
            d_w = cur['weight'] - old.get('weight', 0)
            d_d = cur['dollars'] - old.get('dollars', 0)
            d_sh = (cur['shares'] or 0) - (old.get('shares') or 0)
            if abs(d_w) < 1e-9:
                continue
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
        if len(order_rows) > 1:
            orders_html = (
                '<div style="margin-top: 14px; font-size: 13px; font-weight: bold; color: #333;">'
                f'Orders today &nbsp;<span style="color:#888; font-weight: normal; font-size: 11px;">'
                f'(prior {prior.get("mult", "?")}x &rarr; today {mult}x)</span></div>'
                '<table style="width: 100%; border-collapse: collapse; margin-top: 6px; font-family: Arial, sans-serif;">'
                + ''.join(order_rows) +
                '</table>'
            )
        else:
            orders_html = (
                '<div style="margin-top: 14px; font-size: 12px; color: #888;">'
                'Multiplier changed but target deltas rounded to zero.</div>'
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
        f'{orders_html}'
        f'</div>'
    )
    return block
