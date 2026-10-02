"""Pure PA replay payload helpers. No broker, network or production writes.

The engine supplies fresh Main-anchor, unsplit, whole-share orders and their
PA single-target exits. The browser applies the stager's PA quantity formula.
"""
from __future__ import annotations

import math
import pandas as pd


def execution_candidates(candidates, signal_data, processed, aliases):
    """Keep index signals, but execute on the ETF's own OHLC/ATR, like scan."""
    result, rows, seen = [], dict(signal_data), set()
    for ts, ticker, clean, strat, idx in candidates:
        target = aliases.get(clean, clean)
        if target != clean:
            frame = processed.get(target)
            date = pd.Timestamp(ts)
            if frame is None or date not in frame.index:
                raise ValueError(f"PA execution history missing: {clean}->{target} {date}")
            idx = frame.index.get_loc(date)
            if not isinstance(idx, int):
                # numpy integer is acceptable; duplicate dates are not.
                try:
                    idx = int(idx)
                except (TypeError, ValueError):
                    raise ValueError(f"Duplicate PA execution date: {target} {date}")
            r = frame.iloc[idx]
            rows[(target, idx)] = {
                'atr': r['ATR'], 'close': r['Close'], 'open': r['Open'],
                'high': r['High'], 'low': r['Low'],
                'vol_ratio': r.get('vol_ratio', 0), 'sznl': r.get('Sznl', 50),
                'range_pct': r['RangePct'] * 100,
                'atr_sznl_5d': r.get('atr_sznl_5d', 50),
                'rank_ret_126d': r.get('rank_ret_126d', 50),
                'rank_ret_252d': r.get('rank_ret_252d', 50),
            }
            ticker = target
        key = (ts, target, strat)
        if key not in seen:
            result.append((ts, ticker, target, strat, idx))
            seen.add(key)
    return result, rows


def build_pa_payload(trades, prices, session_dates, anchor, settings):
    """Per-share daily MTM, including exact engine exits and zero sessions.

The signal is staged on the next session, sized from the prior session's PA
equity. This is an EOD approximation to live staging's premarket NLV, rather
than a reconstruction of unrecorded 09:31 account marks/deposits.
"""
    anchor = float(anchor)
    if not math.isfinite(anchor) or anchor <= 0:
        raise ValueError('Invalid Main sizing anchor')
    multiplier = float(settings['risk_multiplier'])
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('Invalid PA risk multiplier')
    dates = pd.DatetimeIndex(session_dates).normalize()
    if dates.empty or not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError('PA replay requires ordered unique exchange sessions')
    lookup = {d: i for i, d in enumerate(dates)}
    records = []
    for _, row in trades.iterrows():
        signal = pd.Timestamp(row['Signal Date']).normalize()
        entry = pd.Timestamp(row['Entry Date']).normalize()
        exit_ = pd.Timestamp(row['Exit Date']).normalize()
        qty = float(row['Shares_flat'])
        if not math.isfinite(qty) or qty < 1 or qty != math.floor(qty):
            raise ValueError('PA basis must contain executable positive integer Main quantities')
        if entry not in lookup or exit_ not in lookup or exit_ < entry:
            raise ValueError(f"PA trade outside exchange calendar: {row['Ticker']} {entry} {exit_}")
        # Signal-close trades size at that session; other orders are staged
        # once on the first exchange session after the signal, even if GTC.
        stage = lookup[entry] if signal == entry else int(dates.searchsorted(signal, side='right'))
        if stage > lookup[entry] or stage >= len(dates):
            raise ValueError('PA staging date follows entry')
        ticker = str(row['Ticker'])
        frame = prices.get(ticker)
        if frame is None:
            raise ValueError(f'Missing PA price history: {ticker}')
        close = frame['Close'].copy()
        close.index = pd.DatetimeIndex(close.index).normalize()
        marks = close.reindex(dates[lookup[entry]:lookup[exit_] + 1])
        if marks.isna().any() or not all(math.isfinite(float(v)) for v in marks):
            raise ValueError(f'Missing PA daily marks: {ticker} {entry} {exit_}')
        sign = -1 if 'SHORT' in str(row['Action']).upper() else 1
        entry_price, exit_price = float(row['Entry Price']), float(row['Exit Price'])
        if not all(math.isfinite(v) and v > 0 for v in (entry_price, exit_price)):
            raise ValueError('Invalid PA execution price')
        values = [float(v) for v in marks]
        values[-1] = exit_price  # realized stop/gap/slippage fill, or latest open mark
        previous = entry_price
        pnl = []
        for value in values:
            pnl.append(sign * (value - previous))
            previous = value
        risk = float(row['Risk_flat_750k']) / qty
        if not math.isfinite(risk) or risk < 0:
            raise ValueError('Invalid PA per-share risk')
        exit_type = str(row.get('Exit Type', ''))
        time_stop = row.get('Time Stop')
        if exit_type == 'Time' and pd.notna(time_stop) and pd.Timestamp(time_stop).normalize() > exit_:
            exit_type = 'As-of mark'
        records.append({
            'ticker': ticker, 'strategy': str(row['Strategy']),
            'direction': 'Short' if sign < 0 else 'Long',
            'signal': signal.strftime('%Y-%m-%d'), 'stage': stage,
            'entry': lookup[entry], 'exit': lookup[exit_],
            'entry_price': entry_price, 'exit_price': exit_price,
            'primary_qty': int(qty), 'risk_per_share': risk,
            'exit_type': exit_type,
            'unit_pnl': pnl,
        })
    return {
        'version': 1, 'asof': dates[-1].strftime('%Y-%m-%d'),
        'dates': [d.strftime('%Y-%m-%d') for d in dates],
        'config': {'primary_anchor': anchor, 'risk_multiplier': multiplier,
                   'per_strategy_daily_cap_bps': settings['per_strategy_daily_cap_bps'],
                   'sizing_clock': 'prior-session EOD equity at staging; quantity held fixed',
                   'execution': 'Main unsplit orders; PA single far target'},
        'trades': records,
        'limitations': [
            'Current rules replayed historically; this is not a live PA account statement or an out-of-sample track record.',
            'No external cash flows or unrecorded premarket account marks. Equity is reinvested only when new orders are staged.',
            'The replay covers the staged swing book. An unfinished trade at the last available bar contributes an as-of mark; the final calendar period is partial. Historical universe selection and current-rule tuning can bias results.',
            'Engine stop/gap/exit slippage is included in execution prices. Borrow, commissions, market impact, failed fills and intraday drawdowns are not fully modeled.',
        ],
    }
