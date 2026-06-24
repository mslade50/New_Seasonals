"""Regression test for the persistent-limit fill-window cap in
strat_backtester.process_signals_fast (execution['fill_window_days']).

OLV (2026-06-24) cancels its GTC close-0.25 ATR limit after 3 trading days
instead of the full 10-day hold. A fill that lands inside the window must be
kept; one that lands after it must drop the signal entirely. The hold reduction
on a kept fill still references execution['hold_days'], so a kept trade is
unchanged by the window.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'pages'))


class _NoOp:
    def __getattr__(self, name):
        def f(*a, **k):
            return self
        return f
    def __call__(self, *a, **k): return self
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cache_data(self, *a, **k):
        def deco(fn): return fn
        return deco
    cache_resource = cache_data
sys.modules['streamlit'] = _NoOp()

import pandas as pd
import numpy as np

from strat_backtester import process_signals_fast


def _olv_strategy(fill_window_days=None):
    execution = {
        'risk_bps': 35, 'slippage_bps': 2,
        'stop_atr': 1.25, 'tgt_atr': 2.5,
        'hold_days': 10,
        'use_stop_loss': True, 'use_take_profit': True,
    }
    if fill_window_days is not None:
        execution['fill_window_days'] = fill_window_days
    return {
        'name': 'TEST OLV',
        'settings': {
            'trade_direction': 'Long',
            'entry_type': 'Limit Order -0.25 ATR (Persistent)',
            'max_one_pos': False,
        },
        'execution': execution,
        'universe_tickers': ['TEST'],
    }


def _build_inputs(touch_day, start_date='2024-01-02'):
    """Long persistent limit = close - 0.25*ATR = 100 - 0.5 = 99.5.

    Only the bar at offset `touch_day` (trading days after signal) dips to/below
    99.5; every earlier bar's Low stays at 99.6 (untouched). 12 bars so any
    in-window fill has room for the reduced hold to play out.
    """
    n = 12
    dates = pd.date_range(start_date, periods=n, freq='B')
    lows = [99.0] + [99.6] * (n - 1)        # idx0 (signal) irrelevant
    for d in range(1, n):
        lows[d] = 99.0 if d >= touch_day else 99.6
    df = pd.DataFrame({
        'Open':  [100.0] * n,
        'High':  [101.0] * n,
        'Low':   lows,
        'Close': [100.0] * n,
    }, index=dates)
    df['ATR'] = 2.0
    df['RangePct'] = 0.02
    df['vol_ratio'] = 1.0
    df['Sznl'] = 50.0
    df['atr_sznl_5d'] = 50.0
    df['rank_ret_126d'] = 50.0
    df['rank_ret_252d'] = 50.0

    candidates = [(int(dates[0].value), 'TEST', 'TEST', 0, 0)]
    signal_data = {
        ('TEST', 0): {
            'atr': 2.0, 'close': 100.0, 'open': 100.0,
            'high': 101.0, 'low': 99.0,
            'vol_ratio': 1.0, 'sznl': 50, 'range_pct': 2.0,
            'atr_sznl_5d': 50.0, 'rank_ret_126d': 50.0, 'rank_ret_252d': 50.0,
        }
    }
    return candidates, signal_data, {'TEST': df}


def run_case(label, touch_day, fill_window_days, expect_fill):
    candidates, signal_data, processed = _build_inputs(touch_day)
    strategies = [_olv_strategy(fill_window_days=fill_window_days)]
    sig_df = process_signals_fast(
        candidates, signal_data, processed, strategies, starting_equity=100_000,
    )
    filled = not sig_df.empty
    detail = ''
    if filled:
        row = sig_df.iloc[0]
        detail = f" Entry={row['Price']:.2f} on {pd.Timestamp(row['Entry Date']).date()}"
    print(f"[{label}] touch=T+{touch_day} window={fill_window_days} "
          f"-> filled={filled}{detail}")
    assert filled == expect_fill, (
        f"[{label}] expected filled={expect_fill}, got {filled}"
    )
    if filled:
        # limit fill price is the anchored limit (99.5), not the open
        assert abs(sig_df.iloc[0]['Price'] - 99.5) < 1e-6


def main():
    # A: touch on T+4, window=3 -> order cancelled before the dip, NO trade.
    run_case('A: late touch, window=3 drops', touch_day=4,
             fill_window_days=3, expect_fill=False)
    # B: same bars, window=10 -> fills on T+4 (proves the bars CAN fill).
    run_case('B: late touch, window=10 fills', touch_day=4,
             fill_window_days=10, expect_fill=True)
    # C: touch on T+3, window=3 -> boundary inclusive, fills.
    run_case('C: boundary touch T+3, window=3 fills', touch_day=3,
             fill_window_days=3, expect_fill=True)
    # D: no fill_window_days -> defaults to hold_days(10), late touch fills.
    run_case('D: default window = hold_days', touch_day=4,
             fill_window_days=None, expect_fill=True)
    print('\nAll OLV fill-window cases passed.')


if __name__ == '__main__':
    main()
