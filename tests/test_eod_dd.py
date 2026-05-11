"""Smoke test for OVS EOD-DD exit in strat_backtester.process_signals_fast.

Constructs a minimal synthetic OVS scenario and asserts the trade exits
with type 'EOD-DD' when entry-day close is more than 0.25 ATR offside,
and exits normally otherwise.
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'pages'))

# Stub streamlit before importing strat_backtester (avoids ScriptRunContext warning)
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


def _ovs_strategy(eod_dd_atr=0.25):
    return {
        'name': 'Overbot Vol Spike',
        'settings': {
            'trade_direction': 'Short',
            'entry_type': 'Limit (Open +/- 0.75 ATR)',
            'max_one_pos': False,
        },
        'execution': {
            'risk_bps': 40, 'slippage_bps': 2,
            'stop_atr': 1.0, 'tgt_atr': 2.0,
            'hold_days': 2,
            'use_stop_loss': False, 'use_take_profit': True,
            'path1_bps': 40, 'path2_bps': 8, 'path2_daily_cap_pct': 0.75,
            'eod_dd_atr': eod_dd_atr,
        },
        'universe_tickers': ['TEST'],
    }


def _build_inputs(entry_close):
    """OVS short entry on day 1 at limit 102.5; entry_close drives EOD-DD."""
    dates = pd.date_range('2024-01-02', periods=5, freq='B')
    df = pd.DataFrame({
        'Open':  [100.0, 101.0, 100.0, 100.0, 100.0],
        'High':  [101.0, max(103.5, entry_close), 102.0, 102.0, 102.0],
        'Low':   [99.0,  100.5, 100.0, 100.0, 100.0],
        'Close': [100.0, entry_close, 100.0, 101.0, 100.5],
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


def run_case(label, entry_close, expected_type):
    candidates, signal_data, processed = _build_inputs(entry_close)
    strategies = [_ovs_strategy()]
    sig_df = process_signals_fast(
        candidates, signal_data, processed, strategies,
        starting_equity=100_000,
    )
    if sig_df.empty:
        raise AssertionError(f"[{label}] empty result frame")
    row = sig_df.iloc[0]
    actual_type = row['Exit Type']
    print(f"[{label}] entry_close={entry_close} -> Exit Type={actual_type!r}, "
          f"Entry={row['Price']}, Exit={row['Exit Price']}, PnL={row['PnL']}")
    assert actual_type == expected_type, (
        f"[{label}] expected Exit Type={expected_type!r}, got {actual_type!r}"
    )
    if expected_type == 'EOD-DD':
        # Exit price should equal the entry-day close
        assert abs(row['Exit Price'] - entry_close) < 1e-6
        # Entry should be the limit fill (102.5) for our setup
        assert abs(row['Price'] - 102.5) < 1e-6
        # Short trade losing: PnL should be negative
        assert row['PnL'] < 0


def main():
    # Case A: EOD-DD fires (close = 103.5, dd = 0.5 ATR > 0.25)
    run_case('A: triggers', 103.5, 'EOD-DD')
    # Case B: EOD-DD does NOT fire (close = 102.6, dd = 0.05 ATR < 0.25)
    run_case('B: does not trigger', 102.6, 'Time')
    print('\nAll EOD-DD cases passed.')


if __name__ == '__main__':
    main()
