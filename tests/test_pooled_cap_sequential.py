"""Pooled direction cap must apply SEQUENTIALLY after the per-strategy cap.

Before 2026-07-16 the engine computed the pooled long/short scale from
placed_risk_by_dir_date accumulated PRE per-strategy trim, so a
single-strategy cluster day that the per-strategy cap already trimmed to
exactly the pooled cap got trimmed AGAIN (~30% understatement vs live,
which applies the caps in sequence on the post-trim totals).

Scenario: one Long strategy stages 2 x 200 bps = 400 bps on one day.
Per-strategy cap 250 bps trims to 250. Pooled long cap 250 bps then sees a
post-trim total exactly AT the cap and must NOT trim further. The buggy
engine divided by the stale 400 bps and booked 156 bps.
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

from strat_backtester import process_signals_fast

EQUITY = 100_000.0


def _strategy():
    return {
        'name': 'CapTest',
        'settings': {
            'trade_direction': 'Long',
            'entry_type': 'T+1 Open',
            'max_one_pos': False,
        },
        'execution': {
            'risk_bps': 200, 'slippage_bps': 0,
            'stop_atr': 1.0, 'tgt_atr': 0.0,
            'hold_days': 2,
            'use_stop_loss': False, 'use_take_profit': False,
        },
        'universe_tickers': ['TESTA', 'TESTB'],
    }


def _ticker_df():
    dates = pd.date_range('2024-01-02', periods=5, freq='B')
    df = pd.DataFrame({
        'Open':  [100.0] * 5,
        'High':  [101.0] * 5,
        'Low':   [99.0] * 5,
        'Close': [100.0] * 5,
    }, index=dates)
    df['ATR'] = 2.0
    df['RangePct'] = 0.02
    df['vol_ratio'] = 1.0
    df['Sznl'] = 50.0
    return df


def test_pooled_cap_uses_post_strategy_trim_totals():
    dates = pd.date_range('2024-01-02', periods=5, freq='B')
    processed = {'TESTA': _ticker_df(), 'TESTB': _ticker_df()}
    candidates = [
        (int(dates[0].value), 'TESTA', 'TESTA', 0, 0),
        (int(dates[0].value), 'TESTB', 'TESTB', 0, 0),
    ]
    sd = {'atr': 2.0, 'close': 100.0, 'open': 100.0, 'high': 101.0,
          'low': 99.0, 'vol_ratio': 1.0, 'sznl': 50, 'range_pct': 2.0}
    signal_data = {('TESTA', 0): dict(sd), ('TESTB', 0): dict(sd)}

    sig_df = process_signals_fast(
        candidates, signal_data, processed, [_strategy()],
        starting_equity=EQUITY, flat_sizing=True,
        cap_bps=250, max_long_risk_bps=250,
    )
    assert len(sig_df) == 2
    total_risk = float(sig_df['Risk $'].sum())
    cap_dollars = EQUITY * 250 / 10000.0   # $2,500
    # per-strategy cap trims 400->250 bps; pooled cap must see the trimmed
    # total (at cap) and leave it alone. The old double-trim booked ~$1,562.
    assert abs(total_risk - cap_dollars) < 5.0, (
        f"expected ~${cap_dollars:.0f} after sequential caps, got ${total_risk:.0f} "
        f"(double-trim bug books ~${cap_dollars * 250 / 400:.0f})"
    )
