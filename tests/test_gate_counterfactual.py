"""Regression tests for the sector-gate counterfactual pass
(scripts/build_trade_ledger.py: gated_strategy_names / strip_sector_gate /
shape_flat_trades). The nogate parquet feeds build_site.build_gate_lab, which
in turn drives the site's gate-history section and the trade-log
"all trades" toggle — the invariants here keep that diff meaningful.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))


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

import numpy as np
import pandas as pd

from build_trade_ledger import (
    gated_strategy_names,
    strip_sector_gate,
    shape_flat_trades,
)


def _book():
    return [
        {'name': 'Oversold Low Volume',
         'execution': {'risk_bps': 35,
                       'sector_loss_gate': {'window_td': 10, 'max_realized_r': -2.0}}},
        {'name': 'Overbot Vol Spike', 'execution': {'risk_bps': 40}},
    ]


def test_gated_names_finds_only_gated_strategies():
    assert gated_strategy_names(_book()) == ['Oversold Low Volume']


def test_prod_book_carries_a_gated_strategy():
    # If the gate leaves the book entirely, the counterfactual pass silently
    # stops producing the parquet — make that visible here instead.
    from strategy_config import STRATEGY_BOOK
    assert gated_strategy_names(STRATEGY_BOOK), \
        "no strategy carries sector_loss_gate — remove the nogate pass or update this test"


def test_strip_removes_gate_without_mutating_source():
    book = _book()
    stripped = strip_sector_gate(book)
    assert gated_strategy_names(stripped) == []
    # untouched original (the same book object runs the baseline passes)
    assert 'sector_loss_gate' in book[0]['execution']
    # other execution config survives
    assert stripped[0]['execution']['risk_bps'] == 35


def test_shape_flat_trades_columns_and_derivations():
    sig = pd.DataFrame({
        'Date': ['2026-06-01', '2026-06-02'],
        'Entry Date': ['2026-06-02', '2026-06-03'],
        'Exit Date': ['2026-06-05', '2026-06-08'],
        'Ticker': ['XOM', 'SPY'],
        'Strategy': ['Oversold Low Volume', 'Overbot Vol Spike'],
        'Action': ['BUY', 'SELL SHORT'],
        'Price': [100.0, 500.0],
        'Exit Price': [104.0, 505.0],
        'PnL': [400.0, -250.0],
        'Risk $': [2625.0, 3000.0],
        'Shares': [100.0, 50.0],
    })
    df = shape_flat_trades(sig)
    assert {'Signal Date', 'Entry Price', 'PnL_flat_750k', 'Risk_flat_750k',
            'Shares_flat', 'Direction', 'Return_Pct', 'R_Multiple',
            'Tier'} <= set(df.columns)
    assert list(df['Direction']) == ['Long', 'Short']
    # long: +4%; short against the move: -1%
    assert np.isclose(df['Return_Pct'].iloc[0], 4.0)
    assert np.isclose(df['Return_Pct'].iloc[1], -1.0)
    assert np.isclose(df['R_Multiple'].iloc[0], 400.0 / 2625.0)
    assert str(df['Signal Date'].dtype).startswith('datetime64')
