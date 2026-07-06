"""sector_loss_gate: OLV same-sector loss gate (2026-07-02).

The rule: skip a new signal when the strategy's realized R in the same sector
over the trailing window_td business days is max_realized_r or worse. Guards
the daily_scan live gate (ledger-driven) and the config entry; the engine-side
candidate gate shares the same semantics (exits strictly before signal date).
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _NoOp:
    def __getattr__(self, name): return self
    def __call__(self, *a, **k): return self
    def __bool__(self): return False
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def cache_data(self, *a, **k):
        def deco(fn): return fn
        return deco
    cache_resource = cache_data


sys.modules['streamlit'] = _NoOp()

import daily_scan
from strategy_config import STRATEGY_BOOK

CFG = {'sector_loss_gate': {'window_td': 10, 'max_realized_r': -2.0}}


def _prime(ledger, smap):
    daily_scan._SLG_CACHE.clear()
    daily_scan._SLG_CACHE['state'] = (ledger, smap)


def _ledger(rows):
    df = pd.DataFrame(rows, columns=['Strategy', 'Ticker', 'Exit Date', 'R_Multiple'])
    df['Exit Date'] = pd.to_datetime(df['Exit Date'])
    return df


SMAP = {'OXY': 'Energy', 'USO': 'Energy', 'XOM': 'Energy', 'AAPL': 'Technology'}


def teardown_function():
    daily_scan._SLG_CACHE.clear()


def test_blocks_after_sector_losses_inside_window():
    _prime(_ledger([
        ('Oversold Low Volume', 'OXY', '2026-06-22', -1.0),
        ('Oversold Low Volume', 'USO', '2026-06-24', -1.1),
    ]), SMAP)
    blocked, note = daily_scan.sector_gate_blocked(
        'Oversold Low Volume', CFG, 'XOM', '2026-06-26')
    assert blocked and 'Energy' in note
    # note must name every contributing exit (auditability: the ledger vintage
    # that produced a block is overwritten by the next rebuild)
    assert 'OXY' in note and 'USO' in note and '-1.10R' in note


def test_passes_when_losses_precede_window():
    _prime(_ledger([
        ('Oversold Low Volume', 'OXY', '2026-05-01', -2.0),
        ('Oversold Low Volume', 'USO', '2026-05-05', -2.0),
    ]), SMAP)
    blocked, _ = daily_scan.sector_gate_blocked(
        'Oversold Low Volume', CFG, 'XOM', '2026-06-26')
    assert not blocked


def test_other_sector_and_other_strategy_do_not_block():
    _prime(_ledger([
        ('Oversold Low Volume', 'AAPL', '2026-06-24', -3.0),      # Tech, not Energy
        ('LT Trend ST OS', 'OXY', '2026-06-24', -3.0),            # other strategy
    ]), SMAP)
    blocked, _ = daily_scan.sector_gate_blocked(
        'Oversold Low Volume', CFG, 'XOM', '2026-06-26')
    assert not blocked


def test_boundary_exactly_at_threshold_passes():
    # rule is strictly-worse-than: sum == max_realized_r does NOT block
    _prime(_ledger([('Oversold Low Volume', 'OXY', '2026-06-24', -2.0)]), SMAP)
    blocked, _ = daily_scan.sector_gate_blocked(
        'Oversold Low Volume', CFG, 'XOM', '2026-06-26')
    assert not blocked


def test_same_day_exits_excluded():
    # exits ON the signal date are unknown at staging time -> excluded
    _prime(_ledger([('Oversold Low Volume', 'OXY', '2026-06-26', -5.0)]), SMAP)
    blocked, _ = daily_scan.sector_gate_blocked(
        'Oversold Low Volume', CFG, 'XOM', '2026-06-26')
    assert not blocked


def test_fail_open_without_ledger_or_sector():
    _prime(None, SMAP)
    assert not daily_scan.sector_gate_blocked('Oversold Low Volume', CFG, 'XOM', '2026-06-26')[0]
    _prime(_ledger([('Oversold Low Volume', 'OXY', '2026-06-24', -9.0)]), SMAP)
    # unknown ticker -> no sector -> pass through
    assert not daily_scan.sector_gate_blocked('Oversold Low Volume', CFG, 'ZZZZ', '2026-06-26')[0]


def test_config_carries_the_gate_on_olv_only():
    with_gate = [s['name'] for s in STRATEGY_BOOK
                 if s['execution'].get('sector_loss_gate')]
    assert with_gate == ['Oversold Low Volume']
    cfg = next(s for s in STRATEGY_BOOK if s['name'] == 'Oversold Low Volume')
    assert cfg['execution']['sector_loss_gate'] == {'window_td': 10, 'max_realized_r': -2.0}
