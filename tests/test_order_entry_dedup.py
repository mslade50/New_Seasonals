"""Filled-order duplicate guard (2026-07-16).

eq_order_entry/pa_order_entry built their duplicate guard solely from
ib.openTrades(), which only sees WORKING orders. A filled naked MOO (the
trend sleeve fills seconds after the open) or a fully-closed intraday
bracket leaves no working leg, so a same-day re-run re-entered — doubling
a ~$450k sleeve invisibly to next month's rebalance delta math. The fix
merges today's ib.fills() orderRefs plus a local placed-orders journal
(survives a TWS/Gateway restart) into the guard sets.

These tests exercise the journal half (pure file I/O). Skipped where the
OneDrive execution dir isn't present (CI).
"""
import os
import sys

import pytest

IBKR_DIR = os.path.join(os.path.expanduser('~'), 'OneDrive', 'trading_ibkr')


@pytest.fixture(scope='module')
def eq_mod():
    if not os.path.isdir(IBKR_DIR):
        pytest.skip(f"live execution dir not present: {IBKR_DIR}")
    sys.path.insert(0, IBKR_DIR)
    try:
        import eq_order_entry
    except ImportError as e:
        pytest.skip(f"eq_order_entry not importable here ({e})")
    return eq_order_entry


def test_journal_roundtrip(eq_mod, tmp_path):
    j = str(tmp_path / 'placed.json')
    fp = ('TQQQ', 'SELL', 100, 'LMT', 55.25)
    sig = 'TQQQ|SELL|3x Leader Gap Fade|2026-07-16'
    eq_mod.journal_placed('2026-07-16', sig, fp, path=j)
    refs, fps = eq_mod.load_placed_today('2026-07-16', path=j)
    assert sig in refs
    assert fp in fps


def test_journal_prunes_prior_days(eq_mod, tmp_path):
    j = str(tmp_path / 'placed.json')
    eq_mod.journal_placed('2026-07-15', 'OLD|SIG', ('A', 'BUY', 1, 'LMT', 1.0), path=j)
    eq_mod.journal_placed('2026-07-16', 'NEW|SIG', ('B', 'BUY', 2, 'LMT', 2.0), path=j)
    refs_today, _ = eq_mod.load_placed_today('2026-07-16', path=j)
    refs_old, _ = eq_mod.load_placed_today('2026-07-15', path=j)
    assert 'NEW|SIG' in refs_today
    # the second write (a new day) pruned the prior day's entries
    assert refs_old == set()


def test_journal_missing_file_is_clean_slate(eq_mod, tmp_path):
    refs, fps = eq_mod.load_placed_today('2026-07-16', path=str(tmp_path / 'nope.json'))
    assert refs == set() and fps == set()


def test_naked_moo_signal_ref_matches_trend_row(eq_mod):
    # the trend sleeve's identity survives a re-run: same symbol/side/
    # strategy/staged-date -> same ref, regardless of qty/price
    a = eq_mod.signal_ref('GLD', 'BUY', 'Trend Sleeve', '2026-07-16')
    b = eq_mod.signal_ref('gld ', 'buy', 'Trend Sleeve', '2026-07-16')
    assert a == b
