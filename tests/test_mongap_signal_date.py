"""Monday-gap kill weekday gating (fixed 2026-07-16).

The MonFri Reversion gap-kill (drop a Friday signal whose Monday open gaps
>0.5 ATR up) was dead in live from 2026-06-09: order_staging gated on
Scan_Date's weekday, but the Monday ~4:47 AM rescan restamps Scan_Date to
Monday, so the check (0 not in [4]) never entered the kill branch. The fix
stamps the signal bar's date as Signal_Date on every staging row and gates
on that via order_staging.mongap_gate_weekday().

Two halves:
1. Config/scanner contract — the MonFri strategy carries the gate spec and
   daily_scan stamps Signal_Date (source-level assertion, runs everywhere).
2. Live gate behavior — mongap_gate_weekday() resolves Friday from a
   restamped Monday row. Skipped when order_staging.py (OneDrive, outside
   this repo) isn't importable, e.g. in CI.
"""
import os
import re
import sys

import pytest

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

from strategy_config import STRATEGY_BOOK

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IBKR_DIR = os.path.join(
    os.path.expanduser('~'), 'OneDrive', 'trading_ibkr'
)


def _monfri():
    for s in STRATEGY_BOOK:
        if s['settings'].get('use_t1_gap_kill'):
            return s
    raise AssertionError("no strategy in STRATEGY_BOOK carries use_t1_gap_kill")


def test_gap_kill_spec_targets_friday_signals():
    s = _monfri()
    st = s['settings']
    assert st.get('t1_gap_kill_signal_weekdays') == [4], (
        "gap-kill weekday spec must be [4]=Friday; the live gate keys on the "
        "SIGNAL bar's weekday via Signal_Date"
    )
    assert float(st.get('t1_gap_kill_atr', 0)) > 0
    assert st.get('t1_gap_kill_dir', 'up') == 'up'


def test_scanner_stamps_signal_date():
    """save_staging_orders must stamp Signal_Date from the signal bar's date
    (row['Date']), not the run clock. Source-level guard: the staging dict
    literal carries the column and derives it from row['Date']."""
    src = open(os.path.join(REPO_ROOT, 'daily_scan.py'), encoding='utf-8').read()
    m = re.search(r'"Signal_Date":\s*str\(row\[[\'"]Date[\'"]\]\)', src)
    assert m, (
        "daily_scan staging rows must stamp Signal_Date=str(row['Date']); "
        "order_staging's Monday-gap kill gates on it"
    )


@pytest.fixture(scope='module')
def order_staging_mod():
    if not os.path.isdir(IBKR_DIR):
        pytest.skip(f"live execution dir not present: {IBKR_DIR}")
    sys.path.insert(0, IBKR_DIR)
    try:
        import order_staging
    except ImportError as e:
        pytest.skip(f"order_staging not importable here ({e})")
    return order_staging


def test_gate_weekday_reads_signal_date(order_staging_mod):
    """The restamp scenario: Friday signal, rescanned Monday. The gate must
    resolve Friday (4) from Signal_Date even though Scan_Date says Monday."""
    row = {
        'Signal_Date': '2026-07-10',   # a Friday
        'Scan_Date': '2026-07-13',     # the Monday AM rescan restamp
    }
    assert order_staging_mod.mongap_gate_weekday(row) == 4


def test_gate_weekday_falls_back_to_scan_date(order_staging_mod):
    # legacy row without the column: fall back to Scan_Date (keeps the old
    # behavior for rows staged before 2026-07-16; sheet clears next scan)
    assert order_staging_mod.mongap_gate_weekday({'Scan_Date': '2026-07-10'}) == 4
    assert order_staging_mod.mongap_gate_weekday(
        {'Signal_Date': '', 'Scan_Date': '2026-07-10'}) == 4


def test_gate_weekday_unparseable_keeps_trade(order_staging_mod):
    # -1 never matches a weekday list -> gate keeps the trade (valid by default)
    assert order_staging_mod.mongap_gate_weekday({}) == -1
    assert order_staging_mod.mongap_gate_weekday(
        {'Signal_Date': 'garbage', 'Scan_Date': 'also garbage'}) == -1
