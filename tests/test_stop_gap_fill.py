"""Regression test for the gap-aware stop fill + slippage model
(strat_backtester._stop_fill_price, added 2026-06-27).

Locks the contract that a stop the bar GAPS THROUGH fills at the open (not the
stop), every stop fill eats STOP_SLIP_BPS, and a gapped fill eats an ADDITIONAL
STOP_GAP_SLIP_BPS. gap_fill=False must reproduce the legacy fill-at-stop.
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
        if len(a) == 1 and callable(a[0]) and not k:
            return a[0]
        def deco(fn): return fn
        return deco
    cache_resource = cache_data
sys.modules['streamlit'] = _NoOp()

from strat_backtester import _stop_fill_price, STOP_SLIP_BPS, STOP_GAP_SLIP_BPS

TOL = 1e-9


def approx(a, b):
    assert abs(a - b) < TOL, f"expected {b}, got {a}"


def test_long_no_gap():
    # open above the stop -> traded down through it intraday -> fill at stop, slip only
    fill, gapped = _stop_fill_price('Long', stop_price=100.0, day_open=101.0)
    assert gapped is False
    approx(fill, 100.0 * (1 - STOP_SLIP_BPS / 1e4))


def test_long_gap_through():
    # open below the stop -> gapped -> fill at open, slip + gap slip
    fill, gapped = _stop_fill_price('Long', stop_price=100.0, day_open=97.0)
    assert gapped is True
    approx(fill, 97.0 * (1 - (STOP_SLIP_BPS + STOP_GAP_SLIP_BPS) / 1e4))
    # and it is strictly worse than the legacy fill-at-stop
    assert fill < 100.0


def test_short_no_gap():
    fill, gapped = _stop_fill_price('Short', stop_price=100.0, day_open=99.0)
    assert gapped is False
    approx(fill, 100.0 * (1 + STOP_SLIP_BPS / 1e4))


def test_short_gap_through():
    # short stop is above; open above it -> gapped -> cover at open, higher
    fill, gapped = _stop_fill_price('Short', stop_price=100.0, day_open=103.0)
    assert gapped is True
    approx(fill, 103.0 * (1 + (STOP_SLIP_BPS + STOP_GAP_SLIP_BPS) / 1e4))
    assert fill > 100.0


def test_legacy_mode_unchanged():
    # gap_fill=False reproduces the old engine exactly: fill at stop, no slip
    for direction, op in [('Long', 90.0), ('Short', 110.0)]:
        fill, gapped = _stop_fill_price(direction, 100.0, op, gap_fill=False)
        approx(fill, 100.0)
        assert gapped is False


def test_custom_bps():
    fill, _ = _stop_fill_price('Long', 100.0, 95.0, slip_bps=5.0, gap_slip_bps=20.0)
    approx(fill, 95.0 * (1 - 25.0 / 1e4))


def main():
    test_long_no_gap()
    test_long_gap_through()
    test_short_no_gap()
    test_short_gap_through()
    test_legacy_mode_unchanged()
    test_custom_bps()
    print("All stop gap-fill cases passed.")


if __name__ == '__main__':
    main()
