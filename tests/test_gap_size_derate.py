"""Large-gap-up size derate (2026-07-21).

Guards the aligned sites: strategy_config carries execution['gap_size_derate']
on the two dip-buys (Monday Dip, SPY QQQ MonFri Reversion), the engine applies
it point-in-time via strat_backtester.gap_derate_mult (sizing step 3b5), and
order_staging (out-of-repo) enforces the SAME rule at the IBKR open off the
GapDerate_* stamps daily_scan writes. This test covers the two in-repo sites;
the live parity is the shared formula (thr/dir semantics) exercised below.

The derate is a SIZING overlay, distinct from SPY QQQ MonFri's t1_gap_kill
(a Friday-only full DROP, a filter in get_historical_mask). The kill runs first
and this half-sizes whatever it leaves standing, so both must stay configured.
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

from strategy_config import STRATEGY_BOOK
import pages.strat_backtester as sb

DERATE_STRATS = {"Monday Dip", "SPY QQQ MonFri Reversion"}
SPEC = {"threshold_atr": 0.25, "mult": 0.5, "dir": "up"}


def _exec_of(name):
    for s in STRATEGY_BOOK:
        if s["name"] == name:
            return s["execution"]
    raise AssertionError(f"strategy {name!r} not in STRATEGY_BOOK")


def test_config_carries_exactly_the_two_derate_strats():
    with_derate = {s["name"]: s["execution"]["gap_size_derate"]
                   for s in STRATEGY_BOOK if s["execution"].get("gap_size_derate")}
    assert set(with_derate) == DERATE_STRATS
    for name in DERATE_STRATS:
        d = with_derate[name]
        assert float(d["threshold_atr"]) == 0.25
        assert float(d["mult"]) == 0.5
        assert str(d["dir"]).lower() == "up"


def test_spy_qqq_keeps_its_friday_gap_kill_alongside_the_derate():
    # The kill (settings) and the derate (execution) are independent layers —
    # neither may silently replace the other.
    s = None
    for x in STRATEGY_BOOK:
        if x["name"] == "SPY QQQ MonFri Reversion":
            s = x
    assert s is not None
    assert s["settings"].get("use_t1_gap_kill") is True
    assert float(s["settings"]["t1_gap_kill_atr"]) == 0.5
    assert s["execution"].get("gap_size_derate") is not None


def test_helper_boundary_up():
    sc, atr = 100.0, 4.0          # 0.25 ATR = 1.00 -> threshold 101.00
    # just BELOW threshold -> full size
    assert sb.gap_derate_mult(SPEC, sc, 100.99, atr) == 1.0
    # exactly AT threshold is NOT a strict gap (> only) -> full size
    assert sb.gap_derate_mult(SPEC, sc, 101.00, atr) == 1.0
    # just ABOVE threshold -> derated
    assert sb.gap_derate_mult(SPEC, sc, 101.01, atr) == 0.5
    # big gap up -> derated
    assert sb.gap_derate_mult(SPEC, sc, 108.00, atr) == 0.5
    # gap DOWN under an 'up' spec -> full size
    assert sb.gap_derate_mult(SPEC, sc, 96.00, atr) == 1.0


def test_helper_direction_down():
    spec = {"threshold_atr": 0.25, "mult": 0.5, "dir": "down"}
    sc, atr = 100.0, 4.0          # threshold 99.00
    assert sb.gap_derate_mult(spec, sc, 99.01, atr) == 1.0   # not through
    assert sb.gap_derate_mult(spec, sc, 98.99, atr) == 0.5   # gapped down
    assert sb.gap_derate_mult(spec, sc, 105.0, atr) == 1.0   # gap up, ignored


def test_helper_fails_open_on_bad_inputs():
    sc, atr = 100.0, 4.0
    assert sb.gap_derate_mult(None, sc, 108.0, atr) == 1.0
    assert sb.gap_derate_mult({}, sc, 108.0, atr) == 1.0
    assert sb.gap_derate_mult(SPEC, sc, 0.0, atr) == 1.0        # no open
    assert sb.gap_derate_mult(SPEC, sc, float("nan"), atr) == 1.0
    assert sb.gap_derate_mult(SPEC, sc, 108.0, 0.0) == 1.0      # no ATR
    assert sb.gap_derate_mult(SPEC, 0.0, 108.0, atr) == 1.0     # no signal close


def test_helper_composes_multiplicatively_with_frag_band():
    # A high-fragility gap-up day: FAMILY4 0.25x frag band x 0.5x derate = 0.125x.
    gd = sb.gap_derate_mult(SPEC, 100.0, 108.0, 4.0)
    assert round(0.25 * gd, 4) == 0.125
