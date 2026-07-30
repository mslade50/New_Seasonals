"""earnings_size_override: pre-earnings flat-replace sizing (OLV 2026-04-30,
St OS Sznl 2026-07-30).

Guards the carrier set and each carrier's window/bps so a config edit can't
silently widen a window or de-scale the override. The machinery itself is
generic (daily_scan sizing 2d + strat_backtester 3b key on the field), so
config invariants are the load-bearing surface.

St OS Sznl rationale (2026-07-30): the no-stop 5d hold straddles imminent
prints — ledger -5..-1 TD cell N=9 avgR -0.50 with every tail loser, vs
+0.32 outside. Small-N risk-appetite haircut to ~15% of normal size, not a
fitted edge rule.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_config import GLOBAL_RISK_MULTIPLIER, STRATEGY_BOOK

# name -> (min_td, max_td, nominal_bps)
EXPECTED = {
    "Oversold Low Volume": (-10, 0, 10),
    "St OS Sznl": (-5, -1, 6),
}


def _carriers():
    return {s["name"]: s["execution"]["earnings_size_override"]
            for s in STRATEGY_BOOK
            if s.get("execution", {}).get("earnings_size_override")}


def test_exactly_the_expected_carriers():
    assert set(_carriers()) == set(EXPECTED)


def test_windows_and_grm_scaled_bps():
    for name, (lo, hi, nominal) in EXPECTED.items():
        eo = _carriers()[name]
        assert eo["min_td"] == lo, name
        assert eo["max_td"] == hi, name
        assert eo["risk_bps"] == nominal * GLOBAL_RISK_MULTIPLIER, name


def test_window_is_pre_earnings_only():
    # Both carriers key on earnings AHEAD (negative offsets); a positive
    # min_td/max_td would target post-earnings days — different mechanism,
    # different study required.
    for name, eo in _carriers().items():
        assert eo["min_td"] <= eo["max_td"] <= 0, name
