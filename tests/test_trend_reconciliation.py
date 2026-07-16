"""Trend-sleeve state vs live-book reconciliation (2026-07-16).

The month-end rebalance computes DELTAS against trend_sleeve_state.json; if
staged orders were never executed (or doubled by a re-run), every future
rebalance is wrong and nothing surfaced it. reconcile_trend diffs the state's
target shares against the live book once Execute_On has passed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daily_execution_report import reconcile_trend


def _state(shares_by_sym, generated="2026-07-10 21:40:00"):
    return {
        "generated": generated,
        "positions": {s: {"shares": n} for s, n in shares_by_sym.items()},
    }


def _book(qty_by_sym):
    return [{"symbol": s, "sec_type": "STK", "position": q}
            for s, q in qty_by_sym.items()]


def test_match_is_quiet():
    st = _state({"GLD": 100, "TLT": 200})
    assert reconcile_trend(st, _book({"GLD": 100, "TLT": 200, "AAPL": 50})) == []


def test_mismatch_and_missing_flagged():
    st = _state({"GLD": 100, "TLT": 200})
    issues = reconcile_trend(st, _book({"GLD": 250}))  # doubled-ish GLD, no TLT
    assert len(issues) == 2
    assert any("GLD" in i and "100" in i and "250" in i for i in issues)
    assert any("TLT" in i and "0" in i for i in issues)


def test_grace_period_before_execute_on(monkeypatch):
    # state generated 'today' -> Execute_On is the NEXT session -> skip
    import daily_execution_report as der
    today = der.et_now().strftime("%Y-%m-%d")
    st = _state({"GLD": 100}, generated=f"{today} 21:40:00")
    assert reconcile_trend(st, _book({})) == []


def test_empty_state_is_quiet():
    assert reconcile_trend({}, _book({"GLD": 100})) == []
