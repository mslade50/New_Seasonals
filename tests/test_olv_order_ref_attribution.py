"""Cap attribution straight from broker order refs, with no reviewed seed.

The reconciled bridge refuses whenever canonical fill coverage stops bracketing
the reviewed seed, which took the OLV cap out of the scan for days at a time.
Order refs already carry the sleeve that staged each entry, so attribution does
not need a continuous fill history.
"""
from __future__ import annotations

import pytest

from olv_sizing import STRATEGY, held_notionals_from_order_refs

ASOF = "2026-09-14T20:00:00+00:00"
ACCOUNT = "U1234567"


def book(*, fills=(), orders=(), positions=(), account=ACCOUNT, error=None):
    return {"accounts": [{
        "key": "primary",
        "broker_account": account,
        "error": error,
        "fills": list(fills),
        "orders": list(orders),
        "positions": list(positions),
    }]}


def ref(symbol, strategy, staged):
    return {"order_ref": f"{symbol}|BUY|{strategy}|{staged}", "symbol": symbol}


def position(symbol, shares, value):
    return {"symbol": symbol, "position": shares, "market_value": value}


def test_symbol_the_sleeve_staged_is_attributed_whole():
    result = held_notionals_from_order_refs(
        book(fills=[ref("AAPL", STRATEGY, "2026-09-10")],
             positions=[position("AAPL", 100, 15_000.0)]),
        ACCOUNT, asof=ASOF)
    assert result == {("AAPL", STRATEGY): 15_000.0}


def test_another_sleeves_symbol_is_left_alone():
    result = held_notionals_from_order_refs(
        book(fills=[ref("MSFT", "Trend", "2026-09-10")],
             positions=[position("MSFT", 50, 9_000.0)]),
        ACCOUNT, asof=ASOF)
    assert result == {}


def test_a_stale_order_ref_falls_out_of_the_lookback():
    result = held_notionals_from_order_refs(
        book(fills=[ref("OLD", STRATEGY, "2026-01-02")],
             positions=[position("OLD", 10, 500.0)]),
        ACCOUNT, asof=ASOF, lookback_days=30)
    assert result == {}


def test_working_orders_count_as_evidence_too():
    result = held_notionals_from_order_refs(
        book(orders=[ref("NVDA", STRATEGY, "2026-09-12")],
             positions=[position("NVDA", 20, 4_000.0)]),
        ACCOUNT, asof=ASOF)
    assert result == {("NVDA", STRATEGY): 4_000.0}


def test_a_flat_position_is_not_attributed():
    result = held_notionals_from_order_refs(
        book(fills=[ref("AAPL", STRATEGY, "2026-09-10")],
             positions=[position("AAPL", 0, 0.0)]),
        ACCOUNT, asof=ASOF)
    assert result == {}


def test_a_short_is_attributed_at_absolute_value():
    result = held_notionals_from_order_refs(
        book(fills=[ref("TSLA", STRATEGY, "2026-09-11")],
             positions=[position("TSLA", -40, -8_000.0)]),
        ACCOUNT, asof=ASOF)
    assert result == {("TSLA", STRATEGY): 8_000.0}


def test_a_missing_market_value_refuses_rather_than_understating_the_cap():
    with pytest.raises(ValueError, match="market value"):
        held_notionals_from_order_refs(
            book(fills=[ref("AAPL", STRATEGY, "2026-09-10")],
                 positions=[{"symbol": "AAPL", "position": 100, "market_value": None}]),
            ACCOUNT, asof=ASOF)


def test_an_unparseable_staged_date_keeps_the_symbol():
    # Dropping it would silently remove a real position from the cap.
    result = held_notionals_from_order_refs(
        book(fills=[ref("AAPL", STRATEGY, "not-a-date")],
             positions=[position("AAPL", 100, 15_000.0)]),
        ACCOUNT, asof=ASOF)
    assert result == {("AAPL", STRATEGY): 15_000.0}


def test_account_mismatch_refuses():
    with pytest.raises(ValueError, match="account mismatch"):
        held_notionals_from_order_refs(book(account="OTHER"), ACCOUNT, asof=ASOF)


def test_an_errored_primary_account_refuses():
    with pytest.raises(ValueError, match="healthy Primary"):
        held_notionals_from_order_refs(book(error="gateway down"), ACCOUNT, asof=ASOF)


def test_a_naive_asof_refuses():
    with pytest.raises(ValueError, match="timezone"):
        held_notionals_from_order_refs(book(), ACCOUNT, asof="2026-09-14T20:00:00")
