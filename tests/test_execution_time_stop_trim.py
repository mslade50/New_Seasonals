"""Partial execution-tab closes may resize a scheduled time stop without STP.

The live executor lives in OneDrive and is absent in CI, so this regression
test skips when that local execution checkout is unavailable.
"""
import os
import sys
from types import SimpleNamespace

import pytest


IBKR_DIR = os.path.join(os.path.expanduser("~"), "OneDrive", "trading_ibkr")


@pytest.fixture(scope="module")
def executor():
    if not os.path.isdir(IBKR_DIR):
        pytest.skip(f"live execution dir not present: {IBKR_DIR}")
    sys.path.insert(0, IBKR_DIR)
    try:
        import execute_order
    except ImportError as exc:
        pytest.skip(f"execute_order not importable here ({exc})")
    return execute_order


@pytest.fixture(scope="module")
def agent():
    if not os.path.isdir(IBKR_DIR):
        pytest.skip(f"live execution dir not present: {IBKR_DIR}")
    sys.path.insert(0, IBKR_DIR)
    try:
        import exec_agent
    except ImportError as exc:
        pytest.skip(f"exec_agent not importable here ({exc})")
    return exec_agent


def _time_stop(qty=100):
    return {
        "order_type": "MKT",
        "tif": "GTC",
        "oca_group": "",
        "oca_type": 1,
        "good_after": "20260731 15:59:00 US/Eastern",
        "good_till": "",
        "outside_rth": True,
        "lmt": 0.0,
        "aux": 0.0,
        "qty": qty,
        "order_ref": "TIME_EXIT",
        "source_key": 123,
    }


def test_time_stop_only_exit_is_valid_protection_and_resizes(executor):
    legs = [_time_stop()]

    assert executor._validate_exit_topology(legs, 100) is None

    scaled, error = executor._scaled_exit_legs(legs, 50)
    assert error is None
    assert scaled[0]["scaled_qty"] == 50
    assert scaled[0]["good_after"] == legs[0]["good_after"]


def test_exposure_adding_actions_reject_target_only_exit(executor):
    target = _time_stop()
    target.update({
        "order_type": "LMT",
        "lmt": 125.0,
        "good_after": "",
    })
    assert executor._validate_exit_topology([target], 100) == (
        "no working price stop or scheduled time stop"
    )


@pytest.mark.parametrize("fraction", [0.25, 0.5])
def test_executor_accepts_time_stop_only_for_add_and_readd(
        executor, monkeypatch, fraction):
    contract = SimpleNamespace(
        symbol="AAPL",
        secType="STK",
        conId=12345,
        exchange="",
        currency="USD",
        lastTradeDateOrContractMonth="",
    )
    position = SimpleNamespace(
        contract=contract,
        position=100,
        account="DU_TEST",
        avgCost=100.0,
    )
    scheduled = executor.MarketOrder("SELL", 100)
    scheduled.orderId = 41
    scheduled.permId = 401
    scheduled.clientId = 17
    scheduled.tif = "GTC"
    scheduled.goodAfterTime = "20260731 15:59:00 US/Eastern"
    scheduled.outsideRth = True
    scheduled.orderRef = "TIME_EXIT"
    trade = SimpleNamespace(
        contract=contract,
        order=scheduled,
        orderStatus=SimpleNamespace(status="Submitted"),
    )

    class FakeIB:
        def positions(self):
            return [position]

        def reqAllOpenOrders(self):
            return None

        def openTrades(self):
            return [trade]

        def sleep(self, _seconds):
            return None

    monkeypatch.setattr(executor, "LIVE_MAX_QTY", 1_000)
    monkeypatch.setattr(executor, "LIVE_MAX_NOTIONAL", 1_000_000)
    identity = {
        "symbol": "AAPL",
        "sec_type": "STK",
        "con_id": 12345,
        "expected_position": 100,
        "fraction": fraction,
    }

    add_ctx, add_error = executor._prepare_fast_position(
        FakeIB(), identity, "primary", partial=False)
    readd_ctx, readd_error = executor._prepare_fast_position(
        FakeIB(), {**identity, "readd": True}, "primary", partial=True)

    assert add_error is None
    assert add_ctx["legs"][0]["good_after"] == scheduled.goodAfterTime
    assert readd_error is None
    assert readd_ctx["legs"][0]["good_after"] == scheduled.goodAfterTime


@pytest.mark.parametrize("fraction", [0.25, 0.5])
def test_agent_accepts_time_stop_only_for_add_and_readd(
        agent, monkeypatch, fraction):
    position = {
        "symbol": "AAPL",
        "sec_type": "STK",
        "con_id": 12345,
        "position": 100,
        "avg_cost": 100.0,
        "market_price": 101.0,
    }
    time_stop = {
        "symbol": "AAPL",
        "sec_type": "STK",
        "con_id": 12345,
        "action": "SELL",
        "order_type": "MKT",
        "qty": 100,
        "status": "PreSubmitted",
        "good_after": "20260731 15:59:00 US/Eastern",
        "oca_group": "",
        "perm_id": 401,
    }
    monkeypatch.setitem(agent._BOOK, "book", {
        "accounts": [{
            "key": "primary",
            "nlv": 750_000,
            "positions": [position],
            "orders": [time_stop],
        }],
    })
    monkeypatch.setattr(agent, "LIVE_MAX_QTY", 1_000)
    identity = {
        "symbol": "AAPL",
        "sec_type": "STK",
        "con_id": 12345,
        "expected_position": 100,
        "fraction": fraction,
    }

    add_ok, add_reasons = agent._validate({
        "type": "add_to_position",
        "account": "primary",
        "payload": identity,
    })
    readd_ok, readd_reasons = agent._validate({
        "type": "trim_readd",
        "account": "primary",
        "payload": {
            **identity,
            "close_order_type": "MKT",
            "readd": True,
            "readd_tif": "DAY",
        },
    })

    assert add_ok, add_reasons
    assert readd_ok, readd_reasons


def test_partial_close_cancels_and_resizes_time_stop_only(executor, capsys):
    contract = SimpleNamespace(
        symbol="AAPL",
        secType="STK",
        conId=12345,
        exchange="",
        currency="USD",
        lastTradeDateOrContractMonth="",
    )
    position = SimpleNamespace(
        contract=contract, position=100, account="DU_TEST"
    )
    scheduled = executor.MarketOrder("SELL", 100)
    scheduled.orderId = 41
    scheduled.permId = 401
    scheduled.clientId = 17
    scheduled.tif = "GTC"
    scheduled.goodAfterTime = "20260731 15:59:00 US/Eastern"
    scheduled.outsideRth = True
    scheduled.orderRef = "TIME_EXIT"
    old_trade = SimpleNamespace(
        contract=contract,
        order=scheduled,
        orderStatus=SimpleNamespace(
            status="Submitted", filled=0, avgFillPrice=0.0
        ),
    )

    class FakeClient:
        def __init__(self):
            self.order_id = 100

        def getReqId(self):
            self.order_id += 1
            return self.order_id

    class FakeIB:
        def __init__(self):
            self.client = FakeClient()
            self.trades = [old_trade]
            self.placed = []

        def positions(self):
            return [position] if position.position else []

        def managedAccounts(self):
            return ["DU_TEST"]

        def reqAllOpenOrders(self):
            # The reservation guard refreshes through this and does list() on
            # the result, so it must return the trades, not None.
            return self.trades

        def openTrades(self):
            return self.trades

        def sleep(self, _seconds):
            return None

        def cancelOrder(self, order):
            old_trade.orderStatus.status = "Cancelled"

        def qualifyContracts(self, *_contracts):
            return None

        def placeOrder(self, placed_contract, order):
            status = "Submitted" if order.goodAfterTime else "Filled"
            filled = 0 if order.goodAfterTime else order.totalQuantity
            trade = SimpleNamespace(
                contract=placed_contract,
                order=order,
                orderStatus=SimpleNamespace(
                    status=status, filled=filled, avgFillPrice=100.0
                ),
            )
            self.placed.append(trade)
            self.trades.append(trade)
            if not order.goodAfterTime:
                position.position -= int(order.totalQuantity)
            return trade

    ib = FakeIB()
    result = executor._do_flatten(
        ib,
        {
            "symbol": "AAPL",
            "sec_type": "STK",
            "con_id": 12345,
            "expected_position": 100,
            "fraction": 0.5,
            "order_type": "MKT",
            "_broker_account": "DU_TEST",
            "_command_id": "flatten-partial-timestop",
        },
        "127.0.0.1",
        7496,
        17,
    )
    payload = capsys.readouterr().out

    assert result == 0
    assert '"state": "executed"' in payload
    assert position.position == 50
    assert old_trade.orderStatus.status == "Cancelled"
    assert len(ib.placed) == 2
    resized = ib.placed[1].order
    assert resized.orderType == "MKT"
    assert resized.totalQuantity == 50
    assert resized.goodAfterTime == scheduled.goodAfterTime
    assert resized.tif == "GTC"
    assert resized.outsideRth is True


@pytest.mark.parametrize(
    ("starting_position", "requested_action", "expected_position"),
    [(25, "SELL", 15), (-25, "BUY", -15)],
)
def test_close_only_uses_closing_side_and_never_touches_orders(
        executor, capsys, starting_position, requested_action,
        expected_position):
    contract = SimpleNamespace(
        symbol="AAPL", secType="STK", conId=12345, exchange="",
        currency="USD", lastTradeDateOrContractMonth="",
    )
    position = SimpleNamespace(
        contract=contract, position=starting_position, account="DU_TEST",
    )

    class FakeIB:
        def __init__(self):
            self.placed = []

        def positions(self):
            return [position] if position.position else []

        def qualifyContracts(self, *_contracts):
            return None

        def managedAccounts(self):
            return ["DU_TEST"]

        # close_only READS the working orders (2026-09-02) to enforce its
        # bare-position rule — a resting exit the same size as the close could
        # fill beside it and reverse the position. What it must never do is
        # change one, so cancel/modify stay fatal here.
        def reqAllOpenOrders(self):
            return []

        def openTrades(self):
            return []

        def cancelOrder(self, _order):
            raise AssertionError("close_only must not cancel working orders")

        def sleep(self, _seconds):
            return None

        def placeOrder(self, placed_contract, order):
            self.placed.append((placed_contract, order))
            if order.action == "SELL":
                position.position -= int(order.totalQuantity)
            else:
                position.position += int(order.totalQuantity)
            return SimpleNamespace(
                order=order,
                orderStatus=SimpleNamespace(
                    status="Filled", filled=order.totalQuantity,
                    avgFillPrice=100.0,
                ),
            )

    ib = FakeIB()
    result = executor._do_close_only(ib, {
        "symbol": "AAPL", "sec_type": "STK", "con_id": 12345,
        "_broker_account": "DU_TEST", "_command_id": "close-only-side",
        "qty": 10, "action": requested_action, "order_type": "MKT",
    })
    payload = capsys.readouterr().out

    assert result == 0
    assert '"state": "executed"' in payload
    assert '"working_orders_untouched": true' in payload
    assert len(ib.placed) == 1
    assert ib.placed[0][1].action == requested_action
    assert int(ib.placed[0][1].totalQuantity) == 10
    assert position.position == expected_position


def test_close_only_rejects_add_side_and_oversize(executor, capsys):
    contract = SimpleNamespace(
        symbol="AAPL", secType="STK", conId=12345, exchange="",
        currency="USD", lastTradeDateOrContractMonth="",
    )
    position = SimpleNamespace(
        contract=contract, position=25, account="DU_TEST",
    )

    class FakeIB:
        def __init__(self):
            self.placed = []

        def positions(self):
            return [position]

        def qualifyContracts(self, *_contracts):
            return None

        def managedAccounts(self):
            return ["DU_TEST"]

        def reqAllOpenOrders(self):
            return []

        def openTrades(self):
            return []

        def placeOrder(self, contract, order):
            self.placed.append((contract, order))
            raise AssertionError("rejected close must not place an order")

    ib = FakeIB()
    executor._do_close_only(ib, {
        "symbol": "AAPL", "sec_type": "STK", "con_id": 12345,
        "_broker_account": "DU_TEST", "_command_id": "close-only-wrong-side",
        "qty": 10, "action": "BUY", "order_type": "MKT",
    })
    wrong_side = capsys.readouterr().out
    executor._do_close_only(ib, {
        "symbol": "AAPL", "sec_type": "STK", "con_id": 12345,
        "_broker_account": "DU_TEST", "_command_id": "close-only-oversize",
        "qty": 26, "action": "SELL", "order_type": "MKT",
    })
    oversize = capsys.readouterr().out

    assert "would add to the live position" in wrong_side
    assert "exceeds held 25" in oversize
    assert ib.placed == []


def test_agent_close_only_ignores_misaligned_working_orders(agent, monkeypatch):
    position = {
        "symbol": "AAPL", "sec_type": "STK", "con_id": 12345,
        "position": 25, "avg_cost": 100.0, "market_price": 101.0,
    }
    monkeypatch.setitem(agent._BOOK, "book", {
        "accounts": [{
            "key": "primary", "nlv": 750_000, "positions": [position],
            "orders": [
                {"symbol": "AAPL", "sec_type": "STK", "con_id": 12345,
                 "action": "SELL", "order_type": "MKT", "qty": 10,
                 "good_after": "20260801 15:59:00 US/Eastern"},
                {"symbol": "AAPL", "sec_type": "STK", "con_id": 12345,
                 "action": "SELL", "order_type": "MKT", "qty": 15,
                 "good_after": "20260802 15:59:00 US/Eastern"},
                {"symbol": "AAPL", "sec_type": "STK", "con_id": 12345,
                 "action": "BUY", "order_type": "LMT", "qty": 10,
                 "lmt": 95.0},
            ],
        }],
    })
    base = {
        "type": "close_only", "account": "primary",
        "payload": {
            "symbol": "AAPL", "sec_type": "STK", "con_id": 12345,
            "qty": 10, "action": "SELL", "order_type": "MKT",
        },
    }

    ok, reasons = agent._validate(base)
    preview = agent._preview(base)
    wrong_ok, wrong_reasons = agent._validate({
        **base, "payload": {**base["payload"], "action": "BUY"},
    })
    large_ok, large_reasons = agent._validate({
        **base, "payload": {**base["payload"], "qty": 26},
    })

    assert ok, reasons
    assert preview["legs"][-1] == "LEAVE ALL WORKING ORDERS UNCHANGED"
    assert "working orders unchanged" in preview["summary"]
    assert not wrong_ok
    assert any("would add" in reason for reason in wrong_reasons)
    assert not large_ok
    assert any("exceeds held 25" in reason for reason in large_reasons)
