"""Every exposed executor ticket with real IBKR order classes and simulated transport."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path

import pytest

from tests.execution_harness import (FIXTURE, SimBroker, bind, install_helpers,
                                     load_executor, position, LimitOrder, StopOrder)


@pytest.fixture(autouse=True)
def helpers(monkeypatch):
    install_helpers(monkeypatch)


def invoke(executor, function, *args):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        code = getattr(executor, function)(*args)
    assert code == 0
    result = json.loads(stream.getvalue())  # rejects multiple terminal documents
    assert result["ok"] == (result["state"] == "executed")
    return result


def entry(asset="STK", kind="LMT", account="PRIMARY", action="BUY"):
    return dict(_command_id="matrix-entry", _broker_account=account, symbol="EUR" if asset == "CASH" else "MES" if asset == "FUT" else "TEST",
                action=action, sec_type=asset, currency="USD", quantity=2,
                entry=1.1 if asset == "CASH" else 100,
                stop=(1.0 if action == "BUY" else 1.2) if asset == "CASH" else (90 if action == "BUY" else 110),
                target=(1.2 if action == "BUY" else 1.0) if asset == "CASH" else (120 if action == "BUY" else 80),
                entry_cap=(101 if action == "BUY" else 99) if kind == "STP_LMT" else None,
                entry_type=kind, fut_expiry="203012", exchange="CME")


ENTRY_CASES = [("STK", k) for k in ("LMT", "STP_LMT", "MKT", "MOO", "MOC")] + [("FUT", k) for k in ("LMT", "MKT", "MOO")] + [("CASH", k) for k in ("LMT", "MKT")]


@pytest.mark.parametrize("account", ["primary", "pa"])
@pytest.mark.parametrize("action", ["BUY", "SELL"])
@pytest.mark.parametrize("asset,kind", ENTRY_CASES)
def test_all_entry_types_build_and_acknowledge_complete_brackets(tmp_path, account, action, asset, kind):
    broker = SimBroker()
    executor = bind(load_executor(), broker, tmp_path)
    p = entry(asset, kind, account.upper(), action)
    p["time_stop"] = "2030-12-20"
    result = invoke(executor, "_do_entry_bracket", broker, p, account)
    assert result["ok"], result
    orders = [row[1] for row in broker.sent]
    assert len(orders) == 4
    parent, children = orders[0], orders[1:]
    assert parent.orderType == {"MOO": "MKT", "STP_LMT": "STP LMT"}.get(kind, kind)
    assert parent.tif == ("OPG" if kind == "MOO" else "DAY")
    assert not parent.transmit and children[-1].transmit
    assert all(o.parentId == parent.orderId and o.account == account.upper() and o.totalQuantity == 2 for o in children)
    assert [o.orderType for o in children] == ["LMT", "STP", "MKT"]
    assert children[-1].goodAfterTime.startswith("20301220 15:59:00")
    if kind == "STP_LMT":
        assert (parent.auxPrice, parent.lmtPrice) == (p["entry"], p["entry_cap"])


@pytest.mark.parametrize("asset,kind", [("FUT", "MOC"), ("FUT", "STP_LMT"), ("CASH", "MOO"), ("CASH", "MOC"), ("CASH", "STP_LMT"), ("STK", "STP")])
def test_unexposed_entry_combinations_reject_without_submission(tmp_path, asset, kind):
    broker = SimBroker()
    result = invoke(bind(load_executor(), broker, tmp_path), "_do_entry_bracket", broker, entry(asset, kind), "primary")
    assert result["state"] == "rejected" and not broker.sent


@pytest.mark.parametrize("asset", ["STK", "FUT", "CASH"])
@pytest.mark.parametrize("account", ["primary", "pa"])
@pytest.mark.parametrize("legs", [("stop",), ("target",), ("time_stop",), ("stop", "target", "time_stop")])
def test_exit_attach_supports_each_leg_subset(tmp_path, asset, account, legs):
    held = position(asset, account.upper())
    broker = SimBroker([held])
    p = dict(_command_id="attach", _broker_account=account.upper(), con_id=42, symbol=held.contract.symbol, sec_type=asset)
    p.update({key: {"stop": 1.0 if asset == "CASH" else 90, "target": 1.2 if asset == "CASH" else 120, "time_stop": "2030-12-20"}[key] for key in legs})
    result = invoke(bind(load_executor(), broker, tmp_path), "_do_exit_attach", broker, p, account)
    assert result["ok"], result
    orders = [row[1] for row in broker.sent]
    assert len(orders) == len(legs)
    assert all(o.account == account.upper() and o.totalQuantity == 100 and o.action == "SELL" for o in orders)
    if len(orders) > 1:
        assert len({o.ocaGroup for o in orders}) == 1 and all(o.ocaType == 1 for o in orders)


def close_payload(held, order_type="MKT"):
    return dict(_command_id="close", _broker_account=held.account, con_id=held.contract.conId,
                symbol=held.contract.symbol, sec_type=held.contract.secType,
                action="SELL" if held.position > 0 else "BUY", qty=abs(held.position),
                order_type=order_type, limit=100, tif="DAY", expected_position=held.position)


@pytest.mark.parametrize("asset", ["STK", "FUT", "CASH"])
@pytest.mark.parametrize("account", ["primary", "pa"])
@pytest.mark.parametrize("order_type", ["MKT", "LMT"])
@pytest.mark.parametrize("operation", ["close_only", "close_resize", "flatten"])
def test_all_close_modes_on_each_asset_account(tmp_path, asset, account, order_type, operation):
    held = position(asset, account.upper())
    broker = SimBroker([held], status="Filled" if order_type == "MKT" else "Submitted", fill=100 if order_type == "MKT" else 0)
    executor = bind(load_executor(), broker, tmp_path)
    p = close_payload(held, order_type)
    args = (broker, p) if operation == "close_only" else (broker, p, "fixture", 0, 7) if operation == "flatten" else (broker, p, account, "fixture", 0, 7)
    result = invoke(executor, "_do_" + operation, *args)
    assert result["ok"], result
    assert len(broker.sent) == 1
    assert broker.sent[0][1].account == account.upper()
    assert broker.sent[0][1].orderType == order_type


@pytest.mark.parametrize("other_only", [True, False])
def test_flatten_never_uses_another_accounts_same_contract(tmp_path, other_only):
    own, other = position(), position(account="PA", quantity=900)
    broker = SimBroker([other] if other_only else [other, own], status="Filled", fill=100)
    result = invoke(bind(load_executor(), broker, tmp_path), "_do_flatten", broker, close_payload(own), "fixture", 0, 7)
    if other_only:
        assert result["state"] == "rejected" and not broker.sent
    else:
        assert result["ok"], result
        assert broker.sent[0][1].account == "PRIMARY"
        assert next(p for p in broker.holdings if p.account == "PA").position == 900


def test_full_flatten_includes_pending_entry_fill_during_cancel(tmp_path):
    held = position()
    broker = SimBroker([held], status="Filled", fill=200)
    executor = bind(load_executor(), broker, tmp_path)
    pending = LimitOrder("BUY", 20, 99, orderId=8, permId=108, clientId=7, account="PRIMARY", tif="DAY")
    from ib_insync import Trade, OrderStatus
    broker.trades.append(Trade(held.contract, pending, OrderStatus(status="Submitted", filled=0, remaining=20)))
    def entry_fill(ib, trade):
        ib.holdings[0] = held._replace(position=110)
        trade.orderStatus.filled = 10
    broker.on_cancel = entry_fill
    result = invoke(executor, "_do_flatten", broker, close_payload(held), "fixture", 0, 7)
    assert result["ok"], result
    assert broker.cancelled == [8]
    assert broker.sent[0][1].totalQuantity == 110
    assert broker.holdings[0].position == 0


@pytest.mark.parametrize("restored_status", ["PendingSubmit", "Inactive"])
def test_partial_flatten_never_claims_unacknowledged_restored_exits(tmp_path, restored_status):
    from ib_insync import Trade, OrderStatus
    held = position()
    held.contract.exchange = "SMART"
    broker = SimBroker([held])
    stop = StopOrder("SELL", 100, 90, orderId=7, permId=107, clientId=7, account="PRIMARY", tif="GTC")
    broker.trades.append(Trade(held.contract, stop, OrderStatus(status="Submitted", filled=0, remaining=100)))
    executor = bind(load_executor(), broker, tmp_path)
    submit = broker.place
    def place(ib, contract, order, **kwargs):
        broker.status, broker.fill = ("Filled", 40) if order.orderType == "MKT" else (restored_status, 0)
        return submit(ib, contract, order, **kwargs)
    executor.guarded_place_order = place
    result = invoke(executor, "_do_flatten", broker, dict(close_payload(held), qty=40), "fixture", 0, 7)
    assert result["state"] == "unknown", result
    assert "re-attach FAILED" in result["detail"]
    assert len(broker.sent) == 2


@pytest.mark.parametrize("operation", ["close_only", "flatten"])
def test_terminal_partial_limit_fill_is_not_a_clean_rejection(tmp_path, operation):
    held = position()
    broker = SimBroker([held], status="Cancelled", fill=10)
    executor = bind(load_executor(), broker, tmp_path)
    args = (broker, close_payload(held, "LMT"))
    if operation == "flatten": args += ("fixture", 0, 7)
    result = invoke(executor, "_do_" + operation, *args)
    assert result["state"] == "unknown", result
    assert result["fill"]["filled"] == 10


def test_partial_flatten_does_not_restore_wrong_direction_after_concurrent_fill(tmp_path):
    from ib_insync import Trade, OrderStatus
    held = position()
    held.contract.exchange = "SMART"
    broker = SimBroker([held], status="Filled", fill=40)
    stop = StopOrder("SELL", 100, 90, orderId=7, permId=107, clientId=7, account="PRIMARY", tif="GTC")
    broker.trades.append(Trade(held.contract, stop, OrderStatus(status="Submitted", filled=0, remaining=100)))
    executor = bind(load_executor(), broker, tmp_path)
    submit = broker.place
    def place(*args, **kwargs):
        result = submit(*args, **kwargs)
        broker.holdings[0] = held._replace(position=-10)
        return result
    executor.guarded_place_order = place
    result = invoke(executor, "_do_flatten", broker, dict(close_payload(held), qty=40), "fixture", 0, 7)
    assert result["state"] == "unknown" and "direction changed" in result["detail"]
    assert len(broker.sent) == 1  # the close only; no SELL exit against the new short


@pytest.mark.parametrize("operation", ["entry_bracket", "exit_attach", "close_only", "flatten", "option_spread"])
def test_pending_ack_is_never_reported_as_executed(tmp_path, operation):
    held = position()
    broker = SimBroker([held], status="PendingSubmit")
    executor = bind(load_executor(), broker, tmp_path)
    if operation == "entry_bracket": args = (broker, entry(), "primary")
    elif operation == "exit_attach": args = (broker, dict(close_payload(held), stop=90), "primary")
    elif operation == "option_spread": args = (broker, option_payload(), "primary")
    elif operation == "close_only": args = (broker, close_payload(held, "LMT"))
    else: args = (broker, close_payload(held, "LMT"), "fixture", 0, 7)
    result = invoke(executor, "_do_" + operation, *args)
    assert result["state"] == "unknown", result
    assert broker.sent


def option_payload(right="C", vertical=False, action="BUY"):
    low, high = 100, 110
    buy_strike, sell_strike = (low, high) if right == "C" else (high, low)
    legs = [dict(side="BUY", right=right, expiry="20301220", strike=buy_strike, ratio=1)]
    if vertical: legs.append(dict(side="SELL", right=right, expiry="20301220", strike=sell_strike, ratio=1))
    return dict(_command_id="option", _broker_account="PRIMARY", symbol="TEST", action=action,
                quantity=2, limit=2, debit_risk=8 if action == "SELL" else 2, tif="DAY", legs=legs)


@pytest.mark.parametrize("right", ["C", "P"])
@pytest.mark.parametrize("vertical,action", [(False, "BUY"), (True, "BUY"), (True, "SELL")])
def test_single_options_and_debit_credit_verticals(tmp_path, right, vertical, action):
    broker = SimBroker()
    result = invoke(bind(load_executor(), broker, tmp_path), "_do_option_spread", broker, option_payload(right, vertical, action), "primary")
    assert result["ok"], result
    contract, order, guard = broker.sent[0]
    assert contract.secType == ("BAG" if vertical else "OPT")
    assert order.orderType == "LMT" and order.action == action
    assert guard["risk_usd"] > 0


@pytest.mark.parametrize("operation,field,value", [
    ("entry_bracket", "quantity", float("nan")), ("entry_bracket", "stop", float("nan")),
    ("entry_bracket", "entry", float("inf")), ("entry_bracket", "expiry", "2030-02-30"),
    ("entry_bracket", "time_stop", "2000-01-01"),
    ("exit_attach", "stop", float("nan")), ("exit_attach", "target", float("inf")),
    ("exit_attach", "time_stop", "2030-02-30"),
    ("option_spread", "quantity", 1.5), ("option_spread", "limit", float("nan")),
])
def test_invalid_numeric_and_calendar_inputs_do_not_reach_broker(tmp_path, operation, field, value):
    held = position()
    broker = SimBroker([held])
    payload = entry() if operation == "entry_bracket" else option_payload() if operation == "option_spread" else dict(close_payload(held), stop=90)
    payload[field] = value
    result = invoke(bind(load_executor(), broker, tmp_path), "_do_" + operation, broker, payload, "primary")
    assert result["state"] == "rejected", result
    assert not broker.sent


def test_delayed_parent_and_child_acknowledgements_wait_without_resending(tmp_path):
    broker = SimBroker(status="PendingSubmit")
    def acknowledge(ib):
        if ib.sleep_count >= 3:
            for trade in ib.trades: trade.orderStatus.status = "Submitted"
    broker.on_sleep = acknowledge
    result = invoke(bind(load_executor(), broker, tmp_path), "_do_entry_bracket", broker, entry(), "primary")
    assert result["ok"], result
    assert len(broker.sent) == 3


@pytest.mark.parametrize("account", ["primary", "pa"])
@pytest.mark.parametrize("short", [False, True])
@pytest.mark.parametrize("readd", [False, True])
def test_add_and_readd_with_real_parent_and_child_orders(tmp_path, account, short, readd):
    from ib_insync import Trade, OrderStatus
    held = position(account=account.upper(), quantity=-100 if short else 100)
    held.contract.exchange = "SMART"
    broker = SimBroker([held])
    exit_side = "BUY" if short else "SELL"
    stop = StopOrder(exit_side, 100, 110 if short else 90, orderId=7, permId=107, clientId=7,
                     account=account.upper(), tif="GTC")
    broker.trades.append(Trade(held.contract, stop, OrderStatus(status="Submitted", filled=0, remaining=100)))
    executor = bind(load_executor(), broker, tmp_path)
    submit = broker.place
    def place(ib, contract, order, **kwargs):
        closes = readd and order.orderType == "MKT" and not order.parentId and order.action == exit_side
        broker.status, broker.fill = ("Filled", 40) if closes else ("Submitted", 0)
        return submit(ib, contract, order, **kwargs)
    executor.guarded_place_order = place
    p = dict(close_payload(held), qty=40, readd=readd)
    kind = "_do_close_resize" if readd else "_do_add_to_position"
    result = invoke(executor, kind, broker, p, account, "fixture", 0, 7)
    assert result["ok"], result
    parents = [o for _, o, _ in broker.sent if not o.parentId and o.action != exit_side]
    assert len(parents) == 1 and parents[0].totalQuantity == 40
    assert parents[0].orderType == ("LMT" if readd else "MKT")
    assert parents[0].tif == "DAY" and parents[0].transmit is False
    children = [o for _, o, _ in broker.sent if o.parentId == parents[0].orderId]
    assert children and children[-1].transmit and all(o.totalQuantity == 40 for o in children)


@pytest.mark.parametrize("account", ["primary", "pa"])
@pytest.mark.parametrize("asset", ["STK", "FUT", "CASH"])
@pytest.mark.parametrize("operation", ["modify", "cancel"])
@pytest.mark.parametrize("order_type", ["LMT", "STP", "STP LMT", "MKT"])
def test_modify_cancel_native_order_variants(tmp_path, account, asset, operation, order_type):
    from ib_insync import Trade, OrderStatus, MarketOrder, StopLimitOrder
    held = position(asset, account.upper())
    held.contract.exchange = "SMART" if asset == "STK" else "CME" if asset == "FUT" else "IDEALPRO"
    broker = SimBroker([held])
    order = (LimitOrder("SELL", 100, 120) if order_type == "LMT" else StopOrder("SELL", 100, 90) if order_type == "STP" else
             StopLimitOrder("SELL", 100, 89, 90) if order_type == "STP LMT" else MarketOrder("SELL", 100))
    order.orderId, order.permId, order.clientId, order.account, order.tif = 7, 107, 7, account.upper(), "GTC"
    if order_type == "MKT": order.goodAfterTime = "20301220 15:59:00 US/Eastern"
    broker.trades.append(Trade(held.contract, order, OrderStatus(status="Submitted", filled=0, remaining=100)))
    executor = bind(load_executor(), broker, tmp_path)
    payload = dict(close_payload(held), order_id=7, perm_id=107, client_id=7, new_qty=70)
    args = (broker, payload, "fixture", 0, 7, account) if operation == "modify" else (broker, payload, "fixture", 0, 7)
    result = invoke(executor, "_do_" + operation, *args)
    assert result["ok"], result
    assert len(broker.sent) == (1 if operation == "modify" else 0)
    assert len(broker.cancelled) == (1 if operation == "cancel" else 0)
    if operation == "modify":
        assert broker.sent[0][1].totalQuantity == 70
        assert broker.sent[0][1].goodAfterTime == order.goodAfterTime


def test_candidate_fixture_matches_current_source_when_available():
    from broker_runtime.prepare_execution_repairs import patch_executor, patch_agent
    root = Path(os.environ.get("IBKR_REVIEW_SOURCE", ""))
    if not (root / "execute_order.py").exists():
        return  # the portable fixtures above still execute all behavior coverage
    for name, patch in (("execute_order", patch_executor), ("exec_agent", patch_agent)):
        raw = (root / (name + ".py")).read_text(encoding="utf-8-sig")
        rendered = patch(raw)
        expected = {n.name: ast.dump(n) for n in ast.parse(rendered).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        for node in ast.parse((FIXTURE / (name + "_core.py")).read_text(encoding="utf-8")).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assert ast.dump(node) == expected[node.name]
