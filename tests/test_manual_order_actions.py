"""Manual actions use fake broker wires only; no live executor is imported."""
import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from broker_runtime import manual_order_actions as manual
from broker_runtime import prepare_manual_order_actions as prepare


def trade(account="PRIMARY", **overrides):
    order = dict(account=account, clientId=99, orderId=7, permId=701,
                 totalQuantity=100, action="SELL", orderType="STP LMT", lmtPrice=90,
                 auxPrice=95, transmit=True, parentId=42, tif="GTC", ocaGroup="mixed",
                 ocaType=1, goodAfterTime="20261001 15:59:00 US/Eastern",
                 orderRef="Legend|strategy", outsideRth=False)
    order.update(overrides)
    return NS(contract=NS(conId=42, symbol="SPY", secType="STK"), order=NS(**order),
              orderStatus=NS(status="Submitted", filled=0))


class Broker:
    def __init__(self, orders):
        self.orders, self.wires = orders, []
        self.fail_after_send = False
        self.acknowledge = True
        self.fill_on_cancel = False
        self.owner = None
        self.disconnected = False
    def reqAllOpenOrders(self): pass
    def sleep(self, _): pass
    def openTrades(self): return self.orders
    def connect(self, host, port, clientId, timeout): self.owner = clientId
    def disconnect(self): self.disconnected = True
    def cancelOrder(self, order):
        self.wires.append(("cancel", copy.deepcopy(order)))
        if self.fail_after_send: raise TimeoutError("lost acknowledgement")
        if self.acknowledge:
            target = next(t for t in self.orders if t.order.permId == order.permId)
            target.orderStatus.status = "Filled" if self.fill_on_cancel else "Cancelled"
            if self.fill_on_cancel: target.orderStatus.filled = order.totalQuantity
    def placeOrder(self, contract, order):
        self.wires.append(("modify", copy.deepcopy(order)))
        if self.fail_after_send: raise TimeoutError("lost acknowledgement")
        if self.acknowledge:
            next(t for t in self.orders if t.order.permId == order.permId).order = order


def forbidden(*a, **k):
    pytest.fail("manual action consulted trading policy")


@pytest.fixture
def setup(tmp_path):
    def build(account="PRIMARY", **overrides):
        selected = trade(account, **overrides)
        ib = Broker([selected])
        ns = dict(__file__=str(tmp_path / "execute_order.py"), _out=lambda **kw: kw,
                  guarded_cancel_order=forbidden, guarded_place_order=forbidden,
                  LIVE_MAX_QTY=1, _max_notional=forbidden, _cluster_symbol=forbidden,
                  _exact_position=forbidden, _capture_exit_legs=forbidden,
                  _orders_for_contract=lambda *a: ib.orders)
        payload = dict(_broker_account=account, _command_id="test-command",
                       con_id=42, client_id=99, order_id=7, perm_id=701)
        return ns, ib, payload
    return build


@pytest.mark.parametrize("account", ["PRIMARY", "PA"])
@pytest.mark.parametrize("modify", [False, True])
def test_manual_action_ignores_strategy_risk_inventory_and_other_journals(setup, tmp_path, account, modify):
    ns, ib, payload = setup(account)
    root = tmp_path / "data" / "position_actions"
    root.mkdir(parents=True)
    prior = dict(version=1, id="automatic", phase="pending", payload=dict(payload))
    (root / "pending.json").write_text(json.dumps(prior))
    (root / "corrupt.json").write_text("corrupt legacy state")
    other = dict(prior, payload=dict(payload, _broker_account="OTHER"))
    (root / "other.json").write_text(json.dumps(other))
    edits = root / "order_edits"
    edits.mkdir()
    (edits / "old.json").write_text(json.dumps(dict(phase="attention", identity=list(manual.life.payload_identity(payload)))))
    payload.update(new_qty=100_000, new_limit=-2.5)
    result = manual.run(ns, ib, payload, "fixture", 0, 99, modify=modify)
    assert result["state"] == "executed" and result["ok"]
    assert len(ib.wires) == 1
    assert json.loads((root / "pending.json").read_text())["phase"] == "attention"
    assert json.loads((root / "other.json").read_text()) == other
    if modify:
        sent = ib.wires[0][1]
        assert sent.totalQuantity == 100_000 and sent.lmtPrice == -2.5
        assert sent.orderRef == "Legend|strategy" and sent.ocaGroup == "mixed"


def test_only_requested_fields_change_and_held_parent_stays_held(setup):
    ns, ib, payload = setup(transmit=False)
    before = vars(ib.orders[0].order).copy()
    result = manual.run(ns, ib, dict(payload, new_stop=105), "fixture", 0, 99, modify=True)
    assert result["ok"]
    assert vars(ib.wires[0][1]) == dict(before, auxPrice=105)


@pytest.mark.parametrize("field,value", [("_broker_account", "OTHER"), ("con_id", 43), ("client_id", 98), ("order_id", 8), ("perm_id", 702)])
def test_never_routes_to_a_different_order(setup, field, value):
    ns, ib, payload = setup()
    payload[field] = value
    result = manual.run(ns, ib, payload, "fixture", 0, 99)
    assert result["state"] == "rejected" and not ib.wires


def test_resolves_placing_client_and_closes_auxiliary_connection(setup):
    ns, main, payload = setup()
    owner = Broker(main.orders)
    ns["IB"] = lambda: owner
    result = manual.run(ns, main, payload, "fixture", 0, 123)
    assert result["ok"] and not main.wires
    assert owner.owner == 99 and owner.disconnected and len(owner.wires) == 1


@pytest.mark.parametrize("modify", [False, True])
@pytest.mark.parametrize("failure", ["fail_after_send", "no_ack"])
def test_unknown_delivery_is_not_retried(setup, modify, failure):
    ns, ib, payload = setup()
    if failure == "fail_after_send": ib.fail_after_send = True
    else: ib.acknowledge = False
    payload["new_qty"] = 200
    result = manual.run(ns, ib, payload, "fixture", 0, 99, modify=modify)
    assert result["state"] == "unknown"
    replay = manual.run(ns, ib, payload, "fixture", 0, 99, modify=modify)
    assert replay["state"] == "unknown" and len(ib.wires) == 1


@pytest.mark.parametrize("modify", [False, True])
def test_same_command_replays_completed_receipt(setup, modify):
    ns, ib, payload = setup()
    payload["new_qty"] = 200
    result = manual.run(ns, ib, payload, "fixture", 0, 99, modify=modify)
    assert result["ok"]
    assert manual.run(ns, ib, payload, "fixture", 0, 99, modify=modify) == result
    assert len(ib.wires) == 1


def test_fill_racing_cancel_is_not_reported_cancelled(setup):
    ns, ib, payload = setup()
    ib.fill_on_cancel = True
    result = manual.run(ns, ib, payload, "fixture", 0, 99)
    assert result["state"] == "unknown" and "Filled" in result["detail"]


def test_new_manual_cancel_is_allowed_after_an_uncertain_modify(setup):
    ns, ib, payload = setup()
    ib.fail_after_send = True
    result = manual.run(ns, ib, dict(payload, new_qty=120), "fixture", 0, 99, modify=True)
    assert result["state"] == "unknown"
    ib.fail_after_send = False
    result = manual.run(ns, ib, dict(payload, _command_id="new-explicit-cancel"), "fixture", 0, 99)
    assert result["ok"] and len(ib.wires) == 2


@pytest.mark.parametrize("value", [float("inf"), float("nan"), "invalid"])
def test_unencodable_modify_never_transmits(setup, value):
    ns, ib, payload = setup()
    result = manual.run(ns, ib, dict(payload, new_qty=value), "fixture", 0, 99, modify=True)
    assert result["state"] == "rejected" and not ib.wires


@pytest.mark.parametrize("account", ["primary", "pa"])
def test_agent_checks_only_address_and_encoding(setup, account):
    _, _, payload = setup()
    payload.update(new_qty=1_000_000, new_limit=-.25)
    assert manual.validate(dict(type="modify", account=account, payload=payload)) == (True, [])
    assert manual.validate(dict(type="cancel", account=account, payload=payload)) == (True, [])


def test_candidate_patches_real_runtime_without_importing_broker(tmp_path):
    source = Path("C:/Users/McKinley Slade/OneDrive/trading_ibkr")
    if not source.exists(): pytest.skip("runtime source unavailable")
    target = tmp_path / "candidate"
    manifest = prepare.prepare(source, target)
    assert set(manifest["candidate"]) == {"manual_order_actions.py", "exec_agent.py", "execute_order.py"}
    before = ast.parse((source / "execute_order.py").read_text(encoding="utf-8-sig"))
    after = ast.parse((target / "execute_order.py").read_text())
    functions = lambda tree: {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    b, a = functions(before), functions(after)
    assert {name for name in b if b[name] != a[name]} == {"_do_cancel", "_do_modify"}
    agent = ast.parse((target / "exec_agent.py").read_text())
    node = next(n for n in agent.body if isinstance(n, ast.FunctionDef) and n.name == "_validate")
    ns = {"manual_order_actions": manual}
    import sys
    from unittest.mock import patch
    with patch.dict(sys.modules, manual_order_actions=manual):
        exec(compile(ast.Module(body=[node], type_ignores=[]), "isolated-agent-validator", "exec"), ns)
        payload = dict(con_id=42, client_id=99, order_id=7, perm_id=701)
        # No agent book, price, strategy, or risk globals exist in this namespace.
        for account in ("primary", "pa"):
            assert ns["_validate"](dict(type="cancel", account=account, payload=payload)) == (True, [])
