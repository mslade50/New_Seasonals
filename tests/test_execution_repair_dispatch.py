"""Exercise the live-shaped dispatcher, without importing or connecting a broker."""
import ast
import copy
import json
import os
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace as N

import pytest

from broker_runtime import prepare_execution_repairs as prepare
from broker_runtime import position_actions as actions
from broker_runtime import position_action_agent as agent
from broker_runtime import order_mutations
from broker_runtime import execution_lifecycle as life
from broker_runtime.execution_contracts import qualify_held, qualify_position
from tests.test_unified_position_actions import Broker, exit_order, namespace, request

SOURCE = Path(os.environ.get("IBKR_REVIEW_SOURCE", "C:/Users/McKinley Slade/OneDrive/trading_ibkr"))


@pytest.fixture
def patched():
    if not (SOURCE / "execute_order.py").exists():
        pytest.skip("reviewed runtime source is required")
    return prepare.patch_executor((SOURCE / "execute_order.py").read_text(encoding="utf-8-sig"))


@pytest.mark.parametrize("account", ["primary", "pa"])
@pytest.mark.parametrize("typ", ["entry_bracket", "exit_attach", "close_only", "close_resize",
                                 "flatten", "option_spread", "add_to_position", "modify", "cancel"])
def test_every_supported_command_reaches_its_handler(patched, account, typ):
    tree = ast.parse(patched)
    nodes = [n for n in tree.body if
             (isinstance(n, ast.FunctionDef) and n.name == "main") or
             (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and
              t.id in {"SUPPORTED", "DISABLED_UNSAFE_MUTATIONS"} for t in n.targets))]
    calls = []
    class Event:
        def __iadd__(self, fn): return self
    broker = N(connect=lambda *a, **k: None, disconnect=lambda: None, errorEvent=Event())
    env = dict(json=json, sys=N(argv=["fixture", json.dumps(dict(
        id="fixture", type=typ, account=account, payload={}))]),
        LIVE_ENABLED=True, LIVE_ACCOUNTS={"primary", "pa"}, LIVE_TYPES={typ},
        PORTS={account: ("fixture", 0, 7)}, IB=lambda: broker,
        _out=lambda *a, **k: (a,k), _resolve_broker_account=lambda *a: account,
        _on_err=lambda *a: None)
    for name in ["entry_bracket", "exit_attach", "close_only", "close_resize", "flatten",
                 "option_spread", "add_to_position", "modify", "cancel"]:
        env["_do_"+name] = lambda *a, name=name: calls.append(name)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "dispatcher", "exec"), env)
    env["main"]()
    assert calls == [typ]
    calls.clear()
    env["LIVE_ENABLED"] = False
    env["main"]()
    assert calls == []


def test_resize_accepts_fresh_ack_even_if_returned_trade_stays_pending(tmp_path):
    broker = Broker(exits=[exit_order(1, 100)])
    ns = namespace(tmp_path, broker)
    original = ns["guarded_place_order"]
    def place(*args, **kwargs):
        result = original(*args, **kwargs)
        if result is broker.orders[0]:
            stale = copy.deepcopy(result)
            stale.orderStatus.status = "PendingSubmit"
            return stale
        return result
    ns["guarded_place_order"] = place
    result = actions.run(ns, broker, request(), "primary", "", 0, 7)
    assert result["ok"] and broker.position.position == 60
    assert len(broker.mutations) == 2


def test_delayed_resize_waits_without_resending(tmp_path):
    broker = Broker(exits=[exit_order(1, 100)])
    ns = namespace(tmp_path, broker)
    original = broker.place
    polls = []
    def place(ib, c, o, **kwargs):
        result = original(ib, c, o, **kwargs)
        if o.orderId == 1:
            result.orderStatus.status = "PendingSubmit"
        return result
    def sleep(_):
        polls.append(1)
        if len(polls) >= 8:
            broker.orders[0].orderStatus.status = "Submitted"
    broker.sleep = sleep
    ns["guarded_place_order"] = place
    result = actions.run(ns, broker, request(), "primary", "", 0, 7)
    assert result["ok"], result
    assert sum(m[1] == 1 for m in broker.mutations) == 1
    assert broker.position.position == 60


def test_fill_during_resize_wait_aborts_before_close(tmp_path):
    broker = Broker(exits=[exit_order(1, 100)])
    ns = namespace(tmp_path, broker)
    original = broker.place
    def place(*a, **kw):
        result = original(*a, **kw)
        result.orderStatus.filled = 5
        return result
    ns["guarded_place_order"] = place
    result = actions.run(ns, broker, request(), "primary", "", 0, 7)
    assert result["state"] == "unknown" and broker.last_close is None


def test_missing_fresh_fill_counters_reject_before_any_mutation(tmp_path):
    broker = Broker(exits=[exit_order(1, 100)])
    ns = namespace(tmp_path, broker)
    original = ns["_orders_for_contract"]
    def snapshot(*a):
        rows = copy.deepcopy(original(*a))
        for t in rows:
            t.orderStatus = N(status="Submitted")
        return rows
    ns["_orders_for_contract"] = snapshot
    result = actions.run(ns, broker, request(), "primary", "", 0, 7)
    assert result["state"] == "rejected" and not broker.mutations
    assert "fill quantities" in result["detail"]


def test_pa_close_uses_same_lifecycle_with_exact_account(tmp_path):
    broker = Broker(exits=[exit_order(1, 100, account="PA")])
    broker.position.account = "PA"
    payload = dict(request(), _broker_account="PA")
    assert agent.applies(dict(account="pa",type="close_resize"))
    result = actions.run(namespace(tmp_path, broker), broker, payload, "pa", "", 0, 7)
    assert result["ok"] and broker.position.position == 60


def test_contract_qualification_preserves_conid_and_hydrates_exchange():
    c = N(conId=42,secType="FUT",exchange="",lastTradeDateOrContractMonth="202609")
    ib = N(qualifyContracts=lambda c: [N(**dict(vars(c),exchange="CME"))])
    assert qualify_held(ib,c).exchange == "CME"
    assert c.exchange == ""
    ib.qualifyContracts = lambda c: [N(**dict(vars(c),conId=43))]
    with pytest.raises(ValueError, match="changed"):
        qualify_held(ib,c)


def test_qualify_immutable_ibkr_position_keeps_quantity_and_account():
    Position = namedtuple("Position", "account contract position avgCost")
    original = Position("PRIMARY", N(conId=42,secType="FUT",exchange=""), 2, 500000)
    ib = N(qualifyContracts=lambda c:[N(**dict(vars(c),exchange="CME"))])
    result = qualify_position(ib, original)
    assert result.contract.exchange == "CME" and original.contract.exchange == ""
    assert result.account == original.account and result.position == 2 and result.avgCost == 500000


def test_order_edit_receipt_blocks_duplicate_after_restart(tmp_path, monkeypatch):
    broker = Broker(exits=[exit_order(1,100)])
    ns = namespace(tmp_path,broker)
    payload = dict(request(),client_id=7,order_id=1,perm_id=101)
    calls=[]
    monkeypatch.setattr(order_mutations.life,"mutate_one",lambda *a,**k:
                        (calls.append(1) or dict(ok=True,state="executed",detail="done")))
    assert order_mutations.run(ns,broker,payload,"",0,7)["ok"]
    assert order_mutations.run(ns,broker,payload,"",0,7)["ok"]
    assert calls == [1]


def edit_fixture(tmp_path):
    broker = Broker(exits=[exit_order(1, 80)])
    ns = namespace(tmp_path, broker)
    ns.update(LIVE_MAX_QTY=1000, LIVE_MAX_FUT_CONTRACTS=20,
              _uncapped_futures=lambda _: False,
              _px=lambda p: p if 0 < p < 1e10 else 0)
    reader = ns["_orders_for_contract"]
    ns["_orders_for_contract"] = lambda *a: copy.deepcopy(reader(*a))
    p = dict(request(), client_id=7, order_id=1, perm_id=101, mutation_kind="exit")
    return broker, ns, p


def test_cancel_observes_native_ack_with_detached_raw_snapshot(tmp_path):
    broker, ns, p = edit_fixture(tmp_path)
    result = life.mutate_one(ns, broker, p, "", 0, 7)
    assert result["ok"] and result["fill"]["status"] == "Cancelled"
    assert len(broker.mutations) == 1


def test_modify_preserves_raw_timing_and_outside_rth(tmp_path):
    broker, ns, p = edit_fixture(tmp_path)
    reader = ns["_orders_for_contract"]
    def fresh(*args):
        rows = reader(*args)
        rows[0].order.goodAfterTime = "20260925 15:59:00 America/New_York"
        rows[0].order.outsideRth = True
        return rows
    ns["_orders_for_contract"] = fresh
    result = life.mutate_one(ns, broker, dict(p, new_stop=91), "", 0, 7, modify=True)
    assert result["ok"], result
    assert broker.orders[0].order.goodAfterTime == "20260925 15:59:00 America/New_York"
    assert broker.orders[0].order.outsideRth is True


@pytest.mark.parametrize("parent", [0, 99])
def test_exit_increase_cannot_overclose_even_for_attached_child(tmp_path, parent):
    broker, ns, p = edit_fixture(tmp_path)
    broker.orders[0].order.parentId = parent
    result = life.mutate_one(ns, broker, dict(p, new_qty=101), "", 0, 7, modify=True)
    assert result["state"] == "rejected" and not broker.mutations
    assert "inventory" in result["detail"]


def test_futures_increase_includes_contract_multiplier(tmp_path):
    broker, ns, p = edit_fixture(tmp_path)
    broker.orders[0].contract.secType = "FUT"
    broker.orders[0].contract.multiplier = "1000"
    broker.orders[0].order.totalQuantity = 1
    result = life.mutate_one(ns, broker, dict(p, new_qty=2), "", 0, 7, modify=True)
    assert result["state"] == "rejected" and not broker.mutations
    assert "notional cap" in result["detail"]


def test_ambiguous_edit_blocks_position_action(tmp_path):
    broker, ns, p = edit_fixture(tmp_path)
    broker.fail_resize = True
    result = order_mutations.run(ns, broker, dict(p, new_stop=91), "", 0, 7, modify=True)
    assert result["state"] == "unknown"
    result = actions.run(ns, broker, dict(request(), _command_id="next"), "primary", "", 0, 7)
    assert result["state"] == "rejected" and "earlier order edit" in result["detail"]
    assert len(broker.mutations) == 1


@pytest.mark.parametrize("account", ["primary", "pa"])
@pytest.mark.parametrize("typ", ["cancel", "modify", "add_to_position", "scheduled_option"])
def test_agent_gate_dispatch_and_legacy_intent_rejection(account, typ):
    path = SOURCE / "exec_agent.py"
    if not path.exists():
        pytest.skip("reviewed agent source required")
    source = prepare.patch_agent(path.read_text(encoding="utf-8-sig"))
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_live_eligible")
    env = dict(LIVE_ENABLED=True, LIVE_ACCOUNTS={"primary", "pa"},
               LIVE_TYPES={"cancel", "modify", "add_to_position", "option_spread"},
               DISABLED_UNSAFE_MUTATIONS={"cancel", "modify", "add_to_position", "trim_readd"})
    exec(compile(ast.Module(body=[node], type_ignores=[]), "agent-gate", "exec"), env)
    cmd = dict(account=account, type=typ, payload=dict(order_type="LMT", pricing_policy="capped_limit_v1"))
    assert env["_live_eligible"](cmd)[0]
    if typ == "scheduled_option":
        cmd["payload"]["order_type"] = "MKT"
        assert not env["_live_eligible"](cmd)[0]
    env["LIVE_ENABLED"] = False
    assert not env["_live_eligible"](cmd)[0]
