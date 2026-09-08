"""Exercise the reviewed broker functions without importing its live module."""
import ast
import copy
import math
import os
from pathlib import Path
from types import SimpleNamespace as N

import pytest

from broker_runtime import position_actions as actions
from broker_runtime import position_action_agent as agent
from broker_runtime import prepare_position_actions as prepare
from tests.test_unified_position_actions import Broker, exit_order, namespace, request


@pytest.fixture
def source():
    path = Path(os.environ.get("IBKR_REVIEW_SOURCE",
                "C:/Users/McKinley Slade/OneDrive/trading_ibkr")) / "execute_order.py"
    if not path.exists():
        pytest.skip("reviewed broker source is installed only on the trading host")
    return path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")


@pytest.fixture
def adapter(source, tmp_path, monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "position_actions", actions)
    broker = Broker(exits=[exit_order(1, 30, "a"), exit_order(2, 30, "a", "LMT"),
                           exit_order(3, 20, "b"), exit_order(4, 20, "b", "LMT")])
    env = namespace(tmp_path, broker)
    env.update(math=math, copy=copy, LIVE_MAX_QTY=1000,
               _px=lambda x: float(x) if x else None,
               _order_source_key=lambda o: str(o.permId),
               _placed_ids=lambda ts: [t.order.orderId for t in ts])
    names = {"_capture_exit_legs", "_fast_qty", "_leg_group_key", "_scaled_exit_legs",
             "_place_exit_legs", "_prepare_position_action_add"}
    nodes = [n for n in ast.parse(prepare.patch_executor(source)).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "reviewed-functions", "exec"), env)

    def order(side, qty, typ, price=None):
        return N(action=side, totalQuantity=qty, orderType=typ, lmtPrice=price, auxPrice=price,
                 parentId=0, transmit=True, tif="DAY", account="", orderRef="", ocaGroup="",
                 ocaType=0, goodAfterTime="", goodTillDate="", outsideRth=False)
    env.update(MarketOrder=lambda s,q: order(s,q,"MKT"),
               LimitOrder=lambda s,q,p: order(s,q,"LMT",p),
               StopOrder=lambda s,q,p: order(s,q,"STP",p))
    wires = []
    def place(ib, contract, order, **kwargs):
        wires.append((copy.deepcopy(order), kwargs))
        existing = next((t for t in broker.orders if t.order.orderId == order.orderId), None)
        if existing:
            existing.order = copy.deepcopy(order)
            return existing
        placed = N(contract=contract, order=order,
                   orderStatus=N(status="Submitted", filled=0))
        broker.orders.append(placed)
        return placed
    env["guarded_place_order"] = place
    return env, broker, wires


def test_real_preparer_accepts_mismatched_coverage_and_preserves_caps(adapter):
    env, broker, _ = adapter
    context, error = env["_prepare_position_action_add"](broker, request(), "primary", partial=False)
    assert error is None and context["qty"] == 40
    _, error = env["_prepare_position_action_add"](broker, dict(request(), qty=1001), "primary", partial=False)
    assert "LIVE_MAX_QTY" in error
    _, error = env["_prepare_position_action_add"](broker, dict(request(), expected_position=90), "primary", partial=False)
    assert "stale position" in error


def test_real_native_brackets_use_equal_parent_child_sizes_and_deferred_release(adapter, tmp_path):
    env, broker, wires = adapter
    context, error = env["_prepare_position_action_add"](broker, dict(request(), qty=21), "primary", partial=False)
    assert error is None
    context["risk_reference"] = 101
    record = dict(id="native", payload=request(), phase="pending")
    actions.stage_add(env, broker, context, 21, record, tmp_path, market=True)
    parents = [(o,k) for o,k in wires if k["mutation_kind"] == "entry"]
    assert [o.totalQuantity for o,_ in parents] == [13, 8]
    for parent, kwargs in parents:
        assert parent.transmit is False and parent.tif == "DAY"
        assert kwargs["risk_usd"] == 11 * parent.totalQuantity
        children = [(o,k) for o,k in wires if o.parentId == parent.orderId]
        assert len(children) == 3  # two staged siblings, then last sibling released
        assert all(o.totalQuantity == parent.totalQuantity for o,_ in children)
        assert [o.transmit for o,_ in children] == [False, False, True]
        assert len({o.ocaGroup for o,_ in children}) == 1
        assert children[0][0].auxPrice == 90 and children[1][0].lmtPrice == 120
        assert all(o.account == "PRIMARY" and o.tif == "GTC" for o,_ in children)
    assert not set(o.ocaGroup for o,_ in wires if o.parentId) & {"a", "b"}


def test_uncertain_native_release_stops_before_next_allocation(adapter, tmp_path):
    env, broker, wires = adapter
    context, _ = env["_prepare_position_action_add"](broker, request(), "primary", partial=False)
    context["risk_reference"] = 101
    original = env["guarded_place_order"]
    def uncertain(ib, c, o, **kwargs):
        result = original(ib, c, o, **kwargs)
        if o.parentId and o.transmit:
            raise TimeoutError("lost release response")
        return result
    env["guarded_place_order"] = uncertain
    record = dict(id="unknown", payload=request(), phase="pending")
    with pytest.raises(ValueError, match="uncertain"):
        actions.stage_add(env, broker, context, 40, record, tmp_path, market=True)
    assert sum(k["mutation_kind"] == "entry" for _,k in wires) == 1
    assert record["phase"] == "mutating"


def test_candidate_keeps_original_pa_functions_and_only_exempts_primary_add(source):
    candidate = prepare.patch_executor(source)
    for name in ("_do_close_resize", "_do_add_to_position"):
        original = prepare.function_source(source, name)
        legacy = prepare.function_source(candidate, "_legacy" + name)
        assert legacy == original.replace("def " + name + "(", "def _legacy" + name + "(", 1)
    assert 'acct == "primary" and t == "add_to_position"' in candidate
    assert prepare.function_source(candidate, "_prepare_fast_position") == prepare.function_source(source, "_prepare_fast_position")
    compile(candidate, "candidate", "exec")


def test_agent_patcher_preserves_live_gates(source, monkeypatch):
    path = Path(os.environ.get("IBKR_REVIEW_SOURCE",
                "C:/Users/McKinley Slade/OneDrive/trading_ibkr")) / "exec_agent.py"
    candidate = prepare.patch_agent(path.read_text(encoding="utf-8-sig").replace("\r\n", "\n"))
    nodes = [n for n in ast.parse(candidate).body if isinstance(n, ast.FunctionDef)
             and n.name in {"_validate", "_live_eligible"}]
    env = dict(LIVE_ENABLED=True, LIVE_ACCOUNTS={"primary", "pa"},
               LIVE_TYPES={"add_to_position", "close_resize", "cancel"},
               DISABLED_UNSAFE_MUTATIONS={"add_to_position", "cancel", "modify", "trim_readd"})
    monkeypatch.setitem(__import__("sys").modules, "position_action_agent", agent)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "agent-functions", "exec"), env)
    cmd = dict(account="primary", type="add_to_position", payload=request())
    assert env["_validate"](cmd)[0] and env["_live_eligible"](cmd)[0]
    assert not env["_live_eligible"](dict(cmd, account="pa"))[0]
    assert not env["_live_eligible"](dict(cmd, type="cancel"))[0]
    env["LIVE_ENABLED"] = False
    assert not env["_live_eligible"](cmd)[0]
    compile(candidate, "agent-candidate", "exec")
