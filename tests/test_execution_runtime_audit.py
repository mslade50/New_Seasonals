"""Regressions from the live-runtime audit; no executor import or broker network."""
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as N

import pytest

from broker_runtime import execution_lifecycle as life
from broker_runtime import position_actions as actions
from tests.test_unified_position_actions import Broker, exit_order, namespace, request


def test_add_qualifies_exact_held_contract_before_staging(tmp_path, monkeypatch):
    broker = Broker(exits=[exit_order(1, 100)])
    broker.position.contract.exchange = ""
    calls = []

    def stage(ns, ib, context, qty, signal, market=False):
        assert context["ref"].exchange == "SMART"
        assert context["ref"].conId == 42
        assert broker.position.contract.exchange == ""  # do not mutate broker cache
        calls.append(qty)
        return {"parent": [{"order_id": 401}], "children": []}, None

    monkeypatch.setattr(life, "stage_attached", stage)
    result = actions.run(namespace(tmp_path, broker), broker, request(), "primary", "", 0, 7, adding=True)
    assert result["ok"], result
    assert calls == [40]


@pytest.mark.parametrize("adding", [False, True])
def test_bad_contract_qualification_stops_before_exit_changes(tmp_path, adding):
    broker = Broker(exits=[exit_order(1, 80)])
    broker.qualifyContracts = lambda c: [N(conId=999, secType="STK", exchange="SMART")]
    result = actions.run(namespace(tmp_path, broker), broker, request(), "primary", "", 0, 7, adding=adding)
    assert result["state"] == "rejected", result
    assert "qualification" in result["detail"]
    assert not broker.mutations


@pytest.mark.parametrize("status", ["PendingSubmit", "ApiPending", ""])
def test_add_requires_child_acknowledgement_even_when_parent_is_submitted(status):
    parent = N(order=N(orderId=1), orderStatus=N(status="Submitted", filled=0))
    child = N(order=N(orderId=2), orderStatus=N(status=status, filled=0))
    broker = N(client=N(getReqId=lambda: 1), sleep=lambda _: None)
    ns = {
        "MarketOrder": lambda *a: N(), "LimitOrder": lambda *a: N(),
        "guarded_place_order": lambda *a, **k: parent,
        "_fast_guard_risk_usd": lambda *a: 100,
        "_place_exit_legs": lambda *a, **k: ([child], None),
        # The actual legacy helper accepts PendingSubmit; this must not suffice.
        "_placement_problem": lambda _: None,
        "_placed_ids": lambda rows: [{"order_id": t.order.orderId} for t in rows],
    }
    context = dict(entry_action="BUY", close_action="SELL", account="PRIMARY", ref=N(), legs=[{}], avg_cost=100)
    result, error = life.stage_attached(ns, broker, context, 10, "fixture", market=True)
    assert error and "acknowledgement" in error
    assert result["parent"] == [{"order_id": 1}]
    assert result["children"] == [{"order_id": 2}]


@pytest.mark.parametrize("missing", [False, True])
def test_attached_child_callback_can_arrive_late_without_a_resubmission(missing):
    parent = N(order=N(orderId=1), orderStatus=N(status="Submitted", filled=0))
    child = N(order=N(orderId=2), orderStatus=N(status="PendingSubmit", filled=0))
    events = []

    def sleep(_):
        events.append("wait")
        child.orderStatus.status = "PreSubmitted"

    def place(*a, **k):
        events.append("parent")
        return parent

    def attach(*a, **k):
        events.append("children")
        return ([] if missing else [child]), None

    broker = N(client=N(getReqId=lambda: 1), sleep=sleep)
    ns = {"MarketOrder": lambda *a: N(), "guarded_place_order": place,
          "_fast_guard_risk_usd": lambda *a: 100, "_place_exit_legs": attach,
          "_placement_problem": lambda _: None,
          "_placed_ids": lambda rows: [{"order_id": t.order.orderId} for t in rows]}
    context = dict(entry_action="BUY", close_action="SELL", account="PRIMARY", ref=N(), legs=[{}], avg_cost=100)
    result, error = life.stage_attached(ns, broker, context, 10, "fixture", market=True)
    assert events.count("parent") == events.count("children") == 1
    if missing:
        assert error and "acknowledgement" in error
    else:
        assert error is None
        assert events == ["parent", "children", "wait"]
        assert result["children"] == [{"order_id": 2}]


@pytest.mark.parametrize("adding", [False, True])
def test_position_action_subprocess_emits_one_real_terminal_result(tmp_path, adding):
    source = Path(os.environ.get("IBKR_REVIEW_SOURCE", "C:/Users/McKinley Slade/OneDrive/trading_ibkr")) / "execute_order.py"
    if not source.exists():
        pytest.skip("reviewed executor source is required for exact _out contract")
    text = source.read_text(encoding="utf-8-sig")
    out_node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "_out")
    actual_out = ast.get_source_segment(text, out_node)
    script = f'''
import json
from pathlib import Path
from broker_runtime import position_actions as actions
from tests.test_unified_position_actions import Broker, exit_order, namespace, request
{actual_out}
root = Path({str(tmp_path)!r})
broker = Broker(exits=[exit_order(1, 100)])
ns = namespace(root, broker)
ns["_out"] = _out
if {adding!r}:
    actions.life.stage_attached = lambda *a, **k: ({{"parent": [{{"order_id": 401}}], "children": []}}, None)
code = actions.run(ns, broker, request(), "primary", "", 0, 7, adding={adding!r})
assert code == 0
record = next(actions.records(root))
assert record["phase"] == "done"
assert record["result"]["ok"] is True
raise SystemExit(code)
'''
    completed = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True, check=False)
    assert completed.returncode == 0, completed.stderr
    lines = completed.stdout.strip().splitlines()
    assert len(lines) == 1, completed.stdout
    assert json.loads(lines[0])["ok"] is True
