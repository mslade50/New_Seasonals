"""Broker evidence recovery never calls place/cancel, including lost replies."""
import asyncio
import copy
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace as N

import pytest

from broker_runtime import broker_reconciliation as obs
from broker_runtime import position_actions as actions
from broker_runtime import position_action_agent as agent
from tests.test_unified_position_actions import Broker, exit_order, namespace, request


class ObservedBroker(Broker):
    def reqAllOpenOrders(self):
        return self.openTrades()

    def reqExecutions(self):
        return []


def stopped(account="PRIMARY", **extra):
    p = dict(request(), _broker_account=account)
    return dict(version=1, id=p["_command_id"], payload=p, phase="attention",
                account_key="primary" if account == "PRIMARY" else "pa", mutation="cancel exit",
                held=100, quantity=40, closing="SELL", legs=[], removed=["old-exit"],
                manual_override="later-edit", **extra)


def evidence(**extra):
    return dict(account="PRIMARY", con_id=42, at=1, position=100, orders=[],
                completed=[], executions=[], **extra)


@pytest.mark.parametrize("account", ["PRIMARY", "PA"])
@pytest.mark.parametrize("mutation", ["cancel exit", "resize exit", ""])
def test_stopped_preclose_retires_without_replaying_or_undoing_manual_edits(tmp_path, account, mutation):
    broker = ObservedBroker(exits=[exit_order(1, 73, account=account)])
    broker.position.account = account
    record = stopped(account)
    record["mutation"] = mutation
    actions.save(tmp_path, record)
    before = copy.deepcopy(broker.orders)
    with actions.operation_lock(tmp_path):
        results = obs.refresh_target(broker, tmp_path, account, 42)
    assert results[record["id"]]["state"] == "rejected"
    assert "before entry/close" in results[record["id"]]["detail"]
    saved = next(actions.records(tmp_path))
    assert saved["phase"] == "done" and saved["manual_override"] == "later-edit"
    assert saved["resolution"]["no_replay"]
    assert broker.orders == before and not broker.mutations


def test_new_requested_action_can_follow_resolved_receipt_but_old_id_cannot_replay(tmp_path):
    broker = ObservedBroker(exits=[exit_order(1, 100)])
    old = stopped()
    actions.save(tmp_path, old)
    ns = namespace(tmp_path, broker)
    result = actions.run(ns, broker, dict(request(), _command_id="new"), "primary", "", 0, 7)
    assert result["ok"] and broker.position.position == 60
    count = len(broker.mutations)
    result = actions.run(ns, broker, request(), "primary", "", 0, 7)
    assert result["state"] == "rejected" and len(broker.mutations) == count


def test_add_is_unblocked_without_replaying_the_old_close(tmp_path, monkeypatch):
    broker = ObservedBroker(exits=[exit_order(1, 100)])
    actions.save(tmp_path, stopped())
    submitted = []
    def attached(ns, ib, context, qty, signal, market=False):
        submitted.append((qty, signal))
        return dict(parent=[dict(order_id=999)], children=[dict(order_id=1000)]), None
    monkeypatch.setattr(actions.life, "stage_attached", attached)
    result = actions.run(namespace(tmp_path, broker), broker, dict(request(), _command_id="new-add"),
                         "primary", "", 0, 7, adding=True)
    assert result["ok"] and len(submitted) == 1
    assert broker.position.position == 100 and not broker.mutations


@pytest.mark.parametrize("method", ["reqPositions", "reqAllOpenOrders", "reqCompletedOrders", "reqExecutions"])
@pytest.mark.parametrize("failure", ["none", "timeout"])
def test_incomplete_requests_leave_original_receipt_unchanged(tmp_path, monkeypatch, method, failure):
    broker = ObservedBroker()
    actions.save(tmp_path, stopped())
    original = actions.record_path(tmp_path, "fixture").read_bytes()
    def fail(*a, **k):
        if failure == "timeout":
            raise TimeoutError("fixture")
        return None
    monkeypatch.setattr(broker, method, fail)
    with pytest.raises((ValueError, TimeoutError)):
        obs.refresh_target(broker, tmp_path, "PRIMARY", 42)
    assert actions.record_path(tmp_path, "fixture").read_bytes() == original
    assert not broker.mutations


def test_book_changing_between_reads_is_not_resolved(tmp_path):
    broker = ObservedBroker()
    actions.save(tmp_path, stopped())
    def executions():
        broker.position.position = 99
        return []
    broker.reqExecutions = executions
    with pytest.raises(ValueError, match="changed"):
        obs.refresh_target(broker, tmp_path, "PRIMARY", 42)
    assert next(actions.records(tmp_path))["phase"] == "attention"


@pytest.mark.parametrize("status", ["PendingCancel", "PendingSubmit", "ApiPending", "Inactive"])
def test_transitional_broker_orders_keep_uncertainty(status):
    trade = exit_order(1, 100)
    trade.orderStatus.status = status
    # Inactive is omitted by this fixture's openTrades; return actual request rows.
    broker = ObservedBroker(exits=[trade])
    broker.reqAllOpenOrders = lambda: [trade]
    with pytest.raises(ValueError, match="transition"):
        obs.capture(broker, "PRIMARY", 42)


@pytest.mark.parametrize("typ", ["MKT", "LMT", "STP", "STP LMT", "MOC"])
@pytest.mark.parametrize("status,filled", [("Filled", 40), ("Cancelled", 11), ("ApiCancelled", 0)])
def test_terminal_close_recovers_by_exact_identity_for_order_types(typ, status, filled):
    trade = exit_order(9, 40, typ=typ)
    trade.orderStatus.status, trade.orderStatus.filled = status, 0
    trade.order.filledQuantity = filled
    broker = ObservedBroker(exits=[trade])
    record = stopped(wire=["PRIMARY", 42, 7, 9, 109])
    record["mutation"] = "submit close"
    result = obs.resolve(record, obs.capture(broker, "PRIMARY", 42))
    assert result["phase"] == "done" and result["result"]["fill"]["filled"] == filled
    assert not broker.mutations


def test_cancelled_completed_order_default_zero_is_not_treated_as_zero_fills():
    trade = exit_order(9, 40)
    trade.orderStatus.status = "Cancelled"
    broker = ObservedBroker(exits=[trade])
    record = stopped(wire=["PRIMARY", 42, 7, 9, 109])
    record["mutation"] = "submit close"
    result = obs.resolve(record, obs.capture(broker, "PRIMARY", 42))
    assert result["result"]["fill"]["filled"] is None
    assert "unavailable" in result["result"]["detail"]


def test_working_or_missing_close_cannot_be_resubmitted():
    trade = exit_order(9, 40)
    broker = ObservedBroker(exits=[trade])
    record = stopped(wire=["PRIMARY", 42, 7, 9, 109])
    record["mutation"] = "submit close"
    with pytest.raises(ValueError):
        obs.resolve(record, obs.capture(broker, "PRIMARY", 42))
    broker.orders.clear()
    with pytest.raises(ValueError):
        obs.resolve(record, obs.capture(broker, "PRIMARY", 42))


def test_complete_execution_evidence_resolves_missing_close_without_inventing_no_fill():
    record = stopped(wire=["PRIMARY", 42, 7, 9, 109])
    record["mutation"] = "submit close"
    ev = evidence()
    ev["executions"] = [dict(identity=record["wire"], cumulative=40, shares=40, exec_id="trade.01")]
    assert obs.resolve(record, ev)["result"]["fill"]["filled"] == 40
    ev["executions"][0]["cumulative"] = 10
    with pytest.raises(ValueError):
        obs.resolve(record, ev)


@pytest.mark.parametrize("account,con_id,perm", [("OTHER", 42, 109), ("PRIMARY", 43, 109), ("PRIMARY", 42, 110)])
def test_same_symbol_other_identity_is_not_evidence(account, con_id, perm):
    trade = exit_order(9, 40, account=account, con_id=con_id)
    trade.order.permId = perm
    trade.orderStatus.status = "Filled"
    record = stopped(wire=["PRIMARY", 42, 7, 9, 109])
    record["mutation"] = "submit close"
    with pytest.raises(ValueError):
        obs.resolve(record, obs.capture(ObservedBroker(exits=[trade]), "PRIMARY", 42))


@pytest.mark.parametrize("modify", [True, False])
def test_lost_manual_edit_acknowledgment_resolves_without_second_send(tmp_path, modify):
    trade = exit_order(9, 100)
    record = stopped()
    record.update(identity=["PRIMARY", 42, 7, 9, 109], modify=modify)
    record["payload"]["new_qty"] = 100
    if not modify:
        trade.orderStatus.status = "Cancelled"
    actions.save(tmp_path / "order_edits", record)
    broker = ObservedBroker(exits=[trade])
    assert obs.refresh_target(broker, tmp_path, "PRIMARY", 42)["fixture"]["ok"]
    assert not broker.mutations


def test_observer_route_never_runs_a_position_mutation(tmp_path):
    broker = ObservedBroker()
    actions.save(tmp_path, stopped())
    result = actions.run(namespace(tmp_path, broker), broker, dict(request(), observe_only=True),
                         "primary", "", 0, 7)
    assert result["state"] == "rejected" and not broker.mutations


def test_agent_checks_stopped_positions_and_edits_once_per_target(tmp_path):
    actions.save(tmp_path, stopped())
    edit = stopped()
    edit.update(id="edit", modify=False, identity=["PRIMARY", 42, 7, 1, 101])
    edit.pop("account_key")
    actions.save(tmp_path / "order_edits", edit)
    sent = []
    async def execute(command):
        sent.append(command)
    ns = dict(_BOOK=dict(book=dict(accounts=[dict(key="primary", broker_account="PRIMARY")])),
              _live_eligible=lambda c: (True, ""), _execute_live=execute)
    asyncio.run(agent.observe_stopped(ns, tmp_path))
    assert len(sent) == 1 and sent[0]["payload"]["observe_only"]


def test_missing_addition_identity_and_restore_uncertainty_stay_unknown():
    record = stopped()
    for mutation in ["stage addition and attached exits", "restore previously cancelled exit rungs"]:
        record["mutation"] = mutation
        with pytest.raises(ValueError):
            obs.resolve(record, evidence())


def test_completed_add_with_attached_exits_reconciles_without_another_entry():
    parent, child = exit_order(9, 40), exit_order(10, 40)
    parent.order.action, parent.orderStatus.status = "BUY", "Filled"
    child.order.parentId = 9
    record = stopped()
    record.update(mutation="stage addition and attached exits", addition_requested=40,
                  add_context=dict(entry_action="BUY", close_action="SELL", legs=[dict(qty=100, source_key="stop")]),
                  addition=[dict(parent=[dict(order_id=9, perm_id=109)], children=[dict(order_id=10, perm_id=110)])])
    broker = ObservedBroker(holdings=140, exits=[parent, child, exit_order(1, 100)])
    assert obs.resolve(record, obs.capture(broker, "PRIMARY", 42))["result"]["fill"]["filled"] == 40
    assert not broker.mutations
    child.order.parentId = 7
    with pytest.raises(ValueError, match="attached exit"):
        obs.resolve(record, obs.capture(broker, "PRIMARY", 42))


def test_partial_add_still_working_is_not_marked_completed():
    parent, child = exit_order(9, 40), exit_order(10, 40)
    parent.order.action, parent.orderStatus.filled = "BUY", 10
    child.order.parentId = 9
    record = stopped()
    record.update(mutation="stage addition and attached exits", addition_requested=40,
                  add_context=dict(entry_action="BUY", close_action="SELL", legs=[dict(qty=100, source_key="stop")]),
                  addition=[dict(parent=[dict(order_id=9, perm_id=109)], children=[dict(order_id=10, perm_id=110)])])
    with pytest.raises(ValueError, match="still working"):
        obs.resolve(record, obs.capture(ObservedBroker(exits=[parent, child]), "PRIMARY", 42))


def test_fresh_snapshot_is_required_even_when_account_and_symbol_are_flat(tmp_path):
    broker = ObservedBroker(holdings=0)
    actions.save(tmp_path, stopped())
    broker.reqCompletedOrders = lambda **kw: None
    with pytest.raises(ValueError):
        obs.refresh_target(broker, tmp_path, "PRIMARY", 42)


def test_request_timeout_is_bounded_and_restored_after_failure():
    broker = ObservedBroker()
    broker.RequestTimeout = 0
    def fail():
        assert broker.RequestTimeout == 8
        raise TimeoutError()
    broker.reqPositions = fail
    with pytest.raises(TimeoutError):
        obs.capture(broker, "PRIMARY", 42)
    assert broker.RequestTimeout == 0


def test_partial_exit_reconciliation_is_retired_without_finishing_old_allocation():
    record = stopped()
    record.update(kind="reconcile_exits", plan=[dict(identity=["PRIMARY", 42, 7, 1, 101], remaining="80")])
    ev = evidence()
    ev["position"] = 0
    result = obs.resolve(record, ev)
    assert result["phase"] == "done" and not result["result"]["ok"]
    assert "no remaining changes replayed" in result["result"]["detail"]


def test_installed_executor_routes_both_handlers_to_updated_module(tmp_path, monkeypatch):
    # Extract functions only: never import the configured live executor.
    path = Path('C:/Users/McKinley Slade/OneDrive/trading_ibkr/execute_order.py')
    if not path.exists():
        pytest.skip('host runtime not available')
    source = path.read_text(encoding='utf-8-sig')
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
             and n.name in {'_do_close_resize', '_do_add_to_position'}]
    assert len(nodes) == 2
    broker = ObservedBroker()
    actions.save(tmp_path, stopped())
    ns = namespace(tmp_path, broker)
    monkeypatch.setitem(sys.modules, 'position_actions', actions)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<installed handlers>', 'exec'), ns)
    for name in ['_do_close_resize', '_do_add_to_position']:
        result = ns[name](broker, dict(request(), observe_only=True), 'primary', '', 0, 7)
        assert result['state'] in {'rejected', 'executed'} and not broker.mutations
    assert next(actions.records(tmp_path))['phase'] == 'done'


def test_candidate_is_pinned_to_reviewed_runtime_and_contains_only_required_modules(tmp_path):
    from broker_runtime import prepare_broker_reconciliation as prep
    runtime = Path('C:/Users/McKinley Slade/OneDrive/trading_ibkr')
    if not runtime.exists():
        pytest.skip('host runtime not available')
    out = tmp_path / 'candidate'
    manifest = prep.prepare(runtime, out)
    assert set(manifest['candidate']) == set(prep.MODULES)
    assert set(p.name for p in out.iterdir()) == set(prep.MODULES) | {'manifest.json'}
    assert manifest == json.loads((out / 'manifest.json').read_text())
    with pytest.raises(ValueError, match='new'):
        prep.prepare(runtime, out)


def test_execution_correction_cannot_overstate_original_cumulative_fill():
    r = stopped(wire=["PRIMARY", 42, 7, 9, 109])
    r["mutation"] = "submit close"
    ev = evidence()
    ev["executions"] = [dict(identity=r["wire"], exec_id="trade.01", cumulative=40, shares=40),
                        dict(identity=r["wire"], exec_id="trade.02", cumulative=10, shares=10)]
    with pytest.raises(ValueError, match="correction"):
        obs.resolve(r, ev)


def test_execution_only_recovery_requires_complete_unique_fill_history():
    r = stopped(wire=["PRIMARY", 42, 7, 9, 109])
    r["mutation"] = "submit close"
    ev = evidence()
    fill = dict(identity=r["wire"], exec_id="trade.01", cumulative=40, shares=40)
    ev["executions"] = [fill, fill.copy()]
    assert obs.resolve(r, ev)["result"]["fill"]["filled"] == 40
    ev["executions"] = [dict(fill, shares=10)]
    with pytest.raises(ValueError, match="incomplete"):
        obs.resolve(r, ev)


def test_zero_permanent_id_requires_matching_command_reference():
    r = stopped(wire=["PRIMARY", 42, 7, 9, 0])
    r["mutation"] = "submit close"
    trade = exit_order(9, 40)
    trade.orderStatus.status = "Filled"
    broker = ObservedBroker(exits=[trade])
    with pytest.raises(ValueError):
        obs.resolve(r, obs.capture(broker, "PRIMARY", 42))
    trade.order.orderRef = "EXEC|fixture|unified-close"
    assert obs.resolve(r, obs.capture(broker, "PRIMARY", 42))["phase"] == "done"


def test_stop_cancellation_failure_keeps_real_coverage_discrepancy_visible(tmp_path):
    r = stopped()
    r["legs"] = [dict(order_type="STP", qty=100)]
    actions.save(tmp_path, r)
    broker = ObservedBroker(holdings=100)
    result = obs.refresh_target(broker, tmp_path, "PRIMARY", 42)[r["id"]]
    assert result["state"] == "unknown" and "cover 0" in result["detail"]
    saved = next(actions.records(tmp_path))
    assert saved["phase"] == "attention"
    assert saved["observation"]["broker_outcome"]["state"] == "rejected"
    assert "before entry/close" in saved["observation"]["broker_outcome"]["detail"]
    assert not broker.mutations
    broker.orders = [exit_order(1, 100)]
    assert obs.refresh_target(broker, tmp_path, "PRIMARY", 42)[r["id"]]["state"] == "rejected"
    assert next(actions.records(tmp_path))["phase"] == "done"


def test_partial_cancelled_close_reports_fill_but_does_not_hide_unprotected_remainder():
    trade, stop = exit_order(9, 40), exit_order(1, 60)
    trade.orderStatus.status = "Cancelled"
    trade.order.filledQuantity = 10
    r = stopped(wire=["PRIMARY", 42, 7, 9, 109])
    r.update(mutation="submit close", legs=[dict(order_type="STP", qty=100)])
    with pytest.raises(obs.CoverageDiscrepancy, match="cover 60.*holds 90") as caught:
        obs.resolve(r, obs.capture(ObservedBroker(holdings=90, exits=[trade, stop]), "PRIMARY", 42))
    assert caught.value.result["fill"]["filled"] == 10


def test_filled_add_with_cancelled_protection_cannot_be_reported_successful():
    parent, child = exit_order(9, 40), exit_order(10, 40)
    parent.order.action, parent.orderStatus.status = "BUY", "Filled"
    child.order.parentId, child.orderStatus.status = 9, "Cancelled"
    r = stopped()
    r.update(mutation="stage addition and attached exits", addition_requested=40,
             add_context=dict(entry_action="BUY", close_action="SELL", legs=[dict(qty=100, source_key="stop")]),
             addition=[dict(parent=[dict(order_id=9, perm_id=109)], children=[dict(order_id=10, perm_id=110)])])
    broker = ObservedBroker(holdings=140, exits=[parent, child, exit_order(1, 100)])
    with pytest.raises(obs.CoverageDiscrepancy, match="cover 100.*holds 140") as caught:
        obs.resolve(r, obs.capture(broker, "PRIMARY", 42))
    assert caught.value.result["fill"]["filled"] == 40


@pytest.mark.parametrize("field,value", [("ocaGroup", "changed"), ("ocaType", 3), ("orderType", "LMT"),
    ("parentId", 900), ("goodAfterTime", "20270917 15:59:00"), ("transmit", False), ("tif", "DAY")])
def test_snapshot_detects_structural_changes(field, value):
    trade = exit_order(1, 100)
    broker = ObservedBroker(exits=[trade])
    def executions():
        setattr(trade.order, field, value)
        return []
    broker.reqExecutions = executions
    with pytest.raises(ValueError, match="changed"):
        obs.capture(broker, "PRIMARY", 42)


def test_blank_account_is_not_silently_treated_as_absent():
    broker = ObservedBroker(exits=[exit_order(1, 100, account="")])
    with pytest.raises(ValueError, match="blank account"):
        obs.capture(broker, "PRIMARY", 42)
    broker.managedAccounts = lambda: ["PRIMARY"]
    assert obs.capture(broker, "PRIMARY", 42)["orders"][0]["identity"][0] == "PRIMARY"
    broker.managedAccounts = lambda: ["PRIMARY", "PA"]
    with pytest.raises(ValueError, match="blank account"):
        obs.capture(broker, "PRIMARY", 42)


def test_real_ib_wrapper_requires_raw_decoder_reader_instead_of_cached_trades():
    broker = ObservedBroker()
    broker.wrapper = N()
    with pytest.raises(ValueError, match="raw broker"):
        obs.capture(broker, "PRIMARY", 42)
    calls = []
    def raw(ib):
        calls.append(ib)
        return []
    obs.capture(broker, "PRIMARY", 42, open_reader=raw)
    assert len(calls) == 2


def test_contradictory_completed_fill_quantity_is_not_ignored():
    trade = exit_order(9, 40)
    trade.orderStatus.status = "Filled"
    trade.order.filledQuantity = 10
    with pytest.raises(ValueError, match="contradicts"):
        obs.capture(ObservedBroker(exits=[trade]), "PRIMARY", 42)


@pytest.fixture
def ib_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
    asyncio.set_event_loop(None)


def test_installed_raw_collector_observes_structural_changes_hidden_by_ib_cache(monkeypatch, ib_event_loop):
    import dataclasses
    import math
    import threading
    from ib_insync import IB, Order, OrderState, Stock
    path = Path('C:/Users/McKinley Slade/OneDrive/trading_ibkr/legend_reservation_guard.py')
    if not path.exists():
        pytest.skip('host runtime not available')
    names = {'_fresh_open_trades', '_clean_raw_broker_order'}
    nodes = [n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    env = dict(deepcopy=copy.deepcopy, SimpleNamespace=N, math=math,
               is_dataclass=dataclasses.is_dataclass, dataclass_fields=dataclasses.fields,
               BrokerMutationBlocked=ValueError, _BROKER_SNAPSHOT_LOCK=threading.RLock())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<raw reader>', 'exec'), env)
    # Construct the actual library wrapper without connecting to a broker.
    ib = IB()
    contract = Stock('TEST', 'SMART', 'USD', conId=42)
    old = Order(orderId=1, clientId=7, permId=101, account='PRIMARY', action='SELL',
                totalQuantity=100, orderType='STP', auxPrice=90, parentId=8,
                ocaGroup='OLD', ocaType=1, tif='GTC')
    state = OrderState(status='Submitted')
    ib.wrapper.openOrder(1, contract, old, state)
    actual = copy.deepcopy(old)
    actual.ocaGroup, actual.parentId, actual.goodAfterTime = 'CURRENT', 9, '20270917 15:59:00'
    def response():
        ib.wrapper.openOrder(1, contract, actual, state)
        ib.wrapper.orderStatus(1, 'Submitted', 0, 100, 0, 101, 9, 0, 7, '', 0)
        return list(ib.wrapper.trades.values())
    monkeypatch.setattr(ib, 'reqAllOpenOrders', response)
    monkeypatch.setattr(ib, 'reqPositions', lambda: [N(account='PRIMARY', contract=contract, position=100)])
    monkeypatch.setattr(ib, 'reqCompletedOrders', lambda **kw: [])
    monkeypatch.setattr(ib, 'reqExecutions', lambda: [])
    ev = obs.capture(ib, 'PRIMARY', 42, open_reader=env['_fresh_open_trades'])
    assert ev['orders'][0]['oca_group'] == 'CURRENT'
    assert ev['orders'][0]['parent'] == 9
    assert ev['orders'][0]['good_after'] == '20270917 15:59:00'
    assert next(iter(ib.wrapper.trades.values())).order.ocaGroup == 'OLD'
