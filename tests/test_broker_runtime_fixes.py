"""Mock broker regressions; deployed modules are parsed, never imported."""
from __future__ import annotations

import ast
import copy
import math
import os
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from broker_runtime import auction_lifecycle as auction
from broker_runtime import execution_lifecycle as life
from broker_runtime import prepare

SOURCE = Path(os.environ.get("IBKR_REVIEW_SOURCE", "C:/Users/McKinley Slade/OneDrive/trading_ibkr"))


def contract(con_id=42):
    return NS(conId=con_id, symbol="UNH", secType="STK", currency="USD", exchange="SMART", lastTradeDateOrContractMonth="")


def trade(*, account="TEST_PRIMARY", con_id=42, client_id=99, order_id=7, perm_id=701, qty=100, action="SELL", status="Submitted", filled=0):
    return NS(contract=contract(con_id), order=NS(account=account, clientId=client_id, orderId=order_id, permId=perm_id,
              totalQuantity=qty, action=action, orderType="STP", parentId=0, tif="GTC", ocaGroup="", lmtPrice=0., auxPrice=95., transmit=True),
              orderStatus=NS(status=status, filled=filled, avgFillPrice=100.))


def payload():
    return {"_broker_account": "TEST_PRIMARY", "_command_id": "fixture", "con_id": 42, "client_id": 99,
            "order_id": 7, "perm_id": 701, "symbol": "UNH", "sec_type": "STK", "action": "SELL", "qty": 40, "order_type": "MKT"}


class Broker:
    def __init__(self, trades=None, positions=None):
        self.trades = list(trades or [])
        self.holdings = list(positions or [])
        self.closed = False
    def reqAllOpenOrders(self): pass
    def sleep(self, seconds): pass
    def openTrades(self): return self.trades
    def positions(self): return self.holdings
    def qualifyContracts(self, value): return [value]
    def disconnect(self): self.closed = True


def out(ok, state, detail, fill=None):
    return {"ok": ok, "state": state, "detail": detail, "fill": fill}


def test_owner_preflight_fails_before_any_mutation():
    first, second = trade(), trade(client_id=88, order_id=8, perm_id=702)
    main = Broker([first, second])
    mutations = []
    other = Broker()
    other.connect = lambda *a, **k: (_ for _ in ()).throw(ConnectionError("occupied owner"))
    ns = {"IB": lambda: other, "guarded_place_order": lambda *a, **k: mutations.append(a)}
    done, error = life.resize(ns, main, "fixture", 0, 99, [(first, 60), (second, 60)], "fixture")
    assert error and done == [] and mutations == [] and other.closed


def test_exact_cancel_does_not_touch_another_account_or_contract():
    other = trade(account="TEST_OTHER")
    broker = Broker([other, trade(con_id=43)])
    ns = {"_out": out, "guarded_cancel_order": lambda *a: pytest.fail("wrong order cancelled")}
    result = life.mutate_one(ns, broker, payload(), "fixture", 0, 99)
    assert result["state"] == "rejected"


def test_cancel_exception_after_send_is_unknown():
    wanted = trade()
    broker = Broker([wanted])
    attempts = []
    def cancel(*args):
        attempts.append(args)
        raise TimeoutError("ambiguous response")
    result = life.mutate_one({"_out": out, "guarded_cancel_order": cancel}, broker, payload(), "fixture", 0, 99)
    assert len(attempts) == 1 and result["state"] == "unknown"


def test_modify_confirmed_exact_order_and_preserves_cache_before_send():
    wanted = trade()
    broker = Broker([wanted])
    def place(conn, cont, order, **kwargs):
        assert wanted.order.auxPrice == 95
        assert kwargs["account"] == "TEST_PRIMARY"
        wanted.order = order
        return wanted
    ns = {"_out": out, "guarded_place_order": place, "_command_signal": lambda *a: "test"}
    result = life.mutate_one(ns, broker, dict(payload(), new_stop=96, mutation_kind="exit"), "fixture", 0, 99, modify=True)
    assert result["state"] == "executed" and wanted.order.auxPrice == 96


def test_auction_claim_survives_date_change_and_ambiguous_failure(tmp_path):
    assert auction.claim(tmp_path, "TEST_PRIMARY", 42, "SPY|SELL|Event|2026-09-01", 100, "MKT", "OPG")
    assert not auction.claim(tmp_path, "TEST_PRIMARY", 42, "SPY|SELL|Event|2026-09-01", 100, "MKT", "OPG")
    with pytest.raises(ValueError, match="different"):
        auction.claim(tmp_path, "TEST_PRIMARY", 43, "SPY|SELL|Event|2026-09-01", 100, "MKT", "OPG")


@pytest.fixture
def executor(monkeypatch):
    if not (SOURCE / "execute_order.py").exists():
        pytest.skip("external-source AST regression requires the reviewed source checkout")
    monkeypatch.setitem(__import__("sys").modules, "execution_lifecycle", life)
    source = prepare.patch_execute((SOURCE / "execute_order.py").read_text(encoding="utf-8-sig").replace("\r\n", "\n"))
    names = {"_exact_position", "_do_close_resize", "_restore_leg_quantities", "_do_flatten"}
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names]
    env = {"math": math, "copy": copy, "_ERRORS": [], "TERMINAL": life.TERMINAL,
           "BrokerMutationBlocked": type("BrokerMutationBlocked", (Exception,), {}), "_out": out,
           "_cluster_symbol": lambda c: None, "_payload_contract_matches": lambda c, p: c.conId == p["con_id"],
           "_command_signal": lambda p, key: "fixture:" + key,
           "MarketOrder": lambda action, qty: NS(action=action, totalQuantity=qty, orderId=999, permId=999, clientId=99, ocaGroup="", account="TEST_PRIMARY"),
           "LimitOrder": lambda action, qty, px: NS(action=action, totalQuantity=qty, lmtPrice=px, orderId=999, permId=999, clientId=99, ocaGroup="", account="TEST_PRIMARY")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "reviewed-executor-functions", "exec"), env)
    return env


@pytest.mark.parametrize("status,filled,expected,lagging", [("Submitted", 0, 60, False), ("Submitted", 10, 60, False), ("Cancelled", 10, 90, False), ("Cancelled", 10, 90, True)])
def test_close_recovery_capacity_includes_working_close(executor, monkeypatch, status, filled, expected, lagging):
    env = executor
    position = NS(account="TEST_PRIMARY", position=100, contract=contract())
    stop = trade()
    broker = Broker([stop], [position])
    def resize(ns, ib, host, port, cid, plan, signal):
        done = []
        for t, qty in plan:
            done.append((t.order.permId, t.order.totalQuantity, qty))
            t.order.totalQuantity = qty
        return done, None
    monkeypatch.setattr(life, "resize", resize)
    def place(ib, cont, order, **kwargs):
        if not lagging:
            position.position -= filled
        close = NS(order=order, contract=cont, orderStatus=NS(status=status, filled=filled, avgFillPrice=100))
        if status not in life.TERMINAL:
            broker.trades.append(close)
        return close
    env.update({"_orders_for_contract": lambda *a: broker.trades,
                "_split_pending_entry_legs": lambda rows: (rows, []),
                "_capture_exit_legs": lambda rows, action: ([{"source_key": str(t.order.permId)} for t in rows], None),
                "_validate_exit_topology": lambda *a, **k: None,
                "_scaled_exit_legs": lambda legs, qty: ([dict(leg, scaled_qty=qty) for leg in legs], None),
                "_order_source_key": lambda order: str(order.permId),
                "_resize_legs_via_owners": lambda *a: resize(env, *a),
                "_confirm_leg_quantities": lambda *a: [], "guarded_place_order": place})
    result = env["_do_close_resize"](broker, payload(), "primary", "fixture", 0, 99)
    assert stop.order.totalQuantity == expected
    working = 40 - filled if status not in life.TERMINAL else 0
    assert stop.order.totalQuantity + working <= 100 - filled
    assert result["state"] == "unknown"


def test_flatten_wrong_account_does_not_place(executor):
    env = executor
    position = NS(account="TEST_OTHER", position=100, contract=contract())
    broker = Broker(positions=[position])
    env["guarded_place_order"] = lambda *a, **k: pytest.fail("wrong-account flatten")
    result = env["_do_flatten"](broker, dict(payload(), qty=100), "fixture", 0, 99)
    assert result["state"] == "rejected"


def test_flatten_working_close_is_unknown(executor):
    env = executor
    position = NS(account="TEST_PRIMARY", position=100, contract=contract())
    broker = Broker(positions=[position])
    def place(ib, cont, order, **kwargs):
        assert order.account == kwargs["account"] == "TEST_PRIMARY"
        return NS(order=order, orderStatus=NS(status="Submitted", filled=0, avgFillPrice=0))
    env.update({"_orders_for_contract": lambda *a: [], "guarded_place_order": place})
    result = env["_do_flatten"](broker, dict(payload(), qty=100), "fixture", 0, 99)
    assert result["state"] == "unknown"


def test_prepare_checks_all_reviewed_hashes_and_writes_new_candidate(tmp_path):
    if not SOURCE.exists():
        pytest.skip("external-source preparation requires the reviewed checkout")
    target = tmp_path / "candidate"
    assert len(prepare.prepare(SOURCE, target)) == 8
    for file in target.glob("*.py"):
        compile(file.read_text(encoding="utf-8"), file.name, "exec")
    with pytest.raises(ValueError, match="new"):
        prepare.prepare(SOURCE, target)


def test_quote_timeout_kills_and_reaps_without_importing_agent():
    if not (SOURCE / "exec_agent.py").exists():
        pytest.skip("requires reviewed external source")
    import asyncio
    source = prepare.patch_agent((SOURCE / "exec_agent.py").read_text(encoding="utf-8-sig").replace("\r\n", "\n"))
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.AsyncFunctionDef) and n.name == "_fetch_option")
    events = []
    class Process:
        returncode = None
        async def communicate(self):
            events.append("communicate")
            return b"{}", None
        def kill(self):
            self.returncode = -1
            events.append("kill")
    process = Process()
    async def spawn(*a, **k): return process
    async def wait(awaitable, timeout):
        awaitable.close()
        raise asyncio.TimeoutError()
    fake_async = NS(create_subprocess_exec=spawn, wait_for=wait, subprocess=NS(PIPE=1, DEVNULL=2))
    env = {"asyncio": fake_async, "sys": NS(executable="fixture"), "OPTION_SCRIPT": "fixture", "json": __import__("json")}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "quote-fixture", "exec"), env)
    private_loop = asyncio.new_event_loop()
    try:
        result = private_loop.run_until_complete(env["_fetch_option"]("SPY"))
    finally:
        private_loop.close()
    assert result.get("error") and events == ["kill", "communicate"]


def test_add_attaches_before_release_and_partial_fill_is_unknown():
    parent = trade(action="BUY", status="Submitted", filled=10, qty=40)
    events = []
    broker = Broker()
    broker.client = NS(getReqId=lambda: 99)
    def place(ib, contract, order, **kwargs):
        assert order.transmit is False
        events.append("untransmitted parent")
        return parent
    def children(*a, **kwargs):
        assert kwargs["parent_id"] == 99 and kwargs["transmit_chain"]
        events.append("attached children and release")
        return [trade(qty=40)], []
    context = {"entry_action": "BUY", "close_action": "SELL", "avg_cost": 100, "account": "TEST_PRIMARY", "ref": contract(), "legs": [{}]}
    ns = {"MarketOrder": lambda action, qty: NS(action=action, totalQuantity=qty), "guarded_place_order": place,
          "_fast_guard_risk_usd": lambda *a: 100, "_place_exit_legs": children,
          "_placement_problem": lambda *a: None, "_placed_ids": lambda rows: [t.order.orderId for t in rows]}
    result, error = life.stage_attached(ns, broker, context, 40, "test", market=True)
    assert events == ["untransmitted parent", "attached children and release"]
    assert "partial" in error and "DO NOT RETRY" in error


def test_stream_deadline_covers_parent_exited_with_inherited_pipe(monkeypatch, tmp_path):
    import queue
    from scripts import automation_supervisor as supervisor
    from scripts import process_tree
    ticks = iter(range(10))
    closed = []
    process = NS(stdout=iter(()), poll=lambda: 0, wait=lambda **kw: 0)
    class Queue:
        def get(self, timeout): raise queue.Empty
        def put(self, item): pass
    monkeypatch.setattr(supervisor, "time", NS(monotonic=lambda: next(ticks)))
    monkeypatch.setattr(supervisor.queue, "Queue", Queue)
    monkeypatch.setattr(supervisor.threading, "Thread", lambda **kw: NS(start=lambda: None, join=lambda **kw: None))
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **kw: process)
    monkeypatch.setattr(process_tree, "ProcessTree", lambda proc: NS(close=lambda: closed.append(True)))
    rc = supervisor.SubprocessClient().stream(["fixture"], cwd=tmp_path, env={}, timeout_seconds=3, logger=NS(line=lambda *a: None))
    assert rc == 124 and closed == [True]


def test_entry_preflight_distinguishes_valid_empty_from_missing(monkeypatch):
    if not (SOURCE / "eq_order_entry.py").exists():
        pytest.skip("requires reviewed external source")
    import pandas as pd
    source = prepare.patch_entry((SOURCE / "eq_order_entry.py").read_text(encoding="utf-8-sig").replace("\r\n", "\n"))
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_run_execution")
    env = {"os": NS(path=NS(join=lambda *a: "fixture.csv", exists=lambda path: False, basename=lambda path: path)),
           "STAGING_FOLDER": "fixture", "pd": NS(read_csv=lambda path: pd.DataFrame()),
           "IB": lambda: pytest.fail("preflight attempted broker connection")}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "entry-preflight-fixture", "exec"), env)
    assert env["_run_execution"](None) == 1
    env["os"].path.exists = lambda path: True
    assert env["_run_execution"](None) == 0
