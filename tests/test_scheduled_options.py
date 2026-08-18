"""Scheduled target-delta option intent: pure gates and crash-safe persistence.

The live agent/executor stay in the local OneDrive execution project, so these
tests skip in CI when that project is unavailable. No test connects to IBKR.
"""
import datetime as dt
import asyncio
import importlib
import json
import os
import sys
from types import SimpleNamespace

import pytest


IBKR_DIR = os.path.join(os.path.expanduser("~"), "OneDrive", "trading_ibkr")


@pytest.fixture(scope="module")
def modules():
    if not os.path.isdir(IBKR_DIR):
        pytest.skip(f"live execution dir not present: {IBKR_DIR}")
    sys.path.insert(0, IBKR_DIR)
    try:
        return importlib.import_module("exec_agent"), importlib.import_module("execute_order")
    except ImportError as exc:
        pytest.skip(f"local execution modules unavailable ({exc})")


def _payload(**overrides):
    payload = {
        "symbol": "SPY", "right": "P", "target_delta": 0.15,
        "delta_tolerance": 0.03, "premium_budget": 1000,
        "order_type": "MKT", "tif": "DAY",
        "execute_date": "2026-08-21", "execute_time": "15:45",
        "timezone": "America/New_York", "grace_minutes": 5,
        "expiry_mode": "min_dte", "min_dte": 30, "expiry": None,
    }
    payload.update(overrides)
    return payload


def test_schedule_validation_accepts_min_dte_and_specific(modules, monkeypatch):
    agent, _executor = modules
    monkeypatch.setattr(agent, "OPTION_ACCOUNTS", {"primary"})
    monkeypatch.setattr(agent, "UNCAPPED_OPTIONS_ACCOUNTS", set())
    monkeypatch.setitem(agent._BOOK, "book", {
        "accounts": [{"key": "primary", "positions": [], "nlv": 1_000_000}],
    })
    now = dt.datetime(2026, 8, 18, 12, 0, tzinfo=agent._ET)

    assert agent._validate_scheduled_option(_payload(), "primary", now=now) == []
    exact = _payload(expiry_mode="specific", min_dte=None, expiry="2026-09-18")
    assert agent._validate_scheduled_option(exact, "primary", now=now) == []


def test_schedule_validation_fails_closed(modules, monkeypatch):
    agent, _executor = modules
    monkeypatch.setattr(agent, "OPTION_ACCOUNTS", {"primary"})
    monkeypatch.setattr(agent, "UNCAPPED_OPTIONS_ACCOUNTS", {"primary"})
    monkeypatch.setitem(agent._BOOK, "book", {"accounts": []})
    now = dt.datetime(2026, 8, 18, 12, 0, tzinfo=agent._ET)
    reasons = agent._validate_scheduled_option(
        _payload(target_delta=0.75, execute_date="2026-08-22"), "primary", now=now)
    assert any("target_delta" in reason for reason in reasons)
    assert any("weekday" in reason for reason in reasons)
    assert any("risk_ack" in reason for reason in reasons)


def test_expiry_delta_and_ask_sizing(modules):
    _agent, executor = modules
    today = dt.date(2026, 8, 21)
    expiries = {"20260828", "20260918", "20261016"}
    assert executor._resolve_option_expiry(expiries, today, "min_dte", min_dte=30) == "20261016"
    assert executor._resolve_option_expiry(
        expiries, today, "specific", expiry="2026-10-16") == "20261016"
    with pytest.raises(ValueError, match="not listed"):
        executor._resolve_option_expiry(expiries, today, "specific", expiry="2026-09-25")

    rows = [
        {"right": "P", "delta": -0.12, "ask": 0.80, "con_id": 12, "market_data_type": 1},
        {"right": "P", "delta": -0.149, "ask": 1.20, "con_id": 15, "market_data_type": 1},
        {"right": "P", "delta": -0.18, "ask": 1.60, "con_id": 18, "market_data_type": 1},
    ]
    selected = executor._select_delta_row(rows, "P", 0.15, 0.03)
    assert selected["con_id"] == 15
    assert executor._market_option_quantity(1000, selected["ask"]) == 8
    with pytest.raises(ValueError, match="not live"):
        executor._select_delta_row([{**rows[1], "market_data_type": 3}], "P", 0.15, 0.03)


def test_schedule_persistence_marks_interrupted_execution_unknown(modules, monkeypatch, tmp_path):
    agent, _executor = modules
    state_path = tmp_path / "scheduled.json"
    monkeypatch.setattr(agent, "SCHEDULED_OPTIONS_PATH", str(state_path))
    agent._SCHEDULES.clear()
    cmd = {"id": "sched-1", "account": "primary", "payload": _payload()}
    rec = agent._store_schedule(cmd)
    assert rec["state"] == "scheduled"
    assert json.loads(state_path.read_text(encoding="utf-8"))[0]["id"] == "sched-1"

    rows = json.loads(state_path.read_text(encoding="utf-8"))
    rows[0]["state"] = "executing"
    state_path.write_text(json.dumps(rows), encoding="utf-8")
    agent._SCHEDULES.clear()
    agent._load_schedules()
    assert agent._SCHEDULES["sched-1"]["state"] == "unknown"
    assert "never auto-retried" in agent._SCHEDULES["sched-1"]["detail"]


def test_dynamic_executor_resolves_and_submits_market_order(
        modules, monkeypatch, capsys):
    _agent, executor = modules
    option_workbench = importlib.import_module("option_workbench")
    selected_expiry = (dt.date.today() + dt.timedelta(days=35)).strftime("%Y%m%d")
    later_expiry = (dt.date.today() + dt.timedelta(days=63)).strftime("%Y%m%d")
    monkeypatch.setattr(executor, "OPTION_ACCOUNTS", {"primary"})
    monkeypatch.setattr(executor, "UNCAPPED_OPTIONS_ACCOUNTS", set())
    monkeypatch.setattr(executor, "LIVE_MAX_OPT_CONTRACTS", 10)
    monkeypatch.setattr(executor, "LIVE_MAX_OPT_RISK", 2500)
    monkeypatch.setattr(executor, "LIVE_MAX_OPT_RISK_BY_ACCT", {})
    monkeypatch.setattr(option_workbench, "_quote_chain", lambda *args: ({
        "expiry": selected_expiry, "dte": 35,
        "strikes": [{"right": "P", "delta": -0.149, "ask": 1.20,
                     "bid": 1.18, "strike": 700.0, "con_id": 15,
                     "market_data_type": 1}],
    }, None))

    class FakeIB:
        def __init__(self):
            self.placed = None

        def qualifyContracts(self, *contracts):
            for contract in contracts:
                contract.conId = 15 if contract.secType == "OPT" else 1
            return list(contracts)

        def reqMarketDataType(self, value):
            assert value == 1

        def reqTickers(self, _contract):
            return [SimpleNamespace(marketPrice=lambda: 750.0, marketDataType=1)]

        def reqSecDefOptParams(self, *_args):
            return [SimpleNamespace(
                tradingClass="SPY", expirations={selected_expiry, later_expiry},
                strikes={700.0, 750.0},
            )]

        def placeOrder(self, contract, order):
            order.orderId = 91
            order.permId = 901
            self.placed = (contract, order)
            return SimpleNamespace(
                order=order,
                orderStatus=SimpleNamespace(
                    status="Submitted", filled=0, avgFillPrice=0,
                ),
            )

        def sleep(self, _seconds):
            return None

    ib = FakeIB()
    payload = _payload(execute_date="2099-08-21", min_dte=30)
    payload["dynamic_selection"] = True
    assert executor._do_dynamic_option_market(ib, payload, "primary") == 0
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["ok"] is True
    assert result["state"] == "executed"
    assert result["fill"]["quantity"] == 8
    contract, order = ib.placed
    assert contract.secType == "OPT"
    assert contract.strike == 700.0
    assert order.orderType == "MKT"
    assert order.totalQuantity == 8


def test_signed_command_handler_persists_and_cancels_schedule(
        modules, monkeypatch, tmp_path):
    agent, _executor = modules
    now = dt.datetime.now(agent._ET)
    execute_at = now.replace(hour=15, minute=45, second=0, microsecond=0)
    if execute_at <= now:
        execute_at += dt.timedelta(days=1)
    while execute_at.weekday() >= 5:
        execute_at += dt.timedelta(days=1)
    payload = _payload(
        execute_date=execute_at.date().isoformat(),
        execute_time=execute_at.strftime("%H:%M"),
    )
    state_path = tmp_path / "handler-schedules.json"
    monkeypatch.setattr(agent, "SCHEDULED_OPTIONS_PATH", str(state_path))
    monkeypatch.setattr(agent, "OPTION_ACCOUNTS", {"primary"})
    monkeypatch.setattr(agent, "UNCAPPED_OPTIONS_ACCOUNTS", set())
    monkeypatch.setattr(agent, "LIVE_ENABLED", True)
    monkeypatch.setattr(agent, "LIVE_ACCOUNTS", {"primary"})
    monkeypatch.setattr(agent, "LIVE_TYPES", {"option_spread"})
    monkeypatch.setattr(agent, "_verify", lambda *_args: True)
    monkeypatch.setattr(agent, "_record_seen", lambda *_args: None)
    monkeypatch.setitem(agent._BOOK, "book", {
        "accounts": [{"key": "primary", "positions": [], "nlv": 1_000_000}],
    })
    agent._SCHEDULES.clear()
    agent._SEEN.clear()

    class FakeWS:
        def __init__(self):
            self.messages = []

        async def send(self, message):
            self.messages.append(json.loads(message))

    ws = FakeWS()
    schedule_cmd = {
        "id": "schedule-handler-1", "type": "scheduled_option",
        "account": "primary", "dry_run": False, "payload": payload,
        "expires_at": (dt.datetime.now().timestamp() + 60) * 1000,
    }
    asyncio.run(agent._handle_command(ws, json.dumps(schedule_cmd), "ignored"))
    assert agent._SCHEDULES["schedule-handler-1"]["state"] == "scheduled"
    assert ws.messages[-1]["state"] == "scheduled"

    cancel_cmd = {
        "id": "schedule-cancel-1", "type": "scheduled_option_cancel",
        "account": "primary", "dry_run": False,
        "payload": {"schedule_id": "schedule-handler-1"},
        "expires_at": (dt.datetime.now().timestamp() + 60) * 1000,
    }
    asyncio.run(agent._handle_command(ws, json.dumps(cancel_cmd), "ignored"))
    assert agent._SCHEDULES["schedule-handler-1"]["state"] == "cancelled"
    assert any(m["id"] == "schedule-handler-1" and m["state"] == "cancelled"
               for m in ws.messages)
    assert ws.messages[-1]["id"] == "schedule-cancel-1"
    assert ws.messages[-1]["state"] == "cancelled"


def test_due_loop_executes_once_through_existing_option_spread_gate(
        modules, monkeypatch, tmp_path):
    agent, _executor = modules
    state_path = tmp_path / "due-schedules.json"
    monkeypatch.setattr(agent, "SCHEDULED_OPTIONS_PATH", str(state_path))
    monkeypatch.setattr(agent, "_live_eligible", lambda _cmd: (True, ""))
    monkeypatch.setattr(agent, "_journal_option", lambda *_args: None)
    calls = []

    async def fake_execute(cmd):
        calls.append(cmd)
        return {
            "ok": True, "state": "executed", "detail": "fake live success",
            "fill": {"order_id": 7, "filled": 8, "avg_fill": 1.21},
        }

    monkeypatch.setattr(agent, "_execute_live", fake_execute)
    now = dt.datetime.now().timestamp()
    agent._SCHEDULES.clear()
    agent._SCHEDULES["due-1"] = {
        "id": "due-1", "account": "primary", "payload": _payload(),
        "state": "scheduled", "created_at": now - 100,
        "execute_at": now - 1, "expires_at": now + 60,
    }

    class FakeWS:
        def __init__(self):
            self.messages = []

        async def send(self, message):
            self.messages.append(json.loads(message))

    ws = FakeWS()

    async def run_once():
        task = asyncio.create_task(agent._scheduled_option_loop(ws))
        await asyncio.sleep(0.05)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    asyncio.run(run_once())
    assert len(calls) == 1
    assert calls[0]["type"] == "option_spread"
    assert calls[0]["payload"]["dynamic_selection"] is True
    assert agent._SCHEDULES["due-1"]["state"] == "executed"
    assert any(m["id"] == "due-1" and m["state"] == "executed" for m in ws.messages)
