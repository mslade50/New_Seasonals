"""Stop-optional entry_bracket + exit_attach + the 2xATR risk_ack gate (2026-07-27).

The live executor/agent live in OneDrive and are absent in CI, so these
regression tests skip when that local execution checkout is unavailable.
"""
import json
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


@pytest.fixture()
def book(agent):
    """One primary account with a bare long AAPL position; restored after."""
    prev = agent._BOOK["book"]
    agent._BOOK["book"] = {"accounts": [{
        "key": "primary", "nlv": 100000,
        "positions": [{"symbol": "AAPL", "sec_type": "STK", "con_id": 111,
                       "position": 100, "avg_cost": 200.0, "market_price": 205.0}],
        "orders": [],
    }]}
    yield agent._BOOK["book"]
    agent._BOOK["book"] = prev


def _cmd(t, payload, account="primary"):
    return {"type": t, "account": account, "payload": payload}


# ---------------- agent: entry_bracket with no stop ----------------

def test_agent_accepts_stopless_entry(agent, book):
    ok, reasons = agent._validate(_cmd("entry_bracket", {
        "symbol": "USO", "sec_type": "STK", "action": "BUY",
        "quantity": 10, "entry": 50, "stop": None, "target": None}))
    assert ok, reasons


def test_agent_still_rejects_bad_stop_and_ordering(agent, book):
    ok, reasons = agent._validate(_cmd("entry_bracket", {
        "symbol": "USO", "sec_type": "STK", "action": "BUY",
        "quantity": 10, "entry": 50, "stop": 0}))
    assert not ok and any("stop must be > 0" in r for r in reasons)
    ok, reasons = agent._validate(_cmd("entry_bracket", {
        "symbol": "USO", "sec_type": "STK", "action": "BUY",
        "quantity": 10, "entry": 50, "stop": 55}))
    assert not ok and any("stop < entry" in r for r in reasons)
    # no stop but a wrong-side target is still rejected. The reason reads
    # "worst fill < target" since the STP_LMT work (2026-08-18) -- for every
    # single-price entry type the worst fill IS the entry, so this is the same
    # rule, reworded.
    ok, reasons = agent._validate(_cmd("entry_bracket", {
        "symbol": "USO", "sec_type": "STK", "action": "BUY",
        "quantity": 10, "entry": 50, "stop": None, "target": 45}))
    assert not ok and any("worst fill < target" in r for r in reasons)


def test_agent_nlv_risk_gate_only_with_stop(agent, book):
    # a stopped entry over 5% NLV risk is rejected; the same size with no stop
    # passes validation (the executor's ATR gate owns the unprotected case)
    ok, reasons = agent._validate(_cmd("entry_bracket", {
        "symbol": "USO", "sec_type": "STK", "action": "BUY",
        "quantity": 200, "entry": 50, "stop": 20}))
    assert not ok and any("% of NLV" in r for r in reasons)
    ok, reasons = agent._validate(_cmd("entry_bracket", {
        "symbol": "USO", "sec_type": "STK", "action": "BUY",
        "quantity": 200, "entry": 50, "stop": None}))
    assert ok, reasons


def test_agent_preview_marks_unprotected(agent, book):
    pv = agent._preview(_cmd("entry_bracket", {
        "symbol": "USO", "sec_type": "STK", "action": "BUY",
        "quantity": 10, "entry": 50, "stop": None, "target": None}))
    assert any("UNPROTECTED" in leg for leg in pv["legs"])
    assert not any(leg.startswith("STOP") for leg in pv["legs"])
    assert "NO STOP" in pv["summary"]
    # stopped preview unchanged (regression)
    pv = agent._preview(_cmd("entry_bracket", {
        "symbol": "USO", "sec_type": "STK", "action": "BUY",
        "quantity": 10, "entry": 50, "stop": 48, "target": 56}))
    assert any(leg.startswith("STOP") for leg in pv["legs"])
    assert "R:R" in pv["summary"]


# ---------------- agent: exit_attach ----------------

def test_agent_exit_attach_valid(agent, book):
    ok, reasons = agent._validate(_cmd("exit_attach", {
        "symbol": "AAPL", "sec_type": "STK", "con_id": 111, "stop": 190}))
    assert ok, reasons
    pv = agent._preview(_cmd("exit_attach", {
        "symbol": "AAPL", "sec_type": "STK", "con_id": 111,
        "stop": 190, "target": 230, "time_stop": "2026-08-14"}))
    assert len(pv["legs"]) == 3
    assert all("SELL 100" in leg for leg in pv["legs"])
    assert "LONG 100" in pv["summary"]


def test_agent_exit_attach_needs_a_leg(agent, book):
    ok, reasons = agent._validate(_cmd("exit_attach", {
        "symbol": "AAPL", "sec_type": "STK", "con_id": 111}))
    assert not ok and any("at least one" in r for r in reasons)


def test_agent_exit_attach_wrong_side_prices(agent, book):
    ok, reasons = agent._validate(_cmd("exit_attach", {
        "symbol": "AAPL", "sec_type": "STK", "con_id": 111, "stop": 210}))
    assert not ok and any("wrong side" in r for r in reasons)
    ok, reasons = agent._validate(_cmd("exit_attach", {
        "symbol": "AAPL", "sec_type": "STK", "con_id": 111, "target": 195}))
    assert not ok and any("wrong side" in r for r in reasons)
    ok, reasons = agent._validate(_cmd("exit_attach", {
        "symbol": "AAPL", "sec_type": "STK", "con_id": 111,
        "stop": 230, "target": 190}))
    assert not ok and any("stop < target" in r for r in reasons)


def test_agent_exit_attach_rejects_working_orders(agent, book):
    book["accounts"][0]["orders"] = [{
        "symbol": "AAPL", "sec_type": "STK", "con_id": 111, "action": "SELL",
        "order_type": "STP", "qty": 100, "aux": 195, "status": "Submitted"}]
    ok, reasons = agent._validate(_cmd("exit_attach", {
        "symbol": "AAPL", "sec_type": "STK", "con_id": 111, "stop": 190}))
    assert not ok and any("already working" in r for r in reasons)
    book["accounts"][0]["orders"] = [{
        "symbol": "AAPL", "sec_type": "STK", "con_id": 111, "action": "BUY",
        "order_type": "LMT", "qty": 50, "lmt": 198, "status": "Submitted"}]
    ok, reasons = agent._validate(_cmd("exit_attach", {
        "symbol": "AAPL", "sec_type": "STK", "con_id": 111, "stop": 190}))
    assert not ok and any("entry/add order already working" in r for r in reasons)


def test_agent_exit_attach_no_position(agent, book):
    ok, reasons = agent._validate(_cmd("exit_attach", {
        "symbol": "TSLA", "sec_type": "STK", "stop": 100}))
    assert not ok and any("no open position" in r for r in reasons)


def test_agent_exit_attach_bad_time_stop(agent, book):
    ok, reasons = agent._validate(_cmd("exit_attach", {
        "symbol": "AAPL", "sec_type": "STK", "con_id": 111, "time_stop": "soon"}))
    assert not ok and any("time_stop must be a date" in r for r in reasons)


# ---------------- executor: build_bracket with stop=None ----------------

def _next_id_gen():
    n = [100]
    def nid():
        n[0] += 1
        return n[0]
    return nid


def test_build_bracket_stopless_naked(executor):
    parent, children = executor.build_bracket("BUY", 10, 50.0, None, None, None, _next_id_gen())
    assert children == []
    assert parent.transmit is True          # nothing else releases a naked parent


def test_build_bracket_stopless_with_target(executor):
    parent, children = executor.build_bracket("BUY", 10, 50.0, None, 60.0, None, _next_id_gen())
    assert parent.transmit is False
    assert len(children) == 1
    assert children[0].orderType == "LMT" and children[0].lmtPrice == 60.0
    assert children[0].transmit is True


def test_build_bracket_with_stop_unchanged(executor):
    parent, children = executor.build_bracket(
        "BUY", 10, 50.0, 48.0, 60.0, None, _next_id_gen(), time_stop_gat="20260814 15:59:00")
    types = [c.orderType for c in children]
    assert types == ["LMT", "STP", "MKT"]
    assert children[-1].transmit is True and parent.transmit is False


def test_build_bracket_entry_parent_types(executor):
    for entry_type, expected, tif in (("LMT", "LMT", "DAY"), ("MKT", "MKT", "DAY"),
                                      ("MOO", "MKT", "OPG"), ("MOC", "MOC", "DAY")):
        parent, children = executor.build_bracket(
            "BUY", 10, 50.0, 48.0, 60.0, None, _next_id_gen(),
            entry_type=entry_type)
        assert parent.orderType == expected
        assert parent.tif == tif
        assert [c.orderType for c in children] == ["LMT", "STP"]


def test_agent_entry_parent_type_validation(agent, book):
    base = {"symbol": "AAPL", "sec_type": "STK", "action": "BUY",
            "quantity": 10, "entry": 200, "stop": 190, "target": 220}
    for entry_type in ("LMT", "MKT", "MOO", "MOC"):
        ok, reasons = agent._validate(_cmd(
            "entry_bracket", {**base, "entry_type": entry_type}))
        assert ok, reasons
    ok, reasons = agent._validate(_cmd(
        "entry_bracket", {**base, "entry_type": "MOC", "sec_type": "FUT"}))
    assert not ok and any("MOC entry supports stocks only" in r for r in reasons)
    ok, reasons = agent._validate(_cmd(
        "entry_bracket", {**base, "entry_type": "MOO", "sec_type": "CASH",
                          "symbol": "EUR", "currency": "USD"}))
    assert not ok and any("MOO entry does not support FX" in r for r in reasons)


# ---------------- executor: risk_ack gate pieces ----------------

def _fake_bars(n=30, base=100.0, rng=2.0):
    bars = []
    for i in range(n):
        px = base + (i % 3)
        bars.append(SimpleNamespace(high=px + rng / 2, low=px - rng / 2, close=px))
    return bars


class _FakeIB:
    def __init__(self, bars, nlv):
        self._bars, self._nlv = bars, nlv
    def reqHistoricalData(self, *a, **k):
        return self._bars
    def accountSummary(self, *a, **k):
        return [SimpleNamespace(tag="NetLiquidation", value=str(self._nlv))]


def test_atr_and_nlv_helpers(executor):
    ib = _FakeIB(_fake_bars(), 100000)
    atr, last = executor._atr_estimate(ib, None, "STK")
    assert atr and atr > 0 and last > 0
    assert executor._nlv(ib) == 100000.0
    # too little history -> no estimate
    atr, last = executor._atr_estimate(_FakeIB(_fake_bars(5), 1), None, "STK")
    assert atr is None and last is None


def test_supported_and_dispatchable(executor):
    assert "exit_attach" in executor.SUPPORTED
    assert executor.RISK_ACK_BPS == 50.0


def test_exit_attach_precontract_rejections(executor, capsys):
    # leg checks run before any IB access, so ib=None is safe here
    executor._do_exit_attach(None, {"symbol": "AAPL"}, "primary")
    out = json.loads(capsys.readouterr().out.strip())
    assert out["ok"] is False and "at least one" in out["detail"]
    executor._do_exit_attach(None, {"symbol": "AAPL", "stop": -5}, "primary")
    out = json.loads(capsys.readouterr().out.strip())
    assert out["ok"] is False and "must be > 0" in out["detail"]
    executor._do_exit_attach(None, {"symbol": "AAPL", "time_stop": "soon"}, "primary")
    out = json.loads(capsys.readouterr().out.strip())
    assert out["ok"] is False and "time_stop" in out["detail"]
