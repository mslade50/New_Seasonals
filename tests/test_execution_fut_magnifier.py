"""Cents-quoted futures (IBKR priceMagnifier 100) in the executor and agent.

XK (mini soybeans) quotes 1280 cents with multiplier 1000: one contract is
$12,800 and a 1.00 move is $10. Magnifier-1 contracts (MES) must not change.
"""
import contextlib
import io
import json
from types import SimpleNamespace as N

import pytest

from tests.execution_harness import (SimBroker, bind, install_helpers, load_agent,
                                     load_executor)

XK_SPEC = N(symbol="YK", exchange="CBOT", multiplier=1000.0, min_tick=0.00125,
            price_magnifier=100)
STALE_XK_SPEC = N(symbol="YK", exchange="CBOT", multiplier=1000.0, min_tick=0.00125)


class CentsBroker(SimBroker):
    def __init__(self, magnifier=100, **kwargs):
        super().__init__(**kwargs)
        self.magnifier = magnifier

    def qualifyContracts(self, contract):
        contract.conId = contract.conId or 77
        contract.multiplier = "1000"
        return [contract]

    def reqContractDetails(self, contract):
        return [N(contract=contract, minTick=0.00125, priceMagnifier=self.magnifier)]


@pytest.fixture(autouse=True)
def helpers(monkeypatch):
    install_helpers(monkeypatch)
    monkeypatch.delenv("LIVE_FUTURES_NOTIONAL_EXEMPT_ACCOUNTS", raising=False)


def use_spec(monkeypatch, spec):
    import sys
    monkeypatch.setitem(sys.modules, "futures_sizing", N(
        get_spec=lambda symbol: spec,
        snap_to_tick=lambda price, tick: round(round(price / tick) * tick, 8)))


def xk_entry(**overrides):
    p = dict(_command_id="xk-entry", _broker_account="PRIMARY", symbol="XK", action="BUY",
             sec_type="FUT", currency="USD", quantity=2, entry=1280, stop=1260, target=1320,
             entry_type="LMT", fut_expiry="203011", exchange="CBOT",
             fut_ib_symbol="YK", fut_trading_class="XK",
             fut_multiplier=1000, fut_min_tick=0.00125, fut_price_magnifier=100)
    p.update(overrides)
    return p


def run_entry(executor, broker, p):
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        assert executor._do_entry_bracket(broker, p, "primary") == 0
    return json.loads(stream.getvalue())


def test_xk_notional_uses_dollars_per_point(tmp_path, monkeypatch):
    use_spec(monkeypatch, XK_SPEC)
    broker = CentsBroker()
    executor = bind(load_executor(), broker, tmp_path)
    executor.LIVE_MAX_NOTIONAL = 30000   # $25,600 true; the unscaled bug read $2.56M
    result = run_entry(executor, broker, xk_entry())
    assert result["ok"], result
    parent, take, stop = (row[1] for row in broker.sent[:3])
    assert parent.lmtPrice == 1280 and take.lmtPrice == 1320 and stop.auxPrice == 1260


def test_xk_notional_cap_reads_exact_dollars(tmp_path, monkeypatch):
    use_spec(monkeypatch, XK_SPEC)
    broker = CentsBroker()
    executor = bind(load_executor(), broker, tmp_path)
    executor.LIVE_MAX_NOTIONAL = 25000
    result = run_entry(executor, broker, xk_entry())
    assert result["state"] == "rejected" and "notional 25600 > 25000" in result["detail"]
    assert not broker.sent


def test_xk_prices_snap_to_the_cents_grid(tmp_path, monkeypatch):
    use_spec(monkeypatch, XK_SPEC)
    broker = CentsBroker()
    executor = bind(load_executor(), broker, tmp_path)
    result = run_entry(executor, broker, xk_entry(entry=1280.1, stop=1259.9, target=1320.06))
    assert result["ok"], result
    parent, take, stop = (row[1] for row in broker.sent[:3])
    assert (parent.lmtPrice, take.lmtPrice, stop.auxPrice) == (1280.125, 1320.0, 1259.875)


@pytest.mark.parametrize("spec,claimed", [(STALE_XK_SPEC, None), (XK_SPEC, 1)])
def test_magnifier_mismatch_with_ibkr_refuses(tmp_path, monkeypatch, spec, claimed):
    use_spec(monkeypatch, spec)
    broker = CentsBroker()
    executor = bind(load_executor(), broker, tmp_path)
    result = run_entry(executor, broker, xk_entry(fut_price_magnifier=claimed))
    assert result["state"] == "rejected"
    assert "price magnifier changed (1 vs IBKR 100)" in result["detail"]
    assert not broker.sent


def test_mes_dollar_math_unchanged(tmp_path):
    broker = SimBroker()
    executor = bind(load_executor(), broker, tmp_path)
    executor.LIVE_MAX_NOTIONAL = 999
    p = dict(xk_entry(), symbol="MES", exchange="CME", fut_ib_symbol="MES", fut_trading_class="MES", entry=100, stop=90, target=120,
             fut_multiplier=5, fut_min_tick=0.25, fut_price_magnifier=None)
    result = run_entry(executor, broker, p)
    assert result["state"] == "rejected" and "notional 1000 > 999" in result["detail"]


@pytest.fixture()
def agent():
    module = load_agent()
    module._SPECS["map"] = {"XK": XK_SPEC, "MES": N(multiplier=5.0)}
    module._BOOK["book"] = {"accounts": [{"key": "primary", "nlv": 100000,
                                          "positions": [], "orders": []}]}
    return module


def test_agent_leg_multiplier_is_dollars_per_point(agent):
    assert agent._leg_multiplier({"sec_type": "FUT", "symbol": "XK"}) == (10.0, None)
    assert agent._leg_multiplier({"sec_type": "FUT", "symbol": "MES"}) == (5.0, None)
    claimed = {"sec_type": "FUT", "symbol": "ZZZ", "fut_multiplier": 1000}
    assert agent._leg_multiplier({**claimed, "fut_price_magnifier": 100}) == (10.0, None)
    for missing in (None, 0, "x", -5):   # missing/invalid reads 1: over-states, never under
        assert agent._leg_multiplier({**claimed, "fut_price_magnifier": missing}) == (1000.0, None)


def test_agent_validate_xk_notional_and_risk(agent):
    ok, reasons = agent._validate({"type": "entry_bracket", "account": "primary",
                                   "payload": xk_entry(quantity=2)})
    assert ok, reasons
    ok, reasons = agent._validate({"type": "entry_bracket", "account": "primary",
                                   "payload": xk_entry(quantity=30)})   # $384k notional
    assert not ok and any("notional $384,000" in r for r in reasons)
