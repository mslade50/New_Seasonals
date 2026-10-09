"""Guard for risk_agent_ibkr: chain shape, key format, 0DTE, graceful failure, blindness.

No real IB: a fake IB object stands in for the private session.
"""
from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import risk_agent_ibkr as rai  # noqa: E402
from risk_agent_grammar import chain_quote_key  # noqa: E402

FORBIDDEN = ["positions", "portfolio", "accountValues", "accountSummary", "reqPositions",
             "openOrders", "reqExecutions", "placeOrder", "cancelOrder", "fills", "trades",
             "reqAccountUpdates"]


def test_source_never_names_account_or_order_methods():
    src = (ROOT / "risk_agent_ibkr.py").read_text(encoding="utf-8")
    hits = [w for w in FORBIDDEN if re.search(rf"\b{w}\b", src)]
    assert not hits, f"blindness guard: forbidden names in risk_agent_ibkr.py: {hits}"
    assert "ib.connect(" not in src and ".connect(" not in src.replace("connectAsync", "")


class FakeIB:
    def __init__(self, today):
        self.today = today
        self.n = 0

    def qualifyContracts(self, *cs):
        for c in cs:
            self.n += 1
            c.conId = 1000 + self.n
        return list(cs)

    def reqMarketDataType(self, t):
        pass

    def sleep(self, s):
        pass

    def cancelMktData(self, c):
        pass

    def reqMktData(self, c, *a):
        if c.secType == "STK":
            return SimpleNamespace(bid=99.9, ask=100.1, last=100.0, close=99.0, time=None,
                                   marketDataType=1, contract=c, marketPrice=lambda: 100.0)
        k = c.strike
        return SimpleNamespace(
            bid=1.0, ask=1.1, last=1.0, close=1.0, time=None, marketDataType=1, contract=c,
            bidSize=5, askSize=7, marketPrice=lambda: 1.05,
            modelGreeks=SimpleNamespace(impliedVol=0.2, delta=0.5, gamma=0.1, theta=-0.2, vega=0.3))

    def reqSecDefOptParams(self, sym, fut, sec, conid):
        exps = {(self.today + dt.timedelta(days=d)).strftime("%Y%m%d") for d in (0, 1, 7, 40)}
        return [SimpleNamespace(exchange="SMART", tradingClass="SPY", expirations=exps,
                                strikes={float(k) for k in range(80, 121)}),
                SimpleNamespace(exchange="CBOE", tradingClass="SPY", expirations=exps, strikes={1.0})]


@pytest.fixture
def fake(monkeypatch):
    today = dt.datetime.now(ZoneInfo("America/New_York")).date()
    ib = FakeIB(today)
    monkeypatch.setattr(rai, "_shared", lambda timeout=8: ib)
    return ib


def test_chain_shape_keys_and_0dte(fake):
    blk = rai.option_chain("SPY", min_dte=0, max_dte=14, max_strikes_per_expiry=10)
    assert blk["spot"] == 100.0 and blk["market_data_type"] == 1
    assert {"underlying", "spot_ts", "asof_utc", "quotes"} <= set(blk)
    today = fake.today
    want = [(today + dt.timedelta(days=d)).isoformat() for d in (0, 1, 7)]
    assert blk["expiries"] == want                     # 0DTE and weekly in, 40 DTE out
    for key, q in blk["quotes"].items():
        exp, strike, right = key.split("|")
        assert key == chain_quote_key(exp, float(strike), right)
        assert right in ("C", "P") and exp in want
        assert {"bid", "ask", "mid", "iv", "delta", "gamma", "theta", "vega", "con_id",
                "bid_size", "ask_size", "quote_ts"} <= set(q)
        assert q["ask"] == 1.1 and q["mid"] == pytest.approx(1.05)
    per_exp = {e: [k for k in blk["quotes"] if k.startswith(e)] for e in want}
    assert all(len(v) <= 20 for v in per_exp.values())           # 10 strikes x C/P
    assert chain_quote_key(want[0], 100.0, "C") in blk["quotes"]  # ATM kept


def test_unreachable_returns_none_and_chain_degrades(monkeypatch):
    monkeypatch.setenv("RISK_AGENT_IB_PORT", "1")
    monkeypatch.setattr(rai, "_SHARED", {"ib": None, "failed": False})
    assert rai.connect(timeout=2) is None
    assert rai.is_available() is False
    blk = rai.option_chain("SPY")
    assert blk["quotes"] == {} and blk["error"] == "ibkr_unavailable"
    assert rai.historical_bid_ask(1, "2026-10-09") is None
    assert rai.quote_contracts([1]) == {}


def test_merge_live_file_replaces_only_that_underlying(tmp_path):
    p = tmp_path / "live.json"
    rai.merge_live_file(p, {"underlying": "SPY", "quotes": {"a": 1}})
    rai.merge_live_file(p, {"underlying": "QQQ", "quotes": {"b": 2}})
    rai.merge_live_file(p, {"underlying": "SPY", "quotes": {"c": 3}})
    import json
    got = json.loads(p.read_text())
    assert got["SPY"]["quotes"] == {"c": 3} and got["QQQ"]["quotes"] == {"b": 2}
