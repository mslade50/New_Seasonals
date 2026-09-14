"""Candidate functions plus actual IBKR value classes; never imports a live runtime."""
import ast
import asyncio
import copy
import datetime
import hashlib
import hmac
import json
import math
import os
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace as N
from zoneinfo import ZoneInfo

from ib_insync import (ComboLeg, Contract, Forex, Future, LimitOrder, MarketOrder,
                       Option, OrderStatus, Position, StopLimitOrder, StopOrder, Stock, Trade)

from broker_runtime import execution_contracts, execution_lifecycle, order_mutations, position_actions, position_action_agent

FIXTURE = Path(__file__).parent / "fixtures" / "execution_runtime"


def _never(*args, **kwargs):
    raise AssertionError("A test must inject the simulated broker boundary")


def _load(name):
    module = ModuleType("inert_" + name)
    env = module.__dict__
    env.update(globals())
    env.update(__name__="inert_" + name, __file__=str(FIXTURE / (name + "_core.py")),
        IB=_never, guarded_place_order=_never, guarded_cancel_order=_never,
        BrokerMutationBlocked=type("BrokerMutationBlocked", (Exception,), {}),
        OPT_COMBO_TICK=.05, OPT_COMM_PER_CONTRACT=.65,
        LIVE_ENABLED=False, LIVE_ACCOUNTS={"primary", "pa"}, LIVE_TYPES=set(),
        LIVE_MAX_QTY=100000, LIVE_MAX_NOTIONAL=1e7, LIVE_MAX_NOTIONAL_BY_ACCT={},
        LIVE_MAX_FUT_CONTRACTS=100, LIVE_MAX_OPT_CONTRACTS=100, LIVE_MAX_OPT_RISK=30000,
        LIVE_MAX_OPT_RISK_BY_ACCT={}, OPTION_ACCOUNTS={"primary"},
        UNCAPPED_FUTURES_ACCOUNTS=set(), UNCAPPED_OPTIONS_ACCOUNTS=set(),
        ALLOWED_FUTURES_EXCHANGES={"CME", "CBOT", "NYMEX", "COMEX"},
        RISK_ACK_BPS=50, TERMINAL={"Filled", "Cancelled", "ApiCancelled", "Inactive"},
        STRATEGY_REF_MAX=32, _ERRORS=[], _BENIGN=set(),
        SUPPORTED={"entry_bracket", "exit_attach", "close_only", "close_resize", "flatten", "option_spread", "add_to_position", "modify", "cancel"},
        DISABLED_UNSAFE_MUTATIONS={"trim_readd", "add_to_position", "modify", "cancel"},
        PORTS={"primary": ("fixture", 0, 7), "pa": ("fixture", 0, 7)},
        MAX_NOTIONAL=250000, MAX_RISK_PCT=.05, MAX_OPT_RISK=25000, MAX_OPT_CONTRACTS=100,
        _BOOK={"book": None}, _SPECS={"map": {}}, _SCHEDULES={}, _SEEN=[], _ET=ZoneInfo("America/New_York"),
        _THIS_DIR=str(FIXTURE), _DIR=str(FIXTURE), STATUS_TOKEN="fixture-only",
        SCHEDULED_OPTIONS_PATH=str(FIXTURE / "must-not-write-schedules.json"),
        OPTIONS_JOURNAL_PATH=str(FIXTURE / "must-not-write-journal.jsonl"),
        _exclusive_file_lock=lambda *a, **k: nullcontext(),
        _blank_account_is_exact=lambda ib, account: getattr(ib, "managedAccounts", lambda: [])() == [account],
        _fresh_open_trades=lambda ib: ib.openTrades(), _cluster_symbol=lambda c: None,
        _POSITION_EXEC_LOCK=None, EXECUTE_TIMEOUT_S=120)
    source = (FIXTURE / (name + "_core.py")).read_text(encoding="utf-8")
    exec(compile(source, str(FIXTURE / (name + "_core.py")), "exec"), env)
    return module


def load_executor():
    return _load("execute_order")


def load_agent():
    return _load("exec_agent")


def install_helpers(monkeypatch):
    for module in (execution_contracts, execution_lifecycle, order_mutations, position_actions, position_action_agent):
        monkeypatch.setitem(sys.modules, module.__name__.split(".")[-1], module)
    monkeypatch.setitem(sys.modules, "futures_sizing", N(
        get_spec=lambda symbol: N(symbol=symbol, exchange="CME", multiplier=5, min_tick=.25),
        snap_to_tick=lambda price, tick: round(round(price / tick) * tick, 8)))


class SimBroker:
    """Simulated transport using immutable IBKR Position and native Order/Trade classes."""
    def __init__(self, positions=(), status="Submitted", fill=0):
        self.holdings = list(positions)
        self.trades = []
        self.sent = []
        self.cancelled = []
        self.status, self.fill = status, fill
        self.sequence = 1000
        self.client = N(getReqId=self.next_id)
        self.on_cancel = None
        self.on_sleep = None
        self.sleep_count = 0

    def next_id(self):
        self.sequence += 1
        return self.sequence

    def qualifyContracts(self, contract):
        if not contract.conId:
            contract.conId = int(contract.strike) + (10000 if contract.right == "P" else 20000) if contract.secType == "OPT" else 42
        if contract.secType == "FUT":
            contract.exchange = contract.exchange or "CME"
            contract.multiplier = contract.multiplier or "5"
        elif contract.secType == "OPT":
            contract.multiplier = "100"
            contract.currency = "USD"
        elif contract.secType == "CASH":
            contract.exchange = "IDEALPRO"
        return [contract]

    def reqContractDetails(self, contract):
        return [N(contract=contract, minTick=.25 if contract.secType == "FUT" else .00005 if contract.secType == "CASH" else .01)]

    def reqTickers(self, contract):
        price = 1.1 if contract.secType == "CASH" else 100
        return [N(marketPrice=lambda: price, ask=price, bid=price, last=price, close=price)]

    def reqPositions(self): return self.holdings
    def positions(self): return self.holdings
    def reqAllOpenOrders(self): return self.openTrades()
    def openTrades(self): return [t for t in self.trades if not t.isDone()]
    def reqCompletedOrders(self, apiOnly=False): return [t for t in self.trades if t.isDone()]
    def managedAccounts(self): return ["PRIMARY", "PA"]
    def accountSummary(self): return [N(account=a, tag="NetLiquidation", value="1000000") for a in self.managedAccounts()]

    def sleep(self, _):
        self.sleep_count += 1
        if self.on_sleep: self.on_sleep(self)

    def place(self, ib, contract, order, **kwargs):
        assert ib is self
        assert order.account in self.managedAccounts()
        if kwargs.get("account") is not None:
            assert kwargs["account"] == order.account
        assert contract.exchange
        if not order.orderId: order.orderId = self.next_id()
        order.clientId = 7
        order.permId = order.permId or order.orderId + 10000
        existing = next((t for t in self.trades if t.order.orderId == order.orderId), None)
        self.sent.append((copy.deepcopy(contract), copy.deepcopy(order), dict(kwargs)))
        if existing:
            existing.order = order
            existing.orderStatus.remaining = order.totalQuantity - existing.orderStatus.filled
            return existing
        filled = min(self.fill, order.totalQuantity)
        trade = Trade(contract, order, OrderStatus(status=self.status, filled=filled, remaining=order.totalQuantity-filled, avgFillPrice=100))
        self.trades.append(trade)
        if filled:
            for index, position in enumerate(self.holdings):
                if position.account == order.account and position.contract.conId == contract.conId:
                    self.holdings[index] = position._replace(position=position.position + (filled if order.action == "BUY" else -filled))
        return trade

    def cancel(self, ib, order):
        self.cancelled.append(order.orderId)
        trade = next(t for t in self.trades if t.order.orderId == order.orderId)
        trade.orderStatus.status = "Cancelled"
        if self.on_cancel: self.on_cancel(self, trade)


def position(asset="STK", account="PRIMARY", quantity=100):
    contract = (Stock("TEST", "", "USD", conId=42) if asset == "STK" else
                Future("MES", "203012", "", conId=42, multiplier="5", currency="USD", tradingClass="MES") if asset == "FUT" else
                Forex("EURUSD", conId=42))
    return Position(account, contract, quantity, 100)


def bind(executor, broker, tmp_path):
    executor.guarded_place_order = broker.place
    executor.guarded_cancel_order = broker.cancel
    executor.POSITION_ACTION_STATE_DIR = tmp_path
    executor._fut_trading_class = lambda symbol: symbol
    return executor
