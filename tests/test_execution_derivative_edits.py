"""Actual topology math plus inert exact-order broker callbacks; no networking."""
import ast
import copy
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace as N

import pytest

from broker_runtime import execution_lifecycle as life
from broker_runtime import order_mutations
from tests.test_execution_repair_dispatch import edit_fixture


def edit(broker, ns, payload, **changes):
    return life.mutate_one(ns, broker, dict(payload, **changes), "", 0, 7,
                           modify=True, account_key="primary")


def market_fixture(tmp_path):
    broker, ns, payload = edit_fixture(tmp_path)
    order = broker.orders[0].order
    order.orderType, order.lmtPrice, order.auxPrice = "MKT", 0, 0
    order.goodAfterTime, order.outsideRth = "20260925 15:59:00 America/New_York", True
    quotes = []
    def quote(contract):
        quotes.append(contract.conId)
        return [N(contract=copy.deepcopy(contract), marketDataType=1,
                  time=datetime.now(timezone.utc), ask=100)]
    broker.reqTickers = quote
    return broker, ns, payload, quotes


def test_scheduled_market_exit_increase_has_fresh_price_and_preserves_timing(tmp_path):
    broker, ns, payload, quotes = market_fixture(tmp_path)
    payload.pop("mutation_kind")  # The direct Save ticket supplies no qualifiers.
    result = edit(broker, ns, payload, new_qty=90)
    assert result["ok"], result
    assert quotes == [42] and len(broker.mutations) == 1
    assert broker.orders[0].order.goodAfterTime == "20260925 15:59:00 America/New_York"
    assert broker.orders[0].order.outsideRth and broker.orders[0].order.orderType == "MKT"


@pytest.mark.parametrize("failure", ["empty", "wrong_contract", "stale", "delayed", "nan", "exception", "notional", "holdings"])
def test_market_exit_increase_rejects_missing_quote_or_capacity_before_send(tmp_path, failure):
    broker, ns, payload, _ = market_fixture(tmp_path)
    original = broker.reqTickers
    def quote(contract):
        if failure == "exception": raise TimeoutError("quote unavailable")
        if failure == "empty": return []
        rows = original(contract)
        if failure == "wrong_contract": rows[0].contract.conId = 43
        if failure == "stale": rows[0].time -= timedelta(minutes=1)
        if failure == "delayed": rows[0].marketDataType = 3
        if failure == "nan": rows[0].ask = float("nan")
        return rows
    broker.reqTickers = quote
    if failure == "notional": ns["_max_notional"] = lambda _: 8999
    result = edit(broker, ns, payload, new_qty=101 if failure == "holdings" else 90)
    assert result["state"] == "rejected" and not broker.mutations, result


@pytest.mark.parametrize("typ", ["LMT", "STP", "MKT"])
def test_existing_price_edits_and_market_reductions_need_no_quote(tmp_path, typ):
    broker, ns, payload = edit_fixture(tmp_path)
    broker.orders[0].order.orderType = typ
    broker.reqTickers = lambda _: pytest.fail("unexpected quote")
    result = edit(broker, ns, payload, new_qty=60 if typ == "MKT" else 90)
    assert result["ok"], result


@pytest.mark.parametrize("uncapped,expected", [(True, True), (False, False)])
def test_futures_increase_uses_existing_account_policy(tmp_path, uncapped, expected):
    broker, ns, payload, quotes = market_fixture(tmp_path)
    broker.orders[0].contract.secType = "FUT"
    broker.orders[0].contract.multiplier = "1000"
    broker.orders[0].order.totalQuantity = 1
    ns["_uncapped_futures"] = lambda _: uncapped
    ns["LIVE_MAX_QTY"], ns["LIVE_MAX_FUT_CONTRACTS"] = 1, 2
    result = edit(broker, ns, payload, new_qty=3)
    assert result["ok"] is expected, result
    assert not quotes  # uncapped has no stock cap; capped fails contract cap first
    assert len(broker.mutations) == int(expected)


@pytest.mark.parametrize("base,currency,cap,expected", [
    ("USD", "JPY", 90, True), ("USD", "JPY", 89, False),
    ("EUR", "USD", 108, True), ("EUR", "USD", 107, False), ("EUR", "JPY", 100000, False)])
@pytest.mark.parametrize("typ", ["MKT", "LMT"])
def test_cash_increase_uses_base_units_and_exact_usd_pair(tmp_path, base, currency, cap, expected, typ):
    broker, ns, payload, quotes = market_fixture(tmp_path)
    c, order = broker.orders[0].contract, broker.orders[0].order
    c.secType, c.symbol, c.currency = "CASH", base, currency
    order.orderType, order.lmtPrice = typ, 150 if base == "USD" else 1.2
    quote = broker.reqTickers
    def price(contract):
        rows = quote(contract)
        rows[0].ask = 1.2
        return rows
    broker.reqTickers = price
    ns["_max_notional"] = lambda _: cap
    result = edit(broker, ns, payload, new_qty=90)
    assert result["ok"] is expected and len(broker.mutations) == int(expected), result
    assert quotes == ([42] if base == "EUR" and currency == "USD" and typ == "MKT" else [])


def option_fixture(tmp_path, *, combo=True, reverse=False, action="BUY", right="C"):
    broker, ns, payload = edit_fixture(tmp_path)
    path = Path(os.environ.get("EXECUTION_CORE_FIXTURE", str(
        Path(__file__).parent / "fixtures/execution_runtime/execute_order_core.py")))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in {"option_combo_risk", "_trusted_option_topology"}]
    assert len(nodes) == 2
    env = dict(OPT_COMM_PER_CONTRACT=.65)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "reviewed-option-topology", "exec"), env)
    ns.update(OPTION_ACCOUNTS={"primary", "pa"}, LIVE_MAX_OPT_CONTRACTS=10,
              _max_opt_risk=lambda _: 2500, _uncapped_options=lambda _: False,
              OPT_COMBO_TICK=.05, Contract=lambda **kw: N(**kw),
              _trusted_option_topology=env["_trusted_option_topology"])
    # Stock caps must never be used by this branch.
    ns["LIVE_MAX_QTY"] = 1
    ns["_max_notional"] = lambda _: pytest.fail("stock cap used for an option")
    def contract(cid, strike):
        return N(conId=cid, secType="OPT", symbol="TEST", tradingClass="TEST", currency="USD",
                 multiplier="100", strike=strike, right=right, exchange="SMART",
                 lastTradeDateOrContractMonth="20261016")
    contracts = {101: contract(101, 100 if right == "C" else 105),
                 102: contract(102, 105 if right == "C" else 100),
                 42: contract(42, 100)}
    original = broker.orders[0]
    if combo:
        original.contract = N(conId=42, secType="BAG", symbol="TEST", currency="USD", exchange="SMART",
            multiplier="", comboLegs=[N(conId=101,ratio=1,action="SELL" if reverse else "BUY",exchange="SMART"),
                                      N(conId=102,ratio=1,action="BUY" if reverse else "SELL",exchange="SMART")])
    else:
        original.contract = copy.deepcopy(contracts[42])
        broker.position.contract = copy.deepcopy(contracts[42])
    original.order.orderType, original.order.totalQuantity = "LMT", 2
    original.order.action, original.order.lmtPrice = action, -1 if reverse else 1
    original.order.goodAfterTime = "20260925 09:35:00 America/New_York"
    original.order.outsideRth = True
    broker.position.position = 10
    broker.reqContractDetails = lambda c: [N(contract=contracts[c.conId],validExchanges="SMART",marketRuleIds="26")]
    broker.reqMarketRule = lambda _: [N(lowEdge=0,increment=.05),N(lowEdge=3,increment=.10)]
    original_place, placed = ns["guarded_place_order"], []
    def place(*args, **kwargs):
        placed.append(kwargs)
        return original_place(*args, **kwargs)
    ns["guarded_place_order"] = place
    payload.pop("mutation_kind")
    return broker, ns, payload, contracts, placed


@pytest.mark.parametrize("action", ["BUY", "SELL"])
@pytest.mark.parametrize("right", ["C", "P"])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("field", ["quantity", "price"])
def test_exact_signed_vertical_edits_derive_real_risk(tmp_path, action, right, reverse, field):
    broker, ns, payload, _, placed = option_fixture(tmp_path, action=action, right=right, reverse=reverse)
    before = copy.deepcopy(broker.orders[0])
    changes = dict(new_qty=3) if field == "quantity" else dict(new_limit=-1.2 if reverse else 1.2)
    result = edit(broker, ns, payload, **changes)
    assert result["ok"], result
    order = broker.orders[0].order
    canonical_action = ("SELL" if action == "BUY" else "BUY") if reverse else action
    unit_risk = abs(order.lmtPrice) if canonical_action == "BUY" else 5-abs(order.lmtPrice)
    assert placed[0]["risk_usd"] == pytest.approx(unit_risk*100*order.totalQuantity + 2.6*order.totalQuantity)
    base = "long" if right == "C" else "short"
    assert placed[0]["portfolio_direction"] == (base if canonical_action == "BUY" else "short" if base == "long" else "long")
    assert broker.orders[0].contract == before.contract and order.goodAfterTime == before.order.goodAfterTime
    assert order.outsideRth and order.action == before.order.action and len(broker.mutations) == 1


@pytest.mark.parametrize("right", ["C", "P"])
@pytest.mark.parametrize("action", ["BUY", "SELL"])
def test_single_option_edit_uses_multiplier_and_covered_exit_direction(tmp_path, action, right):
    broker, ns, payload, _, placed = option_fixture(tmp_path, combo=False, action=action, right=right)
    result = edit(broker, ns, payload, new_qty=3)
    assert result["ok"], result
    assert placed[0]["risk_usd"] == pytest.approx(303.9)
    assert placed[0]["portfolio_direction"] == ("long" if right == "C" else "short")
    assert placed[0]["mutation_kind"] == ("entry" if action == "BUY" else "exit")


@pytest.mark.parametrize("failure", ["qty_cap", "risk_cap", "wrong_id", "multiplier", "ratio", "expiry", "right", "symbol", "non_guaranteed", "zero", "sign", "width", "tick", "legs", "disabled", "nlv"])
def test_option_invalid_risk_or_topology_rejects_before_send(tmp_path, failure):
    broker, ns, payload, contracts, placed = option_fixture(tmp_path, reverse=True)
    changes = dict(new_qty=3)
    if failure == "qty_cap": ns["LIVE_MAX_OPT_CONTRACTS"] = 2
    if failure == "risk_cap": ns["_max_opt_risk"] = lambda _: 1000
    if failure == "wrong_id": contracts[101].conId = 999
    if failure == "multiplier": contracts[101].multiplier = "10"
    if failure == "ratio": broker.orders[0].contract.comboLegs[0].ratio = 2
    if failure == "expiry": contracts[101].lastTradeDateOrContractMonth = "20261120"
    if failure == "right": contracts[101].right = "P"
    if failure == "symbol": contracts[101].symbol = "OTHER"
    if failure == "non_guaranteed": broker.orders[0].order.smartComboRoutingParams = [N(tag="NonGuaranteed",value="1")]
    if failure in {"zero", "sign", "width", "tick"}: changes = dict(new_limit={"zero":0,"sign":1,"width":-5,"tick":-1.03}[failure])
    if failure == "legs": broker.orders[0].contract.comboLegs = []
    if failure == "disabled": ns["OPTION_ACCOUNTS"] = set()
    if failure == "nlv":
        ns["_uncapped_options"], ns["_nlv"] = lambda _: True, lambda *a: None
    result = edit(broker, ns, payload, **changes)
    assert result["state"] == "rejected" and not broker.mutations and not placed, result


def test_reducing_existing_option_risk_needs_no_extra_qualifiers(tmp_path):
    broker, ns, payload, _, _ = option_fixture(tmp_path)
    ns["_uncapped_options"] = lambda _: True
    ns["_nlv"] = lambda *a: pytest.fail("reduction requested NLV")
    assert edit(broker, ns, payload, new_qty=1)["ok"]


def test_credit_price_decrease_checks_greater_max_loss(tmp_path):
    broker, ns, payload, _, _ = option_fixture(tmp_path, action="SELL")
    ns["_max_opt_risk"] = lambda _: 850
    result = edit(broker, ns, payload, new_limit=.5)
    assert result["state"] == "rejected" and not broker.mutations, result


@pytest.mark.parametrize("held", [0, -10, 1])
def test_uncovered_single_option_sell_cannot_be_modified(tmp_path, held):
    broker, ns, payload, _, _ = option_fixture(tmp_path, combo=False, action="SELL")
    broker.position.position = held
    result = edit(broker, ns, payload, new_limit=1.1)
    assert result["state"] == "rejected" and not broker.mutations, result


@pytest.mark.parametrize("combo", [False, True])
def test_option_cancel_does_not_require_supported_edit_topology(tmp_path, combo):
    broker, ns, payload, _, _ = option_fixture(tmp_path, combo=combo)
    broker.orders[0].contract.comboLegs = []
    broker.reqContractDetails = lambda _: pytest.fail("cancel requested option details")
    result = life.mutate_one(ns, broker, payload, "", 0, 7, account_key="primary")
    assert result["ok"] and broker.mutations == [("cancel",1,2)]


def test_single_option_new_price_respects_exact_market_rule(tmp_path):
    broker, ns, payload, _, _ = option_fixture(tmp_path, combo=False)
    result = edit(broker, ns, payload, new_limit=3.05)
    assert result["state"] == "rejected" and not broker.mutations, result
    assert edit(broker, ns, payload, new_limit=3.10)["ok"]


def test_uncertain_option_edit_is_journaled_without_retry(tmp_path):
    broker, ns, payload, _, _ = option_fixture(tmp_path, reverse=True)
    broker.fail_resize = True
    payload = dict(payload, new_qty=3)
    first = order_mutations.run(ns, broker, payload, "", 0, 7, modify=True, account_key="primary")
    second = order_mutations.run(ns, broker, payload, "", 0, 7, modify=True, account_key="primary")
    assert first["state"] == "unknown" and second["state"] == "rejected"
    assert "reconcil" in second["detail"].lower()
    assert len(broker.mutations) == 1
