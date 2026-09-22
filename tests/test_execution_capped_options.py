import ast
import datetime
import math
import os
import sys
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace as N
from zoneinfo import ZoneInfo

import pytest

from broker_runtime import option_limit_pricing as pricing
from broker_runtime import prepare_execution_repairs as prepare
from broker_runtime.execution_contracts import select_front_details
from tests.spent_preparers import skip_execution_repairs


@pytest.mark.parametrize("budget,ask,tick", [(1000,2.03,.05),(1000,3.333,.01),(100,1,.01),
                                           (1000,.19,.05),(30000,11.07,.1)])
def test_rounding_never_exceeds_budget(budget,ask,tick):
    qty,limit=pricing.capped_size(budget,ask,tick)
    assert Decimal(str(limit))*qty*100 <= Decimal(str(budget))
    assert limit >= ask


@pytest.mark.parametrize("budget,ask,tick", [(99,1,.01),(1000,float("nan"),.01),
                                           (float("inf"),1,.01),(1000,1,0)])
def test_invalid_price_or_unaffordable_contract_is_rejected(budget,ask,tick):
    with pytest.raises(ValueError):
        pricing.capped_size(budget,ask,tick)


@pytest.mark.parametrize("ask,expected",[(2.999,3.0),(3.0,3.0),(3.001,3.05),(3.05,3.05),(3.051,3.1)])
def test_market_rule_band_boundaries_and_premium_cap(ask,expected):
    rules=[N(lowEdge=0,increment=.01),N(lowEdge=3,increment=.05)]
    bands=pricing.market_increments(N(reqMarketRule=lambda _:rules),N(exchange="SMART"),
                                   N(validExchanges="SMART",marketRuleIds="26"))
    qty,limit=pricing.capped_market_size(1000,ask,bands)
    assert limit==expected and limit>=ask
    assert Decimal(str(limit))*qty*100 <= 1000
    assert Decimal(str(limit))*(qty+1)*100 > 1000


def test_rounding_that_crosses_band_uses_new_band_grid():
    # 1.02 rounds past the boundary on the 0.05 grid; 1.03 is the next band start.
    bands=pricing.market_increments(
        N(reqMarketRule=lambda _:[N(lowEdge=0,increment=.05),N(lowEdge=1.03,increment=.1)]),
        N(exchange="SMART"),N(validExchanges="SMART",marketRuleIds="26"))
    assert pricing.capped_market_size(1000,1.02,bands)==(9,1.03)
    with pytest.raises(ValueError,match="budget"):
        pricing.capped_market_size(102.99,1.02,bands)


@pytest.mark.parametrize("exchanges,ids",[("CBOE","7"),("SMART,CBOE","26"),
                                        ("SMART,SMART","26,27"),("SMART",""),
                                        ("SMART","0"),("SMART","bad")])
def test_unresolved_exchange_rule_is_rejected_without_fallback(exchanges,ids):
    def should_not_request(_):
        raise AssertionError("ambiguous market rule must not be requested")
    with pytest.raises(ValueError,match="exchange market rule"):
        pricing.market_increments(N(reqMarketRule=should_not_request),N(exchange="SMART"),
                                  N(validExchanges=exchanges,marketRuleIds=ids,minTick=.01))


@pytest.mark.parametrize("rows",[
    [N(lowEdge=0,increment=float("nan"))],
    [N(lowEdge=1,increment=.01)],
    [N(lowEdge=0,increment=.01),N(lowEdge=0,increment=.05)],
    [N(lowEdge=0,increment=.01),N(lowEdge=3,increment=.05),N(lowEdge=2,increment=.1)],
])
def test_malformed_price_bands_fail_closed(rows):
    with pytest.raises(ValueError,match="market rule"):
        pricing.market_increments(N(reqMarketRule=lambda _:rows),N(exchange="SMART"),
                                  N(validExchanges="SMART",marketRuleIds="26"))


def test_old_market_intent_and_missed_window_stay_blocked():
    p=dict(pricing_policy=pricing.POLICY,order_type="LMT",tif="DAY",
           execute_date="2026-09-14",execute_time="15:45",grace_minutes=5)
    now=datetime.datetime(2026,9,14,15,46,tzinfo=ZoneInfo("America/New_York"))
    pricing.validate_intent(p,now)
    with pytest.raises(ValueError,match="legacy"):
        pricing.validate_intent(dict(p,order_type="MKT"),now)
    with pytest.raises(ValueError,match="window"):
        pricing.validate_intent(p,now+datetime.timedelta(minutes=10))


def test_mcl_delivery_month_is_not_last_trade_month():
    def detail(month,last,cid):
        return N(contractMonth=month,realExpirationDate=last,
                 contract=N(conId=cid,lastTradeDateOrContractMonth=last))
    rows=[detail("202609","20260818",1),detail("202610","20260918",2),detail("202611","20261019",3)]
    _,month,upcoming,last=select_front_details(rows,"20260911",5)
    assert month=="202610" and last=="20260918"
    assert upcoming==["202610","202611"]
    with pytest.raises(ValueError,match="unexpired"):
        select_front_details(rows,"20261101",5)


@pytest.fixture
def dynamic(monkeypatch):
    skip_execution_repairs()
    path=Path(os.environ.get("IBKR_REVIEW_SOURCE", "C:/Users/McKinley Slade/OneDrive/trading_ibkr"))/"execute_order.py"
    if not path.exists():
        pytest.skip("reviewed runtime required")
    source=prepare.patch_executor(path.read_text(encoding="utf-8-sig"))
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=="_do_dynamic_option_limit")
    row=dict(ask=1.00,strike=100,con_id=42,delta=.15)
    monkeypatch.setitem(sys.modules,"option_workbench",N(_quote_chain=lambda *a: (dict(strikes=[row]),None)))
    monkeypatch.setitem(sys.modules,"option_limit_pricing",pricing)
    monkeypatch.setattr(pricing,"validate_intent",lambda p: None)
    calls=[]
    contract=N(conId=42,multiplier="100",symbol="TEST",secType="OPT",exchange="SMART")
    quote=N(ask=2.03,marketDataType=1,contract=contract)
    def place(ib,c,o,**kw):
        calls.append((o,kw))
        return N(order=N(orderId=1,permId=2),orderStatus=N(status="Submitted",filled=0,avgFillPrice=0))
    env=dict(math=math,datetime=datetime,OPTION_ACCOUNTS={"primary"},LIVE_MAX_OPT_CONTRACTS=100,
        Stock=lambda *a:N(symbol="TEST",secType="STK",conId=1),
        Option=lambda *a,**k:contract,
        LimitOrder=lambda side,qty,limit:N(action=side,totalQuantity=qty,lmtPrice=limit,orderType="LMT"),
        _uncapped_options=lambda a:False,_max_opt_risk=lambda a:30000,
        _resolve_option_expiry=lambda *a:"20261016",_select_delta_row=lambda *a:row,
        _trusted_option_topology=lambda side,price,qty,*a:(dict(risk_usd=price*qty*100,portfolio_direction="long"),None),
        _command_signal=lambda *a:"fixture",_ERRORS=[],guarded_place_order=place,
        _out=lambda ok,state,detail,fill=None:dict(ok=ok,state=state,detail=detail,fill=fill))
    exec(compile(ast.Module(body=[node],type_ignores=[]),"dynamic-limit","exec"),env)
    def snapshot(c):
        if c.secType != "OPT":
            return [N(marketPrice=lambda:100,marketDataType=1)]
        quote.time=datetime.datetime.now(datetime.timezone.utc)
        return [quote]
    ib=N(qualifyContracts=lambda c:[c],reqMarketDataType=lambda *a:None,
         reqTickers=snapshot,
         reqSecDefOptParams=lambda *a:[N(tradingClass="TEST",expirations=["20261016"],strikes=[100])],
         reqContractDetails=lambda c:[N(contract=c,minTick=.01,validExchanges="CBOE,SMART",marketRuleIds="7,26")],
         reqMarketRule=lambda rule_id:[N(lowEdge=0,increment=.05)],sleep=lambda _:None)
    payload=dict(symbol="TEST",_broker_account="PRIMARY",right="P",target_delta=.15,
                 delta_tolerance=.03,premium_budget=1000,order_type="LMT",tif="DAY",expiry_mode="specific")
    return env,ib,payload,calls,quote


def test_executor_uses_fresh_ask_rounded_limit_and_guard_risk(dynamic):
    env,ib,p,calls,_=dynamic
    result=env["_do_dynamic_option_limit"](ib,p,"primary")
    assert result["ok"],result
    order,guard=calls[0]
    assert (order.orderType,order.totalQuantity,order.lmtPrice)==("LMT",4,2.05)
    assert order.account=="PRIMARY" and order.tif=="DAY"
    assert guard["risk_usd"]==pytest.approx(820)
    assert len(calls)==1


def test_delayed_option_quote_places_nothing(dynamic):
    env,ib,p,calls,quote=dynamic
    quote.marketDataType=3
    result=env["_do_dynamic_option_limit"](ib,p,"primary")
    assert not result["ok"] and calls==[]


def test_executor_uses_smart_premium_band_not_contract_minimum_tick(dynamic):
    env,ib,p,calls,quote=dynamic
    quote.ask=3.03
    requested=[]
    def market_rule(rule_id):
        requested.append(rule_id)
        return [N(lowEdge=0,increment=.01),N(lowEdge=3,increment=.05)]
    ib.reqMarketRule=market_rule
    result=env["_do_dynamic_option_limit"](ib,p,"primary")
    assert result["ok"],result
    assert requested==[26]
    order,guard=calls[0]
    assert (order.totalQuantity,order.lmtPrice)==(3,3.05)
    assert guard["risk_usd"]==pytest.approx(915)


@pytest.mark.parametrize("rules",[None,[],[N(lowEdge=0,increment=0)]])
def test_missing_or_invalid_market_rule_never_submits(dynamic,rules):
    env,ib,p,calls,_=dynamic
    ib.reqMarketRule=lambda _:rules
    result=env["_do_dynamic_option_limit"](ib,p,"primary")
    assert result["state"]=="rejected" and calls==[]


def test_stale_snapshot_never_submits(dynamic):
    env,ib,p,calls,quote=dynamic
    original=ib.reqTickers
    def stale(c):
        ticks=original(c)
        if c.secType=="OPT":
            quote.time-=datetime.timedelta(minutes=1)
        return ticks
    ib.reqTickers=stale
    result=env["_do_dynamic_option_limit"](ib,p,"primary")
    assert result["state"]=="rejected" and calls==[]


def test_snapshot_expiring_during_preflight_never_submits(dynamic):
    env,ib,p,calls,quote=dynamic
    topology=env["_trusted_option_topology"]
    def slow_preflight(*args):
        quote.time-=datetime.timedelta(seconds=31)
        return topology(*args)
    env["_trusted_option_topology"]=slow_preflight
    result=env["_do_dynamic_option_limit"](ib,p,"primary")
    assert result["state"]=="rejected" and calls==[]


def test_market_rule_rounded_contract_exceeds_budget_places_nothing(dynamic):
    env,ib,p,calls,quote=dynamic
    quote.ask=3.03
    p["premium_budget"]=304
    result=env["_do_dynamic_option_limit"](ib,p,"primary")
    assert result["state"]=="rejected" and calls==[]


def test_actual_ib_insync_shapes_use_market_rule_without_connection(dynamic):
    ib_types=pytest.importorskip("ib_insync")
    env,ib,p,calls,_=dynamic
    contract=ib_types.Option("TEST","20261016",100,"P","SMART",conId=42,multiplier="100")
    quote=ib_types.Ticker(contract=contract,marketDataType=1,ask=3.03)
    env["Option"]=lambda *args,**kwargs:contract
    original=ib.reqTickers
    def snapshot(c):
        if c.secType!="OPT":
            return original(c)
        quote.time=datetime.datetime.now(datetime.timezone.utc)
        return [quote]
    ib.reqTickers=snapshot
    ib.reqContractDetails=lambda c:[ib_types.ContractDetails(
        contract=c,minTick=.01,validExchanges="CBOE,SMART",marketRuleIds="7,26")]
    ib.reqMarketRule=lambda rule_id:[ib_types.PriceIncrement(0,.01),ib_types.PriceIncrement(3,.05)]
    result=env["_do_dynamic_option_limit"](ib,p,"primary")
    assert result["ok"],result
    assert calls[0][0].lmtPrice==3.05


def test_lost_submit_ack_is_unknown_never_clean_rejection(dynamic):
    env,ib,p,calls,_=dynamic
    def ambiguous(*a,**kw):
        calls.append(1)
        raise TimeoutError("lost reply")
    env["guarded_place_order"]=ambiguous
    result=env["_do_dynamic_option_limit"](ib,p,"primary")
    assert result["state"]=="unknown" and len(calls)==1
