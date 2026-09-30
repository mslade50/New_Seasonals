"""Real local handoff protocol with inert IB dataclasses; never connects to IBKR."""
import asyncio
import copy
from dataclasses import asdict
import json
import threading
from types import SimpleNamespace as N

import pytest
from ib_insync import Contract, Order, OrderStatus, Trade

from broker_runtime import execution_lifecycle as life
from broker_runtime import manual_order_actions as manual
from broker_runtime import owner_connection as owner
from broker_runtime import position_actions as actions
from tests.test_unified_position_actions import Broker, namespace, request


def exit_trade(oid, typ="STP", side="SELL"):
    return Trade(Contract(conId=42, symbol="MES", secType="FUT", exchange="CME",
                          lastTradeDateOrContractMonth="20261218", multiplier="5"),
                 Order(account="PRIMARY", clientId=927481, orderId=oid, permId=100+oid,
                       action=side, orderType=typ, totalQuantity=7, auxPrice=7700,
                       tif="GTC", ocaGroup="daytrade", ocaType=2, outsideRth=True,
                       goodAfterTime="20260930 14:55:00 America/New_York" if typ=="MKT" else "",
                       orderRef="MES|BUY|OpenBreakout|2026-09-30|ES-1-TIME"),
                 OrderStatus(status="PreSubmitted", filled=0, remaining=7))


class NativeOwner:
    def __init__(self, broker):
        self.broker = broker
        self.wrapper = N(openOrder=lambda *a: None)
        self.wires = []

    async def reqOpenOrdersAsync(self):
        for trade in self.trades():
            if trade.orderStatus.status not in life.TERMINAL:
                self.wrapper.openOrder(trade.order.orderId, trade.contract, trade.order, N())
        return self.trades()

    def trades(self):
        return [t for t in self.broker.orders if t.order.clientId == 927481]

    def placeOrder(self, contract, order):
        self.wires.append(("modify", order.orderId, order.totalQuantity))
        trade = next(t for t in self.trades() if t.order.permId == order.permId)
        trade.order = copy.deepcopy(order)
        trade.orderStatus.remaining = order.totalQuantity - trade.orderStatus.filled
        return trade

    def cancelOrder(self, order):
        self.wires.append(("cancel", order.orderId, order.totalQuantity))
        for trade in self.trades():
            if trade.order.ocaGroup == order.ocaGroup:
                trade.orderStatus.status = "Cancelled"


@pytest.fixture
def active_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("EXECUTION_OWNER_REGISTRY", str(tmp_path / "owners"))
    broker = Broker(holdings=7, exits=[exit_trade(1), exit_trade(2,"MKT")])
    broker.position.contract = copy.deepcopy(broker.orders[0].contract)
    native = NativeOwner(broker)
    callbacks = []
    config = N(host="fixture",port=0,client_id=927481,account="PRIMARY",mode="live",
               authorize=lambda session: None)
    loop = asyncio.new_event_loop()
    transport = N(ib=native,config=config,session="2026-09-30",snapshot_lock=asyncio.Lock())
    server = owner.OwnerServer(transport, lambda t: callbacks.append(owner.identity(t)))
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    asyncio.run_coroutine_threadsafe(server.start(),loop).result(5)
    def occupied(*a, **kw):
        raise TimeoutError()  # The exact blank exception from yesterday.
    ns=namespace(tmp_path / "journal",broker)
    ns["IB"]=lambda: N(connect=occupied,disconnect=lambda:None)
    ns["_out"]=lambda **kw:kw
    ns["guarded_place_order"]=lambda ib,c,o,**kw: ib.placeOrder(c,o) if isinstance(ib,owner.OwnerConnection) else broker.place(ib,c,o,**kw)
    ns["guarded_cancel_order"]=lambda ib,o: ib.cancelOrder(o)
    try:
        yield broker,native,ns,server,callbacks
    finally:
        asyncio.run_coroutine_threadsafe(server.close(),loop).result(5)
        loop.call_soon_threadsafe(loop.stop)
        thread.join(5)
        loop.close()


def edit_payload(oid=1, **changes):
    return dict(_broker_account="PRIMARY",_command_id="manual",symbol="MES",con_id=42,
                client_id=927481,order_id=oid,perm_id=100+oid,**changes)


@pytest.mark.parametrize("typ,oid", [("STP",1),("MKT",2)])
def test_manual_futures_exit_edit_uses_active_owner_preserving_terms(active_owner,typ,oid):
    broker,native,ns,_,callbacks=active_owner
    before=asdict(broker.orders[oid-1].order)
    result=manual.run(ns,broker,edit_payload(oid,new_qty=5),"fixture",0,7,modify=True)
    assert result["ok"],result
    assert native.wires==[("modify",oid,5)]
    assert asdict(broker.orders[oid-1].order)==dict(before,totalQuantity=5)
    assert len(callbacks)==1


def test_occupied_owner_reproduces_blank_failure_without_handoff(active_owner,monkeypatch):
    broker,native,ns,_,_=active_owner
    monkeypatch.setattr(owner,"connect_existing",lambda *a:None)
    result=manual.run(ns,broker,edit_payload(new_qty=5),"fixture",0,7,modify=True)
    assert result["state"]=="rejected" and not native.wires
    assert "client 927481" in result["detail"] and "TimeoutError" in result["detail"]


@pytest.mark.parametrize("held,qty", [(7,2),(-7,2),(7,7)])
def test_close_adjusts_daytrade_oca_before_close(active_owner,held,qty):
    broker,native,ns,_,_=active_owner
    broker.position.position=held
    for t in broker.orders:t.order.action="SELL" if held>0 else "BUY"
    payload=dict(request(),qty=qty,sec_type="FUT",symbol="MES",expected_position=held,
                 action="SELL" if held>0 else "BUY")
    result=actions.run(ns,broker,payload,"primary","fixture",0,7)
    assert result["ok"],result
    assert broker.position.position==(abs(held)-qty)*(1 if held>0 else -1)
    if qty<abs(held):
        assert native.wires==[("modify",1,5),("modify",2,5)]
        assert [t.order.totalQuantity for t in broker.orders[:2]]==[5,5]
    else:
        assert native.wires==[("cancel",1,7)]
        assert all(t.orderStatus.status=="Cancelled" for t in broker.orders[:2])
    assert len(broker.mutations)==1  # Only the close goes on the executor's wire.


@pytest.mark.parametrize("failure", ["wrong_id","wrong_account","filled","auth","new_order","fields"])
def test_owner_refuses_bad_identity_or_changed_fills_before_transmission(active_owner,failure):
    broker,native,_,server,_=active_owner
    message=dict(token=server.token,action="modify",identity=owner.identity(broker.orders[0]),
                 changes=dict(totalQuantity=5),filled_before=0)
    if failure=="wrong_id":message["identity"][4]=999
    if failure=="wrong_account":message["identity"][0]="OTHER"
    if failure=="filled":message["filled_before"]=1
    if failure=="auth":message["token"]="wrong"
    if failure=="new_order":message["action"]="entry"
    if failure=="fields":message["changes"]={"goodAfterTime":""}
    connection=owner.OwnerConnection(dict(server_descriptor(server),token=message.pop("token")),broker)
    with pytest.raises((ValueError,PermissionError)):
        connection.request(message.pop("action"),**message)
    assert not native.wires


def server_descriptor(server):
    return json.loads(server.path.read_text())


def test_stale_registration_never_falls_back_to_duplicate_connection(active_owner):
    broker,_,_,server,_=active_owner
    descriptor=server_descriptor(server)
    descriptor["port"]=1
    server.path.write_text(json.dumps(descriptor))
    with pytest.raises(OSError):owner.connect_existing("fixture",0,927481,broker)


def test_snapshot_roundtrip_preserves_native_order_fields(active_owner):
    broker,_,_,server,_=active_owner
    rows=owner.OwnerConnection(server_descriptor(server),broker).reqAllOpenOrders()
    assert [asdict(t.order) for t in rows]==[asdict(t.order) for t in broker.orders]


def test_futures_stop_price_edit_preserves_scheduled_sibling(active_owner):
    broker,native,ns,_,_=active_owner
    sibling=asdict(broker.orders[1].order)
    result=manual.run(ns,broker,edit_payload(new_stop=7690.25),"fixture",0,7,modify=True)
    assert result["ok"] and broker.orders[0].order.auxPrice==7690.25,result
    assert asdict(broker.orders[1].order)==sibling and len(native.wires)==1


def test_owner_fill_race_stops_close_before_transmission(active_owner):
    broker,native,ns,_,_=active_owner
    place=native.placeOrder
    def racing(c,o):
        trade=place(c,o)
        trade.orderStatus.filled=1
        return trade
    native.placeOrder=racing
    result=actions.run(ns,broker,dict(request(),qty=2,sec_type="FUT",symbol="MES"),
                       "primary","fixture",0,7)
    assert result["state"]=="unknown" and broker.last_close is None,result


def test_manual_timeout_after_send_is_unknown_and_never_replayed(active_owner):
    broker,native,ns,_,_=active_owner
    def disconnected(c,o):
        native.wires.append(("modify",o.orderId,o.totalQuantity))
        raise TimeoutError("lost acknowledgement")
    native.placeOrder=disconnected
    payload=edit_payload(new_qty=5)
    result=manual.run(ns,broker,payload,"fixture",0,7,modify=True)
    assert result["state"]=="unknown",result
    replay=manual.run(ns,broker,payload,"fixture",0,7,modify=True)
    assert replay["state"]=="unknown" and len(native.wires)==1


def test_missing_broker_echo_is_not_cached_as_working(active_owner):
    broker,native,_,server,_=active_owner
    connection=owner.OwnerConnection(server_descriptor(server),broker)
    assert len(connection.reqAllOpenOrders())==2
    native.trades=lambda:[]
    assert connection.reqAllOpenOrders()==[]


def test_raw_guard_snapshot_includes_orders_owned_by_other_clients(active_owner):
    broker,_,_,server,_=active_owner
    other=exit_trade(3)
    other.order.clientId=7
    connection=owner.OwnerConnection(server_descriptor(server),broker,
                                    read_all=lambda ib: ib.orders+[other])
    rows=connection.reqAllOpenOrdersRaw()
    assert len(rows)==3 and other in rows
