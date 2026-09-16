"""Proportional exit repair, with fake brokers and no live connections."""
import copy
from decimal import Decimal
from types import SimpleNamespace as NS

import pytest

from broker_runtime import reconcile_position_exits as reconcile


def order(i, quantity, group="", *, account="PRIMARY", action="SELL", filled=0, parent=0, client=99, con_id=42):
    return NS(contract=NS(conId=con_id, symbol="TEST", secType="STK"),
              order=NS(account=account, clientId=client, orderId=i, permId=1000+i,
                       totalQuantity=quantity, ocaGroup=group, ocaType=1,
                       orderType="LMT" if i % 2 else "MKT", action=action,
                       parentId=parent, lmtPrice=120 if i % 2 else 0, auxPrice=0,
                       tif="GTC", goodAfterTime="" if i % 2 else "20261001 15:59:00 US/Eastern",
                       goodTillDate="", outsideRth=False, orderRef="strategy", transmit=True),
              orderStatus=NS(status="Submitted", filled=filled))


class Broker:
    def __init__(self, held, orders, account):
        self.position = NS(position=held, account=account, contract=NS(conId=42))
        self.orders, self.wires = orders, []
        self.change_position = self.lose_ack = self.cancel_siblings = False
    def reqPositions(self): pass
    def reqAllOpenOrders(self): pass
    def sleep(self, _): pass
    def openTrades(self): return [t for t in self.orders if t.orderStatus.status not in reconcile.life.TERMINAL]
    def cancelOrder(self, selected):
        self.wires.append(("cancel", selected.permId, None))
        for t in self.orders:
            if t.order.permId == selected.permId or (self.cancel_siblings and selected.ocaGroup and t.order.ocaGroup == selected.ocaGroup):
                t.orderStatus.status = "Cancelled"
    def placeOrder(self, contract, edited):
        self.wires.append(("modify", edited.permId, edited.totalQuantity))
        if self.lose_ack: raise TimeoutError("lost acknowledgement")
        next(t for t in self.orders if t.order.permId == edited.permId).order = copy.deepcopy(edited)
        if self.change_position: self.position.position -= 1


def setup(tmp_path, held, orders, account="PRIMARY"):
    ib = Broker(held, orders, account)
    def exact(ib, payload):
        p = ib.position
        return (p, None) if p.account == payload["_broker_account"] and p.contract.conId == payload["con_id"] else (None, "wrong identity")
    ns = dict(POSITION_ACTION_STATE_DIR=tmp_path, _out=lambda **kw: kw,
              _exact_position=exact, _orders_for_contract=lambda ib, *args: ib.openTrades(),
              guarded_place_order=lambda *a, **k: pytest.fail("risk policy consulted"),
              guarded_cancel_order=lambda *a, **k: pytest.fail("risk policy consulted"))
    p = dict(_broker_account=account, con_id=42, _command_id="fixture")
    return ns, ib, p


@pytest.mark.parametrize("account", ["PRIMARY", "PA"])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("held", [50, 100, 150])
def test_scales_oca_groups_once_preserving_all_other_fields(tmp_path, account, sign, held):
    side = "SELL" if sign > 0 else "BUY"
    orders = [order(1, 60, "a", account=account, action=side), order(2, 60, "a", account=account, action=side),
              order(3, 40, "b", account=account, action=side), order(4, 40, "b", account=account, action=side)]
    before = [vars(t.order).copy() for t in orders]
    ns, ib, p = setup(tmp_path, sign*held, orders, account)
    result = reconcile.run(ns, ib, p, "fixture", 0, 99)
    assert result["ok"], result
    expected = [held*.6, held*.6, held*.4, held*.4]
    for t, old, qty in zip(orders, before, expected):
        assert vars(t.order) == dict(old, totalQuantity=qty)
    assert ib.position.position == sign*held
    assert all(kind == "modify" for kind, *_ in ib.wires)
    assert len(ib.wires) == (0 if held == 100 else 4)
    assert reconcile.run(ns, ib, p, "fixture", 0, 99) == result
    assert len(ib.wires) == (0 if held == 100 else 4)


def test_original_sna_shape_normalizes_unequal_siblings(tmp_path):
    quantities = [119,119,171,218,121,121,367,367]
    orders = [order(i+1, q, str(i//2)) for i,q in enumerate(quantities)]
    ns, ib, p = setup(tmp_path, 700, orders)
    result = reconcile.run(ns, ib, p, "fixture", 0, 99)
    assert result["ok"], result
    assert [t.order.totalQuantity for t in orders] == [101,101,185,185,103,103,311,311]
    # The 171 -> 185 increase is last, after all coverage-reducing changes.
    assert ib.wires[-1] == ("modify", 1003, 185)


def test_unequal_siblings_repair_even_when_maximum_coverage_matches(tmp_path):
    orders = [order(1,80,"a"),order(2,100,"a")]
    ns, ib, p = setup(tmp_path,100,orders)
    assert reconcile.run(ns,ib,p,"fixture",0,99)["ok"]
    assert [t.order.totalQuantity for t in orders] == [100,100]


def test_uses_remaining_quantities_and_retains_filled_counts(tmp_path):
    orders = [order(1,120,"a",filled=20),order(2,100,"a"),order(3,100,"b"),order(4,100,"b")]
    ns, ib, p = setup(tmp_path,100,orders)
    assert reconcile.run(ns,ib,p,"fixture",0,99)["ok"]
    assert [t.order.totalQuantity for t in orders] == [70,50,50,50]
    assert orders[0].orderStatus.filled == 20


def test_no_exits_is_a_noop_and_never_creates_orders(tmp_path):
    ns, ib, p = setup(tmp_path,100,[])
    result = reconcile.run(ns,ib,p,"fixture",0,99)
    assert result["ok"] and "No existing exits" in result["detail"] and not ib.wires


def test_excludes_entries_pending_children_other_accounts_and_contracts(tmp_path):
    unrelated = [order(10,80,action="BUY"),order(11,80,parent=10),
                 order(20,80,account="PA"),order(21,80,con_id=43)]
    orders = [order(1,200,"a"),order(2,200,"a"),*unrelated]
    before = copy.deepcopy([vars(t.order) for t in unrelated])
    ns, ib, p = setup(tmp_path,100,orders)
    assert reconcile.run(ns,ib,p,"fixture",0,99)["ok"]
    assert [vars(t.order) for t in unrelated] == before
    assert [t.order.totalQuantity for t in orders[:2]] == [100,100]


@pytest.mark.parametrize("cancel_siblings", [False,True])
def test_rounding_cancels_entire_zero_rungs(tmp_path,cancel_siblings):
    orders = [order(1,1,"a"),order(2,1,"a"),order(3,2,"b"),order(4,2,"b")]
    ns, ib, p = setup(tmp_path,1,orders)
    ib.cancel_siblings=cancel_siblings
    result=reconcile.run(ns,ib,p,"fixture",0,99)
    assert result["ok"],result
    assert [t.orderStatus.status for t in orders[:2]] == ["Cancelled","Cancelled"]
    assert [t.order.totalQuantity for t in orders[2:]] == [1,1]
    assert result["fill"]["cancelled"]==2


def test_fractional_inventory_is_not_rounded_to_whole_shares(tmp_path):
    orders = [order(1,1,"a"),order(2,1,"a"),order(3,1,"b"),order(4,1,"b")]
    ns,ib,p=setup(tmp_path,1.5,orders)
    assert reconcile.run(ns,ib,p,"fixture",0,99)["ok"]
    assert sum(Decimal(str(orders[i].order.totalQuantity)) for i in (0,2))==Decimal("1.5")


@pytest.mark.parametrize("flag",["change_position","lose_ack"])
def test_stops_after_a_race_or_uncertain_delivery_and_never_replays(tmp_path,flag):
    orders=[order(1,100,"a"),order(2,100,"a")]
    ns,ib,p=setup(tmp_path,50,orders)
    setattr(ib,flag,True)
    result=reconcile.run(ns,ib,p,"fixture",0,99)
    assert result["state"]=="unknown" and len(ib.wires)==1
    assert reconcile.run(ns,ib,p,"fixture",0,99)["state"]=="unknown"
    assert len(ib.wires)==1


def test_all_owner_connections_preflight_before_changes(tmp_path):
    ns,ib,p=setup(tmp_path,50,[order(1,100,"a"),order(2,100,"a",client=98)])
    class Unavailable:
        def connect(self,*a,**k):raise ConnectionError("owner busy")
        def disconnect(self):pass
    ns["IB"]=Unavailable
    result=reconcile.run(ns,ib,p,"fixture",0,99)
    assert result["state"]=="rejected" and not ib.wires


def test_partially_filled_working_entry_does_not_resize_its_children(tmp_path):
    ns,ib,p=setup(tmp_path,50,[order(1,100,action="BUY",filled=50),order(2,100,parent=1)])
    result=reconcile.run(ns,ib,p,"fixture",0,99)
    assert result["state"]=="rejected" and not ib.wires
