import copy
import sys
from types import SimpleNamespace as N

import pytest

from broker_runtime.order_edit_context import infer
from broker_runtime import execution_lifecycle as life
from tests.test_unified_position_actions import Broker, exit_order, namespace, request


def test_quantity_save_without_qualifiers_edits_exact_exit(tmp_path):
    broker = Broker(exits=[exit_order(1,100)])
    result = life.mutate_one(namespace(tmp_path,broker), broker,
        dict(request(),order_id=1,perm_id=101,client_id=7,new_qty=90), "",0,7,modify=True)
    assert result["ok"], result
    assert broker.orders[0].order.totalQuantity == 90
    assert broker.orders[0].order.auxPrice == 90
    assert len(broker.mutations) == 1


def test_entry_automatically_derives_direction_and_stop_risk(tmp_path):
    entry = exit_order(1,100,typ="LMT")
    entry.order.action, entry.order.lmtPrice = "BUY", 100
    stop = exit_order(2,100)
    stop.order.parentId = 1
    broker = Broker(holdings=0, exits=[entry,stop])
    edited = copy.deepcopy(entry.order)
    edited.totalQuantity = 80
    context = infer(namespace(tmp_path,broker),broker,entry,edited)
    assert context == dict(mutation_kind="entry", portfolio_direction="long",risk_usd=800)


def test_size_above_stop_coverage_reserves_unprotected_units(tmp_path):
    entry = exit_order(1,100,typ="LMT")
    entry.order.action, entry.order.lmtPrice = "BUY", 100
    stop = exit_order(2,100)
    stop.order.parentId = 1
    broker = Broker(holdings=0,exits=[entry,stop])
    edited = copy.deepcopy(entry.order)
    edited.totalQuantity = 120
    assert infer(namespace(tmp_path,broker),broker,entry,edited)["risk_usd"] == 3000


def test_provenance_wins_over_opposite_net_holding(tmp_path,monkeypatch):
    entry = exit_order(1,100,typ="LMT")
    broker = Broker(holdings=100,exits=[entry])  # SELL entry despite long net holding
    guard = N(_load_config=lambda:dict(reservation_dir="fixture"),_cluster_symbol=lambda _:"TEST",
        _intent_registry=lambda *a:(None,{}),_guarded_order_semantic=lambda *a:"entry")
    monkeypatch.setitem(sys.modules,"legend_reservation_guard",guard)
    ns=namespace(tmp_path,broker); ns["_cluster_symbol"]=lambda _:"TEST"
    context=infer(ns,broker,entry,copy.deepcopy(entry.order))
    assert context["mutation_kind"]=="entry" and context["portfolio_direction"]=="short"


def test_weaker_stop_gets_automatic_risk_but_tighter_stop_needs_no_reservation(tmp_path,monkeypatch):
    stop=exit_order(1,100)
    broker=Broker(exits=[stop])
    guard=N(_load_config=lambda:dict(reservation_dir="fixture"),_cluster_symbol=lambda _:"TEST",
        _intent_registry=lambda *a:(None,{}),_guarded_order_semantic=lambda *a:"exit",
        _exit_modify_requires_capacity=lambda ib,c,o,a,r:(o.auxPrice<90,"long",o))
    monkeypatch.setitem(sys.modules,"legend_reservation_guard",guard)
    ns=namespace(tmp_path,broker); ns["_cluster_symbol"]=lambda _:"TEST"
    edited=copy.deepcopy(stop.order); edited.auxPrice=85
    assert infer(ns,broker,stop,edited)["risk_usd"]==1500
    edited.auxPrice=95
    assert infer(ns,broker,stop,edited)==dict(mutation_kind="exit")
