"""Portable broker simulations: no live executor imports or network access."""
import copy
import json
from types import SimpleNamespace as N

import pytest

from broker_runtime import position_actions as actions
from broker_runtime import position_action_agent as agent


def contract(cid=42):
    return N(conId=cid, symbol="TEST", secType="STK", exchange="SMART")


def exit_order(number, quantity, group="", typ="STP", filled=0, account="PRIMARY", con_id=42, client=7):
    return N(contract=contract(con_id), order=N(
        account=account, clientId=client, orderId=number, permId=100+number,
        totalQuantity=quantity, ocaGroup=group, ocaType=1, orderType=typ, action="SELL",
        tif="GTC", parentId=0, auxPrice=90, lmtPrice=120, goodAfterTime="", goodTillDate="",
        outsideRth=False, orderRef="original", transmit=True),
        orderStatus=N(status="Submitted", filled=filled, avgFillPrice=100))


class Broker:
    def __init__(self, holdings=100, exits=(), status="Filled", filled=None):
        self.position = N(account="PRIMARY", position=holdings, contract=contract(), avgCost=100)
        self.orders = list(exits)
        self.next_id = 1000
        self.client = N(getReqId=self.next)
        self.status, self.fill_quantity = status, filled
        self.mutations = []
        self.fail_lookup = self.fail_send = self.fail_resize = False
        self.last_close = None
    def next(self):
        self.next_id += 1
        return self.next_id
    def reqPositions(self): return self.positions()
    def positions(self): return [self.position] if self.position.position else []
    def reqTickers(self, c): return [N(ask=101, bid=99)]
    def reqAllOpenOrders(self):
        if self.fail_lookup: raise TimeoutError("fixture lookup")
    def openTrades(self): return [t for t in self.orders if t.orderStatus.status not in actions.TERMINAL]
    def reqCompletedOrders(self, apiOnly=False): return [t for t in self.orders if t.orderStatus.status in actions.TERMINAL]
    def sleep(self, _): pass
    def qualifyContracts(self, c): return [c]
    def place(self, ib, c, o, **kwargs):
        self.mutations.append(("place", o.orderId, o.totalQuantity))
        original = next((t for t in self.orders if t.order.orderId == o.orderId), None)
        if original:
            if self.fail_resize: raise TimeoutError("uncertain modification")
            original.order = copy.deepcopy(o)
            return original
        if self.fail_send: raise TimeoutError("uncertain send")
        o.permId, o.clientId = o.orderId + 100, 7
        filled = o.totalQuantity if self.fill_quantity is None else self.fill_quantity
        trade = N(contract=c, order=o, orderStatus=N(status=self.status, filled=filled, avgFillPrice=100))
        self.orders.append(trade)
        self.last_close = trade
        self.position.position += filled if o.action == "BUY" else -filled
        return trade
    def cancel(self, ib, order):
        self.mutations.append(("cancel", order.orderId, order.totalQuantity))
        next(t for t in self.orders if t.order.orderId == order.orderId).orderStatus.status = "Cancelled"


def request(**extra):
    return dict(_command_id="fixture", _broker_account="PRIMARY", con_id=42, symbol="TEST",
                sec_type="STK", expected_position=100, action="SELL", qty=40, order_type="MKT", **extra)


def namespace(tmp_path, broker):
    def exact(ib, p):
        matches = [x for x in ib.positions() if x.account == p["_broker_account"] and x.contract.conId == p["con_id"]]
        return (matches[0], None) if len(matches) == 1 else (None, "exact position unavailable")
    def orders(ib, c, account):
        ib.reqAllOpenOrders()
        return [t for t in ib.openTrades() if t.contract.conId == c.conId and t.order.account == account]
    def capture(rows, side):
        if any(t.order.orderType == "MKT" and not t.order.goodAfterTime for t in rows):
            return None, "another immediate close is working"
        return [dict(source_key=str(t.order.permId), qty=t.order.totalQuantity,
                     oca_group=t.order.ocaGroup, oca_type=t.order.ocaType, order_type=t.order.orderType,
                     aux=t.order.auxPrice, lmt=t.order.lmtPrice, good_after=t.order.goodAfterTime,
                     good_till=t.order.goodTillDate, tif=t.order.tif, outside_rth=t.order.outsideRth,
                     order_ref=t.order.orderRef) for t in rows], None
    def prepare(ib, p, account, partial=False):
        legs, _ = actions.exit_snapshot(ns, ib, broker.position, p)
        return dict(pos=broker.position, ref=broker.position.contract, qty=p["qty"], held=abs(broker.position.position),
                    account="PRIMARY", legs=legs, avg_cost=100, entry_action="BUY", close_action="SELL"), None
    def restore(ib, c, legs, side, qty, account, signal):
        restored = []
        for leg in legs:
            t = exit_order(broker.next(), qty, leg.get("oca_group", ""), leg["order_type"])
            t.order.action = side
            broker.orders.append(t)
            restored.append(t)
            broker.mutations.append(("restore", t.order.orderId, qty))
        return restored, legs
    ns = {"POSITION_ACTION_STATE_DIR": tmp_path, "_exact_position": exact, "_orders_for_contract": orders,
          "_capture_exit_legs": capture, "_prepare_position_action_add": prepare,
          "_max_notional": lambda _: 100000, "_command_signal": lambda p, s: p["_command_id"]+":"+s,
          "_out": lambda ok, state, detail, fill=None: dict(ok=ok, state=state, detail=detail, fill=fill),
          "MarketOrder": lambda action, qty: N(action=action, totalQuantity=qty, permId=0),
          "LimitOrder": lambda action, qty, price: N(action=action, totalQuantity=qty, lmtPrice=price, permId=0),
          "guarded_place_order": broker.place, "guarded_cancel_order": broker.cancel,
          "_place_exit_legs": restore, "_placement_problem": lambda _: None}
    return ns


def run(tmp_path, broker, payload=None, adding=False):
    return actions.run(namespace(tmp_path, broker), broker, payload or request(), "primary", "fixture", 0, 7, adding=adding)


@pytest.mark.parametrize("coverage", [20, 80, 100, 180])
def test_mismatched_oca_coverage_normalizes_without_double_counting(tmp_path, coverage):
    broker = Broker(exits=[exit_order(1, coverage, "pair"), exit_order(2, coverage, "pair", "LMT")])
    result = run(tmp_path, broker)
    assert result["ok"] and broker.position.position == 60
    assert [t.order.totalQuantity for t in broker.orders[:2]] == [60, 60]
    assert broker.mutations[-1][1] == 1001


def test_weighted_rungs_round_to_exact_remainder_and_remove_zero(tmp_path):
    broker = Broker(holdings=5, exits=[exit_order(1, 3, "a"), exit_order(2, 2, "b")])
    p = request()
    p.update(qty=4, expected_position=5)
    assert run(tmp_path, broker, p)["ok"]
    assert broker.orders[0].order.totalQuantity == 1
    assert broker.orders[1].orderStatus.status == "Cancelled"


@pytest.mark.parametrize("weights,total,result", [([3, 2], 51, [31, 20]), ([1, 1, 1], 2, [1, 1, 0]), ([1, 1], 0, [0, 0])])
def test_rounding(weights, total, result):
    assert actions.allocate(weights, total) == result


def test_successful_empty_lookup_closes_but_failed_lookup_places_nothing(tmp_path):
    broker = Broker()
    assert run(tmp_path / "empty", broker)["ok"]
    failed = Broker()
    failed.fail_lookup = True
    result = run(tmp_path / "failed", failed)
    assert result["state"] == "rejected" and not failed.mutations


def test_full_close_cancels_exits_but_leaves_other_account_contract_and_entry(tmp_path):
    owned = exit_order(1, 80)
    other = exit_order(2, 20, account="OTHER")
    future = exit_order(3, 20, con_id=43)
    entry = exit_order(4, 10, typ="LMT")
    entry.order.action = "BUY"
    broker = Broker(exits=[owned, other, future, entry])
    p = request()
    p["qty"] = 100
    result = run(tmp_path, broker, p)
    assert result["ok"] and broker.position.position == 0
    assert owned.orderStatus.status == "Cancelled"
    assert all(t.orderStatus.status == "Submitted" for t in [other, future, entry])


def test_short_close_buys_and_resizes_buy_exits(tmp_path):
    stop = exit_order(1, 70)
    stop.order.action = "BUY"
    broker = Broker(holdings=-100, exits=[stop])
    p = request()
    p.update(action="BUY", expected_position=-100)
    assert run(tmp_path, broker, p)["ok"]
    assert broker.position.position == -60 and stop.order.totalQuantity == 60


def test_partial_fill_then_cancel_restores_only_actual_remaining_quantity(tmp_path):
    broker = Broker(exits=[exit_order(1, 80)], status="Submitted", filled=10)
    assert run(tmp_path, broker)["ok"]
    assert broker.orders[0].order.totalQuantity == 60  # remaining close reserves 30
    broker.last_close.orderStatus.status = "Cancelled"
    result = run(tmp_path, broker, dict(request(), reconcile_only=True))
    assert result["ok"] and broker.orders[0].order.totalQuantity == 90
    assert len([m for m in broker.mutations if m[1] == 1001]) == 1


def test_cancelled_full_close_restores_our_cancelled_exits(tmp_path):
    broker = Broker(exits=[exit_order(1, 80)], status="Cancelled", filled=10)
    p = request()
    p["qty"] = 100
    assert run(tmp_path, broker, p)["ok"]
    assert broker.openTrades()[0].order.totalQuantity == 90


def test_resting_limit_resumes_after_restart_without_another_close(tmp_path):
    broker = Broker(exits=[exit_order(1, 100)], status="Submitted", filled=0)
    p = request()
    p.update(order_type="LMT", limit=110, outside_rth=True)
    result = run(tmp_path, broker, p)
    assert result["ok"] and result["fill"]["filled"] == 0
    broker.last_close.orderStatus.status = "Filled"
    broker.last_close.orderStatus.filled = 40
    broker.position.position = 60
    assert run(tmp_path, broker, dict(p, reconcile_only=True))["ok"]
    assert len([m for m in broker.mutations if m[1] == 1001]) == 1


@pytest.mark.parametrize("failure", ["fail_send", "fail_resize"])
def test_uncertain_mutation_blocks_retries_and_new_actions(tmp_path, failure):
    broker = Broker(exits=[exit_order(1, 100)])
    setattr(broker, failure, True)
    result = run(tmp_path, broker)
    assert result["state"] == "unknown"
    count = len(broker.mutations)
    run(tmp_path, broker, dict(request(), reconcile_only=True))
    run(tmp_path, broker, dict(request(), _command_id="different"))
    assert len(broker.mutations) == count


def test_owner_unavailable_rejects_before_any_mutation(tmp_path):
    broker = Broker(exits=[exit_order(1, 100, client=88)])
    ns = namespace(tmp_path, broker)
    ns["IB"] = lambda: N(connect=lambda *a, **k: (_ for _ in ()).throw(ConnectionError()), disconnect=lambda: None)
    result = actions.run(ns, broker, request(), "primary", "", 0, 7)
    assert not result["ok"] and not broker.mutations


def test_add_uses_one_attached_parent_per_rung_and_normalizes_old_coverage(tmp_path, monkeypatch):
    broker = Broker(exits=[exit_order(1, 30, "a"), exit_order(2, 20, "b")])
    calls = []
    def attached(ns, ib, ctx, qty, signal, market=False):
        calls.append((qty, [x["qty"] for x in ctx["legs"]], market))
        return {"parent": [signal], "children": ["fixture"]}, None
    monkeypatch.setattr(actions.life, "stage_attached", attached)
    p = request()
    p["qty"] = 21
    assert run(tmp_path, broker, p, adding=True)["ok"]
    assert [x.order.totalQuantity for x in broker.orders] == [60, 40]
    assert calls == [(13, [13], True), (8, [8], True)]


def test_readd_only_uses_confirmed_terminal_partial_close(tmp_path, monkeypatch):
    broker = Broker(exits=[exit_order(1, 100)], status="Submitted", filled=10)
    calls = []
    monkeypatch.setattr(actions.life, "stage_attached", lambda ns, ib, ctx, qty, signal, market=False: (calls.append(qty) or {}, None))
    p = dict(request(), readd=True)
    assert run(tmp_path, broker, p)["ok"] and not calls
    broker.last_close.orderStatus.status = "Cancelled"
    assert run(tmp_path, broker, dict(p, reconcile_only=True))["ok"]
    assert calls == [10]
    assert run(tmp_path, broker, dict(p, reconcile_only=True))["ok"]
    assert calls == [10]  # a restart cannot submit the completed re-add twice


def test_readd_never_crosses_the_original_session(tmp_path, monkeypatch):
    broker = Broker(exits=[exit_order(1, 100)], status="Submitted", filled=10)
    p = dict(request(), readd=True)
    assert run(tmp_path, broker, p)["ok"]
    path = actions.record_path(tmp_path, p["_command_id"])
    record = json.loads(path.read_text())
    record["session"] = "2000-01-01"
    actions.save(tmp_path, record)
    broker.last_close.orderStatus.status = "Cancelled"
    monkeypatch.setattr(actions.life, "stage_attached", lambda *a, **k: pytest.fail("late re-add"))
    result = run(tmp_path, broker, dict(p, reconcile_only=True))
    assert result["state"] == "unknown" and "after the re-add session" in result["detail"]


def test_expired_exit_is_revalidated_before_delayed_readd(tmp_path):
    leg = dict(source_key="1", qty=10, order_type="MKT",
               good_after="20000101 15:59:00 US/Eastern")
    broker = Broker()
    with pytest.raises(ValueError, match="expired"):
        actions.stage_add(namespace(tmp_path, broker), broker, {"legs": [leg]},
                          10, {}, tmp_path, market=False)
    assert not broker.mutations


def test_same_command_id_with_changed_quantity_cannot_resubmit(tmp_path):
    broker = Broker()
    assert run(tmp_path, broker)["ok"]
    result = run(tmp_path, broker, dict(request(), qty=20))
    assert result["state"] == "rejected" and len(broker.mutations) == 1
    assert run(tmp_path, broker)["ok"]  # the original receipt was not overwritten


def test_pending_child_identity_includes_owning_client(tmp_path):
    stop = exit_order(1, 100, client=7)
    stop.order.parentId = 50
    unrelated = exit_order(50, 20, client=8, typ="LMT")
    unrelated.order.action = "BUY"
    broker = Broker(exits=[stop, unrelated])
    assert run(tmp_path, broker)["ok"]
    assert stop.order.totalQuantity == 60


def test_partially_filled_entry_blocks_close_before_exit_changes(tmp_path):
    stop = exit_order(1, 100)
    stop.order.parentId = 50
    entry = exit_order(50, 20, filled=10, typ="LMT")
    entry.order.action = "BUY"
    broker = Broker(exits=[stop, entry])
    assert run(tmp_path, broker)["state"] == "rejected"
    assert not broker.mutations


def test_exit_fill_during_owner_lookup_never_submits_close(tmp_path, monkeypatch):
    stop = exit_order(1, 100)
    broker = Broker(exits=[stop])
    original = actions.life.find_exact
    def racing_lookup(*args):
        result = original(*args)
        stop.orderStatus.filled = 10
        broker.position.position = 90
        return result
    monkeypatch.setattr(actions.life, "find_exact", racing_lookup)
    assert run(tmp_path, broker)["state"] == "unknown"
    assert not broker.mutations


def test_corrupt_journal_fails_closed(tmp_path):
    (tmp_path / "broken.json").write_text("{broken")
    broker = Broker()
    assert run(tmp_path, broker)["state"] == "rejected" and not broker.mutations


def test_agent_shape_validation_does_not_depend_on_a_snapshot():
    cmd = {"account": "primary", "type": "close_resize", "payload": request()}
    assert agent.applies(cmd) and agent.validate(cmd) == (True, [])
    assert not agent.validate(dict(cmd, payload=dict(request(), con_id=0)))[0]
    assert not agent.applies(dict(cmd, account="pa"))
