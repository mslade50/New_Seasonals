"""Observe broker evidence and retire stopped workflows; never send an order.

Call under the position-action lock. Fresh request results, not the IB cache,
establish the current state. Missing historical evidence never proves a fill or
cancellation. Resolving a receipt never resumes its old close/re-add plan.
"""
from __future__ import annotations

import copy
import math
import time

try:
    from . import position_actions as actions
except ImportError:
    import position_actions as actions

TERMINAL = {"Filled", "Cancelled", "ApiCancelled"}
STABLE = TERMINAL | {"Submitted", "PreSubmitted"}


def number(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("broker quantity is missing or invalid")
    return value


def order_row(trade, *, completed=False):
    o, s = trade.order, trade.orderStatus
    qty = number(o.totalQuantity)
    # ib_insync completedOrder constructs an OrderStatus with default filled=0.
    # That default is not evidence of zero fills on a cancelled order.
    filled = number(s.filled)
    if completed:
        reported = float(getattr(o, "filledQuantity", float("nan")))
        filled = reported if math.isfinite(reported) and 0 <= reported <= qty else None
    if str(s.status) == "Filled":
        filled = qty
    return dict(identity=[str(o.account), int(trade.contract.conId), int(o.clientId),
                          int(o.orderId), int(o.permId or 0)],
                status=str(s.status), filled=filled, qty=qty,
                action=str(o.action), ref=str(o.orderRef or ""), parent=int(o.parentId or 0),
                limit=float(o.lmtPrice), stop=float(o.auxPrice))


def capture(ib, account, con_id):
    timeout = getattr(ib, "RequestTimeout", None)
    if timeout is not None:
        ib.RequestTimeout = min(timeout, 8) if timeout > 0 else 8
    try:
        return _capture(ib, account, con_id)
    finally:
        if timeout is not None:
            ib.RequestTimeout = timeout


def _capture(ib, account, con_id):
    """Double read detects a moving book; request timeout/None is not empty."""
    def current():
        positions = ib.reqPositions()
        orders = ib.reqAllOpenOrders()
        if positions is None or orders is None:
            raise ValueError("broker position/open-order request did not complete")
        ps = [float(p.position) for p in positions
              if str(p.account) == account and int(p.contract.conId) == con_id]
        if len(ps) > 1 or any(not math.isfinite(p) for p in ps):
            raise ValueError("broker position identity is ambiguous")
        rows = [order_row(t) for t in orders
                if str(t.order.account) == account and int(t.contract.conId) == con_id]
        if any(r["status"] not in STABLE for r in rows):
            raise ValueError("broker still reports an order transition")
        if len({tuple(r["identity"]) for r in rows}) != len(rows):
            raise ValueError("broker returned duplicate order identities")
        return dict(position=ps[0] if ps else 0, orders=sorted(rows, key=lambda r: r["identity"]))
    before = current()
    completed = ib.reqCompletedOrders(apiOnly=False)
    fills = ib.reqExecutions()
    if completed is None or fills is None:
        raise ValueError("broker completed-order/execution request did not complete")
    history = [order_row(t, completed=True) for t in completed
               if str(t.order.account) == account and int(t.contract.conId) == con_id]
    executions = []
    for fill in fills:
        e = fill.execution
        if str(e.acctNumber) == account and int(fill.contract.conId) == con_id:
            executions.append(dict(identity=[account, con_id, int(e.clientId), int(e.orderId),
                                             int(e.permId or 0)], exec_id=str(e.execId),
                                   cumulative=number(e.cumQty), shares=number(e.shares)))
    after = current()
    if before != after:
        raise ValueError("broker state changed during reconciliation; checking again shortly")
    return dict(after, completed=history, executions=executions, at=time.time(),
                account=account, con_id=con_id)


def matches(identity, wanted):
    if identity[:2] != list(wanted[:2]):
        return False
    if wanted[4] > 0:
        return identity[4] == wanted[4]
    return identity[2:4] == list(wanted[2:4])


def lookup(evidence, wanted):
    live = [r for r in evidence["orders"] if matches(r["identity"], wanted)]
    past = [r for r in evidence["completed"] if matches(r["identity"], wanted)]
    if len(live) > 1 or len(past) > 1:
        raise ValueError("exact broker order has conflicting identities")
    if live and past and (live[0]["status"], live[0]["filled"]) != (past[0]["status"], past[0]["filled"]):
        raise ValueError("open/completed broker evidence is still transitioning")
    return (live or past or [None])[0]


def outcome(ok, detail, fill=None):
    return dict(ok=ok, state="executed" if ok else "rejected", detail=detail, fill=fill)


def resolve(record, evidence):
    """Return a terminal receipt only when evidence explains the stopped plan."""
    if record["phase"] == "done":
        return record
    p = record["payload"]
    if evidence["account"] != p["_broker_account"] or evidence["con_id"] != int(p["con_id"]):
        raise ValueError("evidence belongs to another account/contract")
    result = None
    if "modify" in record:
        row = lookup(evidence, record["identity"])
        if row is None:
            raise ValueError("exact edited order is absent from available broker history")
        if not record["modify"]:
            if row["status"] in TERMINAL:
                result = outcome(row["status"] != "Filled", "Broker confirms order " + row["status"],
                                 dict(status=row["status"], filled=row["filled"]))
        else:
            fields = {"new_qty": "qty", "new_limit": "limit", "new_stop": "stop"}
            requested = {v: float(p[k]) for k, v in fields.items() if p.get(k) not in (None, "")}
            if requested and row["status"] in STABLE and all(row[k] == v for k, v in requested.items()):
                result = outcome(True, "Broker readback confirms the requested modification",
                                 dict(status=row["status"], filled=row["filled"]))
            elif row["status"] in TERMINAL:
                result = outcome(False, "Order is terminal; earlier modification was not confirmed; no edit replayed",
                                 dict(status=row["status"], filled=row["filled"]))
    elif record.get("kind") == "reconcile_exits":
        # Retire an interrupted allocation, never continue its stale plan. The
        # next user instruction computes quantities from the current book.
        result = outcome(False, "Stopped exit-allocation workflow reconciled against current broker state; "
                         "completed changes retained, no remaining changes replayed")
    elif not record.get("wire") and not record.get("addition") and record.get("mutation", "") in {"", "resize exit", "cancel exit"}:
        # wire is durably saved BEFORE a close send. Addition has a separate
        # pre-send marker. This proves the stopped workflow never sent either.
        if (number(record["held"]) <= 0 or number(record["quantity"]) <= 0
                or record["closing"] not in {"BUY", "SELL"} or not isinstance(record["legs"], list)):
            raise ValueError("pre-submission journal is incomplete")
        result = outcome(False, "Earlier workflow stopped before entry/close submission. "
                         "Current broker orders and holdings verified; existing changes retained; no trade replayed")
    elif record.get("addition"):
        context = record.get("add_context") or {}
        rungs = actions.groups(context.get("legs", []))
        qty = record.get("addition_requested", record.get("filled") if record.get("wire") else record["quantity"])
        if not rungs or qty is None:
            raise ValueError("addition allocation evidence is incomplete")
        sizes = actions.allocate([r[0]["qty"] for r in rungs], qty)
        expected = [(r, size) for r, size in zip(rungs, sizes) if size]
        if len(expected) != len(record["addition"]):
            raise ValueError("not every intended addition has a broker receipt")
        total_filled = 0
        for (rung, size), allocation in zip(expected, record["addition"]):
            if len(allocation["parent"]) != 1 or len(allocation["children"]) != len(rung):
                raise ValueError("attached addition identities are incomplete")
            observed = []
            for item in allocation["parent"] + allocation["children"]:
                if not item.get("perm_id"):
                    raise ValueError("addition lacks a durable broker identity")
                row = lookup(evidence, [p["_broker_account"], int(p["con_id"]), 0,
                                        item["order_id"], item["perm_id"]])
                if row is None or row["status"] not in STABLE:
                    raise ValueError("addition order is absent or still transitioning")
                observed.append(row)
            parent = observed[0]
            if parent["status"] not in TERMINAL or parent["qty"] != size or parent["action"] != context["entry_action"]:
                raise ValueError("addition entry is still working or its terms changed")
            for child in observed[1:]:
                if child["action"] != context["close_action"] or child["parent"] != parent["identity"][3]:
                    raise ValueError("attached exit identity does not match addition")
            if parent["filled"] is None:
                raise ValueError("terminal addition filled quantity unavailable")
            total_filled += parent["filled"]
        result = outcome(bool(total_filled), "Broker confirms addition entries are terminal; "
                         "current exits retained, no entry or exit replayed", dict(filled=total_filled))
    elif record.get("wire") and record.get("mutation") != "stage addition and attached exits":
        wanted = record["wire"]
        row = lookup(evidence, wanted)
        filled = max([e["cumulative"] for e in evidence["executions"] if matches(e["identity"], wanted)] or [0])
        qty = number(record["quantity"])
        if row:
            if row["qty"] != qty or row["action"] != record["closing"]:
                raise ValueError("close terms changed at broker; inspect exact order")
            filled = max(filled, row["filled"] or 0)
        if filled > qty:
            raise ValueError("broker fill exceeds recorded close quantity")
        if (row and row["status"] in TERMINAL) or filled == qty:
            status = "Filled" if filled == qty else row["status"]
            known = row is None or row["filled"] is not None or filled == qty
            quantity_text = f"{filled:g}/{qty:g} filled" if known else "historical fill quantity unavailable"
            result = outcome(bool(filled), f"Broker confirms close {status}: {quantity_text}. "
                             "Stopped workflow retired; no exit restoration or re-add replayed",
                             dict(status=status, filled=filled if known else None, order_id=wanted[3]))
    if result is None:
        raise ValueError("broker evidence does not yet resolve the attempted operation")
    resolved = copy.deepcopy(record)
    resolved.update(phase="done", result=result,
                    resolution=dict(kind="broker_readback", evidence=evidence, no_replay=True))
    return resolved


def refresh_target(ib, root, account, con_id):
    """Resolve each stopped receipt independently from one consistent snapshot."""
    targets = [(folder, r) for folder in (root, root / "order_edits") for r in actions.records(folder)
               if r["phase"] != "done" and r["payload"]["_broker_account"] == account
               and int(r["payload"]["con_id"]) == int(con_id)
               and not (folder == root and r["phase"] == "pending" and r.get("wire"))]
    if not targets:
        return {}
    evidence = capture(ib, account, int(con_id))
    results = {}
    for folder, record in targets:
        try:
            resolved = resolve(record, evidence)
            actions.save(folder, resolved)
            results[record["id"]] = resolved["result"]
        except ValueError as exc:
            results[record["id"]] = dict(ok=False, state="unknown", detail=str(exc), fill=None)
    return results
