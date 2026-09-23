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
        if completed and filled is not None and filled != qty:
            raise ValueError("completed Filled status contradicts broker filled quantity")
        filled = qty
    return dict(identity=[str(o.account), int(trade.contract.conId), int(o.clientId),
                          int(o.orderId), int(o.permId or 0)],
                status=str(s.status), filled=filled, qty=qty,
                action=str(o.action), ref=str(o.orderRef or ""), parent=int(o.parentId or 0),
                limit=float(o.lmtPrice), stop=float(o.auxPrice), order_type=str(o.orderType),
                oca_group=str(o.ocaGroup or ""), oca_type=int(o.ocaType), tif=str(o.tif),
                good_after=str(o.goodAfterTime or ""), good_till=str(o.goodTillDate or ""),
                outside_rth=bool(o.outsideRth), transmit=bool(o.transmit))


def capture(ib, account, con_id, *, open_reader=None):
    timeout = getattr(ib, "RequestTimeout", None)
    if timeout is not None:
        ib.RequestTimeout = min(timeout, 8) if timeout > 0 else 8
    try:
        return _capture(ib, account, con_id, open_reader=open_reader)
    finally:
        if timeout is not None:
            ib.RequestTimeout = timeout


def _capture(ib, account, con_id, *, open_reader=None):
    """Double read detects a moving book; request timeout/None is not empty."""
    def bound_account(value):
        if str(value or "").strip():
            return str(value).strip()
        if hasattr(ib, "managedAccounts") and ib.managedAccounts() == [account]:
            return account
        raise ValueError("broker returned an ambiguous blank account for this contract")

    def rows_for(trades, *, completed=False):
        rows = []
        for trade in trades:
            if int(trade.contract.conId) != con_id:
                continue
            resolved = bound_account(trade.order.account)
            if resolved == account:
                row = order_row(trade, completed=completed)
                row["identity"][0] = resolved
                rows.append(row)
        return rows

    def current():
        positions = ib.reqPositions()
        if hasattr(ib, "wrapper") and open_reader is None:
            raise ValueError("raw broker order reader is required; cached Trade objects are insufficient")
        orders = open_reader(ib) if open_reader else ib.reqAllOpenOrders()
        if positions is None or orders is None:
            raise ValueError("broker position/open-order request did not complete")
        ps = [float(p.position) for p in positions
              if int(p.contract.conId) == con_id and bound_account(p.account) == account]
        if len(ps) > 1 or any(not math.isfinite(p) for p in ps):
            raise ValueError("broker position identity is ambiguous")
        rows = rows_for(orders)
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
    history = rows_for(completed, completed=True)
    executions = []
    for fill in fills:
        e = fill.execution
        if int(fill.contract.conId) == con_id and bound_account(e.acctNumber) == account:
            executions.append(dict(identity=[account, con_id, int(e.clientId), int(e.orderId),
                                             int(e.permId or 0)], exec_id=str(e.execId),
                                   cumulative=number(e.cumQty), shares=number(e.shares),
                                   ref=str(e.orderRef or "")))
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


def lookup(evidence, wanted, *, ref=None):
    def belongs(row):
        return matches(row["identity"], wanted) and (wanted[4] > 0 or (ref and row["ref"] == ref))
    live = [r for r in evidence["orders"] if belongs(r)]
    past = [r for r in evidence["completed"] if belongs(r)]
    if len(live) > 1 or len(past) > 1:
        raise ValueError("exact broker order has conflicting identities")
    if live and past and (live[0]["status"], live[0]["filled"]) != (past[0]["status"], past[0]["filled"]):
        raise ValueError("open/completed broker evidence is still transitioning")
    return (live or past or [None])[0]


def outcome(ok, detail, fill=None):
    return dict(ok=ok, state="executed" if ok else "rejected", detail=detail, fill=fill)


def check_coverage(record, evidence):
    """Describe current exit discrepancies, independently of receipt resolution."""
    if not (record.get("legs") or record.get("addition") or record.get("kind") == "reconcile_exits"):
        return
    position = evidence["position"]
    closing = "SELL" if position > 0 else "BUY" if position < 0 else record.get("closing")
    if not closing and record.get("kind") == "reconcile_exits":
        original_position = float(record["held"])
        if not math.isfinite(original_position) or original_position == 0:
            raise ValueError("original exit-allocation direction is unavailable")
        closing = "SELL" if original_position > 0 else "BUY"
    live = [r for r in evidence["orders"] if r["status"] not in TERMINAL]
    parents = {(r["identity"][2], r["identity"][3]) for r in live}
    owned = [leg["identity"] for leg in record.get("legs", []) if leg.get("identity")]
    owned.extend(item["identity"] for item in record.get("plan", []) if item.get("identity"))
    for allocation in record.get("addition", []):
        owned.extend([evidence["account"], evidence["con_id"], 0, item["order_id"], item["perm_id"]]
                     for item in allocation.get("children", []) if item.get("perm_id"))
    for row in live:
        if (any(matches(row["identity"], wanted) for wanted in owned)
                and (not position or row["action"] != closing)):
            raise ValueError("record-owned closing order remains working after the position became flat or reversed")
    exits = [r for r in live if r["action"] == closing
             and (not r["parent"] or (r["identity"][2], r["parent"]) not in parents)]
    if not position:
        if exits:
            raise ValueError("position is flat but closing orders are still working")
        return
    required_protection = bool(record.get("addition")) or any(
        l.get("order_type") in {"STP", "STP LMT"} or
        (l.get("order_type") == "MKT" and l.get("good_after")) for l in record.get("legs", []))
    groups = {}
    for row in exits:
        if row["filled"] is None:
            raise ValueError("current closing order activation or fill quantity is unclear")
        remaining = row["qty"] - row["filled"]
        if remaining <= 0:
            raise ValueError("current closing order remaining quantity is inconsistent")
        if row["oca_group"] and row["oca_type"] not in {1, 2}:
            raise ValueError("current OCA group does not establish bounded exit coverage")
        group = row["oca_group"] or tuple(row["identity"])
        groups.setdefault(group, []).append((row, remaining))
    coverage = 0
    for siblings in groups.values():
        quantities = {qty for _, qty in siblings}
        if len(quantities) != 1:
            raise ValueError("current OCA exit quantities disagree")
        if required_protection and not any(
                r["order_type"] in {"STP", "STP LMT"} or
                (r["order_type"] == "MKT" and r["good_after"]) for r, _ in siblings):
            raise ValueError("remaining position has an exit group without its stop/time protection")
        coverage += next(iter(quantities))
    if not math.isclose(coverage, abs(position), rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"current exits cover {coverage:g} units but position holds {abs(position):g}")


def execution_quantity(evidence, wanted, ref):
    fills = [e for e in evidence["executions"] if matches(e["identity"], wanted)
             and (wanted[4] > 0 or e.get("ref") == ref)]
    ids = {}
    for fill in fills:
        execution_id = fill.get("exec_id", "")
        if not execution_id:
            raise ValueError("execution evidence lacks a unique execution ID")
        base, sep, revision = execution_id.rpartition(".")
        if not sep or revision != "01":
            raise ValueError("corrected/unrecognized execution requires terminal order confirmation")
        if base in ids and ids[base] != fill:
            raise ValueError("conflicting execution evidence")
        ids[base] = fill
    total = sum(number(e["shares"]) for e in ids.values())
    cumulative = max([number(e["cumulative"]) for e in ids.values()] or [0])
    if not math.isclose(total, cumulative, rel_tol=0, abs_tol=1e-8):
        raise ValueError("execution history is incomplete or inconsistent")
    return total


def stopped_before_submission_detail(record):
    """Say what the stopped workflow DID, from the record, not what it didn't.

    The old text ("existing changes retained ... no broker changes made") read
    as though nothing had happened, when a full close had in fact cancelled
    every exit and left the position unprotected (RTX/RY/ROST 2026-09-23).
    """
    cancelled, resized = actions.exit_changes(record)
    done = []
    if cancelled:
        done.append(f"removed {len(cancelled)} exit leg(s) ({', '.join(cancelled)})")
    if resized:
        done.append(f"resized {len(resized)} exit leg(s) ({', '.join(resized)})")
    if not done and record.get("mutation") in {"resize exit", "cancel exit"}:
        done.append(f"was mid-way through '{record['mutation']}' (the exact leg is not itemised)")
    if done:
        lead = ("Earlier workflow stopped before entry/close submission, after it "
                + " and ".join(done) + "; those exit changes are still in effect at the broker. "
                "No close or entry was placed and none is replayed")
    else:
        lead = ("Earlier workflow stopped before entry/close submission and before changing any exit. "
                "No trade replayed")
    return lead + ". Current broker orders and holdings verified"


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
        result = outcome(False, stopped_before_submission_detail(record))
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
        result = outcome(bool(total_filled), f"Broker confirms addition entries are terminal: {total_filled:g} filled; "
                         "no entry or exit replayed", dict(filled=total_filled))
    elif record.get("wire") and record.get("mutation") != "stage addition and attached exits":
        wanted = record["wire"]
        ref = record.get("close_order_ref") or f"EXEC|{record['id']}|unified-close"
        row = lookup(evidence, wanted, ref=ref)
        # A terminal order echo is authoritative. Execution-only recovery needs
        # unique, complete, uncorrected fills; max(cumQty) can overstate busts.
        fill_rows = [e for e in evidence["executions"] if matches(e["identity"], wanted)
                     and (wanted[4] > 0 or e.get("ref") == ref)]
        if any(e.get("exec_id", "").rpartition(".")[2] != "01" for e in fill_rows):
            raise ValueError("broker reports an execution correction; quantity needs corrected reconciliation")
        filled = (row["filled"] or 0) if row and row["status"] in TERMINAL else execution_quantity(evidence, wanted, ref)
        qty = number(record["quantity"])
        if row:
            if row["qty"] != qty or row["action"] != record["closing"]:
                raise ValueError("close terms changed at broker; inspect exact order")
            if row["status"] not in TERMINAL:
                raise ValueError("close order is still working at broker")
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
    warnings = []
    if "modify" not in record:
        try:
            check_coverage(record, evidence)
        except ValueError as exc:
            # This is broker state we can explain, not an uncertain execution.
            # Report it without reviving an old plan or vetoing the next user
            # instruction. The next instruction retains its existing rules.
            warnings.append(str(exc))
            result["detail"] += (". Current exit discrepancy: " + str(exc)
                                 + "; this reconciliation made no broker changes")
    resolved = copy.deepcopy(record)
    resolved.update(phase="done", result=result,
                    resolution=dict(kind="broker_readback", evidence=evidence, no_replay=True, warnings=warnings))
    return resolved


def refresh_target(ib, root, account, con_id, *, open_reader=None):
    """Resolve each stopped receipt independently from one consistent snapshot."""
    targets = [(folder, r) for folder in (root, root / "order_edits") for r in actions.records(folder)
               if r["phase"] != "done" and r["payload"]["_broker_account"] == account
               and int(r["payload"]["con_id"]) == int(con_id)
               and not (folder == root and r["phase"] == "pending" and r.get("wire"))]
    if not targets:
        return {}
    evidence = capture(ib, account, int(con_id), open_reader=open_reader)
    results = {}
    for folder, record in targets:
        try:
            resolved = resolve(record, evidence)
            actions.save(folder, resolved)
            results[record["id"]] = resolved["result"]
        except ValueError as exc:
            results[record["id"]] = dict(ok=False, state="unknown", detail=str(exc), fill=None)
            record["observation"] = dict(at=evidence["at"], detail=str(exc))
            actions.save(folder, record)
    return results
