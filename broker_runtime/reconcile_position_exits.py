"""Proportionally reconcile existing exits to live inventory; never place a close."""
from __future__ import annotations

import copy
import hashlib
import json
from decimal import Decimal
from pathlib import Path

try:
    from . import execution_lifecycle as life
    from . import manual_order_actions as manual
except ImportError:
    import execution_lifecycle as life
    import manual_order_actions as manual

TERMS = ("action", "orderType", "lmtPrice", "auxPrice", "tif", "parentId", "ocaGroup",
         "ocaType", "goodAfterTime", "goodTillDate", "outsideRth", "orderRef", "transmit")


def number(value):
    value = Decimal(str(value))
    if not value.is_finite():
        raise ValueError("fresh finite quantities are required")
    return value


def validate(command):
    try:
        if command.get("account") not in {"primary", "pa"}:
            raise ValueError("unknown account")
        if int((command.get("payload") or {}).get("con_id") or 0) <= 0:
            raise ValueError("exact position contract required")
        return True, []
    except (ValueError, TypeError, OverflowError) as exc:
        return False, [str(exc)]


def snapshot(ns, ib, payload):
    ib.reqPositions()
    position, error = ns["_exact_position"](ib, payload)
    if error:
        raise ValueError(error)
    signed = number(position.position)
    if not signed:
        raise ValueError("position is flat; no exits were changed")
    account, con_id = payload["_broker_account"], int(payload["con_id"])
    if str(position.account) != account or int(position.contract.conId) != con_id:
        raise ValueError("position identity changed")
    rows = [t for t in ns["_orders_for_contract"](ib, position.contract, account)
            if str(t.order.account) == account and int(t.contract.conId) == con_id
            and t.orderStatus.status not in life.TERMINAL]
    parents = {(int(t.order.clientId), int(t.order.orderId)): t for t in rows}
    closing = "SELL" if signed > 0 else "BUY"
    exits = {}
    for trade in rows:
        order = trade.order
        if str(order.action).upper() != closing:
            continue
        parent = parents.get((int(order.clientId), int(getattr(order, "parentId", 0) or 0)))
        if parent is not None:
            if number(parent.orderStatus.filled or 0):
                raise ValueError("an entry is partially filled; wait for its bracket to settle before reconciling")
            continue  # This exit belongs to a still-working entry, not held shares.
        key = life.identity(trade)
        if key in exits:
            raise ValueError("duplicate exit identity")
        filled = number(trade.orderStatus.filled)
        remaining = number(order.totalQuantity) - filled
        if filled < 0 or remaining <= 0:
            raise ValueError("an exit has no reliable remaining quantity")
        group = ("oca", str(order.ocaGroup)) if getattr(order, "ocaGroup", "") else ("order", str(key))
        exits[key] = dict(trade=trade, filled=filled, remaining=remaining, group=group,
                          terms={field: copy.deepcopy(getattr(order, field, None)) for field in TERMS})
    return signed, exits


def allocation(exits, held):
    """Count each OCA group once, normalize siblings, and use largest remainder."""
    groups = {}
    for key in sorted(exits):
        groups.setdefault(exits[key]["group"], []).append(key)
    if not groups:
        return {}
    weights = [max(exits[k]["remaining"] for k in keys) for keys in groups.values()]
    # Keep whole-share allocations whole. For existing fractional quantities,
    # use their finest decimal unit without inventing unrepresentable holdings.
    decimals = max(0, *(-v.normalize().as_tuple().exponent for v in [held, *weights]))
    unit = Decimal(1).scaleb(-decimals)
    units = int(held / unit)
    denominator = sum(weights)
    raw = [Decimal(units) * weight / denominator for weight in weights]
    sizes = [int(value) for value in raw]
    rank = sorted(range(len(raw)), key=lambda i: (-(raw[i] - sizes[i]), i))
    for i in rank[:units - sum(sizes)]:
        sizes[i] += 1
    return {key: Decimal(size) * unit for keys, size in zip(groups.values(), sizes) for key in keys}


def run(ns, ib, payload, host, port, cid):
    root = Path(ns.get("POSITION_ACTION_STATE_DIR") or Path(ns["__file__"]).parent / "data" / "position_actions")
    record = None
    attempted = False
    try:
        command_id = str(payload.get("_command_id") or "").strip()
        if not command_id:
            raise ValueError("durable command id required")
        path = root / "order_edits" / (hashlib.sha256(command_id.encode()).hexdigest() + ".json")
        with manual.operation_lock(root):
            if path.exists():
                old = json.loads(path.read_text(encoding="utf-8"))
                if old.get("payload") != payload or old.get("kind") != "reconcile_exits":
                    raise ValueError("command id belongs to a different instruction")
                if old["phase"] == "done":
                    return ns["_out"](**old["result"])
                return ns["_out"](ok=False, state="unknown", detail="Reconcile was already attempted; refresh the broker before a new instruction", fill=None)
            signed, initial = snapshot(ns, ib, payload)
            desired = allocation(initial, abs(signed))
            changed = [k for k in initial if initial[k]["remaining"] != desired[k]]
            # Every reduction precedes every increase; zero allocations cancel
            # their complete rung rather than sending illegal zero-size orders.
            changed.sort(key=lambda k: (desired[k] > initial[k]["remaining"], k))
            record = dict(version=1, id=command_id, payload=dict(payload), kind="reconcile_exits",
                          identity=[payload["_broker_account"], int(payload["con_id"])],
                          phase="pending", completed=[], held=str(signed),
                          plan=[dict(identity=list(k), remaining=str(desired[k])) for k in changed])
            manual.save(path, record)
            expected = {k: item["remaining"] for k, item in initial.items()}
            cancelled = set()
            with life.owners(ns, ib, host, port, cid, [initial[k]["trade"] for k in changed]) as owners:
                native = {life.identity(t): (conn, t) for conn, t in owners}

                def verify():
                    current_signed, current = snapshot(ns, ib, payload)
                    if current_signed != signed:
                        raise ValueError("position changed during reconciliation; refresh before continuing")
                    # IBKR may cancel another zero-size OCA sibling with the
                    # first cancellation. Only broker-confirmed cancels count.
                    for key in set(expected) - set(current) - cancelled:
                        observed = native.get(key, (None, None))[1]
                        if (desired[key] == 0 and observed is not None
                                and observed.orderStatus.status in {"Cancelled", "ApiCancelled"}
                                and number(observed.orderStatus.filled or 0) == initial[key]["filled"]):
                            cancelled.add(key)
                    if set(current) != set(expected) - cancelled:
                        raise ValueError("working exits changed during reconciliation")
                    for key, item in current.items():
                        if (item["filled"] != initial[key]["filled"] or item["remaining"] != expected[key]
                                or item["terms"] != initial[key]["terms"]):
                            raise ValueError("exit quantity, fill, or terms changed during reconciliation")
                    return current

                for key in changed:
                    current = verify()
                    if key in cancelled:
                        continue
                    connection, original = native[key]
                    trade = life.fresh_for_edit(ns, connection, key)
                    if (number(trade.orderStatus.filled) != initial[key]["filled"]
                            or number(trade.order.totalQuantity) - number(trade.orderStatus.filled) != expected[key]
                            or any(getattr(trade.order, field, None) != value for field, value in initial[key]["terms"].items())):
                        raise ValueError("selected exit changed during owner lookup")
                    manual.pause_position_recovery(root, key, command_id)
                    record["phase"] = "mutating"
                    record["next_identity"] = list(key)
                    manual.save(path, record)
                    attempted = True
                    if desired[key] == 0:
                        connection.cancelOrder(original.order)
                        status = life.wait_cancelled(connection, original)
                        if status.status not in {"Cancelled", "ApiCancelled"} or number(status.filled or 0) != initial[key]["filled"]:
                            raise ValueError("zero-rung cancellation did not confirm without a fill")
                        cancelled.add(key)
                    else:
                        order = copy.deepcopy(trade.order)
                        target = initial[key]["filled"] + desired[key]
                        order.totalQuantity = float(target)
                        connection.placeOrder(trade.contract, order)
                        life.wait_modified(connection, key, {"totalQuantity": float(target)},
                                           float(initial[key]["filled"]), ns=ns)
                        expected[key] = desired[key]
                    record["completed"].append(list(key))
                    record["phase"] = "pending"
                    manual.save(path, record)
                current = verify()
                final_groups = {}
                for key, item in current.items():
                    if item["remaining"] != desired[key]:
                        raise ValueError("final exit quantity does not match allocation")
                    final_groups.setdefault(item["group"], set()).add(item["remaining"])
                if initial and (any(len(q) != 1 for q in final_groups.values())
                                or sum(next(iter(q)) for q in final_groups.values()) != abs(signed)):
                    raise ValueError("final exit coverage does not match the position")
            detail = (f"Reconciled exits to {abs(signed):g} units; prices and schedules preserved"
                      if changed else "Exit quantities already match the position" if initial else "No existing exits to reconcile")
            result = dict(ok=True, state="executed", detail=detail,
                          fill=dict(position=float(signed), resized=len(changed) - len(cancelled), cancelled=len(cancelled)))
            record.update(phase="done", result=result)
            manual.save(path, record)
            return ns["_out"](**result)
    except Exception as exc:
        result = dict(ok=False, state="unknown" if attempted else "rejected",
                      detail=f"{'Reconcile stopped after an order change; refresh before retrying' if attempted else 'Nothing changed'}: {exc}", fill=None)
        if record is not None:
            record.update(phase="attention" if attempted else "done", result=result)
            manual.save(path, record)
        return ns["_out"](**result)
