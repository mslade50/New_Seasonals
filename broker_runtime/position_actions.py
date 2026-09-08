"""Unified position actions. Imported only by the reviewed broker candidate.

The order journal is written before each broker mutation. A lost acknowledgement
never authorizes another submission. Pending closes are reconciled by the agent
using the same command id; a new request cannot overtake an unresolved operation.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

try:
    from . import execution_lifecycle as life
except ImportError:
    import execution_lifecycle as life

TERMINAL = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
ACKNOWLEDGED = {"Submitted", "PreSubmitted"}


def whole(value, name="quantity", allow_zero=False):
    number = float(value)
    if not math.isfinite(number) or number != int(number) or number < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} whole number")
    return int(number)


def allocate(weights, total):
    """Largest remainder, stable ties; zero-size rungs are deliberately removed."""
    total = whole(total, allow_zero=True)
    weights = [whole(w) for w in weights]
    if not weights:
        return []
    # Integer arithmetic avoids floating-point tie drift.
    denominator = sum(weights)
    base = [total * w // denominator for w in weights]
    ranks = sorted(range(len(weights)), key=lambda i: (-(total * weights[i] % denominator), i))
    for i in ranks[:total - sum(base)]:
        base[i] += 1
    return base


def groups(legs):
    result = {}
    for leg in legs:
        key = leg.get("oca_group") or leg["source_key"]
        result.setdefault(key, []).append(leg)
    for siblings in result.values():
        if len({whole(leg["qty"]) for leg in siblings}) != 1:
            raise ValueError("OCA siblings have unequal remaining quantities; their exit structure needs review")
        if len(siblings) > 1 and any(leg.get("oca_type", 1) not in (1, 2) for leg in siblings):
            raise ValueError("Non-blocking OCA exits cannot establish bounded closing coverage")
    return list(result.values())


def scaled_legs(legs, total):
    rungs = groups(legs)
    sizes = allocate([rung[0]["qty"] for rung in rungs], total)
    return [dict(leg, scaled_qty=size) for rung, size in zip(rungs, sizes) for leg in rung]


def validate_topology(legs):
    try:
        if not legs or not any(leg["order_type"] == "STP" or
                               (leg["order_type"] == "MKT" and leg["good_after"]) for leg in legs):
            raise ValueError("Add/re-add requires a price stop or scheduled time stop")
        groups(legs)
        for leg in legs:
            for field in ("good_after", "good_till"):
                if not leg.get(field):
                    continue
                match = re.fullmatch(r"(\d{8})[ -](\d{2}:\d{2}:\d{2})(?: ([A-Za-z_/]+))?", str(leg[field]))
                if not match:
                    raise ValueError("Inherited exit timing is unreadable")
                zone = ZoneInfo(match[3] or "America/New_York")
                deadline = datetime.strptime(match[1] + " " + match[2], "%Y%m%d %H:%M:%S").replace(tzinfo=zone)
                if deadline <= datetime.now(zone):
                    raise ValueError("An inherited exit deadline has expired")
        return None
    except (ValueError, TypeError, KeyError) as exc:
        return str(exc)


def validate_total(legs, total):
    try:
        scaled_legs(legs, total)
        return None
    except (ValueError, TypeError) as exc:
        return str(exc)


def journal_root(ns):
    return Path(ns.get("POSITION_ACTION_STATE_DIR") or Path(ns["__file__"]).parent / "data" / "position_actions")


def record_path(root, command_id):
    if not str(command_id or "").strip():
        raise ValueError("durable command id required")
    return root / (hashlib.sha256(command_id.encode()).hexdigest() + ".json")


def save(root, record):
    root.mkdir(parents=True, exist_ok=True)
    target = record_path(root, record["id"])
    temporary = target.with_suffix(".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump({k: v for k, v in record.items() if not k.startswith("_")}, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)


@contextmanager
def operation_lock(root):
    """Serialize this executor's journal and mutations, including across restarts."""
    root.mkdir(parents=True, exist_ok=True)
    with (root / "operations.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def records(root):
    # Corrupt state is an error, never an empty deduplication set.
    for path in sorted(root.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("version") != 1 or record.get("phase") not in {"pending", "mutating", "attention", "done"}:
            raise ValueError("invalid position-action journal")
        yield record


def key(trade):
    return list(life.identity(trade))


def same_order(trade, wanted):
    order = trade.order
    if str(getattr(order, "account", "")) != wanted[0] or int(trade.contract.conId) != wanted[1]:
        return False
    perm = int(getattr(order, "permId", 0) or 0)
    if wanted[4] > 0:
        return perm == wanted[4]
    return int(order.clientId) == wanted[2] and int(order.orderId) == wanted[3]


def current_position(ns, ib, payload, *, permit_flat=False):
    ib.reqPositions()
    rows = [p for p in ib.positions() if str(p.account) == payload["_broker_account"]
            and int(p.contract.conId) == int(payload["con_id"]) and p.position]
    if not rows and permit_flat:
        return None
    position, error = ns["_exact_position"](ib, payload)
    if error:
        raise ValueError(error)
    whole(abs(position.position), "position")
    return position


def exit_snapshot(ns, ib, position, payload):
    rows = ns["_orders_for_contract"](ib, position.contract, payload["_broker_account"])
    closing = "SELL" if position.position > 0 else "BUY"
    parents = {(int(t.order.clientId), int(t.order.orderId)): t for t in rows}
    exits = []
    for trade in rows:
        if str(trade.order.action).upper() != closing:
            continue
        parent = parents.get((int(trade.order.clientId), int(getattr(trade.order, "parentId", 0) or 0)))
        if parent is not None:
            if float(parent.orderStatus.filled or 0):
                raise ValueError("An entry bracket is partially filled; reconcile it before resizing this position")
            continue
        exits.append(trade)
    legs, error = ns["_capture_exit_legs"](exits, closing)
    if error:
        raise ValueError(error)
    for leg, trade in zip(legs, exits):
        leg["identity"] = key(trade)
        leg["filled_before"] = whole(trade.orderStatus.filled or 0, allow_zero=True)
        leg["qty"] = whole(float(trade.order.totalQuantity) - leg["filled_before"])
        # Retain full order objects in broker memory only; journal stores the
        # supported fields from _capture_exit_legs, never credentials.
    groups(legs)
    return legs, exits


def mark_mutating(root, record, detail):
    record["phase"], record["mutation"] = "mutating", detail
    save(root, record)


def adjust_exits(ns, ib, record, total, root, host, port, cid):
    """Resize remaining quantities; cancel zero rungs; restore only our cancellations."""
    payload = record["payload"]
    desired = scaled_legs(record["legs"], total)
    rows = ns["_orders_for_contract"](ib, record["_contract"], payload["_broker_account"])
    plans, recreate = [], []
    for leg in desired:
        found = [t for t in rows if same_order(t, leg["identity"])]
        if len(found) > 1:
            raise ValueError("duplicate exact exit identity")
        if not found:
            if leg["source_key"] not in record["removed"]:
                raise ValueError("An exit disappeared or filled during the action; reconcile before further changes")
            if leg["scaled_qty"]:
                recreate.append(leg)
            continue
        trade = found[0]
        filled = whole(trade.orderStatus.filled or 0, allow_zero=True)
        if filled != leg["filled_before"]:
            raise ValueError("An exit filled during this action; reconcile the remaining allocation")
        target = filled + leg["scaled_qty"]
        if target != whole(trade.order.totalQuantity) or not leg["scaled_qty"]:
            plans.append((trade, target, leg))
    # Resolve ALL clients before the first order changes. Reductions precede
    # increases so normalizing incomplete coverage does not transiently add it.
    plans.sort(key=lambda item: (item[1] > float(item[0].order.totalQuantity), item[1]))
    with life.owners(ns, ib, host, port, cid, [t for t, _, _ in plans]) as resolved:
        for (connection, original), (_, target, leg) in zip(resolved, plans):
            trade = life.find_exact(connection, tuple(leg["identity"]))
            if whole(trade.orderStatus.filled or 0, allow_zero=True) != leg["filled_before"]:
                raise ValueError("exit fill changed during owner lookup")
            mark_mutating(root, record, "cancel exit" if not leg["scaled_qty"] else "resize exit")
            if not leg["scaled_qty"]:
                ns["guarded_cancel_order"](connection, trade.order)
                connection.sleep(0.5)
                if trade.orderStatus.status not in {"Cancelled", "ApiCancelled"}:
                    raise ValueError("exit cancellation was not confirmed")
                record["removed"].append(leg["source_key"])
            else:
                order = copy.deepcopy(trade.order)
                order.totalQuantity, order.transmit = target, True
                placed = ns["guarded_place_order"](
                    connection, trade.contract, order, account=order.account,
                    mutation_kind="modify" if getattr(order, "parentId", 0) else "exit",
                    signal_id=ns["_command_signal"](payload, f"normalize:{order.permId}:{target}"))
                connection.sleep(0.5)
                verified = life.find_exact(connection, tuple(leg["identity"]))
                if placed.orderStatus.status not in ACKNOWLEDGED or whole(verified.order.totalQuantity) != target:
                    raise ValueError("exit resize lacks broker acknowledgement")
            record["phase"] = "pending"
            save(root, record)
    if recreate:
        mark_mutating(root, record, "restore previously cancelled exit rungs")
        # Each restored OCA rung is independent and preserves its prices/timing.
        for rung in groups(recreate):
            size = rung[0]["scaled_qty"]
            normalized = [dict(leg, qty=size) for leg in rung]
            trades, _ = ns["_place_exit_legs"](
                ib, record["_contract"], normalized, record["closing"], size,
                payload["_broker_account"], ns["_command_signal"](payload, "restore-rung"))
            if ns["_placement_problem"](trades) or len(trades) != len(rung):
                raise ValueError("restored exits need broker reconciliation")
            for old, trade in zip(rung, trades):
                stored = next(x for x in record["legs"] if x["source_key"] == old["source_key"])
                stored["identity"] = key(trade)
                stored["filled_before"] = whole(trade.orderStatus.filled or 0, allow_zero=True)
                record["removed"].remove(old["source_key"])
        record["phase"] = "pending"
        save(root, record)


def find_close(ib, record):
    ib.reqAllOpenOrders()
    candidates = list(ib.openTrades())
    candidates.extend(ib.reqCompletedOrders(apiOnly=False))
    found = [t for t in candidates if same_order(t, record["wire"])]
    if not found:
        raise ValueError("close order is absent from broker open/completed orders; delivery remains unknown")
    # Some APIs expose one order in both caches during its terminal transition.
    terminal = [t for t in found if t.orderStatus.status in TERMINAL]
    return terminal[-1] if terminal else found[-1]


def added_risk(context, quantity):
    risks = []
    for rung in groups(context["legs"]):
        stops = [leg for leg in rung if leg["order_type"] == "STP"]
        distance = max(abs(context["risk_reference"] - float(leg["aux"])) for leg in stops) if stops else context["risk_reference"]
        risks.append(distance * quantity)
    return max(risks)


def stage_add(ns, ib, context, quantity, record, root, *, market):
    """Use native attached exits for the addition; never cancel the old bracket."""
    error = validate_topology(context["legs"])
    if error:
        raise ValueError(error)
    mark_mutating(root, record, "stage addition and attached exits")
    results = []
    adapter = dict(ns)
    adapter["_fast_guard_risk_usd"] = added_risk
    # One parent per allocation: IB may expand partial-sized attached children
    # to their parent's size. Every child therefore equals its own parent.
    rungs = groups(context["legs"])
    sizes = allocate([rung[0]["qty"] for rung in rungs], quantity)
    for index, (rung, size) in enumerate(zip(rungs, sizes)):
        if not size:
            continue
        local = dict(context, legs=[dict(leg, qty=size) for leg in rung])
        result, error = life.stage_attached(
            adapter, ib, local, size, ns["_command_signal"](record["payload"], f"attached-add:{index}"), market=market)
        results.append(result)
        record["addition"] = results
        save(root, record)
        if error:
            raise ValueError(error)
    return results


def reconcile(ns, ib, record, root, host, port, cid, close=None):
    if record["phase"] in {"attention", "mutating"}:
        raise ValueError("previous broker mutation needs reconciliation; no automatic repeat")
    position = current_position(ns, ib, record["payload"], permit_flat=True)
    # A stored, qualified contract is recovered from the actual close order
    # when the position has become flat.
    close = close or find_close(ib, record)
    record["_contract"] = close.contract
    if position and (position.position > 0) != record["long"]:
        raise ValueError("position direction changed; stop and reconcile")
    status = str(close.orderStatus.status)
    filled = whole(close.orderStatus.filled or 0, allow_zero=True)
    requested = record["quantity"]
    if filled > requested:
        raise ValueError("broker filled more than the recorded request")
    held = whole(abs(position.position), allow_zero=True) if position else 0
    # Conservative against a lagging positions callback after a partial fill.
    held = min(held, record["held"] - filled)
    working = 0 if status in TERMINAL else requested - filled
    if working > held:
        raise ValueError("working close exceeds current inventory; reconcile immediately")
    if status not in TERMINAL and status not in ACKNOWLEDGED:
        raise ValueError("close acknowledgement is ambiguous; no automatic retry")
    target = held - working
    if record["legs"] and target != record.get("exit_total"):
        adjust_exits(ns, ib, record, target, root, host, port, cid)
        record["exit_total"] = target
    record["filled"], record["status"] = filled, status
    will_readd = status in TERMINAL and record["payload"].get("readd") and filled
    record["phase"] = "pending"
    save(root, record)
    detail = f"Close {status}: {filled}/{requested} filled; exits allocated to {target}"
    if will_readd:
        if datetime.now(ZoneInfo("America/New_York")).date().isoformat() != record["session"]:
            raise ValueError("close completed after the re-add session; no new entry staged")
        if status != "Filled":
            # A cancelled partial close is complete. Only its actually filled
            # quantity is eligible for re-add, never the original request.
            detail += "; partial close is terminal"
        context = dict(record["add_context"], ref=record["_contract"])
        stage_add(ns, ib, context, filled, record, root, market=False)
        detail += f"; DAY re-add staged for {filled} with attached exits"
    result = {"ok": True, "state": "executed", "detail": detail,
            "fill": {"status": status, "filled": filled, "order_id": close.order.orderId,
                     "exits_resized_to": target, "readd": record.get("addition")}}
    record["result"] = result
    if status in TERMINAL:
        record["phase"] = "done"
    save(root, record)
    return result


def run(ns, ib, payload, account_key, host, port, cid, *, adding=False):
    root = journal_root(ns)
    record = None
    try:
        if account_key != "primary":
            raise ValueError("unified position actions are Primary-only")
        if not isinstance(payload.get("readd", False), bool):
            raise ValueError("readd must be boolean")
        if not payload.get("_broker_account") or not payload.get("con_id"):
            raise ValueError("exact account and contract are required")
        with operation_lock(root):
            command_id = str(payload.get("_command_id") or "")
            target = record_path(root, command_id)
            if target.exists():
                saved = json.loads(target.read_text(encoding="utf-8"))
                if saved["payload"] != {k: v for k, v in payload.items() if k != "reconcile_only"}:
                    raise ValueError("command id was reused with a different payload")
                record = saved
                if record["phase"] == "done":
                    return ns["_out"](**record["result"])
                return ns["_out"](**reconcile(ns, ib, record, root, host, port, cid))
            if payload.get("reconcile_only"):
                raise ValueError("no original operation to reconcile")
            for previous in records(root):
                if previous["phase"] != "done" and previous["payload"]["_broker_account"] == payload["_broker_account"] and previous["payload"]["con_id"] == payload["con_id"]:
                    raise ValueError("An earlier position action is unresolved; reconcile it before another")
            position = current_position(ns, ib, payload)
            held = whole(abs(position.position), "position")
            typ = str(payload.get("order_type") or "MKT").upper()
            if typ not in {"MKT", "LMT"}:
                raise ValueError("Close requires MKT or LMT")
            if typ == "LMT" and (not math.isfinite(float(payload.get("limit", 0))) or float(payload.get("limit", 0)) <= 0):
                raise ValueError("Limit close requires a positive finite price")
            if payload.get("outside_rth") and typ != "LMT":
                raise ValueError("Outside-RTH close requires LMT")
            tif = str(payload.get("tif") or "DAY").upper()
            if tif not in {"DAY", "GTC"}:
                raise ValueError("TIF must be DAY or GTC")
            if typ == "MKT":
                tif = "DAY"
            if payload.get("qty") not in (None, ""):
                quantity = whole(payload["qty"])
            else:
                fraction = float(payload.get("fraction", 1))
                if not math.isfinite(fraction) or fraction <= 0:
                    raise ValueError("percentage must be positive and finite")
                quantity = whole(math.floor(held * fraction + 0.5))
            if not adding and quantity > held:
                raise ValueError("Close quantity exceeds current holdings")
            if position.contract.secType not in {"STK", "FUT", "CASH"}:
                raise ValueError("Use a combo ticket to close options")
            if (adding or payload.get("readd")) and position.contract.secType != "STK":
                raise ValueError("Add and re-add currently support stocks only")
            closing = "SELL" if position.position > 0 else "BUY"
            if not adding and payload.get("action") != closing:
                raise ValueError("Requested direction would add to the current position")
            legs, exits = exit_snapshot(ns, ib, position, payload)
            context = None
            if adding or payload.get("readd"):
                if not legs:
                    raise ValueError("Add/re-add requires existing exits to inherit")
                # Reuse all existing exposure/risk gates, with normalized
                # coverage rather than requiring an exact original quantity.
                context, error = ns["_prepare_position_action_add"](ib, dict(payload, qty=quantity), account_key, partial=False)
                if error:
                    raise ValueError(error)
                context["legs"] = legs
                reference = float(context["avg_cost"])
                if adding:
                    quotes = ib.reqTickers(position.contract)
                    if len(quotes) != 1:
                        raise ValueError("A current quote is required for the Add risk check")
                    reference = float(quotes[0].ask if position.position > 0 else quotes[0].bid)
                if not math.isfinite(reference) or reference <= 0:
                    raise ValueError("A positive current price is required for Add/re-add")
                if quantity * reference > ns["_max_notional"](account_key):
                    raise ValueError("Add/re-add exceeds the existing notional cap")
                context["risk_reference"] = reference
            if adding and typ != "MKT":
                raise ValueError("Add currently requires MKT")
            record = {"version": 1, "id": command_id, "payload": dict(payload),
                      "account_key": account_key, "phase": "pending",
                      "held": held, "quantity": quantity, "long": position.position > 0,
                      "closing": closing, "legs": legs, "removed": [],
                      "session": datetime.now(ZoneInfo("America/New_York")).date().isoformat(),
                      "_contract": position.contract}
            if context:
                record["add_context"] = {k: context[k] for k in ("entry_action", "close_action", "avg_cost", "account", "legs", "risk_reference")}
            save(root, record)
            if adding:
                adjust_exits(ns, ib, record, held, root, host, port, cid)
                live = current_position(ns, ib, payload)
                if live.position != (held if record["long"] else -held):
                    raise ValueError("Position changed before Add; no entry submitted")
                result = stage_add(ns, ib, context, quantity, record, root, market=True)
                outcome = dict(ok=True, state="executed",
                               detail="Add submitted with its own proportional attached exits; existing exits retained",
                               fill={"add": result})
                record["phase"], record["result"] = "done", outcome
                save(root, record)
                return ns["_out"](**outcome)
            remaining = held - quantity
            adjust_exits(ns, ib, record, remaining, root, host, port, cid)
            record["exit_total"] = remaining
            live = current_position(ns, ib, payload)
            if live.position != (held if record["long"] else -held):
                raise ValueError("Position changed while exits were adjusted; no close submitted")
            contract = live.contract
            if contract.secType == "STK":
                contract.exchange = "SMART"
            ib.qualifyContracts(contract)
            order = ns["LimitOrder"](closing, quantity, float(payload["limit"])) if typ == "LMT" else ns["MarketOrder"](closing, quantity)
            order.account, order.tif = payload["_broker_account"], tif
            order.outsideRth = bool(payload.get("outside_rth"))
            order.orderId = ib.client.getReqId()
            order.orderRef = ns["_command_signal"](payload, "unified-close")
            record["wire"] = [payload["_broker_account"], int(contract.conId), cid, order.orderId, 0]
            mark_mutating(root, record, "submit close")
            trade = ns["guarded_place_order"](ib, contract, order, mutation_kind="exit",
                                            account=order.account, signal_id=order.orderRef)
            record["wire"][4] = int(trade.order.permId or 0)
            record["phase"] = "pending"
            save(root, record)
            for _ in range(12 if typ == "MKT" else 2):
                ib.sleep(1)
                if trade.orderStatus.status in TERMINAL:
                    break
            return ns["_out"](**reconcile(ns, ib, record, root, host, port, cid, trade))
    except Exception as exc:
        if record is not None:
            record["phase"], record["error"] = "attention", str(exc)
            save(root, record)
        return ns["_out"](False, "unknown" if record is not None else "rejected",
                          f"{'Reconcile in TWS; do not repeat this action' if record is not None else 'Nothing changed'}: {exc}")
