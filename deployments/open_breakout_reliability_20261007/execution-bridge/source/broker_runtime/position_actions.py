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
    from .execution_contracts import qualify_position
except ImportError:
    import execution_lifecycle as life
    from execution_contracts import qualify_position

TERMINAL = {"Filled", "Cancelled", "ApiCancelled", "Inactive"}
ACKNOWLEDGED = {"Submitted", "PreSubmitted"}
OPERATOR_NOTE_MIN = 8


class PreTransmitRefusal(ValueError):
    """A refusal decided BEFORE any broker mutation was attempted.

    Nothing was sent, so the correct report is ``state="rejected"``. The command
    boundary maps this class specifically; a bare ValueError from further in
    still degrades to ``unknown`` because it cannot prove non-delivery.

    ``lock`` carries the structured identity of the blocking action when there
    is one, so the Execution tab can offer to clear exactly that action instead
    of asking the operator to find it.

    ``snapshot`` carries the live book a refusal was decided against, when the
    refusal only makes sense next to the evidence -- a resolve refused because
    the action's own orders are still working has to show WHICH ones.
    """

    def __init__(self, message, lock=None, snapshot=None):
        super().__init__(message)
        self.lock = lock if isinstance(lock, dict) else None
        self.snapshot = snapshot if isinstance(snapshot, dict) else None


def blocking_lock(record, kind):
    """The structured identity of the action that blocks this command.

    Field names are the site contract (`docs/site_execution_schema.md`,
    `lockRejection()`): symbol, action_type, action_id, created_at, discrepancy.
    ``action_type`` is the journalled command type where the record has one;
    records written before this change carry none, so it degrades to the generic
    kind rather than guessing a type from the record's shape.
    """
    payload = record.get("payload") or {}
    con_id = payload.get("con_id") or (record.get("identity") or [None, None])[1]
    return {"symbol": str(payload.get("symbol") or "").upper() or f"conId {con_id}",
            "action_type": str(record.get("command_type") or "").strip() or kind,
            "action_id": str(record.get("id") or ""),
            "created_at": str(record.get("created") or record.get("session") or ""),
            "discrepancy": str((record.get("observation") or {}).get("detail")
                               or record.get("error")
                               or ((record.get("result") or {}).get("detail") if record.get("result") else "")
                               or f"phase {record.get('phase')}"),
            "account": str(record.get("account_key") or "")}


def blocking_summary(lock):
    """One line naming WHICH earlier action blocks this one, and since when.

    The audit's complaint about 'An earlier position action is unresolved' was
    that it never said which one, on what, from when, or what the broker
    actually disagreed about. Kept as prose as well as structure because a
    relay that drops the structured field must still print something useful.
    """
    return (f"blocked by {lock['action_type']} {lock['action_id'] or 'unknown'} "
            f"on {lock['symbol']} (opened {lock['created_at'] or 'unknown time'}): "
            f"{lock['discrepancy']}")


def exit_changes(record):
    """(cancelled keys, resized keys) this record's workflow actually did.

    ``changes`` is written by adjust_exits since 2026-09-23; older records fall
    back to ``removed`` (cancellations only; resizes were never itemised).
    """
    changes = [c for c in (record.get("changes") or []) if isinstance(c, dict)]
    if changes:
        labels = {"cancelled": "", "oca_sibling_cancelled": " [OCA-cancelled by IBKR]",
                  "oca_sibling_absent": " [gone from the book; cancellation NOT confirmed]"}
        # A later restore takes a cancelled rung back out of ``removed``.
        removed = {str(k) for k in (record.get("removed") or [])}
        cancelled = [str(c.get("key")) + labels[c.get("kind")] for c in changes
                     if c.get("kind") in labels and str(c.get("key")) in removed]
        resized = [str(c.get("key")) for c in changes if c.get("kind") == "resized"]
        return cancelled, resized
    return [str(k) for k in (record.get("removed") or [])], []


def exit_change_summary(record):
    """What the stopped workflow left changed at the broker, for the operator."""
    if record is None:
        return ""
    cancelled, resized = exit_changes(record)
    if record.get("wire") or record.get("addition"):
        return ""
    parts = []
    if cancelled:
        parts.append(f"{len(cancelled)} exit leg(s) removed ({', '.join(cancelled)})")
    if resized:
        parts.append(f"{len(resized)} exit leg(s) resized ({', '.join(resized)})")
    if not parts:
        return ""
    text = " -- NO close order was placed; this run " + " and ".join(parts)
    if cancelled and not resized and len(cancelled) >= len(record.get("legs") or []):
        text += "; the position currently has NO exits from this bracket -- re-attach exits or close it in TWS"
    return text


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
                if field == "good_after" and leg.get("order_type") in {"STP", "STP LMT"}:
                    # A stop's goodAfterTime is its ARMING time (day-2 arming,
                    # eq_order_entry and the site's stop_arm=next_session). Once
                    # it has passed the stop is simply live; it is not an
                    # expired deadline and must not refuse an Add/re-add.
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


REPLACE_ATTEMPTS = 5
REPLACE_FIRST_DELAY_S = 0.1
REPLACE_MAX_DELAY_S = 1.0


def replace_with_retry(source, target, *, attempts=REPLACE_ATTEMPTS, sleep=None):
    """os.replace, retried on PermissionError (Windows WinError 5 / 32).

    The journal lives in a OneDrive-synced folder. The sync client (and AV)
    briefly opens the destination after each write, and a rename onto a file
    another process holds open fails with Access is denied. 2026-09-23 NOVT:
    that one refusal, between resizing the exits and sending the close, left
    the action at phase=attention. Backoff 0.1, 0.2, 0.4, 0.8 s (capped at
    1 s); the final failure re-raises the original error unchanged.
    """
    import time
    sleep = sleep or time.sleep
    delay = REPLACE_FIRST_DELAY_S
    for attempt in range(1, attempts + 1):
        try:
            os.replace(source, target)
            return attempt
        except PermissionError:
            if attempt >= attempts:
                raise
            sleep(delay)
            delay = min(delay * 2, REPLACE_MAX_DELAY_S)


def save(root, record):
    root.mkdir(parents=True, exist_ok=True)
    target = record_path(root, record["id"])
    temporary = target.with_suffix(".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump({k: v for k, v in record.items() if not k.startswith("_")}, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    replace_with_retry(temporary, target)


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
    # Keep full raw order echoes and require fresh fill counters. A cached
    # default zero cannot establish that an exit has not partially filled.
    if any(not hasattr(t.orderStatus, "filled") for t in rows):
        raise ValueError("Fresh broker fill quantities are unavailable; refresh before resizing")
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


def oca_sibling_outcome(connection, native, leg):
    """Why a 0-quantity OCA sibling is no longer an open order: cancelled or filled.

    ``native`` is the owning client's own Trade, resolved before the first
    mutation, so its status callbacks are the broker's word on this order. The
    completed-order and execution reads are the fallback when that callback has
    not landed. Returns ("cancelled" | "absent" | "filled" | "working", detail).
    "absent" means the order is gone with no evidence either way; the caller's
    position re-read before the close is what then catches a fill. "working"
    means the native order still reports a live status after the wait -- it is
    NOT treated as removed, and the caller aborts.
    """
    status = str(life.wait_cancelled(connection, native).status)
    filled = float(getattr(native.orderStatus, "filled", 0) or 0)
    if status == "Filled" or filled > float(leg["filled_before"]):
        return "filled", f"status {status}, filled {filled:g}"
    if status in {"Cancelled", "ApiCancelled"}:
        return "cancelled", f"broker status {status}"
    if status not in life.TERMINAL:
        return "working", f"native status still {status or 'unknown'} after the wait"
    wanted = leg["identity"]
    try:
        completed = list(connection.reqCompletedOrders(apiOnly=False) or [])
    except Exception:  # noqa: BLE001 - evidence only
        completed = []
    for trade in completed:
        if not same_order(trade, wanted):
            continue
        done = str(trade.orderStatus.status)
        reported = getattr(trade.order, "filledQuantity", None)
        try:
            reported = float(reported)
        except (TypeError, ValueError):
            reported = float("nan")
        if done == "Filled" or (math.isfinite(reported) and reported > float(leg["filled_before"])):
            return "filled", f"completed order {done}"
        if done in {"Cancelled", "ApiCancelled"}:
            return "cancelled", f"completed order {done}"
    reader = getattr(connection, "reqExecutions", None)
    if callable(reader):
        try:
            for fill in reader() or []:
                if int(getattr(fill.execution, "permId", 0) or 0) == int(wanted[4]):
                    return "filled", "execution report present"
        except Exception:  # noqa: BLE001 - evidence only
            pass
    return "absent", f"no open order, last status {status or 'unknown'}"


def adjust_exits(ns, ib, record, total, root, host, port, cid):
    """Resize remaining quantities; cancel zero rungs; restore only our cancellations."""
    payload = record["payload"]
    desired = scaled_legs(record["legs"], total)
    rows = ns["_orders_for_contract"](ib, record["_contract"], payload["_broker_account"])
    if any(not hasattr(t.orderStatus, "filled") for t in rows):
        raise ValueError("Fresh broker fill quantities are unavailable; refresh before resizing")
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
    changes = record.setdefault("changes", [])
    cancelled_ocas = set()
    with life.owners(ns, ib, host, port, cid, [t for t, _, _ in plans]) as resolved:
        for (connection, original), (_, target, leg) in zip(resolved, plans):
            group = str(leg.get("oca_group") or "")
            try:
                trade = life.fresh_for_edit(ns, connection, tuple(leg["identity"]))
            except ValueError as exc:
                # 2026-09-16..09-23 full closes (MCHP/HXL/SNA/RTX/ENTG/JHX/RY/ROST):
                # every rung scales to 0, rung 1 of an OCA pair is cancelled,
                # IBKR takes its sibling terminal, and find_exact -- which skips
                # terminal rows -- matches ZERO orders. A 0-quantity leg whose
                # OCA sibling THIS run cancelled is exactly what we wanted gone,
                # PROVIDED the broker shows it Cancelled rather than Filled.
                if not leg["scaled_qty"] and group and group in cancelled_ocas:
                    verdict, detail = oca_sibling_outcome(connection, original, leg)
                    if verdict == "filled":
                        raise ValueError(
                            f"exit {leg['source_key']} (OCA {group}) FILLED while its OCA sibling was "
                            f"being cancelled ({detail}); the broker has already reduced the position, "
                            "so no close was submitted") from exc
                    if verdict == "working":
                        raise ValueError(
                            f"exit {leg['source_key']} (OCA {group}) cannot be resolved as an open order yet "
                            f"its broker status is not terminal ({detail}); no close was submitted") from exc
                    record["removed"].append(leg["source_key"])
                    changes.append(dict(kind="oca_sibling_" + verdict, key=leg["source_key"],
                                        oca=group, detail=detail))
                    record["phase"] = "pending"
                    save(root, record)
                    continue
                raise ValueError(
                    f"{exc} [leg {leg['source_key']} -> {leg['scaled_qty']}, "
                    f"OCA {group or 'none'}; "
                    f"{len(record['removed'])} of {len(plans)} leg(s) already cancelled "
                    "this run]") from exc
            if whole(trade.orderStatus.filled or 0, allow_zero=True) != leg["filled_before"]:
                raise ValueError("exit fill changed during owner lookup")
            mark_mutating(root, record, "cancel exit" if not leg["scaled_qty"] else "resize exit")
            if not leg["scaled_qty"]:
                ns["guarded_cancel_order"](connection, trade.order)
                if life.wait_cancelled(connection, original).status not in {"Cancelled", "ApiCancelled"}:
                    raise ValueError("exit cancellation was not confirmed")
                record["removed"].append(leg["source_key"])
                changes.append(dict(kind="cancelled", key=leg["source_key"], oca=group))
                if group:
                    cancelled_ocas.add(group)
            else:
                order = copy.deepcopy(trade.order)
                order.totalQuantity, order.transmit = target, True
                ns["guarded_place_order"](
                    connection, trade.contract, order, account=order.account,
                    mutation_kind="modify" if getattr(order, "parentId", 0) else "exit",
                    signal_id=ns["_command_signal"](payload, f"normalize:{order.permId}:{target}"))
                life.wait_modified(connection, tuple(leg["identity"]),
                                   {"totalQuantity": target}, leg["filled_before"], ns=ns)
                changes.append(dict(kind="resized", key=leg["source_key"], oca=group,
                                    qty=leg["scaled_qty"]))
            record["phase"] = "pending"
            save(root, record)
    zeroed = [leg for _, _, leg in plans if not leg["scaled_qty"]]
    if zeroed:
        # Reverse risk: a close must never fill while an exit we meant to
        # remove is still working, or that exit becomes a naked opposite order.
        after = ns["_orders_for_contract"](ib, record["_contract"], payload["_broker_account"])
        survivors = [leg["source_key"] for leg in zeroed
                     if any(same_order(t, leg["identity"]) and str(t.orderStatus.status) not in TERMINAL
                            for t in after)]
        if survivors:
            raise ValueError(f"exit leg(s) {', '.join(survivors)} still working after cancellation; "
                             "no close was submitted")
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
    record["addition_requested"] = quantity
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
        try:
            from . import broker_reconciliation as observation
        except ImportError:
            import broker_reconciliation as observation
        results = observation.refresh_target(ib, root, record["payload"]["_broker_account"],
                                             record["payload"]["con_id"], open_reader=ns.get("_fresh_open_trades"))
        return results[record["id"]]
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


def open_orders_for(ns, ib, broker_account, con_id):
    """Every order on this exact account/contract, as plain rows.

    Reads through the executor's ``_fresh_open_trades`` when it is available:
    ``ib.openTrades()`` alone returns only THIS client's orders, and the whole
    point of the snapshot is to show the operator what every client left behind
    on the contract. Terminal rows are kept -- a stale working leg that has
    since gone Cancelled is exactly the evidence that clears a lock.
    """
    reader = ns.get("_fresh_open_trades") if hasattr(ns, "get") else None
    if callable(reader):
        trades = list(reader(ib))
    else:
        ib.reqAllOpenOrders()
        trades = list(ib.openTrades())
    rows = []
    for trade in trades:
        order = trade.order
        if (str(getattr(order, "account", "")) != broker_account
                or int(getattr(trade.contract, "conId", 0) or 0) != int(con_id)):
            continue
        rows.append({"client_id": int(getattr(order, "clientId", -1)),
                     "order_id": int(getattr(order, "orderId", 0) or 0),
                     "perm_id": int(getattr(order, "permId", 0) or 0),
                     "action": str(getattr(order, "action", "")),
                     "order_type": str(getattr(order, "orderType", "")),
                     "quantity": float(getattr(order, "totalQuantity", 0) or 0),
                     "filled": float(getattr(trade.orderStatus, "filled", 0) or 0),
                     "status": str(getattr(trade.orderStatus, "status", "")),
                     "order_ref": str(getattr(order, "orderRef", "") or ""),
                     "oca_group": str(getattr(order, "ocaGroup", "") or "")})
    return rows


def durable_keys(client_id, order_id, perm_id):
    """The identity forms an order can be recognised by, newest first.

    Mirrors the executor's ``_order_source_key``: a permId once the broker has
    assigned one, else the placing client's own orderId. Both are emitted so a
    record written before the order had a permId still matches the live row.
    """
    keys = set()
    if int(perm_id or 0) > 0:
        keys.add(f"perm:{int(perm_id)}")
    if int(order_id or 0) > 0:
        keys.add(f"client:{int(client_id or 0)}:order:{int(order_id)}")
    return keys


def action_fingerprint(record):
    """Every orderRef / OCA group / durable key this action is known to own.

    Assembled from whatever the record got as far as writing: the closing
    order's ref and wire identity, each exit leg's ref / OCA group / source key
    / identity, the legs it cancelled on the way (``removed``), the identity of
    the order an order-edit record was editing, and any attached-add results.
    An action that aborted early simply owns fewer keys; it never owns a key it
    did not put on the wire.
    """
    refs, ocas, keys = set(), set(), set()

    def add_wire(value):
        if isinstance(value, (list, tuple)) and len(value) >= 5:
            try:
                keys.update(durable_keys(value[2], value[3], value[4]))
            except (TypeError, ValueError):
                pass

    ref = str(record.get("close_order_ref") or "").strip()
    if ref:
        refs.add(ref)
    add_wire(record.get("wire"))
    add_wire(record.get("identity"))
    for leg in (record.get("legs") or []):
        if not isinstance(leg, dict):
            continue
        leg_ref = str(leg.get("order_ref") or "").strip()
        if leg_ref:
            refs.add(leg_ref)
        oca = str(leg.get("oca_group") or "").strip()
        if oca:
            ocas.add(oca)
        source_key = str(leg.get("source_key") or "").strip()
        if source_key:
            keys.add(source_key)
        add_wire(leg.get("identity"))
    for source_key in (record.get("removed") or []):
        if str(source_key or "").strip():
            keys.add(str(source_key).strip())
    for result in (record.get("addition") or []):
        if not isinstance(result, dict):
            continue
        result_ref = str(result.get("order_ref") or result.get("signal_id") or "").strip()
        if result_ref:
            refs.add(result_ref)
        for wire in (result.get("orders") or result.get("wire") or []):
            add_wire(wire)
    return {"refs": refs, "ocas": ocas, "keys": keys}


def attributed_to_action(row, fingerprint):
    """True when this live order row belongs to the action being resolved.

    Attribution is by the things the action itself stamped: the orderRef it
    minted, the OCA group its exits share, or the durable order identity it
    journalled. A bare symbol match is deliberately NOT enough -- the primary
    account runs the systematic book across ~1060 names and an unrelated OLV or
    OVS bracket on the same contract must not be mistaken for this action's.
    """
    ref = str(row.get("order_ref") or "").strip()
    if ref and ref in fingerprint["refs"]:
        return True
    oca = str(row.get("oca_group") or "").strip()
    if oca and oca in fingerprint["ocas"]:
        return True
    return bool(durable_keys(row.get("client_id"), row.get("order_id"),
                             row.get("perm_id")) & fingerprint["keys"])


def order_label(row):
    """How an order is named back to the operator: permId when it has one."""
    if int(row.get("perm_id") or 0) > 0:
        return f"perm {int(row['perm_id'])}"
    return f"order {int(row.get('order_id') or 0)}"


def positions_for(ib, broker_account, con_id):
    """Held rows on this exact account/contract."""
    ib.reqPositions()
    return [{"account": str(row.account), "con_id": int(row.contract.conId),
             "symbol": str(getattr(row.contract, "symbol", "")),
             "position": float(row.position), "avg_cost": float(getattr(row, "avgCost", 0) or 0)}
            for row in ib.positions()
            if str(row.account) == broker_account and int(row.contract.conId) == int(con_id)]


def resolve(ns, ib, payload, account_key):
    """Operator clearance of ONE unresolved position action. Sends NO order.

    Re-reads the live book for the action's exact account/contract, records the
    positions and working orders it saw into the action's state record, and
    marks the record done with ``resolved_by = "operator"``. This is the site's
    equivalent of the hand-written ``resolution`` blocks that until now could
    only be produced by a one-off script on the trading box, which is why an
    unresolved action latched a lock the Execution tab could create but never
    clear.

    Refuses (pre-transmit, so ``rejected``) when the action is unknown, already
    resolved, on another account, on another symbol, when the operator note is
    missing, or when THE ACTION'S OWN ORDERS ARE STILL WORKING.

    That last refusal is the one with teeth. Clearing the lock is what lets the
    next position command through, and the lock exists precisely because an
    action stopped half-way. If a leg the action placed is still live on the
    book -- same orderRef, same OCA group, or the same durable order identity --
    then it has not stopped, and releasing the lock invites a second command to
    act on a contract that already has a working order on it. The refusal names
    the order ids and returns the snapshot, so the operator can go cancel or
    let them fill and then resolve.

    Working orders that are NOT attributable to this action do not refuse --
    the primary account runs the systematic book across ~1060 names and an
    unrelated OLV or OVS bracket on the same contract is not this action's
    business. They are counted, named in the detail line, and carried in the
    snapshot so the clearance is still made with eyes open.
    """
    root = journal_root(ns)
    try:
        if account_key not in {"primary", "pa"}:
            raise PreTransmitRefusal("unknown execution account")
        action_id = str(payload.get("action_id") or "").strip()
        if not action_id:
            raise PreTransmitRefusal("action_id is required")
        note = str(payload.get("operator_note") or "").strip()
        if len(note) < OPERATOR_NOTE_MIN:
            raise PreTransmitRefusal(
                f"operator_note of at least {OPERATOR_NOTE_MIN} characters is required; "
                "clearing a lock is a judgement and the journal records who made it")
        symbol = str(payload.get("symbol") or "").strip().upper()
        with operation_lock(root):
            target, scope = None, "position action"
            for candidate_root, label in ((root, "position action"),
                                          (root / "order_edits", "order edit")):
                path = record_path(candidate_root, action_id)
                if path.exists():
                    target, scope = path, label
                    break
            if target is None:
                raise PreTransmitRefusal(f"no position action or order edit with id {action_id}")
            record = json.loads(target.read_text(encoding="utf-8"))
            if record.get("phase") == "done":
                raise PreTransmitRefusal(
                    f"{scope} {action_id} is already resolved; nothing to clear")
            record_payload = record.get("payload") or {}
            if str(record.get("account_key") or account_key) != account_key:
                raise PreTransmitRefusal(
                    f"{scope} {action_id} belongs to account {record.get('account_key')}")
            recorded_symbol = str(record_payload.get("symbol") or "").strip().upper()
            if symbol and recorded_symbol and symbol != recorded_symbol:
                raise PreTransmitRefusal(
                    f"{scope} {action_id} is on {recorded_symbol}, not {symbol}")
            broker_account = str(record_payload.get("_broker_account") or "")
            con_id = int(record_payload.get("con_id") or 0)
            if not broker_account or con_id <= 0:
                raise PreTransmitRefusal(
                    f"{scope} {action_id} has no exact account/contract to re-check")
            positions = positions_for(ib, broker_account, con_id)
            orders = open_orders_for(ns, ib, broker_account, con_id)
            fingerprint = action_fingerprint(record)
            for row in orders:
                row["attributed_to_action"] = attributed_to_action(row, fingerprint)
            held = sum(row["position"] for row in positions)
            live = [row for row in orders if row["status"] not in TERMINAL]
            mine = [row for row in live if row["attributed_to_action"]]
            theirs = [row for row in live if not row["attributed_to_action"]]
            snapshot = {"positions": positions, "open_orders": orders}
            if theirs:
                snapshot["unattributed_working"] = {
                    "count": len(theirs),
                    "order_ids": [int(row["order_id"]) for row in theirs],
                    "perm_ids": [int(row["perm_id"]) for row in theirs],
                    "labels": [order_label(row) for row in theirs],
                }
            if mine:
                raise PreTransmitRefusal(
                    f"{scope} {action_id} still has {len(mine)} working order(s) of its own on "
                    f"{recorded_symbol or con_id}: "
                    + ", ".join(f"{order_label(row)} ({row['status']})" for row in mine)
                    + ". Cancel them or let them go terminal before clearing the lock; "
                      "the snapshot records what the book held.",
                    None, snapshot)
            unrelated = ""
            if theirs:
                unrelated = (f" {len(theirs)} unrelated working order(s) remain on the contract "
                             f"and were NOT touched: "
                             + ", ".join(f"{order_label(row)} ({row['status']})" for row in theirs)
                             + ".")
            mine_note = ", none of them this action's" if live else ""
            detail = (f"{scope} {action_id} on {recorded_symbol or con_id} cleared by operator; "
                      f"live book now holds {held:g} unit(s) with {len(live)} working order(s)"
                      f"{mine_note}.{unrelated} "
                      f"Note: {note}")
            record["resolution"] = {
                "at": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
                "resolved_by": "operator", "note": note,
                "positions": positions, "open_orders": orders,
                "previous_phase": record.get("phase"), "previous_error": record.get("error"),
            }
            record["phase"] = "done"
            record["result"] = dict(ok=True, state="executed", detail=detail, fill=None)
            save(target.parent, record)
            # state stays "executed" on the wire: exec_agent only trusts a child
            # result in {executed, rejected, unknown}, and weakening that gate to
            # teach it one read-only word would weaken it for every transmit.
            # The agent renames this one to the site's "resolved" and forwards
            # the snapshot (docs/site_execution_schema.md).
            return ns["_out"](ok=True, state="executed", detail=detail, fill=None,
                              snapshot=snapshot)
    except PreTransmitRefusal as exc:
        # A refusal that was decided against the live book returns that book:
        # "still has working orders" is only actionable next to the rows.
        return ns["_out"](ok=False, state="rejected", detail=f"Nothing sent: {exc}", fill=None,
                          snapshot=getattr(exc, "snapshot", None))
    except Exception as exc:  # noqa: BLE001
        # This path never places an order, so a read failure is still a clean
        # refusal: the record is only written after both reads succeed.
        return ns["_out"](ok=False, state="rejected",
                          detail=f"Nothing sent: resolve could not read the live book ({exc})",
                          fill=None)


def run(ns, ib, payload, account_key, host, port, cid, *, adding=False):
    # Deliver only after the operation has saved its outcome. A closed output
    # pipe cannot change either a completed receipt or an acknowledged pending
    # close that still needs automatic fill/re-add reconciliation.
    outcome = _run(ns, ib, payload, account_key, host, port, cid, adding=adding)
    try:
        return ns["_out"](**outcome)
    except TypeError:
        # An _out without the structured side channel must still report the
        # outcome; losing the whole result is far worse than losing `lock`.
        if "lock" not in outcome:
            raise
        return ns["_out"](**{k: v for k, v in outcome.items() if k != "lock"})


def _run(ns, ib, payload, account_key, host, port, cid, *, adding=False):
    root = journal_root(ns)
    record = None
    try:
        if account_key not in {"primary", "pa"}:
            raise ValueError("unknown execution account")
        if not isinstance(payload.get("readd", False), bool):
            raise ValueError("readd must be boolean")
        if not payload.get("_broker_account") or not payload.get("con_id"):
            raise ValueError("exact account and contract are required")
        with operation_lock(root):
            try:
                from . import broker_reconciliation as observation
            except ImportError:
                import broker_reconciliation as observation
            if payload.get("observe_only"):
                results = observation.refresh_target(ib, root, payload["_broker_account"], payload["con_id"],
                                                     open_reader=ns.get("_fresh_open_trades"))
                return results.get(str(payload.get("_command_id")),
                                   dict(ok=True, state="executed", detail="Broker journal check complete", fill=None))
            command_id = str(payload.get("_command_id") or "")
            target = record_path(root, command_id)
            if target.exists():
                saved = json.loads(target.read_text(encoding="utf-8"))
                if saved["payload"] != {k: v for k, v in payload.items() if k != "reconcile_only"}:
                    raise ValueError("command id was reused with a different payload")
                record = saved
                if record["phase"] == "done":
                    return record["result"]
                return reconcile(ns, ib, record, root, host, port, cid)
            if payload.get("reconcile_only"):
                raise ValueError("no original operation to reconcile")
            observation.refresh_target(ib, root, payload["_broker_account"], payload["con_id"],
                                       open_reader=ns.get("_fresh_open_trades"))
            for previous in records(root):
                if previous["phase"] != "done" and previous["payload"]["_broker_account"] == payload["_broker_account"] and previous["payload"]["con_id"] == payload["con_id"]:
                    lock = blocking_lock(previous, "position action")
                    raise PreTransmitRefusal(
                        "An earlier position action is unresolved; reconcile it before another -- "
                        + blocking_summary(lock), lock)
            for previous in records(root / "order_edits"):
                if (previous["phase"] != "done" and previous["identity"][:2]
                        == [payload["_broker_account"], int(payload["con_id"])]):
                    lock = blocking_lock(previous, "order edit")
                    raise PreTransmitRefusal(
                        "An earlier order edit is unresolved; reconcile it before a position action -- "
                        + blocking_summary(lock), lock)
            # Positions callbacks can omit routing metadata. Resolve the exact
            # held instrument before changing any exits or staging an addition.
            position = qualify_position(ib, current_position(ns, ib, payload))
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
                context["ref"] = position.contract
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
                      "created": datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds"),
                      "command_type": str(ns.get("_COMMAND_TYPE") or ""),
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
                               # Existing fill reconciliation consumes top-level
                               # order_ids; include entry parents, never exits.
                               fill={"add": result, "order_ids": [
                                   parent["order_id"] for allocation in result
                                   for parent in allocation["parent"]]})
                record["phase"], record["result"] = "done", outcome
                save(root, record)
                return outcome
            remaining = held - quantity
            adjust_exits(ns, ib, record, remaining, root, host, port, cid)
            record["exit_total"] = remaining
            live = current_position(ns, ib, payload)
            if live.position != (held if record["long"] else -held):
                raise ValueError("Position changed while exits were adjusted; no close submitted")
            # The fresh inventory check above resolves the same account/conId;
            # use the already-qualified copy without mutating IB's position cache.
            contract = position.contract
            order = ns["LimitOrder"](closing, quantity, float(payload["limit"])) if typ == "LMT" else ns["MarketOrder"](closing, quantity)
            order.account, order.tif = payload["_broker_account"], tif
            order.outsideRth = bool(payload.get("outside_rth"))
            order.orderId = ib.client.getReqId()
            order.orderRef = ns["_command_signal"](payload, "unified-close")
            record["_close_marking"] = True
            record["_mutation_before_close"] = record.get("mutation", "")
            record["close_order_ref"] = order.orderRef
            record["wire"] = [payload["_broker_account"], int(contract.conId), cid, order.orderId, 0]
            mark_mutating(root, record, "submit close")
            # In memory only (underscore keys are never journalled). Set once
            # the durable marker is on disk, immediately before the send: an
            # exception while it is False proves the close never left.
            record["_close_sent"] = True
            trade = ns["guarded_place_order"](ib, contract, order, mutation_kind="exit",
                                            account=order.account, signal_id=order.orderRef)
            record["wire"][4] = int(trade.order.permId or 0)
            record["phase"] = "pending"
            save(root, record)
            for _ in range(12 if typ == "MKT" else 2):
                ib.sleep(1)
                if trade.orderStatus.status in TERMINAL:
                    break
            return reconcile(ns, ib, record, root, host, port, cid, trade)
    except Exception as exc:
        if (record is not None and record.get("_close_marking") and not record.get("_close_sent")
                and not isinstance(exc, PreTransmitRefusal)):
            # The close marker was being written (or its save failed) and the
            # send never happened. Journalling that wire would describe a close
            # that does not exist and pin the record at attention forever (NOVT
            # 2026-09-23); keep the record truthful so readback can retire it.
            record.pop("wire", None)
            record.pop("close_order_ref", None)
            record["mutation"] = record.get("_mutation_before_close", "")
        # A PreTransmitRefusal is decided before this command touches the broker,
        # so an EARLIER command's mutation marker on a reloaded record must not
        # promote it to "unknown".
        attempted = (not isinstance(exc, PreTransmitRefusal)
                     and record is not None
                     and any(record.get(key) for key in ("mutation", "wire", "addition")))
        detail = f"{'Reconcile in TWS; do not repeat this action' if attempted else 'Nothing changed'}: {exc}"
        if attempted:
            detail += exit_change_summary(record)
        outcome = dict(ok=False, state="unknown" if attempted else "rejected", detail=detail, fill=None)
        lock = getattr(exc, "lock", None)
        if isinstance(lock, dict):
            # Site contract: the Clear-lock control needs the structured
            # identity, not just the prose in detail (10318876, 2026-09-23).
            outcome["lock"] = lock
        if record is not None:
            record["phase"], record["error"] = ("attention" if attempted else "done"), str(exc)
            if not attempted:
                # The operation can fail while resolving owners, before the
                # first mutation marker. Preserve that rejection as terminal so
                # a zero-send failure cannot block later manual corrections.
                record["result"] = outcome
            try:
                save(root, record)
            except Exception as journal_exc:  # noqa: BLE001
                # Still report: a lost result is worse than an unsaved error
                # marker, and the on-disk record keeps its last durable phase.
                outcome["detail"] += f" [journal update failed: {journal_exc}]"
        return outcome
