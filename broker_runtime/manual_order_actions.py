"""Operator-directed edits: broker routing and receipts, without trading policy.

Only the explicit cancel/modify command handlers call this module. Automated
entries, exits and position actions continue to use their existing guards.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

try:
    from . import execution_lifecycle as life
except ImportError:
    import execution_lifecycle as life


def validate(command):
    """Check the command address/encoding, never a cached portfolio or risk cap."""
    try:
        if command.get("account") not in {"primary", "pa"}:
            raise ValueError("unknown account")
        payload = dict(command.get("payload") or {})
        life.payload_identity(dict(payload, _broker_account=command["account"]))
        if command["type"] == "modify":
            changes(payload)
        elif command["type"] != "cancel":
            raise ValueError("manual order command must be cancel or modify")
        return True, []
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        return False, [str(exc)]


def changes(payload):
    values = {}
    for key, field in (("new_qty", "totalQuantity"), ("new_limit", "lmtPrice"),
                       ("new_stop", "auxPrice")):
        if payload.get(key) not in (None, ""):
            value = float(payload[key])
            if not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
            values[field] = value
    if not values:
        raise ValueError("no changed fields supplied")
    return values


@contextmanager
def operation_lock(root):
    # Share the position executor's lock, so an automatic recovery cannot race
    # the manual edit. This serializes delivery; it does not veto the action.
    root.mkdir(parents=True, exist_ok=True)
    with (root / "operations.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def save(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pending")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    _replace_with_retry(temporary, path)


def _replace_with_retry(source, target, attempts=5, sleep=None):
    # OneDrive/AV can hold the destination open for a moment (WinError 5);
    # same bounded backoff as position_actions.replace_with_retry.
    import time
    sleep = sleep or time.sleep
    delay = 0.1
    for attempt in range(1, attempts + 1):
        try:
            os.replace(source, target)
            return attempt
        except PermissionError:
            if attempt >= attempts:
                raise
            sleep(delay)
            delay = min(delay * 2, 1.0)


def pause_position_recovery(root, wanted, command_id):
    """A prior automatic close/re-add must not undo the operator's new edit."""
    for path in root.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            payload = record.get("payload") or {}
            if (record.get("phase") != "done"
                    and payload.get("_broker_account") == wanted[0]
                    and int(payload.get("con_id") or 0) == wanted[1]):
                record.update(phase="attention", manual_override=command_id,
                              error="Automatic recovery paused by manual order edit; reconcile before resuming")
                save(path, record)
        except (ValueError, TypeError, AttributeError):
            # Corrupt records already stop the automatic reader. They do not
            # prevent a separately identified manual cancel or modification.
            continue


def fresh_order(ns, connection, wanted, native):
    reader = ns.get("_orders_for_contract")
    if reader is None:
        return life.find_exact(connection, wanted)
    matches = []
    for row in reader(connection, native.contract, wanted[0]):
        try:
            if life.identity(row) == wanted:
                matches.append(row)
        except (ValueError, TypeError):
            continue
    if len(matches) != 1:
        raise ValueError("selected order is no longer uniquely working")
    return matches[0]


def mutate(ns, ib, payload, host, port, cid, modify, before_send):
    attempted = False
    try:
        wanted = life.payload_identity(payload)
        trade = life.find_exact(ib, wanted)
        with life.owners(ns, ib, host, port, cid, [trade]) as resolved:
            connection, native = resolved[0]
            if not modify:
                before_send()
                attempted = True
                connection.cancelOrder(native.order)
                for _ in range(24):
                    if native.orderStatus.status in life.TERMINAL:
                        break
                    connection.sleep(.25)
                status = native.orderStatus.status
                filled = float(native.orderStatus.filled or 0)
                if status not in {"Cancelled", "ApiCancelled"}:
                    raise ValueError(f"cancel not confirmed ({status}, filled {filled:g}); reconcile before retry")
                return dict(ok=True, state="executed", detail="Cancellation confirmed",
                            fill=dict(status=status, filled=filled))
            current = fresh_order(ns, connection, wanted, native)
            changed = changes(payload)
            order = copy.deepcopy(current.order)
            for field, value in changed.items():
                setattr(order, field, value)
            # Retain every unedited broker field, including parent/OCA/routing
            # and transmit state. A modification must not release a held parent.
            before_send()
            attempted = True
            connection.placeOrder(current.contract, order)
            for _ in range(24):
                connection.sleep(.25)
                current = fresh_order(ns, connection, wanted, native)
                if current.orderStatus.status in {"Submitted", "PreSubmitted"} and all(
                        getattr(current.order, field) == value for field, value in changed.items()):
                    return dict(ok=True, state="executed", detail="Exact order modification confirmed",
                                fill=dict(status=current.orderStatus.status,
                                          filled=float(current.orderStatus.filled or 0)))
            raise ValueError("modification not acknowledged; reconcile before retry")
    except Exception as exc:
        return dict(ok=False, state="unknown" if attempted else "rejected",
                    detail=f"{'Broker outcome uncertain; do not resend' if attempted else 'Nothing sent'}: {exc}", fill=None)


def run(ns, ib, payload, host, port, cid, *, modify=False):
    root = Path(ns.get("POSITION_ACTION_STATE_DIR") or Path(ns["__file__"]).parent / "data" / "position_actions")
    record = None
    try:
        wanted = life.payload_identity(payload)
        command_id = str(payload.get("_command_id") or "").strip()
        if not command_id:
            raise ValueError("command id required for delivery receipt")
        path = root / "order_edits" / (hashlib.sha256(command_id.encode()).hexdigest() + ".json")
        with operation_lock(root):
            if path.exists():
                previous = json.loads(path.read_text(encoding="utf-8"))
                if previous["payload"] != payload or previous["modify"] != modify:
                    raise ValueError("command id belongs to a different edit")
                if previous["phase"] == "done":
                    return ns["_out"](**previous["result"])
                record = previous
                try:
                    from . import broker_reconciliation as observation
                except ImportError:
                    import broker_reconciliation as observation
                results = observation.refresh_target(ib, root, wanted[0], wanted[1],
                                                     open_reader=ns.get("_fresh_open_trades"))
                if command_id in results:
                    return ns["_out"](**results[command_id])
                return ns["_out"](ok=False, state="unknown",
                                  detail="This command was already attempted; delivery needs reconciliation", fill=None)
            # A new manual instruction is allowed despite old uncertain edits.
            # Persist the exact command once, immediately before transmission.
            def before_send():
                nonlocal record
                pause_position_recovery(root, wanted, command_id)
                record = dict(version=1, id=command_id, payload=dict(payload), modify=modify,
                              identity=list(wanted), phase="mutating", manual=True)
                save(path, record)
            result = mutate(ns, ib, payload, host, port, cid, modify, before_send)
            if record is not None:
                record.update(result=result, phase="attention" if result["state"] == "unknown" else "done")
                save(path, record)
            return ns["_out"](**result)
    except Exception as exc:
        return ns["_out"](ok=False, state="unknown" if record else "rejected",
                          detail=f"Manual order delivery needs review: {exc}", fill=None)
