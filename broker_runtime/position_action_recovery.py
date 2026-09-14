"""Explicit, audited abandonment of a close that stopped before submission.

This module has no broker mutation API. Call only after reviewing the failed
operation and assigning remaining exit edits to the operator. It does not retry
the close or claim that partially resized exits have reached their target.
"""
import copy
import hashlib
import json
from pathlib import Path

from . import position_actions as actions


def reviewed_abort(record, account, *, now, reason):
    if (record.get("version") != 1 or record.get("phase") != "attention"
            or record.get("mutation") != "resize exit" or "wire" in record
            or record.get("removed") or "add_context" in record
            or record["payload"].get("readd") or not reason.strip()):
        raise ValueError("Only reviewed, interrupted pre-close resizing can be abandoned")
    payload = record["payload"]
    if (account.get("error") or account.get("broker_account") != payload["_broker_account"]
            or not 0 <= now - account["orders_source_at"] <= 60
            or account.get("fills_complete") is not True
            or not 0 <= now - account["fills_source_at"] / 1000 <= 60):
        raise ValueError("Fresh, complete evidence for the exact account is required")
    rows = [o for o in account["orders"] if o["con_id"] == payload["con_id"]]
    legs = actions.scaled_legs(record["legs"], record["held"] - record["quantity"])
    if not legs or len(rows) != len(legs):
        raise ValueError("The complete original exit set must still be working")
    observations = []
    for leg in legs:
        found = [o for o in rows if [o["account"], o["con_id"], o["client_id"],
                                    o["order_id"], o["perm_id"]] == leg["identity"]]
        if len(found) != 1:
            raise ValueError("Exact exit identity missing or duplicated")
        order = found[0]
        if (order["status"] not in actions.ACKNOWLEDGED
                or order["filled"] != leg["filled_before"]
                or order["qty"] != order["filled"] + order["remaining"]
                or order["remaining"] not in {leg["qty"], leg["scaled_qty"]}
                or order["action"] != record["closing"]):
            raise ValueError("An exit has unexpected fills, quantity, or status")
        if order["parent_id"] and any(
                p["order_id"] == order["parent_id"] and p["client_id"] == order["client_id"]
                for p in account["orders"]):
            raise ValueError("An exit still has an active entry parent")
        for field in ("oca_group", "oca_type", "order_type", "tif", "lmt", "aux",
                      "good_after", "good_till", "outside_rth", "order_ref"):
            if (order.get(field) or "") != (leg.get(field) or ""):
                raise ValueError("An exit's terms changed: " + field)
        observations.append(dict(identity=leg["identity"], remaining=order["remaining"],
                                 intended_remaining=leg["scaled_qty"], status=order["status"]))
    result = copy.deepcopy(record)
    result["phase"] = "done"
    result["result"] = dict(ok=False, state="rejected", detail=
        "Aborted before close submission after broker reconciliation. Partial exit "
        "resizes remain; operator is completing edits manually. No close or re-add was submitted.")
    result["resolution"] = dict(kind="reviewed_preclose_abort", at=now, reason=reason,
                                orders_source_at=account["orders_source_at"], exits=observations)
    return result


def abandon(root, command_id, expected_sha256, account, *, now, reason, backup_dir):
    """Serialize against execution and retain exact original bytes before saving."""
    root, backup_dir = Path(root), Path(backup_dir)
    with actions.operation_lock(root):
        path = actions.record_path(root, command_id)
        original = path.read_bytes()
        if hashlib.sha256(original).hexdigest() != expected_sha256:
            raise ValueError("Journal changed since review")
        record = json.loads(original)
        if record["id"] != command_id:
            raise ValueError("Journal command identity mismatch")
        result = reviewed_abort(record, account, now=now, reason=reason)
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup = backup_dir / (path.stem + "." + expected_sha256 + ".json")
        with backup.open("xb") as stream:
            stream.write(original)
        result["resolution"].update(original_sha256=expected_sha256, backup=str(backup))
        actions.save(root, result)
        return result
