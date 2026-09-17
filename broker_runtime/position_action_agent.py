"""Agent-side validation and resumption; never imports the live executor."""
from __future__ import annotations

import asyncio
import json
import math
import time
from pathlib import Path

try:
    from . import position_actions as actions
except ImportError:
    import position_actions as actions

TYPES = {"close_resize", "add_to_position"}


def applies(cmd):
    return cmd.get("account") in {"primary", "pa"} and cmd.get("type") in TYPES


def validate(cmd):
    payload = cmd.get("payload") or {}
    try:
        actions.whole(payload.get("con_id"), "contract id")
        if payload.get("qty") not in (None, ""):
            actions.whole(payload["qty"])
        else:
            fraction = float(payload.get("fraction", 1))
            if not math.isfinite(fraction) or fraction <= 0 or (cmd["type"] == "close_resize" and fraction > 1):
                raise ValueError("invalid percentage")
        if not isinstance(payload.get("readd", False), bool):
            raise ValueError("readd must be boolean")
        if cmd["type"] == "close_resize" and payload.get("action") not in {"BUY", "SELL"}:
            raise ValueError("closing direction required")
        typ = payload.get("order_type", "MKT")
        if typ not in {"MKT", "LMT"} or (cmd["type"] == "add_to_position" and typ != "MKT"):
            raise ValueError("unsupported order type")
        if typ == "LMT":
            price = float(payload.get("limit") or 0)
            if not math.isfinite(price) or price <= 0:
                raise ValueError("positive limit price required")
        if payload.get("outside_rth") and typ != "LMT":
            raise ValueError("outside RTH requires LMT")
        return True, []
    except (ValueError, TypeError, OverflowError) as exc:
        return False, [str(exc)]


def preview(cmd):
    payload = cmd.get("payload") or {}
    amount = f"{payload['qty']} units" if payload.get("qty") else f"{100 * float(payload.get('fraction', 1)):g}%"
    label = "Primary" if cmd["account"] == "primary" else "PA"
    return {"summary": f"{'Add' if cmd['type'] == 'add_to_position' else 'Close'} {amount} of {payload.get('symbol')} on {label}",
            "legs": ["Broker resolves exact inventory and exit groups; quoted size is a request, not a fill.",
                     "Proportional exits use whole-unit rounding; zero rungs are cancelled.",
                     "DAY re-add at original average cost after confirmed closes." if payload.get("readd") else "No re-add."]}


async def loop(ns, ws):
    root = Path(ns["_DIR"]) / "data" / "position_actions"
    delivered = {}
    last_observation = 0
    while True:
        try:
            if time.monotonic() - last_observation >= 30:
                await observe_stopped(ns, root)
                last_observation = time.monotonic()
            for record in actions.records(root):
                command = {"id": record["id"], "type": "close_resize",
                           "account": record["account_key"],
                           "payload": dict(record["payload"], reconcile_only=True)}
                eligible, _ = ns["_live_eligible"](command)
                if not eligible:
                    continue
                if record["phase"] == "done":
                    result = record["result"]
                elif record["phase"] == "pending":
                    result = await ns["_execute_live"](command)
                else:
                    result = {"ok": False, "state": "unknown",
                              "detail": "Position action needs broker reconciliation; do not retry: " + record.get("error", record.get("mutation", ""))}
                signature = json.dumps(result, sort_keys=True)
                if delivered.get(record["id"]) != signature:
                    await ws.send(json.dumps(dict(result, type="result", id=record["id"], at=time.time())))
                    delivered[record["id"]] = signature
            await report_completed_edits(root / "order_edits", ws, delivered)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            ns["log"](f"position-action reconciliation unavailable: {type(exc).__name__}: {exc}")
        await asyncio.sleep(5)


async def observe_stopped(ns, root):
    """One read-only broker check per affected account/contract, including edits."""
    book = (ns.get("_BOOK") or {}).get("book") or {}
    mapping = {a["broker_account"]: a["key"] for a in book.get("accounts", [])
               if a.get("broker_account") and a.get("key") in {"primary", "pa"}}
    targets = {}
    for folder in (root, root / "order_edits"):
        for record in actions.records(folder):
            if record["phase"] == "done" or (folder == root and record["phase"] == "pending" and record.get("wire")):
                continue
            p = record["payload"]
            account = record.get("account_key") or mapping.get(p["_broker_account"])
            if account:
                targets[(account, p["con_id"])] = dict(id=record["id"], type="close_resize", account=account,
                    payload=dict(p, observe_only=True))
    for command in targets.values():
        eligible, _ = ns["_live_eligible"](command)
        if eligible:
            await ns["_execute_live"](command)


async def report_completed_edits(root, ws, delivered):
    """Republish durable terminal edit receipts after a lost result/reconnect.

    This is reporting only. Uncertain edits never call an executor or retry.
    """
    for record in actions.records(root):
        if record["phase"] != "done":
            continue
        result = record["result"]
        if (not isinstance(result, dict) or type(result.get("ok")) is not bool
                or result.get("state") not in {"executed", "rejected"}
                or (result["state"] == "executed") != result["ok"]
                or not isinstance(result.get("detail"), str) or not result["detail"]):
            raise ValueError("invalid terminal order-edit receipt")
        signature = json.dumps(result, sort_keys=True)
        if delivered.get(record["id"]) != signature:
            await ws.send(json.dumps(dict(result, type="result", id=record["id"], at=time.time())))
            delivered[record["id"]] = signature
