"""Expected-exit monitor using named local, verified input artifacts.

No broker, R2, task, or order adapter. Email is available only with explicit
--send. Input inventory must come from the reviewed tagged-inventory adapter;
raw broker positions and theoretical targets cannot establish tranche holdings.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ET = ZoneInfo("America/New_York")
UTC = dt.timezone.utc
MAX_SOURCE_AGE_SECONDS = 90
GRACE_MINUTES = 5


def _stamp(value):
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(value / 1000 if value > 1e12 else value, UTC)
    result = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("timestamp lacks timezone")
    return result.astimezone(UTC)


def _fresh(value, now):
    age = (now - _stamp(value)).total_seconds()
    if not 0 <= age <= MAX_SOURCE_AGE_SECONDS:
        raise ValueError("source is stale or future-dated")


def _key(row):
    values = [row.get(key) for key in ("account", "con_id", "strategy", "tranche_id", "ref_date")]
    if not all(value is not None and str(value).strip() for value in values) or int(values[1]) <= 0:
        raise ValueError("tranche lacks exact account/contract/allocation identity")
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()[:24]


def _sources(inventory, book, fills, now):
    if inventory.get("status") != "known" or not isinstance(inventory.get("tranches"), list):
        raise ValueError("reviewed tagged inventory is unavailable")
    _fresh(inventory.get("asof_utc"), now)
    book = book.get("book", book)
    _fresh(book.get("at"), now)
    accounts = [row for row in book.get("accounts", []) if row.get("key") == "primary"]
    if len(accounts) != 1 or accounts[0].get("error"):
        raise ValueError("exact Primary broker snapshot is incomplete")
    account = accounts[0]
    if not account.get("broker_account") or not isinstance(account.get("positions"), list) or not isinstance(account.get("orders"), list):
        raise ValueError("Primary snapshot has no complete account/position/order identity")
    coverage = fills.get("completeness") or {}
    primary = (coverage.get("accounts") or {}).get("primary") or {}
    if coverage.get("truncated") or coverage.get("merge_error") or coverage.get("incomplete_days") or primary.get("complete") is not True or primary.get("error"):
        raise ValueError("Primary execution source is incomplete")
    if primary.get("broker_account") != account["broker_account"]:
        raise ValueError("Primary execution and position accounts differ")
    _fresh(primary.get("received_at"), now)
    _fresh(primary.get("source_at") or primary.get("complete_through"), now)
    if not isinstance(fills.get("fills"), list):
        raise ValueError("execution rows are unavailable")
    return account


def _closing_fills(rows, obligation):
    effective = {}
    for row in rows:
        if row.get("account_key") != "primary" or row.get("account") != obligation["account"]:
            continue
        exec_id = str(row.get("exec_id") or "")
        match = re.fullmatch(r"(.+)\.(\d+)", exec_id)
        family, revision = (match.group(1), int(match.group(2))) if match else (exec_id, 0)
        if not family:
            raise ValueError("execution has no identity")
        if revision >= effective.get(family, (-1, None))[0]:
            effective[family] = (revision, row)
    total = 0.0
    for _, row in effective.values():
        if int(row.get("con_id") or 0) != int(obligation["con_id"]):
            continue
        ref = str(row.get("order_ref") or "").split("|")
        if len(ref) < 4 or ref[2] != obligation["strategy"] or ref[3] != obligation["ref_date"]:
            continue
        if len(ref) > 4 and ref[4] and ref[4] != obligation["tranche_id"]:
            continue
        sign = {"BOT": 1, "BUY": 1, "SLD": -1, "SELL": -1}.get(str(row.get("side") or "").upper())
        if sign is None or sign * obligation["baseline_signed_qty"] >= 0:
            continue
        quantity = float(row.get("qty") or 0)
        if not math.isfinite(quantity) or quantity <= 0:
            raise ValueError("exit execution has invalid quantity")
        total += quantity
    return total


def evaluate(inventory, book, fills, previous=None, *, now=None):
    now = dt.datetime.now(UTC) if now is None else _stamp(now)
    state = copy.deepcopy(previous or {"schema_version": 1, "obligations": {}, "notifications": {}})
    if state.get("schema_version") != 1 or not isinstance(state.get("obligations"), dict) or not isinstance(state.get("notifications"), dict):
        raise ValueError("monitor state is malformed; prior alert deduplication must be preserved")
    source_error, primary = None, None
    try:
        primary = _sources(inventory, book, fills, now)
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        source_error = str(exc)
    current = {}
    for row in inventory.get("tranches", []) if inventory.get("status") == "known" else []:
        try:
            key = _key(row)
            if row.get("account_key") != "primary":
                raise ValueError("tagged inventory contains a non-Primary tranche")
            if key in current:
                raise ValueError("duplicate tagged tranche")
            current[key] = row
            prior = state["obligations"].get(key, {})
            reopened = prior.get("last_status") == "resolved" and float(row["signed_qty"]) != 0
            state["obligations"][key] = {**row, "baseline_signed_qty": prior.get("baseline_signed_qty", row["signed_qty"]),
                                          "first_seen_at": prior.get("first_seen_at", now.isoformat()),
                                          "episode": int(prior.get("episode", 0)) + int(reopened),
                                          "last_status": prior.get("last_status")}
        except (ValueError, TypeError, KeyError) as exc:
            source_error = str(exc)
    results = []
    def event(kind, record, text):
        fingerprint = hashlib.sha256(json.dumps([kind, record.get("id"), record.get("episode"), record.get("deadline"), record.get("remaining_tagged_qty")]).encode()).hexdigest()
        state["notifications"].setdefault(fingerprint, {"status": "pending", "kind": kind,
                                          "created_at": now.isoformat(), "message": text})
    for key, obligation in sorted(state["obligations"].items()):
        row = {"id": key, "tranche_id": obligation["tranche_id"], "strategy": obligation["strategy"],
               "episode": obligation.get("episode", 0),
               "symbol": obligation.get("symbol"), "con_id": obligation["con_id"],
               "deadline": obligation.get("exit_deadline_utc"), "remaining_tagged_qty": None,
               "broker_net_qty": None, "closing_fill_qty": None}
        due, valid_deadline = False, False
        try:
            deadline = _stamp(row["deadline"])
            valid_deadline = True
            due = now >= deadline + dt.timedelta(minutes=GRACE_MINUTES)
            if source_error:
                raise ValueError(source_error)
            if primary["broker_account"] != obligation["account"]:
                raise ValueError("obligation and Primary snapshot accounts differ")
            positions = [p for p in primary["positions"] if int(p.get("con_id") or 0) == int(obligation["con_id"])]
            if any(p.get("account") != obligation["account"] for p in positions) or len(positions) > 1:
                raise ValueError("broker contract position identity is ambiguous")
            row["broker_net_qty"] = float(positions[0]["position"]) if positions else 0.0
            row["remaining_tagged_qty"] = float(current[key]["signed_qty"]) if key in current else 0.0
            if not all(math.isfinite(value) for value in (row["broker_net_qty"], row["remaining_tagged_qty"], float(obligation["baseline_signed_qty"]))):
                row["broker_net_qty"] = row["remaining_tagged_qty"] = None
                raise ValueError("position quantity is not finite")
            row["closing_fill_qty"] = _closing_fills(fills["fills"], obligation)
            if row["remaining_tagged_qty"] == 0:
                if obligation.get("last_status") != "resolved" and row["closing_fill_qty"] < abs(obligation["baseline_signed_qty"]):
                    raise ValueError("flat tagged inventory lacks matched closing execution evidence")
                row.update(status="resolved", detail="Attributed inventory is flat and closing executions are confirmed")
            elif due:
                row.update(status="missed", detail="Tagged inventory remains open more than five minutes after its explicit deadline")
            else:
                row.update(status="pending", detail="Exit deadline or five-minute confirmation grace has not elapsed")
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            row.update(status="unable_to_verify", detail=str(exc))
        previous_status = obligation.get("last_status")
        obligation["last_status"] = row["status"]
        obligation["last_checked_at"] = now.isoformat()
        if row["status"] == "resolved" and previous_status in {"missed", "unable_to_verify"}:
            event("resolved", row, f"{row['symbol']} / {row['strategy']}: exit obligation resolved")
        elif row["status"] in {"missed", "unable_to_verify"} and (due or not valid_deadline):
            event(row["status"], row, f"{row['symbol']} / {row['strategy']}: {row['detail']}")
        results.append(row)
    if not results and source_error:
        row = {"id": "inventory-coverage", "status": "unable_to_verify", "detail": source_error, "remaining_tagged_qty": None}
        results.append(row)
        event("unable_to_verify", row, "Expected exits cannot be verified: " + source_error)
    counts = {status: sum(row["status"] == status for row in results) for status in ("pending", "missed", "unable_to_verify", "resolved")}
    local = now.astimezone(ET)
    from trading_calendar import TRADING_DAY
    import pandas as pd
    summary_due = local.time().replace(tzinfo=None) >= dt.time(16, 10) and TRADING_DAY.is_on_offset(pd.Timestamp(local.date()))
    if summary_due:
        summary_key = f"summary:{local.date()}"
        state["notifications"].setdefault(summary_key, {"status": "pending", "kind": "summary", "created_at": now.isoformat(),
            "message": f"16:10 ET exit summary: {counts['missed']} missed, {counts['unable_to_verify']} unverified, {counts['pending']} pending, {counts['resolved']} resolved"})
    report = {"schema_version": 1, "generated_at": now.isoformat(), "account_key": "primary",
              "status": "attention" if counts["missed"] else "degraded" if counts["unable_to_verify"] else "ok",
              "counts": counts, "obligations": results, "source_error": source_error,
              "notifications": list(state["notifications"].values()), "summary_due": summary_due}
    return report, state


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".pending", delete=False, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, path)


def deliver_pending(state, state_path, sender):
    pending = [row for row in state["notifications"].values() if row["status"] == "pending"]
    if not pending:
        return True
    for row in pending:
        row["status"] = "sending"
    _write_json(state_path, state)  # Durable claim before SMTP; crash is ambiguous.
    html = "<h2>Expected exit status</h2><ul>" + "".join(f"<li>{escape(row['message'])}</li>" for row in pending) + "</ul>"
    try:
        confirmed = bool(sender("Expected exit status", html))
    except Exception as exc:
        for row in pending:
            row.update(status="delivery_unknown", error=type(exc).__name__)
        _write_json(state_path, state)
        return False
    for row in pending:
        # The existing adapter returns False for both pre-send failures and
        # failures after SMTP DATA. That boolean cannot prove no delivery.
        row["status"] = "sent" if confirmed else "delivery_unknown"
    _write_json(state_path, state)
    return confirmed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inventory", "book", "fills", "state", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--asof", help="explicit timestamp for an offline fixture")
    parser.add_argument("--send", action="store_true", help="explicitly send pending alerts with the existing report email adapter")
    args = parser.parse_args(argv)
    from scripts.automation_supervisor import GlobalFileLock, LockUnavailable
    try:
        # Serialize the full read/evaluate/claim/send/write transaction, not
        # merely the final rename, so two invocations cannot send one alert.
        with GlobalFileLock(args.state.with_suffix(args.state.suffix + ".lock")):
            read = lambda path: json.loads(path.read_text(encoding="utf-8"))
            previous = read(args.state) if args.state.exists() else None
            report, state = evaluate(read(args.inventory), read(args.book), read(args.fills), previous, now=args.asof)
            _write_json(args.state, state)
            _write_json(args.output, report)
            if args.send:
                from daily_execution_report import send_email
                delivered = deliver_pending(state, args.state, send_email)
                report["notifications"] = list(state["notifications"].values())
                _write_json(args.output, report)
                if not delivered:
                    return 2
        print(f"Expected-exit monitor: {report['status']}; {report['counts']}")
        return 0
    except (ValueError, TypeError, OSError, KeyError, LockUnavailable) as exc:
        print(f"Expected-exit monitor failed ({type(exc).__name__}); prior state was preserved")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
