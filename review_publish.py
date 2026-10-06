"""Delivery-gated human review publication. Never approves or executes trades.

One CAS object per product/date holds current immutable proposal versions and
their decision history. Publisher and Pages write the SAME object conditionally,
so supersession and human decisions serialize without a new storage service.
"""
from __future__ import annotations
import copy
import datetime as dt
import hashlib
import json
from broker_runtime import review_sizing
from pathlib import Path
from zoneinfo import ZoneInfo

SCHEMA = "review-inbox.v1"
PREFIX = "review_inbox/v1"
ET = ZoneInfo("America/New_York")

def key(product: str, date: str) -> str:
    if product not in {"pitch", "seasonal"}:
        raise ValueError("unknown product")
    dt.date.fromisoformat(date)
    return f"{PREFIX}/{product}/{date}.json"

def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

def seal(payload: dict) -> dict:
    text = canonical(payload)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return {"id": f"{payload['product']}:{payload['source_idea_id']}:{digest[:16]}",
            "hash": digest, "canonical": text, "payload": copy.deepcopy(payload)}

def deadlines(asof: str, orders: list[dict], product: str) -> dict:
    import exchange_calendars as xc
    import pandas as pd
    calendar = xc.get_calendar("XNYS")
    manual = product == "seasonal" or any(o.get("Manual_Only") or o.get("Sec_Type") != "STK" or o.get("Proxy_Ticker") for o in orders)
    ends = []
    for row in orders:
        date = row.get("Execute_On")
        if not date or not calendar.is_session(date):
            raise ValueError("Execute_On must be an explicit NYSE session; no date is inferred")
        close = calendar.session_close(date).to_pydatetime()
        opening = calendar.session_open(date).to_pydatetime()
        kind = row.get("Entry_Type")
        if kind == "MOO":
            end = opening - dt.timedelta(minutes=5)  # existing Pitch 09:25 gate
        elif kind == "MOC":
            end = close - dt.timedelta(minutes=30)  # existing 15:30 gate, tightened on early closes
        elif kind == "LIMIT":
            end = close  # review first-session entry only; does not extend stage eligibility to GTD expiry
        else:
            raise ValueError("unknown entry type")
        ends.append(end)
    end = min(ends)
    return {"review_deadline": end.isoformat(),
            "execution_deadline": None if manual else end.isoformat(),
            "manual_only": manual,
            "deadline_basis": "NYSE first-session review cutoff; manual venue execution window unverified" if manual else "NYSE first-session close / existing Pitch auction cutoff (early close aware)"}

def build_record(*, product: str, asof: str, ideas: list[dict], receipt: dict,
                 verdict_digest: str, account_value: float, stand_down: dict | None = None,
                 previous: dict | None = None) -> dict:
    key(product, asof)
    if receipt.get("status") != "sent" or receipt.get("date") != asof:
        raise ValueError("confirmed matching sent delivery required")
    if verdict_digest not in {receipt.get("delivery_digest"), receipt.get("verdict_digest")}:
        raise ValueError("delivery verdict does not match prepared ideas")
    sent_at = receipt.get("sent_at")
    if not sent_at or dt.datetime.fromisoformat(sent_at.replace("Z", "+00:00")).tzinfo is None:
        raise ValueError("delivery timestamp missing timezone")
    if previous and (previous.get("schema") != SCHEMA or previous.get("product") != product or previous.get("date") != asof):
        raise ValueError("invalid previous review ledger")
    record = copy.deepcopy(previous) if previous else {"schema": SCHEMA, "product": product, "date": asof, "proposals": {}, "events": []}
    current = []
    for idea in ideas:
        orders = [{k: v for k, v in row.items() if k != "Approve"} for row in idea["orders"]]
        # Reconstruct only from the receipt-bound delivered idea, never rounded
        # reference quantities or a browser-provided budget. Old/incomplete
        # evidence stays reviewable with an explicit account execution block.
        try:
            source_sizing = review_sizing.instruction(idea, orders, product)
            sizing_hash = hashlib.sha256(canonical(source_sizing).encode()).hexdigest()
            bindings = {account: {"account": account, "status": "requires_fresh_account_preview",
                                 "sizing_hash": sizing_hash} for account in review_sizing.ACCOUNTS}
        except (ValueError, TypeError, KeyError) as exc:
            source_sizing = None
            bindings = {account: {"account": account, "status": "blocked", "reason": str(exc)}
                        for account in review_sizing.ACCOUNTS}
        payload = {"schema": 1, "product": product, "source_date": asof,
                   "source_idea_id": idea["idea_id"], "published_at": sent_at,
                   "title": idea["title"], "thesis": idea["thesis"], "grade": idea.get("grade"),
                   "evidence": idea.get("evidence") or {}, "what_kills_it": idea.get("what_kills_it", ""),
                   "account": "Primary + PA (independent account previews)",
                   "account_basis": "Explicit receipt-bound proposal for each account; live equity, sizing and capacity must be separately verified",
                   "sizing_basis": "Fixed publisher ACCOUNT_VALUE basis; original prepared quantities, not live NLV",
                   "publisher_reference_equity": account_value,
                   "source_sizing": source_sizing, "source_sizing_canonical": canonical(source_sizing) if source_sizing else None, "account_proposals": bindings,
                   "execution_accounts": list(review_sizing.ACCOUNTS),
                   "orders": orders, **deadlines(asof, orders, product)}
        # Cloud reconciliation and the direct publisher must converge on the
        # same version. A later delivery of an unchanged current proposal does
        # not move its original publication timestamp or silently reset review.
        for existing_id in record.get("current_ids", []):
            original = record["proposals"][existing_id]["payload"]
            if ({k:v for k,v in original.items() if k != "published_at"}
                    == {k:v for k,v in payload.items() if k != "published_at"}):
                payload["published_at"] = original["published_at"]
                break
        envelope = seal(payload)
        if envelope["id"] in record["proposals"] and record["proposals"][envelope["id"]] != envelope:
            raise ValueError("immutable proposal collision")
        record["proposals"][envelope["id"]] = envelope
        current.append(envelope["id"])
    # A receipt with separate delivery/verdict digests is a directed amendment
    # to the already-delivered day, not an instruction to erase other ideas.
    if stand_down is None and receipt.get("delivery_digest") != receipt.get("verdict_digest"):
        changed_sources = {idea["idea_id"] for idea in ideas}
        current.extend(old_id for old_id in record.get("current_ids", [])
                       if record["proposals"][old_id]["payload"]["source_idea_id"] not in changed_sources)
    record.update({"current_ids": current, "delivery": {k: receipt[k] for k in ("delivery_id", "sent_at")},
                   "verdict_digest": verdict_digest, "stand_down": stand_down is not None,
                   "stand_down_reason": (stand_down or {}).get("reason", ""), "published_at": sent_at})
    return record

def _publish(*, product: str, asof: str, ideas: list[dict], receipt: dict,
            verdict_digest: str, account_value: float, path: Path, use_r2: bool,
            stand_down: dict | None = None) -> None:
    """No network at import. Test/dev paths never import the production adapter."""
    remote_key = key(product, asof)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not use_r2:
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        record = build_record(product=product, asof=asof, ideas=ideas, receipt=receipt,
                              verdict_digest=verdict_digest, account_value=account_value,
                              stand_down=stand_down, previous=previous)
        from research_io import write_json
        write_json(path, record)
        return
    import cache_io
    if not cache_io.is_configured():
        raise RuntimeError("existing R2 configuration unavailable")
    for _ in range(4):
        before = cache_io.head(remote_key)
        etag = (before or {}).get("ETag")
        previous = None
        if before:
            if not etag or not cache_io.download_to_local(remote_key, str(path)):
                raise RuntimeError("review ledger cannot be read")
            if (cache_io.head(remote_key) or {}).get("ETag") != etag:
                continue
            previous = json.loads(path.read_text(encoding="utf-8"))
        record = build_record(product=product, asof=asof, ideas=ideas, receipt=receipt,
                              verdict_digest=verdict_digest, account_value=account_value,
                              stand_down=stand_down, previous=previous)
        from research_io import write_json
        write_json(path, record)
        result, _ = cache_io.conditional_upload_from_local(str(path), remote_key,
                         create_only=before is None, expected_etag=etag)
        if result == "uploaded":
            return
        if result != "precondition_failed":
            raise RuntimeError("review ledger conditional write failed")
    raise RuntimeError("review ledger changed concurrently; rerun delivery reconciliation")

def publish(**kwargs) -> None:
    # Separate invocations on the same workstation must not race through the
    # local staging file. R2 CAS additionally serializes independent machines.
    from research_io import file_lock
    with file_lock(Path(kwargs["path"])):
        _publish(**kwargs)
