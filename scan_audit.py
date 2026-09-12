"""Durable, sanitized scan evidence for the weekly operations review."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import re
import uuid
from zoneinfo import ZoneInfo

MARKER = "SCAN_AUDIT_JSON "


def redact(text):
    text = re.sub(r"https?://[^\s<>]+", "[URL omitted]", str(text))
    return re.sub(r"(?i)(token|password|secret|api[_-]?key)\s*[=:]\s*[^\s,;]+", r"\1=[redacted]", text)


def archive_scan(coverage, signals, *, scope, bookend, email_ok, root=None):
    now = dt.datetime.now(dt.timezone.utc)
    record = {
        "schema_version": 1, "generated_at": now.isoformat(),
        "date_et": now.astimezone(ZoneInfo("America/New_York")).date().isoformat(),
        "scope": scope, "bookend": bookend, "email_accepted": bool(email_ok),
        "coverage": coverage,
        "signals": [{key: row.get(key) for key in (
            "Ticker", "Strategy_Name", "Date", "Scan_Source", "Shares", "Risk_Amt",
            "OLV_Signal_Number", "OLV_Recency_Window", "OLV_Recency_Mult", "OLV_Risk_Budget",
            "OLV_Sizing_Capital", "OLV_Risk_ATR", "Entry_Offset_ATR", "Limit_Price",
            "Pivot_Rule_Version", "Pivot_Matched_Rule", "Sizing_Notes",
        )} for row in signals],
    }
    record = json.loads(json.dumps(record, default=str))
    for item in record["coverage"].get("exceptions", []):
        item["reason"] = redact(item.get("reason", ""))
    body = json.dumps(record, ensure_ascii=True, allow_nan=False, default=str)
    print(MARKER + body)
    directory = Path(root or os.environ.get("NEW_SEASONALS_AUTOMATION_STATE_ROOT")
                     or Path(__file__).resolve().parent / "artifacts" / "automation")
    target = directory / "scan_audits" / record["date_et"] / f"{bookend}-{uuid.uuid4().hex}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body + "\n", encoding="utf-8")
    return target
