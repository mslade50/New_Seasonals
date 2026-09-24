"""Browser-independent, complete public TradingView premarket discovery.

HTTP capture time is a screen observation, not an exchange quote timestamp.
The provider's premarket bar date must match today's session. These records
remain non-executable and still require fresh daily ATR and full source review.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone

import requests

from .schema import PremarketSnapshot, iso_utc, parse_timestamp
from .tradingview import target_session_date

URL = "https://scanner.tradingview.com/america/scan"
RECORD_TYPE = "EP_PUBLIC_PREMARKET_CAPTURE_V1"
SOURCE = "TRADINGVIEW_PUBLIC_BULK"
COLUMNS = ["name", "description", "exchange", "type", "subtype", "close",
           "premarket_close", "premarket_change", "premarket_change_abs",
           "premarket_volume", "premarket_time"]
MAX_ROWS = 10000


def request_body():
    # Match the saved screen's broad price/volume/exchange scope. No change%
    # predicate: apply the existing 5% nomination rule downstream.
    return {"filter": [
        {"left": "exchange", "operation": "in_range", "right": ["NASDAQ", "NYSE"]},
        {"left": "close", "operation": "egreater", "right": 1},
        {"left": "premarket_volume", "operation": "egreater", "right": 100000}],
        "options": {"lang": "en"}, "markets": ["america"],
        "symbols": {"query": {"types": ["stock", "dr"]}, "tickers": []},
        "columns": COLUMNS, "sort": {"sortBy": "name", "sortOrder": "asc"},
        "range": [0, MAX_ROWS]}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Missing or non-finite public premarket metric")
    return float(value)


def normalize(evidence: dict) -> list[dict]:
    if (evidence.get("record_type") != RECORD_TYPE or evidence.get("provider") != "TRADINGVIEW"
            or evidence.get("url") != URL or evidence.get("request") != request_body()):
        raise ValueError("Public premarket capture provenance mismatch")
    captured = parse_timestamp(evidence["captured_at"])
    started = parse_timestamp(evidence["started_at"])
    target = target_session_date(captured, session="premarket").isoformat()
    if (evidence.get("target_session_date") != target or evidence.get("session") != "premarket"
            or not 0 <= (captured - started).total_seconds() <= 90
            or target_session_date(started, session="premarket").isoformat() != target):
        raise ValueError("Public premarket capture session or request duration invalid")
    response = evidence["response"]
    if not isinstance(response, dict):
        raise ValueError("Public premarket response must be an object")
    rows, total = response.get("data"), response.get("totalCount")
    # Empty upstream feeds are an outage, not proof of zero eligible movers.
    if (not isinstance(rows, list) or type(total) is not int
            or total != len(rows) or not 1 <= total < MAX_ROWS):
        raise ValueError("Public premarket response empty, truncated or count mismatch")
    digest = hashlib.sha256(json.dumps(response, sort_keys=True, allow_nan=False).encode()).hexdigest()
    snapshots, identities, symbols = [], set(), set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("d"), list) or len(row["d"]) != len(COLUMNS):
            raise ValueError("Public premarket row schema mismatch")
        symbol, company, exchange, kind, subtype, regular, last, pct, change, volume, stamp = row["d"]
        identity = row.get("s")
        if (not isinstance(symbol, str) or not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-]{0,14}", symbol)
                or exchange not in {"NASDAQ", "NYSE"} or identity != f"{exchange}:{symbol}"
                or identity in identities or symbol in symbols or not isinstance(company, str) or not company.strip()):
            raise ValueError("Public premarket identity invalid or duplicated")
        identities.add(identity); symbols.add(symbol)
        if kind not in {"stock", "dr"} or not isinstance(subtype, str):
            raise ValueError("Public premarket security type malformed")
        if kind == "stock" and subtype != "common":
            continue
        regular, last, pct, change, volume, stamp = map(_number, (regular, last, pct, change, volume, stamp))
        previous = last - change
        if regular < 1 or last <= 0 or previous <= 0 or volume < 100000 or volume != int(volume):
            raise ValueError("Public premarket price or volume invalid")
        # premarket_time is a session-bar timestamp, not the time of last trade.
        bar = datetime.fromtimestamp(stamp, timezone.utc)
        if bar > captured or target_session_date(bar, session="premarket").isoformat() != target:
            raise ValueError("Public premarket bar is not from today's session")
        if abs(100 * change / previous - pct) > max(0.15, abs(pct) * 0.05):
            raise ValueError("Public premarket price and change are inconsistent")
        snapshot = PremarketSnapshot(
            symbol=symbol, observed_at=iso_utc(captured), previous_close=previous, last=last,
            bid=0, ask=0, premarket_volume=int(volume), premarket_open=last,
            premarket_high=last, premarket_low=last, premarket_vwap=0,
            prior_two_day_low=0, atr_14=0, avg_volume_20=0, addv_63=0,
            company_name=company, market_data_status="PUBLIC_BULK_SNAPSHOT", tradeable=False,
            source=SOURCE, price_basis="TRADINGVIEW_EXTENDED_HOURS_REPORTED",
            primary_exchange=exchange, screen_exchange=exchange, session="premarket",
            provider="TRADINGVIEW", target_session_date=target, security_type=kind,
            reported_result_count=total, extracted_row_count=total, source_file_sha256=digest,
            reported_change_pct=pct, reported_move_dollars=change,
            premarket_move_verification_status="VERIFIED",
            premarket_move_verification_source=SOURCE, premarket_move_verified_at=iso_utc(captured),
        ).to_dict()
        # No executable quote exists; omit the derived infinite spread sentinel
        # so evidence is strict JSON rather than JavaScript's nonstandard Infinity.
        snapshot.pop("spread_bps")
        snapshots.append(snapshot)
    if not snapshots:
        raise ValueError("Public premarket response has no verified equity coverage")
    return snapshots


def validate_capture(evidence: dict) -> None:
    if evidence.get("snapshots") != normalize(evidence):
        raise ValueError("Public premarket snapshots differ from retained response")


def capture(*, now_fn=None, post=None) -> dict:
    now_fn = now_fn or (lambda: datetime.now(timezone.utc))
    started = now_fn()
    target = target_session_date(started, session="premarket").isoformat()
    query = request_body()
    response = (post or requests.post)(URL, json=query, timeout=30)
    response.raise_for_status()
    evidence = {"record_type": RECORD_TYPE, "provider": "TRADINGVIEW", "session": "premarket",
                "target_session_date": target, "started_at": iso_utc(started),
                "captured_at": iso_utc(now_fn()), "url": URL, "request": query,
                "response": response.json()}
    evidence["snapshots"] = normalize(evidence)
    return evidence
