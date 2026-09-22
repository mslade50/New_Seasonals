"""Cheap broad-market discovery before bounded, exact daily-history checks."""
from __future__ import annotations

import math
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import requests
import pandas as pd

from .schema import parse_timestamp
from trading_calendar import TRADING_DAY

URL = "https://scanner.tradingview.com/america/scan"
SCHEMA = "EP_SHORT_BULK_DISCOVERY_V1"
COLUMNS = ["name", "description", "exchange", "close", "volume", "average_volume_60d_calc", "SMA50", "time"]
MAX_ROWS = 10000
MIN_ROWS = 100
MAX_HISTORY_TARGETS = 500
NY = ZoneInfo("America/New_York")


def request_for(settings: dict) -> dict:
    # Raw last price and absolute volume are necessary gates, with extra margin
    # for independent providers. Include all types here; listing identity is the
    # equity/ADR gate so the scanner's "stock" type cannot accidentally drop DRs.
    return {"filter": [
        {"left": "close", "operation": "egreater", "right": 0.95 * settings["min_price"]},
        {"left": "volume", "operation": "egreater", "right": 0.9 * settings["min_vol"] * settings["vol_thresh"]}],
        "options": {"lang": "en"}, "markets": ["america"],
        "symbols": {"query": {"types": []}, "tickers": []}, "columns": COLUMNS,
        "sort": {"sortBy": "name", "sortOrder": "asc"}, "range": [0, MAX_ROWS]}


def _positive(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def select_targets(response: dict, listings: dict, settings: dict, signal_date: str) -> tuple[list[str], dict]:
    if not isinstance(response, dict):
        raise ValueError("Malformed bulk-discovery response")
    rows, total = response.get("data"), response.get("totalCount")
    if (not isinstance(rows, list) or not isinstance(total, int) or isinstance(total, bool)
            or total != len(rows) or not MIN_ROWS <= total <= MAX_ROWS):
        raise ValueError("Bulk discovery is empty, incomplete, oversized or unexpectedly small")
    targets, seen, symbols_seen = [], set(), set()
    stats = {"provider_rows": total, "outside_listing_scope": 0, "volume_filtered": 0,
             "below_sma50_filtered": 0, "unknown_metrics_retained": 0,
             "matched_listings": 0, "dated_prior_session": 0, "unverified_dates_retained": 0}
    for item in rows:
        if not isinstance(item, dict):
            raise ValueError("Malformed bulk-discovery row")
        identity, values = item.get("s"), item.get("d")
        if not isinstance(identity, str) or identity in seen or not isinstance(values, list) or len(values) != len(COLUMNS):
            raise ValueError("Malformed or duplicate bulk-discovery identity")
        seen.add(identity)
        symbol, company, exchange, price, volume, avg60, sma50, bar_time = values
        if not isinstance(symbol, str) or not company or identity != f"{exchange}:{symbol}":
            raise ValueError("Bulk-discovery columns do not match security identity")
        listing = listings.get(symbol)
        venues = {"NASDAQ": "NASDAQ", "N": "NYSE", "A": "AMEX", "P": "AMEX", "Z": "CBOE", "V": "IEX"}
        if listing is None or venues.get(listing["exchange"]) != exchange:
            stats["outside_listing_scope"] += 1
            continue
        if symbol in symbols_seen:
            raise ValueError("Duplicate primary listing in bulk discovery")
        symbols_seen.add(symbol)
        stats["matched_listings"] += 1
        try:
            dated = (_positive(bar_time) and
                     datetime.fromtimestamp(bar_time, timezone.utc).astimezone(NY).date().isoformat() == signal_date)
        except (ValueError, OverflowError, OSError):
            dated = False
        if not dated:
            # Do not use stale/unknown metrics to discard a security. A small
            # number of halted/delisted rows can coexist with a fresh feed.
            targets.append(symbol)
            stats["unverified_dates_retained"] += 1
            continue
        stats["dated_prior_session"] += 1
        # Last-60 volume sum is bounded by last-63 volume sum, whether the
        # provider's 60-day window includes the signal bar or ends one bar prior.
        # Unknown metrics remain candidates for the independent history check.
        if _positive(volume) and _positive(avg60) and volume <= 0.9 * settings["vol_thresh"] * (60 / 63) * avg60:
            stats["volume_filtered"] += 1
            continue
        if _positive(price) and _positive(sma50) and price < 0.98 * sma50:
            stats["below_sma50_filtered"] += 1
            continue
        if not all(_positive(v) for v in (price, volume, avg60, sma50)):
            stats["unknown_metrics_retained"] += 1
        targets.append(symbol)
    if not stats["matched_listings"] or stats["dated_prior_session"] / stats["matched_listings"] < 0.9:
        raise ValueError("Bulk discovery is not predominantly dated to the prior NYSE session")
    if len(targets) > MAX_HISTORY_TARGETS:
        raise ValueError(f"Short discovery has {len(targets)} targets, exceeding capacity {MAX_HISTORY_TARGETS}; no truncation or full-market fallback")
    stats.update(shortlisted=len(targets), listed_equities=len(listings))
    return sorted(targets), stats


def validate_discovery(evidence: dict, target: str, listings: dict, settings: dict, *, now: datetime) -> tuple[list[str], dict]:
    if evidence.get("schema") != SCHEMA or evidence.get("session_date") != target or evidence.get("url") != URL:
        raise ValueError("Bulk discovery schema, session or source mismatch")
    captured = parse_timestamp(evidence["captured_at"])
    local = captured.astimezone(NY)
    if captured > now or local.date().isoformat() != target or not time(4) <= local.time() < time(9, 30):
        raise ValueError("Bulk short discovery must be captured in the target premarket")
    if evidence.get("request") != request_for(settings):
        raise ValueError("Bulk discovery query differs from the configured loose prefilter")
    prior = (pd.Timestamp(target) - TRADING_DAY).date().isoformat()
    return select_targets(evidence["response"], listings, settings, prior)


def capture_discovery(target: str, settings: dict) -> dict:
    now = datetime.now(timezone.utc)
    local = now.astimezone(NY)
    if local.date().isoformat() != target or not time(4) <= local.time() < time(9, 30):
        raise ValueError("Short discovery capture is premarket-only")
    query = request_for(settings)
    response = requests.post(URL, json=query, timeout=30)
    response.raise_for_status()
    return {"schema": SCHEMA, "session_date": target, "url": URL, "request": query,
            "captured_at": datetime.now(timezone.utc).isoformat(), "response": response.json()}
