"""Fresh, replayable US-listed equity discovery from Nasdaq Trader directories."""
from __future__ import annotations

import csv
import io
import re
from collections import Counter
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from trading_calendar import TRADING_DAY
from .schema import parse_timestamp

SCHEMA = "NASDAQ_TRADER_EQUITY_UNIVERSE_V1"
URLS = {name: "https://www.nasdaqtrader.com/dynamic/SymDir/" + name
        for name in ("nasdaqlisted.txt", "otherlisted.txt")}
MIN_DIRECTORY_ROWS = 1000
NY = ZoneInfo("America/New_York")
NON_EQUITY = re.compile(r"\b(warrants?|rights?|units?|preferred|preference|notes?|debentures?|bonds?)\b", re.I)


def parse_directory(name: str, content: str, target: str, captured_at: str) -> tuple[list[dict], dict]:
    lines = content.strip().splitlines()
    if len(lines) < 3 or not lines[-1].startswith("File Creation Time:"):
        raise ValueError("Incomplete listing directory: missing timestamp footer")
    stamp = lines[-1].split("|", 1)[0].removeprefix("File Creation Time:").strip()
    created = datetime.strptime(stamp, "%m%d%Y%H:%M").date()
    captured = parse_timestamp(captured_at).astimezone(NY)
    prior = (pd.Timestamp(target) - TRADING_DAY).date()
    if not prior <= created <= captured.date() or captured.date().isoformat() != target:
        raise ValueError("Listing directory is stale or from the wrong session")
    reader = csv.DictReader(io.StringIO("\n".join(lines[:-1])), delimiter="|")
    symbol_key = "Symbol" if name == "nasdaqlisted.txt" else "ACT Symbol"
    required = {symbol_key, "Security Name", "Test Issue", "ETF"}
    if name == "otherlisted.txt":
        required |= {"Exchange", "NASDAQ Symbol"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError("Listing directory schema changed")
    rows = list(reader)
    if len(rows) < MIN_DIRECTORY_ROWS:
        raise ValueError("Listing directory is unexpectedly small")
    eligible, excluded, seen = [], Counter(), set()
    for row in rows:
        if None in row or any(row.get(k) is None for k in required):
            raise ValueError("Malformed listing-directory row")
        symbol, company = row[symbol_key].strip(), row["Security Name"].strip()
        if not symbol or not company or symbol in seen:
            raise ValueError("Empty or duplicate listing identity")
        seen.add(symbol)
        if row["Test Issue"] not in {"Y", "N"} or row["ETF"] not in {"Y", "N"}:
            raise ValueError("Unrecognized listing eligibility flag")
        if row["Test Issue"] == "Y":
            excluded["test_issue"] += 1
        elif row["ETF"] == "Y" or row.get("NextShares") == "Y":
            excluded["exchange_traded_fund"] += 1
        elif NON_EQUITY.search(company):
            excluded["warrant_right_unit_preferred_or_debt"] += 1
        elif not re.fullmatch(r"[A-Z]{1,5}(?:\.[A-Z])?", symbol):
            excluded["unsupported_symbol_format"] += 1
        else:
            eligible.append({"symbol": symbol, "yahoo_symbol": symbol.replace(".", "-"),
                             "company_name": company,
                             "exchange": "NASDAQ" if name == "nasdaqlisted.txt" else row["Exchange"],
                             "directory": name})
    return eligible, {"source_rows": len(rows), "eligible_rows": len(eligible),
                      "excluded": dict(excluded), "file_date": str(created)}


def validate_universe(evidence: dict, target: str, *, now: datetime) -> tuple[dict, dict]:
    if evidence.get("schema") != SCHEMA or evidence.get("session_date") != target:
        raise ValueError("Listing universe schema/session mismatch")
    sources = evidence.get("sources", {})
    if set(sources) != set(URLS):
        raise ValueError("Both listing directories are required; no static-list fallback")
    listings, stats, duplicates, yahoo_seen = {}, {}, 0, {}
    for name, url in URLS.items():
        record = sources[name]
        captured = parse_timestamp(record["captured_at"])
        if record.get("url") != url or captured > now:
            raise ValueError("Listing-directory source or capture time mismatch")
        rows, stats[name] = parse_directory(name, record["content"], target, record["captured_at"])
        for row in rows:
            symbol, yahoo = row["symbol"], row["yahoo_symbol"]
            if symbol in listings:
                if listings[symbol]["company_name"] != row["company_name"]:
                    raise ValueError("Conflicting duplicate listing identity")
                duplicates += 1
                continue
            if yahoo in yahoo_seen:
                raise ValueError("Ambiguous Yahoo symbol mapping")
            yahoo_seen[yahoo] = symbol
            listings[symbol] = row
    if not listings:
        raise ValueError("No eligible listed equities")
    return dict(sorted(listings.items())), {"directories": stats, "duplicates": duplicates,
                                          "listed_equities": len(listings)}


def capture_universe(target: str) -> dict:
    sources = {}
    for name, url in URLS.items():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        sources[name] = {"url": url, "captured_at": datetime.now(timezone.utc).isoformat(),
                         "content": response.text}
    evidence = {"schema": SCHEMA, "session_date": target, "sources": sources}
    validate_universe(evidence, target, now=datetime.now(timezone.utc))
    return evidence
