"""Verified forward-scan exclusions; historical strategy universes are untouched."""
import copy
import datetime as dt
import json
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent / "config" / "live_scan_exclusions.json"


def exclude_retired_symbols(book, *, asof, registry_path=REGISTRY):
    day = dt.date.fromisoformat(str(asof)[:10])
    payload = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported live scan exclusion registry")
    excluded = {}
    for row in payload["exclusions"]:
        ticker = row["ticker"].strip().upper().replace(".", "-")
        if (row.get("status") != "confirmed_delisted" or not row.get("evidence_url", "").startswith("https://")
                or not row.get("reason") or not ticker):
            raise ValueError("live exclusion requires a confirmed delisting and source")
        if dt.date.fromisoformat(row["effective_from"]) <= day:
            excluded[ticker] = row
    result = copy.deepcopy(book)
    removed = set()
    for strategy in result:
        members = strategy["universe_tickers"]
        removed.update(t for t in members if t.upper().replace(".", "-") in excluded)
        strategy["universe_tickers"] = [t for t in members if t.upper().replace(".", "-") not in excluded]
    return result, sorted(removed)
