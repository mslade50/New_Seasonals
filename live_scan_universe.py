"""Verified forward-scan exclusions; historical strategy universes are untouched."""
import copy
import datetime as dt
import json
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent / "config" / "live_scan_exclusions.json"


def canonical_ticker(ticker):
    """Yahoo symbol spelling, with an explicit repair for the dollar-index alias.

    Share-class dots still become hyphens; the dot in DX-Y.NYB is a provider
    suffix, not a share class. This changes spelling, never the instrument.
    """
    value = str(ticker).strip().upper()
    if value in {"DX-Y.NYB", "DX-Y-NYB"}:
        return "DX-Y.NYB"
    return value.replace(".", "-")


def exclude_retired_tickers(tickers, *, asof, registry_path=REGISTRY):
    day = dt.date.fromisoformat(str(asof)[:10])
    payload = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported live scan exclusion registry")
    excluded = {}
    for row in payload["exclusions"]:
        ticker = canonical_ticker(row["ticker"])
        if (row.get("status") != "confirmed_delisted" or not row.get("evidence_url", "").startswith("https://")
                or not row.get("reason") or not ticker):
            raise ValueError("live exclusion requires a confirmed delisting and source")
        if dt.date.fromisoformat(row["effective_from"]) <= day:
            excluded[ticker] = row
    members = [canonical_ticker(t) for t in tickers]
    removed = sorted(set(members) & set(excluded))
    return [t for t in members if t not in excluded], removed


def exclude_retired_symbols(book, *, asof, registry_path=REGISTRY):
    result = copy.deepcopy(book)
    removed = set()
    for strategy in result:
        members, retired = exclude_retired_tickers(
            strategy["universe_tickers"], asof=asof, registry_path=registry_path)
        removed.update(retired)
        strategy["universe_tickers"] = members
    return result, sorted(removed)
