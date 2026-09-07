"""Small local receipts for incomplete source coverage; no external publishing."""
import datetime
import json
import os
from pathlib import Path


def write_status(path, payload):
    payload = {"schema_version": 1, "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), **payload}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return payload


def scan_coverage(requested, loaded, validated, stale, strategies):
    requested = {str(t).upper().replace(".", "-") for t in requested}
    loaded = {str(t).upper().replace(".", "-") for t in loaded}
    valid = {str(t).upper().replace(".", "-") for t in validated}
    missing = sorted(requested - loaded)
    unavailable = sorted(requested - valid)
    by_strategy = []
    for strategy in strategies:
        members = {str(t).upper().replace(".", "-") for t in strategy["universe_tickers"]}
        by_strategy.append({"strategy": strategy.get("id", strategy.get("name")),
                            "requested": len(members), "available": len(members & valid),
                            "unavailable": sorted(members - valid)})
    return {"status": "degraded" if unavailable else "ok", "requested": len(requested),
            "loaded": len(requested & loaded), "available": len(requested & valid),
            "missing": missing, "stale": {str(k): str(v) for k, v in stale.items()},
            "unavailable": unavailable, "strategies": by_strategy,
            "fallback": "Continue current, otherwise executable inputs; unavailable symbols are unknown, not zero signals"}
