"""Build fresh exact-native strategy and structured dead-end snapshots.

The native strategy fingerprint binds the complete configured rule id, traded
universe, direction, entry and exit.  Broad semantic variants are handled by
the separate algorithm-family catalog; this snapshot catches a source that
repackages a rule already present in the configured primary book.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.strategy_discovery.contracts import (
    ContractError,
    load_json,
    parse_timestamp,
    sha256_json,
    validate_catalog,
)
from research.strategy_discovery.pipeline import structural_fingerprint
from research_io import write_json

DEFAULT_OUTPUT = ROOT / "artifacts" / "strategy_discovery" / "catalogs" / "current"
DEFAULT_DEAD_ENDS = ROOT / "data" / "strategy_research" / "dead_ends.json"


def _entry(entry_type: str) -> dict:
    lowered = entry_type.casefold()
    if "signal close" in lowered:
        return {"session_offset": 0, "timing": "CLOSE", "order_type": "MOC", "price_rule": None}
    timing = "INTRADAY" if "persistent" in lowered else "OPEN"
    return {"session_offset": 1, "timing": timing, "order_type": "LIMIT", "price_rule": entry_type}


def native_fingerprint(strategy: dict) -> str:
    execution = strategy["execution"]
    proposal_projection = {
        "direction": strategy["settings"]["trade_direction"].upper(),
        "universe": {
            "asset_class": "LISTED_EQUITY",
            "scope": "configured primary strategy universe",
            "instruments": strategy["universe_tickers"],
        },
        "universe_history": {
            "membership_mode": "FIXED_INSTRUMENTS",
            "includes_delisted": False,
            "models_delisting_returns": False,
            "evidence_reference": "strategy_config.py configured universe",
        },
        "signal": {
            "observation_timing": "CLOSE_FINAL",
            "decision_lead_minutes": None,
            "conditions": [
                {
                    "field": "native_strategy_rule_id",
                    "operator": "==",
                    "value": strategy["id"],
                    "unit": None,
                    "lookback_sessions": None,
                }
            ],
        },
        "entry": _entry(strategy["settings"]["entry_type"]),
        "exit": {
            "time_stop_sessions": execution.get("hold_days"),
            "stop_rule": f"{execution.get('stop_atr')} ATR stop" if execution.get("use_stop_loss") else None,
            "target_rule": f"{execution.get('tgt_atr')} ATR target" if execution.get("use_take_profit") else None,
        },
        "data_requirements": [],
    }
    return structural_fingerprint(proposal_projection)


def _catalog(kind: str, records: list[dict], as_of: str) -> dict:
    return {
        "schema_version": "1.0",
        "snapshot_id": f"{kind.lower()}-{sha256_json(records)[:16]}",
        "catalog_type": kind,
        "generated_at": as_of,
        "as_of": as_of,
        "records_digest": sha256_json(records),
        "records": records,
    }


def build(as_of: str, dead_end_path: Path) -> tuple[dict, dict]:
    parse_timestamp(as_of, "as_of")
    from strategy_config import STRATEGY_BOOK

    strategy_records = sorted(
        (
            {"name": row["name"], "structural_fingerprint": native_fingerprint(row), "active": True}
            for row in STRATEGY_BOOK
        ),
        key=lambda row: row["name"],
    )
    dead_records = []
    if dead_end_path.exists():
        wrapper = load_json(dead_end_path)
        if set(wrapper) != {"schema_version", "records"} or wrapper["schema_version"] != "strategy-dead-ends.v1":
            raise ContractError("structured strategy dead-end registry is invalid")
        if not isinstance(wrapper["records"], list):
            raise ContractError("structured strategy dead-end records must be a list")
        dead_records = sorted(wrapper["records"], key=lambda row: (row.get("decided_at", ""), row.get("name", "")))
    strategy = _catalog("STRATEGY_BOOK", strategy_records, as_of)
    dead = _catalog("DEAD_ENDS", dead_records, as_of)
    validate_catalog(strategy, "STRATEGY_BOOK", "strategy_catalog")
    validate_catalog(dead, "DEAD_ENDS", "dead_end_catalog")
    return strategy, dead


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dead-ends", type=Path, default=DEFAULT_DEAD_ENDS)
    args = parser.parse_args(argv)
    try:
        strategy, dead = build(args.as_of, args.dead_ends)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        strategy_path = args.output_dir / "strategy_book.json"
        dead_path = args.output_dir / "dead_ends.json"
        write_json(strategy_path, strategy)
        write_json(dead_path, dead)
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"STRATEGY CATALOG BUILD FAILED: {exc}", file=sys.stderr)
        return 2
    print(f"strategy catalog: {strategy_path}")
    print(f"dead-end catalog: {dead_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
