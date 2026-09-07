"""Export a source-backed algorithm family catalog, without holdings or network IO."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.strategy_discovery.contracts import ContractError, load_json, sha256_json
from research.strategy_discovery.family_fit import (
    ACTIVE_STATUS,
    MARKETS,
    horizon_bucket,
    publish_family_catalog,
    validate_family_catalog,
)
from scripts.run_strategy_discovery import _local_output_dir

REGISTRY = "research/strategy_discovery/algorithm_family_registry.json"


def source_literal(path: Path, name: str):
    """Read configuration without importing sleeve producers or broker helpers."""
    for node in ast.parse(path.read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            try:
                return ast.literal_eval(node.value)
            except (ValueError, TypeError) as exc:
                raise ContractError(f"{name} must remain an explicit source literal") from exc
    raise ContractError(f"missing source configuration: {name}")


def build_catalog(strategy_book, event_sleeve, trend_universe, registry, source_digests, *, as_of, checked_at=None):
    if registry.get("schema_version") != "algorithm-family-registry.v1":
        raise ContractError("unsupported algorithm family registry")
    records = []

    def row(name, sleeve, direction, instruments, profile, configuration, *, status=ACTIVE_STATUS, cash=False):
        records.append({"name": name, "sleeve": sleeve, "status": status, "direction": direction,
                        "instruments": sorted(set(instruments)), "profile": profile,
                        "configuration_digest": sha256_json(configuration), "cash_gate_supported": cash})

    aliases = {"^GSPC": "SPY", "^NDX": "QQQ"}
    for strategy in strategy_book:
        name = strategy["name"]
        annotation = copy.deepcopy(registry["core"].get(name, {
            "behavior": "UNCLASSIFIED", "markets": sorted(MARKETS), "evidence_refs": ["strategy_config.py: unclassified new strategy"],
        }))
        annotation["horizon"] = horizon_bucket(strategy["execution"].get("hold_days"))
        row(name, "PRIMARY_CORE", strategy["settings"]["trade_direction"].upper(),
            [aliases.get(t, t) for t in strategy["universe_tickers"]], annotation,
            {k: strategy[k] for k in ("settings", "execution", "universe_tickers")})
    for name, cfg in event_sleeve.items():
        annotation = copy.deepcopy(registry["events"].get(name, {
            "behavior": "UNCLASSIFIED", "markets": sorted(MARKETS), "horizon": "VARIABLE",
            "evidence_refs": ["event_sleeve.py: unclassified new Event strategy"],
        }))
        row(name, "EVENT", cfg["side"], [cfg["ticker"]], annotation, cfg)
    row("Monthly Trend", "TREND", "LONG", trend_universe,
        {"behavior": "MOMENTUM", "markets": ["EQUITY_INDEX", "RATES", "COMMODITIES"],
         "horizon": "SWING", "evidence_refs": ["trend_sleeve.py", registry["operating_evidence"]]},
        {"universe": trend_universe, "source_digest": source_digests["trend_sleeve.py"]}, cash=True)
    for ref in registry["references"]:
        row(ref["name"], "REFERENCE", ref["direction"], ref["instruments"],
            {key: copy.deepcopy(ref[key]) for key in ("behavior", "markets", "horizon", "evidence_refs")},
            ref, status=ref["status"])
    records.sort(key=lambda record: record["name"])
    return validate_family_catalog({"schema_version": "algorithm-family-catalog.v1", "as_of": as_of,
        "checked_at": checked_at or registry["checked_at"], "operating_evidence": registry["operating_evidence"],
        "source_digests": source_digests, "records": records, "records_digest": sha256_json(records),
        "runtime_verified_now": False, "positions_used": False})


def catalog_from_source(*, as_of, configured_status_current=False):
    # strategy_config loads native universe configuration only; this command
    # never imports Event/Trend producers or reads their live state files.
    from strategy_config import STRATEGY_BOOK
    registry = load_json(ROOT / REGISTRY)
    source_paths = ["strategy_config.py", "event_sleeve.py", "trend_sleeve.py", REGISTRY, registry["operating_evidence"]]
    source_digests = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in source_paths}
    return build_catalog(STRATEGY_BOOK, source_literal(ROOT / "event_sleeve.py", "EVENT_SLEEVE"),
        source_literal(ROOT / "trend_sleeve.py", "TREND_UNIVERSE"), registry, source_digests, as_of=as_of,
        checked_at=as_of if configured_status_current else None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", required=True, help="UTC decision cutoff; does not refresh operating-status observations.")
    parser.add_argument("--configured-status-current", action="store_true",
                        help="date the configured-algorithm observation at --as-of; does not claim live runtime verification")
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts/strategy_discovery/catalogs"))
    args = parser.parse_args(argv)
    try:
        output = _local_output_dir(args.output_dir, ROOT / "artifacts/strategy_discovery")
        catalog = catalog_from_source(as_of=args.as_of, configured_status_current=args.configured_status_current)
        output.mkdir(parents=True, exist_ok=True)
        path = publish_family_catalog(output, catalog)
    except (ContractError, OSError) as exc:
        print(f"ALGORITHM CATALOG BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(f"algorithm catalog: {path}")
    observation = "configured status observed at cutoff" if args.configured_status_current else "dated status evidence retained"
    print(f"{sum(r['status'] == ACTIVE_STATUS for r in catalog['records'])} active algorithms; {observation}; positions excluded; live runtime not asserted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
