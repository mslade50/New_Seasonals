"""Build the shared-executor reservation config and integrity manifest.

This does not arm live trading. It requires one structurally validated raw-
mutation wrapper, an exact-source integration-test receipt, and hashes every
active executor source so any later change makes the live gate fail closed.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legend_etf.config import STRATEGY_VERSION
from legend_etf.reservations import (
    CRITICAL_RUNTIME_DISTRIBUTIONS,
    GUARD_MODULE_LABEL,
    GUARD_REQUIRED_MARKER_NAME,
    PORTFOLIO_GUARD_MODULE_LABEL,
    PROTOCOL_VERSION,
    REQUIRED_LEGEND_RUNTIME_FILES,
    discover_broker_mutation_files,
    discover_executor_runtime_files,
    file_sha256,
    source_tree_sha256,
    validate_candidate_parity_evidence,
    validate_central_broker_guard,
    validate_integration_receipt,
)
from legend_etf.storage import atomic_write_json

REVIEW_TOKEN = "I_REVIEWED_SHARED_EXECUTOR_RESERVATIONS"


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reservation-dir", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--legend-root", type=Path, default=ROOT)
    parser.add_argument("--executor-root", type=Path, required=True)
    parser.add_argument("--reservation-config", type=Path, required=True)
    parser.add_argument("--integration-receipt", type=Path, required=True)
    parser.add_argument("--candidate-parity-evidence", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviewed-attestation", required=True)
    return parser


def _absolute(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_absolute():  # pragma: no cover - resolve is absolute
        raise ValueError(f"path is not absolute: {path}")
    return resolved


def main() -> int:
    args = make_parser().parse_args()
    if args.reviewed_attestation != REVIEW_TOKEN:
        raise RuntimeError(
            f"refusing unreviewed executor attestation; pass {REVIEW_TOKEN} only "
            "after both mutation paths have reservation tests"
        )
    reservation_dir = _absolute(args.reservation_dir)
    runtime_dir = _absolute(args.runtime_dir)
    config_path = _absolute(args.reservation_config)
    manifest_path = _absolute(args.manifest)
    integration_receipt = _absolute(args.integration_receipt)
    parity_evidence = _absolute(args.candidate_parity_evidence)
    legend_root = _absolute(args.legend_root)
    executor_root = _absolute(args.executor_root)
    sources = discover_executor_runtime_files(executor_root)
    for label in (GUARD_MODULE_LABEL, PORTFOLIO_GUARD_MODULE_LABEL):
        support_path = executor_root / label
        if not support_path.is_file():
            raise FileNotFoundError(
                f"shared executor support module not found: {support_path}"
            )
        if label not in sources:
            raise RuntimeError(f"shared executor inventory omitted {label}")
    raw_mutators = discover_broker_mutation_files(executor_root)
    if set(raw_mutators) != {GUARD_MODULE_LABEL}:
        raise RuntimeError(
            "all raw IBKR mutations must route through legend_reservation_guard.py"
        )
    validate_central_broker_guard(sources[GUARD_MODULE_LABEL])
    budget_text = sources[PORTFOLIO_GUARD_MODULE_LABEL].read_text(encoding="utf-8")
    if (
        'LEGEND_PORTFOLIO_BUDGET_PROTOCOL = '
        '"legend-equity-index-risk-budget-v3"'
        not in budget_text
        or "def reserve_equity_index_capacity" not in budget_text
    ):
        raise RuntimeError("shared executor portfolio guard protocol is incomplete")
    tree_hash = source_tree_sha256(sources)
    validate_integration_receipt(
        integration_receipt,
        executor_root=executor_root,
        expected_source_tree_sha256=tree_hash,
    )
    candidate_parity = validate_candidate_parity_evidence(
        parity_evidence,
        legend_root=legend_root,
    )
    if candidate_parity.get("protocol") != "legend-etf-native-candidate-parity-v1":
        raise RuntimeError("SPY/QQQ deployment requires ETF-native candidate parity")

    config = {
        "protocol": PROTOCOL_VERSION,
        "reservation_dir": str(reservation_dir),
        "legend_runtime_dir": str(runtime_dir),
        "executor_source_tree_sha256": tree_hash,
        "executor_root": str(executor_root),
    }
    required_marker = executor_root / GUARD_REQUIRED_MARKER_NAME
    # Marker first: a crash between these writes blocks mapped mutations instead
    # of silently returning to the pre-activation raw path.
    atomic_write_json(
        required_marker,
        {
            "protocol": PROTOCOL_VERSION,
            "executor_root": str(executor_root),
        },
    )
    atomic_write_json(config_path, config)
    manifest = {
        "protocol": PROTOCOL_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "reservation_dir": str(reservation_dir),
        "legend_runtime_dir": str(runtime_dir),
        "executor_root": str(executor_root),
        "executor_source_tree_sha256": tree_hash,
        "reservation_config": {
            "path": str(config_path),
            "sha256": file_sha256(config_path),
        },
        "guard_required_marker": {
            "path": str(required_marker),
            "sha256": file_sha256(required_marker),
        },
        "executors": [
            {
                "label": label,
                "path": str(source),
                "sha256": file_sha256(source),
            }
            for label, source in sorted(sources.items())
        ],
        "integration_receipt": {
            "path": str(integration_receipt),
            "sha256": file_sha256(integration_receipt),
        },
        "candidate_parity": {
            "path": str(parity_evidence),
            "sha256": file_sha256(parity_evidence),
            "source_tree_sha256": candidate_parity["candidate_pipeline"][
                "source_tree_sha256"
            ],
        },
        "legend_build": {
            "strategy_version": STRATEGY_VERSION,
            "root": str(legend_root),
            "files": [
                {
                    "label": label,
                    "path": str(legend_root / Path(label)),
                    "sha256": file_sha256(legend_root / Path(label)),
                }
                for label in sorted(REQUIRED_LEGEND_RUNTIME_FILES)
            ],
        },
        "python_runtime": {
            "executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "packages": {
                distribution: importlib.metadata.version(distribution)
                for distribution in sorted(CRITICAL_RUNTIME_DISTRIBUTIONS)
            },
        },
    }
    atomic_write_json(manifest_path, manifest)
    print(f"wrote fail-closed executor guard manifest: {manifest_path}")
    print(f"manifest sha256: {file_sha256(manifest_path)}")
    print("live execution remains disabled until the separate dated gate is armed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
