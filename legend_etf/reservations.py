"""Cross-executor account/symbol reservations for the Legend ETF sleeve."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from .storage import (
    LockBusyError,
    atomic_write_json,
    content_hash,
    exclusive_file_lock,
    read_json,
    utc_now_iso,
)

PROTOCOL_VERSION = "legend-account-symbol-lock-v1"
GUARD_MODULE_LABEL = "legend_reservation_guard.py"
PORTFOLIO_GUARD_MODULE_LABEL = "legend_portfolio_budget.py"
INTEGRATION_RECEIPT_PROTOCOL = "legend-shared-executor-integration-v1"
INTEGRATION_REVIEW_TOKEN = "I_REVIEWED_SHARED_EXECUTOR_INTEGRATION_TESTS"
CANDIDATE_PARITY_PROTOCOL = "legend-futures-candidate-parity-v1"
CANDIDATE_PARITY_COUNT = 342
CANDIDATE_PARITY_SESSION_COUNT = 2_660
CANDIDATE_PARITY_RANGE = {"start": "2016-01-01", "end": "2026-08-31"}
CANDIDATE_RATIO_ATOL = 1e-12
CANDIDATE_ATR_ATOL = 1e-10
REQUIRED_INTEGRATION_TESTS = frozenset(
    {
        "two_process_last_capacity_race",
        "crash_before_broker_call",
        "accepted_order_transport_timeout",
        "external_correlated_order_before_legend",
        "external_order_between_legend_roots",
        "capacity_held_until_terminal",
        "cancel_replace_never_frees_capacity",
        "corrupt_or_unknown_instrument_blocks",
        "symbol_quarantine_blocks_foreign_mutation",
        "paper_order_ref_and_oca_time_exit",
    }
)
GUARDED_MUTATION_FUNCTIONS = {
    "placeOrder": "guarded_place_order",
    "cancelOrder": "guarded_cancel_order",
    "reqGlobalCancel": "guarded_global_cancel",
}
MUTATION_PATTERN = re.compile(
    r"\.\s*(?:placeOrder|cancelOrder|reqGlobalCancel)\s*\("
)
CRITICAL_RUNTIME_DISTRIBUTIONS = frozenset(
    {
        "databento",
        "ib-insync",
        "numpy",
        "pandas",
        "keyring",
        "exchange-calendars",
        "python-dotenv",
    }
)
REQUIRED_LEGEND_RUNTIME_FILES = frozenset(
    {
        "legend_etf/__init__.py",
        "legend_etf/calendar.py",
        "legend_etf/config.py",
        "legend_etf/core.py",
        "legend_etf/databento_source.py",
        "legend_etf/ibkr_adapter.py",
        "legend_etf/paths.py",
        "legend_etf/paper_proof.py",
        "legend_etf/portfolio_guard.py",
        "legend_etf/reservations.py",
        "legend_etf/session.py",
        "legend_etf/sizing.py",
        "legend_etf/storage.py",
        "scripts/check_legend_etf_calendar.py",
        "scripts/prepare_legend_etf_signals.py",
        "scripts/run_legend_etf_session.py",
        "scripts/run_legend_etf_task.ps1",
        "requirements-legend-etf.txt",
    }
)
CANDIDATE_PIPELINE_FILES = frozenset(
    {
        "legend_etf/__init__.py",
        "legend_etf/calendar.py",
        "legend_etf/config.py",
        "legend_etf/core.py",
        "legend_etf/databento_source.py",
        "legend_etf/paths.py",
        "legend_etf/reservations.py",
        "legend_etf/storage.py",
        "scripts/prepare_legend_etf_signals.py",
        "scripts/verify_legend_futures_candidate_parity.py",
        "requirements-legend-etf.txt",
    }
)
CANDIDATE_RUNTIME_DISTRIBUTIONS = frozenset(
    {
        "databento",
        "exchange-calendars",
        "numpy",
        "pandas",
        "pyarrow",
        "python-dotenv",
    }
)


def _safe_component(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    if not clean:
        raise ValueError("empty reservation component")
    return clean


def reservation_path(directory: Path, account: str, symbol: str) -> Path:
    return Path(directory) / (
        f"account_{_safe_component(account)}__symbol_{_safe_component(symbol.upper())}.lock"
    )


def quarantine_path(directory: Path, account: str, symbol: str) -> Path:
    return Path(directory) / (
        f"account_{_safe_component(account)}__symbol_"
        f"{_safe_component(symbol.upper())}.quarantine.json"
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized(path: Path) -> str:
    return str(Path(path).expanduser().resolve()).casefold()


def discover_broker_mutation_files(root: Path) -> dict[str, Path]:
    """Discover every active Python source that directly mutates IBKR state."""

    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise RuntimeError(f"shared executor root is missing: {base}")
    discovered: dict[str, Path] = {}
    for source in sorted(base.rglob("*.py")):
        relative = source.relative_to(base)
        lowered_parts = [part.lower() for part in relative.parts]
        if (
            any(part.startswith("_backup") for part in lowered_parts)
            or any(part == "__pycache__" for part in lowered_parts)
            or source.name.lower().startswith("test_")
        ):
            continue
        try:
            text = source.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise RuntimeError(f"could not inspect broker source: {source}") from exc
        if MUTATION_PATTERN.search(text):
            discovered[relative.as_posix()] = source.resolve()
    return discovered


def discover_executor_python_files(root: Path) -> dict[str, Path]:
    """Return every active Python source under the exact executor root."""

    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise RuntimeError(f"shared executor root is missing: {base}")
    discovered: dict[str, Path] = {}
    for source in sorted(base.rglob("*.py")):
        relative = source.relative_to(base)
        lowered_parts = [part.lower() for part in relative.parts]
        if (
            any(part.startswith("_backup") for part in lowered_parts)
            or any(part == "__pycache__" for part in lowered_parts)
            or source.name.lower().startswith("test_")
        ):
            continue
        resolved = source.resolve()
        try:
            resolved.relative_to(base)
        except ValueError as exc:
            raise RuntimeError(f"executor source escapes configured root: {source}") from exc
        discovered[relative.as_posix()] = resolved
    return discovered


def source_tree_sha256(sources: dict[str, Path]) -> str:
    rows = [
        {"label": label, "sha256": file_sha256(source)}
        for label, source in sorted(sources.items())
    ]
    encoded = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def candidate_pipeline_sources(root: Path) -> dict[str, Path]:
    base = Path(root).expanduser().resolve()
    sources = {
        label: (base / Path(label)).resolve()
        for label in CANDIDATE_PIPELINE_FILES
    }
    for label, source in sources.items():
        try:
            source.relative_to(base)
        except ValueError as exc:  # pragma: no cover - constant labels are bounded
            raise RuntimeError(f"candidate source escapes Legend root: {label}") from exc
        if not source.is_file():
            raise FileNotFoundError(f"candidate pipeline source is missing: {source}")
    return sources


def candidate_pipeline_attestation(root: Path) -> dict[str, Any]:
    sources = candidate_pipeline_sources(root)
    return {
        "source_tree_sha256": source_tree_sha256(sources),
        "files": [
            {
                "label": label,
                "path": str(source),
                "sha256": file_sha256(source),
            }
            for label, source in sorted(sources.items())
        ],
        "python_runtime": {
            "executable": str(Path(sys.executable).resolve()),
            "python_version": platform.python_version(),
            "packages": {
                distribution: importlib.metadata.version(distribution)
                for distribution in sorted(CANDIDATE_RUNTIME_DISTRIBUTIONS)
            },
        },
    }


def validate_central_broker_guard(path: Path) -> None:
    """Prove all literal raw IBKR mutations sit in reviewed wrapper functions."""

    source = Path(path)
    try:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        raise RuntimeError(f"central broker guard is unreadable: {source}") from exc
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    required = set(GUARDED_MUTATION_FUNCTIONS.values())
    if not required.issubset(functions):
        raise RuntimeError("central broker guard lacks required wrapper functions")
    seen: set[str] = set()

    class MutationVisitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.function_stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.function_stack.append(node.name)
            self.generic_visit(node)
            self.function_stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            if isinstance(node.func, ast.Attribute):
                raw_name = node.func.attr
                expected_function = GUARDED_MUTATION_FUNCTIONS.get(raw_name)
                if expected_function is not None:
                    current = self.function_stack[-1] if self.function_stack else ""
                    if current != expected_function:
                        raise RuntimeError(
                            f"raw {raw_name} must be inside {expected_function}"
                        )
                    seen.add(raw_name)
            self.generic_visit(node)

    MutationVisitor().visit(tree)
    if seen != set(GUARDED_MUTATION_FUNCTIONS):
        raise RuntimeError("central broker guard does not exercise every raw mutation")


def _read_json_once(path: Path) -> tuple[dict[str, Any], str]:
    source = Path(path)
    try:
        raw = source.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"deployment artifact is missing/invalid: {source}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(  # noqa: TRY004 - invalid deployment artifact
            f"deployment artifact is not an object: {source}"
        )
    return payload, hashlib.sha256(raw).hexdigest()


def validate_integration_receipt(
    path: Path,
    *,
    executor_root: Path,
    expected_source_tree_sha256: str,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    receipt, digest = _read_json_once(path)
    if expected_sha256 is not None:
        expected = str(expected_sha256).strip().lower()
        if len(expected) != 64 or digest != expected:
            raise RuntimeError("shared executor integration receipt changed")
    if set(receipt) != {
        "protocol",
        "created_at",
        "executor_root",
        "source_tree_sha256",
        "passed_tests",
        "reviewed_attestation",
    }:
        raise RuntimeError("shared executor integration receipt schema is invalid")
    if receipt["protocol"] != INTEGRATION_RECEIPT_PROTOCOL:
        raise RuntimeError("shared executor integration receipt protocol mismatch")
    created = pd.Timestamp(receipt["created_at"])
    if created.tz is None:
        raise RuntimeError("shared executor integration receipt must be timestamped")
    if _normalized(Path(receipt["executor_root"])) != _normalized(executor_root):
        raise RuntimeError("integration receipt executor root mismatch")
    if receipt["source_tree_sha256"] != expected_source_tree_sha256:
        raise RuntimeError("integration receipt is not for this executor source tree")
    tests = receipt["passed_tests"]
    if not isinstance(tests, list) or set(tests) != REQUIRED_INTEGRATION_TESTS:
        raise RuntimeError("shared executor integration test set is incomplete")
    if receipt["reviewed_attestation"] != INTEGRATION_REVIEW_TOKEN:
        raise RuntimeError("shared executor integration tests lack review attestation")
    return receipt


def _valid_sha256(value: object) -> bool:
    return bool(re.fullmatch(r"[0-9a-f]{64}", str(value).strip().lower()))


def _validate_parity_input_file(
    record: object, *, label: str, verify_content: bool
) -> Path:
    if not isinstance(record, dict) or set(record) != {
        "path",
        "size",
        "mtime_ns",
        "sha256",
    }:
        raise RuntimeError(f"Legend parity {label} manifest schema is invalid")
    source = Path(str(record["path"] or ""))
    if not source.is_absolute() or not source.is_file():
        raise RuntimeError(f"Legend parity {label} input is missing")
    stat = source.stat()
    if (
        type(record["size"]) is not int
        or type(record["mtime_ns"]) is not int
        or record["size"] != stat.st_size
        or record["mtime_ns"] != stat.st_mtime_ns
        or not _valid_sha256(record["sha256"])
    ):
        raise RuntimeError(f"Legend parity {label} input metadata changed")
    if verify_content and file_sha256(source) != str(record["sha256"]).lower():
        raise RuntimeError(f"Legend parity {label} input content changed")
    return source.resolve()


def validate_candidate_parity_evidence(
    path: Path,
    *,
    legend_root: Path,
    expected_sha256: str | None = None,
    verify_input_files: bool = True,
) -> dict[str, Any]:
    evidence, digest = _read_json_once(path)
    if expected_sha256 is not None:
        expected = str(expected_sha256).strip().lower()
        if len(expected) != 64 or digest != expected:
            raise RuntimeError("Legend candidate-parity evidence changed")
    if set(evidence) != {
        "protocol",
        "status",
        "completed_at",
        "command",
        "inputs",
        "range",
        "candidate_pipeline",
        "full_session_count",
        "counts",
        "duplicate_key_rows",
        "matches",
        "max_deltas",
        "tolerances",
        "runtime_seconds",
    }:
        raise RuntimeError("Legend candidate-parity evidence schema is invalid")
    if evidence.get("protocol") != CANDIDATE_PARITY_PROTOCOL:
        raise RuntimeError("Legend candidate-parity evidence protocol mismatch")
    if evidence.get("status") != "pass":
        raise RuntimeError("Legend candidate-parity evidence is not passing")
    if evidence.get("range") != CANDIDATE_PARITY_RANGE:
        raise RuntimeError("Legend candidate-parity range mismatch")
    completed_at = pd.Timestamp(evidence.get("completed_at"))
    if completed_at.tz is None:
        raise RuntimeError("Legend candidate-parity evidence has no completion time")
    command = evidence.get("command")
    if not isinstance(command, str) or "verify_legend_futures_candidate_parity.py" not in command:
        raise RuntimeError("Legend candidate-parity command is invalid")
    if evidence.get("full_session_count") != CANDIDATE_PARITY_SESSION_COUNT:
        raise RuntimeError("Legend candidate-parity session count mismatch")
    runtime_seconds = evidence.get("runtime_seconds")
    if (
        isinstance(runtime_seconds, bool)
        or not isinstance(runtime_seconds, (int, float))
        or not math.isfinite(float(runtime_seconds))
        or float(runtime_seconds) <= 0
    ):
        raise RuntimeError("Legend candidate-parity runtime is invalid")
    counts = evidence.get("counts")
    if (
        not isinstance(counts, dict)
        or set(counts) != {"golden", "research", "production"}
        or any(type(value) is not int for value in counts.values())
        or any(value != CANDIDATE_PARITY_COUNT for value in counts.values())
    ):
        raise RuntimeError("Legend candidate-parity counts are incomplete")
    duplicates = evidence.get("duplicate_key_rows")
    if (
        not isinstance(duplicates, dict)
        or set(duplicates) != set(counts)
        or any(type(value) is not int or value != 0 for value in duplicates.values())
    ):
        raise RuntimeError("Legend candidate-parity evidence contains duplicates")
    matches = evidence.get("matches")
    expected_matches = {
        "research_direction",
        "research_contract",
        "production_direction",
        "production_contract",
    }
    if (
        not isinstance(matches, dict)
        or set(matches) != expected_matches
        or any(type(value) is not int for value in matches.values())
        or any(value != CANDIDATE_PARITY_COUNT for value in matches.values())
    ):
        raise RuntimeError("Legend candidate-parity attributes are incomplete")
    tolerances = evidence.get("tolerances")
    deltas = evidence.get("max_deltas")
    if (
        not isinstance(tolerances, dict)
        or set(tolerances) != {"ratio_atol", "atr_atol"}
        or float(tolerances["ratio_atol"]) != CANDIDATE_RATIO_ATOL
        or float(tolerances["atr_atol"]) != CANDIDATE_ATR_ATOL
        or not isinstance(deltas, dict)
        or set(deltas) != {"research_ratio", "production_ratio", "research_atr14"}
    ):
        raise RuntimeError("Legend candidate-parity tolerance schema is invalid")
    numeric_deltas = {label: float(value) for label, value in deltas.items()}
    if (
        any(not math.isfinite(value) or value < 0 for value in numeric_deltas.values())
        or numeric_deltas["research_ratio"] > CANDIDATE_RATIO_ATOL
        or numeric_deltas["production_ratio"] > CANDIDATE_RATIO_ATOL
        or numeric_deltas["research_atr14"] > CANDIDATE_ATR_ATOL
    ):
        raise RuntimeError("Legend candidate-parity deltas exceed tolerance")

    inputs = evidence.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != {
        "historical_engine",
        "golden",
        "archive",
    }:
        raise RuntimeError("Legend candidate-parity input manifest is invalid")
    _validate_parity_input_file(
        inputs["historical_engine"],
        label="historical engine",
        verify_content=verify_input_files,
    )
    _validate_parity_input_file(
        inputs["golden"], label="golden", verify_content=verify_input_files
    )
    archive = inputs["archive"]
    if not isinstance(archive, dict) or set(archive) != {
        "path",
        "metadata_manifest_sha256",
        "files",
    }:
        raise RuntimeError("Legend candidate-parity archive manifest is invalid")
    archive_root = Path(str(archive["path"] or ""))
    records = archive["files"]
    if (
        not archive_root.is_absolute()
        or not archive_root.is_dir()
        or not isinstance(records, list)
        or not records
        or not _valid_sha256(archive["metadata_manifest_sha256"])
        or content_hash(records) != archive["metadata_manifest_sha256"]
    ):
        raise RuntimeError("Legend candidate-parity archive evidence is invalid")
    declared_archive_paths = []
    for index, record in enumerate(records):
        declared_archive_paths.append(
            _validate_parity_input_file(
                record,
                label=f"archive file {index}",
                verify_content=verify_input_files,
            )
        )
    actual_archive_paths = [source.resolve() for source in sorted(archive_root.glob("*.parquet"))]
    if [_normalized(path) for path in declared_archive_paths] != [
        _normalized(path) for path in actual_archive_paths
    ]:
        raise RuntimeError("Legend candidate-parity archive file set changed")
    pipeline = evidence.get("candidate_pipeline")
    if pipeline != candidate_pipeline_attestation(legend_root):
        raise RuntimeError(
            "Legend candidate-parity evidence is stale for this source/runtime tree"
        )
    return evidence


def validate_guard_manifest(
    path: Path,
    *,
    expected_reservation_dir: Path | None = None,
    expected_runtime_dir: Path | None = None,
    expected_legend_root: Path | None = None,
    expected_executor_root: Path | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate that every discovered external IBKR mutator is attested."""

    manifest, manifest_digest = _read_json_once(Path(path))
    if expected_sha256 is not None:
        expected = str(expected_sha256).strip().lower()
        if len(expected) != 64 or manifest_digest != expected:
            raise RuntimeError("Legend deployment manifest hash differs from runtime.env")
    if manifest.get("protocol") != PROTOCOL_VERSION:
        raise RuntimeError("shared executor guard protocol mismatch")
    declared_directory = Path(str(manifest.get("reservation_dir") or ""))
    if not str(declared_directory).strip():
        raise RuntimeError("shared executor manifest has no reservation directory")
    if (
        expected_reservation_dir is not None
        and _normalized(declared_directory) != _normalized(expected_reservation_dir)
    ):
        raise RuntimeError(
            "runtime reservation directory differs from the guarded executors"
        )
    declared_runtime = str(manifest.get("legend_runtime_dir") or "").strip()
    if not declared_runtime:
        raise RuntimeError("shared executor manifest has no Legend runtime directory")
    if (
        expected_runtime_dir is not None
        and _normalized(Path(declared_runtime)) != _normalized(expected_runtime_dir)
    ):
        raise RuntimeError("Legend runtime directory differs from deployment manifest")
    executor_root = Path(str(manifest.get("executor_root") or ""))
    if not str(executor_root).strip() or not executor_root.is_dir():
        raise RuntimeError("shared executor manifest has no valid executor root")
    if (
        expected_executor_root is not None
        and _normalized(executor_root) != _normalized(expected_executor_root)
    ):
        raise RuntimeError("shared executor root differs from runtime configuration")
    config = manifest.get("reservation_config")
    if not isinstance(config, dict):
        raise RuntimeError(  # noqa: TRY004 - invalid deployment artifact
            "shared executor manifest has no hashed reservation config"
        )
    config_path = Path(str(config.get("path") or ""))
    config_hash = str(config.get("sha256") or "").strip().lower()
    if not config_path.is_file() or len(config_hash) != 64:
        raise RuntimeError("shared executor reservation config is missing or changed")
    config_payload, actual_config_hash = _read_json_once(config_path)
    if actual_config_hash != config_hash:
        raise RuntimeError("shared executor reservation config is missing or changed")
    config_directory = str(config_payload.get("reservation_dir") or "").strip()
    config_runtime = str(config_payload.get("legend_runtime_dir") or "").strip()
    config_executor_root = str(config_payload.get("executor_root") or "").strip()
    config_tree_hash = str(
        config_payload.get("executor_source_tree_sha256") or ""
    ).strip()
    if (
        config_payload.get("protocol") != PROTOCOL_VERSION
        or not config_directory
        or not config_runtime
        or not config_executor_root
        or len(config_tree_hash) != 64
        or _normalized(Path(config_directory)) != _normalized(declared_directory)
        or _normalized(Path(config_runtime)) != _normalized(Path(declared_runtime))
        or _normalized(Path(config_executor_root)) != _normalized(executor_root)
    ):
        raise RuntimeError("shared executor reservation config does not match manifest")
    sources = discover_executor_python_files(executor_root)
    support_labels = (GUARD_MODULE_LABEL, PORTFOLIO_GUARD_MODULE_LABEL)
    support_sources = {
        label: (executor_root / label).resolve() for label in support_labels
    }
    missing_support = [
        label for label, source in support_sources.items() if not source.is_file()
    ]
    if missing_support:
        raise RuntimeError(
            "shared executor support module(s) missing: "
            + ", ".join(missing_support)
        )
    if not set(support_sources).issubset(sources):
        raise RuntimeError("shared executor support modules are outside source inventory")
    raw_mutators = discover_broker_mutation_files(executor_root)
    if set(raw_mutators) != {GUARD_MODULE_LABEL}:
        raise RuntimeError(
            "all raw IBKR mutations must route through legend_reservation_guard.py"
        )
    validate_central_broker_guard(support_sources[GUARD_MODULE_LABEL])
    budget_text = support_sources[PORTFOLIO_GUARD_MODULE_LABEL].read_text(
        encoding="utf-8"
    )
    if (
        'LEGEND_PORTFOLIO_BUDGET_PROTOCOL = '
        '"legend-equity-index-risk-budget-v2"'
        not in budget_text
        or "def reserve_equity_index_capacity" not in budget_text
    ):
        raise RuntimeError("shared executor portfolio guard protocol is incomplete")
    required_sources = sources
    entries = manifest.get("executors")
    if not isinstance(entries, list):
        raise RuntimeError(  # noqa: TRY004 - invalid deployment artifact
            "shared executor guard manifest has no executors"
        )
    seen: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError(  # noqa: TRY004 - invalid deployment artifact
                "invalid shared executor manifest entry"
            )
        label = str(entry.get("label") or "").replace("\\", "/").strip()
        source = Path(str(entry.get("path") or ""))
        expected = str(entry.get("sha256") or "").strip().lower()
        if label not in required_sources or label in seen:
            raise RuntimeError("shared executor guard labels are invalid or duplicated")
        if _normalized(source) != _normalized(required_sources[label]):
            raise RuntimeError(f"guarded source path does not match executor root: {label}")
        if not source.is_file() or len(expected) != 64:
            raise RuntimeError(f"guarded {label} executor is missing or unhashed")
        if file_sha256(source) != expected:
            raise RuntimeError(
                f"guarded {label} executor changed after reservation validation"
            )
        seen.add(label)
    if seen != set(required_sources):
        missing = sorted(set(required_sources).difference(seen))
        raise RuntimeError(
            "every active IBKR mutation path must honor reservations; missing "
            + ", ".join(missing)
        )
    declared_tree_hash = str(manifest.get("executor_source_tree_sha256") or "")
    actual_tree_hash = source_tree_sha256(required_sources)
    if declared_tree_hash != actual_tree_hash or config_tree_hash != actual_tree_hash:
        raise RuntimeError("shared executor source-tree hash mismatch")
    receipt = manifest.get("integration_receipt")
    if not isinstance(receipt, dict) or set(receipt) != {"path", "sha256"}:
        raise RuntimeError("shared executor manifest has no integration receipt")
    receipt_path = Path(str(receipt["path"] or ""))
    expected_receipt_hash = str(receipt["sha256"] or "").lower()
    validate_integration_receipt(
        receipt_path,
        executor_root=executor_root,
        expected_source_tree_sha256=actual_tree_hash,
        expected_sha256=expected_receipt_hash,
    )
    build = manifest.get("legend_build")
    if not isinstance(build, dict):
        raise RuntimeError(  # noqa: TRY004 - invalid deployment artifact
            "shared executor manifest has no Legend build attestation"
        )
    if str(build.get("strategy_version") or "") != "legend-etf-original-v1":
        raise RuntimeError("Legend build strategy version mismatch")
    declared_legend_root = Path(str(build.get("root") or ""))
    if not str(declared_legend_root).strip() or not declared_legend_root.is_dir():
        raise RuntimeError("Legend build manifest has no valid source root")
    if (
        expected_legend_root is not None
        and _normalized(declared_legend_root) != _normalized(expected_legend_root)
    ):
        raise RuntimeError("executing Legend source root differs from deployment manifest")
    parity = manifest.get("candidate_parity")
    if not isinstance(parity, dict) or set(parity) != {
        "path",
        "sha256",
        "source_tree_sha256",
    }:
        raise RuntimeError("Legend deployment manifest has no candidate-parity proof")
    parity_path = Path(str(parity.get("path") or ""))
    parity_hash = str(parity.get("sha256") or "").strip().lower()
    parity_evidence = validate_candidate_parity_evidence(
        parity_path,
        legend_root=declared_legend_root,
        expected_sha256=parity_hash,
        verify_input_files=False,
    )
    if parity.get("source_tree_sha256") != parity_evidence["candidate_pipeline"][
        "source_tree_sha256"
    ]:
        raise RuntimeError("Legend candidate-parity source-tree hash mismatch")
    files = build.get("files")
    if not isinstance(files, list):
        raise RuntimeError(  # noqa: TRY004 - invalid deployment artifact
            "Legend build attestation has no file list"
        )
    build_labels: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise RuntimeError(  # noqa: TRY004 - invalid deployment artifact
                "invalid Legend build manifest entry"
            )
        label = str(entry.get("label") or "").replace("\\", "/")
        source = Path(str(entry.get("path") or ""))
        expected = str(entry.get("sha256") or "").strip().lower()
        if label not in REQUIRED_LEGEND_RUNTIME_FILES or label in build_labels:
            raise RuntimeError("Legend build labels are invalid or duplicated")
        expected_source = declared_legend_root / Path(label)
        if _normalized(source) != _normalized(expected_source):
            raise RuntimeError(
                f"Legend runtime source path differs from attested root: {label}"
            )
        if not source.is_file() or len(expected) != 64:
            raise RuntimeError(f"Legend runtime file is missing or unhashed: {label}")
        if file_sha256(source) != expected:
            raise RuntimeError(f"Legend runtime file changed after review: {label}")
        build_labels.add(label)
    if build_labels != REQUIRED_LEGEND_RUNTIME_FILES:
        raise RuntimeError("Legend build manifest does not cover every runtime file")
    runtime = manifest.get("python_runtime")
    if not isinstance(runtime, dict):
        raise RuntimeError(  # noqa: TRY004 - invalid deployment artifact
            "shared executor manifest has no Python runtime attestation"
        )
    executable = Path(str(runtime.get("executable") or ""))
    if _normalized(executable) != _normalized(Path(sys.executable)):
        raise RuntimeError("Legend Python executable differs from deployment manifest")
    if str(runtime.get("python_version") or "") != platform.python_version():
        raise RuntimeError("Legend Python version differs from deployment manifest")
    packages = runtime.get("packages")
    if not isinstance(packages, dict) or set(packages) != set(
        CRITICAL_RUNTIME_DISTRIBUTIONS
    ):
        raise RuntimeError("Legend critical package attestation is incomplete")
    for distribution, expected in packages.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Legend critical package is missing: {distribution}"
            ) from exc
        if actual != str(expected):
            raise RuntimeError(
                f"Legend critical package changed: {distribution} {actual} != {expected}"
            )
    return manifest


class ReservationBook:
    """Hold non-blocking locks until the complete live session is closed."""

    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self._held: dict[tuple[str, str], Any] = {}

    def try_acquire(
        self,
        account: str,
        symbol: str,
        *,
        owner_token: str | None = None,
    ) -> tuple[bool, str]:
        key = (str(account), str(symbol).upper())
        if key in self._held:
            return True, "already held by this Legend process"
        manager = exclusive_file_lock(
            reservation_path(self.directory, key[0], key[1])
        )
        try:
            manager.__enter__()
        except LockBusyError:
            return False, "account-symbol reservation is held by another executor"
        quarantine = read_json(quarantine_path(self.directory, key[0], key[1]))
        if (
            isinstance(quarantine, dict)
            and quarantine.get("active") is True
            and (
                not owner_token
                or quarantine.get("owner_token") != owner_token
            )
        ):
            manager.__exit__(None, None, None)
            return False, (
                "account-symbol has a durable unresolved-position quarantine"
            )
        self._held[key] = manager
        return True, "acquired"

    def holds(self, account: str, symbol: str) -> bool:
        return (str(account), str(symbol).upper()) in self._held

    def activate_quarantine(
        self,
        account: str,
        symbol: str,
        *,
        owner_token: str,
        payload: dict[str, Any],
    ) -> None:
        key = (str(account), str(symbol).upper())
        if key not in self._held:
            raise RuntimeError("cannot quarantine an unreserved account-symbol")
        path = quarantine_path(self.directory, key[0], key[1])
        current = read_json(path)
        if (
            isinstance(current, dict)
            and current.get("active") is True
            and current.get("owner_token") != owner_token
        ):
            raise RuntimeError("account-symbol is quarantined by another owner")
        atomic_write_json(
            path,
            {
                "protocol": PROTOCOL_VERSION,
                "active": True,
                "owner": "Legend ETF",
                "owner_token": owner_token,
                "account": key[0],
                "symbol": key[1],
                "activated_at": (
                    current.get("activated_at")
                    if isinstance(current, dict)
                    and current.get("owner_token") == owner_token
                    else utc_now_iso()
                ),
                "updated_at": utc_now_iso(),
                **payload,
            },
        )

    def deactivate_quarantine(
        self, account: str, symbol: str, *, owner_token: str
    ) -> None:
        key = (str(account), str(symbol).upper())
        if key not in self._held:
            raise RuntimeError("cannot clear quarantine without holding reservation")
        path = quarantine_path(self.directory, key[0], key[1])
        current = read_json(path)
        if current is None:
            return
        if not isinstance(current, dict) or current.get("protocol") != PROTOCOL_VERSION:
            raise RuntimeError("account-symbol quarantine artifact is invalid")
        if current.get("active") is not True:
            return
        if current.get("owner_token") != owner_token:
            raise RuntimeError("refusing to clear another owner's quarantine")
        atomic_write_json(
            path,
            {
                **current,
                "active": False,
                "terminal_proved_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
            },
        )

    def close(self) -> None:
        errors: list[BaseException] = []
        for key, manager in reversed(list(self._held.items())):
            try:
                manager.__exit__(None, None, None)
            except (OSError, RuntimeError) as exc:  # pragma: no cover - OS edge
                errors.append(exc)
            finally:
                self._held.pop(key, None)
        if errors:
            raise RuntimeError("one or more symbol reservations did not release")
