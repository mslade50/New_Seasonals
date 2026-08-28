"""Run the streaming intraday v0 study from explicit local research inputs."""

from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.intraday.diagnostics import DEFAULT_COST_GRID_BPS
from research.intraday.eligibility import EligibilityConfig
from research.intraday.streaming import (
    run_streaming_intraday_research,
    write_streaming_research_artifacts,
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_hash(path: Path, expected: str | None, label: str) -> str:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    actual = _sha256_file(resolved)
    if expected and actual.lower() != expected.lower():
        raise ValueError(
            f"{label} SHA-256 mismatch: expected {expected.lower()}, got {actual}"
        )
    return actual


def _load_snapshot_provenance(
    manifest_path: Path | None,
    *,
    data_dir: Path,
    expected_index_sha256: str | None,
) -> tuple[dict[str, object], dict[str, str]]:
    if manifest_path is None:
        if expected_index_sha256:
            raise ValueError("--expected-snapshot-index-sha256 requires --snapshot-manifest")
        return {}, {}
    resolved = manifest_path.resolve()
    manifest_sha256 = _validated_hash(resolved, None, "snapshot manifest")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "intraday-r2-snapshot.v1":
        raise ValueError("snapshot manifest has an unsupported schema_version")
    expected_bars = (resolved.parent / "bars").resolve()
    if data_dir.resolve() != expected_bars:
        raise ValueError(
            f"data directory must be the snapshot bars directory: {expected_bars}"
        )
    index_sha256 = str(payload.get("source_meta_sha256", "")).lower()
    if expected_index_sha256 and index_sha256 != expected_index_sha256.lower():
        raise ValueError(
            "snapshot index SHA-256 mismatch: "
            f"expected {expected_index_sha256.lower()}, got {index_sha256}"
        )
    item_hashes = {
        Path(str(item["relative_path"])).name: str(item["sha256"]).lower()
        for item in payload.get("items", [])
    }
    provenance = {
        "snapshot_manifest_path": str(resolved),
        "snapshot_manifest_sha256": manifest_sha256,
        "snapshot_schema_version": payload.get("schema_version"),
        "snapshot_source_meta_sha256": index_sha256,
        "snapshot_ticker_count": payload.get("ticker_count"),
        "snapshot_total_rows": payload.get("total_rows"),
        "snapshot_total_bytes": payload.get("total_bytes"),
    }
    return provenance, item_hashes


def _validate_used_files_against_snapshot(
    result,
    item_hashes: dict[str, str],
) -> int:
    if not item_hashes:
        return 0
    loaded = result.coverage_audit.loc[result.coverage_audit["status"].eq("loaded")]
    validated = 0
    for row in loaded.itertuples(index=False):
        filename = Path(str(row.input_path)).name
        expected = item_hashes.get(filename)
        if expected is None:
            raise ValueError(f"loaded input is absent from snapshot manifest: {filename}")
        if str(row.input_sha256).lower() != expected:
            raise ValueError(f"loaded input hash differs from snapshot manifest: {filename}")
        validated += 1
    return validated


def _read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"input table does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"input table must be parquet or CSV: {path}")


def _read_universe(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"universe file does not exist: {path}")
    if path.suffix.lower() in {".csv", ".parquet"}:
        frame = _read_table(path)
        columns = {str(column).strip().lower(): column for column in frame.columns}
        if "ticker" not in columns:
            raise ValueError("universe table must contain a ticker column")
        values = frame[columns["ticker"]].tolist()
    elif path.suffix.lower() in {".txt", ".list"}:
        values = path.read_text(encoding="utf-8").splitlines()
    else:
        raise ValueError("universe file must be CSV, parquet, TXT, or LIST")
    tickers = sorted({str(value).upper().strip() for value in values if str(value).strip()})
    if not tickers:
        raise ValueError("universe file contains no tickers")
    return tickers


def _default_output_dir() -> Path:
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    return ROOT / "artifacts" / "intraday_streaming_research" / stamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Research-only streaming intraday v0 runner. Reads local parquets; "
            "never downloads, uploads, stages, or promotes."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Explicit directory containing {TICKER}_15min.parquet files.",
    )
    parser.add_argument("--sector-map", type=Path, required=True)
    universe = parser.add_mutually_exclusive_group(required=True)
    universe.add_argument("--universe-file", type=Path)
    universe.add_argument("--tickers", nargs="+")
    parser.add_argument(
        "--cost-grid-bps",
        nargs="+",
        type=float,
        default=list(DEFAULT_COST_GRID_BPS),
        help="Round-trip cost grid; must include the locked 10 bps primary case.",
    )
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-dollar-volume", type=float, default=25_000_000.0)
    parser.add_argument("--min-completeness", type=float, default=0.95)
    parser.add_argument("--bootstrap-reps", type=int, default=2_000)
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument("--expected-snapshot-index-sha256")
    parser.add_argument("--expected-universe-sha256")
    parser.add_argument("--expected-sector-map-sha256")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sector_map_sha256 = _validated_hash(
        args.sector_map, args.expected_sector_map_sha256, "sector map"
    )
    universe_file_sha256 = (
        _validated_hash(
            args.universe_file,
            args.expected_universe_sha256,
            "universe file",
        )
        if args.universe_file is not None
        else None
    )
    snapshot_provenance, snapshot_item_hashes = _load_snapshot_provenance(
        args.snapshot_manifest,
        data_dir=args.data_dir,
        expected_index_sha256=args.expected_snapshot_index_sha256,
    )
    source_provenance = {
        **snapshot_provenance,
        "sector_map_path": str(args.sector_map.resolve()),
        "sector_map_file_sha256": sector_map_sha256,
        "universe_file_path": (
            str(args.universe_file.resolve()) if args.universe_file is not None else None
        ),
        "universe_file_sha256": universe_file_sha256,
        "expected_hashes_enforced": bool(
            args.expected_snapshot_index_sha256
            or args.expected_universe_sha256
            or args.expected_sector_map_sha256
        ),
    }
    metadata = _read_table(args.sector_map)
    candidates = (
        _read_universe(args.universe_file)
        if args.universe_file is not None
        else sorted({str(ticker).upper().strip() for ticker in args.tickers})
    )
    eligibility_config = EligibilityConfig(
        min_price=args.min_price,
        min_median_dollar_volume=args.min_dollar_volume,
        min_data_completeness=args.min_completeness,
    )
    result = run_streaming_intraday_research(
        args.data_dir,
        metadata,
        candidates,
        eligibility_config=eligibility_config,
        cost_grid_bps=tuple(args.cost_grid_bps),
        bootstrap_reps=args.bootstrap_reps,
        source_provenance=source_provenance,
    )
    result.source_provenance["n_loaded_files_validated_against_snapshot"] = (
        _validate_used_files_against_snapshot(result, snapshot_item_hashes)
    )
    output_dir = write_streaming_research_artifacts(
        result, args.output_dir or _default_output_dir()
    )
    print(f"Research artifacts: {output_dir}")
    print(
        f"Requested {len(result.requested_tickers)} | "
        f"evaluated {len(result.loaded_candidate_tickers)} | "
        f"signals {len(result.signals)} | primary trades {len(result.trades)} | "
        f"execution rejected {len(result.execution_rejections)}"
    )
    primary = result.day_cluster_stats.loc[
        result.day_cluster_stats["primary_cost_case"]
    ]
    if primary.empty:
        print("No testable primary-cost day clusters.")
    else:
        print(primary.to_string(index=False))
    print(
        "Research-only: no network, R2 mutation, production write, broker action, "
        "order, schedule, or automatic promotion occurred."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
