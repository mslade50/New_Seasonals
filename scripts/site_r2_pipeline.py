"""R2-only materialization boundary for private-site production builds.

There are two isolated workspaces in the production workflow:

* generator: pulls canonical inputs from R2 and builds ledger/ideas/risk files;
* assembler: pulls the same canonical inputs plus an immutable, run-scoped
  generated bundle from R2, then assembles ``dist/``.

No repository ``data/`` directory is copied into either workspace.  This
module fails closed on missing required objects, failed uploads, digest drift,
or attempts to use it outside GitHub Actions' marked cloud stages.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import cache_io
from scripts.stage_private_site_cloud_build import STAGE_MARKER


PROVENANCE_PATH = "data/.site-r2-provenance.json"
GENERATED_MANIFEST_NAME = "manifest.json"
GENERATED_PREFIX = "site/builds"
LOCAL_PRIMARY_ENV = "LOCAL_AUTOMATION_PRIMARY"
LOCAL_RUN_TOKEN_ENV = "LOCAL_AUTOMATION_RUN_TOKEN"


@dataclass(frozen=True)
class R2Input:
    name: str
    key: str
    path: str
    required: bool = True


CANONICAL_INPUTS: tuple[R2Input, ...] = (
    R2Input("master_prices", "master_prices.parquet", "data/master_prices.parquet"),
    R2Input("earnings_calendar", "earnings_calendar.parquet", "data/earnings_calendar.parquet"),
    R2Input("atr_seasonal_ranks", "atr_seasonal_ranks.parquet", "atr_seasonal_ranks.parquet"),
    R2Input("analyst_grades", "analyst_grades.parquet", "data/analyst_grades.parquet"),
    R2Input("fragility", "rd2_fragility.parquet", "data/rd2_fragility.parquet"),
    R2Input("risk_environment", "rd2_environment.json", "data/rd2_environment.json"),
    R2Input("exposure_state", "exposure_state.json", "data/exposure_state.json"),
    R2Input("dial_sleeve", "dial_sleeve_paper.json", "data/dial_sleeve_paper.json"),
    R2Input("cboe_putcall", "cboe_putcall.parquet", "data/cboe_putcall.parquet"),
    R2Input("sector_map", "sector_map.parquet", "data/sector_map.parquet"),
    R2Input("trade_console_stats", "trade_console_stats.json", "data/trade_console_stats.json"),
    R2Input(
        "fundamental_daily",
        "fundamental/current/daily_report_latest.json",
        "data/fundamental/current/daily_report_latest.json",
        False,
    ),
    R2Input(
        "fundamental_maps",
        "fundamental/current/company_maps_latest.json",
        "data/fundamental/current/company_maps_latest.json",
        False,
    ),
    R2Input("overflow_universe", "overflow_universe.parquet", "data/overflow_universe.parquet", False),
    R2Input(
        "overflow_earnings",
        "earnings_calendar_overflow.parquet",
        "data/earnings_calendar_overflow.parquet",
        False,
    ),
    R2Input("iv_history", "options/iv_history.parquet", "data/iv_history.parquet", False),
    R2Input(
        "option_surface_history",
        "options/surface_history.parquet",
        "data/option_surface_history.parquet",
        False,
    ),
)

GENERATED_INPUTS: tuple[R2Input, ...] = (
    R2Input("ledger", "backtest_trades_full.parquet", "data/backtest_trades_full.parquet"),
    R2Input("ledger_daily", "backtest_daily_pnl.parquet", "data/backtest_daily_pnl.parquet"),
    # Optional while no strategy carries ``sector_loss_gate``.  The ledger
    # builder intentionally omits this counterfactual in that state and the
    # site hides the retired gate-lab panel.  If a gate is reintroduced, the
    # generated file is still published and provenance-checked normally.
    R2Input(
        "ledger_nogate",
        "backtest_trades_nogate.parquet",
        "data/backtest_trades_nogate.parquet",
        False,
    ),
    R2Input("ledger_ovsext", "backtest_trades_ovsext.parquet", "data/backtest_trades_ovsext.parquet"),
    R2Input("seasonal_ideas", "daily_seasonal_ideas.json", "data/daily_seasonal_ideas.json"),
    R2Input("site_risk", "site_risk.json", "data/site_risk.json"),
    R2Input("atr_downside", "atr_downside_stats.json", "data/atr_downside_stats.json"),
    R2Input("betas", "betas.json", "data/betas.json", False),
)

PUBLISH_GROUPS: dict[str, tuple[R2Input, ...]] = {
    "risk": tuple(i for i in CANONICAL_INPUTS if i.name in {"fragility", "risk_environment", "dial_sleeve"}),
    "exposure": tuple(i for i in CANONICAL_INPUTS if i.name == "exposure_state"),
    "cboe": tuple(i for i in CANONICAL_INPUTS if i.name == "cboe_putcall"),
    "reference": tuple(i for i in CANONICAL_INPUTS if i.name in {"sector_map", "trade_console_stats"}),
}
PUBLISH_GROUPS["bootstrap"] = tuple(
    item
    for group in ("risk", "exposure", "cboe", "reference")
    for item in PUBLISH_GROUPS[group]
)


def _require_github_actions() -> None:
    if os.environ.get("GITHUB_ACTIONS", "").lower() != "true":
        raise RuntimeError("private-site R2 publishing/materialization is GitHub-Actions-only")


def _require_cloud_stage(root: Path, *, empty_runtime: bool) -> dict:
    _require_github_actions()
    if os.environ.get("PRIVATE_SITE_CLOUD_BUILD") != "1":
        raise RuntimeError("PRIVATE_SITE_CLOUD_BUILD=1 is required")
    marker_path = root / STAGE_MARKER
    if not marker_path.is_file():
        raise RuntimeError(f"cloud-stage marker missing: {marker_path}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("mode") != "private-site-cloud-source":
        raise RuntimeError("invalid cloud-stage marker")
    if empty_runtime:
        forbidden = [
            rel
            for rel in ("data", "dist", "dist-shared", "charts", "reports", "scratch", "atr_seasonal_ranks.parquet")
            if (root / rel).exists()
        ]
        if forbidden:
            raise RuntimeError(f"cloud stage is not data-empty: {', '.join(forbidden)}")
    return marker


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _head_metadata(key: str, *, expected_size: int | None = None) -> dict:
    meta = cache_io.head(key)
    if not meta:
        raise RuntimeError(f"R2 HEAD failed after transfer: {key}")
    if "ContentLength" not in meta:
        raise RuntimeError(f"R2 HEAD did not return ContentLength after transfer: {key}")
    try:
        remote_size = int(meta["ContentLength"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"R2 HEAD returned invalid ContentLength after transfer: {key}") from exc
    if expected_size is not None and remote_size != expected_size:
        raise RuntimeError(
            f"R2 HEAD ContentLength mismatch after transfer: {key} "
            f"(local={expected_size}, remote={remote_size})"
        )
    modified = meta.get("LastModified")
    return {
        "etag": str(meta.get("ETag") or "").strip('"'),
        "last_modified": modified.isoformat() if hasattr(modified, "isoformat") else str(modified or ""),
        "size": remote_size,
    }


def _entry(item: R2Input, path: Path, *, key: str | None = None) -> dict:
    r2_key = key or item.key
    return {
        "name": item.name,
        "key": r2_key,
        "path": item.path,
        "required": item.required,
        "sha256": _sha256(path),
        **_head_metadata(r2_key, expected_size=path.stat().st_size),
    }


def _verify_uploaded_file(path: Path, key: str) -> dict:
    """Fail closed unless R2 confirms the exact uploaded byte count."""
    return _head_metadata(key, expected_size=path.stat().st_size)


def _download(root: Path, item: R2Input, *, key: str | None = None) -> dict | None:
    r2_key = key or item.key
    target = root / item.path
    if cache_io.download_to_local(r2_key, str(target)):
        rec = _entry(item, target, key=r2_key)
        print(f"[site-r2] verified {r2_key} sha256={rec['sha256'][:12]}")
        return rec
    if item.required:
        raise RuntimeError(f"required private-site R2 object unavailable: {r2_key}")
    print(f"[site-r2] optional object unavailable: {r2_key}")
    return None


def _write_provenance(root: Path, *, phase: str, run_id: str | None, marker: dict, entries: list[dict]) -> dict:
    payload = {
        "mode": "r2-only",
        "phase": phase,
        "run_id": run_id,
        "source_sha": marker.get("source_sha"),
        "materialized_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "entries": entries,
    }
    path = root / PROVENANCE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def pull_generator(root: Path) -> dict:
    marker = _require_cloud_stage(root, empty_runtime=True)
    entries = [rec for item in CANONICAL_INPUTS if (rec := _download(root, item)) is not None]
    return _write_provenance(root, phase="generator", run_id=None, marker=marker, entries=entries)


def _run_prefix(run_id: str) -> str:
    clean = str(run_id).strip()
    if not clean or not clean.replace("-", "").isalnum():
        raise ValueError("run id must contain only letters, numbers, and hyphens")
    return f"{GENERATED_PREFIX}/{clean}"


def publish_generated(root: Path, run_id: str) -> dict:
    marker = _require_cloud_stage(root, empty_runtime=False)
    prefix = _run_prefix(run_id)
    entries: list[dict] = []
    for item in GENERATED_INPUTS:
        path = root / item.path
        if not path.is_file():
            if item.required:
                raise RuntimeError(f"required generated site input missing: {item.path}")
            print(f"[site-r2] optional generated input absent: {item.path}")
            continue
        key = f"{prefix}/{item.key}"
        if not cache_io.upload_from_local(str(path), key):
            raise RuntimeError(f"failed to publish generated site input: {key}")
        entries.append(_entry(item, path, key=key))

    payload = {
        "mode": "private-site-generated-bundle",
        "run_id": str(run_id),
        "source_sha": marker.get("source_sha"),
        "published_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "entries": entries,
    }
    local_manifest = root / "data" / ".site-generated-bundle.json"
    local_manifest.parent.mkdir(parents=True, exist_ok=True)
    local_manifest.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    manifest_key = f"{prefix}/{GENERATED_MANIFEST_NAME}"
    if not cache_io.upload_from_local(str(local_manifest), manifest_key):
        raise RuntimeError(f"failed to publish generated bundle manifest: {manifest_key}")
    _verify_uploaded_file(local_manifest, manifest_key)
    print(f"[site-r2] published immutable generated bundle {prefix}")
    return payload


def pull_assembler(root: Path, run_id: str) -> dict:
    marker = _require_cloud_stage(root, empty_runtime=True)
    entries = [rec for item in CANONICAL_INPUTS if (rec := _download(root, item)) is not None]

    prefix = _run_prefix(run_id)
    manifest_path = root / "data" / ".site-generated-bundle.json"
    manifest_key = f"{prefix}/{GENERATED_MANIFEST_NAME}"
    if not cache_io.download_to_local(manifest_key, str(manifest_path)):
        raise RuntimeError(f"generated bundle manifest unavailable: {manifest_key}")
    bundle = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bundle.get("mode") != "private-site-generated-bundle" or str(bundle.get("run_id")) != str(run_id):
        raise RuntimeError("generated bundle manifest identity mismatch")
    by_name = {entry.get("name"): entry for entry in bundle.get("entries") or []}
    for item in GENERATED_INPUTS:
        expected = by_name.get(item.name)
        if not expected:
            if item.required:
                raise RuntimeError(f"generated bundle manifest missing {item.name}")
            print(f"[site-r2] optional generated input absent from bundle: {item.name}")
            continue
        expected_key = f"{prefix}/{item.key}"
        if expected.get("key") != expected_key:
            raise RuntimeError(f"generated bundle key mismatch for {item.name}")
        rec = _download(root, item, key=expected_key)
        if rec is None:
            if item.required:
                raise RuntimeError(f"generated bundle object unavailable for {item.name}")
            continue
        if rec["sha256"] != expected.get("sha256"):
            raise RuntimeError(f"generated bundle digest mismatch for {item.name}")
        entries.append(rec)
    return _write_provenance(root, phase="assembler", run_id=str(run_id), marker=marker, entries=entries)


def _require_publish_group_authority(*, local_primary: bool) -> None:
    if not local_primary:
        _require_github_actions()
        return

    if os.environ.get(LOCAL_PRIMARY_ENV, "").strip() != "1":
        raise RuntimeError(f"{LOCAL_PRIMARY_ENV}=1 is required for local-primary publishing")
    if not os.environ.get(LOCAL_RUN_TOKEN_ENV, "").strip():
        raise RuntimeError(f"nonempty {LOCAL_RUN_TOKEN_ENV} is required for local-primary publishing")


def publish_group(root: Path, group: str, *, local_primary: bool = False) -> list[dict]:
    """Publish a bounded canonical group from an explicitly trusted producer.

    GitHub Actions remains the default and only implicit authority.  A local
    primary runner must opt in at the CLI *and* carry two process-scoped
    environment markers.  The ephemeral run token is intentionally read only
    from the environment so it never appears in command lines or logs.
    """
    _require_publish_group_authority(local_primary=local_primary)
    items = PUBLISH_GROUPS[group]
    entries: list[dict] = []
    for item in items:
        path = root / item.path
        if not path.is_file():
            raise RuntimeError(f"site R2 publisher input missing: {item.path}")
        if not cache_io.upload_from_local(str(path), item.key):
            raise RuntimeError(f"failed to publish site R2 input: {item.key}")
        entries.append(_entry(item, path))
    print(f"[site-r2] published group {group}: {len(entries)} object(s)")
    return entries


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    pull = sub.add_parser("pull")
    pull.add_argument("--phase", choices=("generator", "assembler"), required=True)
    pull.add_argument("--root", default=".")
    pull.add_argument("--run-id")
    generated = sub.add_parser("publish-generated")
    generated.add_argument("--root", default=".")
    generated.add_argument("--run-id", required=True)
    publisher = sub.add_parser("publish-group")
    publisher.add_argument("--root", default=".")
    publisher.add_argument("--group", choices=sorted(PUBLISH_GROUPS), required=True)
    publisher.add_argument(
        "--local-primary",
        action="store_true",
        help=(
            "allow the guarded local primary publisher; also requires "
            f"{LOCAL_PRIMARY_ENV}=1 and a nonempty {LOCAL_RUN_TOKEN_ENV}"
        ),
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()

    if args.command == "pull":
        if args.phase == "generator":
            pull_generator(root)
        else:
            if not args.run_id:
                parser.error("--run-id is required for assembler pulls")
            pull_assembler(root, args.run_id)
    elif args.command == "publish-generated":
        publish_generated(root, args.run_id)
    else:
        publish_group(root, args.group, local_primary=args.local_primary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
