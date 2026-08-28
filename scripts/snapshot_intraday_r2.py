"""Create an immutable, read-only research snapshot of the R2 intraday cache.

The script intentionally exposes no upload path.  It downloads the cache
index and every parquet named by that index into a fresh directory beneath
``artifacts/``, validates the parquet envelopes, hashes each payload, and
publishes the manifest last.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = (ROOT / "artifacts").resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cache_io import (
    _client as _r2_client,
)
from cache_io import (
    _r2_creds,
    download_to_local,
    is_configured,
    list_keys,
)

R2_PREFIX = "intraday/15min"
META_KEY = f"{R2_PREFIX}/_meta.parquet"
REQUIRED_META_COLUMNS = {
    "ticker",
    "interval",
    "n_bars",
    "first_ts",
    "last_ts",
    "n_days",
    "size_kb",
}
REQUIRED_BAR_COLUMNS = {"ts", "open", "high", "low", "close", "volume"}
SAFE_TICKER = re.compile(r"^[A-Za-z0-9.^=_-]+$")


class IntradaySnapshotError(RuntimeError):
    """Raised when the remote cache cannot form an auditable snapshot."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _list_r2_inventory(prefix: str) -> dict[str, dict]:
    """Return read-only object fingerprints for a prefix without secret values."""

    client = _r2_client()
    creds = _r2_creds()
    if client is None or creds is None:
        raise IntradaySnapshotError("R2 credentials are not configured")
    inventory: dict[str, dict] = {}
    token: str | None = None
    while True:
        kwargs: dict = {
            "Bucket": creds["R2_BUCKET"],
            "Prefix": prefix,
            "MaxKeys": 1000,
        }
        if token:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        for item in response.get("Contents", []):
            inventory[str(item["Key"])] = {
                "etag": str(item["ETag"]).strip('"').lower(),
                "bytes": int(item["Size"]),
                "last_modified": item["LastModified"].isoformat(),
            }
        if not response.get("IsTruncated"):
            break
        token = response.get("NextContinuationToken")
        if not token:
            raise IntradaySnapshotError("R2 inventory pagination omitted its token")
    return inventory


def _resolve_fresh_output(
    requested: Path | None, *, artifacts_root: Path = ARTIFACTS_ROOT
) -> tuple[Path, Path]:
    root = artifacts_root.resolve()
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    output = (
        requested.resolve()
        if requested is not None
        else (root / "intraday-r2-snapshots" / stamp).resolve()
    )
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"--output-dir must stay under {root}") from exc
    partial = output.with_name(f".{output.name}.partial")
    for path in (output, partial):
        if path.exists():
            raise ValueError(f"snapshot target must not already exist: {path}")
    return output, partial


def _validate_meta(frame: pd.DataFrame, *, min_tickers: int) -> pd.DataFrame:
    missing = REQUIRED_META_COLUMNS.difference(frame.columns)
    if missing:
        raise IntradaySnapshotError(f"intraday meta missing columns: {sorted(missing)}")
    work = frame.copy()
    work["ticker"] = work["ticker"].astype(str).str.strip().str.upper()
    if work["ticker"].duplicated().any():
        duplicate = work.loc[work["ticker"].duplicated(), "ticker"].iloc[0]
        raise IntradaySnapshotError(f"intraday meta has duplicate ticker: {duplicate}")
    invalid = [ticker for ticker in work["ticker"] if not SAFE_TICKER.fullmatch(ticker)]
    if invalid:
        raise IntradaySnapshotError(f"intraday meta has unsafe ticker: {invalid[0]!r}")
    if len(work) < min_tickers:
        raise IntradaySnapshotError(
            f"intraday meta has {len(work)} tickers; minimum is {min_tickers}"
        )
    if set(work["interval"].astype(str)) != {"15min"}:
        raise IntradaySnapshotError("intraday meta contains a non-15min interval")
    numeric = work[["n_bars", "n_days", "size_kb"]].apply(
        pd.to_numeric, errors="coerce"
    )
    if numeric.isna().any().any() or (numeric <= 0).any().any():
        raise IntradaySnapshotError("intraday meta has invalid counts or sizes")
    work[["n_bars", "n_days", "size_kb"]] = numeric
    work["first_ts"] = pd.to_datetime(work["first_ts"], errors="coerce")
    work["last_ts"] = pd.to_datetime(work["last_ts"], errors="coerce")
    if work[["first_ts", "last_ts"]].isna().any().any():
        raise IntradaySnapshotError("intraday meta has invalid timestamps")
    if work["last_ts"].lt(work["first_ts"]).any():
        raise IntradaySnapshotError("intraday meta has last_ts before first_ts")
    return work.sort_values("ticker").reset_index(drop=True)


def _download_with_retries(
    key: str,
    path: Path,
    *,
    download: Callable[[str, str], bool],
    retries: int,
) -> None:
    for attempt in range(1, retries + 1):
        if download(key, str(path)):
            return
        if attempt < retries:
            time.sleep(min(2**attempt, 8))
    raise IntradaySnapshotError(f"failed to download {key} after {retries} attempts")


def snapshot_intraday_cache(
    output_dir: Path,
    *,
    artifacts_root: Path = ARTIFACTS_ROOT,
    min_tickers: int = 150,
    retries: int = 3,
    list_remote_keys: Callable[[str], set[str]] = list_keys,
    download: Callable[[str, str], bool] = download_to_local,
    inventory_remote: Callable[[str], dict[str, dict]] | None = None,
) -> Path:
    """Download and validate a complete cache snapshot, returning its directory."""

    output, partial = _resolve_fresh_output(
        output_dir, artifacts_root=artifacts_root
    )
    partial.mkdir(parents=True)
    inventory_before = (
        inventory_remote(f"{R2_PREFIX}/") if inventory_remote is not None else None
    )
    remote_keys = (
        set(inventory_before)
        if inventory_before is not None
        else set(list_remote_keys(f"{R2_PREFIX}/"))
    )
    if META_KEY not in remote_keys:
        raise IntradaySnapshotError(f"remote cache is missing {META_KEY}")

    meta_path = partial / "_meta.parquet"
    _download_with_retries(META_KEY, meta_path, download=download, retries=retries)
    meta = _validate_meta(pd.read_parquet(meta_path), min_tickers=min_tickers)
    expected_keys = {META_KEY} | {
        f"{R2_PREFIX}/{ticker}.parquet" for ticker in meta["ticker"]
    }
    missing_remote = sorted(expected_keys.difference(remote_keys))
    if missing_remote:
        raise IntradaySnapshotError(
            f"remote cache is missing {len(missing_remote)} indexed objects; "
            f"first={missing_remote[0]}"
        )

    bars_dir = partial / "bars"
    bars_dir.mkdir()
    items: list[dict] = []
    for sequence, row in enumerate(meta.itertuples(index=False), start=1):
        ticker = str(row.ticker)
        key = f"{R2_PREFIX}/{ticker}.parquet"
        local = bars_dir / f"{ticker}_15min.parquet"
        _download_with_retries(key, local, download=download, retries=retries)
        parquet = pq.ParquetFile(local)
        try:
            columns = set(parquet.schema.names)
            missing_columns = REQUIRED_BAR_COLUMNS.difference(columns)
            if missing_columns:
                raise IntradaySnapshotError(
                    f"{ticker} parquet missing columns: {sorted(missing_columns)}"
                )
            n_rows = int(parquet.metadata.num_rows)
        finally:
            parquet.close()
        if n_rows != int(row.n_bars):
            raise IntradaySnapshotError(
                f"{ticker} row count {n_rows} does not match meta {int(row.n_bars)}"
            )
        items.append(
            {
                "sequence": sequence,
                "ticker": ticker,
                "key": key,
                "relative_path": str(local.relative_to(partial)).replace("\\", "/"),
                "bytes": local.stat().st_size,
                "md5": _md5(local),
                "sha256": _sha256(local),
                "n_rows": n_rows,
                "first_ts": row.first_ts,
                "last_ts": row.last_ts,
                "n_days": int(row.n_days),
            }
        )

    inventory_after = (
        inventory_remote(f"{R2_PREFIX}/") if inventory_remote is not None else None
    )
    if inventory_before is not None:
        if inventory_after != inventory_before:
            raise IntradaySnapshotError("R2 inventory changed during the snapshot")
        local_by_key = {item["key"]: item for item in items}
        local_by_key[META_KEY] = {
            "bytes": meta_path.stat().st_size,
            "md5": _md5(meta_path),
        }
        for key in sorted(expected_keys):
            remote = inventory_before[key]
            local = local_by_key[key]
            if int(remote["bytes"]) != int(local["bytes"]):
                raise IntradaySnapshotError(f"remote/local size mismatch for {key}")
            etag = str(remote["etag"]).lower()
            if "-" in etag:
                raise IntradaySnapshotError(
                    f"multipart ETag cannot prove byte identity for {key}"
                )
            if etag != str(local["md5"]).lower():
                raise IntradaySnapshotError(f"remote/local ETag mismatch for {key}")

    completed_utc = pd.Timestamp.now(tz="UTC").isoformat()
    manifest = {
        "schema_version": "intraday-r2-snapshot.v1",
        "research_only": True,
        "read_only_r2": True,
        "no_upload": True,
        "no_order": True,
        "production_writes": False,
        "automatic_promotion": False,
        "completed_utc": completed_utc,
        "source_prefix": R2_PREFIX,
        "source_meta_key": META_KEY,
        "source_meta_sha256": _sha256(meta_path),
        "remote_key_count_under_prefix": len(remote_keys),
        "remote_inventory_verified": inventory_before is not None,
        "remote_inventory": inventory_before,
        "unexpected_remote_keys": sorted(remote_keys.difference(expected_keys)),
        "ticker_count": len(items),
        "total_rows": sum(item["n_rows"] for item in items),
        "total_bytes": sum(item["bytes"] for item in items),
        "items": items,
    }
    (partial / "snapshot_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8"
    )
    partial.rename(output)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read-only R2 intraday snapshot into an immutable artifact directory."
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-tickers", type=int, default=150)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not is_configured():
        raise IntradaySnapshotError("R2 credentials are not configured")
    requested = args.output_dir
    if requested is None:
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
        requested = ARTIFACTS_ROOT / "intraday-r2-snapshots" / stamp
    output = snapshot_intraday_cache(
        requested,
        min_tickers=args.min_tickers,
        retries=args.retries,
        inventory_remote=_list_r2_inventory,
    )
    manifest = json.loads((output / "snapshot_manifest.json").read_text("utf-8"))
    print(f"Immutable intraday snapshot: {output}")
    print(
        f"{manifest['ticker_count']} tickers | {manifest['total_rows']:,} bars | "
        f"{manifest['total_bytes'] / (1024**2):,.1f} MiB"
    )
    print("R2 access was read-only; no production object was uploaded or modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
