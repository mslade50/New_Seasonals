from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.snapshot_intraday_r2 import (
    META_KEY,
    R2_PREFIX,
    IntradaySnapshotError,
    snapshot_intraday_cache,
)


def _bar_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts": pd.date_range("2026-01-02 09:30", periods=2, freq="15min"),
            "open": [10.0, 10.1],
            "high": [10.2, 10.3],
            "low": [9.9, 10.0],
            "close": [10.1, 10.2],
            "volume": [100, 120],
        }
    )


def _remote_payloads(tmp_path: Path, *, include_bar: bool = True) -> dict[str, Path]:
    source = tmp_path / "source"
    source.mkdir()
    meta = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "interval": "15min",
                "n_bars": 2,
                "first_ts": pd.Timestamp("2026-01-02 09:30"),
                "last_ts": pd.Timestamp("2026-01-02 09:45"),
                "n_days": 1,
                "size_kb": 1.0,
            }
        ]
    )
    meta_path = source / "meta.parquet"
    meta.to_parquet(meta_path, index=False)
    payloads = {META_KEY: meta_path}
    if include_bar:
        bar_path = source / "AAA.parquet"
        _bar_frame().to_parquet(bar_path, index=False)
        payloads[f"{R2_PREFIX}/AAA.parquet"] = bar_path
    return payloads


def _fake_download(payloads: dict[str, Path]):
    def download(key: str, destination: str) -> bool:
        source = payloads.get(key)
        if source is None:
            return False
        Path(destination).write_bytes(source.read_bytes())
        return True

    return download


def test_snapshot_is_complete_hashed_and_manifest_last(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    payloads = _remote_payloads(tmp_path)
    output = artifacts / "run"
    result = snapshot_intraday_cache(
        output,
        artifacts_root=artifacts,
        min_tickers=1,
        list_remote_keys=lambda _prefix: set(payloads),
        download=_fake_download(payloads),
    )

    assert result == output
    assert (output / "bars" / "AAA_15min.parquet").is_file()
    manifest = json.loads((output / "snapshot_manifest.json").read_text("utf-8"))
    assert manifest["read_only_r2"] is True
    assert manifest["no_upload"] is True
    assert manifest["production_writes"] is False
    assert manifest["ticker_count"] == 1
    assert manifest["total_rows"] == 2
    assert len(manifest["items"][0]["sha256"]) == 64
    assert len(manifest["items"][0]["md5"]) == 32


def test_snapshot_verifies_unchanged_single_part_remote_inventory(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    payloads = _remote_payloads(tmp_path)
    inventory = {
        key: {
            "etag": __import__("hashlib").md5(
                path.read_bytes(), usedforsecurity=False
            ).hexdigest(),
            "bytes": path.stat().st_size,
            "last_modified": "2026-08-27T00:00:00+00:00",
        }
        for key, path in payloads.items()
    }
    output = snapshot_intraday_cache(
        artifacts / "verified",
        artifacts_root=artifacts,
        min_tickers=1,
        download=_fake_download(payloads),
        inventory_remote=lambda _prefix: inventory,
    )
    manifest = json.loads((output / "snapshot_manifest.json").read_text("utf-8"))
    assert manifest["remote_inventory_verified"] is True


def test_snapshot_fails_when_remote_inventory_changes(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    payloads = _remote_payloads(tmp_path)
    inventory = {
        key: {
            "etag": __import__("hashlib").md5(
                path.read_bytes(), usedforsecurity=False
            ).hexdigest(),
            "bytes": path.stat().st_size,
            "last_modified": "2026-08-27T00:00:00+00:00",
        }
        for key, path in payloads.items()
    }
    calls = 0

    def changing_inventory(_prefix: str) -> dict[str, dict]:
        nonlocal calls
        calls += 1
        if calls == 1:
            return inventory
        changed = {key: dict(value) for key, value in inventory.items()}
        changed[f"{R2_PREFIX}/AAA.parquet"]["etag"] = "0" * 32
        return changed

    with pytest.raises(IntradaySnapshotError, match="changed during"):
        snapshot_intraday_cache(
            artifacts / "changed",
            artifacts_root=artifacts,
            min_tickers=1,
            download=_fake_download(payloads),
            inventory_remote=changing_inventory,
        )


def test_snapshot_fails_when_meta_names_missing_remote_object(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    payloads = _remote_payloads(tmp_path, include_bar=False)
    with pytest.raises(IntradaySnapshotError, match="missing 1 indexed objects"):
        snapshot_intraday_cache(
            artifacts / "run",
            artifacts_root=artifacts,
            min_tickers=1,
            list_remote_keys=lambda _prefix: set(payloads),
            download=_fake_download(payloads),
        )


def test_snapshot_refuses_paths_outside_artifacts(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    with pytest.raises(ValueError, match="must stay under"):
        snapshot_intraday_cache(
            tmp_path / "elsewhere" / "run",
            artifacts_root=artifacts,
            min_tickers=1,
            list_remote_keys=lambda _prefix: set(),
            download=lambda _key, _path: False,
        )


def test_snapshot_refuses_existing_or_partial_target(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    output = artifacts / "run"
    output.mkdir()
    with pytest.raises(ValueError, match="must not already exist"):
        snapshot_intraday_cache(
            output,
            artifacts_root=artifacts,
            min_tickers=1,
            list_remote_keys=lambda _prefix: set(),
            download=lambda _key, _path: False,
        )
