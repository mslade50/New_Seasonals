import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scripts import build_site
from scripts.site_r2_pipeline import CANONICAL_INPUTS, GENERATED_INPUTS, PROVENANCE_PATH
from scripts.stage_private_site_cloud_build import STAGE_MARKER


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _production_stage(root: Path) -> list[dict]:
    (root / STAGE_MARKER).write_text(
        json.dumps({"mode": "private-site-cloud-source", "source_sha": "abc"}),
        encoding="utf-8",
    )
    entries = []
    for item in (*CANONICAL_INPUTS, *GENERATED_INPUTS):
        if not item.required:
            continue
        path = root / item.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"r2:{item.name}".encode())
        entries.append({
            "name": item.name,
            "key": item.key,
            "path": item.path,
            "sha256": _sha256(path),
        })
    provenance = root / PROVENANCE_PATH
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(json.dumps({
        "mode": "r2-only",
        "phase": "assembler",
        "run_id": "123-1",
        "source_sha": "abc",
        "entries": entries,
    }), encoding="utf-8")
    return entries


def test_production_boundary_rejects_digest_drift_and_extra_files(tmp_path, monkeypatch):
    entries = _production_stage(tmp_path)
    monkeypatch.setattr(build_site, "_ROOT", str(tmp_path))
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("PRIVATE_SITE_CLOUD_BUILD", "1")
    monkeypatch.setenv("GITHUB_RUN_ID", "123")
    monkeypatch.setenv("GITHUB_RUN_ATTEMPT", "1")
    monkeypatch.setenv("GITHUB_SHA", "abc")

    assert build_site.load_production_provenance()["mode"] == "r2-only"

    first = tmp_path / entries[0]["path"]
    first.write_bytes(b"local mutation")
    with pytest.raises(RuntimeError, match="digest changed"):
        build_site.load_production_provenance()

    first.write_bytes(f"r2:{entries[0]['name']}".encode())
    extra = tmp_path / "data" / "local-only.parquet"
    extra.write_bytes(b"must never deploy")
    with pytest.raises(RuntimeError, match="unprovenanced file"):
        build_site.load_production_provenance()
