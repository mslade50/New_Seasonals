import json
from pathlib import Path

import pytest

from scripts import site_r2_pipeline as pipeline


def _stage(root: Path, sha="b" * 40):
    root.mkdir(parents=True)
    (root / ".private-site-cloud-stage.json").write_text(
        json.dumps({"mode": "private-site-cloud-source", "source_sha": sha}),
        encoding="utf-8",
    )


def _fake_r2(monkeypatch, initial=None):
    store = dict(initial or {})

    def download(key, local):
        if key not in store:
            return False
        path = Path(local)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(store[key])
        return True

    def upload(local, key):
        path = Path(local)
        if not path.is_file():
            return False
        store[key] = path.read_bytes()
        return True

    def head(key):
        if key not in store:
            return None
        return {"ETag": f'"{key}-etag"', "ContentLength": len(store[key]), "LastModified": "now"}

    monkeypatch.setattr(pipeline.cache_io, "download_to_local", download)
    monkeypatch.setattr(pipeline.cache_io, "upload_from_local", upload)
    monkeypatch.setattr(pipeline.cache_io, "head", head)
    return store


def test_two_workspace_round_trip_is_r2_only(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("PRIVATE_SITE_CLOUD_BUILD", "1")
    canonical = {
        item.key: f"canonical:{item.name}".encode()
        for item in pipeline.CANONICAL_INPUTS
        if item.required
    }
    store = _fake_r2(monkeypatch, canonical)

    generator = tmp_path / "generator"
    _stage(generator)
    generated_provenance = pipeline.pull_generator(generator)
    assert generated_provenance["phase"] == "generator"
    assert {e["name"] for e in generated_provenance["entries"]} >= {
        item.name for item in pipeline.CANONICAL_INPUTS if item.required
    }

    for item in pipeline.GENERATED_INPUTS:
        path = generator / item.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"generated:{item.name}".encode())
    bundle = pipeline.publish_generated(generator, "12345")
    assert bundle["run_id"] == "12345"
    assert "site/builds/12345/manifest.json" in store

    assembler = tmp_path / "assembler"
    _stage(assembler)
    provenance = pipeline.pull_assembler(assembler, "12345")
    assert provenance["phase"] == "assembler"
    assert provenance["run_id"] == "12345"
    assert {e["name"] for e in provenance["entries"]} >= {
        item.name for item in pipeline.GENERATED_INPUTS
    }
    for item in pipeline.GENERATED_INPUTS:
        assert (assembler / item.path).read_bytes() == f"generated:{item.name}".encode()


def test_optional_generated_input_can_be_absent_from_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("PRIVATE_SITE_CLOUD_BUILD", "1")
    canonical = {
        item.key: f"canonical:{item.name}".encode()
        for item in pipeline.CANONICAL_INPUTS
        if item.required
    }
    store = _fake_r2(monkeypatch, canonical)

    generator = tmp_path / "generator"
    _stage(generator)
    pipeline.pull_generator(generator)
    for item in pipeline.GENERATED_INPUTS:
        if item.required:
            path = generator / item.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"generated:{item.name}".encode())

    bundle = pipeline.publish_generated(generator, "optional-123")
    assert {entry["name"] for entry in bundle["entries"]} == {
        item.name for item in pipeline.GENERATED_INPUTS if item.required
    }

    assembler = tmp_path / "assembler"
    _stage(assembler)
    provenance = pipeline.pull_assembler(assembler, "optional-123")
    names = {entry["name"] for entry in provenance["entries"]}
    assert names >= {item.name for item in pipeline.GENERATED_INPUTS if item.required}
    assert "ledger_nogate" not in names
    assert not (assembler / "data/backtest_trades_nogate.parquet").exists()


def test_pull_requires_data_empty_marked_github_workspace(tmp_path, monkeypatch):
    root = tmp_path / "stage"
    _stage(root)
    (root / "data").mkdir()
    (root / "data/local.json").write_text("local", encoding="utf-8")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("PRIVATE_SITE_CLOUD_BUILD", "1")
    _fake_r2(monkeypatch)
    with pytest.raises(RuntimeError, match="not data-empty"):
        pipeline.pull_generator(root)


def test_fundamentals_are_optional_at_the_r2_boundary():
    by_name = {item.name: item for item in pipeline.CANONICAL_INPUTS}
    assert by_name["fundamental_daily"].required is False
    assert by_name["fundamental_maps"].required is False


def test_publish_group_is_github_actions_only(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    with pytest.raises(RuntimeError, match="GitHub-Actions-only"):
        pipeline.publish_group(tmp_path, "reference")


def _write_publish_group(root: Path, group: str):
    for item in pipeline.PUBLISH_GROUPS[group]:
        path = root / item.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"local-primary:{item.name}".encode())


def test_local_primary_publish_requires_cli_opt_in_and_both_process_markers(tmp_path, monkeypatch):
    _write_publish_group(tmp_path, "reference")
    _fake_r2(monkeypatch)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv(pipeline.LOCAL_PRIMARY_ENV, "1")
    monkeypatch.setenv(pipeline.LOCAL_RUN_TOKEN_ENV, "run-123")

    # Environment state alone must never bypass the ordinary cloud-only path.
    with pytest.raises(RuntimeError, match="GitHub-Actions-only"):
        pipeline.publish_group(tmp_path, "reference")

    monkeypatch.delenv(pipeline.LOCAL_PRIMARY_ENV)
    with pytest.raises(RuntimeError, match=f"{pipeline.LOCAL_PRIMARY_ENV}=1"):
        pipeline.publish_group(tmp_path, "reference", local_primary=True)

    monkeypatch.setenv(pipeline.LOCAL_PRIMARY_ENV, "1")
    monkeypatch.setenv(pipeline.LOCAL_RUN_TOKEN_ENV, "   ")
    with pytest.raises(RuntimeError, match=pipeline.LOCAL_RUN_TOKEN_ENV):
        pipeline.publish_group(tmp_path, "reference", local_primary=True)


def test_local_primary_publish_verifies_every_uploaded_content_length(tmp_path, monkeypatch):
    group = "reference"
    _write_publish_group(tmp_path, group)
    store = _fake_r2(monkeypatch)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv(pipeline.LOCAL_PRIMARY_ENV, "1")
    monkeypatch.setenv(pipeline.LOCAL_RUN_TOKEN_ENV, "run-verified")

    entries = pipeline.publish_group(tmp_path, group, local_primary=True)

    assert {entry["key"] for entry in entries} == {item.key for item in pipeline.PUBLISH_GROUPS[group]}
    assert all(entry["size"] == len(store[entry["key"]]) for entry in entries)


def test_local_primary_publish_fails_closed_on_head_size_mismatch(tmp_path, monkeypatch):
    group = "reference"
    _write_publish_group(tmp_path, group)
    _fake_r2(monkeypatch)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv(pipeline.LOCAL_PRIMARY_ENV, "1")
    monkeypatch.setenv(pipeline.LOCAL_RUN_TOKEN_ENV, "run-mismatch")
    monkeypatch.setattr(
        pipeline.cache_io,
        "head",
        lambda key: {"ETag": '"etag"', "ContentLength": 1, "LastModified": "now"},
    )

    with pytest.raises(RuntimeError, match="ContentLength mismatch"):
        pipeline.publish_group(tmp_path, group, local_primary=True)


def test_cloud_stage_guards_are_not_relaxed_by_local_primary_markers(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.setenv(pipeline.LOCAL_PRIMARY_ENV, "1")
    monkeypatch.setenv(pipeline.LOCAL_RUN_TOKEN_ENV, "run-does-not-authorize-build")

    with pytest.raises(RuntimeError, match="GitHub-Actions-only"):
        pipeline.pull_generator(tmp_path)
    with pytest.raises(RuntimeError, match="GitHub-Actions-only"):
        pipeline.publish_generated(tmp_path, "123")
