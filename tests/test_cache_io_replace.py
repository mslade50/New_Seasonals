import os
from pathlib import Path

import pytest

_REAL_REPLACE = os.replace

import cache_io


class FakeClient:
    def __init__(self, payload: bytes = b"new", error: Exception | None = None):
        self.payload = payload
        self.error = error
        self.targets: list[str] = []

    def download_file(self, bucket: str, key: str, target: str) -> None:
        self.targets.append(target)
        # Mimic s3transfer: write an intermediate "<target>.<random>" then rename.
        partial = Path(f"{target}.AbCd1234")
        partial.write_bytes(self.payload)
        if self.error:
            raise self.error
        _REAL_REPLACE(partial, target)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    target = tmp_path / "master_prices.parquet"
    target.write_bytes(b"old")
    sleeps: list[float] = []
    monkeypatch.setattr(cache_io.time, "sleep", sleeps.append)
    monkeypatch.setattr(cache_io, "_r2_creds", lambda: {"R2_BUCKET": "bucket"})

    def install(client: FakeClient) -> None:
        monkeypatch.setattr(cache_io, "_client", lambda: client)

    return target, sleeps, install


def _leftovers(target: Path) -> list[str]:
    return sorted(p.name for p in target.parent.iterdir() if p != target)


def _flaky_replace(monkeypatch, target: Path, failures: int | None):
    real_replace = os.replace
    calls = {"n": 0}

    def fake_replace(src, dst):
        if Path(dst) == target:
            calls["n"] += 1
            if failures is None or calls["n"] <= failures:
                err = PermissionError(13, "The process cannot access the file")
                err.winerror = 32
                raise err
        return real_replace(src, dst)

    monkeypatch.setattr(cache_io.os, "replace", fake_replace)
    return calls


def test_downloads_via_temp_and_replaces_target(setup):
    target, sleeps, install = setup
    client = FakeClient()
    install(client)

    assert cache_io.download_to_local("master_prices.parquet", str(target)) is True
    assert target.read_bytes() == b"new"
    assert client.targets[0] != str(target)
    assert _leftovers(target) == []
    assert sleeps == []


def test_transient_lock_retries_then_succeeds(setup, monkeypatch):
    target, sleeps, install = setup
    install(FakeClient())
    calls = _flaky_replace(monkeypatch, target, failures=3)

    assert cache_io.download_to_local("master_prices.parquet", str(target)) is True
    assert target.read_bytes() == b"new"
    assert calls["n"] == 4
    assert len(sleeps) == 3
    assert _leftovers(target) == []
    assert cache_io.last_download_error() is None


def test_permanent_lock_preserves_target_and_cleans_temp(setup, monkeypatch):
    target, sleeps, install = setup
    install(FakeClient())
    calls = _flaky_replace(monkeypatch, target, failures=None)

    assert cache_io.download_to_local("master_prices.parquet", str(target)) is False
    assert target.read_bytes() == b"old"
    assert calls["n"] == cache_io._REPLACE_ATTEMPTS
    assert len(sleeps) == cache_io._REPLACE_ATTEMPTS - 1
    assert 10 <= sum(sleeps) <= 25
    assert _leftovers(target) == []
    assert "PermissionError" in (cache_io.last_download_error() or "")


def test_non_lock_error_is_not_retried(setup, monkeypatch):
    target, sleeps, install = setup
    install(FakeClient())

    def boom(src, dst):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(cache_io.os, "replace", boom)

    assert cache_io.download_to_local("master_prices.parquet", str(target)) is False
    assert target.read_bytes() == b"old"
    assert sleeps == []
    assert _leftovers(target) == []


def test_download_error_cleans_partial_and_keeps_target(setup):
    target, sleeps, install = setup
    install(FakeClient(error=ConnectionError("reset by peer")))

    assert cache_io.download_to_local("master_prices.parquet", str(target)) is False
    assert target.read_bytes() == b"old"
    assert _leftovers(target) == []
    assert "ConnectionError" in (cache_io.last_download_error() or "")


def test_missing_target_gets_created(tmp_path, monkeypatch):
    target = tmp_path / "sub" / "earnings_calendar.parquet"
    monkeypatch.setattr(cache_io, "_r2_creds", lambda: {"R2_BUCKET": "bucket"})
    monkeypatch.setattr(cache_io, "_client", lambda: FakeClient(b"fresh"))

    assert cache_io.download_to_local("earnings_calendar.parquet", str(target)) is True
    assert target.read_bytes() == b"fresh"
    assert _leftovers(target) == []
