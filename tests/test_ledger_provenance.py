"""Ledger provenance: the parquet metadata must name the commit it was built from.

Both production ledger vintages on 2026-09-04 carried `ledger_git_sha=unknown`
because the deploy builds inside `generator/`, a git-ls-files copy with no
.git directory, so `git rev-parse` failed there. GITHUB_SHA (a default
Actions env var) is now read first; the working tree's HEAD is the fallback;
'unknown' is the last resort.
"""
import os
import subprocess
import sys

import pyarrow.parquet as pq
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import build_trade_ledger as btl


def test_sha_comes_from_github_sha_when_present(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "0123456789abcdef0123456789abcdef01234567")
    # Even with git available the env var wins: it names the checked-out commit
    # regardless of whether the build directory is a git checkout.
    monkeypatch.setattr(btl.subprocess, "run",
                        lambda *a, **k: pytest.fail("git must not be consulted"))
    meta = btl._provenance_meta(5)
    assert meta["ledger_git_sha"] == "0123456789abcdef0123456789abcdef01234567"
    assert meta["ledger_rows"] == "5"


def test_sha_falls_back_to_git_head_without_the_env_var(monkeypatch):
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, 0, stdout="feedfacefeedfacefeedfacefeedfacefeedface\n", stderr="")

    monkeypatch.setattr(btl.subprocess, "run", fake_run)
    assert btl._git_sha() == "feedfacefeedfacefeedfacefeedfacefeedface"
    assert calls and calls[0][0][:2] == ["git", "rev-parse"]
    assert calls[0][1].get("timeout") == 10


def test_sha_fallback_matches_the_real_repo_head(monkeypatch):
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=btl._ROOT,
                              capture_output=True, text=True, timeout=10)
    except Exception:
        pytest.skip("git not available")
    if head.returncode != 0 or not head.stdout.strip():
        pytest.skip("not a git checkout")
    assert btl._git_sha() == head.stdout.strip()


def test_sha_is_unknown_when_neither_source_exists(monkeypatch):
    monkeypatch.delenv("GITHUB_SHA", raising=False)

    def no_git(cmd, **kw):
        raise FileNotFoundError("git")

    monkeypatch.setattr(btl.subprocess, "run", no_git)
    assert btl._git_sha() == "unknown"
    # A failing rev-parse (the generator/ case: a directory with no .git) is
    # the same outcome, not a stray error string.
    monkeypatch.setattr(btl.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 128, stdout="", stderr="fatal: not a git repository"))
    assert btl._git_sha() == "unknown"


def test_blank_env_var_is_not_a_sha(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "   ")
    monkeypatch.setattr(btl.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="abc123\n", stderr=""))
    assert btl._git_sha() == "abc123"


def test_metadata_round_trips_through_the_parquet_schema(tmp_path, monkeypatch):
    import pandas as pd
    monkeypatch.setenv("GITHUB_SHA", "cafebabe" * 5)
    path = str(tmp_path / "ledger.parquet")
    df = pd.DataFrame({"Ticker": ["SPY"], "R_Multiple": [1.0]})
    btl._write_ledger_with_meta(df, path, btl._provenance_meta(len(df)))
    meta = pq.read_schema(path).metadata
    assert meta[b"ledger_git_sha"].decode() == "cafebabe" * 5
    assert meta[b"ledger_rows"].decode() == "1"
    assert b"ledger_build_utc" in meta and b"ledger_source" in meta
