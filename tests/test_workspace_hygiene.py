from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts import workspace_hygiene as hygiene


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def clean_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    _git(tmp_path, "init", "-b", "main")
    _git(tmp_path, "config", "user.name", "Workspace Hygiene Test")
    _git(tmp_path, "config", "user.email", "workspace-hygiene@example.invalid")
    (tmp_path / ".gitignore").write_text(".local/\nartifacts/\n", encoding="utf-8")
    (tmp_path / "tracked.txt").write_text("original\n", encoding="utf-8")
    _git(tmp_path, "add", ".gitignore", "tracked.txt")
    _git(tmp_path, "commit", "-m", "fixture")
    monkeypatch.setattr(hygiene, "ROOT", tmp_path)
    monkeypatch.setattr(hygiene, "ARTIFACT_ROOT", tmp_path / "artifacts")
    return tmp_path


def test_check_allows_only_declared_tracked_change(clean_repo: Path) -> None:
    baseline = clean_repo / ".local" / "baseline.json"
    assert hygiene.write_baseline(baseline, force=False) == 0

    (clean_repo / "tracked.txt").write_text("changed\n", encoding="utf-8")

    assert hygiene.check_baseline(baseline, allowed_values=[]) == 1
    assert hygiene.check_baseline(baseline, allowed_values=["tracked.txt"]) == 0


def test_check_detects_new_untracked_file(clean_repo: Path) -> None:
    baseline = clean_repo / ".local" / "baseline.json"
    assert hygiene.write_baseline(baseline, force=False) == 0

    (clean_repo / "surprise.log").write_text("not in an ignored area\n", encoding="utf-8")

    assert hygiene.check_baseline(baseline, allowed_values=[]) == 1


def test_artifact_dir_is_ignored_and_confined(clean_repo: Path) -> None:
    assert hygiene.prepare_artifact_dir("browser/execution") == 0
    assert (clean_repo / "artifacts" / "browser" / "execution").is_dir()
    assert hygiene.prepare_artifact_dir("../outside") == 2


def test_existing_baseline_requires_force(clean_repo: Path) -> None:
    baseline = clean_repo / ".local" / "baseline.json"
    assert hygiene.write_baseline(baseline, force=False) == 0
    assert hygiene.write_baseline(baseline, force=False) == 2
    assert hygiene.write_baseline(baseline, force=True) == 0
