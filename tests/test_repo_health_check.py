import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from scripts import repo_health_check as health

REPO = Path(__file__).resolve().parents[1]


def _marker(runtime: Path, config_root: Path) -> None:
    (runtime / ".local").mkdir(parents=True)
    (runtime / ".local" / "automation-runtime.json").write_text(
        json.dumps({"config_root": str(config_root), "fallback_ref": "x"}),
        encoding="utf-8",
    )


def test_dev_checkout_resolves_newest_versioned_runtime_with_a_marker(tmp_path):
    dev = tmp_path / "New_Seasonals"
    dev.mkdir()
    _marker(tmp_path / "New_Seasonals-automation-runtime", dev)
    _marker(tmp_path / "New_Seasonals-automation-runtime-v8", dev)
    _marker(tmp_path / "New_Seasonals-automation-runtime-v9", dev)
    (tmp_path / "New_Seasonals-automation-runtime-v11").mkdir()

    runtime, marker = health._resolve_automation_runtime(dev)
    assert runtime == tmp_path / "New_Seasonals-automation-runtime-v9"
    assert marker["config_root"] == str(dev)
    assert health._default_automation_state_root(runtime, marker) == (
        dev / "artifacts" / "automation"
    )

    _marker(tmp_path / "New_Seasonals-automation-runtime-v10", dev)
    runtime, _ = health._resolve_automation_runtime(dev)
    assert runtime == tmp_path / "New_Seasonals-automation-runtime-v10"


def test_runtime_checkout_resolves_itself_and_absent_runtime_falls_back(tmp_path):
    runtime = tmp_path / "New_Seasonals-automation-runtime-v9"
    _marker(runtime, tmp_path / "config")
    assert health._resolve_automation_runtime(runtime)[0] == runtime

    lone = tmp_path / "isolated" / "New_Seasonals"
    lone.mkdir(parents=True)
    assert health._resolve_automation_runtime(lone) == (None, {})
    assert health._default_automation_state_root(None, {}) == (
        health.ROOT / "artifacts" / "automation"
    )


def test_state_root_env_override_is_unchanged(tmp_path):
    env = {**os.environ, "NEW_SEASONALS_AUTOMATION_STATE_ROOT": str(tmp_path / "state")}
    result = subprocess.run(
        [sys.executable, "-c",
         "from scripts import repo_health_check as h; print(h.AUTOMATION_STATE_ROOT); "
         "print(h.AUTOMATION_LOG_DIR)"],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-2:] == [
        str(tmp_path / "state"), str(tmp_path / "state" / "logs")
    ]


def _capture(monkeypatch) -> list:
    reports = []
    monkeypatch.setattr(health, "report", lambda *args: reports.append(args))
    return reports


def test_dev_checkout_reads_breadth_from_the_runtime_data_dir(tmp_path, monkeypatch):
    dev, runtime = tmp_path / "dev", tmp_path / "runtime"
    monkeypatch.setattr(health, "ROOT", dev)
    monkeypatch.setattr(health, "AUTOMATION_RUNTIME", runtime)
    seen = []
    monkeypatch.setattr(health, "_last_index_date",
                        lambda p: seen.append(p) or health.dt.date(2026, 9, 30))
    monkeypatch.setattr(health, "_spy_last_session",
                        lambda p: seen.append(p) or health.dt.date(2026, 9, 30))
    reports = _capture(monkeypatch)

    health.check_local_data()

    assert seen and all(path.parent == runtime / "data" for path in seen)
    breadth = [r for r in reports if r[1] == "data:breadth-alignment"]
    assert breadth[0][0] == "OK"
    assert f"read from pinned runtime {runtime / 'data'}" in breadth[0][2]
    assert all("read from pinned runtime" in r[2] for r in reports)


def test_runtime_checkout_reads_its_own_data_without_a_note(tmp_path, monkeypatch):
    monkeypatch.setattr(health, "ROOT", tmp_path)
    monkeypatch.setattr(health, "AUTOMATION_RUNTIME", tmp_path)
    seen = []
    monkeypatch.setattr(health, "_last_index_date",
                        lambda p: seen.append(p) or health.dt.date(2026, 9, 30))
    monkeypatch.setattr(health, "_spy_last_session",
                        lambda p: seen.append(p) or health.dt.date(2026, 9, 30))
    reports = _capture(monkeypatch)

    health.check_breadth_alignment()

    assert all(path.parent == tmp_path / "data" for path in seen)
    assert reports[-1][0] == "OK"
    assert "pinned runtime" not in reports[-1][2]


def _collect_fixture(tmp_path, monkeypatch, stdout: str, returncode: int = 2) -> list:
    tests = tmp_path / "tests"
    tests.mkdir()
    for name in ("test_ok.py", "test_broken.py", "test_empty.py"):
        (tests / name).write_text("", encoding="utf-8")
    monkeypatch.setattr(health, "ROOT", tmp_path)
    monkeypatch.setattr(health.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=returncode, stdout=stdout, stderr=""))
    return _capture(monkeypatch)


def _error_block(name: str, message: str) -> str:
    return (
        f"_____ ERROR collecting tests/{name} _____\n"
        f"tests/{name}:3: in <module>\n"
        "    import thing\n"
        f"E   {message}\n"
    )


def test_collection_error_and_zero_test_guard_both_report(tmp_path, monkeypatch):
    stdout = (
        "tests/test_ok.py::test_one\n\n"
        "==================== ERRORS ====================\n"
        + _error_block("test_broken.py", "SyntaxError: invalid syntax")
        + "========== short test summary info ==========\n"
        "ERROR tests/test_broken.py\n"
    )
    reports = _collect_fixture(tmp_path, monkeypatch, stdout)

    health.check_test_collection()

    assert ("FAIL", "tests:collect",
            "1 file(s) fail to collect: test_broken.py (SyntaxError: invalid syntax)") in reports
    zero = [r for r in reports if r[0] == "WARN" and "ZERO" in r[2]]
    assert zero and zero[0][2].endswith("test_empty.py")
    assert not any(r[0] == "OK" for r in reports)


def test_missing_ib_insync_is_an_environment_warning_not_a_fail(tmp_path, monkeypatch):
    stdout = (
        "tests/test_ok.py::test_one\n"
        "tests/test_empty.py::test_two\n\n"
        "==================== ERRORS ====================\n"
        + _error_block("test_broken.py", "ModuleNotFoundError: No module named 'ib_insync'")
        + "========== short test summary info ==========\n"
    )
    reports = _collect_fixture(tmp_path, monkeypatch, stdout)

    health.check_test_collection()

    assert not any(r[0] == "FAIL" for r in reports)
    env = [r for r in reports if r[1] == "tests:collect-env"]
    assert env[0][0] == "WARN"
    assert "lacks ib_insync" in env[0][2] and "test_broken.py" in env[0][2]
    assert not any("ZERO" in r[2] for r in reports)


def test_other_missing_module_still_fails(tmp_path, monkeypatch):
    stdout = (
        "tests/test_ok.py::test_one\n"
        "tests/test_empty.py::test_two\n\n"
        "==================== ERRORS ====================\n"
        + _error_block("test_broken.py", "ModuleNotFoundError: No module named 'pandas'")
    )
    reports = _collect_fixture(tmp_path, monkeypatch, stdout)

    health.check_test_collection()

    assert [r[0] for r in reports] == ["FAIL"]
    assert "No module named 'pandas'" in reports[0][2]
