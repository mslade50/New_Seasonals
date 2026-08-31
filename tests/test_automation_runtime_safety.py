from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

from scripts import automation_supervisor as sup

ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts" / "automation_supervisor.py"
R2_ENV_NAMES = (
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
)


def _direct_script_env() -> dict[str, str]:
    """Return a deterministic child environment with no import-path crutch."""

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for name in R2_ENV_NAMES:
        env.pop(name, None)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def _run_supervisor_directly(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SUPERVISOR), *args],
        cwd=cwd,
        env=_direct_script_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=30,
        check=False,
    )


def test_direct_script_r2_download_imports_repo_module_without_pythonpath(tmp_path):
    target = tmp_path / "downloaded.bin"

    result = _run_supervisor_directly(
        "_r2-download",
        "regression/missing.bin",
        str(target),
        "--required",
        cwd=tmp_path,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "required R2 input unavailable: regression/missing.bin" in output
    assert "ModuleNotFoundError" not in output
    assert "No module named 'cache_io'" not in output
    assert not target.exists()


def test_direct_script_pull_intraday_imports_repo_module_without_pythonpath(tmp_path):
    result = _run_supervisor_directly("_pull-intraday", cwd=tmp_path)

    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "canonical intraday metadata is unavailable" in output
    assert "ModuleNotFoundError" not in output
    assert "No module named 'cache_io'" not in output


def test_direct_script_discretionary_gate_imports_scripts_package_without_pythonpath(
    tmp_path,
):
    probe = textwrap.dedent(
        """
        import datetime as dt
        import runpy
        import sys
        from types import SimpleNamespace

        namespace = runpy.run_path(sys.argv[1], run_name="automation_supervisor_probe")
        catalog = namespace["CATALOG"]
        job = next(
            item
            for item in catalog["discretionary"].jobs
            if item.local_gate == "discretionary_delivery_window"
        )
        fake = SimpleNamespace(
            now=lambda: dt.datetime(2026, 8, 31, 12, 35, tzinfo=dt.timezone.utc)
        )
        allowed, detail = namespace["AutomationSupervisor"]._local_gate(fake, job)
        print(f"allowed={allowed} detail={detail}")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", probe, str(SUPERVISOR)],
        cwd=tmp_path,
        env=_direct_script_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=30,
        check=False,
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "allowed=True detail=NYSE delivery gate for 2026-08-31" in output
    assert "ModuleNotFoundError" not in output
    assert "No module named 'scripts'" not in output


def test_hydrated_child_environment_forces_utf8_without_mutating_parent(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / ".env").write_text(
        "R2_ACCOUNT_ID=acct\n"
        "R2_ACCESS_KEY_ID=key\n"
        "R2_SECRET_ACCESS_KEY=secret\n"
        "R2_BUCKET=bucket\n"
        "PYTHONIOENCODING=cp1252\n"
        "PYTHONUTF8=0\n",
        encoding="utf-8",
    )
    exec_env = tmp_path / "exec_agent.env"
    exec_env.write_text("STATUS_TOKEN=status-secret\n", encoding="utf-8")
    gcp = tmp_path / "credentials.json"
    gcp.write_text(json.dumps({"type": "service_account"}), encoding="utf-8")
    base = {
        "UNCHANGED": "yes",
        "PYTHONIOENCODING": "latin-1",
        "PYTHONUTF8": "0",
    }

    child = sup.hydrate_environment(
        config_root=config,
        gcp_json_path=gcp,
        exec_env_path=exec_env,
        base_env=base,
    )

    assert child["PYTHONIOENCODING"] == "utf-8"
    assert child["PYTHONUTF8"] == "1"
    assert base == {
        "UNCHANGED": "yes",
        "PYTHONIOENCODING": "latin-1",
        "PYTHONUTF8": "0",
    }


class _RecordingProcess:
    def __init__(self):
        self.calls: list[dict[str, object]] = []

    def stream(self, argv, *, cwd, env, timeout_seconds, logger):
        self.calls.append(
            {
                "argv": list(argv),
                "cwd": cwd,
                "env": dict(env),
                "timeout_seconds": timeout_seconds,
            }
        )
        return 0


def test_health_child_receives_the_selected_runtime_state_root(tmp_path, monkeypatch):
    runtime = tmp_path / "New_Seasonals-automation-runtime-v3"
    config = tmp_path / "config"
    state_root = runtime / "artifacts" / "automation"
    runtime.mkdir()
    config.mkdir()
    process = _RecordingProcess()
    supervisor = SimpleNamespace(
        process=process,
        env={"PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
    )
    monkeypatch.setattr(
        sup,
        "_make_runtime",
        lambda args, controller_only=False: (supervisor, SimpleNamespace()),
    )

    result = sup.main(
        [
            "health",
            "--config-root",
            str(config),
            "--repo-root",
            str(runtime),
            "--state-root",
            str(state_root),
            "--skip-tests",
            "--skip-automation",
        ]
    )

    assert result == 0
    assert len(process.calls) == 1
    child_env = process.calls[0]["env"]
    assert isinstance(child_env, dict)
    assert child_env["NEW_SEASONALS_AUTOMATION_STATE_ROOT"] == str(state_root.resolve())
    assert "NEW_SEASONALS_AUTOMATION_STATE_ROOT" not in supervisor.env


def test_version_suffixed_pinned_runtime_defaults_health_to_its_own_state_tree(
    tmp_path,
):
    runtime = tmp_path / "New_Seasonals-automation-runtime-v3"
    scripts = runtime / "scripts"
    marker_dir = runtime / ".local"
    scripts.mkdir(parents=True)
    marker_dir.mkdir()
    shutil.copyfile(
        ROOT / "scripts" / "repo_health_check.py", scripts / "repo_health_check.py"
    )
    (runtime / "cache_io.py").write_text(
        "# import stub for default-path probe\n", encoding="utf-8"
    )
    (marker_dir / "automation-runtime.json").write_text("{}\n", encoding="utf-8")
    probe = textwrap.dedent(
        """
        import runpy
        import sys

        namespace = runpy.run_path(sys.argv[1], run_name="repo_health_path_probe")
        print(namespace["AUTOMATION_STATE_ROOT"])
        """
    )
    env = _direct_script_env()
    env.pop("NEW_SEASONALS_AUTOMATION_STATE_ROOT", None)

    result = subprocess.run(
        [sys.executable, "-c", probe, str(scripts / "repo_health_check.py")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert Path(result.stdout.strip()) == runtime / "artifacts" / "automation"


def test_task_runner_sets_and_restores_utf8_and_runtime_state_environment():
    source = (ROOT / "scripts" / "run_local_automation.ps1").read_text(encoding="utf-8")
    try_index = source.index("try {")
    finally_index = source.index("finally {")
    contracts = (
        (
            "PYTHONIOENCODING",
            "$previousPythonIoEncoding",
            "$env:PYTHONIOENCODING = 'utf-8'",
        ),
        ("PYTHONUTF8", "$previousPythonUtf8", "$env:PYTHONUTF8 = '1'"),
        (
            "NEW_SEASONALS_AUTOMATION_STATE_ROOT",
            "$previousAutomationStateRoot",
            (
                "$env:NEW_SEASONALS_AUTOMATION_STATE_ROOT = "
                "(Join-Path $RuntimeRoot 'artifacts\\automation')"
            ),
        ),
    )

    for env_name, previous_variable, assignment in contracts:
        save = (
            f"{previous_variable} = [Environment]::GetEnvironmentVariable("
            f"'{env_name}', 'Process')"
        )
        restore = (
            f"Restore-ProcessEnvironment -Name '{env_name}' -Value {previous_variable}"
        )
        assert source.index(save) < try_index
        assert try_index < source.index(assignment) < finally_index
        assert finally_index < source.index(restore)
