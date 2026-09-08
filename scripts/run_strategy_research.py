"""Scheduled research entry point: unique logs, serialized runs, verified completion."""
from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research_io import file_lock, write_json


def run(root: Path = ROOT, *, execute=subprocess.run) -> int:
    artifacts = root / "artifacts" / "strategy_research_agent"
    artifacts.mkdir(parents=True, exist_ok=True)
    # An old redirected child can retain last_run.log's Windows handle.
    # A new run must never depend on opening that shared file.
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    log_path = artifacts / f"{run_id}.log"
    marker = artifacts / f"{run_id}.marker"
    receipt_path = artifacts / f"{run_id}.json"
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
               CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS="0")
    claude = str(Path.home() / ".local" / "bin" / "claude.exe")
    ps = str(Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe")
    scripts = root / "scripts"
    phases = [
        ("source_collection", [sys.executable, str(scripts / "collect_strategy_sources.py"), "collect"], 1200),
        ("research_agent", [ps, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                            str(scripts / "invoke_strategy_research_agent.ps1"),
                            "-ClaudeExe", claude, "-Model", "opus", "-Effort", "xhigh",
                            "-TimeoutSeconds", "9000"], 9300),
        ("completion_check", [sys.executable, str(scripts / "check_strategy_research_run.py"),
                              "--marker", str(marker)], 120),
    ]
    with file_lock(artifacts / "runner"), log_path.open("x", encoding="utf-8") as log:
        marker.write_text(run_id, encoding="utf-8")
        receipt = {"run_id": run_id, "status": "running", "log": str(log_path),
                   "marker": str(marker), "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "phases": {}}

        def persist():
            write_json(receipt_path, receipt)
            write_json(artifacts / "last_run.json", receipt)

        persist()
        rc = 1
        for phase, command, timeout in phases:
            receipt["phase"] = phase
            persist()
            print(f"Starting {phase}", file=log, flush=True)
            try:
                result = execute(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
                rc = int(result.returncode)
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"{phase} failed: {exc}", file=log, flush=True)
                rc = 1
            receipt["phases"][phase] = rc
            if rc:
                # Alert failure cannot turn the original failure into success.
                try:
                    alert = execute([sys.executable, str(scripts / "send_strategy_research_failure_email.py"),
                                     "--phase", phase, "--summary", f"{phase} failed; inspect {log_path.name}.",
                                     "--send"], cwd=root, env=env, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=120, check=False)
                    receipt["alert_exit_code"] = alert.returncode
                except (OSError, subprocess.SubprocessError) as exc:
                    print(f"Failure alert failed: {exc}", file=log, flush=True)
                    receipt["alert_exit_code"] = 1
                break
        receipt.update(status="success" if rc == 0 else "failure", exit_code=rc,
                       finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        persist()
        print(f"Research {receipt['status']}; exit={rc}", file=log, flush=True)
    print(f"Research {receipt['status']}: {log_path}")
    return rc


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Research launcher failed before verified completion: {exc}", file=sys.stderr)
        raise SystemExit(1)
