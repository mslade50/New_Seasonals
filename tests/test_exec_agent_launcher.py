"""Exercise Windows PowerShell with disposable, broker-free Python agents.

Artifacts are deliberately retained; no test deletes production or test files.
"""
import os
from pathlib import Path
import subprocess
import sys
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_exec_agent.ps1"
POWERSHELL = Path(os.environ.get("SystemRoot", "C:/Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"


@unittest.skipUnless(POWERSHELL.exists(), "Native Windows PowerShell required")
class LauncherTests(unittest.TestCase):
    def run_agent(self, source, *, window="0\nAGENT_END_HOUR=24", env_present=True):
        runtime = ROOT / "artifacts" / "exec-agent-launcher-tests" / uuid.uuid4().hex
        runtime.mkdir(parents=True)
        (runtime / "exec_agent.py").write_text(source, encoding="utf-8")
        if env_present:
            (runtime / "exec_agent.env").write_text("AGENT_START_HOUR=" + window + "\n", encoding="utf-8")
        result = subprocess.run(
            [str(POWERSHELL), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(LAUNCHER),
             "-RuntimeDirectory", str(runtime), "-PythonPath", sys.executable],
            capture_output=True, text=True, timeout=20,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        log_path = runtime / "exec_agent_last_run.log"
        log = log_path.read_text(encoding="utf-8-sig") if log_path.exists() else ""
        return result, log

    def test_abrupt_native_exit_is_failure_with_code_logged(self):
        result, log = self.run_agent("import os\nos._exit(23)\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("python_code=23 unexpected=True", log)

    def test_clean_exit_during_window_is_unexpected_failure(self):
        result, log = self.run_agent("print('fake agent complete')\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("fake agent complete", log)
        self.assertIn("python_code=0 unexpected=True", log)

    def test_stderr_does_not_interrupt_wait_for_child(self):
        result, log = self.run_agent("import sys\nprint('diagnostic', file=sys.stderr, flush=True)\nprint('finished')\nsys.exit(7)\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("diagnostic", log)
        self.assertIn("finished", log)
        self.assertIn("python_code=7", log)

    def test_outside_window_does_not_launch(self):
        result, log = self.run_agent("raise RuntimeError('MUST NOT LAUNCH')\n", window="0\nAGENT_END_HOUR=0")
        self.assertEqual(result.returncode, 0)
        self.assertIn("outside run window; skipped", log)
        self.assertNotIn("MUST NOT LAUNCH", log)

    def test_missing_environment_fails(self):
        result, log = self.run_agent("print('MUST NOT LAUNCH')\n", env_present=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("launcher failure", log)
        self.assertNotIn("MUST NOT LAUNCH", log)


if __name__ == "__main__":
    unittest.main()
