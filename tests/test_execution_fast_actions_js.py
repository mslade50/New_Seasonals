"""Run the Execution-tab fast-action browser contract in the Python suite."""

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_execution_fast_actions_javascript_contract():
    result = subprocess.run(
        [shutil.which("node"), str(ROOT / "tests" / "js" / "test_execution_fast_actions.js")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
