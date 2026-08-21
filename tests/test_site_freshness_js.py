import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
@pytest.mark.parametrize(
    "script",
    [
        "test_site_freshness_guards.js",
        "test_site_manifest.js",
    ],
)
def test_site_freshness_javascript_contract(script):
    result = subprocess.run(
        [shutil.which("node"), str(ROOT / "tests" / "js" / script)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
