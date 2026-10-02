import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def _node(script: str) -> None:
    result = subprocess.run([NODE, str(ROOT / "tests" / "js" / script)],
                            cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_nav_has_idea_check_right_after_pitch() -> None:
    common = (ROOT / "site" / "assets" / "common.js").read_text(encoding="utf-8")
    pitch = common.index('href: "pitch.html"')
    assert common.index('{ href: "idea.html",     label: "Idea Check" },') > pitch
    assert common.index('href: "idea.html"') < common.index('href: "focus.html"')
    assert (ROOT / "site" / "idea.html").exists()


def test_function_never_touches_the_broker() -> None:
    src = (ROOT / "functions" / "idea-check.js").read_text(encoding="utf-8")
    assert "_access.js" in src
    for banned in ("STATUS_TOKEN", "EXEC_BROKER_URL", "hmac"):
        assert banned not in src


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_idea_tab_javascript_contract() -> None:
    _node("test_idea_tab.js")


@pytest.mark.skipif(NODE is None, reason="Node.js is not installed")
def test_idea_check_function_contract() -> None:
    _node("test_idea_check_function.mjs")
