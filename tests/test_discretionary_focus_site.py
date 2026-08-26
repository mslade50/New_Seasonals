"""Contract guards for the research-only Discretionary Focus site surface."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
FOCUS_HTML = SITE / "focus.html"
FOCUS_JS = SITE / "assets" / "focus.js"
FUNCTION = ROOT / "functions" / "discretionary-focus.js"


def test_focus_page_is_registered_and_research_only():
    common = (SITE / "assets" / "common.js").read_text(encoding="utf-8")
    assert '{ href: "focus.html",    label: "Focus" },' in common

    html = FOCUS_HTML.read_text(encoding="utf-8")
    assert 'data-page="focus"' in html
    assert 'assets/common.js' in html and 'assets/focus.js' in html
    assert "Research only" in html
    assert "cannot allocate capital" in html
    assert "<button" not in html.lower()
    assert "<form" not in html.lower()
    assert "<input" not in html.lower()


def test_focus_frontend_has_no_execution_controls():
    source = FOCUS_JS.read_text(encoding="utf-8")
    assert 'FOCUS_ENDPOINT = "/discretionary-focus"' in source
    assert 'FOCUS_SCHEMA = "discretionary-focus.v1"' in source
    assert "validateFocusPayload" in source
    assert "NO_QUALIFIED_SETUP" in source and "EXPIRED" in source and "UNAVAILABLE" in source
    assert "window.location.href =" not in source
    assert "/exec-" not in source
    assert "fetch(\"/fundamental-state\"" not in source


def test_focus_function_is_read_only_no_store_and_uses_exact_key():
    source = FUNCTION.read_text(encoding="utf-8")
    assert 'FOCUS_KEY = "discretionary_focus/current.json"' in source
    assert "onRequestGet" in source and "onRequestPost" not in source
    assert '"Cache-Control": "no-store"' in source
    assert ".put(" not in source and ".delete(" not in source
    assert "payload.focus.length > 2" in source
    assert "payload.research_only !== true" in source


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_focus_javascript_contract():
    result = subprocess.run(
        [shutil.which("node"), str(ROOT / "tests" / "js" / "test_discretionary_focus.js")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
