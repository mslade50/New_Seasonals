"""Run the Execution-tab fast-action browser contract in the Python suite."""

import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _run_js(name):
    result = subprocess.run(
        [shutil.which("node"), str(ROOT / "tests" / "js" / name)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_execution_fast_actions_javascript_contract():
    _run_js("test_execution_fast_actions.js")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_execution_navigation_preserves_edits_without_commands():
    _run_js("test_execution_navigation.js")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_execution_close_types_javascript_contract():
    """close_only / close_resize / flatten tickets and the safe trim routing."""
    _run_js("test_execution_close_types.js")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_execution_dashboard_control_matrix():
    _run_js("test_execution_dashboard_matrix.js")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_tradelog_contract_units():
    _run_js("test_tradelog_units.js")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_execution_manual_orders_javascript_contract():
    _run_js("test_execution_manual_orders.js")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_execution_reconcile_button_contract():
    _run_js("test_execution_reconcile.js")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_execution_lock_resolve_and_risk_prompt_contract():
    """Structured lock rejection, the clear-lock note gate, the
    position_action_resolve payload, the snapshot summary, and the RISK_ACK
    confirmation wording (audit findings C2 + the RISK_ACK cause row)."""
    _run_js("test_execution_lock_resolve.js")


def test_trim_control_is_retired_from_the_site_vocabulary():
    """Finding A2: every layer rejected `trim_readd`, so the control is gone.

    Guarded here as well as in Node so the retirement holds even where Node is
    unavailable (CI skips the JS contracts).
    """
    js = (ROOT / "site" / "assets" / "execution.js").read_text(encoding="utf-8")
    assert "trimReaddPayload" not in js
    assert "function execTrim" not in js
    assert "window.execTrim" not in js
    assert 'sendCommand("trim_readd"' not in js
    assert '"trim_readd"' not in js
    assert "execTrim(" not in js
    assert '"position_action_resolve"' in js
    schema = (ROOT / "docs" / "site_execution_schema.md").read_text(encoding="utf-8")
    assert "position_action_resolve" in schema
    assert "RETIRED 2026-09-21" in schema
