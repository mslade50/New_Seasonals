from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

MIGRATED = (
    "build_earnings_calendar.yml",
    "build_indicator_cache.yml",
    "build_macro_releases.yml",
    "daily_screener.yml",
    "discretionary_focus.yml",
    "execution_report.yml",
    "portfolio_report.yml",
    "risk_report.yml",
    "trend_sleeve.yml",
    "update_cboe_putcall.yml",
    "update_intraday_prices.yml",
    "update_master_prices.yml",
    "verify_fills.yml",
    "weekly_rundown.yml",
)


def _text(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def test_migrated_jobs_are_dispatch_only_correlated_backups() -> None:
    for name in MIGRATED:
        workflow = _text(name)
        assert "  schedule:" not in workflow, name
        assert "  workflow_dispatch:" in workflow, name
        assert "run-name:" in workflow, name
        assert "automation_token:" in workflow, name
        assert "inputs.automation_token" in workflow, name


def test_only_guarded_controller_retains_a_cron_for_migrated_jobs() -> None:
    fallback = _text("local_automation_fallback.yml")
    assert "  schedule:" in fallback
    assert "- cron: '47 * * * *'" in fallback
    assert "scripts/automation_supervisor.py fallback-due" in fallback
    assert "uses: ./.github/workflows/" not in fallback


def test_scanner_dispatch_has_explicit_bookend_and_side_effect_controls() -> None:
    workflow = _text("daily_screener.yml")
    for input_name in ("bookend:", "run_event_sleeve:", "deploy_after_scan:"):
        assert input_name in workflow
    assert "if: inputs.bookend == 'am' && inputs.run_event_sleeve" in workflow
    assert "if: inputs.bookend == 'am'" in workflow
    assert workflow.count("if: inputs.deploy_after_scan") == 2
    assert "uses: ./.github/workflows/deploy_site.yml" in workflow


def test_master_price_dispatch_requires_explicit_am_or_pm_mode() -> None:
    workflow = _text("update_master_prices.yml")
    assert "mode:" in workflow
    assert "- am" in workflow
    assert "- pm" in workflow
    assert 'if [ "${{ inputs.mode }}" = "pm" ]' in workflow
    assert "--exclude-today" in workflow


def test_event_sleeve_remote_recovery_is_dispatch_only() -> None:
    workflow = _text("event_sleeve.yml")
    assert "  schedule:" not in workflow
    assert "  workflow_dispatch:" in workflow
    assert "automation_token:" in workflow
    assert "python event_sleeve.py" in workflow
    assert "cancel-in-progress: false" in workflow
