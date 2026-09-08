from pathlib import Path
import re

from scripts import automation_supervisor as supervisor


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

MIGRATED = (
    "build_earnings_calendar.yml",
    "build_indicator_cache.yml",
    "build_macro_releases.yml",
    "daily_screener.yml",
    "discretionary_focus.yml",
    "event_sleeve.yml",
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


def _dispatch_inputs(name: str) -> set[str]:
    match = re.search(
        r"(?ms)^  workflow_dispatch:\s*\n(.*?)(?=^\S|\Z)",
        _text(name),
    )
    assert match, name
    return set(re.findall(r"(?m)^      ([A-Za-z0-9_]+):\s*$", match.group(1)))


def test_migrated_jobs_are_dispatch_only_correlated_backups() -> None:
    for name in MIGRATED:
        workflow = _text(name)
        assert "  schedule:" not in workflow, name
        assert "  workflow_dispatch:" in workflow, name
        assert "run-name:" in workflow, name
        assert "automation_token:" in workflow, name
        assert "inputs.automation_token" in workflow, name


def test_every_local_job_backup_enables_fail_closed_producer_mode() -> None:
    names = {
        job.workflow.workflow
        for pipeline in supervisor.CATALOG.values()
        for job in pipeline.jobs
        if job.workflow and not job.dispatch_only
    }
    assert names
    for name in names:
        workflow = _text(name).replace("\r\n", "\n")
        assert "\nenv:\n  LOCAL_AUTOMATION_STRICT: '1'\n" in "\n" + workflow, name


def test_only_guarded_controller_retains_a_cron_for_migrated_jobs() -> None:
    fallback = _text("local_automation_fallback.yml")
    assert "  schedule:" in fallback
    assert "- cron: '47 * * * *'" in fallback
    # 2026-09-04 (incident 2026-09-03): ~20-minute spacing across the premarket
    # fallback window 05:20-08:55 ET in both DST regimes (09-13 UTC covers
    # EDT 09:20-12:55 and EST 10:20-13:55), weekdays only.
    assert "- cron: '7 9-13 * * 1-5'" in fallback
    assert "- cron: '27 9-13 * * 1-5'" in fallback
    # 2026-09-04 round 2: the six off-window ticks and, by name, the EDT
    # 13:07Z tick that lands at 09:07 ET inside the discretionary window under
    # the 'general' concurrency group, are documented where the crons live.
    assert "13:07Z tick lands at 09:07 ET" in fallback
    assert "08:50-09:20 ET" in fallback
    assert "'general' concurrency group" in fallback
    for off_window in ("EDT 05:07", "09:27", "EST 04:07", "04:27"):
        assert off_window in fallback
    assert "- cron: '50 12,13 * * 1-5'" in fallback
    assert (
        'scripts/automation_supervisor.py fallback-due --pipeline "$PIPELINE" '
        '--ref "$AUTOMATION_RUNTIME_REF"'
    ) in fallback
    assert "AUTOMATION_RUNTIME_REF: automation-runtime-2026-09-03.1" in fallback
    assert "ref: ${{ env.AUTOMATION_RUNTIME_REF }}" in fallback
    assert "&& 'discretionary' || inputs.pipeline || 'all'" in fallback
    assert "&& 'discretionary' || 'general'" in fallback
    assert "uses: ./.github/workflows/" not in fallback


def test_scanner_dispatch_has_explicit_bookend_and_side_effect_controls() -> None:
    workflow = _text("daily_screener.yml")
    for input_name in ("bookend:", "run_event_sleeve:", "deploy_after_scan:"):
        assert input_name in workflow
    assert "if: inputs.bookend == 'am' && inputs.run_event_sleeve" in workflow
    assert "if: inputs.bookend == 'am'" in workflow
    assert workflow.count("if: inputs.deploy_after_scan") == 2
    assert 'daily_scan.py --scope=all --bookend="${{ inputs.bookend }}"' in workflow
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


def test_cloud_deploy_backups_are_correlated_without_local_builds() -> None:
    for name in ("deploy_site.yml", "deploy_shared_seasonals.yml"):
        workflow = _text(name)
        assert "automation_token:" in workflow
        assert "inputs.automation_token" in workflow
    assert "build_site.py --production" in _text("deploy_site.yml")


def test_pinned_tag_fallback_token_can_deliver_discretionary_focus() -> None:
    workflow = _text("discretionary_focus.yml")
    condition = "(github.ref == 'refs/heads/main' || inputs.automation_token != '')"
    assert workflow.count(condition) == 3


def test_every_supervisor_backup_input_is_declared_by_target_workflow() -> None:
    for pipeline in supervisor.CATALOG.values():
        for job in pipeline.jobs:
            if not job.workflow:
                continue
            declared = _dispatch_inputs(job.workflow.workflow)
            requested = {"automation_token", *job.workflow.input_dict()}
            assert requested <= declared, (
                job.id,
                job.workflow.workflow,
                requested - declared,
            )
