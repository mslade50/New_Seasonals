from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (ROOT / "scripts" / "run_local_automation.ps1").read_text(encoding="utf-8")
INSTALLER = (ROOT / "scripts" / "install_local_automation_tasks.ps1").read_text(encoding="utf-8")


def test_runner_calls_only_the_pinned_supervisor_contract():
    assert "automation_supervisor.py" in RUNNER
    assert "run-pipeline --pipeline $Pipeline --config-root $ConfigRoot --ref" in RUNNER
    assert "health --config-root $ConfigRoot --ref" in RUNNER
    assert "fallback_ref" in RUNNER
    assert "LOCAL_AUTOMATION_PRIMARY" in RUNNER
    assert "LOCAL_AUTOMATION_RUN_TOKEN" in RUNNER
    assert "[Guid]::NewGuid()" in RUNNER
    assert "GH_PAT_NEW_SEASONALS" in RUNNER
    assert "GetEnvironmentVariable('GH_PAT_NEW_SEASONALS', 'User')" in RUNNER
    assert "git_executable" in RUNNER
    assert "rev-parse HEAD" in RUNNER
    assert "status --porcelain --untracked-files=no" in RUNNER


def test_scheduled_runner_never_mutates_or_updates_code():
    lowered = RUNNER.lower()
    for forbidden in (" fetch ", " pull ", " merge ", " reset ", " checkout ", " switch "):
        assert forbidden not in lowered


def test_installer_defines_the_required_local_clock_schedule():
    for pipeline, at, mask in (
        ("premarket", "04:10:00", "62"),
        ("discretionary", "08:35:00", "62"),
        ("execution", "16:30:00", "62"),
        ("postclose", "17:10:00", "62"),
        ("indicator", "03:00:00", "2"),
        ("weekly-rundown", "08:00:00", "1"),
        ("health", "07:30:00", "62"),
    ):
        assert f"Id = '{pipeline}'" in INSTALLER
        assert f"Time = '{at}'" in INSTALLER
        assert f"DaysMask = {mask}" in INSTALLER
    assert "Eastern Standard Time" in INSTALLER


def test_task_settings_are_resilient_and_ignore_overlaps():
    assert "$definition.Settings.Enabled = $false" in INSTALLER
    assert "$definition.Settings.WakeToRun = $true" in INSTALLER
    assert "$definition.Settings.StartWhenAvailable = $true" in INSTALLER
    assert "$definition.Settings.RestartCount = 3" in INSTALLER
    assert "$definition.Settings.RestartInterval = 'PT5M'" in INSTALLER
    assert "$definition.Settings.MultipleInstances = 2" in INSTALLER
    assert "$definition.Principal.LogonType = 2" in INSTALLER


def test_cutover_is_explicit_and_preserves_task_objects():
    assert "-ConfirmCutover" in INSTALLER
    assert "Trigger CBOE Put-Call (GHA workflow_dispatch)" in INSTALLER
    assert "Trigger Update Master Prices (GHA workflow_dispatch)" in INSTALLER
    assert "Trigger Risk Report AM Correction (GHA workflow_dispatch)" in INSTALLER
    assert "Trigger Daily Screener (GHA workflow_dispatch)" in INSTALLER
    assert "Repo Health Check" in INSTALLER
    assert "FallbackRef" in INSTALLER
    assert "Unregister-ScheduledTask" not in INSTALLER
    assert "DeleteTask" not in INSTALLER
    assert "TASK_CREATE (not" in INSTALLER
    assert "task enabled states were rolled back" in INSTALLER


def test_automation_requirements_are_explicit():
    requirements = (ROOT / "scripts" / "requirements-automation.txt").read_text(encoding="utf-8")
    assert "-r ../requirements.txt" in requirements
    assert "exchange-calendars" in requirements
    assert "google-auth" in requirements
