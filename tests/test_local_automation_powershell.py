import json
import re
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = (ROOT / "scripts" / "run_local_automation.ps1").read_text(encoding="utf-8")
INSTALLER = (ROOT / "scripts" / "install_local_automation_tasks.ps1").read_text(encoding="utf-8")
POWERSHELL = Path(r"C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe")
REQUIRES_WINDOWS_POWERSHELL = pytest.mark.skipif(
    not POWERSHELL.is_file(), reason="requires Windows PowerShell"
)


def _extract_function(name: str) -> str:
    match = re.search(
        rf"^function {re.escape(name)} \{{.*?^\}}",
        INSTALLER,
        flags=re.DOTALL | re.MULTILINE,
    )
    assert match is not None, f"PowerShell function not found: {name}"
    return match.group(0)


def _ps_quote(value: Path | str) -> str:
    return str(value).replace("'", "''")


def _run_powershell(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(POWERSHELL),
            "-NoLogo",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            *args,
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_runner_calls_only_the_pinned_supervisor_contract():
    assert "automation_supervisor.py" in RUNNER
    assert "run-pipeline --pipeline $Pipeline --config-root $ConfigRoot --ref" in RUNNER
    assert "health --config-root $ConfigRoot --ref" in RUNNER
    assert (
        "fallback-due --pipeline premarket --job scan_am --config-root $ConfigRoot --ref"
        in RUNNER
    )
    assert "fallback_ref" in RUNNER
    # 2026-09-04: local second chance + stable state root outside the worktree.
    assert "'premarket-retry'" in RUNNER
    assert (
        "run-pipeline --pipeline premarket --retry --config-root $ConfigRoot --ref"
        in RUNNER
    )
    assert "$stateRoot = (Join-Path $ConfigRoot 'artifacts\\automation')" in RUNNER
    assert RUNNER.count("--state-root $stateRoot") == 4
    # 2026-09-04 round 2: a hung primary holding the supervisor lock is exactly
    # when the battery matters, so the recovery probe can never abort the task
    # before it (verify finding 1).
    health_branch = RUNNER.split("if ($Pipeline -eq 'health') {", 1)[1].split(
        "elseif ($Pipeline -eq 'premarket-retry')", 1
    )[0]
    assert "try {" in health_branch
    assert "catch {" in health_branch
    assert health_branch.index("catch {") < health_branch.index(
        "health --config-root $ConfigRoot"
    )
    assert "$recoveryExitCode = 1" in health_branch
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
        ("premarket-retry", "05:45:00", "62"),
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
    # Deletion exists only behind the explicit -PruneSuperseded switch, inside
    # Invoke-Prune (2026-09-04). Nothing else in the installer may delete.
    outside_prune = INSTALLER.replace(_extract_function("Invoke-Prune"), "")
    assert "DeleteTask" not in outside_prune
    assert "DeleteTask" in _extract_function("Invoke-Prune")
    assert "TASK_CREATE (not" in INSTALLER
    assert "task enabled states were rolled back" in INSTALLER


def test_prune_switch_is_default_off_and_whatif_lists_only():
    assert "[switch]$PruneSuperseded" in INSTALLER
    assert "$PruneSuperseded = $true" not in INSTALLER
    assert "SupportsShouldProcess = $true" in INSTALLER
    prune = _extract_function("Invoke-Prune")
    # The refusal has to name the exact listing command, switch included
    # (2026-09-04 round 2): -WhatIf alone still throws.
    assert (
        "Prune requires the explicit -PruneSuperseded switch "
        "(add -PruneSuperseded -WhatIf to list without deleting)"
    ) in prune
    assert "WhatIf: would unregister disabled task" in prune
    candidates = _extract_function("Get-PruneCandidates")
    assert "StartsWith($TaskNamePrefix" in candidates
    assert "keep: rollback generation" in candidates
    assert "keep: ENABLED" in candidates
    cutover = _extract_function("Invoke-Cutover")
    assert "Copy-RetiredRuntimeLogs" in cutover
    assert cutover.index("Copy-RetiredRuntimeLogs") < cutover.index("$entry.task.Enabled = $true")
    assert "if ($PruneSuperseded)" in cutover
    # 2026-09-04 round 2: the incoming generation's cutover-state.json is
    # written BEFORE the copy-forward, so the copy mirrors it at its own
    # cutover instead of at the next one.
    assert cutover.index("'cutover-state.json'") < cutover.index(
        "Copy-RetiredRuntimeLogs -RootFolder"
    )
    # Per-source counters: a running total would misreport the second source.
    copy_logs = _extract_function("Copy-RetiredRuntimeLogs")
    assert copy_logs.index("foreach ($root in $sources)") < copy_logs.index("$copied = 0")
    assert copy_logs.index("$copied = 0") < copy_logs.index("$sourceLogs = Join-Path")
    # 2026-09-04 round 3: the cutover-state mirror runs BEFORE the missing-logs
    # guard, or the freshly prepared incoming generation (no logs directory at
    # all) is never mirrored at its own cutover.
    assert copy_logs.index("cutover-state.json") < copy_logs.index(
        "no runtime logs under"
    )
    assert copy_logs.index("cutover-state.json") < copy_logs.index(
        "Get-ChildItem -LiteralPath $sourceLogs"
    )
    # A retired task whose action carries no quoted -RuntimeRoot is announced,
    # never dropped from the source list in silence.
    assert "carries no quoted -RuntimeRoot in its action" in copy_logs
    assert copy_logs.count("Write-Output") >= 3


def test_cutover_accepts_a_distinct_prior_local_prefix_and_keeps_fixed_legacy_tasks():
    assert "[string]$RetireTaskNamePrefix" in INSTALLER
    assert "$RetireTaskNamePrefix + $spec.Id" in INSTALLER
    assert "foreach ($name in (Get-SupersededTaskNames))" in INSTALLER
    assert "Trigger Daily Screener (GHA workflow_dispatch)" in INSTALLER
    assert "Repo Health Check" in INSTALLER


@REQUIRES_WINDOWS_POWERSHELL
def test_same_new_and_retired_prefix_is_rejected_before_scheduler_access():
    result = _run_powershell(
        ROOT / "scripts" / "install_local_automation_tasks.ps1",
        "-Phase",
        "Status",
        "-SourceRepository",
        str(ROOT),
        "-TaskNamePrefix",
        "New Prefix - ",
        "-RetireTaskNamePrefix",
        "new prefix - ",
    )
    assert result.returncode != 0
    assert "TaskNamePrefix and RetireTaskNamePrefix must be different" in (
        result.stdout + result.stderr
    )


def test_pinned_runtime_validator_output_cannot_pollute_marker_return_value():
    function = re.search(
        r"function Assert-PinnedRuntime \{(?P<body>.*?)\n\}",
        INSTALLER,
        flags=re.DOTALL,
    )
    assert function is not None
    body = function.group("body")
    assert re.search(
        r"\$validationLines\s*=\s*@\(& \$powershell.*?-ValidateOnly 2>&1\)",
        body,
        flags=re.DOTALL,
    )
    assert "$validationExitCode = $LASTEXITCODE" in body
    assert "$validationOutput = ($validationLines | Out-String).Trim()" in body
    assert "return $marker" in body


@REQUIRES_WINDOWS_POWERSHELL
def test_pinned_runtime_returns_only_the_marker_when_validator_writes_stdout(tmp_path):
    runtime = tmp_path / "runtime"
    config = tmp_path / "config"
    runner = runtime / "scripts" / "run_local_automation.ps1"
    marker_path = runtime / ".local" / "automation-runtime.json"
    runner.parent.mkdir(parents=True)
    config.mkdir()
    runner.write_text(
        textwrap.dedent(
            """
            param(
                [string]$Pipeline,
                [string]$RuntimeRoot,
                [string]$ConfigRoot,
                [switch]$ValidateOnly
            )
            Write-Output 'VALIDATION-SUCCESS-MUST-NOT-ESCAPE'
            exit 0
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=runtime, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.email", "automation-test@example.invalid"],
        cwd=runtime,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Automation Test"], cwd=runtime, check=True
    )
    subprocess.run(["git", "add", "scripts/run_local_automation.ps1"], cwd=runtime, check=True)
    subprocess.run(
        ["git", "commit", "-m", "test runtime"],
        cwd=runtime,
        check=True,
        capture_output=True,
        text=True,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=runtime,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    subprocess.run(["git", "tag", "test-runtime"], cwd=runtime, check=True)
    marker_path.parent.mkdir()
    marker_path.write_text(
        json.dumps(
            {
                "pinned_sha": sha,
                "fallback_ref": "test-runtime",
                "config_root": str(config),
            }
        ),
        encoding="utf-8",
    )

    harness = tmp_path / "assert_pinned_runtime.ps1"
    harness.write_text(
        textwrap.dedent(
            f"""
            Set-StrictMode -Version Latest
            $ErrorActionPreference = 'Stop'
            $MutableTrackedState = @()
            {_extract_function('Invoke-GitCapture')}
            {_extract_function('Assert-PinnedRuntime')}
            $script:GitExecutable = (Get-Command 'git.exe' -ErrorAction Stop).Source
            $script:RuntimeRoot = '{_ps_quote(runtime)}'
            $script:ConfigRoot = '{_ps_quote(config)}'
            $result = @(Assert-PinnedRuntime)
            if ($result.Count -ne 1) {{ throw "Expected one result, found $($result.Count)" }}
            if ([string]$result[0].pinned_sha -ne '{sha}') {{ throw 'Wrong marker returned' }}
            Write-Output 'ASSERT_PINNED_RUNTIME_SINGLE_MARKER_OK'
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    result = _run_powershell(harness)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "ASSERT_PINNED_RUNTIME_SINGLE_MARKER_OK"


@REQUIRES_WINDOWS_POWERSHELL
def test_git_capture_accepts_informational_stderr_when_exit_code_is_zero(tmp_path):
    fake_git = tmp_path / "fake_git.cmd"
    fake_git.write_text(
        "@echo off\r\n"
        "echo normal-output\r\n"
        "echo Preparing worktree 1^>^&2\r\n"
        "exit /b 0\r\n",
        encoding="ascii",
    )
    harness = tmp_path / "git_capture.ps1"
    harness.write_text(
        textwrap.dedent(
            f"""
            Set-StrictMode -Version Latest
            $ErrorActionPreference = 'Stop'
            $script:GitExecutable = '{_ps_quote(fake_git)}'
            {_extract_function('Invoke-GitCapture')}
            $result = Invoke-GitCapture -Arguments @('worktree', 'add')
            if ($result.ExitCode -ne 0) {{ throw 'wrong exit code' }}
            if ($result.Output -notmatch 'normal-output') {{ throw 'stdout missing' }}
            if ($result.Output -notmatch 'Preparing worktree') {{ throw 'stderr missing' }}
            Write-Output 'GIT_CAPTURE_STDERR_OK'
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    result = _run_powershell(harness)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "GIT_CAPTURE_STDERR_OK"


def _run_cutover_simulation(
    tmp_path: Path, fail_task: str = "", active_task: str = ""
) -> dict:
    harness = tmp_path / "simulate_cutover.ps1"
    harness.write_text(
        textwrap.dedent(
            f"""
            Set-StrictMode -Version Latest
            $ErrorActionPreference = 'Stop'
            $ConfirmCutover = $true
            $PruneSuperseded = $false
            $PipelineSpecs = @(
                [pscustomobject]@{{ Id = 'premarket' }},
                [pscustomobject]@{{ Id = 'postclose' }}
            )
            $SupersededTasks = @('Fixed Legacy One', 'Fixed Legacy Two')
            $TaskNamePrefix = 'New V3 - '
            $RetireTaskNamePrefix = 'New V2 - '
            $script:RuntimeRoot = '{_ps_quote(tmp_path / "runtime")}'
            $script:ConfigRoot = '{_ps_quote(tmp_path / "config")}'

            {_extract_function('Get-SupersededTaskNames')}
            {_extract_function('Assert-TaskIdle')}
            {_extract_function('Invoke-Cutover')}
            function Copy-RetiredRuntimeLogs {{ param($RootFolder) }}
            function Invoke-Prune {{ throw 'prune must not run in this simulation' }}

            function New-FakeTask {{
                param(
                    [string]$Name,
                    [bool]$Enabled,
                    [bool]$FailOnDisable = $false,
                    [int]$RuntimeState = 3
                )
                $task = [pscustomobject]@{{
                    Name = $Name
                    State = $RuntimeState
                    EnabledState = [pscustomobject]@{{ Value = $Enabled }}
                    FailOnDisable = $FailOnDisable
                }}
                $task | Add-Member -MemberType ScriptProperty -Name Enabled -Value {{
                    return [bool]$this.EnabledState.Value
                }} -SecondValue {{
                    param($value)
                    if ($this.FailOnDisable -and -not [bool]$value) {{
                        throw "simulated disable failure: $($this.Name)"
                    }}
                    $this.EnabledState.Value = [bool]$value
                }}
                return $task
            }}

            $tasks = @{{}}
            foreach ($spec in $PipelineSpecs) {{
                $name = $TaskNamePrefix + $spec.Id
                $tasks[$name] = New-FakeTask -Name $name -Enabled $false `
                    -RuntimeState $(if ($name -eq '{active_task.replace("'", "''")}') {{ 4 }} else {{ 3 }})
            }}
            $tasks['Fixed Legacy One'] = New-FakeTask -Name 'Fixed Legacy One' -Enabled $true `
                -RuntimeState $(if ('Fixed Legacy One' -eq '{active_task.replace("'", "''")}') {{ 4 }} else {{ 3 }})
            $tasks['Fixed Legacy Two'] = New-FakeTask -Name 'Fixed Legacy Two' -Enabled $false `
                -RuntimeState $(if ('Fixed Legacy Two' -eq '{active_task.replace("'", "''")}') {{ 4 }} else {{ 3 }})
            foreach ($spec in $PipelineSpecs) {{
                $name = $RetireTaskNamePrefix + $spec.Id
                $tasks[$name] = New-FakeTask -Name $name -Enabled $true `
                    -FailOnDisable ($name -eq '{fail_task.replace("'", "''")}') `
                    -RuntimeState $(if ($name -eq '{active_task.replace("'", "''")}') {{ 4 }} else {{ 3 }})
            }}
            $root = [pscustomobject]@{{ Tasks = $tasks }}
            $root | Add-Member -MemberType ScriptMethod -Name GetTask -Value {{
                param($name)
                if (-not $this.Tasks.ContainsKey($name)) {{ throw "missing task: $name" }}
                return $this.Tasks[$name]
            }}
            $script:FakeScheduler = [pscustomobject]@{{ Root = $root }}

            function Assert-Administrator {{}}
            function Assert-EasternLocalClock {{}}
            function Assert-PinnedRuntime {{ return [pscustomobject]@{{ pinned_sha = '0123456789abcdef' }} }}
            function Connect-TaskScheduler {{ return $script:FakeScheduler }}
            function Assert-RegisteredTaskDefinition {{ param($Task, $Spec, [string]$IdentitySid) }}
            function Get-TaskOrNull {{
                param($RootFolder, [string]$Name)
                if ($RootFolder.Tasks.ContainsKey($Name)) {{ return $RootFolder.Tasks[$Name] }}
                return $null
            }}

            $outcome = 'success'
            $message = ''
            try {{ $null = @(Invoke-Cutover) }}
            catch {{ $outcome = 'failure'; $message = $_.Exception.Message }}
            $states = @{{}}
            foreach ($name in $tasks.Keys) {{ $states[$name] = [bool]$tasks[$name].Enabled }}
            [pscustomobject]@{{ outcome = $outcome; message = $message; states = $states }} |
                ConvertTo-Json -Depth 4 -Compress
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    result = _run_powershell(harness)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@REQUIRES_WINDOWS_POWERSHELL
def test_cutover_enables_new_prefix_and_disables_prior_prefix_and_fixed_tasks(tmp_path):
    result = _run_cutover_simulation(tmp_path)
    assert result["outcome"] == "success"
    assert result["states"] == {
        "New V3 - premarket": True,
        "New V3 - postclose": True,
        "New V2 - premarket": False,
        "New V2 - postclose": False,
        "Fixed Legacy One": False,
        "Fixed Legacy Two": False,
    }


@REQUIRES_WINDOWS_POWERSHELL
def test_cutover_failure_restores_both_new_and_superseded_enabled_states(tmp_path):
    result = _run_cutover_simulation(tmp_path, fail_task="New V2 - postclose")
    assert result["outcome"] == "failure"
    assert "task enabled states were rolled back" in result["message"]
    assert result["states"] == {
        "New V3 - premarket": False,
        "New V3 - postclose": False,
        "New V2 - premarket": True,
        "New V2 - postclose": True,
        "Fixed Legacy One": True,
        "Fixed Legacy Two": False,
    }


@REQUIRES_WINDOWS_POWERSHELL
def test_cutover_refuses_to_mutate_when_a_superseded_task_is_running(tmp_path):
    result = _run_cutover_simulation(
        tmp_path, active_task="New V2 - premarket"
    )
    assert result["outcome"] == "failure"
    assert "requires every task to be idle" in result["message"]
    assert result["states"] == {
        "New V3 - premarket": False,
        "New V3 - postclose": False,
        "New V2 - premarket": True,
        "New V2 - postclose": True,
        "Fixed Legacy One": True,
        "Fixed Legacy Two": False,
    }


@REQUIRES_WINDOWS_POWERSHELL
def test_prune_candidates_protect_current_rollback_enabled_and_running_tasks(tmp_path):
    harness = tmp_path / "prune_candidates.ps1"
    harness.write_text(
        textwrap.dedent(
            f"""
            Set-StrictMode -Version Latest
            $ErrorActionPreference = 'Stop'
            $TaskNamePrefix = 'New Seasonals Local v9 - '
            $RetireTaskNamePrefix = 'New Seasonals Local v8 - '
            {_extract_function('Get-PruneCandidates')}
            function New-FakeTask {{
                param([string]$Name, [bool]$Enabled, [int]$State = 3)
                return [pscustomobject]@{{ Name = $Name; Enabled = $Enabled; State = $State }}
            }}
            $tasks = @(
                (New-FakeTask -Name 'New Seasonals Local v9 - premarket' -Enabled $true),
                (New-FakeTask -Name 'New Seasonals Local v8 - premarket' -Enabled $false),
                (New-FakeTask -Name 'New Seasonals Local v7 - premarket' -Enabled $false),
                (New-FakeTask -Name 'New Seasonals Local v6 - health' -Enabled $true),
                (New-FakeTask -Name 'New Seasonals Local v5 - postclose' -Enabled $false -State 4),
                (New-FakeTask -Name 'New Seasonals Local - premarket' -Enabled $false),
                (New-FakeTask -Name 'IBKR Daily Order Chain' -Enabled $true),
                (New-FakeTask -Name 'Trigger Daily Screener (GHA workflow_dispatch)' -Enabled $false)
            )
            $root = [pscustomobject]@{{ Items = $tasks }}
            $root | Add-Member -MemberType ScriptMethod -Name GetTasks -Value {{ param($flags) return $this.Items }}
            $rows = @(Get-PruneCandidates -RootFolder $root)
            $rows | ForEach-Object {{ [pscustomobject]@{{ Name = $_.Name; Action = $_.Action }} }} | ConvertTo-Json -Compress
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    result = _run_powershell(harness)
    assert result.returncode == 0, result.stdout + result.stderr
    rows = {row["Name"]: row["Action"] for row in json.loads(result.stdout.strip().splitlines()[-1])}
    assert rows == {
        "New Seasonals Local - premarket": "delete",
        "New Seasonals Local v5 - postclose": "keep: queued or running",
        "New Seasonals Local v6 - health": "keep: ENABLED (never pruned; disable it through a cutover first)",
        "New Seasonals Local v7 - premarket": "delete",
        "New Seasonals Local v8 - premarket": "keep: rollback generation (-RetireTaskNamePrefix)",
    }
    # The current generation and every non-generation task never appear.
    assert "New Seasonals Local v9 - premarket" not in rows
    assert "IBKR Daily Order Chain" not in rows
    assert "Trigger Daily Screener (GHA workflow_dispatch)" not in rows


def test_automation_requirements_are_explicit():
    requirements = (ROOT / "scripts" / "requirements-automation.txt").read_text(encoding="utf-8")
    assert "-r ../requirements.txt" in requirements
    assert "exchange-calendars" in requirements
    assert "google-auth" in requirements
    assert "pytest" in requirements
