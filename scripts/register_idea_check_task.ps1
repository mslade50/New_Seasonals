# Registers the IdeaCheckPoller task: one pass every minute, all day.
#
# Intentionally inert until run by the operator. Registration creates a
# recurring unattended job that reads the R2 idea queue and starts headless
# Claude Code sessions (subscription auth). It places no orders.
#
# NO CONSOLE FLASH: the task does not run cmd.exe directly. It runs
# wscript.exe //B against a one-line VBS launcher that starts the .bat with
# window style 0 (hidden), the same pattern the Daily Pitch task uses. The
# launcher is written to %LOCALAPPDATA%\IdeaCheck at registration time.
#
# Runbook: docs/claude_ref/idea_check.md

$ErrorActionPreference = 'Stop'

$dir      = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo     = Split-Path -Parent $dir
$bat      = Join-Path $dir 'run_idea_check_poller.bat'
$taskName = 'IdeaCheckPoller'

if (-not (Test-Path $bat)) { throw "Cannot find $bat" }

$launchDir = Join-Path $env:LOCALAPPDATA 'IdeaCheck'
New-Item -ItemType Directory -Path $launchDir -Force | Out-Null
$vbs = Join-Path $launchDir 'run_idea_check_poller.vbs'
$q = '""'
$vbsText = @(
    'Option Explicit',
    'Dim shell, code',
    'Set shell = CreateObject("WScript.Shell")',
    "code = shell.Run(`"cmd.exe /c $q$bat$q`", 0, True)",
    'WScript.Quit code'
) -join "`r`n"
Set-Content -LiteralPath $vbs -Value $vbsText -Encoding ASCII

$action = New-ScheduledTaskAction -Execute "$env:SystemRoot\System32\wscript.exe" `
    -Argument "//B //NoLogo `"$vbs`"" -WorkingDirectory $repo

$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).Date `
    -RepetitionInterval (New-TimeSpan -Minutes 1) `
    -RepetitionDuration (New-TimeSpan -Days 3650)

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME `
    -LogonType Interactive -RunLevel Limited

# IgnoreNew: a pass still running (an agent review can take 20 minutes) makes
# the next minute's start a no-op instead of stacking. The poller also holds
# its own lock file, which the poller refreshes before each review.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 8)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null

Write-Host "Registered task '$taskName' -> every 1 minute"
Write-Host "  Launcher: $vbs"
Write-Host "  Command:  cmd /c `"$bat`" (hidden)"
Write-Host "  Log:      scripts\logs\idea_check_last_run.log"
$task = Get-ScheduledTask -TaskName $taskName
$info = $task | Get-ScheduledTaskInfo
Write-Host ("  State: {0}   Next run: {1}" -f $task.State, $info.NextRunTime)
