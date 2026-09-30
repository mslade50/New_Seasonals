# Registers the weekday 07:00 Daily Seasonal run.
#
# Intentionally inert until run by the operator. Registration creates a
# recurring unattended agent session that reads this repo, writes check
# scripts under scratch/seasonal_checks/, and sends the Daily Seasonal email.
# It does NOT place orders: nothing reads the Seasonal Agent tab for placement
# in v1.
#
# Mirrors register_daily_pitch_task.ps1 exactly, including where it points:
# the .bat next to THIS script, in whatever checkout this script is run from,
# with `python` resolved from PATH (no venv, no pinned worktree).
#
# Design: docs/seasonal_agent_design_2026-09-30.md
# Live rule: docs/claude_ref/daily_seasonal.md

$ErrorActionPreference = 'Stop'

$dir      = Split-Path -Parent $MyInvocation.MyCommand.Path
$bat      = Join-Path $dir 'run_daily_seasonal.bat'
$taskName = 'Daily Seasonal (agent)'

if (-not (Test-Path $bat)) { throw "Cannot find $bat" }

$action = New-ScheduledTaskAction -Execute 'cmd.exe' `
    -Argument "/c `"$bat`"" -WorkingDirectory (Split-Path -Parent $dir)

# 07:00, after the 5:10 Daily Pitch (owner decision 2026-09-30). The pitch
# delivers around 6:20, so its journal fingerprints are in this morning's
# seasonal state and the two products never pitch the same trade.
$trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At '7:00AM'

# Interactive, same as the pitch: S4U needs an elevated shell.
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME `
    -LogonType Interactive -RunLevel Limited

# 120 min and NO auto-restart, same reasons as the pitch: a retry cannot tell
# "died before publishing" from "published, then the delivery check failed".
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 120)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null

Write-Host "Registered task '$taskName' -> weekdays 7:00 AM"
Write-Host "  Command: cmd /c `"$bat`""
Write-Host "  Log:     scripts\logs\daily_seasonal_last_run.log"
$task = Get-ScheduledTask -TaskName $taskName
$info = $task | Get-ScheduledTaskInfo
Write-Host ("  State: {0}   Next run: {1}" -f $task.State, $info.NextRunTime)
