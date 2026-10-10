# Registers the Sunday PM Weekly task on the trading desktop:
#
#   PM Weekly    Sunday 16:00   scripts\run_pm_agent.bat (grade, state,
#                               /pm-agent, delivery check)
#
# Run it on DESKTOP-2KI41V6 from the dev\New_Seasonals checkout (the same
# checkout the Risk Agent, Pitch and Seasonal agents run from), as the
# interactive user, so claude.exe finds its login. It places no orders and
# reads no book data. Runbook: docs/claude_ref/pm_agent.md

$ErrorActionPreference = 'Stop'

$dir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $dir
$bat  = Join-Path $dir 'run_pm_agent.bat'
if (-not (Test-Path $bat)) { throw "Cannot find $bat" }

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME `
    -LogonType Interactive -RunLevel Limited

# NO auto-restart (see register_risk_agent_task.ps1). 100 minutes covers the
# 90 minute agent timeout plus grading and the checks. 16:00 leaves the
# 18:30 Market Context and 20:00 Posts runs clear even at the timeout.
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 100)
Register-ScheduledTask -TaskName 'PM Weekly' `
    -Action (New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c `"$bat`"" -WorkingDirectory $repo) `
    -Trigger (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At '4:00PM') `
    -Principal $principal -Settings $settings -Force | Out-Null

$info = Get-ScheduledTask -TaskName 'PM Weekly' | Get-ScheduledTaskInfo
Write-Host ("Registered 'PM Weekly'  next run: {0}" -f $info.NextRunTime)
