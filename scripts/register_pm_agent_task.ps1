# Registers the PM layer tasks on the trading desktop:
#
#   PM Weekly      Sunday 16:00     scripts\run_pm_agent.bat (grade, state,
#                                   /pm-agent, delivery check)
#   PM check-in    weekdays 08:30   scripts\run_pm_daily_check.bat (code only;
#                                   emails only when an exception fires)
#
# Run it on DESKTOP-2KI41V6 from the dev\New_Seasonals checkout (the same
# checkout the Risk Agent, Pitch and Seasonal agents run from), as the
# interactive user, so claude.exe finds its login. Both are read-only: no
# orders, no writes outside PM_AGENT_HOME and R2 pm_agent/.
# Runbook: docs/claude_ref/pm_agent.md

$ErrorActionPreference = 'Stop'

$dir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $dir
$weekly = Join-Path $dir 'run_pm_agent.bat'
$daily  = Join-Path $dir 'run_pm_daily_check.bat'
foreach ($p in $weekly, $daily) { if (-not (Test-Path $p)) { throw "Cannot find $p" } }

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME `
    -LogonType Interactive -RunLevel Limited

# NO auto-restart (see register_risk_agent_task.ps1). 100 minutes covers the
# 90 minute agent timeout plus grading and the checks. 16:00 leaves the
# 18:30 Market Context and 20:00 Posts runs clear even at the timeout.
$wk = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 100)
Register-ScheduledTask -TaskName 'PM Weekly' `
    -Action (New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c `"$weekly`"" -WorkingDirectory $repo) `
    -Trigger (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At '4:00PM') `
    -Principal $principal -Settings $wk -Force | Out-Null

# 08:30: after premarket (04:10), the Pitch (05:10) and the Risk Agent (06:30),
# so their delivery receipts exist; before the 09:30 open. The check skips
# holidays itself and refuses a second check-in for the same date.
$dk = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 20)
Register-ScheduledTask -TaskName 'PM check-in' `
    -Action (New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c `"$daily`"" -WorkingDirectory $repo) `
    -Trigger (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At '8:30AM') `
    -Principal $principal -Settings $dk -Force | Out-Null

foreach ($name in 'PM Weekly', 'PM check-in') {
    $info = Get-ScheduledTask -TaskName $name | Get-ScheduledTaskInfo
    Write-Host ("Registered '{0}'  next run: {1}" -f $name, $info.NextRunTime)
}
