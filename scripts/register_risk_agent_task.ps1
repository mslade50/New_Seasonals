# Registers the weekday 6:15 PM Risk Agent (paper) run.
#
# Intentionally inert until run by the operator. Registration creates a
# recurring unattended agent session that reads this repo, writes check scripts
# under scratch/risk_agent_checks/, and sends the Risk Agent email. It places
# NO orders: the sleeve is paper only.
#
# Runbook: docs/claude_ref/risk_agent.md
# House convention: eyeball several nights of manual output
# (scripts\run_risk_agent.bat) BEFORE registering this.

$ErrorActionPreference = 'Stop'

$dir      = Split-Path -Parent $MyInvocation.MyCommand.Path
$bat      = Join-Path $dir 'run_risk_agent.bat'
$taskName = 'Risk Agent (paper)'

if (-not (Test-Path $bat)) { throw "Cannot find $bat" }

$action = New-ScheduledTaskAction -Execute 'cmd.exe' `
    -Argument "/c `"$bat`"" -WorkingDirectory (Split-Path -Parent $dir)

# 18:15 local, after the 17:10 postclose job has rebuilt the shared risk export
# and prices.
$trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At '6:15PM'

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME `
    -LogonType Interactive -RunLevel Limited

# NO auto-restart, deliberately. A retry cannot tell "died before publishing"
# from "published, then the delivery check failed", so it risks a second email
# and a duplicate journal decision for the same date. The journal also refuses
# a second publish for an asof; check_risk_agent_delivered.py makes a miss loud.
# 100 minutes covers the 90 minute agent timeout plus grading and the checks.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 100)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null

Write-Host "Registered task '$taskName' -> weekdays 6:15 PM"
Write-Host "  Command: cmd /c `"$bat`""
Write-Host "  Log:     scripts\logs\risk_agent_last_run.log"
$task = Get-ScheduledTask -TaskName $taskName
$info = $task | Get-ScheduledTaskInfo
Write-Host ("  State: {0}   Next run: {1}" -f $task.State, $info.NextRunTime)
