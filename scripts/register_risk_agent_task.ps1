# Registers the two weekday Risk Agent (paper) tasks on the trading desktop:
#
#   Risk Agent (paper)      06:30  scripts\run_risk_agent.bat (sync, grade,
#                                  state, /risk-agent, delivery check)
#   Risk Agent open fill    09:36  grade_risk_agent.py --open-fill (option
#                                  orders fill at live IBKR quotes at the open,
#                                  so 0-1 DTE structures get a real entry)
#
# Run it on DESKTOP-2KI41V6 from the dev\New_Seasonals checkout (the same
# checkout the Pitch and Seasonal agents run from), as the interactive user, so
# claude.exe finds its login. It places NO orders: the sleeve is paper only, and
# the IBKR client is read-only market data on its own clientId.
#
# Runbook: docs/claude_ref/risk_agent.md

$ErrorActionPreference = 'Stop'

$dir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $dir
$bat  = Join-Path $dir 'run_risk_agent.bat'
if (-not (Test-Path $bat)) { throw "Cannot find $bat" }

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME `
    -LogonType Interactive -RunLevel Limited
$days = 'Monday','Tuesday','Wednesday','Thursday','Friday'

# NO auto-restart, deliberately. A retry cannot tell "died before publishing"
# from "published, then the delivery check failed", so it risks a second email
# and a duplicate journal decision for the same date. The journal also refuses
# a second publish for an asof; check_risk_agent_delivered.py makes a miss loud.
# 100 minutes covers the 90 minute agent timeout plus grading and the checks.
$main = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 100)
Register-ScheduledTask -TaskName 'Risk Agent (paper)' `
    -Action (New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c `"$bat`"" -WorkingDirectory $repo) `
    -Trigger (New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At '6:30AM') `
    -Principal $principal -Settings $main -Force | Out-Null

# The open fill is idempotent and a no-op outside regular hours or without the
# gateway, so a late start is harmless; it must never run past mid-morning.
$fill = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 15)
$fillCmd = "/c cd /d `"$repo`" && python scripts\grade_risk_agent.py --open-fill >> scripts\logs\risk_agent_open_fill.log 2>&1"
Register-ScheduledTask -TaskName 'Risk Agent open fill' `
    -Action (New-ScheduledTaskAction -Execute 'cmd.exe' -Argument $fillCmd -WorkingDirectory $repo) `
    -Trigger (New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At '9:36AM') `
    -Principal $principal -Settings $fill -Force | Out-Null

foreach ($name in 'Risk Agent (paper)', 'Risk Agent open fill') {
    $info = Get-ScheduledTask -TaskName $name | Get-ScheduledTaskInfo
    Write-Host ("Registered '{0}'  next run: {1}" -f $name, $info.NextRunTime)
}
