# Registers the weekday 7:00 AM Daily Pitch run.
#
# Intentionally inert until run by the operator. Registration creates a
# recurring unattended agent session that reads this repo, writes check
# scripts under scratch/pitch_checks/, and sends the Daily Pitch email.
# It does NOT place orders: placement lives in pitch_moo.py on the trading
# machine, behind an Approve = Y gate and its own activation flag.
#
# Spec: daily_pitch_agent_spec_2026-08-06.html section 9
# Runbook: docs/daily_pitch.md
#
# House convention: eyeball several days of output from manual runs
# (scripts\run_daily_pitch.bat) BEFORE registering this.

$ErrorActionPreference = 'Stop'

$dir      = Split-Path -Parent $MyInvocation.MyCommand.Path
$bat      = Join-Path $dir 'run_daily_pitch.bat'
$taskName = 'Daily Pitch (agent)'

if (-not (Test-Path $bat)) { throw "Cannot find $bat" }

$action = New-ScheduledTaskAction -Execute 'cmd.exe' `
    -Argument "/c `"$bat`"" -WorkingDirectory (Split-Path -Parent $dir)

# 7:00 AM ET: the 4:47 scan chain and the 4:30 risk correction have landed,
# so every input is today's. Delivery lands around 7:45, before the
# pre-market order window and long before the 9:05 approval runner.
$trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At '7:00AM'

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME `
    -LogonType Interactive -RunLevel Limited

# Generous limit: stage C fans out several verification agents and each
# writes and runs a real check script.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 60) `
    -RestartCount 1 -RestartInterval (New-TimeSpan -Minutes 10)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null

Write-Host "Registered task '$taskName' -> weekdays 7:00 AM"
Write-Host "  Command: cmd /c `"$bat`""
Write-Host "  Log:     scripts\logs\daily_pitch_last_run.log"
$task = Get-ScheduledTask -TaskName $taskName
$info = $task | Get-ScheduledTaskInfo
Write-Host ("  State: {0}   Next run: {1}" -f $task.State, $info.NextRunTime)
