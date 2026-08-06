# Registers the weekday 5:10 AM Daily Pitch run.
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

# 5:10 AM ET, NOT 5:00. The 4:47 scan runs 13-16 minutes (measured across
# five AM runs), landing 5:00-5:04, and it is what writes the Order_Staging /
# Overflow tabs and exposure_state.json that the mandatory `overlap` field is
# written from. Starting at 5:00 would race it. 5:10 clears it with margin and
# still delivers around 6:20, hours before the 9:05 approval runner.
$trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At '5:10AM'

# Interactive: S4U needs an elevated shell (0x80070005 otherwise). Matches
# the risk-correction trigger. Fires whenever the desktop is on and logged in;
# a locked screen still counts as logged in.
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME `
    -LogonType Interactive -RunLevel Limited

# 120 min, not 60: the opus/xhigh bake-off run took 69 minutes end to end
# (19 candidates, 7 verifier subagents, 25 check scripts), so a 60-minute
# limit would kill a normal morning mid-falsification.
#
# NO auto-restart, deliberately. A retry cannot tell "died before publishing"
# from "published, then the delivery check failed", so it risks a second
# email and duplicate journal ids for the same date. The spec's own stance is
# that a missed run delivers nothing rather than delivering late, and
# check_pitch_delivered.py makes the miss loud.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 120)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null

Write-Host "Registered task '$taskName' -> weekdays 5:10 AM"
Write-Host "  Command: cmd /c `"$bat`""
Write-Host "  Log:     scripts\logs\daily_pitch_last_run.log"
$task = Get-ScheduledTask -TaskName $taskName
$info = $task | Get-ScheduledTaskInfo
Write-Host ("  State: {0}   Next run: {1}" -f $task.State, $info.NextRunTime)
