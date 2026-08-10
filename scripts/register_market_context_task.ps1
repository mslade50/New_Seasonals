# Registers the Sun-Thu 6:30 PM Market Context brief.
#
# Intentionally inert until run by the operator. Registration creates a
# recurring unattended agent session that reads this repo, writes drill
# scripts under scratch/context_checks/, and posts one message to the report
# card's Slack channel. It touches no book, no broker and no orders: this
# product is position-blind by design.
#
# Spec: market_context_skill_design_2026-08-09.md section 10
#
# House convention, and it matters more here than usual: eyeball several
# evenings of output from manual runs (scripts\run_market_context.bat, with
# send_context_slack.py in --dry-run) and post at least one brief to Slack by
# hand BEFORE registering this. Nugget selection quality is where this product
# lives or dies and it is not something a first scheduled run should discover.

$ErrorActionPreference = 'Stop'

$dir      = Split-Path -Parent $MyInvocation.MyCommand.Path
$bat      = Join-Path $dir 'run_market_context.bat'
$taskName = 'Market Context Brief'

if (-not (Test-Path $bat)) { throw "Cannot find $bat" }

$action = New-ScheduledTaskAction -Execute 'cmd.exe' `
    -Argument "/c `"$bat`"" -WorkingDirectory (Split-Path -Parent $dir)

# 18:30, NOT 18:00. The denali report card holds the interactive Claude
# session at 18:00 and typically finishes well inside 30 minutes; starting
# together would have the two contend. Sunday through Thursday: there is no
# Friday edition, because nobody is previewing Monday on a Friday night.
$trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Sunday,Monday,Tuesday,Wednesday,Thursday -At '6:30PM'

# Interactive: S4U needs an elevated shell (0x80070005 otherwise). Fires
# whenever the desktop is on and logged in; a locked screen still counts.
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME `
    -LogonType Interactive -RunLevel Limited

# 60 minutes covers the sweep (about 10 seconds), the 10-minute stale-price
# retry, and an agent stage that writes and runs 3 to 8 drill scripts.
#
# NO auto-restart, deliberately. A retry cannot tell "died before posting"
# from "posted, then the delivery check failed", so it risks a second brief in
# the channel and duplicate journal records for the same evening. A missed
# evening delivers nothing rather than delivering twice, and
# check_context_delivered.py makes the miss loud.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 60)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null

Write-Host "Registered task '$taskName' -> Sun-Thu 6:30 PM"
Write-Host "  Command: cmd /c `"$bat`""
Write-Host "  Log:     scripts\logs\market_context_last_run.log"
$task = Get-ScheduledTask -TaskName $taskName
$info = $task | Get-ScheduledTaskInfo
Write-Host ("  State: {0}   Next run: {1}" -f $task.State, $info.NextRunTime)
