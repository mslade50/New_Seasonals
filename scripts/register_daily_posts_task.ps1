# Registers the Sun-Fri 8:00 PM Daily Posts drafting run.
#
# Registration creates a recurring unattended agent session that reads this
# repo, writes check scripts under scratch/posts_checks/ and a post queue
# under content/queue/. It posts NOTHING anywhere: the queue is a review
# file, McKinley posts by hand. No book, no broker, no orders, no web.
#
# 8:00 PM ET: well clear of the PM price cron (5:10 PM EDT) and of the
# 6:30 PM market-context session. Saturday skipped - no session to preview,
# weekend posting draws on content/backlog/. A Friday run previews Monday.

$ErrorActionPreference = 'Stop'

$dir      = Split-Path -Parent $MyInvocation.MyCommand.Path
$bat      = Join-Path $dir 'run_daily_posts.bat'
$taskName = 'Daily Posts (agent)'

if (-not (Test-Path $bat)) { throw "Cannot find $bat" }

$action = New-ScheduledTaskAction -Execute 'cmd.exe' `
    -Argument "/c `"$bat`"" -WorkingDirectory (Split-Path -Parent $dir)

$trigger = New-ScheduledTaskTrigger -Weekly `
    -DaysOfWeek Sunday,Monday,Tuesday,Wednesday,Thursday,Friday -At '8:00PM'

# Interactive, not S4U: S4U registration needs an elevated shell
# (0x80070005 otherwise). Fires whenever the machine is on and logged in;
# a locked screen still counts. Same convention as the pitch and context
# tasks.
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME `
    -LogonType Interactive -RunLevel Limited

# 45 minutes covers the pull + grade + state (seconds each) and a drafting
# stage that writes and runs 2-5 check scripts. NO auto-restart: a retry
# cannot tell "died before writing the queue" from "wrote it, then the
# delivery check tripped", and a rerun would overwrite a queue McKinley may
# already be marking. A missed evening delivers nothing, loudly.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 45)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger `
    -Principal $principal -Settings $settings -Force | Out-Null

Write-Host "Registered task '$taskName' -> Sun-Fri 8:00 PM"
Write-Host "  Command: cmd /c `"$bat`""
Write-Host "  Log:     scripts\logs\daily_posts_last_run.log"
$task = Get-ScheduledTask -TaskName $taskName
$info = $task | Get-ScheduledTaskInfo
Write-Host ("  State: {0}   Next run: {1}" -f $task.State, $info.NextRunTime)
