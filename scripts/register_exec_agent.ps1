# Register the ExecAgent scheduled task — runs the execution-bridge agent on the
# trading box during the 05:00-21:00 ET window. Idempotent (-Force replaces).
$dir = "C:\Users\McKinley Slade\OneDrive\trading_ibkr"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
  -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$dir\run_exec_agent.ps1`""

# Launch daily at 5 AM, and again at logon (so it comes back after a reboot+login,
# no admin needed). The agent self-exits outside 05:00-21:00, so the logon trigger
# is safe at any hour.
$t1 = New-ScheduledTaskTrigger -Daily -At 5:00AM
# A stopped wrapper also recovers; IgnoreNew prevents duplicate active instances.
$t1.Repetition = (New-ScheduledTaskTrigger -Once -At 5:00AM `
  -RepetitionInterval (New-TimeSpan -Minutes 5) `
  -RepetitionDuration (New-TimeSpan -Hours 16)).Repetition
$t1.Repetition.StopAtDurationEnd = $false
$t2 = New-ScheduledTaskTrigger -AtLogOn

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
  -RestartInterval (New-TimeSpan -Minutes 1) -RestartCount 3 `
  -ExecutionTimeLimit (New-TimeSpan -Hours 17) `
  -MultipleInstances IgnoreNew -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries

$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName "ExecAgent" -Action $action -Trigger $t1, $t2 `
  -Settings $settings -Principal $principal `
  -Description "Execution-bridge agent: outbound WebSocket to the Cloudflare broker. Self-limited to 05:00-21:00 ET; launched daily at 5 AM and at logon, with 5-minute recovery attempts until 9 PM." -Force

Write-Output "Registered. Next run times:"
Get-ScheduledTask -TaskName "ExecAgent" | Get-ScheduledTaskInfo | Select-Object LastRunTime, NextRunTime
