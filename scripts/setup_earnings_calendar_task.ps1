# setup_earnings_calendar_task.ps1
# One-time setup for Windows Task Scheduler — daily earnings calendar refresh
# Run this script as Administrator

$TaskName = "EarningsCalendarRefresh"
$ScriptPath = "C:\Users\mckin\New_Seasonals\scripts\run_earnings_calendar.ps1"

# Check if running as admin
$currentPrincipal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $currentPrincipal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Error "This script must be run as Administrator. Right-click PowerShell and select 'Run as Administrator'."
    exit 1
}

# Remove existing task if it exists
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Removing existing task..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

# Trigger: weekdays at 5:30 PM ET (before 7 PM overflow scan)
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "05:30PM"

# Action: run the wrapper script
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -NoProfile -File $ScriptPath" -WorkingDirectory "C:\Users\mckin\New_Seasonals"

# Settings — wake from sleep, retry on failure, generous timeout (FMP pulls ~950 tickers at 10/sec ≈ 95s)
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -WakeToRun -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5) -ExecutionTimeLimit (New-TimeSpan -Hours 1) -StartWhenAvailable

# Register
Register-ScheduledTask -TaskName $TaskName -Trigger $Trigger -Action $Action -Settings $Settings -Description "Daily earnings_calendar.parquet refresh from FMP (weekdays 5:30 PM, before overflow scan)" -RunLevel Highest

Write-Host ""
Write-Host "Task registered successfully!" -ForegroundColor Green
Write-Host "  Schedule: Mon-Fri at 5:30 PM"
Write-Host "  Script:   $ScriptPath"
Write-Host "  Wake:     Yes"
Write-Host "  Retry:    3 attempts, 5 min interval"
