# setup_overflow_scan_task.ps1
# One-time setup for Windows Task Scheduler
# Run this script as Administrator

$TaskName = "OverflowDailyScan"
$ScriptPath = "C:\Users\mckin\New_Seasonals\scripts\run_overflow_scan.ps1"

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

# Create the trigger: Weekdays at 7:00 PM ET
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "07:00PM"

# Create the action: Run PowerShell script
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -NoProfile -File $ScriptPath" -WorkingDirectory "C:\Users\mckin\New_Seasonals"

# Settings
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -WakeToRun -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5) -ExecutionTimeLimit (New-TimeSpan -Hours 1) -StartWhenAvailable

# Register the task
Register-ScheduledTask -TaskName $TaskName -Trigger $Trigger -Action $Action -Settings $Settings -Description "Run overflow daily scan (extended universe, local parquet cache)" -RunLevel Highest

Write-Host ""
Write-Host "Task registered successfully!" -ForegroundColor Green
Write-Host "  Schedule: Weekdays at 7:00 PM ET"
Write-Host "  Script:   $ScriptPath"
Write-Host "  Wake:     Yes"
Write-Host "  Retry:    3 attempts, 5 min interval"
Write-Host ""
Write-Host "FIRST RUN: Build the cache manually before the first scheduled run:" -ForegroundColor Yellow
Write-Host "  cd C:\Users\mckin\New_Seasonals" -ForegroundColor Yellow
Write-Host "  python local_overflow_scan.py --rebuild" -ForegroundColor Yellow
