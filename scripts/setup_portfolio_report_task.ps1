# setup_portfolio_report_task.ps1
# One-time setup for Windows Task Scheduler.
# Run this script as Administrator.

$TaskName = "DailyPortfolioReport"
$ScriptPath = "C:\Users\mckin\New_Seasonals\scripts\run_daily_portfolio_report.ps1"

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

# Create the trigger: Weekdays at 5:30 PM ET (after the 5 PM ET overflow scan
# and 5:15 PM ET fill verification have written that day's state).
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "05:30PM"

# Create the action: Run PowerShell script
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -NoProfile -File $ScriptPath" -WorkingDirectory "C:\Users\mckin\New_Seasonals"

# Settings
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -WakeToRun -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5) -ExecutionTimeLimit (New-TimeSpan -Hours 1) -StartWhenAvailable

# Register the task
Register-ScheduledTask -TaskName $TaskName -Trigger $Trigger -Action $Action -Settings $Settings -Description "Daily portfolio health report - combined liquid + overflow backtest, emails to mckinleyslade@gmail.com" -RunLevel Highest

Write-Host ""
Write-Host "Task registered successfully!" -ForegroundColor Green
Write-Host "  Schedule: Weekdays at 5:30 PM ET"
Write-Host "  Script:   $ScriptPath"
Write-Host "  Wake:     Yes"
Write-Host "  Retry:    3 attempts, 5 min interval"
Write-Host ""
Write-Host "FIRST RUN: Test manually before relying on the schedule:" -ForegroundColor Yellow
Write-Host "  cd C:\Users\mckin\New_Seasonals" -ForegroundColor Yellow
Write-Host "  python daily_portfolio_report.py" -ForegroundColor Yellow
