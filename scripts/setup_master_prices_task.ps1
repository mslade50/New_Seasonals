# setup_master_prices_task.ps1
# One-time setup for Windows Task Scheduler
# Run this script as Administrator

$TaskName = "MasterPricesUpdate"
$ScriptPath = "C:\Users\mckin\New_Seasonals\scripts\run_master_prices_update.ps1"

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

# Trigger: weekdays at 6:00 PM
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday,Tuesday,Wednesday,Thursday,Friday -At "06:00PM"

# Action: run the wrapper script
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -NoProfile -File $ScriptPath" -WorkingDirectory "C:\Users\mckin\New_Seasonals"

# Settings
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -WakeToRun -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5) -ExecutionTimeLimit (New-TimeSpan -Hours 1) -StartWhenAvailable

# Register
Register-ScheduledTask -TaskName $TaskName -Trigger $Trigger -Action $Action -Settings $Settings -Description "Daily master_prices.parquet incremental update (weekdays 6 PM)" -RunLevel Highest

Write-Host ""
Write-Host "Task registered successfully!" -ForegroundColor Green
Write-Host "  Schedule: Mon-Fri at 6:00 PM"
Write-Host "  Script:   $ScriptPath"
Write-Host "  Wake:     Yes"
Write-Host "  Retry:    3 attempts, 5 min interval"
