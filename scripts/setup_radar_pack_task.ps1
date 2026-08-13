# setup_radar_pack_task.ps1
# One-time setup for Windows Task Scheduler — weekly radar pack export
# (Sat 10:00 local; fail-soft: a missed run only degrades the radar's
# earnings/seasonal fields to their yfinance/neutral fallbacks).
# Run this script as Administrator.

$TaskName = "RadarPackExport"
$ScriptPath = Join-Path $PSScriptRoot "run_radar_pack.ps1"
$ProjectDir = Split-Path -Parent $PSScriptRoot

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

# Trigger: Saturdays at 10:00 AM local (before the Sun/Mon 08:00 UTC radar screen)
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At "10:00AM"

# Action: run the wrapper script
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-ExecutionPolicy Bypass -NoProfile -File `"$ScriptPath`"" -WorkingDirectory $ProjectDir

# Settings — wake from sleep, retry on failure, generous timeout
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -WakeToRun -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 5) -ExecutionTimeLimit (New-TimeSpan -Hours 1) -StartWhenAvailable

# Register
Register-ScheduledTask -TaskName $TaskName -Trigger $Trigger -Action $Action -Settings $Settings -Description "Weekly radar data pack export (earnings + ATR seasonal ranks) into radar-briefings (Sat 10:00, fail-soft)" -RunLevel Highest

Write-Host ""
Write-Host "Task registered successfully!" -ForegroundColor Green
Write-Host "  Schedule: Saturdays at 10:00 AM"
Write-Host "  Script:   $ScriptPath"
Write-Host "  Wake:     Yes"
Write-Host "  Retry:    3 attempts, 5 min interval"
