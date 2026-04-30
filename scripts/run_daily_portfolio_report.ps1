# run_daily_portfolio_report.ps1
# PowerShell wrapper for Task Scheduler to run the daily portfolio health report.
# Suggested schedule: Weekdays at 5:30 PM ET (after the overflow scan and
# verify_fills have settled the day's activity).

$ErrorActionPreference = "Stop"

# Set environment
$env:PYTHONIOENCODING = "utf-8"
$ProjectDir = "C:\Users\mckin\New_Seasonals"

# Load .env file for email credentials (EMAIL_USER, EMAIL_PASS, GCP_JSON, etc.)
$EnvFile = Join-Path $ProjectDir ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            [System.Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim(), "Process")
        }
    }
}

$LogDir = Join-Path $ProjectDir "logs"
$DateStr = Get-Date -Format "yyyy-MM-dd"
$TimeStr = Get-Date -Format "HHmmss"
$LogFile = Join-Path $LogDir "portfolio_report_${DateStr}_${TimeStr}.log"

# Create log directory if needed
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Log start
$StartTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$StartTime] Starting daily portfolio report..." | Tee-Object -FilePath $LogFile -Append

# Wait for network connectivity (machine may be waking from sleep)
$MaxWaitSec = 120
$Elapsed = 0
while (-not (Test-Connection -ComputerName 8.8.8.8 -Count 1 -Quiet -ErrorAction SilentlyContinue)) {
    if ($Elapsed -ge $MaxWaitSec) {
        "[$((Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))] ERROR: No network after ${MaxWaitSec}s, aborting." | Tee-Object -FilePath $LogFile -Append
        exit 1
    }
    "[$((Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))] Waiting for network..." | Tee-Object -FilePath $LogFile -Append
    Start-Sleep -Seconds 10
    $Elapsed += 10
}
"[$((Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))] Network ready (waited ${Elapsed}s)." | Tee-Object -FilePath $LogFile -Append

# Change to project directory
Set-Location $ProjectDir

try {
    $ErrorActionPreference = "Continue"
    $output = python daily_portfolio_report.py 2>&1
    $output | Tee-Object -FilePath $LogFile -Append
    $ErrorActionPreference = "Stop"

    if ($LASTEXITCODE -ne 0) {
        throw "Python exited with code $LASTEXITCODE"
    }

    $EndTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$EndTime] Daily portfolio report completed successfully." | Tee-Object -FilePath $LogFile -Append
    exit 0
}
catch {
    $ErrorTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ErrorTime] ERROR: $($_.Exception.Message)" | Tee-Object -FilePath $LogFile -Append
    exit 1
}
