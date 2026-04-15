# run_overflow_scan.ps1
# PowerShell wrapper for Task Scheduler to run the local overflow scanner
# Suggested schedule: Weekdays, same times as daily_scan GitHub Actions
#   - 9:13, 17:40, 18:45, 19:30, 20:13 UTC (convert to ET as needed)
# Or a single daily run: Weekdays 4:30 PM ET (20:30 UTC)

$ErrorActionPreference = "Stop"

# Set environment
$env:PYTHONIOENCODING = "utf-8"
$ProjectDir = "C:\Users\mckin\New_Seasonals"

# Load .env file for email credentials
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
$LogFile = Join-Path $LogDir "overflow_scan_${DateStr}_${TimeStr}.log"

# Create log directory if needed
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Log start
$StartTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$StartTime] Starting overflow scan..." | Tee-Object -FilePath $LogFile -Append

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
    $output = python local_overflow_scan.py 2>&1
    $output | Tee-Object -FilePath $LogFile -Append
    $ErrorActionPreference = "Stop"

    if ($LASTEXITCODE -ne 0) {
        throw "Python exited with code $LASTEXITCODE"
    }

    $EndTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$EndTime] Overflow scan completed successfully." | Tee-Object -FilePath $LogFile -Append
    exit 0
}
catch {
    $ErrorTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ErrorTime] ERROR: $($_.Exception.Message)" | Tee-Object -FilePath $LogFile -Append
    exit 1
}
