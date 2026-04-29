# run_master_prices_update.ps1
# PowerShell wrapper for Task Scheduler to update data/master_prices.parquet
# Trigger: Weekdays at 6:00 PM ET via Windows Task Scheduler

$ErrorActionPreference = "Stop"

$env:PYTHONIOENCODING = "utf-8"
$ProjectDir = "C:\Users\mckin\New_Seasonals"
$LogDir = Join-Path $ProjectDir "logs"
$DateStr = Get-Date -Format "yyyy-MM-dd"
$LogFile = Join-Path $LogDir "master_prices_update_$DateStr.log"

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

$StartTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$StartTime] Starting master_prices update..." | Tee-Object -FilePath $LogFile -Append

# Wait for network connectivity (machine may be waking from sleep)
$MaxWaitSec = 300
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

Set-Location $ProjectDir

try {
    $ErrorActionPreference = "Continue"
    $output = python scripts/update_master_prices.py 2>&1
    $output | Tee-Object -FilePath $LogFile -Append
    $ErrorActionPreference = "Stop"

    if ($LASTEXITCODE -ne 0) {
        throw "Python exited with code $LASTEXITCODE"
    }

    $EndTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$EndTime] master_prices update completed successfully." | Tee-Object -FilePath $LogFile -Append
    exit 0
}
catch {
    $ErrorTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ErrorTime] ERROR: $($_.Exception.Message)" | Tee-Object -FilePath $LogFile -Append
    exit 1
}
