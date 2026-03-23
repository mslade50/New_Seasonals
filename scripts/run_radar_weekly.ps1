# run_radar_weekly.ps1
# PowerShell wrapper for Task Scheduler to run the radar weekly summary
# Trigger: Sundays at 8:30 AM ET via Windows Task Scheduler

$ErrorActionPreference = "Stop"

# Set environment
$env:PYTHONIOENCODING = "utf-8"
$ProjectDir = "C:\Users\mckin\New_Seasonals"
$LogDir = Join-Path $ProjectDir "logs"
$DateStr = Get-Date -Format "yyyy-MM-dd"
$LogFile = Join-Path $LogDir "radar_weekly_$DateStr.log"

# Create log directory if needed
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}

# Log start
$StartTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$StartTime] Starting radar weekly summary..." | Tee-Object -FilePath $LogFile -Append

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

# Change to project directory
Set-Location $ProjectDir

try {
    # Run the summary script
    $ErrorActionPreference = "Continue"
    $output = python radar_weekly_summary.py 2>&1
    $output | Tee-Object -FilePath $LogFile -Append
    $ErrorActionPreference = "Stop"

    if ($LASTEXITCODE -ne 0) {
        throw "Python exited with code $LASTEXITCODE"
    }

    # Commit and push the summary so GitHub Actions can pick it up
    "[$((Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))] Committing and pushing summary..." | Tee-Object -FilePath $LogFile -Append
    git add data/radar_weekly_summary.md
    git commit -m "data: radar weekly digest $DateStr"
    git push

    $EndTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$EndTime] Radar weekly summary completed successfully." | Tee-Object -FilePath $LogFile -Append
    exit 0
}
catch {
    $ErrorTime = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "[$ErrorTime] ERROR: $($_.Exception.Message)" | Tee-Object -FilePath $LogFile -Append
    exit 1
}
