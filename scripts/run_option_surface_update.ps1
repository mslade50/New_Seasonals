# Nightly read-only ETF/index option-surface snapshot for the private site.
# Suggested Windows Task Scheduler trigger: weekdays at 5:20 PM ET, after the
# close and while TWS or IB Gateway remains open with market-data entitlements.
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$ProjectDir = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $ProjectDir "logs"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir -Force | Out-Null
}
$LogFile = Join-Path $LogDir ("option_surface_" + (Get-Date -Format "yyyy-MM-dd") + ".log")

Set-Location $ProjectDir
try {
    $output = python scripts/update_option_surface.py 2>&1
    $output | Tee-Object -FilePath $LogFile -Append
    if ($LASTEXITCODE -ne 0) {
        throw "option-surface recorder exited with code $LASTEXITCODE"
    }
    exit 0
}
catch {
    "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] ERROR: $($_.Exception.Message)" |
        Tee-Object -FilePath $LogFile -Append
    exit 1
}
