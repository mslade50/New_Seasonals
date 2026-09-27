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
    # IB writes benign notices (e.g. "Error 200 ... No security definition")
    # to stderr; under "Stop" those become terminating errors, so relax it
    # around the native call and judge success by python's exit code only.
    $ErrorActionPreference = "Continue"
    $output = python scripts/update_option_surface.py 2>&1
    $PyExit = $LASTEXITCODE
    $output | ForEach-Object { "$_" } | Tee-Object -FilePath $LogFile -Append
    $ErrorActionPreference = "Stop"
    "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] python exit code: $PyExit" |
        Tee-Object -FilePath $LogFile -Append
    exit $PyExit
}
catch {
    "[$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))] ERROR: $($_.Exception.Message)" |
        Tee-Object -FilePath $LogFile -Append
    exit 1
}
