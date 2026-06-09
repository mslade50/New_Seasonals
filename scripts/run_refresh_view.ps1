# run_refresh_view.ps1 - nightly trade-ledger + HTML refresh wrapper for Task Scheduler.
# Rebuilds data/backtest_trades_full.parquet (+ daily PnL) and regenerates
# reports/portfolio/backtester_view.html via scripts/refresh_view.py (~1-2 min).
# Unattended: no --open. Logs to C:\Scripts\logs\refresh_view.log.
$ErrorActionPreference = 'Continue'
$proj   = 'C:\Users\mckin\New_Seasonals'
$py     = 'C:\Program Files\Python311\python.exe'
$logdir = 'C:\Scripts\logs'
if (-not (Test-Path $logdir)) { New-Item -ItemType Directory -Force -Path $logdir | Out-Null }
$log = Join-Path $logdir 'refresh_view.log'

Set-Location $proj
$start = Get-Date
Add-Content $log "`r`n==== $($start.ToString('yyyy-MM-dd HH:mm:ss'))  refresh_view START ===="
& $py 'scripts\refresh_view.py' 2>&1 | Out-File -FilePath $log -Append -Encoding utf8
$code = $LASTEXITCODE
$dur = [int]((Get-Date) - $start).TotalSeconds
Add-Content $log "==== $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  DONE exit=$code (${dur}s) ===="
exit $code
