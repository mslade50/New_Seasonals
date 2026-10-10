# Launcher for the minimal Open Breakout runner (python -m open_breakout simple). Windows PowerShell 5.1.
# Meant for a Task Scheduler trigger at 09:20 ET on weekdays. Does NOT touch OpenBreakout_DailyLaunch / daily_launch.ps1.
#   1. session = today in New York; exit 0 if not an XNYS session (is_session.py)
#   2. refuse if any old live-session / shadow-session / LIVE simple runner is alive (shared client id 927481);
#      dry runs (--dry-run / --plan) never block a live launch
#   3. refuse at or after 09:27 (the runner refuses after 09:29:30 anyway); poll for the TWS port (config port, 7496) up to
#      10 min, but never past 09:27; then wait for 09:20 ET if started earlier
#   4. run the runner IN THE FOREGROUND of this script (-Wait) and exit with its exit code; logs to
#      artifacts/open_breakout_runs/<session>-simple/ (dry runs: <session>-simple-dry/)
#   -DryRun  connect and log the orders it would place, never transmit (no live ack is needed to be set by hand).
param([switch]$DryRun, [string]$Session)
$ErrorActionPreference = 'Stop'
$Repo = 'C:\Users\McKinley Slade\dev\New_Seasonals'
$Python = 'C:\Users\McKinley Slade\AppData\Local\Programs\Python\Python310\python.exe'
$Config = Join-Path $Repo 'artifacts\open_breakout_runs\config-20260925-live.json'
Set-Location $Repo

function Get-NyNow { [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow, 'Eastern Standard Time') }
$today = (Get-NyNow).ToString('yyyy-MM-dd')
if (-not $Session) { $Session = $today }
if (-not $DryRun -and $Session -ne $today) { throw "-Session is only for -DryRun (today is $today)" }
$Suffix = if ($DryRun) { 'simple-dry' } else { 'simple' }
$State = Join-Path $Repo "artifacts\open_breakout_runs\$Session-$Suffix"
New-Item -ItemType Directory -Force -Path $State | Out-Null
$Log = Join-Path $State 'launcher.log'
function Write-Log([string]$m) { $l = '{0} ET  {1}' -f (Get-NyNow).ToString('yyyy-MM-dd HH:mm:ss'), $m; Add-Content -Path $Log -Value $l; Write-Host $l }

& $Python 'artifacts/open_breakout_build/is_session.py' $Session | Out-Null
if ($LASTEXITCODE -eq 10) { Write-Log "$Session is not an XNYS session; nothing to do"; exit 0 }
if ($LASTEXITCODE -ne 0) { Write-Log "is_session.py failed ($LASTEXITCODE)"; exit 1 }

$cfg = Get-Content -Raw -Path $Config | ConvertFrom-Json
$port = if ($cfg.port) { [int]$cfg.port } else { 7496 }
$procs = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match 'open_breakout' })
$isDry = { param($p) $p.CommandLine -match '(--dry-run|--plan)' }
$alive = @($procs | Where-Object {
    if ($DryRun) { ($_.CommandLine -match '\ssimple\s') -and (& $isDry $_) }
    else { ($_.CommandLine -match '(live-session|shadow-session)') -or (($_.CommandLine -match '\ssimple\s') -and -not (& $isDry $_)) } })
if ($alive.Count -gt 0) { Write-Log "an open_breakout runner is already alive (PID $($alive.ProcessId -join ', ')); not starting"; exit 3 }

$ny = Get-NyNow
$cutoff = $ny.Date.AddHours(9).AddMinutes(27)
if (-not $DryRun -and $ny -ge $cutoff) { Write-Log "late start at $($ny.ToString('HH:mm:ss')) ET; not starting"; exit 7 }

function Test-Port([int]$p) {
    $c = New-Object System.Net.Sockets.TcpClient
    try { $iar = $c.BeginConnect('127.0.0.1', $p, $null, $null); return ($iar.AsyncWaitHandle.WaitOne(1000) -and $c.Connected) } catch { return $false } finally { $c.Close() }
}
$deadline = (Get-NyNow).AddMinutes(10)
if (-not $DryRun -and $deadline -gt $cutoff) { $deadline = $cutoff }
$noted = $false
while (-not (Test-Port $port)) {
    if ((Get-NyNow) -ge $deadline) { Write-Log "TWS port $port not accepting connections by $($deadline.ToString('HH:mm:ss')) ET; NOT STARTING"; exit 8 }
    if (-not $noted) { Write-Log "waiting for TWS on port $port (until $($deadline.ToString('HH:mm:ss')) ET)"; $noted = $true }
    Start-Sleep -Seconds 5
}
Write-Log "TWS port $port is up"

if (-not $DryRun) {
    $ny = Get-NyNow
    $start = $ny.Date.AddHours(9).AddMinutes(20)
    if ($ny -ge $cutoff) { Write-Log "late start at $($ny.ToString('HH:mm:ss')) ET after waiting for TWS; not starting"; exit 7 }
    if ($ny -lt $start) { Write-Log "waiting for 09:20 ET"; Start-Sleep -Seconds ([int]($start - $ny).TotalSeconds) }
}

$env:OPEN_BREAKOUT_LIVE_ACK = "LIVE $Session $($cfg.account)"   # inherited by the child only
$argList = @('-m', 'open_breakout', 'simple', '--mode', 'live', '--session', $Session)
if ($DryRun) {
    $argList += @('--dry-run', '--client-id', '927486')
    $n = Get-NyNow   # a past session, or today at/after 09:29:30, cannot be a live-style dry run: plan from the last bars instead
    if ($Session -ne $today -or $n -ge $n.Date.AddHours(9).AddMinutes(29).AddSeconds(30)) { $argList += '--plan' }
}
Write-Log "starting: python $($argList -join ' ') (account ending $($cfg.account.Substring($cfg.account.Length - 4)))"
try {
    $p = Start-Process -FilePath $Python -ArgumentList $argList -WorkingDirectory $Repo -WindowStyle Hidden -PassThru -Wait `
        -RedirectStandardOutput (Join-Path $State 'run.stdout.log') -RedirectStandardError (Join-Path $State 'run.stderr.log')
} finally { Remove-Item Env:\OPEN_BREAKOUT_LIVE_ACK -ErrorAction SilentlyContinue }
Write-Log "runner PID $($p.Id) exited with code $($p.ExitCode)"
exit $p.ExitCode
