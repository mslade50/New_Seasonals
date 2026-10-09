# Source for artifacts/open_breakout_runs/daily_launch.ps1; deploy byte-for-byte after offline qualification.
# Unattended daily launch of the open-breakout shadow + LIVE session (Task Scheduler,
# weekdays 08:12 ET, task OpenBreakout_DailyLaunch). Windows PowerShell 5.1.
# LIVE is normal risk sizing from 2026-09-29 (one contract on 2026-09-28); no per-market contract cap,
# only the fat-finger ceiling (60) checked by launch-live.ps1 and the code; the step-6 preflight
# what-ifs a fixed reference size (10) as a warning only
# (the 09:25 arming preflight in the session gates on the planned sizes).
#
#   1. session = today in New York; exit 0 if not an XNYS session (is_session.py)
#   2. log: artifacts/open_breakout_runs/daily_launch_<session>.log (append; never Slack)
#   3. IB Gateway port 7496 listening, else poll every 30 s for up to 10 min      -> exit 2
#   4. no session process running, no runtime.sqlite in <session>-shadow*/-live*   -> exit 3
#   5. valid risk refresh for the session, else run refresh_risk.py once          -> exit 4
#   6. read-only preflight (live config, client 927485), at most 3 attempts, must report ok            -> exit 5
#   7. launch-shadow.ps1 then launch-live.ps1 (each starts one hidden python via Start-Process)
#   8. after 60 s (then polling up to 3 more minutes) both sessions alive + connected -> exit 6
#   Late start (task started at or after 09:20 ET, e.g. StartWhenAvailable)        -> exit 7
#   Bad arguments / unexpected error                                                -> exit 1
# Every failure ends the log with one line: DAILY_LAUNCH FAILED (<code>): <reason>
#
#   -DryRun   steps 1-6 for real (read-only; step 5 may write a new risk_* folder), then runs both
#             launchers with -DryRun and prints the two launch commands. Creates no state dir.
#             Log: daily_launch_<session>_dryrun.log; preflight file ...-live-dryrun.json.
#   -Session  YYYY-MM-DD. Only with -DryRun (a real launch is always today's New York date).
param(
    [string]$Session,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$Repo = 'C:\Users\McKinley Slade\dev\New_Seasonals'
$Python = 'C:\Users\McKinley Slade\AppData\Local\Programs\Python\Python310\python.exe'
$PowerShellExe = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$RunsRel = 'artifacts/open_breakout_runs'
$Runs = Join-Path $Repo 'artifacts\open_breakout_runs'
$Build = Join-Path $Repo 'artifacts\open_breakout_build'
$LiveConfig = 'artifacts/open_breakout_runs/config-20260925-live.json'
$PreflightClient = 927485
$GatewayPort = 7496
$PortWaitSeconds = 600
$PortPollSeconds = 30
$LateCutoff = New-TimeSpan -Hours 9 -Minutes 20
$StatusFirstWait = 60
$StatusExtraWait = 180
$Tmp = [IO.Path]::GetTempPath()

Set-Location $Repo

function Get-NyNow { [TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow, 'Eastern Standard Time') }

$nyStart = Get-NyNow
$today = $nyStart.ToString('yyyy-MM-dd')
$badSession = $null
if (-not $Session) { $Session = $today }
$parsed = [datetime]::MinValue
if (-not [datetime]::TryParseExact($Session, 'yyyy-MM-dd', [Globalization.CultureInfo]::InvariantCulture, 'None', [ref]$parsed)) {
    $badSession = "Session must be YYYY-MM-DD, got '$Session'"
    $Session = $today
} elseif (-not $DryRun -and $Session -ne $today) {
    $badSession = "-Session $Session is allowed only with -DryRun (a real launch uses today's New York date $today)"
    $Session = $today
}

$Log = Join-Path $Runs ("daily_launch_$Session" + $(if ($DryRun) { '_dryrun' } else { '' }) + '.log')
$account = $null

function Hide-Account([string]$Text) {
    if ($account -and $Text) { return $Text.Replace([string]$account, '<ACCT>') }
    return $Text
}

function Write-Log([string]$Message) {
    $line = '{0} ET  {1}' -f (Get-NyNow).ToString('yyyy-MM-dd HH:mm:ss'), (Hide-Account $Message)
    Add-Content -Path $Log -Value $line -Encoding UTF8
    Write-Host $line
}

function Stop-Launch([int]$Code, [string]$Reason) {
    Write-Log "DAILY_LAUNCH FAILED ($Code): $Reason"
    exit $Code
}

function Read-Shared([string]$Path) {
    # The launchers' python grandchildren may inherit these handles; open with full sharing.
    if (-not (Test-Path $Path)) { return '' }
    try {
        $fs = [IO.File]::Open($Path, 'Open', 'Read', 'ReadWrite, Delete')
        try { $sr = New-Object IO.StreamReader($fs); return $sr.ReadToEnd() } finally { $fs.Dispose() }
    } catch { return "<could not read $Path : $($_.Exception.Message)>" }
}

function Add-ChildOutput([string]$Tag, [string]$Text) {
    if (-not $Text) { return }
    foreach ($l in ($Text -split "`r?`n")) { if ($l.Trim()) { Write-Log "  [$Tag] $l" } }
}

# Runs a child with stdout/stderr redirected to temp files (never pipes), waits on the process
# handle only (not on descendants, unlike Start-Process -Wait), kills it on timeout.
function Invoke-Child([string]$Tag, [string]$File, [string[]]$ArgList, [int]$TimeoutSec,
                      [hashtable]$ChildEnv = @{}, [switch]$QuietStdout) {
    $stamp = '{0}_{1}' -f $Tag, [guid]::NewGuid().ToString('N').Substring(0, 8)
    $out = Join-Path $Tmp "daily_launch_$stamp.out"
    $err = Join-Path $Tmp "daily_launch_$stamp.err"
    $saved = @{}
    foreach ($k in $ChildEnv.Keys) {
        $saved[$k] = [Environment]::GetEnvironmentVariable($k, 'Process')
        [Environment]::SetEnvironmentVariable($k, $ChildEnv[$k], 'Process')
    }
    try {
        $p = Start-Process -FilePath $File -ArgumentList $ArgList -WorkingDirectory $Repo -NoNewWindow `
            -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
        $null = $p.Handle
    } finally {
        foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k], 'Process') }
    }
    $timedOut = -not $p.WaitForExit($TimeoutSec * 1000)
    if ($timedOut) {
        try { $p.Kill() } catch { }
        $code = -1
    } else {
        $code = $p.ExitCode
    }
    $stdout = Read-Shared $out
    $stderr = Read-Shared $err
    if (-not $QuietStdout) { Add-ChildOutput $Tag $stdout }
    Add-ChildOutput "$Tag stderr" $stderr
    Remove-Item -Path $out, $err -Force -ErrorAction SilentlyContinue
    if ($timedOut) { Write-Log "$Tag timed out after $TimeoutSec s and was killed" }
    return [pscustomobject]@{ Code = $code; TimedOut = $timedOut; Stdout = $stdout }
}

function Test-GatewayPort {
    try {
        $c = Get-NetTCPConnection -LocalPort $GatewayPort -State Listen -ErrorAction Stop
        if ($c) { return $true }
    } catch { }
    $ns = netstat -ano -p tcp | Select-String -Pattern (":$GatewayPort\s+\S+\s+LISTENING")
    return [bool]$ns
}

function Get-SessionProcesses {
    @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
        Where-Object { $_.CommandLine -match 'open_breakout' -and $_.CommandLine -match '(shadow|live)-session' })
}

function Resolve-Risk([string]$SessionDate) {
    # Same selection rule as launch-live.ps1 / launch-shadow.ps1 Resolve-RiskFile.
    $candidates = Get-ChildItem -Path $Build -Directory -Filter 'risk_*' | Sort-Object Name -Descending
    foreach ($dir in $candidates) {
        $refresh = Join-Path $dir.FullName 'refresh.json'
        if (-not (Test-Path $refresh)) { continue }
        try { $info = Get-Content -Raw -Path $refresh | ConvertFrom-Json } catch { continue }
        if ($info.session -ne $SessionDate) { continue }
        if ($info.PSObject.Properties.Name -contains 'risk_error') { continue }
        if (-not ($info.PSObject.Properties.Name -contains 'legacy_score')) { continue }
        if (-not (Test-Path (Join-Path $Repo $info.path))) { continue }
        return [pscustomobject]@{ Dir = $dir.Name; Info = $info }
    }
    return $null
}

function Test-LateStart {
    $ny = Get-NyNow
    if (-not $DryRun -and $ny.TimeOfDay -ge $LateCutoff) {
        Stop-Launch 7 "late start at $($ny.ToString('HH:mm')) ET (cutoff $($LateCutoff.ToString('hh\:mm'))); not launching. The sessions refuse after 09:30 anyway; launch by hand only if there is time to arm before 09:25."
    }
}

try {
    $mode = if ($DryRun) { 'DRYRUN' } else { 'LIVE' }
    Write-Log "DAILY_LAUNCH START session=$Session mode=$mode pid=$PID user=$env:USERNAME"
    if ($badSession) { Stop-Launch 1 $badSession }
    Test-LateStart

    # 1. XNYS session?
    $r = Invoke-Child 'is_session' $Python @('artifacts/open_breakout_build/is_session.py', $Session) 60
    if ($r.Code -eq 10) {
        Write-Log "DAILY_LAUNCH SKIPPED: $Session is not an XNYS session; nothing to do"
        exit 0
    }
    if ($r.Code -ne 0) { Stop-Launch 1 "is_session.py failed (exit $($r.Code))" }

    # Live config: account only for the child's env ack; never logged.
    $cfg = Get-Content -Raw -Path (Join-Path $Repo $LiveConfig) | ConvertFrom-Json
    $account = [string]$cfg.account
    if (-not $account -or $cfg.mode -ne 'live' -or $cfg.client_id -ne 927481) {
        Stop-Launch 1 "unexpected live config $LiveConfig (mode/client_id/account)"
    }
    if ($cfg.client_id -eq $PreflightClient) { Stop-Launch 1 "preflight client $PreflightClient equals the session client" }

    # 3. Gateway port
    $deadline = (Get-Date).AddSeconds($PortWaitSeconds)
    $announced = $false
    while (-not (Test-GatewayPort)) {
        if ((Get-Date) -ge $deadline) {
            Stop-Launch 2 "IB Gateway port $GatewayPort not listening after $([int]($PortWaitSeconds / 60)) minutes; log in to the Gateway and launch by hand (runbook)"
        }
        if (-not $announced) { Write-Log "port $GatewayPort not listening; polling every $PortPollSeconds s for up to $([int]($PortWaitSeconds / 60)) min"; $announced = $true }
        Start-Sleep -Seconds $PortPollSeconds
    }
    Write-Log "step 3 ok: port $GatewayPort listening"

    # 4. Nothing already running / journaled for this session
    $procs = Get-SessionProcesses
    $mine = @($procs | Where-Object { $_.CommandLine -match ('--session\s+' + [regex]::Escape($Session)) })
    if ($mine.Count -gt 0) {
        Stop-Launch 3 "open_breakout session process already running for $Session (PID $($mine.ProcessId -join ', '))"
    }
    if ($procs.Count -gt 0) {
        Stop-Launch 3 "open_breakout session process for another date still running (PID $($procs.ProcessId -join ', ')); the launchers refuse while one is alive"
    }
    $pattern = '^' + [regex]::Escape($Session) + '-(shadow|live)(-\d)?$'
    $journaled = @(Get-ChildItem -Path $Runs -Directory | Where-Object { $_.Name -match $pattern } |
        Where-Object { Test-Path (Join-Path $_.FullName 'runtime.sqlite') })
    if ($journaled.Count -gt 0) {
        Stop-Launch 3 "state dir already has runtime.sqlite: $($journaled.Name -join ', '); inspect with daily_status.ps1, relaunch by hand with -Attempt N only if no orders were placed"
    }
    Write-Log "step 4 ok: no session process running, no $Session journal"

    # 5. Risk refresh
    $risk = Resolve-Risk $Session
    if (-not $risk) {
        Write-Log "no valid risk refresh for $Session; running refresh_risk.py --session $Session once"
        $r = Invoke-Child 'refresh_risk' $Python @('artifacts/open_breakout_build/refresh_risk.py', '--session', $Session) 180
        Write-Log "refresh_risk.py exit $($r.Code)"
        $risk = Resolve-Risk $Session
    }
    if (-not $risk) {
        Stop-Launch 4 "no risk refresh for $Session without risk_error (R2 may not have the prior session's row yet); rerun refresh_risk.py --session $Session later, then launch by hand"
    }
    Write-Log "step 5 ok: risk $($risk.Dir) risk_latest=$($risk.Info.risk_latest) legacy_score=$($risk.Info.legacy_score) short_gate=$($risk.Info.short_gate)"

    # 6. Read-only preflight, bounded retries with fresh broker proof, ack only in each child
    . (Join-Path $Repo 'scripts/open_breakout_startup_preflight.ps1')
    $pf = Invoke-OpenBreakoutStartupPreflight -Repo $Repo -Python $Python -RunsRel $RunsRel `
        -LiveConfig $LiveConfig -Session $Session -ClientId $PreflightClient -Account $account `
        -LateCutoff $LateCutoff -DryRun:$DryRun
    Write-Log "preflight final: ok=$($pf.ok) stream_age_seconds=$($pf.stream_age_seconds | ConvertTo-Json -Compress) orders_placed=$($pf.orders_placed)"
    # Margin at the reference size is information only here (no manifest before 09:00): warnings never fail.
    Write-Log "preflight margin: basis=$($pf.what_if_basis) qty=$($pf.what_if_qty | ConvertTo-Json -Compress) total=$($pf.margin_total) limit=$($pf.margin_limit) warnings=$($pf.warnings | ConvertTo-Json -Compress)"
    Write-Log 'step 6 ok: preflight passed'

    # 7. Launch
    $launchers = @(
        @{ Tag = 'launch-shadow'; Script = 'artifacts\open_breakout_runs\launch-shadow.ps1' },
        @{ Tag = 'launch-live'; Script = 'artifacts\open_breakout_runs\launch-live.ps1' }
    )
    if ($DryRun) {
        foreach ($l in $launchers) {
            $cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File $($l.Script) -Session $Session"
            Write-Log "DRY RUN would run: $cmd"
            $r = Invoke-Child "$($l.Tag) -DryRun" $PowerShellExe @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $l.Script, '-Session', $Session, '-DryRun') 120
            if ($r.Code -ne 0) { Stop-Launch 6 "$($l.Tag) -DryRun exited $($r.Code)" }
        }
        $left = @(Get-ChildItem -Path $Runs -Directory | Where-Object { $_.Name -match $pattern })
        if ($left.Count -gt 0) { Stop-Launch 1 "dry run left state dirs: $($left.Name -join ', ')" }
        Write-Log "DAILY_LAUNCH DRYRUN OK: steps 1-6 passed for $Session; nothing launched, no state dir created"
        exit 0
    }

    Test-LateStart
    $launchFail = @()
    foreach ($l in $launchers) {
        Write-Log "step 7: $($l.Tag)"
        $r = Invoke-Child $l.Tag $PowerShellExe @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $l.Script, '-Session', $Session) 120
        # A shadow failure does not block the live launch; step 8 then fails the run.
        if ($r.Code -ne 0) { $launchFail += "$($l.Tag) exited $($r.Code)"; Write-Log "$($l.Tag) exited $($r.Code)" }
    }

    # 8. Status
    Start-Sleep -Seconds $StatusFirstWait
    $deadline = (Get-Date).AddSeconds($StatusExtraWait)
    while ($true) {
        $r = Invoke-Child 'status' $PowerShellExe @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', 'artifacts\open_breakout_runs\daily_status.ps1', '-Session', $Session) 120
        if ($r.Code -eq 0) { break }
        $dead = $r.Stdout -match 'alive=False|no runtime.sqlite|no state dir|HALT|FAIL'
        if ($dead -or (Get-Date) -ge $deadline) {
            $why = (($r.Stdout -split "`r?`n") | Where-Object { $_ -match '^STATUS' }) -join ' '
            if ($launchFail.Count -gt 0) { $why = ($launchFail -join '; ') + '; ' + $why }
            Stop-Launch 6 "sessions not both running and connected: $why"
        }
        Write-Log 'not both connected yet; re-checking in 15 s'
        Start-Sleep -Seconds 15
    }
    if ($launchFail.Count -gt 0) { Stop-Launch 6 ($launchFail -join '; ') }

    Write-Log 'step 9: monitor launch through the 09:30 ET gate'
    & "$Repo\scripts\monitor_open_breakout_launch.ps1" `
        -Session $Session -Repo $Repo -TimeoutSeconds 5400
    $monitorExit = $LASTEXITCODE
    if ($monitorExit -ne 0) { Stop-Launch $monitorExit "critical-window launch verification failed (exit $monitorExit)" }

    Write-Log "DAILY_LAUNCH OK: $Session shadow and live running and connected. Attend 09:25-11:30 and 15:50-16:01 ET."
    exit 0
} catch {
    Stop-Launch 1 "unexpected error: $($_.Exception.Message) at line $($_.InvocationInfo.ScriptLineNumber)"
}
