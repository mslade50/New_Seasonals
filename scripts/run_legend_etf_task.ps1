param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("signals", "session", "watchdog")]
    [string]$Mode,
    [string[]]$Accounts = @("primary"),
    [switch]$Live,
    [switch]$ShadowThroughExit,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"
$transcriptStarted = $false
$runtimeRoot = ""
$logPath = ""
$runId = ""
$runStartedAt = ""
try {
$Accounts = @(
    $Accounts |
        ForEach-Object { $_ -split "," } |
        ForEach-Object { $_.Trim().ToLowerInvariant() } |
        Where-Object { $_ }
)
if ($Accounts.Count -eq 0 -or @($Accounts | Where-Object { $_ -notin @("primary", "pa") }).Count -gt 0) {
    throw "Accounts must contain only exact labels: primary, pa."
}
if (@($Accounts | Select-Object -Unique).Count -ne $Accounts.Count) {
    throw "Accounts must not contain duplicates."
}
if ((Get-TimeZone).Id -ne "Eastern Standard Time") {
    throw "Legend ETF tasks require the Windows host timezone Eastern Standard Time."
}

function Write-AtomicJson {
    param(
        [Parameter(Mandatory = $true)]
        [object]$InputObject,
        [Parameter(Mandatory = $true)]
        [string]$LiteralPath
    )
    $parent = Split-Path -Parent $LiteralPath
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $leaf = Split-Path -Leaf $LiteralPath
    $temporary = Join-Path $parent ".$leaf.$([guid]::NewGuid().ToString('N')).tmp"
    $json = $InputObject | ConvertTo-Json -Depth 12
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText(
        $temporary,
        $json + [Environment]::NewLine,
        $encoding
    )
    if (Test-Path -LiteralPath $LiteralPath -PathType Leaf) {
        [System.IO.File]::Replace($temporary, $LiteralPath, $null, $true)
    }
    else {
        [System.IO.File]::Move($temporary, $LiteralPath)
    }
}

function Get-LeaseProcess {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Lease
    )
    $leasePid = 0
    if (-not [int]::TryParse([string]$Lease.pid, [ref]$leasePid) -or $leasePid -le 0) {
        throw "Legend live session lease PID is invalid."
    }
    $leaseRunId = [string]$Lease.run_id
    if ([string]::IsNullOrWhiteSpace($leaseRunId)) {
        throw "Legend live session lease runId is missing."
    }
    try {
        $leaseStart = [DateTimeOffset]::Parse([string]$Lease.process_started_at).ToUniversalTime()
    }
    catch {
        throw "Legend live session lease process-start identity is invalid."
    }
    $process = Get-Process -Id $leasePid -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        return $null
    }
    $actualStart = [DateTimeOffset]$process.StartTime.ToUniversalTime()
    if ([math]::Abs(($actualStart - $leaseStart).TotalSeconds) -gt 2.0) {
        throw "Legend lease PID was reused by a different process; automatic takeover is unsafe."
    }
    $cim = Get-CimInstance Win32_Process -Filter "ProcessId = $leasePid" -ErrorAction Stop
    $runPattern = '--run-id\s+"?' + [regex]::Escape($leaseRunId) + '"?(?:\s|$)'
    if ($null -eq $cim -or [string]$cim.CommandLine -notmatch $runPattern -or
        [string]$cim.CommandLine -notmatch 'run_legend_etf_session\.py') {
        throw "Legend lease process command line does not match its exact runId; automatic takeover is unsafe."
    }
    return $process
}

function Stop-And-ProveLeaseProcess {
    param(
        [object]$Process,
        [string]$SessionTaskName
    )
    $sessionTask = Get-ScheduledTask -TaskName $SessionTaskName -ErrorAction SilentlyContinue
    if ($null -ne $sessionTask -and $sessionTask.State -eq "Running") {
        Stop-ScheduledTask -TaskName $SessionTaskName -ErrorAction Stop
    }
    elseif ($null -ne $Process) {
        # This is an orphaned, exact PID/start-time/runId-bound scheduled child.
        Stop-Process -Id $Process.Id -ErrorAction Stop
    }
    for ($attempt = 0; $attempt -lt 10; $attempt++) {
        Start-Sleep -Milliseconds 500
        $remainingProcess = $(
            if ($null -eq $Process) { $null }
            else { Get-Process -Id $Process.Id -ErrorAction SilentlyContinue }
        )
        $remainingTask = Get-ScheduledTask -TaskName $SessionTaskName -ErrorAction SilentlyContinue
        if ($null -eq $remainingProcess -and
            ($null -eq $remainingTask -or $remainingTask.State -ne "Running")) {
            return
        }
    }
    throw "The exact Legend session owner did not stop; broker reconciliation was not started."
}

function Fence-AbsentLegendOwner {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Lease,
        [Parameter(Mandatory = $true)]
        [string]$LeasePath,
        [Parameter(Mandatory = $true)]
        [string]$WatchdogRunId
    )
    $mutex = New-Object System.Threading.Mutex($false, "Global\NewSeasonals.LegendETF.Runner.v1")
    $acquired = $false
    try {
        try {
            $acquired = $mutex.WaitOne(0)
        }
        catch [System.Threading.AbandonedMutexException] {
            $acquired = $true
        }
        if (-not $acquired) {
            throw "Legend session mutex is still held; automatic takeover is unsafe."
        }
        $Lease.status = "fenced"
        $Lease.fenced_at = (Get-Date).ToUniversalTime().ToString("o")
        $Lease.fenced_by = $WatchdogRunId
        Write-AtomicJson -InputObject $Lease -LiteralPath $LeasePath
    }
    finally {
        if ($acquired) {
            $mutex.ReleaseMutex()
        }
        $mutex.Dispose()
    }
}
$repoRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
    throw "LOCALAPPDATA is required for the machine-global Legend runtime."
}
$runtimeRoot = Join-Path $env:LOCALAPPDATA "NewSeasonals\legend_etf"
$logRoot = Join-Path $runtimeRoot "logs"
New-Item -ItemType Directory -Force -Path $logRoot | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $logRoot "$Mode-$stamp.log"
if ([string]::IsNullOrWhiteSpace($PythonExe)) {
    $PythonExe = (Get-Command python -ErrorAction Stop).Source
}
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Pinned Python interpreter does not exist: $PythonExe"
}

Start-Transcript -Path $logPath | Out-Null
$transcriptStarted = $true
    $calendarRaw = & $PythonExe (Join-Path $PSScriptRoot "check_legend_etf_calendar.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Legend ETF calendar precheck exited with code $LASTEXITCODE"
    }
    $calendar = ($calendarRaw | Out-String) | ConvertFrom-Json
    $entryDate = [string]$calendar.entry_date
    if ($Mode -eq "session") {
        $runId = [guid]::NewGuid().ToString("N")
        $runStartedAt = (Get-Date).ToUniversalTime().ToString("o")
        $activeRun = [pscustomobject]@{
            Mode = "session"
            RunId = $runId
            Status = "running"
            StartedAt = $runStartedAt
            EntryDate = $entryDate
            Live = [bool]$Live
            Accounts = @($Accounts | Sort-Object)
        }
        Write-AtomicJson -InputObject $activeRun -LiteralPath (Join-Path $runtimeRoot "active_session_run.json")
    }
    if (-not [bool]$calendar.full_session) {
        # Corrections/busts can arrive after the morning task.  Both the
        # session task and the 10:40 watchdog therefore run the independent,
        # read-only account-symbol audit even though no entry is permitted.
        if ($Live) {
            $auditArguments = @(
                (Join-Path $PSScriptRoot "run_legend_etf_session.py"),
                "--accounts"
            ) + $Accounts + @("--live", "--audit-only")
            & $PythonExe @auditArguments
            if ($LASTEXITCODE -ne 0) {
                throw "Legend ETF non-entry-day correction audit exited with code $LASTEXITCODE"
            }
        }
        if ($Mode -eq "watchdog") {
            $activePath = Join-Path $runtimeRoot "active_session_run.json"
            $successPath = Join-Path $runtimeRoot "last_success_session.json"
            if (-not (Test-Path -LiteralPath $activePath -PathType Leaf) -or -not (Test-Path -LiteralPath $successPath -PathType Leaf)) {
                throw "Today's expected-skip Legend ETF session receipt pair is missing."
            }
            $active = Get-Content -LiteralPath $activePath -Raw | ConvertFrom-Json
            $receipt = Get-Content -LiteralPath $successPath -Raw | ConvertFrom-Json
            $expectedAccounts = (@($Accounts | Sort-Object) -join ",")
            $activeAccounts = (@($active.Accounts | Sort-Object) -join ",")
            $receiptAccounts = (@($receipt.Accounts | Sort-Object) -join ",")
            if ($active.Status -ne "skipped_non_full_session" -or
                $receipt.Status -ne "skipped_non_full_session" -or
                $active.RunId -ne $receipt.RunId -or
                $active.EntryDate -ne $entryDate -or $receipt.EntryDate -ne $entryDate -or
                [bool]$active.Live -ne [bool]$Live -or [bool]$receipt.Live -ne [bool]$Live -or
                $activeAccounts -ne $expectedAccounts -or $receiptAccounts -ne $expectedAccounts) {
                throw "The scheduled Legend ETF expected-skip receipt does not match."
            }
        }
        $skipReceipt = [pscustomobject]@{
            Mode = $Mode
            Status = "skipped_non_full_session"
            Reason = [string]$calendar.reason
            At = (Get-Date).ToUniversalTime().ToString("o")
            Log = $logPath
            RunId = $runId
            StartedAt = $runStartedAt
            EntryDate = $entryDate
            Live = [bool]$Live
            Accounts = @($Accounts | Sort-Object)
        }
        if ($Mode -eq "session") {
            Write-AtomicJson -InputObject $skipReceipt -LiteralPath (Join-Path $runtimeRoot "active_session_run.json")
        }
        Write-AtomicJson -InputObject $skipReceipt -LiteralPath (Join-Path $runtimeRoot "last_success_$Mode.json")
        Write-Host "Legend ETF skipped: $($calendar.reason) on $entryDate"
        exit 0
    }
    if ($Mode -eq "watchdog") {
        if ($Live) {
            $leasePath = Join-Path $runtimeRoot "live_session_lease.json"
            if (Test-Path -LiteralPath $leasePath -PathType Leaf) {
                $lease = Get-Content -LiteralPath $leasePath -Raw | ConvertFrom-Json
                if ($lease.protocol -ne "legend-live-session-lease-v1") {
                    throw "Legend live session lease protocol is invalid."
                }
                if ($lease.Entry_Date -eq $entryDate -and $lease.status -eq "active") {
                    $sessionTaskName = "NewSeasonals-LegendETF-Session"
                    $leaseProcess = Get-LeaseProcess -Lease $lease
                    Stop-And-ProveLeaseProcess -Process $leaseProcess -SessionTaskName $sessionTaskName
                    Fence-AbsentLegendOwner `
                        -Lease $lease `
                        -LeasePath $leasePath `
                        -WatchdogRunId "watchdog-$entryDate"
                }
            }
            $reconcileArguments = @(
                (Join-Path $PSScriptRoot "run_legend_etf_session.py"),
                "--accounts"
            ) + $Accounts + @("--live", "--reconcile-only", "--run-id", "watchdog-$entryDate")
            & $PythonExe @reconcileArguments
            if ($LASTEXITCODE -ne 0) {
                throw "Legend ETF watchdog broker reconciliation exited with code $LASTEXITCODE"
            }
        }
        $activePath = Join-Path $runtimeRoot "active_session_run.json"
        $successPath = Join-Path $runtimeRoot "last_success_session.json"
        if (-not (Test-Path -LiteralPath $activePath -PathType Leaf) -or -not (Test-Path -LiteralPath $successPath -PathType Leaf)) {
            throw "Today's Legend ETF active/success receipt pair is missing."
        }
        $active = Get-Content -LiteralPath $activePath -Raw | ConvertFrom-Json
        $receipt = Get-Content -LiteralPath $successPath -Raw | ConvertFrom-Json
        $startedAt = [DateTimeOffset]::Parse([string]$active.StartedAt).ToLocalTime()
        $nowLocal = [DateTimeOffset]::Now
        $scheduledBoundary = [DateTimeOffset]::new(
            $nowLocal.Year,
            $nowLocal.Month,
            $nowLocal.Day,
            9,
            27,
            0,
            $nowLocal.Offset
        )
        $expectedAccounts = (@($Accounts | Sort-Object) -join ",")
        $activeAccounts = (@($active.Accounts | Sort-Object) -join ",")
        $receiptAccounts = (@($receipt.Accounts | Sort-Object) -join ",")
        if ($active.Status -ne "completed" -or $receipt.Status -ne "success" -or
            $active.RunId -ne $receipt.RunId -or $startedAt -lt $scheduledBoundary -or
            $active.EntryDate -ne $entryDate -or $receipt.EntryDate -ne $entryDate -or
            [bool]$active.Live -ne [bool]$Live -or [bool]$receipt.Live -ne [bool]$Live -or
            $activeAccounts -ne $expectedAccounts -or $receiptAccounts -ne $expectedAccounts) {
            throw "The scheduled Legend ETF session has no matching completed receipt."
        }
    }
    elseif ($Mode -eq "signals") {
        & $PythonExe (Join-Path $PSScriptRoot "prepare_legend_etf_signals.py")
    }
    else {
        $arguments = @(
            (Join-Path $PSScriptRoot "run_legend_etf_session.py"),
            "--accounts"
        ) + $Accounts
        if (-not [string]::IsNullOrWhiteSpace($runId)) {
            $arguments += @("--run-id", $runId)
        }
        if ($Live) {
            $arguments += "--live"
        }
        if ($ShadowThroughExit) {
            $arguments += "--shadow-through-exit"
        }
        & $PythonExe @arguments
    }
    if ($Mode -ne "watchdog" -and $LASTEXITCODE -ne 0) {
        throw "Legend ETF $Mode task exited with code $LASTEXITCODE"
    }
    $successReceipt = [pscustomobject]@{
        Mode = $Mode
        Status = "success"
        At = (Get-Date).ToUniversalTime().ToString("o")
        Log = $logPath
        RunId = $runId
        StartedAt = $runStartedAt
        EntryDate = $entryDate
        Live = [bool]$Live
        Accounts = @($Accounts | Sort-Object)
    }
    if ($Mode -eq "session") {
        $activeReceipt = $successReceipt | Select-Object *
        $activeReceipt.Status = "completed"
        Write-AtomicJson -InputObject $activeReceipt -LiteralPath (Join-Path $runtimeRoot "active_session_run.json")
    }
    Write-AtomicJson -InputObject $successReceipt -LiteralPath (Join-Path $runtimeRoot "last_success_$Mode.json")
}
catch {
    $originalError = $_.Exception.Message
    try {
        if ([string]::IsNullOrWhiteSpace($runtimeRoot)) {
            if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
                throw "LOCALAPPDATA is unavailable for the failure marker."
            }
            $runtimeRoot = Join-Path $env:LOCALAPPDATA "NewSeasonals\legend_etf"
        }
        New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
        $failurePath = Join-Path $runtimeRoot "latest_failure.json"
        $failureReceipt = [pscustomobject]@{
            Mode = $Mode
            Status = "failure"
            At = (Get-Date).ToUniversalTime().ToString("o")
            Error = $originalError
            Log = $logPath
            RunId = $runId
            StartedAt = $runStartedAt
            Live = [bool]$Live
            Accounts = @($Accounts | Sort-Object)
        }
        Write-AtomicJson -InputObject $failureReceipt -LiteralPath $failurePath
        if ($Mode -eq "session" -and -not [string]::IsNullOrWhiteSpace($runId)) {
            $activeFailure = [pscustomobject]@{
                Mode = "session"
                RunId = $runId
                Status = "failure"
                StartedAt = $runStartedAt
                EntryDate = $entryDate
                FailedAt = (Get-Date).ToUniversalTime().ToString("o")
                Live = [bool]$Live
                Accounts = @($Accounts | Sort-Object)
                Error = $originalError
            }
            Write-AtomicJson -InputObject $activeFailure -LiteralPath (Join-Path $runtimeRoot "active_session_run.json")
        }
    }
    catch {
        # Preserve the original failure even if durable marker I/O also fails.
    }
    $alert = "Legend ETF $Mode FAILED. Review $logPath"
    try {
        & "$env:SystemRoot\System32\msg.exe" $env:USERNAME $alert 2>$null
    }
    catch {
        # The durable failure marker remains even when no interactive session
        # is available for a local popup.
    }
    throw
}
finally {
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
}
