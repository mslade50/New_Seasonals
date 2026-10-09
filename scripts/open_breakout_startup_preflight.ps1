# Read-only startup recovery. Uses the daily launcher's child/log/clock functions.
# A new CLI process establishes fresh broker proof on each attempt; no report is reused.
function Invoke-OpenBreakoutStartupPreflight {
    param(
        [string]$Repo, [string]$Python, [string]$RunsRel, [string]$LiveConfig,
        [string]$Session, [int]$ClientId, [string]$Account,
        [timespan]$LateCutoff, [switch]$DryRun
    )
    $maxAttempts = 3
    $retryDelays = @(2, 5)
    $suffix = if ($DryRun) { '-dryrun' } else { '' }
    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        $now = Get-NyNow
        if (-not $DryRun -and ($now.ToString('yyyy-MM-dd') -ne $Session -or $now.TimeOfDay -ge $LateCutoff)) {
            Stop-Launch 7 'startup preflight reached the 09:20 ET cutoff or session date changed'
        }
        $timeout = 240
        if (-not $DryRun) {
            $timeout = [int][Math]::Min($timeout, [Math]::Floor(($LateCutoff - $now.TimeOfDay).TotalSeconds))
            if ($timeout -lt 1) { Stop-Launch 7 'no preflight time remains before the 09:20 ET cutoff' }
        }
        $pfRel = "$RunsRel/preflight-$Session-live$suffix.json"
        $n = 2
        while (Test-Path (Join-Path $Repo $pfRel)) { $pfRel = "$RunsRel/preflight-$Session-live$suffix-$n.json"; $n++ }
        Write-Log "step 6: preflight attempt $attempt/$maxAttempts client $ClientId -> $pfRel"
        $r = Invoke-Child 'preflight' $Python @('-m', 'open_breakout', 'preflight', '--config', $LiveConfig,
            '--session', $Session, '--client-id', "$ClientId", '--out', $pfRel) $timeout -ChildEnv @{
                OPEN_BREAKOUT_LIVE_ACK = "LIVE $Session $Account"; PYTHONWARNINGS = 'ignore' } -QuietStdout
        $pfPath = Join-Path $Repo $pfRel
        if ($r.TimedOut -or $r.Code -ne 0 -or -not (Test-Path $pfPath)) {
            Add-ChildOutput 'preflight stdout' $r.Stdout
            Stop-Launch 5 "preflight child failed or wrote no report (exit $($r.Code), timeout=$($r.TimedOut)); see $pfRel"
        }
        try { $pf = Get-Content -Raw -Path $pfPath | ConvertFrom-Json }
        catch { Stop-Launch 5 "invalid preflight report: $pfRel" }
        $failures = @($pf.failures)
        $failureText = ConvertTo-Json -InputObject $failures -Compress -Depth 6
        Write-Log "preflight attempt $attempt/$maxAttempts`: ok=$($pf.ok) retryable=$($pf.recovery_retryable) failures=$failureText orders_placed=$($pf.orders_placed) report=$pfRel"
        # Never authorize a session from an absent/ambiguous zero-order assertion.
        if ($null -eq $pf.orders_placed -or $pf.orders_placed -ne 0) {
            Stop-Launch 5 "preflight did not assert zero orders: $pfRel"
        }
        if ($pf.ok -eq $true) {
            if ($failures.Count -ne 0) { Stop-Launch 5 "inconsistent successful preflight: $pfRel" }
            $now = Get-NyNow
            if (-not $DryRun -and ($now.ToString('yyyy-MM-dd') -ne $Session -or $now.TimeOfDay -ge $LateCutoff)) {
                Stop-Launch 7 'preflight completed after the 09:20 ET cutoff or session date changed'
            }
            return $pf
        }
        # Match only the installed runner's explicit transport-proof retry class.
        # Any mixed business/safety failure makes the entire report terminal.
        $retryable = $pf.recovery_retryable -eq $true -and $failures.Count -gt 0
        foreach ($failure in $failures) {
            if ($failure -isnot [string] -or -not (
                $failure -ceq 'RECOVERY_EVIDENCE_CHANGED' -or $failure -ceq 'TRANSPORT_UNHEALTHY' -or
                $failure.StartsWith('RECOVERY_INCOMPLETE:', [StringComparison]::Ordinal) -or
                $failure.StartsWith('STREAM_MISSING:', [StringComparison]::Ordinal) -or
                $failure.StartsWith('STREAM_STALE:', [StringComparison]::Ordinal))) { $retryable = $false }
        }
        if (-not $retryable -or $attempt -ge $maxAttempts) {
            Stop-Launch 5 "preflight not ok after $attempt/$maxAttempts attempts: $failureText; report=$pfRel"
        }
        $delay = $retryDelays[$attempt - 1]
        $now = Get-NyNow
        if (-not $DryRun -and ($now.ToString('yyyy-MM-dd') -ne $Session -or
            $now.AddSeconds($delay).Date -ne $now.Date -or $now.AddSeconds($delay).TimeOfDay -ge $LateCutoff)) {
            Stop-Launch 7 'preflight retry would reach the 09:20 ET cutoff or change session date'
        }
        Write-Log "retrying read-only preflight in $delay seconds with a fresh child and broker recovery proof"
        Start-Sleep -Seconds $delay
    }
}
