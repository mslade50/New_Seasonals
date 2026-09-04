[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Prepare', 'RegisterDisabled', 'Cutover', 'Status', 'Prune')]
    [string]$Phase,

    [string]$SourceRepository = (Split-Path -Parent $PSScriptRoot),

    [string]$RuntimeRoot,

    [string]$ConfigRoot,

    [ValidatePattern('^[0-9a-fA-F]{40}$')]
    [string]$PinnedSha,

    [ValidatePattern('^[A-Za-z0-9._/-]+$')]
    [string]$FallbackRef,

    [string]$RuntimeBranch = 'codex/local-primary-runtime',

    [string]$TaskNamePrefix = 'New Seasonals Local - ',

    [string]$RetireTaskNamePrefix,

    [string]$BootstrapPython,

    [switch]$ConfirmCutover,

    # Default OFF. Unregisters every disabled, idle `New Seasonals Local*`
    # generation that is neither the current -TaskNamePrefix nor the
    # -RetireTaskNamePrefix rollback set. Pair with -WhatIf for a listing.
    [switch]$PruneSuperseded
)

# Installs the local-primary scheduler in explicit, reversible phases:
#   Prepare          creates a pinned branch worktree and its own venv;
#   RegisterDisabled registers all eight tasks disabled and can safely resume
#                    an exact partially completed disabled set;
#   Cutover          validates/enables them, copies the outgoing generation's
#                    runtime logs into the stable state root, then disables an
#                    optional prior local prefix plus the fixed superseded
#                    tasks (deletes none unless -PruneSuperseded);
#   Status           is read-only;
#   Prune            with -PruneSuperseded deletes disabled, idle, non-current
#                    generations; with -WhatIf it only lists them.
# -WhatIf is honoured by RegisterDisabled, Cutover, and Prune (listing only,
# no elevation required). Nothing in this script runs automatically merely
# because it is checked out.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$PipelineSpecs = @(
    [pscustomobject]@{ Id = 'premarket';       Time = '04:10:00'; DaysMask = 62; Description = 'Weekday premarket cache, risk, signal, and cloud-deploy handoff pipeline' },
    # Local second chance for the premarket run (2026-09-03 stall): re-walks the
    # same receipts; success is a no-op, an expired pre-side-effect lease is
    # re-run, indeterminate is never re-run. Needs nothing from GitHub.
    # 05:45, not 05:30: master_prices_am holds a 70-minute lease, so a 04:10
    # claim is still live at 05:30 and the retry would skip the stalled job.
    [pscustomobject]@{ Id = 'premarket-retry'; Time = '05:45:00'; DaysMask = 62; Description = 'Weekday premarket local second chance (receipt-gated re-run, no GitHub dependency)' },
    [pscustomobject]@{ Id = 'discretionary';   Time = '08:35:00'; DaysMask = 62; Description = 'Weekday research-only discretionary focus pipeline' },
    [pscustomobject]@{ Id = 'execution';       Time = '16:30:00'; DaysMask = 62; Description = 'Weekday execution reporting pipeline' },
    [pscustomobject]@{ Id = 'postclose';       Time = '17:10:00'; DaysMask = 62; Description = 'Weekday post-close data, reports, signals, and cloud-deploy handoff pipeline' },
    [pscustomobject]@{ Id = 'indicator';       Time = '03:00:00'; DaysMask = 2;  Description = 'Monday indicator-cache maintenance pipeline' },
    [pscustomobject]@{ Id = 'weekly-rundown'; Time = '08:00:00'; DaysMask = 1;  Description = 'Sunday weekly market rundown pipeline' },
    [pscustomobject]@{ Id = 'health';          Time = '07:30:00'; DaysMask = 62; Description = 'Weekday receipt, data, delivery, and local-trigger health battery' }
)

$SupersededTasks = @(
    'Trigger CBOE Put-Call (GHA workflow_dispatch)',
    'Trigger Update Master Prices (GHA workflow_dispatch)',
    'Trigger Risk Report AM Correction (GHA workflow_dispatch)',
    'Trigger Daily Screener (GHA workflow_dispatch)',
    'Repo Health Check'
)

if ($PSBoundParameters.ContainsKey('RetireTaskNamePrefix')) {
    if ([string]::IsNullOrWhiteSpace($RetireTaskNamePrefix)) {
        throw 'RetireTaskNamePrefix cannot be empty or whitespace when specified'
    }
    if ($TaskNamePrefix.Equals($RetireTaskNamePrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'TaskNamePrefix and RetireTaskNamePrefix must be different'
    }
}

function Get-SupersededTaskNames {
    $names = New-Object 'System.Collections.Generic.List[string]'
    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    foreach ($name in $SupersededTasks) {
        if ($seen.Add($name)) { $names.Add($name) }
    }
    if (-not [string]::IsNullOrWhiteSpace($RetireTaskNamePrefix)) {
        foreach ($spec in $PipelineSpecs) {
            $name = $RetireTaskNamePrefix + $spec.Id
            if ($seen.Add($name)) { $names.Add($name) }
        }
    }
    return $names.ToArray()
}

# These are versioned snapshots in the source repository but runtime outputs
# for the local-primary jobs. Mark them skip-worktree only inside the dedicated
# operational worktree so data refreshes do not invalidate the pinned-code
# guard. Reference/config files under data/ are deliberately excluded.
$MutableTrackedState = @(
    'data/analyst_grades.parquet',
    'data/cboe_putcall.parquet',
    'data/dial_sleeve_paper.json',
    'data/exposure_state.json',
    'data/fragility_63d_history.parquet',
    'data/rd2_environment.json',
    'data/rd2_fragility_simple.parquet',
    'data/rd2_fragility_ts.parquet',
    'data/rd2_fragility.parquet',
    'data/rd2_spy_ohlc.parquet',
    'data/risk_dashboard_signal_state.json',
    'data/signal_fire_history.parquet'
)

function Resolve-AbsoluteDirectory {
    param([Parameter(Mandatory = $true)][string]$Path, [string]$Label, [switch]$MustExist)
    if (-not [IO.Path]::IsPathRooted($Path)) {
        throw "$Label must be an absolute path: $Path"
    }
    $full = [IO.Path]::GetFullPath($Path)
    if ($MustExist -and -not (Test-Path -LiteralPath $full -PathType Container)) {
        throw "$Label does not exist: $full"
    }
    if ($MustExist) {
        return (Resolve-Path -LiteralPath $full).Path
    }
    return $full.TrimEnd('\')
}

function Assert-EasternLocalClock {
    if ([TimeZoneInfo]::Local.Id -ne 'Eastern Standard Time') {
        throw "Task triggers use the machine's local clock; expected Eastern Standard Time, found $([TimeZoneInfo]::Local.Id)"
    }
}

function Assert-Administrator {
    if ([bool]$WhatIfPreference) {
        # A -WhatIf listing mutates nothing and reads the scheduler as the
        # current user; elevation is required only for the real change.
        return
    }
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'RegisterDisabled, Cutover and Prune must run from an elevated Windows PowerShell session'
    }
}

function Get-StableStateRoot {
    # Mirrors scripts/run_local_automation.ps1: runtime logs, the supervisor
    # lock, and the health receipt cache live under the CONFIG root so they
    # survive the deletion of a retired runtime worktree.
    return (Join-Path $script:ConfigRoot 'artifacts\automation')
}

function Get-RuntimeRootFromTask {
    param([Parameter(Mandatory = $true)]$Task)
    # The registered action carries -RuntimeRoot "<path>"; that is the only
    # durable record of where a retired generation wrote its logs.
    try {
        $definition = $Task.Definition
        if ([int]$definition.Actions.Count -lt 1) { return $null }
        $arguments = [string]$definition.Actions.Item(1).Arguments
    }
    catch {
        return $null
    }
    if ($arguments -match '-RuntimeRoot "([^"]+)"') {
        return [IO.Path]::GetFullPath($Matches[1]).TrimEnd('\')
    }
    return $null
}

function Copy-RetiredRuntimeLogs {
    param([Parameter(Mandatory = $true)]$RootFolder)
    # F4 (2026-09-03): every cutover left the previous worktree, and with it
    # the only record of the failed 04:10 run, to be deleted by hand. Copy the
    # outgoing generation's in-worktree logs into the stable state root first.
    # Existing files are never overwritten; a missing source is not an error.
    $stableLogs = Join-Path (Get-StableStateRoot) 'logs'
    $sources = New-Object 'System.Collections.Generic.List[string]'
    $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
    if (-not [string]::IsNullOrWhiteSpace($RetireTaskNamePrefix)) {
        foreach ($spec in $PipelineSpecs) {
            $taskName = $RetireTaskNamePrefix + $spec.Id
            $task = Get-TaskOrNull -RootFolder $RootFolder -Name $taskName
            if ($null -eq $task) { continue }
            $root = Get-RuntimeRootFromTask -Task $task
            if ($root) {
                if ($seen.Add($root)) { $sources.Add($root) }
            }
            else {
                # The registered action is the ONLY durable record of where a
                # retired generation wrote its logs. Losing that must be loud,
                # not a silent drop from the source list (2026-09-04 verify).
                Write-Output "Log copy-forward: retired task '$taskName' carries no quoted -RuntimeRoot in its action; its runtime logs cannot be located and are NOT copied forward"
            }
        }
    }
    # The generation being enabled may itself hold in-worktree logs from a
    # pre-stable-root runtime (v8 wrote under its own artifacts/automation).
    if ($seen.Add($script:RuntimeRoot)) { $sources.Add($script:RuntimeRoot) }
    foreach ($root in $sources) {
        # Counters are per SOURCE: a run that copies six files from a retired
        # v8 worktree and none from the incoming one must say so, not report a
        # running total against the last path it printed (2026-09-04 verify).
        $copied = 0
        $skipped = 0
        $sourceLogs = Join-Path $root 'artifacts\automation\logs'
        if ($sourceLogs.Equals($stableLogs, [StringComparison]::OrdinalIgnoreCase)) { continue }
        # The cutover-state mirror runs BEFORE the missing-logs guard: the
        # generation being enabled is a freshly prepared worktree that has
        # just had its cutover-state.json written and has no logs directory at
        # all (it writes its logs to the stable state root), so a mirror block
        # after the guard never ran for it (2026-09-04 verify, finding 1).
        $stateFile = Join-Path $root 'artifacts\automation\cutover-state.json'
        if (Test-Path -LiteralPath $stateFile -PathType Leaf) {
            $historyDir = Join-Path (Get-StableStateRoot) 'cutovers'
            $historyTarget = Join-Path $historyDir ((Split-Path -Leaf $root) + '-cutover-state.json')
            if (-not (Test-Path -LiteralPath $historyTarget -PathType Leaf)) {
                if ([bool]$WhatIfPreference) {
                    Write-Output "WhatIf: would copy $stateFile -> $historyTarget"
                }
                else {
                    New-Item -ItemType Directory -Path $historyDir -Force | Out-Null
                    Copy-Item -LiteralPath $stateFile -Destination $historyTarget
                }
            }
        }
        if (-not (Test-Path -LiteralPath $sourceLogs -PathType Container)) {
            Write-Output "Log copy-forward: no runtime logs under $sourceLogs (nothing to copy)"
            continue
        }
        $files = @(Get-ChildItem -LiteralPath $sourceLogs -Recurse -File)
        foreach ($file in $files) {
            $relative = $file.FullName.Substring($sourceLogs.Length).TrimStart('\')
            $target = Join-Path $stableLogs $relative
            if (Test-Path -LiteralPath $target -PathType Leaf) { $skipped += 1; continue }
            if ([bool]$WhatIfPreference) {
                Write-Output "WhatIf: would copy $($file.FullName) -> $target"
                $copied += 1
                continue
            }
            $targetDir = Split-Path -Parent $target
            if (-not (Test-Path -LiteralPath $targetDir -PathType Container)) {
                New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
            }
            Copy-Item -LiteralPath $file.FullName -Destination $target
            $copied += 1
        }
        $copyVerb = if ([bool]$WhatIfPreference) { 'would be copied' } else { 'copied' }
        Write-Output "Log copy-forward from $sourceLogs -> $stableLogs : $copied file(s) $copyVerb, $skipped already present"
    }
}

function Get-PruneCandidates {
    param([Parameter(Mandatory = $true)]$RootFolder)
    # Every registered task that looks like a local-primary generation
    # ('New Seasonals Local - x' for v1, 'New Seasonals Local vN - x' after)
    # and is not the current prefix. The -RetireTaskNamePrefix generation is
    # the rollback path and is protected. Enabled or non-idle tasks are never
    # deletion candidates: enabled means somebody's live writer.
    $rows = @()
    foreach ($task in @($RootFolder.GetTasks(1))) {
        $name = [string]$task.Name
        if ($name -notmatch '^New Seasonals Local( v\d+)? - ') { continue }
        if ($name.StartsWith($TaskNamePrefix, [StringComparison]::OrdinalIgnoreCase)) { continue }
        $enabled = [bool]$task.Enabled
        $state = [int]$task.State
        $action = 'delete'
        if (-not [string]::IsNullOrWhiteSpace($RetireTaskNamePrefix) -and
            $name.StartsWith($RetireTaskNamePrefix, [StringComparison]::OrdinalIgnoreCase)) {
            $action = 'keep: rollback generation (-RetireTaskNamePrefix)'
        }
        elseif ($enabled) {
            $action = 'keep: ENABLED (never pruned; disable it through a cutover first)'
        }
        elseif ($state -eq 2 -or $state -eq 4) {
            $action = 'keep: queued or running'
        }
        $rows += [pscustomobject]@{ Name = $name; Enabled = $enabled; State = $state; Action = $action }
    }
    return @($rows | Sort-Object Name)
}

function Invoke-Prune {
    if (-not $PruneSuperseded) {
        throw 'Prune requires the explicit -PruneSuperseded switch (add -PruneSuperseded -WhatIf to list without deleting)'
    }
    Assert-Administrator
    $scheduler = Connect-TaskScheduler
    $candidates = Get-PruneCandidates -RootFolder $scheduler.Root
    if ($candidates.Count -eq 0) {
        Write-Output "Prune: no superseded 'New Seasonals Local*' tasks outside prefix '$TaskNamePrefix'."
        return
    }
    $deleted = 0
    foreach ($row in $candidates) {
        if ($row.Action -ne 'delete') {
            Write-Output "Prune: $($row.Name) : $($row.Action)"
            continue
        }
        if ([bool]$WhatIfPreference) {
            Write-Output "WhatIf: would unregister disabled task '$($row.Name)'"
            $deleted += 1
            continue
        }
        # Re-read immediately before the mutation; refuse if anything changed.
        $current = $scheduler.Root.GetTask($row.Name)
        if ([bool]$current.Enabled) { throw "Prune refused: task became enabled: $($row.Name)" }
        Assert-TaskIdle -Task $current -Label "prune candidate $($row.Name)"
        $scheduler.Root.DeleteTask($row.Name, 0)
        if ($null -ne (Get-TaskOrNull -RootFolder $scheduler.Root -Name $row.Name)) {
            throw "Prune failed: task still registered after DeleteTask: $($row.Name)"
        }
        Write-Output "Unregistered disabled task: $($row.Name)"
        $deleted += 1
    }
    $verb = if ([bool]$WhatIfPreference) { 'would unregister' } else { 'unregistered' }
    Write-Output "Prune complete: $verb $deleted task(s); current prefix '$TaskNamePrefix' untouched."
}

function Invoke-GitCapture {
    param([Parameter(Mandatory = $true)][string[]]$Arguments, [switch]$AllowExitOne)
    # Windows PowerShell 5.1 wraps native stderr in a non-terminating
    # NativeCommandError. With the installer's global Stop preference, Git's
    # normal progress messages (for example "Preparing worktree") otherwise
    # abort a successful command before LASTEXITCODE can be inspected.
    $priorErrorActionPreference = $ErrorActionPreference
    $lines = @()
    $code = 1
    try {
        $ErrorActionPreference = 'Continue'
        $lines = @(& $script:GitExecutable @Arguments 2>&1)
        $code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $priorErrorActionPreference
    }
    $output = ($lines | Out-String).Trim()
    if ($code -ne 0 -and -not ($AllowExitOne -and $code -eq 1)) {
        throw "Git command failed (exit $code): git $($Arguments -join ' ')`n$output"
    }
    return [pscustomobject]@{ Output = $output; ExitCode = $code }
}

function Assert-PinnedRuntime {
    $markerPath = Join-Path $script:RuntimeRoot '.local\automation-runtime.json'
    if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
        throw "Runtime is not prepared: $markerPath is missing"
    }
    $marker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json
    $head = (Invoke-GitCapture -Arguments @('-C', $script:RuntimeRoot, 'rev-parse', 'HEAD')).Output
    if (-not $head.Equals([string]$marker.pinned_sha, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Runtime HEAD $head differs from pinned SHA $($marker.pinned_sha)"
    }
    if ([string]::IsNullOrWhiteSpace([string]$marker.fallback_ref)) {
        throw 'Runtime marker has no immutable GitHub fallback ref'
    }
    $fallbackCommit = (Invoke-GitCapture -Arguments @(
        '-C', $script:RuntimeRoot, 'rev-parse', "$([string]$marker.fallback_ref)^{commit}"
    )).Output
    if (-not $fallbackCommit.Equals([string]$marker.pinned_sha, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Fallback ref $($marker.fallback_ref) resolves to $fallbackCommit, not pinned SHA $($marker.pinned_sha)"
    }
    $dirty = (Invoke-GitCapture -Arguments @('-C', $script:RuntimeRoot, 'status', '--porcelain', '--untracked-files=no')).Output
    if ($dirty) {
        throw 'Runtime worktree has tracked changes'
    }
    foreach ($path in $MutableTrackedState) {
        $entry = (Invoke-GitCapture -Arguments @(
            '-C', $script:RuntimeRoot, 'ls-files', '-v', '--', $path
        )).Output
        if ($entry -and -not $entry.StartsWith('S ')) {
            throw "Mutable runtime state is not protected with skip-worktree: $path"
        }
    }
    if (-not ([IO.Path]::GetFullPath([string]$marker.config_root)).Equals($script:ConfigRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Runtime marker ConfigRoot differs from the requested ConfigRoot'
    }

    $runner = Join-Path $script:RuntimeRoot 'scripts\run_local_automation.ps1'
    $powershell = 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
    if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) { throw "Runner missing: $runner" }
    if (-not (Test-Path -LiteralPath $powershell -PathType Leaf)) { throw "Windows PowerShell missing: $powershell" }
    # Capture the validator's success-stream message instead of allowing it to
    # become a second function return value alongside $marker. Cutover needs
    # Assert-PinnedRuntime to return exactly one marker object.
    $validationLines = @(& $powershell -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $runner `
        -Pipeline premarket -RuntimeRoot $script:RuntimeRoot -ConfigRoot $script:ConfigRoot -ValidateOnly 2>&1)
    $validationExitCode = $LASTEXITCODE
    $validationOutput = ($validationLines | Out-String).Trim()
    if ($validationExitCode -ne 0) {
        throw "Pinned runtime validation failed with exit code $validationExitCode`n$validationOutput"
    }
    return $marker
}

function Connect-TaskScheduler {
    $service = New-Object -ComObject 'Schedule.Service'
    $service.Connect()
    return [pscustomobject]@{ Service = $service; Root = $service.GetFolder('\') }
}

function Get-TaskOrNull {
    param($RootFolder, [string]$Name)
    try { return $RootFolder.GetTask($Name) }
    catch {
        # ITaskFolder::GetTask reports a missing task as HRESULT 0x80070002.
        # Access, RPC, and scheduler-service failures must abort cutover; treating
        # them as "not present" could leave a legacy writer enabled.
        if ($_.Exception.HResult -eq -2147024894) { return $null }
        throw
    }
}

function Assert-TaskIdle {
    param(
        [Parameter(Mandatory = $true)]$Task,
        [Parameter(Mandatory = $true)][string]$Label
    )
    # TASK_STATE_QUEUED=2 and TASK_STATE_RUNNING=4. Disabling a task does not
    # stop an existing instance, so crossing either state during cutover could
    # leave old and new writers active at the same time.
    $state = [int]$Task.State
    if ($state -eq 2 -or $state -eq 4) {
        throw "Cutover requires every task to be idle; $Label is queued or running (state=$state)"
    }
}

function Quote-TaskArgument {
    param([string]$Value)
    if ($Value.Contains('"')) { throw 'Task arguments may not contain a double quote' }
    return '"' + $Value + '"'
}

function Get-ExpectedTaskArguments {
    param([Parameter(Mandatory = $true)]$Spec)
    $runner = Join-Path $script:RuntimeRoot 'scripts\run_local_automation.ps1'
    return @(
        '-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
        '-File', (Quote-TaskArgument $runner),
        '-Pipeline', (Quote-TaskArgument $Spec.Id),
        '-RuntimeRoot', (Quote-TaskArgument $script:RuntimeRoot),
        '-ConfigRoot', (Quote-TaskArgument $script:ConfigRoot)
    ) -join ' '
}

function Resolve-AccountSid {
    param([Parameter(Mandatory = $true)][string]$Account)
    if ($Account -match '^S-1-') { return $Account }
    try {
        $ntAccount = New-Object Security.Principal.NTAccount($Account)
        return ($ntAccount.Translate([Security.Principal.SecurityIdentifier])).Value
    }
    catch {
        throw "Unable to resolve scheduled-task account '$Account' to a SID: $($_.Exception.Message)"
    }
}

function Assert-RegisteredTaskDefinition {
    param(
        [Parameter(Mandatory = $true)]$Task,
        [Parameter(Mandatory = $true)]$Spec,
        [Parameter(Mandatory = $true)][string]$IdentitySid
    )
    $definition = $Task.Definition
    $expectedPowerShell = 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
    $registeredSid = Resolve-AccountSid -Account ([string]$definition.Principal.UserId)
    if (-not $registeredSid.Equals($IdentitySid, [StringComparison]::OrdinalIgnoreCase) -or
        [int]$definition.Principal.LogonType -ne 2 -or
        [int]$definition.Principal.RunLevel -ne 1) {
        throw "Task principal differs from the guarded S4U definition: $($Task.Name)"
    }
    if ([int]$definition.Actions.Count -ne 1) {
        throw "Task must have exactly one action: $($Task.Name)"
    }
    $action = $definition.Actions.Item(1)
    if (-not ([IO.Path]::GetFullPath([string]$action.Path)).Equals($expectedPowerShell, [StringComparison]::OrdinalIgnoreCase) -or
        -not ([IO.Path]::GetFullPath([string]$action.WorkingDirectory)).Equals($script:RuntimeRoot, [StringComparison]::OrdinalIgnoreCase) -or
        [string]$action.Arguments -cne (Get-ExpectedTaskArguments -Spec $Spec)) {
        throw "Task action differs from the pinned runtime definition: $($Task.Name)"
    }
    if ([int]$definition.Triggers.Count -ne 1) {
        throw "Task must have exactly one trigger: $($Task.Name)"
    }
    $trigger = $definition.Triggers.Item(1)
    $expectedAt = [DateTime]::ParseExact($Spec.Time, 'HH:mm:ss', [Globalization.CultureInfo]::InvariantCulture)
    $actualAt = [DateTime]::Parse([string]$trigger.StartBoundary, [Globalization.CultureInfo]::InvariantCulture)
    if ([int]$trigger.Type -ne 3 -or
        [int]$trigger.DaysOfWeek -ne [int]$Spec.DaysMask -or
        [int]$trigger.WeeksInterval -ne 1 -or
        -not [bool]$trigger.Enabled -or
        $actualAt.TimeOfDay -ne $expectedAt.TimeOfDay) {
        throw "Task trigger differs from the expected Eastern schedule: $($Task.Name)"
    }
    $settings = $definition.Settings
    if (-not [bool]$settings.WakeToRun -or
        -not [bool]$settings.StartWhenAvailable -or
        [int]$settings.RestartCount -ne 3 -or
        [string]$settings.RestartInterval -ne 'PT5M' -or
        [int]$settings.MultipleInstances -ne 2 -or
        [string]$settings.ExecutionTimeLimit -ne 'PT6H' -or
        [bool]$settings.DisallowStartIfOnBatteries -or
        [bool]$settings.StopIfGoingOnBatteries) {
        throw "Task settings differ from the guarded definition: $($Task.Name)"
    }
}

function Register-DisabledTasks {
    Assert-Administrator
    Assert-EasternLocalClock
    $null = Assert-PinnedRuntime
    $scheduler = Connect-TaskScheduler
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $identityName = $identity.Name
    $identitySid = $identity.User.Value
    $windowsPowerShell = 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
    $runner = Join-Path $script:RuntimeRoot 'scripts\run_local_automation.ps1'

    foreach ($spec in $PipelineSpecs) {
        $taskName = $TaskNamePrefix + $spec.Id
        $existing = Get-TaskOrNull -RootFolder $scheduler.Root -Name $taskName
        if ([bool]$WhatIfPreference) {
            if ($null -ne $existing) {
                Write-Output "WhatIf: already registered (enabled=$($existing.Enabled)); would validate, not overwrite: $taskName"
            }
            else {
                Write-Output "WhatIf: would register disabled S4U task '$taskName' at $($spec.Time) local Eastern, DaysOfWeek mask $($spec.DaysMask): $(Get-ExpectedTaskArguments -Spec $spec)"
            }
            continue
        }
        if ($null -ne $existing) {
            # A prior interrupted registration may have completed a prefix of
            # the set. Resume only across exact, still-disabled definitions;
            # never overwrite, repair, or silently adopt a conflicting task.
            Assert-RegisteredTaskDefinition -Task $existing -Spec $spec -IdentitySid $identitySid
            if ($existing.Enabled) {
                throw "Existing guarded task is enabled; refusing registration resume: $taskName"
            }
            Write-Output "Already registered disabled (validated): $taskName"
            continue
        }
        $definition = $scheduler.Service.NewTask(0)
        $definition.RegistrationInfo.Author = $identityName
        $definition.RegistrationInfo.Description = $spec.Description + ' (local Eastern time; registered disabled)'

        # S4U requires no stored password, preserves the user's local profile,
        # and is already proven on this machine's existing unattended trigger
        # tasks. API authentication is explicit process state, not Windows
        # integrated network authentication. Registration requires elevation.
        $definition.Principal.UserId = $identityName
        $definition.Principal.LogonType = 2
        $definition.Principal.RunLevel = 1

        $definition.Settings.Enabled = $false
        $definition.Settings.WakeToRun = $true
        $definition.Settings.StartWhenAvailable = $true
        $definition.Settings.RestartCount = 3
        $definition.Settings.RestartInterval = 'PT5M'
        $definition.Settings.MultipleInstances = 2
        $definition.Settings.ExecutionTimeLimit = 'PT6H'
        $definition.Settings.DisallowStartIfOnBatteries = $false
        $definition.Settings.StopIfGoingOnBatteries = $false

        # TASK_TRIGGER_WEEKLY. StartBoundary has no offset by design: Windows
        # Task Scheduler evaluates it against this Eastern-time machine's local
        # clock and follows daylight-saving transitions.
        $trigger = $definition.Triggers.Create(3)
        $at = [DateTime]::ParseExact($spec.Time, 'HH:mm:ss', [Globalization.CultureInfo]::InvariantCulture)
        $trigger.StartBoundary = (Get-Date -Hour $at.Hour -Minute $at.Minute -Second 0).ToString('s')
        $trigger.WeeksInterval = 1
        $trigger.DaysOfWeek = $spec.DaysMask
        $trigger.Enabled = $true

        $action = $definition.Actions.Create(0)
        $action.Path = $windowsPowerShell
        $action.WorkingDirectory = $script:RuntimeRoot
        $action.Arguments = Get-ExpectedTaskArguments -Spec $spec

        # TASK_CREATE + TASK_LOGON_S4U.  TASK_CREATE (not
        # CREATE_OR_UPDATE) is intentional: this installer never overwrites or
        # deletes an existing task.
        $null = $scheduler.Root.RegisterTaskDefinition($taskName, $definition, 2, $identityName, $null, 2, $null)
        $registered = $scheduler.Root.GetTask($taskName)
        if ($registered.Enabled) {
            throw "Task was not registered disabled: $taskName"
        }
        Assert-RegisteredTaskDefinition -Task $registered -Spec $spec -IdentitySid $identitySid
        Write-Output "Registered disabled: $taskName"
    }
}

function Show-CutoverPlan {
    param([Parameter(Mandatory = $true)]$RootFolder)
    # -WhatIf view of Cutover: what would be enabled, disabled, copied, pruned.
    foreach ($spec in $PipelineSpecs) {
        $taskName = $TaskNamePrefix + $spec.Id
        $task = Get-TaskOrNull -RootFolder $RootFolder -Name $taskName
        if ($null -eq $task) { Write-Output "WhatIf: would FAIL, required registered task is missing: $taskName" }
        else { Write-Output "WhatIf: would enable $taskName (currently enabled=$($task.Enabled))" }
    }
    foreach ($name in (Get-SupersededTaskNames)) {
        $task = Get-TaskOrNull -RootFolder $RootFolder -Name $name
        if ($null -eq $task) { Write-Output "WhatIf: superseded task not present (nothing changed): $name" }
        else { Write-Output "WhatIf: would disable superseded task $name (currently enabled=$($task.Enabled))" }
    }
    # Same order as the real Cutover: the incoming cutover-state.json is
    # written before the log copy-forward, and the copy-forward mirrors each
    # source's cutover-state.json before its missing-logs guard, so a freshly
    # prepared runtime with no logs directory is still mirrored in this run.
    # -WhatIf writes nothing, so the incoming file does not exist yet and only
    # the RETIRED generations' state files appear in the listing below.
    Write-Output "WhatIf: cutover-state.json would be written under $(Join-Path $script:RuntimeRoot 'artifacts\automation') first, then mirrored to $(Join-Path (Get-StableStateRoot) 'cutovers') by the log copy-forward below; a -WhatIf run does not write it, so only already-existing state files are listed"
    Copy-RetiredRuntimeLogs -RootFolder $RootFolder
    if ($PruneSuperseded) {
        foreach ($row in (Get-PruneCandidates -RootFolder $RootFolder)) {
            if ($row.Action -eq 'delete') { Write-Output "WhatIf: would unregister disabled task '$($row.Name)' after cutover" }
            else { Write-Output "WhatIf: prune keeps $($row.Name) : $($row.Action)" }
        }
    }
}

function Invoke-Cutover {
    if (-not $ConfirmCutover) {
        throw 'Cutover requires the explicit -ConfirmCutover switch'
    }
    Assert-Administrator
    Assert-EasternLocalClock
    $marker = Assert-PinnedRuntime
    $scheduler = Connect-TaskScheduler
    if ([bool]$WhatIfPreference) {
        Show-CutoverPlan -RootFolder $scheduler.Root
        return
    }
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $identityName = $identity.Name
    $identitySid = $identity.User.Value
    $newTaskState = @()
    foreach ($spec in $PipelineSpecs) {
        $taskName = $TaskNamePrefix + $spec.Id
        $task = Get-TaskOrNull -RootFolder $scheduler.Root -Name $taskName
        if ($null -eq $task) { throw "Required registered task is missing: $taskName" }
        Assert-RegisteredTaskDefinition -Task $task -Spec $spec -IdentitySid $identitySid
        $newTaskState += [pscustomobject]@{
            task = $task
            name = $taskName
            enabled = [bool]$task.Enabled
        }
    }

    $supersededState = @()
    foreach ($name in (Get-SupersededTaskNames)) {
        $task = Get-TaskOrNull -RootFolder $scheduler.Root -Name $name
        $supersededState += [pscustomobject]@{
            name = $name
            present = ($null -ne $task)
            enabled = if ($null -ne $task) { [bool]$task.Enabled } else { $false }
        }
    }

    # Re-read and gate every task immediately before the first enabled-state
    # mutation. This deliberately narrows the race after the definition and
    # state snapshots above; the receipt contract remains the second line of
    # defense if Windows changes state after this check.
    foreach ($entry in $newTaskState) {
        $current = $scheduler.Root.GetTask($entry.name)
        Assert-TaskIdle -Task $current -Label "new task $($entry.name)"
    }
    foreach ($entry in $supersededState) {
        if (-not $entry.present) { continue }
        $current = $scheduler.Root.GetTask($entry.name)
        Assert-TaskIdle -Task $current -Label "superseded task $($entry.name)"
    }
    # Write the incoming generation's cutover-state.json FIRST, then copy
    # logs forward: Copy-RetiredRuntimeLogs mirrors every source's
    # cutover-state.json into the stable state root, and the incoming runtime
    # is one of its sources. Written afterwards it would only be mirrored at
    # the NEXT cutover, so this generation's own record was missing from
    # <state root>\cutovers for its whole life (2026-09-04 verify).
    $stateDir = Join-Path $script:RuntimeRoot 'artifacts\automation'
    New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
    [pscustomobject]@{
        cutover_at = [DateTime]::UtcNow.ToString('o')
        pinned_sha = [string]$marker.pinned_sha
        new_tasks = @($newTaskState | ForEach-Object {
            [pscustomobject]@{ name = $_.name; enabled = $_.enabled }
        })
        legacy_tasks = $supersededState
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $stateDir 'cutover-state.json') -Encoding UTF8

    # Every task is idle, so no log is mid-write: preserve the outgoing
    # generation's runtime logs (and both generations' cutover state) before
    # any writer changes hands.
    Copy-RetiredRuntimeLogs -RootFolder $scheduler.Root

    # Availability first: enable and verify every local primary task before
    # disabling superseded tasks. Any exception rolls both sets back to their
    # pre-cutover enabled states, avoiding a half-cutover with duplicate writers.
    try {
        foreach ($entry in $newTaskState) { $entry.task.Enabled = $true }
        foreach ($entry in $newTaskState) {
            if (-not $entry.task.Enabled) { throw "Failed to enable new task: $($entry.name)" }
        }

        foreach ($entry in $supersededState) {
            if (-not $entry.present) {
                Write-Output "Superseded task not present (nothing changed): $($entry.name)"
                continue
            }
            $legacy = $scheduler.Root.GetTask($entry.name)
            $legacy.Enabled = $false
            if ($legacy.Enabled) { throw "Failed to disable superseded task: $($entry.name)" }
            Write-Output "Disabled superseded task: $($entry.name)"
        }
    }
    catch {
        $cutoverError = $_.Exception.Message
        $rollbackErrors = @()
        foreach ($entry in $newTaskState) {
            try { $entry.task.Enabled = [bool]$entry.enabled }
            catch { $rollbackErrors += "new task $($entry.name): $($_.Exception.Message)" }
        }
        foreach ($entry in $supersededState) {
            if (-not $entry.present) { continue }
            try {
                $legacy = $scheduler.Root.GetTask($entry.name)
                $legacy.Enabled = [bool]$entry.enabled
            }
            catch { $rollbackErrors += "superseded task $($entry.name): $($_.Exception.Message)" }
        }
        if ($rollbackErrors.Count -gt 0) {
            throw "Cutover failed ($cutoverError); rollback also had errors: $($rollbackErrors -join '; ')"
        }
        throw "Cutover failed and task enabled states were rolled back: $cutoverError"
    }
    Write-Output "Cutover complete at pinned commit $($marker.pinned_sha). No tasks were deleted."
    if ($PruneSuperseded) {
        # Only after a fully successful cutover, and never the rollback set.
        Invoke-Prune
    }
}

function Show-TaskStatus {
    $scheduler = Connect-TaskScheduler
    foreach ($spec in $PipelineSpecs) {
        $name = $TaskNamePrefix + $spec.Id
        $task = Get-TaskOrNull -RootFolder $scheduler.Root -Name $name
        if ($null -eq $task) {
            Write-Output "$name : NOT REGISTERED"
        }
        else {
            Write-Output "$name : enabled=$($task.Enabled) next=$($task.NextRunTime) lastResult=$($task.LastTaskResult)"
        }
    }
    foreach ($name in (Get-SupersededTaskNames)) {
        $task = Get-TaskOrNull -RootFolder $scheduler.Root -Name $name
        if ($null -eq $task) { Write-Output "$name : NOT PRESENT" }
        else { Write-Output "$name : enabled=$($task.Enabled)" }
    }
    Write-Output "Stable state root (logs, lock, health receipts): $(Get-StableStateRoot)"
    $others = @(Get-PruneCandidates -RootFolder $scheduler.Root)
    Write-Output "Other 'New Seasonals Local*' generations registered: $($others.Count) (list with -Phase Prune -PruneSuperseded -WhatIf)"
}

$SourceRepository = Resolve-AbsoluteDirectory -Path $SourceRepository -Label 'SourceRepository' -MustExist
if (-not $RuntimeRoot) {
    $RuntimeRoot = Join-Path (Split-Path -Parent $SourceRepository) 'New_Seasonals-automation-runtime'
}
$RuntimeRoot = Resolve-AbsoluteDirectory -Path $RuntimeRoot -Label 'RuntimeRoot'
if (-not $ConfigRoot) { $ConfigRoot = $SourceRepository }
$ConfigRoot = Resolve-AbsoluteDirectory -Path $ConfigRoot -Label 'ConfigRoot' -MustExist

$gitCommand = Get-Command 'git.exe' -ErrorAction Stop
$script:GitExecutable = $gitCommand.Source
$script:RuntimeRoot = $RuntimeRoot
$script:ConfigRoot = $ConfigRoot

if ($PruneSuperseded -and $Phase -notin @('Cutover', 'Prune')) {
    throw '-PruneSuperseded is only meaningful with -Phase Cutover or -Phase Prune'
}

switch ($Phase) {
    'Prepare' {
        if ([bool]$WhatIfPreference) { throw 'Prepare does not support -WhatIf (it creates a worktree and a venv)' }
        if (-not $PinnedSha) { throw 'Prepare requires -PinnedSha with the tested origin/main commit' }
        if (-not $FallbackRef) { throw 'Prepare requires -FallbackRef with an immutable remote tag at PinnedSha' }
        if (-not (Test-Path -LiteralPath (Join-Path $SourceRepository '.git'))) {
            throw "SourceRepository is not a Git worktree: $SourceRepository"
        }
        if (-not (Test-Path -LiteralPath (Join-Path $ConfigRoot '.env') -PathType Leaf)) {
            throw "ConfigRoot must contain the machine-local .env file: $ConfigRoot"
        }

        $null = Invoke-GitCapture -Arguments @('-C', $SourceRepository, 'fetch', '--quiet', 'origin', 'main')
        $originMain = (Invoke-GitCapture -Arguments @('-C', $SourceRepository, 'rev-parse', 'origin/main')).Output
        if (-not $originMain.Equals($PinnedSha, [StringComparison]::OrdinalIgnoreCase)) {
            throw "PinnedSha must equal verified origin/main (origin/main=$originMain, requested=$PinnedSha)"
        }
        $remoteTag = (Invoke-GitCapture -Arguments @(
            '-C', $SourceRepository, 'ls-remote', '--exit-code', '--refs', 'origin', "refs/tags/$FallbackRef"
        )).Output
        if (-not $remoteTag) {
            throw "FallbackRef is not an existing remote tag: $FallbackRef"
        }
        # Require a lightweight immutable tag whose remote object is the exact
        # tested commit. A stale local tag is not evidence for what GitHub will
        # execute when `gh workflow run --ref` resolves the remote ref.
        $remoteTagCommit = ($remoteTag -split '\s+')[0]
        if (-not $remoteTagCommit.Equals($PinnedSha, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Remote FallbackRef $FallbackRef resolves to $remoteTagCommit, not PinnedSha $PinnedSha"
        }
        $localTag = Invoke-GitCapture -Arguments @(
            '-C', $SourceRepository, 'show-ref', '--verify', '--quiet', "refs/tags/$FallbackRef"
        ) -AllowExitOne
        if ($localTag.ExitCode -ne 0) {
            $null = Invoke-GitCapture -Arguments @(
                '-C', $SourceRepository, 'fetch', '--quiet', 'origin', "refs/tags/$FallbackRef:refs/tags/$FallbackRef"
            )
        }
        $fallbackCommit = (Invoke-GitCapture -Arguments @(
            '-C', $SourceRepository, 'rev-parse', "$FallbackRef^{commit}"
        )).Output
        if (-not $fallbackCommit.Equals($PinnedSha, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Local FallbackRef $FallbackRef resolves to $fallbackCommit, not PinnedSha $PinnedSha"
        }

        if (Test-Path -LiteralPath $RuntimeRoot) {
            # A failed package download may leave a correctly pinned worktree
            # without a complete venv. Resume that exact state, but never move,
            # replace, or repair a different checkout implicitly.
            if (-not (Test-Path -LiteralPath (Join-Path $RuntimeRoot '.git'))) {
                throw "RuntimeRoot exists but is not the expected Git worktree: $RuntimeRoot"
            }
            $existingHead = (Invoke-GitCapture -Arguments @('-C', $RuntimeRoot, 'rev-parse', 'HEAD')).Output
            $existingBranch = (Invoke-GitCapture -Arguments @('-C', $RuntimeRoot, 'branch', '--show-current')).Output
            $existingDirty = (Invoke-GitCapture -Arguments @(
                '-C', $RuntimeRoot, 'status', '--porcelain', '--untracked-files=no'
            )).Output
            if (-not $existingHead.Equals($PinnedSha, [StringComparison]::OrdinalIgnoreCase) -or
                $existingBranch -ne $RuntimeBranch -or $existingDirty) {
                throw "Existing RuntimeRoot is not the clean requested branch/SHA; refusing implicit replacement: $RuntimeRoot"
            }
        }
        else {
            $branchCheck = Invoke-GitCapture -Arguments @(
                '-C', $SourceRepository, 'show-ref', '--verify', '--quiet', "refs/heads/$RuntimeBranch"
            ) -AllowExitOne
            if ($branchCheck.ExitCode -eq 0) {
                throw "Runtime branch already exists; refusing to replace it: $RuntimeBranch"
            }

            $null = Invoke-GitCapture -Arguments @(
                '-C', $SourceRepository, 'worktree', 'add', '-b', $RuntimeBranch, $RuntimeRoot, $PinnedSha
            )
        }

        if (-not $BootstrapPython) {
            $BootstrapPython = (Get-Command 'python.exe' -ErrorAction Stop).Source
        }
        if (-not [IO.Path]::IsPathRooted($BootstrapPython) -or -not (Test-Path -LiteralPath $BootstrapPython -PathType Leaf)) {
            throw "BootstrapPython must be an absolute existing executable: $BootstrapPython"
        }

        $existingMutable = @()
        foreach ($path in $MutableTrackedState) {
            $tracked = (Invoke-GitCapture -Arguments @(
                '-C', $script:RuntimeRoot, 'ls-files', '--', $path
            )).Output
            if ($tracked) { $existingMutable += $path }
        }
        if ($existingMutable.Count -gt 0) {
            $indexArguments = @(
                '-C', $script:RuntimeRoot, 'update-index', '--skip-worktree', '--'
            ) + $existingMutable
            $null = Invoke-GitCapture -Arguments $indexArguments
        }
        $venvPython = Join-Path $RuntimeRoot '.venv\Scripts\python.exe'
        if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
            & $BootstrapPython -m venv (Join-Path $RuntimeRoot '.venv')
            if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
                throw 'Dedicated automation virtual environment creation failed'
            }
        }
        & $venvPython -m pip install --disable-pip-version-check -r (Join-Path $RuntimeRoot 'scripts\requirements-automation.txt')
        if ($LASTEXITCODE -ne 0) { throw 'Dedicated automation dependency installation failed' }

        $markerDir = Join-Path $RuntimeRoot '.local'
        New-Item -ItemType Directory -Path $markerDir -Force | Out-Null
        [pscustomobject]@{
            mode = 'pinned-local-automation-runtime'
            pinned_sha = $PinnedSha.ToLowerInvariant()
            fallback_ref = $FallbackRef
            runtime_branch = $RuntimeBranch
            runtime_root = $RuntimeRoot
            config_root = $ConfigRoot
            git_executable = $script:GitExecutable
            mutable_tracked_state = $MutableTrackedState
            prepared_at = [DateTime]::UtcNow.ToString('o')
        } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $markerDir 'automation-runtime.json') -Encoding UTF8

        $null = Assert-PinnedRuntime
        Write-Output "Prepared pinned runtime $RuntimeRoot at $PinnedSha. No tasks were registered."
    }
    'RegisterDisabled' { Register-DisabledTasks }
    'Cutover' { Invoke-Cutover }
    'Status' { Show-TaskStatus }
    'Prune' { Invoke-Prune }
}
