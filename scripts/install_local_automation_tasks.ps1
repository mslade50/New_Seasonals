[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Prepare', 'RegisterDisabled', 'Cutover', 'Status')]
    [string]$Phase,

    [string]$SourceRepository = (Split-Path -Parent $PSScriptRoot),

    [string]$RuntimeRoot,

    [string]$ConfigRoot,

    [ValidatePattern('^[0-9a-fA-F]{40}$')]
    [string]$PinnedSha,

    [string]$RuntimeBranch = 'codex/local-primary-runtime',

    [string]$TaskNamePrefix = 'New Seasonals Local - ',

    [string]$BootstrapPython,

    [switch]$ConfirmCutover
)

# Installs the local-primary scheduler in explicit, reversible phases:
#   Prepare          creates a pinned branch worktree and its own venv;
#   RegisterDisabled atomically registers all six tasks disabled;
#   Cutover          validates/enables them, then disables four legacy GHA
#                    dispatch triggers (never deletes any task);
#   Status           is read-only.
# Nothing in this script runs automatically merely because it is checked out.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$PipelineSpecs = @(
    [pscustomobject]@{ Id = 'premarket';       Time = '04:10:00'; DaysMask = 62; Description = 'Weekday premarket cache, risk, signal, and cloud-deploy handoff pipeline' },
    [pscustomobject]@{ Id = 'discretionary';   Time = '08:35:00'; DaysMask = 62; Description = 'Weekday research-only discretionary focus pipeline' },
    [pscustomobject]@{ Id = 'execution';       Time = '16:30:00'; DaysMask = 62; Description = 'Weekday execution reporting pipeline' },
    [pscustomobject]@{ Id = 'postclose';       Time = '17:10:00'; DaysMask = 62; Description = 'Weekday post-close data, reports, signals, and cloud-deploy handoff pipeline' },
    [pscustomobject]@{ Id = 'indicator';       Time = '03:00:00'; DaysMask = 2;  Description = 'Monday indicator-cache maintenance pipeline' },
    [pscustomobject]@{ Id = 'weekly-rundown'; Time = '08:00:00'; DaysMask = 1;  Description = 'Sunday weekly market rundown pipeline' }
)

$LegacyDispatchTasks = @(
    'Trigger CBOE Put-Call (GHA workflow_dispatch)',
    'Trigger Update Master Prices (GHA workflow_dispatch)',
    'Trigger Risk Report AM Correction (GHA workflow_dispatch)',
    'Trigger Daily Screener (GHA workflow_dispatch)'
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
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'RegisterDisabled and Cutover must run from an elevated Windows PowerShell session'
    }
}

function Invoke-GitCapture {
    param([Parameter(Mandatory = $true)][string[]]$Arguments, [switch]$AllowExitOne)
    $output = (& $script:GitExecutable @Arguments 2>&1 | Out-String).Trim()
    $code = $LASTEXITCODE
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
    $dirty = (Invoke-GitCapture -Arguments @('-C', $script:RuntimeRoot, 'status', '--porcelain', '--untracked-files=no')).Output
    if ($dirty) {
        throw 'Runtime worktree has tracked changes'
    }
    if (-not ([IO.Path]::GetFullPath([string]$marker.config_root)).Equals($script:ConfigRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Runtime marker ConfigRoot differs from the requested ConfigRoot'
    }

    $runner = Join-Path $script:RuntimeRoot 'scripts\run_local_automation.ps1'
    $powershell = 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
    if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) { throw "Runner missing: $runner" }
    if (-not (Test-Path -LiteralPath $powershell -PathType Leaf)) { throw "Windows PowerShell missing: $powershell" }
    & $powershell -NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File $runner `
        -Pipeline premarket -RuntimeRoot $script:RuntimeRoot -ConfigRoot $script:ConfigRoot -ValidateOnly
    if ($LASTEXITCODE -ne 0) {
        throw "Pinned runtime validation failed with exit code $LASTEXITCODE"
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
    catch { return $null }
}

function Quote-TaskArgument {
    param([string]$Value)
    if ($Value.Contains('"')) { throw 'Task arguments may not contain a double quote' }
    return '"' + $Value + '"'
}

function Register-DisabledTasks {
    Assert-Administrator
    Assert-EasternLocalClock
    $null = Assert-PinnedRuntime
    $scheduler = Connect-TaskScheduler
    $identityName = [Security.Principal.WindowsIdentity]::GetCurrent().Name
    $windowsPowerShell = 'C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe'
    $runner = Join-Path $script:RuntimeRoot 'scripts\run_local_automation.ps1'

    # Preflight all names so a conflict cannot leave a partially registered set.
    foreach ($spec in $PipelineSpecs) {
        $taskName = $TaskNamePrefix + $spec.Id
        if ($null -ne (Get-TaskOrNull -RootFolder $scheduler.Root -Name $taskName)) {
            throw "Task already exists; refusing to overwrite it: $taskName"
        }
    }

    foreach ($spec in $PipelineSpecs) {
        $taskName = $TaskNamePrefix + $spec.Id
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
        $action.Arguments = @(
            '-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
            '-File', (Quote-TaskArgument $runner),
            '-Pipeline', (Quote-TaskArgument $spec.Id),
            '-RuntimeRoot', (Quote-TaskArgument $script:RuntimeRoot),
            '-ConfigRoot', (Quote-TaskArgument $script:ConfigRoot)
        ) -join ' '

        # TASK_CREATE + TASK_LOGON_S4U.  TASK_CREATE (not
        # CREATE_OR_UPDATE) is intentional: this installer never overwrites or
        # deletes an existing task.
        $null = $scheduler.Root.RegisterTaskDefinition($taskName, $definition, 2, $identityName, $null, 2, $null)
        $registered = $scheduler.Root.GetTask($taskName)
        if ($registered.Enabled) {
            throw "Task was not atomically registered disabled: $taskName"
        }
        Write-Output "Registered disabled: $taskName"
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
    $newTasks = @()
    foreach ($spec in $PipelineSpecs) {
        $taskName = $TaskNamePrefix + $spec.Id
        $task = Get-TaskOrNull -RootFolder $scheduler.Root -Name $taskName
        if ($null -eq $task) { throw "Required registered task is missing: $taskName" }
        $newTasks += $task
    }

    $legacyState = @()
    foreach ($name in $LegacyDispatchTasks) {
        $task = Get-TaskOrNull -RootFolder $scheduler.Root -Name $name
        $legacyState += [pscustomobject]@{
            name = $name
            present = ($null -ne $task)
            enabled = if ($null -ne $task) { [bool]$task.Enabled } else { $false }
        }
    }
    $stateDir = Join-Path $script:RuntimeRoot 'artifacts\automation'
    New-Item -ItemType Directory -Path $stateDir -Force | Out-Null
    [pscustomobject]@{
        cutover_at = [DateTime]::UtcNow.ToString('o')
        pinned_sha = [string]$marker.pinned_sha
        legacy_tasks = $legacyState
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $stateDir 'cutover-state.json') -Encoding UTF8

    # Availability first: enable and verify every local primary task before
    # disabling any legacy dispatcher.  A partial failure can cause a duplicate
    # backup dispatch, but cannot silently remove the production schedule.
    foreach ($task in $newTasks) { $task.Enabled = $true }
    foreach ($task in $newTasks) {
        if (-not $task.Enabled) { throw "Failed to enable new task: $($task.Name)" }
    }

    foreach ($entry in $legacyState) {
        if (-not $entry.present) {
            Write-Output "Legacy task not present (nothing changed): $($entry.name)"
            continue
        }
        $legacy = $scheduler.Root.GetTask($entry.name)
        $legacy.Enabled = $false
        if ($legacy.Enabled) { throw "Failed to disable legacy dispatch task: $($entry.name)" }
        Write-Output "Disabled legacy dispatch task: $($entry.name)"
    }
    Write-Output "Cutover complete at pinned commit $($marker.pinned_sha). No tasks were deleted."
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
    foreach ($name in $LegacyDispatchTasks) {
        $task = Get-TaskOrNull -RootFolder $scheduler.Root -Name $name
        if ($null -eq $task) { Write-Output "$name : NOT PRESENT" }
        else { Write-Output "$name : enabled=$($task.Enabled)" }
    }
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

switch ($Phase) {
    'Prepare' {
        if (-not $PinnedSha) { throw 'Prepare requires -PinnedSha with the tested origin/main commit' }
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
            runtime_branch = $RuntimeBranch
            runtime_root = $RuntimeRoot
            config_root = $ConfigRoot
            git_executable = $script:GitExecutable
            prepared_at = [DateTime]::UtcNow.ToString('o')
        } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $markerDir 'automation-runtime.json') -Encoding UTF8

        $null = Assert-PinnedRuntime
        Write-Output "Prepared pinned runtime $RuntimeRoot at $PinnedSha. No tasks were registered."
    }
    'RegisterDisabled' { Register-DisabledTasks }
    'Cutover' { Invoke-Cutover }
    'Status' { Show-TaskStatus }
}
