[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('premarket', 'discretionary', 'execution', 'postclose', 'indicator', 'weekly-rundown', 'health')]
    [string]$Pipeline,

    [Parameter(Mandatory = $true)]
    [string]$RuntimeRoot,

    [Parameter(Mandatory = $true)]
    [string]$ConfigRoot,

    [switch]$ValidateOnly
)

# Thin, deliberately boring Task Scheduler boundary.  Code is pinned during
# an explicit install/upgrade; scheduled runs never fetch, merge, reset,
# checkout, or otherwise mutate their own source tree.
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-ExistingDirectory {
    param([Parameter(Mandatory = $true)][string]$Path, [string]$Label)
    if (-not [IO.Path]::IsPathRooted($Path)) {
        throw "$Label must be an absolute path: $Path"
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label does not exist: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Restore-ProcessEnvironment {
    param([string]$Name, [AllowNull()][string]$Value)
    if ($null -eq $Value) {
        Remove-Item -LiteralPath "Env:$Name" -ErrorAction SilentlyContinue
    }
    else {
        Set-Item -LiteralPath "Env:$Name" -Value $Value
    }
}

$RuntimeRoot = Resolve-ExistingDirectory -Path $RuntimeRoot -Label 'RuntimeRoot'
$ConfigRoot = Resolve-ExistingDirectory -Path $ConfigRoot -Label 'ConfigRoot'
$scriptRuntimeRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $scriptRuntimeRoot.Equals($RuntimeRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Runner/runtime mismatch: script=$scriptRuntimeRoot argument=$RuntimeRoot"
}

$markerPath = Join-Path $RuntimeRoot '.local\automation-runtime.json'
if (-not (Test-Path -LiteralPath $markerPath -PathType Leaf)) {
    throw "Pinned runtime marker is missing: $markerPath"
}
$marker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json

$markedRuntime = [IO.Path]::GetFullPath([string]$marker.runtime_root)
$markedConfig = [IO.Path]::GetFullPath([string]$marker.config_root)
if (-not $markedRuntime.Equals($RuntimeRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Pinned runtime marker does not match RuntimeRoot'
}
if (-not $markedConfig.Equals($ConfigRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Pinned runtime marker does not match ConfigRoot'
}
if ([string]$marker.pinned_sha -notmatch '^[0-9a-fA-F]{40}$') {
    throw 'Pinned runtime marker contains an invalid commit SHA'
}
if ([string]$marker.fallback_ref -notmatch '^[A-Za-z0-9._/-]+$') {
    throw 'Pinned runtime marker contains an invalid GitHub fallback ref'
}

$gitExecutable = [string]$marker.git_executable
if (-not [IO.Path]::IsPathRooted($gitExecutable) -or -not (Test-Path -LiteralPath $gitExecutable -PathType Leaf)) {
    throw 'Pinned runtime marker contains an invalid Git executable path'
}

$head = (& $gitExecutable -C $RuntimeRoot rev-parse HEAD 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to verify pinned runtime HEAD: $head"
}
if (-not $head.Equals([string]$marker.pinned_sha, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Runtime HEAD is not the tested pinned commit (expected $($marker.pinned_sha), got $head)"
}
$trackedChanges = (& $gitExecutable -C $RuntimeRoot status --porcelain --untracked-files=no 2>&1 | Out-String).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "Unable to verify runtime worktree state: $trackedChanges"
}
if ($trackedChanges) {
    throw 'Runtime worktree has tracked changes; refusing unattended execution'
}
$mutableState = @($marker.mutable_tracked_state)
if ($mutableState.Count -eq 0) {
    throw 'Pinned runtime marker does not declare mutable tracked state'
}
foreach ($path in $mutableState) {
    $entry = (& $gitExecutable -C $RuntimeRoot ls-files -v -- ([string]$path) 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or ($entry -and -not $entry.StartsWith('S '))) {
        throw "Mutable runtime state is not protected with skip-worktree: $path"
    }
}

$pythonExecutable = Join-Path $RuntimeRoot '.venv\Scripts\python.exe'
$supervisor = Join-Path $RuntimeRoot 'scripts\automation_supervisor.py'
foreach ($requiredFile in @($pythonExecutable, $supervisor)) {
    if (-not (Test-Path -LiteralPath $requiredFile -PathType Leaf)) {
        throw "Required pinned runtime file is missing: $requiredFile"
    }
}

if ($ValidateOnly) {
    Write-Output "Validated pinned local-automation runtime at $($head.Substring(0, 12))."
    return
}

$previousPrimary = [Environment]::GetEnvironmentVariable('LOCAL_AUTOMATION_PRIMARY', 'Process')
$previousToken = [Environment]::GetEnvironmentVariable('LOCAL_AUTOMATION_RUN_TOKEN', 'Process')
$previousGhToken = [Environment]::GetEnvironmentVariable('GH_TOKEN', 'Process')
$previousPythonIoEncoding = [Environment]::GetEnvironmentVariable('PYTHONIOENCODING', 'Process')
$previousPythonUtf8 = [Environment]::GetEnvironmentVariable('PYTHONUTF8', 'Process')
$previousAutomationStateRoot = [Environment]::GetEnvironmentVariable('NEW_SEASONALS_AUTOMATION_STATE_ROOT', 'Process')
$exitCode = 1
try {
    # Both values are scoped to this PowerShell process and inherited by the
    # supervisor's children.  The token is never copied to disk or printed.
    $env:LOCAL_AUTOMATION_PRIMARY = '1'
    $env:LOCAL_AUTOMATION_RUN_TOKEN = [Guid]::NewGuid().ToString('N')
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUTF8 = '1'
    $env:NEW_SEASONALS_AUTOMATION_STATE_ROOT = (Join-Path $RuntimeRoot 'artifacts\automation')
    if ([string]::IsNullOrWhiteSpace($previousGhToken)) {
        $userPat = [Environment]::GetEnvironmentVariable('GH_PAT_NEW_SEASONALS', 'User')
        if (-not [string]::IsNullOrWhiteSpace($userPat)) {
            # gh.exe reads GH_TOKEN.  The value stays process-scoped and is
            # neither copied to the runtime nor written to the task action.
            $env:GH_TOKEN = $userPat
        }
    }

    Write-Output "Starting local automation pipeline '$Pipeline' from pinned commit $($head.Substring(0, 12))."
    if ($Pipeline -eq 'health') {
        & $pythonExecutable -u $supervisor health --config-root $ConfigRoot --ref ([string]$marker.fallback_ref)
    }
    else {
        & $pythonExecutable -u $supervisor run-pipeline --pipeline $Pipeline --config-root $ConfigRoot --ref ([string]$marker.fallback_ref)
    }
    $exitCode = $LASTEXITCODE
}
finally {
    Restore-ProcessEnvironment -Name 'LOCAL_AUTOMATION_PRIMARY' -Value $previousPrimary
    Restore-ProcessEnvironment -Name 'LOCAL_AUTOMATION_RUN_TOKEN' -Value $previousToken
    Restore-ProcessEnvironment -Name 'GH_TOKEN' -Value $previousGhToken
    Restore-ProcessEnvironment -Name 'PYTHONIOENCODING' -Value $previousPythonIoEncoding
    Restore-ProcessEnvironment -Name 'PYTHONUTF8' -Value $previousPythonUtf8
    Restore-ProcessEnvironment -Name 'NEW_SEASONALS_AUTOMATION_STATE_ROOT' -Value $previousAutomationStateRoot
}

if ($exitCode -ne 0) {
    throw "Local automation pipeline '$Pipeline' failed with exit code $exitCode"
}
Write-Output "Local automation pipeline '$Pipeline' completed."
